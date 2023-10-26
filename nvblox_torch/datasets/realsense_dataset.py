#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#


import numpy as np
import pyrealsense2 as rs


def setup_realsense(clipping_distance_m):
    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == "RGB Camera":
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    if device_product_line == "L500":
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)

    # We will be removing the background of objects more than
    #  clipping_distance_m meters away
    clipping_distance = clipping_distance_m / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
    intrinsics = color_profile.get_intrinsics()

    realsense = {
        "pipeline": pipeline,
        "align": align,
        "clipping_distance": clipping_distance,
        "depth_scale": depth_scale,
        "intrinsics": intrinsics,
    }

    return realsense


def get_next_depth_and_color(realsense, filter_color=False):
    # Get frameset of color and depth
    frames = realsense["pipeline"].wait_for_frames()
    # frames.get_depth_frame() is a 640x360 depth image

    # Align the depth frame to color frame
    aligned_frames = realsense["align"].process(frames)

    # Get aligned frames
    aligned_depth_frame = (
        aligned_frames.get_depth_frame()
    )  # aligned_depth_frame is a 640x480 depth image
    color_frame = aligned_frames.get_color_frame()

    # Validate that both frames are valid
    if not aligned_depth_frame or not color_frame:
        return None, None

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Remove background - Set pixels further than clipping_distance to grey
    grey_color = 153
    depth_image_3d = np.dstack(
        (depth_image, depth_image, depth_image)
    )  # depth image is 1 channel, color is 3 channels

    if filter_color:
        color_image = np.where(
            (depth_image_3d > realsense["clipping_distance"]) | (depth_image_3d <= 0),
            grey_color,
            color_image,
        )

    depth_image[depth_image > realsense["clipping_distance"]] = 0.0
    depth_image = depth_image.astype(np.float32) * realsense["depth_scale"]

    return depth_image, color_image


def get_intrinsics_matrix(realsense):
    intrinsics = realsense["intrinsics"]
    intrinsic_matrix = np.array(
        [
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1],
        ]
    )
    return intrinsic_matrix


from transforms3d import affines, euler


def get_default_pose(realsense):
    # eul = [-2.2, 0.08750253722127824, 1.4509891417591223]
    # p = [ 6, 0,  4]
    eul = [-2.0, 0, 0]
    p = [1.0, -1.5, 0.5]
    R = euler.euler2mat(*eul)
    T = affines.compose(p, R, [1, 1, 1])
    return np.array(T)


def stop_realsense(realsense):
    realsense["pipeline"].stop()
    del realsense["pipeline"]


import torch


class RealsenseDataloader:
    def __init__(self, max_steps=1e6, clipping_distance_m=2.5):
        self.max_steps = int(max_steps)
        self.step = 0
        self.realsense = setup_realsense(clipping_distance_m)

    def __len__(self):
        return self.max_steps

    def __iter__(self):
        if self.step > 0:
            raise IndexError("RealsenseDataloader can only be iterated once")
        self.step = 0
        return self

    def __next__(self):
        if self.step >= self.max_steps:
            raise StopIteration
        depth_np, rgb_np = get_next_depth_and_color(self.realsense, filter_color=False)
        intrinsics_np = get_intrinsics_matrix(self.realsense)
        pose_np = get_default_pose(self.realsense)

        rgb = torch.from_numpy(rgb_np)
        rgba = torch.cat([rgb, torch.ones_like(rgb[:, :, 0:1]) * 255], dim=-1)
        depth = torch.from_numpy(depth_np).float()
        pose = torch.from_numpy(pose_np).float()
        intrinsics = torch.from_numpy(intrinsics_np).float()

        return {
            "rgba": rgba,
            "depth": depth,
            "pose": pose,
            "intrinsics": intrinsics,
            "raw_rgb": rgb_np,
            "raw_depth": depth_np,
            "rgba_nvblox": rgba.permute((1, 2, 0)).contiguous(),
        }

    def get_data(self):
        return self.__next__()

    def get_raw_data(self):
        depth_np, rgb_np = get_next_depth_and_color(self.realsense, filter_color=False)
        return depth_np, rgb_np

    def stop_device(self):
        stop_realsense(self.realsense)
        del self.realsense
