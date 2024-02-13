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
import math

import cv2
import torch
from transforms3d import affines, euler

from nvblox_torch.mapper import Mapper
from nvblox_torch.scene import Scene


def get_global_pose(device="cuda"):
    eul = [math.radians(90), math.radians(180), -math.radians(45)]
    p = [4, 4, 2]

    R = euler.euler2mat(*eul)
    T = affines.compose(p, R, [1, 1, 1])
    return torch.Tensor(T).to(device)


def get_intrinsics(h_fov, height, width, device="cuda"):
    fx = width / (2 * math.tan(math.radians(h_fov) / 2))
    fy = fx  # square pixels
    return torch.tensor([[fx, 0, width / 2], [0, fy, height / 2], [0, 0, 1]], device=device)


def create_dummy_map():
    scene = Scene()

    scene.set_aabb([-5.5, -5.5, -0.5], [5.5, 5.5, 5.5])
    scene.add_plane_boundaries(-5.0, 5.0, -5.0, 5.0)
    scene.add_ground_level(0.0)
    scene.add_ceiling(5.0)
    scene.add_primitive("cube", [0.0, 0.0, 2.0, 2.0, 2.0, 2.0])
    scene.add_primitive("sphere", [0.0, 0.0, 2.0, 2.0])
    return scene


def test_py_dummy_map():
    scene = create_dummy_map()
    mapper = Mapper(voxel_sizes=[0.02], integrator_types=["tsdf"])
    mapper.build_from_scene(scene, 0)

    batch_size = 10
    tensor_args = {"device": "cuda", "dtype": torch.float32}
    query_spheres = torch.zeros((batch_size, 4), **tensor_args) + 0.5
    query_spheres[:, 3] = 0.001
    out_points = torch.zeros((batch_size, 4), **tensor_args) + 0.0

    r = mapper.query_sdf(query_spheres, out_points, True, 0)
    print(r)

    camera_pose = get_global_pose()
    intrinsics = get_intrinsics(90, 480, 640, "cpu")
    depth = mapper.render_depth_image(
        0, camera_pose, intrinsics, 480, 640, max_ray_length=12.0, max_steps=100
    )

    depth_np = (depth.clamp(0, 10.0) / depth.max()).detach().cpu().numpy()
    # cv2.imshow("Depth", depth_np)
    # cv2.waitKey(0)


if __name__ == "__main__":
    test_py_dummy_map()
