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


import argparse
import sys

import cv2
import numpy as np
import torch
from transforms3d import affines, euler

import examples.realsense_utils as rsutils


def get_default_camera_pose(device="cuda"):
    eul = [-2.0, 0, 0]
    p = [1.0, -1.5, 0.5]
    R = euler.euler2mat(*eul)
    T = affines.compose(p, R, [1, 1, 1])
    return torch.Tensor(T).to(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clipping-distance", type=float, default=1.50)
    args = parser.parse_args()

    realsense = rsutils.setup_realsense(args.clipping_distance)

    # Streaming loop
    try:
        intrinsics = rsutils.get_intrinsics_matrix(realsense)
        pose = get_default_camera_pose()

        while True:
            depth_image_np, color_image_np = rsutils.get_next_depth_and_color(realsense)

            depth_image = torch.from_numpy(depth_image_np).cuda()
            color_image = torch.from_numpy(color_image_np).cuda()

            # Render images:
            #   depth align to color on left
            #   depth on right
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image_np, alpha=100), cv2.COLORMAP_JET
            )
            images = np.hstack((color_image_np, depth_colormap))

            cv2.namedWindow("Align Example", cv2.WINDOW_NORMAL)
            cv2.imshow("Align Example", images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord("q") or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        rsutils.stop_realsense(realsense)
