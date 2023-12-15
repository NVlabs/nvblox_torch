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
import os

import imageio
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from transforms3d.quaternions import quat2mat


class Sun3dDataset(Dataset):
    def __init__(self, root) -> None:
        super().__init__()
        self.root = root

        cam_intrinsics_np = np.loadtxt(os.path.join(self.root, "camera-intrinsics.txt"))
        self.camera_intrinsics = torch.from_numpy(cam_intrinsics_np).float()
        self.sequence_name = list(
            sorted(
                [
                    d
                    for d in os.listdir(self.root)
                    if os.path.isdir(os.path.join(self.root, d))
                ]
            )
        )[0]
        self.seq_dir = os.path.join(self.root, self.sequence_name)

        self.frame_names = list(
            sorted(set([f.split(".")[0] for f in os.listdir(self.seq_dir)]))
        )

    def __len__(self):
        return len(self.frame_names)

    def __getitem__(self, index):
        """ rgba: 4xHxW, depth HxW
        """
        frame_name = self.frame_names[index]
        rgb_np = imageio.imread(os.path.join(self.seq_dir, f"{frame_name}.color.png"))
        depth_np = imageio.imread(os.path.join(self.seq_dir, f"{frame_name}.depth.png"))
        pose_np = np.loadtxt(os.path.join(self.seq_dir, f"{frame_name}.pose.txt"))

        depth_np = depth_np.astype(np.float32) / 1000

        rgb = torch.from_numpy(rgb_np)
        rgba = torch.cat([rgb, torch.ones_like(rgb[:, :, 0:1]) * 255], dim=-1).permute(2,0,1)
        depth = torch.from_numpy(depth_np).float()
        pose = torch.from_numpy(pose_np).float()

        eigen_quat = [0.707106769, 0.707106769, 0, 0]
        sun3d_to_nvblox_T = torch.eye(4)
        sun3d_to_nvblox_T[:3, :3] = torch.tensor(quat2mat(eigen_quat))

        nvblox_pose = sun3d_to_nvblox_T @ pose

        # // Rotate the world frame since Y is up in the normal 3D match dasets.
        # Eigen::Quaternionf q_L_O = Eigen::Quaternionf::FromTwoVectors(Vector3f(0, 1, 0), Vector3f(0, 0, 1));

        return {
            "rgba": rgba,
            "depth": depth,
            "pose": nvblox_pose,
            "intrinsics": self.camera_intrinsics,
        }
