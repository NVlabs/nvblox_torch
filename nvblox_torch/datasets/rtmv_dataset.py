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
import json
import os

import imageio
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from transforms3d.quaternions import quat2mat


class RtmvDataset(Dataset):
    def __init__(self, root) -> None:
        super().__init__()
        self.root = root
        self.frame_names = list(
            sorted(
                set(
                    [
                        f.split(".")[0]
                        for f in os.listdir(self.root)
                        if "combined" not in f
                    ]
                )
            )
        )
        self.json_files = [
            os.path.join(self.root, f"{f}.json") for f in self.frame_names
        ]
        self.rgb_files = [os.path.join(self.root, f"{f}.png") for f in self.frame_names]
        self.depth_files = [
            os.path.join(self.root, f"{f}.depth.png") for f in self.frame_names
        ]

    def load_json(self, json_file):
        with open(json_file, "r") as fp:
            obj = json.load(fp)

        camera_pose = torch.tensor(obj["camera_data"]["cam2world"], dtype=torch.float32)
        intrinsics_dict = obj["camera_data"]["intrinsics"]
        intrinsics = torch.tensor(
            [
                [intrinsics_dict["fx"], 0, intrinsics_dict["cx"]],
                [0, intrinsics_dict["fy"], intrinsics_dict["cy"]],
                [0, 0, 1],
            ]
        )
        return intrinsics, camera_pose

    def __len__(self):
        return len(self.frame_names)

    def __getitem__(self, index):
        # frame_name = self.frame_names[index]

        rgba_np = imageio.imread(self.rgb_files[index])
        depth_np = imageio.imread(self.depth_files[index])
        intrinsics, pose = self.load_json(self.json_files[index])

        depth_np = depth_np.astype(np.float32) / 1000
        rgba = torch.from_numpy(rgba_np).permute((2, 0, 1))
        depth = torch.from_numpy(depth_np).float()

        # eigen_quat = [0.707106769, 0.707106769, 0, 0]
        # sun3d_to_nvblox_T = torch.eye(4)
        # sun3d_to_nvblox_T[:3, :3] = torch.tensor(quat2mat(eigen_quat))

        # nvblox_pose = sun3d_to_nvblox_T @ pose

        # // Rotate the world frame since Y is up in the normal 3D match dasets.
        # Eigen::Quaternionf q_L_O = Eigen::Quaternionf::FromTwoVectors(Vector3f(0, 1, 0), Vector3f(0, 0, 1));

        return {"rgba": rgba, "depth": depth, "pose": pose, "intrinsics": intrinsics}
