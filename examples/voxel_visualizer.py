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
import open3d as o3d
import torch


class SimpleVoxelVisualizer:
    def __init__(self, voxel_size, external=True) -> None:
        self.external = external
        if external:
            self.ext_vis = o3d.visualization.ExternalVisualizer()
            self.last_pcd = None
        self.last_map = None
        self.vis = o3d.visualization.Visualizer()
        self.voxel_size = voxel_size
        self.time = 0
        self.start()

    def start(self):
        self.vis.create_window()

        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0.5, 0.5, 0.5])
        opt.point_size = 1.0
        opt.show_coordinate_frame = True

    def end(self):
        o3d.visualization.draw_geometries([self.last_map])
        if self.external:
            self.ext_vis.set(self.last_pcd, path="pcd", time=0)

    def add_points(self, centers, colors):
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centers))
        pcd.colors = o3d.utility.Vector3dVector(colors)

        voxel_grid = o3d.geometry.VoxelGrid().create_from_point_cloud(
            pcd, voxel_size=self.voxel_size
        )
        self.vis.add_geometry(voxel_grid)
        self.last_map = voxel_grid
        if self.external:
            self.last_pcd = pcd
            self.time += 1

    def show(self, clear_after=True):
        view_control = self.vis.get_view_control()
        at = np.array([0, 0, 0])
        eye = np.array([4, 4, -4])
        front = at - eye
        up = np.array([0, 0, 1])
        view_control.set_front(front)
        view_control.set_lookat(at)
        view_control.set_up(up)

        self.vis.poll_events()
        self.vis.update_renderer()
        if clear_after:
            self.vis.clear_geometries()

    def __del__(self):
        self.vis.destroy_window()
