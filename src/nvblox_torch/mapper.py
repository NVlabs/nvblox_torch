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
from typing import List, Optional

import torch

from nvblox_torch.sdf_query import (
    SdfSphereBlox,
    SdfSphereCostMultiBlox,
    SdfSphereTrajectoryCostMultiBlox,
)
from nvblox_torch.util_file import get_binary_path, join_path


def get_nvblox_mapper_class():
    # print(f"Looking for class in: {cpp_build_path}")
    torch.classes.load_library(join_path(get_binary_path(), "libpy_nvblox.so"))
    Mapper = torch.classes.pynvblox.Mapper
    return Mapper


class Mapper:
    """
    This wrapper gives slightly more control over the PyTorch API compared to the C++ exposed object
    (e.g. supporting keyword arguments)
    """

    def __init__(
        self,
        voxel_sizes: List[int],
        integrator_types: List[str],
        layer_parameters: List[float] = [2.0],
        free_on_destruction: bool = False,
        cuda_device_id: int = 0,
    ) -> None:
        # Initialize c_mapper with layers = len(voxel_sizes)
        self._c_mapper = get_nvblox_mapper_class()(
            voxel_sizes, integrator_types, layer_parameters, free_on_destruction
        )
        self._voxel_sizes = voxel_sizes
        self._integrator_types = integrator_types

    def add_depth_frame(self, depth_frame, pose, intrinsics, mapper_id):
        assert 0 <= mapper_id < len(self._voxel_sizes)
        intrinsics = intrinsics.to("cpu")
        self._c_mapper.integrate_depth(depth_frame, pose, intrinsics, mapper_id)

    def add_color_frame(self, color_frame, pose, intrinsics, mapper_id):
        assert 0 <= mapper_id < len(self._voxel_sizes)
        intrinsics = intrinsics.to("cpu")
        self._c_mapper.integrate_color(color_frame, pose, intrinsics, mapper_id)

    def update_esdf(self, mapper_id=-1):
        assert -1 <= mapper_id < len(self._voxel_sizes)

        self._c_mapper.update_esdf(mapper_id)

    def update_mesh(self, mapper_id=-1):
        assert -1 <= mapper_id < len(self._voxel_sizes)
        self._c_mapper.update_mesh(mapper_id)

    def full_update(self, depth_frame, color_frame, pose, intrinsics, mapper_id):
        assert 0 <= mapper_id < len(self._voxel_sizes)
        intrinsics = intrinsics.to("cpu")
        self._c_mapper.full_update(depth_frame, color_frame, pose, intrinsics, mapper_id)

    def decay_occupancy(self, mapper_id=-1):
        assert -1 <= mapper_id < len(self._voxel_sizes)
        # assert (
        #    self._integrator_types[mapper_id] == "occupancy"
        # ), "Only occupancy integrator supports occupancy decay"
        self._c_mapper.decay_occupancy(mapper_id)

    def update_hashmaps(self):
        self._c_mapper.update_hashmaps()

    def clear(self, mapper_id=-1):
        assert -1 <= mapper_id < len(self._voxel_sizes)
        self._c_mapper.clear(mapper_id)

    def save_mesh(self, mesh_fname, mapper_id):
        assert 0 <= mapper_id < len(self._voxel_sizes)
        self._c_mapper.update_mesh(mapper_id)
        self._c_mapper.output_mesh_ply(mesh_fname, mapper_id)

    def save_map(self, map_fname, mapper_id):
        assert 0 <= mapper_id < len(self._voxel_sizes)
        self._c_mapper.output_blox_map(map_fname, mapper_id)

    def render_depth_image(
        self,
        mapper_id,
        camera_pose,
        intrinsics,
        height,
        width,
        max_ray_length=20.0,
        max_steps=100,
    ):
        assert 0 <= mapper_id < len(self._voxel_sizes)
        assert self._integrator_types[mapper_id] == "tsdf"

        intrinsics = intrinsics.to("cpu")
        return self._c_mapper.render_depth_image(
            camera_pose, intrinsics, height, width, max_ray_length, max_steps, mapper_id
        )

    def render_depth_and_color_image(
        self,
        mapper_id,
        camera_pose,
        intrinsics,
        height,
        width,
        max_ray_length=20.0,
        max_steps=100,
    ):
        assert 0 <= mapper_id < len(self._voxel_sizes)
        assert self._integrator_types[mapper_id] == "tsdf"

        intrinsics = intrinsics.to("cpu")
        depth, color = self._c_mapper.render_depth_and_color_image(
            camera_pose, intrinsics, height, width, max_ray_length, max_steps, mapper_id
        )
        return depth, color

    def create_dummy_map(self, mapper_id):
        assert 0 <= mapper_id < len(self._voxel_sizes)
        self._c_mapper.create_dummy_map(mapper_id)

    def load_from_file(self, filename, mapper_id):
        assert 0 <= mapper_id < len(self._voxel_sizes)
        return self._c_mapper.load_from_file(filename, mapper_id)

    def get_occupied_voxels_on_grid(self, min_coord, max_coord, voxel_size):
        ...

    def query_sdf(
        self,
        sphere_position_rad: torch.Tensor,
        out_spheres: Optional[torch.Tensor] = None,
        write_closest_point: bool = True,
        mapper_id=-1,
    ):
        assert -1 <= mapper_id < len(self._voxel_sizes)
        distance = SdfSphereBlox.apply(
            sphere_position_rad,
            self._c_mapper,
            out_spheres,
            write_closest_point,
            mapper_id,
        )

        return distance

    def query_tsdf(
        self,
        sphere_position_rad: torch.Tensor,
        outputs: Optional[torch.Tensor] = None,
        mapper_id=-1,
    ):
        assert mapper_id == -1, "Currently only multiple mapper query is supported"

        if outputs is None:
            num_queries = sphere_position_rad.shape[0]
            outputs = torch.zeros(
                (2, num_queries), dtype=torch.float32, device=sphere_position_rad.device
            )

        self._c_mapper.query_multi_tsdf(outputs, sphere_position_rad, num_queries)
        return outputs

    def query_occupancy(
        self,
        sphere_position_rad: torch.Tensor,
        outputs: Optional[torch.Tensor] = None,
        mapper_id=-1,
    ):
        assert mapper_id == -1, "Currently only multiple mapper query is supported"

        if outputs is None:
            num_queries = sphere_position_rad.shape[0]
            outputs = torch.zeros(
                [num_queries], dtype=torch.float32, device=sphere_position_rad.device
            )

        self._c_mapper.query_multi_occupancy(outputs, sphere_position_rad, num_queries)
        return outputs

    def query_sphere_sdf_cost(
        self,
        sphere_position_rad: torch.Tensor,
        out_distance: torch.Tensor,
        out_grad: torch.Tensor,
        sparsity_idx: torch.Tensor,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        blox_pose: torch.Tensor,
        blox_enable: torch.Tensor,
    ):
        distance = SdfSphereCostMultiBlox.apply(
            sphere_position_rad,
            out_distance,
            out_grad,
            sparsity_idx,
            weight,
            activation_distance,
            self._c_mapper,
            blox_pose,
            blox_enable,
        )
        return distance

    def query_sphere_trajectory_sdf_cost(
        self,
        sphere_position_rad: torch.Tensor,
        out_distance: torch.Tensor,
        out_grad: torch.Tensor,
        sparsity_idx: torch.Tensor,
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        speed_dt: torch.Tensor,
        blox_pose: torch.Tensor,
        blox_enable: torch.Tensor,
        sweep_steps: int = 0,
        enable_speed_metric: bool = False,
        return_loss: bool = False,
        use_experimental: bool = False,
    ):
        distance = SdfSphereTrajectoryCostMultiBlox.apply(
            sphere_position_rad,
            out_distance,
            out_grad,
            sparsity_idx,
            weight,
            activation_distance,
            speed_dt,
            self._c_mapper,
            blox_pose,
            blox_enable,
            sweep_steps,
            enable_speed_metric,
            return_loss,
            use_experimental,
        )
        return distance

    def build_from_scene(self, scene, mapper_id):
        assert 0 <= mapper_id < len(self._voxel_sizes)
        self._c_mapper.build_from_scene(scene.get_c_scene(), mapper_id)

    @property
    def num_layers(self) -> int:
        return self._c_mapper.num_layers

    def get_mesh(self, mapper_id: int = 0):
        out = self._c_mapper.get_mesh(mapper_id)
        mesh_data = {
            "vertices": out[0],
            "normals": out[1],
            "colors": out[2],
            "triangles": out[3],
        }
        if mesh_data["normals"].shape[0] == 0:
            mesh_data["normals"] = None
        if mesh_data["colors"].shape[0] == 0:
            mesh_data["colors"] = None

        return mesh_data
