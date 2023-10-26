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
import torch

from nvblox_torch.mapper import Mapper
from nvblox_torch.scene import Scene


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
    assert torch.norm(r - 0.3160).item() < 1e-3
