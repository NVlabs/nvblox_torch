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

from nvblox_torch.util_file import get_binary_path, join_path


def get_nvblox_scene_class():
    torch.classes.load_library(join_path(get_binary_path(), "libpy_nvblox.so"))
    Scene = torch.classes.pynvblox.Scene
    return Scene


class Scene:
    def __init__(self, c_scene=None):
        if c_scene is None:
            SceneCls = get_nvblox_scene_class()
            self._c_scene = SceneCls()
        else:
            self._c_scene = c_scene

    def set_aabb(self, low, high):
        self._c_scene.set_aabb(low, high)

    def add_plane_boundaries(self, x_min, x_max, y_min, y_max):
        self._c_scene.add_plane_boundaries(x_min, x_max, y_min, y_max)

    def add_ground_level(self, level):
        self._c_scene.add_ground_level(level)

    def add_ceiling(self, ceiling):
        self._c_scene.add_ceiling(ceiling)

    def add_primitive(self, type, params):
        self._c_scene.add_primitive(type, params)

    def create_dummy_map(self):
        self._c_scene.create_dummy_map()

    def get_c_scene(self):
        return self._c_scene
