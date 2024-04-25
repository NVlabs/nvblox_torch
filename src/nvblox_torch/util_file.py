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


# get paths
def get_module_path():
    path = os.path.dirname(__file__)
    return path


def get_cpp_build_path():
    path = os.path.join(get_module_path(), "cpp/build/")
    return path


def get_binary_path():
    path = os.path.join(get_module_path(), "bin/")
    return path


def join_path(path1, path2):
    return os.path.join(path1, path2)
