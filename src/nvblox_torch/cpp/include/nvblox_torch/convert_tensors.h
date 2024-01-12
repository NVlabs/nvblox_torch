/*
 * Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */
#pragma once
#include <torch/script.h>

#include "nvblox/core/types.h"
#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/image.h"


namespace pynvblox {

torch::Tensor init_depth_image_tensor(int64_t height, int64_t width, torch::DeviceType device);

torch::Tensor init_color_image_tensor(int64_t height, int64_t width, torch::DeviceType device);


nvblox::DepthImage copy_depth_image_from_tensor(torch::Tensor depth_image_t);

nvblox::ColorImage copy_color_image_from_tensor(torch::Tensor color_image_t);


torch::Tensor copy_depth_image_to_tensor(const nvblox::DepthImage& depth_image);

torch::Tensor copy_color_image_to_tensor(const nvblox::ColorImage& color_image);


nvblox::DepthImageView make_depth_image_view(torch::Tensor depth_image_t);

nvblox::ColorImageView make_color_image_view(torch::Tensor color_image_t);


nvblox::Transform copy_transform_from_tensor(torch::Tensor transform_t);

nvblox::Camera camera_from_intrinsics_tensor(torch::Tensor intrinsics_t, int height, int width);

nvblox::MemoryType memory_type_from_torch_device(torch::DeviceType device_type);

torch::DeviceType memory_type_to_torch_device(nvblox::MemoryType memory_type);

} // namespace pynvblox

