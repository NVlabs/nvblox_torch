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
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <nvblox/sensors/image.h>

#include "nvblox_torch/convert_tensors.h"

TEST(ConvertTensors, ImageConversion) {
  // Dummy image.
  nvblox::DepthImage image_1(2, 2, nvblox::MemoryType::kHost);
  image_1(0, 0) = 0.f;
  image_1(0, 1) = 1.f;
  image_1(1, 0) = 0.f;
  image_1(1, 1) = 1.f;

  // Convert to a tensor.
  torch::Tensor image_t = pynvblox::copy_depth_image_to_tensor(image_1);

  // Back to an image
  nvblox::DepthImage image_2 = pynvblox::copy_depth_image_from_tensor(image_t);

  constexpr float kEps = 1e-6;
  EXPECT_NEAR(image_2(0, 0), 0.f, kEps);
  EXPECT_NEAR(image_2(0, 1), 1.f, kEps);
  EXPECT_NEAR(image_2(1, 0), 0.f, kEps);
  EXPECT_NEAR(image_2(1, 1), 1.f, kEps);
}

TEST(ConvertTensors, TransformConversion) {
  torch::Tensor transform_t = torch::eye(4);
  nvblox::Transform transform =
      pynvblox::copy_transform_from_tensor(transform_t);

  const float diff =
      (transform.matrix() - nvblox::Transform::Identity().matrix()).sum();
  constexpr float kEps = 1e-6;
  EXPECT_LT(diff, kEps);
}

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
