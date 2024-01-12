##
## Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
##
## NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
## property and proprietary rights in and to this material, related
## documentation and any modifications thereto. Any use, reproduction,
## disclosure or distribution of this material and related documentation
## without an express license agreement from NVIDIA CORPORATION or
## its affiliates is strictly prohibited.
##

cd src/nvblox_torch/cpp && mkdir -p build && cd build && cmake -DCMAKE_PREFIX_PATH="$1" -DCMAKE_CUDA_COMPILER=$(which nvcc) .. && make -j32 && cd ../../../../
