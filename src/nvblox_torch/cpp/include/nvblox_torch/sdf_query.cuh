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

#include <nvblox/core/indexing.h>
#include <nvblox/core/types.h>
#include <nvblox/mapper/mapper.h>
#include "nvblox/map/layer.h"
#include "nvblox/map/voxels.h"
#include <nvblox/gpu_hash/internal/cuda/gpu_indexing.cuh>
#include <nvblox/io/ply_writer.h>
#include <nvblox/primitives/scene.h>
#include <nvblox/utils/timing.h>


namespace pynvblox {
	namespace sdf {

    __global__ void queryDistancesKernel(
    int64_t num_queries, 
	nvblox::Index3DDeviceHashMapType<nvblox::EsdfBlock> block_hash,
    float block_size, 
	const float* query_spheres,
	float* out_closest_point,
	const bool write_closest_point = false);

	__global__ void queryDistancesMultiMapperKernel(
    int64_t num_mappers,
    int64_t num_queries, 
	  nvblox::Index3DDeviceHashMapType<nvblox::EsdfBlock>* hashes,
    float* block_sizes, 
	const float* query_spheres,
	float* out_closest_point,
	const bool write_closest_point);

	__global__ void queryTSDFMultiMapperKernel(
    int64_t num_mappers,
    int64_t num_queries, 
	  nvblox::Index3DDeviceHashMapType<nvblox::TsdfBlock>* hashes,
    float* block_sizes, 
    const float* query_spheres,
    float* out_tsdf,
    float* out_weight); 

	__global__ void queryOccupancyMultiMapperKernel(
    int64_t num_mappers,
    int64_t num_queries, 
	nvblox::Index3DDeviceHashMapType<nvblox::OccupancyBlock>* hashes,
    float* block_sizes, 
    const float* query_spheres,
    float* out_log_odds); 

	}
}