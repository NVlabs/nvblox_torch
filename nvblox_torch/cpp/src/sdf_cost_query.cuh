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
		namespace cost{


	__global__ void sphereDistanceCostMultiKernel(
	const float *sphere_pos_rad, 
	float *out_distance,
	float *out_grad,
	uint8_t *sparsity_idx,
	const float *weight,
	const float *activation_distance,
	nvblox::Index3DDeviceHashMapType<nvblox::EsdfBlock>* block_hash,
	const float *blox_pose,
    const uint8_t *blox_enable,
	const float *block_sizes,
	const int batch_size, const int horizon, const int nspheres, 
	const bool write_grad,
	const int num_mappers);

	__global__ void sphereTrajectoryDistanceCostMultiKernel(
	const float *sphere_pos_rad, 
	float *out_distance,
	float *out_grad,
	uint8_t *sparsity_idx,
	const float *weight,
	const float *activation_distance,
	const float *speed_dt,
	nvblox::Index3DDeviceHashMapType<nvblox::EsdfBlock>* block_hash,
	const float *blox_pose,
	const uint8_t *blox_enable,
    const float *block_sizes,
	const int batch_size, const int horizon, const int nspheres,
	const int sweep_steps, const bool enable_speed_metric, 
	const bool write_grad,
	const int num_mappers);


	__global__ void sphereSweptTrajectoryDistanceCostMultiKernel(
	const float *sphere_pos_rad, 
	float *out_distance,
	float *out_grad,
	uint8_t *sparsity_idx,
	const float *weight,
	const float *activation_distance,
	const float *speed_dt,
	nvblox::Index3DDeviceHashMapType<nvblox::EsdfBlock>* block_hash,
	const float *blox_pose,
	const uint8_t *blox_enable,
    const float *block_sizes,
	const int batch_size, const int horizon, const int nspheres,
	const int sweep_steps, const bool enable_speed_metric, 
	const bool write_grad,
	const int num_mappers);
}
	
}

}