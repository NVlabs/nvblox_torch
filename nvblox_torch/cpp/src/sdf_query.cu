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
#include "sdf_query.cuh"

#include <c10/cuda/CUDAStream.h>
#include <iostream>


namespace pynvblox {
  namespace sdf{

__global__ void queryDistancesKernel(
    int64_t num_queries, 
	  nvblox::Index3DDeviceHashMapType<nvblox::EsdfBlock> block_hash,
    float block_size, 
	const float* query_spheres,
	float* out_closest_point,
	const bool write_closest_point) 
{

  // Figure out which point this thread should be querying. 
  size_t query_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (query_index >= num_queries) {
    return;
  }
  
  constexpr int kNumVoxelsPerBlock = 8;
  const float voxel_size = block_size / kNumVoxelsPerBlock;
  nvblox::Vector3f query_location;

  // read data into vector3f:
  float radius = query_spheres[query_index*4 + 3];
  float distance = -100.0f;
  query_location(0) = query_spheres[query_index*4 + 0];
  query_location(1) = query_spheres[query_index*4 + 1];
  query_location(2) = query_spheres[query_index*4 + 2];
    
  nvblox::Vector3f closest_pt;

  // Get the correct block from the hash.
  nvblox::EsdfVoxel* esdf_voxel;
  if (!nvblox::getVoxelAtPosition<nvblox::EsdfVoxel>(block_hash, query_location, block_size,
                                     &esdf_voxel) ||
      !esdf_voxel->observed) {
    // This voxel is outside of the map or not observed. Mark it as 100 meters
    // behind a surface.
    
    closest_pt = query_location;
  } else {
    // Get the distance of the relevant voxel.
	distance = voxel_size * sqrt(esdf_voxel->squared_distance_vox) ;

    // If it's outside, we set the value to be negative
    if (!esdf_voxel->is_inside) {
		distance = -distance;
    }
	distance += radius;
    // TODO(helen): quick hack to get ~approx location of parent, should be
    // within a voxel of where it actually should be.
    // The parent location is relative to the location of the current voxel.
	if (write_closest_point)
	{
		// closest_pt =
    //    query_location +
    //    voxel_size * esdf_voxel->parent_direction.cast<float>();
    closest_pt =
        -voxel_size * esdf_voxel->parent_direction.cast<float>();
 
	}
	}
  // write distance and closest point out:
  out_closest_point[query_index*4 + 3] = distance;
  if (write_closest_point)
  {
	  out_closest_point[query_index*4 + 0] = closest_pt(0);
	  out_closest_point[query_index*4 + 1] = closest_pt(1);
	  out_closest_point[query_index*4 + 2] = closest_pt(2);
	  
  }

}

const float MAX_DST = 100.0f;

__global__ void queryDistancesMultiMapperKernel(
    int64_t num_mappers,
    int64_t num_queries, 
	  nvblox::Index3DDeviceHashMapType<nvblox::EsdfBlock>* hashes,
    float* block_sizes, 
	const float* query_spheres,
	float* out_closest_point,
	const bool write_closest_point) 
{
  // Figure out which point this thread should be querying. 
  size_t query_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (query_index >= num_queries) {
    return;
  }
  
  constexpr int kNumVoxelsPerBlock = 8;
  nvblox::Vector3f query_location;

  // read data into vector3f:
  float radius = query_spheres[query_index*4 + 3];
  float distance = -MAX_DST;
  float min_distance = MAX_DST;
  query_location(0) = query_spheres[query_index*4 + 0];
  query_location(1) = query_spheres[query_index*4 + 1];
  query_location(2) = query_spheres[query_index*4 + 2];
    
  nvblox::Vector3f closest_pt;

  for (int i = 0; i < num_mappers; i++)
  {
    const float block_size = block_sizes[i];

    const float voxel_size = block_size / kNumVoxelsPerBlock;

    // Get the correct block from the hash.
    nvblox::EsdfVoxel* esdf_voxel;
    if (!nvblox::getVoxelAtPosition<nvblox::EsdfVoxel>(hashes[i], query_location, block_size,
                                      &esdf_voxel) ||
        !esdf_voxel->observed) {
      // This voxel is outside of the map or not observed. Mark it as 100 meters
      // behind a surface.
      
      closest_pt = query_location;
    } else {
      // Get the distance of the relevant voxel.
      distance = voxel_size * sqrt(esdf_voxel->squared_distance_vox) ;

      // If it's outside, we set the value to be negative
      if (!esdf_voxel->is_inside) {
        distance = -distance;
      }
      distance += radius;
      // TODO(helen): quick hack to get ~approx location of parent, should be
      // within a voxel of where it actually should be.
      // The parent location is relative to the location of the current voxel.
      if (write_closest_point)
      {
        // closest_pt =
        //    query_location +
        //    voxel_size * esdf_voxel->parent_direction.cast<float>();
        closest_pt =
            -voxel_size * esdf_voxel->parent_direction.cast<float>();
    
      }
    }
    if (distance < min_distance || min_distance == -MAX_DST)
    {
      min_distance = distance;
      // write distance and closest point out:
      out_closest_point[query_index*4 + 3] = distance;
      if (write_closest_point)
      {
        out_closest_point[query_index*4 + 0] = closest_pt(0);
        out_closest_point[query_index*4 + 1] = closest_pt(1);
        out_closest_point[query_index*4 + 2] = closest_pt(2);
      }
    }
  }
}


__global__ void queryTSDFMultiMapperKernel(
    int64_t num_mappers,
    int64_t num_queries, 
	  nvblox::Index3DDeviceHashMapType<nvblox::TsdfBlock>* hashes,
    float* block_sizes, 
    const float* query_spheres,
    float* out_tsdf,
    float* out_weight) 
{
  // Figure out which point this thread should be querying. 
  size_t query_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (query_index >= num_queries) {
    return;
  }
  
  constexpr int kNumVoxelsPerBlock = 8;
  nvblox::Vector3f query_location;

  // read data into vector3f:
  float radius = query_spheres[query_index*4 + 3];
  // TODO: See if we want to get rid of the radius
  query_location(0) = query_spheres[query_index*4 + 0];
  query_location(1) = query_spheres[query_index*4 + 1];
  query_location(2) = query_spheres[query_index*4 + 2];
  
  float weight = 0;
  float distance = -MAX_DST;
  float abs_distance;

  float min_abs_distance = MAX_DST;
  float max_weight = 0;

  for (int i = 0; i < num_mappers; i++)
  {
      const float block_size = block_sizes[i];

    const float voxel_size = block_size / kNumVoxelsPerBlock;

    // Get the correct block from the hash.
    nvblox::TsdfVoxel* tsdf_voxel;
    if (nvblox::getVoxelAtPosition<nvblox::TsdfVoxel>(hashes[i], query_location, block_size, &tsdf_voxel)) {
      abs_distance = tsdf_voxel->distance > 0 ? tsdf_voxel->distance : -tsdf_voxel->distance;
      if (abs_distance < min_abs_distance)
      {
        // Get the distance of the relevant voxel.
        distance = tsdf_voxel->distance;
        weight = tsdf_voxel->weight;
      }
    }
  }
  out_tsdf[query_index] = distance;
  out_weight[query_index] = weight;
}

__global__ void queryOccupancyMultiMapperKernel(
    int64_t num_mappers,
    int64_t num_queries, 
	  nvblox::Index3DDeviceHashMapType<nvblox::OccupancyBlock>* hashes,
    float* block_sizes, 
    const float* query_spheres,
    float* out_log_odds) 
{
  // Figure out which point this thread should be querying. 
  size_t query_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (query_index >= num_queries) {
    return;
  }
  
  constexpr int kNumVoxelsPerBlock = 8;
  nvblox::Vector3f query_location;

  // read data into vector3f:
  float radius = query_spheres[query_index*4 + 3];
  // TODO: See if we want to get rid of the radius
  query_location(0) = query_spheres[query_index*4 + 0];
  query_location(1) = query_spheres[query_index*4 + 1];
  query_location(2) = query_spheres[query_index*4 + 2];
  
  float max_log_odds = nvblox::logOddsFromProbability(0);

  for (int i = 0; i < num_mappers; i++)
  {
    const float block_size = block_sizes[i];
    const float voxel_size = block_size / kNumVoxelsPerBlock;

    // Get the correct block from the hash.
    nvblox::OccupancyVoxel* occupancy_voxel;
    if (nvblox::getVoxelAtPosition<nvblox::OccupancyVoxel>(hashes[i], query_location, block_size, &occupancy_voxel)) {
      if (occupancy_voxel->log_odds > max_log_odds) {
        max_log_odds = occupancy_voxel->log_odds;
      }
    }
  }
  out_log_odds[query_index] = max_log_odds;
}

} // namespace sdf
} // namespace nvblox
