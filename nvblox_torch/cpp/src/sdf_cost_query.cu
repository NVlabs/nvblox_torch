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
#include "nvblox_torch/helper_math.h"
#include "nvblox_torch/sdf_query.cuh"

#include <c10/cuda/CUDAStream.h>
#include <iostream>


namespace pynvblox {
  namespace sdf{
  namespace cost{
/**
 * @brief Compute length of sphere
 *
 * @param v1
 * @param v2
 * @return float
 */
__device__ __forceinline__ float sphere_length(const float4 &v1,
                                               const float4 &v2) {
  return norm3df(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

template<typename scalar_t>
__device__ __forceinline__ void load_layer_pose(const scalar_t *obb_mat,
float3 &position, float4 &quat)
{ // obb_mat has x,y,z, qw, qx, qy, qz, 0 with an extra 0 padding for better use of memory
  float4 temp = *(float4 *)&obb_mat[0];
  position.x = temp.x;
  position.y = temp.y;
  position.z = temp.z;
  quat.w = temp.w;
  temp = *(float4 *)&obb_mat[4];
  quat.x = temp.x;
  quat.y = temp.y;
  quat.z = temp.z;
}

__device__ __forceinline__ float sphere_distance(const float4 &v1,
                                                 const float4 &v2) {
  return max(0.0f, sphere_length(v1, v2) - v1.w - v2.w);
}

__device__ __forceinline__ void transform_sphere_quat(const float3 p, 
                      const float4 q,
                      const float4 &sphere_pos, 
                      float4 &C) {
  // do dot product:
  // new_p = q * p * q_inv + obs_p

  if(q.x!= 0 || q.y != 0 || q.z!=0)
  {

  C.x = p.x + q.w * q.w * sphere_pos.x + 2 * q.y * q.w * sphere_pos.z -
        2 * q.z * q.w * sphere_pos.y + q.x * q.x * sphere_pos.x +
        2 * q.y * q.x * sphere_pos.y + 2 * q.z * q.x * sphere_pos.z -
        q.z * q.z * sphere_pos.x - q.y * q.y * sphere_pos.x;
  C.y = p.y + 2 * q.x * q.y * sphere_pos.x + q.y * q.y * sphere_pos.y +
        2 * q.z * q.y * sphere_pos.z + 2 * q.w * q.z * sphere_pos.x -
        q.z * q.z * sphere_pos.y + q.w * q.w * sphere_pos.y - 2 * q.x * q.w * sphere_pos.z -
        q.x * q.x * sphere_pos.y;
  C.z = p.z + 2 * q.x * q.z * sphere_pos.x + 2 * q.y * q.z * sphere_pos.y +
        q.z * q.z * sphere_pos.z - 2 * q.w * q.y * sphere_pos.x - q.y * q.y * sphere_pos.z +
        2 * q.w * q.x * sphere_pos.y - q.x * q.x * sphere_pos.z + q.w * q.w * sphere_pos.z;
  }
  else
  {
    C.x = p.x + sphere_pos.x;
    C.y = p.y + sphere_pos.y;
    C.z = p.z + sphere_pos.z;
  }
  C.w = sphere_pos.w;
}


__device__ __forceinline__ void
inv_transform_vec_quat(
                      const float3 p, 
                      const float4 q,
                       const float4 &sphere_pos, float3 &C) {
  // do dot product:
  // new_p = q * p * q_inv + obs_p
  if(q.x != 0 || q.y!= 0 || q.z!=0)
  {
  C.x =  q.w *  q.w * sphere_pos.x - 2 * q.y *  q.w * sphere_pos.z +
        2 * q.z *  q.w * sphere_pos.y + q.x * q.x * sphere_pos.x +
        2 * q.y * q.x * sphere_pos.y + 2 * q.z * q.x * sphere_pos.z -
        q.z * q.z * sphere_pos.x - q.y * q.y * sphere_pos.x;
  C.y = 2 * q.x * q.y * sphere_pos.x + q.y * q.y * sphere_pos.y +
        2 * q.z * q.y * sphere_pos.z - 2 *  q.w * q.z * sphere_pos.x -
        q.z * q.z * sphere_pos.y +  q.w *  q.w * sphere_pos.y + 2 * q.x *  q.w * sphere_pos.z -
        q.x * q.x * sphere_pos.y;
  C.z = 2 * q.x * q.z * sphere_pos.x + 2 * q.y * q.z * sphere_pos.y +
        q.z * q.z * sphere_pos.z + 2 *  q.w * q.y * sphere_pos.x - q.y * q.y * sphere_pos.z -
        2 *  q.w * q.x * sphere_pos.y - q.x * q.x * sphere_pos.z +  q.w *  q.w * sphere_pos.z;
  }
  else
  {
    C.x = sphere_pos.x;
    C.y = sphere_pos.y;
    C.z = sphere_pos.z;
  }
}


__device__ __forceinline__ void
inv_transform_vec_quat_add(const float3 p, 
                           const float4 q, // x,y,z, qw, qx,qy,qz
                           const float4 &sphere_pos, float3 &C) {
  // do dot product:
  // new_p = q * p * q_inv + obs_p
  float3 temp_C = make_float3(0.0);
  inv_transform_vec_quat(p, q, sphere_pos, temp_C);
  C = C + temp_C;
}


__device__ __forceinline__ void scale_eta_metric(const float4 &sphere, const float4 &cl_pt, 
const float eta,
float4 &sum_pt,
const float sign)
{
    // compute distance:
    float sph_dist = 0;
    sph_dist = sphere_length(sphere, cl_pt);
    
    if (sph_dist == 0)
    {
      sum_pt.x = 0;
      sum_pt.y = 0;
      sum_pt.z = 0;
      sum_pt.w = sphere.w;
      
      return;
    }
    sum_pt.x = sign*(sphere.x - cl_pt.x) / sph_dist;
    sum_pt.y = sign*(sphere.y - cl_pt.y) / sph_dist;
    sum_pt.z = sign*(sphere.z - cl_pt.z) / sph_dist;
    if (eta > 0.0)
    {


    if (sph_dist > eta) {
      sum_pt.w = sph_dist - 0.5 * eta;

    } else if (sph_dist <= eta) {
      
      sum_pt.w = (0.5 / eta) * (sph_dist) * (sph_dist);
      const float scale = (1 / eta) * (sph_dist);
      sum_pt.x = scale * sum_pt.x;
      sum_pt.y = scale * sum_pt.y;
      sum_pt.z = scale * sum_pt.z;
    }

    }
    else
    {
      sum_pt.w = sph_dist;
    }
}


__device__ __forceinline__ void scale_eta_metric_vector(const float4 &delta_vector,
const float distance, 
const float eta,
float4 &sum_pt,
const float sign)
{
    // compute distance:
    float sph_dist = distance;
    
    if (sph_dist == delta_vector.w)
    {
      // when sphere is intersecting with a voxel, there is no gradient information, we 
      // hence add a random noise as the gradient
      sum_pt.x = delta_vector.x;
      sum_pt.y = delta_vector.y;
      sum_pt.z = delta_vector.z;
      sum_pt.w = delta_vector.w;
      
      return;
    }
    sum_pt.x = -1.0 * sign *(delta_vector.x);
    sum_pt.y = -1.0 * sign *(delta_vector.y);
    sum_pt.z = -1.0 * sign *(delta_vector.z);
    if (eta > 0.0)
    {


    if (sph_dist > eta) {
      sum_pt.w = sph_dist - 0.5 * eta;

    } else if (sph_dist <= eta) {
      
      sum_pt.w = (0.5 / eta) * (sph_dist) * (sph_dist);
      const float scale = (1 / eta) * (sph_dist);
      sum_pt.x = scale * sum_pt.x;
      sum_pt.y = scale * sum_pt.y;
      sum_pt.z = scale * sum_pt.z;
    }

    }
    else
    {
      sum_pt.w = sph_dist;
    }
}


__device__ __forceinline__ float get_distance( 
    const nvblox::Index3DDeviceHashMapType<nvblox::EsdfBlock> hash,
    const float block_size,
    const float voxel_size,
    const nvblox::Vector3f &query_location)
    {
    float current_distance = 0;
    nvblox::EsdfVoxel* esdf_voxel;

    if (!nvblox::getVoxelAtPosition<nvblox::EsdfVoxel>(hash, query_location, block_size,
                                      &esdf_voxel) ||
        !esdf_voxel->observed) {
      // This voxel is outside of the map or not observed. 
      current_distance = 0.0;
    } 
    else
    {
      current_distance = -1.0 * voxel_size * sqrt(esdf_voxel->squared_distance_vox);
 
      if (esdf_voxel->is_inside) {
	    	current_distance  = -1 * current_distance; 
        }
    }
    return current_distance;
    }
__device__ __forceinline__ void
compute_fd_gradient(
    const nvblox::Index3DDeviceHashMapType<nvblox::EsdfBlock> hash,
    float4 &cl_pt,
    const float voxel_size,
    const float4 loc_sphere,
    const float block_size)
    {

    // use finite difference to compute gradient:
    const float eps =  block_size;
    nvblox::Vector3f query_location;
    query_location(0) = loc_sphere.x + eps;
    query_location(1) = loc_sphere.y;
    query_location(2) = loc_sphere.z;
    float d1, d0;
    d1 = get_distance(hash,  block_size, voxel_size, query_location);
    query_location(0) = loc_sphere.x - eps;
    d0 = get_distance(hash,  block_size, voxel_size, query_location);

    cl_pt.x =  (d1 - d0) / (2 * eps);

    query_location(0) = loc_sphere.x ;
    query_location(1) = loc_sphere.y + eps;
    query_location(2) = loc_sphere.z;
    
    d1 = get_distance(hash,  block_size, voxel_size, query_location);
    query_location(1) = loc_sphere.y - eps;
    d0 = get_distance(hash,  block_size, voxel_size, query_location);

    cl_pt.y =(d1 - d0) / (2 * eps);

    query_location(0) = loc_sphere.x ;
    query_location(1) = loc_sphere.y;
    query_location(2) = loc_sphere.z + eps;
    
    d1 = get_distance(hash,  block_size, voxel_size, query_location);
    query_location(2) = loc_sphere.z - eps;
    d0 = get_distance(hash,  block_size, voxel_size, query_location);

    cl_pt.z = (d1 - d0) / (2* eps);
    if (isnan(cl_pt.z))
    {
      cl_pt.z = 0.0;
    }
    if (isnan(cl_pt.x))
    {
      cl_pt.x = 0.0;
    }
    if (isnan(cl_pt.y))
    {
      cl_pt.y = 0.0;
    }
    
    if (cl_pt.z == 0 && cl_pt.x == 0 && cl_pt.y == 0)
    {
      cl_pt.x = -0.001;
      cl_pt.z = -0.001;
    }
    }

__device__ __forceinline__ void
compute_voxel_distance_grad(
  nvblox::EsdfVoxel* esdf_voxel,
  float4 &sum_grad,
  float &signed_distance,
  const float4 &loc_sphere,
  const float &eta,
  const float voxel_size,
  const nvblox::Index3DDeviceHashMapType<nvblox::EsdfBlock> hash,
  const float block_size

)
{
  nvblox::Vector3f closest_pt;

  float sign = -1.0;
  float distance = 0.0;
  // Get the distance of the relevant voxel.
  distance = voxel_size * sqrt(esdf_voxel->squared_distance_vox);

    // If it's outside, we set the value to be negative
  if (esdf_voxel->is_inside) {
		sign = 1.0;  
  }
    
  distance = (distance * sign) + loc_sphere.w;// + loc_sphere.w;
  if (distance > 0.0)
  {
      //grad_pt = -1.0 * ( sign * voxel_size * esdf_voxel->parent_direction.cast<float>());

      closest_pt = voxel_size * esdf_voxel->parent_direction.cast<float>();
      // eta scaling here:


      float4 cl_pt = make_float4(closest_pt(0), closest_pt(1), closest_pt(2), 0.0);

     if (false)
     {

     

      if (distance != loc_sphere.w)
      {
      cl_pt = ((distance)  / norm3df(cl_pt.x, cl_pt.y, cl_pt.z)) * cl_pt;
      }

      

      cl_pt += loc_sphere;
      
      cl_pt.w = 0.0;

      scale_eta_metric(loc_sphere, cl_pt, eta, sum_grad, sign);
     }
     if (true)
     {
      float scale = norm3df(cl_pt.x, cl_pt.y, cl_pt.z);
      if (scale > 0.0)
      {
      cl_pt = (1.0  / scale ) * cl_pt;
      }
      else
      {
        // do finite difference to compute gradient:
        compute_fd_gradient(hash, cl_pt, voxel_size, loc_sphere, block_size);

      }
      cl_pt.w = loc_sphere.w;
      scale_eta_metric_vector(cl_pt, distance, eta, sum_grad, sign);
      
     }
      //sum_grad.x = cl_pt.x;
      //sum_grad.y = cl_pt.y;
      //sum_grad.z = cl_pt.z;
      
      //sum_grad.w = distance;      
      //sum_grad.x = sphere_length(loc_sphere, cl_pt);
      //sum_grad.y = 0.0;
      //sum_grad.z = 0.0; 

  }
  signed_distance = distance;

}

/**
 * @brief Scales the Collision across the trajectory by sphere velocity. This is
 * implemented from CHOMP motion planner (ICRA 2009). We use central difference
 * to compute the velocity and acceleration of the sphere.
 *
 * @param sphere_0_cache
 * @param sphere_1_cache
 * @param sphere_2_cache
 * @param dt
 * @param transform_back
 * @param max_dist
 * @param max_grad
 * @return void
 */
__device__ __forceinline__ void
scale_speed_metric(const float4 &sphere_0_cache, const float4 &sphere_1_cache,
                   const float4 &sphere_2_cache, const float &dt,
                   const bool &transform_back, float &max_dist,
                   float3 &max_grad) {

  float3 norm_vel_vec = make_float3(sphere_2_cache.x - sphere_0_cache.x,
                                    sphere_2_cache.y - sphere_0_cache.y,
                                    sphere_2_cache.z - sphere_0_cache.z);

  norm_vel_vec = (0.5 / dt) * norm_vel_vec;
  const float sph_vel = length(norm_vel_vec);

  if (transform_back) {

    float3 sph_acc_vec = make_float3(
        sphere_0_cache.x + sphere_2_cache.x - 2.0 * sphere_1_cache.x,
        sphere_0_cache.y + sphere_2_cache.y - 2.0 * sphere_1_cache.y,
        sphere_0_cache.z + sphere_2_cache.z - 2.0 * sphere_1_cache.z);

    sph_acc_vec = (1.0 / (dt * dt)) * sph_acc_vec;
    norm_vel_vec = norm_vel_vec * (1.0 / sph_vel);

    const float3 curvature_vec = (sph_acc_vec) / (sph_vel * sph_vel);

    // compute orthogonal projection:
    float orth_proj[9] = {0.0};

    // load float3 into array for easier matmul later:
    float vel_arr[3];
    vel_arr[0] = norm_vel_vec.x;
    vel_arr[1] = norm_vel_vec.y;
    vel_arr[2] = norm_vel_vec.z;

// calculate projection ( I - (v * v^T)):
#pragma unroll 3
    for (int i = 0; i < 3; i++) {
#pragma unroll 3
      for (int j = 0; j < 3; j++) {
        orth_proj[i * 3 + j] = -1 * vel_arr[i] * vel_arr[j];
      }
    }
    orth_proj[0] += 1;
    orth_proj[4] += 1;
    orth_proj[8] += 1;

    // load float3 into array for easier matmul later
    // two matmuls:
    float orth_pt[3];; // orth_proj(3x3) * max_grad(3x1)
    float orth_curve[3]; // max_dist(1) * orth_proj (3x3) * curvature_vec (3x1)

#pragma unroll 3
    for (int i = 0; i < 3; i++) // matrix - vector product
    {
      orth_pt[i] = orth_proj[i * 3 + 0] * max_grad.x +
                    orth_proj[i * 3 + 1] * max_grad.y +
                    orth_proj[i * 3 + 2] * max_grad.z;

      orth_curve[i] = max_dist * (orth_proj[i * 3 + 0] * curvature_vec.x +
                                   orth_proj[i * 3 + 1] * curvature_vec.y +
                                   orth_proj[i * 3 + 2] * curvature_vec.z);
    }

    // max_grad =  sph_vel * ((orth_proj * max_grad) - max_dist *  orth_proj *
    // curvature)

    max_grad.x = sph_vel * (orth_pt[0] - orth_curve[0]);
    max_grad.y = sph_vel * (orth_pt[1] - orth_curve[1]);
    max_grad.z = sph_vel * (orth_pt[2] - orth_curve[2]);
  }
  max_dist = sph_vel * max_dist;
}


__global__ void sphereDistanceCostMultiKernel(
	const float *sphere_pos_rad, 
	float *out_distance,
	float *out_grad,
	uint8_t *sparsity_idx,
	const float *weight,
	const float *activation_distance,
	nvblox::Index3DDeviceHashMapType<nvblox::EsdfBlock>* hashes,
  const float *blox_pose,// pose of robot w.r.t nvblox world  origin w_T_rbase
  const uint8_t *blox_enable,
  const float *block_sizes,
	const int batch_size, const int horizon, const int nspheres, 
	const bool write_grad,
  const int num_mappers
)
  {
  const int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int b_idx = t_idx / (horizon * nspheres);
  const int h_idx = (t_idx - b_idx * (horizon * nspheres)) / nspheres;
  const int lsph_idx = (t_idx - b_idx * horizon * nspheres - h_idx * nspheres);
  if (lsph_idx >= nspheres || b_idx >= batch_size || h_idx >= horizon) {
    return;
  }
  const int sph_idx =
      b_idx * horizon * nspheres + h_idx * nspheres + lsph_idx;

  const float eta = activation_distance[0];
  
  // Load sphere_pos_rad input
  float4 sphere_cache = *(float4 *)&sphere_pos_rad[sph_idx * 4];
  if (sphere_cache.w < 0.0) {
    return;
  }
  sphere_cache.w += eta;

  float4 loc_sphere = make_float4(0.0, 0.0, 0.0, 0.0);

  

  constexpr int kNumVoxelsPerBlock = 8;
  
  nvblox::Vector3f query_location;

  // read data into vector3f:
  float radius = sphere_cache.w;

  float3 global_grad = make_float3(0,0,0);

  float max_distance = 0.0f;
  // Get the correct block from the hash.
  nvblox::EsdfVoxel* esdf_voxel;

  float4 obb_quat = make_float4(0.0);
  float3 obb_pos = make_float3(0.0);
  
  float signed_distance = 0.0;
  for (int i = 0; i < num_mappers; i++)
  {
    if (blox_enable[i] == 0)
    {
      continue;
    }
    signed_distance =0.0;
  load_layer_pose(&blox_pose[i*8], obb_pos, obb_quat);
  // transform sphere to nvblox base frame:
  transform_sphere_quat(obb_pos, obb_quat, sphere_cache,
                          loc_sphere);

  query_location(0) = loc_sphere.x;
  query_location(1) = loc_sphere.y;
  query_location(2) = loc_sphere.z;
  float4 sum_grad = make_float4(0.0,0.0,0.0,0.0);

  const float block_size = block_sizes[i];
  const float voxel_size = block_size / kNumVoxelsPerBlock;
  if (!nvblox::getVoxelAtPosition<nvblox::EsdfVoxel>(hashes[i], query_location, block_size,
                                     &esdf_voxel) ||
      !esdf_voxel->observed) {
    // This voxel is outside of the map or not observed. Mark it as 100 meters
    // behind a surface.
  } else {

     float4 sum_grad = make_float4(0.0,0.0,0.0,0.0);

    compute_voxel_distance_grad(
      esdf_voxel, sum_grad, 
    signed_distance,
    loc_sphere, eta, voxel_size, hashes[i], block_size);
    max_distance += sum_grad.w;

    if(sum_grad.w > 0.0)
    { 
      inv_transform_vec_quat_add(obb_pos, obb_quat,sum_grad, global_grad);
    }
    
  
  }
  
  }


  // sparsity opt:
  if (max_distance == 0) {
    if (sparsity_idx[sph_idx] == 0) {
      return;
    }
    sparsity_idx[sph_idx] = 0;
    if (write_grad) {
      *(float3 *)&out_grad[sph_idx * 4] = global_grad; // max_grad is all zeros
    }
    out_distance[sph_idx] = 0.0;
    return;
  }
  // else max_dist != 0
  max_distance = weight[0] * max_distance;

  if (write_grad) {
    *(float3 *)&out_grad[sph_idx * 4] = weight[0] * global_grad;
  }
  out_distance[sph_idx] = max_distance;
  sparsity_idx[sph_idx] = 1;

}

__global__ void sphereDistanceCostMultiKernel_map1(
	const float *sphere_pos_rad, 
	float *out_distance,
	float *out_grad,
	uint8_t *sparsity_idx,
	const float *weight,
	const float *activation_distance,
	nvblox::Index3DDeviceHashMapType<nvblox::EsdfBlock>* hashes,
  const float *blox_pose,// pose of robot w.r.t nvblox world  origin w_T_rbase
  const uint8_t *blox_enable,
  const float *block_sizes,
	const int batch_size, const int horizon, const int nspheres, 
	const bool write_grad
)
  {
    const int num_mappers = 1;

  const int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int b_idx = t_idx / (horizon * nspheres);
  const int h_idx = (t_idx - b_idx * (horizon * nspheres)) / nspheres;
  const int lsph_idx = (t_idx - b_idx * horizon * nspheres - h_idx * nspheres);
  if (lsph_idx >= nspheres || b_idx >= batch_size || h_idx >= horizon) {
    return;
  }
  const int sph_idx =
      b_idx * horizon * nspheres + h_idx * nspheres + lsph_idx;

  const float eta = activation_distance[0];
  
  // Load sphere_pos_rad input
  float4 sphere_cache = *(float4 *)&sphere_pos_rad[sph_idx * 4];
  if (sphere_cache.w < 0.0) {
    return;
  }
  sphere_cache.w += eta;

  float4 loc_sphere = make_float4(0.0, 0.0, 0.0, 0.0);

  

  constexpr int kNumVoxelsPerBlock = 8;
  
  nvblox::Vector3f query_location;

  // read data into vector3f:
  float radius = sphere_cache.w;

  float3 global_grad = make_float3(0,0,0);

  float max_distance = 0.0f;
  // Get the correct block from the hash.
  nvblox::EsdfVoxel* esdf_voxel;

  float4 obb_quat = make_float4(0.0);
  float3 obb_pos = make_float3(0.0);
  
  float signed_distance = 0.0;

  #pragma unroll 1
  for (int i = 0; i < num_mappers; i++)
  {
    if (blox_enable[i] == 0)
    {
      continue;
    }
    signed_distance =0.0;
  load_layer_pose(&blox_pose[i*8], obb_pos, obb_quat);
  // transform sphere to nvblox base frame:
  transform_sphere_quat(obb_pos, obb_quat, sphere_cache,
                          loc_sphere);

  query_location(0) = loc_sphere.x;
  query_location(1) = loc_sphere.y;
  query_location(2) = loc_sphere.z;
  float4 sum_grad = make_float4(0.0,0.0,0.0,0.0);

  const float block_size = block_sizes[i];
  const float voxel_size = block_size / kNumVoxelsPerBlock;
  if (!nvblox::getVoxelAtPosition<nvblox::EsdfVoxel>(hashes[i], query_location, block_size,
                                     &esdf_voxel) ||
      !esdf_voxel->observed) {
    // This voxel is outside of the map or not observed. Mark it as 100 meters
    // behind a surface.
  } else {

     float4 sum_grad = make_float4(0.0,0.0,0.0,0.0);

    compute_voxel_distance_grad(
      esdf_voxel, sum_grad, 
    signed_distance,
    loc_sphere, eta, voxel_size, hashes[i], block_size);
    max_distance += sum_grad.w;

    if(sum_grad.w > 0.0)
    { 
      inv_transform_vec_quat_add(obb_pos, obb_quat,sum_grad, global_grad);
    }
    
  
  }
  
  }


  // sparsity opt:
  if (max_distance == 0) {
    if (sparsity_idx[sph_idx] == 0) {
      return;
    }
    sparsity_idx[sph_idx] = 0;
    if (write_grad) {
      *(float3 *)&out_grad[sph_idx * 4] = global_grad; // max_grad is all zeros
    }
    out_distance[sph_idx] = 0.0;
    return;
  }
  // else max_dist != 0
  max_distance = weight[0] * max_distance;

  if (write_grad) {
    *(float3 *)&out_grad[sph_idx * 4] = weight[0] * global_grad;
  }
  out_distance[sph_idx] = max_distance;
  sparsity_idx[sph_idx] = 1;

}

__global__ void sphereDistanceCostMultiKernel_map2(
	const float *sphere_pos_rad, 
	float *out_distance,
	float *out_grad,
	uint8_t *sparsity_idx,
	const float *weight,
	const float *activation_distance,
	nvblox::Index3DDeviceHashMapType<nvblox::EsdfBlock>* hashes,
  const float *blox_pose,// pose of robot w.r.t nvblox world  origin w_T_rbase
  const uint8_t *blox_enable,
  const float *block_sizes,
	const int batch_size, const int horizon, const int nspheres, 
	const bool write_grad
)
  {
    const int num_mappers = 2;

  const int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int b_idx = t_idx / (horizon * nspheres);
  const int h_idx = (t_idx - b_idx * (horizon * nspheres)) / nspheres;
  const int lsph_idx = (t_idx - b_idx * horizon * nspheres - h_idx * nspheres);
  if (lsph_idx >= nspheres || b_idx >= batch_size || h_idx >= horizon) {
    return;
  }
  const int sph_idx =
      b_idx * horizon * nspheres + h_idx * nspheres + lsph_idx;

  const float eta = activation_distance[0];
  
  // Load sphere_pos_rad input
  float4 sphere_cache = *(float4 *)&sphere_pos_rad[sph_idx * 4];
  if (sphere_cache.w < 0.0) {
    return;
  }
  sphere_cache.w += eta;

  float4 loc_sphere = make_float4(0.0, 0.0, 0.0, 0.0);

  

  constexpr int kNumVoxelsPerBlock = 8;
  
  nvblox::Vector3f query_location;

  // read data into vector3f:
  float radius = sphere_cache.w;

  float3 global_grad = make_float3(0,0,0);

  float max_distance = 0.0f;
  // Get the correct block from the hash.
  nvblox::EsdfVoxel* esdf_voxel;

  float4 obb_quat = make_float4(0.0);
  float3 obb_pos = make_float3(0.0);
  
  float signed_distance = 0.0;

  #pragma unroll 2
  for (int i = 0; i < num_mappers; i++)
  {
    if (blox_enable[i] == 0)
    {
      continue;
    }
    signed_distance =0.0;
  load_layer_pose(&blox_pose[i*8], obb_pos, obb_quat);
  // transform sphere to nvblox base frame:
  transform_sphere_quat(obb_pos, obb_quat, sphere_cache,
                          loc_sphere);

  query_location(0) = loc_sphere.x;
  query_location(1) = loc_sphere.y;
  query_location(2) = loc_sphere.z;
  float4 sum_grad = make_float4(0.0,0.0,0.0,0.0);

  const float block_size = block_sizes[i];
  const float voxel_size = block_size / kNumVoxelsPerBlock;
  if (!nvblox::getVoxelAtPosition<nvblox::EsdfVoxel>(hashes[i], query_location, block_size,
                                     &esdf_voxel) ||
      !esdf_voxel->observed) {
    // This voxel is outside of the map or not observed. Mark it as 100 meters
    // behind a surface.
  } else {

     float4 sum_grad = make_float4(0.0,0.0,0.0,0.0);

    compute_voxel_distance_grad(
      esdf_voxel, sum_grad, 
    signed_distance,
    loc_sphere, eta, voxel_size, hashes[i], block_size);
    max_distance += sum_grad.w;

    if(sum_grad.w > 0.0)
    { 
      inv_transform_vec_quat_add(obb_pos, obb_quat,sum_grad, global_grad);
    }
    
  
  }
  
  }


  // sparsity opt:
  if (max_distance == 0) {
    if (sparsity_idx[sph_idx] == 0) {
      return;
    }
    sparsity_idx[sph_idx] = 0;
    if (write_grad) {
      *(float3 *)&out_grad[sph_idx * 4] = global_grad; // max_grad is all zeros
    }
    out_distance[sph_idx] = 0.0;
    return;
  }
  // else max_dist != 0
  max_distance = weight[0] * max_distance;

  if (write_grad) {
    *(float3 *)&out_grad[sph_idx * 4] = weight[0] * global_grad;
  }
  out_distance[sph_idx] = max_distance;
  sparsity_idx[sph_idx] = 1;

}
__global__ void sphereTrajectoryDistanceCostMultiKernel(
	const float *sphere_pos_rad, 
	float *out_distance,
	float *out_grad,
	uint8_t *sparsity_idx,
	const float *weight,
	const float *activation_distance,
	const float *speed_dt,
	nvblox::Index3DDeviceHashMapType<nvblox::EsdfBlock>* hashes,
  const float *blox_pose,// pose of robot w.r.t nvblox world  origin w_T_rbase
  const uint8_t *blox_enable,
  const float *block_sizes,
	const int batch_size, const int horizon, const int nspheres,
	const int sweep_steps, const bool enable_speed_metric, 
	const bool write_grad,
  const int num_mappers)
{
  const int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int b_idx = t_idx / (horizon * nspheres);
  const int h_idx = (t_idx - b_idx * (horizon * nspheres)) / nspheres;
  const int lsph_idx = (t_idx - b_idx * horizon * nspheres - h_idx * nspheres);
  if (lsph_idx >= nspheres || b_idx >= batch_size || h_idx >= horizon) {
    return;
  }
  const int sph_idx =
      b_idx * horizon * nspheres + h_idx * nspheres + lsph_idx;

  const float eta = activation_distance[0];
  
  // Load sphere_pos_rad input
  float4 sphere_cache = *(float4 *)&sphere_pos_rad[sph_idx * 4];
  if (sphere_cache.w < 0.0) {
    return;
  }
  sphere_cache.w += eta;

  float4 loc_sphere = make_float4(0.0, 0.0, 0.0, 0.0);

  float radius = sphere_cache.w;


  constexpr int kNumVoxelsPerBlock = 8;
  
  nvblox::Vector3f query_location;

  // read data into vector3f:
  float max_distance = 0.0f;
  float signed_distance = 0.0f;
  
  float3 global_grad = make_float3(0,0,0);

  float4 obb_quat = make_float4(0.0);
  float3 obb_pos = make_float3(0.0);
  float current_distance = 0.0f;
  #pragma unroll
  for(int m = 0; m<num_mappers; m++)
  {
    if (blox_enable[m] == 0)
    {
      continue;
    }
    current_distance = 0.0;
    signed_distance = 0.0;
      load_layer_pose(&blox_pose[m*8], obb_pos, obb_quat);
    transform_sphere_quat(obb_pos, obb_quat, sphere_cache,
                          loc_sphere);

  // transform sphere to nvblox base frame:

  query_location(0) = loc_sphere.x;
  query_location(1) = loc_sphere.y;
  query_location(2) = loc_sphere.z;
  const float block_size = block_sizes[m];
  const float voxel_size = block_size / kNumVoxelsPerBlock;
  
  // Get the correct block from the hash.
  nvblox::EsdfVoxel* esdf_voxel;
  if (!nvblox::getVoxelAtPosition<nvblox::EsdfVoxel>(hashes[m], query_location, block_size,
                                     &esdf_voxel) ||
      !esdf_voxel->observed) {
    // This voxel is outside of the map or not observed. Mark it as 100 meters
    // behind a surface.
  } else {

    float4 sum_grad = make_float4(0.0,0.0,0.0,0.0);

    compute_voxel_distance_grad(esdf_voxel, sum_grad,
    signed_distance, 
    loc_sphere, eta, voxel_size, hashes[m], block_size);
    max_distance += sum_grad.w;
    if (sum_grad.w > 0.0)
    {
      inv_transform_vec_quat_add(obb_pos, obb_quat,sum_grad, global_grad);
    }
    }
 
    
  }

// sparsity opt:
if (max_distance == 0) {
    if (sparsity_idx[sph_idx] == 0) {
      return;
    }
    sparsity_idx[sph_idx] = 0;
    if (write_grad) {
      *(float3 *)&out_grad[sph_idx * 4] = global_grad; // max_grad is all zeros
    }
    out_distance[sph_idx] = 0.0;
    return;
  }
if(enable_speed_metric)
{


bool sweep_back = false;
bool sweep_fwd = false;
float4 sphere_0_cache, sphere_2_cache;

const int b_addrs =
      b_idx * horizon * nspheres; // + h_idx * n_spheres + sph_idx;

if (h_idx > 0) {
    sphere_0_cache =
        *(float4 *)&sphere_pos_rad[b_addrs * 4 + (h_idx - 1) * nspheres * 4 +
                                   sph_idx * 4];
    float sphere_0_distance = sphere_distance(sphere_0_cache, sphere_cache);
    if (sphere_0_distance > 0.0)
  {
    sweep_back = true;
  }
  }

  if (h_idx < horizon - 1) {
    sphere_2_cache =
        *(float4 *)&sphere_pos_rad[b_addrs * 4 + (h_idx + 1) * nspheres * 4 +
                                   sph_idx * 4];
    float sphere_2_distance = sphere_distance(sphere_2_cache, sphere_cache);
    if(sphere_2_distance>0.0)
    {

    
    sweep_fwd = true;
    }
  }
    
 if (sweep_back && sweep_fwd) {
    const float dt = speed_dt[0];

    scale_speed_metric(sphere_0_cache, sphere_cache, sphere_2_cache, dt,
                       write_grad, max_distance, global_grad);
  }

}

  // else max_dist != 0
  max_distance = weight[0] * max_distance;

  if (write_grad) {
    *(float3 *)&out_grad[sph_idx * 4] = weight[0] * global_grad;
  }
  out_distance[sph_idx] = max_distance;
  sparsity_idx[sph_idx] = 1;

}

__global__ void sphereTrajectoryDistanceCostMultiKernel_map1(
	const float *sphere_pos_rad, 
	float *out_distance,
	float *out_grad,
	uint8_t *sparsity_idx,
	const float *weight,
	const float *activation_distance,
	const float *speed_dt,
	nvblox::Index3DDeviceHashMapType<nvblox::EsdfBlock>* hashes,
  const float *blox_pose,// pose of robot w.r.t nvblox world  origin w_T_rbase
  const uint8_t *blox_enable,
  const float *block_sizes,
	const int batch_size, const int horizon, const int nspheres,
	const int sweep_steps, const bool enable_speed_metric, 
	const bool write_grad)
{
  const int num_mappers = 1;
  const int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int b_idx = t_idx / (horizon * nspheres);
  const int h_idx = (t_idx - b_idx * (horizon * nspheres)) / nspheres;
  const int lsph_idx = (t_idx - b_idx * horizon * nspheres - h_idx * nspheres);
  if (lsph_idx >= nspheres || b_idx >= batch_size || h_idx >= horizon) {
    return;
  }
  const int sph_idx =
      b_idx * horizon * nspheres + h_idx * nspheres + lsph_idx;

  const float eta = activation_distance[0];
  
  // Load sphere_pos_rad input
  float4 sphere_cache = *(float4 *)&sphere_pos_rad[sph_idx * 4];
  if (sphere_cache.w < 0.0) {
    return;
  }
  sphere_cache.w += eta;

  float4 loc_sphere = make_float4(0.0, 0.0, 0.0, 0.0);

  float radius = sphere_cache.w;


  constexpr int kNumVoxelsPerBlock = 8;
  
  nvblox::Vector3f query_location;

  // read data into vector3f:
  float max_distance = 0.0f;
  float signed_distance = 0.0f;
  
  float3 global_grad = make_float3(0,0,0);

  float4 obb_quat = make_float4(0.0);
  float3 obb_pos = make_float3(0.0);
  float current_distance = 0.0f;
  #pragma unroll 1
  for(int m = 0; m<num_mappers; m++)
  {
    if (blox_enable[m] == 0)
    {
      continue;
    }
    current_distance = 0.0;
    signed_distance = 0.0;
      load_layer_pose(&blox_pose[m*8], obb_pos, obb_quat);
    transform_sphere_quat(obb_pos, obb_quat, sphere_cache,
                          loc_sphere);

  // transform sphere to nvblox base frame:

  query_location(0) = loc_sphere.x;
  query_location(1) = loc_sphere.y;
  query_location(2) = loc_sphere.z;
  const float block_size = block_sizes[m];
  const float voxel_size = block_size / kNumVoxelsPerBlock;
  
  // Get the correct block from the hash.
  nvblox::EsdfVoxel* esdf_voxel;
  if (!nvblox::getVoxelAtPosition<nvblox::EsdfVoxel>(hashes[m], query_location, block_size,
                                     &esdf_voxel) ||
      !esdf_voxel->observed) {
    // This voxel is outside of the map or not observed. Mark it as 100 meters
    // behind a surface.
  } else {

    float4 sum_grad = make_float4(0.0,0.0,0.0,0.0);

    compute_voxel_distance_grad(esdf_voxel, sum_grad,
    signed_distance, 
    loc_sphere, eta, voxel_size, hashes[m], block_size);
    max_distance += sum_grad.w;
    if (sum_grad.w > 0.0)
    {
      inv_transform_vec_quat_add(obb_pos, obb_quat,sum_grad, global_grad);
    }
    }
 
    
  }

// sparsity opt:
if (max_distance == 0) {
    if (sparsity_idx[sph_idx] == 0) {
      return;
    }
    sparsity_idx[sph_idx] = 0;
    if (write_grad) {
      *(float3 *)&out_grad[sph_idx * 4] = global_grad; // max_grad is all zeros
    }
    out_distance[sph_idx] = 0.0;
    return;
  }
if(enable_speed_metric)
{


bool sweep_back = false;
bool sweep_fwd = false;
float4 sphere_0_cache, sphere_2_cache;

const int b_addrs =
      b_idx * horizon * nspheres; // + h_idx * n_spheres + sph_idx;

if (h_idx > 0) {
    sphere_0_cache =
        *(float4 *)&sphere_pos_rad[b_addrs * 4 + (h_idx - 1) * nspheres * 4 +
                                   sph_idx * 4];
    float sphere_0_distance = sphere_distance(sphere_0_cache, sphere_cache);
    if (sphere_0_distance > 0.0)
  {
    sweep_back = true;
  }
  }

  if (h_idx < horizon - 1) {
    sphere_2_cache =
        *(float4 *)&sphere_pos_rad[b_addrs * 4 + (h_idx + 1) * nspheres * 4 +
                                   sph_idx * 4];
    float sphere_2_distance = sphere_distance(sphere_2_cache, sphere_cache);
    if(sphere_2_distance>0.0)
    {

    
    sweep_fwd = true;
    }
  }
    
 if (sweep_back && sweep_fwd) {
    const float dt = speed_dt[0];

    scale_speed_metric(sphere_0_cache, sphere_cache, sphere_2_cache, dt,
                       write_grad, max_distance, global_grad);
  }

}

  // else max_dist != 0
  max_distance = weight[0] * max_distance;

  if (write_grad) {
    *(float3 *)&out_grad[sph_idx * 4] = weight[0] * global_grad;
  }
  out_distance[sph_idx] = max_distance;
  sparsity_idx[sph_idx] = 1;

}
__global__ void sphereTrajectoryDistanceCostMultiKernel_map2(
	const float *sphere_pos_rad, 
	float *out_distance,
	float *out_grad,
	uint8_t *sparsity_idx,
	const float *weight,
	const float *activation_distance,
	const float *speed_dt,
	nvblox::Index3DDeviceHashMapType<nvblox::EsdfBlock>* hashes,
  const float *blox_pose,// pose of robot w.r.t nvblox world  origin w_T_rbase
  const uint8_t *blox_enable,
  const float *block_sizes,
	const int batch_size, const int horizon, const int nspheres,
	const int sweep_steps, const bool enable_speed_metric, 
	const bool write_grad
  )
{
  const int num_mappers = 2;
  const int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int b_idx = t_idx / (horizon * nspheres);
  const int h_idx = (t_idx - b_idx * (horizon * nspheres)) / nspheres;
  const int lsph_idx = (t_idx - b_idx * horizon * nspheres - h_idx * nspheres);
  if (lsph_idx >= nspheres || b_idx >= batch_size || h_idx >= horizon) {
    return;
  }
  const int sph_idx =
      b_idx * horizon * nspheres + h_idx * nspheres + lsph_idx;

  const float eta = activation_distance[0];
  
  // Load sphere_pos_rad input
  float4 sphere_cache = *(float4 *)&sphere_pos_rad[sph_idx * 4];
  if (sphere_cache.w < 0.0) {
    return;
  }
  sphere_cache.w += eta;

  float4 loc_sphere = make_float4(0.0, 0.0, 0.0, 0.0);

  float radius = sphere_cache.w;


  constexpr int kNumVoxelsPerBlock = 8;
  
  nvblox::Vector3f query_location;

  // read data into vector3f:
  float max_distance = 0.0f;
  float signed_distance = 0.0f;
  
  float3 global_grad = make_float3(0,0,0);

  float4 obb_quat = make_float4(0.0);
  float3 obb_pos = make_float3(0.0);
  float current_distance = 0.0f;
  #pragma unroll 2
  for(int m = 0; m<num_mappers; m++)
  {
    if (blox_enable[m] == 0)
    {
      continue;
    }
    current_distance = 0.0;
    signed_distance = 0.0;
      load_layer_pose(&blox_pose[m*8], obb_pos, obb_quat);
    transform_sphere_quat(obb_pos, obb_quat, sphere_cache,
                          loc_sphere);

  // transform sphere to nvblox base frame:

  query_location(0) = loc_sphere.x;
  query_location(1) = loc_sphere.y;
  query_location(2) = loc_sphere.z;
  const float block_size = block_sizes[m];
  const float voxel_size = block_size / kNumVoxelsPerBlock;
  
  // Get the correct block from the hash.
  nvblox::EsdfVoxel* esdf_voxel;
  if (!nvblox::getVoxelAtPosition<nvblox::EsdfVoxel>(hashes[m], query_location, block_size,
                                     &esdf_voxel) ||
      !esdf_voxel->observed) {
    // This voxel is outside of the map or not observed. Mark it as 100 meters
    // behind a surface.
  } else {

    float4 sum_grad = make_float4(0.0,0.0,0.0,0.0);

    compute_voxel_distance_grad(esdf_voxel, sum_grad,
    signed_distance, 
    loc_sphere, eta, voxel_size, hashes[m], block_size);
    max_distance += sum_grad.w;
    if (sum_grad.w > 0.0)
    {
      inv_transform_vec_quat_add(obb_pos, obb_quat,sum_grad, global_grad);
    }
    }
 
    
  }

  // sparsity opt:
  if (max_distance == 0) {
    if (sparsity_idx[sph_idx] == 0) {
      return;
    }
    sparsity_idx[sph_idx] = 0;
    if (write_grad) {
      *(float3 *)&out_grad[sph_idx * 4] = global_grad; // max_grad is all zeros
    }
    out_distance[sph_idx] = 0.0;
    return;
  }
if(enable_speed_metric)
{


bool sweep_back = false;
bool sweep_fwd = false;
float4 sphere_0_cache, sphere_2_cache;

const int b_addrs =
      b_idx * horizon * nspheres; // + h_idx * n_spheres + sph_idx;

if (h_idx > 0) {
    sphere_0_cache =
        *(float4 *)&sphere_pos_rad[b_addrs * 4 + (h_idx - 1) * nspheres * 4 +
                                   sph_idx * 4];
    float sphere_0_distance = sphere_distance(sphere_0_cache, sphere_cache);
    if (sphere_0_distance > 0.0)
  {
    sweep_back = true;
  }
  }

  if (h_idx < horizon - 1) {
    sphere_2_cache =
        *(float4 *)&sphere_pos_rad[b_addrs * 4 + (h_idx + 1) * nspheres * 4 +
                                   sph_idx * 4];
    float sphere_2_distance = sphere_distance(sphere_2_cache, sphere_cache);
    if(sphere_2_distance>0.0)
    {

    
    sweep_fwd = true;
    }
  }
    
 if (sweep_back && sweep_fwd) {
    const float dt = speed_dt[0];

    scale_speed_metric(sphere_0_cache, sphere_cache, sphere_2_cache, dt,
                       write_grad, max_distance, global_grad);
  }

}

  // else max_dist != 0
  max_distance = weight[0] * max_distance;

  if (write_grad) {
    *(float3 *)&out_grad[sph_idx * 4] = weight[0] * global_grad;
  }
  out_distance[sph_idx] = max_distance;
  sparsity_idx[sph_idx] = 1;

}

__global__ void sphereSweptTrajectoryDistanceCostMultiKernel(
	const float *sphere_pos_rad, 
	float *out_distance,
	float *out_grad,
	uint8_t *sparsity_idx,
	const float *weight,
	const float *activation_distance,
	const float *speed_dt,
	nvblox::Index3DDeviceHashMapType<nvblox::EsdfBlock>* hashes,
  const float *blox_pose,// pose of robot w.r.t nvblox world  origin w_T_rbase
  const uint8_t *blox_enable,
  const float *block_sizes,
	const int batch_size, const int horizon, const int nspheres,
	const int sweep_steps, const bool enable_speed_metric, 
	const bool write_grad,
  const int num_mappers)
{
  const int t_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int b_idx = t_idx / (horizon * nspheres);
  const int h_idx = (t_idx - b_idx * (horizon * nspheres)) / nspheres;
  const int lsph_idx = (t_idx - b_idx * horizon * nspheres - h_idx * nspheres);
  if (lsph_idx >= nspheres || b_idx >= batch_size || h_idx >= horizon) {
    return;
  }
  const int sph_idx =
      b_idx * horizon * nspheres + h_idx * nspheres + lsph_idx;

  const float eta = activation_distance[0];
  
  // Load sphere_pos_rad input
  float4 sphere_1_cache = *(float4 *)&sphere_pos_rad[sph_idx * 4];
  if (sphere_1_cache.w < 0.0) {
    return;
  }
  sphere_1_cache.w += eta;

  const int sw_steps = sweep_steps;
  const float fl_sw_steps = sw_steps;

  bool sweep_back = false;
  bool sweep_fwd = false;
  float4 sphere_0_cache, sphere_2_cache;
  float signed_distance = 0.0;
  float sphere_0_distance, sphere_2_distance, sphere_0_len, sphere_2_len;

  const int b_addrs =
        b_idx * horizon * nspheres; 

  if (h_idx > 0) 
  {
    sphere_0_cache =
        *(float4 *)&sphere_pos_rad[b_addrs * 4 + (h_idx - 1) * nspheres * 4 +
                                   sph_idx * 4];
    sphere_0_cache.w += eta;
    sphere_0_distance = sphere_distance(sphere_0_cache, sphere_1_cache);
    sphere_0_len = sphere_0_distance + sphere_0_cache.w * 2;
    if (sphere_0_distance > 0.0) {
      sweep_back = true;
    }
  }

  if (h_idx < horizon - 1) 
  {
    sphere_2_cache =
        *(float4 *)&sphere_pos_rad[b_addrs * 4 + (h_idx + 1) * nspheres * 4 +
                                   sph_idx * 4];
    sphere_2_cache.w += eta;
    sphere_2_distance = sphere_distance(sphere_2_cache, sphere_1_cache);
    sphere_2_len = sphere_2_distance + sphere_2_cache.w * 2;
    if (sphere_2_distance > 0.0) {
      sweep_fwd = true;
    }
  }
    

  float4 loc_sphere = make_float4(0.0, 0.0, 0.0, 0.0);

  float4 loc_sphere_0, loc_sphere_1, loc_sphere_2;
  float k0 = 0.0;



  float radius = sphere_1_cache.w;


  constexpr int kNumVoxelsPerBlock = 8;
  
  nvblox::Vector3f query_location;

  // read data into vector3f:
  float max_distance = 0.0f;
  
  float3 global_grad = make_float3(0,0,0);

  float4 obb_quat = make_float4(0.0);
  float3 obb_pos = make_float3(0.0);

  for(int m = 0; m<num_mappers; m++)
  {
    if (blox_enable[m] == 0)
    {
      continue;
    }
    signed_distance = 0.0;
    const float block_size = block_sizes[m];
    const float voxel_size = block_size / kNumVoxelsPerBlock;

    load_layer_pose(&blox_pose[m*8], obb_pos, obb_quat);
    transform_sphere_quat(obb_pos, obb_quat, sphere_1_cache,
                          loc_sphere_1);

    // transform sphere to nvblox base frame:

    float curr_jump_distance = 0.0;
    float4 sum_grad = make_float4(0.0, 0.0, 0.0, 0.0);




    query_location(0) = loc_sphere_1.x;
    query_location(1) = loc_sphere_1.y;
    query_location(2) = loc_sphere_1.z;
    
    // Get the correct block from the hash.
    nvblox::EsdfVoxel* esdf_voxel;
    if (!nvblox::getVoxelAtPosition<nvblox::EsdfVoxel>(hashes[m], query_location, block_size,
                                      &esdf_voxel) ||
        !esdf_voxel->observed) {
      // This voxel is outside of the map or not observed. 
      curr_jump_distance = loc_sphere_1.w;// jump by sphere radius
    } 
    else 
    {

      compute_voxel_distance_grad(esdf_voxel, sum_grad, 
      signed_distance,
      loc_sphere_1, eta, voxel_size, hashes[m], block_size);
      // update jump distance:
      curr_jump_distance = fabsf(signed_distance);
    }

    const float jump_mid_distance = curr_jump_distance;
    float4 t_grad = make_float4(0.0, 0.0, 0.0, 0.0);

    if (sweep_back && jump_mid_distance < sphere_0_distance/2)
    {

      transform_sphere_quat(obb_pos, obb_quat, sphere_0_cache, loc_sphere_0);
      for(int j=0; j<sw_steps; j++)
      {
        signed_distance = 0.0;

        if(curr_jump_distance >= (sphere_0_len/2)) {
          break;
        }
        k0 = 1 - (curr_jump_distance/sphere_0_len);
        // compute collision
        const float4 interpolated_sphere =
        (k0)*loc_sphere_1 + (1 - k0) * loc_sphere_0;
        query_location(0) = interpolated_sphere.x;
        query_location(1) = interpolated_sphere.y;
        query_location(2) = interpolated_sphere.z;
        if (!nvblox::getVoxelAtPosition<nvblox::EsdfVoxel>(hashes[m], query_location, block_size,
                                      &esdf_voxel) ||
            !esdf_voxel->observed) {
          // This voxel is outside of the map or not observed. 
          curr_jump_distance += loc_sphere_1.w;// jump by sphere radius
        } 
        else 
        {

          compute_voxel_distance_grad(esdf_voxel, t_grad, 
          signed_distance,
          interpolated_sphere, eta, voxel_size, hashes[m], block_size);
          // update jump distance:
          curr_jump_distance += fabsf(signed_distance);
          sum_grad += t_grad;
        }


      }
    }
    if(sweep_fwd && jump_mid_distance < (sphere_2_len / 2))
    {
      curr_jump_distance = jump_mid_distance;
      transform_sphere_quat(obb_pos, obb_quat, sphere_2_cache, loc_sphere_2);
      for(int j=0; j< sw_steps; j++)
      {
        if (curr_jump_distance >= (sphere_2_len)/2)
        {
          break;
        }
        k0 = 1 - (curr_jump_distance/sphere_2_len);
        // compute collision
        const float4 interpolated_sphere =
        (k0)*loc_sphere_1 + (1 - k0) * loc_sphere_2;
        query_location(0) = interpolated_sphere.x;
        query_location(1) = interpolated_sphere.y;
        query_location(2) = interpolated_sphere.z;
        if (!nvblox::getVoxelAtPosition<nvblox::EsdfVoxel>(hashes[m], query_location, block_size,
                                      &esdf_voxel) ||
            !esdf_voxel->observed) {
          // This voxel is outside of the map or not observed. 
          curr_jump_distance += loc_sphere_1.w;// jump by sphere radius
        } 
        else 
        {

          compute_voxel_distance_grad(esdf_voxel, t_grad, 
          signed_distance,
          interpolated_sphere, eta, voxel_size, hashes[m], block_size);
          // update jump distance:
          curr_jump_distance += fabsf(signed_distance);
          sum_grad += t_grad;
        }
      }
    }

    if (sum_grad.w > 0.0)
    {
      max_distance += sum_grad.w;

      inv_transform_vec_quat_add(obb_pos, obb_quat,sum_grad, global_grad);
    }
  
    
  }


  // sparsity opt:
if (max_distance == 0) {
    if (sparsity_idx[sph_idx] == 0) {
      return;
    }
    sparsity_idx[sph_idx] = 0;
    if (write_grad) {
      *(float3 *)&out_grad[sph_idx * 4] = global_grad; // max_grad is all zeros
    }
    out_distance[sph_idx] = 0.0;
    return;
}
if(enable_speed_metric)
{


 if (sweep_back && sweep_fwd) {
    const float dt = speed_dt[0];

    scale_speed_metric(sphere_0_cache, sphere_1_cache, sphere_2_cache, dt,
                       write_grad, max_distance, global_grad);
  }

}

  // else max_dist != 0
  max_distance = weight[0] * max_distance;

  if (write_grad) {
    *(float3 *)&out_grad[sph_idx * 4] = weight[0] * global_grad;
  }
  out_distance[sph_idx] = max_distance;
  sparsity_idx[sph_idx] = 1;

}



}
  
}

}