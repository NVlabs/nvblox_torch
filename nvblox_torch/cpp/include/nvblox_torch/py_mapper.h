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

#include <torch/custom_class.h> // This is the file that contains info about torch+class
#include <ATen/ATen.h>

#include <vector>
#include <nvblox/mapper/mapper.h>
#include "nvblox/core/types.h"
#include "nvblox/io/mesh_io.h"
#include "nvblox/io/ply_writer.h"
#include "nvblox/rays/sphere_tracer.h"
#include "nvblox/mesh/mesh.h"
#include <nvblox/core/indexing.h>
#include "nvblox/map/layer.h"
#include "nvblox/map/voxels.h"
#include <nvblox/primitives/scene.h>
#include <nvblox/utils/timing.h>
#include <nvblox/gpu_hash/internal/cuda/gpu_indexing.cuh>


#include "nvblox_torch/convert_tensors.h"
#include "nvblox_torch/py_scene.h"



namespace pynvblox {
using HashMap = nvblox::Index3DDeviceHashMapType<nvblox::EsdfBlock>;

struct Mapper : torch::CustomClassHolder {

  Mapper(std::vector<double> voxel_size_m, 
  std::vector<std::string> projective_layer_type,
  std::vector<double> layer_parameters={1.0f},
  bool free_on_destruction=false);

  ~Mapper();
  //std::shared_ptr<nvblox::Mapper> mapper_;
  std::vector<std::shared_ptr<nvblox::Mapper>> mappers_;

  std::vector<double> voxel_size_m_;
  std::vector<double> layer_parameters_;
  float * voxel_size_m_gpu_; 
  
  std::vector<std::string> projective_layer_type_;
  HashMap * cuda_hashes_;
  HashMap * cpu_hashes_;
  int num_layers;
  bool hash_init_ = false;
  bool hash_update_ = false;
  bool free_on_destruct_ = false;
  void initHashMaps();
  void updateHashMaps();

  void integrateDepth(torch::Tensor depth_frame_t, torch::Tensor T_L_C_t, torch::Tensor intrinsics_t, long mapper_id=-1);

  void integrateColor(torch::Tensor color_frame_t, torch::Tensor T_L_C_t, torch::Tensor intrinsics_t, long mapper_id=-1);

  void updateEsdf(long mapper_id=-1);

  void updateMesh(long mapper_id=-1);

  void fullUpdate(torch::Tensor depth_frame_t, torch::Tensor color_frame_t, torch::Tensor T_L_C_t, torch::Tensor intrinsics_t, long mapper_id);

  void decayOccupancy(long mapper_id=-1);

  void clear(long mapper_id=-1);

  void addMapper(double voxel_size_m, std::string projective_layer_type, std::vector<double> layer_parameters = {1.0f});

  torch::Tensor renderDepthImage(
    torch::Tensor camera_pose, 
    torch::Tensor intrinsics, 
    int64_t img_height, 
    int64_t img_width,
    double max_ray_length,
    int64_t max_steps,
    long mapper_id);

  std::vector<torch::Tensor> renderDepthAndColorImage(
    torch::Tensor camera_pose, 
    torch::Tensor intrinsics, 
    int64_t img_height, 
    int64_t img_width,
    double max_ray_length,
    int64_t max_steps,
    long mapper_id);

  torch::Tensor getAllOccupiedVoxels(long mapper_id);

  std::vector<torch::Tensor> querySdf(
    torch::Tensor closest_point,
    const torch::Tensor sphere_position_rad,
    const int64_t batch_size,
    const bool write_closest_point,
    long mapper_id=-1);

  std::vector<torch::Tensor> queryMultiSdf(
    torch::Tensor closest_point,
		const torch::Tensor sphere_position_rad,
		const int64_t batch_size,
		const bool write_closest_point);

  std::vector<torch::Tensor> queryMultiTsdf(
    torch::Tensor outputs,
		const torch::Tensor sphere_position_rad,
		const int64_t batch_size);

  std::vector<torch::Tensor> queryMultiOccupancy(
    torch::Tensor outputs,
		const torch::Tensor sphere_position_rad,
		const int64_t batch_size);


  std::vector<torch::Tensor> querySphereSdfMultiCost(
		const torch::Tensor sphere_position_rad,
    torch::Tensor out_distance,
    torch::Tensor out_grad,
    torch::Tensor sparsity_idx,
    const torch::Tensor weight,
    const torch::Tensor activation_distance,
    const torch::Tensor blox_pose,
    const torch::Tensor blox_enable,
    const int64_t batch_size, const int64_t horizon, 
    const int64_t n_spheres, const bool write_grad) ;

  std::vector<torch::Tensor> querySphereTrajectorySdfMultiCost(
		const torch::Tensor sphere_position_rad,
    torch::Tensor out_distance,
    torch::Tensor out_grad,
    torch::Tensor sparsity_idx,
    const torch::Tensor weight,
    const torch::Tensor activation_distance,
    const torch::Tensor speed_dt,
    const torch::Tensor blox_pose,
    const torch::Tensor blox_enable,
    const int64_t batch_size, const int64_t horizon, 
    const int64_t n_spheres, const int64_t sweep_steps, const bool enable_speed_metric, const bool write_grad);
  
  bool outputMeshPly(std::string mesh_output_path, long mapper_id=0);
  bool outputBloxMap(std::string blox_output_path, long mapper_id=0);

  std::vector<torch::Tensor> getMesh(long mapper_id=0);
  void loadFromFile(std::string file_path, long mapper_id=0);

  void buildFromScene(c10::intrusive_ptr<Scene> scene, long mapper_id=0);

  c10::intrusive_ptr<Mapper> clone() const {
    return c10::make_intrusive<Mapper>(voxel_size_m_, projective_layer_type_, layer_parameters_,free_on_destruct_);
  }

};

} // namespace pynvblox