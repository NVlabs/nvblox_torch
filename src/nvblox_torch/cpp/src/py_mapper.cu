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
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include "nvblox_torch/py_mapper.h"
#include "nvblox_torch/sdf_query.cuh"
#include "nvblox_torch/sdf_cost_query.cuh"



namespace pynvblox {


// declare nvblox variables here:
Mapper::Mapper(std::vector<double> voxel_size_m, 
std::vector<std::string> projective_layer_type,
std::vector<double> layer_parameters,
bool free_on_destruction
) {
  layer_parameters_ = layer_parameters;

  free_on_destruct_ = free_on_destruction;
  // Initialize the mapper 
  voxel_size_m_ = voxel_size_m;
  projective_layer_type_ = projective_layer_type;

  for (int i=0; i < voxel_size_m.size(); i++) {
    addMapper(voxel_size_m_[i], projective_layer_type_[i], layer_parameters);
  }
  // create a gpu buffer that has the voxel_sizes:
  float * voxel_size_cpu_;
  voxel_size_cpu_ = (float*) malloc(sizeof(float)*voxel_size_m.size());
  for (int i =0; i<voxel_size_m.size(); i ++ )
  {
    voxel_size_cpu_[i] = (float) mappers_[i]->esdf_layer().block_size();
  }
  cudaError_t cudaStatus = cudaMalloc((void**) &voxel_size_m_gpu_, sizeof(float) * 
  voxel_size_m.size());

  cudaMemcpy(voxel_size_m_gpu_, voxel_size_cpu_, sizeof(float) * voxel_size_m.size(), 
  cudaMemcpyHostToDevice);
  num_layers = mappers_.size();
  initHashMaps();
  updateHashMaps();
} // initialize with nothing?

void Mapper::addMapper(double voxel_size_m, std::string projective_layer_type,
std::vector<double> layer_parameters)
{
  //std::cout << "Setting up mapper." << std::endl;
  nvblox::ProjectiveLayerType layer_type;
  if (projective_layer_type == "tsdf") {
    layer_type = nvblox::ProjectiveLayerType::kTsdf;
    //std::cout << "    TSDF projective layer" << std::endl;
  }
  else { // projective_layer_type == occupancy
    layer_type = nvblox::ProjectiveLayerType::kOccupancy;
    //std::cout << "    Occupancy projective layer" << std::endl;
  }

  auto mapper = std::make_shared<nvblox::Mapper>(
    voxel_size_m, nvblox::MemoryType::kDevice, layer_type);

  // Default parameters
  // TODO: Expose these and other similar parameters to Python side
  mapper->mesh_integrator().min_weight(layer_parameters[0]);
  mapper->color_integrator().max_integration_distance_m(10.0f);
  mapper->esdf_integrator().max_esdf_distance_m(2.0f);
  mapper->esdf_integrator().max_site_distance_vox(1.73);
  //mapper->esdf_integrator().min_weight(2.0f);

  //if (projective_layer_type == "tsdf") {
    mapper->tsdf_integrator().max_integration_distance_m(15.0f);
    mapper->tsdf_integrator().view_calculator().raycast_subsampling_factor(2);
    mapper->tsdf_decay_integrator().decay_factor(0.001);
    mapper->tsdf_decay_integrator().decayed_weight_threshold(0.001);
    mapper->tsdf_decay_integrator().set_free_distance_on_decayed(true);
    mapper->tsdf_decay_integrator().deallocate_decayed_blocks(false);
  //}
  //else {
    mapper->occupancy_integrator().occupied_region_half_width_m(voxel_size_m / 4);
    mapper->occupancy_integrator().max_integration_distance_m(15.0f);
    mapper->occupancy_decay_integrator().free_region_decay_probability(0.55);
    mapper->occupancy_decay_integrator().occupied_region_decay_probability(0.25);
  //}
  mappers_.push_back(mapper);
}

void Mapper::integrateDepth(torch::Tensor depth_frame_t, torch::Tensor T_L_C_t, torch::Tensor intrinsics_t, long mapper_id)
{
  auto mapper = mappers_[mapper_id];

  int height = depth_frame_t.sizes()[0];
  int width = depth_frame_t.sizes()[1];

  nvblox::Transform T_L_C = copy_transform_from_tensor(T_L_C_t);
  nvblox::Camera camera = camera_from_intrinsics_tensor(intrinsics_t, height, width);

  nvblox::DepthImage depth_frame = copy_depth_image_from_tensor(depth_frame_t);
  mapper->integrateDepth(depth_frame, T_L_C, camera);
}

void Mapper::integrateColor(torch::Tensor color_frame_t, torch::Tensor T_L_C_t, torch::Tensor intrinsics_t, long mapper_id)
{
  auto mapper = mappers_[mapper_id];

  int height = color_frame_t.sizes()[0];
  int width = color_frame_t.sizes()[1];
  nvblox::Transform T_L_C = copy_transform_from_tensor(T_L_C_t);
  nvblox::Camera camera = camera_from_intrinsics_tensor(intrinsics_t, height, width);

  nvblox::ColorImage color_frame = copy_color_image_from_tensor(color_frame_t);
  mapper->integrateColor(color_frame, T_L_C, camera);
}

void Mapper::updateEsdf(long mapper_id)
{
  if (mapper_id >= 0) {
    mappers_[mapper_id]->updateEsdf();
  }
  else {
    for (auto & mapper : mappers_) {
      mapper->updateEsdf();
    }
  }
  hash_update_ = false;
}

void Mapper::updateMesh(long mapper_id) {
  if (mapper_id >= 0) {
    mappers_[mapper_id]->updateMesh();
  }
  else {
    for (auto & mapper : mappers_) {
      mapper->updateMesh();
    }
  }
  hash_update_ = false;

}

void Mapper::fullUpdate(torch::Tensor depth_frame_t, torch::Tensor color_frame_t, torch::Tensor T_L_C_t, torch::Tensor intrinsics_t, long mapper_id) {
  auto mapper = mappers_[mapper_id];

  int height = depth_frame_t.sizes()[0];
  int width = depth_frame_t.sizes()[1];

  nvblox::Transform T_L_C = copy_transform_from_tensor(T_L_C_t);
  nvblox::Camera camera = camera_from_intrinsics_tensor(intrinsics_t, height, width);

  nvblox::DepthImage depth_frame = copy_depth_image_from_tensor(depth_frame_t);
  mapper->integrateDepth(depth_frame, T_L_C, camera);
  nvblox::ColorImage color_frame = copy_color_image_from_tensor(color_frame_t);
  mapper->integrateColor(color_frame, T_L_C, camera);

  mapper->updateEsdf();
  mapper->updateMesh();
}

void Mapper::decayOccupancy(long mapper_id) {
  if (mapper_id >= 0) {
    mappers_[mapper_id]->decayOccupancy();
    mappers_[mapper_id]->decayTsdf();
    
  }
  else {
    for (auto & mapper : mappers_) {
      mapper->decayOccupancy();
      mapper->decayTsdf();
      
    }
  }
}

void Mapper::clear(long mapper_id) {
  if (mapper_id >= 0) {
    mappers_[mapper_id]->occupancy_layer().clear();
    mappers_[mapper_id]->tsdf_layer().clear();
    mappers_[mapper_id]->esdf_layer().clear();
    mappers_[mapper_id]->color_layer().clear();
    mappers_[mapper_id]->mesh_layer().clear();
  }
  else {
    for (auto & mapper : mappers_) {
    mapper->occupancy_layer().clear();
    mapper->tsdf_layer().clear();    
     mapper->esdf_layer().clear();
  mapper->color_layer().clear();
  mapper->mesh_layer().clear();
    }
  }
  // TODO: Revive these after PyTorch c++11 ABI wheels are available
 
}

torch::Tensor Mapper::renderDepthImage(
  torch::Tensor camera_pose, 
  torch::Tensor intrinsics, 
  int64_t img_height, 
  int64_t img_width,
  double max_ray_length,
  int64_t max_steps,
  long mapper_id)
{
  auto mapper = mappers_[mapper_id];

  // TODO: This 4.0 is the default truncation distance in projective_integrator_base.h
  // This should be made a global constant and somehow set accordingly.
  double truncation_distance_m = voxel_size_m_[mapper_id] * 4.0;

  nvblox::Transform T_S_C = copy_transform_from_tensor(camera_pose);
  nvblox::Camera camera = camera_from_intrinsics_tensor(intrinsics, img_height, img_width);

  nvblox::SphereTracer sphere_tracer_gpu;
  sphere_tracer_gpu.maximum_ray_length_m(max_ray_length);
  sphere_tracer_gpu.maximum_steps(max_steps);

  nvblox::TsdfLayer &layer = mapper->tsdf_layer();
  torch::DeviceType device = torch::kCUDA; // Currently SphereTracer only supports GPU)


  torch::Tensor depth_image_t = init_depth_image_tensor(img_height, img_width, device);
  nvblox::DepthImageView depth_image_view = make_depth_image_view(depth_image_t);
  sphere_tracer_gpu.renderImageOnGPU(
      camera, T_S_C, layer, truncation_distance_m, &depth_image_view, 
      nvblox::MemoryType::kDevice);

  return depth_image_t;
}

std::vector<torch::Tensor> Mapper::renderDepthAndColorImage(
  torch::Tensor camera_pose, 
  torch::Tensor intrinsics, 
  int64_t img_height, 
  int64_t img_width,
  double max_ray_length,
  int64_t max_steps,
  long mapper_id)
{
  auto mapper = mappers_[mapper_id];
  // TODO: This 4.0 is the default truncation distance in projective_integrator_base.h
  // This should be made a global constant and somehow set accordingly.
  double truncation_distance_m = voxel_size_m_[mapper_id] * 4.0;

  nvblox::Transform T_S_C = copy_transform_from_tensor(camera_pose);
  nvblox::Camera camera = camera_from_intrinsics_tensor(intrinsics, img_height, img_width);

  nvblox::SphereTracer sphere_tracer_gpu;
  sphere_tracer_gpu.maximum_ray_length_m(max_ray_length);
  sphere_tracer_gpu.maximum_steps(max_steps);

  nvblox::TsdfLayer &tsdf_layer = mapper->tsdf_layer();
  nvblox::ColorLayer &color_layer = mapper->color_layer();

  torch::DeviceType device = torch::kCUDA; // Currently SphereTracer only supports GPU

  torch::Tensor depth_image_t = init_depth_image_tensor(img_height, img_width, device);
  nvblox::DepthImageView depth_image_view = make_depth_image_view(depth_image_t);
  torch::Tensor color_image_t = init_color_image_tensor(img_height, img_width, device);
  nvblox::ColorImageView color_image_view = make_color_image_view(color_image_t);

  sphere_tracer_gpu.renderRgbdImageOnGPU(
    camera, T_S_C, tsdf_layer, color_layer,
    truncation_distance_m, &depth_image_view, &color_image_view, nvblox::MemoryType::kDevice);

  return {depth_image_t, color_image_t};
}

/*
__global__ void isBlockOccupied(int num_blocks,
                                const VoxelBlock<TsdfVoxel>** blocks,
                                float cutoff_distance, float min_weight,
                                      bool* meshable) {
  dim3 voxel_index = threadIdx;
  // This for loop allows us to have fewer threadblocks than there are
  // blocks in this computation. We assume the threadblock size is constant
  // though to make our lives easier.
  for (int block_index = blockIdx.x; block_index < num_blocks;
       block_index += gridDim.x) {
    // Get the correct voxel for this index.
    const TsdfVoxel& voxel =
        blocks[block_index]
            ->voxels[voxel_index.z][voxel_index.y][voxel_index.x];
    if (fabs(voxel.distance) <= cutoff_distance && voxel.weight >= min_weight) {
      meshable[block_index] = true;
    }
  }
}*/


torch::Tensor Mapper::getAllOccupiedVoxels(long mapper_id) {
  auto mapper = mappers_[mapper_id];
  auto mapper_type = projective_layer_type_[mapper_id];
  int num_voxels = 1;

  /*

  if (mapper_type == "occupancy") {
    nvblox::TsdfLayer layer = mapper->tsdf_layer();
    //std::vector<nvblox::TsdfBlock*> = layer.getAllBlockPointers();
    std::vector<nvblox::Index3D> occupied_voxels;

    const std::vector<nvblox::Index3D> all_tsdf_blocks = layer.getAllBlockIndices();
    for (nvblox::Index3D index : all_tsdf_blocks) {
      nvblox::TsdfBlock block = layer.getBlockAtIndex(index).get();
      for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
          for (int z = 0; z < 8; z++) {
            
          }
        }
      }
        if(voxel.weight > 0 && voxel.distance < layer.block_size()) {
          occupied_voxels.push_back(...);
        }
    }
  }
  */

  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, 0);
  torch::Tensor voxels = torch::zeros({num_voxels, 4}, options);

  return voxels;
}

void Mapper::updateHashMaps()
{
  //cudaStream_t stream = at::cuda::getCurrentCUDAStream(); 

  for (int i = 0; i < num_layers; i++) {
    cpu_hashes_[i] = mappers_[i]->esdf_layer().getGpuLayerView().getHash().impl_;
  }
  cudaMemcpy(cuda_hashes_, cpu_hashes_, sizeof(HashMap) * num_layers, cudaMemcpyHostToDevice);
  hash_update_ = true;
}


void Mapper::initHashMaps()
{
  const int num_mappers = mappers_.size();
  cpu_hashes_ = (HashMap*) malloc(sizeof(HashMap) * num_mappers);
  cudaError_t cudaStatus = cudaMalloc((void**) &cuda_hashes_, sizeof(HashMap) * num_mappers);
  for (int i = 0; i < num_mappers; i++) {
    cpu_hashes_[i] = mappers_[i]->esdf_layer().getGpuLayerView().getHash().impl_;
  }
  cudaMemcpy(cuda_hashes_, cpu_hashes_, sizeof(HashMap) * num_mappers, cudaMemcpyHostToDevice);
  hash_init_ = true;
  num_layers = num_mappers;
}

bool Mapper::outputMeshPly(std::string mesh_output_path, long mapper_id) {
  auto mapper = mappers_[mapper_id];
  return nvblox::io::outputMeshLayerToPly(mapper->mesh_layer(), mesh_output_path.c_str());
}

bool Mapper::outputBloxMap(std::string blox_output_path, long mapper_id)
{
  auto mapper = mappers_[mapper_id];
  const bool result =  mapper->saveLayerCake(blox_output_path);
  return result;
}


std::vector<torch::Tensor> Mapper::getMesh(long mapper_id)
{
  auto mapper = mappers_[mapper_id];

  nvblox::Mesh layer_mesh = nvblox::Mesh::fromLayer(mapper->mesh_layer());
  auto options_f32 =
  torch::TensorOptions()
    .dtype(torch::kFloat32)
    .device(torch::kCPU)
    .requires_grad(false);
  auto options_i32 =
  torch::TensorOptions()
    .dtype(torch::kInt32)
    .device(torch::kCPU)
    .requires_grad(false);
  
  // create torch tensors on cpu:
  std::vector<torch::Tensor> output;
  torch::Tensor vert_tensor = torch::zeros({long(layer_mesh.vertices.size()), 3}, options_f32);
  torch::Tensor color_tensor = torch::zeros({long(layer_mesh.colors.size()), 4}, options_f32);
  torch::Tensor triangle_tensor = torch::zeros({long(layer_mesh.triangles.size()/3), 3}, options_i32);
  torch::Tensor normal_tensor = torch::zeros({long(layer_mesh.normals.size()),3}, options_f32);
  
  for(int i=0; i<layer_mesh.vertices.size(); i++)
  {
    vert_tensor[i][0] = layer_mesh.vertices[i](0);
    vert_tensor[i][1] = layer_mesh.vertices[i](1);
    vert_tensor[i][2] = layer_mesh.vertices[i](2);
    
  }
  if (layer_mesh.colors.size() > 0)
  {
    for(int i=0; i<layer_mesh.colors.size(); i++)
    {
      color_tensor[i][0] = layer_mesh.colors[i].r;
      color_tensor[i][1] = layer_mesh.colors[i].g;
      color_tensor[i][2] = layer_mesh.colors[i].b;
      color_tensor[i][3] = layer_mesh.colors[i].a;
      
    }

  }
  if (layer_mesh.normals.size() > 0)
  {
    for(int i=0; i<layer_mesh.normals.size(); i++)
    {
    normal_tensor[i][0] = layer_mesh.normals[i][0];
    normal_tensor[i][1] = layer_mesh.normals[i][1];
    normal_tensor[i][2] = layer_mesh.normals[i][2];
    }
  }
  for(int i=0; i< int(layer_mesh.triangles.size()/3); i++)
  {
    triangle_tensor[i][0] = layer_mesh.triangles[i*3];
    triangle_tensor[i][1] = layer_mesh.triangles[i*3 + 1];
    triangle_tensor[i][2] = layer_mesh.triangles[i*3 + 2];
  }
  output.push_back(vert_tensor);
  output.push_back(normal_tensor);
  output.push_back(color_tensor);
  output.push_back(triangle_tensor);
  return output;
}


void Mapper::buildFromScene(c10::intrusive_ptr<Scene> scene, long mapper_id) {
  auto mapper = mappers_[mapper_id];
  float voxel_size = voxel_size_m_[mapper_id];

  nvblox::TsdfLayer gt_tsdf(voxel_size, nvblox::MemoryType::kHost);
  scene->scene_->generateLayerFromScene<nvblox::TsdfVoxel>(4 * voxel_size, &gt_tsdf);
  mapper->tsdf_layer() = std::move(gt_tsdf);
  // Set the max computed distance to 5 meters.
  mapper->esdf_integrator().max_esdf_distance_m(5.0f);
  // Generate the ESDF from everything in the TSDF.
  mapper->updateEsdf();
}


std::vector<torch::Tensor> Mapper::queryMultiSdf(
    torch::Tensor closest_point,
		const torch::Tensor sphere_position_rad,
		const int64_t batch_size,
		const bool write_closest_point) {
  
  const int num_queries = batch_size;
  const int num_mappers = mappers_.size();
  /*
  using HashMap = nvblox::Index3DDeviceHashMapType<nvblox::EsdfBlock>;
  HashMap * hashes;
  HashMap * cudaHashes;
  hashes = (HashMap*) malloc(sizeof(HashMap) * num_mappers);
  cudaError_t cudaStatus = cudaMalloc((void**) &cudaHashes, sizeof(HashMap) * num_mappers);
  for (int i = 0; i < num_mappers; i++) {
    hashes[i] = mappers_[i]->esdf_layer().getGpuLayerView().getHash().impl_;
  }
  cudaMemcpy(cudaHashes, hashes, sizeof(HashMap) * num_mappers, cudaMemcpyHostToDevice);
  */
  // Call a kernel.
	cudaStream_t stream = at::cuda::getCurrentCUDAStream(); 

  constexpr int kNumThreads = 128;
  int num_blocks = num_queries / kNumThreads + 1;
  // Call the kernel.

  pynvblox::sdf::queryDistancesMultiMapperKernel<<<num_blocks, kNumThreads, 0, stream>>>(
      num_mappers, num_queries, cuda_hashes_, //gpu_layer_view.getHash().impl_,
      voxel_size_m_gpu_,
	  sphere_position_rad.data_ptr<float>(),
	  closest_point.data_ptr<float>(),
	  write_closest_point);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  //free(hashes);
  //cudaFree(cudaHashes);

  return {closest_point};
}

std::vector<torch::Tensor> Mapper::queryMultiTsdf(
    torch::Tensor outputs,
		const torch::Tensor sphere_position_rad,
		const int64_t batch_size) {

  const int num_queries = batch_size;
  const int num_mappers = mappers_.size();
  
  using HashMap = nvblox::Index3DDeviceHashMapType<nvblox::TsdfBlock>;
  HashMap * hashes;
  HashMap * cudaHashes;
  hashes = (HashMap*) malloc(sizeof(HashMap) * num_mappers);
  cudaError_t cudaStatus = cudaMalloc((void**) &cudaHashes, sizeof(HashMap) * num_mappers);
  for (int i = 0; i < num_mappers; i++) {
    hashes[i] = mappers_[i]->tsdf_layer().getGpuLayerView().getHash().impl_;
  }
  cudaMemcpy(cudaHashes, hashes, sizeof(HashMap) * num_mappers, cudaMemcpyHostToDevice);
  // Call a kernel.
	cudaStream_t stream = at::cuda::getCurrentCUDAStream(); 

  constexpr int kNumThreads = 128;
  int num_blocks = num_queries / kNumThreads + 1;
  // Call the kernel.

  float * out_tsdf = outputs.data_ptr<float>();
  float * out_weight = outputs.data_ptr<float>() + num_queries;

  pynvblox::sdf::queryTSDFMultiMapperKernel<<<num_blocks, kNumThreads, 0, stream>>>(
      num_mappers, num_queries, cudaHashes, //gpu_layer_view.getHash().impl_,
      voxel_size_m_gpu_, // TODO: support different block sizes
      sphere_position_rad.data_ptr<float>(),
      out_tsdf,
      out_weight);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  free(hashes);
  cudaFree(cudaHashes);

  return {outputs};
}

std::vector<torch::Tensor> Mapper::queryMultiOccupancy(
    torch::Tensor outputs,
		const torch::Tensor sphere_position_rad,
		const int64_t batch_size) {

  const int num_queries = batch_size;
  const int num_mappers = mappers_.size();
  using HashMap = nvblox::Index3DDeviceHashMapType<nvblox::OccupancyBlock>;
  HashMap * hashes;
  HashMap * cudaHashes;
  hashes = (HashMap*) malloc(sizeof(HashMap) * num_mappers);
  cudaError_t cudaStatus = cudaMalloc((void**) &cudaHashes, sizeof(HashMap) * num_mappers);
  for (int i = 0; i < num_mappers; i++) {
    hashes[i] = mappers_[i]->occupancy_layer().getGpuLayerView().getHash().impl_;
  }
  cudaMemcpy(cudaHashes, hashes, sizeof(HashMap) * num_mappers, cudaMemcpyHostToDevice);
  // Call a kernel.
	cudaStream_t stream = at::cuda::getCurrentCUDAStream(); 

  constexpr int kNumThreads = 128;
  int num_blocks = num_queries / kNumThreads + 1;
  // Call the kernel.

  float * out_log_odds = outputs.data_ptr<float>();

  pynvblox::sdf::queryOccupancyMultiMapperKernel<<<num_blocks, kNumThreads, 0, stream>>>(
      num_mappers, num_queries, cudaHashes, //gpu_layer_view.getHash().impl_,
      voxel_size_m_gpu_, // TODO: support different block sizes
      sphere_position_rad.data_ptr<float>(),
      out_log_odds);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  free(hashes);
  cudaFree(cudaHashes);

  return {outputs};
}

std::vector<torch::Tensor> Mapper::querySdf(
    torch::Tensor closest_point,
		const torch::Tensor sphere_position_rad,
		const int64_t batch_size,
		const bool write_closest_point,
    long mapper_id) {
  // TODO: Edit this to query all the mappers
  auto mapper = mappers_[mapper_id];

  const int num_queries = batch_size;
  // GPU hash transfer timer
  nvblox::GPULayerView<nvblox::EsdfBlock> gpu_layer_view =
      mapper->esdf_layer().getGpuLayerView();

  // Call a kernel.
	cudaStream_t stream = at::cuda::getCurrentCUDAStream(); 

  constexpr int kNumThreads = 128;
  int num_blocks = num_queries / kNumThreads + 1;
  // Call the kernel.

  pynvblox::sdf::queryDistancesKernel<<<num_blocks, kNumThreads, 0, stream>>>(
      num_queries, gpu_layer_view.getHash().impl_,
      mapper->esdf_layer().block_size(), 
	  sphere_position_rad.data_ptr<float>(),
	  closest_point.data_ptr<float>(),
	  write_closest_point);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {closest_point};
}




std::vector<torch::Tensor> Mapper::querySphereSdfMultiCost(
		const torch::Tensor sphere_position_rad,
    torch::Tensor out_distance,
    torch::Tensor out_grad,
    torch::Tensor sparsity_idx,
    const torch::Tensor weight,
    const torch::Tensor activation_distance,
    const torch::Tensor blox_pose,
    const torch::Tensor blox_enable,
    const int64_t batch_size, const int64_t horizon, 
    const int64_t n_spheres, const bool write_grad) 
  {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(); 

  const at::cuda::OptionalCUDAGuard guard(sphere_position_rad.device());
  assert(hash_update_);
  assert(hash_init_);
  const int num_mappers = num_layers;


  const int bnh_spheres = n_spheres * batch_size * horizon; //
  

  // Call a kernel.

  constexpr int kNumThreads = 128;
  int num_blocks = (bnh_spheres + kNumThreads -1) / kNumThreads;
  // Call the kernel.
  
  if (num_mappers == 1)
  {
    pynvblox::sdf::cost::sphereDistanceCostMultiKernel_map1<<<num_blocks, kNumThreads, 0, stream>>>(
    sphere_position_rad.data_ptr<float>(),
    out_distance.data_ptr<float>(),
    out_grad.data_ptr<float>(),
    sparsity_idx.data_ptr<uint8_t>(),
    weight.data_ptr<float>(),
    activation_distance.data_ptr<float>(),
    cuda_hashes_,
    blox_pose.data_ptr<float>(),
    blox_enable.data_ptr<uint8_t>(),
    voxel_size_m_gpu_, 
	  batch_size, 
    horizon,
    n_spheres,
    write_grad);
  }
  else if (num_mappers == 2)
  {
    pynvblox::sdf::cost::sphereDistanceCostMultiKernel_map2<<<num_blocks, kNumThreads, 0, stream>>>(
    sphere_position_rad.data_ptr<float>(),
    out_distance.data_ptr<float>(),
    out_grad.data_ptr<float>(),
    sparsity_idx.data_ptr<uint8_t>(),
    weight.data_ptr<float>(),
    activation_distance.data_ptr<float>(),
    cuda_hashes_,
    blox_pose.data_ptr<float>(),
    blox_enable.data_ptr<uint8_t>(),
    voxel_size_m_gpu_, 
	  batch_size, 
    horizon,
    n_spheres,
    write_grad);
  }
  else
  {
    pynvblox::sdf::cost::sphereDistanceCostMultiKernel<<<num_blocks, kNumThreads, 0, stream>>>(
    sphere_position_rad.data_ptr<float>(),
    out_distance.data_ptr<float>(),
    out_grad.data_ptr<float>(),
    sparsity_idx.data_ptr<uint8_t>(),
    weight.data_ptr<float>(),
    activation_distance.data_ptr<float>(),
    cuda_hashes_,
    blox_pose.data_ptr<float>(),
    blox_enable.data_ptr<uint8_t>(),
    voxel_size_m_gpu_, 
	  batch_size, 
    horizon,
    n_spheres,
    write_grad,
    num_mappers);
  }
  
  
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out_distance, out_grad, sparsity_idx};
}


std::vector<torch::Tensor> Mapper::querySphereTrajectorySdfMultiCost(
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
    const int64_t n_spheres, const int64_t sweep_steps, const bool enable_speed_metric, 
    const bool write_grad,
    const bool use_experimental) 
  {
  assert(hash_init_);
  assert(hash_update_);
  
	cudaStream_t stream = at::cuda::getCurrentCUDAStream(); 
  
  
  const at::cuda::OptionalCUDAGuard guard(sphere_position_rad.device());

  const int num_mappers = num_layers;
  const int bnh_spheres = n_spheres * batch_size * horizon; //

  
  // Call a kernel.

  constexpr int kNumThreads = 128;
  int num_blocks = (bnh_spheres + kNumThreads -1) / kNumThreads;

  if (sweep_steps <= 1 || !use_experimental)
  {
    // Call the kernel.
  if (num_mappers == 1)
  {

  
  
  pynvblox::sdf::cost::sphereTrajectoryDistanceCostMultiKernel_map1<<<num_blocks, kNumThreads, 0, stream>>>(
    sphere_position_rad.data_ptr<float>(),
    out_distance.data_ptr<float>(),
    out_grad.data_ptr<float>(),
    sparsity_idx.data_ptr<uint8_t>(),
    weight.data_ptr<float>(),
    activation_distance.data_ptr<float>(),
    speed_dt.data_ptr<float>(),
    cuda_hashes_,
    blox_pose.data_ptr<float>(),
    blox_enable.data_ptr<uint8_t>(),
    voxel_size_m_gpu_,
	  batch_size, 
    horizon,
    n_spheres,
    sweep_steps,
    enable_speed_metric,
    write_grad);
  }
  else if (num_mappers == 2)
  {
   
    pynvblox::sdf::cost::sphereTrajectoryDistanceCostMultiKernel_map2<<<num_blocks, kNumThreads, 0, stream>>>(
    sphere_position_rad.data_ptr<float>(),
    out_distance.data_ptr<float>(),
    out_grad.data_ptr<float>(),
    sparsity_idx.data_ptr<uint8_t>(),
    weight.data_ptr<float>(),
    activation_distance.data_ptr<float>(),
    speed_dt.data_ptr<float>(),
    cuda_hashes_,
    blox_pose.data_ptr<float>(),
    blox_enable.data_ptr<uint8_t>(),
    voxel_size_m_gpu_,
	  batch_size, 
    horizon,
    n_spheres,
    sweep_steps,
    enable_speed_metric,
    write_grad
    );
  }
  else
  {
    pynvblox::sdf::cost::sphereTrajectoryDistanceCostMultiKernel<<<num_blocks, kNumThreads, 0, stream>>>(
    sphere_position_rad.data_ptr<float>(),
    out_distance.data_ptr<float>(),
    out_grad.data_ptr<float>(),
    sparsity_idx.data_ptr<uint8_t>(),
    weight.data_ptr<float>(),
    activation_distance.data_ptr<float>(),
    speed_dt.data_ptr<float>(),
    cuda_hashes_,
    blox_pose.data_ptr<float>(),
    blox_enable.data_ptr<uint8_t>(),
    voxel_size_m_gpu_,
	  batch_size, 
    horizon,
    n_spheres,
    sweep_steps,
    enable_speed_metric,
    write_grad,
    num_mappers);
    
  }
  }

  else
  {
    // Call the kernel.
  if (num_mappers == 1)
  {

  
  pynvblox::sdf::cost::sphereSweptTrajectoryDistanceCostMultiKernel_map1<<<num_blocks, kNumThreads, 0, stream>>>(
    sphere_position_rad.data_ptr<float>(),
    out_distance.data_ptr<float>(),
    out_grad.data_ptr<float>(),
    sparsity_idx.data_ptr<uint8_t>(),
    weight.data_ptr<float>(),
    activation_distance.data_ptr<float>(),
    speed_dt.data_ptr<float>(),
    cuda_hashes_,
    blox_pose.data_ptr<float>(),
    blox_enable.data_ptr<uint8_t>(),
    voxel_size_m_gpu_,
	  batch_size, 
    horizon,
    n_spheres,
    sweep_steps,
    enable_speed_metric,
    write_grad);
 
  }
  else if (num_mappers == 2)
  {
    pynvblox::sdf::cost::sphereSweptTrajectoryDistanceCostMultiKernel_map2<<<num_blocks, kNumThreads, 0, stream>>>(
    sphere_position_rad.data_ptr<float>(),
    out_distance.data_ptr<float>(),
    out_grad.data_ptr<float>(),
    sparsity_idx.data_ptr<uint8_t>(),
    weight.data_ptr<float>(),
    activation_distance.data_ptr<float>(),
    speed_dt.data_ptr<float>(),
    cuda_hashes_,
    blox_pose.data_ptr<float>(),
    blox_enable.data_ptr<uint8_t>(),
    voxel_size_m_gpu_,
	  batch_size, 
    horizon,
    n_spheres,
    sweep_steps,
    enable_speed_metric,
    write_grad);
    
  }
  else
  {
    pynvblox::sdf::cost::sphereSweptTrajectoryDistanceCostMultiKernel<<<num_blocks, kNumThreads, 0, stream>>>(
    sphere_position_rad.data_ptr<float>(),
    out_distance.data_ptr<float>(),
    out_grad.data_ptr<float>(),
    sparsity_idx.data_ptr<uint8_t>(),
    weight.data_ptr<float>(),
    activation_distance.data_ptr<float>(),
    speed_dt.data_ptr<float>(),
    cuda_hashes_,
    blox_pose.data_ptr<float>(),
    blox_enable.data_ptr<uint8_t>(),
    voxel_size_m_gpu_,
	  batch_size, 
    horizon,
    n_spheres,
    sweep_steps,
    enable_speed_metric,
    write_grad,
    num_mappers);
   
  }
  }
  
  
  
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return {out_distance, out_grad, sparsity_idx};
}



void Mapper::loadFromFile(std::string file_path, long mapper_id)
{
	// TODO: How to load?
  //mapper_.reset(new RgbdMapper(file_path, MemoryType::kDevice));
  mappers_[mapper_id]->loadMap(file_path.c_str());
  hash_update_ = false;
  updateHashMaps();
}

Mapper::~Mapper()
{
  if (free_on_destruct_)
  {
    free(cpu_hashes_);
    cudaFree(cuda_hashes_);
    cudaFree(voxel_size_m_gpu_);
  }
  
}
} // namespace pynvblox