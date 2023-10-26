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

#include "py_scene.h"
#include "py_mapper.h"

// TODO: Create separate wrapper around the mapper like we did for the scene
namespace pynvblox {

TORCH_LIBRARY(pynvblox, m) {
  m.class_<Scene>("Scene")
    .def(torch::init())
    .def("set_aabb", &Scene::setAABB)
    .def("add_plane_boundaries", &Scene::addPlaneBoundaries)
    .def("add_ground_level", &Scene::addGroundLevel)
    .def("add_ceiling", &Scene::addCeiling)
    .def("add_primitive", &Scene::addPrimitive)
    .def("create_dummy_map", &Scene::createDummyMap)
    ;

  // TODO: In future, can break this NvBlox up into a Mapper, Renderer, what have you
  // For now, they all wrap a single RgbdMapper, so no reason to break up
  m.class_<Mapper>("Mapper")
    .def(torch::init<std::vector<double>, std::vector<std::string>, bool>())
    // Mapping methods
    .def("integrate_depth", &Mapper::integrateDepth)
    .def("integrate_color", &Mapper::integrateColor)
    .def("update_esdf", &Mapper::updateEsdf)
    .def("update_mesh", &Mapper::updateMesh)
    .def("full_update", &Mapper::fullUpdate)
    .def("clear", &Mapper::clear)
    .def("update_hashmaps", &Mapper::updateHashMaps)
    // Occupancy integrator methods
    .def("decay_occupancy", &Mapper::decayOccupancy)
    // Rendering methods
    .def("render_depth_image", &Mapper::renderDepthImage)
    .def("render_depth_and_color_image", &Mapper::renderDepthAndColorImage)
    .def("get_all_occupied_voxels", &Mapper::getAllOccupiedVoxels)
    // Query methods
    .def("query_sdf", &Mapper::querySdf)
    .def("query_multi_sdf", &Mapper::queryMultiSdf)
    .def("query_multi_tsdf", &Mapper::queryMultiTsdf)
    .def("query_multi_occupancy", &Mapper::queryMultiOccupancy)
    .def("query_sphere_sdf_cost", &Mapper::querySphereSdfMultiCost)
    .def("query_sphere_trajectory_sdf_cost", &Mapper::querySphereTrajectorySdfMultiCost)
    // File methods
    .def("output_mesh_ply", &Mapper::outputMeshPly)
    .def("load_from_file", &Mapper::loadFromFile)
    .def("get_mesh", &Mapper::getMesh)
    // Attributes
    .def("build_from_scene", &Mapper::buildFromScene)
    ;
}

} // namespace pynvblox
