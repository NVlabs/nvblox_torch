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
#include "py_scene.h"

#include <nvblox/primitives/scene.h>


namespace pynvblox {
    
void Scene::setAABB(std::vector<double> low, std::vector<double> high) {
// TODO: improve the error checking, add exceptions etc
if (low.size() != 3 || high.size() !=3) {
    std::cerr << "Scene::setAABB expects two vectors of length 3. Ignoring invalid request to setAABB." << std::endl;
    return;
}
scene_->aabb() = nvblox::AxisAlignedBoundingBox(nvblox::Vector3f(low[0], low[1], low[2]),
                                        nvblox::Vector3f(high[0], high[1], high[2]));
}

void Scene::addPlaneBoundaries(double x_min, double x_max, double y_min, double y_max) {
scene_->addPlaneBoundaries(x_min, x_max, y_min, y_max);
}

void Scene::addGroundLevel(double level) {
scene_->addGroundLevel(level);
}

void Scene::addCeiling(double ceiling) {
scene_->addCeiling(ceiling);
}

void Scene::addPrimitive(std::string type, std::vector<double> prim_params) {
if (type == "cube") {
    if (prim_params.size() != 6) {
        std::cerr << "Scene::addPrimitive excepts 6 parameters for type 'cube'. Ignoring invalid request to addPrimitive" << std::endl;
        return;
    }
    scene_->addPrimitive(std::make_unique<nvblox::primitives::Cube>(
        nvblox::Vector3f(prim_params[0], prim_params[1], prim_params[2]), 
        nvblox::Vector3f(prim_params[3], prim_params[4], prim_params[5])));
} else if (type == "sphere") {
    if (prim_params.size() != 4) {
        std::cerr << "Scene::addPrimitive excepts 4 parameters for type 'sphere'. Ignoring invalid request to addPrimitive" << std::endl;
        return;
    }
    scene_->addPrimitive(
        std::make_unique<nvblox::primitives::Sphere>(nvblox::Vector3f(prim_params[0], prim_params[1], prim_params[2]), prim_params[3]));
}
else {
    std::cerr << "Scene::addPrimitive received invalid primitive type: " << type << std::endl;
    return;
}
}

void Scene::createDummyMap() {
      // Create a map that's a box with a sphere in the middle.
  scene_->aabb() = nvblox::AxisAlignedBoundingBox(nvblox::Vector3f(-5.5f, -5.5f, -0.5f),
                                         nvblox::Vector3f(5.5f, 5.5f, 5.5f));
  scene_->addPlaneBoundaries(-5.0f, 5.0f, -5.0f, 5.0f);
  scene_->addGroundLevel(0.0f);
  scene_->addCeiling(5.0f);
  scene_->addPrimitive(std::make_unique<nvblox::primitives::Cube>(
      nvblox::Vector3f(0.0f, 0.0f, 2.0f), nvblox::Vector3f(2.0f, 2.0f, 2.0f)));
  scene_->addPrimitive(
      std::make_unique<nvblox::primitives::Sphere>(nvblox::Vector3f(0.0f, 0.0f, 2.0f), 2.0f));
}

} // namespace pynvblox

