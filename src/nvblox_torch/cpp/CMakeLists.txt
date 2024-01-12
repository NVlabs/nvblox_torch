#[[
Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited.
]]
cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(nvblox_torch_cpp LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_CXX_STANDARD 17) # pytorch 2.0 requires 17 while pytorch 1.x works with 14.
set(CMAKE_CXX_STANDARD_REQUIRED True) 

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
    set(CUDA_ARCHITECTURE_FLAGS "70")
else()
    set(CUDA_ARCHITECTURE_FLAGS "-gencode=arch=compute_70,code=sm_70 ")
endif()

option(BUILD_TESTING "Build tests" OFF)

# This asks include files from nvblox to not provide implementations
# that cause linking against ABI-incompatible stuff
ADD_DEFINITIONS(-DPRE_CXX11_ABI_LINKABLE)

SET(nvblox_link_options "-Wl,--disable-new-dtags")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(PkgConfig REQUIRED)
pkg_check_modules(glog REQUIRED libglog)
# Download thirdparty deps
#message(STATUS "Downloading 3rdparty dependencies")
#message(STATUS "Downloading Eigen")
include(thirdparty/eigen/eigen.cmake)
#find_package(Glog REQUIRED)

#find_package(Eigen3 REQUIRED) # we need to import eigen from nvblox?


find_package(CUDA REQUIRED)

find_package(nvblox REQUIRED)


# compile cuda library:# Pass options to NVCC
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --disable-warnings --generate-line-info -lineinfo -Xcudafe --display_error_number")


set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --compiler-options -fPIC -std=c++14")

############
# INCLUDES #
############

# Internal includes
include_directories(include)

# External includes
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${GLOG_INCLUDE_DIRS} ${gflags_INCLUDE_DIRS})

#############
# LIBRARIES #
#############

### Nvblox-torch python wrapper
add_library(py_nvblox SHARED src/py_nvblox.cu src/py_mapper.cu
            src/py_scene.cpp src/convert_tensors.cpp src/sdf_query.cu src/sdf_cost_query.cu)

            target_link_libraries(py_nvblox
nvblox_lib
nvblox_eigen
${glog_LIBRARIES}
${TORCH_LIBRARIES}
${CUDA_LIBRARIES}
#${CUDA_nv}
#${CUDA_nvToolsExt_LIBRARY} 
)
set_target_properties(py_nvblox PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

#########
# TESTS #
#########
include(CTest)

if(BUILD_TESTING)
    find_package(GTest REQUIRED)
    enable_testing()
    add_subdirectory(tests)
endif()
