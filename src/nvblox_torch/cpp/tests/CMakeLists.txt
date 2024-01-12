#[[
Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited.
]]

# Add some tests.
set(TEST_OPTIONS DISCOVERY_TIMEOUT 30)
function (add_nvblox_torch_test test_name)
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name} ${GTEST_LIBRARIES} py_nvblox nvblox_lib)
    gtest_discover_tests(${test_name} ${TEST_OPTIONS})
endfunction()

add_nvblox_torch_test(test_tensor_conversions)
