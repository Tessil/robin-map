cmake_minimum_required(VERSION 3.1)

project(tsl_robin_map)

add_library(tsl_robin_map INTERFACE)
# Use tsl::robin_map as target, more consistent with other libraries conventions (Boost, Qt, ...)
add_library(tsl::robin_map ALIAS tsl_robin_map)

target_include_directories(tsl_robin_map INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_sources(tsl_robin_map INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include/tsl/robin_growth_policy.h"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/include/tsl/robin_hash.h"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/include/tsl/robin_map.h"
                                       "${CMAKE_CURRENT_SOURCE_DIR}/include/tsl/robin_set.h")

if(MSVC)
    target_sources(tsl_robin_map INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/tsl_robin_map.natvis")
endif()

if(${CMAKE_VERSION} VERSION_GREATER "3.7") 
    # Only available since version 3.8
    target_compile_features(tsl_robin_map INTERFACE cxx_std_11)
endif()
