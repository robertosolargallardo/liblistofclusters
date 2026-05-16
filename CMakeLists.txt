cmake_minimum_required(VERSION 3.20)

project(liblistofclusters
    VERSION 0.1.0
    DESCRIPTION "Header-only C++ template library implementing the Chavez-Navarro List of Clusters metric-space index"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(LISTOFCLUSTERS_BUILD_TESTS    "Build unit tests"                  ON)
option(LISTOFCLUSTERS_BUILD_BENCH    "Build benchmarks"                  OFF)
option(LISTOFCLUSTERS_BUILD_PYTHON   "Build Python wrapper"              OFF)
option(LISTOFCLUSTERS_ENABLE_ASAN    "Enable AddressSanitizer in tests"  ON)
option(LISTOFCLUSTERS_ENABLE_UBSAN   "Enable UndefinedBehaviorSanitizer in tests" ON)

# Header-only INTERFACE library. Consumers link with
# `target_link_libraries(my_target PRIVATE listofclusters::listofclusters)`.
add_library(listofclusters INTERFACE)
add_library(listofclusters::listofclusters ALIAS listofclusters)

target_include_directories(listofclusters INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

target_compile_features(listofclusters INTERFACE cxx_std_23)

if(LISTOFCLUSTERS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(LISTOFCLUSTERS_BUILD_BENCH)
    add_subdirectory(bench)
endif()

if(LISTOFCLUSTERS_BUILD_PYTHON)
    add_subdirectory(python)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(DIRECTORY include/listofclusters
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS listofclusters EXPORT listofclustersTargets)

install(EXPORT listofclustersTargets
        FILE listofclustersTargets.cmake
        NAMESPACE listofclusters::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/listofclusters)
