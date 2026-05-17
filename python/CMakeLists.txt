cmake_minimum_required(VERSION 3.20)
project(listofclusters_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find Python development components and nanobind.
find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import nanobind; print(nanobind.cmake_dir())"
  OUTPUT_VARIABLE nanobind_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(nanobind CONFIG REQUIRED)

# The library headers (header-only).
set(LIB_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../include")

nanobind_add_module(_listofclusters
  STABLE_ABI
  src/_listofclusters.cc
)
target_include_directories(_listofclusters PRIVATE "${LIB_INCLUDE}")
target_compile_options(_listofclusters PRIVATE -O3)

# Optional BLAS fast path: on Apple link Accelerate framework (cblas_sgemm
# for the batched pairwise distance hot path in detail/batched_distance.hh).
# On other platforms try to find a BLAS library; if none, the NEON/AVX2
# fallback in the same header is still used.
if(APPLE)
  target_link_libraries(_listofclusters PRIVATE "-framework Accelerate")
  target_compile_definitions(_listofclusters PRIVATE LISTOFCLUSTERS_USE_BLAS=1)
else()
  find_package(BLAS QUIET)
  if(BLAS_FOUND)
    target_link_libraries(_listofclusters PRIVATE ${BLAS_LIBRARIES})
    target_compile_definitions(_listofclusters PRIVATE LISTOFCLUSTERS_USE_BLAS=1)
  endif()
endif()

install(TARGETS _listofclusters LIBRARY DESTINATION listofclusters)
