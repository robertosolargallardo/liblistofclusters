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

install(TARGETS _listofclusters LIBRARY DESTINATION listofclusters)
