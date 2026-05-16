# Example apps that consume the library. All three depend on Armadillo.
# Disabled by default — enable with -DLISTOFCLUSTERS_BUILD_EXAMPLES=ON.

find_package(Armadillo REQUIRED)

foreach(ex cache dbscan baseline_brute_force)
    file(GLOB _src ${CMAKE_CURRENT_SOURCE_DIR}/${ex}/*.cc)
    add_executable(${ex} ${_src})
    target_link_libraries(${ex} PRIVATE
        listofclusters::listofclusters
        ${ARMADILLO_LIBRARIES})
    target_include_directories(${ex} PRIVATE ${ARMADILLO_INCLUDE_DIRS})
endforeach()
