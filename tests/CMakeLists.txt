# Tests use no third-party framework yet — the smoke test is self-contained
# and uses std::vector<double> so it has no Armadillo dependency.

add_executable(test_smoke test_smoke.cc)
target_link_libraries(test_smoke PRIVATE listofclusters::listofclusters)
target_compile_options(test_smoke PRIVATE -Wall -Wextra -Wpedantic)

if(LISTOFCLUSTERS_ENABLE_ASAN OR LISTOFCLUSTERS_ENABLE_UBSAN)
    set(_san_flags "")
    if(LISTOFCLUSTERS_ENABLE_ASAN)
        list(APPEND _san_flags address)
    endif()
    if(LISTOFCLUSTERS_ENABLE_UBSAN)
        list(APPEND _san_flags undefined)
    endif()
    list(JOIN _san_flags "," _san_joined)
    target_compile_options(test_smoke PRIVATE -fsanitize=${_san_joined})
    target_link_options(test_smoke PRIVATE -fsanitize=${_san_joined})
endif()

add_test(NAME smoke COMMAND test_smoke)
