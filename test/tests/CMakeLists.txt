set(MAIN_TEST "main.cpp")

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/log_out")
configure_file(test_config.h.in ${CMAKE_BINARY_DIR}/generated/test_config.h)
include_directories(${CMAKE_BINARY_DIR}/generated/ ${CMAKE_CURRENT_SOURCE_DIR})

function(addTestItem test_name source lib_list run_param)
    add_executable(${test_name} ${MAIN_TEST} ${source})
    target_link_libraries(${test_name}
        ${lib_list}
        libgtest
        ${GCTA_SYSTEM_LIBS}
    )
    add_dependencies(${test_name} gcta64)
    if("${run_param}" STREQUAL "")
        add_test(NAME ${test_name} COMMAND ${test_name})
    else()
        add_test(NAME ${test_name} COMMAND ${test_name} "${run_param}")
    endif()
endfunction(addTestItem)

addTestItem(hello_test hello_test.cpp "" "")
addTestItem(table_test table_test.cpp "tables" "")
addTestItem(logger_test test_logger.cpp "logger" "")
addTestItem(thread_test test_thread.cpp "threadpool;logger" "")
addTestItem(buffer_test test_buffer.cpp "logger" "")
addTestItem(matrix_test test_matrix.cpp "logger" "")
addTestItem(chisq_test test_chisq.cpp "statlib;acat;utils;mainV1;logger" "")
addTestItem(covar_test test_covar.cpp "covar;pheno;optionio;statlib;utils;logger" "")
addTestItem(pheno_test test_pheno.cpp "pheno;optionio;utils;logger" "")
addTestItem(marker_test test_marker.cpp "marker;optionio;utils;logger" "")
addTestItem(geno_test test_geno.cpp "geno;marker;pheno;optionio;tables;utils;Pgenlib;logger" "")
addTestItem(grm_test test_grm.cpp "grm;geno;marker;pheno;optionio;tables;mem;utils;Pgenlib;mainV1;logger" "")
addTestItem(fastfam_test test_fastfam.cpp "fastfam;covar;geno;marker;pheno;optionio;statlib;acat;tables;mem;utils;Pgenlib;logger" "")

# Dedicated bounds-checked target to catch std::vector out-of-bounds accesses in FastFAM
add_executable(fastfam_bounds_test ${MAIN_TEST} test_fastfam_bounds.cpp "${PROJECT_SOURCE_DIR}/src/FastFAM.cpp")
target_compile_definitions(fastfam_bounds_test PRIVATE
    _LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_FAST
    _GLIBCXX_ASSERTIONS=1
)
target_link_libraries(fastfam_bounds_test
    covar geno marker pheno optionio statlib acat tables mem utils Pgenlib logger
    libgtest
    ${GCTA_SYSTEM_LIBS}
)
add_test(NAME fastfam_bounds_test COMMAND fastfam_bounds_test)
