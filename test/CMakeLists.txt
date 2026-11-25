find_package(Threads REQUIRED)

option(USE_SYSTEM_GTEST "Prefer an installed system GoogleTest over the vendored C++11/14 GTest header" OFF)

if(USE_SYSTEM_GTEST)
    find_package(GTest REQUIRED)
    add_library(libgtest INTERFACE)
    target_link_libraries(libgtest INTERFACE GTest::GTest Threads::Threads)
else()
    add_library(libgtest INTERFACE)
    target_include_directories(libgtest INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries(libgtest INTERFACE Threads::Threads)
endif()

include_directories("${PROJECT_SOURCE_DIR}/main" "${PROJECT_SOURCE_DIR}/submods/Pgenlib")

add_subdirectory(tests)
