cmake_minimum_required(VERSION 3.18.0)

project(SIMD LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# FetchContent_Declare(
#     armadillo
#     GIT_REPOSITORY https://gitlab.com/conradsnicta/armadillo-code.git
#     GIT_TAG 15.2.x
# )

FetchContent_Declare(
    xsimd
    GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
    GIT_TAG 14.0.0
    GIT_SHALLOW TRUE
)

FetchContent_Declare(
    highway
    GIT_REPOSITORY https://github.com/google/highway.git
    GIT_TAG 1.4.0
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(highway xsimd)
find_package(OpenMP REQUIRED)
find_package(TBB REQUIRED)
find_package(benchmark REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(project_config INTERFACE)

target_compile_options(project_config INTERFACE
    -O3
    -march=native
    -Wall
    -Wextra
)

target_include_directories(project_config INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    # $<$<PLATFORM_ID:Darwin>:$<BUILD_INTERFACE:/opt/homebrew/opt/libomp/include>>
)

target_link_libraries(project_config INTERFACE
    OpenMP::OpenMP_CXX
    xsimd
    TBB::tbb
    benchmark::benchmark
    Eigen3::Eigen
    # armadillo
    hwy
    hwy_contrib
)

set(MAIN_SOURCES
    src/bench_axpy.cpp
    src/bench_dot.cpp
    src/bench_nrm2.cpp
    src/bench_sumexp.cpp
)

foreach(SOURCE_FILE IN LISTS MAIN_SOURCES)
    get_filename_component(TARGET_NAME ${SOURCE_FILE} NAME_WE)
    add_executable(${TARGET_NAME} ${SOURCE_FILE})
    target_link_libraries(${TARGET_NAME} PRIVATE project_config)
endforeach()
