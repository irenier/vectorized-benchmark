cmake_minimum_required(VERSION 4.2.0)

project(SIMD LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
    armadillo
    GIT_REPOSITORY https://gitlab.com/conradsnicta/armadillo-code.git
    GIT_TAG 15.2.x
)

FetchContent_Declare(
    xsimd
    GIT_REPOSITORY https://github.com/xtensor-stack/xsimd.git
    GIT_TAG 14.0.0
)

FetchContent_Declare(
    highway
    GIT_REPOSITORY https://github.com/google/highway.git
    GIT_TAG 1.3.0
)

FetchContent_MakeAvailable(armadillo highway xsimd)
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

# target_include_directories(project_config INTERFACE
#     $<$<PLATFORM_ID:Darwin>:$<BUILD_INTERFACE:/opt/homebrew/opt/libomp/include>>
# )

target_link_libraries(project_config INTERFACE
    OpenMP::OpenMP_CXX
    xsimd
    TBB::tbb
    benchmark::benchmark
    Eigen3::Eigen
    # armadillo
    hwy
)

set(MAIN_SOURCES
    src/main.cpp
)

foreach(SOURCE_FILE IN LISTS MAIN_SOURCES)
    get_filename_component(TARGET_NAME ${SOURCE_FILE} NAME_WE)
    add_executable(${TARGET_NAME} ${SOURCE_FILE})
    target_link_libraries(${TARGET_NAME} PRIVATE project_config)
endforeach()