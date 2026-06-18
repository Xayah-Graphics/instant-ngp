cmake_minimum_required(VERSION 4.3.0)
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "451f2fe2-a8a2-47c3-bc32-94786d8fc91b")
project(InstantNGP VERSION 1.0.0 LANGUAGES CXX CUDA)

option(NGP_BUILD_BENCHMARKS "Build optional benchmark runners" OFF)

# ============================================================================
# C++ Configuration
# ============================================================================
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API ON)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)


# ============================================================================
# CUDA Configuration
# ============================================================================
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (MSVC)
    add_compile_options(
            $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    )
endif ()

# ============================================================================
# Dependencies
# ============================================================================
find_package(CUDAToolkit 13.2 REQUIRED)

set(NGP_TRAIN_PROFILE "baseline" CACHE STRING "Compile-time Instant NGP train profile")
set(NGP_TRAIN_PROFILES
        baseline
        batch_network_2e17
        batch_network_2e19
        batch_samples_x8
        batch_samples_x24
        batch_rays_2e11
        batch_rays_2e13
        density_skip_8
        density_skip_32
        density_uniform_div2
        density_uniform_div8
        density_nonuniform_div2
        density_nonuniform_div8
        density_decay_090
        density_decay_099
        raymarch_steps_512
        raymarch_steps_2048
        raymarch_random_samples_8
        raymarch_random_samples_32
        raymarch_transmittance_1em3
        raymarch_transmittance_1em5
        raymarch_thickness_0p005
        raymarch_thickness_0p02
        hash_log2_18
        hash_log2_20
        hash_base_8
        hash_base_32
        optimizer_lr_5em3
        optimizer_lr_2em2
        optimizer_l2_0
        optimizer_l2_1em5
        optimizer_loss_scale_64
        optimizer_loss_scale_256
        kernel_threads_256
        kernel_grid_forward_256
        kernel_grid_backward_128
        kernel_grid_backward_512
        kernel_mlp_forward_iters_4
)
set_property(CACHE NGP_TRAIN_PROFILE PROPERTY STRINGS ${NGP_TRAIN_PROFILES})
if (NOT NGP_TRAIN_PROFILE IN_LIST NGP_TRAIN_PROFILES)
    message(FATAL_ERROR "Unknown NGP_TRAIN_PROFILE='${NGP_TRAIN_PROFILE}'. Supported profiles: ${NGP_TRAIN_PROFILES}")
endif ()
string(MAKE_C_IDENTIFIER "${NGP_TRAIN_PROFILE}" NGP_TRAIN_PROFILE_DEFINITION_SUFFIX)
string(TOUPPER "${NGP_TRAIN_PROFILE_DEFINITION_SUFFIX}" NGP_TRAIN_PROFILE_DEFINITION_SUFFIX)

function(download_if_needed url out_path)
    if (EXISTS "${out_path}")
        message(STATUS "Using cached file: ${out_path}")
        return()
    endif ()

    get_filename_component(out_dir "${out_path}" DIRECTORY)
    file(MAKE_DIRECTORY "${out_dir}")

    message(STATUS "Downloading ${url}")
    file(DOWNLOAD
            "${url}"
            "${out_path}"
            TLS_VERIFY ON
            STATUS dl_status
    )

    list(GET dl_status 0 dl_code)
    list(GET dl_status 1 dl_msg)
    if (NOT dl_code EQUAL 0)
        message(FATAL_ERROR "Failed to download ${url}: [${dl_code}] ${dl_msg}")
    endif ()
endfunction()

download_if_needed(
        "https://github.com/nlohmann/json/releases/download/v3.12.0/json.hpp"
        "${CMAKE_CURRENT_BINARY_DIR}/vendor/json/json.hpp"
)
download_if_needed(
        "https://raw.githubusercontent.com/nothings/stb/28d546d5eb77d4585506a20480f4de2e706dff4c/stb_image_write.h"
        "${CMAKE_CURRENT_BINARY_DIR}/vendor/stb/stb_image_write.h"
)

# ============================================================================
# Core Library
# ============================================================================
add_library(ngp-train)
target_sources(
        ngp-train
        PRIVATE
        ngp.train.cpp
        ngp.train.cu
        ngp.train.config.h
        ngp.train.h
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        ngp.train.ixx
)
target_include_directories(ngp-train PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/vendor")
target_compile_definitions(ngp-train PRIVATE "NGP_TRAIN_PROFILE_${NGP_TRAIN_PROFILE_DEFINITION_SUFFIX}=1")
target_link_libraries(ngp-train PRIVATE CUDA::cublas CUDA::cublasLt)
set_property(TARGET ngp-train PROPERTY CXX_MODULE_STD ON)


# ============================================================================
# Optional Benchmarks
# ============================================================================
if (NGP_BUILD_BENCHMARKS)
    download_if_needed(
            "https://raw.githubusercontent.com/nothings/stb/28d546d5eb77d4585506a20480f4de2e706dff4c/stb_image.h"
            "${CMAKE_CURRENT_BINARY_DIR}/vendor/stb/stb_image.h"
    )

    add_library(benchmark-xcli)
    target_sources(
            benchmark-xcli
            PRIVATE
            benchmarks/xcli.cpp
            PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            benchmarks/xcli.ixx
    )
    set_property(TARGET benchmark-xcli PROPERTY CXX_MODULE_STD ON)

    add_library(benchmark-ngp-dataset)
    target_sources(
            benchmark-ngp-dataset
            PRIVATE
            benchmarks/ngp.dataset.cpp
            PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            benchmarks/ngp.dataset.ixx
    )
    target_include_directories(benchmark-ngp-dataset PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/vendor")
    set_property(TARGET benchmark-ngp-dataset PROPERTY CXX_MODULE_STD ON)

    foreach (profile IN LISTS NGP_TRAIN_PROFILES)
        string(MAKE_C_IDENTIFIER "${profile}" profile_definition_suffix)
        string(TOUPPER "${profile_definition_suffix}" profile_definition_suffix)

        add_library(ngp-train-${profile})
        target_sources(
                ngp-train-${profile}
                PRIVATE
                ngp.train.cpp
                ngp.train.cu
                ngp.train.config.h
                ngp.train.h
                PUBLIC
                FILE_SET cxx_modules TYPE CXX_MODULES FILES
                ngp.train.ixx
        )
        target_include_directories(ngp-train-${profile} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/vendor")
        target_compile_definitions(ngp-train-${profile} PRIVATE "NGP_TRAIN_PROFILE_${profile_definition_suffix}=1")
        target_link_libraries(ngp-train-${profile} PRIVATE CUDA::cublas CUDA::cublasLt)
        set_property(TARGET ngp-train-${profile} PROPERTY CXX_MODULE_STD ON)

        add_executable(instant-ngp-benchmark-${profile} benchmarks/main.cpp)
        target_include_directories(instant-ngp-benchmark-${profile} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/vendor")
        target_compile_definitions(instant-ngp-benchmark-${profile} PRIVATE "NGP_TRAIN_PROFILE_NAME=\"${profile}\"")
        target_link_libraries(instant-ngp-benchmark-${profile} PRIVATE benchmark-xcli benchmark-ngp-dataset ngp-train-${profile} CUDA::cudart)
        set_property(TARGET instant-ngp-benchmark-${profile} PROPERTY CXX_MODULE_STD ON)
    endforeach ()
endif ()
