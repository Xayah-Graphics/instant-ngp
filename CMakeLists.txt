cmake_minimum_required(VERSION 4.3.0)
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "451f2fe2-a8a2-47c3-bc32-94786d8fc91b")
project(InstantNGP VERSION 1.0.0 LANGUAGES CXX CUDA)


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

set(NGP_DENSITY_NETWORK_WIDTH 64 CACHE STRING "Compile-time width of the density FullyFusedMLP.")
set(NGP_RGB_NETWORK_WIDTH 64 CACHE STRING "Compile-time width of the RGB FullyFusedMLP.")
if (NOT NGP_DENSITY_NETWORK_WIDTH MATCHES "^(16|32|64|128)$")
    message(FATAL_ERROR "NGP_DENSITY_NETWORK_WIDTH must be one of 16, 32, 64, or 128.")
endif ()
if (NOT NGP_RGB_NETWORK_WIDTH MATCHES "^(16|32|64|128)$")
    message(FATAL_ERROR "NGP_RGB_NETWORK_WIDTH must be one of 16, 32, 64, or 128.")
endif ()


# ============================================================================
# Dependencies
# ============================================================================
find_package(CUDAToolkit 13.2 REQUIRED)

set(NGP_CUTLASS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../instant-ngp/dependencies/cutlass" CACHE PATH "Pinned CUTLASS root for the network module.")
if (NOT EXISTS "${NGP_CUTLASS_ROOT}/include/cutlass/cutlass.h")
    message(FATAL_ERROR "CUTLASS headers are missing at '${NGP_CUTLASS_ROOT}'.")
endif ()
if (NOT EXISTS "${NGP_CUTLASS_ROOT}/tools/util/include")
    message(FATAL_ERROR "CUTLASS utility headers are missing at '${NGP_CUTLASS_ROOT}/tools/util/include'.")
endif ()

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
        "https://raw.githubusercontent.com/nothings/stb/28d546d5eb77d4585506a20480f4de2e706dff4c/stb_image.h"
        "${CMAKE_CURRENT_BINARY_DIR}/vendor/stb/stb_image.h"
)
download_if_needed(
        "https://raw.githubusercontent.com/nothings/stb/28d546d5eb77d4585506a20480f4de2e706dff4c/stb_image_write.h"
        "${CMAKE_CURRENT_BINARY_DIR}/vendor/stb/stb_image_write.h"
)

# ============================================================================
# Libraries
# ============================================================================
add_library(instant-ngp)
target_sources(
        instant-ngp
        PRIVATE
        instant-ngp.cpp
        instant-ngp.cu
        network.cu
        PUBLIC
        instant-ngp.h
)
target_link_libraries(instant-ngp PUBLIC CUDA::cudart CUDA::cuda_driver CUDA::cublas)
target_compile_definitions(
        instant-ngp
        PRIVATE
        NGP_DENSITY_NETWORK_WIDTH=${NGP_DENSITY_NETWORK_WIDTH}
        NGP_RGB_NETWORK_WIDTH=${NGP_RGB_NETWORK_WIDTH}
)
target_compile_options(
        instant-ngp
        PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)
target_include_directories(
        instant-ngp
        PRIVATE
        "${NGP_CUTLASS_ROOT}/include"
        "${NGP_CUTLASS_ROOT}/tools/util/include"
        "${CMAKE_CURRENT_BINARY_DIR}/vendor"
)
set_property(TARGET instant-ngp PROPERTY CXX_MODULE_STD ON)


# ============================================================================
# Executable
# ============================================================================
add_executable(instant-ngp-app main.cpp)
target_link_libraries(instant-ngp-app PRIVATE instant-ngp)
target_include_directories(instant-ngp-app PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/vendor")
set_property(TARGET instant-ngp-app PROPERTY CXX_MODULE_STD ON)
