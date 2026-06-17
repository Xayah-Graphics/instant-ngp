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
set_property(CACHE NGP_TRAIN_PROFILE PROPERTY STRINGS baseline)
set(NGP_TRAIN_PROFILES baseline)
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
add_library(xcli)
target_sources(
        xcli
        PRIVATE
        xcli.cpp
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        xcli.ixx
)
set_property(TARGET xcli PROPERTY CXX_MODULE_STD ON)

add_library(ngp-dataset)
target_sources(
        ngp-dataset
        PRIVATE
        ngp.dataset.cpp
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        ngp.dataset.ixx
)
target_include_directories(ngp-dataset PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/vendor")
set_property(TARGET ngp-dataset PROPERTY CXX_MODULE_STD ON)

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
# Executable
# ============================================================================
add_executable(instant-ngp-app main.cpp)
target_link_libraries(instant-ngp-app PRIVATE xcli ngp-dataset ngp-train)
set_property(TARGET instant-ngp-app PROPERTY CXX_MODULE_STD ON)
