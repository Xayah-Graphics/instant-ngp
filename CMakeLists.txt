cmake_minimum_required(VERSION 4.4.0)
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD "f35a9ac6-8463-4d38-8eec-5d6008153e7d")
project(InstantNGP VERSION 1.2.1 LANGUAGES CXX CUDA)

option(NGP_BUILD_APP "Build the optional Instant NGP executable" OFF)
option(NGP_BUILD_SPECTRA_PROVIDER "Build the Spectra Neural Field Provider" OFF)

if (NGP_BUILD_SPECTRA_PROVIDER AND NOT WIN32)
    message(FATAL_ERROR "The Spectra Neural Field Provider is supported only on Windows")
endif ()

# ============================================================================
# C++ Configuration
# ============================================================================
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


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

function(download_if_needed url out_path sha256)
    string(TOLOWER "${sha256}" expected_sha256)
    if (EXISTS "${out_path}")
        file(SHA256 "${out_path}" existing_sha256)
        string(TOLOWER "${existing_sha256}" existing_sha256)
        if (NOT existing_sha256 STREQUAL expected_sha256)
            message(FATAL_ERROR "Cached file hash mismatch for ${out_path}: expected ${expected_sha256}, got ${existing_sha256}")
        endif ()
        message(STATUS "Using cached file: ${out_path}")
        return()
    endif ()

    get_filename_component(out_dir "${out_path}" DIRECTORY)
    file(MAKE_DIRECTORY "${out_dir}")

    message(STATUS "Downloading ${url}")
    set(tmp_path "${out_path}.download")
    file(REMOVE "${tmp_path}")
    file(DOWNLOAD
            "${url}"
            "${tmp_path}"
            TLS_VERIFY ON
            EXPECTED_HASH "SHA256=${sha256}"
            STATUS dl_status
    )

    list(GET dl_status 0 dl_code)
    list(GET dl_status 1 dl_msg)
    if (NOT dl_code EQUAL 0)
        file(REMOVE "${tmp_path}")
        message(FATAL_ERROR "Failed to download ${url}: [${dl_code}] ${dl_msg}")
    endif ()
    file(RENAME "${tmp_path}" "${out_path}")
endfunction()

download_if_needed(
        "https://github.com/nlohmann/json/releases/download/v3.12.0/json.hpp"
        "${CMAKE_CURRENT_BINARY_DIR}/vendor/json/json.hpp"
        "AAF127C04CB31C406E5B04A63F1AE89369FCCDE6D8FA7CDDA1ED4F32DFC5DE63"
)
download_if_needed(
        "https://raw.githubusercontent.com/nothings/stb/28d546d5eb77d4585506a20480f4de2e706dff4c/stb_image_write.h"
        "${CMAKE_CURRENT_BINARY_DIR}/vendor/stb/stb_image_write.h"
        "CBD5F0AD7A9CF4468AFFB36354A1D2338034F2C12473CF1A8E32053CB6914A05"
)

# ============================================================================
# Core Library
# ============================================================================
add_library(libngp)
target_sources(
        libngp
        PRIVATE
        ngp.train.cpp
        ngp.train.cu
        ngp.train.config.h
        ngp.train.h
        PUBLIC
        FILE_SET cxx_modules TYPE CXX_MODULES FILES
        ngp.train.ixx
)
target_include_directories(libngp PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/vendor")
target_compile_definitions(libngp PRIVATE "NGP_TRAIN_PROFILE_${NGP_TRAIN_PROFILE_DEFINITION_SUFFIX}=1")
target_link_libraries(libngp PRIVATE CUDA::cublas CUDA::cublasLt)
set_property(TARGET libngp PROPERTY CXX_MODULE_STD ON)


# ============================================================================
# Optional Dataset Loading
# ============================================================================
if (NGP_BUILD_APP OR NGP_BUILD_SPECTRA_PROVIDER)
    set(NGP_DATASET_COMMIT "bc636fbfc7507e8173156393f71c0b569f5fe95b")
    set(NGP_DATASET_VENDOR_DIR "${CMAKE_CURRENT_BINARY_DIR}/vendor/dataset/${NGP_DATASET_COMMIT}")
    download_if_needed(
            "https://raw.githubusercontent.com/Xayah-Graphics/dataset/${NGP_DATASET_COMMIT}/nerf_synthetic.ixx"
            "${NGP_DATASET_VENDOR_DIR}/nerf_synthetic.ixx"
            "0E2D0EB41BAD298103BBB873C5C4DA5BB3AF3A40E913DD92CB4BCF4CF621C179"
    )
    download_if_needed(
            "https://raw.githubusercontent.com/Xayah-Graphics/dataset/${NGP_DATASET_COMMIT}/nerf_synthetic.cpp"
            "${NGP_DATASET_VENDOR_DIR}/nerf_synthetic.cpp"
            "F786677A654C47CF7253BFB3F49D4A7E34112DF6E108E2A3C4C59DBF1B4BC27E"
    )
    download_if_needed(
            "https://raw.githubusercontent.com/Xayah-Graphics/dataset/${NGP_DATASET_COMMIT}/dd_nerf.ixx"
            "${NGP_DATASET_VENDOR_DIR}/dd_nerf.ixx"
            "FC0DD63059EA6743CBBD0CA3AD4401185AD0169B1F75013879DDD7BBAE29F216"
    )
    download_if_needed(
            "https://raw.githubusercontent.com/Xayah-Graphics/dataset/${NGP_DATASET_COMMIT}/dd_nerf.cpp"
            "${NGP_DATASET_VENDOR_DIR}/dd_nerf.cpp"
            "C41104BEB22DA837FE361EF3FEDFC91BFE99A6A52CA2F604A76D30CDB587967D"
    )
    download_if_needed(
            "https://raw.githubusercontent.com/nothings/stb/28d546d5eb77d4585506a20480f4de2e706dff4c/stb_image.h"
            "${CMAKE_CURRENT_BINARY_DIR}/vendor/stb/stb_image.h"
            "594C2FE35D49488B4382DBFAEC8F98366DEFCA819D916AC95BECF3E75F4200B3"
    )

    add_library(dataset-loader)
    target_sources(
            dataset-loader
            PRIVATE
            "${NGP_DATASET_VENDOR_DIR}/nerf_synthetic.cpp"
            "${NGP_DATASET_VENDOR_DIR}/dd_nerf.cpp"
            PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            "${NGP_DATASET_VENDOR_DIR}/nerf_synthetic.ixx"
            "${NGP_DATASET_VENDOR_DIR}/dd_nerf.ixx"
    )
    target_include_directories(dataset-loader PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/vendor")
    set_property(TARGET dataset-loader PROPERTY CXX_MODULE_STD ON)
endif ()


# ============================================================================
# Optional App
# ============================================================================
if (NGP_BUILD_APP)
    set(NGP_UTIL_COMMIT "bb261075e7c8a4722b23f42c3be01aa24024731a")
    set(NGP_UTIL_VENDOR_DIR "${CMAKE_CURRENT_BINARY_DIR}/vendor/util/${NGP_UTIL_COMMIT}")
    download_if_needed(
            "https://raw.githubusercontent.com/Xayah-Graphics/util/${NGP_UTIL_COMMIT}/xcli.ixx"
            "${NGP_UTIL_VENDOR_DIR}/xcli.ixx"
            "2F80115AA7B739BDE4BF990EC1C0D0AFD85CF6B5610AF9A9D22A8C6686898F8D"
    )
    download_if_needed(
            "https://raw.githubusercontent.com/Xayah-Graphics/util/${NGP_UTIL_COMMIT}/xcli.cpp"
            "${NGP_UTIL_VENDOR_DIR}/xcli.cpp"
            "A1386CE6782CF267CD9AFB002FA66E70B7BF019CFA19ADC9C5EF0470BE445E69"
    )

    add_library(xayah-util-xcli)
    target_sources(
            xayah-util-xcli
            PRIVATE
            "${NGP_UTIL_VENDOR_DIR}/xcli.cpp"
            PUBLIC
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            "${NGP_UTIL_VENDOR_DIR}/xcli.ixx"
    )
    set_property(TARGET xayah-util-xcli PROPERTY CXX_MODULE_STD ON)

    add_executable(instant_ngp main.cpp)
    target_include_directories(instant_ngp PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/vendor")
    target_compile_definitions(instant_ngp PRIVATE "NGP_TRAIN_PROFILE_NAME=\"${NGP_TRAIN_PROFILE}\"")
    target_link_libraries(instant_ngp PRIVATE xayah-util-xcli dataset-loader libngp CUDA::cudart)
    set_property(TARGET instant_ngp PROPERTY CXX_MODULE_STD ON)
    if (MSVC)
        target_link_options(instant_ngp PRIVATE "/WHOLEARCHIVE:${CMAKE_CURRENT_BINARY_DIR}/@cmake_cxx_std.lib")
    endif ()
endif ()


# ============================================================================
# Optional Spectra Provider
# ============================================================================
if (NGP_BUILD_SPECTRA_PROVIDER)
    find_package(SpectraSDK 2.0.5 CONFIG REQUIRED)

    spectra_add_provider(
            instant-ngp-provider
            MODULE ngp.provider
            TYPE ngp::Provider
    )
    target_sources(
            instant-ngp-provider
            PRIVATE
            FILE_SET cxx_modules TYPE CXX_MODULES FILES
            ngp.provider.ixx
    )
    target_include_directories(instant-ngp-provider PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/vendor")
    target_link_libraries(instant-ngp-provider PRIVATE dataset-loader libngp CUDA::cudart)
    set(bundle_directory "${PROJECT_BINARY_DIR}/spectra/instant-ngp")
    set_target_properties(
            instant-ngp-provider
            PROPERTIES
            CXX_MODULE_STD ON
            RUNTIME_OUTPUT_DIRECTORY "${bundle_directory}"
    )
    add_custom_target(
            instant-ngp-bundle
            COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different "${CMAKE_CURRENT_SOURCE_DIR}/spectra" "${bundle_directory}"
            SOURCES
            spectra/instant-ngp.spectra
            VERBATIM
    )
    add_dependencies(instant-ngp-provider instant-ngp-bundle)
    if (MSVC)
        target_link_options(instant-ngp-provider PRIVATE "/WHOLEARCHIVE:${CMAKE_CURRENT_BINARY_DIR}/@cmake_cxx_std.lib")
    endif ()
endif ()
