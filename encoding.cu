#include "encoding.cuh"

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace ngp {

    __global__ void hash_grid_encoding_kernel(
        const float* input,
        std::uint32_t batch_size,
        std::uint32_t input_dimensions,
        std::uint32_t level_count,
        std::uint32_t features_per_level,
        std::uint32_t base_resolution,
        float log2_per_level_scale,
        HashGridEncoding::Storage storage,
        const std::uint32_t* level_offsets,
        const __half* parameters,
        __half* output
    ) {
        const std::uint32_t sample_index = blockIdx.x * blockDim.x + threadIdx.x;
        const std::uint32_t level_index = blockIdx.y;
        if (sample_index >= batch_size || level_index >= level_count) {
            return;
        }

        float local_position[4] = {};
        std::uint32_t local_grid_position[4] = {};
        for (std::uint32_t dimension_index = 0; dimension_index < input_dimensions; ++dimension_index) {
            float scaled_position = fmaf(
                exp2f(static_cast<float>(level_index) * log2_per_level_scale) * static_cast<float>(base_resolution) - 1.0f,
                input[dimension_index * batch_size + sample_index],
                0.5f
            );
            const float floored_position = floorf(scaled_position);
            local_grid_position[dimension_index] = static_cast<std::uint32_t>(static_cast<int>(floored_position));
            local_position[dimension_index] = scaled_position - floored_position;
        }

        const float scale = exp2f(static_cast<float>(level_index) * log2_per_level_scale) * static_cast<float>(base_resolution) - 1.0f;
        const std::uint32_t resolution = static_cast<std::uint32_t>(ceilf(scale)) + 1u;
        const std::uint32_t hashmap_size = level_offsets[level_index + 1u] - level_offsets[level_index];
        float interpolated_features[8] = {};

        const std::uint32_t corner_count = 1u << input_dimensions;
        for (std::uint32_t corner_index = 0; corner_index < corner_count; ++corner_index) {
            float weight = 1.0f;
            std::uint32_t grid_corner_position[4] = {};

            for (std::uint32_t dimension_index = 0; dimension_index < input_dimensions; ++dimension_index) {
                if ((corner_index & (1u << dimension_index)) == 0u) {
                    weight *= 1.0f - local_position[dimension_index];
                    grid_corner_position[dimension_index] = local_grid_position[dimension_index];
                } else {
                    weight *= local_position[dimension_index];
                    grid_corner_position[dimension_index] = local_grid_position[dimension_index] + 1u;
                }
            }

            std::uint32_t dense_stride = 1u;
            std::uint32_t dense_index = 0u;
            constexpr std::uint32_t max_bases[] = {
                0x0u,
                0xFFFFFFFFu,
                0xFFFFu,
                0x659u,
                0xFFu,
                0x54u,
                0x28u,
                0x17u,
                0xFu,
                0xBu,
                0x9u,
            };

            if (resolution <= max_bases[input_dimensions]) {
                for (std::uint32_t dimension_index = 0; dimension_index < input_dimensions; ++dimension_index) {
                    dense_index += grid_corner_position[dimension_index] * dense_stride;
                    dense_stride *= resolution;
                }
            } else {
                dense_stride = 0xFFFFFFFFu;
            }

            std::uint32_t final_index = dense_index;
            if (storage == HashGridEncoding::Storage::Hash && hashmap_size < dense_stride) {
                constexpr std::uint32_t coherent_prime_factors[7] = {
                    1u,
                    2654435761u,
                    805459861u,
                    3674653429u,
                    2097192037u,
                    1434869437u,
                    2165219737u,
                };
                final_index = 0u;
                for (std::uint32_t dimension_index = 0; dimension_index < input_dimensions; ++dimension_index) {
                    final_index ^= grid_corner_position[dimension_index] * coherent_prime_factors[dimension_index];
                }
            }

            const std::uint32_t parameter_base = (level_offsets[level_index] + (final_index % hashmap_size)) * features_per_level;
            for (std::uint32_t feature_index = 0; feature_index < features_per_level; ++feature_index) {
                interpolated_features[feature_index] += weight * __half2float(parameters[parameter_base + feature_index]);
            }
        }

        const std::uint32_t output_base = level_index * features_per_level;
        for (std::uint32_t feature_index = 0; feature_index < features_per_level; ++feature_index) {
            output[(output_base + feature_index) * batch_size + sample_index] = __float2half_rn(interpolated_features[feature_index]);
        }
    }

    __global__ void spherical_harmonics_encoding_kernel(const float* input, std::uint32_t batch_size, std::uint32_t degree, __half* output) {
        const std::uint32_t sample_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (sample_index >= batch_size) {
            return;
        }

        const float x = input[sample_index];
        const float y = input[batch_size + sample_index];
        const float z = input[batch_size * 2u + sample_index];

        const float xy = x * y;
        const float xz = x * z;
        const float yz = y * z;
        const float x2 = x * x;
        const float y2 = y * y;
        const float z2 = z * z;
        const float x4 = x2 * x2;
        const float y4 = y2 * y2;
        const float z4 = z2 * z2;
        const float x6 = x4 * x2;
        const float y6 = y4 * y2;
        const float z6 = z4 * z2;

        output[sample_index] = __float2half_rn(0.28209479177387814f);
        if (degree <= 1u) {
            return;
        }

        output[batch_size + sample_index] = __float2half_rn(-0.48860251190291987f * y);
        output[batch_size * 2u + sample_index] = __float2half_rn(0.48860251190291987f * z);
        output[batch_size * 3u + sample_index] = __float2half_rn(-0.48860251190291987f * x);
        if (degree <= 2u) {
            return;
        }

        output[batch_size * 4u + sample_index] = __float2half_rn(1.0925484305920792f * xy);
        output[batch_size * 5u + sample_index] = __float2half_rn(-1.0925484305920792f * yz);
        output[batch_size * 6u + sample_index] = __float2half_rn(0.94617469575755997f * z2 - 0.31539156525251999f);
        output[batch_size * 7u + sample_index] = __float2half_rn(-1.0925484305920792f * xz);
        output[batch_size * 8u + sample_index] = __float2half_rn(0.54627421529603959f * x2 - 0.54627421529603959f * y2);
        if (degree <= 3u) {
            return;
        }

        output[batch_size * 9u + sample_index] = __float2half_rn(0.59004358992664352f * y * (-3.0f * x2 + y2));
        output[batch_size * 10u + sample_index] = __float2half_rn(2.8906114426405538f * xy * z);
        output[batch_size * 11u + sample_index] = __float2half_rn(0.45704579946446572f * y * (1.0f - 5.0f * z2));
        output[batch_size * 12u + sample_index] = __float2half_rn(0.3731763325901154f * z * (5.0f * z2 - 3.0f));
        output[batch_size * 13u + sample_index] = __float2half_rn(0.45704579946446572f * x * (1.0f - 5.0f * z2));
        output[batch_size * 14u + sample_index] = __float2half_rn(1.4453057213202769f * z * (x2 - y2));
        output[batch_size * 15u + sample_index] = __float2half_rn(0.59004358992664352f * x * (-x2 + 3.0f * y2));
        if (degree <= 4u) {
            return;
        }

        output[batch_size * 16u + sample_index] = __float2half_rn(2.5033429417967046f * xy * (x2 - y2));
        output[batch_size * 17u + sample_index] = __float2half_rn(1.7701307697799304f * yz * (-3.0f * x2 + y2));
        output[batch_size * 18u + sample_index] = __float2half_rn(0.94617469575756008f * xy * (7.0f * z2 - 1.0f));
        output[batch_size * 19u + sample_index] = __float2half_rn(0.66904654355728921f * yz * (3.0f - 7.0f * z2));
        output[batch_size * 20u + sample_index] = __float2half_rn(-3.1735664074561294f * z2 + 3.7024941420321507f * z4 + 0.31735664074561293f);
        output[batch_size * 21u + sample_index] = __float2half_rn(0.66904654355728921f * xz * (3.0f - 7.0f * z2));
        output[batch_size * 22u + sample_index] = __float2half_rn(0.47308734787878004f * (x2 - y2) * (7.0f * z2 - 1.0f));
        output[batch_size * 23u + sample_index] = __float2half_rn(1.7701307697799304f * xz * (-x2 + 3.0f * y2));
        output[batch_size * 24u + sample_index] = __float2half_rn(-3.7550144126950569f * x2 * y2 + 0.62583573544917614f * x4 + 0.62583573544917614f * y4);
        if (degree <= 5u) {
            return;
        }

        output[batch_size * 25u + sample_index] = __float2half_rn(0.65638205684017015f * y * (10.0f * x2 * y2 - 5.0f * x4 - y4));
        output[batch_size * 26u + sample_index] = __float2half_rn(8.3026492595241645f * xy * z * (x2 - y2));
        output[batch_size * 27u + sample_index] = __float2half_rn(-0.48923829943525038f * y * (3.0f * x2 - y2) * (9.0f * z2 - 1.0f));
        output[batch_size * 28u + sample_index] = __float2half_rn(4.7935367849733241f * xy * z * (3.0f * z2 - 1.0f));
        output[batch_size * 29u + sample_index] = __float2half_rn(0.45294665119569694f * y * (14.0f * z2 - 21.0f * z4 - 1.0f));
        output[batch_size * 30u + sample_index] = __float2half_rn(0.1169503224534236f * z * (-70.0f * z2 + 63.0f * z4 + 15.0f));
        output[batch_size * 31u + sample_index] = __float2half_rn(0.45294665119569694f * x * (14.0f * z2 - 21.0f * z4 - 1.0f));
        output[batch_size * 32u + sample_index] = __float2half_rn(2.3967683924866621f * z * (x2 - y2) * (3.0f * z2 - 1.0f));
        output[batch_size * 33u + sample_index] = __float2half_rn(-0.48923829943525038f * x * (x2 - 3.0f * y2) * (9.0f * z2 - 1.0f));
        output[batch_size * 34u + sample_index] = __float2half_rn(2.0756623148810411f * z * (-6.0f * x2 * y2 + x4 + y4));
        output[batch_size * 35u + sample_index] = __float2half_rn(0.65638205684017015f * x * (10.0f * x2 * y2 - x4 - 5.0f * y4));
        if (degree <= 6u) {
            return;
        }

        output[batch_size * 36u + sample_index] = __float2half_rn(1.3663682103838286f * xy * (-10.0f * x2 * y2 + 3.0f * x4 + 3.0f * y4));
        output[batch_size * 37u + sample_index] = __float2half_rn(2.3666191622317521f * yz * (10.0f * x2 * y2 - 5.0f * x4 - y4));
        output[batch_size * 38u + sample_index] = __float2half_rn(2.0182596029148963f * xy * (x2 - y2) * (11.0f * z2 - 1.0f));
        output[batch_size * 39u + sample_index] = __float2half_rn(-0.92120525951492349f * yz * (3.0f * x2 - y2) * (11.0f * z2 - 3.0f));
        output[batch_size * 40u + sample_index] = __float2half_rn(0.92120525951492349f * xy * (-18.0f * z2 + 33.0f * z4 + 1.0f));
        output[batch_size * 41u + sample_index] = __float2half_rn(0.58262136251873131f * yz * (30.0f * z2 - 33.0f * z4 - 5.0f));
        output[batch_size * 42u + sample_index] = __float2half_rn(6.6747662381009842f * z2 - 20.024298714302954f * z4 + 14.684485723822165f * z6 - 0.31784601133814211f);
        output[batch_size * 43u + sample_index] = __float2half_rn(0.58262136251873131f * xz * (30.0f * z2 - 33.0f * z4 - 5.0f));
        output[batch_size * 44u + sample_index] = __float2half_rn(0.46060262975746175f * (x2 - y2) * (11.0f * z2 * (3.0f * z2 - 1.0f) - 7.0f * z2 + 1.0f));
        output[batch_size * 45u + sample_index] = __float2half_rn(-0.92120525951492349f * xz * (x2 - 3.0f * y2) * (11.0f * z2 - 3.0f));
        output[batch_size * 46u + sample_index] = __float2half_rn(0.50456490072872406f * (11.0f * z2 - 1.0f) * (-6.0f * x2 * y2 + x4 + y4));
        output[batch_size * 47u + sample_index] = __float2half_rn(2.3666191622317521f * xz * (10.0f * x2 * y2 - x4 - 5.0f * y4));
        output[batch_size * 48u + sample_index] = __float2half_rn(10.247761577878714f * x2 * y4 - 10.247761577878714f * x4 * y2 + 0.6831841051919143f * x6 - 0.6831841051919143f * y6);
        if (degree <= 7u) {
            return;
        }

        output[batch_size * 49u + sample_index] = __float2half_rn(0.70716273252459627f * y * (-21.0f * x2 * y4 + 35.0f * x4 * y2 - 7.0f * x6 + y6));
        output[batch_size * 50u + sample_index] = __float2half_rn(5.2919213236038001f * xy * z * (-10.0f * x2 * y2 + 3.0f * x4 + 3.0f * y4));
        output[batch_size * 51u + sample_index] = __float2half_rn(-0.51891557872026028f * y * (13.0f * z2 - 1.0f) * (-10.0f * x2 * y2 + 5.0f * x4 + y4));
        output[batch_size * 52u + sample_index] = __float2half_rn(4.1513246297620823f * xy * z * (x2 - y2) * (13.0f * z2 - 3.0f));
        output[batch_size * 53u + sample_index] = __float2half_rn(-0.15645893386229404f * y * (3.0f * x2 - y2) * (13.0f * z2 * (11.0f * z2 - 3.0f) - 27.0f * z2 + 3.0f));
        output[batch_size * 54u + sample_index] = __float2half_rn(0.44253269244498261f * xy * z * (-110.0f * z2 + 143.0f * z4 + 15.0f));
        output[batch_size * 55u + sample_index] = __float2half_rn(0.090331607582517306f * y * (-135.0f * z2 + 495.0f * z4 - 429.0f * z6 + 5.0f));
        output[batch_size * 56u + sample_index] = __float2half_rn(0.068284276912004949f * z * (315.0f * z2 - 693.0f * z4 + 429.0f * z6 - 35.0f));
        output[batch_size * 57u + sample_index] = __float2half_rn(0.090331607582517306f * x * (-135.0f * z2 + 495.0f * z4 - 429.0f * z6 + 5.0f));
        output[batch_size * 58u + sample_index] = __float2half_rn(0.07375544874083044f * z * (x2 - y2) * (143.0f * z2 * (3.0f * z2 - 1.0f) - 187.0f * z2 + 45.0f));
        output[batch_size * 59u + sample_index] = __float2half_rn(-0.15645893386229404f * x * (x2 - 3.0f * y2) * (13.0f * z2 * (11.0f * z2 - 3.0f) - 27.0f * z2 + 3.0f));
        output[batch_size * 60u + sample_index] = __float2half_rn(1.0378311574405206f * z * (13.0f * z2 - 3.0f) * (-6.0f * x2 * y2 + x4 + y4));
        output[batch_size * 61u + sample_index] = __float2half_rn(-0.51891557872026028f * x * (13.0f * z2 - 1.0f) * (-10.0f * x2 * y2 + x4 + 5.0f * y4));
        output[batch_size * 62u + sample_index] = __float2half_rn(2.6459606618019f * z * (15.0f * x2 * y4 - 15.0f * x4 * y2 + x6 - y6));
        output[batch_size * 63u + sample_index] = __float2half_rn(0.70716273252459627f * x * (-35.0f * x2 * y4 + 21.0f * x4 * y2 - x6 + 7.0f * y6));
    }

    HashGridEncoding::HashGridEncoding(const Config& config) : config_{config} {
        if (config_.input_dimensions < 2u || config_.input_dimensions > 4u) {
            throw std::runtime_error{"HashGridEncoding requires input_dimensions in {2, 3, 4}."};
        }
        if (config_.level_count == 0u) {
            throw std::runtime_error{"HashGridEncoding requires level_count > 0."};
        }
        if (config_.features_per_level != 1u && config_.features_per_level != 2u && config_.features_per_level != 4u && config_.features_per_level != 8u) {
            throw std::runtime_error{"HashGridEncoding requires features_per_level in {1, 2, 4, 8}."};
        }
        if (config_.base_resolution == 0u) {
            throw std::runtime_error{"HashGridEncoding requires base_resolution > 0."};
        }
        if (!(config_.per_level_scale > 0.0f)) {
            throw std::runtime_error{"HashGridEncoding requires per_level_scale > 0."};
        }
        if (config_.storage == Storage::Hash && config_.log2_hashmap_size >= 31u) {
            throw std::runtime_error{"HashGridEncoding requires log2_hashmap_size < 31 for hash storage."};
        }

        output_width_ = config_.level_count * config_.features_per_level;
        log2_per_level_scale_ = std::log2(config_.per_level_scale);

        level_offsets_cpu_.resize(config_.level_count + 1u);
        std::uint32_t running_offset = 0u;
        for (std::uint32_t level_index = 0; level_index < config_.level_count; ++level_index) {
            const float scale = std::exp2(static_cast<float>(level_index) * log2_per_level_scale_) * static_cast<float>(config_.base_resolution) - 1.0f;
            const std::uint32_t resolution = static_cast<std::uint32_t>(std::ceil(scale)) + 1u;

            std::uint64_t params_in_level = 1u;
            for (std::uint32_t dimension_index = 0; dimension_index < config_.input_dimensions; ++dimension_index) {
                params_in_level *= resolution;
                if (params_in_level > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) / 2u) {
                    throw std::runtime_error{"HashGridEncoding level resolution overflows the supported parameter index range."};
                }
            }

            params_in_level = ((params_in_level + 7u) / 8u) * 8u;
            if (config_.storage == Storage::Tiled) {
                std::uint64_t tiled_limit = 1u;
                for (std::uint32_t dimension_index = 0; dimension_index < config_.input_dimensions; ++dimension_index) {
                    tiled_limit *= config_.base_resolution;
                    if (tiled_limit > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) {
                        throw std::runtime_error{"HashGridEncoding tiled storage overflows the supported parameter index range."};
                    }
                }
                params_in_level = std::min(params_in_level, tiled_limit);
            } else if (config_.storage == Storage::Hash) {
                params_in_level = std::min(params_in_level, static_cast<std::uint64_t>(1u) << config_.log2_hashmap_size);
            }

            if (params_in_level > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) - running_offset) {
                throw std::runtime_error{"HashGridEncoding total parameter offsets overflow the supported range."};
            }

            level_offsets_cpu_[level_index] = running_offset;
            running_offset += static_cast<std::uint32_t>(params_in_level);
        }
        level_offsets_cpu_[config_.level_count] = running_offset;

        parameter_count_ = static_cast<std::size_t>(running_offset) * static_cast<std::size_t>(config_.features_per_level);
        if (parameter_count_ == 0u) {
            throw std::runtime_error{"HashGridEncoding computed an empty parameter buffer."};
        }

        parameters_.resize(parameter_count_);
        level_offsets_gpu_.resize(level_offsets_cpu_.size());
        level_offsets_gpu_.copy_from_host(level_offsets_cpu_.data());
    }

    void HashGridEncoding::upload_parameters(std::span<const float> host_parameters) {
        if (host_parameters.size() != parameter_count_) {
            throw std::runtime_error{"HashGridEncoding parameter upload size does not match parameter_count()."};
        }

        std::vector<__half> converted_parameters(parameter_count_);
        for (std::size_t parameter_index = 0; parameter_index < parameter_count_; ++parameter_index) {
            converted_parameters[parameter_index] = __float2half_rn(host_parameters[parameter_index]);
        }

        parameters_.copy_from_host(converted_parameters.data());
        has_parameters_ = true;
    }

    void HashGridEncoding::initialize_parameters(std::uint64_t seed, float scale) {
        if (!(scale > 0.0f)) {
            throw std::runtime_error{"HashGridEncoding requires scale > 0."};
        }

        std::vector<float> host_parameters(parameter_count_);
        std::mt19937_64 random_engine(seed);
        std::uniform_real_distribution<float> distribution(-1e-4f * scale, 1e-4f * scale);
        for (float& parameter : host_parameters) {
            parameter = distribution(random_engine);
        }

        upload_parameters(host_parameters);
    }

    void HashGridEncoding::encode(cudaStream_t stream, const float* input, std::uint32_t batch_size, __half* output) const {
        if (input == nullptr) {
            throw std::runtime_error{"HashGridEncoding requires a non-null input pointer."};
        }
        if (output == nullptr) {
            throw std::runtime_error{"HashGridEncoding requires a non-null output pointer."};
        }
        if (!has_parameters_) {
            throw std::runtime_error{"HashGridEncoding requires parameters to be uploaded or initialized before encode()."};
        }
        if (batch_size == 0u) {
            throw std::runtime_error{"HashGridEncoding requires batch_size > 0."};
        }
        if (batch_size % required_batch_granularity() != 0u) {
            throw std::runtime_error{"HashGridEncoding requires batch_size to be a multiple of 256."};
        }

        constexpr std::uint32_t block_size = 256u;
        const dim3 blocks{
            (batch_size + block_size - 1u) / block_size,
            config_.level_count,
            1u,
        };
        hash_grid_encoding_kernel<<<blocks, block_size, 0, stream>>>(
            input,
            batch_size,
            config_.input_dimensions,
            config_.level_count,
            config_.features_per_level,
            config_.base_resolution,
            log2_per_level_scale_,
            config_.storage,
            level_offsets_gpu_.data(),
            parameters_.data(),
            output
        );

        const cudaError_t launch_status = cudaGetLastError();
        if (launch_status != cudaSuccess) {
            throw std::runtime_error{std::string{"HashGridEncoding launch failed: "} + cudaGetErrorString(launch_status)};
        }
    }

    SphericalHarmonicsEncoding::SphericalHarmonicsEncoding(const Config& config) : config_{config} {
        if (config_.input_dimensions != 3u) {
            throw std::runtime_error{"SphericalHarmonicsEncoding requires input_dimensions == 3."};
        }
        if (config_.degree == 0u) {
            throw std::runtime_error{"SphericalHarmonicsEncoding requires degree > 0."};
        }
        if (config_.degree > 8u) {
            throw std::runtime_error{"SphericalHarmonicsEncoding only supports degree <= 8."};
        }

        output_width_ = config_.degree * config_.degree;
    }

    void SphericalHarmonicsEncoding::encode(cudaStream_t stream, const float* input, std::uint32_t batch_size, __half* output) const {
        if (input == nullptr) {
            throw std::runtime_error{"SphericalHarmonicsEncoding requires a non-null input pointer."};
        }
        if (output == nullptr) {
            throw std::runtime_error{"SphericalHarmonicsEncoding requires a non-null output pointer."};
        }
        if (batch_size == 0u) {
            throw std::runtime_error{"SphericalHarmonicsEncoding requires batch_size > 0."};
        }
        if (batch_size % required_batch_granularity() != 0u) {
            throw std::runtime_error{"SphericalHarmonicsEncoding requires batch_size to be a multiple of 256."};
        }

        constexpr std::uint32_t block_size = 256u;
        const std::uint32_t block_count = (batch_size + block_size - 1u) / block_size;
        spherical_harmonics_encoding_kernel<<<block_count, block_size, 0, stream>>>(input, batch_size, config_.degree, output);

        const cudaError_t launch_status = cudaGetLastError();
        if (launch_status != cudaSuccess) {
            throw std::runtime_error{std::string{"SphericalHarmonicsEncoding launch failed: "} + cudaGetErrorString(launch_status)};
        }
    }

} // namespace ngp
