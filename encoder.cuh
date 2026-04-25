#pragma once

#include "instant-ngp.h"
#include <variant>

namespace ngp::encoding {

    enum class GridType {
        Hash,
        Dense,
        Tiled,
    };

    inline constexpr std::uint32_t max_n_levels = 128u;

    struct ParamsOffsetTable final {
        std::uint32_t data[max_n_levels + 1u] = {};
    };

    template <typename T, std::uint32_t N_POS_DIMS = 3u, std::uint32_t N_FEATURES_PER_LEVEL = 2u>
    struct GridEncodingTemplated final {
        GridEncodingTemplated(std::uint32_t n_features, std::uint32_t log2_hashmap_size, std::uint32_t base_resolution, float per_level_scale, bool stochastic_interpolation, GridType grid_type);

        void encode(cudaStream_t stream, const legacy::GPUMatrixDynamic<float>& input, legacy::GPUMatrixDynamic<T>& output);
        void backward(cudaStream_t stream, const legacy::GPUMatrixDynamic<float>& input, const legacy::GPUMatrixDynamic<T>& dL_doutput, network::detail::GradientMode param_gradients_mode = network::detail::GradientMode::Overwrite);
        void initialize_params(legacy::math::pcg32& rng, float* params_full_precision, float scale = 1.0f);

        std::uint32_t n_features                     = 0u;
        std::uint32_t n_levels                       = 0u;
        std::size_t n_params                         = 0u;
        ParamsOffsetTable offset_table               = {};
        std::uint32_t base_resolution                = 0u;
        std::uint32_t input_width                    = N_POS_DIMS;
        std::uint32_t output_width                   = 0u;
        legacy::MatrixLayout preferred_output_layout = legacy::SoA;
        float per_level_scale                        = 0.0f;
        bool stochastic_interpolation                = false;
        GridType grid_type                           = GridType::Hash;
        float max_level                              = 1000.0f;
        T* params                                    = nullptr;
        T* gradients                                 = nullptr;
    };

    template <typename T>
    std::variant<GridEncodingTemplated<T, 3u, 1u>, GridEncodingTemplated<T, 3u, 2u>, GridEncodingTemplated<T, 3u, 4u>, GridEncodingTemplated<T, 3u, 8u>> create_position_encoding(std::uint32_t n_dims_to_encode, const InstantNGP::NetworkConfig::HashGridConfig& config);

    template <typename T>
    struct SphericalHarmonicsEncoding final {
        SphericalHarmonicsEncoding(std::uint32_t degree, std::uint32_t n_dims_to_encode);

        void encode(cudaStream_t stream, const legacy::GPUMatrixDynamic<float>& input, legacy::GPUMatrixDynamic<T>& output);

        std::uint32_t degree                         = 0u;
        std::uint32_t input_width                    = 3u;
        std::uint32_t output_width                   = 0u;
        legacy::MatrixLayout preferred_output_layout = legacy::SoA;
    };

    template <typename T>
    SphericalHarmonicsEncoding<T> create_direction_encoding(std::uint32_t n_dims_to_encode, const InstantNGP::NetworkConfig::DirectionEncodingConfig& config);

} // namespace ngp::encoding
