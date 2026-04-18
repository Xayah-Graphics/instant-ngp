#ifndef INSTANT_NGP_H
#define INSTANT_NGP_H

#include "math-mat.h"
#include "math-vec.h"
#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace ngp {
    struct Dataset {
        enum class Type { NerfSynthetic };

        struct Frame {
            std::vector<uint8_t> rgba;
            uint32_t width                             = 0;
            uint32_t height                            = 0;
            float focal_length_x                       = 0.0f;
            float focal_length_y                       = 0.0f;
            std::array<float, 16> transform_matrix_4x4 = {};
        };

        std::vector<Frame> train;
        std::vector<Frame> validation;
        std::vector<Frame> test;
    };

    struct Runtime {};


    class InstantNGP final {
    public:
        InstantNGP();
        ~InstantNGP();
        InstantNGP(const InstantNGP&)                = delete;
        InstantNGP& operator=(const InstantNGP&)     = delete;
        InstantNGP(InstantNGP&&) noexcept            = default;
        InstantNGP& operator=(InstantNGP&&) noexcept = default;

        void load_dataset(const std::filesystem::path& dataset_path, Dataset::Type dataset_type);
        void upload_dataset(const Dataset& dataset);

    private:
        Dataset dataset_ = {};
        Runtime runtime_ = {};
    };

} // namespace ngp

#endif // INSTANT_NGP_H
