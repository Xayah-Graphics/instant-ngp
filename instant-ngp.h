#ifndef INSTANT_NGP_H
#define INSTANT_NGP_H

#include "common.cuh"
#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace ngp {

    struct Runtime final {
        struct Dataset final {
            enum class Type {
                NerfSynthetic,
            };

            struct CPU final {
                struct Frame final {
                    std::vector<std::uint8_t> rgba = {};
                    std::uint32_t width = 0;
                    std::uint32_t height = 0;
                    float focal_length_x = 0.0f;
                    float focal_length_y = 0.0f;
                    std::array<float, 16> transform_matrix_4x4 = {};
                };

                std::vector<Frame> train = {};
                std::vector<Frame> validation = {};
                std::vector<Frame> test = {};
            };

            struct GPU final {
                struct Frame final {
                    // const std::uint8_t* pixels = nullptr;
                    // legacy::Int2 resolution = {};
                    // float focal_length = 0.0f;
                    // legacy::Mat4x3 camera = {};
                };

                struct Train final {
                    // std::vector<legacy::GpuBuffer<std::uint8_t>> pixels = {};
                    // legacy::GpuBuffer<Frame> frames = {};
                };

                Train train = {};
            };

            CPU cpu = {};
            GPU gpu = {};
        };

        Dataset dataset = {};
    };

    class InstantNGP final {
    public:
        InstantNGP();
        ~InstantNGP();
        InstantNGP(const InstantNGP&) = delete;
        InstantNGP& operator=(const InstantNGP&) = delete;
        InstantNGP(InstantNGP&&) noexcept = default;
        InstantNGP& operator=(InstantNGP&&) noexcept = default;

        void load_dataset(const std::filesystem::path& dataset_path, Runtime::Dataset::Type dataset_type);

    private:
        Runtime runtime_ = {};
    };

} // namespace ngp

#endif // INSTANT_NGP_H
