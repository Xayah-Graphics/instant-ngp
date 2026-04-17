#ifndef INSTANT_NGP_H
#define INSTANT_NGP_H

#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

enum class DatasetType {
    NerfSynthetic,
};

struct Dataset {
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
    int32_t aabb_scale = 1;
};


class InstantNGP final {
public:
    InstantNGP();
    ~InstantNGP();
    InstantNGP(const InstantNGP&)                = delete;
    InstantNGP& operator=(const InstantNGP&)     = delete;
    InstantNGP(InstantNGP&&) noexcept            = default;
    InstantNGP& operator=(InstantNGP&&) noexcept = default;

    void load_dataset(const std::filesystem::path& dataset_path, DatasetType dataset_type);

private:
    Dataset dataset_ = {};
};


#endif // INSTANT_NGP_H
