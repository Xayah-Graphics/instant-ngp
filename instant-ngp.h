#ifndef INSTANT_NGP_H
#define INSTANT_NGP_H

#include <array>
#include <filesystem>

enum class DatasetType {
    NerfSynthetic,
};

struct Dataset {
    struct NerfFrame {
        std::vector<uint8_t> rgba;
        uint32_t width;
        uint32_t height;
        float focal_length_x;
        float focal_length_y;
        std::array<float, 16> transform_matrix_4x4 = {};
    };

    std::vector<NerfFrame> train;
    std::vector<NerfFrame> validation;
    std::vector<NerfFrame> test;
    int32_t aabb_scale;
};


class InstantNGP final {
public:
    InstantNGP();
    ~InstantNGP();
    InstantNGP(const InstantNGP&)                = delete;
    InstantNGP& operator=(const InstantNGP&)     = delete;
    InstantNGP(InstantNGP&&) noexcept            = default;
    InstantNGP& operator=(InstantNGP&&) noexcept = default;

protected:
    void load_dataset(const std::filesystem::path& dataset_path, DatasetType dataset_type);
};


#endif // INSTANT_NGP_H
