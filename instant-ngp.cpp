#include "instant-ngp.h"

import std;

InstantNGP::InstantNGP() {
    std::print("Hello, Instant NGP!\n");
}
InstantNGP::~InstantNGP() {
    std::print("Bye!\n");
}
void InstantNGP::load_dataset(const std::filesystem::path& dataset_path, DatasetType dataset_type) {}
