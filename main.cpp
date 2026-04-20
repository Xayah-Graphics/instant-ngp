#include "instant-ngp.h"
import std;

int main() {
    ngp::InstantNGP ngp;
    ngp.load_dataset("../data/nerf-synthetic/chair", ngp::InstantNGP::DatasetType::NerfSynthetic);
    return 0;
}
