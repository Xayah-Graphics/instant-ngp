#include "instant-ngp.h"
import std;

int main() {
    ngp::InstantNGP ngp;
    ngp.load_dataset("../data/nerf-synthetic/chair", ngp::Dataset::Type::NerfSynthetic);
    return 0;
}
