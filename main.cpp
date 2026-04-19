#include "instant-ngp.h"
import std;

int main() {
    ngp::InstantNGP ngp;
    ngp.load_dataset("data/nerf-synthetic/chair", ngp::Runtime::Dataset::Type::NerfSynthetic);
    return 0;
}
