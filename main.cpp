#include "instant-ngp.h"
import std;

int main() {
    InstantNGP ngp;
    ngp.load_dataset("../data/nerf-synthetic/chair", Dataset::Type::NerfSynthetic);
    return 0;
}
