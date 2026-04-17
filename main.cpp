#include "instant-ngp.h"
import std;

int main() {
    InstantNGP ngp;
    ngp.load_dataset("../data/nerf-synthetic/chair", DatasetType::NerfSynthetic);
    return 0;
}
