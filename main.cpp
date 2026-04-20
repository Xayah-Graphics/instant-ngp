#include "instant-ngp.h"
import std;

int main() {
    ngp::InstantNGP::NetworkConfig network_config{};
    network_config.density_network.n_hidden_layers = 2;
    ngp::InstantNGP ngp{network_config};
    ngp.load_dataset("../data/nerf-synthetic/chair", ngp::InstantNGP::DatasetType::NerfSynthetic);
    ngp.train(10);
    return 0;
}
