![banner](https://github.com/Xayah-Graphics/imagebed/blob/78855d9c1638920ce1bac9302d56cf49305d518d/instant-ngp.png)
# [SIGGRAPH 2022] Instant Neural Graphics Primitives with a Multiresolution Hash Encoding. Thomas Müller, et al.

[![Linux Build (Arch)](https://github.com/Xayah-Graphics/instant-ngp/actions/workflows/arch-build.yml/badge.svg)](https://github.com/Xayah-Graphics/instant-ngp/actions/workflows/arch-build.yml)
[![Windows Build](https://github.com/Xayah-Graphics/instant-ngp/actions/workflows/windows-build.yml/badge.svg)](https://github.com/Xayah-Graphics/instant-ngp/actions/workflows/windows-build.yml)

## 1. Algorithm Pipeline

On dev...

## 2. Build Instruction

#### Build

- CMake 4.3.0 or higher
- Ninja build system (for CXX std module support)
- A C++23 compliant compiler (tested on Arch Linux with gcc/g++ 15.2.1, Windows with MSVC 17.14.29)
- NVIDIA CUDA 13.2 or higher

```
cmake -B build -S . -G Ninja
cmake --build build -j 30
```

#### Validation, Inference, And Test

Run periodic whole-split validation during training:

```bash
./build/instant-ngp-app \
    --scene data/nerf-synthetic/chair \
    --steps 50000 \
    --validation-interval 10000 \
    --validation-dir validation
```

This writes one CSV per validation step, for example `validation/step_10000.csv`.

Render a single view during training:

```bash
./build/instant-ngp-app \
    --scene data/nerf-synthetic/chair \
    --steps 50000 \
    --inference-interval 10000 \
    --inference-dir inference \
    --inference-width 800 \
    --inference-height 800 \
    --inference-focal-length 1111.1110311937682 \
    --inference-transform "-0.012968253344297409,0.44961410760879517,-0.8931288123130798,-3.6003170013427734,-0.9999159574508667,-0.005831199698150158,0.011583293788135052,0.046693746000528336,-4.656612873077393e-10,0.8932039141654968,0.44965195655822754,1.8126047849655151,0,0,0,1"
```

Benchmark the whole test split after training:

```bash
./build/instant-ngp-app \
    --scene data/nerf-synthetic/chair \
    --steps 50000 \
    --test-report test_metrics.csv
```

