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

