#pragma once

#include "fastcma.h"
#include <vector>
#include <string>

namespace fastcma {
inline std::string version_string() {
    unsigned int v = fastcma_version();
    unsigned int major = (v >> 16) & 0xFF;
    unsigned int minor = (v >> 8) & 0xFF;
    unsigned int patch = v & 0xFF;
    return std::to_string(major) + "." + std::to_string(minor) + "." + std::to_string(patch);
}

inline double minimize_sphere(size_t dim, double sigma = 0.4, size_t maxfevals = 20000, unsigned long long seed = 42ULL) {
    std::vector<double> xmin(dim, 0.0);
    return fastcma_sphere(dim, sigma, maxfevals, seed, xmin.data());
}
} // namespace fastcma
