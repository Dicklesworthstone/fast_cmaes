#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Version packed as major<<16 | minor<<8 | patch (e.g., 0.1.4 -> 0x000104)
unsigned int fastcma_version(void);

// Minimize sphere starting from 0.6 in each dim; fills xmin if not null. Returns best f.
double fastcma_sphere(size_t dim, double sigma, size_t maxfevals, unsigned long long seed, double* xmin);

#ifdef __cplusplus
}
#endif
