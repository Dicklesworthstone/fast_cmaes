#include "fastcma.h"
#include <math.h>
#include <stdio.h>

int main(void) {
    double xmin[4];
    double f = fastcma_sphere(4, 0.4, 20000, 42ULL, xmin);
    if (!isfinite(f)) {
        fprintf(stderr, "f is not finite\n");
        return 1;
    }
    if (f >= 1e-5) {
        fprintf(stderr, "f too large: %g\n", f);
        return 2;
    }
    return 0;
}
