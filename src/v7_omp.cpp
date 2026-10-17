// src/v7_omp.cpp -- V7: OpenMP, many cores (owner: Khadija). Specification: Section 5.7.
// Same loop body as V6. The n and co loops are combined into one loop so it can be parallelised.
// Visual Studio supports only OpenMP 2.0: no collapse, and the loop variable is a signed int.
// No CUDA code in this file.
#include "common.h"
#include <omp.h>

void v7_run(const ConvParams& p, const float* in, const float* w, float* out, int threads) {
    const int total = p.N * p.Cout;
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int i = 0; i < total; ++i) {
        int n = i / p.Cout, co = i % p.Cout;
        for (int oy = 0; oy < p.Ho; ++oy) {
            for (int ox = 0; ox < p.Wo; ++ox) {
                float acc = 0.0f;
                for (int c = 0; c < p.Cin; ++c) {
                    for (int r = 0; r < p.K; ++r) {
                        int iy = oy * p.S - p.P + r;
                        if (iy < 0 || iy >= p.H) continue;              // outside the image counts as 0
                        for (int s = 0; s < p.K; ++s) {
                            int ix = ox * p.S - p.P + s;
                            if (ix < 0 || ix >= p.W) continue;
                            acc += in[(((size_t)n * p.Cin + c) * p.H + iy) * p.W + ix]
                                 * w[(((size_t)co * p.Cin + c) * p.K + r) * p.K + s];
                        }
                    }
                }
                out[(((size_t)n * p.Cout + co) * p.Ho + oy) * p.Wo + ox] = acc;
            }
        }
    }
}
