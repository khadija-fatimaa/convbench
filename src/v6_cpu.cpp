// src/v6_cpu.cpp -- V6: plain C++, one core (owner: Khadija). Specification: Section 5.6.
// Loop order n, co, oy, ox, then c, r, s. No OpenMP, no SIMD intrinsics, no manual unrolling.
// No CUDA code in this file.
#include "common.h"

void v6_run(const ConvParams& p, const float* in, const float* w, float* out) {
    for (int n = 0; n < p.N; ++n) {
        for (int co = 0; co < p.Cout; ++co) {
            for (int oy = 0; oy < p.Ho; ++oy) {
                for (int ox = 0; ox < p.Wo; ++ox) {
                    float acc = 0.0f;
                    for (int c = 0; c < p.Cin; ++c) {
                        for (int r = 0; r < p.K; ++r) {
                            int iy = oy * p.S - p.P + r;
                            if (iy < 0 || iy >= p.H) continue;          // outside the image counts as 0
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
}
