// src/v2_implicit.cu -- V2 implicit GEMM (owner: Mania). STUB.
#include "common.h"

size_t v2_workspace_bytes(const ConvParams& p) { (void)p; return 0; }

void v2_run(const ConvParams& p, const float* in, const float* w, float* out, float* ws) {
    (void)p; (void)in; (void)w; (void)out; (void)ws;
    printf("V2: not implemented\n");
}
