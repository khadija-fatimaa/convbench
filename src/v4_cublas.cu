// src/v4_cublas.cu -- V4 im2col + cuBLAS (owner: Mania). STUB.
#include "common.h"

size_t v4_workspace_bytes(const ConvParams& p) { (void)p; return 0; }

void v4_init()    { printf("V4 init: not implemented\n"); }
void v4_destroy() { }

void v4_run(const ConvParams& p, const float* in, const float* w, float* out, float* ws) {
    (void)p; (void)in; (void)w; (void)out; (void)ws;
    printf("V4: not implemented\n");
}
