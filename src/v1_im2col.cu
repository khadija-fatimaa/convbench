// src/v1_im2col.cu -- im2col kernel + V1 (owner: Mania). STUB.
#include "common.h"

size_t v1_workspace_bytes(const ConvParams& p) { (void)p; return 0; }

void im2col_launch(const ConvParams& p, const float* in, float* col) {
    (void)p; (void)in; (void)col;
    printf("im2col_launch: not implemented\n");
}

void v1_run(const ConvParams& p, const float* in, const float* w, float* out, float* ws) {
    (void)p; (void)in; (void)w; (void)out; (void)ws;
    printf("V1: not implemented\n");
}
