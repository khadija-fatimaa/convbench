// src/v3_direct.cu -- V3 direct convolution (owner: Khadija). STUB.
#include "common.h"

static int g_cob = 0;

size_t v3_workspace_bytes(const ConvParams& p) { (void)p; return 0; }

void v3_set_cob(int cob) { g_cob = cob; }

void v3_run(const ConvParams& p, const float* in, const float* w, float* out, float* ws) {
    (void)p; (void)in; (void)w; (void)out; (void)ws;
    printf("V3: not implemented (cob = %d)\n", g_cob);
}
