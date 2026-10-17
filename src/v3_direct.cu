// src/v3_direct.cu -- V3: direct convolution on the GPU (owner: Khadija). Specification: Section 5.4.
// cob = 0 runs the naive kernel; cob = 4, 8 or 16 runs the tiled kernel with that many
// output channels per block. No workspace is needed.
#include "common.h"

static int g_cob = 0;

size_t v3_workspace_bytes(const ConvParams& p) { (void)p; return 0; }

void v3_set_cob(int cob) { g_cob = cob; }

// ---------------------------------------------------------------------------
// Step 1: naive kernel. One thread per output element, grid-stride loop,
// ox changes fastest so neighbouring threads write neighbouring addresses.
// ---------------------------------------------------------------------------
__global__ void v3_naive_kernel(ConvParams p, const float* __restrict__ in,
                                const float* __restrict__ w, float* __restrict__ out) {
    const size_t total  = (size_t)p.N * p.Cout * p.Ho * p.Wo;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        int ox = (int)(idx % p.Wo);
        size_t t = idx / p.Wo;
        int oy = (int)(t % p.Ho);  t /= p.Ho;
        int co = (int)(t % p.Cout);
        int n  = (int)(t / p.Cout);

        float acc = 0.0f;
        for (int c = 0; c < p.Cin; ++c) {
            for (int r = 0; r < p.K; ++r) {
                int iy = oy * p.S - p.P + r;
                if (iy < 0 || iy >= p.H) continue;                      // outside the image counts as 0
                for (int s = 0; s < p.K; ++s) {
                    int ix = ox * p.S - p.P + s;
                    if (ix < 0 || ix >= p.W) continue;
                    acc += in[(((size_t)n * p.Cin + c) * p.H + iy) * p.W + ix]
                         * w[(((size_t)co * p.Cin + c) * p.K + r) * p.K + s];
                }
            }
        }
        out[idx] = acc;   // idx is already the NCHW output index
    }
}

// ---------------------------------------------------------------------------
// Step 2: tiled kernel. A 16 x 16 block computes a 16 x 16 tile of output
// pixels for one image and COB output channels. For each input channel the
// block loads the input tile and the COB x K x K weights into shared memory.
// ---------------------------------------------------------------------------
template <int COB>
__global__ void v3_tiled_kernel(ConvParams p, const float* __restrict__ in,
                                const float* __restrict__ w, float* __restrict__ out) {
    __shared__ float in_tile[37][37];     // IT = 15*S + K <= 37 because S <= 2 and K <= 7
    __shared__ float w_tile[16][7][7];    // COB <= 16, K <= 7

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * 16 + tx;
    const int K = p.K, S = p.S;
    const int IT = 15 * S + K;                       // input tile side
    const int KK = K * K;

    const int groups = (p.Cout + COB - 1) / COB;
    const int n   = blockIdx.z / groups;
    const int co0 = (blockIdx.z % groups) * COB;

    const int y0 = blockIdx.y * 16 * S - p.P;        // top-left input position of the tile
    const int x0 = blockIdx.x * 16 * S - p.P;

    float acc[COB];
    #pragma unroll
    for (int j = 0; j < COB; ++j) acc[j] = 0.0f;

    for (int c = 0; c < p.Cin; ++c) {
        const float* in_c = in + ((size_t)n * p.Cin + c) * p.H * p.W;

        // (a) all 256 threads load the IT x IT input tile, row by row, 0 outside the image
        for (int i = tid; i < IT * IT; i += 256) {
            int ly = i / IT, lx = i % IT;
            int iy = y0 + ly, ix = x0 + lx;
            in_tile[ly][lx] = (iy >= 0 && iy < p.H && ix >= 0 && ix < p.W)
                            ? in_c[(size_t)iy * p.W + ix] : 0.0f;
        }
        // (b) load the COB x K x K weights for this input channel, 0 for co >= Cout
        for (int i = tid; i < COB * KK; i += 256) {
            int j = i / KK, rs = i % KK;
            int r = rs / K, s = rs % K;
            int co = co0 + j;
            w_tile[j][r][s] = (co < p.Cout)
                            ? w[(((size_t)co * p.Cin + c) * K + r) * K + s] : 0.0f;
        }
        __syncthreads();                                              // (c)

        // (d) accumulate
        for (int r = 0; r < K; ++r) {
            for (int s = 0; s < K; ++s) {
                float v = in_tile[ty * S + r][tx * S + s];
                #pragma unroll
                for (int j = 0; j < COB; ++j) acc[j] += v * w_tile[j][r][s];
            }
        }
        __syncthreads();                                              // (e)
    }

    const int oy = blockIdx.y * 16 + ty;
    const int ox = blockIdx.x * 16 + tx;
    if (oy < p.Ho && ox < p.Wo) {
        #pragma unroll
        for (int j = 0; j < COB; ++j) {
            int co = co0 + j;
            if (co < p.Cout)
                out[(((size_t)n * p.Cout + co) * p.Ho + oy) * p.Wo + ox] = acc[j];
        }
    }
}

template <int COB>
static void launch_tiled(const ConvParams& p, const float* in, const float* w, float* out) {
    int groups = (p.Cout + COB - 1) / COB;
    dim3 block(16, 16);
    dim3 grid((p.Wo + 15) / 16, (p.Ho + 15) / 16, p.N * groups);
    v3_tiled_kernel<COB><<<grid, block>>>(p, in, w, out);
}

void v3_run(const ConvParams& p, const float* in, const float* w, float* out, float* ws) {
    (void)ws;   // V3 needs no workspace
    switch (g_cob) {
        case 0: {
            size_t total = (size_t)p.N * p.Cout * p.Ho * p.Wo;
            int blocks = (int)((total + 255) / 256);
            v3_naive_kernel<<<blocks, 256>>>(p, in, w, out);
            break;
        }
        case 4:  launch_tiled<4>(p, in, w, out);  break;
        case 8:  launch_tiled<8>(p, in, w, out);  break;
        case 16: launch_tiled<16>(p, in, w, out); break;
        default:
            fprintf(stderr, "V3: unsupported cob value %d (use 0, 4, 8 or 16)\n", g_cob);
            exit(1);
    }
}
