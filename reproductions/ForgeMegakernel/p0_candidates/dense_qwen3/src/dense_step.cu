#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cmath>

namespace cg = cooperative_groups;
constexpr int H = 1024, D = 128, Q = 2048, I = 3072;
constexpr int V = 151936, S = 128, L = 28, NW = 8, NB = 128;

struct Params {
    const int64_t *token;
    const __nv_bfloat16 *ln1, *wq, *wk, *wv, *qn, *kn, *wo;
    const __nv_bfloat16 *ln2, *wg, *wu, *wd, *fnorm, *embed;
    const __nv_bfloat16 *kcache, *vcache;
    float *logits;
    int64_t *next_token;
    __nv_bfloat16 *k_write, *v_write;
    float *z;
};

__device__ __forceinline__ float bf(__nv_bfloat16 x) {
    return __bfloat162float(x);
}
__device__ __forceinline__ float warp_sum(float x) {
    for (int delta = 16; delta; delta >>= 1)
        x += __shfl_down_sync(0xffffffff, x, delta);
    return x;
}
__device__ __forceinline__ float warp_max(float x) {
    for (int delta = 16; delta; delta >>= 1)
        x = fmaxf(x, __shfl_down_sync(0xffffffff, x, delta));
    return x;
}

// One warp owns one output row. All CTAs stay resident for grid barriers.
__device__ void gemv(const __nv_bfloat16 *w, const float *x, float *y,
                     int rows, int cols, const float *residual = nullptr) {
    int lane = threadIdx.x & 31;
    int warp = blockIdx.x * NW + (threadIdx.x >> 5);
    for (int row = warp; row < rows; row += gridDim.x * NW) {
        float acc = 0.f;
        const __nv_bfloat16 *r = w + (size_t)row * cols;
        for (int col = lane; col < cols; col += 32)
            acc = fmaf(bf(r[col]), x[col], acc);
        acc = warp_sum(acc);
        if (lane == 0) y[row] = residual ? residual[row] + acc : acc;
    }
}

__device__ void rms1024(cg::grid_group grid, const float *x,
                        const __nv_bfloat16 *w, float *out, float *scale_store) {
    __shared__ float partial[256];
    if (blockIdx.x == 0) {
        float sum = 0.f;
        for (int j = threadIdx.x; j < H; j += 256)
            sum = fmaf(x[j], x[j], sum);
        partial[threadIdx.x] = sum;
        __syncthreads();
        for (int d = 128; d; d >>= 1) {
            if (threadIdx.x < d) partial[threadIdx.x] += partial[threadIdx.x + d];
            __syncthreads();
        }
        if (threadIdx.x == 0) *scale_store = rsqrtf(partial[0] / H + 1.e-6f);
    }
    grid.sync();
    float scale = *scale_store;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x;
         j < H; j += gridDim.x * blockDim.x)
        out[j] = (x[j] * scale) * bf(w[j]);
    grid.sync();
}

__global__ void decode(Params p) {
    cg::grid_group grid = cg::this_grid();
    float *x = p.z;
    float *xn = p.z + 1024;
    float *q = p.z + 2048;
    float *k = p.z + 4096;
    float *v = p.z + 5120;
    float *qnorm = p.z + 6144;
    float *knorm = p.z + 8192;
    float *att = p.z + 9216;
    float *gate = p.z + 11264;
    float *up = p.z + 14336;
    float *mlp = p.z + 17408;
    float *scores = p.z + 20480;
    float *probs = p.z + 22544;
    float *scale_store = p.z + 24608;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp = blockIdx.x * NW + (threadIdx.x >> 5);

    int64_t token = *p.token;
    for (int j = tid; j < H; j += gridDim.x * blockDim.x)
        x[j] = bf(p.embed[token * H + j]);
    grid.sync();

    for (int layer = 0; layer < L; ++layer) {
        rms1024(grid, x, p.ln1 + layer * H, xn, scale_store);
        gemv(p.wq + (size_t)layer * Q * H, xn, q, Q, H);
        gemv(p.wk + (size_t)layer * H * H, xn, k, H, H);
        gemv(p.wv + (size_t)layer * H * H, xn, v, H, H);
        grid.sync();

        // Per-head Q/K RMS norm before rotary embedding.
        if (warp < 24) {
            bool is_q = warp < 16;
            int head = is_q ? warp : warp - 16;
            float *src = (is_q ? q : k) + head * D;
            float *dst = (is_q ? qnorm : knorm) + head * D;
            const __nv_bfloat16 *weight = (is_q ? p.qn : p.kn) + layer * D;
            float sum = 0.f;
            for (int d = lane; d < D; d += 32)
                sum = fmaf(src[d], src[d], sum);
            sum = warp_sum(sum);
            float scale = rsqrtf(__shfl_sync(0xffffffff, sum, 0) / D + 1.e-6f);
            for (int d = lane; d < D; d += 32)
                dst[d] = (src[d] * scale) * bf(weight[d]);
        }
        grid.sync();
        for (int j = tid; j < Q + H; j += gridDim.x * blockDim.x) {
            bool is_q = j < Q;
            int idx = is_q ? j : j - Q;
            int d = idx % D;
            int other = idx - d + ((d + 64) % D);
            float a = is_q ? qnorm[idx] : knorm[idx];
            float b = is_q ? qnorm[other] : knorm[other];
            if (d < 64) b = -b;
            double angle = double(S) * pow(1000000.0, -double((d % 64) * 2) / D);
            float value = a * float(cos(angle)) + b * float(sin(angle));
            if (is_q) q[idx] = value;
            else {
                k[idx] = value;
                p.k_write[layer * H + idx] = __float2bfloat16_rn(value);
            }
        }
        for (int j = tid; j < H; j += gridDim.x * blockDim.x)
            p.v_write[layer * H + j] = __float2bfloat16_rn(v[j]);
        grid.sync();

        // Each warp scores one (query head, position) pair.
        for (int pair = warp; pair < 16 * (S + 1); pair += gridDim.x * NW) {
            int head = pair / (S + 1);
            int t = pair % (S + 1);
            int kv_head = head / 2;
            float acc = 0.f;
            for (int d = lane; d < D; d += 32) {
                float kv = t == S ? k[kv_head * D + d] :
                    bf(p.kcache[((size_t)layer * S + t) * H + kv_head * D + d]);
                acc = fmaf(q[head * D + d], kv, acc);
            }
            acc = warp_sum(acc);
            if (lane == 0) scores[pair] = acc * 0.08838834764831845f;
        }
        grid.sync();

        if (warp < 16) {
            int base = warp * (S + 1);
            float mx = -INFINITY;
            for (int t = lane; t <= S; t += 32)
                mx = fmaxf(mx, scores[base + t]);
            mx = warp_max(mx);
            mx = __shfl_sync(0xffffffff, mx, 0);
            float sum = 0.f;
            for (int t = lane; t <= S; t += 32)
                sum += expf(scores[base + t] - mx);
            sum = warp_sum(sum);
            sum = __shfl_sync(0xffffffff, sum, 0);
            for (int t = lane; t <= S; t += 32)
                probs[base + t] = expf(scores[base + t] - mx) / sum;
        }
        grid.sync();

        for (int j = tid; j < Q; j += gridDim.x * blockDim.x) {
            int head = j / D;
            int d = j % D;
            int kv_head = head / 2;
            float sum = 0.f;
            for (int t = 0; t < S; ++t)
                sum = fmaf(probs[head * (S + 1) + t],
                           bf(p.vcache[((size_t)layer * S + t) * H + kv_head * D + d]), sum);
            att[j] = fmaf(probs[head * (S + 1) + S], v[kv_head * D + d], sum);
        }
        grid.sync();
        gemv(p.wo + (size_t)layer * H * Q, att, x, H, Q, x);
        grid.sync();

        rms1024(grid, x, p.ln2 + layer * H, xn, scale_store);
        gemv(p.wg + (size_t)layer * I * H, xn, gate, I, H);
        gemv(p.wu + (size_t)layer * I * H, xn, up, I, H);
        grid.sync();
        for (int j = tid; j < I; j += gridDim.x * blockDim.x)
            mlp[j] = (gate[j] / (1.f + expf(-gate[j]))) * up[j];
        grid.sync();
        gemv(p.wd + (size_t)layer * H * I, mlp, x, H, I, x);
        grid.sync();
    }

    rms1024(grid, x, p.fnorm, xn, scale_store);
    gemv(p.embed, xn, p.logits, V, H);
    grid.sync();
    if (blockIdx.x == 0) {
        __shared__ float max_val[256];
        __shared__ int max_idx[256];
        float best = -INFINITY;
        int best_idx = 0;
        for (int j = threadIdx.x; j < V; j += blockDim.x) {
            float value = p.logits[j];
            if (value > best || (value == best && j < best_idx)) {
                best = value;
                best_idx = j;
            }
        }
        max_val[threadIdx.x] = best;
        max_idx[threadIdx.x] = best_idx;
        __syncthreads();
        for (int d = 128; d; d >>= 1) {
            if (threadIdx.x < d) {
                float val = max_val[threadIdx.x + d];
                int idx = max_idx[threadIdx.x + d];
                if (val > max_val[threadIdx.x] ||
                    (val == max_val[threadIdx.x] && idx < max_idx[threadIdx.x])) {
                    max_val[threadIdx.x] = val;
                    max_idx[threadIdx.x] = idx;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) *p.next_token = max_idx[0];
    }
}

void run(at::Tensor token, at::Tensor ln1, at::Tensor wq, at::Tensor wk,
         at::Tensor wv, at::Tensor qn, at::Tensor kn, at::Tensor wo,
         at::Tensor ln2, at::Tensor wg, at::Tensor wu, at::Tensor wd,
         at::Tensor fnorm, at::Tensor embed, at::Tensor kcache,
         at::Tensor vcache, at::Tensor logits, at::Tensor next_token,
         at::Tensor k_write, at::Tensor v_write, at::Tensor scratch) {
    Params p{};
    p.token = token.data_ptr<int64_t>();
    p.ln1 = (const __nv_bfloat16*)ln1.data_ptr<at::BFloat16>();
    p.wq = (const __nv_bfloat16*)wq.data_ptr<at::BFloat16>();
    p.wk = (const __nv_bfloat16*)wk.data_ptr<at::BFloat16>();
    p.wv = (const __nv_bfloat16*)wv.data_ptr<at::BFloat16>();
    p.qn = (const __nv_bfloat16*)qn.data_ptr<at::BFloat16>();
    p.kn = (const __nv_bfloat16*)kn.data_ptr<at::BFloat16>();
    p.wo = (const __nv_bfloat16*)wo.data_ptr<at::BFloat16>();
    p.ln2 = (const __nv_bfloat16*)ln2.data_ptr<at::BFloat16>();
    p.wg = (const __nv_bfloat16*)wg.data_ptr<at::BFloat16>();
    p.wu = (const __nv_bfloat16*)wu.data_ptr<at::BFloat16>();
    p.wd = (const __nv_bfloat16*)wd.data_ptr<at::BFloat16>();
    p.fnorm = (const __nv_bfloat16*)fnorm.data_ptr<at::BFloat16>();
    p.embed = (const __nv_bfloat16*)embed.data_ptr<at::BFloat16>();
    p.kcache = (const __nv_bfloat16*)kcache.data_ptr<at::BFloat16>();
    p.vcache = (const __nv_bfloat16*)vcache.data_ptr<at::BFloat16>();
    p.logits = logits.data_ptr<float>();
    p.next_token = next_token.data_ptr<int64_t>();
    p.k_write = (__nv_bfloat16*)k_write.data_ptr<at::BFloat16>();
    p.v_write = (__nv_bfloat16*)v_write.data_ptr<at::BFloat16>();
    p.z = scratch.data_ptr<float>();
    void *args[] = {&p};
    cudaError_t err = cudaLaunchCooperativeKernel((void*)decode,
        dim3(NB), dim3(256), args, 0, at::cuda::getCurrentCUDAStream().stream());
    TORCH_CHECK(err == cudaSuccess, "cooperative decode launch: ", cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
