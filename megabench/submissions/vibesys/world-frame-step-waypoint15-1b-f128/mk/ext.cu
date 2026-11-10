#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "host_util.h"
#include "mega_kernel.cuh"
#include <vector>
#include <map>

static Params g_params;
static std::vector<const void*> g_key;

#define T(name) (d[name].cast<torch::Tensor>())
#define PTR(t) ((t).data_ptr())

void mega_run(py::dict d, bool timeline) {
  const char* in_names[] = {"noise", "mouse", "button", "scroll", "k_cache", "v_cache", "noise_fc1", "noise_fc2", "ctrl_fc1", "ctrl_fc2",
                            "attn_cond_bias", "attn_cond_w", "mlp_cond_bias", "mlp_cond_w", "wq", "wk", "wv", "wo", "v_lamb", "fuse_x",
                            "fuse_c", "fuse_out", "fc1", "fc2", "out_norm_w", "unpatch_w", "unpatch_b", "patch_w"};
  std::vector<const void*> key;
  for (auto n : in_names) key.push_back(PTR(T(n)));
  for (auto n : {"x", "h", "att", "mlp", "patch_in", "qbuf", "kbuf", "vbuf"}) key.push_back(PTR(T(n)));
  Params& P = g_params;
  if (key != g_key) {
    g_key = key;
    P = Params();
    P.tmA_h = make_map(PTR(T("h")), 2048, 512, 64, 128);
    P.tmA_att = make_map(PTR(T("att")), 2048, 512, 64, 128);
    P.tmA_mlp = make_map(PTR(T("mlp")), 8192, 512, 64, 128);
    P.tmA_patch = make_map(PTR(T("patch_in")), 128, 512, 64, 128);
    P.tmQ = make_map(PTR(T("qbuf")), 2048, 512, 64, 128);
    P.tmB_wq = make_map(PTR(T("wq")), 2048, 24ull * 2048, 64, 128);
    P.tmB_wk = make_map(PTR(T("wk")), 2048, 24ull * 1024, 64, 128);
    P.tmB_wv = make_map(PTR(T("wv")), 2048, 24ull * 1024, 64, 128);
    P.tmB_wo = make_map(PTR(T("wo")), 2048, 24ull * 2048, 64, 128);
    P.tmB_fc1 = make_map(PTR(T("fc1")), 2048, 24ull * 8192, 64, 256);
    P.tmB_fc2 = make_map(PTR(T("fc2")), 8192, 24ull * 2048, 64, 128);
    P.tmB_fx = make_map(PTR(T("fuse_x")), 2048, 8ull * 2048, 64, 64);
    P.tmB_fo = make_map(PTR(T("fuse_out")), 2048, 8ull * 2048, 64, 128);
    P.tmB_patch = make_map(PTR(T("patch_w")), 128, 2048, 64, 128);
    P.tmKc = make_map(PTR(T("k_cache")), 1024, 24ull * 16 * 512, 64, 128);
    P.tmVc = make_map(PTR(T("v_cache")), 1024, 24ull * 16 * 512, 64, 128);
    P.tmKb = make_map(PTR(T("kbuf")), 1024, 512, 64, 128);
    P.tmVb = make_map(PTR(T("vbuf")), 1024, 512, 64, 128);
    P.noise = (const bf16*)PTR(T("noise")); P.mouse = (const bf16*)PTR(T("mouse")); P.button = (const bf16*)PTR(T("button"));
    P.scroll = (const bf16*)PTR(T("scroll"));
    P.noise_fc1 = (const float*)PTR(T("noise_fc1")); P.noise_fc2 = (const float*)PTR(T("noise_fc2"));
    P.ctrl_fc1 = (const bf16*)PTR(T("ctrl_fc1")); P.ctrl_fc2 = (const bf16*)PTR(T("ctrl_fc2"));
    P.attn_cond_bias = (const bf16*)PTR(T("attn_cond_bias")); P.attn_cond_w = (const bf16*)PTR(T("attn_cond_w"));
    P.mlp_cond_bias = (const bf16*)PTR(T("mlp_cond_bias")); P.mlp_cond_w = (const bf16*)PTR(T("mlp_cond_w"));
    P.v_lamb = (const bf16*)PTR(T("v_lamb")); P.fuse_c = (const bf16*)PTR(T("fuse_c")); P.out_norm_w = (const bf16*)PTR(T("out_norm_w"));
    P.unpatch_w = (const bf16*)PTR(T("unpatch_w")); P.unpatch_b = (const bf16*)PTR(T("unpatch_b"));
    P.x = (bf16*)PTR(T("x")); P.h = (bf16*)PTR(T("h")); P.att = (bf16*)PTR(T("att")); P.mlp = (bf16*)PTR(T("mlp"));
    P.patch_in = (bf16*)PTR(T("patch_in")); P.qbuf = (bf16*)PTR(T("qbuf")); P.kbuf = (bf16*)PTR(T("kbuf")); P.vbuf = (bf16*)PTR(T("vbuf"));
    P.vfirst = (bf16*)PTR(T("vfirst")); P.acc = (float*)PTR(T("acc"));
    P.fourier = (const float*)PTR(T("fourier")); P.rope_cos = (const float*)PTR(T("rope_cos")); P.rope_sin = (const float*)PTR(T("rope_sin"));
    P.n_h1 = (float*)PTR(T("n_h1")); P.cond = (bf16*)PTR(T("cond")); P.ctrl_h1 = (bf16*)PTR(T("ctrl_h1")); P.ctrl = (bf16*)PTR(T("ctrl"));
    P.mod = (bf16*)PTR(T("mod")); P.outmod = (bf16*)PTR(T("outmod")); P.cvec = (bf16*)PTR(T("cvec"));
    P.vis = (const int*)PTR(T("vis")); P.nvis = (const int*)PTR(T("nvis"));
    P.bar = (unsigned long long*)PTR(T("bar"));
  }
  P.latent = (bf16*)PTR(T("latent")); P.k_write = (bf16*)PTR(T("k_write")); P.v_write = (bf16*)PTR(T("v_write"));
  P.stop = d.contains("stop") ? d["stop"].cast<int>() : -1; P.stop_layer = d.contains("stop_layer") ? d["stop_layer"].cast<int>() : -1; P.only_pass = d.contains("only_pass") ? d["only_pass"].cast<int>() : -1;
  P.tl = timeline ? (unsigned long long*)PTR(T("tl")) : nullptr;
  size_t smem = GEMM_STAGE_SMEM + 1024 + 256 + 1024;
  cudaFuncSetAttribute(mega_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  void* args[] = {&P};
  cudaError_t e = cudaLaunchCooperativeKernel((void*)mega_kernel, dim3(148), dim3(256), args, smem, at::cuda::getCurrentCUDAStream());
  if (e != cudaSuccess) throw std::runtime_error(std::string("launch failed: ") + cudaGetErrorString(e));
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("mega_run", &mega_run); }
