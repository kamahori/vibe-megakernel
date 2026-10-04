#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace {
constexpr int NB=64, NT=128, L=34, H=2560, D=256, QH=8, KH=4;
constexpr int F=10240, T=128, VOC=262144;
enum {WQ,WK,WV,WO,WG,WU,WD};
enum {LN1,LN2,LN3,LN4,QN,KN,FN};
enum {X=0,XN=2560,Q=5120,K=7168,V=8192,A=9216,
      TMP=11264,G=13824,U=24064,ACT=34304};
struct Args {
  const int64_t* token;
  const __nv_bfloat16* embed;
  const __nv_bfloat16* norm[7];
  const uint8_t* weight[7];
  const float* scale[7];
  const __nv_bfloat16 *kc,*vc;
  float *s,*logits;
  int64_t* next;
  __nv_bfloat16 *kw,*vw;
  int* sync;
  int phase;
};
__device__ __forceinline__ float bf(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ void barrier(int* sync, int& phase) {
  __syncthreads();
  if(threadIdx.x==0) {
    __threadfence();
    int old=atomicAdd(sync,1);
    if(old==NB-1) {
      atomicExch(sync,0);
      __threadfence();
      atomicExch(sync+1,phase+1);
    } else {
      while(atomicAdd(sync+1,0)<phase+1) {}
    }
  }
  ++phase;
  __syncthreads();
}
__device__ void rms(const float* x,float* y,const __nv_bfloat16* w,
                    int n,float* red) {
  int tid=threadIdx.x;
  float sum=0;
  for(int i=tid;i<n;i+=NT) sum=fmaf(x[i],x[i],sum);
  red[tid]=sum;
  __syncthreads();
  for(int step=NT/2;step;step>>=1) {
    if(tid<step) red[tid]+=red[tid+step];
    __syncthreads();
  }
  float inv=rsqrtf(red[0]/n+1.e-6f);
  for(int i=tid;i<n;i+=NT) y[i]=x[i]*inv*(1.f+bf(w[i]));
  __syncthreads();
}
template<int R,int C>
__device__ void linear(const Args& a,int name,int layer,const float* x,float* y) {
  int lane=threadIdx.x&31, warp=threadIdx.x>>5;
  for(int row=blockIdx.x*4+warp;row<R;row+=NB*4) {
    const uint8_t* w=a.weight[name]+((int64_t)layer*R+row)*(C/2);
    float sum=0;
    for(int c=lane;c<C/2;c+=32) {
      uint8_t q=w[c];
      sum=fmaf(float(int(q&15)-8),x[2*c],sum);
      sum=fmaf(float(int(q>>4)-8),x[2*c+1],sum);
    }
    for(int step=16;step;step>>=1)
      sum+=__shfl_down_sync(0xffffffff,sum,step);
    if(lane==0) y[row]=sum*a.scale[name][(int64_t)layer*R+row];
  }
}
__device__ void lm_head(const Args& a,const float* x) {
  int lane=threadIdx.x&31,warp=threadIdx.x>>5;
  for(int row=blockIdx.x*4+warp;row<VOC;row+=NB*4) {
    const __nv_bfloat16* w=a.embed+(int64_t)row*H;
    float sum=0;
    for(int c=lane;c<H;c+=32) sum=fmaf(bf(w[c]),x[c],sum);
    for(int step=16;step;step>>=1)
      sum+=__shfl_down_sync(0xffffffff,sum,step);
    if(lane==0) a.logits[row]=sum;
  }
}
__device__ void rope_and_writes(const Args& a,int layer,float* red) {
  float* s=a.s;
  bool local=(layer+1)%6!=0;
  double theta=local?10000.:1000000.;
  double position=local?128.:16.;
  for(int h=0;h<QH;++h) {
    rms(s+Q+h*D,s+Q+h*D,a.norm[QN]+layer*D,D,red);
    for(int d=threadIdx.x;d<D;d+=NT) {
      int j=d&127;
      double phase=position*pow(theta,-double(2*j)/D);
      float co=float(cos(phase)),si=float(sin(phase));
      float self=s[Q+h*D+d],other=s[Q+h*D+(d^128)];
      s[A+h*D+d]=self*co+(d<128?-other:other)*si;
    }
    __syncthreads();
    for(int d=threadIdx.x;d<D;d+=NT) s[Q+h*D+d]=s[A+h*D+d];
    __syncthreads();
  }
  for(int h=0;h<KH;++h) {
    rms(s+K+h*D,s+K+h*D,a.norm[KN]+layer*D,D,red);
    for(int d=threadIdx.x;d<D;d+=NT) {
      int j=d&127;
      double phase=position*pow(theta,-double(2*j)/D);
      float co=float(cos(phase)),si=float(sin(phase));
      float self=s[K+h*D+d],other=s[K+h*D+(d^128)];
      s[TMP+h*D+d]=self*co+(d<128?-other:other)*si;
    }
    __syncthreads();
    for(int d=threadIdx.x;d<D;d+=NT) {
      float k=s[TMP+h*D+d];
      s[K+h*D+d]=k;
      a.kw[(layer*KH+h)*D+d]=__float2bfloat16_rn(k);
      a.vw[(layer*KH+h)*D+d]=__float2bfloat16_rn(s[V+h*D+d]);
    }
    __syncthreads();
  }
}
__device__ void attention(const Args& a,int layer,float* scores) {
  int h=blockIdx.x,group=h/2,t=threadIdx.x;
  const float* s=a.s;
  if(t<T) {
    float dot=0;
    const __nv_bfloat16* k=a.kc+(((int64_t)layer*T+t)*KH+group)*D;
    for(int d=0;d<D;++d) dot=fmaf(s[Q+h*D+d],bf(k[d]),dot);
    scores[t]=dot*0.0625f;
  }
  if(t==0) {
    float dot=0;
    for(int d=0;d<D;++d) dot=fmaf(s[Q+h*D+d],s[K+group*D+d],dot);
    scores[T]=dot*0.0625f;
  }
  __syncthreads();
  if(t==0) {
    float mx=scores[0];
    for(int i=1;i<=T;++i) mx=fmaxf(mx,scores[i]);
    float den=0;
    for(int i=0;i<=T;++i) { scores[i]=expf(scores[i]-mx);den+=scores[i]; }
    for(int i=0;i<=T;++i) scores[i]/=den;
  }
  __syncthreads();
  for(int d=t;d<D;d+=NT) {
    float sum=0;
    for(int i=0;i<T;++i) {
      const __nv_bfloat16* v=a.vc+(((int64_t)layer*T+i)*KH+group)*D;
      sum=fmaf(scores[i],bf(v[d]),sum);
    }
    a.s[A+h*D+d]=fmaf(scores[T],s[V+group*D+d],sum);
  }
}
__global__ void decode(Args a) {
  __shared__ float red[NT],scores[T+1];
  int tid=threadIdx.x,phase=a.phase;
  float* s=a.s;
  if(blockIdx.x==0) {
    int64_t tok=*a.token;
    for(int i=tid;i<H;i+=NT) s[X+i]=bf(a.embed[tok*H+i])*sqrtf(float(H));
  }
  barrier(a.sync,phase);
  for(int layer=0;layer<L;++layer) {
    if(blockIdx.x==0) rms(s+X,s+XN,a.norm[LN1]+layer*H,H,red);
    barrier(a.sync,phase);
    linear<QH*D,H>(a,WQ,layer,s+XN,s+Q);
    linear<KH*D,H>(a,WK,layer,s+XN,s+K);
    linear<KH*D,H>(a,WV,layer,s+XN,s+V);
    barrier(a.sync,phase);
    if(blockIdx.x==0) rope_and_writes(a,layer,red);
    barrier(a.sync,phase);
    if(blockIdx.x<QH) attention(a,layer,scores);
    barrier(a.sync,phase);
    linear<H,QH*D>(a,WO,layer,s+A,s+TMP);
    barrier(a.sync,phase);
    if(blockIdx.x==0) {
      rms(s+TMP,s+TMP,a.norm[LN2]+layer*H,H,red);
      for(int i=tid;i<H;i+=NT) s[X+i]+=s[TMP+i];
      __syncthreads();
      rms(s+X,s+XN,a.norm[LN3]+layer*H,H,red);
    }
    barrier(a.sync,phase);
    linear<F,H>(a,WG,layer,s+XN,s+G);
    linear<F,H>(a,WU,layer,s+XN,s+U);
    barrier(a.sync,phase);
    if(blockIdx.x==0)
      for(int i=tid;i<F;i+=NT) {
        float g=s[G+i];
        float z=0.7978845608028654f*(g+0.044715f*g*g*g);
        s[ACT+i]=(0.5f*g*(1.f+tanhf(z)))*s[U+i];
      }
    barrier(a.sync,phase);
    linear<H,F>(a,WD,layer,s+ACT,s+TMP);
    barrier(a.sync,phase);
    if(blockIdx.x==0) {
      rms(s+TMP,s+TMP,a.norm[LN4]+layer*H,H,red);
      for(int i=tid;i<H;i+=NT) s[X+i]+=s[TMP+i];
    }
    barrier(a.sync,phase);
  }
  if(blockIdx.x==0) rms(s+X,s+XN,a.norm[FN],H,red);
  barrier(a.sync,phase);
  lm_head(a,s+XN);
  barrier(a.sync,phase);
  if(blockIdx.x==0) {
    float mx=-INFINITY;int idx=0;
    for(int i=tid;i<VOC;i+=NT) {
      float v=a.logits[i];
      if(v>mx) {mx=v;idx=i;}
    }
    s[tid]=mx;
    reinterpret_cast<int*>(s+NT)[tid]=idx;
    __syncthreads();
    for(int step=NT/2;step;step>>=1) {
      if(tid<step) {
        float v=s[tid+step];
        int j=reinterpret_cast<int*>(s+NT)[tid+step];
        int self=reinterpret_cast<int*>(s+NT)[tid];
        if(v>s[tid]||(v==s[tid]&&j<self)) {
          s[tid]=v;reinterpret_cast<int*>(s+NT)[tid]=j;
        }
      }
      __syncthreads();
    }
    if(tid==0) *a.next=reinterpret_cast<int*>(s+NT)[0];
  }
}
template<typename P> P* ptr(const torch::Tensor& x) {
  return reinterpret_cast<P*>(x.data_ptr());
}
void step(pybind11::dict inputs,torch::Tensor scratch,torch::Tensor sync,
          torch::Tensor logits,torch::Tensor next,
          torch::Tensor kw,torch::Tensor vw,int phase) {
  Args a{};
  auto get=[&](const char* key)->torch::Tensor {
    return inputs[pybind11::str(key)].cast<torch::Tensor>();
  };
  a.token=ptr<int64_t>(get("token"));
  a.embed=ptr<__nv_bfloat16>(get("embed"));
  const char* norms[]={"ln1","ln2","ln3","ln4","qn","kn","fnorm"};
  const char* weights[]={"wq","wk","wv","wo","wg","wu","wd"};
  const char* scales[]={"wq_scale","wk_scale","wv_scale","wo_scale",
                        "wg_scale","wu_scale","wd_scale"};
  for(int i=0;i<7;++i) {
    a.norm[i]=ptr<__nv_bfloat16>(get(norms[i]));
    a.weight[i]=ptr<uint8_t>(get(weights[i]));
    a.scale[i]=ptr<float>(get(scales[i]));
  }
  a.kc=ptr<__nv_bfloat16>(get("kcache"));
  a.vc=ptr<__nv_bfloat16>(get("vcache"));
  a.s=ptr<float>(scratch);a.sync=ptr<int>(sync);
  a.logits=ptr<float>(logits);a.next=ptr<int64_t>(next);
  a.kw=ptr<__nv_bfloat16>(kw);a.vw=ptr<__nv_bfloat16>(vw);
  a.phase=phase;
  decode<<<NB,NT,0,at::cuda::getCurrentCUDAStream().stream()>>>(a);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("step",&step); }
