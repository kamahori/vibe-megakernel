#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
using B = __nv_bfloat16;
__device__ float f(B x) { return __bfloat162float(x); }
__device__ float ws(float x) {
  for(int d=16;d;d>>=1) x += __shfl_down_sync(0xffffffff,x,d);
  return __shfl_sync(0xffffffff,x,0);
}
// All 256 threads participate. Each warp computes one output row at a time.
__device__ void mv(const B* w,const float* x,float* y,int rows,int cols,float* stage) {
  for(int i=threadIdx.x;i<cols;i+=256) stage[i]=x[i];
  __syncthreads();
  int lane=threadIdx.x%32, warp=threadIdx.x/32;
  for(int r=warp;r<rows;r+=8) {
    float s=0;
    for(int c=lane;c<cols;c+=32) s += f(w[(size_t)r*cols+c])*stage[c];
    s=ws(s); if(lane==0) y[r]=s;
  }
  __syncthreads();
}
__device__ void norm(const float* x,const B* w,float* y,int n,float* tmp) {
  float s=0; for(int i=threadIdx.x;i<n;i+=256) s+=x[i]*x[i];
  s=ws(s); if(threadIdx.x%32==0) tmp[threadIdx.x/32]=s;
  __syncthreads();
  if(threadIdx.x==0) { float z=0;for(int i=0;i<8;++i) z+=tmp[i];tmp[8]=rsqrtf(z/n+1e-6f); }
  __syncthreads(); float scale=tmp[8];
  for(int i=threadIdx.x;i<n;i+=256) y[i]=x[i]*scale*f(w[i]);
  __syncthreads();
}
__device__ void headnorm(float* x,const B* w,int heads) {
  int lane=threadIdx.x%32,warp=threadIdx.x/32;
  for(int h=warp;h<heads;h+=8) {
    float s=0;for(int d=lane;d<128;d+=32) s+=x[h*128+d]*x[h*128+d];
    float scale=rsqrtf(ws(s)/128+1e-6f);
    for(int d=lane;d<128;d+=32) x[h*128+d]=x[h*128+d]*scale*f(w[d]);
  } __syncthreads();
}
extern "C" __global__ void dense_step(const int64_t* token,const B* ln1,const B* wq,const B* wk,const B* wv,const B* qn,const B* kn,const B* wo,const B* ln2,const B* wg,const B* wu,const B* wd,const B* fn,const B* embed,const B* kc,const B* vc,float* logits,int64_t* next,B* kw,B* vw,float* scratch) {
  __shared__ float stage[3072];
  __shared__ float tmp[16];
  __shared__ float best[256];
  __shared__ int indices[256];
  float* x=scratch;float* xn=x+1024;float* q=xn+1024;float* k=q+2048;
  float* v=k+1024;float* o=v+1024;float* a=o+2048;float* gate=a+1024;float* up=gate+3072;
  for(int i=threadIdx.x;i<1024;i+=256) x[i]=f(embed[(size_t)*token*1024+i]);
  __syncthreads();
  for(int l=0;l<28;++l) {
    norm(x,ln1+l*1024,xn,1024,tmp);
    mv(wq+(size_t)l*2048*1024,xn,q,2048,1024,stage);
    mv(wk+(size_t)l*1024*1024,xn,k,1024,1024,stage);
    mv(wv+(size_t)l*1024*1024,xn,v,1024,1024,stage);
    headnorm(q,qn+l*128,16);headnorm(k,kn+l*128,8);
    // Cache the unrotated vectors so both halves read the original values.
    for(int i=threadIdx.x;i<2048;i+=256) o[i]=q[i];
    for(int i=threadIdx.x;i<1024;i+=256) a[i]=k[i];
    __syncthreads();
    for(int i=threadIdx.x;i<2048;i+=256) {
      int d=i%128,base=i-d;double angle=128.0*pow(1000000.0,-2.0*(d%64)/128.0);
      float c=(float)cos(angle),s=(float)sin(angle);
      q[i]=o[i]*c+(d<64?-o[base+d+64]:o[base+d-64])*s;
      if(i<1024) { k[i]=a[i]*c+(d<64?-a[base+d+64]:a[base+d-64])*s;
        kw[l*1024+i]=__float2bfloat16(k[i]);vw[l*1024+i]=__float2bfloat16(v[i]); }
    } __syncthreads();
    // One warp owns a query head. Its 129 scores occupy a private stage slice.
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
    for(int h=warp;h<16;h+=8) {
      int g=h/2;float maximum=-INFINITY;
      for(int t=0;t<129;++t) {
        float s=0;for(int d=lane;d<128;d+=32) {
          float key=t==128?k[g*128+d]:f(kc[((size_t)l*128+t)*1024+g*128+d]);
          s+=q[h*128+d]*key;
        } s=ws(s)*0.08838834764831845f;
        if(lane==0) stage[warp*129+t]=s;maximum=fmaxf(maximum,s);
      }
      __syncwarp();float den=0;
      for(int t=0;t<129;++t) den+=expf(stage[warp*129+t]-maximum);
      for(int d=lane;d<128;d+=32) {
        float z=0;for(int t=0;t<129;++t) {
          float value=t==128?v[g*128+d]:f(vc[((size_t)l*128+t)*1024+g*128+d]);
          z+=expf(stage[warp*129+t]-maximum)/den*value;
        } o[h*128+d]=z;
      } __syncwarp();
    } __syncthreads();
    mv(wo+(size_t)l*1024*2048,o,a,1024,2048,stage);
    for(int i=threadIdx.x;i<1024;i+=256) x[i]+=a[i];
    __syncthreads();norm(x,ln2+l*1024,xn,1024,tmp);
    mv(wg+(size_t)l*3072*1024,xn,gate,3072,1024,stage);
    mv(wu+(size_t)l*3072*1024,xn,up,3072,1024,stage);
    for(int i=threadIdx.x;i<3072;i+=256) gate[i]=(gate[i]/(1.0f+expf(-gate[i])))*up[i];
    __syncthreads();mv(wd+(size_t)l*1024*3072,gate,a,1024,3072,stage);
    for(int i=threadIdx.x;i<1024;i+=256) x[i]+=a[i];
    __syncthreads();
  }
  norm(x,fn,xn,1024,tmp);mv(embed,xn,logits,151936,1024,stage);
  float b=-INFINITY;int bi=0;
  for(int i=threadIdx.x;i<151936;i+=256) if(logits[i]>b || (logits[i]==b && i<bi)) {b=logits[i];bi=i;}
  best[threadIdx.x]=b;indices[threadIdx.x]=bi;__syncthreads();
  if(threadIdx.x==0) { for(int i=1;i<256;++i) if(best[i]>b || (best[i]==b && indices[i]<bi)) { b=best[i];bi=indices[i]; } *next=bi; }
}
