#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <stdint.h>
#include <math.h>
constexpr int B=64,T=256,S=5,H=4096,K=1024,I=14336,V=128256,C=128,L=32,QH=32,KH=8,D=128,SEQ=C+S;
enum {TOKEN,DRAFT,LN1,LN2,FNORM,WQ,WK,WV,WO,WG,WU,WD,EMBED,LM,KCACHE,VCACHE,LOGITS,ACCEPT,COMMITTED,TOKENS,LENGTH,FEATURES,KWRITE,VWRITE,SCRATCH};
struct P {
 int64_t *token,*draft;
 __nv_bfloat16 *ln1,*ln2,*fnorm,*wq,*wk,*wv,*wo,*wg,*wu,*wd,*embed,*lm,*kc,*vc;
 float *logits; int64_t *accept,*committed,*tokens,*length;
 __nv_bfloat16 *features,*kwrite,*vwrite; float *scratch;
};
struct W {
 float *x,*xn,*q,*k,*v,*a,*lin,*gate,*up,*score,*prob,*feat,*scale;
 int *greedy,*bar_count,*bar_epoch;
};
__device__ __forceinline__ W workspace(float* base) {
 W w;
 w.x=base; base+=S*H; w.xn=base; base+=S*H; w.q=base; base+=S*H;
 w.k=base; base+=S*K; w.v=base; base+=S*K; w.a=base; base+=S*H;
 w.lin=base; base+=S*I; w.gate=base; base+=S*I; w.up=base; base+=S*I;
 w.score=base; base+=S*QH*SEQ; w.prob=base; base+=S*QH*SEQ;
 w.feat=base; base+=3*S*H; w.scale=base; base+=S;
 w.greedy=(int*)base; base+=S; w.bar_count=(int*)base; base++;
 w.bar_epoch=(int*)base; return w;
}
__device__ __forceinline__ void barrier(W w) {
 cooperative_groups::this_grid().sync();
}
__device__ __forceinline__ float warp_sum(float x) {
 for(int off=16;off;off>>=1) x+=__shfl_down_sync(0xffffffff,x,off);
 return x;
}
__device__ __forceinline__ float warp_max(float x) {
 for(int off=16;off;off>>=1) x=fmaxf(x,__shfl_down_sync(0xffffffff,x,off));
 return x;
}
__device__ void matvec(const float* x,const __nv_bfloat16* weight,float* out,int in,int rows) {
 int lane=threadIdx.x&31,warp=threadIdx.x>>5;
 for(int job=blockIdx.x*8+warp;job<S*rows;job+=B*8) {
  int step=job/rows,row=job%rows;
  const __nv_bfloat16* wr=weight+(size_t)row*in;
  const float* xr=x+(size_t)step*in;
  float sum=0.f;
  for(int j=lane;j<in;j+=32) sum=fmaf((float)wr[j],xr[j],sum);
  sum=warp_sum(sum);
  if(lane==0) out[job]=sum;
 }
}
__device__ void norm(const float* x,const __nv_bfloat16* weight,float* out,int width,W w) {
 if(blockIdx.x<S) {
  float sum=0.f;
  for(int j=threadIdx.x;j<width;j+=T) {float z=x[blockIdx.x*width+j];sum=fmaf(z,z,sum);}
  __shared__ float sums[T];
  sums[threadIdx.x]=sum; __syncthreads();
  for(int off=T/2;off;off>>=1) {
   if(threadIdx.x<off) sums[threadIdx.x]+=sums[threadIdx.x+off];
   __syncthreads();
  }
  if(threadIdx.x==0) w.scale[blockIdx.x]=rsqrtf(sums[0]/width+1.e-5f);
 }
 barrier(w);
 for(int j=blockIdx.x*T+threadIdx.x;j<S*width;j+=B*T)
  out[j]=x[j]*w.scale[j/width]*(float)weight[j%width];
 barrier(w);
}
__device__ void rotate(float* z,int width,int step,int head,int half) {
 int loc=step*width+head*D+half;
 float a=z[loc],b=z[loc+64];
 float phase=(float)(C+step)*powf(500000.f,-(float)(2*half)/D);
 float cs=cosf(phase),sn=sinf(phase);
 z[loc]=a*cs-b*sn; z[loc+64]=b*cs+a*sn;
}
__global__ void persistent(P p) {
 W w=workspace(p.scratch);
 for(int j=blockIdx.x*T+threadIdx.x;j<S*H;j+=B*T) {
  int s=j/H,i=j%H; int64_t tok=s==0?*p.token:p.draft[s-1];
  w.x[j]=(float)p.embed[(size_t)tok*H+i];
 }
 barrier(w);
 for(int layer=0;layer<L;layer++) {
  norm(w.x,p.ln1+(size_t)layer*H,w.xn,H,w);
  matvec(w.xn,p.wq+(size_t)layer*H*H,w.q,H,H);
  matvec(w.xn,p.wk+(size_t)layer*K*H,w.k,H,K);
  matvec(w.xn,p.wv+(size_t)layer*K*H,w.v,H,K);
  barrier(w);
  for(int j=blockIdx.x*T+threadIdx.x;j<S*(QH+KH)*64;j+=B*T) {
   int half=j%64,head=(j/64)%(QH+KH),step=j/(64*(QH+KH));
   if(head<QH) rotate(w.q,H,step,head,half);
   else rotate(w.k,K,step,head-QH,half);
  }
  barrier(w);
  for(int j=blockIdx.x*T+threadIdx.x;j<S*K;j+=B*T) {
   p.kwrite[(size_t)layer*S*K+j]=(__nv_bfloat16)w.k[j];
   p.vwrite[(size_t)layer*S*K+j]=(__nv_bfloat16)w.v[j];
  }
  int lane=threadIdx.x&31,warp=threadIdx.x>>5;
  for(int job=blockIdx.x*8+warp;job<S*QH;job+=B*8) {
   int s=job/QH,head=job%QH,group=head/4;
   for(int pos=0;pos<=C+s;pos++) {
    float acc=0.f;
    for(int d=lane;d<D;d+=32) {
     float key=pos<C?(float)p.kc[((size_t)layer*C+pos)*K+group*D+d]:w.k[(pos-C)*K+group*D+d];
     acc=fmaf(w.q[s*H+head*D+d],key,acc);
    }
    acc=warp_sum(acc);
    if(lane==0) w.score[job*SEQ+pos]=acc*0.08838834764831845f;
   }
  }
  barrier(w);
  for(int job=blockIdx.x*8+warp;job<S*QH;job+=B*8) {
   int s=job/QH;float mx=-INFINITY;
   for(int pos=lane;pos<=C+s;pos+=32) mx=fmaxf(mx,w.score[job*SEQ+pos]);
   mx=warp_max(mx);mx=__shfl_sync(0xffffffff,mx,0);
   float sum=0.f;
   for(int pos=lane;pos<=C+s;pos+=32) sum+=expf(w.score[job*SEQ+pos]-mx);
   sum=warp_sum(sum);sum=__shfl_sync(0xffffffff,sum,0);
   for(int pos=lane;pos<=C+s;pos+=32) w.prob[job*SEQ+pos]=expf(w.score[job*SEQ+pos]-mx)/sum;
  }
  barrier(w);
  for(int j=blockIdx.x*T+threadIdx.x;j<S*H;j+=B*T) {
   int s=j/H,head=(j%H)/D,d=j%D,group=head/4;float sum=0.f;
   for(int pos=0;pos<=C+s;pos++) {
    float val=pos<C?(float)p.vc[((size_t)layer*C+pos)*K+group*D+d]:w.v[(pos-C)*K+group*D+d];
    sum=fmaf(w.prob[(s*QH+head)*SEQ+pos],val,sum);
   }
   w.a[j]=sum;
  }
  barrier(w);
  matvec(w.a,p.wo+(size_t)layer*H*H,w.lin,H,H);
  barrier(w);
  for(int j=blockIdx.x*T+threadIdx.x;j<S*H;j+=B*T) w.x[j]+=w.lin[j];
  barrier(w);
  norm(w.x,p.ln2+(size_t)layer*H,w.xn,H,w);
  matvec(w.xn,p.wg+(size_t)layer*I*H,w.gate,H,I);
  matvec(w.xn,p.wu+(size_t)layer*I*H,w.up,H,I);
  barrier(w);
  for(int j=blockIdx.x*T+threadIdx.x;j<S*I;j+=B*T) {
   float z=w.gate[j];w.gate[j]=(z/(1.f+expf(-z)))*w.up[j];
  }
  barrier(w);
  matvec(w.gate,p.wd+(size_t)layer*H*I,w.lin,I,H);
  barrier(w);
  for(int j=blockIdx.x*T+threadIdx.x;j<S*H;j+=B*T) w.x[j]+=w.lin[j];
  barrier(w);
  if(layer==7||layer==15||layer==31) {
   int slot=layer==7?0:layer==15?1:2;
   for(int j=blockIdx.x*T+threadIdx.x;j<S*H;j+=B*T) w.feat[slot*S*H+j]=w.x[j];
   barrier(w);
  }
 }
 norm(w.x,p.fnorm,w.xn,H,w);
 matvec(w.xn,p.lm,p.logits,H,V);
 barrier(w);
 if(blockIdx.x<S) {
  float best=-INFINITY;int bestidx=0;
  for(int j=threadIdx.x;j<V;j+=T) {
   float z=p.logits[blockIdx.x*V+j];
   if(z>best||(z==best&&j<bestidx)) {best=z;bestidx=j;}
  }
  __shared__ float maxima[T];__shared__ int indices[T];
  maxima[threadIdx.x]=best;indices[threadIdx.x]=bestidx;
  __syncthreads();
  for(int off=T/2;off;off>>=1) {
   if(threadIdx.x<off) {
    float other=maxima[threadIdx.x+off];int idx=indices[threadIdx.x+off];
    if(other>maxima[threadIdx.x]||(other==maxima[threadIdx.x]&&idx<indices[threadIdx.x])) {
     maxima[threadIdx.x]=other;indices[threadIdx.x]=idx;
    }
   }
   __syncthreads();
  }
  if(threadIdx.x==0) w.greedy[blockIdx.x]=indices[0];
 }
 barrier(w);
 if(blockIdx.x==0&&threadIdx.x==0) {
  int accepted=0;
  while(accepted<4&&w.greedy[accepted]==p.draft[accepted]) accepted++;
  *p.accept=accepted;*p.committed=accepted+1;*p.length=C+accepted+1;
  for(int i=0;i<S;i++) p.tokens[i]=i<accepted?p.draft[i]:i==accepted?w.greedy[i]:-1;
 }
 barrier(w);
 int accepted=(int)*p.accept;
 for(int j=blockIdx.x*T+threadIdx.x;j<3*H;j+=B*T)
  p.features[j]=(__nv_bfloat16)w.feat[(j/H)*S*H+accepted*H+j%H];
 for(int j=blockIdx.x*T+threadIdx.x;j<L*S*K;j+=B*T)
  if((j/K)%S>accepted) {p.kwrite[j]=(__nv_bfloat16)0.f;p.vwrite[j]=(__nv_bfloat16)0.f;}
}
extern "C" int launch(void** args,cudaStream_t stream) {
 P p{};
 p.token=(int64_t*)args[TOKEN];p.draft=(int64_t*)args[DRAFT];
 p.ln1=(__nv_bfloat16*)args[LN1];p.ln2=(__nv_bfloat16*)args[LN2];
 p.fnorm=(__nv_bfloat16*)args[FNORM];p.wq=(__nv_bfloat16*)args[WQ];
 p.wk=(__nv_bfloat16*)args[WK];p.wv=(__nv_bfloat16*)args[WV];
 p.wo=(__nv_bfloat16*)args[WO];p.wg=(__nv_bfloat16*)args[WG];
 p.wu=(__nv_bfloat16*)args[WU];p.wd=(__nv_bfloat16*)args[WD];
 p.embed=(__nv_bfloat16*)args[EMBED];p.lm=(__nv_bfloat16*)args[LM];
 p.kc=(__nv_bfloat16*)args[KCACHE];p.vc=(__nv_bfloat16*)args[VCACHE];
 p.logits=(float*)args[LOGITS];p.accept=(int64_t*)args[ACCEPT];
 p.committed=(int64_t*)args[COMMITTED];p.tokens=(int64_t*)args[TOKENS];
 p.length=(int64_t*)args[LENGTH];p.features=(__nv_bfloat16*)args[FEATURES];
 p.kwrite=(__nv_bfloat16*)args[KWRITE];p.vwrite=(__nv_bfloat16*)args[VWRITE];
 p.scratch=(float*)args[SCRATCH];
 void* launch_args[] = {&p};
 return (int)cudaLaunchCooperativeKernel((void*)persistent,dim3(B),dim3(T),launch_args,0,stream);
}
