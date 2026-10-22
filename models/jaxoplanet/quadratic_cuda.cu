// Double-precision CUDA reduction; preserve the ordered 128-cadence sum.
// Upstream scalar formula attribution is in quadratic_cuda.h.
#include <cuda_runtime.h>
#include "quadratic_cuda.h"
#include "xla/ffi/api/ffi.h"
namespace ffi=xla::ffi;
using D=ffi::Buffer<ffi::F64>;
using B=ffi::Buffer<ffi::PRED>;
using R=ffi::ResultBuffer<ffi::F64>;
constexpr int threads=128;
template<int K>
__global__ void First(int64_t n,const double* theta,const double* z,const double* zdot,
                      const double* y,const double* var,const bool* mask,
                      const double* design,const bool* front,double* scratch) {
  constexpr int V=K+6;
  int64_t lane=blockIdx.y,t=int64_t(blockIdx.x)*threads+threadIdx.x,i=lane*n+t;
  double values[V]={};
  if(t<n)koala_quad::cadence<K>(theta+lane*(K+5),z[i],zdot[i],y[i],var[i],mask[i],design+K*i,front[i],values);
  __shared__ double shared[V][threads];
  for(int k=0;k<V;++k)shared[k][threadIdx.x]=values[k];
  __syncthreads();
  for(int stride=threads/2;stride;stride/=2) {
    if(threadIdx.x<stride)for(int k=0;k<V;++k)shared[k][threadIdx.x]+=shared[k][threadIdx.x+stride];
    __syncthreads();
  }
  if(threadIdx.x==0)for(int k=0;k<V;++k)scratch[(lane*gridDim.x+blockIdx.x)*V+k]=shared[k][0];
}
__global__ void Second(int blocks,int values,const double* scratch,double* output) {
  int lane=blockIdx.x,k=threadIdx.x;
  if(k<values) {
    double sum=0.;for(int b=0;b<blocks;++b)sum+=scratch[(lane*blocks+b)*values+k];
    output[lane*values+k]=sum;
  }
}
template<int K>
void Launch(cudaStream_t stream,int blocks,int64_t lanes,int64_t n,D& theta,D& z,D& zdot,D& y,D& var,
            B& mask,D& design,B& front,R& scratch) {
  First<K><<<dim3(blocks,lanes),threads,0,stream>>>(n,theta.typed_data(),z.typed_data(),zdot.typed_data(),
      y.typed_data(),var.typed_data(),mask.typed_data(),design.typed_data(),front.typed_data(),scratch->typed_data());
}
ffi::Error Calculate(cudaStream_t stream,D theta,D z,D zdot,D y,D var,B mask,D design,B front,R out,R scratch) {
  int64_t n=z.dimensions().back(),columns=design.dimensions().back();
  if(columns<1 || columns>8 || theta.dimensions().back()!=columns+5)
    return ffi::Error::InvalidArgument("koala quadratic CUDA expects 1-8 trend columns and matching parameters");
  int64_t lanes=theta.element_count()/(columns+5),values=columns+6;
  if(z.element_count()!=lanes*n || zdot.element_count()!=lanes*n || y.element_count()!=lanes*n ||
     var.element_count()!=lanes*n || mask.element_count()!=lanes*n ||
     design.element_count()!=lanes*n*columns || front.element_count()!=lanes*n)
    return ffi::Error::InvalidArgument("koala quadratic CUDA batch shape mismatch");
  int blocks=(n+threads-1)/threads;
  if (lanes == 0) return ffi::Error::Success();
  if (out->element_count() != lanes*values || scratch->element_count() != lanes*blocks*values)
    return ffi::Error::InvalidArgument("koala quadratic CUDA output shape mismatch");
  if (blocks > 0) switch(columns) {
    case 1: Launch<1>(stream,blocks,lanes,n,theta,z,zdot,y,var,mask,design,front,scratch); break;
    case 2: Launch<2>(stream,blocks,lanes,n,theta,z,zdot,y,var,mask,design,front,scratch); break;
    case 3: Launch<3>(stream,blocks,lanes,n,theta,z,zdot,y,var,mask,design,front,scratch); break;
    case 4: Launch<4>(stream,blocks,lanes,n,theta,z,zdot,y,var,mask,design,front,scratch); break;
    case 5: Launch<5>(stream,blocks,lanes,n,theta,z,zdot,y,var,mask,design,front,scratch); break;
    case 6: Launch<6>(stream,blocks,lanes,n,theta,z,zdot,y,var,mask,design,front,scratch); break;
    case 7: Launch<7>(stream,blocks,lanes,n,theta,z,zdot,y,var,mask,design,front,scratch); break;
    case 8: Launch<8>(stream,blocks,lanes,n,theta,z,zdot,y,var,mask,design,front,scratch); break;
  }
  Second<<<lanes,32,0,stream>>>(blocks,values,scratch->typed_data(),out->typed_data());
  auto error=cudaGetLastError();
  if(error!=cudaSuccess)return ffi::Error::Internal(cudaGetErrorString(error));
  return ffi::Error::Success();
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(koala_quadratic_likelihood,Calculate,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>()
       .Arg<D>().Arg<D>().Arg<D>().Arg<D>().Arg<D>().Arg<B>().Arg<D>().Arg<B>().Ret<D>().Ret<D>());

__global__ void PartsKernel(int64_t n,const double* input,double* output) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<n) {
    auto p=koala_quad::partials(input[4*i],input[4*i+1],input[4*i+2],input[4*i+3]);
    output[5*i]=p.value;output[5*i+1]=p.dr;output[5*i+2]=p.dz;
    output[5*i+3]=p.du1;output[5*i+4]=p.du2;
  }
}
ffi::Error PartsCall(cudaStream_t stream,D input,R output) {
  int64_t n=input.element_count()/4;
  if(input.dimensions().back()!=4 || output->element_count()!=5*n)
    return ffi::Error::InvalidArgument("koala quadratic CUDA partials shape mismatch");
  if (n == 0) return ffi::Error::Success();
  PartsKernel<<<(n+127)/128,128,0,stream>>>(n,input.typed_data(),output->typed_data());
  auto error=cudaGetLastError();
  return error==cudaSuccess?ffi::Error::Success():ffi::Error::Internal(cudaGetErrorString(error));
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(koala_quadratic_partials,PartsCall,
    ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>().Arg<D>().Ret<D>());
