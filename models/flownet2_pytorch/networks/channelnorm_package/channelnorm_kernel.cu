#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>

#include "channelnorm_kernel.cuh"

#define CUDA_NUM_THREADS 512 

#define DIM0(TENSOR) ((TENSOR).x)
#define DIM1(TENSOR) ((TENSOR).y)
#define DIM2(TENSOR) ((TENSOR).z)
#define DIM3(TENSOR) ((TENSOR).w)

#define DIM3_INDEX(TENSOR, xx, yy, zz, ww) ((TENSOR)[((xx) * (TENSOR##_stride.x)) + ((yy) * (TENSOR##_stride.y)) + ((zz) * (TENSOR##_stride.z)) + ((ww) * (TENSOR##_stride.w))])

using at::Half;

template <typename scalar_t>
__global__ void kernel_channelnorm_update_output(
    const int n, 
    const scalar_t* __restrict__ input1,
    const long4 input1_size,
    const long4 input1_stride,
    scalar_t* __restrict__ output, 
    const long4 output_size,
    const long4 output_stride,
    int norm_deg) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    int dim_b = DIM0(output_size);
    int dim_c = DIM1(output_size);
    int dim_h = DIM2(output_size);
    int dim_w = DIM3(output_size);
    int dim_chw = dim_c * dim_h * dim_w;

    int b = ( index / dim_chw ) % dim_b;
    int y = ( index / dim_w )   % dim_h;
    int x = ( index          )  % dim_w;

    int i1dim_c = DIM1(input1_size);
    int i1dim_h = DIM2(input1_size);
    int i1dim_w = DIM3(input1_size);
    int i1dim_chw = i1dim_c * i1dim_h * i1dim_w;
    int i1dim_hw  = i1dim_h * i1dim_w;

    float result = 0.0;

    for (int c = 0; c < i1dim_c; ++c) {
        int i1Index = b * i1dim_chw + c * i1dim_hw + y * i1dim_w + x;
        scalar_t val = input1[i1Index];
        result += static_cast<float>(val * val);
    }
    result = sqrt(result);
    output[index] = static_cast<scalar_t>(result);
}


template <typename scalar_t>
__global__ void kernel_channelnorm_backward_input1(
    const int n,
    const scalar_t* __restrict__ input1, const long4 input1_size, const long4 input1_stride,
    const scalar_t* __restrict__ output, const long4 output_size, const long4 output_stride, 
    const scalar_t* __restrict__ gradOutput, const long4 gradOutput_size, const long4 gradOutput_stride,
    scalar_t* __restrict__ gradInput, const long4 gradInput_size, const long4 gradInput_stride, 
    int norm_deg) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    float val = 0.0;

    int dim_b = DIM0(gradInput_size);
    int dim_c = DIM1(gradInput_size);
    int dim_h = DIM2(gradInput_size);
    int dim_w = DIM3(gradInput_size);
    int dim_chw = dim_c * dim_h * dim_w;
    int dim_hw  = dim_h * dim_w;

    int b = ( index / dim_chw ) % dim_b;
    int y = ( index / dim_w )   % dim_h;
    int x = ( index          )  % dim_w;


    int outIndex = b * dim_hw + y * dim_w + x;
    val = static_cast<float>(gradOutput[outIndex]) * static_cast<float>(input1[index]) / (static_cast<float>(output[outIndex])+1e-9);
    gradInput[index] = static_cast<scalar_t>(val);

}

void channelnorm_kernel_forward(
    at::Tensor& input1, 
    at::Tensor& output, 
    int norm_deg) {

    const long4 input1_size = make_long4(input1.size(0), input1.size(1), input1.size(2), input1.size(3));
    const long4 input1_stride = make_long4(input1.stride(0), input1.stride(1), input1.stride(2), input1.stride(3));

    const long4 output_size = make_long4(output.size(0), output.size(1), output.size(2), output.size(3));
    const long4 output_stride = make_long4(output.stride(0), output.stride(1), output.stride(2), output.stride(3));

    int n = output.numel();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input1.type(), "channelnorm_forward", ([&] {

      kernel_channelnorm_update_output<scalar_t><<< (n + CUDA_NUM_THREADS - 1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream() >>>(
//at::globalContext().getCurrentCUDAStream() >>>(
          n,
          input1.data<scalar_t>(), 
          input1_size,
          input1_stride, 
          output.data<scalar_t>(),
          output_size,
          output_stride, 
          norm_deg);

    }));

      // TODO: ATen-equivalent check

     // THCudaCheck(cudaGetLastError());
}

void channelnorm_kernel_backward(
    at::Tensor& input1, 
    at::Tensor& output,
    at::Tensor& gradOutput, 
    at::Tensor& gradInput1, 
    int norm_deg) {

    const long4 input1_size = make_long4(input1.size(0), input1.size(1), input1.size(2), input1.size(3));
    const long4 input1_stride = make_long4(input1.stride(0), input1.stride(1), input1.stride(2), input1.stride(3));

    const long4 output_size = make_long4(output.size(0), output.size(1), output.size(2), output.size(3));
    const long4 output_stride = make_long4(output.stride(0), output.stride(1), output.stride(2), output.stride(3));

    const long4 gradOutput_size = make_long4(gradOutput.size(0), gradOutput.size(1), gradOutput.size(2), gradOutput.size(3));
    const long4 gradOutput_stride = make_long4(gradOutput.stride(0), gradOutput.stride(1), gradOutput.stride(2), gradOutput.stride(3));

    const long4 gradInput1_size = make_long4(gradInput1.size(0), gradInput1.size(1), gradInput1.size(2), gradInput1.size(3));
    const long4 gradInput1_stride = make_long4(gradInput1.stride(0), gradInput1.stride(1), gradInput1.stride(2), gradInput1.stride(3));

    int n = gradInput1.numel();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input1.type(), "channelnorm_backward_input1", ([&] {

      kernel_channelnorm_backward_input1<scalar_t><<< (n + CUDA_NUM_THREADS - 1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream() >>>(
//at::globalContext().getCurrentCUDAStream() >>>(
          n, 
          input1.data<scalar_t>(),
          input1_size,
          input1_stride,
          output.data<scalar_t>(),
          output_size,
          output_stride,
          gradOutput.data<scalar_t>(),
          gradOutput_size,
          gradOutput_stride, 
          gradInput1.data<scalar_t>(),
          gradInput1_size,
          gradInput1_stride,
          norm_deg
    );

    }));

    // TODO: Add ATen-equivalent check

//    THCudaCheck(cudaGetLastError());
}
