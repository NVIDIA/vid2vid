#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>

#define CUDA_NUM_THREADS 512 
#define THREADS_PER_BLOCK 64 

#define DIM0(TENSOR) ((TENSOR).x)
#define DIM1(TENSOR) ((TENSOR).y)
#define DIM2(TENSOR) ((TENSOR).z)
#define DIM3(TENSOR) ((TENSOR).w)

#define DIM3_INDEX(TENSOR, xx, yy, zz, ww) ((TENSOR)[((xx) * (TENSOR##_stride.x)) + ((yy) * (TENSOR##_stride.y)) + ((zz) * (TENSOR##_stride.z)) + ((ww) * (TENSOR##_stride.w))])

template <typename scalar_t>
__global__ void kernel_resample2d_update_output(const int n, 
                                               const scalar_t* __restrict__ input1, const long4 input1_size, const long4 input1_stride,
                                               const scalar_t* __restrict__ input2, const long4 input2_size, const long4 input2_stride, 
                                               scalar_t* __restrict__ output, const long4 output_size, const long4 output_stride, int kernel_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    scalar_t val = 0.0f;

    int dim_b = DIM0(output_size);
    int dim_c = DIM1(output_size);
    int dim_h = DIM2(output_size);
    int dim_w = DIM3(output_size);
    int dim_chw = dim_c * dim_h * dim_w;
    int dim_hw  = dim_h * dim_w;

    int b = ( index / dim_chw ) % dim_b;
    int c = ( index / dim_hw )  % dim_c;
    int y = ( index / dim_w )   % dim_h;
    int x = ( index          )  % dim_w;

    scalar_t dx = DIM3_INDEX(input2, b, 0, y, x);
    scalar_t dy = DIM3_INDEX(input2, b, 1, y, x);

    scalar_t xf = static_cast<scalar_t>(x) + dx;
    scalar_t yf = static_cast<scalar_t>(y) + dy;
    scalar_t alpha = xf - floor(xf); // alpha
    scalar_t beta = yf - floor(yf); // beta

    int xL = max(min( int (floor(xf)),    dim_w-1), 0);
    int xR = max(min( int (floor(xf)+1), dim_w -1), 0);
    int yT = max(min( int (floor(yf)),    dim_h-1), 0);
    int yB = max(min( int (floor(yf)+1),  dim_h-1), 0);

    for (int fy = 0; fy < kernel_size; fy += 1) {
        for (int fx = 0; fx < kernel_size; fx += 1) {
            val += static_cast<float>((1. - alpha)*(1. - beta) * DIM3_INDEX(input1, b, c, yT + fy, xL + fx));
            val += static_cast<float>((alpha)*(1. - beta) * DIM3_INDEX(input1, b, c, yT + fy, xR + fx));
            val += static_cast<float>((1. - alpha)*(beta) * DIM3_INDEX(input1, b, c, yB + fy, xL + fx));
            val += static_cast<float>((alpha)*(beta) * DIM3_INDEX(input1, b, c, yB + fy, xR + fx));
        }
    }

    output[index] = val;

}


template <typename scalar_t>
__global__ void kernel_resample2d_backward_input1(
    const int n, const scalar_t* __restrict__ input1, const long4 input1_size, const long4 input1_stride,
    const scalar_t* __restrict__ input2, const long4 input2_size, const long4 input2_stride,
    const scalar_t* __restrict__ gradOutput, const long4 gradOutput_size, const long4 gradOutput_stride,
    scalar_t* __restrict__ gradInput, const long4 gradInput_size, const long4 gradInput_stride, int kernel_size) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    int dim_b = DIM0(gradOutput_size);
    int dim_c = DIM1(gradOutput_size);
    int dim_h = DIM2(gradOutput_size);
    int dim_w = DIM3(gradOutput_size);
    int dim_chw = dim_c * dim_h * dim_w;
    int dim_hw  = dim_h * dim_w;

    int b = ( index / dim_chw ) % dim_b;
    int c = ( index / dim_hw )  % dim_c;
    int y = ( index / dim_w )   % dim_h;
    int x = ( index          )  % dim_w;

    scalar_t dx = DIM3_INDEX(input2, b, 0, y, x);
    scalar_t dy = DIM3_INDEX(input2, b, 1, y, x);

    scalar_t xf = static_cast<scalar_t>(x) + dx;
    scalar_t yf = static_cast<scalar_t>(y) + dy;
    scalar_t alpha = xf - int(xf); // alpha
    scalar_t beta = yf - int(yf); // beta

    int idim_h = DIM2(input1_size);
    int idim_w = DIM3(input1_size);

    int xL = max(min( int (floor(xf)),    idim_w-1), 0);
    int xR = max(min( int (floor(xf)+1), idim_w -1), 0);
    int yT = max(min( int (floor(yf)),    idim_h-1), 0);
    int yB = max(min( int (floor(yf)+1),  idim_h-1), 0);

    for (int fy = 0; fy < kernel_size; fy += 1) {
        for (int fx = 0; fx < kernel_size; fx += 1) {
            atomicAdd(&DIM3_INDEX(gradInput, b, c, (yT + fy), (xL + fx)), (1-alpha)*(1-beta) * DIM3_INDEX(gradOutput, b, c, y, x));
            atomicAdd(&DIM3_INDEX(gradInput, b, c, (yT + fy), (xR + fx)),   (alpha)*(1-beta) * DIM3_INDEX(gradOutput, b, c, y, x));
            atomicAdd(&DIM3_INDEX(gradInput, b, c, (yB + fy), (xL + fx)),   (1-alpha)*(beta) * DIM3_INDEX(gradOutput, b, c, y, x));
            atomicAdd(&DIM3_INDEX(gradInput, b, c, (yB + fy), (xR + fx)),     (alpha)*(beta) * DIM3_INDEX(gradOutput, b, c, y, x));
        }
    }

}

template <typename scalar_t>
__global__ void kernel_resample2d_backward_input2(
    const int n, const scalar_t* __restrict__ input1, const long4 input1_size, const long4 input1_stride,
    const scalar_t* __restrict__ input2, const long4 input2_size, const long4 input2_stride,
    const scalar_t* __restrict__ gradOutput, const long4 gradOutput_size, const long4 gradOutput_stride,
    scalar_t* __restrict__ gradInput, const long4 gradInput_size, const long4 gradInput_stride, int kernel_size) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= n) {
        return;
    }

    scalar_t output = 0.0;
    int kernel_rad = (kernel_size - 1)/2;

    int dim_b = DIM0(gradInput_size);
    int dim_c = DIM1(gradInput_size);
    int dim_h = DIM2(gradInput_size);
    int dim_w = DIM3(gradInput_size);
    int dim_chw = dim_c * dim_h * dim_w;
    int dim_hw  = dim_h * dim_w;

    int b = ( index / dim_chw ) % dim_b;
    int c = ( index / dim_hw )  % dim_c;
    int y = ( index / dim_w )   % dim_h;
    int x = ( index          )  % dim_w;

    int odim_c = DIM1(gradOutput_size);

    scalar_t dx = DIM3_INDEX(input2, b, 0, y, x);
    scalar_t dy = DIM3_INDEX(input2, b, 1, y, x);

    scalar_t xf = static_cast<scalar_t>(x) + dx;
    scalar_t yf = static_cast<scalar_t>(y) + dy;

    int xL = max(min( int (floor(xf)),    dim_w-1), 0);
    int xR = max(min( int (floor(xf)+1), dim_w -1), 0);
    int yT = max(min( int (floor(yf)),    dim_h-1), 0);
    int yB = max(min( int (floor(yf)+1),  dim_h-1), 0);
    
    if (c % 2) {
        float gamma = 1 - (xf - floor(xf)); // alpha
        for (int i = 0; i <= 2*kernel_rad; ++i) {
            for (int j = 0; j <= 2*kernel_rad; ++j) {
                for (int ch = 0; ch < odim_c; ++ch) {
                    output += (gamma) * DIM3_INDEX(gradOutput, b, ch, y, x) * DIM3_INDEX(input1, b, ch, (yB + j), (xL + i));
                    output -= (gamma) * DIM3_INDEX(gradOutput, b, ch, y, x) * DIM3_INDEX(input1, b, ch, (yT + j), (xL + i));
                    output += (1-gamma) * DIM3_INDEX(gradOutput, b, ch, y, x) * DIM3_INDEX(input1, b, ch, (yB + j), (xR + i));
                    output -= (1-gamma) * DIM3_INDEX(gradOutput, b, ch, y, x) * DIM3_INDEX(input1, b, ch, (yT + j), (xR + i));
                }
            }
        }
    }
    else {
        float gamma = 1 - (yf - floor(yf)); // alpha
        for (int i = 0; i <= 2*kernel_rad; ++i) {
            for (int j = 0; j <= 2*kernel_rad; ++j) {
                for (int ch = 0; ch < odim_c; ++ch) {
                    output += (gamma) * DIM3_INDEX(gradOutput, b, ch, y, x) * DIM3_INDEX(input1, b, ch, (yT + j), (xR + i));
                    output -= (gamma) * DIM3_INDEX(gradOutput, b, ch, y, x) * DIM3_INDEX(input1, b, ch, (yT + j), (xL + i));
                    output += (1-gamma) * DIM3_INDEX(gradOutput, b, ch, y, x) * DIM3_INDEX(input1, b, ch, (yB + j), (xR + i));
                    output -= (1-gamma) * DIM3_INDEX(gradOutput, b, ch, y, x) * DIM3_INDEX(input1, b, ch, (yB + j), (xL + i));
                }
            }
        }

    }

    gradInput[index] = output;

}

void resample2d_kernel_forward(
    at::Tensor& input1, 
    at::Tensor& input2,
    at::Tensor& output, 
    int kernel_size) {

    int n = output.numel();

    const long4 input1_size = make_long4(input1.size(0), input1.size(1), input1.size(2), input1.size(3));
    const long4 input1_stride = make_long4(input1.stride(0), input1.stride(1), input1.stride(2), input1.stride(3));

    const long4 input2_size = make_long4(input2.size(0), input2.size(1), input2.size(2), input2.size(3));
    const long4 input2_stride = make_long4(input2.stride(0), input2.stride(1), input2.stride(2), input2.stride(3));

    const long4 output_size = make_long4(output.size(0), output.size(1), output.size(2), output.size(3));
    const long4 output_stride = make_long4(output.stride(0), output.stride(1), output.stride(2), output.stride(3));

    // TODO: when atomicAdd gets resolved, change to AT_DISPATCH_FLOATING_TYPES_AND_HALF
//    AT_DISPATCH_FLOATING_TYPES(input1.type(), "resample_forward_kernel", ([&] {

        kernel_resample2d_update_output<float><<< (n + CUDA_NUM_THREADS - 1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream() >>>(
//at::globalContext().getCurrentCUDAStream() >>>(
            n,
            input1.data<float>(),
            input1_size,
            input1_stride, 
            input2.data<float>(),
            input2_size,
            input2_stride,
            output.data<float>(),
            output_size,
            output_stride,
            kernel_size);

//    }));

        // TODO: ATen-equivalent check

       //    THCudaCheck(cudaGetLastError());

}

void resample2d_kernel_backward(
    at::Tensor& input1,
    at::Tensor& input2,
    at::Tensor& gradOutput,
    at::Tensor& gradInput1,
    at::Tensor& gradInput2,
    int kernel_size) {

    int n = gradOutput.numel();

    const long4 input1_size = make_long4(input1.size(0), input1.size(1), input1.size(2), input1.size(3));
    const long4 input1_stride = make_long4(input1.stride(0), input1.stride(1), input1.stride(2), input1.stride(3));

    const long4 input2_size = make_long4(input2.size(0), input2.size(1), input2.size(2), input2.size(3));
    const long4 input2_stride = make_long4(input2.stride(0), input2.stride(1), input2.stride(2), input2.stride(3));

    const long4 gradOutput_size = make_long4(gradOutput.size(0), gradOutput.size(1), gradOutput.size(2), gradOutput.size(3));
    const long4 gradOutput_stride = make_long4(gradOutput.stride(0), gradOutput.stride(1), gradOutput.stride(2), gradOutput.stride(3));

    const long4 gradInput1_size = make_long4(gradInput1.size(0), gradInput1.size(1), gradInput1.size(2), gradInput1.size(3));
    const long4 gradInput1_stride = make_long4(gradInput1.stride(0), gradInput1.stride(1), gradInput1.stride(2), gradInput1.stride(3));

//    AT_DISPATCH_FLOATING_TYPES(input1.type(), "resample_backward_input1", ([&] {

        kernel_resample2d_backward_input1<float><<< (n + CUDA_NUM_THREADS - 1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream() >>>(
//at::globalContext().getCurrentCUDAStream() >>>(
            n, 
            input1.data<float>(), 
            input1_size,
            input1_stride,
            input2.data<float>(),
            input2_size, 
            input2_stride,
            gradOutput.data<float>(),
            gradOutput_size,
            gradOutput_stride,
            gradInput1.data<float>(),
            gradInput1_size,
            gradInput1_stride, 
            kernel_size
        );

//    }));

    const long4 gradInput2_size = make_long4(gradInput2.size(0), gradInput2.size(1), gradInput2.size(2), gradInput2.size(3));
    const long4 gradInput2_stride = make_long4(gradInput2.stride(0), gradInput2.stride(1), gradInput2.stride(2), gradInput2.stride(3));

    n = gradInput2.numel();

//    AT_DISPATCH_FLOATING_TYPES(gradInput2.type(), "resample_backward_input2", ([&] {


        kernel_resample2d_backward_input2<float><<< (n + CUDA_NUM_THREADS - 1)/CUDA_NUM_THREADS, CUDA_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream() >>>(
//at::globalContext().getCurrentCUDAStream() >>>(
            n, 
            input1.data<float>(), 
            input1_size, 
            input1_stride,
            input2.data<float>(), 
            input2_size,
            input2_stride,
            gradOutput.data<float>(),
            gradOutput_size,
            gradOutput_stride,
            gradInput2.data<float>(),
            gradInput2_size,
            gradInput2_stride,
            kernel_size
       );

//    }));

    // TODO: Use the ATen equivalent to get last error

    //    THCudaCheck(cudaGetLastError());

}
