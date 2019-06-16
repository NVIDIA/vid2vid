#include <ATen/ATen.h>
#include <torch/torch.h>

#include "resample2d_kernel.cuh"

int resample2d_cuda_forward(
    at::Tensor& input1,
    at::Tensor& input2, 
    at::Tensor& output,
    int kernel_size) {
      resample2d_kernel_forward(input1, input2, output, kernel_size);
    return 1;
}

int resample2d_cuda_backward(
    at::Tensor& input1, 
    at::Tensor& input2,
    at::Tensor& gradOutput,
    at::Tensor& gradInput1, 
    at::Tensor& gradInput2, 
    int kernel_size) {
        resample2d_kernel_backward(input1, input2, gradOutput, gradInput1, gradInput2, kernel_size);
    return 1;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &resample2d_cuda_forward, "Resample2D forward (CUDA)");
  m.def("backward", &resample2d_cuda_backward, "Resample2D backward (CUDA)");
}

