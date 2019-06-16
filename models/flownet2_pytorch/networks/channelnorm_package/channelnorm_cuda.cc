#include <torch/torch.h>
#include <ATen/ATen.h>

#include "channelnorm_kernel.cuh"

int channelnorm_cuda_forward(
    at::Tensor& input1, 
    at::Tensor& output,
    int norm_deg) {

    channelnorm_kernel_forward(input1, output, norm_deg);
    return 1;
}


int channelnorm_cuda_backward(
    at::Tensor& input1, 
    at::Tensor& output,
    at::Tensor& gradOutput,
    at::Tensor& gradInput1,
    int norm_deg) {

    channelnorm_kernel_backward(input1, output, gradOutput, gradInput1, norm_deg);
    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &channelnorm_cuda_forward, "Channel norm forward (CUDA)");
  m.def("backward", &channelnorm_cuda_backward, "Channel norm backward (CUDA)");
}

