#pragma once

#include <ATen/ATen.h>

void resample2d_kernel_forward(
    at::Tensor& input1,
    at::Tensor& input2,
    at::Tensor& output,
    int kernel_size);

void resample2d_kernel_backward(
    at::Tensor& input1,
    at::Tensor& input2,
    at::Tensor& gradOutput,
    at::Tensor& gradInput1, 
    at::Tensor& gradInput2, 
    int kernel_size);
