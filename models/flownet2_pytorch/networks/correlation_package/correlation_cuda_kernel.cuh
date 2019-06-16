#pragma once

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <cuda_runtime.h>

int correlation_forward_cuda_kernel(at::Tensor& output,
    int ob,
    int oc,
    int oh,
    int ow,
    int osb,
    int osc,
    int osh,
    int osw,

    at::Tensor& input1,
    int ic,
    int ih,
    int iw,
    int isb,
    int isc,
    int ish,
    int isw,

    at::Tensor& input2,
    int gc,
    int gsb,
    int gsc,
    int gsh,
    int gsw,

    at::Tensor& rInput1,
    at::Tensor& rInput2,
    int pad_size,
    int kernel_size,
    int max_displacement,
    int stride1,
    int stride2,
    int corr_type_multiply,
    cudaStream_t stream);


int correlation_backward_cuda_kernel(   
    at::Tensor& gradOutput,
    int gob,
    int goc,
    int goh,
    int gow,
    int gosb,
    int gosc,
    int gosh,
    int gosw,

    at::Tensor& input1,
    int ic,
    int ih,
    int iw,
    int isb,
    int isc,
    int ish,
    int isw,

    at::Tensor& input2,
    int gsb,
    int gsc,
    int gsh,
    int gsw,

    at::Tensor& gradInput1, 
    int gisb,
    int gisc,
    int gish,
    int gisw,

    at::Tensor& gradInput2,
    int ggc,
    int ggsb,
    int ggsc,
    int ggsh,
    int ggsw,

    at::Tensor& rInput1,
    at::Tensor& rInput2,
    int pad_size,
    int kernel_size,
    int max_displacement,
    int stride1,
    int stride2,
    int corr_type_multiply,
    cudaStream_t stream);
