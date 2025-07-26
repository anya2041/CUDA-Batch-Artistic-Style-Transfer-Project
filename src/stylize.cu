// stylize.cu
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>

#define CUDA_CHECK(x) do { \
  cudaError_t err = (x); \
  if (err != cudaSuccess) { \
    std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__ << " - " \
              << cudaGetErrorString(err) << std::endl; \
    std::exit(EXIT_FAILURE); \
  } \
} while (0)

__global__ void stylize_kernel(const unsigned char* __restrict__ input,
                               const unsigned char* __restrict__ style,
                               unsigned char* __restrict__ output,
                               int num_pixels, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pixels) return;

    int base = idx * 3; // BGR
    float ia = alpha;
    float sa = 1.0f - alpha;

    output[base + 0] = (unsigned char)(ia * input[base + 0] + sa * style[base + 0]);
    output[base + 1] = (unsigned char)(ia * input[base + 1] + sa * style[base + 1]);
    output[base + 2] = (unsigned char)(ia * input[base + 2] + sa * style[base + 2]);
}

void run_style_transfer_batch(const std::vector<cv::Mat>& inputs,
                              const cv::Mat& style,
                              std::vector<cv::Mat>& outputs) {
    CV_Assert(!inputs.empty());
    CV_Assert(style.type() == CV_8UC3);

    const float alpha = 0.6f;

    for (size_t i = 0; i < inputs.size(); ++i) {
        CV_Assert(inputs[i].type() == CV_8UC3);
        CV_Assert(inputs[i].isContinuous());

        cv::Mat resized_style;
        cv::resize(style, resized_style, inputs[i].size());
        CV_Assert(resized_style.isContinuous());

        const int rows = inputs[i].rows;
        const int cols = inputs[i].cols;
        const int num_pixels = rows * cols;
        const size_t bytes = static_cast<size_t>(num_pixels) * 3 * sizeof(unsigned char);

        cv::Mat out_img(rows, cols, CV_8UC3);

        unsigned char *d_input = nullptr, *d_style = nullptr, *d_output = nullptr;
        CUDA_CHECK(cudaMalloc(&d_input,  bytes));
        CUDA_CHECK(cudaMalloc(&d_style,  bytes));
        CUDA_CHECK(cudaMalloc(&d_output, bytes));

        CUDA_CHECK(cudaMemcpy(d_input,  inputs[i].ptr<unsigned char>(),  bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_style,  resized_style.ptr<unsigned char>(), bytes, cudaMemcpyHostToDevice));

        int threads = 256;
        int blocks  = (num_pixels + threads - 1) / threads;

        stylize_kernel<<<blocks, threads>>>(d_input, d_style, d_output, num_pixels, alpha);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(out_img.ptr<unsigned char>(), d_output, bytes, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_style));
        CUDA_CHECK(cudaFree(d_output));

        outputs[i] = out_img;
    }
}
