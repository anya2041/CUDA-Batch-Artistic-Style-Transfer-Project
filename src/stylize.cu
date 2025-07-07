#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <vector>

__global__ void stylize_kernel(const uchar3* input, const uchar3* style, uchar3* output, int img_size, float alpha) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < img_size) {
        output[idx].x = alpha * input[idx].x + (1 - alpha) * style[idx].x;
        output[idx].y = alpha * input[idx].y + (1 - alpha) * style[idx].y;
        output[idx].z = alpha * input[idx].z + (1 - alpha) * style[idx].z;
    }
}

void run_style_transfer_batch(const std::vector<cv::Mat>& inputs, const cv::Mat& style, std::vector<cv::Mat>& outputs) {
    float alpha = 0.6f;
    int img_size = inputs[0].rows * inputs[0].cols;
    for (size_t i = 0; i < inputs.size(); ++i) {
        cv::Mat resized_style;
        cv::resize(style, resized_style, inputs[i].size());
        cv::Mat out_img(inputs[i].rows, inputs[i].cols, inputs[i].type());

        uchar3 *d_input, *d_style, *d_output;
        cudaMalloc(&d_input, img_size * sizeof(uchar3));
        cudaMalloc(&d_style, img_size * sizeof(uchar3));
        cudaMalloc(&d_output, img_size * sizeof(uchar3));

        cudaMemcpy(d_input, inputs[i].ptr(), img_size * sizeof(uchar3), cudaMemcpyHostToDevice);
        cudaMemcpy(d_style, resized_style.ptr(), img_size * sizeof(uchar3), cudaMemcpyHostToDevice);

        int threads = 256;
        int blocks = (img_size + threads - 1) / threads;
        stylize_kernel<<<blocks, threads>>>(d_input, d_style, d_output, img_size, alpha);

        cudaMemcpy(out_img.ptr(), d_output, img_size * sizeof(uchar3), cudaMemcpyDeviceToHost);

        cudaFree(d_input); cudaFree(d_style); cudaFree(d_output);

        outputs[i] = out_img;
    }
}
