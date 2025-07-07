#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>

extern void run_style_transfer_batch(const std::vector<cv::Mat>&, const cv::Mat&, std::vector<cv::Mat>&);

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <input_dir> <style_image> <output_dir>" << std::endl;
        return 1;
    }
    std::string input_dir = argv[1], style_path = argv[2], output_dir = argv[3];

    std::vector<cv::Mat> input_images;
    for (const auto& entry : std::filesystem::directory_iterator(input_dir)) {
        cv::Mat img = cv::imread(entry.path().string());
        if (!img.empty()) input_images.push_back(img);
    }
    cv::Mat style_img = cv::imread(style_path);
    if (style_img.empty()) {
        std::cerr << "Failed to load style image." << std::endl;
        return 1;
    }
    std::vector<cv::Mat> output_images(input_images.size());
    run_style_transfer_batch(input_images, style_img, output_images);

    for (size_t i = 0; i < output_images.size(); ++i) {
        std::string out_path = output_dir + "/stylized_" + std::to_string(i) + ".jpg";
        cv::imwrite(out_path, output_images[i]);
    }
    std::cout << "Batch style transfer complete. Output saved to " << output_dir << std::endl;
    return 0;
}
