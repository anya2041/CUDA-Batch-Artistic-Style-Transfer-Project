# CUDA Batch Artistic Style Transfer

This project implements a CUDA-accelerated batch artistic style transfer system. It applies the style of a reference image (e.g., a famous painting) to a batch of input images, leveraging GPU acceleration for high performance.

## Features
- Batch processing of 100+ images
- Customizable style image
- CUDA-accelerated style blending
- Output of stylized images and logs

## How to Run

1. Clone the repository:
git clone <your_repository_url>

text
2. Build the project:

cd <project_folder> && make

text
3. Run the pipeline:

./run.sh

text
4. View outputs in the `output/` directory.

## Requirements

- CUDA-capable GPU
- CUDA Toolkit
- OpenCV (for image I/O)

## Project Structure

| File/Folder   | Purpose                               |
|---------------|---------------------------------------|
| src/          | CUDA and host code                    |
| data/         | Input and style images                |
| output/       | Stylized images and logs              |
| Makefile      | Build instructions                    |
| run.sh        | Pipeline execution script             |

2. Makefile

makefile
CUDA_PATH ?= /usr/local/cuda
NVCC = $(CUDA_PATH)/bin/nvcc
CXX = g++
CXXFLAGS = -O2 -std=c++11 `pkg-config --cflags opencv4`
LDFLAGS = `pkg-config --libs opencv4` -lcudart

all: stylize

stylize: src/main.cpp src/stylize.cu
	$(NVCC) -o bin/stylize src/main.cpp src/stylize.cu $(CXXFLAGS) $(LDFLAGS)

clean:
	rm -rf bin/stylize output/*

3. run.sh

bash
#!/bin/bash
mkdir -p output
bin/stylize data/input/ data/style/style.jpg output/
