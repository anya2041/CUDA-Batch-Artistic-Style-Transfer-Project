CUDA_PATH ?= "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0"
NVCC = $(CUDA_PATH)/bin/nvcc.exe
CXX = g++
CXXFLAGS = -O2 -std=c++17 -IC:/Users/Anya/Downloads/opencv/build/include
LDFLAGS = -LC:/Users/Anya/Downloads/opencv/build/x64/vc16/lib -lopencv_world4110 -lcudart

SRC_DIR = src
BIN_DIR = bin

all: $(BIN_DIR)/stylize.exe

$(BIN_DIR)/stylize.exe: $(SRC_DIR)/main.cpp $(SRC_DIR)/stylize.cu
	$(NVCC) -o $(BIN_DIR)/stylize.exe $(SRC_DIR)/main.cpp $(SRC_DIR)/stylize.cu $(CXXFLAGS) $(LDFLAGS)

clean:
	del /Q $(BIN_DIR)\*.exe
