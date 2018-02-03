CC=g++
NVCC=/usr/local/cuda-8.0/bin/nvcc

#GPU CONFIGURATION
GC_MAIN=example_radix_select.cu
GC_EXE=gpu_run
#NVCC_FLAGS = --ptxas-options=-v -gencode arch=compute_35,code=sm_35 -rdc=true
ARCH = -gencode arch=compute_61,code=sm_61
#ARCH = -gencode arch=compute_35,code=sm_35

all: gpu_cc

gpu_cc:
	$(NVCC) -std=c++11 $(ARCH) $(GC_MAIN) -o $(GC_EXE) -I cub-1.7.4/
	
clean:
	rm -rf $(GC_EXE)  
