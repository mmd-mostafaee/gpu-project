nvcc_options= -gencode arch=compute_30,code=sm_30 -lm -D --compiler-options -Wall 
sources = kernel.cu

all: kernel kernel_optimized

kernel: $(sources) Makefile Header.h
	nvcc -o kernel $(sources) $(nvcc_options)

kernel_optimized: $(sources) Makefile Header.h
	nvcc -o kernel_optimized $(sources) $(nvcc_options) -O3

clean:
	rm kernel
	rm kernel_optimized