# SDK_PATH=/home/apostolis/NVIDIA_GPU_Computing_SDK
SDK_PATH=/cuda_sdk
CUDA_INSTALL_PATH=/usr/local/cuda
CFLAGS= -I. -I$(CUDA_INSTALL_PATH)/include -I$(SDK_PATH)/C/common/inc/
LDFLAGS= -L$(CUDA_INSTALL_PATH)/lib64 -L$(SDK_PATH)/C/lib -lcutil_x86_64 -lcudart -lcublas



all:
	
	nvcc -O3 $(CUDAFLAGS) $(CFLAGS) $(LDFLAGS) cpp_cmeans.cu -o cuda_exec
	

clean:
	rm -f exec
