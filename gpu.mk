#GPU build script

#please put here cuda install path. 
#on Linux type "which nvcc" and if the result is
#                  /exports/applications/apps/cuda/rhel5/4.2/cuda/bin/nvcc
#then the install path is 
CUDA_INSTALL_PATH= /exports/applications/apps/cuda/rhel5/4.2/cuda

# Tested CUDA versions are
# 4.2, 4.0rc2, 4.0
# We tested the code on Tesla M2050
#============================================================================================
# You should not modify the lines below

CUDA_COMPILER=nvcc
CUDA_INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include -I/usr/local/include
CUDA_LIB =  -L$(CUDA_INSTALL_PATH)/lib -L$(CUDA_INSTALL_PATH)/lib64  -lcublas -lm -arch sm_20 -lgomp
CUDA_COMILER_FLAGS= -O3 -w 


gpu_console:  
	nvcc -O3 -w $(CUDA_INCLUDES) $(FRONTENDFOLDER)gpu_console.cu      $(CUDA_LIB)  -o $(BUILD_FOLDER)gpu_console

gpu_test: gpu_console
	./$(BUILD_FOLDER)gpu_console -d datasets/small.csv  -r results/small_gpu.txt -v true -p double -a 1 -n 3
	./$(BUILD_FOLDER)gpu_console -d datasets/small.csv  -r results/small_2_gpu.txt -v true -p double -s 1000 -b 64  -a 1 -n 2
	./$(BUILD_FOLDER)gpu_console -d datasets/small.csv  -r results/small_3_gpu.txt -v true -p double -s 1000 -b 64 -u 1 -a 1 -n 2

gpu: gpu_test 
