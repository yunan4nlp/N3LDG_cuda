#export CUDA_BIN_PATH=/usr/local/cuda-8.0 
rm  CMakeCache.txt  CMakeFiles  cmake_install.cmake  Makefile  src  tensor.out -rf
path=/home/yunan/cpu
#cmake .. -DEIGEN3_DIR=${path}/eigen -DN3LDG_DIR=${path}/N3LDG -DDEVICE_TYPE=EIGEN -DMKL=TRUE
cmake .. -DEIGEN3_DIR=${path}/eigen -DN3LDG_DIR=${path}/N3LDG -DDEVICE_TYPE=EIGEN -DCMAKE_BUILD_TYPE=TRUE -DMKL=TRUE
#cmake .. -DEIGEN3_DIR=/home/yunan/gpu_NN/eigen -DN3L_DIR=/home/yunan/for20/N3LDG_cuda/N3LDG -DDEVICE_TYPE=CUDA
