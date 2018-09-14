#cmake .. -DEIGEN3_DIR=/home/yunan/gpu_NN/eigen -DN3LDG_DIR=/home/yunan/gpu_NN/N3LDG -DDEVICE_TYPE=CUDA -DCMAKE_BUILD_TYPE=Debug
rm CMakeCache.txt  CMakeFiles  Makefile cmake_install.cmake  src  tensor.out -rf
cmake .. -DN3LDG_DIR=/home/yunan/cpu/N3LDG -DDEVICE_TYPE=CUDA -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
