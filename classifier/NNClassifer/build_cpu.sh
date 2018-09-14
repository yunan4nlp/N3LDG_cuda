#cmake .. -DEIGEN3_DIR=/home/yunan/gpu_NN/eigen -DN3LDG_DIR=/home/yunan/gpu_NN/N3LDG -DDEVICE_TYPE=CUDA -DCMAKE_BUILD_TYPE=Debug
rm CMakeCache.txt  CMakeFiles  Makefile cmake_install.cmake  src  tensor.out -rf
path=/home/yunan/cpu
#cmake .. -DEIGEN3_DIR=${path}/eigen -DN3LDG_DIR=${path}/N3LDG -DDEVICE_TYPE=EIGEN -DCMAKE_BUILD_TYPE=Debug
cmake .. -DEIGEN3_DIR=${path}/eigen -DN3LDG_DIR=${path}/N3LDG -DDEVICE_TYPE=EIGEN -DMKL=TRUE



cmake .. -DEIGEN3_DIR=E:\vs_workspace2\N3LDG_cpu\eigen -DN3LDG_DIR=E:\vs_workspace2\N3LDG_cpu\N3LDG -DDEVICE_TYPE=EIGEN

