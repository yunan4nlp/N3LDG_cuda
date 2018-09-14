if(DEVICE_TYPE MATCHES CUDA)
	cuda_add_executable(TestTensor TestTensor.cpp)
endif()

if(DEVICE_TYPE MATCHES EIGEN)
	add_executable(TestTensor TestTensor.cpp)
endif()
target_link_libraries(TestTensor ${LIBS})
