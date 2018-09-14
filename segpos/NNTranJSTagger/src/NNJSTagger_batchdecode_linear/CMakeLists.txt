
include_directories(
  model
)

if(DEVICE_TYPE MATCHES CUDA)
	cuda_add_executable(NNJSTagger_batchdecode_linear NNJSTagger.cpp)
endif()

if(DEVICE_TYPE MATCHES EIGEN)
	add_executable(NNJSTagger_batchdecode_linear NNJSTagger.cpp)
	target_link_libraries(NNJSTagger_batchdecode_linear ${MKL_LIBS})
endif()

target_link_libraries(NNJSTagger_batchdecode_linear ${LIBS})
