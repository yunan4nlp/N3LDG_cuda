
include_directories(
  model
)


if(DEVICE_TYPE MATCHES CUDA)
	cuda_add_executable(NNJSTagger_batchdecode NNJSTagger.cpp)
endif()

if(DEVICE_TYPE MATCHES EIGEN)
	add_executable(NNJSTagger_batchdecode NNJSTagger.cpp)
endif()

target_link_libraries(NNJSTagger_batchdecode ${LIBS})

