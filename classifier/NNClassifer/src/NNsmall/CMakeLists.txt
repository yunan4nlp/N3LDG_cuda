
include_directories(
  model
)

if(DEVICE_TYPE MATCHES CUDA)
	cuda_add_executable(NNsmall NNCNNLabeler.cpp)
	target_link_libraries(NNsmall ${EXTRA_LIBS})
endif()

if(DEVICE_TYPE MATCHES EIGEN)
	add_executable(NNsmall NNCNNLabeler.cpp)
	target_link_libraries(NNsmall ${EXTRA_LIBS})
endif()
target_link_libraries(NNsmall ${LIBS})
