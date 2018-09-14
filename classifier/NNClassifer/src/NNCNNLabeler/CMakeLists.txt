
include_directories(
  model
)

if(DEVICE_TYPE MATCHES CUDA)
	cuda_add_executable(NNCNNLabeler NNCNNLabeler.cpp)
endif()

if(DEVICE_TYPE MATCHES EIGEN)
	add_executable(NNCNNLabeler NNCNNLabeler.cpp)
endif()


target_link_libraries(NNCNNLabeler ${LIBS})
