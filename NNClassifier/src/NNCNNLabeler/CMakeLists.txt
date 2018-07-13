
include_directories(
  model
)

cuda_add_executable(NNCNNLabeler NNCNNLabeler.cpp)
target_link_libraries(NNCNNLabeler ${LIBS})

set (EXTRA_LIBS ${EXTRA_LIBS} gpu)
target_link_libraries (NNCNNLabeler ${EXTRA_LIBS})
