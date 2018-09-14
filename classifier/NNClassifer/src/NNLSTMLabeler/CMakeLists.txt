
include_directories(
  model
)
add_executable(NNLSTMLabeler NNCNNLabeler.cpp)
target_link_libraries(NNLSTMLabeler ${LIBS})


#cuda_add_executable(NNCNNLabeler NNCNNLabeler.cpp)
#target_link_libraries(NNCNNLabeler ${LIBS})

#set (EXTRA_LIBS ${EXTRA_LIBS} gpu)
#target_link_libraries (NNCNNLabeler ${EXTRA_LIBS})

