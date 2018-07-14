cuda_add_executable(TestTensor TestTensor.cpp)
target_link_libraries(TestTensor ${LIBS})

set (EXTRA_LIBS ${EXTRA_LIBS} gpu)
target_link_libraries (TestTensor ${EXTRA_LIBS})
