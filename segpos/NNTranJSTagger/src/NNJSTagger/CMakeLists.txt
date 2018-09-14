
include_directories(
  model
)

#add_executable(NNJSTagger NNJSTagger.cpp)
cuda_add_executable(NNJSTagger NNJSTagger.cpp)
target_link_libraries(NNJSTagger ${LIBS})

set (EXTRA_LIBS ${EXTRA_LIBS} gpu)
target_link_libraries (NNJSTagger ${EXTRA_LIBS})
