
include_directories(
  model
)

cuda_add_executable(NNCNNLabeler_drop NNCNNLabeler.cpp)

target_link_libraries(NNCNNLabeler_drop ${LIBS})
target_link_libraries (NNCNNLabeler_drop ${EXTRA_LIBS})

