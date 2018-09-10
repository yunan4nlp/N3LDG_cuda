#ifndef LDG_TENSOR
#define LDG_TENSOR

//#include <cuda.h>
//#include <cuda_runtime.h>
#include <memory>

#include "MyLib.h"
#include "Shape.h"
#include "Type.h"


namespace LDG {
class Tensor {
public:
  Tensor() : shape_(), v(NULL), device_type(CPU) {
  }

  Tensor(const Shape& shape) : shape_(shape), v(NULL) {
  }

  ~Tensor() {
    if(device_type == CUDA && v != NULL)
      //cudaFree(v);
    if(device_type == CPU && v != NULL)
      delete v;
    v = NULL;
  }

  /**
   * Returns the shape of the Tensor.
   * @return Shape of the Tensor.
   */
  const Shape& shape() const {
    return shape_;
  }

  int device_type;
  dtype *v;
  Shape shape_;

  const void* get_handle() const {
    return handle_.get();
  }

  int col() const {
    return shape_.dims()[1];
  }

  int row() const {
    return shape_.dims()[0];
  }


  std::shared_ptr<void> handle_;
};

typedef  Tensor* PTensor;
}


#endif // !LDG_TENSOR
