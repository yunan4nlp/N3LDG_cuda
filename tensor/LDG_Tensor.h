#ifndef LDG_TENSOR
#define LDG_TENSOR

//#include <cuda.h>
//#include <cuda_runtime.h>

#include "MyLib.h"
#include "Shape.h"
#include "Type.h"


namespace LDG {
class Tensor {
public:
    Tensor() : shape_(), v(NULL), device_type(CPU) {}

    Tensor(const Shape& shape) : shape_(shape), v(NULL) {}

    ~Tensor() {
        if(device_type == CUDA) {}
            //cudaFree(v);
        else if(device_type == CPU)
            delete []v;
    }

    /**
     * Returns the shape of the Tensor.
     * @return Shape of the Tensor.
     */
    const Shape& shape() const {
        return shape_;
    }

    int col() const {
        return shape_.dims()[1];
    }

    int row() const {
        return shape_.dims()[0];
    }

    int device_type;
    dtype *v;
    Shape shape_;
};

typedef  Tensor* PTensor;
}


#endif // !LDG_TENSOR