#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h> 
#include <curand_kernel.h>

#include "MyLib.h"

void Fadd_impl(const dtype* x, const dtype* y,  dtype* r, int size);

void Fadd_impl(const dtype* x, dtype** y,  dtype* r, int x_size, int size);

void Fadd_impl(dtype** x, dtype** y,  dtype** r, int dim0, int size);

void Fsubtract_impl(const dtype* x, const dtype* y, dtype* r, int size);

void Fmultiply_impl(const dtype* x, const dtype* y, dtype* r, int size);

void Fmultiply_impl(dtype** x, dtype** y, dtype** r, int dim0, int size);

void Fdivide_impl(const dtype* x, const dtype* y, dtype* r, int size);

void Fmultiply_scalar_impl(const dtype* x, const dtype y, dtype* r, int size);

void Fadd_scalar_impl(const dtype* x, const dtype y, dtype* r, int size);

void Fsquare_impl(const dtype* x, dtype* r, int size);

void Fsqrt_impl(const dtype* x, dtype* r, int size);

void Ftanh_impl(const dtype* x, dtype* r, int size);

void Dtanh_impl(const dtype* x, const dtype* y, dtype* r, int size);

void Ftanh_impl(dtype** x, dtype** r, int dim0, int size);

void Dtanh_impl(dtype** x, dtype** y, dtype** r, int dim0, int size);

void Fsigmoid_impl(dtype** x, dtype** r, int dim0, int size);

void Dsigmoid_impl(dtype** x, dtype** y, dtype** r, int dim0, int size);

void Fsigmoid_impl(const dtype* x, dtype* r, int size);

void Dsigmoid_impl(const dtype* x, const dtype* y, dtype* r, int size);

void concat_impl(const dtype *src, dtype* dst, int offset, int dim);

void unconcat_impl(const dtype *src, dtype* dst, int offset, int dim);

void Ftranspose_impl(const dtype* x, dtype* r, int dim0, int dim1, int size);


void set_col_impl(dtype* x, int dim0, int col, int size, dtype val);

void get_col_impl(const dtype* x, dtype* r, int dim0, int col, int size);

void get_cols_impl(const dtype* x, dtype* r, int xdim0, int xdim1, int r_size, int* cols, int col_num);

void FLookup_impl(const dtype* x, dtype** r, int xdim0, int xdim1, int r_size, int* cols, int col_num);

void DLookup_impl(dtype* gx, dtype** loss, int gxdim0, int gxdim1, int l_size, int* cols, int col_num);

void Fadd_col_impl(dtype* x, const dtype* y, int col, int dim0, int size);


void Favgpooling_impl(dtype** x, dtype* y, int n, int r, int s);

void Davgpooling_impl(const dtype* gy, int gy_size, int gx_size, int n, dtype** gx);

void Fsumpooling_impl(dtype** x, dtype* y, int n, int r, int s);

void Dsumpooling_impl(const dtype* gy, int gy_size, int gx_size, dtype** gx);

void Fmaxpooling_impl(dtype** x, dtype* y, int n, int r, int s, int* index);

void Dmaxpooling_impl(const dtype* gy, dtype** gx, int* index, int dim);

void Fminpooling_impl(dtype** x, dtype* y, int n, int r, int s, int* index);

void Dminpooling_impl(const dtype* gy, dtype** gx, int* index, int dim);
