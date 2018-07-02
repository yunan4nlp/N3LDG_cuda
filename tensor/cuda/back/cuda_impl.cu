#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda_impl.cuh"

void CudaDevice::random_uniform(LDG::Tensor &t, const Shape &shape, float lower, float upper) {}

void CudaDevice::random_bernoulli(LDG::Tensor &t, const Shape &shape, float p) {}

void CudaDevice:: random_normal(LDG::Tensor &t, const Shape &shape, float mean, float sd) {}
void CudaDevice:: random_log_normal(LDG::Tensor &t, const Shape &shape, float mean, float sd) {}

void CudaDevice:: Fequal(const LDG::Tensor& x, LDG::Tensor& r) {}
void CudaDevice:: Ftanh(const LDG::Tensor& x, LDG::Tensor& r) {}
void CudaDevice:: Fsigmoid(const LDG::Tensor& x, LDG::Tensor& r) {}
void CudaDevice:: Frelu(const LDG::Tensor& x, LDG::Tensor& r) {}
void CudaDevice:: Fleaky_relu(const LDG::Tensor& x, LDG::Tensor& r) {}
void CudaDevice:: Fexp(const LDG::Tensor& x, LDG::Tensor& r) {}
void CudaDevice:: Flog(const LDG::Tensor& x, LDG::Tensor& r) {}

void CudaDevice:: Dequal(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {}
void CudaDevice:: Dtanh(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {}
void CudaDevice:: Dleaky_relu(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {}
void CudaDevice:: Dsigmoid(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {}
void CudaDevice:: Drelu(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {}
void CudaDevice:: Dexp(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {}
void CudaDevice:: Dlog(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {}

__global__ void my_add_kernel(const dtype *x, const dtype *y, dtype *r)  {  
	int i = blockIdx.x;  
	r[i] = x[i] + y[i];  
}

void CudaDevice:: Fadd(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {
	int size = x.shape().size();
	int numBlocks = 8;
	int numThreadsPerBlock = 8;
	dim3 dimGrid(numBlocks,1,1);  
	dim3 dimBlock(numThreadsPerBlock,1,1); 
	my_add_kernel<<<2,1,1>>>(x.v, y.v, r.v);
}

void CudaDevice:: Fsubtract(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {}
void CudaDevice:: Fmultiply(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {}
void CudaDevice:: Fdivide(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {}
void CudaDevice:: Fmatmul(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {}

void CudaDevice:: Dadd(const LDG::Tensor& x, const LDG::Tensor& y, const LDG::Tensor& r,
		const LDG::Tensor& gr, LDG::Tensor& gx, LDG::Tensor& gy) {}
void CudaDevice:: Dsubtract(const LDG::Tensor& x, const LDG::Tensor& y, const LDG::Tensor& r,
		const LDG::Tensor& gr, LDG::Tensor& gx, LDG::Tensor& gy) {}
void CudaDevice:: Dmultiply(const LDG::Tensor& x, const LDG::Tensor& y, const LDG::Tensor& r,
		const LDG::Tensor& gr, LDG::Tensor& gx, LDG::Tensor& gy) {}
void CudaDevice:: Ddivide(const LDG::Tensor& x, const LDG::Tensor& y, const LDG::Tensor& r,
		const LDG::Tensor& gr, LDG::Tensor& gx, LDG::Tensor& gy) {}
void CudaDevice:: Dmatmul(const LDG::Tensor& x, const LDG::Tensor& y, const LDG::Tensor& r,
		const LDG::Tensor& gr, LDG::Tensor& gx, LDG::Tensor& gy) {}


void CudaDevice:: Fadd(const vector<LDG::PTensor>& vec_x, LDG::Tensor& r) {}
void CudaDevice:: Fsubstract(const vector<LDG::PTensor>& vec_x, LDG::Tensor& r) {}
void CudaDevice:: Fmultiply(const vector<LDG::PTensor>& vec_x, LDG::Tensor& r) {}
void CudaDevice:: Fdivide(const vector<LDG::PTensor>& vec_x, LDG::Tensor& r) {}
void CudaDevice:: Fmatmul(const vector<LDG::PTensor>& vec_x, LDG::Tensor& r) {}

void CudaDevice:: Dadd(const vector<LDG::PTensor>& vec_x, const LDG::Tensor& r,
		const LDG::Tensor& gr, vector<LDG::Tensor>& vec_gx) {}
void CudaDevice:: Dsubtract(const vector<LDG::PTensor>& vec_x, const LDG::Tensor& r,
		const LDG::Tensor& gr, vector<LDG::Tensor>& vec_gx) {}
void CudaDevice:: Dmultiply(const vector<LDG::PTensor>& vec_x, const LDG::Tensor& r,
		const LDG::Tensor& gr, vector<LDG::Tensor>& vec_gx) {}
void CudaDevice:: Ddivide(const vector<LDG::PTensor>& vec_x, const LDG::Tensor& r,
		const LDG::Tensor& gr, vector<LDG::Tensor>& vec_gx) {}
void CudaDevice:: Dmatmul(const vector<LDG::PTensor>& vec_x, const LDG::Tensor& r,
		const LDG::Tensor& gr, vector<LDG::Tensor>& vec_gx) {}

void CudaDevice:: Fsoftmax(const LDG::Tensor& x, LDG::Tensor& r) {}
void CudaDevice:: Dsoftmax(const LDG::Tensor& x, const LDG::Tensor& r, const LDG::Tensor& gr,
		LDG::Tensor& gx) {}

void CudaDevice:: concat(const vector<LDG::PTensor>& vec_x, const int& dim, LDG::Tensor& r) {}
void CudaDevice:: unconcat(const LDG::Tensor& x, const int& dim, vector<LDG::Tensor>& vec_r) {}
