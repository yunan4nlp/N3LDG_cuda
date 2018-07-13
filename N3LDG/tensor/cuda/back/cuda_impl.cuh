#ifndef CUDA_DEVICE
#define CUDA_DEVICE

#include "Device.h"
#include "MyLib.h"
#include <iostream>


class CudaDevice : public Device{
	public:
	 void random_uniform(LDG::Tensor &t, const Shape &shape, float lower, float upper); 

	 void random_bernoulli(LDG::Tensor &t, const Shape &shape, float p);
	 void random_normal(LDG::Tensor &t, const Shape &shape, float mean, float sd);
	 void random_log_normal(LDG::Tensor &t, const Shape &shape, float mean, float sd);

	 void Fequal(const LDG::Tensor& x, LDG::Tensor& r);
	 void Ftanh(const LDG::Tensor& x, LDG::Tensor& r);
	 void Fsigmoid(const LDG::Tensor& x, LDG::Tensor& r);
	 void Frelu(const LDG::Tensor& x, LDG::Tensor& r);
	 void Fleaky_relu(const LDG::Tensor& x, LDG::Tensor& r);
	 void Fexp(const LDG::Tensor& x, LDG::Tensor& r);
	 void Flog(const LDG::Tensor& x, LDG::Tensor& r);

	 void Dequal(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r);
	 void Dtanh(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r);
	 void Dleaky_relu(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r);
	 void Dsigmoid(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r);
	 void Drelu(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r);
	 void Dexp(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r);
	 void Dlog(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r);

	 void Fadd(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r);
	 void Fsubtract(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r);
	 void Fmultiply(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r);
	 void Fdivide(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r);
	 void Fmatmul(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r);

	 void Dadd(const LDG::Tensor& x, const LDG::Tensor& y, const LDG::Tensor& r,
		const LDG::Tensor& gr, LDG::Tensor& gx, LDG::Tensor& gy);
	 void Dsubtract(const LDG::Tensor& x, const LDG::Tensor& y, const LDG::Tensor& r,
		const LDG::Tensor& gr, LDG::Tensor& gx, LDG::Tensor& gy);
	 void Dmultiply(const LDG::Tensor& x, const LDG::Tensor& y, const LDG::Tensor& r,
		const LDG::Tensor& gr, LDG::Tensor& gx, LDG::Tensor& gy);
	 void Ddivide(const LDG::Tensor& x, const LDG::Tensor& y, const LDG::Tensor& r,
		const LDG::Tensor& gr, LDG::Tensor& gx, LDG::Tensor& gy);
	 void Dmatmul(const LDG::Tensor& x, const LDG::Tensor& y, const LDG::Tensor& r,
		const LDG::Tensor& gr, LDG::Tensor& gx, LDG::Tensor& gy);


	 void Fadd(const vector<LDG::PTensor>& vec_x, LDG::Tensor& r);
	 void Fsubstract(const vector<LDG::PTensor>& vec_x, LDG::Tensor& r);
	 void Fmultiply(const vector<LDG::PTensor>& vec_x, LDG::Tensor& r);
	 void Fdivide(const vector<LDG::PTensor>& vec_x, LDG::Tensor& r);
	 void Fmatmul(const vector<LDG::PTensor>& vec_x, LDG::Tensor& r);

	 void Dadd(const vector<LDG::PTensor>& vec_x, const LDG::Tensor& r,
		const LDG::Tensor& gr, vector<LDG::Tensor>& vec_gx);
	 void Dsubtract(const vector<LDG::PTensor>& vec_x, const LDG::Tensor& r,
		const LDG::Tensor& gr, vector<LDG::Tensor>& vec_gx);
	 void Dmultiply(const vector<LDG::PTensor>& vec_x, const LDG::Tensor& r,
		const LDG::Tensor& gr, vector<LDG::Tensor>& vec_gx);
	 void Ddivide(const vector<LDG::PTensor>& vec_x, const LDG::Tensor& r,
		const LDG::Tensor& gr, vector<LDG::Tensor>& vec_gx);
	 void Dmatmul(const vector<LDG::PTensor>& vec_x, const LDG::Tensor& r,
		const LDG::Tensor& gr, vector<LDG::Tensor>& vec_gx);

	 void Fsoftmax(const LDG::Tensor& x, LDG::Tensor& r);
	 void Dsoftmax(const LDG::Tensor& x, const LDG::Tensor& r, const LDG::Tensor& gr,
		LDG::Tensor& gx);

	 void concat(const vector<LDG::PTensor>& vec_x, const int& dim, LDG::Tensor& r);
	 void unconcat(const LDG::Tensor& x, const int& dim, vector<LDG::Tensor>& vec_r);
};

#endif
