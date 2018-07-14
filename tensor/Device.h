#ifndef LDG_DEVICE
#define LDG_DEVICE

#include "LDG_Tensor.h"
#include "MyLib.h"


class Device {
public:
	int device_type;
public:
	virtual void init(LDG::Tensor &t, const Shape &shape) = 0;
	virtual void malloc(LDG::Tensor &t, const Shape &shape) = 0;
	virtual void set(LDG::Tensor &t, dtype val) = 0;
	virtual void set(LDG::Tensor &t, const vector<dtype>& vec_val) = 0;
	virtual void set(LDG::Tensor &t, const dtype* host_data, int h_size) = 0;
	virtual void zero(LDG::Tensor &t) = 0;
	virtual void set_col(LDG::Tensor &t, int col, dtype val) = 0;
	virtual void copy_data(const LDG::Tensor &src, LDG::Tensor& dst) = 0;
	virtual void get_col(const LDG::Tensor& x, int col, LDG::Tensor& r) = 0;
        virtual dtype &at(const LDG::Tensor &x, int col_i, int row_i) {}

	virtual void random_uniform(LDG::Tensor &t, const Shape &shape, float lower, float upper) = 0;
	virtual void random_bernoulli(LDG::Tensor &t, const Shape &shape, float p) = 0;
	virtual void random_normal(LDG::Tensor &t, const Shape &shape, float mean, float sd) = 0;
	virtual void random_log_normal(LDG::Tensor &t, const Shape &shape, float mean, float sd) = 0;

	virtual void Fequal(const LDG::Tensor& x, LDG::Tensor& r) = 0;
	virtual void Ftanh(const LDG::Tensor& x, LDG::Tensor& r) = 0;
	virtual void Fsigmoid(const LDG::Tensor& x, LDG::Tensor& r) = 0;
	virtual void Frelu(const LDG::Tensor& x, LDG::Tensor& r) = 0;
	virtual void Fleaky_relu(const LDG::Tensor& x, LDG::Tensor& r) = 0;
	virtual void Fexp(const LDG::Tensor& x, LDG::Tensor& r) = 0;
	virtual void Flog(const LDG::Tensor& x, LDG::Tensor& r) = 0;
	
	virtual void Fsquare(const LDG::Tensor& x, LDG::Tensor& r) = 0;
	virtual void Fsqrt(const LDG::Tensor& x, LDG::Tensor& r) = 0;

	virtual void Dequal(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) = 0;
	virtual void Dtanh(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) = 0;
	virtual void Dleaky_relu(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) = 0;
	virtual void Dsigmoid(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) = 0;
	virtual void Drelu(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) = 0;
	virtual void Dexp(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) = 0;
	virtual void Dlog(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) = 0;

	virtual void Fadd(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) = 0;
	virtual void Fsubtract(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) = 0;
	virtual void Fmultiply(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) = 0;
	virtual void Fdivide(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) = 0;

	virtual void Fmatmul(const LDG::Tensor &x, const LDG::Tensor &y, LDG::Tensor &r, bool tx = false, bool ty = false) = 0;


	virtual void Fadd_scalar(const LDG::Tensor& x, const dtype y, LDG::Tensor& r) = 0;
	virtual void Fmultiply_scalar(const LDG::Tensor& x, const dtype y, LDG::Tensor& r) = 0;


	virtual void Fadd_col(LDG::Tensor& x, const LDG::Tensor& y_col, int col) = 0;

	virtual void Dadd(const LDG::Tensor& x, const LDG::Tensor& y, const LDG::Tensor& r,
		const LDG::Tensor& gr, LDG::Tensor& gx, LDG::Tensor& gy) = 0;
	virtual void Dsubtract(const LDG::Tensor& x, const LDG::Tensor& y, const LDG::Tensor& r,
		const LDG::Tensor& gr, LDG::Tensor& gx, LDG::Tensor& gy) = 0;
	virtual void Dmultiply(const LDG::Tensor& x, const LDG::Tensor& y, const LDG::Tensor& r,
		const LDG::Tensor& gr, LDG::Tensor& gx, LDG::Tensor& gy) = 0;
	virtual void Ddivide(const LDG::Tensor& x, const LDG::Tensor& y, const LDG::Tensor& r,
		const LDG::Tensor& gr, LDG::Tensor& gx, LDG::Tensor& gy) = 0;
	virtual void Dmatmul(const LDG::Tensor& x, const LDG::Tensor& y, const LDG::Tensor& r,
		const LDG::Tensor& gr, LDG::Tensor& gx, LDG::Tensor& gy) = 0;


	virtual void Fadd(const vector<LDG::PTensor>& vec_x, LDG::Tensor& r) = 0;
	virtual void Fsubstract(const vector<LDG::PTensor>& vec_x, LDG::Tensor& r) = 0;
	virtual void Fmultiply(const vector<LDG::PTensor>& vec_x, LDG::Tensor& r) = 0;
	virtual void Fdivide(const vector<LDG::PTensor>& vec_x, LDG::Tensor& r) = 0;
	virtual void Fmatmul(const vector<LDG::PTensor>& vec_x, LDG::Tensor& r) = 0;

	virtual void Dadd(const vector<LDG::PTensor>& vec_x, const LDG::Tensor& r,
		const LDG::Tensor& gr, vector<LDG::Tensor>& vec_gx) = 0;
	virtual void Dsubtract(const vector<LDG::PTensor>& vec_x, const LDG::Tensor& r,
		const LDG::Tensor& gr, vector<LDG::Tensor>& vec_gx) = 0;
	virtual void Dmultiply(const vector<LDG::PTensor>& vec_x, const LDG::Tensor& r,
		const LDG::Tensor& gr, vector<LDG::Tensor>& vec_gx) = 0;
	virtual void Ddivide(const vector<LDG::PTensor>& vec_x, const LDG::Tensor& r,
		const LDG::Tensor& gr, vector<LDG::Tensor>& vec_gx) = 0;
	virtual void Dmatmul(const vector<LDG::PTensor>& vec_x, const LDG::Tensor& r,
		const LDG::Tensor& gr, vector<LDG::Tensor>& vec_gx) = 0;

	virtual void Fsoftmax(const LDG::Tensor& x, LDG::Tensor& r) = 0;
	virtual void Dsoftmax(const LDG::Tensor& x, const LDG::Tensor& r, const LDG::Tensor& gr,
		LDG::Tensor& gx) = 0;

	virtual void concat(const vector<LDG::PTensor>& vec_x, LDG::Tensor& r) = 0;
	virtual void unconcat(const LDG::Tensor& r, vector<LDG::PTensor>& vec_x) = 0;

	virtual void Ftranspose(const LDG::Tensor& x, LDG::Tensor& r) = 0;

	virtual void FSumPooling(const LDG::Tensor &x, LDG::Tensor &y) = 0;

	virtual void FAvgPooling(const LDG::Tensor &x, LDG::Tensor &y) = 0;

	virtual void FMaxPooling(const LDG::Tensor &x, LDG::Tensor &y, int *index) = 0;

        virtual void FMinPooling(const LDG::Tensor& x, LDG::Tensor& y, int *index) = 0;

	virtual void DMaxPooling(const LDG::Tensor& x, const LDG::Tensor& y, const LDG::Tensor& gy, LDG::Tensor& gx, int *index) = 0;

	virtual void DMinPooling(const LDG::Tensor& x, const LDG::Tensor& y, const LDG::Tensor& gy, LDG::Tensor& gx, int *index) = 0;
	void unaryExp(const LDG::Tensor& x, LDG::Tensor& r, 
			Device *dev, void (Device::*f)(const LDG::Tensor&, LDG::Tensor& )) {
		(dev->*f)(x, r);
	}

	void binaryExp(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r, 
			Device *dev, void (Device::*f)(const LDG::Tensor&, const LDG::Tensor&, LDG::Tensor& )) {
		(dev->*f)(x, y, r);
	}
};

#endif // ! Device