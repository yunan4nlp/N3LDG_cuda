#ifndef LDG_TENSOR_CPU_CPU_IMPL_H
#define LDG_TENSOR_CPU_CPU_IMPL_H

#include "Device.h"

class CpuDevice : public Device {
public:
    void malloc(LDG::Tensor &t, const Shape &shape) {
        t.device_type = DeviceType::CPU;
        t.shape_ = shape;
        t.v = new dtype[sizeof(dtype) * shape.size()];
    }

    void set(LDG::Tensor &t, dtype val) {
        for (int i = 0; i < t.shape().size(); ++i) {
            t.v[i] = val;
        }
    }

    void set(LDG::Tensor &t, const vector<dtype>& vec_val) {
        int size = t.shape().size();
        assert(size == vec_val.size());
        for (int i = 0; i < size; ++i) {
            t.v[i] = vec_val.at(i);
        }
    }

    void set(LDG::Tensor &t, const dtype* host_data, int h_size) {
        abort();
    }

    void zero(LDG::Tensor &t) {
        for (int i = 0; i < t.shape().size(); ++i) {
            t.v[i] = 0.0f;
        }
    }

    void set_col(LDG::Tensor &t, int col, dtype val) {
        for (int i = 0; i < t.row(); ++i) {
            at(t, col, i) = val;
        }
    }

    void set_row(LDG::Tensor &t, int row, dtype val) {
        for (int i = 0; i < t.col(); ++i) {
            at(t, row, i) = val;
        }
    }
    void copy_data(const LDG::Tensor &src, LDG::Tensor& dst) {
        if(!src.shape().has_same_dims(dst.shape())
                || src.device_type != dst.device_type) {
            abort();
        }

        for (int i = 0; i < t.shape().size(); ++i) {
            dst.v[i] = src.v[i];
        }
    }
    void get_row(const LDG::Tensor& x, int row, LDG::Tensor& r) {}

    void random_uniform(LDG::Tensor &t, const Shape &shape, float lower, float upper) {}
    void random_bernoulli(LDG::Tensor &t, const Shape &shape, float p) {}
    void random_normal(LDG::Tensor &t, const Shape &shape, float mean, float sd) {}
    void random_log_normal(LDG::Tensor &t, const Shape &shape, float mean, float sd) {}

    void Fequal(const LDG::Tensor& x, LDG::Tensor& r) {}
    void Ftanh(const LDG::Tensor& x, LDG::Tensor& r) {}
    void Fsigmoid(const LDG::Tensor& x, LDG::Tensor& r) {}
    void Frelu(const LDG::Tensor& x, LDG::Tensor& r) {}
    void Fleaky_relu(const LDG::Tensor& x, LDG::Tensor& r) {}
    void Fexp(const LDG::Tensor& x, LDG::Tensor& r) {}
    void Flog(const LDG::Tensor& x, LDG::Tensor& r) {}

    void Fsquare(const LDG::Tensor& x, LDG::Tensor& r) {}
    void Fsqrt(const LDG::Tensor& x, LDG::Tensor& r) {}

    void Dequal(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {}
    void Dtanh(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {}
    void Dleaky_relu(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {}
    void Dsigmoid(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {}
    void Drelu(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {}
    void Dexp(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {}
    void Dlog(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {}

    void Fadd(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {}
    void Fsubtract(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {}
    void Fmultiply(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {}
    void Fdivide(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {}
    void Fmatmul(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r, bool ta=false, bool tb=false) {}


    void Fadd_scalar(const LDG::Tensor& x, const dtype y, LDG::Tensor& r) {}
    void Fmultiply_scalar(const LDG::Tensor& x, const dtype y, LDG::Tensor& r) {}

    void Dadd(const LDG::Tensor& x, const LDG::Tensor& y, const LDG::Tensor& r,
            const LDG::Tensor& gr, LDG::Tensor& gx, LDG::Tensor& gy) {}
    void Dsubtract(const LDG::Tensor& x, const LDG::Tensor& y, const LDG::Tensor& r,
            const LDG::Tensor& gr, LDG::Tensor& gx, LDG::Tensor& gy) {}
    void Dmultiply(const LDG::Tensor& x, const LDG::Tensor& y, const LDG::Tensor& r,
            const LDG::Tensor& gr, LDG::Tensor& gx, LDG::Tensor& gy) {}
    void Ddivide(const LDG::Tensor& x, const LDG::Tensor& y, const LDG::Tensor& r,
            const LDG::Tensor& gr, LDG::Tensor& gx, LDG::Tensor& gy) {}
    void Dmatmul(const LDG::Tensor& x, const LDG::Tensor& y, const LDG::Tensor& r,
            const LDG::Tensor& gr, LDG::Tensor& gx, LDG::Tensor& gy) {}


    void Fadd(const vector<LDG::PTensor>& vec_x, LDG::Tensor& r) {}
    void Fsubstract(const vector<LDG::PTensor>& vec_x, LDG::Tensor& r) {}
    void Fmultiply(const vector<LDG::PTensor>& vec_x, LDG::Tensor& r) {}
    void Fdivide(const vector<LDG::PTensor>& vec_x, LDG::Tensor& r) {}
    void Fmatmul(const vector<LDG::PTensor>& vec_x, LDG::Tensor& r) {}

    void Dadd(const vector<LDG::PTensor>& vec_x, const LDG::Tensor& r,
            const LDG::Tensor& gr, vector<LDG::Tensor>& vec_gx) {}
    void Dsubtract(const vector<LDG::PTensor>& vec_x, const LDG::Tensor& r,
            const LDG::Tensor& gr, vector<LDG::Tensor>& vec_gx) {}
    void Dmultiply(const vector<LDG::PTensor>& vec_x, const LDG::Tensor& r,
            const LDG::Tensor& gr, vector<LDG::Tensor>& vec_gx) {}
    void Ddivide(const vector<LDG::PTensor>& vec_x, const LDG::Tensor& r,
            const LDG::Tensor& gr, vector<LDG::Tensor>& vec_gx) {}
    void Dmatmul(const vector<LDG::PTensor>& vec_x, const LDG::Tensor& r,
            const LDG::Tensor& gr, vector<LDG::Tensor>& vec_gx) {}

    void Fsoftmax(const LDG::Tensor& x, LDG::Tensor& r) {}
    void Dsoftmax(const LDG::Tensor& x, const LDG::Tensor& r, const LDG::Tensor& gr,
            LDG::Tensor& gx) {}

    void concat(const vector<LDG::PTensor>& vec_x, LDG::Tensor& r) {}
    void unconcat(const LDG::Tensor& r, vector<LDG::PTensor>& vec_x) {}

    void Ftranspose(const LDG::Tensor& x, LDG::Tensor& r) {}

    void to_cpu(const LDG::Tensor &gpu_tensor, LDG::Tensor &cpu_tensor) {}
    void show_val(const LDG::Tensor &t) {}
    void unaryExp(const LDG::Tensor& x, LDG::Tensor& r, 
            Device *dev, void (Device::*f)(const LDG::Tensor&, LDG::Tensor& )) {
        (dev->*f)(x, r);
    }

    void binaryExp(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r, 
            Device *dev, void (Device::*f)(const LDG::Tensor&, const LDG::Tensor&, LDG::Tensor& )) {
        (dev->*f)(x, y, r);
    }

protected:
    dtype &at(const LDG::Tensor &x, int col_i) {
        assert(x.shape().dims().size() == 1 && col_i < x.col());
        return x.v[col_i];
    }

    dtype &at(const LDG::Tensor &x, int col_i, int row_i) {
        assert(x.shape().dims().size() == 2 && col_i < x.col() &&
                row_i < x.row());
        return x.v[x.row() * col_i + row_i];
    }

};

#endif
