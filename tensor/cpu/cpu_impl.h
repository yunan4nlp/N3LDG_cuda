#ifndef LDG_TENSOR_CPU_CPU_IMPL_H
#define LDG_TENSOR_CPU_CPU_IMPL_H

#include "Device.h"
#include "Eigen/Dense"
#include <unsupported/Eigen/CXX11/Tensor>

class CpuDevice : public Device {
public:
    void malloc(LDG::Tensor &t, const Shape &shape) {
        t.device_type = DeviceType::CPU;
        t.shape_ = shape;
        t.v = new dtype[shape.size()];
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
        assert(t.shape().size() == h_size);
        for (int i = 0; i < h_size; ++i) {
            t.v[i] = host_data[i];
        }
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
            at(t, i, row) = val;
        }
    }

    void copy_data(const LDG::Tensor &src, LDG::Tensor& dst) {
        if(!src.shape().has_same_dims(dst.shape())
                || src.device_type != dst.device_type) {
            abort();
        }

        for (int i = 0; i < src.shape().size(); ++i) {
            dst.v[i] = src.v[i];
        }
    }

    //git_col
    void get_row(const LDG::Tensor& x, int col, LDG::Tensor& r) {
        assert(col < x.col() && x.row() == r.row());
        for (int i = 0; i < r.row(); ++i) {
            at(r, i) = at(x, col, i);
        }
    }

    void random_uniform(LDG::Tensor &t, const Shape &shape, float lower,
            float upper) {
        t.device_type = DeviceType::CPU;
        t.shape_ = shape;
        t.v = new dtype[shape.size()];
        int size = shape.size();
        dtype min = lower, max = upper;
        for (int i = 0; i < size; i++) {
            t.v[i] = (dtype(rand()) / RAND_MAX) * (max - min) + min;
        }
    }

    void random_bernoulli(LDG::Tensor &t, const Shape &shape, float p) {}
    void random_normal(LDG::Tensor &t, const Shape &shape, float mean, float sd) {}
    void random_log_normal(LDG::Tensor &t, const Shape &shape, float mean, float sd) {}

    void Fequal(const LDG::Tensor& x, LDG::Tensor& r) {}
    void Ftanh(const LDG::Tensor& x, LDG::Tensor& r) {
        assert(x.shape().size() == r.shape().size());
        to_vec(r) = to_vec(x).tanh();
    }

    void Fsigmoid(const LDG::Tensor& x, LDG::Tensor& r) {
        assert(x.shape().size() == r.shape().size());
        to_vec(r) = to_vec(x).sigmoid();
    }

    void Frelu(const LDG::Tensor& x, LDG::Tensor& r) {}
    void Fleaky_relu(const LDG::Tensor& x, LDG::Tensor& r) {}
    void Fexp(const LDG::Tensor& x, LDG::Tensor& r) {}
    void Flog(const LDG::Tensor& x, LDG::Tensor& r) {}

    void Fsquare(const LDG::Tensor& x, LDG::Tensor& r) {
        assert(x.shape().size() == r.shape().size());
        to_vec(r) = to_vec(x).unaryExpr([](dtype x) -> dtype {
                return x * x; });
    }

    void Fsqrt(const LDG::Tensor& x, LDG::Tensor& r) {
        assert(x.shape().size() == r.shape().size());
        to_vec(r) = to_vec(x).unaryExpr([](dtype x) -> dtype {
                return sqrt(x); });
    }

    void Dequal(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {}

    void Dtanh(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {
        assert(x.shape().size() == r.shape().size() &&
                y.shape().size() == r.shape().size());
        to_vec(r) = to_vec(y).unaryExpr([](dtype y) -> dtype {
                return (1 + y) * (1 - y); });
    }

    void Dleaky_relu(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {}
    void Dsigmoid(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {}
    void Drelu(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {}
    void Dexp(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {}
    void Dlog(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {}

    void Fadd(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {
        assert(x.shape().size() == r.shape().size() &&
                y.shape().size() == r.shape().size());
        to_vec(r) = to_vec(x) + to_vec(y);
    }

    void Fsubtract(const LDG::Tensor& x, const LDG::Tensor& y,
            LDG::Tensor& r) {
        assert(x.shape().size() == r.shape().size() &&
                y.shape().size() == r.shape().size());
        to_vec(r) = to_vec(x) - to_vec(y);
    }

    void Fmultiply(const LDG::Tensor& x, const LDG::Tensor& y,
            LDG::Tensor& r) {
        assert(x.shape().size() == r.shape().size() &&
                y.shape().size() == r.shape().size());
        to_vec(r) = to_vec(x) * to_vec(y);
    }

    void Fdivide(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {
        assert(x.shape().size() == r.shape().size() &&
                y.shape().size() == r.shape().size());
        to_vec(r) = to_vec(x) / to_vec(y);
    }

    void Fmatmul(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r,
            bool ta=false, bool tb=false) {
        if (ta && tb) {
            to_mat(r) = to_mat(x).transpose() * to_mat(y).transpose();
        } else if (ta && !tb) {
            to_mat(r) = to_mat(x).transpose() * to_mat(y);
        } else if (!ta && tb) {
            to_mat(r) = to_mat(x) * to_mat(y).transpose();
        } else {
            to_mat(r) = to_mat(x) * to_mat(y);
        }
    }

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

    void Fsoftmax(const LDG::Tensor& x, LDG::Tensor& r) {
        int nDim = x.shape().dims()[0];
        int memsize = nDim * sizeof(dtype);

        int optLabel = -1;
        for (int i = 0; i < nDim; ++i) {
            if (optLabel < 0 || x.v[i] > x.v[optLabel])
                optLabel = i;
        }

        std::vector<dtype> scores;
        scores.resize(nDim);

        dtype sum2 = 0, maxScore = x.v[optLabel];
        for (int i = 0; i < nDim; ++i) {
            scores[i] = -1e10;
            scores[i] = exp(x.v[i] - maxScore);
            sum2 += scores[i];
        } 

        for (int i = 0; i < nDim; ++i) {
            r.v[i] = scores[i] / sum2;
        }
    }

    void Dsoftmax(const LDG::Tensor& x, const LDG::Tensor& r, const LDG::Tensor& gr,
            LDG::Tensor& gx) {}

    void concat(const vector<LDG::PTensor>& vec_x, LDG::Tensor& r) {
        int offset = 0;
        for (const LDG::PTensor &t : vec_x) {
            int size = t->shape().size();
            assert(offset + size <= t->shape().size());
            for (int i = 0; i < size; ++i) {
                r.v[offset + i] = t->v[i];
            }
        }
    }

    void unconcat(const LDG::Tensor& r, vector<LDG::PTensor>& vec_x) {
        int offset = 0;
        for (LDG::PTensor &t : vec_x) {
            int size = t->shape().size();
            assert(offset + size <= t->shape().size());
            for (int i = 0; i < size; ++i) {
                r.v[offset + i] = t->v[i];
                t->v[i] = r.v[offset + i];
            }
        }
    }

    void Ftranspose(const LDG::Tensor& x, LDG::Tensor& r) {
        assert(x.shape().size() == r.shape().size());
        r.shape_ = Shape({x.shape().dims()[1], x.shape().dims()[0]});
        to_mat(r) = to_mat(x).transpose();
    }

    void to_cpu(const LDG::Tensor &gpu_tensor, LDG::Tensor &cpu_tensor) {
        assert(gpu_tensor.shape().size() == cpu_tensor.shape().size());
        cpu_tensor.shape_ = gpu_tensor.shape_;
        for (int i = 0; i < gpu_tensor.shape().size(); ++i) {
            cpu_tensor.v[i] = gpu_tensor.v[i];
        }
    }

    void show_val(const LDG::Tensor &t) {
        for (int i = 0; i < t.shape().size(); i++) {
            std::cout << t.v[i] << " ,";
        }
        std::cout << endl;
    }

    void unaryExp(const LDG::Tensor& x, LDG::Tensor& r, 
            CpuDevice *dev, void (CpuDevice::*f)(const LDG::Tensor&, LDG::Tensor& )) {
        (dev->*f)(x, r);
    }

    void binaryExp(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r, 
            CpuDevice *dev, void (CpuDevice::*f)(const LDG::Tensor&,
                const LDG::Tensor&,
                LDG::Tensor& )) {
        (dev->*f)(x, y, r);
    }

private:
    typedef Eigen::TensorMap<Eigen::Tensor<dtype, 1>> Vec;
    typedef Eigen::Map<Matrix<dtype, Dynamic, Dynamic, ColMajor>> Mat;

    dtype &at(const LDG::Tensor &x, int row_i) {
        assert(row_i < x.row());
        return x.v[row_i];
    }

    dtype &at(const LDG::Tensor &x, int col_i, int row_i) {
        assert(x.shape().dims().size() == 2 && col_i < x.col() &&
                row_i < x.row());
        return x.v[x.row() * col_i + row_i];
    }

    Vec to_vec(const LDG::Tensor &x) {
        return Vec(x.v, x.shape().size());
    }

    Mat to_mat(const LDG::Tensor &x) {
        assert(x.shape().dims().size() == 2);
        return Mat(x.v, x.row(), x.col());
    }
};

#endif
