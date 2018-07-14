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
        to_vec(t).setConstant(val);
    }

    void set(LDG::Tensor &t, const vector<dtype>& vec_val) {
        int size = t.shape().size();
        assert(size == vec_val.size());
        to_vec(t) = Vec(const_cast<dtype*>(vec_val.data()), size);
    }

    void set(LDG::Tensor &t, const dtype* host_data, int h_size) {
        assert(t.shape().size() == h_size);
        to_vec(t) = Vec(const_cast<dtype*>(host_data), h_size);
    }

    void zero(LDG::Tensor &t) {
        to_vec(t).setZero();
    }

    void set_col(LDG::Tensor &t, int col, dtype val) {
        Vec(&at(t, col, 0), t.row()).setConstant(val);
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

        to_vec(dst) = to_vec(src);
    }

    void get_col(const LDG::Tensor& x, int col, LDG::Tensor& r) {
        assert(col < x.col() && x.row() == r.row());
        to_vec(r) = Vec(x.v, x.row());
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

    void Frelu(const LDG::Tensor& x, LDG::Tensor& r) {
        assert(x.shape().size() == r.shape().size());
        to_vec(r) = to_vec(x).unaryExpr([](dtype v) -> dtype {
            return v > 0.0f ? v : 0.0f;});
    }

    void Fleaky_relu(const LDG::Tensor& x, LDG::Tensor& r) {
        assert(x.shape().size() == r.shape().size());
        to_vec(r) = to_vec(x).unaryExpr([](dtype v) -> dtype {
            return v > 0.0f ? v : v * -0.1;});
    }

    void Fexp(const LDG::Tensor& x, LDG::Tensor& r) {
        assert(x.shape().size() == r.shape().size());
        to_vec(r) = to_vec(x).unaryExpr([](dtype v) -> dtype {
              return exp(v);});
    }

    void Flog(const LDG::Tensor& x, LDG::Tensor& r) {
        assert(x.shape().size() == r.shape().size());
        to_vec(r) = to_vec(x).unaryExpr([](dtype v) -> dtype {
            return log(v);});
    }

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

    void Dleaky_relu(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {

    }

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

    void Fadd_scalar(const LDG::Tensor& x, const dtype y, LDG::Tensor& r) {
        to_vec(r) = to_vec(x) + y;
    }

    void Fmultiply_scalar(const LDG::Tensor& x, const dtype y, LDG::Tensor& r) {
        to_vec(r) = y * to_vec(x);
    }

    void Fadd_col(LDG::Tensor& x, const LDG::Tensor& y_col, int col) {
        auto x_dims = x.shape().dims();
        auto y_dims = y_col.shape().dims();
        int size = x.shape().size();
        int dim0 = x_dims[0];
        int dim1 = x_dims[1];
        if(col >= dim1) {
                cout << "col index beyond x dim" << endl;	
                return;
        }

        if (y_dims[1] != 1) {
                cout << "y is not a vector" << endl;
                return;
        }

        Vec(x.v + col * dim0, dim0) += Vec(y_col.v, dim0);
    }

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

        int optLabel = -1;
        for (int i = 0; i < nDim; ++i) {
            if (optLabel < 0 || x.v[i] > x.v[optLabel])
                optLabel = i;
        }

        dtype maxScore = x.v[optLabel];
        to_vec(r) = (to_vec(x) - maxScore).exp();
        dtype sum2 = 0.0f;
        for (int i = 0; i < nDim; ++i) {
            sum2 += r.v[i];
        }
        to_vec(r) = to_vec(r) / sum2;
    }

    void Dsoftmax(const LDG::Tensor& x, const LDG::Tensor& r, const LDG::Tensor& gr,
            LDG::Tensor& gx) {}

    void concat(const vector<LDG::PTensor>& vec_x, LDG::Tensor& r) {
        int offset = 0;
        for (const LDG::PTensor &t : vec_x) {
            int size = t->shape().size();
            assert(offset + size <= r.shape().size());
            for (int i = 0; i < size; ++i) {
                r.v[offset + i] = t->v[i];
            }

            offset += size;
        }
    }

    void unconcat(const LDG::Tensor& r, vector<LDG::PTensor>& vec_x) {
        int offset = 0;
        for (LDG::PTensor &t : vec_x) {
            int size = t->shape().size();
            assert(offset + size <= r.shape().size());
            for (int i = 0; i < size; ++i) {
                t->v[i] = r.v[offset + i];
            }

            offset += size;
        }
    }

    void Ftranspose(const LDG::Tensor& x, LDG::Tensor& r) {
        assert(x.shape().size() == r.shape().size());
        r.shape_ = Shape({x.shape().dims()[1], x.shape().dims()[0]});
        to_mat(r) = to_mat(x).transpose();
    }

    void FSumPooling(const LDG::Tensor& x, LDG::Tensor& y) {
      int n = x.col();
      for (int i = 0; i < n; ++i) {
        to_vec(y) += Vec(&at(x, i, 0), x.row());
      }
    }

    virtual void FAvgPooling(const LDG::Tensor &x, LDG::Tensor &y) {
      int n = x.col();
      for (int i = 0; i < n; ++i) {
        to_vec(y) += Vec(&at(x, i, 0), x.row());
      }
      to_vec(y) = (1.0f / n) * to_vec(y);
    }

    void FMaxPooling(const LDG::Tensor& x, LDG::Tensor& y, int *index) {
      int n = x.col();
      for (int i = 0; i < x.row(); ++i) {
          int max_i = 0;
          float max = at(x, 0, i);
          for (int j = 1; j < n; ++j) {
              if (max < at(x, j, i)) {
                  max_i = j;
                  max = at(x, j, i);
              }
          }
          index[i] = max_i;
          y.v[i] = max;
      }
    }

    void FMinPooling(const LDG::Tensor& x, LDG::Tensor& y, int *index) {
      int n = x.col();
      for (int i = 0; i < x.row(); ++i) {
          int min_i = 0;
          float min = at(x, 0, i);
          for (int j = 1; j < n; ++j) {
              if (min > at(x, j, i)) {
                  min_i = j;
                  min = at(x, j, i);
              }
          }
          index[i] = min_i;
          y.v[i] = min;
      }
    }

    void DMaxPooling(const LDG::Tensor& x, const LDG::Tensor& y, 
            const LDG::Tensor& gy, LDG::Tensor& gx, int *index) {
        for (int i = 0; i < gy.row(); ++i) {
            at(gx, index[i], i) += gy.v[i];
        }
    }

    void DMinPooling(const LDG::Tensor& x, const LDG::Tensor& y, 
            const LDG::Tensor& gy, LDG::Tensor& gx, int *index) {
        for (int i = 0; i < gy.row(); ++i) {
            at(gx, index[i], i) += gy.v[i];
        }
    }

    void to_cpu(const LDG::Tensor &gpu_tensor, LDG::Tensor &cpu_tensor) {
        assert(gpu_tensor.shape().size() == cpu_tensor.shape().size());
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

    void init(LDG::Tensor &t, const Shape &shape) {
            malloc(t, shape);
            zero(t);
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

    dtype &at(const LDG::Tensor &x, int row_i) {
        assert(row_i < x.row());
        return x.v[row_i];
    }

    dtype &at(const LDG::Tensor &x, int col_i, int row_i) {
        assert(x.shape().dims().size() == 2 && col_i < x.col() &&
                row_i < x.row());
        return x.v[x.row() * col_i + row_i];
    }


private:
    typedef Eigen::TensorMap<Eigen::Tensor<dtype, 1>> Vec;
    typedef Eigen::Map<Matrix<dtype, Dynamic, Dynamic, ColMajor>> Mat;

    Vec to_vec(const LDG::Tensor &x) {
        return Vec(x.v, x.shape().size());
    }

    Mat to_mat(const LDG::Tensor &x) {
        assert(x.shape().dims().size() == 2);
        return Mat(x.v, x.row(), x.col());
    }
};

#endif
