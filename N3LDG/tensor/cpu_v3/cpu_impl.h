#ifndef LDG_TENSOR_CPU_CPU_IMPL_H
#define LDG_TENSOR_CPU_CPU_IMPL_H

#include "Device.h"
#include "Eigen/Dense"
#include <unsupported/Eigen/CXX11/Tensor>
#include <functional>

class CpuDevice : public Device {
public:
    CpuDevice() {
        Device::device_type = CPU;
    }

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
        to_vec(r) = Vec(x.v + col * x.row(), x.row());
    }

    void get_cols(const LDG::Tensor& x, int* cols, int col_num,
            LDG::Tensor& r) {
        assert(r.row() == x.row());
        for (int i = 0; i < col_num; ++i) {
            int col_i = cols[i];
            Vec(r.v + i * x.row(), x.row()) =
                Vec(x.v + col_i * x.row(), x.row());
        }
    }

    void FLookup(const LDG::Tensor& x, int* cols, int col_num,
            vector<LDG::PTensor>& vec_r) {
        for (int i = 0; i < col_num; ++i) {
            to_vec(*vec_r.at(i++)) = Vec(x.v + cols[i] * x.row());
        }
    }

    void FLookup(const LDG::Tensor& x, const vector<int>& vec_cols,
            vector<LDG::PTensor>& vec_r) {
        int i = 0;
        for (int col : vec_cols) {
            to_vec(*vec_r.at(i++)) = Vec(x.v + col * x.row());
        }
    }

    void DLookup(const LDG::Tensor& gx, int* cols, int col_num,
            vector<LDG::PTensor>& vec_loss) {
        for (int i = 0; i < col_num; ++i) {
            Vec(gx.v + cols[i] * gx.row()) += to_vec(*vec_loss.at(i++));
        }
    }

    void DLookup(const LDG::Tensor &gx, const vector<int>& vec_cols,
            vector<LDG::PTensor> &vec_loss) {
        int i = 0;
        for (int col : vec_cols) {
            Vec(gx.v + col * gx.row()) += to_vec(*vec_loss.at(i++));
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

    void Ftanh(const vector<LDG::PTensor>& vec_x, vector<LDG::PTensor>& vec_r) {
        for (int i = 0; i < vec_x.size(); ++i) {
            Ftanh(*vec_x.at(i), *vec_r.at(i));
        }
    }

    void Fsigmoid(const LDG::Tensor& x, LDG::Tensor& r) {
        assert(x.shape().size() == r.shape().size());
        to_vec(r) = to_vec(x).sigmoid();
    }

    void Fsigmoid(const vector<LDG::PTensor>& vec_x, vector<LDG::PTensor>& vec_r) {
        for (int i = 0; i < vec_x.size(); ++i) {
            Fsigmoid(*vec_x.at(i), *vec_r.at(i));
        }
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

    void Dtanh(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) override {
        assert(x.shape().size() == r.shape().size() &&
                y.shape().size() == r.shape().size());
        to_vec(r) = to_vec(y).unaryExpr([](dtype y) -> dtype {
                return (1 + y) * (1 - y); });
    }

    void Dtanh(const vector<LDG::PTensor>& vec_x,
            const vector<LDG::PTensor>& vec_y, vector<LDG::PTensor>& vec_r) {
        for (int i = 0; i < vec_x.size(); ++i) {
            Dtanh(*vec_x.at(i), *vec_y.at(i), *vec_r.at(i));
        }
    }

    void Dleaky_relu(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {}

    void Dsigmoid(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {
        to_vec(r) = to_vec(y).unaryExpr([](dtype y) -> dtype { 
                return y * (1 - y); });
    }

    void Dsigmoid(const vector<LDG::PTensor>& vec_x,
            const vector<LDG::PTensor>& vec_y, vector<LDG::PTensor>& vec_r) {
        for (int i = 0; i < vec_x.size(); ++i) {
            Dsigmoid(*vec_x.at(i), *vec_y.at(i), *vec_r.at(i));
        }
    }

    void Fadd(const LDG::Tensor& x, const vector<LDG::PTensor>& vec_y, LDG::Tensor& r) {
        for (int i = 0; i < vec_y.size(); ++i) {
            Vec(r.v + i * r.row(), r.row()) = Vec(x.v + i * r.row(), r.row()) +
                to_vec(*vec_y.at(i));
        }
    }

    void Drelu(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {}
    void Dexp(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {}
    void Dlog(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {}

    void Fadd(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {
        assert(x.shape().size() == r.shape().size() &&
                y.shape().size() == r.shape().size());
        to_vec(r) = to_vec(x) + to_vec(y);
    }

    virtual LDG::Tensor Fadd(const LDG::Tensor& x, const LDG::Tensor& y) {
        LDG::Tensor r;
        init(r, Shape({x.row(), x.col()}));
        Fadd(x, y, r);
        return r;
    }

    void Fsubtract(const LDG::Tensor& x, const LDG::Tensor& y,
            LDG::Tensor& r) {
        assert(x.shape().size() == r.shape().size() &&
                y.shape().size() == r.shape().size());
        to_vec(r) = to_vec(x) - to_vec(y);
    }

    void Fmultiply(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {
        assert(x.shape().size() == r.shape().size() &&
                y.shape().size() == r.shape().size());
        to_vec(r) = to_vec(x) * to_vec(y);
    }

    void Fmultiply(const vector<LDG::PTensor>& vec_x,
            const vector<LDG::PTensor>& vec_y, vector<LDG::PTensor>& vec_r) {
        for (int i = 0; i < vec_r.size(); ++i) {
            Fmultiply(*vec_x.at(i), *vec_y.at(i), *vec_r.at(i));
        }
    }

    void Fdivide(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {
        assert(x.shape().size() == r.shape().size() &&
                y.shape().size() == r.shape().size());
        to_vec(r) = to_vec(x) / to_vec(y);
    }

    void Fmatmul(const vector<LDG::PTensor> &vec_x,
            const vector<LDG::PTensor> &vec_y, LDG::Tensor &r, bool tx = false,
            bool ty = false) {
        int x_size = vec_x.size();
        int y_size = vec_y.size();
        int x_dim = vec_x[0]->shape()[0];
        int y_dim = vec_y[0]->shape()[0];
        LDG::Tensor x, y;
        init(x, Shape({x_dim, x_size}));
        init(y, Shape({y_dim, y_size}));
        concat(vec_x, x);
        concat(vec_y, y);

        Fmatmul(x, y, r, tx, ty);
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

    void Fmatmul(const vector<LDG::PTensor> &vec_x,
            const vector<LDG::PTensor> &vec_y, vector<LDG::PTensor> &vec_r,
            bool tx = false, bool ty = false) {
        for (int i = 0; i < vec_x.size(); ++i) {
            Fmatmul(*vec_x.at(i), *vec_y.at(i), *vec_r.at(i));
        }
    }

    void Fmatmul(const LDG::Tensor &x, const vector<LDG::PTensor> &vec_y,
            vector<LDG::PTensor> &vec_r, bool tx = false, bool ty = false) {
        assert (!tx && !ty);
        LDG::Tensor y;
        init(y, Shape({vec_y.at(0)->row(), vec_y.size()}));
        concat(vec_y, y);

        LDG::Tensor r;
        init(r, Shape({x.row(), vec_y.size()}));
        Fmatmul(x, y, r);

        for (int i = 0; i < vec_r.size(); ++i) {
            to_vec(*vec_r.at(i)) = Vec(r.v + i * r.row(), r.row());
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

    void Fadd(const vector<LDG::PTensor>& vec_x,
            const vector<LDG::PTensor>& vec_y, vector<LDG::PTensor>& vec_r) {
        for (int i = 0; i < vec_x.size(); ++i) {
            Fadd(*vec_x.at(i), *vec_y.at(i), *vec_r.at(i));
        }
    }

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

    void FSumPooling(const vector<LDG::PTensor>& vec_x, LDG::Tensor& y) {
        int n = vec_x.size();
        for (int i = 0; i < n; ++i) {
            to_vec(y) += Vec(vec_x.at(i)->v, vec_x.at(i)->row());
        }
    }

    void DSumPooling(const LDG::Tensor& gy, vector<LDG::PTensor>& vec_gx) {
        int n = vec_gx.size();
        for (int i = 0; i < n; ++i) {
            to_vec(*vec_gx.at(i)) += to_vec(gy);
        }
    }

    virtual void FAvgPooling(const vector<LDG::PTensor>& vec_x, LDG::Tensor &y) {
      int n = vec_x.size();
      for (int i = 0; i < n; ++i) {
        to_vec(y) += to_vec(*vec_x.at(i));
      }
      to_vec(y) = (1.0f / n) * to_vec(y);
    }

    void DAvgPooling(const LDG::Tensor& gy, vector<LDG::PTensor>& vec_gx) {
        int n = vec_gx.size();
        for (int i = 0; i < n; ++i) {
            to_vec(*vec_gx.at(i)) += 1.0f / n * to_vec(gy);
        }
    }

    void FMaxPooling(const vector<LDG::PTensor>& vec_x, LDG::Tensor& y,
            int *index) {
      int n = vec_x.size();
      for (int i = 0; i < vec_x.size(); ++i) {
          int max_i = 0;
          float max = vec_x.at(0)->v[i];
          for (int j = 1; j < n; ++j) {
              if (max < vec_x.at(j)->v[i]) {
                  max_i = j;
                  max = vec_x.at(j)->v[i];
              }
          }
          index[i] = max_i;
          y.v[i] = max;
      }
    }

    void FMinPooling(const vector<LDG::PTensor> &vec_x, LDG::Tensor& y,
            int *index) {
      int n = vec_x.size();
      for (int i = 0; i < vec_x.size(); ++i) {
          int min_i = 0;
          float min = vec_x.at(0)->v[i];
          for (int j = 1; j < n; ++j) {
              if (min > vec_x.at(j)->v[i]) {
                  min_i = j;
                  min = vec_x.at(j)->v[i];
              }
          }
          index[i] = min_i;
          y.v[i] = min;
      }
    }

    void DMaxPooling(const LDG::Tensor& gy, vector<LDG::PTensor>& vec_gx,
            int* index) {
        for (int i = 0; i < gy.row(); ++i) {
            vec_gx.at(index[i])->v[i] += gy.v[i];
        }
    }

    void DMinPooling(const LDG::Tensor& gy, vector<LDG::PTensor>& vec_gx,
            int* index) {
        for (int i = 0; i < gy.row(); ++i) {
            vec_gx.at(index[i])->v[i] += gy.v[i];
        }
    }

    void copy_tensor(const LDG::Tensor &src, LDG::Tensor& dst) {
        malloc(dst, src.shape());
        to_vec(dst) = to_vec(src);
    }

    void to_cpu(const LDG::Tensor &gpu_tensor, LDG::Tensor &cpu_tensor) {
        assert(gpu_tensor.shape().size() == cpu_tensor.shape().size());
        for (int i = 0; i < gpu_tensor.shape().size(); ++i) {
            cpu_tensor.v[i] = gpu_tensor.v[i];
        }
    }

    void to_gpu(const LDG::Tensor &cpu_tensor, LDG::Tensor& gpu_tensor) {
        assert(false);
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
