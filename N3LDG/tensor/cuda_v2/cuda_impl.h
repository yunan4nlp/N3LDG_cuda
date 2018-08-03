#ifndef CUDA_DEVICE
#define CUDA_DEVICE

#include "Device.h"
#include "MyLib.h"
#include "kernel.cuh"
#include <iostream>
#include <iomanip>
#include <chrono>


class CudaDevice : public Device {
	private:
		cublasHandle_t handle;

	public:

	public:
		CudaDevice() {
			Device::device_type = CUDA;
			cublasCreate(&handle);
		}


	public:
		void show_val(const LDG::Tensor &t) {
			int size = t.shape().size();
			dtype host_data[size];
			int memsize = sizeof(dtype) * size;
			cudaMemcpy(host_data, t.v, memsize, cudaMemcpyDeviceToHost);

			for (int i = 0; i < size; i++) {
				std::cout << setprecision(10) << host_data[i] << " ,";
			}
			std::cout << endl;
		}

		void init(LDG::Tensor &t, const Shape &shape) {
			malloc(t, shape);
			zero(t);
		}

		void copy_data(const LDG::Tensor &src, LDG::Tensor& dst) {
			if(src.shape().has_same_dims(dst.shape())
					&& src.device_type == dst.device_type) {
				int memsize = src.shape().size() * sizeof(dtype);
				cudaMemcpy(dst.v, src.v, memsize, cudaMemcpyDeviceToDevice);
			} else
				cout << "copy error"  << endl;
		}


		void set(LDG::Tensor &t, const dtype* host_data, int h_size) {
			int size = t.shape().size();
			if (size == h_size) {
				int memsize = sizeof(dtype) * size;
				cudaMemcpy(t.v, host_data, memsize, cudaMemcpyHostToDevice);
			} else
				cout << "set size not match" << endl;
		}

		void set_col(LDG::Tensor &t, int col, dtype val) {
			int dim0 = t.shape()[0];
			int dim1 = t.shape()[1];
			if(col < dim1) {
				set_col_impl(t.v, dim0, col, t.shape().size(), val);
			} else
				std::cout << "set col beyond dim1 " << endl;
		}

		void malloc(LDG::Tensor &t, const Shape &shape) {
			t.device_type = CUDA;
			t.shape_ = shape;
			int size = shape.size();
			int memsize = sizeof(dtype) * size;
			cudaMalloc((void **)&t.v, memsize);
		}

		void zero(LDG::Tensor &t) {
			set(t, 0);
		}

		void set(LDG::Tensor &t, dtype val) {
			int size = t.shape().size();
			dtype* zero_host = new dtype[size];
			for (int idx = 0; idx < size; idx++)
				zero_host[idx] = val;
			set(t, zero_host, size);
			delete []zero_host;
		}

		void set(LDG::Tensor &t, const vector<dtype>& vec_val) {
			int size = t.shape().size();
			if (vec_val.size() == size) {
				dtype array_val[size];
				for (int idx = 0; idx < size; idx++) {
					array_val[idx] = vec_val[idx];
				}
				set(t, array_val, size);	
			} else 
				cout << "set error dim is not match" << endl;
		}

		void get_col(const LDG::Tensor& x, int col, LDG::Tensor& r) {
			int dim0 = x.shape()[0];
			int dim1 = x.shape()[1];
			if(dim0 == r.shape()[0]) {
				if(col < dim1) {
					get_col_impl(x.v, r.v, dim0, col, x.shape().size());
				} else
					cout << "get col, col beyond" << endl;
			} else
				cout << "get col dims are not matched" << endl;
		}

		void get_cols(const LDG::Tensor& x, int* cols, int col_num, LDG::Tensor& r) {
			int *gpu_cols;
			int memsize = sizeof(int) * col_num;
			cudaMalloc((void **)&gpu_cols, memsize);
			cudaMemcpy(gpu_cols, cols, memsize, cudaMemcpyHostToDevice);

			int xdim0 = x.shape()[0];
			int xdim1 = x.shape()[1];
			int r_size = r.shape().size();

			if(xdim0 == r.shape()[0]) {
				get_cols_impl(x.v, r.v, xdim0, xdim1, r_size, gpu_cols, col_num);
			} else
				cout << "get col dims are not matched" << endl;
			cudaFree(gpu_cols);
		}

		void FLookup(const LDG::Tensor& x, const vector<int>& vec_cols, vector<LDG::PTensor>& vec_r) {
			int size = vec_cols.size();
			int cols[size];
			for(int idx = 0; idx < size; idx++) {
				cols[idx] = vec_cols[idx];
			}
			FLookup(x, cols, size, vec_r);
		}

		void DLookup(const LDG::Tensor& gx, const vector<int>& vec_cols, vector<LDG::PTensor>& vec_loss) {
			int size = vec_cols.size();
			int cols[size];
			for(int idx = 0; idx < size; idx++) {
				cols[idx] = vec_cols[idx];
			}
			DLookup(gx, cols, size, vec_loss);
		}

		void FLookup(const LDG::Tensor& x, int* cols, int col_num, vector<LDG::PTensor>& vec_r) {
			if(vec_r.size() != col_num)
				cout << "error vec_r size is not matched." << endl;

			int *gpu_cols;
			int memsize = sizeof(int) * col_num;
			cudaMalloc((void **)&gpu_cols, memsize);
			cudaMemcpy(gpu_cols, cols, memsize, cudaMemcpyHostToDevice);

			dtype** v_data_r;
			cudaMalloc((void **)&v_data_r, sizeof(dtype*) * col_num);
			for(int i = 0; i < col_num; i++)
				cudaMemcpy(v_data_r + i, &vec_r[i]->v, sizeof(dtype*) ,cudaMemcpyHostToDevice);

			int xdim0 = x.shape()[0];
			int xdim1 = x.shape()[1];
			int rdim0 = vec_r[0]->shape()[0];
			int r_size = rdim0 * col_num;
			
			if(xdim0 == rdim0) {
				FLookup_impl(x.v, v_data_r, xdim0, xdim1, r_size, gpu_cols, col_num);
			} else
				cout << "get col dims are not matched" << endl;
			cudaFree(gpu_cols);
			cudaFree(v_data_r);
		}


		void DLookup(const LDG::Tensor& gx, int* cols, int col_num, vector<LDG::PTensor>& vec_loss) {
			if(vec_loss.size() != col_num) {
				cout << "error vec_loss size is not matched." << endl;
			}

			int *gpu_cols;
			int memsize = sizeof(int) * col_num;
			cudaMalloc((void **)&gpu_cols, memsize);
			cudaMemcpy(gpu_cols, cols, memsize, cudaMemcpyHostToDevice);

			dtype** v_data_loss;
			cudaMalloc((void **)&v_data_loss, sizeof(dtype*) * col_num);
			for(int i = 0; i < col_num; i++)
				cudaMemcpy(v_data_loss + i, &vec_loss[i]->v, sizeof(dtype*) ,cudaMemcpyHostToDevice);

			int gxdim0 = gx.shape()[0];
			int gxdim1 = gx.shape()[1];
			int ldim0 = vec_loss[0]->shape()[0];
			int l_size = ldim0 * col_num;
			
			if(gxdim0 == ldim0) {
				DLookup_impl(gx.v, v_data_loss, gxdim0, gxdim1, l_size, gpu_cols, col_num);
			} else
				cout << "get col dims are not matched" << endl;

			cudaFree(gpu_cols);
			cudaFree(v_data_loss);
		}

		void random_uniform(LDG::Tensor &t, const Shape &shape, float lower, float upper) {
			t.shape_ = shape;
			int size = shape.size();
			dtype *host_data = new dtype[size];
			dtype min = lower, max = upper;
			for (int i = 0; i < size; i++) {
				host_data[i] = (dtype(rand()) / RAND_MAX) * (max - min) + min;
			}
			int memsize = sizeof(dtype) * size;
			t.device_type = CUDA;
			cudaMalloc((void **)&t.v, memsize);
			cudaMemcpy(t.v, host_data, memsize, cudaMemcpyHostToDevice);
			delete []host_data;
		} 

		void random_bernoulli(LDG::Tensor &t, const Shape &shape, float p){}

		void random_normal(LDG::Tensor &t, const Shape &shape, float mean, float sd){}

		void random_log_normal(LDG::Tensor &t, const Shape &shape, float mean, float sd){}


		void Fequal(const LDG::Tensor& x, LDG::Tensor& r){}

		void Ftanh(const LDG::Tensor& x, LDG::Tensor& r) {
			int x_size = x.shape().size();
			int r_size = r.shape().size();
			if(x_size == r_size)
				Ftanh_impl(x.v, r.v, x_size);
			else
				std::cout << "error, tanh dim is not match" << std::endl;
		}

		void Ftanh(const vector<LDG::PTensor>& vec_x, vector<LDG::PTensor>& vec_r) {
			const int n_x = vec_x.size();
			const int n_r = vec_r.size();
			if(n_x == n_r){
				dtype** v_data_x;
				cudaMalloc((void **)&v_data_x, sizeof(dtype*) * n_x);
				for(int i = 0; i < n_x; i++)
					cudaMemcpy(v_data_x + i, &vec_x[i]->v, sizeof(dtype*) ,cudaMemcpyHostToDevice);

				dtype** v_data_r;
				cudaMalloc((void **)&v_data_r, sizeof(dtype*) * n_r);
				for(int i = 0; i < n_r; i++)
					cudaMemcpy(v_data_r + i, &vec_r[i]->v, sizeof(dtype*) ,cudaMemcpyHostToDevice);
				int dim0 = vec_x[0]->shape()[0];
				int size = dim0 * n_x;

				Ftanh_impl(v_data_x, v_data_r, dim0, size);

				cudaFree(v_data_x);
				cudaFree(v_data_r);
			} else 
				std::cout << "error the number of Ftanh tensors is not matched "<<endl;
		}

		void Fsigmoid(const vector<LDG::PTensor>& vec_x, vector<LDG::PTensor>& vec_r) {
			const int n_x = vec_x.size();
			const int n_r = vec_r.size();
			if(n_x == n_r){
				dtype** v_data_x;
				cudaMalloc((void **)&v_data_x, sizeof(dtype*) * n_x);
				for(int i = 0; i < n_x; i++)
					cudaMemcpy(v_data_x + i, &vec_x[i]->v, sizeof(dtype*) ,cudaMemcpyHostToDevice);

				dtype** v_data_r;
				cudaMalloc((void **)&v_data_r, sizeof(dtype*) * n_r);
				for(int i = 0; i < n_r; i++)
					cudaMemcpy(v_data_r + i, &vec_r[i]->v, sizeof(dtype*) ,cudaMemcpyHostToDevice);
				int dim0 = vec_x[0]->shape()[0];
				int size = dim0 * n_x;

				Fsigmoid_impl(v_data_x, v_data_r, dim0, size);

				cudaFree(v_data_x);
				cudaFree(v_data_r);
			} else 
				std::cout << "error the number of Fsigmoid tensors is not matched "<<endl;
		}

		void Dsigmoid(const vector<LDG::PTensor>& vec_x, const vector<LDG::PTensor>& vec_y, vector<LDG::PTensor>& vec_r) {
			const int n_x = vec_x.size();
			const int n_y = vec_y.size();
			const int n_r = vec_r.size();
			if(n_x == n_r && n_y == n_r) {
				dtype** v_data_x;
				cudaMalloc((void **)&v_data_x, sizeof(dtype*) * n_x);
				for(int i = 0; i < n_x; i++)
					cudaMemcpy(v_data_x + i, &vec_x[i]->v, sizeof(dtype*) ,cudaMemcpyHostToDevice);

				dtype** v_data_y;
				cudaMalloc((void **)&v_data_y, sizeof(dtype*) * n_y);
				for(int i = 0; i < n_y; i++)
					cudaMemcpy(v_data_y + i, &vec_y[i]->v, sizeof(dtype*) ,cudaMemcpyHostToDevice);

				dtype** v_data_r;
				cudaMalloc((void **)&v_data_r, sizeof(dtype*) * n_r);
				for(int i = 0; i < n_r; i++)
					cudaMemcpy(v_data_r + i, &vec_r[i]->v, sizeof(dtype*) ,cudaMemcpyHostToDevice);

				int dim0 = vec_x[0]->shape()[0];
				int size = dim0 * n_x;

				Dsigmoid_impl(v_data_x, v_data_y, v_data_r, dim0, size);

				cudaFree(v_data_x);
				cudaFree(v_data_y);
				cudaFree(v_data_r);
			} else 
				std::cout << "error the number of Dsigmoid tensors is not matched "<<endl;
		}


		void Dtanh(const vector<LDG::PTensor>& vec_x, const vector<LDG::PTensor>& vec_y, vector<LDG::PTensor>& vec_r) {
			const int n_x = vec_x.size();
			const int n_y = vec_y.size();
			const int n_r = vec_r.size();
			if(n_x == n_r && n_y == n_r) {
				dtype** v_data_x;
				cudaMalloc((void **)&v_data_x, sizeof(dtype*) * n_x);
				for(int i = 0; i < n_x; i++)
					cudaMemcpy(v_data_x + i, &vec_x[i]->v, sizeof(dtype*) ,cudaMemcpyHostToDevice);

				dtype** v_data_y;
				cudaMalloc((void **)&v_data_y, sizeof(dtype*) * n_y);
				for(int i = 0; i < n_y; i++)
					cudaMemcpy(v_data_y + i, &vec_y[i]->v, sizeof(dtype*) ,cudaMemcpyHostToDevice);

				dtype** v_data_r;
				cudaMalloc((void **)&v_data_r, sizeof(dtype*) * n_r);
				for(int i = 0; i < n_r; i++)
					cudaMemcpy(v_data_r + i, &vec_r[i]->v, sizeof(dtype*) ,cudaMemcpyHostToDevice);

				int dim0 = vec_x[0]->shape()[0];
				int size = dim0 * n_x;

				Dtanh_impl(v_data_x, v_data_y, v_data_r, dim0, size);

				cudaFree(v_data_x);
				cudaFree(v_data_y);
				cudaFree(v_data_r);
			} else 
				std::cout << "error the number of Dtanh tensors is not matched "<<endl;
		}

		void Fsigmoid(const LDG::Tensor& x, LDG::Tensor& r){
			int x_size = x.shape().size();
			int r_size = r.shape().size();
			if(x_size == r_size)
				Fsigmoid_impl(x.v, r.v, x_size);
			else
				std::cout << "error, sigmoid dim is not match" << std::endl;
		}

		void Frelu(const LDG::Tensor& x, LDG::Tensor& r){}
		void Fleaky_relu(const LDG::Tensor& x, LDG::Tensor& r){}
		void Fexp(const LDG::Tensor& x, LDG::Tensor& r){}
		void Flog(const LDG::Tensor& x, LDG::Tensor& r){}


		void Fsquare(const LDG::Tensor& x, LDG::Tensor& r) {
			int x_size = x.shape().size();
			int r_size = r.shape().size();
			if(x_size == r_size)
				Fsquare_impl(x.v, r.v, x_size);
			else
				std::cout << "error, square dim is not match" << std::endl;
		}

		void Fsqrt(const LDG::Tensor& x, LDG::Tensor& r) {
			int x_size = x.shape().size();
			int r_size = r.shape().size();
			if(x_size == r_size)
				Fsqrt_impl(x.v, r.v, x_size);
			else
				std::cout << "error, sqrt dim is not match" << std::endl;
		}

		void Dequal(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r){}

		void Dtanh(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {
			int x_size = x.shape().size();
			int y_size = y.shape().size();
			int r_size = r.shape().size();
			if(x_size == y_size && x_size == r_size)
				Dtanh_impl(x.v, y.v, r.v, x_size);
			else
				std::cout << "error, dtanh dim is not match" << std::endl;
		}

		void Dleaky_relu(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r){}

		void Dsigmoid(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r){
			int x_size = x.shape().size();
			int y_size = y.shape().size();
			int r_size = r.shape().size();
			if(x_size == y_size && x_size == r_size)
				Dsigmoid_impl(x.v, y.v, r.v, x_size);
			else
				std::cout << "error, dsigmoid dim is not match" << std::endl;
		}

		void Drelu(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r){}
		void Dexp(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r){}
		void Dlog(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r){}

		virtual LDG::Tensor Fadd(const LDG::Tensor& x, const LDG::Tensor& y) {
			int x_size = x.shape().size();
			int y_size = y.shape().size();
			LDG::Tensor r;
			if(x_size == y_size) {
				init(r, x.shape());
				Fadd_impl(x.v, y.v, r.v, x_size);
			}
			else
				std::cout << "error, add dim is not match" << std::endl;
			return r;
		}

		void Fadd(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r){
			int x_size = x.shape().size();
			int y_size = y.shape().size();
			int r_size = r.shape().size();
			if(x_size == y_size && x_size == r_size)
				Fadd_impl(x.v, y.v, r.v, x_size);
			else
				std::cout << "error, add dim is not match" << std::endl;
		}

		virtual void Fadd(const LDG::Tensor& x, const vector<LDG::PTensor>& vec_y, LDG::Tensor& r) {
			int n = vec_y.size();
			int x_size = x.shape().size();
			int r_size = r.shape().size();
			assert(x_size == r_size);
			for(int idx = 0; idx < n; idx++){
				assert(x_size == vec_y[idx]->shape().size());
			}

			dtype** v_data_y;
			cudaMalloc((void **)&v_data_y, sizeof(dtype*) * n);
			for(int i = 0; i < n; i++)
				cudaMemcpy(v_data_y + i, &vec_y[i]->v, sizeof(dtype*) ,cudaMemcpyHostToDevice);
			Fadd_impl(x.v, v_data_y, r.v, n, x_size);
			cudaFree(v_data_y);
		}

		void Fadd(const vector<LDG::PTensor>& vec_x, const vector<LDG::PTensor>& vec_y, vector<LDG::PTensor>& vec_r) {
			int size_x = vec_x.size();
			int size_y = vec_y.size();
			int size_r = vec_r.size();

			if(size_x == size_y && size_x == size_y) {
				int n = size_x;
				for(int idx = 0; idx < n; idx++) {
					int xdim0 = vec_x[idx]->shape()[0];	
					int ydim0 = vec_y[idx]->shape()[0];	
					int rdim0 = vec_r[idx]->shape()[0];	
					assert(xdim0 == ydim0 && xdim0 == rdim0);
				}
				dtype** v_data_x;
				cudaMalloc((void **)&v_data_x, sizeof(dtype*) * n);
				for(int i = 0; i < n; i++)
					cudaMemcpy(v_data_x + i, &vec_x[i]->v, sizeof(dtype*) ,cudaMemcpyHostToDevice);

				dtype** v_data_y;
				cudaMalloc((void **)&v_data_y, sizeof(dtype*) * n);
				for(int i = 0; i < n; i++)
					cudaMemcpy(v_data_y + i, &vec_y[i]->v, sizeof(dtype*) ,cudaMemcpyHostToDevice);

				dtype** v_data_r;
				cudaMalloc((void **)&v_data_r, sizeof(dtype*) * n);
				for(int i = 0; i < n; i++)
					cudaMemcpy(v_data_r + i, &vec_r[i]->v, sizeof(dtype*) ,cudaMemcpyHostToDevice);

				int dim0 = vec_x[0]->shape()[0];
				Fadd_impl(v_data_x, v_data_y, v_data_r, dim0, dim0 * n);
				cudaFree(v_data_x);
				cudaFree(v_data_y);
				cudaFree(v_data_r);
			} else {
				cout << "Fadd size is not matched" << endl;
			}
		}

		void Fsubtract(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r){
			int x_size = x.shape().size();
			int y_size = y.shape().size();
			int r_size = r.shape().size();
			if(x_size == y_size && x_size == r_size)
				Fsubtract_impl(x.v, y.v, r.v, x_size);
			else
				std::cout << "error, subtract dim is not match" << std::endl;
		}

		void Fmultiply(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {
			int x_size = x.shape().size();
			int y_size = y.shape().size();
			int r_size = r.shape().size();
			if(x_size == y_size && x_size == r_size)
				Fmultiply_impl(x.v, y.v, r.v, x_size);
			else
				std::cout << "error, multiply dim is not match" << std::endl;

		}
		void Fadd_scalar(const LDG::Tensor& x, const dtype y, LDG::Tensor& r) {
			int x_size = x.shape().size();
			Fadd_scalar_impl(x.v, y, r.v, x_size);
		}


		void Fmultiply_scalar(const LDG::Tensor& x, const dtype y, LDG::Tensor& r) {
			int x_size = x.shape().size();
			Fmultiply_scalar_impl(x.v, y, r.v, x_size);
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
					
			Fadd_col_impl(x.v, y_col.v, col, dim0, size);
		}

		void Fdivide(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {
			int x_size = x.shape().size();
			int y_size = y.shape().size();
			int r_size = r.shape().size();
			if(x_size == y_size && x_size == r_size)
				Fdivide_impl(x.v, y.v, r.v, x_size);
			else
				std::cout << "error, divide dim is not match" << std::endl;
		}

		void Fmatmul(const vector<LDG::PTensor> &vec_x, const vector<LDG::PTensor> &vec_y, LDG::Tensor &r,
				bool tx = false, bool ty = false) {
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

		void Fmatmul(const LDG::Tensor &x, const LDG::Tensor &y, LDG::Tensor &r,
				bool tx = false, bool ty = false) {
			int m = tx ? x.shape()[1] : x.shape()[0];
			int n = tx ?  x.shape()[0] : x.shape()[1];
			int k = ty ? y.shape()[0] : y.shape()[1];

			dtype alpha = 1;
			dtype beta =  0;

			cublasOperation_t transx = tx ? CUBLAS_OP_T : CUBLAS_OP_N;
			int ldx = tx ? n : m;

			cublasOperation_t transy = ty ? CUBLAS_OP_T : CUBLAS_OP_N;
			int ldy = ty ? k : n;

#if USE_FLOAT
			(cublasSgemm(handle, transx, transy, m, k, n,
						 &alpha, x.v, ldx, y.v, ldy, &beta, r.v, m));
#else                   
			(cublasDgemm(handle, transx, transy, m, k, n,
						 &alpha, x.v, ldx, y.v, ldy, &beta, r.v, m));
#endif                  
		}

		void Fmatmul(const vector<LDG::PTensor> &vec_x, const vector<LDG::PTensor> &vec_y, 
				vector<LDG::PTensor> &vec_r,
				bool tx = false, bool ty = false) {
			int size_x = vec_x.size();
			int size_y = vec_y.size();
			int size_r = vec_r.size();
			assert(size_x == size_y && size_x == size_r);
			for (int idx = 0; idx < size_x; idx++) {
				LDG::PTensor x = vec_x[idx];
				LDG::PTensor y = vec_y[idx];
				LDG::PTensor r = vec_r[idx];

				int m = tx ? x->shape()[1] : x->shape()[0];
				int n = tx ?  x->shape()[0] : x->shape()[1];
				int k = ty ? y->shape()[0] : y->shape()[1];

				dtype alpha = 1;
				dtype beta =  0;

				cublasOperation_t transx = tx ? CUBLAS_OP_T : CUBLAS_OP_N;
				int ldx = tx ? n : m;

				cublasOperation_t transy = ty ? CUBLAS_OP_T : CUBLAS_OP_N;
				int ldy = ty ? k : n;

#if USE_FLOAT
				(cublasSgemm(handle, transx, transy, m, k, n,
							 &alpha, x->v, ldx, y->v, ldy, &beta, r->v, m));
#else                   
				(cublasDgemm(handle, transx, transy, m, k, n,
							 &alpha, x->v, ldx, y->v, ldy, &beta, r->v, m));
#endif                  
			}
		}

		void Fmatmul(const LDG::Tensor &x, const vector<LDG::PTensor> &vec_y, vector<LDG::PTensor> &vec_r, bool tx = false, bool ty = false) {
			int size_y = vec_y.size();
			int size_r = vec_r.size();
			assert(size_y == size_r);
			for (int idx = 0; idx < size_y; idx++) {
				LDG::PTensor y = vec_y[idx];
				LDG::PTensor r = vec_r[idx];

				int m = tx ? x.shape()[1] : x.shape()[0];
				int n = tx ?  x.shape()[0] : x.shape()[1];
				int k = ty ? y->shape()[0] : y->shape()[1];

				dtype alpha = 1;
				dtype beta =  0;

				cublasOperation_t transx = tx ? CUBLAS_OP_T : CUBLAS_OP_N;
				int ldx = tx ? n : m;

				cublasOperation_t transy = ty ? CUBLAS_OP_T : CUBLAS_OP_N;
				int ldy = ty ? k : n;

#if USE_FLOAT
				(cublasSgemm(handle, transx, transy, m, k, n,
							 &alpha, x.v, ldx, y->v, ldy, &beta, r->v, m));
#else                   
				(cublasDgemm(handle, transx, transy, m, k, n,
							 &alpha, x.v, ldx, y->v, ldy, &beta, r->v, m));
#endif                  
			}

		}

		void Fmultiply(const vector<LDG::PTensor>& vec_x, const vector<LDG::PTensor>& vec_y, vector<LDG::PTensor>& vec_r) {
			int size_x = vec_x.size();
			int size_y = vec_y.size();
			int size_r = vec_r.size();

			if(size_x == size_y && size_x == size_y) {
				int n = size_x;
				for(int idx = 0; idx < n; idx++) {
					int xdim0 = vec_x[idx]->shape()[0];	
					int ydim0 = vec_y[idx]->shape()[0];	
					int rdim0 = vec_r[idx]->shape()[0];	
					assert(xdim0 == ydim0 && xdim0 == rdim0);
				}
				int dim0 = vec_x[0]->shape()[0];
				dtype** v_data_x;
				cudaMalloc((void **)&v_data_x, sizeof(dtype*) * n);
				for(int i = 0; i < n; i++)
					cudaMemcpy(v_data_x + i, &vec_x[i]->v, sizeof(dtype*) ,cudaMemcpyHostToDevice);

				dtype** v_data_y;
				cudaMalloc((void **)&v_data_y, sizeof(dtype*) * n);
				for(int i = 0; i < n; i++)
					cudaMemcpy(v_data_y + i, &vec_y[i]->v, sizeof(dtype*) ,cudaMemcpyHostToDevice);

				dtype** v_data_r;
				cudaMalloc((void **)&v_data_r, sizeof(dtype*) * n);
				for(int i = 0; i < n; i++)
					cudaMemcpy(v_data_r + i, &vec_r[i]->v, sizeof(dtype*) ,cudaMemcpyHostToDevice);

				Fmultiply_impl(v_data_x, v_data_y, v_data_r, dim0, dim0 * n);

				cudaFree(v_data_x);
				cudaFree(v_data_y);
				cudaFree(v_data_r);
			} else{
				cout << "Fmultiply size is not matched" << endl;
			}
		}

		void Fsoftmax(const LDG::Tensor& x, LDG::Tensor& r) {
			int nDim = x.shape()[0];
			int memsize = nDim * sizeof(dtype);
			dtype x_host_data[nDim], r_host_data[nDim], scores[nDim];
			cudaMemcpy(x_host_data, x.v, memsize, cudaMemcpyDeviceToHost);

			int optLabel = -1;
			for (int i = 0; i < nDim; ++i) {
				if (optLabel < 0 || x_host_data[i] > x_host_data[optLabel])
					optLabel = i;
			}

			dtype sum2 = 0, maxScore = x_host_data[optLabel];
			for (int i = 0; i < nDim; ++i) {
				scores[i] = -1e10;
				scores[i] = exp(x_host_data[i] - maxScore);
				sum2 += scores[i];
			} 

			for (int i = 0; i < nDim; ++i) {
				r_host_data[i] = scores[i] / sum2;
			}
			cudaMemcpy(r.v, r_host_data,  memsize, cudaMemcpyHostToDevice);
		}

		void Dsoftmax(const LDG::Tensor& x, const LDG::Tensor& r, const LDG::Tensor& gr,
				LDG::Tensor& gx){
		}

		void concat(const vector<LDG::PTensor>& vec_x, LDG::Tensor& r){
			int max_size = vec_x.size();	
			int offset = 0;
			for(int idx = 0; idx < max_size; idx++) {
				int dim_0 = vec_x[idx]->shape()[0];
				concat_impl(vec_x[idx]->v, r.v, offset, dim_0);
				offset += dim_0;
			}
		}

		void unconcat(const LDG::Tensor& r, vector<LDG::PTensor>& vec_x) {
			int max_size = vec_x.size();
			int offset = 0;
			for(int idx = 0; idx < max_size; idx++) {
				int dim_0 = vec_x[idx]->shape()[0];
				unconcat_impl(r.v, vec_x[idx]->v, offset, dim_0);
				offset += dim_0;
			}
				
		}

		void to_cpu(const LDG::Tensor &gpu_tensor, LDG::Tensor &cpu_tensor) {
			if (gpu_tensor.device_type == CUDA && cpu_tensor.device_type == CPU) {
				if(gpu_tensor.shape().has_same_dims(cpu_tensor.shape())) {
					int memsize = gpu_tensor.shape().size() * sizeof(dtype);
					cudaMemcpy(cpu_tensor.v, gpu_tensor.v, memsize, cudaMemcpyDeviceToHost);		
				} else {
					cout << "gpu: " << gpu_tensor.shape().to_string() << " ";
					cout << "cpu: " << cpu_tensor.shape().to_string() << endl;
					cout << "to_cpu dims are not match." << endl;
				}
			} else {
				cout << "to_cpu tensor type is error" << endl;
			}
		}

		void to_gpu(const LDG::Tensor &cpu_tensor, LDG::Tensor& gpu_tensor) {
			if (gpu_tensor.device_type == CUDA && cpu_tensor.device_type == CPU) {
				if(gpu_tensor.shape().has_same_dims(cpu_tensor.shape())) {
					int memsize = cpu_tensor.shape().size() * sizeof(dtype);
					cudaMemcpy(gpu_tensor.v, cpu_tensor.v, memsize, cudaMemcpyHostToDevice);		
				} else {
					cout << "to_gpu dims are not match." << endl;
				}
			} else {
				cout << "to_cpu tensor type is error" << endl;
			}
		}

		void Ftranspose(const LDG::Tensor& x, LDG::Tensor& r) {
			int x_size = x.shape().size();
			int r_size = r.shape().size();
			int dim0 = x.shape()[0];
			int dim1 = x.shape()[1];
			r.shape_ = Shape({dim1, dim0});
			if(x_size == r_size)
				Ftranspose_impl(x.v, r.v, dim0, dim1, x_size);
			else
				std::cout << "error, transpose dim is not match" << std::endl;
		}

		void FAvgPooling(const vector<LDG::PTensor>& vec_x, LDG::Tensor& y) {
			const int n = vec_x.size();
			const int r = y.shape().size();
			const int s = y.shape().lower_volume(1);
			dtype** v_data_x;
			cudaMalloc((void **)&v_data_x, sizeof(dtype*) * n);
			for(int i = 0; i < n; i++)
				cudaMemcpy(v_data_x + i, &vec_x[i]->v, sizeof(dtype*) ,cudaMemcpyHostToDevice);
			Favgpooling_impl(v_data_x, y.v, n, r, s);
			cudaFree(v_data_x);
		}

		void DAvgPooling(const LDG::Tensor& gy, vector<LDG::PTensor>& vec_gx) {
			const int n = vec_gx.size();
			dtype** v_data_gx;
			cudaMalloc((void **)&v_data_gx, sizeof(dtype*) * n);
			for(int i = 0; i < n; i++)
				cudaMemcpy(v_data_gx + i, &vec_gx[i]->v, sizeof(dtype*) ,cudaMemcpyHostToDevice);
			const int gx_size = n * vec_gx[0]->shape()[0];
			const int gy_size = gy.shape().size();
			Davgpooling_impl(gy.v, gy_size, gx_size, n, v_data_gx);

			cudaFree(v_data_gx);
		}

		void FSumPooling(const vector<LDG::PTensor>& vec_x, LDG::Tensor& y) {
			const int n = vec_x.size();
			const int r = y.shape().size();
			const int s = y.shape().lower_volume(1);
			dtype** v_data_x;
			cudaMalloc((void **)&v_data_x, sizeof(dtype*) * n);
			for(int i = 0; i < n; i++)
				cudaMemcpy(v_data_x + i, &vec_x[i]->v, sizeof(dtype*) ,cudaMemcpyHostToDevice);
			Fsumpooling_impl(v_data_x, y.v, n, r, s);
			cudaFree(v_data_x);
		}

		void DSumPooling(const LDG::Tensor& gy, vector<LDG::PTensor>& vec_gx) {
			const int n = vec_gx.size();
			dtype** v_data_gx;
			cudaMalloc((void **)&v_data_gx, sizeof(dtype*) * n);
			for(int i = 0; i < n; i++)
				cudaMemcpy(v_data_gx + i, &vec_gx[i]->v, sizeof(dtype*) ,cudaMemcpyHostToDevice);
			const int gx_size = n * vec_gx[0]->shape()[0];
			const int gy_size = gy.shape().size();
			Dsumpooling_impl(gy.v, gy_size, gx_size, v_data_gx);

			cudaFree(v_data_gx);
		}
		
		void FMaxPooling(const vector<LDG::PTensor>& vec_x, LDG::Tensor& y, int* index) {
			const int n = vec_x.size();
			const int r = y.shape().size();
			const int s = y.shape().lower_volume(1);
			dtype** v_data_x;
			cudaMalloc((void **)&v_data_x, sizeof(dtype*) * n);
			for(int i = 0; i < n; i++)
				cudaMemcpy(v_data_x + i, &vec_x[i]->v, sizeof(dtype*) ,cudaMemcpyHostToDevice);

			int *gpu_index_data;
			cudaMalloc((void **)&gpu_index_data, sizeof(int) * r);
			Fmaxpooling_impl(v_data_x, y.v, n, r, s, gpu_index_data);
			cudaMemcpy(index, gpu_index_data, sizeof(int) * r, cudaMemcpyDeviceToHost);

			cudaFree(v_data_x);
			cudaFree(gpu_index_data);
		}

		void DMaxPooling(const LDG::Tensor& gy, vector<LDG::PTensor>& vec_gx, int* index) {
			const int n = vec_gx.size();
			dtype** v_data_gx;
			cudaMalloc((void **)&v_data_gx, sizeof(dtype*) * n);
			for(int i = 0; i < n; i++)
				cudaMemcpy(v_data_gx + i, &vec_gx[i]->v, sizeof(dtype*) ,cudaMemcpyHostToDevice);

			const int dim0 = gy.shape()[0];
			int *gpu_index_data;
			cudaMalloc((void **)&gpu_index_data, sizeof(int) * dim0);
			cudaMemcpy(gpu_index_data, index, sizeof(int) * dim0, cudaMemcpyHostToDevice);
			Dmaxpooling_impl(gy.v, v_data_gx, gpu_index_data, dim0);

			cudaFree(v_data_gx);
			cudaFree(gpu_index_data);
		}

		void FMinPooling(const vector<LDG::PTensor>& vec_x, LDG::Tensor &y, int* index) {
			const int n = vec_x.size();
			const int r = y.shape().size();
			const int s = y.shape().lower_volume(1);
			dtype** v_data_x;
			cudaMalloc((void **)&v_data_x, sizeof(dtype*) * n);
			for(int i = 0; i < n; i++)
				cudaMemcpy(v_data_x + i, &vec_x[i]->v, sizeof(dtype*) ,cudaMemcpyHostToDevice);

			int *gpu_index_data;
			cudaMalloc((void **)&gpu_index_data, sizeof(int) * r);
			Fminpooling_impl(v_data_x, y.v, n, r, s, gpu_index_data);
			cudaMemcpy(index, gpu_index_data, sizeof(int) * r, cudaMemcpyDeviceToHost);

			cudaFree(v_data_x);
			cudaFree(gpu_index_data);
		}

		void DMinPooling(const LDG::Tensor& gy, vector<LDG::PTensor>& vec_gx, int* index) {
			const int n = vec_gx.size();
			dtype** v_data_gx;
			cudaMalloc((void **)&v_data_gx, sizeof(dtype*) * n);
			for(int i = 0; i < n; i++)
				cudaMemcpy(v_data_gx + i, &vec_gx[i]->v, sizeof(dtype*) ,cudaMemcpyHostToDevice);

			const int dim0 = gy.shape()[0];
			int *gpu_index_data;
			cudaMalloc((void **)&gpu_index_data, sizeof(int) * dim0);
			cudaMemcpy(gpu_index_data, index, sizeof(int) * dim0, cudaMemcpyHostToDevice);
			Dminpooling_impl(gy.v, v_data_gx, gpu_index_data, dim0);

			cudaFree(v_data_gx);
			cudaFree(gpu_index_data);
		}
};

#endif
