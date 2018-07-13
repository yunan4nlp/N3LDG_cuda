#ifndef CUDA_DEVICE
#define CUDA_DEVICE

#include "Device.h"
#include "MyLib.h"
#include "kernel.cuh"
#include <iostream>
#include <iomanip>


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
			int dim0 = t.shape().dims()[0];
			int dim1 = t.shape().dims()[1];
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
			int dim0 = x.shape().dims()[0];
			int dim1 = x.shape().dims()[1];
			if(dim0 == r.shape().dims()[0]) {
				if(col < dim1) {
					get_col_impl(x.v, r.v, dim0, col, x.shape().size());
				} else
					cout << "get col, col beyond" << endl;
			} else
				cout << "get col dims are not matched" << endl;
		}


		void random_uniform(LDG::Tensor &t, const Shape &shape, float lower, float upper) {
			t.device_type = CUDA;
			t.shape_ = shape;
			int size = shape.size();
			dtype *host_data = new dtype[size];
			dtype min = lower, max = upper;
			for (int i = 0; i < size; i++) {
				host_data[i] = (dtype(rand()) / RAND_MAX) * (max - min) + min;
			}
			int memsize = sizeof(dtype) * size;
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

		void Fadd(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r){
			int x_size = x.shape().size();
			int y_size = y.shape().size();
			int r_size = r.shape().size();
			if(x_size == y_size && x_size == r_size)
				Fadd_impl(x.v, y.v, r.v, x_size);
			else
				std::cout << "error, add dim is not match" << std::endl;
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

		void Fmatmul(const LDG::Tensor &x, const LDG::Tensor &y, LDG::Tensor &r,
				bool tx = false, bool ty = false) {
			int m = tx ? x.shape().dims()[1] : x.shape().dims()[0];
			int n = tx ?  x.shape().dims()[0] : x.shape().dims()[1];
			int k = ty ? y.shape().dims()[0] : y.shape().dims()[1];

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


		void Dadd(const LDG::Tensor& x, const LDG::Tensor& y, const LDG::Tensor& r,
				const LDG::Tensor& gr, LDG::Tensor& gx, LDG::Tensor& gy) {

		}

		void Dsubtract(const LDG::Tensor& x, const LDG::Tensor& y, const LDG::Tensor& r,
				const LDG::Tensor& gr, LDG::Tensor& gx, LDG::Tensor& gy){}
		void Dmultiply(const LDG::Tensor& x, const LDG::Tensor& y, const LDG::Tensor& r,
				const LDG::Tensor& gr, LDG::Tensor& gx, LDG::Tensor& gy){}
		void Ddivide(const LDG::Tensor& x, const LDG::Tensor& y, const LDG::Tensor& r,
				const LDG::Tensor& gr, LDG::Tensor& gx, LDG::Tensor& gy){}
		void Dmatmul(const LDG::Tensor& x, const LDG::Tensor& y, const LDG::Tensor& r,
				const LDG::Tensor& gr, LDG::Tensor& gx, LDG::Tensor& gy){}


		void Fadd(const vector<LDG::PTensor>& vec_x, LDG::Tensor& r){}
		void Fsubstract(const vector<LDG::PTensor>& vec_x, LDG::Tensor& r){}
		void Fmultiply(const vector<LDG::PTensor>& vec_x, LDG::Tensor& r){}
		void Fdivide(const vector<LDG::PTensor>& vec_x, LDG::Tensor& r){}
		void Fmatmul(const vector<LDG::PTensor>& vec_x, LDG::Tensor& r){}

		void Dadd(const vector<LDG::PTensor>& vec_x, const LDG::Tensor& r,
				const LDG::Tensor& gr, vector<LDG::Tensor>& vec_gx){}
		void Dsubtract(const vector<LDG::PTensor>& vec_x, const LDG::Tensor& r,
				const LDG::Tensor& gr, vector<LDG::Tensor>& vec_gx){}
		void Dmultiply(const vector<LDG::PTensor>& vec_x, const LDG::Tensor& r,
				const LDG::Tensor& gr, vector<LDG::Tensor>& vec_gx){}
		void Ddivide(const vector<LDG::PTensor>& vec_x, const LDG::Tensor& r,
				const LDG::Tensor& gr, vector<LDG::Tensor>& vec_gx){}
		void Dmatmul(const vector<LDG::PTensor>& vec_x, const LDG::Tensor& r,
				const LDG::Tensor& gr, vector<LDG::Tensor>& vec_gx){}

		void Fsoftmax(const LDG::Tensor& x, LDG::Tensor& r) {
			int nDim = x.shape().dims()[0];
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
				int dim_0 = vec_x[idx]->shape().dims()[0];
				concat_impl(vec_x[idx]->v, r.v, offset, dim_0);
				offset += dim_0;
			}
		}

		void unconcat(const LDG::Tensor& r, vector<LDG::PTensor>& vec_x) {
			int max_size = vec_x.size();
			int offset = 0;
			for(int idx = 0; idx < max_size; idx++) {
				int dim_0 = vec_x[idx]->shape().dims()[0];
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

		void to_gpu(const LDG::Tensor &cpu_tensor, LDG::Tensor &gpu_tensor) {
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
			int dim0 = x.shape().dims()[0];
			int dim1 = x.shape().dims()[1];
			r.shape_ = Shape({dim1, dim0});
			if(x_size == r_size)
				Ftranspose_impl(x.v, r.v, dim0, dim1, x_size);
			else
				std::cout << "error, transpose dim is not match" << std::endl;
		}

		void FSumPooling(const LDG::Tensor& x, LDG::Tensor& y) {
			const int n = x.shape()[1];
			const int r = y.shape().size();
			const int s = y.shape().lower_volume(1);
			Fsumpooling_impl(x.v, y.v, n, r, s);
		}

		void FAvgPooling(const LDG::Tensor& x, LDG::Tensor& y) {
			const int n = x.shape()[1];
			const int r = y.shape().size();
			const int s = y.shape().lower_volume(1);
			Favgpooling_impl(x.v, y.v, n, r, s);
		}
		
		void FMaxPooling(const LDG::Tensor& x, LDG::Tensor& y, int* index) {
			const int n = x.shape()[1];
			const int r = y.shape().size();
			const int s = y.shape().lower_volume(1);
			int *gpu_index_data;
			cudaMalloc((void **)&gpu_index_data, sizeof(int) * r);
			Fmaxpooling_impl(x.v, y.v, n, r, s, gpu_index_data);
			cudaMemcpy(index, gpu_index_data, sizeof(int) * r, cudaMemcpyDeviceToHost);
			cudaFree(gpu_index_data);
		}

		void DMaxPooling(const LDG::Tensor& x, const LDG::Tensor& y, 
				const LDG::Tensor& gy, LDG::Tensor& gx, int* index) {
			const int dim0 = x.shape().dims()[0];
			int *gpu_index_data;
			cudaMalloc((void **)&gpu_index_data, sizeof(int) * dim0);
			cudaMemcpy(gpu_index_data, index, sizeof(int) * dim0, cudaMemcpyHostToDevice);
			Dmaxpooling_impl(x.v, y.v, gy.v, gx.v, gpu_index_data, dim0);
			cudaFree(gpu_index_data);
		}

		void FMinPooling(const LDG::Tensor& x, LDG::Tensor& y, int* index) {
			const int n = x.shape()[1];
			const int r = y.shape().size();
			const int s = y.shape().lower_volume(1);
			int *gpu_index_data;
			cudaMalloc((void **)&gpu_index_data, sizeof(int) * r);
			Fminpooling_impl(x.v, y.v, n, r, s, gpu_index_data);
			cudaMemcpy(index, gpu_index_data, sizeof(int) * r, cudaMemcpyDeviceToHost);
			cudaFree(gpu_index_data);
		}

		void DMinPooling(const LDG::Tensor& x, const LDG::Tensor& y, 
				const LDG::Tensor& gy, LDG::Tensor& gx, int* index) {
			const int dim0 = x.shape().dims()[0];
			int *gpu_index_data;
			cudaMalloc((void **)&gpu_index_data, sizeof(int) * dim0);
			cudaMemcpy(gpu_index_data, index, sizeof(int) * dim0, cudaMemcpyHostToDevice);
			Dminpooling_impl(x.v, y.v, gy.v, gx.v, gpu_index_data, dim0);
			cudaFree(gpu_index_data);
		}
};

#endif
