#ifndef CUDA_DEVICE
#define CUDA_DEVICE

#include <iostream>
#include <iomanip>
#include <chrono>

#include "memory_pool.h"
#include "Device.h"
#include "MyLib.h"
#include "kernel.cuh"

static void *my_alloc(std::size_t size) {
	void *ptr;
	cudaMalloc((void **)&ptr, size);
	return ptr;
}

static void my_delete(void *ptr) {
	cudaFree(ptr);
}

class CudaDevice : public Device {
	private:
		cublasHandle_t handle;

		MemoryPool *mem_pool;
	public:

	public:
		CudaDevice() {
			Device::device_type = CUDA;
			cublasCreate(&handle);

			mem_pool = new MemoryPool(my_alloc, my_delete);
		}

		~CudaDevice() {
			delete mem_pool;
		}

	public:
		void init(LDG::Tensor &t, const Shape &shape) {
			malloc(t, shape);
			zero(t);
		}

		/*
		void copy_data(const LDG::Tensor &src, LDG::Tensor& dst) {
			if(src.shape().has_same_dims(dst.shape())
					&& src.device_type == dst.device_type) {
				int memsize = src.shape().size() * sizeof(dtype);
				cudaMemcpy(dst.v, src.v, memsize, cudaMemcpyDeviceToDevice);
			} else
				cout << "copy error"  << endl;
		}
		*/


		void set(LDG::Tensor &t, const dtype* host_data, int h_size) {
			int size = t.shape().size();
			if (size == h_size) {
				int memsize = sizeof(dtype) * size;
				cudaMemcpy(MDATA(t), host_data, memsize, cudaMemcpyHostToDevice);
			} else
				cout << "set size not match" << endl;
		}

		void set_col(LDG::Tensor &t, int col, dtype val) {
			int dim0 = t.shape()[0];
			int dim1 = t.shape()[1];
			if(col < dim1) {
				set_col_impl(MDATA(t), dim0, col, t.shape().size(), val);
			} else
				std::cout << "set col beyond dim1 " << endl;
		}

		void malloc(LDG::Tensor &t, const Shape &shape) {
			t.device_type = CUDA;
			t.shape_ = shape;
			int size = shape.size();
			int memsize = sizeof(dtype) * size;
			//cudaMalloc((void **)&t.v, memsize);

			t.handle_ = mem_pool->allocate(memsize);
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
				cudaMemcpy(MDATA(t), vec_val.data(), sizeof(dtype) * size, cudaMemcpyHostToDevice);
			} else 
				cout << "set error dim is not match" << endl;
		}

		void get_col(const LDG::Tensor& x, int col, LDG::Tensor& r) {
			int dim0 = x.shape()[0];
			int dim1 = x.shape()[1];
			malloc(r, Shape({dim0, 1}));

			if(col < dim1) {
				get_col_impl(CDATA(x), MDATA(r), dim0, col, x.shape().size());
			} else
				cout << "get col, col beyond" << endl;
		}

		void get_cols(const LDG::Tensor& x, int* cols, int col_num, LDG::Tensor& r) {
			int memsize = sizeof(int) * col_num;

			std::shared_ptr<void> gpu_ptrs = mem_pool->allocate(memsize); 
			int *gpu_cols = static_cast<int *>(gpu_ptrs.get());

			cudaMemcpy(gpu_cols, cols, memsize, cudaMemcpyHostToDevice);

			int xdim0 = x.shape()[0];
			int xdim1 = x.shape()[1];
			malloc(r, Shape({xdim0 * col_num, 1}));

			int r_size = r.shape().size();

			for(int idx = 0; idx < col_num; idx++)
				assert(cols[idx] < xdim1);

			malloc(r, Shape({xdim0, col_num}));
			get_cols_impl(CDATA(x), MDATA(r), xdim0, xdim1, r_size, gpu_cols, col_num);
		}

		void FLookup(const LDG::Tensor& x, const vector<int>& vec_cols, vector<LDG::PTensor>& vec_r) {
			int size = vec_cols.size();
			int cols[size];
			for(int idx = 0; idx < size; idx++) {
				cols[idx] = vec_cols[idx];
			}
			FLookup(x, cols, size, vec_r);
		}

		void DLookup(LDG::Tensor& gx, const vector<int>& vec_cols, vector<LDG::PTensor>& vec_loss) {
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

			int memsize = sizeof(int) * col_num;
			std::shared_ptr<void> cols_ptrs = mem_pool->allocate(memsize); 
			int *gpu_cols = static_cast<int *>(cols_ptrs.get());
			cudaMemcpy(gpu_cols, cols, memsize, cudaMemcpyHostToDevice);

			std::shared_ptr<void> r_ptrs = mem_pool->allocate(sizeof(dtype*) * col_num); 
			dtype **v_data_r = static_cast<dtype **>(r_ptrs.get());

			for(int i = 0; i < col_num; i++) {
				const dtype* tmp_ptr = CDATA(*vec_r[i]);
				cudaMemcpy(v_data_r + i, &(tmp_ptr), sizeof(dtype*) ,cudaMemcpyHostToDevice);
			}

			int xdim0 = x.shape()[0];
			int xdim1 = x.shape()[1];
			int rdim0 = vec_r[0]->shape()[0];
			int r_size = rdim0 * col_num;
			
			if(xdim0 == rdim0) {
				FLookup_impl(CDATA(x), v_data_r, xdim0, xdim1, r_size, gpu_cols, col_num);
			} else
				cout << "get col dims are not matched" << endl;
		}


		void DLookup(LDG::Tensor& gx, int* cols, int col_num, vector<LDG::PTensor>& vec_loss) {
			if(vec_loss.size() != col_num) {
				cout << "error vec_loss size is not matched." << endl;
			}


			int memsize = sizeof(int) * col_num;
			std::shared_ptr<void> cols_ptrs = mem_pool->allocate(memsize); 
			int *gpu_cols = static_cast<int *>(cols_ptrs.get());

			cudaMemcpy(gpu_cols, cols, memsize, cudaMemcpyHostToDevice);


			std::shared_ptr<void> loss_ptrs = mem_pool->allocate(sizeof(dtype*) * col_num); 
			dtype **v_data_loss = static_cast<dtype **>(loss_ptrs.get());

			for(int i = 0; i < col_num; i++) {
				const dtype* tmp_ptr = CDATA(*vec_loss[i]);
				cudaMemcpy(v_data_loss + i, &(tmp_ptr), sizeof(dtype*) ,cudaMemcpyHostToDevice);
			}

			int gxdim0 = gx.shape()[0];
			int gxdim1 = gx.shape()[1];
			int ldim0 = vec_loss[0]->shape()[0];
			int l_size = ldim0 * col_num;
			
			if(gxdim0 == ldim0) {
				DLookup_impl(MDATA(gx), v_data_loss, gxdim0, gxdim1, l_size, gpu_cols, col_num);
			} else
				cout << "get col dims are not matched" << endl;
		}

		void random_uniform(LDG::Tensor &t, const Shape &shape, float lower, float upper) {
			int size = shape.size();
			int memsize = sizeof(dtype) * size;
			dtype *host_data = new dtype[size];
			dtype min = lower, max = upper;
			for (int i = 0; i < size; i++) {
				host_data[i] = (dtype(rand()) / RAND_MAX) * (max - min) + min;
			}

			malloc(t, shape);
			//cudaMemcpy(t.v, host_data, memsize, cudaMemcpyHostToDevice);

			cudaMemcpy(MDATA(t), host_data, memsize, cudaMemcpyHostToDevice);
			delete []host_data;
		} 

		void random_bernoulli(LDG::Tensor &t, const Shape &shape, float p){}

		void random_normal(LDG::Tensor &t, const Shape &shape, float mean, float sd){}

		void random_log_normal(LDG::Tensor &t, const Shape &shape, float mean, float sd){}


		void Fequal(const LDG::Tensor& x, LDG::Tensor& r){}

		void Ftanh(const LDG::Tensor& x, LDG::Tensor& r) {
			int x_size = x.shape().size();
			malloc(r, x.shape());
			Ftanh_impl(CDATA(x), MDATA(r), x_size);
		}

		void Ftanh(const vector<LDG::PTensor>& vec_x, vector<LDG::PTensor>& vec_r) {
			const int n_x = vec_x.size();
			const int n_r = vec_r.size();
			if(n_x == n_r){
				int n = n_x;
				for(int idx = 0; idx < n; idx++)
					malloc(*vec_r[idx], vec_x[idx]->shape());

				int mem_size = sizeof(dtype*) * n;

				vector<const dtype*> vec_ptr_x(n);
				for(int i = 0; i < n; i++) {
					vec_ptr_x[i] = (CDATA(*vec_x[i]));
				}
				std::shared_ptr<void> x_ptrs = mem_pool->allocate(mem_size); 
				dtype **v_data_x = static_cast<dtype **>(x_ptrs.get());
				cudaMemcpy(v_data_x, (dtype**)vec_ptr_x.data(), mem_size, cudaMemcpyHostToDevice);


				vector<const dtype*> vec_ptr_r(n);
				for(int i = 0; i < n; i++) {
					vec_ptr_r[i] = (MDATA(*vec_r[i]));
				}
				std::shared_ptr<void> r_ptrs = mem_pool->allocate(mem_size); 
				dtype **v_data_r = static_cast<dtype **>(r_ptrs.get());
				cudaMemcpy(v_data_r, (dtype**)vec_ptr_r.data(), mem_size, cudaMemcpyHostToDevice);

				int dim0 = vec_x[0]->shape()[0];
				int size = dim0 * n_x;

				Ftanh_impl(v_data_x, v_data_r, dim0, size);
			} else 
				std::cout << "error the number of Ftanh tensors is not matched "<<endl;
		}

		void Fsigmoid(const vector<LDG::PTensor>& vec_x, vector<LDG::PTensor>& vec_r) {
			const int n_x = vec_x.size();
			const int n_r = vec_r.size();
			if(n_x == n_r){
				int n = n_r;
				for(int idx = 0; idx < n_r; idx++)
					malloc(*vec_r[idx], vec_x[idx]->shape());

				int mem_size = sizeof(dtype*) * n;

				vector<const dtype*> vec_ptr_x(n);
				for(int i = 0; i < n; i++) {
					vec_ptr_x[i] = (CDATA(*vec_x[i]));
				}
				std::shared_ptr<void> x_ptrs = mem_pool->allocate(mem_size); 
				dtype **v_data_x = static_cast<dtype **>(x_ptrs.get());
				cudaMemcpy(v_data_x, (dtype**)vec_ptr_x.data(), mem_size, cudaMemcpyHostToDevice);


				vector<const dtype*> vec_ptr_r(n);
				for(int i = 0; i < n; i++) {
					vec_ptr_r[i] = (MDATA(*vec_r[i]));
				}
				std::shared_ptr<void> r_ptrs = mem_pool->allocate(mem_size); 
				dtype **v_data_r = static_cast<dtype **>(r_ptrs.get());
				cudaMemcpy(v_data_r, (dtype**)vec_ptr_r.data(), mem_size, cudaMemcpyHostToDevice);


				int dim0 = vec_x[0]->shape()[0];
				int size = dim0 * n_x;

				Fsigmoid_impl(v_data_x, v_data_r, dim0, size);
			} else 
				std::cout << "error the number of Fsigmoid tensors is not matched "<<endl;
		}

		void Dsigmoid(const vector<LDG::PTensor>& vec_x, const vector<LDG::PTensor>& vec_y, vector<LDG::PTensor>& vec_r) {
			const int n_x = vec_x.size();
			const int n_y = vec_y.size();
			const int n_r = vec_r.size();
			if(n_x == n_r && n_y == n_r) {

				for(int idx = 0; idx < n_x; idx++) {
					int xdim0 = vec_x[idx]->shape()[0];	
					int ydim0 = vec_y[idx]->shape()[0];	
					assert(xdim0 == ydim0);
					malloc(*vec_r[idx], vec_x[idx]->shape());
				}
				int n = n_x;

				int mem_size = sizeof(dtype*) * n;

				vector<const dtype*> vec_ptr_x(n);
				for(int i = 0; i < n; i++) {
					vec_ptr_x[i] = (CDATA(*vec_x[i]));
				}
				std::shared_ptr<void> x_ptrs = mem_pool->allocate(mem_size); 
				dtype **v_data_x = static_cast<dtype **>(x_ptrs.get());
				cudaMemcpy(v_data_x, (dtype**)vec_ptr_x.data(), mem_size, cudaMemcpyHostToDevice);

				vector<const dtype*> vec_ptr_y(n);
				for(int i = 0; i < n; i++) {
					vec_ptr_y[i] = (CDATA(*vec_y[i]));
				}
				std::shared_ptr<void> y_ptrs = mem_pool->allocate(mem_size); 
				dtype **v_data_y = static_cast<dtype **>(y_ptrs.get());
				cudaMemcpy(v_data_y, (dtype**)vec_ptr_y.data(), mem_size, cudaMemcpyHostToDevice);

				vector<const dtype*> vec_ptr_r(n);
				for(int i = 0; i < n; i++) {
					vec_ptr_r[i] = (MDATA(*vec_r[i]));
				}
				std::shared_ptr<void> r_ptrs = mem_pool->allocate(mem_size); 
				dtype **v_data_r = static_cast<dtype **>(r_ptrs.get());
				cudaMemcpy(v_data_r, (dtype**)vec_ptr_r.data(), mem_size, cudaMemcpyHostToDevice);

				int dim0 = vec_x[0]->shape()[0];
				int size = dim0 * n_x;

				Dsigmoid_impl(v_data_x, v_data_y, v_data_r, dim0, size);
			} else 
				std::cout << "error the number of Dsigmoid tensors is not matched "<<endl;
		}


		void Dtanh(const vector<LDG::PTensor>& vec_x, const vector<LDG::PTensor>& vec_y, vector<LDG::PTensor>& vec_r) {
			const int n_x = vec_x.size();
			const int n_y = vec_y.size();
			const int n_r = vec_r.size();
			if(n_x == n_r && n_y == n_r) {

				for(int idx = 0; idx < n_x; idx++) {
					int xdim0 = vec_x[idx]->shape()[0];	
					int ydim0 = vec_y[idx]->shape()[0];	
					assert(xdim0 == ydim0);
					malloc(*vec_r[idx], vec_x[idx]->shape());
				}
				int n = n_x;

				int mem_size = sizeof(dtype*) * n;

				vector<const dtype*> vec_ptr_x(n);
				for(int i = 0; i < n; i++) {
					vec_ptr_x[i] = (CDATA(*vec_x[i]));
				}

				std::shared_ptr<void> x_ptrs = mem_pool->allocate(mem_size); 
				dtype **v_data_x = static_cast<dtype **>(x_ptrs.get());
				cudaMemcpy(v_data_x, (dtype**)vec_ptr_x.data(), mem_size, cudaMemcpyHostToDevice);

				vector<const dtype*> vec_ptr_y(n);
				for(int i = 0; i < n; i++) {
					vec_ptr_y[i] = (CDATA(*vec_y[i]));
				}
				std::shared_ptr<void> y_ptrs = mem_pool->allocate(mem_size); 
				dtype **v_data_y = static_cast<dtype **>(y_ptrs.get());
				cudaMemcpy(v_data_y, (dtype**)vec_ptr_y.data(), mem_size, cudaMemcpyHostToDevice);

				vector<const dtype*> vec_ptr_r(n);
				for(int i = 0; i < n; i++) {
					vec_ptr_r[i] = (MDATA(*vec_r[i]));
				}
				std::shared_ptr<void> r_ptrs = mem_pool->allocate(mem_size); 
				dtype **v_data_r = static_cast<dtype **>(r_ptrs.get());
				cudaMemcpy(v_data_r, (dtype**)vec_ptr_r.data(), mem_size, cudaMemcpyHostToDevice);

				int dim0 = vec_x[0]->shape()[0];
				int size = dim0 * n_x;

				Dtanh_impl(v_data_x, v_data_y, v_data_r, dim0, size);
			} else 
				std::cout << "error the number of Dtanh tensors is not matched "<<endl;
		}

		void Fsigmoid(const LDG::Tensor& x, LDG::Tensor& r){
			int x_size = x.shape().size();
			malloc(r, x.shape());
			Fsigmoid_impl(CDATA(x), MDATA(r), x_size);
		}

		void Frelu(const LDG::Tensor& x, LDG::Tensor& r){}
		void Fleaky_relu(const LDG::Tensor& x, LDG::Tensor& r){}
		void Fexp(const LDG::Tensor& x, LDG::Tensor& r){}
		void Flog(const LDG::Tensor& x, LDG::Tensor& r){}


		void Fsquare(const LDG::Tensor& x, LDG::Tensor& r) {
			int x_size = x.shape().size();
			malloc(r, x.shape());
			Fsquare_impl(CDATA(x), MDATA(r), x_size);
		}

		void Fsqrt(const LDG::Tensor& x, LDG::Tensor& r) {
			int x_size = x.shape().size();
			malloc(r, x.shape());
			Fsqrt_impl(CDATA(x), MDATA(r), x_size);
		}

		void Dequal(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r){}

		void Dtanh(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {
			int x_size = x.shape().size();
			int y_size = y.shape().size();
			malloc(r, x.shape());
			if(x_size == y_size)
				Dtanh_impl(CDATA(x), CDATA(y), MDATA(r), x_size);
			else
				std::cout << "error, dtanh dim is not match" << std::endl;
		}

		void Dleaky_relu(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r){}

		void Dsigmoid(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r){
			int x_size = x.shape().size();
			int y_size = y.shape().size();
			malloc(r, x.shape());
			if(x_size == y_size)
				Dsigmoid_impl(CDATA(x), CDATA(y), MDATA(r), x_size);
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
				malloc(r, x.shape());
				Fadd_impl(CDATA(x), CDATA(y), MDATA(r), x_size);
			}
			else
				std::cout << "error, add dim is not match" << std::endl;
			return r;
		}

		void Fadd(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {
			int x_size = x.shape().size();
			int y_size = y.shape().size();
			if(x_size == y_size) {
				malloc(r, x.shape());
				Fadd_impl(CDATA(x), CDATA(y), MDATA(r), x_size);
			}
			else
				std::cout << "error, add dim is not match" << std::endl;
		}

		virtual void Fadd_inplace(LDG::Tensor& x, const LDG::Tensor& y) {
			int x_size = x.shape().size();
			int y_size = y.shape().size();
			if(x_size == y_size) {
				Fadd_inplace_impl(MDATA(x), CDATA(y), x_size);
			} else 
				std::cout << "error, add dim is not match" << std::endl;
		}

		virtual void Fadd(const LDG::Tensor& x, const vector<LDG::PTensor>& vec_y, LDG::Tensor& r) {
			int n = vec_y.size();
			int x_size = x.shape().size();
			malloc(r, x.shape());
			for(int idx = 0; idx < n; idx++){
				assert(x_size == vec_y[idx]->shape().size());
			}


			int mem_size = sizeof(dtype*) * n;

			vector<const dtype*> vec_ptr_y(n);
			for(int i = 0; i < n; i++) {
				vec_ptr_y[i] = (CDATA(*vec_y[i]));
			}
			std::shared_ptr<void> y_ptrs = mem_pool->allocate(mem_size); 
			dtype **v_data_y = static_cast<dtype **>(y_ptrs.get());
			cudaMemcpy(v_data_y, (dtype**)vec_ptr_y.data(), mem_size, cudaMemcpyHostToDevice);

			Fadd_impl(CDATA(x), v_data_y, MDATA(r), n, x_size);
		}

		void Fadd(const vector<LDG::PTensor>& vec_x, const vector<LDG::PTensor>& vec_y, vector<LDG::PTensor>& vec_r) {
			int size_x = vec_x.size();
			int size_y = vec_y.size();
			int size_r = vec_r.size();

			if(size_x == size_y) {
				int n = size_x;
				for(int idx = 0; idx < n; idx++) {
					int xdim0 = vec_x[idx]->shape()[0];	
					int ydim0 = vec_y[idx]->shape()[0];	
					assert(xdim0 == ydim0);
					malloc(*vec_r[idx], vec_x[idx]->shape());
				}

				int mem_size = sizeof(dtype*) * n;

				vector<const dtype*> vec_ptr_x(n);
				for(int i = 0; i < n; i++) {
					vec_ptr_x[i] = (CDATA(*vec_x[i]));
				}
				std::shared_ptr<void> x_ptrs = mem_pool->allocate(mem_size); 
				dtype **v_data_x = static_cast<dtype **>(x_ptrs.get());
				cudaMemcpy(v_data_x, (dtype**)vec_ptr_x.data(), mem_size, cudaMemcpyHostToDevice);


				vector<const dtype*> vec_ptr_y(n);
				for(int i = 0; i < n; i++) {
					vec_ptr_y[i] = (CDATA(*vec_y[i]));
				}
				std::shared_ptr<void> y_ptrs = mem_pool->allocate(mem_size); 
				dtype **v_data_y = static_cast<dtype **>(y_ptrs.get());
				cudaMemcpy(v_data_y, (dtype**)vec_ptr_y.data(), mem_size, cudaMemcpyHostToDevice);


				vector<const dtype*> vec_ptr_r(n);
				for(int i = 0; i < n; i++) {
					vec_ptr_r[i] = (MDATA(*vec_r[i]));
				}
				std::shared_ptr<void> r_ptrs = mem_pool->allocate(mem_size); 
				dtype **v_data_r = static_cast<dtype **>(r_ptrs.get());
				cudaMemcpy(v_data_r, (dtype**)vec_ptr_r.data(), mem_size, cudaMemcpyHostToDevice);


				int size = vec_x[0]->shape().size();
				Fadd_impl(v_data_x, v_data_y, v_data_r, size, size * n);
			} else {
				cout << "Fadd size is not matched" << endl;
			}
		}

		void Fadd_inplace(vector<LDG::PTensor>& vec_x, const vector<LDG::PTensor>& vec_y) {
			int size_x = vec_x.size();
			int size_y = vec_y.size();

			if(size_x == size_y) {
				int n = size_x;
				for(int idx = 0; idx < n; idx++) {
					int xdim0 = vec_x[idx]->shape()[0];	
					int ydim0 = vec_y[idx]->shape()[0];	
					assert(xdim0 == ydim0);
				}

				int mem_size = sizeof(dtype*) * n;

				vector<const dtype*> vec_ptr_x(n);
				for(int i = 0; i < n; i++) {
					vec_ptr_x[i] = (CDATA(*vec_x[i]));
				}
				std::shared_ptr<void> x_ptrs = mem_pool->allocate(mem_size); 
				dtype **v_data_x = static_cast<dtype **>(x_ptrs.get());
				cudaMemcpy(v_data_x, (dtype**)vec_ptr_x.data(), mem_size, cudaMemcpyHostToDevice);


				vector<const dtype*> vec_ptr_y(n);
				for(int i = 0; i < n; i++) {
					vec_ptr_y[i] = (CDATA(*vec_y[i]));
				}
				std::shared_ptr<void> y_ptrs = mem_pool->allocate(mem_size); 
				dtype **v_data_y = static_cast<dtype **>(y_ptrs.get());
				cudaMemcpy(v_data_y, (dtype**)vec_ptr_y.data(), mem_size, cudaMemcpyHostToDevice);

				int size = vec_x[0]->shape().size();
				Fadd_inplace_impl(v_data_x, v_data_y, size, size * n);
			} else {
				cout << "Fadd size is not matched" << endl;
			}
		}

		void Fadd_inplace(LDG::Tensor& x, const vector<LDG::PTensor>& vec_y) {
			int n = vec_y.size();
			int x_size = x.shape().size();
			for(int idx = 0; idx < n; idx++){
				assert(x_size == vec_y[idx]->shape().size());
			}

			int mem_size = sizeof(dtype*) * n;

			vector<const dtype*> vec_ptr_y(n);
			for(int i = 0; i < n; i++) {
				vec_ptr_y[i] = (CDATA(*vec_y[i]));
			}
			std::shared_ptr<void> y_ptrs = mem_pool->allocate(mem_size); 
			dtype **v_data_y = static_cast<dtype **>(y_ptrs.get());
			cudaMemcpy(v_data_y, (dtype**)vec_ptr_y.data(), mem_size, cudaMemcpyHostToDevice);

			Fadd_inplace_impl(MDATA(x), v_data_y, n, x_size);
		}

		void Fsubtract(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r){
			int x_size = x.shape().size();
			int y_size = y.shape().size();
			if(x_size == y_size) {
				malloc(r, x.shape());
				Fsubtract_impl(CDATA(x), CDATA(y), MDATA(r), x_size);
			}
			else
				std::cout << "error, subtract dim is not match" << std::endl;
		}

		void Fsubtract_inplace(LDG::Tensor& x, const LDG::Tensor& y) {
			int x_size = x.shape().size();
			int y_size = y.shape().size();
			if(x_size == y_size) {
				Fsubtract_inplace_impl(MDATA(x), CDATA(y), x_size);
			}
			else
				std::cout << "error, subtract dim is not match" << std::endl;
		}

		void Fmultiply(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {
			int x_size = x.shape().size();
			int y_size = y.shape().size();
			if(x_size == y_size) {
				malloc(r, x.shape());
				Fmultiply_impl(CDATA(x), CDATA(y), MDATA(r), x_size);
			}
			else
				std::cout << "error, multiply dim is not match" << std::endl;

		}

		void Fmultiply_inplace(LDG::Tensor& x, const LDG::Tensor& y) {
			int x_size = x.shape().size();
			int y_size = y.shape().size();
			if(x_size == y_size) {
				Fmultiply_inplace_impl(MDATA(x), CDATA(y), x_size);
			}
			else
				std::cout << "error, multiply dim is not match" << std::endl;

		}

		void Fadd_scalar(const LDG::Tensor& x, const dtype y, LDG::Tensor& r) {
			int x_size = x.shape().size();
			malloc(r, x.shape());
			Fadd_scalar_impl(CDATA(x), y, MDATA(r), x_size);
		}

		void Fadd_scalar_inplace(LDG::Tensor& x, const dtype y) {
			int x_size = x.shape().size();
			Fadd_scalar_inplace_impl(MDATA(x), y, x_size);
		}


		void Fmultiply_scalar(const LDG::Tensor& x, const dtype y, LDG::Tensor& r) {
			int x_size = x.shape().size();
			malloc(r, x.shape());
			Fmultiply_scalar_impl(CDATA(x), y, MDATA(r), x_size);
		}

		void Fmultiply_scalar(const LDG::Tensor& x, const LDG::Tensor &scalar, LDG::Tensor& r) {
			int x_size = x.shape().size();
			malloc(r, x.shape());
			assert(scalar.shape().size() == 1);
			Fmultiply_scalar_impl(CDATA(x), CDATA(scalar), MDATA(r), x_size);
		}

		void Fmultiply_scalar_inplace(LDG::Tensor& x, const dtype y) {
			int x_size = x.shape().size();
			Fmultiply_scalar_inplace_impl(MDATA(x), y, x_size);
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
					
			Fadd_col_impl(MDATA(x), CDATA(y_col), col, dim0, size);
		}

		void Fdivide(const LDG::Tensor& x, const LDG::Tensor& y, LDG::Tensor& r) {
			int x_size = x.shape().size();
			int y_size = y.shape().size();
			if(x_size == y_size) {
				malloc(r, x.shape());
				Fdivide_impl(CDATA(x), CDATA(y), MDATA(r), x_size);
			}
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
			malloc(x, Shape({x_dim, x_size}));
			malloc(y, Shape({y_dim, y_size}));
			concat(vec_x, x);
			concat(vec_y, y);

			Fmatmul(x, y, r, tx, ty);
		} 
		
		void Fmatmul(const vector<LDG::PTensor> &vec_x, const vector<LDG::PTensor> &vec_y, vector<LDG::PTensor> &vec_r,
				bool tx = false, bool ty = false) {
			int x_size = vec_x.size();
			int y_size = vec_y.size();
			int r_size = vec_y.size();
			int x_dim = vec_x[0]->shape()[0];
			int y_dim = vec_y[0]->shape()[0];
			int r_dim = vec_r[0]->shape()[0];
			LDG::Tensor x, y, r;
			malloc(x, Shape({x_dim, x_size}));
			malloc(y, Shape({y_dim, y_size}));
			malloc(r, Shape({r_dim, r_size}));
			concat(vec_x, x);
			concat(vec_y, y);
			concat(vec_r, r);

			Fmatmul(x, y, r, tx, ty);
			unconcat(r, vec_r);
		} 

		void Fmatmul(const LDG::Tensor &x, const LDG::Tensor &y, LDG::Tensor &r,
				bool tx = false, bool ty = false) {
			int m = tx ? x.shape()[1] : x.shape()[0];
			int n = tx ?  x.shape()[0] : x.shape()[1];
			int k = ty ? y.shape()[0] : y.shape()[1];

			malloc(r, Shape({m, k}));

			dtype alpha = 1;
			dtype beta =  0;

			cublasOperation_t transx = tx ? CUBLAS_OP_T : CUBLAS_OP_N;
			int ldx = tx ? n : m;

			cublasOperation_t transy = ty ? CUBLAS_OP_T : CUBLAS_OP_N;
			int ldy = ty ? k : n;

#if USE_FLOAT
			(cublasSgemm(handle, transx, transy, m, k, n,
						 &alpha, CDATA(x), ldx, CDATA(y), ldy, &beta, MDATA(r), m));
#else                   
			(cublasDgemm(handle, transx, transy, m, k, n,
						 &alpha, CDATA(x), ldx, CDATA(y), ldy, &beta, MDATA(r), m));
#endif                  
		}

		void Fmatmul(const LDG::Tensor &x, const vector<LDG::PTensor> &vec_y, vector<LDG::PTensor> &vec_r, bool tx = false, bool ty = false) {
			int size_y = vec_y.size();
			int size_r = vec_r.size();
			assert(size_y == size_r);
			int m = tx ? x.shape()[1] : x.shape()[0];
			int n = tx ?  x.shape()[0] : x.shape()[1];
			for (int idx = 0; idx < size_y; idx++) {
				LDG::PTensor ptr_y = vec_y[idx];
				LDG::PTensor ptr_r = vec_r[idx];

				int k = ty ? ptr_y->shape()[0] : ptr_y->shape()[1];
				malloc(*ptr_r, Shape({m, k}));
			}

			LDG::Tensor y;
			malloc(y, Shape({vec_y[0]->shape()[0], size_y}));
			concat(vec_y, y);

			LDG::Tensor r;
			malloc(r, Shape({vec_r[0]->shape()[0], size_r}));
			concat(vec_r, r);

			dtype alpha = 1;
			dtype beta =  0;

			int k = ty ? y.shape()[0] : y.shape()[1];

			cublasOperation_t transx = tx ? CUBLAS_OP_T : CUBLAS_OP_N;
			int ldx = tx ? n : m;

			cublasOperation_t transy = ty ? CUBLAS_OP_T : CUBLAS_OP_N;
			int ldy = ty ? k : n;

#if USE_FLOAT
			(cublasSgemm(handle, transx, transy, m, k, n,
						 &alpha, CDATA(x), ldx, CDATA(y), ldy, &beta, MDATA(r), m));
#else                   
			(cublasDgemm(handle, transx, transy, m, k, n,
						 &alpha, CDATA(x), ldx, CDATA(y), ldy, &beta, MDATA(r), m));
#endif                  
			unconcat(r, vec_r);
		}

		void Fmultiply(const vector<LDG::PTensor>& vec_x, const vector<LDG::PTensor>& vec_y, vector<LDG::PTensor>& vec_r) {
			int size_x = vec_x.size();
			int size_y = vec_y.size();
			int size_r = vec_r.size();

			if(size_x == size_y) {
				int n = size_x;
				for(int idx = 0; idx < n; idx++) {
					int xdim0 = vec_x[idx]->shape()[0];	
					int ydim0 = vec_y[idx]->shape()[0];	
					int rdim0 = vec_r[idx]->shape()[0];	
					assert(xdim0 == ydim0);
					malloc(*vec_r[idx], vec_x[idx]->shape());
				}
				int mem_size = sizeof(dtype*) * n;

				vector<const dtype*> vec_ptr_x(n);
				for(int i = 0; i < n; i++) {
					vec_ptr_x[i] = (CDATA(*vec_x[i]));
				}
				std::shared_ptr<void> x_ptrs = mem_pool->allocate(mem_size); 
				dtype **v_data_x = static_cast<dtype **>(x_ptrs.get());
				cudaMemcpy(v_data_x, (dtype**)vec_ptr_x.data(), mem_size, cudaMemcpyHostToDevice);

				vector<const dtype*> vec_ptr_y(n);
				for(int i = 0; i < n; i++) {
					vec_ptr_y[i] = (CDATA(*vec_y[i]));
				}
				std::shared_ptr<void> y_ptrs = mem_pool->allocate(mem_size); 
				dtype **v_data_y = static_cast<dtype **>(y_ptrs.get());
				cudaMemcpy(v_data_y, (dtype**)vec_ptr_y.data(), mem_size, cudaMemcpyHostToDevice);

				vector<const dtype*> vec_ptr_r(n);
				for(int i = 0; i < n; i++) {
					vec_ptr_r[i] = (MDATA(*vec_r[i]));
				}
				std::shared_ptr<void> r_ptrs = mem_pool->allocate(mem_size); 
				dtype **v_data_r = static_cast<dtype **>(r_ptrs.get());
				cudaMemcpy(v_data_r, (dtype**)vec_ptr_r.data(), mem_size, cudaMemcpyHostToDevice);

				int size = vec_x[0]->shape().size();
				Fmultiply_impl(v_data_x, v_data_y, v_data_r, size, size * n);
			} else {
				cout << "Fmultiply size is not matched" << endl;
			}
		}

		void Fmultiply_inplace(vector<LDG::PTensor>& vec_x, const vector<LDG::PTensor>& vec_y) {
			int size_x = vec_x.size();
			int size_y = vec_y.size();

			if(size_x == size_y) {
				int n = size_x;
				for(int idx = 0; idx < n; idx++) {
					int xdim0 = vec_x[idx]->shape()[0];	
					int ydim0 = vec_y[idx]->shape()[0];	
					assert(xdim0 == ydim0);
				}

				int mem_size = sizeof(dtype*) * n;

				vector<const dtype*> vec_ptr_x(n);
				for(int i = 0; i < n; i++) {
					vec_ptr_x[i] = (CDATA(*vec_x[i]));
				}
				std::shared_ptr<void> x_ptrs = mem_pool->allocate(mem_size); 
				dtype **v_data_x = static_cast<dtype **>(x_ptrs.get());
				cudaMemcpy(v_data_x, (dtype**)vec_ptr_x.data(), mem_size, cudaMemcpyHostToDevice);


				vector<const dtype*> vec_ptr_y(n);
				for(int i = 0; i < n; i++) {
					vec_ptr_y[i] = (CDATA(*vec_y[i]));
				}
				std::shared_ptr<void> y_ptrs = mem_pool->allocate(mem_size); 
				dtype **v_data_y = static_cast<dtype **>(y_ptrs.get());
				cudaMemcpy(v_data_y, (dtype**)vec_ptr_y.data(), mem_size, cudaMemcpyHostToDevice);

				int size = vec_x[0]->shape().size();
				Fmultiply_inplace_impl(v_data_x, v_data_y, size, size * n);
			} else {
				cout << "Fmultiply size is not matched" << endl;
			}
		}
/*
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
		*/

		void concat(const vector<LDG::PTensor>& vec_x, LDG::Tensor& r) {
			int max_size = vec_x.size();	
			int offset = 0;
			for(int idx = 0; idx < max_size; idx++) {
				int dim_0 = vec_x[idx]->shape()[0];
				concat_impl(CDATA(*vec_x[idx]), MDATA(r), offset, dim_0);
				offset += dim_0;
			}
		}

		void unconcat(const LDG::Tensor& r, vector<LDG::PTensor>& vec_x) {
			int max_size = vec_x.size();
			int offset = 0;
			for(int idx = 0; idx < max_size; idx++) {
				int dim_0 = vec_x[idx]->shape()[0];
				unconcat_impl(CDATA(r), MDATA(*vec_x[idx]), offset, dim_0);
				offset += dim_0;
			}
		}

		/*
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
		*/

		/*
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
		*/

		void FAvgPooling(const vector<LDG::PTensor>& vec_x, LDG::Tensor& y) {
			const int n = vec_x.size();
			for(int idx = 1; idx < n; idx++) {
				assert(vec_x[idx]->shape().size() == vec_x[0]->shape().size());
			}
			malloc(y, vec_x[0]->shape());

			const int r = y.shape().size();
			const int s = y.shape().lower_volume(1);

			int mem_size = sizeof(dtype*) * n;

			vector<const dtype*> vec_ptr_x(n);
			for(int i = 0; i < n; i++) {
				vec_ptr_x[i] = (CDATA(*vec_x[i]));
			}
			std::shared_ptr<void> x_ptrs = mem_pool->allocate(mem_size); 
			dtype **v_data_x = static_cast<dtype **>(x_ptrs.get());
			cudaMemcpy(v_data_x, (dtype**)vec_ptr_x.data(), mem_size, cudaMemcpyHostToDevice);

			Favgpooling_impl(v_data_x, MDATA(y), n, r, s);
		}

		void DAvgPooling(const LDG::Tensor& gy, vector<LDG::PTensor>& vec_gx) {
			const int n = vec_gx.size();

			int mem_size = sizeof(dtype*) * n;

			vector<const dtype*> vec_ptr_gx(n);
			for(int i = 0; i < n; i++) {
				vec_ptr_gx[i] = (CDATA(*vec_gx[i]));
			}
			std::shared_ptr<void> gx_ptrs = mem_pool->allocate(mem_size); 
			dtype **v_data_gx = static_cast<dtype **>(gx_ptrs.get());
			cudaMemcpy(v_data_gx, (dtype**)vec_ptr_gx.data(), mem_size, cudaMemcpyHostToDevice);

			const int gx_size = n * vec_gx[0]->shape()[0];
			const int gy_size = gy.shape().size();

			Davgpooling_impl(CDATA(gy), gy_size, gx_size, n, v_data_gx);
		}

		void FSumPooling(const vector<LDG::PTensor>& vec_x, LDG::Tensor& y) {
			const int n = vec_x.size();
			for(int idx = 1; idx < n; idx++) {
				assert(vec_x[idx]->shape().size() == vec_x[0]->shape().size());
			}
			malloc(y, vec_x[0]->shape());

			const int r = y.shape().size();
			const int s = y.shape().lower_volume(1);

			int mem_size = sizeof(dtype*) * n;

			vector<const dtype*> vec_ptr_x(n);
			for(int i = 0; i < n; i++) {
				vec_ptr_x[i] = (CDATA(*vec_x[i]));
			}
			std::shared_ptr<void> x_ptrs = mem_pool->allocate(mem_size); 
			dtype **v_data_x = static_cast<dtype **>(x_ptrs.get());
			cudaMemcpy(v_data_x, (dtype**)vec_ptr_x.data(), mem_size, cudaMemcpyHostToDevice);


			Fsumpooling_impl(v_data_x, MDATA(y), n, r, s);
		}

		void DSumPooling(const LDG::Tensor& gy, vector<LDG::PTensor>& vec_gx) {
			const int n = vec_gx.size();

			int mem_size = sizeof(dtype*) * n;
			vector<const dtype*> vec_ptr_gx(n);
			for(int i = 0; i < n; i++) {
				vec_ptr_gx[i] = (CDATA(*vec_gx[i]));
			}
			std::shared_ptr<void> gx_ptrs = mem_pool->allocate(mem_size); 
			dtype **v_data_gx = static_cast<dtype **>(gx_ptrs.get());
			cudaMemcpy(v_data_gx, (dtype**)vec_ptr_gx.data(), mem_size, cudaMemcpyHostToDevice);

			const int gx_size = n * vec_gx[0]->shape()[0];
			const int gy_size = gy.shape().size();

			Dsumpooling_impl(CDATA(gy), gy_size, gx_size, v_data_gx);
		}
		
		void FMaxPooling(const vector<LDG::PTensor>& vec_x, LDG::Tensor& y, int* index) {
			const int n = vec_x.size();
			for(int idx = 1; idx < n; idx++) {
				assert(vec_x[idx]->shape().size() == vec_x[0]->shape().size());
			}
			malloc(y, vec_x[0]->shape());
			const int r = y.shape().size();
			const int s = y.shape().lower_volume(1);

			int mem_size = sizeof(dtype*) * n;

			vector<const dtype*> vec_ptr_x(n);
			for(int i = 0; i < n; i++) {
				vec_ptr_x[i] = (CDATA(*vec_x[i]));
			}
			std::shared_ptr<void> x_ptrs = mem_pool->allocate(mem_size); 
			dtype **v_data_x = static_cast<dtype **>(x_ptrs.get());
			cudaMemcpy(v_data_x, (dtype**)vec_ptr_x.data(), mem_size, cudaMemcpyHostToDevice);


			std::shared_ptr<void> index_ptrs = mem_pool->allocate(sizeof(int) * r); 
			int *gpu_index_data = static_cast<int *>(index_ptrs.get());

			Fmaxpooling_impl(v_data_x, MDATA(y), n, r, s, gpu_index_data);
			cudaMemcpy(index, gpu_index_data, sizeof(int) * r, cudaMemcpyDeviceToHost);
		}

		void DMaxPooling(const LDG::Tensor& gy, vector<LDG::PTensor>& vec_gx, int* index) {
			const int n = vec_gx.size();
			std::shared_ptr<void> gx_ptrs = mem_pool->allocate(sizeof(dtype*) * n); 
			dtype **v_data_gx = static_cast<dtype **>(gx_ptrs.get());

			for(int i = 0; i < n; i++) {
				dtype* tmp_ptr = MDATA(*vec_gx[i]);
				cudaMemcpy(v_data_gx + i, &(tmp_ptr), sizeof(dtype*) ,cudaMemcpyHostToDevice);
			}

			const int dim0 = gy.shape()[0];

			std::shared_ptr<void> index_ptrs = mem_pool->allocate(sizeof(int) * dim0); 
			int *gpu_index_data = static_cast<int *>(index_ptrs.get());

			cudaMemcpy(gpu_index_data, index, sizeof(int) * dim0, cudaMemcpyHostToDevice);
			Dmaxpooling_impl(CDATA(gy), v_data_gx, gpu_index_data, dim0);
		}

		void FMinPooling(const vector<LDG::PTensor>& vec_x, LDG::Tensor &y, int* index) {
			const int n = vec_x.size();
			for(int idx = 1; idx < n; idx++) {
				assert(vec_x[idx]->shape().size() == vec_x[0]->shape().size());
			}
			malloc(y, vec_x[0]->shape());

			const int r = y.shape().size();
			const int s = y.shape().lower_volume(1);

			int mem_size = sizeof(dtype*) * n;

			vector<const dtype*> vec_ptr_x(n);
			for(int i = 0; i < n; i++) {
				vec_ptr_x[i] = (CDATA(*vec_x[i]));
			}
			std::shared_ptr<void> x_ptrs = mem_pool->allocate(mem_size); 
			dtype **v_data_x = static_cast<dtype **>(x_ptrs.get());
			cudaMemcpy(v_data_x, (dtype**)vec_ptr_x.data(), mem_size, cudaMemcpyHostToDevice);

			std::shared_ptr<void> index_ptrs = mem_pool->allocate(sizeof(int) * r); 
			int *gpu_index_data = static_cast<int *>(index_ptrs.get());

			Fminpooling_impl(v_data_x, MDATA(y), n, r, s, gpu_index_data);
			cudaMemcpy(index, gpu_index_data, sizeof(int) * r, cudaMemcpyDeviceToHost);
		}

		void DMinPooling(const LDG::Tensor& gy, vector<LDG::PTensor>& vec_gx, int* index) {
			const int n = vec_gx.size();
			int mem_size = sizeof(dtype*) * n;

			vector<const dtype*> vec_ptr_gx(n);
			for(int i = 0; i < n; i++) {
				vec_ptr_gx[i] = (CDATA(*vec_gx[i]));
			}
			std::shared_ptr<void> gx_ptrs = mem_pool->allocate(mem_size); 
			dtype **v_data_gx = static_cast<dtype **>(gx_ptrs.get());
			cudaMemcpy(v_data_gx, (dtype**)vec_ptr_gx.data(), mem_size, cudaMemcpyHostToDevice);

			const int dim0 = gy.shape()[0];
			std::shared_ptr<void> index_ptrs = mem_pool->allocate(sizeof(int) * dim0); 
			int *gpu_index_data = static_cast<int *>(index_ptrs.get());

			cudaMemcpy(gpu_index_data, index, sizeof(int) * dim0, cudaMemcpyHostToDevice);
			Dminpooling_impl(CDATA(gy), v_data_gx, gpu_index_data, dim0);
		}

		void copy_tensor(const LDG::Tensor &src, LDG::Tensor& dst) {
			malloc(dst, src.shape());
			cudaMemcpyAsync(
					MDATA(dst),
				   	CDATA(src), 
					sizeof(dtype) * src.shape().size(), 
					cudaMemcpyDeviceToDevice, 0);
		}

		vector<dtype> to_vector(const LDG::Tensor& x) {
			const std::uint32_t size = x.shape().size();
			vector<dtype> ret(size);
			cudaMemcpy(
					ret.data(), 
					CDATA(x), 
					sizeof(dtype) * size, 
					cudaMemcpyDeviceToHost);
			return ret;
		}
};

#endif
