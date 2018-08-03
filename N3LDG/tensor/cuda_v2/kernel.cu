#include "kernel.cuh"


#define THREADS_PER_BLOCK 1024

__global__ void Fadd_kernel(const dtype* x, const dtype* y, dtype* r, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		r[index] = x[index] + y[index];
	}
}

void Fadd_impl(const dtype* x, const dtype* y, dtype* r, int size) {
	Fadd_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, y, r, size);
	cudaDeviceSynchronize();
}

__global__ void Fadd_kernel(const dtype* x, dtype** y, dtype* r, int count, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		dtype sum = 0;
		int offset = 0;
		for(int idx = 0; idx < count; idx++) {	
			int global = index + offset;
			int idx = global / size;
			int idy = global % size;
			sum += (x[index] + y[idx][idy]);
			offset += size;
		}
		r[index] = sum;
	}
}

void Fadd_impl(const dtype* x, dtype** y, dtype* r, int count, int size) {
	Fadd_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, y, r, count, size);
	cudaDeviceSynchronize();
}

__global__ void Fadd_kernel(dtype** x, dtype** y, dtype** r, int dim0, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		int idx = index / dim0;
		int idy = index % dim0;
		r[idx][idy] = x[idx][idy] + y[idx][idy];
	}
}

void Fadd_impl(dtype** x, dtype** y,  dtype** r, int dim0, int size) {
	Fadd_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, y, r, dim0, size);
	cudaDeviceSynchronize();
}

__global__ void Fsubtract_kernel(const dtype* x, const dtype* y, dtype* r, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		r[index] = x[index] - y[index];
	}
}

void Fsubtract_impl(const dtype* x, const dtype* y, dtype* r, int size) {
	Fsubtract_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, y, r, size);
	cudaDeviceSynchronize();
}

__global__ void Fmultiply_kernel(const dtype* x, const dtype* y, dtype* r, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		r[index] = x[index] * y[index];
	}
}

void Fmultiply_impl(const dtype* x, const dtype* y, dtype* r, int size) {
	Fmultiply_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, y, r, size);
	cudaDeviceSynchronize();
}

__global__ void Fmultiply_kernel(dtype** x, dtype** y, dtype** r, int dim0, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		int idx = index / dim0;
		int idy = index % dim0;
		r[idx][idy] = x[idx][idy] * y[idx][idy];
	}
}

void Fmultiply_impl(dtype** x, dtype** y, dtype** r, int dim0, int size) {
	Fmultiply_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, y, r, dim0, size);
	cudaDeviceSynchronize();
}

__global__ void Fdivide_kernel(const dtype* x, const dtype* y, dtype* r, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		r[index] = x[index] / y[index];
	}
}

void Fdivide_impl(const dtype* x, const dtype* y, dtype* r, int size) {
	Fdivide_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, y, r, size);
	cudaDeviceSynchronize();
}

__global__ void Fmultiply_scalar_kernel(const dtype* x, const dtype y, dtype* r, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		r[index] = x[index] * y;
	}
}

void Fmultiply_scalar_impl(const dtype* x, const dtype y, dtype* r, int size) {
	Fmultiply_scalar_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, y, r, size);
	cudaDeviceSynchronize();
}

__global__ void Fadd_scalar_kernel(const dtype* x, const dtype y, dtype* r, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		r[index] = x[index] + y;
	}
}

void Fadd_scalar_impl(const dtype* x, const dtype y, dtype* r, int size) {
	Fadd_scalar_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, y, r, size);
	cudaDeviceSynchronize();
}

__global__ void Fsquare_kernel(const dtype* x, dtype* r, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		r[index] = x[index] * x[index];
	}
}

void Fsquare_impl(const dtype* x, dtype* r, int size) {
	Fsquare_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, r, size);
	cudaDeviceSynchronize();
}

__global__ void Ftanh_kernel(const dtype* x, dtype* r, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		r[index] = tanh(x[index]);
	}
}

void Ftanh_impl(const dtype* x, dtype* r, int size) {
	Ftanh_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, r, size);
	cudaDeviceSynchronize();
}

__global__ void Ftanh_kernel(dtype** x, dtype** r, int dim0, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		int idx = index / dim0;
		int idy = index % dim0;
		r[idx][idy] = tanh(x[idx][idy]);
	}
}

void Ftanh_impl(dtype** x, dtype** r, int dim0, int size) {
	Ftanh_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, r, dim0, size);
	cudaDeviceSynchronize();
}

__global__ void Fsigmoid_kernel(dtype** x, dtype** r, int dim0, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		int idx = index / dim0;
		int idy = index % dim0;
		r[idx][idy] = 1.0 / (1.0 + exp(-x[idx][idy]));
	}
}

void Fsigmoid_impl(dtype** x, dtype** r, int dim0, int size) {
	Fsigmoid_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, r, dim0, size);
	cudaDeviceSynchronize();
}

__global__ void Dsigmoid_kernel(dtype** x, dtype** y, dtype** r, int dim0, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		int idx = index / dim0;
		int idy = index % dim0;
		r[idx][idy] = (1 - y[idx][idy]) * y[idx][idy];
	}
}

void Dsigmoid_impl(dtype** x, dtype** y, dtype** r, int dim0, int size){
	Dsigmoid_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, y, r, dim0, size);
	cudaDeviceSynchronize();
}


__global__ void Dtanh_kernel(const dtype* x, const dtype* y, dtype* r, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		r[index] = (1 + y[index]) * (1 - y[index]);
	}
}

void Dtanh_impl(const dtype* x, const dtype* y, dtype* r, int size) {
	Dtanh_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, y, r, size);
	cudaDeviceSynchronize();
}

__global__ void Dtanh_kernel(dtype** x, dtype** y, dtype** r, int dim0, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		int idx = index / dim0;
		int idy = index % dim0;
		r[idx][idy] = (1 + y[idx][idy]) * (1 - y[idx][idy]);
	}
}

void Dtanh_impl(dtype** x, dtype** y, dtype** r, int dim0, int size){
	Dtanh_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, y, r, dim0, size);
	cudaDeviceSynchronize();
}

__global__ void Fsigmoid_kernel(const dtype* x, dtype* r, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		r[index] = 1.0 / (1.0 + exp(-x[index]));
	}
}

void Fsigmoid_impl(const dtype* x, dtype* r, int size) {
	Fsigmoid_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, r, size);
	cudaDeviceSynchronize();
}

__global__ void Dsigmoid_kernel(const dtype* x, const dtype* y, dtype* r, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		r[index] = (1 - y[index]) * y[index];
	}
}

void Dsigmoid_impl(const dtype* x, const dtype* y, dtype* r, int size) {
	Dsigmoid_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, y, r, size);
	cudaDeviceSynchronize();
}

__global__ void Fsqrt_kernel(const dtype* x, dtype* r, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		r[index] = sqrt(x[index]);
	}
}

void Fsqrt_impl(const dtype* x, dtype* r, int size) {
	Fsqrt_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, r, size);
	cudaDeviceSynchronize();
}

__global__ void concat_kernel(const dtype *src, dtype* dst, int offset, int dim) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < dim) {
		dst[offset + index] = src[index];
	}
}

void concat_impl(const dtype *src, dtype* dst, int offset, int dim) {
	concat_kernel<<<(dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(src, dst, offset, dim);
	cudaDeviceSynchronize();
}

__global__ void unconcat_kernel(const dtype *src, dtype* dst, int offset, int dim) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < dim) {
		dst[index] = src[offset + index];
	}
}

void unconcat_impl(const dtype *src, dtype* dst, int offset, int dim) {
	unconcat_kernel<<<(dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(src, dst, offset, dim);
	cudaDeviceSynchronize();
}

__global__ void Ftranspose_kernel(const dtype* x, dtype* r, int dim0, int dim1, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		r[index] = x[index % dim0 * dim1 + index / dim0];
	}
}

void Ftranspose_impl(const dtype* x, dtype* r, int dim0, int dim1, int size) {
	Ftranspose_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, r, dim0, dim1, size);
	cudaDeviceSynchronize();
}

__global__ void set_col_kernel(dtype* x, int dim0, int col, int size, dtype val) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int i = index + col * dim0;
	if (i < size && index < dim0) {
		x[i] = val;
	}
}

void set_col_impl(dtype* x, int dim0, int col, int size, dtype val) {
	set_col_kernel<<<(dim0 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, dim0, col, size, val);
	cudaDeviceSynchronize();
}


__global__ void FLookup_kernel(const dtype* x, dtype** r, int xdim0, int xdim1, int r_size, int* cols, int col_num) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < r_size) {
		int col_index = index / xdim0;
		if(col_index < col_num) {
			int col = cols[col_index];
			int offset = index % xdim0;
			int x_index = col * xdim0 + offset;
			if(x_index < xdim0 * xdim1) {
			   	r[col_index][offset] = x[x_index];
			}
		}
	}
}

void FLookup_impl(const dtype* x, dtype** r, int xdim0, int xdim1, int r_size, int* cols, int col_num) {
	FLookup_kernel<<<(r_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>> 
		(x, r, xdim0, xdim1, r_size, cols, col_num);	
}

__global__ void DLookup_kernel(dtype* gx, dtype** loss, int gxdim0, int gxdim1, int l_size, int* cols, int col_num) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < l_size) {
		int col_index = index / gxdim0;
		if(col_index < col_num) {
			int col = cols[col_index];
			int offset = index % gxdim0;
			int gx_index = col * gxdim0 + offset;
			if(gx_index < gxdim0 * gxdim1) {
				atomicAdd(gx + gx_index, loss[col_index][offset]);
			   	//gx[gx_index] += loss[col_index][offset];
			}
		}
	}
}

void DLookup_impl(dtype* gx, dtype** loss, int gxdim0, int gxdim1, int l_size, int* cols, int col_num) {
	DLookup_kernel<<<(l_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>> 
		(gx, loss, gxdim0, gxdim1, l_size, cols, col_num);	
}

__global__ void get_cols_kernel(const dtype* x, dtype* r, int xdim0, int xdim1, int r_size, int* cols, int col_num) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < r_size) {
		int col_index = index / xdim0;
		if(col_index < col_num) {
			int col = cols[col_index];
			int offset = index % xdim0;
			int x_index = col * xdim0 + offset;
			if(x_index < xdim0 * xdim1) {
			   	r[index] = x[x_index];
			}
		}
	}
}

void get_cols_impl(const dtype* x, dtype* r, int xdim0, int xdim1, int r_size, int* cols, int col_num) {
	get_cols_kernel<<<(r_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>> 
		(x, r, xdim0, xdim1, r_size, cols, col_num);	
}

__global__ void get_col_kernel(const dtype* x, dtype* r, int dim0, int col, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int i = index + col * dim0;
	if (i < size && index < dim0) {
		r[index] = x[i];
	}
}

void get_col_impl(const dtype* x, dtype* r, int dim0, int col, int size) {
	get_col_kernel<<<(dim0 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, r, dim0, col, size);
	cudaDeviceSynchronize();
}

__global__ void Fadd_col_kernel(dtype* x, const dtype* y, int col, int dim0, int size){
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int i = index + col * dim0;
	if (i < size && index < dim0) {
		x[i] = x[i] + y[index];
	}
}

void Fadd_col_impl(dtype* x, const dtype* y, int col, int dim0, int size) {
	Fadd_col_kernel<<<(dim0 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, y, col, dim0, size);
	cudaDeviceSynchronize();
}


template<int BLOCK_SIZE>
__global__ void Favgpooling_kernel(
		dtype **px, int skip, int n, dtype *py) {
	__shared__ dtype temp[BLOCK_SIZE];
	const int bid = blockIdx.x;
	const int tid = threadIdx.x;
	//px += bid % skip + (bid / skip) * skip * n;
	int index_start = bid % skip + (bid / skip) * skip * n;
	temp[tid] = 0;
	for (int i = tid; i < n; i += BLOCK_SIZE) {
		int global = index_start + i * skip;
		int idx = global / skip;
		int idy = global % skip; 
		temp[tid] += px[idx][idy];
	}
	::__syncthreads();
#define REDUCE(k) \
	if (BLOCK_SIZE >= k << 1) { \
		if (tid < k) temp[tid] += temp[tid + k]; \
		::__syncthreads(); \
	}
	REDUCE(512)
		REDUCE(256)
		REDUCE(128)
		REDUCE(64)
		REDUCE(32)
		REDUCE(16)
		REDUCE(8)
		REDUCE(4)
		REDUCE(2)
		REDUCE(1)
#undef REDUCE
		if (tid == 0) py[bid] = temp[0] / n;
}

void Favgpooling_impl(dtype** x, dtype* y, int n, int r, int s) {
	int block_size = THREADS_PER_BLOCK;
	while (block_size >> 1 >= n) block_size >>= 1;
	switch (block_size) {
#define CASE(k) \
		case k: ::Favgpooling_kernel<k><<<r, k>>>(x, s, n, y); break
				CASE(1024);
				CASE(512);
				CASE(256);
				CASE(128);
				CASE(64);
				CASE(32);
				CASE(16);
				CASE(8);
				CASE(4);
				CASE(2);
				CASE(1);
#undef CASE
	}
	cudaDeviceSynchronize();
}

__global__ void Davgpooling_kernel(const dtype* gy, int gy_size, int gx_size, int n, dtype** gx) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i < gx_size) {
		int idx = i / gy_size;
		int idy = i % gy_size;
		atomicAdd(gx[idx] + idy, gy[idy] / n);
		//gx[idx][idy] += (gy[idy] / n);
	}
}

void Davgpooling_impl(const dtype* gy, int gy_size, int gx_size, int n, dtype** gx) {
	Davgpooling_kernel<<<(gx_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(gy, gy_size, gx_size, n, gx);
	cudaDeviceSynchronize();
}

template<int BLOCK_SIZE>
__global__ void Fsumpooling_kernel(
		dtype **px, int skip, int n, dtype *py) {
	__shared__ dtype temp[BLOCK_SIZE];
	const int bid = blockIdx.x;
	const int tid = threadIdx.x;
	//px += bid % skip + (bid / skip) * skip * n;
	int index_start = bid % skip + (bid / skip) * skip * n;
	temp[tid] = 0;
	for (int i = tid; i < n; i += BLOCK_SIZE) {
		int global = index_start + i * skip;
		int idx = global / skip;
		int idy = global % skip; 
		dtype val = px[idx][idy];	
		temp[tid] += val;
	}
	::__syncthreads();
#define REDUCE(k) \
	if (BLOCK_SIZE >= k << 1) { \
		if (tid < k) temp[tid] += temp[tid + k]; \
		::__syncthreads(); \
	}
	REDUCE(512)
		REDUCE(256)
		REDUCE(128)
		REDUCE(64)
		REDUCE(32)
		REDUCE(16)
		REDUCE(8)
		REDUCE(4)
		REDUCE(2)
		REDUCE(1)
#undef REDUCE
		if (tid == 0) py[bid] = temp[0];
}

void Fsumpooling_impl(dtype** x, dtype* y, int n, int r, int s) {
	int block_size = THREADS_PER_BLOCK;
	while (block_size >> 1 >= n) block_size >>= 1;
	switch (block_size) {
#define CASE(k) \
		case k: ::Fsumpooling_kernel<k><<<r, k>>>(x, s, n, y); break
				CASE(1024);
				CASE(512);
				CASE(256);
				CASE(128);
				CASE(64);
				CASE(32);
				CASE(16);
				CASE(8);
				CASE(4);
				CASE(2);
				CASE(1);
#undef CASE
	}
	cudaDeviceSynchronize();
}

__global__ void Dsumpooling_kernel(const dtype* gy, int gy_size, int gx_size, dtype** gx) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i < gx_size) {
		int idx = i / gy_size;
		int idy = i % gy_size;
		atomicAdd(gx[idx] + idy, gy[idy]);
		//gx[idx][idy] += gy[idy];
	}
}

void Dsumpooling_impl(const dtype* gy, int gy_size, int gx_size, dtype** gx) {
	Dsumpooling_kernel<<<(gx_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(gy, gy_size, gx_size, gx);
	cudaDeviceSynchronize();
}

template<int BLOCK_SIZE>
__global__ void Fmaxpooling_kernel(
		dtype **px, int skip, int n, dtype *py, int* index) {
	__shared__ dtype temp[BLOCK_SIZE];
	__shared__ int temp_index[BLOCK_SIZE];
	const int bid = blockIdx.x;
	const int tid = threadIdx.x;
	//px += bid % skip + (bid / skip) * skip * n;
	dtype thread_max = NEGATIVE_INFINITY;
	int index_start = bid % skip + (bid / skip) * skip * n;
	int index_max;
	for (int i = tid; i < n; i += BLOCK_SIZE) {
		int global = index_start + i * skip;
		int idx = global / skip;
		int idy = global % skip; 
		dtype val = px[idx][idy];	
		if(val > thread_max) {
			thread_max = val;
			index_max = index_start + i * skip;
		}
	}
	temp[tid] = thread_max;
	temp_index[tid] = index_max;
	::__syncthreads();
#define REDUCE(k) \
	if (BLOCK_SIZE >= k << 1) { \
		if (tid < k) if(temp[tid + k] > temp[tid]) {temp[tid] = temp[tid + k]; temp_index[tid] = temp_index[tid + k];} \
		::__syncthreads(); \
	}
	REDUCE(512)
		REDUCE(256)
		REDUCE(128)
		REDUCE(64)
		REDUCE(32)
		REDUCE(16)
		REDUCE(8)
		REDUCE(4)
		REDUCE(2)
		REDUCE(1)
#undef REDUCE
		if (tid == 0) {py[bid] = temp[0]; index[bid] = temp_index[0];}
}

void Fmaxpooling_impl(dtype** x, dtype* y, int n, int r, int s, int* index){
	int block_size = THREADS_PER_BLOCK;
	while (block_size >> 1 >= n) block_size >>= 1;
	switch (block_size) {
#define CASE(k) \
		case k: ::Fmaxpooling_kernel<k><<<r, k>>>(x, s, n, y, index); break
				CASE(1024);
				CASE(512);
				CASE(256);
				CASE(128);
				CASE(64);
				CASE(32);
				CASE(16);
				CASE(8);
				CASE(4);
				CASE(2);
				CASE(1);
#undef CASE
	}
	cudaDeviceSynchronize();
}

__global__ void Dmaxpooling_kernel(const dtype* gy, dtype** gx, int* index, int dim) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i < dim) {
		int idx = index[i] / dim;
		int idy = index[i] % dim;

		atomicAdd(gx[idx] + idy, gy[i]);
		//gx[idx][idy] += gy[i];
	}
}

void Dmaxpooling_impl(const dtype* gy, dtype** gx, int* index, int dim) {
	Dmaxpooling_kernel<<<(dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(gy, gx, index, dim);
	cudaDeviceSynchronize();
}

template<int BLOCK_SIZE>
__global__ void Fminpooling_kernel(
		dtype **px, int skip, int n, dtype *py, int* index) {
	__shared__ dtype temp[BLOCK_SIZE];
	__shared__ int temp_index[BLOCK_SIZE];
	const int bid = blockIdx.x;
	const int tid = threadIdx.x;
	//px += bid % skip + (bid / skip) * skip * n;
	dtype thread_min = POSITIVE_INFINITY;
	int index_start = bid % skip + (bid / skip) * skip * n;
	int index_min;
	for (int i = tid; i < n; i += BLOCK_SIZE) {
		int global = index_start + i * skip;
		int idx = global / skip;
		int idy = global % skip; 
		dtype val = px[idx][idy];	
		if(val <  thread_min) {
			thread_min = val;
			index_min = index_start + i * skip;
		}
	}
	temp[tid] = thread_min;
	temp_index[tid] = index_min;
	::__syncthreads();
#define REDUCE(k) \
	if (BLOCK_SIZE >= k << 1) { \
		if (tid < k) if(temp[tid + k] < temp[tid]) {temp[tid] = temp[tid + k]; temp_index[tid] = temp_index[tid + k];} \
		::__syncthreads(); \
	}
	REDUCE(512)
		REDUCE(256)
		REDUCE(128)
		REDUCE(64)
		REDUCE(32)
		REDUCE(16)
		REDUCE(8)
		REDUCE(4)
		REDUCE(2)
		REDUCE(1)
#undef REDUCE
		if (tid == 0) {py[bid] = temp[0]; index[bid] = temp_index[0];}
}

void Fminpooling_impl(dtype** x, dtype* y, int n, int r, int s, int* index) {
	int block_size = THREADS_PER_BLOCK;
	while (block_size >> 1 >= n) block_size >>= 1;
	switch (block_size) {
#define CASE(k) \
		case k: ::Fminpooling_kernel<k><<<r, k>>>(x, s, n, y, index); break
				CASE(1024);
				CASE(512);
				CASE(256);
				CASE(128);
				CASE(64);
				CASE(32);
				CASE(16);
				CASE(8);
				CASE(4);
				CASE(2);
				CASE(1);
#undef CASE
	}
}

__global__ void Dminpooling_kernel(const dtype* gy, dtype** gx, int* index, int dim) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i < dim) {
		int idx = index[i] / dim;
		int idy = index[i] % dim;
		atomicAdd(gx[idx] + idy, gy[i]);
		//gx[idx][idy] += gy[i];
	}
}

void Dminpooling_impl(const dtype* gy, dtype** gx, int* index, int dim) {
	Dminpooling_kernel<<<(dim + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(gy, gx, index, dim);
	cudaDeviceSynchronize();
}
