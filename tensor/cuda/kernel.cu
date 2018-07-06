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
__global__ void Fsumpooling_kernel(
		const dtype *px, int skip, int n, dtype *py) {
	__shared__ dtype temp[BLOCK_SIZE];
	const int bid = blockIdx.x;
	const int tid = threadIdx.x;
	px += bid % skip + (bid / skip) * skip * n;
	temp[tid] = 0;
	for (int i = tid; i < n; i += BLOCK_SIZE) temp[tid] += px[i * skip];
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

void Fsumpooling_impl(const dtype* x, dtype* y, int n, int r, int s) {
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
}


template<std::uint32_t BLOCK_SIZE>
__global__ void Fmaxpooling_kernel(
		const dtype *px, int skip, int n, dtype *py) {
	__shared__ dtype temp[BLOCK_SIZE];
	const int bid = blockIdx.x;
	const int tid = threadIdx.x;
	px += bid % skip + (bid / skip) * skip * n;
	dtype thread_max = NEGATIVE_INFINITY;
	for (int i = tid; i < n; i += BLOCK_SIZE) {
#if USE_FLOAT
		thread_max = fmaxf(px[i * skip], thread_max);
#else
		thread_max = fmax(px[i * skip], thread_max);
#endif

	}
	temp[tid] = thread_max;
	::__syncthreads();

#if USE_FLOAT

#define REDUCE(k) \
	if (BLOCK_SIZE >= k << 1) { \
		if (tid < k) temp[tid] = fmaxf(temp[tid + k], temp[tid]); \
		::__syncthreads(); \
	}
#else
#define REDUCE(k) \
	if (BLOCK_SIZE >= k << 1) { \
		if (tid < k) temp[tid] = fmax(temp[tid + k], temp[tid]); \
		::__syncthreads(); \
	}
#endif

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

void Fmaxpooling_impl(const dtype* x, dtype* y, int n, int r, int s) {
	int block_size = THREADS_PER_BLOCK;
	while (block_size >> 1 >= n) block_size >>= 1;
	switch (block_size) {
#define CASE(k) \
		case k: ::Fmaxpooling_kernel<k><<<r, k>>>(x, s, n, y); break
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

template<int BLOCK_SIZE>
__global__ void Dmaxpooling_kernel(
		const dtype *px, const dtype *py, const dtype *pgy,
		int skip, int n, dtype *pgx) {
	__shared__ int argmax_val[BLOCK_SIZE];
	const int bid = blockIdx.x;
	const int tid = threadIdx.x;
	const dtype max_val = py[bid];
	px += bid % skip + (bid / skip) * skip * n;
	pgx += bid % skip + (bid / skip) * skip * n;
	int thread_argmax = n;
	for (int i = tid; i < n; i += BLOCK_SIZE) {
		if (px[i * skip] == max_val) {
			thread_argmax = min(i, thread_argmax);
		}
	}
	argmax_val[tid] = thread_argmax;
	::__syncthreads();
#define REDUCE(k) \
	if (BLOCK_SIZE >= k << 1) { \
		if (tid < k) argmax_val[tid] = min(argmax_val[tid + k], argmax_val[tid]); \
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
		if (tid == 0) pgx[argmax_val[0] * skip] += pgy[bid];
}

void Dmaxpooling_impl(const dtype* x, const dtype* y, const dtype* gy, dtype* gx, int n, int r, int s) {
	int block_size = THREADS_PER_BLOCK;
	while (block_size >> 1 >= n) block_size >>= 1;
	switch (block_size) {
#define CASE(k) \
		case k: ::Dmaxpooling_kernel<k><<<r, k>>>(x, y, gy, s, n, gx); break
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

