#include "kernel.cuh"


#define THREADS_PER_BLOCK 512

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
}

__global__ void Fadd_scalar_kernel(const dtype* x, const dtype y, dtype* r, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		r[index] = x[index] + y;
	}
}

void Fadd_scalar_impl(const dtype* x, const dtype y, dtype* r, int size) {
	Fadd_scalar_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, y, r, size);
}

__global__ void Fsquare_kernel(const dtype* x, dtype* r, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		r[index] = x[index] * x[index];
	}
}

void Fsquare_impl(const dtype* x, dtype* r, int size) {
	Fsquare_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, r, size);
}

__global__ void Ftanh_kernel(const dtype* x, dtype* r, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		r[index] = tanh(x[index]);
	}
}

void Ftanh_impl(const dtype* x, dtype* r, int size) {
	Ftanh_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, r, size);
}

__global__ void Dtanh_kernel(const dtype* x, const dtype* y, dtype* r, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		r[index] = (1 + y[index]) * (1 - y[index]);
	}
}

void Dtanh_impl(const dtype* x, const dtype* y, dtype* r, int size) {
	Dtanh_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, y, r, size);
}


__global__ void Fsqrt_kernel(const dtype* x, dtype* r, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size) {
		r[index] = sqrt(x[index]);
	}
}

void Fsqrt_impl(const dtype* x, dtype* r, int size) {
	Fsqrt_kernel<<<(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, r, size);
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
}

__global__ void set_col_kernel(dtype* x, int dim0, int col, int size, dtype val) {
	    int index = threadIdx.x + blockIdx.x * blockDim.x;
		    int i = col + index * dim0;
			    if (i < size) {
					        //printf("i: %d, col: %d, index: %d, dim0 %d\n", i, col, index, dim0);
					        x[i] = val;
							    }
}

void set_col_impl(dtype* x, int dim0, int dim1, int col, int size, dtype val) {
	    set_col_kernel<<<(dim1 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, dim0, col, size, val);
}


__global__ void set_row_kernel(dtype* x, int dim0, int row, int size, dtype val) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int i = index + row * dim0;
	if (i < size && index < dim0) {
		x[i] = val;
	}
}

void set_row_impl(dtype* x, int dim0, int row, int size, dtype val) {
	set_row_kernel<<<(dim0 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, dim0, row, size, val);
}

__global__ void get_row_kernel(const dtype* x, dtype* r, int dim0, int row, int size) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int i = index + row * dim0;
	if (i < size && index < dim0) {
		r[index] = x[i];
	}
}

void get_row_impl(const dtype* x, dtype* r, int dim0, int row, int size) {
	get_row_kernel<<<(dim0 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(x, r, dim0, row, size);
}

