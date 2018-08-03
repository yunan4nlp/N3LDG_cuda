#ifndef CUDA_MEMPOOL_H
#define CUDA_MEMPOOL_H

#include "memory_pool.h"

static void *my_alloc(std::size_t size) {
	void *ptr;
	cudaMalloc((void **)&ptr, size);
	return ptr;
}

static void my_delete(void *ptr) {
	cudaFree(ptr);
}

extern MemoryPool CUDA_MEM_POOL(my_alloc, my_delete);

#endif
