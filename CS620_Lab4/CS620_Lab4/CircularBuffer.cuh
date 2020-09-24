#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"

template<class T>
class CircularBuffer
{
public:
	unsigned int n;
	__host__ CircularBuffer() {}
	__host__ CircularBuffer(unsigned int size) {
		n = size;
		it_start = 0;
		it_end = 0;
	}
	__host__ ~CircularBuffer() {}
	__host__ void MallocBuffer() {
		if (cudaMallocManaged(&buffer, n * sizeof(T)) != cudaSuccess) {
			printf("\nbuffer cudaMallocManaged failed");
		}
	}
	__host__ void free() {
		if (cudaFree(buffer) != cudaSuccess) {
			printf("\nbuffer cudaFree failed");
		}
	}
	__host__ __device__ inline T& operator[](unsigned int i) { 
		return buffer[i % n];
	}

	__host__  inline T& end() { return buffer[(it_end-1)%n]; }
	__host__  inline T& begin() { return buffer[it_start%n]; }
	__host__
	inline void push(const T& data) {
		buffer[it_end % n] = data;	
		it_end++;
		if (it_end > it_start + n)
			it_start++;
	}
	__host__
	inline T& pop() {
		T& res = buffer[it_start%n];
		buffer[it_start%n] = T(NULL);
		if(it_start < it_end)
			it_start++;
		return res;
	}
	__host__  __device__ void print() {
		for (int i = 0; i < n; i++) {
			printf("Buffer[%i] = %i  ", i, int(buffer[i]));
		}
		printf("\n");
	}

private:
	unsigned int it_start, it_end;
	T* buffer;
};

