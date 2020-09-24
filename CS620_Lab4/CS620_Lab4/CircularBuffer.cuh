#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"

template<class T>
class CircularBuffer
{
public:
	unsigned int n;

	__host__ CircularBuffer(unsigned int size) {
		n = size;
		buffer = new T[n];
		cudaError_t e = cudaMalloc((void**)&d_buffer, n * sizeof(T));
		if (e != cudaSuccess) {
			printf("d_buffer cudaMalloc failed");
		}
		it_start = 0;
		it_end = 0;
	}
	__host__ ~CircularBuffer() { freeDeviceBuffer(); delete[] buffer; }

	__host__ __device__ inline T& operator[](unsigned int i) { 
		
#ifdef __CUDA_ARCH__
		return d_buffer[i % n];
#else
		return buffer[i % n];
#endif
	}

	__host__  inline T& end() { return buffer[(it_end-1)%n]; }
	__host__  inline T& begin() { return buffer[it_start%n]; }
	__host__
	inline void push(const T& data) {
		buffer[it_end%n] = data;
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
	__host__  void print() {
		for (int i = 0; i < n; i++) {
			printf("Buffer[%i] = %i  ", i, int(buffer[i]));
		}
		printf("\n");
	}
	__host__ void uploadToDevice() {
		cudaError_t e = cudaMemcpy(d_buffer, buffer, n*sizeof(T), cudaMemcpyHostToDevice);
		if (e != cudaSuccess) {
			printf("buffer to d_buffer cudaMemcpy failed: %s" , cudaGetErrorString(e));
		}
	}
	__host__ void freeDeviceBuffer() {
		cudaError_t e = cudaFree(d_buffer);
		if (e != cudaSuccess) {
			printf("\nd_buffer cudaFree failed");
		}
	}
private:
	unsigned int it_start, it_end;
	T* buffer, *d_buffer;
};

