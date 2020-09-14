
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define EXPLICIT_MEMORY_CONTROL

const int n = 1e5;
void FillArray(int *a, int *b) {
	for (int i = 0; i < n; i++)
		a[i] = i, b[i] = i*10;
}
__global__ void ArrayAdd(const int* d_a, const int* d_b, int* d_c) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		d_c[i] = d_a[i] + d_b[i];
		//printf("a[%i] = %i, b[%i] = %i, c[%i] = %i\n", i, d_a[i], i, d_b[i], i, d_c[i]);
	}
}

int main() {
	int *h_a = new int[n], *h_b = new int[n], *h_c = new int[n];
	int* d_a, * d_b, * d_c;
	cudaEvent_t startT, stopT;
	float time;
	cudaEventCreate(&startT);
	cudaEventCreate(&stopT);
	cudaEventRecord(startT, 0);
#ifdef EXPLICIT_MEMORY_CONTROL
	FillArray(h_a, h_b);
	cudaMalloc((void**)&d_a, n * sizeof(int));
	cudaMalloc((void**)&d_b, n * sizeof(int));
	cudaMalloc((void**)&d_c, n * sizeof(int));
	cudaMemcpy(d_a, h_a, n * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, n * sizeof(int), cudaMemcpyHostToDevice);
#else
	cudaMallocManaged(&d_a, n * sizeof(int));
	cudaMallocManaged(&d_b, n * sizeof(int));
	cudaMallocManaged(&d_c, n * sizeof(int));
	FillArray(d_a, d_b);
#endif
	ArrayAdd <<<n/64+1, 64>>> (d_a, d_b, d_c);
	if (cudaGetLastError() != cudaSuccess)
		printf("Launch Kernal Failed\n");

	cudaEventRecord(stopT, 0);
	cudaEventSynchronize(stopT);
	cudaEventElapsedTime(&time, startT, stopT);
	cudaEventDestroy(startT);
	cudaEventDestroy(stopT);
	printf("time used: %f miliseconds", time);
	delete[] h_a, h_b, h_c;
	cudaFree(d_c);
	cudaFree(d_a);
	cudaFree(d_b);
	return 0;
}