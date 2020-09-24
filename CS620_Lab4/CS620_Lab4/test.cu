#include "CircularBuffer.cuh"
#include <iostream>
#include <string>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

using namespace std;

__global__
void MyKernel(CircularBuffer<int> buffer) {
	const int index = blockDim.x * blockIdx.x + threadIdx.x;
	//printf("hello from thread %i\n", index);
	return;
}



int main() {
	CircularBuffer<int> CBuffer(5);
	CBuffer.MallocBuffer();
	for (int i = 0; i < 10; i++) {
		CBuffer.push(i);
		MyKernel << <8, 8 >> > (CBuffer);
		cudaDeviceSynchronize();
		printf("\nDone\n");
	}
	return 0;
}