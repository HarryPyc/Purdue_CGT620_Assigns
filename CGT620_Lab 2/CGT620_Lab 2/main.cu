#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cstdio>

__global__ void myKernel() {
	printf("Hello World from %i of %i and thread %i\n",
		blockIdx.x, blockDim.x, threadIdx.x);
}
int main(int argc, char** argv) {
	myKernel <<<16, 16>>> ();
	return 0;
}