#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#pragma warning(disable:4996)
#define _USE_MATH_DEFINES
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <vector>
#include "CBuffer.cuh"

using namespace std;

int width, height;
unsigned long int imageSize = 0;
unsigned char* h_imgIn, * h_imgOut;

__global__ void  ImgProcKernel(unsigned char* d_imgIn, unsigned char* d_imgOut, int width, int height, int components)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = components * (row * width + col);
	float gray = 0.299f * (float)d_imgIn[offset] + 0.587f * (float)d_imgIn[offset + 1] + 0.144f * (float)d_imgIn[offset + 2];
	d_imgOut[offset] = (unsigned char)gray;
	d_imgOut[offset + 1] = (unsigned char)gray;
	d_imgOut[offset + 2] = (unsigned char)gray;
}


// Helper function for using CUDA 
cudaError_t ImgProcCUDA(unsigned char* h_imgIn, unsigned char* h_imgOut, int* width, int* height, int components) {
	unsigned char* d_imgIn = 0;
	unsigned char* d_imgOut = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!\nDo you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&d_imgIn, imageSize * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&d_imgOut, imageSize * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input image from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(d_imgIn, h_imgIn, imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU.
	const int TILE = 32;
	dim3 dimGrid(ceil((float)*width / TILE), ceil((float)*height / TILE));
	dim3 dimBlock(TILE, TILE, 1);
	ImgProcKernel << <dimGrid, dimBlock >> > (d_imgIn, d_imgOut, *width, *height, components);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output image from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(h_imgOut, d_imgOut, imageSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(d_imgIn);
	cudaFree(d_imgOut);

	return cudaStatus;
}


int main()
{
	//read the image first
	int components = 0;
	int requiredComponents = 3;
	//this loads the file, returns its resultion and number of componnents
	cout << "\nReading input image";
	h_imgIn = stbi_load("images/alebrije.png", &(width), &(height), &components, requiredComponents);
	if (!h_imgIn) {
		cout << "Cannot read input image, invalid path?" << endl;
		exit(-1);
	}
	imageSize = width * height * components;
	h_imgOut = (unsigned char*)malloc(imageSize * sizeof(unsigned char));
	if (h_imgOut == NULL) {
		cout << "malloc failed" << endl;
		exit(-1);
	}

	cout << "\nProcessing the image";
	// Run the memory copies, kernel, copy back from the helper
	cudaError_t cudaStatus = ImgProcCUDA(h_imgIn, h_imgOut, &width, &height, components);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Cuda krenel failed!");
		return 1;
	}
	cout << "\nSaving the image";
	//save the output image
	int result = stbi_write_png("images/result.png", width, height, components, h_imgOut, 0);
	if (!result) {
		cout << "Something went wrong during writing. Invalid path?" << endl;
		return 0;
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	cout << "\nDone\n";
	return 0;
}

