#include "CircularBuffer.cuh"
#include <iostream>
#include <string>
#include <time.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#define PI 3.1415926
using namespace std;
const unsigned int n = 4, total_frames = 50;
enum DecayFunction {
	Rect, Tri, Gauss
};

__global__
void MotionBlur(CircularBuffer<unsigned char*> images, unsigned char* d_imgout,  int w, int h, int index, float* decay) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (col >= w || row >= h)
		return;
	int offset = 3 * (row * w + col);
	d_imgout[offset] = 0; d_imgout[offset + 1] = 0; d_imgout[offset + 2] = 0;
	if (index >= n - 1) {
		for (int i = 0; i < n; i++) {
			const unsigned char* d_imagein = images[index - i];
			d_imgout[offset] += d_imagein[offset] * decay[i];
			d_imgout[offset + 1] += d_imagein[offset + 1] * decay[i];
			d_imgout[offset + 2] += d_imagein[offset + 2] * decay[i];
		}
	}
	else {
		const unsigned char* d_imagein = images[index];
		d_imgout[offset] = d_imagein[offset];
		d_imgout[offset + 1] = d_imagein[offset + 1];
		d_imgout[offset + 2] = d_imagein[offset + 2];
	}
}

void GenerateDecayFunction(float* decay, DecayFunction func) {
	const float sigma = 1.f;
	switch(func){
	case Rect:
		for (int i = 0; i < n; i++)
			decay[i] = 1.f / float(n);
		break;
	case Tri:
		for (int i = 0; i < n; i++)
			decay[i] = (1.f - float(i) / float(n)) / float(n + 1) * 2.f;
		break;
	case Gauss:
		float sum = 0.f;
		for (int i = 0; i < n; i++) {
			decay[i] = 1.f / (sigma * sqrt(2.f * PI)) * exp(-(i * i) / (2.f * sigma * sigma));
			sum += decay[i];
		}
		for (int i = 0; i < n; i++) {
			decay[i] /= sum;
		}
		break;
	}
}
string IntTo4Digits(int i) {
	string res = "0000", num = to_string(i);
	const int offset = res.size() - num.size();
	for (int j = 0; j < num.size(); j++) {
		res[j + offset] = num[j];
	}
	return res;
}

int main() {
	float* decay;
	cudaMallocManaged(&decay, n * sizeof(float));
	GenerateDecayFunction(decay, Tri);
	CircularBuffer<unsigned char*> ImageBuffer(n);
	ImageBuffer.MallocBuffer();
	int components = 0, requiredComponents = 3, w, h;
	stbi_load("images/originals/stickman0000.bmp", &w, &h, &components, requiredComponents);
	const unsigned int imageSize = w * h * 3;
	unsigned char* h_imgin, *h_imgout, *d_imgin, *d_imgout;
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc((void**)&d_imgout, imageSize * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "d_imgout cudaMalloc failed!");
		goto Error;
	}
	clock_t t;
	t = clock();
	//processing image
	for (int i = 0; i < total_frames; i++) {
		if (i >= n) {
			unsigned char* d_imgpop = ImageBuffer.pop();
			cudaFree(d_imgpop);
		}
		//Read Image
		string file = "images/originals/stickman" + IntTo4Digits(i) + ".bmp";
		h_imgin = stbi_load(file.c_str(), &w, &h, &components, requiredComponents);
		if (!h_imgin) {
			cout << "Cannot read image" + file << endl;
			goto Error;
		}
		
		cudaStatus = cudaMalloc((void**)&d_imgin, imageSize * sizeof(unsigned char));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "d_img cudaMalloc failed!");
			goto Error;
		}
		cudaStatus = cudaMemcpy(d_imgin, h_imgin, imageSize * sizeof(unsigned char), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "d_img to h_img cudaMemcpy failed!");
			goto Error;
		}
		ImageBuffer.push(d_imgin);

		//Process Image
		const int TILE = 16;
		dim3 dimGrid(w / TILE + 1, h / TILE + 1);
		dim3 dimBlock(TILE, TILE, 1);
		MotionBlur << <dimGrid, dimBlock >> > (ImageBuffer, d_imgout,  w, h, i, decay);
		
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

		//Save image
		h_imgout = new unsigned char[imageSize];
		cudaStatus = cudaMemcpy(h_imgout, d_imgout, imageSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		file = "images/result" + IntTo4Digits(i) + ".bmp";
		int result = stbi_write_bmp(file.c_str(), w, h, components, h_imgout);
		if (!result) {
			cout << "Something went wrong during writing. Invalid path?" << endl;
			goto Error;
		}
		cout << "Save Image" + file << endl;
		delete[] h_imgin, h_imgout;

	}

	t = clock() - t;
	printf("time used: %f", t / (float) CLOCKS_PER_SEC);
Error:
	for (int i = 0; i < n; i++)
		cudaFree(ImageBuffer[i]);
	cudaFree(d_imgout);
	ImageBuffer.free();
	cudaFree(decay);

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	cout << "\nDone\n";
	return 0;
}