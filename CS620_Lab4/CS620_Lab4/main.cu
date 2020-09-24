#include "CircularBuffer.cuh"
#include <iostream>
#include <string>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

using namespace std;

__global__
void MotionBlur(CircularBuffer<unsigned char*> images, unsigned char* d_imgout, unsigned char* acc_buffer, int w, int h, int index) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (col >= w || row >= h)
		return;
	int offset = 3 * (row * w + col);
	const unsigned char* d_imagein = images[index];
	d_imgout[offset] = d_imagein[offset];
	d_imgout[offset+1] = d_imagein[offset+1];
	d_imgout[offset+2] = d_imagein[offset+2];
}



int main() {
	//how many images should be accumulated
	const unsigned int n = 5, total_frames = 39;
	CircularBuffer<unsigned char*> ImageBuffer(n);
	int components = 0, requiredComponents = 3, w, h;
	stbi_load("images/originals/teapot0.bmp", &w, &h, &components, requiredComponents);
	const unsigned int imageSize = w * h * 3;
	unsigned char* h_imgin, *h_imgout, *d_imgin, *d_imgout, *acc_buffer;
	cudaError_t cudaStatus;
	cudaStatus = cudaMalloc((void**)&d_imgout, imageSize * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "d_imgout cudaMalloc failed!");
		goto Error;
	}
	cudaStatus = cudaMalloc((void**)&acc_buffer, imageSize * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "acc_buffer cudaMalloc failed!");
		goto Error;
	}
	//processing image
	for (int i = 0; i < total_frames; i++) {
		/*if (i >= n) {
			unsigned char* d_imgpop = ImageBuffer.pop();
			cudaFree(d_imgpop);
		}*/
		//Read Image
		string file = "images/originals/teapot" + to_string(i) + ".bmp";
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
		ImageBuffer.uploadToDevice();

		//Process Image
		const int TILE = 16;
		dim3 dimGrid(w / TILE + 1, h / TILE + 1);
		dim3 dimBlock(TILE, TILE, 1);
		MotionBlur << <dimGrid, dimBlock >> > (ImageBuffer, d_imgout, acc_buffer, w, h, i);
		
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

		file = "images/result" + to_string(i) + ".bmp";
		int result = stbi_write_bmp(file.c_str(), w, h, components, h_imgout);
		if (!result) {
			cout << "Something went wrong during writing. Invalid path?" << endl;
			goto Error;
		}
		cout << "Save Image" + file << endl;
		delete[] h_imgin, h_imgout;

	}

Error:
	for (int i = 0; i < n; i++)
		cudaFree(ImageBuffer[i]);
	cudaFree(d_imgout);
	cudaFree(acc_buffer);

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	cout << "\nDone\n";
	return 0;
}