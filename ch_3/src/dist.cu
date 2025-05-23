//
//Written by Abhinav Kalsi on 05/09/2025
//

#include <cmath>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cstdio>

#include "dist_kernels.hpp"
#include "cuda_utils.hpp"

using namespace cuda_utils;

__global__ void normalize_and_calc_dist(float* arr, size_t size, float ref_point) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < size) {
		arr[idx] = std::fabsf((static_cast<float>(idx)/ (size - 1)) - ref_point);
	}
}


__host__ void results_gpu(float* arr, float ref_point, const size_t size, size_t GRID_SIZE, size_t TPB, bool profiling) {
	float* d_arr{};

	checkCudaError(cudaMalloc(&d_arr, size * sizeof(float)));  //need to build RAII for cudaMalloc.
	
	Timing kernel_time;
	
	kernel_time.start();
	normalize_and_calc_dist <<<GRID_SIZE, TPB>>> (d_arr, size, ref_point);
	kernel_time.stop();
	checkCudaError(cudaGetLastError());
	checkCudaError(cudaDeviceSynchronize());
	
	Timing memcpy_time;
	memcpy_time.start();
	checkCudaError(cudaMemcpy(arr, d_arr, size * sizeof(float), cudaMemcpyDeviceToHost));
	memcpy_time.stop();

	checkCudaError(cudaFree(d_arr));
	if (profiling) {
		std::printf("Performance Measurment using Cuda Events\n");
		std::printf("----------------------------------------\n");
		std::printf("Elapsed Time kernel: %.5fms\n", kernel_time.elapsed_time());
		std::printf("Elapsed Time memcpy: %.5fms\n", memcpy_time.elapsed_time());
		std::printf("\n");
	}
	
	//for(int i = 0; i<size; ++i) printf("arr[%d]: %f\n", i, arr[i]);

	//return arr;
}