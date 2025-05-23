//
//Written by Abhinav Kalsi on 05/15/2025
//

#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <typeinfo>
#include <cstdint>
#include <ctime>

#define TPB 64

using namespace std::chrono_literals;

__global__ void calc_dist(float* arr, int length_arr, float ref) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < length_arr) 
		arr[idx] = std::fabsf(static_cast<float>(idx) / (length_arr - 1) - ref);
}


int main() {
	int size = 128;
	float ref = 0.5f;

	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cudaError_t err;

	float* arr = 0;

	err = cudaMallocManaged(&arr, size * sizeof(float));
	if (err != cudaSuccess) {
		std::printf("cudaMalloc error: %s\n", cudaGetErrorString(err));
		return -1;
	} 

	clock_t start_ctime = clock();
	cudaEventRecord(start);
	calc_dist<<<(size+TPB-1)/TPB, TPB>>>(arr, size, ref); 
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::printf("kernel launch error: %s\n", cudaGetErrorString(err));
		return -1;
	} 

	cudaEventRecord(end);
	clock_t end_ctime = clock();
	
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		std::printf("kernel execution error: %s\n", cudaGetErrorString(err));
		return -1;
	} 
	
	cudaEventSynchronize(end);

	for (int i = 0; i < size; ++i) std::printf("Array element %d is: %.3f\n", i, arr[i]);

	float elapsed_time = 0;
	cudaEventElapsedTime(&elapsed_time, start, end);
	std::printf("Elapsed time for calc_dist kernel: %.5fms\n", elapsed_time);
	std::printf("Elapsed time according to ctime for the kernel: %.5fms\n", (end_ctime - start_ctime)*1000/CLOCKS_PER_SEC);

	cudaEventDestroy(start);
	cudaEventDestroy(end);

	cudaFree(arr);

	std::printf("\n\nDevice Info...\n");
	
	int device_count;
	cudaGetDeviceCount(&device_count);

	std::printf("Number of devices: %d\n", device_count);
	for (int i = 0; i < device_count; ++i) {
		std::printf("-----------------\n");
		
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);

		std::printf("Device Number: %d\n", i);
		std::printf("Device Name: %s\n", prop.name);
		std::printf("Device Compute Capability: %d.%d\n", prop.major, prop.minor);
		std::printf("Device Max Thread per Block: %d\n", prop.maxThreadsPerBlock);
		std::printf("Device Global Memory: %lld\n", prop.totalGlobalMem);
		std::printf("Device Shared Memory per Block: %lld\n", prop.sharedMemPerBlock);
	}

	std::printf("\nlearning <chrono>...\n");
	std::chrono::seconds sec{ 3 };
	auto sec2{ 3s };
	std::printf("sec: %llds\n", sec.count());
	std::printf("sec2: %llds\n", sec2.count());
	std::printf("milliseonds rep type: %s\n", typeid(std::chrono::milliseconds::rep).name());
	std::printf("milliseconds ratio num: %jd\n", std::chrono::milliseconds::period::num);
	std::printf("milliseconds ratio den: %jd\n", std::chrono::milliseconds::period::den);
	
	//coustom duration
	std::chrono::duration<int32_t, std::ratio<1, 2>> half_second;
	half_second = 10s;

	using frame = std::chrono::duration<int64_t, std::ratio<1, 60>>;

	frame frame1(5);
	std::printf("value of frame(5) + half_second(10s) in ms: %lldms\n", std::chrono::duration_cast<std::chrono::milliseconds>(half_second + frame1));

	half_second = std::chrono::duration<int32_t, std::ratio<1, 2>>(8);
	std::printf("value of half second after change to 8 half_second: %llds\n", std::chrono::duration_cast<std::chrono::seconds>(half_second));

	std::chrono::time_point<std::chrono::steady_clock> start_chrono = std::chrono::steady_clock::now();
	std::printf("");
	auto end_chrono = std::chrono::steady_clock::now();
	std::printf("printf() overhead: %lldns\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end_chrono - start_chrono).count());

	std::getchar();
	return 0;
}