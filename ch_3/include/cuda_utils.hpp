//
//Written by Abhinav Kalsi on 05/16/2025
//

#ifndef CUDA_UTILS_HPP
#define CUDA_UTILS_HPP

#include <cstdio>
#include <cstdlib>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace cuda_utils
{
	void checkCudaError(cudaError_t err) {
		if (err != cudaSuccess) {
			std::fprintf(stderr, "Cuda Error :%s\n", cudaGetErrorString(err));
			std::exit(1);
		}
	}

	class Timing {
	private:
		cudaEvent_t start_;
		cudaEvent_t stop_;
		float elapsed_time_;

	public:
		Timing() { checkCudaError(cudaEventCreate(&start_)); checkCudaError(cudaEventCreate(&stop_)); };

		Timing(const Timing&) = delete;
		Timing& operator=(const Timing&) = delete;

		Timing(Timing&&) noexcept = delete;
		Timing& operator=(Timing&&) = delete;

		void start() { checkCudaError(cudaEventRecord(start_)); }

		void stop() { checkCudaError(cudaEventRecord(stop_)); }

		float elapsed_time() {
			checkCudaError(cudaEventSynchronize(stop_));
			checkCudaError(cudaEventElapsedTime(&elapsed_time_, start_, stop_));
			return elapsed_time_;
		}

		~Timing() { checkCudaError(cudaEventDestroy(start_)); checkCudaError(cudaEventDestroy(stop_)); };
	};

}// namespace cuda_utils

#endif //CUDA_UTILS_HPP

