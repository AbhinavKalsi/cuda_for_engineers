#ifndef DIST_KERNELS_HPP
#define DIST_KERNELS_HPP

#include <cstddef>

void results_gpu(float*, float, const size_t, size_t, size_t, bool);

//__global__ void normalize_and_calc_dist(float* arr, size_t size, float ref_point);

#endif // DIST_KERNELS_HPP
