//
//Written by Abhinav Kalsi on 05/09/2025
//


#ifndef DIST_HPP
#define DIST_HPP

#include <cstdio>
#include <array>
#include <algorithm>
#include <span>
#include <cassert>
#include <concepts>
#include <type_traits>

#include "dist_kernels.hpp"


template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

template<typename T>
constexpr inline T abs_constexpr(T x) noexcept requires Numeric<T> {
	return x < 0.0f ? -x : x;    //std::abs is not constexpr in MSVC
}


template<size_t size_arr>
constexpr void arr_transform(std::array<float, size_arr>& arr) noexcept {
	int arr_size{ static_cast<int>(arr.size()) };
	for (auto& element : arr) 
		element /= (arr_size - 1);   
}

//template<size_t size_arr>
constexpr inline void calc_dist(std::span<float> arr, float ref_point) noexcept {
	for (auto& element : arr)
		element = abs_constexpr(element - ref_point);
}

template<size_t size_arr>
constexpr inline auto arr_assign() noexcept {
	std::array<float, size_arr> arr{};
	for (int i{}; i < arr.size(); ++i)
		arr[i] = static_cast<float>(i);

	return arr;             // NRVO
}



template<size_t size_arr>
[[nodiscard]]
constexpr auto compute_at_compile_time(const float ref_point) {
	std::array<float, size_arr> arr = arr_assign<size_arr>();

	arr_transform<size_arr>(arr);
	calc_dist(arr, ref_point);

	return arr;           //NRVO
}

template<size_t size_arr>
[[nodiscard]]
auto compute_at_run_time_cpu(const float ref_point) {
	std::array<float, size_arr> arr{ 0 };
	for (int i{}; i < size_arr; ++i) arr[i] = static_cast<float>(i);

	arr_transform<size_arr>(arr);
	calc_dist(arr, ref_point);

	std::printf("Runtime calculations done! \n");
	return arr;            // NRVO
}

template<size_t size>
[[nodiscard]]
std::array<float, size> compute_on_gpu(float ref_point, size_t GRID_SIZE, size_t TPB, bool profiling = false) {
	std::array<float, size> arr = arr_assign<size>();
	
	results_gpu(arr.data(), ref_point, size, GRID_SIZE, TPB, profiling);

	//std::copy(h_arr, h_arr + size, arr.begin());
	return arr;  // NRVO
}

const enum class Method {Compile_time, Run_time_CPU, GPU};

template<size_t size>
[[nodiscard]]
constexpr std::array<float, size> compute(Method method, float ref_point) {
	static_assert(size > 1 && "size must be greater than 1!");
	//static_assert(method != Method::GPU && "Method::GPU require additional Two arguments(GRID_SIZE, TPB)");
	switch (method) {
		case Method::Compile_time : return compute_at_compile_time<size>(ref_point);
		case Method::Run_time_CPU: return compute_at_run_time_cpu<size>(ref_point);
		case Method::GPU: throw "Method::GPU require additional Two arguments(GRID_SIZE, TPB)";
	}
}	
 
template<size_t size>
[[nodiscard]]
std::array<float, size> compute(Method method, float ref_point, size_t GRID_SIZE, size_t TPB, bool profiling = false) {
	assert(size > 1 && "size msut be greater than 1!");
	assert(method == Method::GPU && "Method::Compile_time/Run_time_CPU invoked with extra parameters, remove GRID_SIZE and TPB!");
	return compute_on_gpu<size>(ref_point, GRID_SIZE, TPB, profiling);
}


template<size_t size>
void print_result(const std::array<float, size>& arr, const float ref_point) {
	std::printf("Reference point: %.3f\n", ref_point);
	for (auto element : arr)
		std::printf("The distance from Reference Point is %.3f\n", element);
}

template<size_t size>
void print_result_top10(const std::array<float, size>& arr, const float ref_point) {
	std::printf("Reference point: %.3f\n", ref_point);
	for (size_t i{}; i < std::min(size, static_cast<size_t>(10)); ++i)
		std::printf("The distance from Reference Point is %.3f\n", arr[i]);
}

#endif  //DIST_HPP
