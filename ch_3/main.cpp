
//
//Calculating distance from a reference point for an array.
//

#include <cstdio>

#include "dist.hpp"
#include "utils.hpp"

namespace {
	constexpr size_t size_arr{ 128 };   // i think this should be inside main()
	constexpr float ref_point{ 0.500f };   // same for this as well.
	constexpr size_t TPB{ 64 };
	constexpr size_t GRID_SIZE{ (size_arr+TPB-1) / TPB };
}


auto main() -> int {

	using namespace utils;

	Timing ct_run_time, rt_cpu_time, gpu_time;
	
	ct_run_time.start();
	constexpr auto arr1{ compute<size_arr>(Method::Compile_time, ref_point) };
	ct_run_time.end();

	rt_cpu_time.start();
	auto arr2{ compute<size_arr>(Method::Run_time_CPU, ref_point) };
	rt_cpu_time.end();
	
	gpu_time.start();
	auto arr3{ compute<size_arr>(Method::GPU, ref_point, GRID_SIZE, TPB, true) };
	gpu_time.end(); //if compute_on_gpu() is not sync, result will be wrong. 


	std::printf("Time took to run compute_at_compile_time() : %lldms\n", ct_run_time.elapsed_time());
	print_result<size_arr>(arr1, ref_point);
	std::printf("\n\n");

	std::printf("Time took to run compute_at_run_time() : %lldms\n", rt_cpu_time.elapsed_time());
	print_result_top10<size_arr>(arr2, ref_point);
	std::printf("\n\n");

	std::printf("Time took to run compute_on_gpu() : %lldms\n", gpu_time.elapsed_time());
	print_result_top10<size_arr>(arr3, ref_point);
	
	return 0;
}