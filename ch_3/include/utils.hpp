//
//Written by Abhinav Kalsi on 05/17/2025
//

#ifndef UTILS_HPP
#define UTILS_HPP

#include <chrono>
 
namespace utils
{
	class Timing {
	private:
		std::chrono::time_point<std::chrono::steady_clock> start_;
		std::chrono::time_point<std::chrono::steady_clock> end_;
		std::chrono::milliseconds elapsed_time_;
	public:

		Timing() = default;

		Timing(const Timing&) = delete;
		Timing& operator=(const Timing&) = delete;

		Timing(Timing&&) noexcept = delete;
		Timing& operator=(Timing&&) = delete;

		void start() noexcept { start_ = std::chrono::steady_clock::now(); }
		
		void end() noexcept { end_ = std::chrono::steady_clock::now(); }
		
		long long elapsed_time() noexcept {
			elapsed_time_ = std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_);
			return elapsed_time_.count(); 
		}
	};
} //namespace utils


#endif //UTILS_HPP