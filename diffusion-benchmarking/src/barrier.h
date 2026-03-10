#pragma once

#include <atomic>

template <bool busy_wait, typename num_t>
class barrier_t
{
	num_t num_threads_;
	std::atomic<num_t>& count_;
	num_t generation_;

public:
	barrier_t(num_t num_threads, std::atomic<num_t>& count) : num_threads_(num_threads), count_(count), generation_(0)
	{}

	void arrive_and_wait()
	{
		generation_++;

		auto val = count_.fetch_add(1, std::memory_order_acq_rel) + 1;

		if constexpr (!busy_wait)
			count_.notify_all();

		while (val < generation_ * num_threads_)
		{
			if constexpr (!busy_wait)
				count_.wait(val, std::memory_order_acquire);

			val = count_.load(std::memory_order_acquire);
		}
	}

	void arrive()
	{
		generation_++;
		count_.fetch_add(1, std::memory_order_relaxed);
	}

	void wait()
	{
		auto val = count_.load(std::memory_order_acquire);

		if constexpr (!busy_wait)
			count_.notify_all();

		while (val < generation_ * num_threads_)
		{
			if constexpr (!busy_wait)
				count_.wait(val, std::memory_order_acquire);

			val = count_.load(std::memory_order_acquire);
		}
	}
};
