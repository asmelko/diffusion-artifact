#include "blocked_thomas_solver_t.h"

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <thread>

#include "noarr/structures/base/structs_common.hpp"
#include "noarr/structures/extra/shortcuts.hpp"
#include "omp_helper.h"

// a helper using for accessing static constexpr variables
using alg = blocked_thomas_solver_t<double, true>;

template <typename real_t, bool aligned_x>
void blocked_thomas_solver_t<real_t, aligned_x>::precompute_values(real_t*& a, real_t*& b1, index_t shape, index_t dims,
																   index_t n, index_t& counters_count,
																   std::unique_ptr<aligned_atomic<long>[]>& counters,
																   index_t& max_threads)
{
	// allocate memory for a and b1
	a = (real_t*)std::malloc(this->problem_.substrates_count * sizeof(real_t));
	b1 = (real_t*)std::malloc(this->problem_.substrates_count * sizeof(real_t));

	// compute a
	for (index_t s = 0; s < this->problem_.substrates_count; s++)
		a[s] = -this->problem_.dt * this->problem_.diffusion_coefficients[s] / (shape * shape);

	// compute b1
	for (index_t s = 0; s < this->problem_.substrates_count; s++)
		b1[s] = 1 + this->problem_.decay_rates[s] * this->problem_.dt / dims
				+ 2 * this->problem_.dt * this->problem_.diffusion_coefficients[s] / (shape * shape);

	index_t blocks_count = n / block_size_; // TODO: handle remainder
	max_threads = std::max<index_t>(get_max_threads(), blocks_count);

	counters_count = (max_threads + blocks_count - 1) / blocks_count;
	counters = std::make_unique<aligned_atomic<long>[]>(counters_count);

	for (index_t i = 0; i < counters_count; i++)
	{
		counters[i].value.store(0, std::memory_order_relaxed);
	}
}

template <typename real_t, bool aligned_x>
void blocked_thomas_solver_t<real_t, aligned_x>::prepare(const max_problem_t& problem)
{
	this->problem_ = problems::cast<std::int32_t, real_t>(problem);

	auto substrates_layout = get_substrates_layout<3>();

	if (aligned_x)
		this->substrates_ = (real_t*)std::aligned_alloc(alignment_size_, (substrates_layout | noarr::get_size()));
	else
		this->substrates_ = (real_t*)std::malloc((substrates_layout | noarr::get_size()));

	// Initialize substrates
	solver_utils::initialize_substrate(substrates_layout, this->substrates_, this->problem_);
}

template <typename real_t, bool aligned_x>
void blocked_thomas_solver_t<real_t, aligned_x>::tune(const nlohmann::json& params)
{
	block_size_ = params.contains("block_size") ? (std::size_t)params["block_size"] : 100;
	alignment_size_ = params.contains("alignment_size") ? (std::size_t)params["alignment_size"] : 64;
	x_tile_size_ = params.contains("x_tile_size") ? (std::size_t)params["x_tile_size"] : 48;
}

template <typename real_t, bool aligned_x>
void blocked_thomas_solver_t<real_t, aligned_x>::initialize()
{
	if (this->problem_.dims >= 1)
		precompute_values(ax_, b1x_, this->problem_.dx, this->problem_.dims, this->problem_.nx, countersx_count_,
						  countersx_, max_threadsx_);
	if (this->problem_.dims >= 2)
		precompute_values(ay_, b1y_, this->problem_.dy, this->problem_.dims, this->problem_.ny, countersy_count_,
						  countersy_, max_threadsy_);
	if (this->problem_.dims >= 3)
		precompute_values(az_, b1z_, this->problem_.dz, this->problem_.dims, this->problem_.nz, countersz_count_,
						  countersz_, max_threadsz_);

	auto scratch_layout = get_scratch_layout(std::max({ max_threadsx_, max_threadsy_, max_threadsz_ }));

	a_scratch_ = (real_t*)std::malloc((scratch_layout | noarr::get_size()));
	c_scratch_ = (real_t*)std::malloc((scratch_layout | noarr::get_size()));
}

template <typename real_t, bool aligned_x>
auto blocked_thomas_solver_t<real_t, aligned_x>::get_diagonal_layout(const problem_t<index_t, real_t>& problem,
																	 index_t n)
{
	if constexpr (aligned_x)
	{
		std::size_t size = n * sizeof(real_t);
		std::size_t size_padded = (size + alignment_size_ - 1) / alignment_size_ * alignment_size_;
		size_padded /= sizeof(real_t);

		return noarr::scalar<real_t>() ^ noarr::vectors<'i', 's'>(size_padded, problem.substrates_count)
			   ^ noarr::slice<'i'>(n);
	}
	else
	{
		return noarr::scalar<real_t>() ^ noarr::vectors<'i', 's'>(n, problem.substrates_count);
	}
}

template <typename index_t, typename real_t, typename counter_t, typename density_layout_t, typename scratch_layout_t>
static void solve_slice_x_1d(real_t* __restrict__ densities, const real_t* __restrict__ ac,
							 const real_t* __restrict__ b1, real_t* __restrict__ a_data, real_t* __restrict__ c_data,
							 aligned_atomic<counter_t>* __restrict__ counters, const density_layout_t dens_l,
							 const scratch_layout_t scratch_l, const index_t block_size)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'x'>();

	const index_t blocks_count = n / block_size;

	const auto tid = get_thread_num();
	const auto thread_count = get_num_threads();

	auto a = noarr::make_bag(scratch_l, a_data);
	auto c = noarr::make_bag(scratch_l, c_data);
	auto d = noarr::make_bag(dens_l, densities);

	const index_t counter_count = (thread_count + blocks_count - 1) / blocks_count;

	for (index_t counter_idx = tid; counter_idx < counter_count; counter_idx += thread_count)
	{
		counters[counter_idx].value.store(0, std::memory_order_relaxed);
	}

#pragma omp barrier

	for (index_t idx = tid; idx < substrates_count * blocks_count; idx += thread_count)
	{
		const index_t s = idx / blocks_count;
		const index_t block_idx = idx % blocks_count;

		// First part of modified thomas algorithm
		// We modify equations in blocks to have the first and the last equation form the tridiagonal system
		{
			const index_t block_start = block_idx * block_size;
			const index_t block_end = block_start + block_size;

			// Normalize the first and the second equation
			index_t i;
			for (i = block_start; i < block_start + 2; i++)
			{
				const auto state = noarr::idx<'t', 'i'>(tid, i - block_start);

				const auto a_tmp = ac[s] * (i == 0 ? 0 : 1);
				const auto b_tmp = b1[s] + (i == 0 || i == n - 1 ? ac[s] : 0);
				const auto c_tmp = ac[s] * (i == n - 1 ? 0 : 1);

				a[state] = a_tmp / b_tmp;
				c[state] = c_tmp / b_tmp;

				d.template at<'x', 's'>(i, s) /= b_tmp;
			}

			// Process the lower diagonal (forward)
			for (; i < block_end; i++)
			{
				const auto state = noarr::idx<'t', 'i'>(tid, i - block_start);
				const auto prev_state = noarr::idx<'t', 'i'>(tid, i - block_start - 1);

				const auto a_tmp = ac[s] * (i == 0 ? 0 : 1);
				const auto b_tmp = b1[s] + (i == n - 1 ? ac[s] : 0);
				const auto c_tmp = ac[s] * (i == n - 1 ? 0 : 1);

				const auto r = 1 / (b_tmp - a_tmp * c[prev_state]);

				a[state] = r * (0 - a_tmp * a[prev_state]);
				c[state] = r * c_tmp;

				d.template at<'x', 's'>(i, s) =
					r * (d.template at<'x', 's'>(i, s) - a_tmp * d.template at<'x', 's'>(i - 1, s));
			}

			// Process the upper diagonal (backward)
			for (i = block_end - 3; i >= block_start + 1; i--)
			{
				const auto state = noarr::idx<'t', 'i'>(tid, i - block_start);
				const auto next_state = noarr::idx<'t', 'i'>(tid, i - block_start + 1);

				d.template at<'x', 's'>(i, s) -= c[state] * d.template at<'x', 's'>(i + 1, s);

				a[state] = a[state] - c[state] * a[next_state];
				c[state] = 0 - c[state] * c[next_state];
			}

			// Process the first row (backward)
			{
				const auto state = noarr::idx<'t', 'i'>(tid, i - block_start);
				const auto next_state = noarr::idx<'t', 'i'>(tid, i - block_start + 1);

				const auto r = 1 / (1 - c[state] * a[next_state]);

				d.template at<'x', 's'>(i, s) =
					r * (d.template at<'x', 's'>(i, s) - c[state] * d.template at<'x', 's'>(i + 1, s));

				a[state] = r * a[state];
				c[state] = r * (0 - c[state] * c[next_state]);
			}
		}

		{
			const counter_t visit_count = s / counter_count;
			const counter_t last_epoch_end = visit_count * (blocks_count + 1);
			const counter_t this_epoch_end = (visit_count + 1) * (blocks_count + 1);

			auto& counter = counters[s % counter_count].value;

			auto current_value = counter.load(std::memory_order_relaxed);

			// sleep if a thread is too much ahead
			while (current_value < last_epoch_end)
			{
				counter.wait(current_value, std::memory_order_relaxed);
				current_value = counter.load(std::memory_order_relaxed);
			}

			// increment the counter
			current_value = counter.fetch_add(1, std::memory_order_relaxed);

			// Second part of modified thomas algorithm
			// We solve the system of equations that are composed of the first and last row of each block
			// We do it using Thomas Algorithm
			if (current_value == this_epoch_end - 2)
			{
				// bags for the middle phase
				auto get_i = [block_size, tid, thread_count, block_idx](index_t equation_idx) {
					const index_t equation_block_idx = equation_idx / 2;
					const index_t equation_block_offset = equation_idx % 2 * (block_size - 1);

					const index_t equation_tid = (tid + equation_block_idx - block_idx + thread_count) % thread_count;
					const index_t equation_i = equation_block_offset;
					const index_t equation_dens_i = equation_block_idx * block_size + equation_i;

					return std::make_tuple(equation_tid, equation_i, equation_dens_i);
				};

				for (index_t equation_idx = 1; equation_idx < blocks_count * 2; equation_idx++)
				{
					const auto [tid, i, dens_i] = get_i(equation_idx);
					const auto [prev_tid, prev_i, prev_dens_i] = get_i(equation_idx - 1);

					const auto state = noarr::idx<'t', 'i'>(tid, i);
					const auto prev_state = noarr::idx<'t', 'i'>(prev_tid, prev_i);

					const auto r = 1 / (1 - a[state] * c[prev_state]);

					c[state] *= r;

					d.template at<'x', 's'>(dens_i, s) =
						r * (d.template at<'x', 's'>(dens_i, s) - a[state] * d.template at<'x', 's'>(prev_dens_i, s));
				}

				for (index_t equation_idx = blocks_count * 2 - 2; equation_idx >= 0; equation_idx--)
				{
					const auto [tid, i, dens_i] = get_i(equation_idx);
					const auto [next_tid, next_i, next_dens_i] = get_i(equation_idx + 1);

					const auto state = noarr::idx<'t', 'i'>(tid, i);

					d.template at<'x', 's'>(dens_i, s) =
						d.template at<'x', 's'>(dens_i, s) - c[state] * d.template at<'x', 's'>(next_dens_i, s);
				}

				// notify all threads that the last thread has finished
				counter.store(this_epoch_end, std::memory_order_release);
				counter.notify_all();
			}
			else
			{
				// wait for the last thread to finish
				while (current_value < this_epoch_end)
				{
					counter.wait(current_value, std::memory_order_acquire);
					current_value = counter.load(std::memory_order_acquire);
				}
			}
		}

		// Final part of modified thomas algorithm
		// Solve the rest of the unknowns
		{
			const index_t block_start = block_idx * block_size;
			const index_t block_end = block_start + block_size;

			for (index_t i = block_start + 1; i < block_end - 1; i++)
			{
				const auto state = noarr::idx<'t', 'i'>(tid, i - block_start);

				d.template at<'x', 's'>(i, s) = d.template at<'x', 's'>(i, s)
												- a[state] * d.template at<'x', 's'>(block_start, s)
												- c[state] * d.template at<'x', 's'>(block_end - 1, s);
			}
		}
	}
}

template <typename index_t, typename real_t, typename counter_t, typename density_layout_t, typename scratch_layout_t>
static void solve_slice_x_2d_and_3d(real_t* __restrict__ densities, const real_t* __restrict__ ac,
									const real_t* __restrict__ b1, real_t* __restrict__ a_data,
									real_t* __restrict__ c_data, aligned_atomic<counter_t>* __restrict__ counters,
									const density_layout_t dens_l, const scratch_layout_t scratch_l,
									const index_t block_size)
{
	constexpr char dim = 'x';
	const index_t n = dens_l | noarr::get_length<dim>();
	const index_t yz_len = dens_l | noarr::get_length<'m'>();

	const index_t blocks_count = n / block_size;

	const auto tid = get_thread_num();
	const auto thread_count = get_num_threads();

	auto a = noarr::make_bag(scratch_l, a_data);
	auto c = noarr::make_bag(scratch_l, c_data);

	const index_t counters_count = (thread_count + blocks_count - 1) / blocks_count;

	const auto blocked_l = dens_l ^ noarr::into_blocks<'x', 'x', 'l'>(block_size);
	const auto linearized =
		noarr::merge_blocks<'s', 'm'>() ^ noarr::merge_blocks<'m', 'x'>() ^ noarr::step<'x'>(tid, thread_count);

	noarr::traverser(blocked_l).order(linearized).template for_dims<'x'>([=](auto state) {
		const auto s = noarr::get_index<'s'>(state);
		const auto yz = noarr::get_index<'m'>(state);
		const auto block_idx = noarr::get_index<'x'>(state);
		const auto system_idx = s * yz_len + yz;

		auto d = noarr::make_bag(dens_l ^ noarr::fix<'s', 'm'>(s, yz), densities);

		// First part of modified thomas algorithm
		// We modify equations in blocks to have the first and the last equation form the tridiagonal system
		{
			const index_t block_start = block_idx * block_size;
			const index_t block_end = block_start + block_size;

			// Normalize the first and the second equation
			index_t i;
			for (i = block_start; i < block_start + 2; i++)
			{
				const auto state = noarr::idx<'t', 'i'>(tid, i - block_start);

				const auto a_tmp = ac[s] * (i == 0 ? 0 : 1);
				const auto b_tmp = b1[s] + (i == 0 || i == n - 1 ? ac[s] : 0);
				const auto c_tmp = ac[s] * (i == n - 1 ? 0 : 1);

				a[state] = a_tmp / b_tmp;
				c[state] = c_tmp / b_tmp;

				d.template at<dim>(i) /= b_tmp;
			}

			// Process the lower diagonal (forward)
			for (; i < block_end; i++)
			{
				const auto state = noarr::idx<'t', 'i'>(tid, i - block_start);
				const auto prev_state = noarr::idx<'t', 'i'>(tid, i - block_start - 1);

				const auto a_tmp = ac[s] * (i == 0 ? 0 : 1);
				const auto b_tmp = b1[s] + (i == n - 1 ? ac[s] : 0);
				const auto c_tmp = ac[s] * (i == n - 1 ? 0 : 1);

				const auto r = 1 / (b_tmp - a_tmp * c[prev_state]);

				a[state] = r * (0 - a_tmp * a[prev_state]);
				c[state] = r * c_tmp;

				d.template at<dim>(i) = r * (d.template at<dim>(i) - a_tmp * d.template at<'x'>(i - 1));
			}

			// Process the upper diagonal (backward)
			for (i = block_end - 3; i >= block_start + 1; i--)
			{
				const auto state = noarr::idx<'t', 'i'>(tid, i - block_start);
				const auto next_state = noarr::idx<'t', 'i'>(tid, i - block_start + 1);

				d.template at<dim>(i) -= c[state] * d.template at<'x'>(i + 1);

				a[state] = a[state] - c[state] * a[next_state];
				c[state] = 0 - c[state] * c[next_state];
			}

			// Process the first row (backward)
			{
				const auto state = noarr::idx<'t', 'i'>(tid, i - block_start);
				const auto next_state = noarr::idx<'t', 'i'>(tid, i - block_start + 1);

				const auto r = 1 / (1 - c[state] * a[next_state]);

				d.template at<dim>(i) = r * (d.template at<dim>(i) - c[state] * d.template at<dim>(i + 1));

				a[state] = r * a[state];
				c[state] = r * (0 - c[state] * c[next_state]);
			}
		}

		{
			const counter_t visit_count = system_idx / counters_count;
			const counter_t last_epoch_end = visit_count * (blocks_count + 1);
			const counter_t this_epoch_end = (visit_count + 1) * (blocks_count + 1);

			auto& counter = counters[system_idx % counters_count].value;

			auto current_value = counter.load(std::memory_order::relaxed);

			// sleep if a thread is too much ahead
			while (current_value < last_epoch_end)
			{
				// counter.wait(current_value, std::memory_order::relaxed);
				current_value = counter.load(std::memory_order::relaxed);
			}

			// increment the counter
			current_value = counter.fetch_add(1, std::memory_order::relaxed);

			// Second part of modified thomas algorithm
			// We solve the system of equations that are composed of the first and last row of each block
			// We do it using Thomas Algorithm
			if (current_value == this_epoch_end - 2)
			{
				// bags for the middle phase
				auto get_i = [block_size, tid, thread_count, block_idx](index_t equation_idx) {
					const index_t equation_block_idx = equation_idx / 2;
					const index_t equation_block_offset = equation_idx % 2 * (block_size - 1);

					const index_t equation_tid = (tid + equation_block_idx - block_idx + thread_count) % thread_count;
					const index_t equation_i = equation_block_offset;
					const index_t equation_dens_i = equation_block_idx * block_size + equation_i;

					return std::make_tuple(equation_tid, equation_i, equation_dens_i);
				};

				for (index_t equation_idx = 1; equation_idx < blocks_count * 2; equation_idx++)
				{
					const auto [tid, i, dens_i] = get_i(equation_idx);
					const auto [prev_tid, prev_i, prev_dens_i] = get_i(equation_idx - 1);

					const auto state = noarr::idx<'t', 'i'>(tid, i);
					const auto prev_state = noarr::idx<'t', 'i'>(prev_tid, prev_i);

					const auto r = 1 / (1 - a[state] * c[prev_state]);

					c[state] *= r;

					d.template at<dim>(dens_i) =
						r * (d.template at<dim>(dens_i) - a[state] * d.template at<dim>(prev_dens_i));
				}

				for (index_t equation_idx = blocks_count * 2 - 2; equation_idx >= 0; equation_idx--)
				{
					const auto [tid, i, dens_i] = get_i(equation_idx);
					const auto [next_tid, next_i, next_dens_i] = get_i(equation_idx + 1);

					const auto state = noarr::idx<'t', 'i'>(tid, i);

					d.template at<dim>(dens_i) =
						d.template at<dim>(dens_i) - c[state] * d.template at<dim>(next_dens_i);
				}


				// notify all threads that the last thread has finished
				counter.store(this_epoch_end, std::memory_order::release);
				// counter.notify_all();
			}
			else
			{
				// wait for the last thread to finish
				while (current_value < this_epoch_end)
				{
					// counter.wait(current_value, std::memory_order::relaxed);
					current_value = counter.load(std::memory_order::acquire);
				}
			}
		}

		// Final part of modified thomas algorithm
		// Solve the rest of the unknowns
		{
			const index_t block_start = block_idx * block_size;
			const index_t block_end = block_start + block_size;

			for (index_t i = block_start + 1; i < block_end - 1; i++)
			{
				const auto state = noarr::idx<'t', 'i'>(tid, i - block_start);

				d.template at<dim>(i) = d.template at<dim>(i) - a[state] * d.template at<dim>(block_start)
										- c[state] * d.template at<dim>(block_end - 1);
			}
		}
	});
}

template <typename index_t, typename real_t, typename counter_t, typename density_layout_t, typename scratch_layout_t>
static void solve_slice_y_2d(real_t* __restrict__ densities, const real_t* __restrict__ ac,
							 const real_t* __restrict__ b1, real_t* __restrict__ a_data, real_t* __restrict__ c_data,
							 aligned_atomic<counter_t>* __restrict__ counters, const density_layout_t dens_l,
							 const scratch_layout_t scratch_l, const index_t block_size, const index_t x_tile_size)
{
	constexpr char dim = 'y';
	const index_t n = dens_l | noarr::get_length<dim>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();

	const index_t blocks_count = n / block_size;

	const auto tid = get_thread_num();
	const auto thread_count = get_num_threads();

	auto a = noarr::make_bag(scratch_l, a_data);
	auto c = noarr::make_bag(scratch_l, c_data);

	const index_t counters_count = (thread_count + blocks_count - 1) / blocks_count;

	const auto blocked_dens_l = dens_l ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(x_tile_size);
	const auto blocked_l = blocked_dens_l ^ noarr::into_blocks<'y', 'Y', 'l'>(block_size);
	const auto linearized = noarr::merge_blocks<'X', 'Y', 't'>() ^ noarr::merge_blocks<'s', 't', 't'>()
							^ noarr::step<'t'>(tid, thread_count);

	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

	noarr::traverser(blocked_l).order(linearized).template for_dims<'t'>([=](auto state) {
		const index_t s = noarr::get_index<'s'>(state);
		const index_t X = noarr::get_index<'X'>(state);
		const index_t block_idx = noarr::get_index<'Y'>(state);
		const std::size_t system_idx = s * X_len + X;

		auto d = noarr::make_bag(blocked_dens_l ^ noarr::fix<'s', 'X', 'b'>(s, X, 0), densities);

		const auto remainder = x_len % x_tile_size;
		const auto x_len_remainder = remainder == 0 ? x_tile_size : remainder;
		const auto x_len = X == X_len - 1 ? x_len_remainder : x_tile_size;

		// First part of modified thomas algorithm
		// We modify equations in blocks to have the first and the last equation form the tridiagonal system
		{
			const index_t block_start = block_idx * block_size;
			const index_t block_end = block_start + block_size;

			// Normalize the first and the second equation
			index_t i;
			for (i = block_start; i < block_start + 2; i++)
			{
				const auto state = noarr::idx<'t', 'i'>(tid, i - block_start);

				const auto a_tmp = ac[s] * (i == 0 ? 0 : 1);
				const auto b_tmp = b1[s] + (i == 0 || i == n - 1 ? ac[s] : 0);
				const auto c_tmp = ac[s] * (i == n - 1 ? 0 : 1);

				a[state] = a_tmp / b_tmp;
				c[state] = c_tmp / b_tmp;

				for (index_t x = 0; x < x_len; x++)
					d.template at<dim, 'x'>(i, x) /= b_tmp;
			}

			// Process the lower diagonal (forward)
			for (; i < block_end; i++)
			{
				const auto state = noarr::idx<'t', 'i'>(tid, i - block_start);
				const auto prev_state = noarr::idx<'t', 'i'>(tid, i - block_start - 1);

				const auto a_tmp = ac[s] * (i == 0 ? 0 : 1);
				const auto b_tmp = b1[s] + (i == n - 1 ? ac[s] : 0);
				const auto c_tmp = ac[s] * (i == n - 1 ? 0 : 1);

				const auto r = 1 / (b_tmp - a_tmp * c[prev_state]);

				a[state] = r * (0 - a_tmp * a[prev_state]);
				c[state] = r * c_tmp;

				for (index_t x = 0; x < x_len; x++)
					d.template at<dim, 'x'>(i, x) =
						r * (d.template at<dim, 'x'>(i, x) - a_tmp * d.template at<dim, 'x'>(i - 1, x));
			}

			// Process the upper diagonal (backward)
			for (i = block_end - 3; i >= block_start + 1; i--)
			{
				const auto state = noarr::idx<'t', 'i'>(tid, i - block_start);
				const auto next_state = noarr::idx<'t', 'i'>(tid, i - block_start + 1);

				for (index_t x = 0; x < x_len; x++)
					d.template at<dim, 'x'>(i, x) -= c[state] * d.template at<dim, 'x'>(i + 1, x);

				a[state] = a[state] - c[state] * a[next_state];
				c[state] = 0 - c[state] * c[next_state];
			}

			// Process the first row (backward)
			{
				const auto state = noarr::idx<'t', 'i'>(tid, i - block_start);
				const auto next_state = noarr::idx<'t', 'i'>(tid, i - block_start + 1);

				const auto r = 1 / (1 - c[state] * a[next_state]);

				for (index_t x = 0; x < x_len; x++)
					d.template at<dim, 'x'>(i, x) =
						r * (d.template at<dim, 'x'>(i, x) - c[state] * d.template at<dim, 'x'>(i + 1, x));

				a[state] = r * a[state];
				c[state] = r * (0 - c[state] * c[next_state]);
			}
		}

		{
			const counter_t visit_count = system_idx / counters_count;
			const counter_t last_epoch_end = visit_count * (blocks_count + 1);
			const counter_t this_epoch_end = (visit_count + 1) * (blocks_count + 1);

			auto& counter = counters[system_idx % counters_count].value;

			auto current_value = counter.load(std::memory_order::relaxed);

			// sleep if a thread is too much ahead
			while (current_value < last_epoch_end)
			{
				// counter.wait(current_value, std::memory_order::relaxed);
				current_value = counter.load(std::memory_order::relaxed);
			}

			// increment the counter
			current_value = counter.fetch_add(1, std::memory_order::relaxed);

			// Second part of modified thomas algorithm
			// We solve the system of equations that are composed of the first and last row of each block
			// We do it using Thomas Algorithm
			if (current_value == this_epoch_end - 2)
			{
				// bags for the middle phase
				auto get_i = [block_size, tid, thread_count, block_idx](index_t equation_idx) {
					const index_t equation_block_idx = equation_idx / 2;
					const index_t equation_block_offset = equation_idx % 2 * (block_size - 1);

					const index_t equation_tid = (tid + equation_block_idx - block_idx + thread_count) % thread_count;
					const index_t equation_i = equation_block_offset;
					const index_t equation_dens_i = equation_block_idx * block_size + equation_i;

					return std::make_tuple(equation_tid, equation_i, equation_dens_i);
				};

				for (index_t equation_idx = 1; equation_idx < blocks_count * 2; equation_idx++)
				{
					const auto [tid, i, dens_i] = get_i(equation_idx);
					const auto [prev_tid, prev_i, prev_dens_i] = get_i(equation_idx - 1);

					const auto state = noarr::idx<'t', 'i'>(tid, i);
					const auto prev_state = noarr::idx<'t', 'i'>(prev_tid, prev_i);

					const auto r = 1 / (1 - a[state] * c[prev_state]);

					c[state] *= r;

					for (index_t x = 0; x < x_len; x++)
						d.template at<dim, 'x'>(dens_i, x) =
							r
							* (d.template at<dim, 'x'>(dens_i, x) - a[state] * d.template at<dim, 'x'>(prev_dens_i, x));
				}

				for (index_t equation_idx = blocks_count * 2 - 2; equation_idx >= 0; equation_idx--)
				{
					const auto [tid, i, dens_i] = get_i(equation_idx);
					const auto [next_tid, next_i, next_dens_i] = get_i(equation_idx + 1);

					const auto state = noarr::idx<'t', 'i'>(tid, i);

					for (index_t x = 0; x < x_len; x++)
						d.template at<dim, 'x'>(dens_i, x) =
							d.template at<dim, 'x'>(dens_i, x) - c[state] * d.template at<dim, 'x'>(next_dens_i, x);
				}


				// notify all threads that the last thread has finished
				counter.store(this_epoch_end, std::memory_order::release);
				// counter.notify_all();
			}
			else
			{
				// wait for the last thread to finish
				while (current_value < this_epoch_end)
				{
					// counter.wait(current_value, std::memory_order::relaxed);
					current_value = counter.load(std::memory_order::acquire);
				}
			}
		}

		// Final part of modified thomas algorithm
		// Solve the rest of the unknowns
		{
			const index_t block_start = block_idx * block_size;
			const index_t block_end = block_start + block_size;

			for (index_t i = block_start + 1; i < block_end - 1; i++)
			{
				const auto state = noarr::idx<'t', 'i'>(tid, i - block_start);

				for (index_t x = 0; x < x_len; x++)
					d.template at<dim, 'x'>(i, x) = d.template at<dim, 'x'>(i, x)
													- a[state] * d.template at<dim, 'x'>(block_start, x)
													- c[state] * d.template at<dim, 'x'>(block_end - 1, x);
			}
		}
	});
}

template <typename index_t, typename real_t, typename counter_t, typename density_layout_t, typename scratch_layout_t>
static void solve_slice_y_3d(real_t* __restrict__ densities, const real_t* __restrict__ ac,
							 const real_t* __restrict__ b1, real_t* __restrict__ a_data, real_t* __restrict__ c_data,
							 aligned_atomic<counter_t>* __restrict__ counters, const density_layout_t dens_l,
							 const scratch_layout_t scratch_l, const index_t block_size, const index_t x_tile_size)
{
	constexpr char dim = 'y';
	const index_t n = dens_l | noarr::get_length<dim>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t z_len = dens_l | noarr::get_length<'z'>();

	const index_t blocks_count = n / block_size;

	const auto tid = get_thread_num();
	const auto thread_count = get_num_threads();

	auto a = noarr::make_bag(scratch_l, a_data);
	auto c = noarr::make_bag(scratch_l, c_data);

	const index_t counters_count = (thread_count + blocks_count - 1) / blocks_count;

	const auto blocked_dens_l = dens_l ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(x_tile_size);
	const auto blocked_l = blocked_dens_l ^ noarr::into_blocks<'y', 'Y', 'l'>(block_size);
	const auto linearized = noarr::merge_blocks<'X', 'Y', 't'>() ^ noarr::merge_blocks<'z', 't', 't'>()
							^ noarr::merge_blocks<'s', 't', 't'>() ^ noarr::step<'t'>(tid, thread_count);

	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

	noarr::traverser(blocked_l).order(linearized).template for_dims<'t'>([=](auto state) {
		const index_t s = noarr::get_index<'s'>(state);
		const index_t z = noarr::get_index<'z'>(state);
		const index_t X = noarr::get_index<'X'>(state);
		const index_t block_idx = noarr::get_index<'Y'>(state);
		const std::size_t system_idx = s * z_len * X_len + z * X_len + X;

		auto d = noarr::make_bag(blocked_dens_l ^ noarr::fix<'s', 'z', 'X', 'b'>(s, z, X, 0), densities);

		const auto remainder = x_len % x_tile_size;
		const auto x_len_remainder = remainder == 0 ? x_tile_size : remainder;
		const auto x_len = X == X_len - 1 ? x_len_remainder : x_tile_size;

		// First part of modified thomas algorithm
		// We modify equations in blocks to have the first and the last equation form the tridiagonal system
		{
			const index_t block_start = block_idx * block_size;
			const index_t block_end = block_start + block_size;

			// Normalize the first and the second equation
			index_t i;
			for (i = block_start; i < block_start + 2; i++)
			{
				const auto state = noarr::idx<'t', 'i'>(tid, i - block_start);

				const auto a_tmp = ac[s] * (i == 0 ? 0 : 1);
				const auto b_tmp = b1[s] + (i == 0 || i == n - 1 ? ac[s] : 0);
				const auto c_tmp = ac[s] * (i == n - 1 ? 0 : 1);

				a[state] = a_tmp / b_tmp;
				c[state] = c_tmp / b_tmp;

				for (index_t x = 0; x < x_len; x++)
					d.template at<dim, 'x'>(i, x) /= b_tmp;
			}

			// Process the lower diagonal (forward)
			for (; i < block_end; i++)
			{
				const auto state = noarr::idx<'t', 'i'>(tid, i - block_start);
				const auto prev_state = noarr::idx<'t', 'i'>(tid, i - block_start - 1);

				const auto a_tmp = ac[s] * (i == 0 ? 0 : 1);
				const auto b_tmp = b1[s] + (i == n - 1 ? ac[s] : 0);
				const auto c_tmp = ac[s] * (i == n - 1 ? 0 : 1);

				const auto r = 1 / (b_tmp - a_tmp * c[prev_state]);

				a[state] = r * (0 - a_tmp * a[prev_state]);
				c[state] = r * c_tmp;

				for (index_t x = 0; x < x_len; x++)
					d.template at<dim, 'x'>(i, x) =
						r * (d.template at<dim, 'x'>(i, x) - a_tmp * d.template at<dim, 'x'>(i - 1, x));
			}

			// Process the upper diagonal (backward)
			for (i = block_end - 3; i >= block_start + 1; i--)
			{
				const auto state = noarr::idx<'t', 'i'>(tid, i - block_start);
				const auto next_state = noarr::idx<'t', 'i'>(tid, i - block_start + 1);

				for (index_t x = 0; x < x_len; x++)
					d.template at<dim, 'x'>(i, x) -= c[state] * d.template at<dim, 'x'>(i + 1, x);

				a[state] = a[state] - c[state] * a[next_state];
				c[state] = 0 - c[state] * c[next_state];
			}

			// Process the first row (backward)
			{
				const auto state = noarr::idx<'t', 'i'>(tid, i - block_start);
				const auto next_state = noarr::idx<'t', 'i'>(tid, i - block_start + 1);

				const auto r = 1 / (1 - c[state] * a[next_state]);

				for (index_t x = 0; x < x_len; x++)
					d.template at<dim, 'x'>(i, x) =
						r * (d.template at<dim, 'x'>(i, x) - c[state] * d.template at<dim, 'x'>(i + 1, x));

				a[state] = r * a[state];
				c[state] = r * (0 - c[state] * c[next_state]);
			}
		}

		{
			const counter_t visit_count = system_idx / counters_count;
			const counter_t last_epoch_end = visit_count * (blocks_count + 1);
			const counter_t this_epoch_end = (visit_count + 1) * (blocks_count + 1);

			auto& counter = counters[system_idx % counters_count].value;

			auto current_value = counter.load(std::memory_order::relaxed);

			// sleep if a thread is too much ahead
			while (current_value < last_epoch_end)
			{
				// counter.wait(current_value, std::memory_order::relaxed);
				current_value = counter.load(std::memory_order::relaxed);
			}

			// increment the counter
			current_value = counter.fetch_add(1, std::memory_order::relaxed);

			// Second part of modified thomas algorithm
			// We solve the system of equations that are composed of the first and last row of each block
			// We do it using Thomas Algorithm
			if (current_value == this_epoch_end - 2)
			{
				// bags for the middle phase
				auto get_i = [block_size, tid, thread_count, block_idx](index_t equation_idx) {
					const index_t equation_block_idx = equation_idx / 2;
					const index_t equation_block_offset = equation_idx % 2 * (block_size - 1);

					const index_t equation_tid = (tid + equation_block_idx - block_idx + thread_count) % thread_count;
					const index_t equation_i = equation_block_offset;
					const index_t equation_dens_i = equation_block_idx * block_size + equation_i;

					return std::make_tuple(equation_tid, equation_i, equation_dens_i);
				};

				for (index_t equation_idx = 1; equation_idx < blocks_count * 2; equation_idx++)
				{
					const auto [tid, i, dens_i] = get_i(equation_idx);
					const auto [prev_tid, prev_i, prev_dens_i] = get_i(equation_idx - 1);

					const auto state = noarr::idx<'t', 'i'>(tid, i);
					const auto prev_state = noarr::idx<'t', 'i'>(prev_tid, prev_i);

					const auto r = 1 / (1 - a[state] * c[prev_state]);

					c[state] *= r;

					for (index_t x = 0; x < x_len; x++)
						d.template at<dim, 'x'>(dens_i, x) =
							r
							* (d.template at<dim, 'x'>(dens_i, x) - a[state] * d.template at<dim, 'x'>(prev_dens_i, x));
				}

				for (index_t equation_idx = blocks_count * 2 - 2; equation_idx >= 0; equation_idx--)
				{
					const auto [tid, i, dens_i] = get_i(equation_idx);
					const auto [next_tid, next_i, next_dens_i] = get_i(equation_idx + 1);

					const auto state = noarr::idx<'t', 'i'>(tid, i);

					for (index_t x = 0; x < x_len; x++)
						d.template at<dim, 'x'>(dens_i, x) =
							d.template at<dim, 'x'>(dens_i, x) - c[state] * d.template at<dim, 'x'>(next_dens_i, x);
				}


				// notify all threads that the last thread has finished
				counter.store(this_epoch_end, std::memory_order::release);
				// counter.notify_all();
			}
			else
			{
				// wait for the last thread to finish
				while (current_value < this_epoch_end)
				{
					// counter.wait(current_value, std::memory_order::relaxed);
					current_value = counter.load(std::memory_order::acquire);
				}
			}
		}

		// Final part of modified thomas algorithm
		// Solve the rest of the unknowns
		{
			const index_t block_start = block_idx * block_size;
			const index_t block_end = block_start + block_size;

			for (index_t i = block_start + 1; i < block_end - 1; i++)
			{
				const auto state = noarr::idx<'t', 'i'>(tid, i - block_start);

				for (index_t x = 0; x < x_len; x++)
					d.template at<dim, 'x'>(i, x) = d.template at<dim, 'x'>(i, x)
													- a[state] * d.template at<dim, 'x'>(block_start, x)
													- c[state] * d.template at<dim, 'x'>(block_end - 1, x);
			}
		}
	});
}

template <typename index_t, typename real_t, typename counter_t, typename density_layout_t, typename scratch_layout_t>
static void solve_slice_z_3d(real_t* __restrict__ densities, const real_t* __restrict__ ac,
							 const real_t* __restrict__ b1, real_t* __restrict__ a_data, real_t* __restrict__ c_data,
							 aligned_atomic<counter_t>* __restrict__ counters, const density_layout_t dens_l,
							 const scratch_layout_t scratch_l, const index_t block_size, const index_t x_tile_size)
{
	constexpr char dim = 'z';
	const index_t n = dens_l | noarr::get_length<dim>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();

	const index_t blocks_count = n / block_size;

	const auto tid = get_thread_num();
	const auto thread_count = get_num_threads();

	auto a = noarr::make_bag(scratch_l, a_data);
	auto c = noarr::make_bag(scratch_l, c_data);

	const index_t counters_count = (thread_count + blocks_count - 1) / blocks_count;

	const auto blocked_dens_l = dens_l ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(x_tile_size);
	const auto blocked_l = blocked_dens_l ^ noarr::into_blocks<'z', 'Z', 'l'>(block_size);
	const auto linearized = noarr::merge_blocks<'X', 'Z', 't'>() ^ noarr::merge_blocks<'y', 't', 't'>()
							^ noarr::merge_blocks<'s', 't', 't'>() ^ noarr::step<'t'>(tid, thread_count);

	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

	noarr::traverser(blocked_l).order(linearized).template for_dims<'t'>([=](auto state) {
		const index_t s = noarr::get_index<'s'>(state);
		const index_t y = noarr::get_index<'y'>(state);
		const index_t X = noarr::get_index<'X'>(state);
		const index_t block_idx = noarr::get_index<'Z'>(state);
		const std::size_t system_idx = s * y_len * X_len + y * X_len + X;

		auto d = noarr::make_bag(blocked_dens_l ^ noarr::fix<'s', 'y', 'X', 'b'>(s, y, X, 0), densities);

		const auto remainder = x_len % x_tile_size;
		const auto x_len_remainder = remainder == 0 ? x_tile_size : remainder;
		const auto x_len = X == X_len - 1 ? x_len_remainder : x_tile_size;

		// First part of modified thomas algorithm
		// We modify equations in blocks to have the first and the last equation form the tridiagonal system
		{
			const index_t block_start = block_idx * block_size;
			const index_t block_end = block_start + block_size;

			// Normalize the first and the second equation
			index_t i;
			for (i = block_start; i < block_start + 2; i++)
			{
				const auto state = noarr::idx<'t', 'i'>(tid, i - block_start);

				const auto a_tmp = ac[s] * (i == 0 ? 0 : 1);
				const auto b_tmp = b1[s] + (i == 0 || i == n - 1 ? ac[s] : 0);
				const auto c_tmp = ac[s] * (i == n - 1 ? 0 : 1);

				a[state] = a_tmp / b_tmp;
				c[state] = c_tmp / b_tmp;

				for (index_t x = 0; x < x_len; x++)
					d.template at<dim, 'x'>(i, x) /= b_tmp;
			}

			// Process the lower diagonal (forward)
			for (; i < block_end; i++)
			{
				const auto state = noarr::idx<'t', 'i'>(tid, i - block_start);
				const auto prev_state = noarr::idx<'t', 'i'>(tid, i - block_start - 1);

				const auto a_tmp = ac[s] * (i == 0 ? 0 : 1);
				const auto b_tmp = b1[s] + (i == n - 1 ? ac[s] : 0);
				const auto c_tmp = ac[s] * (i == n - 1 ? 0 : 1);

				const auto r = 1 / (b_tmp - a_tmp * c[prev_state]);

				a[state] = r * (0 - a_tmp * a[prev_state]);
				c[state] = r * c_tmp;

				for (index_t x = 0; x < x_len; x++)
					d.template at<dim, 'x'>(i, x) =
						r * (d.template at<dim, 'x'>(i, x) - a_tmp * d.template at<dim, 'x'>(i - 1, x));
			}

			// Process the upper diagonal (backward)
			for (i = block_end - 3; i >= block_start + 1; i--)
			{
				const auto state = noarr::idx<'t', 'i'>(tid, i - block_start);
				const auto next_state = noarr::idx<'t', 'i'>(tid, i - block_start + 1);

				for (index_t x = 0; x < x_len; x++)
					d.template at<dim, 'x'>(i, x) -= c[state] * d.template at<dim, 'x'>(i + 1, x);

				a[state] = a[state] - c[state] * a[next_state];
				c[state] = 0 - c[state] * c[next_state];
			}

			// Process the first row (backward)
			{
				const auto state = noarr::idx<'t', 'i'>(tid, i - block_start);
				const auto next_state = noarr::idx<'t', 'i'>(tid, i - block_start + 1);

				const auto r = 1 / (1 - c[state] * a[next_state]);

				for (index_t x = 0; x < x_len; x++)
					d.template at<dim, 'x'>(i, x) =
						r * (d.template at<dim, 'x'>(i, x) - c[state] * d.template at<dim, 'x'>(i + 1, x));

				a[state] = r * a[state];
				c[state] = r * (0 - c[state] * c[next_state]);
			}
		}

		{
			const counter_t visit_count = system_idx / counters_count;
			const counter_t last_epoch_end = visit_count * (blocks_count + 1);
			const counter_t this_epoch_end = (visit_count + 1) * (blocks_count + 1);

			auto& counter = counters[system_idx % counters_count].value;

			auto current_value = counter.load(std::memory_order::relaxed);

			// sleep if a thread is too much ahead
			while (current_value < last_epoch_end)
			{
				// counter.wait(current_value, std::memory_order::relaxed);
				current_value = counter.load(std::memory_order::relaxed);
			}

			// increment the counter
			current_value = counter.fetch_add(1, std::memory_order::relaxed);

			// Second part of modified thomas algorithm
			// We solve the system of equations that are composed of the first and last row of each block
			// We do it using Thomas Algorithm
			if (current_value == this_epoch_end - 2)
			{
				// bags for the middle phase
				auto get_i = [block_size, tid, thread_count, block_idx](index_t equation_idx) {
					const index_t equation_block_idx = equation_idx / 2;
					const index_t equation_block_offset = equation_idx % 2 * (block_size - 1);

					const index_t equation_tid = (tid + equation_block_idx - block_idx + thread_count) % thread_count;
					const index_t equation_i = equation_block_offset;
					const index_t equation_dens_i = equation_block_idx * block_size + equation_i;

					return std::make_tuple(equation_tid, equation_i, equation_dens_i);
				};

				for (index_t equation_idx = 1; equation_idx < blocks_count * 2; equation_idx++)
				{
					const auto [tid, i, dens_i] = get_i(equation_idx);
					const auto [prev_tid, prev_i, prev_dens_i] = get_i(equation_idx - 1);

					const auto state = noarr::idx<'t', 'i'>(tid, i);
					const auto prev_state = noarr::idx<'t', 'i'>(prev_tid, prev_i);

					const auto r = 1 / (1 - a[state] * c[prev_state]);

					c[state] *= r;

					for (index_t x = 0; x < x_len; x++)
						d.template at<dim, 'x'>(dens_i, x) =
							r
							* (d.template at<dim, 'x'>(dens_i, x) - a[state] * d.template at<dim, 'x'>(prev_dens_i, x));
				}

				for (index_t equation_idx = blocks_count * 2 - 2; equation_idx >= 0; equation_idx--)
				{
					const auto [tid, i, dens_i] = get_i(equation_idx);
					const auto [next_tid, next_i, next_dens_i] = get_i(equation_idx + 1);

					const auto state = noarr::idx<'t', 'i'>(tid, i);

					for (index_t x = 0; x < x_len; x++)
						d.template at<dim, 'x'>(dens_i, x) =
							d.template at<dim, 'x'>(dens_i, x) - c[state] * d.template at<dim, 'x'>(next_dens_i, x);
				}


				// notify all threads that the last thread has finished
				counter.store(this_epoch_end, std::memory_order::release);
				// counter.notify_all();
			}
			else
			{
				// wait for the last thread to finish
				while (current_value < this_epoch_end)
				{
					// counter.wait(current_value, std::memory_order::relaxed);
					current_value = counter.load(std::memory_order::acquire);
				}
			}
		}

		// Final part of modified thomas algorithm
		// Solve the rest of the unknowns
		{
			const index_t block_start = block_idx * block_size;
			const index_t block_end = block_start + block_size;

			for (index_t i = block_start + 1; i < block_end - 1; i++)
			{
				const auto state = noarr::idx<'t', 'i'>(tid, i - block_start);

				for (index_t x = 0; x < x_len; x++)
					d.template at<dim, 'x'>(i, x) = d.template at<dim, 'x'>(i, x)
													- a[state] * d.template at<dim, 'x'>(block_start, x)
													- c[state] * d.template at<dim, 'x'>(block_end - 1, x);
			}
		}
	});
}

template <typename real_t, bool aligned_x>
void blocked_thomas_solver_t<real_t, aligned_x>::solve_x()
{
	if (this->problem_.dims == 1)
	{
#pragma omp parallel num_threads(max_threadsx_)
		solve_slice_x_1d<index_t>(this->substrates_, ax_, b1x_, a_scratch_, c_scratch_, countersx_.get(),
								  get_substrates_layout<1>(), get_scratch_layout(max_threadsx_), block_size_);
	}
	else if (this->problem_.dims == 2)
	{
#pragma omp parallel num_threads(max_threadsx_)
		{
			solve_slice_x_2d_and_3d<index_t>(this->substrates_, ax_, b1x_, a_scratch_, c_scratch_, countersx_.get(),
											 get_substrates_layout<2>() ^ noarr::rename<'y', 'm'>(),
											 get_scratch_layout(max_threadsx_), block_size_);

#pragma omp for
			for (index_t i = 0; i < countersy_count_; i++)
			{
				countersy_[i].value.store(0, std::memory_order_relaxed);
			}
		}
	}
	else if (this->problem_.dims == 3)
	{
#pragma omp parallel num_threads(max_threadsx_)
		{
			solve_slice_x_2d_and_3d<index_t>(this->substrates_, ax_, b1x_, a_scratch_, c_scratch_, countersx_.get(),
											 get_substrates_layout<3>() ^ noarr::merge_blocks<'z', 'y', 'm'>(),
											 get_scratch_layout(max_threadsx_), block_size_);

#pragma omp for
			for (index_t i = 0; i < countersy_count_; i++)
			{
				countersy_[i].value.store(0, std::memory_order_relaxed);
			}
		}
	}
}

template <typename real_t, bool aligned_x>
void blocked_thomas_solver_t<real_t, aligned_x>::solve_y()
{
	if (this->problem_.dims == 2)
	{
#pragma omp parallel num_threads(max_threadsy_)
		{
			solve_slice_y_2d<index_t>(this->substrates_, ay_, b1y_, a_scratch_, c_scratch_, countersy_.get(),
									  get_substrates_layout<2>(), get_scratch_layout(max_threadsx_), block_size_,
									  x_tile_size_);
#pragma omp for
			for (index_t i = 0; i < countersx_count_; i++)
			{
				countersx_[i].value.store(0, std::memory_order_relaxed);
			}
		}
	}
	else if (this->problem_.dims == 3)
	{
#pragma omp parallel num_threads(max_threadsy_)
		{
			solve_slice_y_3d<index_t>(this->substrates_, ay_, b1y_, a_scratch_, c_scratch_, countersy_.get(),
									  get_substrates_layout<3>(), get_scratch_layout(max_threadsx_), block_size_,
									  x_tile_size_);
#pragma omp for
			for (index_t i = 0; i < countersz_count_; i++)
			{
				countersz_[i].value.store(0, std::memory_order_relaxed);
			}
		}
	}
}

template <typename real_t, bool aligned_x>
void blocked_thomas_solver_t<real_t, aligned_x>::solve_z()
{
#pragma omp parallel num_threads(max_threadsz_)
	{
		solve_slice_z_3d<index_t>(this->substrates_, az_, b1z_, a_scratch_, c_scratch_, countersz_.get(),
								  get_substrates_layout<3>(), get_scratch_layout(max_threadsz_), block_size_,
								  x_tile_size_);
#pragma omp for
		for (index_t i = 0; i < countersx_count_; i++)
		{
			countersx_[i].value.store(0, std::memory_order_relaxed);
		}
	}
}

template <typename real_t, bool aligned_x>
void blocked_thomas_solver_t<real_t, aligned_x>::solve()
{
// 	if (this->problem_.dims == 1)
// 	{
// #pragma omp parallel

// 		for (index_t i = 0; i < this->problem_.iterations; i++)
// 			solve_slice_x_1d<index_t>(this->substrates_, ax_, b1x_, a_scratch_, c_scratch_, countersx_.get(),
// 									  get_substrates_layout<1>(), get_scratch_layout(max_threadsx_), block_size_);
// 	}
// 	if (this->problem_.dims == 2)
// 	{
// #pragma omp parallel
// 		for (index_t i = 0; i < this->problem_.iterations; i++)
// 		{
// 			solve_slice_x_2d_and_3d<index_t>(this->substrates_, ax_, b1x_, a_scratch_, c_scratch_, countersx_.get(),
// 											 get_substrates_layout<3>() ^ noarr::merge_blocks<'z', 'y', 'm'>(),
// 											 get_scratch_layout(max_threadsx_), block_size_);
// #pragma omp barrier
// 			solve_slice_y_2d<index_t>(this->substrates_, ay_, b1y_, a_scratch_, c_scratch_, countersy_.get(),
// 									  get_substrates_layout<2>(), get_scratch_layout(max_threadsx_), block_size_,
// 									  x_tile_size_);
// 		}
// 	}
// 	if (this->problem_.dims == 3)
// 	{
// #pragma omp parallel
// 		for (index_t i = 0; i < this->problem_.iterations; i++)
// 		{
// 			solve_slice_x_2d_and_3d<index_t>(this->substrates_, ax_, b1x_, a_scratch_, c_scratch_, countersx_.get(),
// 											 get_substrates_layout<3>() ^ noarr::merge_blocks<'z', 'y', 'm'>(),
// 											 get_scratch_layout(max_threadsx_), block_size_);
// #pragma omp barrier
// 			solve_slice_y_3d<index_t>(this->substrates_, ay_, b1y_, a_scratch_, c_scratch_, countersy_.get(),
// 									  get_substrates_layout<3>(), get_scratch_layout(max_threadsx_), block_size_,
// 									  x_tile_size_);
// #pragma omp barrier
// 			solve_slice_z_3d<index_t>(this->substrates_, az_, b1z_, a_scratch_, c_scratch_, countersz_.get(),
// 									  get_substrates_layout<3>(), get_scratch_layout(max_threadsz_), block_size_,
// 									  x_tile_size_);
// 		}
// 	}
}

template <typename real_t, bool aligned_x>
blocked_thomas_solver_t<real_t, aligned_x>::blocked_thomas_solver_t()
	: ax_(nullptr),
	  b1x_(nullptr),
	  ay_(nullptr),
	  b1y_(nullptr),
	  az_(nullptr),
	  b1z_(nullptr),
	  a_scratch_(nullptr),
	  c_scratch_(nullptr)
{}

template <typename real_t, bool aligned_x>
blocked_thomas_solver_t<real_t, aligned_x>::~blocked_thomas_solver_t()
{
	if (b1x_)
	{
		std::free(ax_);
		std::free(b1x_);
	}
	if (b1y_)
	{
		std::free(ay_);
		std::free(b1y_);
	}
	if (b1z_)
	{
		std::free(az_);
		std::free(b1z_);
	}

	if (a_scratch_)
	{
		std::free(a_scratch_);
		std::free(c_scratch_);
	}
}

template class blocked_thomas_solver_t<float, false>;
template class blocked_thomas_solver_t<double, false>;

template class blocked_thomas_solver_t<float, true>;
template class blocked_thomas_solver_t<double, true>;
