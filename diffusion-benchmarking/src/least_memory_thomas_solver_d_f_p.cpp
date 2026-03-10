#include "least_memory_thomas_solver_d_f_p.h"

#include <cstddef>
#include <iostream>

#include "barrier.h"
#include "noarr/structures/extra/funcs.hpp"
#include "omp_helper.h"
#include "perf_utils.h"
#include "vector_transpose_helper.h"


template <typename real_t, bool aligned_x>
thread_id_t<typename least_memory_thomas_solver_d_f_p<real_t, aligned_x>::index_t> least_memory_thomas_solver_d_f_p<
	real_t, aligned_x>::get_thread_id() const
{
	thread_id_t<typename least_memory_thomas_solver_d_f_p<real_t, aligned_x>::index_t> id;

	const index_t tid = get_thread_num();
	const index_t group_size = cores_division_[1] * cores_division_[2];

	const index_t substrate_group_tid = tid % group_size;

	id.group = tid / group_size;

	id.x = substrate_group_tid % cores_division_[0];
	id.y = (substrate_group_tid / cores_division_[0]) % cores_division_[1];
	id.z = substrate_group_tid / (cores_division_[0] * cores_division_[1]);

	return id;
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_f_p<real_t, aligned_x>::set_block_bounds(index_t n, index_t group_size,
																		   index_t& block_size,
																		   std::vector<index_t>& group_block_lengths,
																		   std::vector<index_t>& group_block_offsets)
{
	if (group_size == 1)
	{
		block_size = n;
		group_block_lengths = { n };
		group_block_offsets = { 0 };
		return;
	}

	block_size = n / group_size;

	group_block_lengths.clear();
	group_block_offsets.clear();

	for (index_t i = 0; i < group_size; i++)
	{
		if (i < n % group_size)
			group_block_lengths.push_back(block_size + 1);
		else
			group_block_lengths.push_back(block_size);
	}

	group_block_offsets.resize(group_size);

	for (index_t i = 0; i < group_size; i++)
	{
		if (i == 0)
			group_block_offsets[i] = 0;
		else
			group_block_offsets[i] = group_block_offsets[i - 1] + group_block_lengths[i - 1];
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_f_p<real_t, aligned_x>::precompute_values(
	std::unique_ptr<real_t*[]>& a, std::unique_ptr<real_t*[]>& b, std::unique_ptr<real_t*[]>& c,
	std::unique_ptr<real_t*[]>& rf, std::unique_ptr<real_t*[]>& rb, index_t n, index_t shape, index_t dims,
	index_t counters_count, std::unique_ptr<std::unique_ptr<aligned_atomic<index_t>>[]>& counters,
	std::unique_ptr<std::unique_ptr<std::barrier<>>[]>& barriers, index_t group_size,
	const std::vector<index_t> group_block_lengths, const std::vector<index_t> group_block_offsets, char dim)
{
	a = std::make_unique<real_t*[]>(get_max_threads());
	b = std::make_unique<real_t*[]>(get_max_threads());
	c = std::make_unique<real_t*[]>(get_max_threads());
	rf = std::make_unique<real_t*[]>(get_max_threads());
	rb = std::make_unique<real_t*[]>(get_max_threads());

	counters = std::make_unique<std::unique_ptr<aligned_atomic<index_t>>[]>(counters_count);
	barriers = std::make_unique<std::unique_ptr<std::barrier<>>[]>(counters_count);

#pragma omp parallel
	{
		auto tid = get_thread_id();

		auto arrays_layout = noarr::scalar<real_t*>() ^ get_thread_distribution_layout();

		real_t*& a_t = arrays_layout | noarr::get_at<'y', 'z', 'g'>(a.get(), tid.y, tid.z, tid.group);
		real_t*& b_t = arrays_layout | noarr::get_at<'y', 'z', 'g'>(b.get(), tid.y, tid.z, tid.group);
		real_t*& c_t = arrays_layout | noarr::get_at<'y', 'z', 'g'>(c.get(), tid.y, tid.z, tid.group);
		real_t*& rf_t = arrays_layout | noarr::get_at<'y', 'z', 'g'>(rf.get(), tid.y, tid.z, tid.group);
		real_t*& rb_t = arrays_layout | noarr::get_at<'y', 'z', 'g'>(rb.get(), tid.y, tid.z, tid.group);

		auto diag_l = get_diagonal_layout(this->problem_, n);

		if (aligned_x)
		{
			a_t = (real_t*)std::aligned_alloc(alignment_size_, diag_l | noarr::get_size());
			b_t = (real_t*)std::aligned_alloc(alignment_size_, diag_l | noarr::get_size());
			c_t = (real_t*)std::aligned_alloc(alignment_size_, diag_l | noarr::get_size());
			rf_t = (real_t*)std::aligned_alloc(alignment_size_, diag_l | noarr::get_size());
			rb_t = (real_t*)std::aligned_alloc(alignment_size_, diag_l | noarr::get_size());
		}
		else
		{
			a_t = (real_t*)std::malloc(diag_l | noarr::get_size());
			b_t = (real_t*)std::malloc(diag_l | noarr::get_size());
			c_t = (real_t*)std::malloc(diag_l | noarr::get_size());
			rf_t = (real_t*)std::malloc(diag_l | noarr::get_size());
			rb_t = (real_t*)std::malloc(diag_l | noarr::get_size());
		}

		auto a_diag = noarr::make_bag(diag_l, a_t);
		auto b_diag = noarr::make_bag(diag_l, b_t);
		auto c_diag = noarr::make_bag(diag_l, c_t);
		auto rf_diag = noarr::make_bag(diag_l, rf_t);
		auto rb_diag = noarr::make_bag(diag_l, rb_t);

		for (index_t s = 0; s < this->problem_.substrates_count; s++)
		{
			for (std::size_t block_idx = 0; block_idx < group_block_lengths.size(); block_idx++)
			{
				index_t i_begin = group_block_offsets[block_idx];
				index_t i_end = i_begin + group_block_lengths[block_idx];

				real_t a = -this->problem_.dt * this->problem_.diffusion_coefficients[s] / (shape * shape);
				real_t b = 1 + this->problem_.decay_rates[s] * this->problem_.dt / dims - 2 * a;
				real_t b0 = 1 + this->problem_.decay_rates[s] * this->problem_.dt / dims - a;

				for (index_t i = i_begin; i < i_begin + 2; i++)
				{
					index_t global_i = i;
					real_t a_tmp = global_i == 0 ? 0 : a;
					real_t b_tmp = (global_i == 0 || global_i == n - 1) ? b0 : b;
					real_t c_tmp = global_i == n - 1 ? 0 : a;

					auto idx = noarr::idx<'s', 'i'>(s, i);

					a_diag[idx] = a_tmp;
					b_diag[idx] = b_tmp;
					c_diag[idx] = c_tmp;
					rf_diag[idx] = 0;

					// if (get_thread_num() == 0)
					// 	std::cout << dim << " " << i << " a: " << a_diag[idx] << " b: " << b_diag[idx]
					// 			  << " c: " << c_diag[idx] << " rf: " << rf_diag[idx] << std::endl;
				}

				for (index_t i = i_begin + 2; i < i_end; i++)
				{
					index_t global_i = i;
					real_t a_tmp = global_i == 0 ? 0 : a;
					real_t b_tmp = (global_i == 0 || global_i == n - 1) ? b0 : b;
					real_t c_tmp = global_i == n - 1 ? 0 : a;

					auto idx = noarr::idx<'s', 'i'>(s, i);
					auto prev_idx = noarr::idx<'s', 'i'>(s, i - 1);

					a_diag[idx] = a_tmp;
					b_diag[idx] = b_tmp;
					c_diag[idx] = c_tmp;

					rf_diag[idx] = a_diag[idx] / b_diag[prev_idx];

					// if (get_thread_num() == 0)
					// 	std::cout << dim << " " << i << " a: " << a_diag[idx] << " b: " << b_diag[idx]
					// 			  << " c: " << c_diag[idx] << " rf: " << rf_diag[idx];

					a_diag[idx] = -a_diag[prev_idx] * rf_diag[idx];
					b_diag[idx] = b_diag[idx] - c_diag[prev_idx] * rf_diag[idx];

					// if (get_thread_num() == 0)
					// 	std::cout << " a': " << a_diag[idx] << " b': " << b_diag[idx] << " c: " << c_diag[idx]
					// 			  << std::endl;
				}

				if (use_alt_blocked_)
				{
					rb_diag[noarr::idx<'s', 'i'>(s, i_begin)] = 0;
					rb_diag[noarr::idx<'s', 'i'>(s, i_end - 1)] = 0;

					for (index_t i = i_begin + 1; i < i_end - 1; i++)
					{
						auto idx = noarr::idx<'s', 'i'>(s, i);
						auto idx0 = noarr::idx<'s', 'i'>(s, i_begin);

						rb_diag[idx] = c_diag[idx0] / b_diag[idx];

						b_diag[idx0] = b_diag[idx0] - a_diag[idx] * rb_diag[idx];
						c_diag[idx0] = -c_diag[idx] * rb_diag[idx];

						// if (get_thread_num() == 0)
						// 	std::cout << dim << " " << i << " a'': " << a_diag[idx] << " b': " << b_diag[idx]
						// 			  << " c': " << c_diag[idx] << " rb: " << rb_diag[idx] << std::endl;
					}
				}
				else
				{
					for (index_t i = i_end - 1; i >= i_end - 2; i--)
					{
						auto idx = noarr::idx<'s', 'i'>(s, i);
						rb_diag[idx] = 0;
					}

					for (index_t i = i_end - 3; i >= i_begin; i--)
					{
						auto idx = noarr::idx<'s', 'i'>(s, i);
						auto prev_idx = noarr::idx<'s', 'i'>(s, i + 1);

						rb_diag[idx] = c_diag[idx] / b_diag[prev_idx];

						if (i != i_begin)
							a_diag[idx] = a_diag[idx] - a_diag[prev_idx] * rb_diag[idx];
						else
							b_diag[idx] = b_diag[idx] - a_diag[prev_idx] * rb_diag[idx];

						c_diag[idx] = -c_diag[prev_idx] * rb_diag[idx];

						// if (get_thread_num() == 0)
						// 	std::cout << dim << " " << i << " a'': " << a_diag[idx] << " b': " << b_diag[idx]
						// 			  << " c': " << c_diag[idx] << " rb: " << rb_diag[idx] << std::endl;
					}
				}

				for (index_t i = i_begin + 1; i < i_end - 1; i++)
				{
					auto idx = noarr::idx<'s', 'i'>(s, i);

					b_diag[idx] = 1 / b_diag[idx];

					// if (get_thread_num() == 0)
					// 	std::cout << dim << " " << i << " a'': " << a_diag[idx] << " b'': " << b_diag[idx]
					// 			  << " c': " << c_diag[idx] << std::endl;
				}
			}

			auto get_idx = [&](index_t equation_idx) {
				auto idx = equation_idx / 2;
				return group_block_offsets[idx] + (group_block_lengths[idx] - 1) * (equation_idx % 2);
			};

			index_t equations = group_block_lengths.size();

			for (index_t equation_idx = 1; equation_idx < equations * 2; equation_idx++)
			{
				index_t i = get_idx(equation_idx);
				index_t prev_i = get_idx(equation_idx - 1);

				auto idx = noarr::idx<'s', 'i'>(s, i);
				auto prev_idx = noarr::idx<'s', 'i'>(s, prev_i);

				a_diag[idx] /= b_diag[prev_idx];
				b_diag[idx] -= c_diag[prev_idx] * a_diag[idx];

				// if (get_thread_num() == 0)
				// 	std::cout << dim << " m " << i << " rf: " << a_diag[idx] << " b: " << b_diag[idx]
				// 			  << " c: " << c_diag[idx] << std::endl;
			}

			for (index_t equation_idx = 0; equation_idx < equations * 2; equation_idx++)
			{
				index_t i = get_idx(equation_idx);
				auto idx = noarr::idx<'s', 'i'>(s, i);

				b_diag[idx] = 1 / b_diag[idx];

				// if (get_thread_num() == 0)
				// 	std::cout << dim << " m " << i << " rf: " << a_diag[idx] << " b': " << b_diag[idx]
				// 			  << " c: " << c_diag[idx] << std::endl;
			}

			// for (index_t i = 0; i < n; i++)
			// {
			// 	auto idx = noarr::idx<'s', 'i'>(s, i);
			// 	if (get_thread_num() == 0)
			// 		std::cout << dim << " f " << i << " a: " << a_diag[idx] << " b: " << b_diag[idx]
			// 				  << " c: " << c_diag[idx] << " rf: " << rf_diag[idx] << " rb: " << rb_diag[idx]
			// 				  << std::endl;
			// }
		}

		index_t lane_id;
		if (dim == 'y')
			lane_id = tid.z + tid.group * cores_division_[2];
		else
			lane_id = tid.y + tid.group * cores_division_[1];

		index_t dim_id = dim == 'y' ? tid.y : tid.z;

		if (dim_id == 0)
		{
			counters[lane_id] = std::make_unique<aligned_atomic<index_t>>(0);
			barriers[lane_id] = std::make_unique<std::barrier<>>(group_size);
		}
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_f_p<real_t, aligned_x>::precompute_values(real_t*& rf, real_t*& b, real_t*& c,
																			index_t shape, index_t dims, index_t n)
{
	auto layout = get_diagonal_layout(this->problem_, n);

	if (aligned_x)
	{
		rf = (real_t*)std::aligned_alloc(alignment_size_, (layout | noarr::get_size()));
		b = (real_t*)std::aligned_alloc(alignment_size_, (layout | noarr::get_size()));
		c = (real_t*)std::aligned_alloc(alignment_size_, (layout | noarr::get_size()));
	}
	else
	{
		rf = (real_t*)std::malloc((layout | noarr::get_size()));
		b = (real_t*)std::malloc((layout | noarr::get_size()));
		c = (real_t*)std::malloc((layout | noarr::get_size()));
	}

	auto rf_diag = noarr::make_bag(layout, rf);
	auto b_diag = noarr::make_bag(layout, b);
	auto c_diag = noarr::make_bag(layout, c);

	for (index_t s = 0; s < this->problem_.substrates_count; s++)
	{
		real_t a = -this->problem_.dt * this->problem_.diffusion_coefficients[s] / (shape * shape);

		rf_diag.template at<'i', 's'>(0, s) = 0;
		b_diag.template at<'i', 's'>(0, s) = 1 + this->problem_.decay_rates[s] * this->problem_.dt / dims - a;
		c_diag.template at<'i', 's'>(0, s) = a;

		for (index_t i = 1; i < n; i++)
		{
			real_t b = 1 + this->problem_.decay_rates[s] * this->problem_.dt / dims - 2 * a;
			if (i == n - 1)
				b += a;

			rf_diag.template at<'i', 's'>(i, s) = a / b_diag.template at<'i', 's'>(i - 1, s);
			b_diag.template at<'i', 's'>(i, s) = b - a * rf_diag.template at<'i', 's'>(i, s);
			c_diag.template at<'i', 's'>(i, s) = a;
		}


		for (index_t i = 0; i < n; i++)
		{
			b_diag.template at<'i', 's'>(i, s) = 1 / b_diag.template at<'i', 's'>(i, s);
		}
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_f_p<real_t, aligned_x>::precompute_values(std::unique_ptr<real_t*[]>& rf,
																			std::unique_ptr<real_t*[]>& b,
																			std::unique_ptr<real_t*[]>& c,
																			index_t shape, index_t dims, index_t n)
{
	rf = std::make_unique<real_t*[]>(get_max_threads());
	b = std::make_unique<real_t*[]>(get_max_threads());
	c = std::make_unique<real_t*[]>(get_max_threads());

#pragma omp parallel
	{
		real_t*& rf_t = rf[get_thread_num()];
		real_t*& b_t = b[get_thread_num()];
		real_t*& c_t = c[get_thread_num()];

		precompute_values(rf_t, b_t, c_t, shape, dims, n);
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_f_p<real_t, aligned_x>::prepare(const max_problem_t& problem)
{
	this->problem_ = problems::cast<std::int32_t, real_t>(problem);

	if (this->problem_.dims == 2)
		cores_division_[2] = 1;

	cores_division_[0] = 1;

	if (partial_blocking_)
	{
		cores_division_[1] = 1;
		cores_division_[2] = 1;
	}

	set_block_bounds(this->problem_.ny, cores_division_[1], group_blocks_[1], group_block_lengthsy_,
					 group_block_offsetsy_);

	set_block_bounds(this->problem_.nz, cores_division_[2], group_blocks_[2], group_block_lengthsz_,
					 group_block_offsetsz_);

	{
		substrate_groups_ = get_max_threads() / (cores_division_[0] * cores_division_[1] * cores_division_[2]);

		auto ss_len = (this->problem_.substrates_count + substrate_step_ - 1) / substrate_step_;

		for (index_t group_id = 0; group_id < substrate_groups_; group_id++)
		{
			const auto [ss_begin, ss_end] = evened_work_distribution(ss_len, substrate_groups_, group_id);

			const auto s_begin = ss_begin * substrate_step_;
			const auto s_end = std::min(this->problem_.substrates_count, ss_end * substrate_step_);

			group_block_lengthss_.push_back(std::max(s_end - s_begin, 0));
			group_block_offsetss_.push_back(s_begin);
		}
	}

	if (use_thread_distributed_allocation_)
	{
		thread_substrate_array_ = std::make_unique<real_t*[]>(get_max_threads());

#pragma omp parallel
		{
			const auto tid = get_thread_id();

			if (group_block_lengthss_[tid.group] != 0)
			{
				auto arrays_layout = noarr::scalar<real_t*>() ^ get_thread_distribution_layout();

				real_t*& substrates_t =
					arrays_layout
					| noarr::get_at<'y', 'z', 'g'>(thread_substrate_array_.get(), tid.y, tid.z, tid.group);

				auto dens_t_l =
					get_blocked_substrate_layout(this->problem_.nx, group_block_lengthsy_[tid.y],
												 group_block_lengthsz_[tid.z], group_block_lengthss_[tid.group]);

				substrates_t = (real_t*)std::aligned_alloc(alignment_size_, (dens_t_l | noarr::get_size()));

				if (problem.gaussian_pulse)
				{
					omp_trav_for_each(noarr::traverser(dens_t_l), [&](auto state) {
						index_t s = noarr::get_index<'s'>(state) + group_block_offsetss_[tid.group];
						index_t x = noarr::get_index<'x'>(state);
						index_t y = noarr::get_index<'y'>(state) + group_block_offsetsy_[tid.y];
						index_t z = noarr::get_index<'z'>(state) + group_block_offsetsz_[tid.z];

						(dens_t_l | noarr::get_at(substrates_t, state)) =
							solver_utils::gaussian_analytical_solution(s, x, y, z, this->problem_);
					});
				}
				else
				{
					omp_trav_for_each(noarr::traverser(dens_t_l), [&](auto state) {
						auto s_idx = noarr::get_index<'s'>(state);

						(dens_t_l | noarr::get_at(substrates_t, state)) = problem.initial_conditions[s_idx];
					});
				}
			}
		}
	}
	else
	{
		auto substrates_layout = get_substrates_layout<3>();

		if (aligned_x)
			this->substrates_ = (real_t*)std::aligned_alloc(alignment_size_, (substrates_layout | noarr::get_size()));
		else
			this->substrates_ = (real_t*)std::malloc((substrates_layout | noarr::get_size()));

		// Initialize substrates
		solver_utils::initialize_substrate(substrates_layout, this->substrates_, this->problem_);
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_f_p<real_t, aligned_x>::tune(const nlohmann::json& params)
{
	alignment_size_ = params.contains("alignment_size") ? (std::size_t)params["alignment_size"] : 64;
	substrate_step_ =
		params.contains("substrate_step") ? (index_t)params["substrate_step"] : this->problem_.substrates_count;

	cores_division_ = params.contains("cores_division") ? (std::array<index_t, 3>)params["cores_division"]
														: std::array<index_t, 3> { 1, 2, 2 };

	{
		using simd_tag = hn::ScalableTag<real_t>;
		simd_tag d;
		std::size_t vector_length = hn::Lanes(d) * sizeof(real_t);
		alignment_size_ = std::max(alignment_size_, vector_length);
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_f_p<real_t, aligned_x>::initialize()
{
	if (this->problem_.dims >= 1)
	{
		precompute_values(thread_rf_x_, thread_bx_, thread_cx_, this->problem_.dx, this->problem_.dims,
						  this->problem_.nx);
		// else
		// 	precompute_values(ax_, b1x_, cx_, this->problem_.dx, this->problem_.dims, this->problem_.nx);
	}
	if (this->problem_.dims >= 2)
	{
		if (cores_division_[1] == 1)
		{
			precompute_values(thread_rf_y_, thread_by_, thread_cy_, this->problem_.dy, this->problem_.dims,
							  this->problem_.ny);
			// else
			// 	precompute_values(ay_, b1y_, cy_, this->problem_.dy, this->problem_.dims, this->problem_.ny);
		}
		else
		{
			countersy_count_ = cores_division_[0] * cores_division_[2] * substrate_groups_;

			precompute_values(thread_ay_, thread_by_, thread_cy_, thread_rf_y_, thread_rb_y_, this->problem_.ny,
							  this->problem_.dy, this->problem_.dims, countersy_count_, countersy_, barriersy_,
							  cores_division_[1], group_block_lengthsy_, group_block_offsetsy_, 'y');
			// else
			// 	precompute_values(ay_, b1y_, a_scratchy_, c_scratchy_, this->problem_.dy, this->problem_.dims,
			// 					  this->problem_.ny, countersy_count_, countersy_, barriersy_, cores_division_[1]);
		}
	}
	if (this->problem_.dims >= 3)
	{
		if (cores_division_[2] == 1)
		{
			precompute_values(thread_rf_z_, thread_bz_, thread_cz_, this->problem_.dz, this->problem_.dims,
							  this->problem_.nz);
			// else
			// 	precompute_values(az_, b1z_, cz_, this->problem_.dz, this->problem_.dims, this->problem_.nz);
		}
		else
		{
			countersz_count_ = cores_division_[0] * cores_division_[1] * substrate_groups_;

			precompute_values(thread_az_, thread_bz_, thread_cz_, thread_rf_z_, thread_rb_z_, this->problem_.nz,
							  this->problem_.dz, this->problem_.dims, countersz_count_, countersz_, barriersz_,
							  cores_division_[2], group_block_lengthsz_, group_block_offsetsz_, 'z');
			// else
			// 	precompute_values(az_, b1z_, a_scratchz_, c_scratchz_, this->problem_.dz, this->problem_.dims,
			// 					  this->problem_.nz, countersz_count_, countersz_, barriersz_, cores_division_[2]);
		}
	}
}

template <typename real_t, bool aligned_x>
auto least_memory_thomas_solver_d_f_p<real_t, aligned_x>::get_thread_distribution_layout() const
{
	return noarr::vectors<'y', 'z', 'g'>(cores_division_[1], cores_division_[2], substrate_groups_);
}

template <typename real_t, bool aligned_x>
auto least_memory_thomas_solver_d_f_p<real_t, aligned_x>::get_diagonal_layout(const problem_t<index_t, real_t>& problem,
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

template <typename index_t, typename vec_t, typename simd_tag, typename density_bag_t, typename diag_bag_t>
constexpr static vec_t z_forward_inside_y_blocked_vectorized(const density_bag_t d, vec_t data, simd_tag t,
															 const index_t y, const index_t z, const diag_bag_t rf)

{
	if (z < 2)
		return data;

	vec_t prev = hn::Load(t, &(d.template at<'y', 'z'>(y, z - 1)));

	auto idx = noarr::idx<'i'>(z);

	return hn::MulAdd(hn::Set(t, -rf[idx]), prev, data);
}

template <typename index_t, typename vec_t, typename simd_tag, typename density_bag_t, typename diag_bag_t>
constexpr static vec_t z_forward_inside_y_blocked_alt_vectorized(const density_bag_t d, vec_t data, simd_tag t,
																 const index_t y, const index_t z, const diag_bag_t rf,
																 const diag_bag_t rb)
{
	if (z == 0)
		return data;

	auto idx = noarr::idx<'i'>(z);

	vec_t prev = hn::Load(t, &(d.template at<'y', 'z'>(y, z - 1)));
	data = hn::MulAdd(hn::Set(t, -rf[idx]), prev, data);

	vec_t data0 = hn::Load(t, &(d.template at<'y', 'z'>(y, 0)));
	data0 = hn::MulAdd(hn::Set(t, -rb[idx]), data, data0);

	hn::Store(data0, t, &(d.template at<'y', 'z'>(y, 0)));

	return data;
}

template <typename vec_t, typename index_t, typename simd_tag, typename dens_bag_t, typename diag_bag_t>
constexpr static void z_backward_blocked_vectorized(const dens_bag_t d, const diag_bag_t rb, simd_tag t,
													const index_t simd_length, const index_t y_len, const index_t z_len)

{
	auto blocked_dens_l = d.structure() ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(simd_length);
	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

	for (index_t y = 0; y < y_len; y++)
	{
		for (index_t X = 0; X < X_len; X++)
		{
			const auto d_blocked =
				noarr::make_bag(blocked_dens_l ^ noarr::fix<'b', 'X', 'x'>(noarr::lit<0>, X, noarr::lit<0>), d.data());

			vec_t prev = hn::Load(t, &d_blocked.template at<'z', 'y'>(z_len - 2, y));

			for (index_t i = z_len - 3; i >= 0; i--)
			{
				auto idx = noarr::idx<'i'>(i);

				vec_t curr = hn::Load(t, &d_blocked.template at<'z', 'y'>(i, y));
				curr = hn::MulAdd(prev, hn::Set(t, -rb[idx]), curr);

				hn::Store(curr, t, &d_blocked.template at<'z', 'y'>(i, y));

				prev = curr;
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t, typename barrier_t>
constexpr static void synchronize_z_blocked(real_t* __restrict__ densities, const real_t* __restrict__ rf_data,
											const real_t* __restrict__ b_data, const real_t* __restrict__ c_data,
											const density_layout_t dens_l, const diagonal_layout_t diag_l,
											const index_t tid, const index_t coop_size, barrier_t& barrier)
{
	barrier.arrive();

	const auto d = noarr::make_bag(dens_l, densities);
	const auto rf = noarr::make_bag(diag_l, rf_data);
	const auto b = noarr::make_bag(diag_l, b_data);
	const auto c = noarr::make_bag(diag_l, c_data);

	const index_t x_len = d | noarr::get_length<'x'>();
	const index_t y_len = d | noarr::get_length<'y'>();
	const index_t n = d | noarr::get_length<'z'>();

	const index_t block_size = n / coop_size;

	const index_t block_size_y = y_len / coop_size;
	const index_t y_begin = tid * block_size_y + std::min(tid, y_len % coop_size);
	const index_t y_end = y_begin + block_size_y + ((tid < y_len % coop_size) ? 1 : 0);

	barrier.wait();

	// #pragma omp critical
	// 	std::cout << "Thread " << tid << " s_begin: " << s << " block_y_begin: " << y_begin << " block_y_end: " << y_end
	// 			  << " block_size: " << block_size_y << std::endl;

	auto get_i = [block_size, n, coop_size](index_t equation_idx) {
		const index_t block_idx = equation_idx / 2;
		const auto block_start = block_idx * block_size + std::min(block_idx, n % coop_size);
		const auto actual_block_size = (block_idx < n % coop_size) ? block_size + 1 : block_size;
		const auto i = block_start + (equation_idx % 2) * (actual_block_size - 1);
		return i;
	};

	for (index_t equation_idx = 1; equation_idx < coop_size * 2; equation_idx++)
	{
		const index_t i = get_i(equation_idx);
		const index_t prev_i = get_i(equation_idx - 1);
		const auto state = noarr::idx<'i'>(i);

		for (index_t y = y_begin; y < y_end; y++)
		{
			for (index_t x = 0; x < x_len; x++)
			{
				d.template at<'x', 'y', 'z'>(x, y, i) =
					d.template at<'x', 'y', 'z'>(x, y, i) - rf[state] * d.template at<'x', 'y', 'z'>(x, y, prev_i);
			}
		}
	}

	for (index_t y = y_begin; y < y_end; y++)
	{
		for (index_t x = 0; x < x_len; x++)
		{
			d.template at<'x', 'y', 'z'>(x, y, n - 1) *= b.template at<'i'>(n - 1);
		}
	}

	for (index_t equation_idx = coop_size * 2 - 2; equation_idx >= 0; equation_idx--)
	{
		const index_t i = get_i(equation_idx);
		const index_t next_i = get_i(equation_idx + 1);
		const auto state = noarr::idx<'i'>(i);

		for (index_t y = y_begin; y < y_end; y++)
		{
			for (index_t x = 0; x < x_len; x++)
			{
				d.template at<'x', 'y', 'z'>(x, y, i) =
					b[state]
					* (d.template at<'x', 'y', 'z'>(x, y, i) - c[state] * d.template at<'x', 'y', 'z'>(x, y, next_i));
			}
		}
	}

	barrier.arrive_and_wait();
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename thread_distribution_l, typename barrier_t>
constexpr static void synchronize_z_blocked_distributed(
	real_t** __restrict__ densities, const real_t* __restrict__ rf_data, const real_t* __restrict__ b_data,
	const real_t* __restrict__ c_data, const density_layout_t dens_l, const diagonal_layout_t diag_l,
	const thread_distribution_l dist_l, const index_t n, const index_t tid, const index_t coop_size, barrier_t& barrier)
{
	barrier.arrive();

	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	const auto rf = noarr::make_bag(diag_l, rf_data);
	const auto b = noarr::make_bag(diag_l, b_data);
	const auto c = noarr::make_bag(diag_l, c_data);

	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();

	const index_t block_size = n / coop_size;

	const index_t block_size_y = y_len / coop_size;
	const index_t y_begin = tid * block_size_y + std::min(tid, y_len % coop_size);
	const index_t y_end = y_begin + block_size_y + ((tid < y_len % coop_size) ? 1 : 0);

	// #pragma omp critical
	// 	std::cout << "Thread " << tid  << " block_begin: " << y_begin << " block_end: " << y_end
	// 			  << " block_size: " << block_size_y << std::endl;

	barrier.wait();

	auto get_i = [block_size, n, coop_size, dens_l](index_t equation_idx) {
		const index_t block_idx = equation_idx / 2;
		const auto block_start = block_idx * block_size + std::min(block_idx, n % coop_size);
		const auto actual_block_size = (block_idx < n % coop_size) ? block_size + 1 : block_size;
		const auto offset = (equation_idx % 2) * (actual_block_size - 1);
		const auto i = block_start + offset;
		return std::make_tuple(i, block_idx,
							   dens_l ^ noarr::set_length<'z'>(actual_block_size) ^ noarr::fix<'z'>(offset));
	};

	for (index_t y = y_begin; y < y_end; y++)
	{
		for (index_t equation_idx = 1; equation_idx < coop_size * 2; equation_idx++)
		{
			const auto [i, block_idx, curr_dens_l] = get_i(equation_idx);
			const auto [prev_i, prev_block_idx, prev_dens_l] = get_i(equation_idx - 1);
			const auto state = noarr::idx<'i'>(i);

			const auto d = noarr::make_bag(curr_dens_l, dist_l | noarr::get_at<'z'>(densities, block_idx));
			const auto prev_d = noarr::make_bag(prev_dens_l, dist_l | noarr::get_at<'z'>(densities, prev_block_idx));

			for (index_t x = 0; x < x_len; x += simd_length)
			{
				simd_t curr = hn::Load(t, &d.template at<'x', 'y'>(x, y));
				simd_t prev = hn::Load(t, &prev_d.template at<'x', 'y'>(x, y));

				curr = hn::MulAdd(prev, hn::Set(t, -rf[state]), curr);

				hn::Store(curr, t, &d.template at<'x', 'y'>(x, y));
			}
		}

		for (index_t x = 0; x < x_len; x += simd_length)
		{
			const auto [i, block_idx, curr_dens_l] = get_i(coop_size * 2 - 1);

			const auto d = noarr::make_bag(curr_dens_l, dist_l | noarr::get_at<'z'>(densities, block_idx));

			simd_t curr = hn::Load(t, &d.template at<'x', 'y'>(x, y));
			curr = hn::Mul(curr, hn::Set(t, b.template at<'i'>(i)));

			hn::Store(curr, t, &d.template at<'x', 'y'>(x, y));
		}

		for (index_t equation_idx = coop_size * 2 - 2; equation_idx >= 0; equation_idx--)
		{
			const auto [i, block_idx, curr_dens_l] = get_i(equation_idx);
			const auto [next_i, next_block_idx, next_dens_l] = get_i(equation_idx + 1);
			const auto state = noarr::idx<'i'>(i);

			const auto d = noarr::make_bag(curr_dens_l, dist_l | noarr::get_at<'z'>(densities, block_idx));
			const auto next_d = noarr::make_bag(next_dens_l, dist_l | noarr::get_at<'z'>(densities, next_block_idx));

			for (index_t x = 0; x < x_len; x += simd_length)
			{
				simd_t curr = hn::Load(t, &d.template at<'x', 'y'>(x, y));
				simd_t next = hn::Load(t, &next_d.template at<'x', 'y'>(x, y));

				curr = hn::MulAdd(next, hn::Set(t, -c[state]), curr);
				curr = hn::Mul(curr, hn::Set(t, b[state]));

				hn::Store(curr, t, &d.template at<'x', 'y'>(x, y));
			}
		}
	}

	barrier.arrive_and_wait();
}

template <typename vec_t, typename index_t, typename simd_tag, typename dens_bag_t, typename diag_bag_t>
constexpr static void z_blocked_end_vectorized(const dens_bag_t d, const diag_bag_t a, const diag_bag_t b,
											   const diag_bag_t c, simd_tag t, const index_t simd_length,
											   const index_t y_len, const index_t z_len)
{
	auto blocked_dens_l = d.structure() ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(simd_length);
	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

	for (index_t y = 0; y < y_len; y++)
	{
		for (index_t X = 0; X < X_len; X++)
		{
			const auto d_blocked =
				noarr::make_bag(blocked_dens_l ^ noarr::fix<'b', 'X', 'x'>(noarr::lit<0>, X, noarr::lit<0>), d.data());

			vec_t begins = hn::Load(t, &d_blocked.template at<'z', 'y'>(0, y));
			vec_t ends = hn::Load(t, &d_blocked.template at<'z', 'y'>(z_len - 1, y));

			for (index_t i = 1; i < z_len - 1; i++)
			{
				const auto state = noarr::idx<'i'>(i);

				vec_t data = hn::Load(t, &d_blocked.template at<'z', 'y'>(i, y));
				data = hn::MulAdd(begins, hn::Set(t, -a[state]), data);
				data = hn::MulAdd(ends, hn::Set(t, -c[state]), data);
				data = hn::Mul(data, hn::Set(t, b[state]));

				hn::Store(data, t, &d_blocked.template at<'z', 'y'>(i, y));
			}
		}
	}
}

template <typename vec_t, typename index_t, typename simd_tag, typename dens_bag_t, typename diag_bag_t>
constexpr static void z_blocked_alt_end_vectorized(const dens_bag_t d, const diag_bag_t a, const diag_bag_t b,
												   const diag_bag_t c, simd_tag t, const index_t simd_length,
												   const index_t y_len, const index_t z_len)
{
	auto blocked_dens_l = d.structure() ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(simd_length);
	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

	for (index_t y = 0; y < y_len; y++)
	{
		for (index_t X = 0; X < X_len; X++)
		{
			const auto d_blocked =
				noarr::make_bag(blocked_dens_l ^ noarr::fix<'b', 'X', 'x'>(noarr::lit<0>, X, noarr::lit<0>), d.data());

			vec_t begins = hn::Load(t, &d_blocked.template at<'z', 'y'>(0, y));
			vec_t prev = hn::Load(t, &d_blocked.template at<'z', 'y'>(z_len - 1, y));

			for (index_t i = z_len - 2; i >= 1; i--)
			{
				const auto state = noarr::idx<'i'>(i);

				vec_t data = hn::Load(t, &d_blocked.template at<'z', 'y'>(i, y));
				data = hn::MulAdd(begins, hn::Set(t, -a[state]), data);
				data = hn::MulAdd(prev, hn::Set(t, -c[state]), data);
				data = hn::Mul(data, hn::Set(t, b[state]));

				prev = data;

				hn::Store(data, t, &d_blocked.template at<'z', 'y'>(i, y));
			}
		}
	}
}

template <typename vec_t, typename index_t, typename simd_tag, typename dens_bag_t, typename diag_bag_t>
constexpr static void y_backward_blocked_vectorized(const dens_bag_t d, const diag_bag_t rb, simd_tag t,
													const index_t z, const index_t simd_length, const index_t y_len)

{
	auto blocked_dens_l = d.structure() ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(simd_length);
	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

	for (index_t X = 0; X < X_len; X++)
	{
		const auto d_blocked =
			noarr::make_bag(blocked_dens_l ^ noarr::fix<'b', 'X', 'x'>(noarr::lit<0>, X, noarr::lit<0>), d.data());

		vec_t prev = hn::Load(t, &d_blocked.template at<'z', 'y'>(z, y_len - 2));

		for (index_t i = y_len - 3; i >= 0; i--)
		{
			auto idx = noarr::idx<'i'>(i);

			vec_t curr = hn::Load(t, &d_blocked.template at<'z', 'y'>(z, i));
			curr = hn::MulAdd(prev, hn::Set(t, -rb[idx]), curr);

			hn::Store(curr, t, &d_blocked.template at<'z', 'y'>(z, i));

			prev = curr;

			// #pragma omp critical
			// 			std::cout << "b " << z << " " << i << " " << X * simd_length << " rb: " << rb[idx] << std::endl;
		}
	}
}


template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t, typename barrier_t>
constexpr static void synchronize_y_blocked(real_t* __restrict__ densities, const real_t* __restrict__ rf_data,
											const real_t* __restrict__ b_data, const real_t* __restrict__ c_data,
											const density_layout_t dens_l, const diagonal_layout_t diag_l,
											const index_t z, const index_t tid, const index_t coop_size,
											barrier_t& barrier)
{
	barrier.arrive();

	const auto d = noarr::make_bag(dens_l, densities);
	const auto rf = noarr::make_bag(diag_l, rf_data);
	const auto b = noarr::make_bag(diag_l, b_data);
	const auto c = noarr::make_bag(diag_l, c_data);

	const index_t x_len = d | noarr::get_length<'x'>();
	const index_t n = d | noarr::get_length<'y'>();

	const index_t block_size = n / coop_size;

	const index_t block_size_x = x_len / coop_size;
	const auto x_begin = tid * block_size_x + std::min(tid, x_len % coop_size);
	const auto x_end = x_begin + block_size_x + ((tid < x_len % coop_size) ? 1 : 0);

	// #pragma omp critical
	// 	std::cout << "Thread " << tid << " block_begin: " << x_begin << " block_end: " << x_end
	// 			  << " block_size: " << block_size_x << std::endl;

	barrier.wait();

	auto get_i = [block_size, n, coop_size](index_t equation_idx) {
		const index_t block_idx = equation_idx / 2;
		const auto block_start = block_idx * block_size + std::min(block_idx, n % coop_size);
		const auto actual_block_size = (block_idx < n % coop_size) ? block_size + 1 : block_size;
		const auto i = block_start + (equation_idx % 2) * (actual_block_size - 1);
		return i;
	};

	for (index_t equation_idx = 1; equation_idx < coop_size * 2; equation_idx++)
	{
		const index_t i = get_i(equation_idx);
		const index_t prev_i = get_i(equation_idx - 1);
		const auto state = noarr::idx<'i'>(i);

		for (index_t x = x_begin; x < x_end; x++)
		{
			d.template at<'x', 'y', 'z'>(x, i, z) =
				d.template at<'x', 'y', 'z'>(x, i, z) - rf[state] * d.template at<'x', 'y', 'z'>(x, prev_i, z);

			// #pragma omp critical
			// 			std::cout << "mf " << z << " " << i << " " << x << " rf: " << rf[state]
			// 					  << " d: " << d.template at<'x', 'z', 'y'>(x, z, i) << std::endl;
		}
	}

	for (index_t x = x_begin; x < x_end; x++)
	{
		d.template at<'x', 'y', 'z'>(x, n - 1, z) *= b.template at<'i'>(n - 1);
	}

	for (index_t equation_idx = coop_size * 2 - 2; equation_idx >= 0; equation_idx--)
	{
		const index_t i = get_i(equation_idx);
		const index_t next_i = get_i(equation_idx + 1);
		const auto state = noarr::idx<'i'>(i);

		for (index_t x = x_begin; x < x_end; x++)
		{
			d.template at<'x', 'y', 'z'>(x, i, z) =
				b[state]
				* (d.template at<'x', 'y', 'z'>(x, i, z) - c[state] * d.template at<'x', 'y', 'z'>(x, next_i, z));

			// #pragma omp critical
			// 			std::cout << "mb " << z << " " << i << " " << x << " b: " << b[state] << " c: " << c[state]
			// 					  << " d: " << d.template at<'x', 'z', 'y'>(x, z, i) << std::endl;
		}
	}

	barrier.arrive_and_wait();
}


template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename thread_distribution_l, typename barrier_t>
constexpr static void synchronize_y_blocked_distributed(
	real_t** __restrict__ densities, const real_t* __restrict__ rf_data, const real_t* __restrict__ b_data,
	const real_t* __restrict__ c_data, const density_layout_t dens_l, const diagonal_layout_t diag_l,
	const thread_distribution_l dist_l, const index_t n, const index_t z, const index_t tid, const index_t coop_size,
	barrier_t& barrier)
{
	barrier.arrive();

	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	const auto rf = noarr::make_bag(diag_l, rf_data);
	const auto b = noarr::make_bag(diag_l, b_data);
	const auto c = noarr::make_bag(diag_l, c_data);

	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t x_simd_len = (x_len + simd_length - 1) / simd_length;

	const index_t block_size = n / coop_size;

	const index_t block_size_x = x_simd_len / coop_size;
	const auto x_simd_begin = tid * block_size_x + std::min(tid, x_simd_len % coop_size);
	const auto x_simd_end = x_simd_begin + block_size_x + ((tid < x_simd_len % coop_size) ? 1 : 0);

	// #pragma omp critical
	// 	std::cout << "Thread " << tid << " block_begin: " << x_begin << " block_end: " << x_end
	// 			  << " block_size: " << block_size_x << std::endl;

	barrier.wait();

	auto get_i = [block_size, n, coop_size, dens_l](index_t equation_idx) {
		const index_t block_idx = equation_idx / 2;
		const auto block_start = block_idx * block_size + std::min(block_idx, n % coop_size);
		const auto actual_block_size = (block_idx < n % coop_size) ? block_size + 1 : block_size;
		const auto offset = (equation_idx % 2) * (actual_block_size - 1);
		const auto i = block_start + offset;
		return std::make_tuple(i, block_idx,
							   dens_l ^ noarr::set_length<'y'>(actual_block_size) ^ noarr::fix<'y'>(offset));
	};

	for (index_t x = x_simd_begin; x < x_simd_end; x++)
	{
		simd_t prev;

		{
			const auto [prev_i, prev_block_idx, prev_dens_l] = get_i(0);
			const auto prev_d = noarr::make_bag(prev_dens_l, dist_l | noarr::get_at<'y'>(densities, prev_block_idx));

			prev = hn::Load(t, &prev_d.template at<'x', 'z'>(x * simd_length, z));
		}

		for (index_t equation_idx = 1; equation_idx < coop_size * 2; equation_idx++)
		{
			const auto [i, block_idx, curr_dens_l] = get_i(equation_idx);
			const auto state = noarr::idx<'i'>(i);

			const auto d = noarr::make_bag(curr_dens_l, dist_l | noarr::get_at<'y'>(densities, block_idx));

			simd_t curr = hn::Load(t, &d.template at<'x', 'z'>(x * simd_length, z));

			curr = hn::MulAdd(prev, hn::Set(t, -rf[state]), curr);

			hn::Store(curr, t, &d.template at<'x', 'z'>(x * simd_length, z));

			prev = curr;

			// #pragma omp critical
			// 			std::cout << "mf " << z << " " << i << " " << x << " rf: " << rf[state]
			// 					  << " d: " << d.template at<'x', 'z', 'y'>(x, z, i) << std::endl;
		}

		{
			const auto [i, block_idx, curr_dens_l] = get_i(coop_size * 2 - 1);

			const auto d = noarr::make_bag(curr_dens_l, dist_l | noarr::get_at<'y'>(densities, block_idx));

			prev = hn::Mul(prev, hn::Set(t, b.template at<'i'>(i)));

			hn::Store(prev, t, &d.template at<'x', 'z'>(x * simd_length, z));
		}

		for (index_t equation_idx = coop_size * 2 - 2; equation_idx >= 0; equation_idx--)
		{
			const auto [i, block_idx, curr_dens_l] = get_i(equation_idx);
			const auto state = noarr::idx<'i'>(i);

			const auto d = noarr::make_bag(curr_dens_l, dist_l | noarr::get_at<'y'>(densities, block_idx));

			simd_t curr = hn::Load(t, &d.template at<'x', 'z'>(x * simd_length, z));

			curr = hn::MulAdd(prev, hn::Set(t, -c[state]), curr);
			curr = hn::Mul(curr, hn::Set(t, b[state]));

			hn::Store(curr, t, &d.template at<'x', 'z'>(x * simd_length, z));

			prev = curr;

			// #pragma omp critical
			// 			std::cout << "mb " << z << " " << i << " " << x << " b: " << b[state] << " c: " << c[state]
			// 					  << " d: " << d.template at<'x', 'z', 'y'>(x, z, i) << std::endl;
		}
	}

	barrier.arrive_and_wait();
}

template <typename vec_t, typename index_t, typename simd_tag, typename dens_bag_t, typename diag_bag_t,
		  typename z_func_t>
constexpr static void y_blocked_end_vectorized(const dens_bag_t d, const diag_bag_t a, const diag_bag_t b,
											   const diag_bag_t c, simd_tag t, const index_t z,
											   const index_t simd_length, const index_t y_len, z_func_t&& z_forward)
{
	auto blocked_dens_l = d.structure() ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(simd_length);
	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

	for (index_t X = 0; X < X_len; X++)
	{
		const auto d_blocked =
			noarr::make_bag(blocked_dens_l ^ noarr::fix<'b', 'X', 'x'>(noarr::lit<0>, X, noarr::lit<0>), d.data());

		vec_t begins = hn::Load(t, &d_blocked.template at<'z', 'y'>(z, 0));
		vec_t ends = hn::Load(t, &d_blocked.template at<'z', 'y'>(z, y_len - 1));

		for (index_t i = 1; i < y_len - 1; i++)
		{
			const auto state = noarr::idx<'i'>(i);

			vec_t data = hn::Load(t, &d_blocked.template at<'z', 'y'>(z, i));
			data = hn::MulAdd(begins, hn::Set(t, -a[state]), data);
			data = hn::MulAdd(ends, hn::Set(t, -c[state]), data);
			data = hn::Mul(data, hn::Set(t, b[state]));

			data = z_forward(data, d_blocked, i);

			hn::Store(data, t, &d_blocked.template at<'z', 'y'>(z, i));

			// #pragma omp critical
			// 			std::cout << "f " << z << " " << i << " " << X * simd_length << " a: " << a[state] << " b: " <<
			// b[state]
			// 					  << " c: " << c[state] << std::endl;
		}

		begins = z_forward(begins, d_blocked, 0);
		hn::Store(begins, t, &d_blocked.template at<'z', 'y'>(z, 0));

		ends = z_forward(ends, d_blocked, y_len - 1);
		hn::Store(ends, t, &d_blocked.template at<'z', 'y'>(z, y_len - 1));
	}
}

template <typename vec_t, typename index_t, typename simd_tag, typename dens_bag_t, typename diag_bag_t,
		  typename z_func_t>
constexpr static void y_blocked_alt_end_vectorized(const dens_bag_t d, const diag_bag_t a, const diag_bag_t b,
												   const diag_bag_t c, simd_tag t, const index_t z,
												   const index_t simd_length, const index_t y_len, z_func_t&& z_forward)
{
	auto blocked_dens_l = d.structure() ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(simd_length);
	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

	for (index_t X = 0; X < X_len; X++)
	{
		const auto d_blocked =
			noarr::make_bag(blocked_dens_l ^ noarr::fix<'b', 'X', 'x'>(noarr::lit<0>, X, noarr::lit<0>), d.data());

		vec_t begins = hn::Load(t, &d_blocked.template at<'z', 'y'>(z, 0));
		vec_t prev = hn::Load(t, &d_blocked.template at<'z', 'y'>(z, y_len - 1));

		for (index_t i = y_len - 2; i >= 1; i--)
		{
			const auto state = noarr::idx<'i'>(i);

			vec_t data = hn::Load(t, &d_blocked.template at<'z', 'y'>(z, i));
			data = hn::MulAdd(begins, hn::Set(t, -a[state]), data);
			data = hn::MulAdd(prev, hn::Set(t, -c[state]), data);
			data = hn::Mul(data, hn::Set(t, b[state]));

			prev = z_forward(prev, d_blocked, i + 1);

			hn::Store(prev, t, &d_blocked.template at<'z', 'y'>(z, i + 1));

			prev = data;

			// #pragma omp critical
			// 			std::cout << "f " << z << " " << i << " " << X * simd_length << " a: " << a[state] << " b: " <<
			// b[state]
			// 					  << " c: " << c[state] << std::endl;
		}

		prev = z_forward(prev, d_blocked, 1);
		hn::Store(prev, t, &d_blocked.template at<'z', 'y'>(z, 1));

		begins = z_forward(begins, d_blocked, 0);
		hn::Store(begins, t, &d_blocked.template at<'z', 'y'>(z, 0));
	}
}

template <typename index_t, typename real_t, typename density_bag_t, typename diag_bag_t>
constexpr static real_t y_forward_inside_x(const density_bag_t d, const index_t z, const index_t y, const index_t x,
										   real_t data, const diag_bag_t rf)

{
	real_t prev;

	if (y == 0)
		prev = 0;
	else
		prev = d.template at<'z', 'x', 'y'>(z, x, y - 1);

	return data - rf.template at<'i'>(y) * prev;
}

template <typename index_t, typename real_t, typename density_bag_t, typename diag_bag_t>
constexpr static real_t y_forward_inside_x_blocked(const density_bag_t d, const index_t z, const index_t y,
												   const index_t x, real_t data, const diag_bag_t rf)

{
	if (y < 2)
		return data;

	real_t prev = d.template at<'z', 'x', 'y'>(z, x, y - 1);

	// #pragma omp critical
	// 	std::cout << "f " << z << " " << y << " " << x << " rf: " << rf.template at<'i'>(y) << std::endl;

	return data - rf.template at<'i'>(y) * prev;
}

template <typename index_t, typename real_t, typename density_bag_t, typename diag_bag_t>
constexpr static real_t y_forward_inside_x_blocked_alt(const density_bag_t d, const index_t z, const index_t y,
													   const index_t x, real_t data, const diag_bag_t rf,
													   const diag_bag_t rb)

{
	if (y == 0)
		return data;

	real_t prev = d.template at<'z', 'x', 'y'>(z, x, y - 1);
	data = data - rf.template at<'i'>(y) * prev;

	real_t data0 = d.template at<'z', 'x', 'y'>(z, x, 0);
	data0 = data0 - rb.template at<'i'>(y) * data;

	d.template at<'z', 'x', 'y'>(z, x, 0) = data0;

	// #pragma omp critical
	// 	std::cout << "f " << z << " " << y << " " << x << " rf: " << rf.template at<'i'>(y) << std::endl;

	return data;
}

template <typename real_t, typename index_t, typename density_bag_t, typename diag_bag_t>
constexpr static void x_forward(const density_bag_t d, const index_t z, const index_t y, const diag_bag_t rf)

{
	const index_t x_len = d | noarr::get_length<'x'>();

	real_t prev = d.template at<'z', 'y', 'x'>(z, y, 0);

	for (index_t i = 1; i < x_len; i++)
	{
		real_t curr = d.template at<'z', 'y', 'x'>(z, y, i);
		curr = curr - rf.template at<'i'>(i) * prev;
		d.template at<'z', 'y', 'x'>(z, y, i) = curr;

		prev = curr;
	}
}

template <typename real_t, typename index_t, typename density_bag_t, typename diag_bag_t, typename y_func_t>
constexpr static void x_backward(const density_bag_t d, const diag_bag_t b, const diag_bag_t c, const index_t z,
								 const index_t y, y_func_t&& y_forward)

{
	const index_t x_len = d | noarr::get_length<'x'>();

	real_t prev = d.template at<'z', 'x', 'y'>(z, x_len - 1, y);

	prev *= b.template at<'i'>(x_len - 1);

	for (index_t i = x_len - 2; i >= 0; i--)
	{
		auto idx = noarr::idx<'i'>(i);
		real_t curr = d.template at<'z', 'x', 'y'>(z, i, y);
		curr = (curr - c[idx] * prev) * b[idx];

		prev = y_forward(i + 1, y, prev);

		d.template at<'z', 'x', 'y'>(z, i + 1, y) = prev;

		prev = curr;
	}

	prev = y_forward(0, y, prev);

	d.template at<'z', 'x', 'y'>(z, 0, y) = prev;
}

template <typename index_t, typename vec_t, typename simd_tag, typename diagonal_bag_t>
constexpr static vec_t x_forward_vectorized(simd_tag, const index_t, const diagonal_bag_t, vec_t& last)
{
	return last;
}

template <typename index_t, typename vec_t, typename simd_tag, typename diagonal_bag_t, typename... vec_pack_t>
constexpr static vec_t x_forward_vectorized(simd_tag t, const index_t x, const diagonal_bag_t rf, vec_t& prev,
											vec_t& first, vec_pack_t&&... vec_pack)
{
	first = hn::MulAdd(prev, hn::Set(t, -rf.template at<'i'>(x)), first);
	return x_forward_vectorized(t, x + 1, rf, first, std::forward<vec_pack_t>(vec_pack)...);
}

template <typename index_t, typename vec_t, typename simd_tag, typename diagonal_bag_t>
constexpr static void x_forward_vectorized(simd_tag, const index_t, const index_t, const diagonal_bag_t, vec_t&)
{}

template <typename index_t, typename vec_t, typename simd_tag, typename diagonal_bag_t, typename... vec_pack_t>
constexpr static void x_forward_vectorized(simd_tag t, const index_t length, const index_t x, const diagonal_bag_t rf,
										   vec_t& prev, vec_t& first, vec_pack_t&&... vec_pack)
{
	if (length == 0)
		return;

	first = hn::MulAdd(prev, hn::Set(t, -rf.template at<'i'>(x)), first);
	x_forward_vectorized(t, length - 1, x + 1, rf, first, std::forward<vec_pack_t>(vec_pack)...);
}

template <typename index_t, typename vec_t, typename simd_tag, typename diagonal_bag_t>
constexpr static void x_backward_vectorized(const diagonal_bag_t, const diagonal_bag_t, simd_tag, const index_t, vec_t&)
{}

template <typename index_t, typename vec_t, typename simd_tag, typename diagonal_bag_t, typename... vec_pack_t>
constexpr static vec_t x_backward_vectorized(const diagonal_bag_t b, const diagonal_bag_t c, simd_tag t,
											 const index_t x, vec_t& first, vec_t& prev, vec_pack_t&&... vec_pack)
{
	x_backward_vectorized(b, c, t, x + 1, prev, std::forward<vec_pack_t>(vec_pack)...);
	first = hn::Mul(hn::MulAdd(prev, hn::Set(t, -c.template at<'i'>(x)), first), hn::Set(t, b.template at<'i'>(x)));
	return first;
}

template <typename index_t, typename vec_t, typename simd_tag, typename diagonal_bag_t>
constexpr static void x_backward_vectorized(const diagonal_bag_t, const diagonal_bag_t, simd_tag, const index_t,
											const index_t, vec_t&)
{}

template <typename index_t, typename vec_t, typename simd_tag, typename diagonal_bag_t, typename... vec_pack_t>
constexpr static vec_t x_backward_vectorized(const diagonal_bag_t b, const diagonal_bag_t c, simd_tag t,
											 const index_t length, const index_t x, vec_t& first, vec_t& prev,
											 vec_pack_t&&... vec_pack)
{
	if (length == 0)
		return first;

	x_backward_vectorized(b, c, t, length - 1, x + 1, prev, std::forward<vec_pack_t>(vec_pack)...);
	first = hn::Mul(hn::MulAdd(prev, hn::Set(t, -c.template at<'i'>(x)), first), hn::Set(t, b.template at<'i'>(x)));
	return first;
}


template <typename index_t, typename vec_t, typename simd_tag, typename dens_bag_t, typename diag_bag_t>
constexpr static vec_t z_forward_inside_y_vectorized(const dens_bag_t d, simd_tag t, const index_t y, const index_t z,
													 vec_t data, const diag_bag_t rf)
{
	vec_t prev;

	if (z == 0)
		prev = hn::Zero(t);
	else
		prev = hn::Load(t, &(d.template at<'y', 'z'>(y, z - 1)));

	auto idx = noarr::idx<'i'>(z);

	return hn::MulAdd(hn::Set(t, -rf[idx]), prev, data);
}

template <typename vec_t, typename index_t, typename simd_tag, typename dens_bag_t, typename diag_bag_t>
constexpr static void z_backward_vectorized(const dens_bag_t d, const diag_bag_t b, const diag_bag_t c, simd_tag t,
											const index_t simd_length, const index_t y_len, const index_t z_len)
{
	auto blocked_dens_l = d.structure() ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(simd_length);
	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

	for (index_t y = 0; y < y_len; y++)
	{
		for (index_t X = 0; X < X_len; X++)
		{
			const auto d_blocked = noarr::make_bag(
				blocked_dens_l ^ noarr::fix<'y', 'b', 'X', 'x'>(y, noarr::lit<0>, X, noarr::lit<0>), d.data());

			vec_t prev = hn::Load(t, &d_blocked.template at<'z'>(z_len - 1));

			prev = hn::Mul(prev, hn::Set(t, b.template at<'i'>(z_len - 1)));
			hn::Store(prev, t, &d_blocked.template at<'z'>(z_len - 1));

			for (index_t i = z_len - 2; i >= 0; i--)
			{
				auto idx = noarr::idx<'i'>(i);

				vec_t curr = hn::Load(t, &d_blocked.template at<'z'>(i));
				curr = hn::Mul(hn::MulAdd(prev, hn::Set(t, -c[idx]), curr), hn::Set(t, b[idx]));
				hn::Store(curr, t, &d_blocked.template at<'z'>(i));

				prev = curr;
			}
		}
	}
}

template <typename index_t, typename simd_tag, typename diag_bag_t, typename... vec_pack_t>
constexpr static void y_forward_inside_x_vectorized_inner(simd_tag, const index_t, const diag_bag_t, hn::Vec<simd_tag>&)
{}

template <typename index_t, typename simd_tag, typename diag_bag_t, typename... vec_pack_t>
constexpr static void y_forward_inside_x_vectorized_inner(simd_tag t, const index_t y, const diag_bag_t rf,
														  hn::Vec<simd_tag>& prev, hn::Vec<simd_tag>& first,
														  vec_pack_t&&... rows)
{
	auto idx = noarr::idx<'i'>(y);

	first = hn::MulAdd(hn::Set(t, -rf[idx]), prev, first);

	y_forward_inside_x_vectorized_inner(t, y + 1, rf, first, std::forward<vec_pack_t>(rows)...);
}

template <typename index_t, typename simd_tag, typename density_bag_t, typename diag_bag_t, typename... vec_pack_t>
constexpr static void y_forward_inside_x_vectorized(const density_bag_t d, simd_tag t, const index_t z,
													const index_t y_offset, const index_t x, const diag_bag_t rf,
													vec_pack_t&&... rows)
{
	hn::Vec<simd_tag> prev;

	if (y_offset == 0)
		prev = hn::Zero(t);
	else
		prev = hn::Load(t, &(d.template at<'z', 'y', 'x'>(z, y_offset - 1, x)));

	y_forward_inside_x_vectorized_inner(t, y_offset, rf, prev, std::forward<vec_pack_t>(rows)...);
}

template <typename index_t, typename simd_tag, typename density_bag_t, typename diag_bag_t, typename... vec_pack_t>
constexpr static void y_forward_inside_x_blocked_vectorized(const density_bag_t d, simd_tag t, const index_t z,
															const index_t y_offset, const index_t x,
															const diag_bag_t rf, hn::Vec<simd_tag>& first,
															vec_pack_t&&... rows)
{
	const index_t y_begin = std::max(y_offset, 2);

	if (y_begin == y_offset)
	{
		auto prev = hn::Load(t, &(d.template at<'z', 'y', 'x'>(z, y_begin - 1, x)));
		y_forward_inside_x_vectorized_inner(t, y_begin, rf, prev, first, std::forward<vec_pack_t>(rows)...);
	}
	else
	{
		y_forward_inside_x_vectorized_inner(t, y_begin, rf, std::forward<vec_pack_t>(rows)...);
	}
}

template <typename index_t, typename vec_t, typename simd_tag, typename diag_bag_t, typename... vec_pack_t>
constexpr static void y_forward_inside_x_blocked_alt_vectorized_inner(simd_tag, const index_t, const diag_bag_t,
																	  const diag_bag_t, vec_t&, vec_t&)
{}

template <typename index_t, typename vec_t, typename simd_tag, typename diag_bag_t, typename... vec_pack_t>
constexpr static void y_forward_inside_x_blocked_alt_vectorized_inner(simd_tag t, const index_t y, const diag_bag_t rf,
																	  const diag_bag_t rb, vec_t& data0, vec_t& prev,
																	  vec_t& first, vec_pack_t&&... rows)
{
	auto idx = noarr::idx<'i'>(y);

	first = hn::MulAdd(hn::Set(t, -rf[idx]), prev, first);
	data0 = hn::MulAdd(hn::Set(t, -rb[idx]), first, data0);

	y_forward_inside_x_blocked_alt_vectorized_inner(t, y + 1, rf, rb, data0, first, std::forward<vec_pack_t>(rows)...);
}

template <typename index_t, typename simd_tag, typename density_bag_t, typename diag_bag_t, typename... vec_pack_t>
constexpr static void y_forward_inside_x_blocked_alt_vectorized(const density_bag_t d, simd_tag t, const index_t z,
																const index_t y_offset, const index_t x,
																const diag_bag_t rf, const diag_bag_t rb,
																hn::Vec<simd_tag>& first, vec_pack_t&&... rows)
{
	const index_t y_begin = std::max(y_offset, 1);

	if (y_offset != 0)
	{
		auto prev = hn::Load(t, &(d.template at<'z', 'y', 'x'>(z, y_begin - 1, x)));
		auto data0 = hn::Load(t, &(d.template at<'z', 'y', 'x'>(z, 0, x)));

		y_forward_inside_x_blocked_alt_vectorized_inner(t, y_begin, rf, rb, data0, prev, first,
														std::forward<vec_pack_t>(rows)...);

		hn::Store(data0, t, &(d.template at<'z', 'y', 'x'>(z, 0, x)));
	}
	else
	{
		y_forward_inside_x_blocked_alt_vectorized_inner(t, y_begin, rf, rb, first, first,
														std::forward<vec_pack_t>(rows)...);
	}
}

template <typename vec_t, typename index_t, typename simd_tag, typename dens_bag_t, typename diag_bag_t,
		  typename z_func_t>
constexpr static void y_backward_vectorized(const dens_bag_t d, const diag_bag_t b, const diag_bag_t c, simd_tag t,
											const index_t z, const index_t simd_length, const index_t y_len,
											z_func_t&& z_forward)
{
	auto blocked_dens_l = d.structure() ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(simd_length);
	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

	for (index_t X = 0; X < X_len; X++)
	{
		const auto d_blocked =
			noarr::make_bag(blocked_dens_l ^ noarr::fix<'b', 'X', 'x'>(noarr::lit<0>, X, noarr::lit<0>), d.data());

		vec_t prev = hn::Load(t, &d_blocked.template at<'z', 'y'>(z, y_len - 1));

		prev = hn::Mul(prev, hn::Set(t, b.template at<'i'>(y_len - 1)));

		for (index_t i = y_len - 2; i >= 0; i--)
		{
			auto idx = noarr::idx<'i'>(i);

			vec_t curr = hn::Load(t, &d_blocked.template at<'z', 'y'>(z, i));
			curr = hn::Mul(hn::MulAdd(prev, hn::Set(t, -c[idx]), curr), hn::Set(t, b[idx]));

			prev = z_forward(prev, d_blocked, i + 1);

			hn::Store(prev, t, &d_blocked.template at<'z', 'y'>(z, i + 1));

			prev = curr;
		}

		prev = z_forward(prev, d_blocked, 0);

		hn::Store(prev, t, &d_blocked.template at<'z', 'y'>(z, 0));
	}
}

template <typename simd_t, typename simd_tag, typename index_t, typename density_bag_t, typename... simd_pack_t>
constexpr static void load(const density_bag_t d, simd_tag t, const index_t x, const index_t y, const index_t z,
						   simd_t& first, simd_pack_t&&... vec_pack)
{
	first = hn::Load(t, &(d.template at<'z', 'y', 'x'>(z, y, x)));
	if constexpr (sizeof...(vec_pack) > 0)
	{
		load(d, t, x, y + 1, z, std::forward<simd_pack_t>(vec_pack)...);
	}
}

template <typename simd_t, typename simd_tag, typename index_t, typename density_bag_t, typename... simd_pack_t>
constexpr static void store(const density_bag_t d, simd_tag t, const index_t x, const index_t y, const index_t z,
							const simd_t& first, const simd_pack_t&... vec_pack)
{
	hn::Store(first, t, &(d.template at<'z', 'y', 'x'>(z, y, x)));
	if constexpr (sizeof...(vec_pack) > 0)
	{
		store(d, t, x, y + 1, z, vec_pack...);
	}
}

template <typename simd_t, typename simd_tag, typename index_t, typename density_bag_t, typename diag_bag_t,
		  typename y_func_t, typename... simd_pack_t>
constexpr static void xy_fused_transpose_part(const density_bag_t d, simd_tag t, const index_t y_offset,
											  const index_t y_len, const index_t z, const diag_bag_t b,
											  const diag_bag_t c, const diag_bag_t rf, y_func_t&& y_forward,
											  simd_pack_t&&... vec_pack)
{
	const index_t n = d | noarr::get_length<'x'>();

	constexpr index_t simd_length = sizeof...(vec_pack);

	const index_t full_n = (n + simd_length - 1) / simd_length * simd_length;

	for (index_t y = y_offset; y < y_len; y += simd_length)
	{
		simd_t prev = hn::Zero(t);

		// forward substitution until last simd_length elements
		for (index_t i = 0; i < full_n - simd_length; i += simd_length)
		{
			load(d, t, i, y, z, std::forward<simd_pack_t>(vec_pack)...);

			// transposition to enable vectorization
			transpose(std::forward<simd_pack_t>(vec_pack)...);

			// actual forward substitution (vectorized)
			prev = x_forward_vectorized(t, i, rf, prev, std::forward<simd_pack_t>(vec_pack)...);

			// transposition back to the original form
			// transpose(rows);

			store(d, t, i, y, z, std::forward<simd_pack_t>(vec_pack)...);
		}

		// we are aligned to the vector size, so we can safely continue
		// here we fuse the end of forward substitution and the beginning of backwards propagation
		{
			load(d, t, full_n - simd_length, y, z, std::forward<simd_pack_t>(vec_pack)...);

			// transposition to enable vectorization
			transpose(std::forward<simd_pack_t>(vec_pack)...);

			index_t remainder_work = n % simd_length;
			remainder_work += remainder_work == 0 ? simd_length : 0;

			// the rest of forward part
			x_forward_vectorized(t, remainder_work, full_n - simd_length, rf, prev,
								 std::forward<simd_pack_t>(vec_pack)...);

			prev = hn::Zero(t);

			// the begin of backward part
			prev = x_backward_vectorized(b, c, t, remainder_work, full_n - simd_length,
										 std::forward<simd_pack_t>(vec_pack)..., prev);

			// transposition back to the original form
			transpose(std::forward<simd_pack_t>(vec_pack)...);

			y_forward(y, full_n - simd_length, t, std::forward<simd_pack_t>(vec_pack)...);

			store(d, t, full_n - simd_length, y, z, std::forward<simd_pack_t>(vec_pack)...);
		}

		// we continue with backwards substitution
		for (index_t i = full_n - simd_length * 2; i >= 0; i -= simd_length)
		{
			load(d, t, i, y, z, std::forward<simd_pack_t>(vec_pack)...);

			// transposition to enable vectorization
			// transpose(rows);

			// backward propagation
			prev = x_backward_vectorized(b, c, t, i, std::forward<simd_pack_t>(vec_pack)..., prev);

			// transposition back to the original form
			transpose(std::forward<simd_pack_t>(vec_pack)...);

			y_forward(y, i, t, std::forward<simd_pack_t>(vec_pack)...);

			store(d, t, i, y, z, std::forward<simd_pack_t>(vec_pack)...);
		}
	}
}

template <typename real_t, typename index_t, typename density_bag_t, typename diag_bag_t, typename y_func_t,
		  typename y_func_scalar_t>
constexpr void xy_fused_transpose_part_2(const density_bag_t d, const index_t z, const diag_bag_t b, const diag_bag_t c,
										 const diag_bag_t rf, const index_t y_offset, y_func_t&& y_forward,
										 y_func_scalar_t&& y_forward_s)
{
	constexpr index_t simd_length = 2;

	const index_t y_len = d | noarr::get_length<'y'>();

	const index_t simd_y_len = (y_len - y_offset) / simd_length * simd_length;

	{
		using simd_tag = hn::FixedTag<real_t, 2>;
		simd_tag t;
		using simd_t = hn::Vec<simd_tag>;

		simd_t vec0, vec1;

		xy_fused_transpose_part<simd_t>(d, t, y_offset, y_offset + simd_y_len, z, b, c, rf,
										std::forward<y_func_t>(y_forward), vec0, vec1);
	}

	for (index_t y = y_offset + simd_y_len; y < y_len; y++)
	{
		x_forward<real_t>(d, z, y, rf);

		x_backward<real_t>(d, b, c, z, y, std::move(y_forward_s));
	}
}

template <typename real_t, typename index_t, typename density_bag_t, typename diag_bag_t, typename y_func_t,
		  typename y_func_scalar_t>
constexpr void xy_fused_transpose_part_4(const density_bag_t d, const index_t z, const diag_bag_t b, const diag_bag_t c,
										 const diag_bag_t rf, const index_t y_offset, y_func_t&& y_forward,
										 y_func_scalar_t&& y_forward_s)
{
	constexpr index_t simd_length = 4;

	const index_t y_len = d | noarr::get_length<'y'>();

	const index_t simd_y_len = (y_len - y_offset) / simd_length * simd_length;

	{
		using simd_tag = hn::FixedTag<real_t, 4>;
		simd_tag t;
		using simd_t = hn::Vec<simd_tag>;

		simd_t vec0, vec1, vec2, vec3;

		xy_fused_transpose_part<simd_t>(d, t, y_offset, y_offset + simd_y_len, z, b, c, rf,
										std::forward<y_func_t>(y_forward), vec0, vec1, vec2, vec3);
	}

	xy_fused_transpose_part_2<real_t>(d, z, b, c, rf, y_offset + simd_y_len, std::forward<y_func_t>(y_forward),
									  std::forward<y_func_scalar_t>(y_forward_s));
}

template <typename real_t, typename index_t, typename density_bag_t, typename diag_bag_t, typename y_func_t,
		  typename y_func_scalar_t>
constexpr void xy_fused_transpose_part_8(const density_bag_t d, const index_t z, const diag_bag_t b, const diag_bag_t c,
										 const diag_bag_t rf, const index_t y_offset, y_func_t&& y_forward,
										 y_func_scalar_t&& y_forward_s)
{
	constexpr index_t simd_length = 8;

	const index_t y_len = d | noarr::get_length<'y'>();

	const index_t simd_y_len = (y_len - y_offset) / simd_length * simd_length;

	{
		using simd_tag = hn::FixedTag<real_t, 8>;
		simd_tag t;
		using simd_t = hn::Vec<simd_tag>;

		simd_t vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7;

		xy_fused_transpose_part<simd_t>(d, t, y_offset, y_offset + simd_y_len, z, b, c, rf,
										std::forward<y_func_t>(y_forward), vec0, vec1, vec2, vec3, vec4, vec5, vec6,
										vec7);
	}

	xy_fused_transpose_part_4<real_t>(d, z, b, c, rf, y_offset + simd_y_len, std::forward<y_func_t>(y_forward),
									  std::forward<y_func_scalar_t>(y_forward_s));
}

template <typename real_t, typename index_t, typename density_bag_t, typename diag_bag_t, typename y_func_t,
		  typename y_func_scalar_t>
constexpr void xy_fused_transpose_part_16(const density_bag_t d, const index_t z, const diag_bag_t b,
										  const diag_bag_t c, const diag_bag_t rf, const index_t y_offset,
										  y_func_t&& y_forward, y_func_scalar_t&& y_forward_s)
{
	constexpr index_t simd_length = 16;

	const index_t y_len = d | noarr::get_length<'y'>();

	const index_t simd_y_len = (y_len - y_offset) / simd_length * simd_length;

	{
		using simd_tag = hn::FixedTag<real_t, 16>;
		simd_tag t;
		using simd_t = hn::Vec<simd_tag>;

		simd_t vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, vec9, vec10, vec11, vec12, vec13, vec14, vec15;

		xy_fused_transpose_part<simd_t>(d, t, y_offset, y_offset + simd_y_len, z, b, c, rf,
										std::forward<y_func_t>(y_forward), vec0, vec1, vec2, vec3, vec4, vec5, vec6,
										vec7, vec8, vec9, vec10, vec11, vec12, vec13, vec14, vec15);
	}

	xy_fused_transpose_part_8<real_t>(d, z, b, c, rf, y_offset + simd_y_len, std::forward<y_func_t>(y_forward),
									  std::forward<y_func_scalar_t>(y_forward_s));
}

template <typename real_t, typename index_t, typename density_bag_t, typename diag_bag_t, typename y_func_t,
		  typename y_func_scalar_t,
		  std::enable_if_t<HWY_MAX_LANES_V(hn::Vec<hn::ScalableTag<real_t>>) == 2, bool> = true>
constexpr void xy_fused_transpose_part_dispatch(const density_bag_t d, const index_t z, const diag_bag_t b,
												const diag_bag_t c, const diag_bag_t rf, y_func_t&& y_forward,
												y_func_scalar_t&& y_forward_s)
{
	xy_fused_transpose_part_2<real_t>(d, z, b, c, rf, 0, std::forward<y_func_t>(y_forward),
									  std::forward<y_func_scalar_t>(y_forward_s));
}

template <typename real_t, typename index_t, typename density_bag_t, typename diag_bag_t, typename y_func_t,
		  typename y_func_scalar_t,
		  std::enable_if_t<HWY_MAX_LANES_V(hn::Vec<hn::ScalableTag<real_t>>) == 4, bool> = true>
constexpr void xy_fused_transpose_part_dispatch(const density_bag_t d, const index_t z, const diag_bag_t b,
												const diag_bag_t c, const diag_bag_t rf, y_func_t&& y_forward,
												y_func_scalar_t&& y_forward_s)
{
	HWY_LANES_CONSTEXPR index_t max_length = hn::Lanes(hn::ScalableTag<real_t> {});
	HWY_LANES_CONSTEXPR index_t simd_length = std::min(16, max_length);

	if (simd_length == 2)
	{
		xy_fused_transpose_part_2<real_t>(d, z, b, c, rf, 0, std::forward<y_func_t>(y_forward),
										  std::forward<y_func_scalar_t>(y_forward_s));
	}
	else if (simd_length == 4)
	{
		xy_fused_transpose_part_4<real_t>(d, z, b, c, rf, 0, std::forward<y_func_t>(y_forward),
										  std::forward<y_func_scalar_t>(y_forward_s));
	}
}

template <typename real_t, typename index_t, typename density_bag_t, typename diag_bag_t, typename y_func_t,
		  typename y_func_scalar_t,
		  std::enable_if_t<HWY_MAX_LANES_V(hn::Vec<hn::ScalableTag<real_t>>) == 8, bool> = true>
constexpr void xy_fused_transpose_part_dispatch(const density_bag_t d, const index_t z, const diag_bag_t b,
												const diag_bag_t c, const diag_bag_t rf, y_func_t&& y_forward,
												y_func_scalar_t&& y_forward_s)
{
	HWY_LANES_CONSTEXPR index_t max_length = hn::Lanes(hn::ScalableTag<real_t> {});
	HWY_LANES_CONSTEXPR index_t simd_length = std::min(16, max_length);

	if (simd_length == 2)
	{
		xy_fused_transpose_part_2<real_t>(d, z, b, c, rf, 0, std::forward<y_func_t>(y_forward),
										  std::forward<y_func_scalar_t>(y_forward_s));
	}
	else if (simd_length == 4)
	{
		xy_fused_transpose_part_4<real_t>(d, z, b, c, rf, 0, std::forward<y_func_t>(y_forward),
										  std::forward<y_func_scalar_t>(y_forward_s));
	}
	else if (simd_length == 8)
	{
		xy_fused_transpose_part_8<real_t>(d, z, b, c, rf, 0, std::forward<y_func_t>(y_forward),
										  std::forward<y_func_scalar_t>(y_forward_s));
	}
}

template <typename real_t, typename index_t, typename density_bag_t, typename diag_bag_t, typename y_func_t,
		  typename y_func_scalar_t,
		  std::enable_if_t<HWY_MAX_LANES_V(hn::Vec<hn::ScalableTag<real_t>>) >= 16, bool> = true>
constexpr void xy_fused_transpose_part_dispatch(const density_bag_t d, const index_t z, const diag_bag_t b,
												const diag_bag_t c, const diag_bag_t rf, y_func_t&& y_forward,
												y_func_scalar_t&& y_forward_s)
{
	HWY_LANES_CONSTEXPR index_t max_length = hn::Lanes(hn::ScalableTag<real_t> {});
	HWY_LANES_CONSTEXPR index_t simd_length = sizeof(real_t) == 8 ? std::min(8, max_length) : std::min(16, max_length);

	if (simd_length == 2)
	{
		xy_fused_transpose_part_2<real_t>(d, z, b, c, rf, 0, std::forward<y_func_t>(y_forward),
										  std::forward<y_func_scalar_t>(y_forward_s));
	}
	else if (simd_length == 4)
	{
		xy_fused_transpose_part_4<real_t>(d, z, b, c, rf, 0, std::forward<y_func_t>(y_forward),
										  std::forward<y_func_scalar_t>(y_forward_s));
	}
	else if (simd_length == 8)
	{
		xy_fused_transpose_part_8<real_t>(d, z, b, c, rf, 0, std::forward<y_func_t>(y_forward),
										  std::forward<y_func_scalar_t>(y_forward_s));
	}
	else if (simd_length == 16)
	{
		xy_fused_transpose_part_16<real_t>(d, z, b, c, rf, 0, std::forward<y_func_t>(y_forward),
										   std::forward<y_func_scalar_t>(y_forward_s));
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_xy_fused_transpose_partial(real_t* __restrict__ densities, const real_t* __restrict__ bx,
												   const real_t* __restrict__ cx, const real_t* __restrict__ rfx,
												   const real_t* __restrict__ by, const real_t* __restrict__ cy,
												   const real_t* __restrict__ rfy, const density_layout_t dens_l,
												   const diagonal_layout_t diagx_l, const diagonal_layout_t diagy_l,
												   const index_t s_begin, const index_t s_end)
{
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	const index_t y_len = dens_l | noarr::get_length<'y'>();
	const index_t z_len = dens_l | noarr::get_length<'z'>();

#pragma omp for schedule(static) nowait collapse(2)
	for (index_t s = s_begin; s < s_end; s++)
	{
		for (index_t z = 0; z < z_len; z++)
		{
			auto bx_bag = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), bx);
			auto cx_bag = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), cx);
			auto rfx_bag = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), rfx);

			auto by_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s), by);
			auto cy_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s), cy);
			auto rfy_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s), rfy);

			const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s'>(s), densities);

			{
				auto y_forward = [z, d, rfy_bag](index_t y_offset, index_t x, auto tag, auto&&... vec_pack) {
					y_forward_inside_x_vectorized(d, tag, z, y_offset, x, rfy_bag,
												  std::forward<decltype(vec_pack)>(vec_pack)...);
				};

				auto y_forward_scalar = [z, d, rfy_bag](index_t x, index_t y, real_t data) {
					return y_forward_inside_x(d, z, y, x, data, rfy_bag);
				};

				xy_fused_transpose_part_dispatch<real_t>(d, z, bx_bag, cx_bag, rfx_bag, std::move(y_forward),
														 std::move(y_forward_scalar));
			}

			auto empty_f = [](auto data, auto, auto) { return data; };
			y_backward_vectorized<simd_t>(d, by_bag, cy_bag, t, z, simd_length, y_len, std::move(empty_f));
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_z_3d_intrinsics_partial(real_t* __restrict__ densities, const real_t* __restrict__ b,
												const real_t* __restrict__ c, const real_t* __restrict__ rf,
												const density_layout_t dens_l, const diagonal_layout_t diag_l,
												const index_t s_begin, const index_t s_end)
{
	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();
	const index_t n = dens_l | noarr::get_length<'z'>();

	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

#pragma omp for schedule(static) nowait collapse(3)
	for (index_t s = s_begin; s < s_end; s++)
	{
		for (index_t y = 0; y < y_len; y++)
		{
			for (index_t x = 0; x < x_len; x += simd_length)
			{
				auto b_bag = noarr::make_bag(diag_l ^ noarr::fix<'s'>(s), b);
				auto c_bag = noarr::make_bag(diag_l ^ noarr::fix<'s'>(s), c);
				auto rf_bag = noarr::make_bag(diag_l ^ noarr::fix<'s'>(s), rf);

				const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s'>(s), densities);

				simd_t prev = hn::Zero(t);

				for (index_t i = 0; i < n; i++)
				{
					auto idx = noarr::idx<'i'>(i);

					simd_t curr = hn::Load(t, &d.template at<'z', 'y', 'x'>(i, y, x));
					curr = hn::MulAdd(hn::Set(t, -rf_bag[idx]), prev, curr);

					hn::Store(curr, t, &d.template at<'z', 'y', 'x'>(i, y, x));

					prev = curr;
				}

				prev = hn::Mul(prev, hn::Set(t, b_bag.template at<'i'>(n - 1)));
				hn::Store(prev, t, &d.template at<'z', 'y', 'x'>(n - 1, y, x));

				for (index_t i = n - 2; i >= 0; i--)
				{
					auto idx = noarr::idx<'i'>(i);

					simd_t curr = hn::Load(t, &d.template at<'z', 'y', 'x'>(i, y, x));
					curr = hn::Mul(hn::MulAdd(prev, hn::Set(t, -c_bag[idx]), curr), hn::Set(t, b_bag[idx]));
					hn::Store(curr, t, &d.template at<'z', 'y', 'x'>(i, y, x));

					prev = curr;
				}
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
constexpr static void solve_slice_xyz_fused_transpose(real_t* __restrict__ densities, const real_t* __restrict__ bx,
													  const real_t* __restrict__ cx, const real_t* __restrict__ rfx,
													  const real_t* __restrict__ by, const real_t* __restrict__ cy,
													  const real_t* __restrict__ rfy, const real_t* __restrict__ bz,
													  const real_t* __restrict__ cz, const real_t* __restrict__ rfz,
													  const density_layout_t dens_l, const diagonal_layout_t diagx_l,
													  const diagonal_layout_t diagy_l, const diagonal_layout_t diagz_l,
													  const index_t s_begin, const index_t s_end)
{
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	const index_t z_len = dens_l | noarr::get_length<'z'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();

	for (index_t s = s_begin; s < s_end; s++)
	{
		auto bx_bag = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), bx);
		auto cx_bag = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), cx);
		auto rfx_bag = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), rfx);

		auto by_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s), by);
		auto cy_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s), cy);
		auto rfy_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s), rfy);

		auto bz_bag = noarr::make_bag(diagz_l ^ noarr::fix<'s'>(s), bz);
		auto cz_bag = noarr::make_bag(diagz_l ^ noarr::fix<'s'>(s), cz);
		auto rfz_bag = noarr::make_bag(diagz_l ^ noarr::fix<'s'>(s), rfz);

		const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s'>(s), densities);

		for (index_t z = 0; z < z_len; z++)
		{
			{
				auto y_forward = [d, z, rfy_bag](index_t y_offset, index_t x, auto tag, auto&&... vec_pack) {
					y_forward_inside_x_vectorized(d, tag, z, y_offset, x, rfy_bag,
												  std::forward<decltype(vec_pack)>(vec_pack)...);
				};

				auto y_forward_scalar = [d, z, rfy_bag](index_t x, index_t y, real_t data) {
					return y_forward_inside_x(d, z, y, x, data, rfy_bag);
				};


				xy_fused_transpose_part_dispatch<real_t>(d, z, bx_bag, cx_bag, rfx_bag, std::move(y_forward),
														 std::move(y_forward_scalar));
			}

			auto z_forward = [t, z, rfz_bag](simd_t data, auto d, index_t y) {
				return z_forward_inside_y_vectorized(d, t, y, z, data, rfz_bag);
			};

			y_backward_vectorized<simd_t>(d, by_bag, cy_bag, t, z, simd_length, y_len, std::move(z_forward));
		}

		z_backward_vectorized<simd_t>(d, bz_bag, cz_bag, t, simd_length, y_len, z_len);
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
constexpr static void solve_slice_xy_fused_transpose(real_t* __restrict__ densities, const real_t* __restrict__ bx,
													 const real_t* __restrict__ cx, const real_t* __restrict__ rfx,
													 const real_t* __restrict__ by, const real_t* __restrict__ cy,
													 const real_t* __restrict__ rfy, const density_layout_t dens_l,
													 const diagonal_layout_t diagx_l, const diagonal_layout_t diagy_l,
													 const index_t s_begin, const index_t s_end)
{
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	for (index_t s = s_begin; s < s_end; s++)
	{
		auto bx_bag = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), bx);
		auto cx_bag = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), cx);
		auto rfx_bag = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), rfx);

		auto by_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s), by);
		auto cy_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s), cy);
		auto rfy_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s), rfy);

		const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s'>(s), densities);

		const index_t y_len = dens_l | noarr::get_length<'y'>();

		{
			auto y_forward = [d, rfy_bag](index_t y_offset, index_t x, auto tag, auto&&... vec_pack) {
				y_forward_inside_x_vectorized(d, tag, 0, y_offset, x, rfy_bag,
											  std::forward<decltype(vec_pack)>(vec_pack)...);
			};

			auto y_forward_scalar = [d, rfy_bag](index_t x, index_t y, real_t data) {
				return y_forward_inside_x(d, 0, y, x, data, rfy_bag);
			};

			xy_fused_transpose_part_dispatch<real_t>(d, 0, bx_bag, cx_bag, rfx_bag, std::move(y_forward),
													 std::move(y_forward_scalar));
		}

		auto empty_f = [](auto data, auto, auto) { return data; };
		y_backward_vectorized<simd_t>(d, by_bag, cy_bag, t, 0, simd_length, y_len, std::move(empty_f));
	}
}


template <bool alt_blocked, typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename sync_func_t>
constexpr static void solve_slice_xyz_fused_transpose_blocked(
	real_t* __restrict__ densities, const real_t* __restrict__ bx, const real_t* __restrict__ cx,
	const real_t* __restrict__ rfx, const real_t* __restrict__ by, const real_t* __restrict__ cy,
	const real_t* __restrict__ rfy, const real_t* __restrict__ az, const real_t* __restrict__ bz,
	const real_t* __restrict__ cz, const real_t* __restrict__ rfz, const real_t* __restrict__ rbz,
	const density_layout_t dens_l, const diagonal_layout_t diagx_l, const diagonal_layout_t diagy_l,
	const diagonal_layout_t diagz_l, const index_t s_begin, const index_t s_end, const index_t z_begin,
	const index_t z_end, sync_func_t&& synchronize_blocked_z)
{
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	for (index_t s = s_begin; s < s_end; s++)
	{
		auto bx_bag = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), bx);
		auto cx_bag = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), cx);
		auto rfx_bag = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), rfx);

		auto by_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s), by);
		auto cy_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s), cy);
		auto rfy_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s), rfy);

		auto slicei = noarr::slice<'i'>(z_begin, z_end - z_begin);
		auto az_bag = noarr::make_bag(diagz_l ^ noarr::fix<'s'>(s) ^ slicei, az);
		auto bz_bag = noarr::make_bag(diagz_l ^ noarr::fix<'s'>(s) ^ slicei, bz);
		auto cz_bag = noarr::make_bag(diagz_l ^ noarr::fix<'s'>(s) ^ slicei, cz);
		auto rfz_bag = noarr::make_bag(diagz_l ^ noarr::fix<'s'>(s) ^ slicei, rfz);
		auto rbz_bag = noarr::make_bag(diagz_l ^ noarr::fix<'s'>(s) ^ slicei, rbz);

		const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s'>(s), densities);

		const index_t y_len = dens_l | noarr::get_length<'y'>();
		const index_t z_len = z_end - z_begin;

		for (index_t z = 0; z < z_len; z++)
		{
			{
				auto y_forward = [d, z, rfy_bag](index_t y_offset, index_t x, auto tag, auto&&... vec_pack) {
					y_forward_inside_x_vectorized(d, tag, z, y_offset, x, rfy_bag,
												  std::forward<decltype(vec_pack)>(vec_pack)...);
				};

				auto y_forward_scalar = [d, z, rfy_bag](index_t x, index_t y, real_t data) {
					return y_forward_inside_x(d, z, y, x, data, rfy_bag);
				};

				xy_fused_transpose_part_dispatch<real_t>(d, z, bx_bag, cx_bag, rfx_bag, std::move(y_forward),
														 std::move(y_forward_scalar));
			}

			if constexpr (alt_blocked)
			{
				auto z_forward = [t, z, rfz_bag, rbz_bag](simd_t data, auto d, index_t y) {
					return z_forward_inside_y_blocked_alt_vectorized(d, data, t, y, z, rfz_bag, rbz_bag);
				};

				y_backward_vectorized<simd_t>(d, by_bag, cy_bag, t, z, simd_length, y_len, std::move(z_forward));
			}
			else
			{
				auto z_forward = [t, z, rfz_bag](simd_t data, auto d, index_t y) {
					return z_forward_inside_y_blocked_vectorized(d, data, t, y, z, rfz_bag);
				};

				y_backward_vectorized<simd_t>(d, by_bag, cy_bag, t, z, simd_length, y_len, std::move(z_forward));
			}
		}

		if constexpr (!alt_blocked)
			z_backward_blocked_vectorized<simd_t>(d, rbz_bag, t, simd_length, y_len, z_len);

		synchronize_blocked_z(s);

		if constexpr (alt_blocked)
			z_blocked_alt_end_vectorized<simd_t>(d, az_bag, bz_bag, cz_bag, t, simd_length, y_len, z_len);
		else
			z_blocked_end_vectorized<simd_t>(d, az_bag, bz_bag, cz_bag, t, simd_length, y_len, z_len);
	}
}

template <bool alt_blocked, typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename sync_func_t>
constexpr static void solve_slice_xy_fused_transpose_blocked(
	real_t* __restrict__ densities, const real_t* __restrict__ bx, const real_t* __restrict__ cx,
	const real_t* __restrict__ rfx, const real_t* __restrict__ ay, const real_t* __restrict__ by,
	const real_t* __restrict__ cy, const real_t* __restrict__ rfy, const real_t* __restrict__ rby,
	const density_layout_t dens_l, const diagonal_layout_t diagx_l, const diagonal_layout_t diagy_l,
	const index_t s_begin, const index_t s_end, const index_t y_begin, const index_t y_end,
	sync_func_t&& synchronize_blocked_y)
{
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	for (index_t s = s_begin; s < s_end; s++)
	{
		auto bx_bag = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), bx);
		auto cx_bag = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), cx);
		auto rfx_bag = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), rfx);

		auto sliceyi = noarr::slice<'i'>(y_begin, y_end - y_begin);
		auto ay_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s) ^ sliceyi, ay);
		auto by_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s) ^ sliceyi, by);
		auto cy_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s) ^ sliceyi, cy);
		auto rfy_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s) ^ sliceyi, rfy);
		auto rby_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s) ^ sliceyi, rby);

		const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s'>(s), densities);

		const index_t y_len = y_end - y_begin;

		{
			if constexpr (alt_blocked)
			{
				auto y_forward = [d, rfy_bag, rby_bag](index_t y_offset, index_t x, auto tag, auto&&... vec_pack) {
					y_forward_inside_x_blocked_alt_vectorized(d, tag, 0, y_offset, x, rfy_bag, rby_bag,
															  std::forward<decltype(vec_pack)>(vec_pack)...);
				};

				auto y_forward_scalar = [d, rfy_bag, rby_bag](index_t x, index_t y, real_t data) {
					return y_forward_inside_x_blocked_alt(d, 0, y, x, data, rfy_bag, rby_bag);
				};

				xy_fused_transpose_part_dispatch<real_t>(d, 0, bx_bag, cx_bag, rfx_bag, std::move(y_forward),
														 std::move(y_forward_scalar));
			}
			else
			{
				auto y_forward = [d, rfy_bag](index_t y_offset, index_t x, auto tag, auto&&... vec_pack) {
					y_forward_inside_x_blocked_vectorized(d, tag, 0, y_offset, x, rfy_bag,
														  std::forward<decltype(vec_pack)>(vec_pack)...);
				};
				auto y_forward_scalar = [d, rfy_bag](index_t x, index_t y, real_t data) {
					return y_forward_inside_x_blocked(d, 0, y, x, data, rfy_bag);
				};

				xy_fused_transpose_part_dispatch<real_t>(d, 0, bx_bag, cx_bag, rfx_bag, std::move(y_forward),
														 std::move(y_forward_scalar));
			}
		}

		if constexpr (!alt_blocked)
			y_backward_blocked_vectorized<simd_t>(d, rby_bag, t, 0, simd_length, y_len);

		synchronize_blocked_y(0, s);

		auto empty_f = [](auto data, auto, auto) { return data; };
		if constexpr (alt_blocked)
			y_blocked_alt_end_vectorized<simd_t>(d, ay_bag, by_bag, cy_bag, t, 0, simd_length, y_len,
												 std::move(empty_f));
		else
			y_blocked_end_vectorized<simd_t>(d, ay_bag, by_bag, cy_bag, t, 0, simd_length, y_len, std::move(empty_f));
	}
}

template <bool alt_blocked, typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename sync_func_y_t, typename sync_func_z_t>
constexpr static void solve_slice_xyz_fused_transpose_blocked(
	real_t* __restrict__ densities, const real_t* __restrict__ bx, const real_t* __restrict__ cx,
	const real_t* __restrict__ rfx, const real_t* __restrict__ ay, const real_t* __restrict__ by,
	const real_t* __restrict__ cy, const real_t* __restrict__ rfy, const real_t* __restrict__ rby,
	const real_t* __restrict__ az, const real_t* __restrict__ bz, const real_t* __restrict__ cz,
	const real_t* __restrict__ rfz, const real_t* __restrict__ rbz, const density_layout_t dens_l,
	const diagonal_layout_t diagx_l, const diagonal_layout_t diagy_l, const diagonal_layout_t diagz_l,
	const index_t s_begin, const index_t s_end, const index_t y_begin, const index_t y_end, const index_t z_begin,
	const index_t z_end, sync_func_y_t&& synchronize_blocked_y, sync_func_z_t&& synchronize_blocked_z)
{
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	for (index_t s = s_begin; s < s_end; s++)
	{
		auto bx_bag = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), bx);
		auto cx_bag = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), cx);
		auto rfx_bag = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), rfx);

		auto sliceyi = noarr::slice<'i'>(y_begin, y_end - y_begin);
		auto ay_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s) ^ sliceyi, ay);
		auto by_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s) ^ sliceyi, by);
		auto cy_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s) ^ sliceyi, cy);
		auto rfy_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s) ^ sliceyi, rfy);
		auto rby_bag = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s) ^ sliceyi, rby);

		auto slicezi = noarr::slice<'i'>(z_begin, z_end - z_begin);
		auto az_bag = noarr::make_bag(diagz_l ^ noarr::fix<'s'>(s) ^ slicezi, az);
		auto bz_bag = noarr::make_bag(diagz_l ^ noarr::fix<'s'>(s) ^ slicezi, bz);
		auto cz_bag = noarr::make_bag(diagz_l ^ noarr::fix<'s'>(s) ^ slicezi, cz);
		auto rfz_bag = noarr::make_bag(diagz_l ^ noarr::fix<'s'>(s) ^ slicezi, rfz);
		auto rbz_bag = noarr::make_bag(diagz_l ^ noarr::fix<'s'>(s) ^ slicezi, rbz);

		const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s'>(s), densities);

		const index_t y_len = y_end - y_begin;
		const index_t z_len = z_end - z_begin;

		for (index_t z = 0; z < z_len; z++)
		{
			{
				if constexpr (alt_blocked)
				{
					auto y_forward = [z, d, rfy_bag, rby_bag](index_t y_offset, index_t x, auto tag,
															  auto&&... vec_pack) {
						y_forward_inside_x_blocked_alt_vectorized(d, tag, z, y_offset, x, rfy_bag, rby_bag,
																  std::forward<decltype(vec_pack)>(vec_pack)...);
					};
					auto y_forward_scalar = [d, z, rfy_bag, rby_bag](index_t x, index_t y, real_t data) {
						return y_forward_inside_x_blocked_alt(d, z, y, x, data, rfy_bag, rby_bag);
					};

					xy_fused_transpose_part_dispatch<real_t>(d, z, bx_bag, cx_bag, rfx_bag, std::move(y_forward),
															 std::move(y_forward_scalar));
				}
				else
				{
					auto y_forward = [z, d, rfy_bag](index_t y_offset, index_t x, auto tag, auto&&... vec_pack) {
						y_forward_inside_x_blocked_vectorized(d, tag, z, y_offset, x, rfy_bag,
															  std::forward<decltype(vec_pack)>(vec_pack)...);
					};
					auto y_forward_scalar = [d, z, rfy_bag](index_t x, index_t y, real_t data) {
						return y_forward_inside_x_blocked(d, z, y, x, data, rfy_bag);
					};

					xy_fused_transpose_part_dispatch<real_t>(d, z, bx_bag, cx_bag, rfx_bag, std::move(y_forward),
															 std::move(y_forward_scalar));
				}
			}

			if constexpr (!alt_blocked)
				y_backward_blocked_vectorized<simd_t>(d, rby_bag, t, z, simd_length, y_len);

			synchronize_blocked_y(z, s);

			if constexpr (alt_blocked)
			{
				auto z_forward = [t, z, rfz_bag, rbz_bag](simd_t data, auto d, index_t y) {
					return z_forward_inside_y_blocked_alt_vectorized(d, data, t, y, z, rfz_bag, rbz_bag);
				};

				y_blocked_alt_end_vectorized<simd_t>(d, ay_bag, by_bag, cy_bag, t, z, simd_length, y_len,
													 std::move(z_forward));
			}
			else
			{
				auto z_forward = [t, z, rfz_bag](simd_t data, auto d, index_t y) {
					return z_forward_inside_y_blocked_vectorized(d, data, t, y, z, rfz_bag);
				};

				y_blocked_end_vectorized<simd_t>(d, ay_bag, by_bag, cy_bag, t, z, simd_length, y_len,
												 std::move(z_forward));
			}
		}

		if constexpr (!alt_blocked)
			z_backward_blocked_vectorized<simd_t>(d, rbz_bag, t, simd_length, y_len, z_len);

		synchronize_blocked_z(s);

		if constexpr (alt_blocked)
			z_blocked_alt_end_vectorized<simd_t>(d, az_bag, bz_bag, cz_bag, t, simd_length, y_len, z_len);
		else
			z_blocked_end_vectorized<simd_t>(d, az_bag, bz_bag, cz_bag, t, simd_length, y_len, z_len);
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_f_p<real_t, aligned_x>::solve_x()
{}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_f_p<real_t, aligned_x>::solve_y()
{}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_f_p<real_t, aligned_x>::solve_z()
{}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_f_p<real_t, aligned_x>::solve()
{
	if (partial_blocking_)
	{
		if (this->problem_.dims == 3)
		{
			solve_partial_blocked_3d();
		}

		return;
	}

	if (!(cores_division_[0] == 1 && cores_division_[1] == 1 && cores_division_[2] == 1))
	{
		if (this->problem_.dims == 2)
		{
			solve_blocked_2d();
		}
		else if (this->problem_.dims == 3)
		{
			if (cores_division_[1] == 1)
				solve_blocked_3d_z();
			else
				solve_blocked_3d_yz();
		}

		return;
	}

#pragma omp parallel
	{
		perf_counter counter("lstmfpai");

		auto tid = get_thread_id();
		auto work_id = tid.group;

		auto s_global_begin = group_block_offsetss_[work_id];
		auto s_global_end = s_global_begin + group_block_lengthss_[work_id];

		for (index_t s = 0; s < group_block_lengthss_[work_id]; s += substrate_step_)
		{
			auto s_end = std::min(s + substrate_step_, group_block_lengthss_[work_id]);

			// #pragma omp critical
			// 			std::cout << "Thread " << get_thread_num() << " s_begin: " << s_global_begin + s
			// 					  << " s_end: " << s_global_begin + s_end << " group: " << tid.group << std::endl;

			for (index_t i = 0; i < this->problem_.iterations; i++)
			{
				if (use_thread_distributed_allocation_)
				{
					auto s_slice = noarr::slice<'s'>(s_global_begin, s_global_end - s_global_begin);

					if (this->problem_.dims == 3)
						solve_slice_xyz_fused_transpose(
							thread_substrate_array_[work_id], thread_bx_[work_id], thread_cx_[work_id],
							thread_rf_x_[work_id], thread_by_[work_id], thread_cy_[work_id], thread_rf_y_[work_id],
							thread_bz_[work_id], thread_cz_[work_id], thread_rf_z_[work_id], get_substrates_layout<3>(),
							get_diagonal_layout(this->problem_, this->problem_.nx) ^ s_slice,
							get_diagonal_layout(this->problem_, this->problem_.ny) ^ s_slice,
							get_diagonal_layout(this->problem_, this->problem_.nz) ^ s_slice, s, s_end);
					else if (this->problem_.dims == 2)
						solve_slice_xy_fused_transpose(
							thread_substrate_array_[work_id], thread_bx_[work_id], thread_cx_[work_id],
							thread_rf_x_[work_id], thread_by_[work_id], thread_cy_[work_id], thread_rf_y_[work_id],
							get_substrates_layout<3>(),
							get_diagonal_layout(this->problem_, this->problem_.nx) ^ s_slice,
							get_diagonal_layout(this->problem_, this->problem_.ny) ^ s_slice, s, s_end);
				}
				else
				{
					if (this->problem_.dims == 3)
						solve_slice_xyz_fused_transpose(
							this->substrates_, thread_bx_[0], thread_cx_[0], thread_rf_x_[0], thread_by_[0],
							thread_cy_[0], thread_rf_y_[0], thread_bz_[0], thread_cz_[0], thread_rf_z_[0],
							get_substrates_layout<3>(), get_diagonal_layout(this->problem_, this->problem_.nx),
							get_diagonal_layout(this->problem_, this->problem_.ny),
							get_diagonal_layout(this->problem_, this->problem_.nz), s_global_begin + s,
							s_global_begin + s_end);
					else if (this->problem_.dims == 2)
						solve_slice_xy_fused_transpose(this->substrates_, thread_bx_[0], thread_cx_[0], thread_rf_x_[0],
													   thread_by_[0], thread_cy_[0], thread_rf_y_[0],
													   get_substrates_layout<3>(),
													   get_diagonal_layout(this->problem_, this->problem_.nx),
													   get_diagonal_layout(this->problem_, this->problem_.ny),
													   s_global_begin + s, s_global_begin + s_end);
				}
			}
		}
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_f_p<real_t, aligned_x>::solve_blocked_2d()
{
	for (index_t i = 0; i < countersy_count_; i++)
	{
		countersy_[i]->value = 0;
	}

#pragma omp parallel
	{
		perf_counter counter("lstmfpai");

		const thread_id_t<index_t> tid = get_thread_id();

		const auto block_y_begin = group_block_offsetsy_[tid.y];
		const auto block_y_end = block_y_begin + group_block_lengthsy_[tid.y];

		barrier_t<true, index_t> barrier(cores_division_[1], countersy_[tid.group]->value);

		const auto block_s_begin = group_block_offsetss_[tid.group];
		const auto block_s_end = block_s_begin + group_block_lengthss_[tid.group];

		const index_t group_size = cores_division_[1];

		for (index_t s = 0; s < group_block_lengthss_[tid.group]; s += substrate_step_)
		{
			auto s_end = std::min(s + substrate_step_, group_block_lengthss_[tid.group]);

			// #pragma omp critical
			// 			std::cout << "Thread " << get_thread_num() << " s_begin: " << s << " s_end: " << s_end
			// 					  << " block_y_begin: " << block_y_begin << " block_y_end: " << block_y_end
			// 					  << " group: " << tid.group << std::endl;

			for (index_t i = 0; i < this->problem_.iterations; i++)
			{
				if (use_thread_distributed_allocation_)
				{
					auto s_slice = noarr::slice<'s'>(block_s_begin, block_s_end - block_s_begin);

					const auto thread_num = get_thread_num();

					auto diag_x = get_diagonal_layout(this->problem_, this->problem_.nx) ^ s_slice;
					auto diag_y = get_diagonal_layout(this->problem_, this->problem_.ny) ^ s_slice;

					auto dist_l = noarr::scalar<real_t*>() ^ get_thread_distribution_layout()
								  ^ noarr::fix<'z', 'g'>(tid.z, tid.group);

					auto dens_l = get_blocked_substrate_layout<'y'>(this->problem_.nx, group_block_lengthsy_[tid.y],
																	group_block_lengthsz_[tid.z],
																	group_block_lengthss_[tid.group]);

					auto sync_y = [densities = thread_substrate_array_.get(), rf_data = thread_ay_[thread_num],
								   b_data = thread_by_[thread_num], c_data = thread_cy_[thread_num], dens_l, diag_y,
								   dist_l, n = this->problem_.ny, tid = tid.y, group_size,
								   &barrier](index_t z, index_t s) {
						synchronize_y_blocked_distributed(densities, rf_data, b_data, c_data,
														  dens_l ^ noarr::fix<'s'>(s), diag_y ^ noarr::fix<'s'>(s),
														  dist_l, n, z, tid, group_size, barrier);
					};

					auto current_densities = dist_l | noarr::get_at<'y'>(thread_substrate_array_.get(), tid.y);

					if (use_alt_blocked_)
						solve_slice_xy_fused_transpose_blocked<true>(
							current_densities, thread_bx_[thread_num], thread_cx_[thread_num], thread_rf_x_[thread_num],
							thread_ay_[thread_num], thread_by_[thread_num], thread_cy_[thread_num],
							thread_rf_y_[thread_num], thread_rb_y_[thread_num],
							dens_l ^ noarr::set_length<'y'>(block_y_end - block_y_begin), diag_x, diag_y, s, s_end,
							block_y_begin, block_y_end, std::move(sync_y));
					else
						solve_slice_xy_fused_transpose_blocked<false>(
							current_densities, thread_bx_[thread_num], thread_cx_[thread_num], thread_rf_x_[thread_num],
							thread_ay_[thread_num], thread_by_[thread_num], thread_cy_[thread_num],
							thread_rf_y_[thread_num], thread_rb_y_[thread_num],
							dens_l ^ noarr::set_length<'y'>(block_y_end - block_y_begin), diag_x, diag_y, s, s_end,
							block_y_begin, block_y_end, std::move(sync_y));
				}
				else
				{
					auto diag_x = get_diagonal_layout(this->problem_, this->problem_.nx);
					auto diag_y = get_diagonal_layout(this->problem_, this->problem_.ny);

					auto dens_l = get_substrates_layout<3>();

					auto sync_y = [densities = this->substrates_, rf_data = thread_ay_[0], b_data = thread_by_[0],
								   c_data = thread_cy_[0], dens_l, diag_y, tid = tid.y, group_size,
								   &barrier](index_t z, index_t s) {
						synchronize_y_blocked(densities, rf_data, b_data, c_data, dens_l ^ noarr::fix<'s'>(s),
											  diag_y ^ noarr::fix<'s'>(s), z, tid, group_size, barrier);
					};

					if (use_alt_blocked_)
						solve_slice_xy_fused_transpose_blocked<true>(
							this->substrates_, thread_bx_[0], thread_cx_[0], thread_rf_x_[0], thread_ay_[0],
							thread_by_[0], thread_cy_[0], thread_rf_y_[0], thread_rb_y_[0],
							dens_l ^ noarr::slice<'y'>(block_y_begin, block_y_end - block_y_begin), diag_x, diag_y,
							block_s_begin + s, block_s_begin + s_end, block_y_begin, block_y_end, std::move(sync_y));
					else
						solve_slice_xy_fused_transpose_blocked<false>(
							this->substrates_, thread_bx_[0], thread_cx_[0], thread_rf_x_[0], thread_ay_[0],
							thread_by_[0], thread_cy_[0], thread_rf_y_[0], thread_rb_y_[0],
							dens_l ^ noarr::slice<'y'>(block_y_begin, block_y_end - block_y_begin), diag_x, diag_y,
							block_s_begin + s, block_s_begin + s_end, block_y_begin, block_y_end, std::move(sync_y));
				}
			}
		}
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_f_p<real_t, aligned_x>::solve_blocked_3d_z()
{
	for (index_t i = 0; i < countersz_count_; i++)
	{
		countersz_[i]->value = 0;
	}

#pragma omp parallel
	{
		perf_counter counter("lstmfpai");

		const thread_id_t<index_t> tid = get_thread_id();

		const auto block_z_begin = group_block_offsetsz_[tid.z];
		const auto block_z_end = block_z_begin + group_block_lengthsz_[tid.z];

		const auto lane_id_z = tid.y + tid.group * cores_division_[1];

		barrier_t<true, index_t> barrier_z(cores_division_[2], countersz_[lane_id_z]->value);

		const auto block_s_begin = group_block_offsetss_[tid.group];
		const auto block_s_end = block_s_begin + group_block_lengthss_[tid.group];

		for (index_t s = 0; s < group_block_lengthss_[tid.group]; s += substrate_step_)
		{
			auto s_end = std::min(s + substrate_step_, group_block_lengthss_[tid.group]);

			// #pragma omp critical
			// 			std::cout << "Thread " << get_thread_num() << " [0, " << tid.y << ", " << tid.z << "]
			// s_begin: "
			// << s
			// 					  << " s_end: " << s + s_step_length << " block_y_begin: " << block_y_begin
			// 					  << " block_y_end: " << block_y_end << " block_z_begin: " << block_z_begin
			// 					  << " block_z_end: " << block_z_end << " group: " << tid.group << " lane_y: " <<
			// lane_id_y
			// 					  << " lane_z: " << lane_id_z << std::endl;

			for (index_t i = 0; i < this->problem_.iterations; i++)
			{
				if (use_thread_distributed_allocation_)
				{
					auto s_slice = noarr::slice<'s'>(block_s_begin, block_s_end - block_s_begin);

					const auto thread_num = get_thread_num();

					auto diag_x = get_diagonal_layout(this->problem_, this->problem_.nx) ^ s_slice;
					auto diag_y = get_diagonal_layout(this->problem_, this->problem_.ny) ^ s_slice;
					auto diag_z = get_diagonal_layout(this->problem_, this->problem_.nz) ^ s_slice;

					auto dist_l = noarr::scalar<real_t*>() ^ get_thread_distribution_layout()
								  ^ noarr::fix<'y', 'g'>(tid.y, tid.group);

					auto dens_l = get_blocked_substrate_layout<'z'>(this->problem_.nx, group_block_lengthsy_[tid.y],
																	group_block_lengthsz_[tid.z],
																	group_block_lengthss_[tid.group]);

					auto sync_z = [densities = thread_substrate_array_.get(), rf_data = thread_az_[thread_num],
								   b_data = thread_bz_[thread_num], c_data = thread_cz_[thread_num], dens_l,
								   diag_l = diag_z, dist_l, n = this->problem_.nz, tid = tid.z,
								   group_size = cores_division_[2], &barrier = barrier_z](index_t s) {
						synchronize_z_blocked_distributed(densities, rf_data, b_data, c_data,
														  dens_l ^ noarr::fix<'s'>(s), diag_l ^ noarr::fix<'s'>(s),
														  dist_l, n, tid, group_size, barrier);
					};

					auto current_densities = dist_l | noarr::get_at<'z'>(thread_substrate_array_.get(), tid.z);

					if (use_alt_blocked_)
						solve_slice_xyz_fused_transpose_blocked<true>(
							current_densities, thread_bx_[thread_num], thread_cx_[thread_num], thread_rf_x_[thread_num],
							thread_by_[thread_num], thread_cy_[thread_num], thread_rf_y_[thread_num],
							thread_az_[thread_num], thread_bz_[thread_num], thread_cz_[thread_num],
							thread_rf_z_[thread_num], thread_rb_z_[thread_num],
							dens_l ^ noarr::set_length<'z'>(block_z_end - block_z_begin), diag_x, diag_y, diag_z, s,
							s_end, block_z_begin, block_z_end, std::move(sync_z));
					else
						solve_slice_xyz_fused_transpose_blocked<false>(
							current_densities, thread_bx_[thread_num], thread_cx_[thread_num], thread_rf_x_[thread_num],
							thread_by_[thread_num], thread_cy_[thread_num], thread_rf_y_[thread_num],
							thread_az_[thread_num], thread_bz_[thread_num], thread_cz_[thread_num],
							thread_rf_z_[thread_num], thread_rb_z_[thread_num],
							dens_l ^ noarr::set_length<'z'>(block_z_end - block_z_begin), diag_x, diag_y, diag_z, s,
							s_end, block_z_begin, block_z_end, std::move(sync_z));
				}
				else
				{
					auto diag_x = get_diagonal_layout(this->problem_, this->problem_.nx);
					auto diag_y = get_diagonal_layout(this->problem_, this->problem_.ny);
					auto diag_z = get_diagonal_layout(this->problem_, this->problem_.nz);

					auto dens_l = get_substrates_layout<3>();

					auto sync_z = [densities = this->substrates_, rf_data = thread_az_[0], b_data = thread_bz_[0],
								   c_data = thread_cz_[0], dens_l, diag_l = diag_z, tid = tid.z,
								   group_size = cores_division_[2], &barrier = barrier_z](index_t s) {
						synchronize_z_blocked(densities, rf_data, b_data, c_data, dens_l ^ noarr::fix<'s'>(s),
											  diag_l ^ noarr::fix<'s'>(s), tid, group_size, barrier);
					};

					if (use_alt_blocked_)
						solve_slice_xyz_fused_transpose_blocked<true>(
							this->substrates_, thread_bx_[0], thread_cx_[0], thread_rf_x_[0], thread_by_[0],
							thread_cy_[0], thread_rf_y_[0], thread_az_[0], thread_bz_[0], thread_cz_[0],
							thread_rf_z_[0], thread_rb_z_[0],
							dens_l ^ noarr::slice<'z'>(block_z_begin, block_z_end - block_z_begin), diag_x, diag_y,
							diag_z, block_s_begin + s, block_s_begin + s_end, block_z_begin, block_z_end,
							std::move(sync_z));
					else
						solve_slice_xyz_fused_transpose_blocked<false>(
							this->substrates_, thread_bx_[0], thread_cx_[0], thread_rf_x_[0], thread_by_[0],
							thread_cy_[0], thread_rf_y_[0], thread_az_[0], thread_bz_[0], thread_cz_[0],
							thread_rf_z_[0], thread_rb_z_[0],
							dens_l ^ noarr::slice<'z'>(block_z_begin, block_z_end - block_z_begin), diag_x, diag_y,
							diag_z, block_s_begin + s, block_s_begin + s_end, block_z_begin, block_z_end,
							std::move(sync_z));
				}
			}
		}
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_f_p<real_t, aligned_x>::solve_blocked_3d_yz()
{
	for (index_t i = 0; i < countersy_count_; i++)
	{
		countersy_[i]->value = 0;
	}

	for (index_t i = 0; i < countersz_count_; i++)
	{
		countersz_[i]->value = 0;
	}

#pragma omp parallel
	{
		perf_counter counter("lstmfpai");

		const thread_id_t<index_t> tid = get_thread_id();

		const auto block_y_begin = group_block_offsetsy_[tid.y];
		const auto block_y_end = block_y_begin + group_block_lengthsy_[tid.y];

		const auto block_z_begin = group_block_offsetsz_[tid.z];
		const auto block_z_end = block_z_begin + group_block_lengthsz_[tid.z];

		const auto lane_id_y = tid.z + tid.group * cores_division_[2];

		barrier_t<true, index_t> barrier_y(cores_division_[1], countersy_[lane_id_y]->value);
		// auto& barrier_y = *barriersy_[lane_id_y];

		const auto lane_id_z = tid.y + tid.group * cores_division_[1];

		barrier_t<true, index_t> barrier_z(cores_division_[2], countersz_[lane_id_z]->value);
		// auto& barrier_z = *barriersz_[lane_id_z];

		const auto block_s_begin = group_block_offsetss_[tid.group];
		const auto block_s_end = block_s_begin + group_block_lengthss_[tid.group];

		for (index_t s = 0; s < group_block_lengthss_[tid.group]; s += substrate_step_)
		{
			auto s_end = std::min(s + substrate_step_, group_block_lengthss_[tid.group]);

			// #pragma omp critical
			// 			std::cout << "Thread " << get_thread_num() << " [0, " << tid.y << ", " << tid.z << "]
			// s_begin:
			// 	"
			// << s
			// 					  << " s_end: " << s + s_step_length << " block_y_begin: " << block_y_begin
			// 					  << " block_y_end: " << block_y_end << " block_z_begin: " << block_z_begin
			// 					  << " block_z_end: " << block_z_end << " group: " << tid.group << " lane_y: " <<
			// lane_id_y
			// 					  << " lane_z: " << lane_id_z << std::endl;

			for (index_t i = 0; i < this->problem_.iterations; i++)
			{
				if (use_thread_distributed_allocation_)
				{
					auto s_slice = noarr::slice<'s'>(block_s_begin, block_s_end - block_s_begin);

					const auto thread_num = get_thread_num();

					auto diag_x = get_diagonal_layout(this->problem_, this->problem_.nx) ^ s_slice;
					auto diag_y = get_diagonal_layout(this->problem_, this->problem_.ny) ^ s_slice;
					auto diag_z = get_diagonal_layout(this->problem_, this->problem_.nz) ^ s_slice;

					auto dist_l =
						noarr::scalar<real_t*>() ^ get_thread_distribution_layout() ^ noarr::fix<'g'>(tid.group);

					auto dens_l = get_blocked_substrate_layout<'*'>(this->problem_.nx, group_block_lengthsy_[tid.y],
																	group_block_lengthsz_[tid.z],
																	group_block_lengthss_[tid.group]);

					auto sync_y = [densities = thread_substrate_array_.get(), rf_data = thread_ay_[thread_num],
								   b_data = thread_by_[thread_num], c_data = thread_cy_[thread_num],
								   dens_l = dens_l ^ noarr::set_length<'z'>(block_z_end - block_z_begin), diag_y,
								   dist_l = dist_l ^ noarr::fix<'z'>(tid.z), n = this->problem_.ny, tid = tid.y,
								   group_size = cores_division_[1], &barrier = barrier_y](index_t z, index_t s) {
						synchronize_y_blocked_distributed(densities, rf_data, b_data, c_data,
														  dens_l ^ noarr::fix<'s'>(s), diag_y ^ noarr::fix<'s'>(s),
														  dist_l, n, z, tid, group_size, barrier);
					};

					auto sync_z = [densities = thread_substrate_array_.get(), rf_data = thread_az_[thread_num],
								   b_data = thread_bz_[thread_num], c_data = thread_cz_[thread_num],
								   dens_l = dens_l ^ noarr::set_length<'y'>(block_y_end - block_y_begin),
								   diag_l = diag_z, dist_l = dist_l ^ noarr::fix<'y'>(tid.y), n = this->problem_.nz,
								   tid = tid.z, group_size = cores_division_[2], &barrier = barrier_z](index_t s) {
						synchronize_z_blocked_distributed(densities, rf_data, b_data, c_data,
														  dens_l ^ noarr::fix<'s'>(s), diag_l ^ noarr::fix<'s'>(s),
														  dist_l, n, tid, group_size, barrier);
					};

					auto current_densities =
						dist_l | noarr::get_at<'y', 'z'>(thread_substrate_array_.get(), tid.y, tid.z);

					if (use_alt_blocked_)
						solve_slice_xyz_fused_transpose_blocked<true>(
							current_densities, thread_bx_[thread_num], thread_cx_[thread_num], thread_rf_x_[thread_num],
							thread_ay_[thread_num], thread_by_[thread_num], thread_cy_[thread_num],
							thread_rf_y_[thread_num], thread_rb_y_[thread_num], thread_az_[thread_num],
							thread_bz_[thread_num], thread_cz_[thread_num], thread_rf_z_[thread_num],
							thread_rb_z_[thread_num],
							dens_l ^ noarr::set_length<'y'>(block_y_end - block_y_begin)
								^ noarr::set_length<'z'>(block_z_end - block_z_begin),
							diag_x, diag_y, diag_z, s, s_end, block_y_begin, block_y_end, block_z_begin, block_z_end,
							std::move(sync_y), std::move(sync_z));
					else
						solve_slice_xyz_fused_transpose_blocked<false>(
							current_densities, thread_bx_[thread_num], thread_cx_[thread_num], thread_rf_x_[thread_num],
							thread_ay_[thread_num], thread_by_[thread_num], thread_cy_[thread_num],
							thread_rf_y_[thread_num], thread_rb_y_[thread_num], thread_az_[thread_num],
							thread_bz_[thread_num], thread_cz_[thread_num], thread_rf_z_[thread_num],
							thread_rb_z_[thread_num],
							dens_l ^ noarr::set_length<'y'>(block_y_end - block_y_begin)
								^ noarr::set_length<'z'>(block_z_end - block_z_begin),
							diag_x, diag_y, diag_z, s, s_end, block_y_begin, block_y_end, block_z_begin, block_z_end,
							std::move(sync_y), std::move(sync_z));
				}
				else
				{
					auto diag_x = get_diagonal_layout(this->problem_, this->problem_.nx);
					auto diag_y = get_diagonal_layout(this->problem_, this->problem_.ny);
					auto diag_z = get_diagonal_layout(this->problem_, this->problem_.nz);

					auto dens_l = get_substrates_layout<3>();

					auto sync_y = [densities = this->substrates_, rf_data = thread_ay_[0], b_data = thread_by_[0],
								   c_data = thread_cy_[0],
								   dens_l = dens_l ^ noarr::slice<'z'>(block_z_begin, block_z_end - block_z_begin),
								   diag_y, tid = tid.y, group_size = cores_division_[1],
								   &barrier = barrier_y](index_t z, index_t s) {
						synchronize_y_blocked(densities, rf_data, b_data, c_data, dens_l ^ noarr::fix<'s'>(s),
											  diag_y ^ noarr::fix<'s'>(s), z, tid, group_size, barrier);
					};

					auto sync_z = [densities = this->substrates_, rf_data = thread_az_[0], b_data = thread_bz_[0],
								   c_data = thread_cz_[0],
								   dens_l = dens_l ^ noarr::slice<'y'>(block_y_begin, block_y_end - block_y_begin),
								   diag_l = diag_z, tid = tid.z, group_size = cores_division_[2],
								   &barrier = barrier_z](index_t s) {
						synchronize_z_blocked(densities, rf_data, b_data, c_data, dens_l ^ noarr::fix<'s'>(s),
											  diag_l ^ noarr::fix<'s'>(s), tid, group_size, barrier);
					};

					if (use_alt_blocked_)
						solve_slice_xyz_fused_transpose_blocked<true>(
							this->substrates_, thread_bx_[0], thread_cx_[0], thread_rf_x_[0], thread_ay_[0],
							thread_by_[0], thread_cy_[0], thread_rf_y_[0], thread_rb_y_[0], thread_az_[0],
							thread_bz_[0], thread_cz_[0], thread_rf_z_[0], thread_rb_z_[0],
							dens_l ^ noarr::slice<'y'>(block_y_begin, block_y_end - block_y_begin)
								^ noarr::slice<'z'>(block_z_begin, block_z_end - block_z_begin),
							diag_x, diag_y, diag_z, block_s_begin + s, block_s_begin + s_end, block_y_begin,
							block_y_end, block_z_begin, block_z_end, std::move(sync_y), std::move(sync_z));
					else
						solve_slice_xyz_fused_transpose_blocked<false>(
							this->substrates_, thread_bx_[0], thread_cx_[0], thread_rf_x_[0], thread_ay_[0],
							thread_by_[0], thread_cy_[0], thread_rf_y_[0], thread_rb_y_[0], thread_az_[0],
							thread_bz_[0], thread_cz_[0], thread_rf_z_[0], thread_rb_z_[0],
							dens_l ^ noarr::slice<'y'>(block_y_begin, block_y_end - block_y_begin)
								^ noarr::slice<'z'>(block_z_begin, block_z_end - block_z_begin),
							diag_x, diag_y, diag_z, block_s_begin + s, block_s_begin + s_end, block_y_begin,
							block_y_end, block_z_begin, block_z_end, std::move(sync_y), std::move(sync_z));
				}
			}
		}
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_f_p<real_t, aligned_x>::solve_partial_blocked_3d()
{
#pragma omp parallel
	{
		perf_counter counter("lstmfppai");

		for (index_t s = 0; s < this->problem_.substrates_count; s += substrate_step_)
		{
			for (index_t i = 0; i < this->problem_.iterations; i++)
			{
				auto s_step_length = std::min(substrate_step_, this->problem_.substrates_count - s);

				solve_slice_xy_fused_transpose_partial(
					this->substrates_, thread_bx_[0], thread_cx_[0], thread_rf_x_[0], thread_by_[0], thread_cy_[0],
					thread_rf_y_[0], get_substrates_layout<3>(), get_diagonal_layout(this->problem_, this->problem_.nx),
					get_diagonal_layout(this->problem_, this->problem_.ny), s, s + s_step_length);
#pragma omp barrier
				solve_slice_z_3d_intrinsics_partial(
					this->substrates_, thread_bz_[0], thread_cz_[0], thread_rf_z_[0], get_substrates_layout<3>(),
					get_diagonal_layout(this->problem_, this->problem_.nz), s, s + s_step_length);
#pragma omp barrier
			}
		}
	}
}

template <typename real_t, bool aligned_x>
least_memory_thomas_solver_d_f_p<real_t, aligned_x>::least_memory_thomas_solver_d_f_p(
	bool use_alt_blocked, bool use_thread_distributed_allocation, bool partial_blocking)
	: countersy_count_(0),
	  countersz_count_(0),
	  use_alt_blocked_(use_alt_blocked),
	  use_thread_distributed_allocation_(use_thread_distributed_allocation),
	  partial_blocking_(partial_blocking)
{}

template <typename real_t, bool aligned_x>
least_memory_thomas_solver_d_f_p<real_t, aligned_x>::~least_memory_thomas_solver_d_f_p()
{
	for (index_t i = 0; i < get_max_threads(); i++)
	{
		if (thread_cx_)
		{
			std::free(thread_rf_x_[i]);
			std::free(thread_bx_[i]);
			std::free(thread_cx_[i]);
		}

		if (thread_cy_)
		{
			std::free(thread_rf_y_[i]);
			std::free(thread_by_[i]);
			std::free(thread_cy_[i]);
		}

		if (thread_cz_)
		{
			std::free(thread_rf_z_[i]);
			std::free(thread_bz_[i]);
			std::free(thread_cz_[i]);
		}

		if (thread_ax_)
		{
			std::free(thread_ax_[i]);
			std::free(thread_rb_x_[i]);
		}

		if (thread_ay_)
		{
			std::free(thread_ay_[i]);
			std::free(thread_rb_y_[i]);
		}

		if (thread_az_)
		{
			std::free(thread_az_[i]);
			std::free(thread_rb_z_[i]);
		}

		if (thread_substrate_array_)
		{
			std::free(thread_substrate_array_[i]);
		}
	}
}

template <typename real_t, bool aligned_x>
double least_memory_thomas_solver_d_f_p<real_t, aligned_x>::access(std::size_t s, std::size_t x, std::size_t y,
																   std::size_t z) const
{
	if (!use_thread_distributed_allocation_)
		return base_solver<real_t, least_memory_thomas_solver_d_f_p<real_t, aligned_x>>::access(s, x, y, z);

	index_t block_idx_y = 0;
	while ((index_t)y >= group_block_offsetsy_[block_idx_y] + group_block_lengthsy_[block_idx_y])
	{
		block_idx_y++;
	}
	y -= group_block_offsetsy_[block_idx_y];

	index_t block_idx_z = 0;
	while ((index_t)z >= group_block_offsetsz_[block_idx_z] + group_block_lengthsz_[block_idx_z])
	{
		block_idx_z++;
	}
	z -= group_block_offsetsz_[block_idx_z];

	index_t block_idx_s = 0;
	while ((index_t)s >= group_block_offsetss_[block_idx_s] + group_block_lengthss_[block_idx_s])
	{
		block_idx_s++;
	}
	s -= group_block_offsetss_[block_idx_s];

	auto dist_l = noarr::scalar<real_t*>() ^ get_thread_distribution_layout()
				  ^ noarr::fix<'s', 'y', 'z'>(block_idx_s, block_idx_y, block_idx_z);

	auto density =
		dist_l | noarr::get_at<'g', 'y', 'z'>(thread_substrate_array_.get(), block_idx_s, block_idx_y, block_idx_z);

	auto dens_l = get_blocked_substrate_layout(this->problem_.nx, group_block_lengthsy_[block_idx_y],
											   group_block_lengthsz_[block_idx_z], group_block_lengthss_[block_idx_s]);

	return dens_l | noarr::get_at<'x', 'y', 'z', 's'>(density, x, y, z, s);
}

template class least_memory_thomas_solver_d_f_p<float, true>;
template class least_memory_thomas_solver_d_f_p<double, true>;
