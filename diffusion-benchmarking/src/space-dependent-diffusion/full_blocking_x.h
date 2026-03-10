#pragma once

#include "../vector_transpose_helper.h"
#include "noarr/structures.hpp"
#include "noarr/structures/interop/bag.hpp"
#include "noarr/structures/structs/blocks.hpp"
#include "noarr/structures_extended.hpp"


template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename thread_distribution_l>
constexpr static void synchronize_x_blocked_distributed_remainder_nf(
	real_t** __restrict__ densities, real_t** __restrict__ a_data, real_t** __restrict__ c_data,
	const density_layout_t dens_l, const diagonal_layout_t diag_l, const thread_distribution_l dist_l, const index_t n,
	const index_t n_alignment, const index_t y_begin, const index_t y_end, const index_t z, const index_t coop_size)
{
	const index_t block_size = n / coop_size;

	const auto ddiag_l = diag_l ^ noarr::merge_blocks<'Y', 'y'>() ^ noarr::fix<'z'>(z);
	const auto ddesn_l = dens_l ^ noarr::fix<'z'>(z);

	// #pragma omp critical
	// 	std::cout << "Thread " << tid << " block_begin: " << x_simd_begin << " block_end: " << x_simd_end
	// 			  << " block_size: " << block_size_x << std::endl;

	auto get_i = [block_size, n, n_alignment, coop_size](index_t equation_idx) {
		const index_t block_idx = equation_idx / 2;
		const auto actual_block_size = (block_idx < n % coop_size) ? block_size + 1 : block_size;
		const auto offset = (equation_idx % 2) * (actual_block_size - 1);
		const auto actual_block_size_aligned = (actual_block_size + n_alignment - 1) / n_alignment * n_alignment;

		return std::make_tuple(block_idx, noarr::set_length<'x'>(actual_block_size_aligned) ^ noarr::fix<'x'>(offset),
							   noarr::set_length<'x'>(actual_block_size) ^ noarr::fix<'x'>(offset));
	};

	for (index_t y = y_begin; y < y_end; y++)
	{
		real_t prev_c;
		real_t prev_d;

		{
			const auto [prev_block_idx, fix_dens, fix_diag] = get_i(0);
			const auto prev_c_bag =
				noarr::make_bag(ddiag_l ^ fix_diag, dist_l | noarr::get_at<'x'>(c_data, prev_block_idx));
			const auto prev_d_bag =
				noarr::make_bag(ddesn_l ^ fix_dens, dist_l | noarr::get_at<'x'>(densities, prev_block_idx));

			prev_c = prev_c_bag.template at<'y'>(y);
			prev_d = prev_d_bag.template at<'y'>(y);
		}

		for (index_t equation_idx = 1; equation_idx < coop_size * 2; equation_idx++)
		{
			const auto [block_idx, fix_dens, fix_diag] = get_i(equation_idx);

			const auto a = noarr::make_bag(ddiag_l ^ fix_diag, dist_l | noarr::get_at<'x'>(a_data, block_idx));
			const auto c = noarr::make_bag(ddiag_l ^ fix_diag, dist_l | noarr::get_at<'x'>(c_data, block_idx));
			const auto d = noarr::make_bag(ddesn_l ^ fix_dens, dist_l | noarr::get_at<'x'>(densities, block_idx));

			real_t curr_a = a.template at<'y'>(y);
			real_t curr_c = c.template at<'y'>(y);
			real_t curr_d = d.template at<'y'>(y);

			real_t r = 1 / (1 - prev_c * curr_a);

			curr_d = r * (curr_d - prev_d * curr_a);
			curr_c = r * curr_c;

			c.template at<'y'>(y) = curr_c;
			d.template at<'y'>(y) = curr_d;

			prev_c = curr_c;
			prev_d = curr_d;

			// #pragma omp critical
			// 				{
			// 					for (index_t l = 0; l < simd_length; l++)
			// 						std::cout << "mb " << z << " " << y + l << " " << equation_idx << " "
			// 								  << hn::ExtractLane(curr_a, l) << " " << hn::ExtractLane(r, l) << " "
			// 								  << hn::ExtractLane(curr_d, l) << std::endl;
			// 				}
		}

		for (index_t equation_idx = coop_size * 2 - 2; equation_idx >= 0; equation_idx--)
		{
			const auto [block_idx, fix_dens, fix_diag] = get_i(equation_idx);

			const auto c = noarr::make_bag(ddiag_l ^ fix_diag, dist_l | noarr::get_at<'x'>(c_data, block_idx));
			const auto d = noarr::make_bag(ddesn_l ^ fix_dens, dist_l | noarr::get_at<'x'>(densities, block_idx));

			real_t curr_c = c.template at<'y'>(y);
			real_t curr_d = d.template at<'y'>(y);

			curr_d = curr_d - prev_d * curr_c;

			d.template at<'y'>(y) = curr_d;

			prev_d = curr_d;

			// #pragma omp critical
			// 				{
			// 					for (index_t l = 0; l < simd_length; l++)
			// 						std::cout << "mf " << z << " " << y + l << " " << equation_idx << " "
			// 								  << hn::ExtractLane(curr_c, l) << " " << hn::ExtractLane(curr_d, l) <<
			// std::endl;
			// 				}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename thread_distribution_l, typename barrier_t>
constexpr static void synchronize_x_blocked_distributed_nf(
	real_t** __restrict__ densities, real_t** __restrict__ a_data, real_t** __restrict__ c_data,
	const density_layout_t dens_l, const diagonal_layout_t diag_l, const thread_distribution_l dist_l, const index_t n,
	const index_t n_alignment, const index_t z_begin, const index_t z_end, const index_t tid, const index_t coop_size,
	barrier_t& barrier)
{
	barrier.arrive();

	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	const index_t y_len = dens_l | noarr::get_length<'y'>();
	const index_t simd_y_len = y_len / simd_length;

	const index_t block_size = n / coop_size;

	const index_t z_len = z_end - z_begin;
	const index_t block_size_z = z_len / coop_size;
	const index_t t_z_begin = z_begin + tid * block_size_z + std::min(tid, z_len % coop_size);
	const index_t t_z_end = t_z_begin + block_size_z + ((tid < z_len % coop_size) ? 1 : 0);

	// #pragma omp critical
	// 	std::cout << "Thread " << tid << " block_begin: " << x_simd_begin << " block_end: " << x_simd_end
	// 			  << " block_size: " << block_size_x << std::endl;

	barrier.wait();

	auto get_i = [block_size, n, n_alignment, coop_size](index_t equation_idx, index_t y) {
		const index_t block_idx = equation_idx / 2;
		const auto actual_block_size = (block_idx < n % coop_size) ? block_size + 1 : block_size;
		const auto offset = (equation_idx % 2) * (actual_block_size - 1);

		const auto transposed_y_offset = offset % simd_length;
		const auto transposed_x = offset - transposed_y_offset;

		const auto actual_block_size_aligned = (actual_block_size + n_alignment - 1) / n_alignment * n_alignment;

		return std::make_tuple(block_idx,
							   noarr::set_length<'x'>(actual_block_size_aligned)
								   ^ noarr::fix<'x', 'y'>(transposed_x, y * simd_length + transposed_y_offset),
							   noarr::set_length<'x'>(actual_block_size)
								   ^ noarr::fix<'x', 'Y', 'y'>(offset, y, noarr::lit<0>));
	};

	for (index_t z = t_z_begin; z < t_z_end; z++)
	{
		for (index_t y = 0; y < simd_y_len; y++)
		{
			simd_t prev_c;
			simd_t prev_d;

			{
				const auto [prev_block_idx, fix_dens, fix_diag] = get_i(0, y);
				const auto prev_c_bag =
					noarr::make_bag(diag_l ^ fix_diag, dist_l | noarr::get_at<'x'>(c_data, prev_block_idx));
				const auto prev_d_bag =
					noarr::make_bag(dens_l ^ fix_dens, dist_l | noarr::get_at<'x'>(densities, prev_block_idx));

				prev_c = hn::Load(t, &prev_c_bag.template at<'z'>(z - z_begin));
				prev_d = hn::Load(t, &prev_d_bag.template at<'z'>(z));
			}

			for (index_t equation_idx = 1; equation_idx < coop_size * 2; equation_idx++)
			{
				const auto [block_idx, fix_dens, fix_diag] = get_i(equation_idx, y);

				const auto a = noarr::make_bag(diag_l ^ fix_diag, dist_l | noarr::get_at<'x'>(a_data, block_idx));
				const auto c = noarr::make_bag(diag_l ^ fix_diag, dist_l | noarr::get_at<'x'>(c_data, block_idx));
				const auto d = noarr::make_bag(dens_l ^ fix_dens, dist_l | noarr::get_at<'x'>(densities, block_idx));

				simd_t curr_a = hn::Load(t, &a.template at<'z'>(z - z_begin));
				simd_t curr_c = hn::Load(t, &c.template at<'z'>(z - z_begin));
				simd_t curr_d = hn::Load(t, &d.template at<'z'>(z));

				simd_t r = hn::Div(hn::Set(t, 1), hn::NegMulAdd(prev_c, curr_a, hn::Set(t, 1)));

				curr_d = hn::Mul(r, hn::NegMulAdd(prev_d, curr_a, curr_d));
				curr_c = hn::Mul(r, curr_c);

				hn::Store(curr_c, t, &c.template at<'z'>(z - z_begin));
				hn::Store(curr_d, t, &d.template at<'z'>(z));

				prev_c = curr_c;
				prev_d = curr_d;

				// #pragma omp critical
				// 				{
				// 					for (index_t l = 0; l < simd_length; l++)
				// 						std::cout << "mb " << z << " " << y + l << " " << equation_idx << " "
				// 								  << hn::ExtractLane(curr_a, l) << " " << hn::ExtractLane(r, l) << " "
				// 								  << hn::ExtractLane(curr_d, l) << std::endl;
				// 				}
			}

			for (index_t equation_idx = coop_size * 2 - 2; equation_idx >= 0; equation_idx--)
			{
				const auto [block_idx, fix_dens, fix_diag] = get_i(equation_idx, y);

				const auto c = noarr::make_bag(diag_l ^ fix_diag, dist_l | noarr::get_at<'x'>(c_data, block_idx));
				const auto d = noarr::make_bag(dens_l ^ fix_dens, dist_l | noarr::get_at<'x'>(densities, block_idx));

				simd_t curr_c = hn::Load(t, &c.template at<'z'>(z - z_begin));
				simd_t curr_d = hn::Load(t, &d.template at<'z'>(z));

				curr_d = hn::NegMulAdd(prev_d, curr_c, curr_d);

				hn::Store(curr_d, t, &d.template at<'z'>(z));

				prev_d = curr_d;

				// #pragma omp critical
				// 				{
				// 					for (index_t l = 0; l < simd_length; l++)
				// 						std::cout << "mf " << z << " " << y + l << " " << equation_idx << " "
				// 								  << hn::ExtractLane(curr_c, l) << " " << hn::ExtractLane(curr_d, l) <<
				// std::endl;
				// 				}
			}
		}

		synchronize_x_blocked_distributed_remainder_nf(
			densities, a_data, c_data, dens_l ^ noarr::slice<'z'>(z_begin, z_len), diag_l, dist_l, n, n_alignment,
			simd_y_len * simd_length, y_len, z - z_begin, coop_size);
	}

	barrier.arrive_and_wait();
}

template <bool begin, typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename scratch_layout_t>
static void solve_block_x_remainder_nf(real_t* __restrict__ densities, const real_t* __restrict__ a,
									   const real_t* __restrict__ b, const real_t* __restrict__ c,
									   real_t* __restrict__ a_scratch, real_t* __restrict__ c_scratch,
									   const density_layout_t dens_l, const diagonal_layout_t diag_l,
									   const scratch_layout_t scratch_l, const index_t n, const index_t s,
									   const index_t z, const index_t y_begin, const index_t y_end)
{
	auto merge_l = noarr::merge_blocks<'Y', 'y'>();
	auto a_scratch_bag = noarr::make_bag(scratch_l ^ merge_l, a_scratch);
	auto c_scratch_bag = noarr::make_bag(scratch_l ^ merge_l, c_scratch);

	auto diag_fix = noarr::fix<'s'>(s) ^ merge_l;
	auto a_bag = noarr::make_bag(diag_l ^ diag_fix, a);
	auto b_bag = noarr::make_bag(diag_l ^ diag_fix, b);
	auto c_bag = noarr::make_bag(diag_l ^ diag_fix, c);
	auto d_bag = noarr::make_bag(dens_l, densities);

	if constexpr (begin)
	{
		for (index_t y = y_begin; y < y_end; y++)
		{
			// Normalize the first and the second equation
			for (index_t i = 0; i < 2; i++)
			{
				const auto idx = noarr::idx<'z', 'y', 'x'>(z, y, i);

				const auto r = 1 / b_bag[idx];

				a_scratch_bag[idx] = a_bag[idx] * r;
				c_scratch_bag[idx] = c_bag[idx] * r;
				d_bag[idx] = d_bag[idx] * r;

				// #pragma omp critical
				// 				std::cout << "f0: " << z_begin + z << " " << y_begin + i << " " << x << " " <<
				// d_bag[idx] << " "
				// 						  << b_bag[idx] << std::endl;
			}

			// Process the lower diagonal (forward)
			for (index_t i = 2; i < n; i++)
			{
				const auto prev_idx = noarr::idx<'z', 'y', 'x'>(z, y, i - 1);
				const auto idx = noarr::idx<'z', 'y', 'x'>(z, y, i);

				const auto r = 1 / (b_bag[idx] - a_bag[idx] * c_scratch_bag[prev_idx]);

				a_scratch_bag[idx] = r * (0 - a_bag[idx] * a_scratch_bag[prev_idx]);
				c_scratch_bag[idx] = r * c_bag[idx];

				d_bag[idx] = r * (d_bag[idx] - a_bag[idx] * d_bag[prev_idx]);


				// #pragma omp critical
				// 				std::cout << "f1: " << z_begin + z << " " << i + y_begin << " " << x << " " <<
				// d_bag[idx] << " "
				// 						  << a_bag[idx] << " " << b_bag[idx] << " " << c_scratch_bag[idx] << std::endl;
			}

			// Process the upper diagonal (backward)
			for (index_t i = n - 3; i >= 1; i--)
			{
				const auto idx = noarr::idx<'z', 'y', 'x'>(z, y, i);
				const auto next_idx = noarr::idx<'z', 'y', 'x'>(z, y, i + 1);

				d_bag[idx] = d_bag[idx] - c_scratch_bag[idx] * d_bag[next_idx];

				a_scratch_bag[idx] = a_scratch_bag[idx] - c_scratch_bag[idx] * a_scratch_bag[next_idx];
				c_scratch_bag[idx] = 0 - c_scratch_bag[idx] * c_scratch_bag[next_idx];


				// #pragma omp critical
				// 				std::cout << "b0: " << z_begin + z << " " << i + y_begin << " " << x << " " <<
				// d_bag[idx] << std::endl;
			}

			// Process the first row (backward)
			{
				const auto idx = noarr::idx<'z', 'y', 'x'>(z, y, 0);
				const auto next_idx = noarr::idx<'z', 'y', 'x'>(z, y, 1);

				const auto r = 1 / (1 - c_scratch_bag[idx] * a_scratch_bag[next_idx]);

				d_bag[idx] = r * (d_bag[idx] - c_scratch_bag[idx] * d_bag[next_idx]);

				a_scratch_bag[idx] = r * a_scratch_bag[idx];
				c_scratch_bag[idx] = r * (0 - c_scratch_bag[idx] * c_scratch_bag[next_idx]);


				// #pragma omp critical
				// 			std::cout << "b1: " << z_begin + z << " " << y_begin << " " << x << " " << d_bag[idx] <<
				// std::endl;
			}
		}
	}
	else
	{
		// Final part of modified thomas algorithm
		// Solve the rest of the unknowns
		for (index_t y = y_begin; y < y_end; y++)
		{
			for (index_t i = 1; i < n - 1; i++)
			{
				const auto idx_begin = noarr::idx<'z', 'y', 'x'>(z, y, 0);
				const auto idx = noarr::idx<'z', 'y', 'x'>(z, y, i);
				const auto idx_end = noarr::idx<'z', 'y', 'x'>(z, y, n - 1);

				d_bag[idx] = d_bag[idx] - a_scratch_bag[idx] * d_bag[idx_begin] - c_scratch_bag[idx] * d_bag[idx_end];

				// #pragma omp critical
				// 						std::cout << "l: " << z_begin +z << " " << i << " " << x << " "
				// 								  << d.template at<'s', 'x', 'z', 'y'>(s, x, z, i) << " " <<
				// a[state] << " " << c[state]
				// 								  << std::endl;
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename scratch_layout_t, typename sync_func_t>
static void solve_block_x_transpose_nf(real_t* __restrict__ densities, const real_t* __restrict__ a,
									   const real_t* __restrict__ b, const real_t* __restrict__ c,
									   real_t* __restrict__ a_scratch, real_t* __restrict__ c_scratch,
									   const density_layout_t dens_l, const diagonal_layout_t diag_l,
									   const scratch_layout_t scratch_l, const index_t x_begin, const index_t x_end,
									   const index_t z_begin, const index_t z_end, const index_t s, const index_t,
									   sync_func_t&& synchronize_blocked_x)
{
	auto blocked_dens_l = dens_l ^ noarr::fix<'s'>(s);

	const index_t n = x_end - x_begin;
	const index_t y_len = blocked_dens_l | noarr::get_length<'y'>();

	auto a_scratch_bag = noarr::make_bag(scratch_l ^ noarr::fix<'y'>(noarr::lit<0>), a_scratch);
	auto c_scratch_bag = noarr::make_bag(scratch_l ^ noarr::fix<'y'>(noarr::lit<0>), c_scratch);

	const auto step_len = z_end - z_begin;

	auto diag_fix = noarr::fix<'s', 'y'>(s, noarr::lit<0>) ^ noarr::slice<'z'>(z_begin, step_len);
	auto a_bag = noarr::make_bag(diag_l ^ diag_fix, a);
	auto b_bag = noarr::make_bag(diag_l ^ diag_fix, b);
	auto c_bag = noarr::make_bag(diag_l ^ diag_fix, c);
	auto d_bag = noarr::make_bag(blocked_dens_l ^ noarr::slice<'z'>(z_begin, step_len), densities);

	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag d;
	constexpr index_t simd_length = hn::Lanes(d);
	using simd_t = hn::Vec<simd_tag>;

	simd_t d_rows[simd_length];

	const index_t simd_y_len = y_len / simd_length;

	for (index_t z = 0; z < step_len; z++)
	{
		const index_t full_n = (n + simd_length - 1) / simd_length * simd_length;

		for (index_t y = 0; y < simd_y_len; y++)
		{
			// vector registers that hold the to be transposed x*yz plane

			simd_t d_prev = hn::Zero(d);
			simd_t a_scratch_prev = hn::Set(d, -1);
			simd_t c_scratch_prev = hn::Zero(d);

			// forward substitution until last simd_length elements
			for (index_t i = 0; i < full_n - simd_length; i += simd_length)
			{
				// aligned loads
				for (index_t v = 0; v < simd_length; v++)
					d_rows[v] = hn::Load(d, &d_bag.template at<'z', 'y', 'x'>(z, y * simd_length + v, i));

				// transposition to enable vectorization
				transpose(d_rows);

				for (index_t v = 0; v < simd_length; v++)
				{
					const index_t x = i + v;

					const auto idx = noarr::idx<'z', 'Y', 'x'>(z, y, x);

					simd_t a_curr = hn::Load(d, &(a_bag[idx]));
					simd_t b_curr = hn::Load(d, &(b_bag[idx]));
					simd_t c_curr = hn::Load(d, &(c_bag[idx]));
					simd_t a_scratch_curr = hn::Load(d, &a_scratch_bag[idx]);
					simd_t c_scratch_curr = hn::Load(d, &c_scratch_bag[idx]);

					{
						simd_t r = hn::Div(hn::Set(d, 1), hn::NegMulAdd(a_curr, c_scratch_prev, b_curr));

						a_scratch_curr = hn::Mul(r, hn::NegMulAdd(a_curr, a_scratch_prev, hn::Set(d, 0)));
						c_scratch_curr = hn::Mul(r, c_curr);
						d_rows[v] = hn::Mul(r, hn::NegMulAdd(a_curr, d_prev, d_rows[v]));

						// #pragma omp critical
						// 						{
						// 							for (index_t l = 0; l < simd_length; l++)
						// 								std::cout << "f " << z_begin + z << " " << y * simd_length + l
						// << " " << x_begin + x
						// 										  << " " << hn::ExtractLane(a_curr, l) << " " <<
						// hn::ExtractLane(b_curr, l)
						// 										  << " " << hn::ExtractLane(r, l) << " " <<
						// hn::ExtractLane(d_rows[v], l)
						// 										  << std::endl;
						// 						}
					}

					if (x >= 1)
					{
						d_prev = d_rows[v];
						a_scratch_prev = a_scratch_curr;
						c_scratch_prev = c_scratch_curr;
					}
					hn::Store(a_scratch_curr, d, &a_scratch_bag[idx]);
					hn::Store(c_scratch_curr, d, &c_scratch_bag[idx]);
				}

				// aligned stores
				for (index_t v = 0; v < simd_length; v++)
					hn::Store(d_rows[v], d, &(d_bag.template at<'z', 'y', 'x'>(z, y * simd_length + v, i)));
			}

			// we are aligned to the vector size, so we can safely continue
			// here we fuse the end of forward substitution and the beginning of backwards propagation
			{
				for (index_t v = 0; v < simd_length; v++)
					d_rows[v] =
						hn::Load(d, &(d_bag.template at<'z', 'y', 'x'>(z, y * simd_length + v, full_n - simd_length)));

				// transposition to enable vectorization
				transpose(d_rows);

				index_t remainder_work = n % simd_length;
				remainder_work += remainder_work == 0 ? simd_length : 0;

				// the rest of forward part
				for (index_t v = 0; v < remainder_work; v++)
				{
					const index_t x = full_n - simd_length + v;

					const auto idx = noarr::idx<'z', 'Y', 'x'>(z, y, x);

					simd_t a_curr = hn::Load(d, &(a_bag[idx]));
					simd_t b_curr = hn::Load(d, &(b_bag[idx]));
					simd_t c_curr = hn::Load(d, &(c_bag[idx]));
					simd_t a_scratch_curr = hn::Load(d, &a_scratch_bag[idx]);
					simd_t c_scratch_curr = hn::Load(d, &c_scratch_bag[idx]);

					// if (x < 2)
					// {
					// 	simd_t r = hn::Div(hn::Set(d, 1), b_curr);

					// 	a_scratch_curr = hn::Mul(a_curr, r);
					// 	c_scratch_curr = hn::Mul(c_curr, r);
					// 	d_rows[v] = hn::Mul(d_rows[v], r);
					// }
					// else
					{
						simd_t r = hn::Div(hn::Set(d, 1), hn::NegMulAdd(a_curr, c_scratch_prev, b_curr));

						a_scratch_curr = hn::Mul(r, hn::NegMulAdd(a_curr, a_scratch_prev, hn::Set(d, 0)));
						c_scratch_curr = hn::Mul(r, c_curr);
						d_rows[v] = hn::Mul(r, hn::NegMulAdd(a_curr, d_prev, d_rows[v]));

						// #pragma omp critical
						// 						{
						// 							for (index_t l = 0; l < simd_length; l++)
						// 								std::cout << "f " << z_begin + z << " " << y * simd_length + l
						// << " " << x_begin + x
						// 										  << " " << hn::ExtractLane(a_curr, l) << " " <<
						// hn::ExtractLane(b_curr, l)
						// 										  << " " << hn::ExtractLane(r, l) << " " <<
						// hn::ExtractLane(d_rows[v], l)
						// 										  << std::endl;
						// 						}
					}

					if (x != 0 && x != n - 1)
					{
						d_prev = d_rows[v];
						a_scratch_prev = a_scratch_curr;
						c_scratch_prev = c_scratch_curr;
					}

					hn::Store(a_scratch_curr, d, &a_scratch_bag[idx]);
					hn::Store(c_scratch_curr, d, &c_scratch_bag[idx]);
				}

				// the begin of backward part
				for (index_t v = remainder_work - 3; v >= 0; v--)
				{
					const index_t x = full_n - simd_length + v;

					const auto idx = noarr::idx<'z', 'Y', 'x'>(z, y, x);

					simd_t a_scratch_curr = hn::Load(d, &a_scratch_bag[idx]);
					simd_t c_scratch_curr = hn::Load(d, &c_scratch_bag[idx]);

					if (x <= n - 3 && x >= 1)
					{
						d_rows[v] = hn::NegMulAdd(c_scratch_curr, d_prev, d_rows[v]);

						// #pragma omp critical
						// 						{
						// 							for (index_t l = 0; l < simd_length; l++)
						// 								std::cout << "b " << z_begin + z << " " << y + l << " " <<
						// x_begin + x << " "
						// 										  << hn::ExtractLane(a_scratch_curr, l) << " "
						// 										  << hn::ExtractLane(c_scratch_curr, l) << " " << 1 << "
						// "
						// 										  << hn::ExtractLane(d_rows[v], l) << std::endl;
						// 						}

						a_scratch_curr = hn::NegMulAdd(c_scratch_curr, a_scratch_prev, a_scratch_curr);
						c_scratch_curr = hn::NegMulAdd(c_scratch_curr, c_scratch_prev, hn::Set(d, 0));
					}
					else if (x == 0)
					{
						simd_t r = hn::Div(hn::Set(d, 1), hn::NegMulAdd(c_scratch_curr, a_scratch_prev, hn::Set(d, 1)));

						d_rows[v] = hn::Mul(r, hn::NegMulAdd(c_scratch_curr, d_prev, d_rows[v]));

						// #pragma omp critical
						// 						{
						// 							for (index_t l = 0; l < simd_length; l++)
						// 								std::cout << "b " << z_begin + z << " " << y + l << " " <<
						// x_begin + x << " "
						// 										  << hn::ExtractLane(a_scratch_curr, l) << " "
						// 										  << hn::ExtractLane(c_scratch_curr, l) << " " <<
						// hn::ExtractLane(r, l) << " "
						// 										  << hn::ExtractLane(d_rows[v], l) << std::endl;
						// 						}

						a_scratch_curr = hn::Mul(r, a_scratch_curr);
						c_scratch_curr = hn::Mul(r, hn::NegMulAdd(c_scratch_curr, c_scratch_prev, hn::Set(d, 0)));
					}

					d_prev = d_rows[v];
					a_scratch_prev = a_scratch_curr;
					c_scratch_prev = c_scratch_curr;
					hn::Store(a_scratch_curr, d, &a_scratch_bag[idx]);
					hn::Store(c_scratch_curr, d, &c_scratch_bag[idx]);
				}

				// aligned stores
				for (index_t v = 0; v < simd_length; v++)
					hn::Store(d_rows[v], d,
							  &(d_bag.template at<'z', 'y', 'x'>(z, y * simd_length + v, full_n - simd_length)));
			}

			// we continue with backwards substitution
			for (index_t i = full_n - simd_length * 2; i >= 0; i -= simd_length)
			{
				// aligned loads
				for (index_t v = 0; v < simd_length; v++)
					d_rows[v] = hn::Load(d, &(d_bag.template at<'z', 'y', 'x'>(z, y * simd_length + v, i)));

				// backward propagation
				for (index_t v = simd_length - 1; v >= 0; v--)
				{
					const index_t x = i + v;

					const auto idx = noarr::idx<'z', 'Y', 'x'>(z, y, x);

					simd_t a_scratch_curr = hn::Load(d, &a_scratch_bag[idx]);
					simd_t c_scratch_curr = hn::Load(d, &c_scratch_bag[idx]);

					if (x <= n - 3 && x >= 1)
					{
						d_rows[v] = hn::NegMulAdd(c_scratch_curr, d_prev, d_rows[v]);

						// #pragma omp critical
						// 						{
						// 							for (index_t l = 0; l < simd_length; l++)
						// 								std::cout << "b " << z_begin + z << " " << y + l << " " <<
						// x_begin + x << " "
						// 										  << hn::ExtractLane(a_scratch_curr, l) << " "
						// 										  << hn::ExtractLane(c_scratch_curr, l) << " " << 1 << "
						// "
						// 										  << hn::ExtractLane(d_rows[v], l) << std::endl;
						// 						}

						a_scratch_curr = hn::NegMulAdd(c_scratch_curr, a_scratch_prev, a_scratch_curr);
						c_scratch_curr = hn::NegMulAdd(c_scratch_curr, c_scratch_prev, hn::Set(d, 0));
					}
					else if (x == 0)
					{
						simd_t r = hn::Div(hn::Set(d, 1), hn::NegMulAdd(c_scratch_curr, a_scratch_prev, hn::Set(d, 1)));
						d_rows[v] = hn::Mul(r, hn::NegMulAdd(c_scratch_curr, d_prev, d_rows[v]));

						// #pragma omp critical
						// 						{
						// 							for (index_t l = 0; l < simd_length; l++)
						// 								std::cout << "b " << z_begin + z << " " << y + l << " " <<
						// x_begin + x << " "
						// 										  << hn::ExtractLane(a_scratch_curr, l) << " "
						// 										  << hn::ExtractLane(c_scratch_curr, l) << " " <<
						// hn::ExtractLane(r, l) << " "
						// 										  << hn::ExtractLane(d_rows[v], l) << std::endl;
						// 						}

						a_scratch_curr = hn::Mul(r, a_scratch_curr);
						c_scratch_curr = hn::Mul(r, hn::NegMulAdd(c_scratch_curr, c_scratch_prev, hn::Set(d, 0)));
					}

					d_prev = d_rows[v];
					a_scratch_prev = a_scratch_curr;
					c_scratch_prev = c_scratch_curr;
					hn::Store(a_scratch_curr, d, &a_scratch_bag[idx]);
					hn::Store(c_scratch_curr, d, &c_scratch_bag[idx]);
				}

				// aligned stores
				for (index_t v = 0; v < simd_length; v++)
					hn::Store(d_rows[v], d, &(d_bag.template at<'z', 'y', 'x'>(z, y * simd_length + v, i)));
			}
		}

		solve_block_x_remainder_nf<true>(densities, a, b, c, a_scratch, c_scratch,
										 blocked_dens_l ^ noarr::slice<'z'>(z_begin, step_len), diag_l, scratch_l, n, s,
										 z, simd_y_len * simd_length, y_len);
	}

	synchronize_blocked_x(z_begin, z_end);

	for (index_t z = 0; z < step_len; z++)
	{
		for (index_t y = 0; y < simd_y_len; y++)
		{
			const index_t full_n = (n + simd_length - 1) / simd_length * simd_length;

			const simd_t begin_unknowns = hn::Load(d, &(d_bag.template at<'y', 'z', 'x'>(y * simd_length, z, 0)));

			const auto transposed_y_offset = (n - 1) % simd_length;
			const auto transposed_x = (n - 1) - transposed_y_offset;

			const simd_t end_unknowns = hn::Load(
				d, &(d_bag.template at<'y', 'z', 'x'>(y * simd_length + transposed_y_offset, z, transposed_x)));

			for (index_t i = 0; i < full_n; i += simd_length)
			{
				for (index_t v = 0; v < simd_length; v++)
					d_rows[v] = hn::Load(d, &(d_bag.template at<'z', 'y', 'x'>(z, y * simd_length + v, i)));

				for (index_t v = 0; v < simd_length; v++)
				{
					index_t x = i + v;

					const auto idx = noarr::idx<'z', 'Y', 'x'>(z, y, x);

					if (x > 0 && x < n - 1)
					{
						simd_t a_scratch_curr = hn::Load(d, &a_scratch_bag[idx]);
						simd_t c_scratch_curr = hn::Load(d, &c_scratch_bag[idx]);

						d_rows[v] = hn::NegMulAdd(a_scratch_curr, begin_unknowns, d_rows[v]);
						d_rows[v] = hn::NegMulAdd(c_scratch_curr, end_unknowns, d_rows[v]);

						// #pragma omp critical
						// 						{
						// 							for (index_t l = 0; l < simd_length; l++)
						// 								std::cout << "e " << z_begin + z << " " << y + l << " " <<
						// x_begin + x << " "
						// 										  << hn::ExtractLane(a_scratch_curr, l) << " "
						// 										  << hn::ExtractLane(c_scratch_curr, l) << " " <<
						// hn::ExtractLane(d_rows[v], l)
						// 										  << std::endl;
						// 						}
					}
				}

				transpose(d_rows);

				for (index_t v = 0; v < simd_length; v++)
					hn::Store(d_rows[v], d, &(d_bag.template at<'z', 'y', 'x'>(z, y * simd_length + v, i)));
			}
		}

		solve_block_x_remainder_nf<false>(densities, a, b, c, a_scratch, c_scratch,
										  blocked_dens_l ^ noarr::slice<'z'>(z_begin, step_len), diag_l, scratch_l, n,
										  s, z, simd_y_len * simd_length, y_len);
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename scratch_layout_t>
static void solve_slice_x_2d_and_3d_transpose_l_nf(real_t* __restrict__ densities, const real_t* __restrict__ a,
												   const real_t* __restrict__ b, const real_t* __restrict__ c,
												   real_t* __restrict__ b_scratch, const density_layout_t dens_l,
												   const diagonal_layout_t diag_l, const scratch_layout_t scratch_l,
												   const index_t s, const index_t z, index_t n)
{
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag d;
	constexpr index_t simd_length = hn::Lanes(d);
	using simd_t = hn::Vec<simd_tag>;

	simd_t d_rows[simd_length];

	const index_t y_len = dens_l | noarr::get_length<'y'>();

	const index_t Y_len = y_len / simd_length;

	// vectorized body
	{
		const index_t full_n = (n + simd_length - 1) / simd_length * simd_length;

		for (index_t Y = 0; Y < Y_len; Y++)
		{
			// vector registers that hold the to be transposed x*yz plane

			simd_t c_prev = hn::Zero(d);
			simd_t d_prev = hn::Zero(d);
			simd_t scratch_prev = hn::Zero(d);

			// forward substitution until last simd_length elements
			for (index_t i = 0; i < full_n - simd_length; i += simd_length)
			{
				// aligned loads
				for (index_t v = 0; v < simd_length; v++)
				{
					d_rows[v] = hn::Load(
						d, &(dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, z, Y * simd_length + v, i, s)));
				}

				// transposition to enable vectorization
				transpose(d_rows);

				for (index_t v = 0; v < simd_length; v++)
				{
					simd_t a_curr =
						hn::Load(d, &(diag_l | noarr::get_at<'z', 'Y', 'y', 'x', 's'>(a, z, Y, 0, i + v, s)));
					simd_t b_curr =
						hn::Load(d, &(diag_l | noarr::get_at<'z', 'Y', 'y', 'x', 's'>(b, z, Y, 0, i + v, s)));

					auto r = hn::Mul(a_curr, scratch_prev);

					scratch_prev = hn::Div(hn::Set(d, 1), hn::NegMulAdd(c_prev, r, b_curr));
					hn::Store(scratch_prev, d, &(scratch_l | noarr::get_at<'x', 'v'>(b_scratch, i + v, 0)));

					d_rows[v] = hn::NegMulAdd(d_prev, r, d_rows[v]);

					d_prev = d_rows[v];
					c_prev = hn::Load(d, &(diag_l | noarr::get_at<'z', 'Y', 'y', 'x', 's'>(c, z, Y, 0, i + v, s)));
				}

				// aligned stores
				for (index_t v = 0; v < simd_length; v++)
				{
					hn::Store(d_rows[v], d,
							  &(dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, z, Y * simd_length + v, i, s)));
				}
			}

			// we are aligned to the vector size, so we can safely continue
			// here we fuse the end of forward substitution and the beginning of backwards propagation
			{
				for (index_t v = 0; v < simd_length; v++)
				{
					d_rows[v] = hn::Load(d, &(dens_l
											  | noarr::get_at<'z', 'y', 'x', 's'>(densities, z, Y * simd_length + v,
																				  full_n - simd_length, s)));
				}

				// transposition to enable vectorization
				transpose(d_rows);

				index_t remainder_work = n % simd_length;
				remainder_work += remainder_work == 0 ? simd_length : 0;

				// the rest of forward part
				{
					for (index_t v = 0; v < remainder_work; v++)
					{
						simd_t a_curr = hn::Load(
							d, &(diag_l
								 | noarr::get_at<'z', 'Y', 'y', 'x', 's'>(a, z, Y, 0, full_n - simd_length + v, s)));
						simd_t b_curr = hn::Load(
							d, &(diag_l
								 | noarr::get_at<'z', 'Y', 'y', 'x', 's'>(b, z, Y, 0, full_n - simd_length + v, s)));

						auto r = hn::Mul(a_curr, scratch_prev);

						scratch_prev = hn::Div(hn::Set(d, 1), hn::NegMulAdd(c_prev, r, b_curr));
						hn::Store(scratch_prev, d,
								  &(scratch_l | noarr::get_at<'x', 'v'>(b_scratch, full_n - simd_length + v, 0)));

						d_rows[v] = hn::NegMulAdd(d_prev, r, d_rows[v]);

						d_prev = d_rows[v];
						c_prev = hn::Load(
							d, &(diag_l
								 | noarr::get_at<'z', 'Y', 'y', 'x', 's'>(c, z, Y, 0, full_n - simd_length + v, s)));
					}
				}

				{
					d_prev = hn::Mul(d_prev, scratch_prev);
					d_rows[remainder_work - 1] = d_prev;
				}

				// the begin of backward part
				{
					for (index_t v = remainder_work - 2; v >= 0; v--)
					{
						simd_t c_curr = hn::Load(
							d, &(diag_l
								 | noarr::get_at<'z', 'Y', 'y', 'x', 's'>(c, z, Y, 0, full_n - simd_length + v, s)));

						auto scratch =
							hn::Load(d, &(scratch_l | noarr::get_at<'x', 'v'>(b_scratch, full_n - simd_length + v, 0)));
						d_rows[v] = hn::Mul(hn::NegMulAdd(d_prev, c_curr, d_rows[v]), scratch);

						d_prev = d_rows[v];
					}
				}

				// transposition back to the original form
				transpose(d_rows);

				// aligned stores
				for (index_t v = 0; v < simd_length; v++)
				{
					hn::Store(d_rows[v], d,
							  &(dens_l
								| noarr::get_at<'z', 'y', 'x', 's'>(densities, z, Y * simd_length + v,
																	full_n - simd_length, s)));
				}
			}

			// we continue with backwards substitution
			for (index_t i = full_n - simd_length * 2; i >= 0; i -= simd_length)
			{
				// aligned loads
				for (index_t v = 0; v < simd_length; v++)
				{
					d_rows[v] = hn::Load(
						d, &(dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, z, Y * simd_length + v, i, s)));
				}

				// backward propagation
				{
					for (index_t v = simd_length - 1; v >= 0; v--)
					{
						simd_t c_curr =
							hn::Load(d, &(diag_l | noarr::get_at<'z', 'Y', 'y', 'x', 's'>(c, z, Y, 0, i + v, s)));

						auto scratch = hn::Load(d, &(scratch_l | noarr::get_at<'x', 'v'>(b_scratch, i + v, 0)));
						d_rows[v] = hn::Mul(hn::NegMulAdd(d_prev, c_curr, d_rows[v]), scratch);

						d_prev = d_rows[v];
					}
				}

				// transposition back to the original form
				transpose(d_rows);

				// aligned stores
				for (index_t v = 0; v < simd_length; v++)
				{
					hn::Store(d_rows[v], d,
							  &(dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, z, Y * simd_length + v, i, s)));
				}
			}
		}
	}

	// yz remainder
	{
		auto remainder_diag_l = diag_l ^ noarr::fix<'Y'>(Y_len);
		auto a_bag = noarr::make_bag(remainder_diag_l, a);
		auto b_bag = noarr::make_bag(remainder_diag_l, b);
		auto c_bag = noarr::make_bag(remainder_diag_l, c);

		const index_t y_remainder = y_len - Y_len * simd_length;

		auto d = noarr::make_bag(dens_l ^ noarr::slice<'y'>(Y_len * simd_length, y_remainder), densities);

		auto scratch = noarr::make_bag(scratch_l ^ noarr::fix<'v'>(0), b_scratch);

		for (index_t y = 0; y < y_remainder; y++)
		{
			{
				auto idx = noarr::idx<'s', 'z', 'y', 'x'>(s, z, y, 0);
				scratch[idx] = 1 / b_bag[idx];
			}

			for (index_t i = 1; i < n; i++)
			{
				auto idx = noarr::idx<'s', 'z', 'y', 'x'>(s, z, y, i);
				auto prev_idx = noarr::idx<'s', 'z', 'y', 'x'>(s, z, y, i - 1);

				auto r = a_bag[idx] * scratch[prev_idx];

				scratch[idx] = 1 / (b_bag[idx] - c_bag[prev_idx] * r);

				d[idx] -= r * d[prev_idx];

				// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
			}

			{
				auto idx = noarr::idx<'s', 'z', 'y', 'x'>(s, z, y, n - 1);
				d[idx] *= scratch[idx];

				// std::cout << "n-1: " << (dens_l | noarr::get_at<'x', 's'>(densities, n - 1, s)) << std::endl;
			}

			for (index_t i = n - 2; i >= 0; i--)
			{
				auto idx = noarr::idx<'s', 'z', 'y', 'x'>(s, z, y, i);
				auto next_idx = noarr::idx<'s', 'z', 'y', 'x'>(s, z, y, i + 1);

				d[idx] = (d[idx] - c_bag[idx] * d[next_idx]) * scratch[idx];

				// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
			}
		}
	}
}
