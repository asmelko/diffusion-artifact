#include "cyclic_reduction_solver_t.h"

#include <cstddef>
#include <cstdlib>
#include <iostream>

#include "noarr/structures/base/structs_common.hpp"
#include "noarr/structures/extra/shortcuts.hpp"
#include "noarr/structures/extra/traverser.hpp"
#include "omp_helper.h"

template <typename real_t, bool aligned_x>
void cyclic_reduction_solver_t<real_t, aligned_x>::precompute_values(real_t*& a, real_t*& b1, index_t shape,
																	 index_t dims)
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
}

template <typename real_t, bool aligned_x>
void cyclic_reduction_solver_t<real_t, aligned_x>::prepare(const max_problem_t& problem)
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
void cyclic_reduction_solver_t<real_t, aligned_x>::tune(const nlohmann::json& params)
{
	x_tile_size_ = params.contains("x_tile_size") ? (std::size_t)params["x_tile_size"] : 48;
	alignment_size_ = params.contains("alignment_size") ? (std::size_t)params["alignment_size"] : 64;
}

template <typename real_t, bool aligned_x>
void cyclic_reduction_solver_t<real_t, aligned_x>::initialize()
{
	if (this->problem_.dims >= 1)
		precompute_values(ax_, b1x_, this->problem_.dx, this->problem_.dims);
	if (this->problem_.dims >= 2)
		precompute_values(ay_, b1y_, this->problem_.dy, this->problem_.dims);
	if (this->problem_.dims >= 3)
		precompute_values(az_, b1z_, this->problem_.dz, this->problem_.dims);

	auto max_n = std::max({ this->problem_.nx, this->problem_.ny, this->problem_.nz });

	for (index_t i = 0; i < get_max_threads(); i++)
	{
		a_scratch_.push_back((real_t*)std::malloc(max_n * sizeof(real_t)));
		b_scratch_.push_back((real_t*)std::malloc(max_n * sizeof(real_t)));
		c_scratch_.push_back((real_t*)std::malloc(max_n * sizeof(real_t)));
	}
}

template <typename real_t, bool aligned_x>
auto cyclic_reduction_solver_t<real_t, aligned_x>::get_diagonal_layout(const problem_t<index_t, real_t>& problem,
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

template <char dim, typename index_t, typename real_t, typename density_layout_t, typename order_t>
inline static void outer_divide(real_t* __restrict__ densities, const real_t ac, const real_t b1,
								real_t* __restrict__ a, real_t* __restrict__ b, real_t* __restrict__ c,
								const density_layout_t dens_l, const order_t order, index_t n)
{
	for (index_t i = 1; i < n; i += 2)
	{
		const auto a_tmp = ac;
		const auto a_low_tmp = a_tmp * (i == 1 ? 0 : 1);

		const auto c_tmp = a_tmp;
		const auto c_high_tmp = a_tmp * (i == n - 2 ? 0 : 1);

		const auto b_low_tmp = b1 + ((i == 1) ? a_tmp : 0);
		const auto b_tmp = b1 + ((i == n - 1) ? a_tmp : 0);
		const auto b_high_tmp = b1 + ((i == n - 2) ? a_tmp : 0);

		const real_t alpha = a_tmp / b_low_tmp;
		const real_t beta = (i == n - 1) ? 0 : (c_tmp / b_high_tmp);

		a[i / 2] = -alpha * a_low_tmp;
		b[i / 2] = b_tmp - (alpha + beta) * c_tmp;
		c[i / 2] = -beta * c_high_tmp;

		noarr::traverser(dens_l).order(order ^ noarr::fix<dim>(i)).for_each([=](auto state) {
			auto prev_state = noarr::neighbor<dim>(state, -1);
			auto next_state = noarr::neighbor<dim>(state, 1);

			(dens_l | noarr::get_at(densities, state)) -=
				alpha * (dens_l | noarr::get_at(densities, prev_state))
				+ beta * ((i == n - 1) ? 0 : (dens_l | noarr::get_at(densities, next_state)));
		});
	}
}

template <char dim, typename index_t, typename real_t, typename density_layout_t, typename order_t>
inline static void outer_join(real_t* __restrict__ densities, const real_t ac, const real_t b1,
							  const density_layout_t dens_l, const order_t order, index_t n)
{
	// the first unknown of each step does not have (i - stride) dependency
	noarr::traverser(dens_l).order(order ^ noarr::fix<dim>(0)).for_each([=](auto state) {
		auto next_state = noarr::neighbor<dim>(state, 1);
		(dens_l | noarr::get_at(densities, state)) =
			((dens_l | noarr::get_at(densities, state)) - ac * (dens_l | noarr::get_at(densities, next_state)))
			/ (b1 + ac);
	});

	for (index_t i = 2; i < n; i += 2)
	{
		if (i + 1 < n)
		{
			noarr::traverser(dens_l).order(order ^ noarr::fix<dim>(i)).for_each([=](auto state) {
				auto prev_state = noarr::neighbor<dim>(state, -1);
				auto next_state = noarr::neighbor<dim>(state, 1);

				(dens_l | noarr::get_at(densities, state)) =
					((dens_l | noarr::get_at(densities, state)) - ac * (dens_l | noarr::get_at(densities, prev_state))
					 - ac * (dens_l | noarr::get_at(densities, next_state)))
					/ b1;
			});
		}
		else
		{
			noarr::traverser(dens_l).order(order ^ noarr::fix<dim>(i)).for_each([=](auto state) {
				auto prev_state = noarr::neighbor<dim>(state, -1);

				(dens_l | noarr::get_at(densities, state)) =
					((dens_l | noarr::get_at(densities, state)) - ac * (dens_l | noarr::get_at(densities, prev_state)))
					/ b1;
			});
		}
	}
}

template <char dim, typename index_t, typename real_t, typename density_layout_t, typename order_t>
inline static void inner_cyclic_reduction(real_t* __restrict__ densities, real_t* __restrict__ a,
										  real_t* __restrict__ b, real_t* __restrict__ c, const density_layout_t dens_l,
										  const order_t order, index_t steps, index_t n)
{
	for (index_t step = 0; step < steps; step++)
	{
		index_t stride = 1 << step;
		for (index_t i = 2 * stride - 1; i < n; i += 2 * stride)
		{
			if (i + stride < n)
			{
				real_t alpha = a[i] / b[i - stride];
				real_t beta = c[i] / b[i + stride];

				a[i] = -alpha * a[i - stride];
				b[i] -= alpha * c[i - stride] + beta * a[i + stride];
				c[i] = -beta * c[i + stride];

				noarr::traverser(dens_l).order(order ^ noarr::fix<dim>(i)).for_each([=](auto state) {
					auto prev_state = noarr::neighbor<dim>(state, -stride);
					auto next_state = noarr::neighbor<dim>(state, stride);

					(dens_l | noarr::get_at(densities, state)) -=
						alpha * (dens_l | noarr::get_at(densities, prev_state))
						+ beta * (dens_l | noarr::get_at(densities, next_state));
				});
			}
			else
			{
				real_t alpha = a[i] / b[i - stride];

				a[i] = -alpha * a[i - stride];
				b[i] -= alpha * c[i - stride];
				c[i] = 0;

				noarr::traverser(dens_l).order(order ^ noarr::fix<dim>(i)).for_each([=](auto state) {
					auto prev_state = noarr::neighbor<dim>(state, -stride);

					(dens_l | noarr::get_at(densities, state)) -=
						alpha * (dens_l | noarr::get_at(densities, prev_state));
				});
			}
		}
	}

	// the first solved unknown
	{
		index_t i = (1 << steps) - 1;
		noarr::traverser(dens_l).order(order ^ noarr::fix<dim>(i)).for_each([=](auto state) {
			(dens_l | noarr::get_at(densities, state)) /= b[i];
		});
	}

	for (index_t step = steps - 1; step >= 0; step--)
	{
		index_t stride = 1 << step;

		index_t i = stride - 1;

		// the first unknown of each step does not have (i - stride) dependency
		noarr::traverser(dens_l).order(order ^ noarr::fix<dim>(i)).for_each([=](auto state) {
			auto next_state = noarr::neighbor<dim>(state, stride);
			(dens_l | noarr::get_at(densities, state)) =
				((dens_l | noarr::get_at(densities, state)) - c[i] * (dens_l | noarr::get_at(densities, next_state)))
				/ b[i];
		});

		i += 2 * stride;

		for (; i < n; i += 2 * stride)
		{
			if (i + stride < n)
			{
				noarr::traverser(dens_l).order(order ^ noarr::fix<dim>(i)).for_each([=](auto state) {
					auto prev_state = noarr::neighbor<dim>(state, -stride);
					auto next_state = noarr::neighbor<dim>(state, stride);

					(dens_l | noarr::get_at(densities, state)) =
						((dens_l | noarr::get_at(densities, state))
						 - a[i] * (dens_l | noarr::get_at(densities, prev_state))
						 - c[i] * (dens_l | noarr::get_at(densities, next_state)))
						/ b[i];
				});
			}
			else
			{
				noarr::traverser(dens_l).order(order ^ noarr::fix<dim>(i)).for_each([=](auto state) {
					auto prev_state = noarr::neighbor<dim>(state, -stride);

					(dens_l | noarr::get_at(densities, state)) =
						((dens_l | noarr::get_at(densities, state))
						 - a[i] * (dens_l | noarr::get_at(densities, prev_state)))
						/ b[i];
				});
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t>
static void solve_slice_x_1d(real_t* __restrict__ densities, const real_t* __restrict__ ac,
							 const real_t* __restrict__ b1, real_t* __restrict__ a, real_t* __restrict__ b,
							 real_t* __restrict__ c, const density_layout_t dens_l)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'x'>();

	const index_t all_steps = (int)std::log2(n);
	const index_t inner_steps = all_steps - 1;
	const index_t inner_n = n / 2;

	// we are halving the number of unknowns into new arrays a, b, c
	// but we are preserving densities array, so its indices need to be adjusted
	const auto inner_dens_step = noarr::step<'x'>(1, 2);

	const auto order = noarr::neutral_proto();

#pragma omp for schedule(static) nowait
	for (index_t s = 0; s < substrates_count; s++)
	{
		const auto fix = noarr::fix<'s'>(s);

		outer_divide<'x'>(densities, ac[s], b1[s], a, b, c, dens_l ^ fix, order, n);

		inner_cyclic_reduction<'x'>(densities, a, b, c, dens_l ^ inner_dens_step ^ fix, order, inner_steps, inner_n);

		outer_join<'x'>(densities, ac[s], b1[s], dens_l ^ fix, order, n);
	}
}

template <typename index_t, typename real_t, typename density_layout_t>
static void solve_slice_x_2d_and_3d(real_t* __restrict__ densities, const real_t* __restrict__ ac,
									const real_t* __restrict__ b1, real_t* __restrict__ a, real_t* __restrict__ b,
									real_t* __restrict__ c, const density_layout_t dens_l)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'x'>();
	const index_t m = dens_l | noarr::get_length<'m'>();

	const index_t all_steps = (int)std::log2(n);
	const index_t inner_steps = all_steps - 1;
	const index_t inner_n = n / 2;

	// we are halving the number of unknowns into new arrays a, b, c
	// but we are preserving densities array, so its indices need to be adjusted
	const auto inner_dens_step = noarr::step<'x'>(1, 2);

	const auto order = noarr::neutral_proto();

#pragma omp for schedule(static) collapse(2) nowait
	for (index_t s = 0; s < substrates_count; s++)
	{
		for (index_t yz = 0; yz < m; yz++)
		{
			const auto fix = noarr::fix<'s', 'm'>(s, yz);

			outer_divide<'x'>(densities, ac[s], b1[s], a, b, c, dens_l ^ fix, order, n);

			inner_cyclic_reduction<'x'>(densities, a, b, c, dens_l ^ inner_dens_step ^ fix, order, inner_steps,
										inner_n);

			outer_join<'x'>(densities, ac[s], b1[s], dens_l ^ fix, order, n);
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t>
static void solve_slice_y_2d(real_t* __restrict__ densities, const real_t* __restrict__ ac,
							 const real_t* __restrict__ b1, real_t* __restrict__ a, real_t* __restrict__ b,
							 real_t* __restrict__ c, const density_layout_t dens_l, std::size_t x_tile_size)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'y'>();

	const index_t all_steps = (int)std::log2(n);
	const index_t inner_steps = all_steps - 1;
	const index_t inner_n = n / 2;

	// we are halving the number of unknowns into new arrays a, b, c
	// but we are preserving densities array, so its indices need to be adjusted
	const auto inner_dens_step = noarr::step<'y'>(1, 2);

	const auto order = noarr::neutral_proto();

	auto blocked_dens_l = dens_l ^ noarr::into_blocks_static<'x', 'b', 'X', 'x'>(x_tile_size);

#pragma omp for schedule(static) nowait
	for (index_t s = 0; s < substrates_count; s++)
	{
		{
			auto body_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<0>);
			const index_t X_len = body_dens_l | noarr::get_length<'X'>();

			for (index_t X = 0; X < X_len; X++)
			{
				const auto fix = noarr::fix<'s', 'X'>(s, X);

				outer_divide<'y'>(densities, ac[s], b1[s], a, b, c, body_dens_l ^ fix, order, n);

				inner_cyclic_reduction<'y'>(densities, a, b, c, body_dens_l ^ inner_dens_step ^ fix, order, inner_steps,
											inner_n);

				outer_join<'y'>(densities, ac[s], b1[s], body_dens_l ^ fix, order, n);
			}
		}

		{
			auto border_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<1>);

			const auto fix = noarr::fix<'s', 'X'>(s, noarr::lit<0>);

			outer_divide<'y'>(densities, ac[s], b1[s], a, b, c, border_dens_l ^ fix, order, n);

			inner_cyclic_reduction<'y'>(densities, a, b, c, border_dens_l ^ inner_dens_step ^ fix, order, inner_steps,
										inner_n);

			outer_join<'y'>(densities, ac[s], b1[s], border_dens_l ^ fix, order, n);
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t>
static void solve_slice_y_3d(real_t* __restrict__ densities, const real_t* __restrict__ ac,
							 const real_t* __restrict__ b1, real_t* __restrict__ a, real_t* __restrict__ b,
							 real_t* __restrict__ c, const density_layout_t dens_l, std::size_t x_tile_size)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'y'>();
	const index_t z_len = dens_l | noarr::get_length<'z'>();

	const index_t all_steps = (int)std::log2(n);
	const index_t inner_steps = all_steps - 1;
	const index_t inner_n = n / 2;

	// we are halving the number of unknowns into new arrays a, b, c
	// but we are preserving densities array, so its indices need to be adjusted
	const auto inner_dens_step = noarr::step<'y'>(1, 2);

	const auto order = noarr::neutral_proto();

	auto blocked_dens_l = dens_l ^ noarr::into_blocks_static<'x', 'b', 'X', 'x'>(x_tile_size);

#pragma omp for schedule(static) nowait collapse(2)
	for (index_t s = 0; s < substrates_count; s++)
	{
		for (index_t z = 0; z < z_len; z++)
		{
			{
				auto body_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<0>);
				const index_t X_len = body_dens_l | noarr::get_length<'X'>();

				for (index_t X = 0; X < X_len; X++)
				{
					const auto fix = noarr::fix<'s', 'z', 'X'>(s, z, X);

					outer_divide<'y'>(densities, ac[s], b1[s], a, b, c, body_dens_l ^ fix, order, n);

					inner_cyclic_reduction<'y'>(densities, a, b, c, body_dens_l ^ inner_dens_step ^ fix, order,
												inner_steps, inner_n);

					outer_join<'y'>(densities, ac[s], b1[s], body_dens_l ^ fix, order, n);
				}
			}

			{
				auto border_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<1>);

				const auto fix = noarr::fix<'s', 'z', 'X'>(s, z, noarr::lit<0>);

				outer_divide<'y'>(densities, ac[s], b1[s], a, b, c, border_dens_l ^ fix, order, n);

				inner_cyclic_reduction<'y'>(densities, a, b, c, border_dens_l ^ inner_dens_step ^ fix, order,
											inner_steps, inner_n);

				outer_join<'y'>(densities, ac[s], b1[s], border_dens_l ^ fix, order, n);
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t>
static void solve_slice_z_3d(real_t* __restrict__ densities, const real_t* __restrict__ ac,
							 const real_t* __restrict__ b1, real_t* __restrict__ a, real_t* __restrict__ b,
							 real_t* __restrict__ c, const density_layout_t dens_l, index_t x_tile_size)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'z'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();

	const index_t all_steps = (int)std::log2(n);
	const index_t inner_steps = all_steps - 1;
	const index_t inner_n = n / 2;

	// we are halving the number of unknowns into new arrays a, b, c
	// but we are preserving densities array, so its indices need to be adjusted
	const auto inner_dens_step = noarr::step<'z'>(1, 2);

	const auto order = noarr::neutral_proto();

	auto blocked_dens_l = dens_l ^ noarr::into_blocks_static<'x', 'b', 'X', 'x'>(x_tile_size);

#pragma omp for schedule(static) nowait collapse(2)
	for (index_t s = 0; s < substrates_count; s++)
	{
		for (index_t y = 0; y < y_len; y++)
		{
			{
				auto body_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<0>);
				const index_t X_len = body_dens_l | noarr::get_length<'X'>();

				for (index_t X = 0; X < X_len; X++)
				{
					const auto fix = noarr::fix<'s', 'y', 'X'>(s, y, X);

					outer_divide<'z'>(densities, ac[s], b1[s], a, b, c, body_dens_l ^ fix, order, n);

					inner_cyclic_reduction<'z'>(densities, a, b, c, body_dens_l ^ inner_dens_step ^ fix, order,
												inner_steps, inner_n);

					outer_join<'z'>(densities, ac[s], b1[s], body_dens_l ^ fix, order, n);
				}
			}

			{
				auto border_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<1>);

				const auto fix = noarr::fix<'s', 'y', 'X'>(s, y, noarr::lit<0>);

				outer_divide<'z'>(densities, ac[s], b1[s], a, b, c, border_dens_l ^ fix, order, n);

				inner_cyclic_reduction<'z'>(densities, a, b, c, border_dens_l ^ inner_dens_step ^ fix, order,
											inner_steps, inner_n);

				outer_join<'z'>(densities, ac[s], b1[s], border_dens_l ^ fix, order, n);
			}
		}
	}
}

template <typename real_t, bool aligned_x>
void cyclic_reduction_solver_t<real_t, aligned_x>::solve_x()
{
	if (this->problem_.dims == 1)
	{
#pragma omp parallel
		solve_slice_x_1d<index_t>(this->substrates_, ax_, b1x_, a_scratch_[get_thread_num()],
								  b_scratch_[get_thread_num()], c_scratch_[get_thread_num()],
								  get_substrates_layout<1>());
	}
	else if (this->problem_.dims == 2)
	{
#pragma omp parallel
		solve_slice_x_2d_and_3d<index_t>(this->substrates_, ax_, b1x_, a_scratch_[get_thread_num()],
										 b_scratch_[get_thread_num()], c_scratch_[get_thread_num()],
										 get_substrates_layout<2>() ^ noarr::rename<'y', 'm'>());
	}
	else if (this->problem_.dims == 3)
	{
#pragma omp parallel
		solve_slice_x_2d_and_3d<index_t>(this->substrates_, ax_, b1x_, a_scratch_[get_thread_num()],
										 b_scratch_[get_thread_num()], c_scratch_[get_thread_num()],
										 get_substrates_layout<3>() ^ noarr::merge_blocks<'z', 'y', 'm'>());
	}
}

template <typename real_t, bool aligned_x>
void cyclic_reduction_solver_t<real_t, aligned_x>::solve_y()
{
	if (this->problem_.dims == 2)
	{
#pragma omp parallel
		solve_slice_y_2d<index_t>(this->substrates_, ay_, b1y_, a_scratch_[get_thread_num()],
								  b_scratch_[get_thread_num()], c_scratch_[get_thread_num()],
								  get_substrates_layout<2>(), x_tile_size_);
	}
	else if (this->problem_.dims == 3)
	{
#pragma omp parallel
		solve_slice_y_3d<index_t>(this->substrates_, ay_, b1y_, a_scratch_[get_thread_num()],
								  b_scratch_[get_thread_num()], c_scratch_[get_thread_num()],
								  get_substrates_layout<3>(), x_tile_size_);
	}
}

template <typename real_t, bool aligned_x>
void cyclic_reduction_solver_t<real_t, aligned_x>::solve_z()
{
#pragma omp parallel
	solve_slice_z_3d<index_t>(this->substrates_, az_, b1z_, a_scratch_[get_thread_num()], b_scratch_[get_thread_num()],
							  c_scratch_[get_thread_num()], get_substrates_layout<3>(), x_tile_size_);
}

template <typename real_t, bool aligned_x>
void cyclic_reduction_solver_t<real_t, aligned_x>::solve()
{
	if (this->problem_.dims == 1)
	{
#pragma omp parallel
		for (index_t i = 0; i < this->problem_.iterations; i++)
			solve_slice_x_1d<index_t>(this->substrates_, ax_, b1x_, a_scratch_[get_thread_num()],
									  b_scratch_[get_thread_num()], c_scratch_[get_thread_num()],
									  get_substrates_layout<1>());
	}
	if (this->problem_.dims == 2)
	{
#pragma omp parallel
		for (index_t i = 0; i < this->problem_.iterations; i++)
		{
			solve_slice_x_2d_and_3d<index_t>(this->substrates_, ax_, b1x_, a_scratch_[get_thread_num()],
											 b_scratch_[get_thread_num()], c_scratch_[get_thread_num()],
											 get_substrates_layout<2>() ^ noarr::rename<'y', 'm'>());
#pragma omp barrier
			solve_slice_y_2d<index_t>(this->substrates_, ax_, b1x_, a_scratch_[get_thread_num()],
									  b_scratch_[get_thread_num()], c_scratch_[get_thread_num()],
									  get_substrates_layout<2>(), x_tile_size_);
#pragma omp barrier
		}
	}
	if (this->problem_.dims == 3)
	{
#pragma omp parallel
		for (index_t i = 0; i < this->problem_.iterations; i++)
		{
			solve_slice_x_2d_and_3d<index_t>(this->substrates_, ax_, b1x_, a_scratch_[get_thread_num()],
											 b_scratch_[get_thread_num()], c_scratch_[get_thread_num()],
											 get_substrates_layout<3>() ^ noarr::merge_blocks<'z', 'y', 'm'>());
#pragma omp barrier
			solve_slice_y_3d<index_t>(this->substrates_, ax_, b1x_, a_scratch_[get_thread_num()],
									  b_scratch_[get_thread_num()], c_scratch_[get_thread_num()],
									  get_substrates_layout<3>(), x_tile_size_);
#pragma omp barrier
			solve_slice_z_3d<index_t>(this->substrates_, ax_, b1x_, a_scratch_[get_thread_num()],
									  b_scratch_[get_thread_num()], c_scratch_[get_thread_num()],
									  get_substrates_layout<3>(), x_tile_size_);
#pragma omp barrier
		}
	}
}

template <typename real_t, bool aligned_x>
cyclic_reduction_solver_t<real_t, aligned_x>::cyclic_reduction_solver_t()
	: ax_(nullptr), b1x_(nullptr), ay_(nullptr), b1y_(nullptr), az_(nullptr), b1z_(nullptr)
{}

template <typename real_t, bool aligned_x>
cyclic_reduction_solver_t<real_t, aligned_x>::~cyclic_reduction_solver_t()
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

	for (std::size_t i = 0; i < b_scratch_.size(); i++)
	{
		std::free(a_scratch_[i]);
		std::free(b_scratch_[i]);
		std::free(c_scratch_[i]);
	}
}

template class cyclic_reduction_solver_t<float, false>;
template class cyclic_reduction_solver_t<double, false>;

template class cyclic_reduction_solver_t<float, true>;
template class cyclic_reduction_solver_t<double, true>;
