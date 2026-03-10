#include "least_memory_thomas_solver_t.h"

#include <cstddef>
#include <iostream>

#include "../perf_utils.h"
#include "../vector_transpose_helper.h"

template <typename real_t, bool aligned_x>
void sdd_least_memory_thomas_solver_t<real_t, aligned_x>::precompute_values(real_t*& a, real_t*& b, real_t*& c,
																			index_t shape, index_t n, index_t dims,
																			char dim, auto substrates_layout)
{
	if (aligned_x)
	{
		a = (real_t*)std::aligned_alloc(alignment_size_, (substrates_layout | noarr::get_size()));
		b = (real_t*)std::aligned_alloc(alignment_size_, (substrates_layout | noarr::get_size()));
		c = (real_t*)std::aligned_alloc(alignment_size_, (substrates_layout | noarr::get_size()));
	}
	else
	{
		a = (real_t*)std::malloc((substrates_layout | noarr::get_size()));
		b = (real_t*)std::malloc((substrates_layout | noarr::get_size()));
		c = (real_t*)std::malloc((substrates_layout | noarr::get_size()));
	}

	auto a_bag = noarr::make_bag(substrates_layout, a);
	auto b_bag = noarr::make_bag(substrates_layout, b);
	auto c_bag = noarr::make_bag(substrates_layout, c);

	auto get_diffusion_coefficients = [&](index_t, index_t, index_t, index_t s) {
		return this->problem_.diffusion_coefficients[s];
	};

	for (index_t s = 0; s < this->problem_.substrates_count; s++)
		for (index_t x = 0; x < this->problem_.nx; x++)
			for (index_t y = 0; y < this->problem_.ny; y++)
				for (index_t z = 0; z < this->problem_.nz; z++)
				{
					auto idx = noarr::idx<'x', 'y', 'z', 's'>(x, y, z, s);

					auto dim_idx = dim == 'x' ? x : (dim == 'y' ? y : z);

					if (dim_idx == 0)
					{
						a_bag[idx] = 0;
						b_bag[idx] = 1 + this->problem_.dt * this->problem_.decay_rates[s] / dims
									 + 1 * this->problem_.dt * get_diffusion_coefficients(x, y, z, s) / (shape * shape);
						c_bag[idx] = -this->problem_.dt * get_diffusion_coefficients(x, y, z, s) / (shape * shape);
					}
					else if (dim_idx == n - 1)
					{
						a_bag[idx] = -this->problem_.dt * get_diffusion_coefficients(x, y, z, s) / (shape * shape);
						b_bag[idx] = 1 + this->problem_.dt * this->problem_.decay_rates[s] / dims
									 + 1 * this->problem_.dt * get_diffusion_coefficients(x, y, z, s) / (shape * shape);
						c_bag[idx] = 0;
					}
					else
					{
						a_bag[idx] = -this->problem_.dt * get_diffusion_coefficients(x, y, z, s) / (shape * shape);
						b_bag[idx] = 1 + this->problem_.dt * this->problem_.decay_rates[s] / dims
									 + 2 * this->problem_.dt * get_diffusion_coefficients(x, y, z, s) / (shape * shape);
						c_bag[idx] = -this->problem_.dt * get_diffusion_coefficients(x, y, z, s) / (shape * shape);
					}
				}
}

template <typename real_t, bool aligned_x>
void sdd_least_memory_thomas_solver_t<real_t, aligned_x>::prepare(const max_problem_t& problem)
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
void sdd_least_memory_thomas_solver_t<real_t, aligned_x>::tune(const nlohmann::json& params)
{
	x_tile_size_ = params.contains("x_tile_size") ? (std::size_t)params["x_tile_size"] : 48;
	continuous_x_diagonal_ = params.contains("continuous_x_diagonal") ? (bool)params["continuous_x_diagonal"] : false;

	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag d;
	std::size_t vector_length = hn::Lanes(d) * sizeof(real_t);
	alignment_size_ = vector_length;
}

template <typename real_t, bool aligned_x>
void sdd_least_memory_thomas_solver_t<real_t, aligned_x>::initialize()
{
	if (continuous_x_diagonal_)
		precompute_values(ax_, bx_, cx_, this->problem_.dx, this->problem_.nx, this->problem_.dims, 'x',
						  get_diag_layout_x() ^ noarr::merge_blocks<'Y', 'y', 'y'>()
							  ^ noarr::into_blocks_static<'y', 'b', 'z', 'y'>(this->problem_.ny)
							  ^ noarr::fix<'b'>(noarr::lit<0>));
	else
		precompute_values(ax_, bx_, cx_, this->problem_.dx, this->problem_.nx, this->problem_.dims, 'x',
						  get_substrates_layout());

	precompute_values(ay_, by_, cy_, this->problem_.dy, this->problem_.ny, this->problem_.dims, 'y',
					  get_substrates_layout());
	precompute_values(az_, bz_, cz_, this->problem_.dz, this->problem_.nz, this->problem_.dims, 'z',
					  get_substrates_layout());

	auto diag_lx = get_scratch_layout<'x'>();
	auto diag_ly = get_scratch_layout<'y'>();
	auto diag_lz = get_scratch_layout<'z'>();
	auto max_size =
		std::max({ (diag_lx | noarr::get_size()), (diag_ly | noarr::get_size()), (diag_lz | noarr::get_size()) });

	for (int i = 0; i < get_max_threads(); i++)
	{
		if (aligned_x)
			b_scratch_.push_back((real_t*)std::aligned_alloc(alignment_size_, (max_size)));
		else
			b_scratch_.push_back((real_t*)std::malloc((max_size)));
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename scratch_layout_t>
static void solve_slice_x_2d_and_3d_transpose_l(real_t* __restrict__ densities, const real_t* __restrict__ a,
												const real_t* __restrict__ b, const real_t* __restrict__ c,
												real_t* __restrict__ b_scratch, const density_layout_t dens_l,
												const diagonal_layout_t diag_l, const scratch_layout_t scratch_l,
												const index_t s, index_t n)
{
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag d;
	constexpr index_t simd_length = hn::Lanes(d);
	using simd_t = hn::Vec<simd_tag>;

	simd_t d_rows[simd_length];

	const index_t m = dens_l | noarr::get_length<'m'>();

	const index_t Y_len = m / simd_length;

	// vectorized body
	{
		const index_t full_n = (n + simd_length - 1) / simd_length * simd_length;

#pragma omp for schedule(static) nowait
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
					d_rows[v] =
						hn::Load(d, &(dens_l | noarr::get_at<'m', 'x', 's'>(densities, Y * simd_length + v, i, s)));
				}

				// transposition to enable vectorization
				transpose(d_rows);

				for (index_t v = 0; v < simd_length; v++)
				{
					simd_t a_curr = hn::Load(d, &(diag_l | noarr::get_at<'Y', 'y', 'x', 's'>(a, Y, 0, i + v, s)));
					simd_t b_curr = hn::Load(d, &(diag_l | noarr::get_at<'Y', 'y', 'x', 's'>(b, Y, 0, i + v, s)));

					auto r = hn::Mul(a_curr, scratch_prev);

					scratch_prev = hn::Div(hn::Set(d, 1), hn::NegMulAdd(c_prev, r, b_curr));
					hn::Store(scratch_prev, d, &(scratch_l | noarr::get_at<'x', 'v'>(b_scratch, i + v, 0)));

					d_rows[v] = hn::NegMulAdd(d_prev, r, d_rows[v]);

					d_prev = d_rows[v];
					c_prev = hn::Load(d, &(diag_l | noarr::get_at<'Y', 'y', 'x', 's'>(c, Y, 0, i + v, s)));
				}

				// aligned stores
				for (index_t v = 0; v < simd_length; v++)
				{
					hn::Store(d_rows[v], d,
							  &(dens_l | noarr::get_at<'m', 'x', 's'>(densities, Y * simd_length + v, i, s)));
				}
			}

			// we are aligned to the vector size, so we can safely continue
			// here we fuse the end of forward substitution and the beginning of backwards propagation
			{
				for (index_t v = 0; v < simd_length; v++)
				{
					d_rows[v] = hn::Load(
						d, &(dens_l
							 | noarr::get_at<'m', 'x', 's'>(densities, Y * simd_length + v, full_n - simd_length, s)));
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
							d, &(diag_l | noarr::get_at<'Y', 'y', 'x', 's'>(a, Y, 0, full_n - simd_length + v, s)));
						simd_t b_curr = hn::Load(
							d, &(diag_l | noarr::get_at<'Y', 'y', 'x', 's'>(b, Y, 0, full_n - simd_length + v, s)));

						auto r = hn::Mul(a_curr, scratch_prev);

						scratch_prev = hn::Div(hn::Set(d, 1), hn::NegMulAdd(c_prev, r, b_curr));
						hn::Store(scratch_prev, d,
								  &(scratch_l | noarr::get_at<'x', 'v'>(b_scratch, full_n - simd_length + v, 0)));

						d_rows[v] = hn::NegMulAdd(d_prev, r, d_rows[v]);

						d_prev = d_rows[v];
						c_prev = hn::Load(
							d, &(diag_l | noarr::get_at<'Y', 'y', 'x', 's'>(c, Y, 0, full_n - simd_length + v, s)));
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
							d, &(diag_l | noarr::get_at<'Y', 'y', 'x', 's'>(c, Y, 0, full_n - simd_length + v, s)));

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
					hn::Store(
						d_rows[v], d,
						&(dens_l
						  | noarr::get_at<'m', 'x', 's'>(densities, Y * simd_length + v, full_n - simd_length, s)));
				}
			}

			// we continue with backwards substitution
			for (index_t i = full_n - simd_length * 2; i >= 0; i -= simd_length)
			{
				// aligned loads
				for (index_t v = 0; v < simd_length; v++)
				{
					d_rows[v] =
						hn::Load(d, &(dens_l | noarr::get_at<'m', 'x', 's'>(densities, Y * simd_length + v, i, s)));
				}

				// backward propagation
				{
					for (index_t v = simd_length - 1; v >= 0; v--)
					{
						simd_t c_curr = hn::Load(d, &(diag_l | noarr::get_at<'Y', 'y', 'x', 's'>(c, Y, 0, i + v, s)));

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
							  &(dens_l | noarr::get_at<'m', 'x', 's'>(densities, Y * simd_length + v, i, s)));
				}
			}
		}
	}

	// yz remainder
	{
		auto remainder_diag_l = diag_l ^ noarr::fix<'Y'>(Y_len) ^ noarr::rename<'y', 'm'>();
		auto a_bag = noarr::make_bag(remainder_diag_l, a);
		auto b_bag = noarr::make_bag(remainder_diag_l, b);
		auto c_bag = noarr::make_bag(remainder_diag_l, c);

		const index_t m_remainder = m - Y_len * simd_length;

		auto d = noarr::make_bag(dens_l ^ noarr::slice<'m'>(Y_len * simd_length, m_remainder), densities);

		auto scratch = noarr::make_bag(scratch_l ^ noarr::fix<'v'>(0), b_scratch);

#pragma omp for schedule(static) nowait
		for (index_t yz = 0; yz < m_remainder; yz++)
		{
			{
				auto idx = noarr::idx<'s', 'm', 'x'>(s, yz, 0);
				scratch[idx] = 1 / b_bag[idx];
			}

			for (index_t i = 1; i < n; i++)
			{
				auto idx = noarr::idx<'s', 'm', 'x'>(s, yz, i);
				auto prev_idx = noarr::idx<'s', 'm', 'x'>(s, yz, i - 1);

				auto r = a_bag[idx] * scratch[prev_idx];

				scratch[idx] = 1 / (b_bag[idx] - c_bag[prev_idx] * r);

				d[idx] -= r * d[prev_idx];

				// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
			}

			{
				auto idx = noarr::idx<'s', 'm', 'x'>(s, yz, n - 1);
				d[idx] *= scratch[idx];

				// std::cout << "n-1: " << (dens_l | noarr::get_at<'x', 's'>(densities, n - 1, s)) << std::endl;
			}

			for (index_t i = n - 2; i >= 0; i--)
			{
				auto idx = noarr::idx<'s', 'm', 'x'>(s, yz, i);
				auto next_idx = noarr::idx<'s', 'm', 'x'>(s, yz, i + 1);

				d[idx] = (d[idx] - c_bag[idx] * d[next_idx]) * scratch[idx];

				// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_x_2d_and_3d_transpose(real_t* __restrict__ densities, const real_t* __restrict__ a,
											  const real_t* __restrict__ b, const real_t* __restrict__ c,
											  real_t* __restrict__ b_scratch, const density_layout_t dens_l,
											  const diagonal_layout_t diag_l, const index_t s, index_t n)
{
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag d;
	constexpr index_t simd_length = hn::Lanes(d);
	using simd_t = hn::Vec<simd_tag>;

	simd_t a_rows[simd_length];
	simd_t b_rows[simd_length];
	simd_t c_rows[simd_length];
	simd_t d_rows[simd_length];

	auto blocked_dens_l = dens_l ^ noarr::into_blocks_static<'m', 'b', 'm', 'v'>(simd_length);

	// vectorized body
	{
		const index_t full_n = (n + simd_length - 1) / simd_length * simd_length;

		auto body_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<0>);
		const index_t m = body_dens_l | noarr::get_length<'m'>();

#pragma omp for schedule(static) nowait
		for (index_t yz = 0; yz < m; yz++)
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
					a_rows[v] = hn::Load(d, &(body_dens_l | noarr::get_at<'m', 'v', 'x', 's'>(a, yz, v, i, s)));
					b_rows[v] = hn::Load(d, &(body_dens_l | noarr::get_at<'m', 'v', 'x', 's'>(b, yz, v, i, s)));
					c_rows[v] = hn::Load(d, &(body_dens_l | noarr::get_at<'m', 'v', 'x', 's'>(c, yz, v, i, s)));
					d_rows[v] = hn::Load(d, &(body_dens_l | noarr::get_at<'m', 'v', 'x', 's'>(densities, yz, v, i, s)));
				}

				// transposition to enable vectorization
				transpose(a_rows);
				transpose(b_rows);
				transpose(c_rows);
				transpose(d_rows);

				for (index_t v = 0; v < simd_length; v++)
				{
					auto r = hn::Mul(a_rows[v], scratch_prev);

					scratch_prev = hn::Div(hn::Set(d, 1), hn::NegMulAdd(c_prev, r, b_rows[v]));
					hn::Store(scratch_prev, d, &(diag_l | noarr::get_at<'x', 'v'>(b_scratch, i + v, 0)));

					d_rows[v] = hn::NegMulAdd(d_prev, r, d_rows[v]);

					d_prev = d_rows[v];
					c_prev = c_rows[v];
				}

				// aligned stores
				for (index_t v = 0; v < simd_length; v++)
				{
					hn::Store(d_rows[v], d, &(body_dens_l | noarr::get_at<'m', 'v', 'x', 's'>(densities, yz, v, i, s)));
				}
			}

			// we are aligned to the vector size, so we can safely continue
			// here we fuse the end of forward substitution and the beginning of backwards propagation
			{
				for (index_t v = 0; v < simd_length; v++)
				{
					a_rows[v] = hn::Load(
						d, &(body_dens_l | noarr::get_at<'m', 'v', 'x', 's'>(a, yz, v, full_n - simd_length, s)));
					b_rows[v] = hn::Load(
						d, &(body_dens_l | noarr::get_at<'m', 'v', 'x', 's'>(b, yz, v, full_n - simd_length, s)));
					c_rows[v] = hn::Load(
						d, &(body_dens_l | noarr::get_at<'m', 'v', 'x', 's'>(c, yz, v, full_n - simd_length, s)));
					d_rows[v] =
						hn::Load(d, &(body_dens_l
									  | noarr::get_at<'m', 'v', 'x', 's'>(densities, yz, v, full_n - simd_length, s)));
				}


				// transposition to enable vectorization
				transpose(a_rows);
				transpose(b_rows);
				transpose(c_rows);
				transpose(d_rows);

				index_t remainder_work = n % simd_length;
				remainder_work += remainder_work == 0 ? simd_length : 0;

				// the rest of forward part
				{
					for (index_t v = 0; v < remainder_work; v++)
					{
						auto r = hn::Mul(a_rows[v], scratch_prev);

						scratch_prev = hn::Div(hn::Set(d, 1), hn::NegMulAdd(c_prev, r, b_rows[v]));
						hn::Store(scratch_prev, d,
								  &(diag_l | noarr::get_at<'x', 'v'>(b_scratch, full_n - simd_length + v, 0)));

						d_rows[v] = hn::NegMulAdd(d_prev, r, d_rows[v]);

						d_prev = d_rows[v];
						c_prev = c_rows[v];
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
						auto scratch =
							hn::Load(d, &(diag_l | noarr::get_at<'x', 'v'>(b_scratch, full_n - simd_length + v, 0)));
						d_rows[v] = hn::Mul(hn::NegMulAdd(d_prev, c_rows[v], d_rows[v]), scratch);

						d_prev = d_rows[v];
					}
				}

				// transposition back to the original form
				transpose(d_rows);

				// aligned stores
				for (index_t v = 0; v < simd_length; v++)
				{
					hn::Store(
						d_rows[v], d,
						&(body_dens_l | noarr::get_at<'m', 'v', 'x', 's'>(densities, yz, v, full_n - simd_length, s)));
				}
			}

			// we continue with backwards substitution
			for (index_t i = full_n - simd_length * 2; i >= 0; i -= simd_length)
			{
				// aligned loads
				for (index_t v = 0; v < simd_length; v++)
				{
					a_rows[v] = hn::Load(d, &(body_dens_l | noarr::get_at<'m', 'v', 'x', 's'>(a, yz, v, i, s)));
					b_rows[v] = hn::Load(d, &(body_dens_l | noarr::get_at<'m', 'v', 'x', 's'>(b, yz, v, i, s)));
					c_rows[v] = hn::Load(d, &(body_dens_l | noarr::get_at<'m', 'v', 'x', 's'>(c, yz, v, i, s)));
					d_rows[v] = hn::Load(d, &(body_dens_l | noarr::get_at<'m', 'v', 'x', 's'>(densities, yz, v, i, s)));
				}

				// transposition back to the original form
				transpose(a_rows);
				transpose(b_rows);
				transpose(c_rows);

				// backward propagation
				{
					for (index_t v = simd_length - 1; v >= 0; v--)
					{
						auto scratch = hn::Load(d, &(diag_l | noarr::get_at<'x', 'v'>(b_scratch, i + v, 0)));
						d_rows[v] = hn::Mul(hn::NegMulAdd(d_prev, c_rows[v], d_rows[v]), scratch);

						d_prev = d_rows[v];
					}
				}

				// transposition back to the original form
				transpose(d_rows);

				// aligned stores
				for (index_t v = 0; v < simd_length; v++)
				{
					hn::Store(d_rows[v], d, &(body_dens_l | noarr::get_at<'m', 'v', 'x', 's'>(densities, yz, v, i, s)));
				}
			}
		}
	}

	// yz remainder
	{
		auto rem_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<1>) ^ noarr::fix<'m'>(noarr::lit<0>);
		const index_t v_len = rem_dens_l | noarr::get_length<'v'>();

		auto a_bag = noarr::make_bag(rem_dens_l, a);
		auto b_bag = noarr::make_bag(rem_dens_l, b);
		auto c_bag = noarr::make_bag(rem_dens_l, c);

		auto d = noarr::make_bag(rem_dens_l, densities);

		auto scratch = noarr::make_bag(diag_l, b_scratch);

#pragma omp for schedule(static) nowait
		for (index_t yz = 0; yz < v_len; yz++)
		{
			{
				auto idx = noarr::idx<'s', 'v', 'x'>(s, yz, 0);
				scratch[idx] = 1 / b_bag[idx];
			}

			for (index_t i = 1; i < n; i++)
			{
				auto idx = noarr::idx<'s', 'v', 'x'>(s, yz, i);
				auto prev_idx = noarr::idx<'s', 'v', 'x'>(s, yz, i - 1);

				auto r = a_bag[idx] * scratch[prev_idx];

				scratch[idx] = 1 / (b_bag[idx] - c_bag[prev_idx] * r);

				d[idx] -= r * d[prev_idx];

				// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
			}

			{
				auto idx = noarr::idx<'s', 'v', 'x'>(s, yz, n - 1);
				d[idx] *= scratch[idx];

				// std::cout << "n-1: " << (dens_l | noarr::get_at<'x', 's'>(densities, n - 1, s)) << std::endl;
			}

			for (index_t i = n - 2; i >= 0; i--)
			{
				auto idx = noarr::idx<'s', 'v', 'x'>(s, yz, i);
				auto next_idx = noarr::idx<'s', 'v', 'x'>(s, yz, i + 1);

				d[idx] = (d[idx] - c_bag[idx] * d[next_idx]) * scratch[idx];

				// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_y_2d(real_t* __restrict__ densities, const real_t* __restrict__ a, const real_t* __restrict__ b,
							 const real_t* __restrict__ c, real_t* __restrict__ b_scratch,
							 const density_layout_t dens_l, const diagonal_layout_t diag_l, index_t s_idx,
							 index_t x_tile_size)
{
	const index_t n = dens_l | noarr::get_length<'y'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();

	auto blocked_dens_l = dens_l ^ noarr::fix<'s'>(s_idx) ^ noarr::into_blocks_dynamic<'x', 'x', 'v', 'b'>(x_tile_size)
						  ^ noarr::fix<'b'>(noarr::lit<0>);

	const index_t x_block_len = blocked_dens_l | noarr::get_length<'x'>();

	auto a_bag = noarr::make_bag(blocked_dens_l, a);
	auto b_bag = noarr::make_bag(blocked_dens_l, b);
	auto c_bag = noarr::make_bag(blocked_dens_l, c);

	auto d = noarr::make_bag(blocked_dens_l, densities);

	auto scratch = noarr::make_bag(diag_l, b_scratch);

#pragma omp for schedule(static) nowait
	for (index_t x = 0; x < x_block_len; x++)
	{
		const auto remainder = x_len % x_tile_size;
		const auto x_len_remainder = remainder == 0 ? x_tile_size : remainder;
		const auto tile_size = x == x_block_len - 1 ? x_len_remainder : x_tile_size;

		for (index_t s = 0; s < tile_size; s++)
		{
			auto idx = noarr::idx<'v', 'y', 'x'>(s, 0, x);
			scratch[idx] = 1 / b_bag[idx];
		}

		for (index_t i = 1; i < n; i++)
		{
			for (index_t s = 0; s < tile_size; s++)
			{
				auto idx = noarr::idx<'v', 'y', 'x'>(s, i, x);
				auto prev_idx = noarr::idx<'v', 'y', 'x'>(s, i - 1, x);

				auto r = a_bag[idx] * scratch[prev_idx];

				scratch[idx] = 1 / (b_bag[idx] - c_bag[prev_idx] * r);

				d[idx] -= r * d[prev_idx];

				// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
			}
		}

		for (index_t s = 0; s < tile_size; s++)
		{
			auto idx = noarr::idx<'v', 'y', 'x'>(s, n - 1, x);
			d[idx] *= scratch[idx];

			// std::cout << "n-1: " << (dens_l | noarr::get_at<'x', 's'>(densities, n - 1, s)) << std::endl;
		}

		for (index_t i = n - 2; i >= 0; i--)
		{
			for (index_t s = 0; s < tile_size; s++)
			{
				auto idx = noarr::idx<'v', 'y', 'x'>(s, i, x);
				auto next_idx = noarr::idx<'v', 'y', 'x'>(s, i + 1, x);

				d[idx] = (d[idx] - c_bag[idx] * d[next_idx]) * scratch[idx];

				// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_y_3d(real_t* __restrict__ densities, const real_t* __restrict__ a, const real_t* __restrict__ b,
							 const real_t* __restrict__ c, real_t* __restrict__ b_scratch,
							 const density_layout_t dens_l, const diagonal_layout_t diag_l, const index_t s_idx,
							 index_t x_tile_size)
{
	const index_t n = dens_l | noarr::get_length<'y'>();
	const index_t z_len = dens_l | noarr::get_length<'z'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();

	auto blocked_dens_l = dens_l ^ noarr::fix<'s'>(s_idx) ^ noarr::into_blocks_dynamic<'x', 'x', 'v', 'b'>(x_tile_size)
						  ^ noarr::fix<'b'>(noarr::lit<0>);

	const index_t x_block_len = blocked_dens_l | noarr::get_length<'x'>();

	auto a_bag = noarr::make_bag(blocked_dens_l, a);
	auto b_bag = noarr::make_bag(blocked_dens_l, b);
	auto c_bag = noarr::make_bag(blocked_dens_l, c);

	auto d = noarr::make_bag(blocked_dens_l, densities);

	auto scratch = noarr::make_bag(diag_l, b_scratch);

#pragma omp for schedule(static) nowait collapse(2)
	for (index_t z = 0; z < z_len; z++)
		for (index_t x = 0; x < x_block_len; x++)
		{
			const auto remainder = x_len % x_tile_size;
			const auto x_len_remainder = remainder == 0 ? x_tile_size : remainder;
			const auto tile_size = x == x_block_len - 1 ? x_len_remainder : x_tile_size;

			for (index_t s = 0; s < tile_size; s++)
			{
				auto idx = noarr::idx<'v', 'z', 'y', 'x'>(s, z, 0, x);
				scratch[idx] = 1 / b_bag[idx];
			}

			for (index_t i = 1; i < n; i++)
				for (index_t s = 0; s < tile_size; s++)
				{
					auto idx = noarr::idx<'v', 'z', 'y', 'x'>(s, z, i, x);
					auto prev_idx = noarr::idx<'v', 'z', 'y', 'x'>(s, z, i - 1, x);

					auto r = a_bag[idx] * scratch[prev_idx];

					scratch[idx] = 1 / (b_bag[idx] - c_bag[prev_idx] * r);

					d[idx] -= r * d[prev_idx];

					// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
				}

			for (index_t s = 0; s < tile_size; s++)
			{
				auto idx = noarr::idx<'v', 'z', 'y', 'x'>(s, z, n - 1, x);
				d[idx] *= scratch[idx];

				// std::cout << "n-1: " << (dens_l | noarr::get_at<'x', 's'>(densities, n - 1, s)) << std::endl;
			}

			for (index_t i = n - 2; i >= 0; i--)
				for (index_t s = 0; s < tile_size; s++)
				{
					auto idx = noarr::idx<'v', 'z', 'y', 'x'>(s, z, i, x);
					auto next_idx = noarr::idx<'v', 'z', 'y', 'x'>(s, z, i + 1, x);

					d[idx] = (d[idx] - c_bag[idx] * d[next_idx]) * scratch[idx];

					// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
				}
		}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_z_3d(real_t* __restrict__ densities, const real_t* __restrict__ a, const real_t* __restrict__ b,
							 const real_t* __restrict__ c, real_t* __restrict__ b_scratch,
							 const density_layout_t dens_l, const diagonal_layout_t diag_l, const index_t s_idx,
							 index_t x_tile_size)
{
	const index_t n = dens_l | noarr::get_length<'z'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();

	auto blocked_dens_l = dens_l ^ noarr::fix<'s'>(s_idx) ^ noarr::into_blocks_dynamic<'x', 'x', 'v', 'b'>(x_tile_size)
						  ^ noarr::fix<'b'>(noarr::lit<0>);

	const index_t x_block_len = blocked_dens_l | noarr::get_length<'x'>();

	auto a_bag = noarr::make_bag(blocked_dens_l, a);
	auto b_bag = noarr::make_bag(blocked_dens_l, b);
	auto c_bag = noarr::make_bag(blocked_dens_l, c);

	auto d = noarr::make_bag(blocked_dens_l, densities);

	auto scratch = noarr::make_bag(diag_l, b_scratch);

#pragma omp for schedule(static) nowait collapse(2)
	for (index_t y = 0; y < y_len; y++)
		for (index_t x = 0; x < x_block_len; x++)
		{
			const auto remainder = x_len % x_tile_size;
			const auto x_len_remainder = remainder == 0 ? x_tile_size : remainder;
			const auto tile_size = x == x_block_len - 1 ? x_len_remainder : x_tile_size;

			for (index_t s = 0; s < tile_size; s++)
			{
				auto idx = noarr::idx<'v', 'z', 'y', 'x'>(s, 0, y, x);
				scratch[idx] = 1 / b_bag[idx];
			}

			for (index_t i = 1; i < n; i++)
				for (index_t s = 0; s < tile_size; s++)
				{
					auto idx = noarr::idx<'v', 'z', 'y', 'x'>(s, i, y, x);
					auto prev_idx = noarr::idx<'v', 'z', 'y', 'x'>(s, i - 1, y, x);

					auto r = a_bag[idx] * scratch[prev_idx];

					scratch[idx] = 1 / (b_bag[idx] - c_bag[prev_idx] * r);

					d[idx] -= r * d[prev_idx];

					// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
				}

			for (index_t s = 0; s < tile_size; s++)
			{
				auto idx = noarr::idx<'v', 'z', 'y', 'x'>(s, n - 1, y, x);
				d[idx] *= scratch[idx];

				// std::cout << "n-1: " << (dens_l | noarr::get_at<'x', 's'>(densities, n - 1, s)) << std::endl;
			}

			for (index_t i = n - 2; i >= 0; i--)
				for (index_t s = 0; s < tile_size; s++)
				{
					auto idx = noarr::idx<'v', 'z', 'y', 'x'>(s, i, y, x);
					auto next_idx = noarr::idx<'v', 'z', 'y', 'x'>(s, i + 1, y, x);

					d[idx] = (d[idx] - c_bag[idx] * d[next_idx]) * scratch[idx];

					// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
				}
		}
}

template <typename real_t, bool aligned_x>
void sdd_least_memory_thomas_solver_t<real_t, aligned_x>::solve_x()
{
	if (this->problem_.dims == 1) {}
	else if (this->problem_.dims == 2)
	{
#pragma omp parallel
		for (index_t s = 0; s < this->problem_.substrates_count; s++)
		{
			if (continuous_x_diagonal_)
				solve_slice_x_2d_and_3d_transpose_l<index_t>(
					this->substrates_, ax_, bx_, cx_, b_scratch_[get_thread_num()],
					get_substrates_layout<2>() ^ noarr::rename<'y', 'm'>(), get_diag_layout_x<2>(),
					get_scratch_layout<'x'>(), s, this->problem_.nx);
			else
				solve_slice_x_2d_and_3d_transpose<index_t>(this->substrates_, ax_, bx_, cx_,
														   b_scratch_[get_thread_num()],
														   get_substrates_layout<2>() ^ noarr::rename<'y', 'm'>(),
														   get_scratch_layout<'x'>(), s, this->problem_.nx);
		}
	}
	else if (this->problem_.dims == 3)
	{
#pragma omp parallel
		for (index_t s = 0; s < this->problem_.substrates_count; s++)
			if (continuous_x_diagonal_)
				solve_slice_x_2d_and_3d_transpose_l<index_t>(
					this->substrates_, ax_, bx_, cx_, b_scratch_[get_thread_num()],
					get_substrates_layout<3>() ^ noarr::merge_blocks<'z', 'y', 'm'>(), get_diag_layout_x<3>(),
					get_scratch_layout<'x'>(), s, this->problem_.nx);
			else
				solve_slice_x_2d_and_3d_transpose<index_t>(
					this->substrates_, ax_, bx_, cx_, b_scratch_[get_thread_num()],
					get_substrates_layout<3>() ^ noarr::merge_blocks<'z', 'y', 'm'>(), get_scratch_layout<'x'>(), s,
					this->problem_.nx);
	}
}

template <typename real_t, bool aligned_x>
void sdd_least_memory_thomas_solver_t<real_t, aligned_x>::solve_y()
{
	if (this->problem_.dims == 2)
	{
#pragma omp parallel
		for (index_t s = 0; s < this->problem_.substrates_count; s++)
			solve_slice_y_2d<index_t>(this->substrates_, ay_, by_, cy_, b_scratch_[get_thread_num()],
									  get_substrates_layout<2>(), get_scratch_layout<'y'>(), s, x_tile_size_);
	}
	else if (this->problem_.dims == 3)
	{
#pragma omp parallel
		for (index_t s = 0; s < this->problem_.substrates_count; s++)
			solve_slice_y_3d<index_t>(this->substrates_, ay_, by_, cy_, b_scratch_[get_thread_num()],
									  get_substrates_layout<3>(), get_scratch_layout<'y'>(), s, x_tile_size_);
	}
}

template <typename real_t, bool aligned_x>
void sdd_least_memory_thomas_solver_t<real_t, aligned_x>::solve_z()
{
#pragma omp parallel
	{
		for (index_t s = 0; s < this->problem_.substrates_count; s++)
			solve_slice_z_3d<index_t>(this->substrates_, az_, bz_, cz_, b_scratch_[get_thread_num()],
									  get_substrates_layout<3>(), get_scratch_layout<'z'>(), s, x_tile_size_);
	}
}

template <typename real_t, bool aligned_x>
void sdd_least_memory_thomas_solver_t<real_t, aligned_x>::solve()
{
	if (this->problem_.dims == 1) {}
	if (this->problem_.dims == 2)
	{
#pragma omp parallel
		{
			perf_counter counter("sdd-lstmt");

			for (index_t s = 0; s < this->problem_.substrates_count; s++)
			{
				for (index_t i = 0; i < this->problem_.iterations; i++)
				{
					if (continuous_x_diagonal_)
						solve_slice_x_2d_and_3d_transpose_l<index_t>(
							this->substrates_, ax_, bx_, cx_, b_scratch_[get_thread_num()],
							get_substrates_layout<2>() ^ noarr::rename<'y', 'm'>(), get_diag_layout_x<2>(),
							get_scratch_layout<'x'>(), s, this->problem_.nx);
					else
						solve_slice_x_2d_and_3d_transpose<index_t>(
							this->substrates_, ax_, bx_, cx_, b_scratch_[get_thread_num()],
							get_substrates_layout<2>() ^ noarr::rename<'y', 'm'>(), get_scratch_layout<'x'>(), s,
							this->problem_.nx);
#pragma omp barrier
					solve_slice_y_2d<index_t>(this->substrates_, ay_, by_, cy_, b_scratch_[get_thread_num()],
											  get_substrates_layout<2>(), get_scratch_layout<'y'>(), s, x_tile_size_);
#pragma omp barrier
				}
			}
		}
	}
	if (this->problem_.dims == 3)
	{
#pragma omp parallel
		{
			perf_counter counter("sdd-lstmt");

			for (index_t s = 0; s < this->problem_.substrates_count; s++)
			{
				for (index_t i = 0; i < this->problem_.iterations; i++)
				{
					if (continuous_x_diagonal_)
						solve_slice_x_2d_and_3d_transpose_l<index_t>(
							this->substrates_, ax_, bx_, cx_, b_scratch_[get_thread_num()],
							get_substrates_layout<3>() ^ noarr::merge_blocks<'z', 'y', 'm'>(), get_diag_layout_x<3>(),
							get_scratch_layout<'x'>(), s, this->problem_.nx);
					else
						solve_slice_x_2d_and_3d_transpose<index_t>(
							this->substrates_, ax_, bx_, cx_, b_scratch_[get_thread_num()],
							get_substrates_layout<3>() ^ noarr::merge_blocks<'z', 'y', 'm'>(),
							get_scratch_layout<'x'>(), s, this->problem_.nx);
#pragma omp barrier
					solve_slice_y_3d<index_t>(this->substrates_, ay_, by_, cy_, b_scratch_[get_thread_num()],
											  get_substrates_layout<3>(), get_scratch_layout<'y'>(), s, x_tile_size_);
#pragma omp barrier
					solve_slice_z_3d<index_t>(this->substrates_, az_, bz_, cz_, b_scratch_[get_thread_num()],
											  get_substrates_layout<3>(), get_scratch_layout<'z'>(), s, x_tile_size_);
#pragma omp barrier
				}
			}
		}
	}
}

template <typename real_t, bool aligned_x>
sdd_least_memory_thomas_solver_t<real_t, aligned_x>::sdd_least_memory_thomas_solver_t()
{}

template <typename real_t, bool aligned_x>
sdd_least_memory_thomas_solver_t<real_t, aligned_x>::~sdd_least_memory_thomas_solver_t()
{
	if (ax_)
	{
		std::free(ax_);
		std::free(bx_);
		std::free(cx_);
		for (auto& b_scratch : b_scratch_)
		{
			std::free(b_scratch);
		}
	}

	if (ay_)
	{
		std::free(ay_);
		std::free(by_);
		std::free(cy_);
	}

	if (az_)
	{
		std::free(az_);
		std::free(bz_);
		std::free(cz_);
	}
}

template class sdd_least_memory_thomas_solver_t<float, false>;
template class sdd_least_memory_thomas_solver_t<double, false>;

template class sdd_least_memory_thomas_solver_t<float, true>;
template class sdd_least_memory_thomas_solver_t<double, true>;
