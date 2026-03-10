#include "least_memory_thomas_solver_t.h"

#include <cstddef>

#include "perf_utils.h"
#include "vector_transpose_helper.h"

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_t<real_t, aligned_x>::precompute_values(real_t*& a, real_t*& b1, real_t*& b,
																		index_t shape, index_t dims, index_t n)
{
	auto layout = get_diagonal_layout(this->problem_, n);

	if (aligned_x)
	{
		b = (real_t*)std::aligned_alloc(alignment_size_, (layout | noarr::get_size()));
	}
	else
	{
		b = (real_t*)std::malloc((layout | noarr::get_size()));
	}

	a = (real_t*)std::malloc(this->problem_.substrates_count * sizeof(real_t));
	b1 = (real_t*)std::malloc(this->problem_.substrates_count * sizeof(real_t));

	auto b_diag = noarr::make_bag(layout, b);

	// compute a
	for (index_t s = 0; s < this->problem_.substrates_count; s++)
		a[s] = -this->problem_.dt * this->problem_.diffusion_coefficients[s] / (shape * shape);

	// compute b1
	for (index_t s = 0; s < this->problem_.substrates_count; s++)
		b1[s] = 1 + this->problem_.decay_rates[s] * this->problem_.dt / dims
				+ 2 * this->problem_.dt * this->problem_.diffusion_coefficients[s] / (shape * shape);

	// compute b_i
	{
		std::array<index_t, 2> indices = { 0, n - 1 };

		for (index_t i : indices)
			for (index_t s = 0; s < this->problem_.substrates_count; s++)
				b_diag.template at<'i', 's'>(i, s) =
					1 + this->problem_.decay_rates[s] * this->problem_.dt / dims
					+ this->problem_.dt * this->problem_.diffusion_coefficients[s] / (shape * shape);

		for (index_t i = 1; i < n - 1; i++)
			for (index_t s = 0; s < this->problem_.substrates_count; s++)
				b_diag.template at<'i', 's'>(i, s) =
					1 + this->problem_.decay_rates[s] * this->problem_.dt / dims
					+ 2 * this->problem_.dt * this->problem_.diffusion_coefficients[s] / (shape * shape);
	}

	// compute b_i'
	{
		for (index_t s = 0; s < this->problem_.substrates_count; s++)
			b_diag.template at<'i', 's'>(0, s) = 1 / b_diag.template at<'i', 's'>(0, s);

		for (index_t i = 1; i < n; i++)
			for (index_t s = 0; s < this->problem_.substrates_count; s++)
			{
				b_diag.template at<'i', 's'>(i, s) =
					1 / (b_diag.template at<'i', 's'>(i, s) - a[s] * a[s] * b_diag.template at<'i', 's'>(i - 1, s));
			}
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_t<real_t, aligned_x>::prepare(const max_problem_t& problem)
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
void least_memory_thomas_solver_t<real_t, aligned_x>::tune(const nlohmann::json& params)
{
	x_tile_size_ = params.contains("x_tile_size") ? (std::size_t)params["x_tile_size"] : 48;
	alignment_size_ = params.contains("alignment_size") ? (std::size_t)params["alignment_size"] : 64;
	substrate_step_ =
		params.contains("substrate_step") ? (index_t)params["substrate_step"] : this->problem_.substrates_count;

	if (vectorized_x_)
	{
		using simd_tag = hn::ScalableTag<real_t>;
		simd_tag d;
		std::size_t vector_length = hn::Lanes(d) * sizeof(real_t);
		alignment_size_ = std::max(alignment_size_, vector_length);
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_t<real_t, aligned_x>::initialize()
{
	if (this->problem_.dims >= 1)
		precompute_values(ax_, b1x_, bx_, this->problem_.dx, this->problem_.dims, this->problem_.nx);
	if (this->problem_.dims >= 2)
		precompute_values(ay_, b1y_, by_, this->problem_.dy, this->problem_.dims, this->problem_.ny);
	if (this->problem_.dims >= 3)
		precompute_values(az_, b1z_, bz_, this->problem_.dz, this->problem_.dims, this->problem_.nz);
}

template <typename real_t, bool aligned_x>
auto least_memory_thomas_solver_t<real_t, aligned_x>::get_diagonal_layout(const problem_t<index_t, real_t>& problem,
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

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_x_1d(real_t* __restrict__ densities, const real_t* __restrict__ a,
							 const real_t* __restrict__ b1, const real_t* __restrict__ b, const density_layout_t dens_l,
							 const diagonal_layout_t diag_l, const index_t s_begin, const index_t s_end)
{
	const index_t n = dens_l | noarr::get_length<'x'>();

#pragma omp for schedule(static) nowait
	for (index_t s = s_begin; s < s_end; s++)
	{
		const real_t a_s = a[s];
		const real_t b1_s = b1[s];
		real_t b_tmp = a_s / (b1_s + a_s);

		for (index_t i = 1; i < n; i++)
		{
			(dens_l | noarr::get_at<'x', 's'>(densities, i, s)) -=
				(dens_l | noarr::get_at<'x', 's'>(densities, i - 1, s)) * b_tmp;

			b_tmp = a_s / (b1_s - a_s * b_tmp);

			// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
		}

		{
			(dens_l | noarr::get_at<'x', 's'>(densities, n - 1, s)) =
				(dens_l | noarr::get_at<'x', 's'>(densities, n - 1, s))
				* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));

			// std::cout << "n-1: " << (dens_l | noarr::get_at<'x', 's'>(densities, n - 1, s)) << std::endl;
		}

		for (index_t i = n - 2; i >= 0; i--)
		{
			(dens_l | noarr::get_at<'x', 's'>(densities, i, s)) =
				((dens_l | noarr::get_at<'x', 's'>(densities, i, s))
				 - a_s * (dens_l | noarr::get_at<'x', 's'>(densities, i + 1, s)))
				* (diag_l | noarr::get_at<'i', 's'>(b, i, s));

			// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_x_2d_and_3d_transpose(real_t* __restrict__ densities, const real_t* __restrict__ a,
											  const real_t* __restrict__ b1, const real_t* __restrict__ b,
											  const density_layout_t dens_l, const diagonal_layout_t diag_l,
											  const index_t s_begin, const index_t s_end)
{
	const index_t n = dens_l | noarr::get_length<'x'>();

	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag d;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(d);
	using simd_t = hn::Vec<simd_tag>;

	auto blocked_dens_l = dens_l ^ noarr::into_blocks_static<'m', 'b', 'm', 'v'>(simd_length);

	for (index_t s = s_begin; s < s_end; s++)
	{
		const real_t a_s = a[s];
		const real_t b1_s = b1[s];

		// vectorized body
		{
			const index_t full_n = (n + simd_length - 1) / simd_length * simd_length;

			auto body_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<0>);
			const index_t m = body_dens_l | noarr::get_length<'m'>();

#pragma omp for schedule(static) nowait
			for (index_t yz = 0; yz < m; yz++)
			{
				// vector registers that hold the to be transposed x*yz plane
				simd_t* rows = new simd_t[simd_length];

				simd_t prev = hn::Zero(d);
				real_t b_tmp = a_s / (b1_s + a_s);

				// forward substitution until last simd_length elements
				for (index_t i = 0; i < full_n - simd_length; i += simd_length)
				{
					// aligned loads
					for (index_t v = 0; v < simd_length; v++)
						rows[v] =
							hn::Load(d, &(body_dens_l | noarr::get_at<'m', 'v', 'x', 's'>(densities, yz, v, i, s)));

					// transposition to enable vectorization
					transpose(rows);

					// actual forward substitution (vectorized)
					{
						rows[0] = hn::MulAdd(prev, hn::Set(d, -b_tmp), rows[0]);

						for (index_t v = 1; v < simd_length; v++)
						{
							rows[v] = hn::MulAdd(rows[v - 1], hn::Set(d, -b_tmp), rows[v]);
							b_tmp = a_s / (b1_s - a_s * b_tmp);
						};

						prev = rows[simd_length - 1];
					}

					// transposition back to the original form
					// transpose(rows);

					// aligned stores
					for (index_t v = 0; v < simd_length; v++)
						hn::Store(rows[v], d,
								  &(body_dens_l | noarr::get_at<'m', 'v', 'x', 's'>(densities, yz, v, i, s)));
				}

				// we are aligned to the vector size, so we can safely continue
				// here we fuse the end of forward substitution and the beginning of backwards propagation
				{
					// aligned loads
					for (index_t v = 0; v < simd_length; v++)
						rows[v] = hn::Load(
							d, &(body_dens_l
								 | noarr::get_at<'m', 'v', 'x', 's'>(densities, yz, v, full_n - simd_length, s)));

					// transposition to enable vectorization
					transpose(rows);

					index_t remainder_work = n % simd_length;
					remainder_work += remainder_work == 0 ? simd_length : 0;

					// the rest of forward part
					{
						rows[0] = hn::MulAdd(prev, hn::Set(d, -b_tmp), rows[0]);

						if (n > simd_length)
							b_tmp = a_s / (b1_s - a_s * b_tmp);

						for (index_t v = 1; v < remainder_work; v++)
						{
							rows[v] = hn::MulAdd(rows[v - 1], hn::Set(d, -b_tmp), rows[v]);
							b_tmp = a_s / (b1_s - a_s * b_tmp);
						}
					}

					// the begin of backward part
					{
						auto b_tmp = hn::Load(d, &(diag_l | noarr::get_at<'i', 's'>(b, full_n - simd_length, s)));

						rows[remainder_work - 1] =
							hn::Mul(rows[remainder_work - 1], hn::Set(d, hn::ExtractLane(b_tmp, remainder_work - 1)));
						for (index_t v = remainder_work - 2; v >= 0; v--)
						{
							rows[v] = hn::Mul(hn::MulAdd(rows[v + 1], hn::Set(d, -a_s), rows[v]),
											  hn::Set(d, hn::ExtractLane(b_tmp, v)));
						}

						prev = rows[0];
					}

					// transposition back to the original form
					transpose(rows);

					// aligned stores
					for (index_t v = 0; v < simd_length; v++)
						hn::Store(rows[v], d,
								  &(body_dens_l
									| noarr::get_at<'m', 'v', 'x', 's'>(densities, yz, v, full_n - simd_length, s)));
				}

				// we continue with backwards substitution
				for (index_t i = full_n - simd_length * 2; i >= 0; i -= simd_length)
				{
					// aligned loads
					for (index_t v = 0; v < simd_length; v++)
						rows[v] =
							hn::Load(d, &(body_dens_l | noarr::get_at<'m', 'v', 'x', 's'>(densities, yz, v, i, s)));

					// transposition to enable vectorization
					// transpose(rows);

					// backward propagation
					{
						auto b_tmp = hn::Load(d, &(diag_l | noarr::get_at<'i', 's'>(b, i, s)));

						rows[simd_length - 1] = hn::Mul(hn::MulAdd(prev, hn::Set(d, -a_s), rows[simd_length - 1]),
														hn::Set(d, hn::ExtractLane(b_tmp, simd_length - 1)));
						for (index_t v = simd_length - 2; v >= 0; v--)
						{
							rows[v] = hn::Mul(hn::MulAdd(rows[v + 1], hn::Set(d, -a_s), rows[v]),
											  hn::Set(d, hn::ExtractLane(b_tmp, v)));
						}

						prev = rows[0];
					}

					// transposition back to the original form
					transpose(rows);

					// aligned stores
					for (index_t v = 0; v < simd_length; v++)
						hn::Store(rows[v], d,
								  &(body_dens_l | noarr::get_at<'m', 'v', 'x', 's'>(densities, yz, v, i, s)));
				}

				delete[] rows;
			}
		}

		// yz remainder
		{
			auto rem_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<1>);
			const index_t v_len = rem_dens_l | noarr::get_length<'v'>();

#pragma omp for schedule(static) nowait
			for (index_t yz = 0; yz < v_len; yz++)
			{
				const real_t a_s = a[s];
				const real_t b1_s = b1[s];
				real_t b_tmp = a_s / (b1_s + a_s);

				for (index_t i = 1; i < n; i++)
				{
					(rem_dens_l | noarr::get_at<'m', 'v', 'x', 's'>(densities, noarr::lit<0>, yz, i, s)) -=
						(rem_dens_l | noarr::get_at<'m', 'v', 'x', 's'>(densities, noarr::lit<0>, yz, i - 1, s))
						* b_tmp;

					b_tmp = a_s / (b1_s - a_s * b_tmp);
				}

				{
					(rem_dens_l | noarr::get_at<'m', 'v', 'x', 's'>(densities, noarr::lit<0>, yz, n - 1, s)) =
						(rem_dens_l | noarr::get_at<'m', 'v', 'x', 's'>(densities, noarr::lit<0>, yz, n - 1, s))
						* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
				}

				for (index_t i = n - 2; i >= 0; i--)
				{
					(rem_dens_l | noarr::get_at<'m', 'v', 'x', 's'>(densities, noarr::lit<0>, yz, i, s)) =
						((rem_dens_l | noarr::get_at<'m', 'v', 'x', 's'>(densities, noarr::lit<0>, yz, i, s))
						 - a_s
							   * (rem_dens_l
								  | noarr::get_at<'m', 'v', 'x', 's'>(densities, noarr::lit<0>, yz, i + 1, s)))
						* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
				}
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_x_2d_and_3d(real_t* __restrict__ densities, const real_t* __restrict__ a,
									const real_t* __restrict__ b1, const real_t* __restrict__ b,
									const density_layout_t dens_l, const diagonal_layout_t diag_l,
									const index_t s_begin, const index_t s_end)
{
	const index_t n = dens_l | noarr::get_length<'x'>();
	const index_t m = dens_l | noarr::get_length<'m'>();

#pragma omp for schedule(static) collapse(2) nowait
	for (index_t s = s_begin; s < s_end; s++)
	{
		for (index_t yz = 0; yz < m; yz++)
		{
			const real_t a_s = a[s];
			const real_t b1_s = b1[s];
			real_t b_tmp = a_s / (b1_s + a_s);

			for (index_t i = 1; i < n; i++)
			{
				(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, i, s)) -=
					(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, i - 1, s)) * b_tmp;

				b_tmp = a_s / (b1_s - a_s * b_tmp);
			}

			{
				(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, n - 1, s)) =
					(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, n - 1, s))
					* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
			}

			for (index_t i = n - 2; i >= 0; i--)
			{
				(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, i, s)) =
					((dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, i, s))
					 - a_s * (dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, i + 1, s)))
					* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_y_2d(real_t* __restrict__ densities, const real_t* __restrict__ a,
							 const real_t* __restrict__ b1, const real_t* __restrict__ b, const density_layout_t dens_l,
							 const diagonal_layout_t diag_l, std::size_t x_tile_size, const index_t s_begin,
							 const index_t s_end)
{
	const index_t n = dens_l | noarr::get_length<'y'>();

	auto blocked_dens_l = dens_l ^ noarr::into_blocks_static<'x', 'b', 'X', 'x'>(x_tile_size);

	for (index_t s = s_begin; s < s_end; s++)
	{
		// body
		{
			auto body_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<0>);
			const index_t x_len = body_dens_l | noarr::get_length<'x'>();
			const index_t X_len = body_dens_l | noarr::get_length<'X'>();

#pragma omp for schedule(static) nowait
			for (index_t X = 0; X < X_len; X++)
			{
				const real_t a_s = a[s];
				const real_t b1_s = b1[s];
				real_t b_tmp = a_s / (b1_s + a_s);

				for (index_t i = 1; i < n; i++)
				{
					for (index_t x = 0; x < x_len; x++)
					{
						(body_dens_l | noarr::get_at<'X', 'x', 'y', 's'>(densities, X, x, i, s)) -=
							(body_dens_l | noarr::get_at<'X', 'x', 'y', 's'>(densities, X, x, i - 1, s)) * b_tmp;
					}

					b_tmp = a_s / (b1_s - a_s * b_tmp);
				}

				for (index_t x = 0; x < x_len; x++)
				{
					(body_dens_l | noarr::get_at<'X', 'x', 'y', 's'>(densities, X, x, n - 1, s)) =
						(body_dens_l | noarr::get_at<'X', 'x', 'y', 's'>(densities, X, x, n - 1, s))
						* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
				}

				for (index_t i = n - 2; i >= 0; i--)
				{
					for (index_t x = 0; x < x_len; x++)
					{
						(body_dens_l | noarr::get_at<'X', 'x', 'y', 's'>(densities, X, x, i, s)) =
							((body_dens_l | noarr::get_at<'X', 'x', 'y', 's'>(densities, X, x, i, s))
							 - a_s * (body_dens_l | noarr::get_at<'X', 'x', 'y', 's'>(densities, X, x, i + 1, s)))
							* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
					}
				}
			}
		}

		// remainder
		{
			auto border_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<1>);
			const index_t x_len = border_dens_l | noarr::get_length<'x'>();

#pragma omp single
			{
				const real_t a_s = a[s];
				const real_t b1_s = b1[s];
				real_t b_tmp = a_s / (b1_s + a_s);

				for (index_t i = 1; i < n; i++)
				{
					for (index_t x = 0; x < x_len; x++)
					{
						(border_dens_l | noarr::get_at<'X', 'x', 'y', 's'>(densities, noarr::lit<0>, x, i, s)) -=
							(border_dens_l | noarr::get_at<'X', 'x', 'y', 's'>(densities, noarr::lit<0>, x, i - 1, s))
							* b_tmp;
					}

					b_tmp = a_s / (b1_s - a_s * b_tmp);
				}

				for (index_t x = 0; x < x_len; x++)
				{
					(border_dens_l | noarr::get_at<'X', 'x', 'y', 's'>(densities, noarr::lit<0>, x, n - 1, s)) =
						(border_dens_l | noarr::get_at<'X', 'x', 'y', 's'>(densities, noarr::lit<0>, x, n - 1, s))
						* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
				}

				for (index_t i = n - 2; i >= 0; i--)
				{
					for (index_t x = 0; x < x_len; x++)
					{
						(border_dens_l | noarr::get_at<'X', 'x', 'y', 's'>(densities, noarr::lit<0>, x, i, s)) =
							((border_dens_l | noarr::get_at<'X', 'x', 'y', 's'>(densities, noarr::lit<0>, x, i, s))
							 - a_s
								   * (border_dens_l
									  | noarr::get_at<'X', 'x', 'y', 's'>(densities, noarr::lit<0>, x, i + 1, s)))
							* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
					}
				}
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_y_3d(real_t* __restrict__ densities, const real_t* __restrict__ a,
							 const real_t* __restrict__ b1, const real_t* __restrict__ b, const density_layout_t dens_l,
							 const diagonal_layout_t diag_l, std::size_t x_tile_size, const index_t s_begin,
							 const index_t s_end)
{
	const index_t n = dens_l | noarr::get_length<'y'>();
	const index_t z_len = dens_l | noarr::get_length<'z'>();

	auto blocked_dens_l = dens_l ^ noarr::into_blocks_static<'x', 'b', 'X', 'x'>(x_tile_size);

#pragma omp for schedule(static) collapse(2) nowait
	for (index_t s = s_begin; s < s_end; s++)
	{
		for (index_t z = 0; z < z_len; z++)
		{
			// body
			{
				auto body_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<0>);
				const index_t x_len = body_dens_l | noarr::get_length<'x'>();
				const index_t X_len = body_dens_l | noarr::get_length<'X'>();

				for (index_t X = 0; X < X_len; X++)
				{
					const real_t a_s = a[s];
					const real_t b1_s = b1[s];
					real_t b_tmp = a_s / (b1_s + a_s);

					for (index_t i = 1; i < n; i++)
					{
						for (index_t x = 0; x < x_len; x++)
						{
							(body_dens_l | noarr::get_at<'z', 'X', 'x', 'y', 's'>(densities, z, X, x, i, s)) -=
								(body_dens_l | noarr::get_at<'z', 'X', 'x', 'y', 's'>(densities, z, X, x, i - 1, s))
								* b_tmp;
						}

						b_tmp = a_s / (b1_s - a_s * b_tmp);
					}

					for (index_t x = 0; x < x_len; x++)
					{
						(body_dens_l | noarr::get_at<'z', 'X', 'x', 'y', 's'>(densities, z, X, x, n - 1, s)) =
							(body_dens_l | noarr::get_at<'z', 'X', 'x', 'y', 's'>(densities, z, X, x, n - 1, s))
							* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
					}

					for (index_t i = n - 2; i >= 0; i--)
					{
						for (index_t x = 0; x < x_len; x++)
						{
							(body_dens_l | noarr::get_at<'z', 'X', 'x', 'y', 's'>(densities, z, X, x, i, s)) =
								((body_dens_l | noarr::get_at<'z', 'X', 'x', 'y', 's'>(densities, z, X, x, i, s))
								 - a_s
									   * (body_dens_l
										  | noarr::get_at<'z', 'X', 'x', 'y', 's'>(densities, z, X, x, i + 1, s)))
								* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
						}
					}
				}
			}

			// remainder
			{
				auto border_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<1>);
				const index_t x_len = border_dens_l | noarr::get_length<'x'>();

				{
					const real_t a_s = a[s];
					const real_t b1_s = b1[s];
					real_t b_tmp = a_s / (b1_s + a_s);

					for (index_t i = 1; i < n; i++)
					{
						for (index_t x = 0; x < x_len; x++)
						{
							(border_dens_l
							 | noarr::get_at<'z', 'X', 'x', 'y', 's'>(densities, z, noarr::lit<0>, x, i, s)) -=
								(border_dens_l
								 | noarr::get_at<'z', 'X', 'x', 'y', 's'>(densities, z, noarr::lit<0>, x, i - 1, s))
								* b_tmp;
						}

						b_tmp = a_s / (b1_s - a_s * b_tmp);
					}

					for (index_t x = 0; x < x_len; x++)
					{
						(border_dens_l
						 | noarr::get_at<'z', 'X', 'x', 'y', 's'>(densities, z, noarr::lit<0>, x, n - 1, s)) =
							(border_dens_l
							 | noarr::get_at<'z', 'X', 'x', 'y', 's'>(densities, z, noarr::lit<0>, x, n - 1, s))
							* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
					}

					for (index_t i = n - 2; i >= 0; i--)
					{
						for (index_t x = 0; x < x_len; x++)
						{
							(border_dens_l
							 | noarr::get_at<'z', 'X', 'x', 'y', 's'>(densities, z, noarr::lit<0>, x, i, s)) =
								((border_dens_l
								  | noarr::get_at<'z', 'X', 'x', 'y', 's'>(densities, z, noarr::lit<0>, x, i, s))
								 - a_s
									   * (border_dens_l
										  | noarr::get_at<'z', 'X', 'x', 'y', 's'>(densities, z, noarr::lit<0>, x,
																				   i + 1, s)))
								* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
						}
					}
				}
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_z_3d(real_t* __restrict__ densities, const real_t* __restrict__ a,
							 const real_t* __restrict__ b1, const real_t* __restrict__ b, const density_layout_t dens_l,
							 const diagonal_layout_t diag_l, std::size_t x_tile_size, const index_t s_begin,
							 const index_t s_end)
{
	const index_t n = dens_l | noarr::get_length<'z'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();

	auto blocked_dens_l = dens_l ^ noarr::into_blocks_static<'x', 'b', 'X', 'x'>(x_tile_size);

#pragma omp for schedule(static) collapse(2) nowait
	for (index_t s = s_begin; s < s_end; s++)
	{
		for (index_t y = 0; y < y_len; y++)
		{
			// body
			{
				auto body_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<0>);
				const index_t x_len = body_dens_l | noarr::get_length<'x'>();
				const index_t X_len = body_dens_l | noarr::get_length<'X'>();

				for (index_t X = 0; X < X_len; X++)
				{
					const real_t a_s = a[s];
					const real_t b1_s = b1[s];
					real_t b_tmp = a_s / (b1_s + a_s);

					for (index_t i = 1; i < n; i++)
					{
						for (index_t x = 0; x < x_len; x++)
						{
							(body_dens_l | noarr::get_at<'y', 'X', 'x', 'z', 's'>(densities, y, X, x, i, s)) -=
								(body_dens_l | noarr::get_at<'y', 'X', 'x', 'z', 's'>(densities, y, X, x, i - 1, s))
								* b_tmp;
						}

						b_tmp = a_s / (b1_s - a_s * b_tmp);
					}

					for (index_t x = 0; x < x_len; x++)
					{
						(body_dens_l | noarr::get_at<'y', 'X', 'x', 'z', 's'>(densities, y, X, x, n - 1, s)) =
							(body_dens_l | noarr::get_at<'y', 'X', 'x', 'z', 's'>(densities, y, X, x, n - 1, s))
							* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
					}

					for (index_t i = n - 2; i >= 0; i--)
					{
						for (index_t x = 0; x < x_len; x++)
						{
							(body_dens_l | noarr::get_at<'y', 'X', 'x', 'z', 's'>(densities, y, X, x, i, s)) =
								((body_dens_l | noarr::get_at<'y', 'X', 'x', 'z', 's'>(densities, y, X, x, i, s))
								 - a_s
									   * (body_dens_l
										  | noarr::get_at<'y', 'X', 'x', 'z', 's'>(densities, y, X, x, i + 1, s)))
								* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
						}
					}
				}
			}

			// remainder
			{
				auto border_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<1>);
				const index_t x_len = border_dens_l | noarr::get_length<'x'>();

				{
					const real_t a_s = a[s];
					const real_t b1_s = b1[s];
					real_t b_tmp = a_s / (b1_s + a_s);

					for (index_t i = 1; i < n; i++)
					{
						for (index_t x = 0; x < x_len; x++)
						{
							(border_dens_l
							 | noarr::get_at<'y', 'X', 'x', 'z', 's'>(densities, y, noarr::lit<0>, x, i, s)) -=
								(border_dens_l
								 | noarr::get_at<'y', 'X', 'x', 'z', 's'>(densities, y, noarr::lit<0>, x, i - 1, s))
								* b_tmp;
						}

						b_tmp = a_s / (b1_s - a_s * b_tmp);
					}

					for (index_t x = 0; x < x_len; x++)
					{
						(border_dens_l
						 | noarr::get_at<'y', 'X', 'x', 'z', 's'>(densities, y, noarr::lit<0>, x, n - 1, s)) =
							(border_dens_l
							 | noarr::get_at<'y', 'X', 'x', 'z', 's'>(densities, y, noarr::lit<0>, x, n - 1, s))
							* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
					}

					for (index_t i = n - 2; i >= 0; i--)
					{
						for (index_t x = 0; x < x_len; x++)
						{
							(border_dens_l
							 | noarr::get_at<'y', 'X', 'x', 'z', 's'>(densities, y, noarr::lit<0>, x, i, s)) =
								((border_dens_l
								  | noarr::get_at<'y', 'X', 'x', 'z', 's'>(densities, y, noarr::lit<0>, x, i, s))
								 - a_s
									   * (border_dens_l
										  | noarr::get_at<'y', 'X', 'x', 'z', 's'>(densities, y, noarr::lit<0>, x,
																				   i + 1, s)))
								* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
						}
					}
				}
			}
		}
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_t<real_t, aligned_x>::solve_x()
{
	if (vectorized_x_)
	{
		if (this->problem_.dims == 1)
		{
#pragma omp parallel
			solve_slice_x_1d<index_t>(this->substrates_, ax_, b1x_, bx_, get_substrates_layout<1>(),
									  get_diagonal_layout(this->problem_, this->problem_.nx), 0,
									  this->problem_.substrates_count);
		}
		else if (this->problem_.dims == 2)
		{
#pragma omp parallel
			solve_slice_x_2d_and_3d_transpose<index_t>(
				this->substrates_, ax_, b1x_, bx_, get_substrates_layout<2>() ^ noarr::rename<'y', 'm'>(),
				get_diagonal_layout(this->problem_, this->problem_.nx), 0, this->problem_.substrates_count);
		}
		else if (this->problem_.dims == 3)
		{
#pragma omp parallel
			solve_slice_x_2d_and_3d_transpose<index_t>(
				this->substrates_, ax_, b1x_, bx_, get_substrates_layout<3>() ^ noarr::merge_blocks<'z', 'y', 'm'>(),
				get_diagonal_layout(this->problem_, this->problem_.nx), 0, this->problem_.substrates_count);
		}
	}
	else
	{
		if (this->problem_.dims == 1)
		{
#pragma omp parallel
			solve_slice_x_1d<index_t>(this->substrates_, ax_, b1x_, bx_, get_substrates_layout<1>(),
									  get_diagonal_layout(this->problem_, this->problem_.nx), 0,
									  this->problem_.substrates_count);
		}
		else if (this->problem_.dims == 2)
		{
#pragma omp parallel
			solve_slice_x_2d_and_3d<index_t>(
				this->substrates_, ax_, b1x_, bx_, get_substrates_layout<2>() ^ noarr::rename<'y', 'm'>(),
				get_diagonal_layout(this->problem_, this->problem_.nx), 0, this->problem_.substrates_count);
		}
		else if (this->problem_.dims == 3)
		{
#pragma omp parallel
			solve_slice_x_2d_and_3d<index_t>(
				this->substrates_, ax_, b1x_, bx_, get_substrates_layout<3>() ^ noarr::merge_blocks<'z', 'y', 'm'>(),
				get_diagonal_layout(this->problem_, this->problem_.nx), 0, this->problem_.substrates_count);
		}
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_t<real_t, aligned_x>::solve_y()
{
	if (this->problem_.dims == 2)
	{
#pragma omp parallel
		solve_slice_y_2d<index_t>(this->substrates_, ay_, b1y_, by_, get_substrates_layout<2>(),
								  get_diagonal_layout(this->problem_, this->problem_.ny), x_tile_size_, 0,
								  this->problem_.substrates_count);
	}
	else if (this->problem_.dims == 3)
	{
#pragma omp parallel
		solve_slice_y_3d<index_t>(this->substrates_, ay_, b1y_, by_, get_substrates_layout<3>(),
								  get_diagonal_layout(this->problem_, this->problem_.ny), x_tile_size_, 0,
								  this->problem_.substrates_count);
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_t<real_t, aligned_x>::solve_z()
{
#pragma omp parallel
	solve_slice_z_3d<index_t>(this->substrates_, az_, b1z_, bz_, get_substrates_layout<3>(),
							  get_diagonal_layout(this->problem_, this->problem_.nz), x_tile_size_, 0,
							  this->problem_.substrates_count);
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_t<real_t, aligned_x>::solve()
{
	if (this->problem_.dims == 1)
	{
#pragma omp parallel
		{
			perf_counter counter("lstmt");

			for (index_t i = 0; i < this->problem_.iterations; i++)
				solve_slice_x_1d<index_t>(this->substrates_, ax_, b1x_, bx_, get_substrates_layout<1>(),
										  get_diagonal_layout(this->problem_, this->problem_.nx), 0,
										  this->problem_.substrates_count);
		}
	}
	if (this->problem_.dims == 2)
	{
#pragma omp parallel
		{
			perf_counter counter("lstmt");

			for (index_t s = 0; s < this->problem_.substrates_count; s += substrate_step_)
			{
				for (index_t i = 0; i < this->problem_.iterations; i++)
				{
					auto s_step_length = std::min(substrate_step_, this->problem_.substrates_count - s);

					if (vectorized_x_)
						solve_slice_x_2d_and_3d_transpose<index_t>(
							this->substrates_, ax_, b1x_, bx_, get_substrates_layout<2>() ^ noarr::rename<'y', 'm'>(),
							get_diagonal_layout(this->problem_, this->problem_.nx), s, s + s_step_length);
					else
						solve_slice_x_2d_and_3d<index_t>(
							this->substrates_, ax_, b1x_, bx_, get_substrates_layout<2>() ^ noarr::rename<'y', 'm'>(),
							get_diagonal_layout(this->problem_, this->problem_.nx), s, s + s_step_length);
#pragma omp barrier
					solve_slice_y_2d<index_t>(this->substrates_, ay_, b1y_, by_, get_substrates_layout<2>(),
											  get_diagonal_layout(this->problem_, this->problem_.ny), x_tile_size_, s,
											  s + s_step_length);
#pragma omp barrier
				}
			}
		}
	}
	if (this->problem_.dims == 3)
	{
#pragma omp parallel
		{
			perf_counter counter("lstmt");

			for (index_t s = 0; s < this->problem_.substrates_count; s += substrate_step_)
			{
				for (index_t i = 0; i < this->problem_.iterations; i++)
				{
					auto s_step_length = std::min(substrate_step_, this->problem_.substrates_count - s);

					if (vectorized_x_)
						solve_slice_x_2d_and_3d_transpose<index_t>(
							this->substrates_, ax_, b1x_, bx_,
							get_substrates_layout<3>() ^ noarr::merge_blocks<'z', 'y', 'm'>(),
							get_diagonal_layout(this->problem_, this->problem_.nx), s, s + s_step_length);
					else
						solve_slice_x_2d_and_3d<index_t>(
							this->substrates_, ax_, b1x_, bx_,
							get_substrates_layout<3>() ^ noarr::merge_blocks<'z', 'y', 'm'>(),
							get_diagonal_layout(this->problem_, this->problem_.nx), s, s + s_step_length);
#pragma omp barrier
					solve_slice_y_3d<index_t>(this->substrates_, ay_, b1y_, by_, get_substrates_layout<3>(),
											  get_diagonal_layout(this->problem_, this->problem_.ny), x_tile_size_, s,
											  s + s_step_length);
#pragma omp barrier
					solve_slice_z_3d<index_t>(this->substrates_, az_, b1z_, bz_, get_substrates_layout<3>(),
											  get_diagonal_layout(this->problem_, this->problem_.nz), x_tile_size_, s,
											  s + s_step_length);
#pragma omp barrier
				}
			}
		}
	}
}

template <typename real_t, bool aligned_x>
least_memory_thomas_solver_t<real_t, aligned_x>::least_memory_thomas_solver_t(bool vectorized_x)
	: ax_(nullptr),
	  b1x_(nullptr),
	  bx_(nullptr),
	  ay_(nullptr),
	  b1y_(nullptr),
	  by_(nullptr),
	  az_(nullptr),
	  b1z_(nullptr),
	  bz_(nullptr),
	  vectorized_x_(vectorized_x)
{}

template <typename real_t, bool aligned_x>
least_memory_thomas_solver_t<real_t, aligned_x>::~least_memory_thomas_solver_t()
{
	if (bx_)
	{
		std::free(bx_);
		std::free(ax_);
		std::free(b1x_);
	}
	if (by_)
	{
		std::free(by_);
		std::free(ay_);
		std::free(b1y_);
	}
	if (bz_)
	{
		std::free(bz_);
		std::free(az_);
		std::free(b1z_);
	}
}

template class least_memory_thomas_solver_t<float, false>;
template class least_memory_thomas_solver_t<double, false>;

template class least_memory_thomas_solver_t<float, true>;
template class least_memory_thomas_solver_t<double, true>;
