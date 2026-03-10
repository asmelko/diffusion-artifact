#include "least_memory_thomas_solver_d_t.h"

#include <cstddef>
#include <iostream>

#include "perf_utils.h"
#include "vector_transpose_helper.h"

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_t<real_t, aligned_x>::precompute_values(real_t*& a, real_t*& b1, real_t*& c,
																		  index_t shape, index_t dims, index_t n)
{
	auto layout = get_diagonal_layout(this->problem_, n);

	if (aligned_x)
	{
		c = (real_t*)std::aligned_alloc(alignment_size_, (layout | noarr::get_size()));
	}
	else
	{
		c = (real_t*)std::malloc((layout | noarr::get_size()));
	}

	a = (real_t*)std::malloc(this->problem_.substrates_count * sizeof(real_t));
	b1 = (real_t*)std::malloc(this->problem_.substrates_count * sizeof(real_t));

	auto c_diag = noarr::make_bag(layout, c);

	// compute a
	for (index_t s = 0; s < this->problem_.substrates_count; s++)
		a[s] = -this->problem_.dt * this->problem_.diffusion_coefficients[s] / (shape * shape);

	// compute b1
	for (index_t s = 0; s < this->problem_.substrates_count; s++)
		b1[s] = 1 + this->problem_.decay_rates[s] * this->problem_.dt / dims
				+ 2 * this->problem_.dt * this->problem_.diffusion_coefficients[s] / (shape * shape);

	// compute c_i'
	for (index_t s = 0; s < this->problem_.substrates_count; s++)
	{
		c_diag.template at<'i', 's'>(0, s) = a[s] / (b1[s] + a[s]);

		for (index_t i = 1; i < n - 1; i++)
		{
			const real_t r = 1 / (b1[s] - a[s] * c_diag.template at<'i', 's'>(i - 1, s));
			c_diag.template at<'i', 's'>(i, s) = a[s] * r;
		}
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_t<real_t, aligned_x>::prepare(const max_problem_t& problem)
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
void least_memory_thomas_solver_d_t<real_t, aligned_x>::tune(const nlohmann::json& params)
{
	x_tile_size_ = params.contains("x_tile_size") ? (std::size_t)params["x_tile_size"] : 48;
	alignment_size_ = params.contains("alignment_size") ? (std::size_t)params["alignment_size"] : 64;
	alignment_multiple_ = alignment_size_;
	substrate_step_ =
		params.contains("substrate_step") ? (index_t)params["substrate_step"] : this->problem_.substrates_count;

	if (use_intrinsics_)
	{
		using simd_tag = hn::ScalableTag<real_t>;
		simd_tag d;
		std::size_t vector_length = hn::Lanes(d) * sizeof(real_t);
		alignment_size_ = std::max(alignment_size_, vector_length);
		x_tile_size_ = (x_tile_size_ + hn::Lanes(d) - 1) / hn::Lanes(d) * hn::Lanes(d);
		x_tile_size_ = std::min(x_tile_size_, 4 * hn::Lanes(d));
		alignment_multiple_ = std::max(alignment_size_, vector_length * x_tile_size_ / hn::Lanes(d));
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_t<real_t, aligned_x>::initialize()
{
	if (this->problem_.dims >= 1)
		precompute_values(ax_, b1x_, cx_, this->problem_.dx, this->problem_.dims, this->problem_.nx);
	if (this->problem_.dims >= 2)
		precompute_values(ay_, b1y_, cy_, this->problem_.dy, this->problem_.dims, this->problem_.ny);
	if (this->problem_.dims >= 3)
		precompute_values(az_, b1z_, cz_, this->problem_.dz, this->problem_.dims, this->problem_.nz);
}

template <typename real_t, bool aligned_x>
auto least_memory_thomas_solver_d_t<real_t, aligned_x>::get_diagonal_layout(const problem_t<index_t, real_t>& problem,
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
							 const real_t* __restrict__ b1, const real_t* __restrict__ back_c,
							 const density_layout_t dens_l, const diagonal_layout_t diag_l, const index_t s_begin,
							 const index_t s_end)
{
	const index_t n = dens_l | noarr::get_length<'x'>();

#pragma omp for schedule(static) nowait
	for (index_t s = s_begin; s < s_end; s++)
	{
		const real_t a_s = a[s];
		const real_t b1_s = b1[s];
		const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s'>(s), densities);
		const auto c = noarr::make_bag(diag_l ^ noarr::fix<'s'>(s), back_c);

		real_t a_tmp = 0;
		real_t b_tmp = b1_s + a_s;
		real_t c_tmp = a_s;
		real_t prev = 0;

		for (index_t i = 0; i < n; i++)
		{
			const real_t r = 1 / (b_tmp - a_tmp * c_tmp);

			real_t curr = d.template at<'x'>(i);
			curr = r * (curr - a_tmp * prev);
			d.template at<'x'>(i) = curr;

			c_tmp = a_s * r;

			// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << " c: " << c_tmp
			// 		  << " curr: " << curr << " prev: " << prev << std::endl;

			a_tmp = a_s;
			b_tmp = b1_s + (i == n - 2 ? a_s : 0);
			prev = curr;
		}

		for (index_t i = n - 2; i >= 0; i--)
		{
			real_t curr = d.template at<'x'>(i);
			curr = curr - c.template at<'i'>(i) * prev;
			d.template at<'x'>(i) = curr;

			// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s))
			// 		  << " c: " << c.template at<'i'>(i) << " curr: " << curr << " prev: " << prev << std::endl;

			prev = curr;
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_x_2d_and_3d_transpose(real_t* __restrict__ densities, const real_t* __restrict__ a,
											  const real_t* __restrict__ b1, const real_t* __restrict__ back_c,
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
				real_t a_tmp = 0;
				real_t b_tmp = b1_s + a_s;
				real_t c_tmp = a_s;
				simd_t prev = hn::Zero(d);
				// vector registers that hold the to be transposed x*yz plane
				simd_t* rows = new simd_t[simd_length + 1];

				// forward substitution until last simd_length elements
				for (index_t i = 0; i < full_n - simd_length; i += simd_length)
				{
					rows[0] = prev;

					// aligned loads
					for (index_t v = 0; v < simd_length; v++)
						rows[v + 1] =
							hn::Load(d, &(body_dens_l | noarr::get_at<'m', 'v', 'x', 's'>(densities, yz, v, i, s)));

					// transposition to enable vectorization
					transpose(rows + 1);

					// actual forward substitution (vectorized)
					{
						for (index_t v = 1; v < simd_length + 1; v++)
						{
							const real_t r = 1 / (b_tmp - a_tmp * c_tmp);

							rows[v] = hn::Mul(hn::MulAdd(rows[v - 1], hn::Set(d, -a_tmp), rows[v]), hn::Set(d, r));

							a_tmp = a_s;
							b_tmp = b1_s;
							c_tmp = a_s * r;
						};

						prev = rows[simd_length];
					}

					// transposition back to the original form
					// transpose(rows + 1);

					// aligned stores
					for (index_t v = 0; v < simd_length; v++)
						hn::Store(rows[v + 1], d,
								  &(body_dens_l | noarr::get_at<'m', 'v', 'x', 's'>(densities, yz, v, i, s)));
				}

				// we are aligned to the vector size, so we can safely continue
				// here we fuse the end of forward substitution and the beginning of backwards propagation
				{
					rows[0] = prev;

					// aligned loads
					for (index_t v = 0; v < simd_length; v++)
						rows[v + 1] = hn::Load(
							d, &(body_dens_l
								 | noarr::get_at<'m', 'v', 'x', 's'>(densities, yz, v, full_n - simd_length, s)));

					// transposition to enable vectorization
					transpose(rows + 1);

					index_t remainder_work = n % simd_length;
					remainder_work += remainder_work == 0 ? simd_length : 0;

					// the rest of forward part
					{
						if (remainder_work == 1)
							b_tmp = b1_s + a_s;

						for (index_t v = 0; v < remainder_work; v++)
						{
							const real_t r = 1 / (b_tmp - a_tmp * c_tmp);

							rows[v + 1] = hn::Mul(hn::MulAdd(rows[v], hn::Set(d, -a_tmp), rows[v + 1]), hn::Set(d, r));

							a_tmp = a_s;
							b_tmp = b1_s + (v == remainder_work - 2 ? a_s : 0);
							c_tmp = a_s * r;
						}
					}

					// the begin of backward part
					{
						auto c = hn::Load(d, &(diag_l | noarr::get_at<'i', 's'>(back_c, full_n - simd_length, s)));

						for (index_t v = remainder_work - 2; v >= 0; v--)
						{
							rows[v + 1] = hn::NegMulAdd(rows[v + 2], hn::Set(d, hn::ExtractLane(c, v)), rows[v + 1]);
						}

						prev = rows[1];
					}

					// transposition back to the original form
					transpose(rows + 1);

					// aligned stores
					for (index_t v = 0; v < simd_length; v++)
						hn::Store(rows[v + 1], d,
								  &(body_dens_l
									| noarr::get_at<'m', 'v', 'x', 's'>(densities, yz, v, full_n - simd_length, s)));
				}

				// we continue with backwards substitution
				for (index_t i = full_n - simd_length * 2; i >= 0; i -= simd_length)
				{
					rows[simd_length] = prev;

					// aligned loads
					for (index_t v = 0; v < simd_length; v++)
						rows[v] =
							hn::Load(d, &(body_dens_l | noarr::get_at<'m', 'v', 'x', 's'>(densities, yz, v, i, s)));

					// transposition to enable vectorization
					// transpose(rows);

					// backward propagation
					{
						auto c = hn::Load(d, &(diag_l | noarr::get_at<'i', 's'>(back_c, i, s)));

						for (index_t v = simd_length - 1; v >= 0; v--)
						{
							rows[v] = hn::NegMulAdd(rows[v + 1], hn::Set(d, hn::ExtractLane(c, v)), rows[v]);
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
				const auto d = noarr::make_bag(rem_dens_l ^ noarr::fix<'s', 'm', 'v'>(s, noarr::lit<0>, yz), densities);
				const auto c = noarr::make_bag(diag_l ^ noarr::fix<'s'>(s), back_c);

				real_t a_tmp = 0;
				real_t b_tmp = b1_s + a_s;
				real_t c_tmp = a_s;
				real_t prev = 0;

				for (index_t i = 0; i < n; i++)
				{
					const real_t r = 1 / (b_tmp - a_tmp * c_tmp);

					real_t curr = d.template at<'x'>(i);
					curr = r * (curr - a_tmp * prev);
					d.template at<'x'>(i) = curr;

					a_tmp = a_s;
					b_tmp = b1_s + (i == n - 2 ? a_s : 0);
					c_tmp = a_s * r;
					prev = curr;
				}

				for (index_t i = n - 2; i >= 0; i--)
				{
					real_t curr = d.template at<'x'>(i);
					curr = curr - c.template at<'i'>(i) * prev;
					d.template at<'x'>(i) = curr;

					prev = curr;
				}
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_x_2d_and_3d(real_t* __restrict__ densities, const real_t* __restrict__ a,
									const real_t* __restrict__ b1, const real_t* __restrict__ back_c,
									const density_layout_t dens_l, const diagonal_layout_t diag_l,
									const index_t s_begin, const index_t s_end)
{
	constexpr char dim = 'x';
	const index_t n = dens_l | noarr::get_length<dim>();
	const index_t m = dens_l | noarr::get_length<'m'>();

#pragma omp for schedule(static) collapse(2) nowait
	for (index_t s = s_begin; s < s_end; s++)
	{
		for (index_t yz = 0; yz < m; yz++)
		{
			const real_t a_s = a[s];
			const real_t b1_s = b1[s];
			const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s', 'm'>(s, yz), densities);
			const auto c = noarr::make_bag(diag_l ^ noarr::fix<'s'>(s), back_c);

			real_t a_tmp = 0;
			real_t b_tmp = b1_s + a_s;
			real_t c_tmp = a_s;
			real_t prev = 0;

			for (index_t i = 0; i < n; i++)
			{
				const real_t r = 1 / (b_tmp - a_tmp * c_tmp);

				real_t curr = d.template at<dim>(i);
				curr = r * (curr - a_tmp * prev);
				d.template at<dim>(i) = curr;

				a_tmp = a_s;
				b_tmp = b1_s + (i == n - 2 ? a_s : 0);
				c_tmp = a_s * r;
				prev = curr;
			}

			for (index_t i = n - 2; i >= 0; i--)
			{
				real_t curr = d.template at<dim>(i);
				curr = curr - c.template at<'i'>(i) * prev;
				d.template at<dim>(i) = curr;

				prev = curr;
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_y_2d(real_t* __restrict__ densities, const real_t* __restrict__ a,
							 const real_t* __restrict__ b1, const real_t* __restrict__ back_c,
							 const density_layout_t dens_l, const diagonal_layout_t diag_l, std::size_t x_tile_size,
							 const index_t s_begin, const index_t s_end)
{
	constexpr char dim = 'y';
	const index_t n = dens_l | noarr::get_length<dim>();

	auto blocked_dens_l = dens_l ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(x_tile_size);
	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

#pragma omp for schedule(static) nowait collapse(2)
	for (index_t s = s_begin; s < s_end; s++)
	{
		for (index_t X = 0; X < X_len; X++)
		{
			const real_t a_s = a[s];
			const real_t b1_s = b1[s];
			const auto d = noarr::make_bag(blocked_dens_l ^ noarr::fix<'s', 'b', 'X'>(s, noarr::lit<0>, X), densities);
			const auto c = noarr::make_bag(diag_l ^ noarr::fix<'s'>(s), back_c);

			const index_t remainder = (dens_l | noarr::get_length<'x'>()) % x_tile_size;
			const index_t x_len_remainder = remainder == 0 ? x_tile_size : remainder;
			const index_t x_len = X == X_len - 1 ? x_len_remainder : x_tile_size;

			real_t c_tmp = a_s;

			{
				const real_t r = 1 / (b1_s + a_s);

				for (index_t x = 0; x < x_len; x++)
				{
					d.template at<dim, 'x'>(0, x) *= r;
				}

				c_tmp = a_s * r;
			}

			for (index_t i = 1; i < n; i++)
			{
				const real_t b_tmp = b1_s + (i == n - 1 ? a_s : 0);
				const real_t r = 1 / (b_tmp - a_s * c_tmp);

				for (index_t x = 0; x < x_len; x++)
				{
					d.template at<dim, 'x'>(i, x) =
						r * (d.template at<dim, 'x'>(i, x) - a_s * d.template at<dim, 'x'>(i - 1, x));
				}

				c_tmp = a_s * r;
			}

			for (index_t i = n - 2; i >= 0; i--)
			{
				const auto back_c = c.template at<'i'>(i);

				for (index_t x = 0; x < x_len; x++)
					d.template at<dim, 'x'>(i, x) -= back_c * d.template at<dim, 'x'>(i + 1, x);
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_y_2d_intrinsics(real_t* __restrict__ densities, const real_t* __restrict__ a,
										const real_t* __restrict__ b1, const real_t* __restrict__ back_c,
										const density_layout_t dens_l, const diagonal_layout_t diag_l,
										const index_t s_begin, const index_t s_end)
{
	constexpr char dim = 'y';
	const index_t n = dens_l | noarr::get_length<dim>();

	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	auto blocked_dens_l = dens_l ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(simd_length);
	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

#pragma omp for schedule(static) nowait collapse(2)
	for (index_t s = s_begin; s < s_end; s++)
	{
		for (index_t X = 0; X < X_len; X++)
		{
			const real_t a_s = a[s];
			const real_t b1_s = b1[s];
			const auto d = noarr::make_bag(blocked_dens_l ^ noarr::fix<'s', 'b', 'X'>(s, noarr::lit<0>, X), densities);
			const auto c = noarr::make_bag(diag_l ^ noarr::fix<'s'>(s), back_c);

			real_t a_tmp = 0;
			real_t b_tmp = b1_s + a_s;
			real_t c_tmp = a_s;
			simd_t prev = hn::Zero(t);

			for (index_t i = 0; i < n; i++)
			{
				const real_t r = 1 / (b_tmp - a_tmp * c_tmp);

				simd_t curr = hn::Load(t, &d.template at<dim, 'x'>(i, 0));
				curr = hn::Mul(hn::MulAdd(hn::Set(t, -a_tmp), prev, curr), hn::Set(t, r));
				hn::Store(curr, t, &d.template at<dim, 'x'>(i, 0));

				a_tmp = a_s;
				b_tmp = b1_s + (i == n - 2 ? a_s : 0);
				c_tmp = a_s * r;
				prev = curr;
			}

			for (index_t i = n - 2; i >= 0; i--)
			{
				simd_t curr = hn::Load(t, &d.template at<dim, 'x'>(i, 0));
				curr = hn::MulAdd(hn::Set(t, -c.template at<'i'>(i)), prev, curr);
				hn::Store(curr, t, &d.template at<dim, 'x'>(i, 0));

				prev = curr;
			}
		}
	}
}


template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_y_3d(real_t* __restrict__ densities, const real_t* __restrict__ a,
							 const real_t* __restrict__ b1, const real_t* __restrict__ back_c,
							 const density_layout_t dens_l, const diagonal_layout_t diag_l, std::size_t x_tile_size,
							 const index_t s_begin, const index_t s_end)
{
	constexpr char dim = 'y';
	const index_t z_len = dens_l | noarr::get_length<'z'>();
	const index_t n = dens_l | noarr::get_length<dim>();

	auto blocked_dens_l = dens_l ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(x_tile_size);
	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

#pragma omp for schedule(static) nowait collapse(3)
	for (index_t s = s_begin; s < s_end; s++)
	{
		for (index_t z = 0; z < z_len; z++)
		{
			for (index_t X = 0; X < X_len; X++)
			{
				const real_t a_s = a[s];
				const real_t b1_s = b1[s];
				const auto d =
					noarr::make_bag(blocked_dens_l ^ noarr::fix<'s', 'z', 'b', 'X'>(s, z, noarr::lit<0>, X), densities);
				const auto c = noarr::make_bag(diag_l ^ noarr::fix<'s'>(s), back_c);

				const index_t remainder = (dens_l | noarr::get_length<'x'>()) % x_tile_size;
				const index_t x_len_remainder = remainder == 0 ? x_tile_size : remainder;
				const index_t x_len = X == X_len - 1 ? x_len_remainder : x_tile_size;

				real_t c_tmp = a_s;

				{
					const real_t r = 1 / (b1_s + a_s);

					for (index_t x = 0; x < x_len; x++)
					{
						d.template at<dim, 'x'>(0, x) *= r;
					}

					c_tmp = a_s * r;
				}

				for (index_t i = 1; i < n; i++)
				{
					const real_t b_tmp = b1_s + (i == n - 1 ? a_s : 0);
					const real_t r = 1 / (b_tmp - a_s * c_tmp);

					for (index_t x = 0; x < x_len; x++)
					{
						d.template at<dim, 'x'>(i, x) =
							r * (d.template at<dim, 'x'>(i, x) - a_s * d.template at<dim, 'x'>(i - 1, x));
					}

					c_tmp = a_s * r;
				}

				for (index_t i = n - 2; i >= 0; i--)
				{
					const auto back_c = c.template at<'i'>(i);

					for (index_t x = 0; x < x_len; x++)
						d.template at<dim, 'x'>(i, x) -= back_c * d.template at<dim, 'x'>(i + 1, x);
				}
			}
		}
	}
}

template <int x_tile_multiple, typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_y_3d_intrinsics(real_t* __restrict__ densities, const real_t* __restrict__ a,
										const real_t* __restrict__ b1, const real_t* __restrict__ back_c,
										const density_layout_t dens_l, const diagonal_layout_t diag_l,
										const index_t s_begin, const index_t s_end)
{
	constexpr char dim = 'y';
	const index_t z_len = dens_l | noarr::get_length<'z'>();
	const index_t n = dens_l | noarr::get_length<dim>();

	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	auto blocked_dens_l = dens_l ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(simd_length * x_tile_multiple);
	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

#pragma omp for schedule(static) nowait collapse(3)
	for (index_t s = s_begin; s < s_end; s++)
	{
		for (index_t z = 0; z < z_len; z++)
		{
			for (index_t X = 0; X < X_len; X++)
			{
				const real_t a_s = a[s];
				const real_t b1_s = b1[s];
				const auto d =
					noarr::make_bag(blocked_dens_l ^ noarr::fix<'s', 'z', 'b', 'X'>(s, z, noarr::lit<0>, X), densities);
				const auto c = noarr::make_bag(diag_l ^ noarr::fix<'s'>(s), back_c);

				real_t a_tmp = 0;
				real_t b_tmp = b1_s + a_s;
				real_t c_tmp = a_s;
				simd_t prev[x_tile_multiple];

				for (index_t x = 0; x < x_tile_multiple; x++)
					prev[x] = hn::Zero(t);

				for (index_t i = 0; i < n; i++)
				{
					const real_t r = 1 / (b_tmp - a_tmp * c_tmp);

					for (index_t x = 0; x < x_tile_multiple; x++)
					{
						simd_t curr = hn::Load(t, &d.template at<dim, 'x'>(i, x * simd_length));
						curr = hn::Mul(hn::MulAdd(hn::Set(t, -a_tmp), prev[x], curr), hn::Set(t, r));
						hn::Store(curr, t, &d.template at<dim, 'x'>(i, x * simd_length));

						prev[x] = curr;
					}

					a_tmp = a_s;
					b_tmp = b1_s + (i == n - 2 ? a_s : 0);
					c_tmp = a_s * r;
				}

				for (index_t i = n - 2; i >= 0; i--)
				{
					for (index_t x = 0; x < x_tile_multiple; x++)
					{
						simd_t curr = hn::Load(t, &d.template at<dim, 'x'>(i, x * simd_length));
						curr = hn::MulAdd(hn::Set(t, -c.template at<'i'>(i)), prev[x], curr);
						hn::Store(curr, t, &d.template at<dim, 'x'>(i, x * simd_length));

						prev[x] = curr;
					}
				}
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_y_3d_intrinsics_dispatch(real_t* __restrict__ densities, const real_t* __restrict__ a,
												 const real_t* __restrict__ b1, const real_t* __restrict__ back_c,
												 const density_layout_t dens_l, const diagonal_layout_t diag_l,
												 const index_t s_begin, const index_t s_end, const index_t x_tile_size)
{
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);

	const index_t x_tile_multiple = x_tile_size / simd_length;

	if (x_tile_multiple == 1)
	{
		solve_slice_y_3d_intrinsics<1>(densities, a, b1, back_c, dens_l, diag_l, s_begin, s_end);
	}
	else if (x_tile_multiple == 2)
	{
		solve_slice_y_3d_intrinsics<2>(densities, a, b1, back_c, dens_l, diag_l, s_begin, s_end);
	}
	else if (x_tile_multiple == 3)
	{
		solve_slice_y_3d_intrinsics<3>(densities, a, b1, back_c, dens_l, diag_l, s_begin, s_end);
	}
	else if (x_tile_multiple == 4)
	{
		solve_slice_y_3d_intrinsics<4>(densities, a, b1, back_c, dens_l, diag_l, s_begin, s_end);
	}
	else
	{
		throw std::runtime_error("Unsupported x_tile_size for intrinsics");
	}
}


template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_xy_fused(real_t* __restrict__ densities, const real_t* __restrict__ ax,
								 const real_t* __restrict__ b1x, const real_t* __restrict__ back_cx,
								 const real_t* __restrict__ ay, const real_t* __restrict__ b1y,
								 const real_t* __restrict__ back_cy, const density_layout_t dens_l,
								 const diagonal_layout_t diagx_l, const diagonal_layout_t diagy_l,
								 const index_t s_begin, const index_t s_end)
{
	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();
	const index_t z_len = dens_l | noarr::get_length<'z'>();

#pragma omp for schedule(static) collapse(2) nowait
	for (index_t s = s_begin; s < s_end; s++)
	{
		for (index_t z = 0; z < z_len; z++)
		{
			const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s', 'z'>(s, z), densities);
			const auto c = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), back_cx);
			const auto cy = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s), back_cy);

			const real_t a_s = ax[s];
			const real_t b1_s = b1x[s];

			const real_t ay_s = ay[s];
			const real_t b1y_s = b1y[s];

			real_t c_tmp_y = ay_s;

			real_t r_y;

			for (index_t y = 0; y < y_len; y++)
			{
				real_t a_tmp = 0;
				real_t b_tmp = b1_s + a_s;
				real_t c_tmp = a_s;
				real_t prev = 0;

				for (index_t i = 0; i < x_len; i++)
				{
					const real_t r = 1 / (b_tmp - a_tmp * c_tmp);

					real_t curr = d.template at<'x', 'y'>(i, y);
					curr = r * (curr - a_tmp * prev);
					d.template at<'x', 'y'>(i, y) = curr;

					a_tmp = a_s;
					b_tmp = b1_s + (i == x_len - 2 ? a_s : 0);
					c_tmp = a_s * r;
					prev = curr;
				}

				for (index_t i = x_len - 2; i >= 0; i--)
				{
					real_t curr = d.template at<'x', 'y'>(i, y);
					curr = curr - c.template at<'i'>(i) * prev;
					d.template at<'x', 'y'>(i, y) = curr;

					real_t prev_y = prev;
					prev = curr;

					if (y == 0)
					{
						r_y = 1 / (b1y_s + ay_s);

						prev_y *= r_y;

						d.template at<'x', 'y'>(i + 1, 0) = prev_y;
					}
					else
					{
						const real_t b_tmp = b1y_s + (y == y_len - 1 ? ay_s : 0);
						r_y = 1 / (b_tmp - ay_s * c_tmp_y);

						prev_y = r_y * (prev_y - ay_s * d.template at<'x', 'y'>(i + 1, y - 1));
						d.template at<'x', 'y'>(i + 1, y) = prev_y;
					}
				}

				if (y == 0)
				{
					r_y = 1 / (b1y_s + ay_s);

					prev *= r_y;

					d.template at<'x', 'y'>(0, 0) = prev;
				}
				else
				{
					const real_t b_tmp = b1y_s + (y == y_len - 1 ? ay_s : 0);
					r_y = 1 / (b_tmp - ay_s * c_tmp_y);

					prev = r_y * (prev - ay_s * d.template at<'x', 'y'>(0, y - 1));
					d.template at<'x', 'y'>(0, y) = prev;
				}

				c_tmp_y = ay_s * r_y;
			}

			for (index_t i = y_len - 2; i >= 0; i--)
			{
				const auto back_c = cy.template at<'i'>(i);

				for (index_t x = 0; x < x_len; x++)
					d.template at<'y', 'x'>(i, x) -= back_c * d.template at<'y', 'x'>(i + 1, x);
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_xy_fused_transpose(real_t* __restrict__ densities, const real_t* __restrict__ ax,
										   const real_t* __restrict__ b1x, const real_t* __restrict__ back_cx,
										   const real_t* __restrict__ ay, const real_t* __restrict__ b1y,
										   const real_t* __restrict__ back_cy, const density_layout_t dens_l,
										   const diagonal_layout_t diagx_l, const diagonal_layout_t diagy_l,
										   const index_t s_begin, const index_t s_end)
{
	const index_t n = dens_l | noarr::get_length<'x'>();

	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	const index_t z_len = dens_l | noarr::get_length<'z'>();
	const index_t y_len_full = dens_l | noarr::get_length<'y'>();

	auto blocked_dens_l = dens_l ^ noarr::into_blocks_static<'y', 'b', 'y', 'v'>(simd_length);

#pragma omp for schedule(static) nowait collapse(2)
	for (index_t s = s_begin; s < s_end; s++)
	{
		for (index_t z = 0; z < z_len; z++)
		{
			const real_t a_s = ax[s];
			const real_t b1_s = b1x[s];

			const real_t ay_s = ay[s];
			const real_t b1y_s = b1y[s];

			real_t ay_tmp = 0;
			real_t by_tmp = b1y_s + ay_s;
			real_t cy_tmp = ay_s;

			// vectorized body
			{
				const index_t full_n = (n + simd_length - 1) / simd_length * simd_length;

				auto body_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<0>);
				const index_t y_len = body_dens_l | noarr::get_length<'y'>();

				for (index_t y = 0; y < y_len; y++)
				{
					real_t a_tmp = 0;
					real_t b_tmp = b1_s + a_s;
					real_t c_tmp = a_s;
					simd_t prev = hn::Zero(t);
					// vector registers that hold the to be transposed x*yz plane
					simd_t* rows = new simd_t[simd_length + 1];

					// forward substitution until last simd_length elements
					for (index_t i = 0; i < full_n - simd_length; i += simd_length)
					{
						rows[0] = prev;

						// aligned loads
						for (index_t v = 0; v < simd_length; v++)
							rows[v + 1] = hn::Load(
								t, &(body_dens_l | noarr::get_at<'z', 'y', 'v', 'x', 's'>(densities, z, y, v, i, s)));

						// transposition to enable vectorization
						transpose(rows + 1);

						// actual forward substitution (vectorized)
						{
							for (index_t v = 1; v < simd_length + 1; v++)
							{
								const real_t r = 1 / (b_tmp - a_tmp * c_tmp);

								rows[v] = hn::Mul(hn::MulAdd(rows[v - 1], hn::Set(t, -a_tmp), rows[v]), hn::Set(t, r));

								a_tmp = a_s;
								b_tmp = b1_s;
								c_tmp = a_s * r;
							};

							prev = rows[simd_length];
						}

						// transposition back to the original form
						transpose(rows + 1);

						// aligned stores
						for (index_t v = 0; v < simd_length; v++)
							hn::Store(
								rows[v + 1], t,
								&(body_dens_l | noarr::get_at<'z', 'y', 'v', 'x', 's'>(densities, z, y, v, i, s)));
					}

					// we are aligned to the vector size, so we can safely continue
					// here we fuse the end of forward substitution and the beginning of backwards propagation
					{
						rows[0] = prev;

						// aligned loads
						for (index_t v = 0; v < simd_length; v++)
							rows[v + 1] = hn::Load(t, &(body_dens_l
														| noarr::get_at<'z', 'y', 'v', 'x', 's'>(
															densities, z, y, v, full_n - simd_length, s)));

						// transposition to enable vectorization
						transpose(rows + 1);

						index_t remainder_work = n % simd_length;
						remainder_work += remainder_work == 0 ? simd_length : 0;

						// the rest of forward part
						{
							if (remainder_work == 1)
								b_tmp = b1_s + a_s;

							for (index_t v = 0; v < remainder_work; v++)
							{
								const real_t r = 1 / (b_tmp - a_tmp * c_tmp);

								rows[v + 1] =
									hn::Mul(hn::MulAdd(rows[v], hn::Set(t, -a_tmp), rows[v + 1]), hn::Set(t, r));

								a_tmp = a_s;
								b_tmp = b1_s + (v == remainder_work - 2 ? a_s : 0);
								c_tmp = a_s * r;
							}
						}

						// the begin of backward part
						{
							auto c =
								hn::Load(t, &(diagx_l | noarr::get_at<'i', 's'>(back_cx, full_n - simd_length, s)));

							for (index_t v = simd_length - 2; v >= 0; v--)
							{
								rows[v + 1] =
									hn::NegMulAdd(rows[v + 2], hn::Set(t, hn::ExtractLane(c, v)), rows[v + 1]);
							}

							prev = rows[1];
						}

						// transposition back to the original form
						transpose(rows + 1);

						{
							real_t ay_bl_tmp = ay_tmp;
							real_t by_bl_tmp = by_tmp;
							real_t cy_bl_tmp = cy_tmp;

							if (y == 0)
								rows[0] = hn::Zero(t);
							else
								rows[0] =
									hn::Load(t, &(body_dens_l
												  | noarr::get_at<'z', 'y', 'v', 'x', 's'>(
													  densities, z, y - 1, simd_length - 1, full_n - simd_length, s)));


							for (index_t v = 0; v < simd_length; v++)
							{
								const real_t r = 1 / (by_bl_tmp - ay_bl_tmp * cy_bl_tmp);

								rows[v + 1] =
									hn::Mul(hn::MulAdd(hn::Set(t, -ay_bl_tmp), rows[v], rows[v + 1]), hn::Set(t, r));

								ay_bl_tmp = ay_s;
								by_bl_tmp = b1y_s + (y * simd_length + v == y_len_full - 2 ? ay_s : 0);
								cy_bl_tmp = ay_s * r;
							}
						}

						// aligned stores
						for (index_t v = 0; v < simd_length; v++)
							hn::Store(rows[v + 1], t,
									  &(body_dens_l
										| noarr::get_at<'z', 'y', 'v', 'x', 's'>(densities, z, y, v,
																				 full_n - simd_length, s)));
					}

					// we continue with backwards substitution
					for (index_t i = full_n - simd_length * 2; i >= 0; i -= simd_length)
					{
						rows[simd_length] = prev;

						// aligned loads
						for (index_t v = 0; v < simd_length; v++)
							rows[v] = hn::Load(
								t, &(body_dens_l | noarr::get_at<'z', 'y', 'v', 'x', 's'>(densities, z, y, v, i, s)));

						// transposition to enable vectorization
						transpose(rows);

						// backward propagation
						{
							auto c = hn::Load(t, &(diagx_l | noarr::get_at<'i', 's'>(back_cx, i, s)));

							for (index_t v = simd_length - 1; v >= 0; v--)
							{
								rows[v] = hn::NegMulAdd(rows[v + 1], hn::Set(t, hn::ExtractLane(c, v)), rows[v]);
							}

							prev = rows[0];
						}

						// transposition back to the original form
						transpose(rows);

						{
							real_t ay_bl_tmp = ay_tmp;
							real_t by_bl_tmp = by_tmp;
							real_t cy_bl_tmp = cy_tmp;

							simd_t prev_y;

							if (y == 0)
								prev_y = hn::Zero(t);
							else
								prev_y = hn::Load(t, &(body_dens_l
													   | noarr::get_at<'z', 'y', 'v', 'x', 's'>(
														   densities, z, y - 1, simd_length - 1, i, s)));

							for (index_t v = 0; v < simd_length; v++)
							{
								const real_t r = 1 / (by_bl_tmp - ay_bl_tmp * cy_bl_tmp);

								rows[v] = hn::Mul(hn::MulAdd(hn::Set(t, -ay_bl_tmp), prev_y, rows[v]), hn::Set(t, r));
								prev_y = rows[v];

								ay_bl_tmp = ay_s;
								by_bl_tmp = b1y_s + (y * simd_length + v == y_len_full - 2 ? ay_s : 0);
								cy_bl_tmp = ay_s * r;
							}
						}

						// aligned stores
						for (index_t v = 0; v < simd_length; v++)
							hn::Store(
								rows[v], t,
								&(body_dens_l | noarr::get_at<'z', 'y', 'v', 'x', 's'>(densities, z, y, v, i, s)));
					}

					for (index_t v = 0; v < simd_length; v++)
					{
						const real_t r = 1 / (by_tmp - ay_tmp * cy_tmp);

						ay_tmp = ay_s;
						by_tmp = b1y_s + (y * simd_length + v == y_len_full - 2 ? ay_s : 0);
						cy_tmp = ay_s * r;
					}

					delete[] rows;
				}
			}

			const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s', 'z'>(s, z), densities);

			// yz remainder
			{
				auto rem_dens_l = blocked_dens_l ^ noarr::fix<'b'>(noarr::lit<1>);
				const index_t v_len = rem_dens_l | noarr::get_length<'v'>();
				const auto c = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), back_cx);

				real_t r_y;


				for (index_t y = y_len_full - v_len; y < y_len_full; y++)
				{
					real_t a_tmp = 0;
					real_t b_tmp = b1_s + a_s;
					real_t c_tmp = a_s;
					real_t prev = 0;

					for (index_t i = 0; i < n; i++)
					{
						const real_t r = 1 / (b_tmp - a_tmp * c_tmp);

						real_t curr = d.template at<'y', 'x'>(y, i);
						curr = r * (curr - a_tmp * prev);
						d.template at<'y', 'x'>(y, i) = curr;

						a_tmp = a_s;
						b_tmp = b1_s + (i == n - 2 ? a_s : 0);
						c_tmp = a_s * r;
						prev = curr;
					}

					for (index_t i = n - 2; i >= 0; i--)
					{
						real_t curr = d.template at<'y', 'x'>(y, i);
						curr = curr - c.template at<'i'>(i) * prev;
						d.template at<'y', 'x'>(y, i) = curr;

						real_t prev_y = prev;
						prev = curr;

						if (y == 0)
						{
							r_y = 1 / (b1y_s + ay_s);

							prev_y *= r_y;

							d.template at<'x', 'y'>(i + 1, 0) = prev_y;
						}
						else
						{
							const real_t b_tmp = b1y_s + (y == y_len_full - 1 ? ay_s : 0);
							r_y = 1 / (b_tmp - ay_s * cy_tmp);

							prev_y = r_y * (prev_y - ay_s * d.template at<'x', 'y'>(i + 1, y - 1));
							d.template at<'x', 'y'>(i + 1, y) = prev_y;
						}
					}

					if (y == 0)
					{
						r_y = 1 / (b1y_s + ay_s);

						prev *= r_y;

						d.template at<'x', 'y'>(0, 0) = prev;
					}
					else
					{
						const real_t b_tmp = b1y_s + (y == y_len_full - 1 ? ay_s : 0);
						r_y = 1 / (b_tmp - ay_s * cy_tmp);

						prev = r_y * (prev - ay_s * d.template at<'x', 'y'>(0, y - 1));
						d.template at<'x', 'y'>(0, y) = prev;
					}

					cy_tmp = ay_s * r_y;
				}
			}

			{
				auto blocked_dens_l = dens_l ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(simd_length);
				const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

				const auto c = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s), back_cy);

				for (index_t X = 0; X < X_len; X++)
				{
					const auto d = noarr::make_bag(
						blocked_dens_l ^ noarr::fix<'s', 'z', 'b', 'X'>(s, z, noarr::lit<0>, X), densities);

					simd_t prev = hn::Load(t, &d.template at<'y', 'x'>(y_len_full - 1, 0));

					for (index_t i = y_len_full - 2; i >= 0; i--)
					{
						simd_t curr = hn::Load(t, &d.template at<'y', 'x'>(i, 0));
						curr = hn::MulAdd(hn::Set(t, -c.template at<'i'>(i)), prev, curr);
						hn::Store(curr, t, &d.template at<'y', 'x'>(i, 0));

						prev = curr;
					}
				}
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_z_3d(real_t* __restrict__ densities, const real_t* __restrict__ a,
							 const real_t* __restrict__ b1, const real_t* __restrict__ back_c,
							 const density_layout_t dens_l, const diagonal_layout_t diag_l, std::size_t x_tile_size,
							 const index_t s_begin, const index_t s_end)
{
	constexpr char dim = 'z';
	const index_t y_len = dens_l | noarr::get_length<'y'>();
	const index_t n = dens_l | noarr::get_length<dim>();

	auto blocked_dens_l = dens_l ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(x_tile_size);
	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

#pragma omp for schedule(static) nowait collapse(3)
	for (index_t s = s_begin; s < s_end; s++)
	{
		for (index_t y = 0; y < y_len; y++)
		{
			for (index_t X = 0; X < X_len; X++)
			{
				const real_t a_s = a[s];
				const real_t b1_s = b1[s];
				const auto d =
					noarr::make_bag(blocked_dens_l ^ noarr::fix<'s', 'y', 'b', 'X'>(s, y, noarr::lit<0>, X), densities);
				const auto c = noarr::make_bag(diag_l ^ noarr::fix<'s'>(s), back_c);

				const index_t remainder = (dens_l | noarr::get_length<'x'>()) % x_tile_size;
				const index_t x_len_remainder = remainder == 0 ? x_tile_size : remainder;
				const index_t x_len = X == X_len - 1 ? x_len_remainder : x_tile_size;

				real_t c_tmp = a_s;

				{
					const real_t r = 1 / (b1_s + a_s);

					for (index_t x = 0; x < x_len; x++)
					{
						d.template at<dim, 'x'>(0, x) *= r;
					}

					c_tmp = a_s * r;
				}

				for (index_t i = 1; i < n; i++)
				{
					const real_t b_tmp = b1_s + (i == n - 1 ? a_s : 0);
					const real_t r = 1 / (b_tmp - a_s * c_tmp);

					for (index_t x = 0; x < x_len; x++)
					{
						d.template at<dim, 'x'>(i, x) =
							r * (d.template at<dim, 'x'>(i, x) - a_s * d.template at<dim, 'x'>(i - 1, x));
					}

					c_tmp = a_s * r;
				}

				for (index_t i = n - 2; i >= 0; i--)
				{
					const auto back_c = c.template at<'i'>(i);

					for (index_t x = 0; x < x_len; x++)
						d.template at<dim, 'x'>(i, x) -= back_c * d.template at<dim, 'x'>(i + 1, x);
				}
			}
		}
	}
}

template <int x_tile_multiple, typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_z_3d_intrinsics(real_t* __restrict__ densities, const real_t* __restrict__ a,
										const real_t* __restrict__ b1, const real_t* __restrict__ back_c,
										const density_layout_t dens_l, const diagonal_layout_t diag_l,
										const index_t s_begin, const index_t s_end)
{
	constexpr char dim = 'z';
	const index_t y_len = dens_l | noarr::get_length<'y'>();
	const index_t n = dens_l | noarr::get_length<dim>();

	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);
	using simd_t = hn::Vec<simd_tag>;

	auto blocked_dens_l = dens_l ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(simd_length * x_tile_multiple);
	const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

#pragma omp for schedule(static) nowait collapse(3)
	for (index_t s = s_begin; s < s_end; s++)
	{
		for (index_t y = 0; y < y_len; y++)
		{
			for (index_t X = 0; X < X_len; X++)
			{
				const real_t a_s = a[s];
				const real_t b1_s = b1[s];
				const auto d =
					noarr::make_bag(blocked_dens_l ^ noarr::fix<'s', 'y', 'b', 'X'>(s, y, noarr::lit<0>, X), densities);
				const auto c = noarr::make_bag(diag_l ^ noarr::fix<'s'>(s), back_c);

				real_t a_tmp = 0;
				real_t b_tmp = b1_s + a_s;
				real_t c_tmp = a_s;
				simd_t prev[x_tile_multiple];

				for (index_t x = 0; x < x_tile_multiple; x++)
					prev[x] = hn::Zero(t);

				for (index_t i = 0; i < n; i++)
				{
					const real_t r = 1 / (b_tmp - a_tmp * c_tmp);

					for (index_t x = 0; x < x_tile_multiple; x++)
					{
						simd_t curr = hn::Load(t, &d.template at<dim, 'x'>(i, x * simd_length));
						curr = hn::Mul(hn::MulAdd(hn::Set(t, -a_tmp), prev[x], curr), hn::Set(t, r));
						hn::Store(curr, t, &d.template at<dim, 'x'>(i, x * simd_length));

						prev[x] = curr;
					}

					a_tmp = a_s;
					b_tmp = b1_s + (i == n - 2 ? a_s : 0);
					c_tmp = a_s * r;
				}

				for (index_t i = n - 2; i >= 0; i--)
				{
					for (index_t x = 0; x < x_tile_multiple; x++)
					{
						simd_t curr = hn::Load(t, &d.template at<dim, 'x'>(i, x * simd_length));
						curr = hn::MulAdd(hn::Set(t, -c.template at<'i'>(i)), prev[x], curr);
						hn::Store(curr, t, &d.template at<dim, 'x'>(i, x * simd_length));

						prev[x] = curr;
					}
				}
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_z_3d_intrinsics_dispatch(real_t* __restrict__ densities, const real_t* __restrict__ a,
												 const real_t* __restrict__ b1, const real_t* __restrict__ back_c,
												 const density_layout_t dens_l, const diagonal_layout_t diag_l,
												 const index_t s_begin, const index_t s_end, const index_t x_tile_size)
{
	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag t;
	HWY_LANES_CONSTEXPR index_t simd_length = hn::Lanes(t);

	const index_t x_tile_multiple = x_tile_size / simd_length;

	if (x_tile_multiple == 1)
	{
		solve_slice_z_3d_intrinsics<1>(densities, a, b1, back_c, dens_l, diag_l, s_begin, s_end);
	}
	else if (x_tile_multiple == 2)
	{
		solve_slice_z_3d_intrinsics<2>(densities, a, b1, back_c, dens_l, diag_l, s_begin, s_end);
	}
	else if (x_tile_multiple == 3)
	{
		solve_slice_z_3d_intrinsics<3>(densities, a, b1, back_c, dens_l, diag_l, s_begin, s_end);
	}
	else if (x_tile_multiple == 4)
	{
		solve_slice_z_3d_intrinsics<4>(densities, a, b1, back_c, dens_l, diag_l, s_begin, s_end);
	}
	else
	{
		throw std::runtime_error("Unsupported x_tile_size for intrinsics");
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_t<real_t, aligned_x>::solve_x()
{
	if (use_intrinsics_)
	{
		if (use_fused_ && this->problem_.dims > 1)
		{
#pragma omp parallel
			solve_slice_xy_fused_transpose<index_t>(
				this->substrates_, ax_, b1x_, cx_, ay_, b1y_, cy_, get_substrates_layout<3>(),
				get_diagonal_layout(this->problem_, this->problem_.nx),
				get_diagonal_layout(this->problem_, this->problem_.ny), 0, this->problem_.substrates_count);
		}
		else
		{
			if (this->problem_.dims == 1)
			{
#pragma omp parallel
				solve_slice_x_1d<index_t>(this->substrates_, ax_, b1x_, cx_, get_substrates_layout<1>(),
										  get_diagonal_layout(this->problem_, this->problem_.nx), 0,
										  this->problem_.substrates_count);
			}
			else if (this->problem_.dims == 2)
			{
#pragma omp parallel
				solve_slice_x_2d_and_3d_transpose<index_t>(
					this->substrates_, ax_, b1x_, cx_, get_substrates_layout<2>() ^ noarr::rename<'y', 'm'>(),
					get_diagonal_layout(this->problem_, this->problem_.nx), 0, this->problem_.substrates_count);
			}
			else if (this->problem_.dims == 3)
			{
#pragma omp parallel
				solve_slice_x_2d_and_3d_transpose<index_t>(
					this->substrates_, ax_, b1x_, cx_,
					get_substrates_layout<3>() ^ noarr::merge_blocks<'z', 'y', 'm'>(),
					get_diagonal_layout(this->problem_, this->problem_.nx), 0, this->problem_.substrates_count);
			}
		}
	}
	else
	{
		if (use_fused_ && this->problem_.dims > 1)
		{
#pragma omp parallel
			solve_slice_xy_fused<index_t>(this->substrates_, ax_, b1x_, cx_, ay_, b1y_, cy_, get_substrates_layout<3>(),
										  get_diagonal_layout(this->problem_, this->problem_.nx),
										  get_diagonal_layout(this->problem_, this->problem_.ny), 0,
										  this->problem_.substrates_count);
		}
		else
		{
			if (this->problem_.dims == 1)
			{
#pragma omp parallel
				solve_slice_x_1d<index_t>(this->substrates_, ax_, b1x_, cx_, get_substrates_layout<1>(),
										  get_diagonal_layout(this->problem_, this->problem_.nx), 0,
										  this->problem_.substrates_count);
			}
			else if (this->problem_.dims == 2)
			{
#pragma omp parallel
				solve_slice_x_2d_and_3d<index_t>(
					this->substrates_, ax_, b1x_, cx_, get_substrates_layout<2>() ^ noarr::rename<'y', 'm'>(),
					get_diagonal_layout(this->problem_, this->problem_.nx), 0, this->problem_.substrates_count);
			}
			else if (this->problem_.dims == 3)
			{
#pragma omp parallel
				solve_slice_x_2d_and_3d<index_t>(this->substrates_, ax_, b1x_, cx_,
												 get_substrates_layout<3>() ^ noarr::merge_blocks<'z', 'y', 'm'>(),
												 get_diagonal_layout(this->problem_, this->problem_.nx), 0,
												 this->problem_.substrates_count);
			}
		}
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_t<real_t, aligned_x>::solve_y()
{
	if (use_fused_)
		return;

	if (use_intrinsics_)
	{
#pragma omp parallel
		solve_slice_y_3d_intrinsics_dispatch<index_t>(this->substrates_, ay_, b1y_, cy_, get_substrates_layout<3>(),
													  get_diagonal_layout(this->problem_, this->problem_.ny), 0,
													  this->problem_.substrates_count, x_tile_size_);
	}
	else
	{
		if (this->problem_.dims == 2)
		{
#pragma omp parallel
			solve_slice_y_2d<index_t>(this->substrates_, ay_, b1y_, cy_, get_substrates_layout<2>(),
									  get_diagonal_layout(this->problem_, this->problem_.ny), x_tile_size_, 0,
									  this->problem_.substrates_count);
		}
		else if (this->problem_.dims == 3)
		{
#pragma omp parallel
			solve_slice_y_3d<index_t>(this->substrates_, ay_, b1y_, cy_, get_substrates_layout<3>(),
									  get_diagonal_layout(this->problem_, this->problem_.ny), x_tile_size_, 0,
									  this->problem_.substrates_count);
		}
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_t<real_t, aligned_x>::solve_z()
{
	if (use_intrinsics_)
	{
#pragma omp parallel
		solve_slice_z_3d_intrinsics_dispatch<index_t>(this->substrates_, az_, b1z_, cz_, get_substrates_layout<3>(),
													  get_diagonal_layout(this->problem_, this->problem_.nz), 0,
													  this->problem_.substrates_count, x_tile_size_);
	}
	else
	{
#pragma omp parallel
		solve_slice_z_3d<index_t>(this->substrates_, az_, b1z_, cz_, get_substrates_layout<3>(),
								  get_diagonal_layout(this->problem_, this->problem_.nz), x_tile_size_, 0,
								  this->problem_.substrates_count);
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_t<real_t, aligned_x>::solve()
{
	if (this->problem_.dims == 1)
	{
		solve_1d();
	}
	else if (this->problem_.dims == 2)
	{
		solve_2d();
	}
	else if (this->problem_.dims == 3)
	{
		if (use_fused_)
			solve_3d_fused();
		else
			solve_3d();
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_t<real_t, aligned_x>::solve_1d()
{
	if (!use_intrinsics_)
	{
#pragma omp parallel
		{
			perf_counter counter("lstmdta");

			for (index_t i = 0; i < this->problem_.iterations; i++)
				solve_slice_x_1d<index_t>(this->substrates_, ax_, b1x_, cx_, get_substrates_layout<1>(),
										  get_diagonal_layout(this->problem_, this->problem_.nx), 0,
										  this->problem_.substrates_count);
		}
	}
	else
	{
#pragma omp parallel
		{
			perf_counter counter("lstmdtai");

			for (index_t i = 0; i < this->problem_.iterations; i++)
				solve_slice_x_1d<index_t>(this->substrates_, ax_, b1x_, cx_, get_substrates_layout<1>(),
										  get_diagonal_layout(this->problem_, this->problem_.nx), 0,
										  this->problem_.substrates_count);
		}
	}
}


template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_t<real_t, aligned_x>::solve_2d()
{
	if (!use_intrinsics_)
	{
#pragma omp parallel
		{
			perf_counter counter("lstmdta");

			for (index_t s = 0; s < this->problem_.substrates_count; s += substrate_step_)
			{
				for (index_t i = 0; i < this->problem_.iterations; i++)
				{
					auto s_step_length = std::min(substrate_step_, this->problem_.substrates_count - s);

					solve_slice_x_2d_and_3d<index_t>(
						this->substrates_, ax_, b1x_, cx_, get_substrates_layout<2>() ^ noarr::rename<'y', 'm'>(),
						get_diagonal_layout(this->problem_, this->problem_.nx), s, s + s_step_length);
#pragma omp barrier
					solve_slice_y_2d<index_t>(this->substrates_, ay_, b1y_, cy_, get_substrates_layout<2>(),
											  get_diagonal_layout(this->problem_, this->problem_.ny), x_tile_size_, s,
											  s + s_step_length);
#pragma omp barrier
				}
			}
		}
	}
	else
	{
#pragma omp parallel
		{
			perf_counter counter("lstmdtai");

			for (index_t s = 0; s < this->problem_.substrates_count; s += substrate_step_)
			{
				for (index_t i = 0; i < this->problem_.iterations; i++)
				{
					auto s_step_length = std::min(substrate_step_, this->problem_.substrates_count - s);

					solve_slice_x_2d_and_3d_transpose<index_t>(
						this->substrates_, ax_, b1x_, cx_, get_substrates_layout<2>() ^ noarr::rename<'y', 'm'>(),
						get_diagonal_layout(this->problem_, this->problem_.nx), s, s + s_step_length);
#pragma omp barrier
					solve_slice_y_2d_intrinsics<index_t>(this->substrates_, ay_, b1y_, cy_, get_substrates_layout<2>(),
														 get_diagonal_layout(this->problem_, this->problem_.ny), s,
														 s + s_step_length);
#pragma omp barrier
				}
			}
		}
	}
}


template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_t<real_t, aligned_x>::solve_3d()
{
	if (!use_intrinsics_)
	{
#pragma omp parallel
		{
			perf_counter counter("lstmdta");

			for (index_t s = 0; s < this->problem_.substrates_count; s += substrate_step_)
			{
				for (index_t i = 0; i < this->problem_.iterations; i++)
				{
					auto s_step_length = std::min(substrate_step_, this->problem_.substrates_count - s);

					solve_slice_x_2d_and_3d<index_t>(this->substrates_, ax_, b1x_, cx_,
													 get_substrates_layout<3>() ^ noarr::merge_blocks<'z', 'y', 'm'>(),
													 get_diagonal_layout(this->problem_, this->problem_.nx), s,
													 s + s_step_length);
#pragma omp barrier
					solve_slice_y_3d<index_t>(this->substrates_, ay_, b1y_, cy_, get_substrates_layout<3>(),
											  get_diagonal_layout(this->problem_, this->problem_.ny), x_tile_size_, s,
											  s + s_step_length);
#pragma omp barrier
					solve_slice_z_3d<index_t>(this->substrates_, az_, b1z_, cz_, get_substrates_layout<3>(),
											  get_diagonal_layout(this->problem_, this->problem_.nz), x_tile_size_, s,
											  s + s_step_length);
#pragma omp barrier
				}
			}
		}
	}
	else
	{
#pragma omp parallel
		{
			perf_counter counter("lstmdtai");

			for (index_t s = 0; s < this->problem_.substrates_count; s += substrate_step_)
			{
				for (index_t i = 0; i < this->problem_.iterations; i++)
				{
					auto s_step_length = std::min(substrate_step_, this->problem_.substrates_count - s);

					solve_slice_x_2d_and_3d_transpose<index_t>(
						this->substrates_, ax_, b1x_, cx_,
						get_substrates_layout<3>() ^ noarr::merge_blocks<'z', 'y', 'm'>(),
						get_diagonal_layout(this->problem_, this->problem_.nx), s, s + s_step_length);
#pragma omp barrier
					solve_slice_y_3d_intrinsics_dispatch<index_t>(
						this->substrates_, ay_, b1y_, cy_, get_substrates_layout<3>(),
						get_diagonal_layout(this->problem_, this->problem_.ny), s, s + s_step_length, x_tile_size_);
#pragma omp barrier
					solve_slice_z_3d_intrinsics_dispatch<index_t>(
						this->substrates_, az_, b1z_, cz_, get_substrates_layout<3>(),
						get_diagonal_layout(this->problem_, this->problem_.nz), s, s + s_step_length, x_tile_size_);
#pragma omp barrier
				}
			}
		}
	}
}

template <typename real_t, bool aligned_x>
void least_memory_thomas_solver_d_t<real_t, aligned_x>::solve_3d_fused()
{
	if (!use_intrinsics_)
	{
#pragma omp parallel
		{
			perf_counter counter("lstmdtfa");

			for (index_t s = 0; s < this->problem_.substrates_count; s += substrate_step_)
			{
				for (index_t i = 0; i < this->problem_.iterations; i++)
				{
					auto s_step_length = std::min(substrate_step_, this->problem_.substrates_count - s);

					solve_slice_xy_fused<index_t>(
						this->substrates_, ax_, b1x_, cx_, ay_, b1y_, cy_, get_substrates_layout<3>(),
						get_diagonal_layout(this->problem_, this->problem_.nx),
						get_diagonal_layout(this->problem_, this->problem_.ny), s, s + s_step_length);
#pragma omp barrier
					solve_slice_z_3d<index_t>(this->substrates_, az_, b1z_, cz_, get_substrates_layout<3>(),
											  get_diagonal_layout(this->problem_, this->problem_.nz), x_tile_size_, s,
											  s + s_step_length);
#pragma omp barrier
				}
			}
		}
	}
	else
	{
#pragma omp parallel
		{
			perf_counter counter("lstmdtfai");

			for (index_t s = 0; s < this->problem_.substrates_count; s += substrate_step_)
			{
				for (index_t i = 0; i < this->problem_.iterations; i++)
				{
					auto s_step_length = std::min(substrate_step_, this->problem_.substrates_count - s);

					solve_slice_xy_fused_transpose<index_t>(
						this->substrates_, ax_, b1x_, cx_, ay_, b1y_, cy_, get_substrates_layout<3>(),
						get_diagonal_layout(this->problem_, this->problem_.nx),
						get_diagonal_layout(this->problem_, this->problem_.ny), s, s + s_step_length);
#pragma omp barrier
					solve_slice_z_3d_intrinsics_dispatch<index_t>(
						this->substrates_, az_, b1z_, cz_, get_substrates_layout<3>(),
						get_diagonal_layout(this->problem_, this->problem_.nz), s, s + s_step_length, x_tile_size_);
#pragma omp barrier
				}
			}
		}
	}
}

template <typename real_t, bool aligned_x>
least_memory_thomas_solver_d_t<real_t, aligned_x>::least_memory_thomas_solver_d_t(bool use_intrinsics, bool use_fused)
	: ax_(nullptr),
	  b1x_(nullptr),
	  cx_(nullptr),
	  ay_(nullptr),
	  b1y_(nullptr),
	  cy_(nullptr),
	  az_(nullptr),
	  b1z_(nullptr),
	  cz_(nullptr),
	  use_intrinsics_(use_intrinsics),
	  use_fused_(use_fused)
{}

template <typename real_t, bool aligned_x>
least_memory_thomas_solver_d_t<real_t, aligned_x>::~least_memory_thomas_solver_d_t()
{
	if (cx_)
	{
		std::free(cx_);
		std::free(ax_);
		std::free(b1x_);
	}
	if (cy_)
	{
		std::free(cy_);
		std::free(ay_);
		std::free(b1y_);
	}
	if (cz_)
	{
		std::free(cz_);
		std::free(az_);
		std::free(b1z_);
	}
}

template class least_memory_thomas_solver_d_t<float, false>;
template class least_memory_thomas_solver_d_t<double, false>;

template class least_memory_thomas_solver_d_t<float, true>;
template class least_memory_thomas_solver_d_t<double, true>;
