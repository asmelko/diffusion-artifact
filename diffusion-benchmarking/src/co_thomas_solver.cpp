#include "co_thomas_solver.h"

#include <atomic>
#include <cstddef>
#include <iostream>

#include "omp_helper.h"
#include "vector_transpose_helper.h"

template <typename real_t, bool aligned_x>
void co_thomas_solver<real_t, aligned_x>::precompute_values(real_t*& a, real_t*& b1, real_t*& c, index_t shape,
															index_t dims, index_t n)
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
void co_thomas_solver<real_t, aligned_x>::prepare(const max_problem_t& problem)
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

int t = 0;
int division_size = 32;

template <typename real_t, bool aligned_x>
void co_thomas_solver<real_t, aligned_x>::tune(const nlohmann::json& params)
{
	x_tile_size_ = params.contains("x_tile_size") ? (std::size_t)params["x_tile_size"] : 48;
	alignment_size_ = params.contains("alignment_size") ? (std::size_t)params["alignment_size"] : 64;
	substrate_step_ =
		params.contains("substrate_step") ? (index_t)params["substrate_step"] : this->problem_.substrates_count;

	if (use_intrinsics_)
	{
		using simd_tag = hn::ScalableTag<real_t>;
		simd_tag d;
		x_tile_size_ = (x_tile_size_ + hn::Lanes(d) - 1) / hn::Lanes(d) * hn::Lanes(d);
		std::size_t vector_length = hn::Lanes(d) * sizeof(real_t);
		alignment_size_ = std::max(alignment_size_, vector_length * x_tile_size_ / hn::Lanes(d));
	}

	if (params.contains("t"))
		t = (int)params["t"];

	if (params.contains("div"))
		division_size = (int)params["div"];
}

template <typename real_t, bool aligned_x>
void co_thomas_solver<real_t, aligned_x>::initialize()
{
	if (this->problem_.dims >= 1)
		precompute_values(ax_, b1x_, cx_, this->problem_.dx, this->problem_.dims, this->problem_.nx);
	if (this->problem_.dims >= 2)
		precompute_values(ay_, b1y_, cy_, this->problem_.dy, this->problem_.dims, this->problem_.ny);
	if (this->problem_.dims >= 3)
		precompute_values(az_, b1z_, cz_, this->problem_.dz, this->problem_.dims, this->problem_.nz);

	{
		const auto threads = get_max_threads();

		const index_t threads_per_s = threads / this->problem_.substrates_count;

		max_cores_groups_ = threads / threads_per_s;
	}

	auto scratch_layout = get_scratch_layout();

	a_scratch_ = (real_t*)std::malloc((scratch_layout | noarr::get_size()));
	c_scratch_ = (real_t*)std::malloc((scratch_layout | noarr::get_size()));

	countersz_count_ = max_cores_groups_;
	countersz_ = std::make_unique<aligned_atomic<long>[]>(countersz_count_);
}

template <typename real_t, bool aligned_x>
auto co_thomas_solver<real_t, aligned_x>::get_diagonal_layout(const problem_t<index_t, real_t>& problem, index_t n)
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

template <typename real_t>
struct inside_data
{
	const real_t a_s, b1_s;
	real_t& c_tmp;
};

template <typename index_t, typename real_t, typename density_bag_t>
static real_t z_forward_inside_y(const density_bag_t d, const index_t s, const index_t z, const index_t y,
								 const index_t x, const real_t a_s, const real_t b1_s, real_t c_tmp, real_t data)

{
	const index_t z_len = d | noarr::get_length<'z'>();

	real_t r;

	if (z == 0)
	{
		r = 1 / (b1_s + a_s);

		data *= r;

		d.template at<'s', 'z', 'x', 'y'>(s, z, x, y) = data;
	}
	else
	{
		const real_t b_tmp = b1_s + (z == z_len - 1 ? a_s : 0);
		r = 1 / (b_tmp - a_s * c_tmp);

		data = r * (data - a_s * d.template at<'s', 'z', 'x', 'y'>(s, z - 1, x, y));
		d.template at<'s', 'z', 'x', 'y'>(s, z, x, y) = data;
	}

	return r;
}

template <typename index_t, typename density_bag_t, typename diag_bag_t>
static void z_backward(const density_bag_t d, const diag_bag_t c, const index_t s)

{
	const index_t x_len = d | noarr::get_length<'x'>();
	const index_t y_len = d | noarr::get_length<'y'>();
	const index_t z_len = d | noarr::get_length<'z'>();

	for (index_t i = z_len - 2; i >= 0; i--)
	{
		const auto back_c = c.template at<'s', 'i'>(s, i);

		for (index_t y = 0; y < y_len; y++)
		{
			for (index_t x = 0; x < x_len; x++)
				d.template at<'s', 'z', 'y', 'x'>(s, i, y, x) -=
					back_c * d.template at<'s', 'z', 'y', 'x'>(s, i + 1, y, x);
		}
	}
}

template <typename index_t, typename real_t, typename scratch_bag_t>
struct inside_data_blocked
{
	const index_t begin;
	const real_t a_s, b1_s;
	scratch_bag_t c;
};

template <bool update_c, typename index_t, typename real_t, typename density_bag_t>
static void y_forward_inside_x(const density_bag_t d, const index_t s, const index_t z, const index_t y,
							   const index_t x, const real_t a_s, const real_t b1_s, real_t& c_tmp, real_t data)

{
	const index_t y_len = d | noarr::get_length<'y'>();

	real_t r;

	if (y == 0)
	{
		r = 1 / (b1_s + a_s);

		data *= r;

		d.template at<'s', 'z', 'x', 'y'>(s, z, x, 0) = data;
	}
	else
	{
		const real_t b_tmp = b1_s + (y == y_len - 1 ? a_s : 0);
		r = 1 / (b_tmp - a_s * c_tmp);

		data = r * (data - a_s * d.template at<'s', 'z', 'x', 'y'>(s, z, x, y - 1));
		d.template at<'s', 'z', 'x', 'y'>(s, z, x, y) = data;
	}

	if constexpr (update_c)
	{
		c_tmp = a_s * r;
	}
}

template <typename index_t, typename real_t, typename density_bag_t>
static void x_forward(const density_bag_t d, const index_t s, const index_t z, const index_t y, const real_t a_s,
					  const real_t b1_s, real_t& a_tmp, real_t& b_tmp, real_t& c_tmp, real_t& prev)

{
	const index_t x_len = d | noarr::get_length<'x'>();

	for (index_t i = 0; i < x_len; i++)
	{
		const real_t r = 1 / (b_tmp - a_tmp * c_tmp);

		real_t curr = d.template at<'s', 'z', 'y', 'x'>(s, z, y, i);
		curr = r * (curr - a_tmp * prev);
		d.template at<'s', 'z', 'y', 'x'>(s, z, y, i) = curr;

		a_tmp = a_s;
		b_tmp = b1_s + (i == x_len - 2 ? a_s : 0);
		c_tmp = a_s * r;
		prev = curr;
	}
}

template <typename index_t, typename real_t, typename density_bag_t, typename diag_bag_t>
static void x_backward(const density_bag_t d, const diag_bag_t c, const index_t s, const index_t z, const index_t y,
					   real_t& prev, inside_data<real_t> y_data)

{
	const index_t x_len = d | noarr::get_length<'x'>();

	for (index_t i = x_len - 2; i >= 0; i--)
	{
		real_t curr = d.template at<'s', 'z', 'x', 'y'>(s, z, i, y);
		curr = curr - c.template at<'s', 'i'>(s, i) * prev;
		// d.template at<'s', 'z', 'x', 'y'>(s, z, i, y) = curr;

		y_forward_inside_x<false>(d, s, z, y, i + 1, y_data.a_s, y_data.b1_s, y_data.c_tmp, prev);

		prev = curr;
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_2d_naive(real_t* __restrict__ densities, const real_t* __restrict__ ax,
						   const real_t* __restrict__ b1x, const real_t* __restrict__ back_cx,
						   const real_t* __restrict__ ay, const real_t* __restrict__ b1y,
						   const real_t* __restrict__ back_cy, const density_layout_t dens_l,
						   const diagonal_layout_t diagx_l, const diagonal_layout_t diagy_l)
{
	const index_t s_len = dens_l | noarr::get_length<'s'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();

#pragma omp parallel for
	for (index_t s = 0; s < s_len; s++)
	{
		for (index_t y = 0; y < y_len; y++)
		{
			const real_t a_s = ax[s];
			const real_t b1_s = b1x[s];
			const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s', 'y'>(s, y), densities);
			const auto c = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), back_cx);
			constexpr char dim = 'x';

			real_t a_tmp = 0;
			real_t b_tmp = b1_s + a_s;
			real_t c_tmp = a_s;
			real_t prev = 0;

			for (index_t i = 0; i < x_len; i++)
			{
				const real_t r = 1 / (b_tmp - a_tmp * c_tmp);

				real_t curr = d.template at<dim>(i);
				curr = r * (curr - a_tmp * prev);
				d.template at<dim>(i) = curr;

				a_tmp = a_s;
				b_tmp = b1_s + (i == x_len - 2 ? a_s : 0);
				c_tmp = a_s * r;
				prev = curr;
			}

			for (index_t i = x_len - 2; i >= 0; i--)
			{
				real_t curr = d.template at<dim>(i);
				curr = curr - c.template at<'i'>(i) * prev;
				d.template at<dim>(i) = curr;

				prev = curr;
			}
		}

		{
			const real_t a_s = ay[s];
			const real_t b1_s = b1y[s];
			const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s'>(s), densities);
			const auto c = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s), back_cy);
			constexpr char dim = 'y';

			real_t c_tmp = a_s;

			{
				const real_t r = 1 / (b1_s + a_s);

				for (index_t x = 0; x < x_len; x++)
				{
					d.template at<dim, 'x'>(0, x) *= r;
				}

				c_tmp = a_s * r;
			}

			for (index_t i = 1; i < y_len; i++)
			{
				const real_t b_tmp = b1_s + (i == y_len - 1 ? a_s : 0);
				const real_t r = 1 / (b_tmp - a_s * c_tmp);

				for (index_t x = 0; x < x_len; x++)
				{
					d.template at<dim, 'x'>(i, x) =
						r * (d.template at<dim, 'x'>(i, x) - a_s * d.template at<dim, 'x'>(i - 1, x));
				}

				c_tmp = a_s * r;
			}

			for (index_t i = y_len - 2; i >= 0; i--)
			{
				const auto back_c = c.template at<'i'>(i);

				for (index_t x = 0; x < x_len; x++)
					d.template at<dim, 'x'>(i, x) -= back_c * d.template at<dim, 'x'>(i + 1, x);
			}
		}
	}
}


template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_2d_tile(real_t* __restrict__ densities, const real_t* __restrict__ ax, const real_t* __restrict__ b1x,
						  const real_t* __restrict__ back_cx, const real_t* __restrict__ ay,
						  const real_t* __restrict__ b1y, const real_t* __restrict__ back_cy,
						  const density_layout_t dens_l, const diagonal_layout_t diagx_l,
						  const diagonal_layout_t diagy_l, const index_t x_tile_size)
{
	const index_t s_len = dens_l | noarr::get_length<'s'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();

#pragma omp parallel for
	for (index_t s = 0; s < s_len; s++)
	{
		for (index_t y = 0; y < y_len; y++)
		{
			const real_t a_s = ax[s];
			const real_t b1_s = b1x[s];
			const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s', 'y'>(s, y), densities);
			const auto c = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), back_cx);
			constexpr char dim = 'x';

			real_t a_tmp = 0;
			real_t b_tmp = b1_s + a_s;
			real_t c_tmp = a_s;
			real_t prev = 0;

			for (index_t i = 0; i < x_len; i++)
			{
				const real_t r = 1 / (b_tmp - a_tmp * c_tmp);

				real_t curr = d.template at<dim>(i);
				curr = r * (curr - a_tmp * prev);
				d.template at<dim>(i) = curr;

				a_tmp = a_s;
				b_tmp = b1_s + (i == x_len - 2 ? a_s : 0);
				c_tmp = a_s * r;
				prev = curr;
			}

			for (index_t i = x_len - 2; i >= 0; i--)
			{
				real_t curr = d.template at<dim>(i);
				curr = curr - c.template at<'i'>(i) * prev;
				d.template at<dim>(i) = curr;

				prev = curr;
			}
		}

		auto blocked_dens_l = dens_l ^ noarr::into_blocks_dynamic<'x', 'X', 'x', 'b'>(x_tile_size);
		const index_t X_len = blocked_dens_l | noarr::get_length<'X'>();

		for (index_t X = 0; X < X_len; X++)
		{
			const real_t a_s = ay[s];
			const real_t b1_s = b1y[s];
			const auto d = noarr::make_bag(blocked_dens_l ^ noarr::fix<'s', 'b', 'X'>(s, noarr::lit<0>, X), densities);
			const auto c = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s), back_cy);
			constexpr char dim = 'y';

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

			for (index_t i = 1; i < y_len; i++)
			{
				const real_t b_tmp = b1_s + (i == y_len - 1 ? a_s : 0);
				const real_t r = 1 / (b_tmp - a_s * c_tmp);

				for (index_t x = 0; x < x_len; x++)
				{
					d.template at<dim, 'x'>(i, x) =
						r * (d.template at<dim, 'x'>(i, x) - a_s * d.template at<dim, 'x'>(i - 1, x));
				}

				c_tmp = a_s * r;
			}

			for (index_t i = y_len - 2; i >= 0; i--)
			{
				const auto back_c = c.template at<'i'>(i);

				for (index_t x = 0; x < x_len; x++)
					d.template at<dim, 'x'>(i, x) -= back_c * d.template at<dim, 'x'>(i + 1, x);
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_2d_fused(real_t* __restrict__ densities, const real_t* __restrict__ ax,
						   const real_t* __restrict__ b1x, const real_t* __restrict__ back_cx,
						   const real_t* __restrict__ ay, const real_t* __restrict__ b1y,
						   const real_t* __restrict__ back_cy, const density_layout_t dens_l,
						   const diagonal_layout_t diagx_l, const diagonal_layout_t diagy_l)
{
	const index_t s_len = dens_l | noarr::get_length<'s'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();

	const auto d = noarr::make_bag(dens_l, densities);
	const auto cx = noarr::make_bag(diagx_l, back_cx);
	const auto cy = noarr::make_bag(diagy_l, back_cy);

#pragma omp parallel for
	for (index_t s = 0; s < s_len; s++)
	{
		const real_t ax_s = ax[s];
		const real_t b1x_s = b1x[s];

		const real_t ay_s = ay[s];
		const real_t b1y_s = b1y[s];

		const index_t z = 0;

		{
			real_t c_tmp_y = ay_s;

			for (index_t y = 0; y < y_len; y++)
			{
				real_t a_tmp = 0;
				real_t b_tmp = b1x_s + ax_s;
				real_t c_tmp = ax_s;
				real_t prev = 0;

				x_forward(d, s, z, y, ax_s, b1x_s, a_tmp, b_tmp, c_tmp, prev);

				x_backward(d, cx, s, z, y, prev, inside_data<real_t> { ay_s, b1y_s, c_tmp_y });

				y_forward_inside_x<true>(d, s, z, y, 0, ay_s, b1y_s, c_tmp_y, prev);
			}

			for (index_t i = y_len - 2; i >= 0; i--)
			{
				const auto back_c = cy.template at<'s', 'i'>(s, i);

				for (index_t x = 0; x < x_len; x++)
				{
					d.template at<'s', 'z', 'y', 'x'>(s, z, i, x) -=
						back_c * d.template at<'s', 'z', 'y', 'x'>(s, z, i + 1, x);
				}
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t>
static void xf_part(density_layout_t d, const real_t a_s, const real_t b1_s, index_t& c_tmp, index_t x_begin,
					const index_t x_end, const index_t y)
{
	const index_t x_len = d | noarr::get_length<'x'>();

	if (x_begin == 0)
	{
		const real_t r = 1 / (b1_s + a_s);

		d.template at<'y', 'x'>(y, 0) *= r;

		c_tmp = a_s * r;

		x_begin++;
	}

	for (index_t i = x_begin; i < x_end; i++)
	{
		const real_t b_tmp = b1_s + (i == x_len - 1 ? a_s : 0);
		const real_t r = 1 / (b_tmp - a_s * c_tmp);

		d.template at<'y', 'x'>(y, i) = r * (d.template at<'y', 'x'>(y, i) - a_s * d.template at<'y', 'x'>(y, i - 1));

		c_tmp = a_s * r;
	}
}

template <typename index_t, typename density_layout_t, typename diagonal_layout_t>
static void xb_part(density_layout_t d, diagonal_layout_t cx, index_t x_begin, const index_t x_end, const index_t y)
{
	const index_t x_len = d | noarr::get_length<'x'>();

	for (index_t i = std::min(x_end - 1, x_len - 2); i >= x_begin; i--)
	{
		const auto back_c = cx.template at<'i'>(i);

		d.template at<'y', 'x'>(y, i) -= back_c * d.template at<'y', 'x'>(y, i + 1);
	}
}

template <typename index_t, typename real_t, typename density_layout_t>
static void yf_part(density_layout_t d, const real_t a_s, const real_t b1_s, index_t& c_tmp, index_t x_begin,
					const index_t x_end, index_t y_begin, const index_t y_end)
{
	const index_t y_len = d | noarr::get_length<'y'>();

	constexpr char dim = 'y';

	if (y_begin == 0)
	{
		const real_t r = 1 / (b1_s + a_s);

		for (index_t x = x_begin; x < x_end; x++)
		{
			d.template at<dim, 'x'>(0, x) *= r;
		}

		c_tmp = a_s * r;

		y_begin++;
	}

	for (index_t i = y_begin; i < y_end; i++)
	{
		const real_t b_tmp = b1_s + (i == y_len - 1 ? a_s : 0);
		const real_t r = 1 / (b_tmp - a_s * c_tmp);

		for (index_t x = x_begin; x < x_end; x++)
		{
			d.template at<dim, 'x'>(i, x) =
				r * (d.template at<dim, 'x'>(i, x) - a_s * d.template at<dim, 'x'>(i - 1, x));
		}

		c_tmp = a_s * r;
	}
}

template <typename index_t, typename density_layout_t, typename diagonal_layout_t>
static void yb_part(density_layout_t d, diagonal_layout_t c, index_t x_begin, const index_t x_end, index_t y_begin,
					const index_t y_end)
{
	const index_t y_len = d | noarr::get_length<'y'>();

	constexpr char dim = 'y';

	for (index_t i = std::min(y_end - 1, y_len - 2); i >= y_begin; i--)
	{
		const auto back_c = c.template at<'i'>(i);

		for (index_t x = x_begin; x < x_end; x++)
			d.template at<dim, 'x'>(i, x) -= back_c * d.template at<dim, 'x'>(i + 1, x);
	}
}

template <typename index_t, typename real_t, typename density_layout_t>
static void solve_xf_block(real_t* __restrict__ densities, const real_t a_s, const real_t b1_s,
						   const density_layout_t dens_l, const index_t s, index_t& c_tmp, index_t x_begin,
						   const index_t x_end, const index_t y_begin, const index_t y_end)
{
	index_t y_size = y_end - y_begin;
	index_t x_size = x_end - x_begin;

	if (y_size > division_size || x_size > division_size)
	{
		if (x_size > division_size && y_size > division_size)
		{
			index_t c_tmp_copy = c_tmp;
			solve_xf_block(densities, a_s, b1_s, dens_l, s, c_tmp_copy, x_begin, x_begin + x_size / 2, y_begin,
						   y_begin + y_size / 2);
			solve_xf_block(densities, a_s, b1_s, dens_l, s, c_tmp_copy, x_begin + x_size / 2, x_end, y_begin,
						   y_begin + y_size / 2);

			solve_xf_block(densities, a_s, b1_s, dens_l, s, c_tmp, x_begin, x_begin + x_size / 2, y_begin + y_size / 2,
						   y_end);
			solve_xf_block(densities, a_s, b1_s, dens_l, s, c_tmp, x_begin + x_size / 2, x_end, y_begin + y_size / 2,
						   y_end);
		}
		else if (x_size > division_size)
		{
			solve_xf_block(densities, a_s, b1_s, dens_l, s, c_tmp, x_begin, x_begin + x_size / 2, y_begin, y_end);
			solve_xf_block(densities, a_s, b1_s, dens_l, s, c_tmp, x_begin + x_size / 2, x_end, y_begin, y_end);
		}
		else
		{
			index_t c_tmp_copy = c_tmp;
			solve_xf_block(densities, a_s, b1_s, dens_l, s, c_tmp_copy, x_begin, x_end, y_begin, y_begin + y_size / 2);
			solve_xf_block(densities, a_s, b1_s, dens_l, s, c_tmp, x_begin, x_end, y_begin + y_size / 2, y_end);
		}

		return;
	}

	// std::cout << "[" << x_begin << ", " << x_end << ") x [" << y_begin << ", " << y_end << ") xf" << std::endl;


	const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s'>(s), densities);

	for (index_t y = y_begin; y < y_end; y++)
	{
		xf_part(d, a_s, b1_s, c_tmp, x_begin, x_end, y);
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_xb_yf_block(real_t* __restrict__ densities, const real_t ax_s, const real_t b1x_s, const real_t ay_s,
							  const real_t b1y_s, const density_layout_t dens_l, const index_t s, diagonal_layout_t cx,
							  index_t& cx_tmp, index_t& cy_tmp, index_t x_begin, const index_t x_end,
							  const index_t y_begin, const index_t y_end)
{
	index_t y_size = y_end - y_begin;
	index_t x_size = x_end - x_begin;

	if (y_size > division_size || x_size > division_size)
	{
		if (x_size > division_size && y_size > division_size)
		{
			index_t cx_tmp_copy = cx_tmp;
			index_t cy_tmp_copy = cy_tmp;
			solve_xb_yf_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cx_tmp_copy, cy_tmp_copy,
							  x_begin + x_size / 2, x_end, y_begin, y_begin + y_size / 2);
			solve_xb_yf_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cx_tmp_copy, cy_tmp, x_begin,
							  x_size / 2, y_begin, y_begin + y_size / 2);

			solve_xb_yf_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cx_tmp, cy_tmp_copy,
							  x_begin + x_size / 2, x_end, y_begin + y_size / 2, y_end);
			solve_xb_yf_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cx_tmp, cy_tmp, x_begin,
							  x_begin + x_size / 2, y_size / 2, y_end);
		}

		return;
	}

	// std::cout << "[" << x_begin << ", " << x_end << ") x [" << y_begin << ", " << y_end << ") xb yf" << std::endl;

	const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s'>(s), densities);

	for (index_t y = y_begin; y < y_end; y++)
	{
		xb_part(d, cx, x_begin, x_end, y);
	}

	yf_part(d, ay_s, b1y_s, cy_tmp, x_begin, x_end, y_begin, y_end);
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_xf_xb_yf_block(real_t* __restrict__ densities, const real_t ax_s, const real_t b1x_s,
								 const real_t ay_s, const real_t b1y_s, const density_layout_t dens_l, const index_t s,
								 diagonal_layout_t cx, index_t& cx_tmp, index_t& cy_tmp, index_t x_begin,
								 const index_t x_end, const index_t y_begin, const index_t y_end)
{
	index_t y_size = y_end - y_begin;
	index_t x_size = x_end - x_begin;

	if (y_size > division_size || x_size > division_size)
	{
		if (x_size > division_size && y_size > division_size)
		{
			index_t cx_tmp_copy = cx_tmp;
			index_t cy_tmp_copy = cy_tmp;
			solve_xf_block(densities, ax_s, b1x_s, dens_l, s, cx_tmp_copy, x_begin, x_begin + x_size / 2, y_begin,
						   y_begin + y_size / 2);
			solve_xf_xb_yf_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cx_tmp_copy, cy_tmp_copy,
								 x_begin + x_size / 2, x_end, y_begin, y_begin + y_size / 2);
			solve_xb_yf_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cx_tmp_copy, cy_tmp, x_begin,
							  x_begin + x_size / 2, y_begin, y_begin + y_size / 2);


			solve_xf_block(densities, ax_s, b1x_s, dens_l, s, cx_tmp_copy, x_begin, x_begin + x_size / 2,
						   y_begin + y_size / 2, y_end);
			solve_xf_xb_yf_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cx_tmp_copy, cy_tmp_copy,
								 x_begin + x_size / 2, x_end, y_begin + y_size / 2, y_end);
			solve_xb_yf_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cx_tmp_copy, cy_tmp, x_begin,
							  x_begin + x_size / 2, y_begin + y_size / 2, y_end);
		}
		// else if (x_size > division_size)
		// {
		// 	solve_xf_block(densities, ax_s, b1x_s, dens_l, s, cx_tmp, x_begin, x_begin + x_size / 2, y_begin, y_begin +
		// y_size / 2); 	solve_xf_xb_yf_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cx_tmp, cy_tmp,
		// x_begin
		// + x_size / 2, x_end, 						 y_begin, y_begin + y_size / 2);
		// }
		// else
		// {
		// 	index_t cx_tmp_copy = cx_tmp;
		// 	solve_xf_xb_yf_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cx_tmp_copy, cy_tmp, x_begin,
		// 						 x_end, y_begin, y_begin + y_size / 2);
		// 	solve_xf_xb_yf_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cx_tmp, cy_tmp, x_begin, x_end,
		// 						 y_size / 2, y_end);
		// }

		return;
	}

	// std::cout << "[" << x_begin << ", " << x_end << ") x [" << y_begin << ", " << y_end << ") xf xb yf" << std::endl;

	const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s'>(s), densities);

	for (index_t y = y_begin; y < y_end; y++)
	{
		xf_part(d, ax_s, b1x_s, cx_tmp, x_begin, x_end, y);

		xb_part(d, cx, x_begin, x_end, y);
	}

	yf_part(d, ay_s, b1y_s, cy_tmp, x_begin, x_end, y_begin, y_end);
}


template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_xb_yf_yb_block(real_t* __restrict__ densities, const real_t ax_s, const real_t b1x_s,
								 const real_t ay_s, const real_t b1y_s, const density_layout_t dens_l, const index_t s,
								 diagonal_layout_t cx, diagonal_layout_t cy, index_t& cx_tmp, index_t& cy_tmp,
								 index_t x_begin, const index_t x_end, const index_t y_begin, const index_t y_end)
{
	index_t y_size = y_end - y_begin;
	index_t x_size = x_end - x_begin;

	if (y_size > division_size || x_size > division_size)
	{
		if (x_size > division_size && y_size > division_size)
		{
			index_t cx_tmp_copy = cx_tmp;
			index_t cy_tmp_copy = cy_tmp;

			solve_xb_yf_yb_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cy, cx_tmp, cy_tmp_copy,
								 x_begin + x_size / 2, x_end, y_begin + y_size / 2, y_end);
			solve_xb_yf_yb_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cy, cx_tmp, cy_tmp, x_begin,
								 x_begin + x_size / 2, y_begin + y_size / 2, y_end);

			solve_xb_yf_yb_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cy, cx_tmp_copy, cy_tmp_copy,
								 x_begin + x_size / 2, x_end, y_begin, y_begin + y_size / 2);
			solve_xb_yf_yb_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cy, cx_tmp_copy, cy_tmp, x_begin,
								 x_begin + x_size / 2, y_begin, y_begin + y_size / 2);
		}

		return;
	}


	// std::cout << "[" << x_begin << ", " << x_end << ") x [" << y_begin << ", " << y_end << ") xb yf yb" << std::endl;

	const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s'>(s), densities);

	for (index_t y = y_begin; y < y_end; y++)
	{
		xb_part(d, cx, x_begin, x_end, y);
	}

	yf_part(d, ay_s, b1y_s, cy_tmp, x_begin, x_end, y_begin, y_end);

	yb_part(d, cy, x_begin, x_end, y_begin, y_end);
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_yb_block(real_t* __restrict__ densities, const real_t ax_s, const real_t b1x_s, const real_t ay_s,
						   const real_t b1y_s, const density_layout_t dens_l, const index_t s, diagonal_layout_t cx,
						   diagonal_layout_t cy, index_t& cx_tmp, index_t& cy_tmp, index_t x_begin, const index_t x_end,
						   const index_t y_begin, const index_t y_end)
{
	index_t y_size = y_end - y_begin;
	index_t x_size = x_end - x_begin;

	if (y_size > division_size || x_size > division_size)
	{
		if (x_size > division_size && y_size > division_size)
		{
			index_t cy_tmp_copy = cy_tmp;

			solve_yb_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cy, cx_tmp, cy_tmp_copy, x_begin,
						   x_begin + x_size / 2, y_begin + y_size / 2, y_end);
			solve_yb_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cy, cx_tmp, cy_tmp_copy, x_begin,
						   x_begin + x_size / 2, y_begin, y_begin + y_size / 2);

			solve_yb_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cy, cx_tmp, cy_tmp, x_begin + x_size / 2,
						   x_end, y_size / 2, y_end);
			solve_yb_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cy, cx_tmp, cy_tmp, x_begin + x_size / 2,
						   x_end, y_begin, y_begin + y_size / 2);
		}

		return;
	}


	// std::cout << "[" << x_begin << ", " << x_end << ") x [" << y_begin << ", " << y_end << ") yb" << std::endl;

	const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s'>(s), densities);

	yb_part(d, cy, x_begin, x_end, y_begin, y_end);
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_xf_xb_yf_yb_block(real_t* __restrict__ densities, const real_t ax_s, const real_t b1x_s,
									const real_t ay_s, const real_t b1y_s, const density_layout_t dens_l,
									const index_t s, diagonal_layout_t cx, diagonal_layout_t cy, index_t& cx_tmp,
									index_t& cy_tmp, index_t x_begin, const index_t x_end, const index_t y_begin,
									const index_t y_end)
{
	index_t y_size = y_end - y_begin;
	index_t x_size = x_end - x_begin;

	if (y_size > division_size || x_size > division_size)
	{
		if (x_size > division_size && y_size > division_size)
		{
			index_t cx_tmp_copy = cx_tmp;
			index_t cy_tmp_copy = cy_tmp;
			solve_xf_block(densities, ax_s, b1x_s, dens_l, s, cx_tmp_copy, x_begin, x_begin + x_size / 2, y_begin,
						   y_begin + y_size / 2);
			solve_xf_xb_yf_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cx_tmp_copy, cy_tmp_copy,
								 x_begin + x_size / 2, x_end, y_begin, y_begin + y_size / 2);
			solve_xb_yf_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cx_tmp_copy, cy_tmp, x_begin,
							  x_begin + x_size / 2, y_begin, y_begin + y_size / 2);


			solve_xf_block(densities, ax_s, b1x_s, dens_l, s, cx_tmp_copy, x_begin, x_begin + x_size / 2,
						   y_begin + y_size / 2, y_end);
			solve_xf_xb_yf_yb_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cy, cx_tmp_copy, cy_tmp_copy,
									x_begin + x_size / 2, x_end, y_begin + y_size / 2, y_end);
			solve_xb_yf_yb_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cy, cx_tmp_copy, cy_tmp, x_begin,
								 x_begin + x_size / 2, y_begin + y_size / 2, y_end);

			solve_yb_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cy, cx_tmp_copy, cy_tmp, x_begin,
						   x_begin + x_size / 2, y_begin, y_begin + y_size / 2);

			solve_yb_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cy, cx_tmp_copy, cy_tmp_copy,
						   x_begin + x_size / 2, x_end, y_begin, y_begin + y_size / 2);
		}

		return;
	}


	// std::cout << "[" << x_begin << ", " << x_end << ") x [" << y_begin << ", " << y_end << ") xf xb yf yb" <<
	// std::endl;

	const auto d = noarr::make_bag(dens_l ^ noarr::fix<'s'>(s), densities);

	for (index_t y = y_begin; y < y_end; y++)
	{
		xf_part(d, ax_s, b1x_s, cx_tmp, x_begin, x_end, y);

		xb_part(d, cx, x_begin, x_end, y);
	}

	yf_part(d, ay_s, b1y_s, cy_tmp, x_begin, x_end, y_begin, y_end);

	yb_part(d, cy, x_begin, x_end, y_begin, y_end);
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_2d_co(real_t* __restrict__ densities, const real_t* __restrict__ ax, const real_t* __restrict__ b1x,
						const real_t* __restrict__ back_cx, const real_t* __restrict__ ay,
						const real_t* __restrict__ b1y, const real_t* __restrict__ back_cy,
						const density_layout_t dens_l, const diagonal_layout_t diagx_l, const diagonal_layout_t diagy_l)
{
	const index_t s_len = dens_l | noarr::get_length<'s'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();

#pragma omp parallel for
	for (index_t s = 0; s < s_len; s++)
	{
		const real_t ax_s = ax[s];
		const real_t b1x_s = b1x[s];

		const real_t ay_s = ay[s];
		const real_t b1y_s = b1y[s];

		index_t cx_tmp = ax_s;
		index_t cy_tmp = ay_s;

		index_t cx_tmp_copy = ax_s;
		index_t cy_tmp_copy = ay_s;

		auto cx = noarr::make_bag(diagx_l ^ noarr::fix<'s'>(s), back_cx);
		auto cy = noarr::make_bag(diagy_l ^ noarr::fix<'s'>(s), back_cy);

		solve_xf_block(densities, ax_s, b1x_s, dens_l, s, cx_tmp, 0, x_len / 2, 0, y_len / 2);

		solve_xf_xb_yf_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cx_tmp, cy_tmp, x_len / 2, x_len, 0,
							 y_len / 2);

		solve_xb_yf_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cx_tmp, cy_tmp_copy, 0, x_len / 2, 0,
						  y_len / 2);

		solve_xf_block(densities, ax_s, b1x_s, dens_l, s, cx_tmp_copy, 0, x_len / 2, y_len / 2, y_len);

		solve_xf_xb_yf_yb_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cy, cx_tmp_copy, cy_tmp, x_len / 2,
								x_len, y_len / 2, y_len);

		solve_xb_yf_yb_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cy, cx_tmp_copy, cy_tmp_copy, 0,
							 x_len / 2, y_len / 2, y_len);

		solve_yb_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cy, cx_tmp, cy_tmp_copy, 0, x_len / 2, 0,
					   y_len / 2);

		solve_yb_block(densities, ax_s, b1x_s, ay_s, b1y_s, dens_l, s, cx, cy, cx_tmp, cy_tmp, x_len / 2, x_len, 0,
					   y_len / 2);
	}
}

template <typename real_t, bool aligned_x>
void co_thomas_solver<real_t, aligned_x>::solve_x()
{}

template <typename real_t, bool aligned_x>
void co_thomas_solver<real_t, aligned_x>::solve_y()
{}

template <typename real_t, bool aligned_x>
void co_thomas_solver<real_t, aligned_x>::solve_z()
{}

template <typename real_t, bool aligned_x>
void co_thomas_solver<real_t, aligned_x>::solve()
{
	if (t == 0)
		solve_2d_naive<index_t>(this->substrates_, ax_, b1x_, cx_, ay_, b1y_, cy_, get_substrates_layout<2>(),
								get_diagonal_layout(this->problem_, this->problem_.nx),
								get_diagonal_layout(this->problem_, this->problem_.ny));


	if (t == 1)
		solve_2d_tile<index_t>(this->substrates_, ax_, b1x_, cx_, ay_, b1y_, cy_, get_substrates_layout<2>(),
							   get_diagonal_layout(this->problem_, this->problem_.nx),
							   get_diagonal_layout(this->problem_, this->problem_.ny), x_tile_size_);

	if (t == 2)
		solve_2d_fused<index_t>(this->substrates_, ax_, b1x_, cx_, ay_, b1y_, cy_, get_substrates_layout<2>(),
								get_diagonal_layout(this->problem_, this->problem_.nx),
								get_diagonal_layout(this->problem_, this->problem_.ny));

	if (t == 3)
		solve_2d_co<index_t>(this->substrates_, ax_, b1x_, cx_, ay_, b1y_, cy_, get_substrates_layout<2>(),
							 get_diagonal_layout(this->problem_, this->problem_.nx),
							 get_diagonal_layout(this->problem_, this->problem_.ny));
}

template <typename real_t, bool aligned_x>
co_thomas_solver<real_t, aligned_x>::co_thomas_solver(bool use_intrinsics, bool use_fused)
	: ax_(nullptr),
	  b1x_(nullptr),
	  cx_(nullptr),
	  ay_(nullptr),
	  b1y_(nullptr),
	  cy_(nullptr),
	  az_(nullptr),
	  b1z_(nullptr),
	  cz_(nullptr),
	  a_scratch_(nullptr),
	  c_scratch_(nullptr),
	  use_intrinsics_(use_intrinsics),
	  use_blocked_(use_fused)
{}

template <typename real_t, bool aligned_x>
co_thomas_solver<real_t, aligned_x>::~co_thomas_solver()
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

	if (a_scratch_)
	{
		std::free(a_scratch_);
		std::free(c_scratch_);
	}
}

template class co_thomas_solver<float, false>;
template class co_thomas_solver<double, false>;

template class co_thomas_solver<float, true>;
template class co_thomas_solver<double, true>;
