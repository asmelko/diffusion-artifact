#include "partial_blocking.h"

#include <cstddef>
#include <iostream>

#include "../perf_utils.h"
#include "../vector_transpose_helper.h"

template <typename real_t, bool aligned_x>
void sdd_partial_blocking<real_t, aligned_x>::precompute_values(real_t*& a, real_t*& b, real_t*& c, index_t shape,
																index_t n, index_t dims, char dim,
																auto substrates_layout)
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
void sdd_partial_blocking<real_t, aligned_x>::prepare(const max_problem_t& problem)
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
void sdd_partial_blocking<real_t, aligned_x>::tune(const nlohmann::json& params)
{
	x_tile_size_ = params.contains("x_tile_size") ? (std::size_t)params["x_tile_size"] : 48;
	continuous_x_diagonal_ = params.contains("continuous_x_diagonal") ? (bool)params["continuous_x_diagonal"] : false;

	using simd_tag = hn::ScalableTag<real_t>;
	simd_tag d;
	std::size_t vector_length = hn::Lanes(d) * sizeof(real_t);

	alignment_size_ = params.contains("alignment_size") ? (std::size_t)params["alignment_size"] : vector_length;
}

template <typename real_t, bool aligned_x>
void sdd_partial_blocking<real_t, aligned_x>::initialize()
{
	if (continuous_x_diagonal_)
		precompute_values(ax_, bx_, cx_, this->problem_.dx, this->problem_.nx, this->problem_.dims, 'x',
						  get_diag_layout_x() ^ noarr::merge_blocks<'Y', 'y', 'y'>());
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


template <typename real_t, typename index_t, typename density_bag_t, typename diag_bag_t, typename scratch_t>
constexpr static void x_forward(const density_bag_t d, const index_t y, const index_t z, const index_t s,
								const index_t n, const diag_bag_t a, const diag_bag_t b, const diag_bag_t c,
								const scratch_t b_scratch)

{
	auto diag_l = a.structure() ^ noarr::merge_blocks<'Y', 'y'>();
	auto a_bag = noarr::make_bag(diag_l, a.data());
	auto b_bag = noarr::make_bag(diag_l, b.data());
	auto c_bag = noarr::make_bag(diag_l, c.data());

	{
		auto idx = noarr::idx<'s', 'z', 'y', 'x', 'v'>(s, z, y, 0, noarr::lit<0>);
		b_scratch[idx] = 1 / b_bag[idx];
	}

	for (index_t i = 1; i < n; i++)
	{
		auto idx = noarr::idx<'s', 'z', 'y', 'x', 'v'>(s, z, y, i, noarr::lit<0>);
		auto prev_idx = noarr::idx<'s', 'z', 'y', 'x', 'v'>(s, z, y, i - 1, noarr::lit<0>);

		auto r = a_bag[idx] * b_scratch[prev_idx];

		b_scratch[idx] = 1 / (b_bag[idx] - c_bag[prev_idx] * r);

		d[idx] -= r * d[prev_idx];

		// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
	}
}

template <typename real_t, typename index_t, typename density_bag_t, typename diag_bag_t, typename scratch_t>
constexpr static void x_backward(const density_bag_t d, const index_t y, const index_t z, const index_t s,
								 const index_t n, const diag_bag_t c, scratch_t b_scratch)

{
	auto diag_l = c.structure() ^ noarr::merge_blocks<'Y', 'y'>();
	auto c_bag = noarr::make_bag(diag_l, c.data());

	{
		auto idx = noarr::idx<'s', 'z', 'y', 'x', 'v'>(s, z, y, n - 1, noarr::lit<0>);

		d[idx] *= b_scratch[idx];

		// std::cout << "n-1: " << (dens_l | noarr::get_at<'x', 's'>(densities, n - 1, s)) << std::endl;
	}

	for (index_t i = n - 2; i >= 0; i--)
	{
		auto idx = noarr::idx<'s', 'z', 'y', 'x', 'v'>(s, z, y, i, noarr::lit<0>);
		auto next_idx = noarr::idx<'s', 'z', 'y', 'x', 'v'>(s, z, y, i + 1, noarr::lit<0>);

		d[idx] = (d[idx] - c_bag[idx] * d[next_idx]) * b_scratch[idx];

		// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
	}
}

template <typename index_t, typename simd_t, typename simd_tag, typename diagonal_t, typename scratch_t,
		  typename... vec_pack_t>
constexpr static void x_forward_vectorized(simd_tag d, const index_t x, const index_t y, const index_t z,
										   const index_t s, const diagonal_t a, const diagonal_t b, const diagonal_t c,
										   const scratch_t b_scratch, simd_t& scratch_prev, simd_t& c_prev,
										   simd_t& d_prev, simd_t& d_first, vec_pack_t&... vec_pack)
{
	HWY_LANES_CONSTEXPR index_t max_length = hn::Lanes(hn::ScalableTag<hn::TFromD<simd_tag>> {});
	HWY_LANES_CONSTEXPR index_t simd_length = std::min(16, max_length);

	const auto idx = noarr::idx<'s', 'z', 'Y', 'y', 'x', 'v'>(s, z, y / simd_length, y % simd_length, x, noarr::lit<0>);

	simd_t a_curr = hn::Load(d, &a[idx]);
	simd_t b_curr = hn::Load(d, &b[idx]);

	auto r = hn::Mul(a_curr, scratch_prev);

	scratch_prev = hn::Div(hn::Set(d, 1), hn::NegMulAdd(c_prev, r, b_curr));
	hn::Store(scratch_prev, d, &b_scratch[idx]);

	d_first = hn::NegMulAdd(d_prev, r, d_first);

	c_prev = hn::Load(d, &c[idx]);

	if constexpr (sizeof...(vec_pack_t) != 0)
		x_forward_vectorized(d, x + 1, y, z, s, a, b, c, b_scratch, scratch_prev, c_prev, d_first, vec_pack...);
}

template <typename index_t, typename simd_t, typename simd_tag, typename diagonal_t, typename scratch_t,
		  typename... vec_pack_t>
constexpr static void x_forward_vectorized(simd_tag d, const index_t length, const index_t x, const index_t y,
										   const index_t z, const index_t s, const diagonal_t a, const diagonal_t b,
										   const diagonal_t c, const scratch_t b_scratch, simd_t& scratch_prev,
										   simd_t& c_prev, simd_t& d_prev, simd_t& d_first, vec_pack_t&... vec_pack)
{
	if (length == 0)
		return;

	HWY_LANES_CONSTEXPR index_t max_length = hn::Lanes(hn::ScalableTag<hn::TFromD<simd_tag>> {});
	HWY_LANES_CONSTEXPR index_t simd_length = std::min(16, max_length);

	const auto idx = noarr::idx<'s', 'z', 'Y', 'y', 'x', 'v'>(s, z, y / simd_length, y % simd_length, x, noarr::lit<0>);

	simd_t a_curr = hn::Load(d, &a[idx]);
	simd_t b_curr = hn::Load(d, &b[idx]);

	auto r = hn::Mul(a_curr, scratch_prev);

	scratch_prev = hn::Div(hn::Set(d, 1), hn::NegMulAdd(c_prev, r, b_curr));
	hn::Store(scratch_prev, d, &b_scratch[idx]);

	d_first = hn::NegMulAdd(d_prev, r, d_first);

	c_prev = hn::Load(d, &c[idx]);

	if constexpr (sizeof...(vec_pack_t) != 0)
		x_forward_vectorized(d, length - 1, x + 1, y, z, s, a, b, c, b_scratch, scratch_prev, c_prev, d_first,
							 vec_pack...);
}

template <typename index_t, typename simd_t, typename simd_tag, typename diagonal_t, typename scratch_t,
		  typename... vec_pack_t>
constexpr static simd_t x_backward_vectorized(simd_tag d, const index_t x, const index_t y, const index_t z,
											  const index_t s, const diagonal_t c, scratch_t b_scratch, simd_t& d_first,
											  simd_t& d_prev, vec_pack_t&... vec_pack)
{
	if constexpr (sizeof...(vec_pack_t) != 0)
		x_backward_vectorized(d, x + 1, y, z, s, c, b_scratch, d_prev, vec_pack...);

	HWY_LANES_CONSTEXPR index_t max_length = hn::Lanes(hn::ScalableTag<hn::TFromD<simd_tag>> {});
	HWY_LANES_CONSTEXPR index_t simd_length = std::min(16, max_length);

	const auto idx = noarr::idx<'s', 'z', 'Y', 'y', 'x', 'v'>(s, z, y / simd_length, y % simd_length, x, noarr::lit<0>);

	simd_t c_curr = hn::Load(d, &c[idx]);
	simd_t scratch = hn::Load(d, &b_scratch[idx]);

	d_first = hn::Mul(hn::NegMulAdd(d_prev, c_curr, d_first), scratch);

	return d_first;
}


template <typename index_t, typename simd_t, typename simd_tag, typename diagonal_t, typename scratch_t,
		  typename... vec_pack_t>
constexpr static simd_t x_backward_vectorized(simd_tag d, const index_t length, const index_t x, const index_t y,
											  const index_t z, const index_t s, const diagonal_t c, scratch_t b_scratch,
											  simd_t& d_first, simd_t& d_prev, vec_pack_t&... vec_pack)
{
	if (length == 0)
		return d_first;

	if constexpr (sizeof...(vec_pack_t) != 0)
		x_backward_vectorized(d, length - 1, x + 1, y, z, s, c, b_scratch, d_prev, vec_pack...);

	HWY_LANES_CONSTEXPR index_t max_length = hn::Lanes(hn::ScalableTag<hn::TFromD<simd_tag>> {});
	HWY_LANES_CONSTEXPR index_t simd_length = std::min(16, max_length);

	const auto idx = noarr::idx<'s', 'z', 'Y', 'y', 'x', 'v'>(s, z, y / simd_length, y % simd_length, x, noarr::lit<0>);

	simd_t c_curr = hn::Load(d, &c[idx]);
	simd_t scratch = hn::Load(d, &b_scratch[idx]);

	d_first = hn::Mul(hn::NegMulAdd(d_prev, c_curr, d_first), scratch);

	return d_first;
}

template <typename index_t, typename simd_tag, typename scratch_t, typename simd_t, typename... vec_pack_t>
constexpr static void x_forward_vectorized_multi(simd_tag d, const index_t x, const index_t y, const index_t z,
												 const index_t s, simd_t& scratch_prev, const scratch_t b_scratch,
												 simd_t& a_first, vec_pack_t&... a_vec_pack, simd_t& b_first,
												 vec_pack_t&... b_vec_pack, simd_t& c_prev, vec_pack_t&... c_vec_pack,
												 simd_t& c_ignore, simd_t& d_prev, simd_t& d_first,
												 vec_pack_t&... d_vec_pack)
{
	const auto idx = noarr::idx<'s', 'z', 'y', 'x'>(s, z, y, x);

	auto r = hn::Mul(a_first, scratch_prev);

	scratch_prev = hn::Div(hn::Set(d, 1), hn::NegMulAdd(c_prev, r, b_first));
	hn::Store(scratch_prev, d, &b_scratch[idx]);

	d_first = hn::NegMulAdd(d_prev, r, d_first);

	if constexpr (sizeof...(vec_pack_t) != 0)
		x_forward_vectorized_multi<index_t, simd_tag, scratch_t, vec_pack_t...>(
			d, x + 1, y, z, s, scratch_prev, b_scratch, a_vec_pack..., b_vec_pack..., c_vec_pack..., c_ignore, d_first,
			d_vec_pack...);
}

template <typename index_t, typename simd_tag, typename scratch_t, typename simd_t, typename... vec_pack_t>
constexpr static void x_forward_vectorized_multi(simd_tag d, const index_t length, const index_t x, const index_t y,
												 const index_t z, const index_t s, simd_t& scratch_prev,
												 const scratch_t b_scratch, simd_t& a_first, vec_pack_t&... a_vec_pack,
												 simd_t& b_first, vec_pack_t&... b_vec_pack, simd_t& c_prev,
												 vec_pack_t&... c_vec_pack, simd_t& c_ignore, simd_t& d_prev,
												 simd_t& d_first, vec_pack_t&... d_vec_pack)
{
	if (length == 0)
		return;

	const auto idx = noarr::idx<'s', 'z', 'y', 'x'>(s, z, y, x);

	auto r = hn::Mul(a_first, scratch_prev);

	scratch_prev = hn::Div(hn::Set(d, 1), hn::NegMulAdd(c_prev, r, b_first));
	hn::Store(scratch_prev, d, &b_scratch[idx]);

	d_first = hn::NegMulAdd(d_prev, r, d_first);

	if constexpr (sizeof...(vec_pack_t) != 0)
		x_forward_vectorized_multi<index_t, simd_tag, scratch_t, vec_pack_t...>(
			d, length - 1, x + 1, y, z, s, scratch_prev, b_scratch, a_vec_pack..., b_vec_pack..., c_vec_pack...,
			c_ignore, d_first, d_vec_pack...);
}

template <typename index_t, typename simd_tag, typename scratch_t, typename simd_t, typename... vec_pack_t>
constexpr static simd_t x_backward_vectorized_multi(simd_tag d, const index_t x, const index_t y, const index_t z,
													const index_t s, scratch_t b_scratch, simd_t& c_first,
													vec_pack_t&... c_vec_pack, simd_t& d_first, simd_t& d_prev,
													vec_pack_t&... vec_pack)
{
	if constexpr (sizeof...(vec_pack_t) != 0)
		x_backward_vectorized_multi<index_t, simd_tag, scratch_t, vec_pack_t...>(d, x + 1, y, z, s, b_scratch,
																				 c_vec_pack..., d_prev, vec_pack...);

	const auto idx = noarr::idx<'s', 'z', 'y', 'x'>(s, z, y, x);

	auto scratch = hn::Load(d, &b_scratch[idx]);

	d_first = hn::Mul(hn::NegMulAdd(d_prev, c_first, d_first), scratch);

	return d_first;
}

template <typename index_t, typename simd_tag, typename scratch_t, typename simd_t, typename... vec_pack_t>
constexpr static simd_t x_backward_vectorized_multi(simd_tag d, const index_t length, const index_t x, const index_t y,
													const index_t z, const index_t s, scratch_t b_scratch,
													simd_t& c_first, vec_pack_t&... c_vec_pack, simd_t& d_first,
													simd_t& d_prev, vec_pack_t&... vec_pack)
{
	if (length == 0)
		return d_first;

	if constexpr (sizeof...(vec_pack_t) != 0)
		x_backward_vectorized_multi<index_t, simd_tag, scratch_t, vec_pack_t...>(
			d, length - 1, x + 1, y, z, s, b_scratch, c_vec_pack..., d_prev, vec_pack...);

	const auto idx = noarr::idx<'s', 'z', 'y', 'x'>(s, z, y, x);

	auto scratch = hn::Load(d, &b_scratch[idx]);

	d_first = hn::Mul(hn::NegMulAdd(d_prev, c_first, d_first), scratch);

	return d_first;
}

template <typename simd_t, typename simd_tag, typename index_t, typename density_bag_t, typename... simd_pack_t>
constexpr static void load(const density_bag_t d, simd_tag t, const index_t x, const index_t y, const index_t z,
						   const index_t s, simd_t& first, simd_pack_t&... vec_pack)
{
	first = hn::Load(t, &(d.template at<'s', 'z', 'y', 'x'>(s, z, y, x)));
	if constexpr (sizeof...(vec_pack) > 0)
	{
		load(d, t, x, y + 1, z, s, vec_pack...);
	}
}

template <typename simd_t, typename simd_tag, typename index_t, typename density_bag_t, typename... simd_pack_t>
constexpr static void store(const density_bag_t d, simd_tag t, const index_t x, const index_t y, const index_t z,
							const index_t s, const simd_t& first, const simd_pack_t&... vec_pack)
{
	hn::Store(first, t, &(d.template at<'s', 'z', 'y', 'x'>(s, z, y, x)));
	if constexpr (sizeof...(vec_pack) > 0)
	{
		store(d, t, x, y + 1, z, s, vec_pack...);
	}
}

template <typename T>
constexpr inline decltype(auto) id(T&& t)
{
	return std::forward<T>(t);
}

template <typename... Args>
constexpr decltype(auto) get_last(Args&&... args)
{
	return (id(std::forward<Args>(args)), ...);
}

template <typename simd_t, typename simd_tag, typename index_t, typename density_bag_t, typename diag_bag_t,
		  typename scratch_bag_t, typename... simd_pack_t>
constexpr static void xy_fused_transpose_part(const density_bag_t d, simd_tag t, const index_t y_offset,
											  const index_t y_len, const index_t s, const index_t z, const index_t n,
											  const diag_bag_t a, const diag_bag_t b, const diag_bag_t c,
											  const scratch_bag_t b_scratch, simd_pack_t... vec_pack)
{
	constexpr index_t simd_length = sizeof...(vec_pack);

	const index_t full_n = (n + simd_length - 1) / simd_length * simd_length;

	for (index_t y = y_offset; y < y_len; y += simd_length)
	{
		simd_t c_prev = hn::Zero(t);
		simd_t d_prev = hn::Zero(t);
		simd_t scratch_prev = hn::Zero(t);

		// forward substitution until last simd_length elements
		for (index_t i = 0; i < full_n - simd_length; i += simd_length)
		{
			load(d, t, i, y, z, s, vec_pack...);

			// transposition to enable vectorization
			transpose(vec_pack...);

			// actual forward substitution (vectorized)
			x_forward_vectorized(t, i, y, z, s, a, b, c, b_scratch, scratch_prev, c_prev, d_prev, vec_pack...);

			d_prev = get_last(vec_pack...);

			// transposition back to the original form
			// transpose(rows);

			store(d, t, i, y, z, s, vec_pack...);
		}

		// we are aligned to the vector size, so we can safely continue
		// here we fuse the end of forward substitution and the beginning of backwards propagation
		{
			load(d, t, full_n - simd_length, y, z, s, vec_pack...);

			// transposition to enable vectorization
			transpose(vec_pack...);

			index_t remainder_work = n % simd_length;
			remainder_work += remainder_work == 0 ? simd_length : 0;

			// the rest of forward part
			x_forward_vectorized(t, remainder_work, full_n - simd_length, y, z, s, a, b, c, b_scratch, scratch_prev,
								 c_prev, d_prev, vec_pack...);

			d_prev = hn::Zero(t);

			// the begin of backward part
			d_prev = x_backward_vectorized(t, remainder_work, full_n - simd_length, y, z, s, c, b_scratch, vec_pack...,
										   d_prev);

			// transposition back to the original form
			transpose(vec_pack...);

			store(d, t, full_n - simd_length, y, z, s, vec_pack...);
		}

		// we continue with backwards substitution
		for (index_t i = full_n - simd_length * 2; i >= 0; i -= simd_length)
		{
			load(d, t, i, y, z, s, vec_pack...);

			// transposition to enable vectorization
			// transpose(rows);

			// backward propagation
			d_prev = x_backward_vectorized(t, i, y, z, s, c, b_scratch, vec_pack..., d_prev);

			// transposition back to the original form
			transpose(vec_pack...);

			store(d, t, i, y, z, s, vec_pack...);
		}
	}
}

template <typename simd_t, typename simd_tag, typename index_t, typename density_bag_t, typename diag_bag_t,
		  typename scratch_bag_t, typename... simd_pack_t>
constexpr static void xy_fused_transpose_part_multi(const density_bag_t d, simd_tag t, const index_t y_offset,
													const index_t y_len, const index_t s, const index_t z,
													const index_t n, const diag_bag_t a, const diag_bag_t b,
													const diag_bag_t c, const scratch_bag_t b_scratch,
													simd_pack_t... a_vec_pack, simd_pack_t... b_vec_pack,
													simd_pack_t... c_vec_pack, simd_pack_t... d_vec_pack)
{
	constexpr index_t simd_length = sizeof...(d_vec_pack);

	const index_t full_n = (n + simd_length - 1) / simd_length * simd_length;

	for (index_t y = y_offset; y < y_len; y += simd_length)
	{
		simd_t c_prev = hn::Zero(t);
		simd_t d_prev = hn::Zero(t);
		simd_t scratch_prev = hn::Zero(t);


		// forward substitution until last simd_length elements
		for (index_t i = 0; i < full_n - simd_length; i += simd_length)
		{
			load(a, t, i, y, z, s, a_vec_pack...);
			load(b, t, i, y, z, s, b_vec_pack...);
			load(c, t, i, y, z, s, c_vec_pack...);
			load(d, t, i, y, z, s, d_vec_pack...);

			// transposition to enable vectorization
			transpose(a_vec_pack...);
			transpose(b_vec_pack...);
			transpose(c_vec_pack...);
			transpose(d_vec_pack...);

			// actual forward substitution (vectorized)
			x_forward_vectorized_multi<index_t, simd_tag, scratch_bag_t, simd_pack_t...>(
				t, i, y, z, s, scratch_prev, b_scratch, a_vec_pack..., b_vec_pack..., c_prev, c_vec_pack..., d_prev,
				d_vec_pack...);

			c_prev = get_last(c_vec_pack...);
			d_prev = get_last(d_vec_pack...);

			// transposition back to the original form
			// transpose(rows);

			store(d, t, i, y, z, s, d_vec_pack...);
		}

		// we are aligned to the vector size, so we can safely continue
		// here we fuse the end of forward substitution and the beginning of backwards propagation
		{
			load(a, t, full_n - simd_length, y, z, s, a_vec_pack...);
			load(b, t, full_n - simd_length, y, z, s, b_vec_pack...);
			load(c, t, full_n - simd_length, y, z, s, c_vec_pack...);
			load(d, t, full_n - simd_length, y, z, s, d_vec_pack...);

			// transposition to enable vectorization
			transpose(a_vec_pack...);
			transpose(b_vec_pack...);
			transpose(c_vec_pack...);
			transpose(d_vec_pack...);

			index_t remainder_work = n % simd_length;
			remainder_work += remainder_work == 0 ? simd_length : 0;

			// the rest of forward part
			x_forward_vectorized_multi<index_t, simd_tag, scratch_bag_t, simd_pack_t...>(
				t, remainder_work, full_n - simd_length, y, z, s, scratch_prev, b_scratch, a_vec_pack..., b_vec_pack...,
				c_prev, c_vec_pack..., d_prev, d_vec_pack...);

			d_prev = hn::Zero(t);

			// the begin of backward part
			d_prev = x_backward_vectorized_multi<index_t, simd_tag, scratch_bag_t, simd_pack_t...>(
				t, remainder_work, full_n - simd_length, y, z, s, b_scratch, c_vec_pack..., d_vec_pack..., d_prev);

			// transposition back to the original form
			transpose(d_vec_pack...);

			store(d, t, full_n - simd_length, y, z, s, d_vec_pack...);
		}

		// we continue with backwards substitution
		for (index_t i = full_n - simd_length * 2; i >= 0; i -= simd_length)
		{
			load(c, t, i, y, z, s, c_vec_pack...);
			load(d, t, i, y, z, s, d_vec_pack...);

			// transposition to enable vectorization
			transpose(c_vec_pack...);

			// backward propagation
			d_prev = x_backward_vectorized_multi<index_t, simd_tag, scratch_bag_t, simd_pack_t...>(
				t, i, y, z, s, b_scratch, c_vec_pack..., d_vec_pack..., d_prev);

			// transposition back to the original form
			transpose(d_vec_pack...);

			store(d, t, i, y, z, s, d_vec_pack...);
		}
	}
}


template <bool multi, typename real_t, typename index_t, typename density_bag_t, typename diag_bag_t,
		  typename scratch_bag_t>
constexpr void xy_fused_transpose_part_2(const density_bag_t d, const index_t s, const index_t z, const index_t n,
										 const diag_bag_t a, const diag_bag_t b, const diag_bag_t c,
										 const scratch_bag_t b_scratch, const index_t y_offset)
{
	constexpr index_t simd_length = 2;

	const index_t y_len = d | noarr::get_length<'y'>();

	const index_t simd_y_len = (y_len - y_offset) / simd_length * simd_length;

	if constexpr (!multi)
	{
		using simd_tag = hn::FixedTag<real_t, 2>;
		simd_tag t;
		using simd_t = hn::Vec<simd_tag>;

		simd_t vec0, vec1;

		xy_fused_transpose_part<simd_t>(d, t, y_offset, y_offset + simd_y_len, s, z, n, a, b, c, b_scratch, vec0, vec1);
	}
	else
	{
		using simd_tag = hn::FixedTag<real_t, 2>;
		simd_tag t;
		using simd_t = hn::Vec<simd_tag>;

		simd_t avec0, avec1;
		simd_t bvec0, bvec1;
		simd_t cvec0, cvec1;
		simd_t dvec0, dvec1;

		xy_fused_transpose_part_multi<simd_t, simd_tag, index_t, density_bag_t, diag_bag_t, scratch_bag_t, simd_t,
									  simd_t>(d, t, y_offset, y_offset + simd_y_len, s, z, n, a, b, c, b_scratch, avec0,
											  avec1, bvec0, bvec1, cvec0, cvec1, dvec0, dvec1);
	}

	for (index_t y = y_offset + simd_y_len; y < y_len; y++)
	{
		x_forward<real_t>(d, y, z, s, n, a, b, c, b_scratch);

		x_backward<real_t>(d, y, z, s, n, c, b_scratch);
	}
}

template <bool multi, typename real_t, typename index_t, typename density_bag_t, typename diag_bag_t,
		  typename scratch_bag_t>
constexpr void xy_fused_transpose_part_4(const density_bag_t d, const index_t s, const index_t z, const index_t n,
										 const diag_bag_t a, const diag_bag_t b, const diag_bag_t c,
										 const scratch_bag_t b_scratch, const index_t y_offset)
{
	constexpr index_t simd_length = 4;

	const index_t y_len = d | noarr::get_length<'y'>();

	const index_t simd_y_len = (y_len - y_offset) / simd_length * simd_length;

	if constexpr (!multi)
	{
		using simd_tag = hn::FixedTag<real_t, 4>;
		simd_tag t;
		using simd_t = hn::Vec<simd_tag>;

		simd_t vec0, vec1, vec2, vec3;

		xy_fused_transpose_part<simd_t>(d, t, y_offset, y_offset + simd_y_len, s, z, n, a, b, c, b_scratch, vec0, vec1,
										vec2, vec3);
	}
	else
	{
		using simd_tag = hn::FixedTag<real_t, 4>;
		simd_tag t;
		using simd_t = hn::Vec<simd_tag>;

		simd_t avec0, avec1, avec2, avec3;
		simd_t bvec0, bvec1, bvec2, bvec3;
		simd_t cvec0, cvec1, cvec2, cvec3;
		simd_t dvec0, dvec1, dvec2, dvec3;

		xy_fused_transpose_part_multi<simd_t, simd_tag, index_t, density_bag_t, diag_bag_t, scratch_bag_t, simd_t,
									  simd_t, simd_t, simd_t>(
			d, t, y_offset, y_offset + simd_y_len, s, z, n, a, b, c, b_scratch, avec0, avec1, avec2, avec3, bvec0,
			bvec1, bvec2, bvec3, cvec0, cvec1, cvec2, cvec3, dvec0, dvec1, dvec2, dvec3);
	}

	if (y_offset + simd_y_len < y_len)
		xy_fused_transpose_part_2<multi, real_t>(d, s, z, n, a, b, c, b_scratch, y_offset + simd_y_len);
}

template <bool multi, typename real_t, typename index_t, typename density_bag_t, typename diag_bag_t,
		  typename scratch_bag_t>
constexpr void xy_fused_transpose_part_8(const density_bag_t d, const index_t s, const index_t z, const index_t n,
										 const diag_bag_t a, const diag_bag_t b, const diag_bag_t c,
										 const scratch_bag_t b_scratch, const index_t y_offset)
{
	constexpr index_t simd_length = 8;

	const index_t y_len = d | noarr::get_length<'y'>();

	const index_t simd_y_len = (y_len - y_offset) / simd_length * simd_length;

	if constexpr (!multi)
	{
		using simd_tag = hn::FixedTag<real_t, 8>;
		simd_tag t;
		using simd_t = hn::Vec<simd_tag>;

		simd_t vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7;

		xy_fused_transpose_part<simd_t>(d, t, y_offset, y_offset + simd_y_len, s, z, n, a, b, c, b_scratch, vec0, vec1,
										vec2, vec3, vec4, vec5, vec6, vec7);
	}
	else
	{
		using simd_tag = hn::FixedTag<real_t, 8>;
		simd_tag t;
		using simd_t = hn::Vec<simd_tag>;

		simd_t avec0, avec1, avec2, avec3, avec4, avec5, avec6, avec7;
		simd_t bvec0, bvec1, bvec2, bvec3, bvec4, bvec5, bvec6, bvec7;
		simd_t cvec0, cvec1, cvec2, cvec3, cvec4, cvec5, cvec6, cvec7;
		simd_t dvec0, dvec1, dvec2, dvec3, dvec4, dvec5, dvec6, dvec7;

		xy_fused_transpose_part_multi<simd_t, simd_tag, index_t, density_bag_t, diag_bag_t, scratch_bag_t, simd_t,
									  simd_t, simd_t, simd_t, simd_t, simd_t, simd_t, simd_t>(
			d, t, y_offset, y_offset + simd_y_len, s, z, n, a, b, c, b_scratch, avec0, avec1, avec2, avec3, avec4,
			avec5, avec6, avec7, bvec0, bvec1, bvec2, bvec3, bvec4, bvec5, bvec6, bvec7, cvec0, cvec1, cvec2, cvec3,
			cvec4, cvec5, cvec6, cvec7, dvec0, dvec1, dvec2, dvec3, dvec4, dvec5, dvec6, dvec7);
	}

	if (y_offset + simd_y_len < y_len)
		xy_fused_transpose_part_4<multi, real_t>(d, s, z, n, a, b, c, b_scratch, y_offset + simd_y_len);
}

template <bool multi, typename real_t, typename index_t, typename density_bag_t, typename diag_bag_t,
		  typename scratch_bag_t>
constexpr void xy_fused_transpose_part_16(const density_bag_t d, const index_t s, const index_t z, const index_t n,
										  const diag_bag_t a, const diag_bag_t b, const diag_bag_t c,
										  const scratch_bag_t b_scratch, const index_t y_offset)
{
	constexpr index_t simd_length = 16;

	const index_t y_len = d | noarr::get_length<'y'>();

	const index_t simd_y_len = (y_len - y_offset) / simd_length * simd_length;

	if constexpr (!multi)
	{
		using simd_tag = hn::FixedTag<real_t, 16>;
		simd_tag t;
		using simd_t = hn::Vec<simd_tag>;

		simd_t vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, vec9, vec10, vec11, vec12, vec13, vec14, vec15;

		xy_fused_transpose_part<simd_t>(d, t, y_offset, y_offset + simd_y_len, s, z, n, a, b, c, b_scratch, vec0, vec1,
										vec2, vec3, vec4, vec5, vec6, vec7, vec8, vec9, vec10, vec11, vec12, vec13,
										vec14, vec15);
	}
	else
	{
		using simd_tag = hn::FixedTag<real_t, 16>;
		simd_tag t;
		using simd_t = hn::Vec<simd_tag>;

		simd_t avec0, avec1, avec2, avec3, avec4, avec5, avec6, avec7, avec8, avec9, avec10, avec11, avec12, avec13,
			avec14, avec15;
		simd_t bvec0, bvec1, bvec2, bvec3, bvec4, bvec5, bvec6, bvec7, bvec8, bvec9, bvec10, bvec11, bvec12, bvec13,
			bvec14, bvec15;
		simd_t cvec0, cvec1, cvec2, cvec3, cvec4, cvec5, cvec6, cvec7, cvec8, cvec9, cvec10, cvec11, cvec12, cvec13,
			cvec14, cvec15;
		simd_t dvec0, dvec1, dvec2, dvec3, dvec4, dvec5, dvec6, dvec7, dvec8, dvec9, dvec10, dvec11, dvec12, dvec13,
			dvec14, dvec15;

		xy_fused_transpose_part_multi<simd_t, simd_tag, index_t, density_bag_t, diag_bag_t, scratch_bag_t, simd_t,
									  simd_t, simd_t, simd_t, simd_t, simd_t, simd_t, simd_t, simd_t, simd_t, simd_t,
									  simd_t, simd_t, simd_t, simd_t, simd_t>(
			d, t, y_offset, y_offset + simd_y_len, s, z, n, a, b, c, b_scratch, avec0, avec1, avec2, avec3, avec4,
			avec5, avec6, avec7, avec8, avec9, avec10, avec11, avec12, avec13, avec14, avec15, bvec0, bvec1, bvec2,
			bvec3, bvec4, bvec5, bvec6, bvec7, bvec8, bvec9, bvec10, bvec11, bvec12, bvec13, bvec14, bvec15, cvec0,
			cvec1, cvec2, cvec3, cvec4, cvec5, cvec6, cvec7, cvec8, cvec9, cvec10, cvec11, cvec12, cvec13, cvec14,
			cvec15, dvec0, dvec1, dvec2, dvec3, dvec4, dvec5, dvec6, dvec7, dvec8, dvec9, dvec10, dvec11, dvec12,
			dvec13, dvec14, dvec15);
	}

	if (y_offset + simd_y_len < y_len)
		xy_fused_transpose_part_8<multi, real_t>(d, s, z, n, a, b, c, b_scratch, y_offset + simd_y_len);
}

template <bool multi, typename real_t, typename index_t, typename density_bag_t, typename diag_bag_t,
		  typename scratch_bag_t, std::enable_if_t<HWY_MAX_LANES_V(hn::Vec<hn::ScalableTag<real_t>>) == 2, bool> = true>
constexpr void xy_fused_transpose_part_dispatch(const density_bag_t d, const index_t s, const index_t z,
												const index_t n, const diag_bag_t a, const diag_bag_t b,
												const diag_bag_t c, const scratch_bag_t b_scratch)
{
	xy_fused_transpose_part_2<multi, real_t>(d, s, z, n, a, b, c, b_scratch, 0);
}

template <bool multi, typename real_t, typename index_t, typename density_bag_t, typename diag_bag_t,
		  typename scratch_bag_t, std::enable_if_t<HWY_MAX_LANES_V(hn::Vec<hn::ScalableTag<real_t>>) == 4, bool> = true>
constexpr void xy_fused_transpose_part_dispatch(const density_bag_t d, const index_t s, const index_t z,
												const index_t n, const diag_bag_t a, const diag_bag_t b,
												const diag_bag_t c, const scratch_bag_t b_scratch)
{
	HWY_LANES_CONSTEXPR index_t max_length = hn::Lanes(hn::ScalableTag<real_t> {});
	HWY_LANES_CONSTEXPR index_t simd_length = std::min(16, max_length);

	if HWY_LANES_CONSTEXPR (simd_length == 2)
	{
		xy_fused_transpose_part_2<multi, real_t>(d, s, z, n, a, b, c, b_scratch, 0);
	}
	else if HWY_LANES_CONSTEXPR (simd_length == 4)
	{
		xy_fused_transpose_part_4<multi, real_t>(d, s, z, n, a, b, c, b_scratch, 0);
	}
}

template <bool multi, typename real_t, typename index_t, typename density_bag_t, typename diag_bag_t,
		  typename scratch_bag_t, std::enable_if_t<HWY_MAX_LANES_V(hn::Vec<hn::ScalableTag<real_t>>) == 8, bool> = true>
constexpr void xy_fused_transpose_part_dispatch(const density_bag_t d, const index_t s, const index_t z,
												const index_t n, const diag_bag_t a, const diag_bag_t b,
												const diag_bag_t c, const scratch_bag_t b_scratch)
{
	HWY_LANES_CONSTEXPR index_t max_length = hn::Lanes(hn::ScalableTag<real_t> {});
	HWY_LANES_CONSTEXPR index_t simd_length = std::min(16, max_length);

	if HWY_LANES_CONSTEXPR (simd_length == 2)
	{
		xy_fused_transpose_part_2<multi, real_t>(d, s, z, n, a, b, c, b_scratch, 0);
	}
	else if HWY_LANES_CONSTEXPR (simd_length == 4)
	{
		xy_fused_transpose_part_4<multi, real_t>(d, s, z, n, a, b, c, b_scratch, 0);
	}
	else if HWY_LANES_CONSTEXPR (simd_length == 8)
	{
		xy_fused_transpose_part_8<multi, real_t>(d, s, z, n, a, b, c, b_scratch, 0);
	}
}

template <bool multi, typename real_t, typename index_t, typename density_bag_t, typename diag_bag_t,
		  typename scratch_bag_t,
		  std::enable_if_t<HWY_MAX_LANES_V(hn::Vec<hn::ScalableTag<real_t>>) >= 16, bool> = true>
constexpr void xy_fused_transpose_part_dispatch(const density_bag_t d, const index_t s, const index_t z,
												const index_t n, const diag_bag_t a, const diag_bag_t b,
												const diag_bag_t c, const scratch_bag_t b_scratch)
{
	HWY_LANES_CONSTEXPR index_t max_length = hn::Lanes(hn::ScalableTag<real_t> {});
	HWY_LANES_CONSTEXPR index_t simd_length = std::min(16, max_length);

	if HWY_LANES_CONSTEXPR (simd_length == 2)
	{
		xy_fused_transpose_part_2<multi, real_t>(d, s, z, n, a, b, c, b_scratch, 0);
	}
	else if HWY_LANES_CONSTEXPR (simd_length == 4)
	{
		xy_fused_transpose_part_4<multi, real_t>(d, s, z, n, a, b, c, b_scratch, 0);
	}
	else if HWY_LANES_CONSTEXPR (simd_length == 8)
	{
		xy_fused_transpose_part_8<multi, real_t>(d, s, z, n, a, b, c, b_scratch, 0);
	}
	else if HWY_LANES_CONSTEXPR (simd_length == 16)
	{
		xy_fused_transpose_part_16<multi, real_t>(d, s, z, n, a, b, c, b_scratch, 0);
	}
}


template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t,
		  typename scratch_layout_t>
static void solve_slice_x_2d_and_3d_transpose_l(real_t* __restrict__ densities, const real_t* __restrict__ a,
												const real_t* __restrict__ b, const real_t* __restrict__ c,
												real_t* __restrict__ b_scratch, const density_layout_t dens_l,
												const diagonal_layout_t diag_l, const scratch_layout_t scratch_l,
												const index_t s, const index_t z, index_t n)
{
	const auto a_bag = noarr::make_bag(diag_l, a);
	const auto b_bag = noarr::make_bag(diag_l, b);
	const auto c_bag = noarr::make_bag(diag_l, c);

	const auto d_bag = noarr::make_bag(dens_l, densities);

	const auto b_scratch_bag = noarr::make_bag(scratch_l, b_scratch);

	xy_fused_transpose_part_dispatch<false, real_t>(d_bag, s, z, n, a_bag, b_bag, c_bag, b_scratch_bag);
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_x_2d_and_3d_transpose(real_t* __restrict__ densities, const real_t* __restrict__ a,
											  const real_t* __restrict__ b, const real_t* __restrict__ c,
											  real_t* __restrict__ b_scratch, const density_layout_t dens_l,
											  const diagonal_layout_t diag_l, const index_t s, const index_t z,
											  index_t n)
{
	const auto a_bag = noarr::make_bag(dens_l ^ noarr::fix<'s', 'z'>(s, z), a);
	const auto b_bag = noarr::make_bag(dens_l ^ noarr::fix<'s', 'z'>(s, z), b);
	const auto c_bag = noarr::make_bag(dens_l ^ noarr::fix<'s', 'z'>(s, z), c);

	const auto d_bag = noarr::make_bag(dens_l ^ noarr::fix<'s', 'z'>(s, z) ^ noarr::slice<'x'>(n), densities);

	const auto b_scratch_bag = noarr::make_bag(diag_l ^ noarr::fix<'v'>(noarr::lit<0>), b_scratch);

	xy_fused_transpose_part_dispatch<true, real_t>(d_bag, s, z, n, a_bag, b_bag, c_bag, b_scratch_bag);
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_y_3d(real_t* __restrict__ densities, const real_t* __restrict__ a, const real_t* __restrict__ b,
							 const real_t* __restrict__ c, real_t* __restrict__ b_scratch,
							 const density_layout_t dens_l, const diagonal_layout_t diag_l, const index_t s_idx,
							 const index_t z, index_t x_tile_size)
{
	const index_t n = dens_l | noarr::get_length<'y'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();

	auto a_bag = noarr::make_bag(dens_l, a);
	auto b_bag = noarr::make_bag(dens_l, b);
	auto c_bag = noarr::make_bag(dens_l, c);
	auto d = noarr::make_bag(dens_l, densities);
	auto scratch = noarr::make_bag(diag_l, b_scratch);

	const index_t x_block_len = (n + x_tile_size - 1) / x_tile_size;

	for (index_t X = 0; X < x_block_len; X++)
	{
		const auto remainder = x_len % x_tile_size;
		const auto x_len_remainder = remainder == 0 ? x_tile_size : remainder;
		const auto tile_size = X == x_block_len - 1 ? x_len_remainder : x_tile_size;

		for (index_t x = 0; x < tile_size; x++)
		{
			scratch[noarr::idx<'v', 'y'>(x, 0)] =
				1 / b_bag[noarr::idx<'s', 'z', 'y', 'x'>(s_idx, z, 0, X * x_tile_size + x)];
		}

		for (index_t i = 1; i < n; i++)
			for (index_t s = 0; s < tile_size; s++)
			{
				auto idx = noarr::idx<'s', 'z', 'y'>(s_idx, z, i);
				auto prev_idx = noarr::idx<'s', 'z', 'y'>(s_idx, z, i - 1);

				auto scratch_idx = noarr::idx<'v'>(s);
				auto dens_idx = noarr::idx<'x'>(X * x_tile_size + s);

				auto r = a_bag[idx & dens_idx] * scratch[prev_idx & scratch_idx];

				scratch[idx & scratch_idx] = 1 / (b_bag[idx & dens_idx] - c_bag[prev_idx & dens_idx] * r);

				d[idx & dens_idx] -= r * d[prev_idx & dens_idx];

				// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
			}

		for (index_t s = 0; s < tile_size; s++)
		{
			auto idx = noarr::idx<'s', 'z', 'y'>(s_idx, z, n - 1);

			auto scratch_idx = noarr::idx<'v'>(s);
			auto dens_idx = noarr::idx<'x'>(X * x_tile_size + s);

			d[idx & dens_idx] *= scratch[idx & scratch_idx];

			// std::cout << "n-1: " << (dens_l | noarr::get_at<'x', 's'>(densities, n - 1, s)) << std::endl;
		}

		for (index_t i = n - 2; i >= 0; i--)
			for (index_t s = 0; s < tile_size; s++)
			{
				auto idx = noarr::idx<'s', 'z', 'y'>(s_idx, z, i);
				auto next_idx = noarr::idx<'s', 'z', 'y'>(s_idx, z, i + 1);

				auto scratch_idx = noarr::idx<'v'>(s);
				auto dens_idx = noarr::idx<'x'>(X * x_tile_size + s);

				d[idx & dens_idx] =
					(d[idx & dens_idx] - c_bag[idx & dens_idx] * d[next_idx & dens_idx]) * scratch[idx & scratch_idx];

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

	auto a_bag = noarr::make_bag(dens_l, a);
	auto b_bag = noarr::make_bag(dens_l, b);
	auto c_bag = noarr::make_bag(dens_l, c);
	auto d = noarr::make_bag(dens_l, densities);
	auto scratch = noarr::make_bag(diag_l, b_scratch);

	const index_t x_block_len = (n + x_tile_size - 1) / x_tile_size;

#pragma omp for schedule(static) nowait collapse(2)
	for (index_t y = 0; y < y_len; y++)
		for (index_t X = 0; X < x_block_len; X++)
		{
			const auto remainder = x_len % x_tile_size;
			const auto x_len_remainder = remainder == 0 ? x_tile_size : remainder;
			const auto tile_size = X == x_block_len - 1 ? x_len_remainder : x_tile_size;

			for (index_t s = 0; s < tile_size; s++)
			{
				scratch[noarr::idx<'v', 'z'>(s, 0)] =
					1 / b_bag[noarr::idx<'s', 'z', 'y', 'x'>(s_idx, 0, y, X * x_tile_size + s)];
			}

			for (index_t i = 1; i < n; i++)
				for (index_t s = 0; s < tile_size; s++)
				{
					auto idx = noarr::idx<'s', 'z', 'y'>(s_idx, i, y);
					auto prev_idx = noarr::idx<'s', 'z', 'y'>(s_idx, i - 1, y);

					auto scratch_idx = noarr::idx<'v'>(s);
					auto dens_idx = noarr::idx<'x'>(X * x_tile_size + s);

					auto r = a_bag[idx & dens_idx] * scratch[prev_idx & scratch_idx];

					scratch[idx & scratch_idx] = 1 / (b_bag[idx & dens_idx] - c_bag[prev_idx & dens_idx] * r);

					d[idx & dens_idx] -= r * d[prev_idx & dens_idx];

					// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
				}

			for (index_t s = 0; s < tile_size; s++)
			{
				auto idx = noarr::idx<'s', 'z', 'y'>(s_idx, n - 1, y);

				auto scratch_idx = noarr::idx<'v'>(s);
				auto dens_idx = noarr::idx<'x'>(X * x_tile_size + s);

				d[idx & dens_idx] *= scratch[idx & scratch_idx];

				// std::cout << "n-1: " << (dens_l | noarr::get_at<'x', 's'>(densities, n - 1, s)) << std::endl;
			}

			for (index_t i = n - 2; i >= 0; i--)
				for (index_t s = 0; s < tile_size; s++)
				{
					auto idx = noarr::idx<'s', 'z', 'y'>(s_idx, i, y);
					auto next_idx = noarr::idx<'s', 'z', 'y'>(s_idx, i + 1, y);

					auto scratch_idx = noarr::idx<'v'>(s);
					auto dens_idx = noarr::idx<'x'>(X * x_tile_size + s);

					d[idx & dens_idx] = (d[idx & dens_idx] - c_bag[idx & dens_idx] * d[next_idx & dens_idx])
										* scratch[idx & scratch_idx];

					// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
				}
		}
}

template <typename real_t, bool aligned_x>
void sdd_partial_blocking<real_t, aligned_x>::solve_x()
{
	if (this->problem_.dims == 1) {}
	else if (this->problem_.dims == 2)
	{
#pragma omp parallel for schedule(static)
		for (index_t s = 0; s < this->problem_.substrates_count; s++)
			for (index_t i = 0; i < this->problem_.iterations; i++)
				if (continuous_x_diagonal_)
					solve_slice_x_2d_and_3d_transpose_l<index_t>(
						this->substrates_, ax_, bx_, cx_, b_scratch_[get_thread_num()], get_substrates_layout<2>(),
						get_diag_layout_x(), get_scratch_layout<'x'>(), s, 0, this->problem_.nx);
				else
					solve_slice_x_2d_and_3d_transpose<index_t>(this->substrates_, ax_, bx_, cx_,
															   b_scratch_[get_thread_num()], get_substrates_layout<2>(),
															   get_scratch_layout<'x'>(), s, 0, this->problem_.nx);
	}
	else if (this->problem_.dims == 3)
	{
#pragma omp parallel
		for (index_t s = 0; s < this->problem_.substrates_count; s++)
			for (index_t i = 0; i < this->problem_.iterations; i++)
#pragma omp for schedule(static) nowait
				for (index_t z = 0; z < this->problem_.nz; z++)
					if (continuous_x_diagonal_)
						solve_slice_x_2d_and_3d_transpose_l<index_t>(
							this->substrates_, ax_, bx_, cx_, b_scratch_[get_thread_num()], get_substrates_layout<3>(),
							get_diag_layout_x(), get_scratch_layout<'x'>(), s, z, this->problem_.nx);
					else
						solve_slice_x_2d_and_3d_transpose<index_t>(
							this->substrates_, ax_, bx_, cx_, b_scratch_[get_thread_num()], get_substrates_layout<3>(),
							get_scratch_layout<'x'>(), s, z, this->problem_.nx);
	}
}

template <typename real_t, bool aligned_x>
void sdd_partial_blocking<real_t, aligned_x>::solve_y()
{
	if (this->problem_.dims == 2)
	{
#pragma omp parallel for schedule(static)
		for (index_t s = 0; s < this->problem_.substrates_count; s++)
			for (index_t i = 0; i < this->problem_.iterations; i++)
				solve_slice_y_3d<index_t>(this->substrates_, ay_, by_, cy_, b_scratch_[get_thread_num()],
										  get_substrates_layout<2>(), get_scratch_layout<'y'>(), s, 0, x_tile_size_);
	}
	else if (this->problem_.dims == 3)
	{
#pragma omp parallel
		for (index_t s = 0; s < this->problem_.substrates_count; s++)
			for (index_t i = 0; i < this->problem_.iterations; i++)
#pragma omp for schedule(static) nowait
				for (index_t z = 0; z < this->problem_.nz; z++)
					solve_slice_y_3d<index_t>(this->substrates_, ay_, by_, cy_, b_scratch_[get_thread_num()],
											  get_substrates_layout<3>(), get_scratch_layout<'y'>(), s, z,
											  x_tile_size_);
	}
}

template <typename real_t, bool aligned_x>
void sdd_partial_blocking<real_t, aligned_x>::solve_z()
{
#pragma omp parallel
	{
		for (index_t s = 0; s < this->problem_.substrates_count; s++)
			for (index_t i = 0; i < this->problem_.iterations; i++)
				solve_slice_z_3d<index_t>(this->substrates_, az_, bz_, cz_, b_scratch_[get_thread_num()],
										  get_substrates_layout<3>(), get_scratch_layout<'z'>(), s, x_tile_size_);
	}
}

template <typename real_t, bool aligned_x>
void sdd_partial_blocking<real_t, aligned_x>::solve()
{
	if (this->problem_.dims == 1) {}
	if (this->problem_.dims == 2)
	{
#pragma omp parallel
		{
			perf_counter counter("sdd-pb");

#pragma omp for schedule(static) nowait
			for (index_t s = 0; s < this->problem_.substrates_count; s++)
			{
				for (index_t i = 0; i < this->problem_.iterations; i++)
				{
					if (continuous_x_diagonal_)
						solve_slice_x_2d_and_3d_transpose_l<index_t>(
							this->substrates_, ax_, bx_, cx_, b_scratch_[get_thread_num()], get_substrates_layout<2>(),
							get_diag_layout_x(), get_scratch_layout<'x'>(), s, 0, this->problem_.nx);
					else
						solve_slice_x_2d_and_3d_transpose<index_t>(
							this->substrates_, ax_, bx_, cx_, b_scratch_[get_thread_num()], get_substrates_layout<2>(),
							get_scratch_layout<'x'>(), s, 0, this->problem_.nx);

					solve_slice_y_3d<index_t>(this->substrates_, ay_, by_, cy_, b_scratch_[get_thread_num()],
											  get_substrates_layout<2>(), get_scratch_layout<'y'>(), s, 0,
											  x_tile_size_);
				}
			}
		}
	}
	if (this->problem_.dims == 3)
	{
#pragma omp parallel
		{
			perf_counter counter("sdd-pb");

			for (index_t s = 0; s < this->problem_.substrates_count; s++)
			{
				for (index_t i = 0; i < this->problem_.iterations; i++)
				{
#pragma omp for schedule(static) nowait
					for (index_t z = 0; z < this->problem_.nz; z++)
					{
						if (continuous_x_diagonal_)
							solve_slice_x_2d_and_3d_transpose_l<index_t>(
								this->substrates_, ax_, bx_, cx_, b_scratch_[get_thread_num()],
								get_substrates_layout<3>(), get_diag_layout_x(), get_scratch_layout<'x'>(), s, z,
								this->problem_.nx);
						else
							solve_slice_x_2d_and_3d_transpose<index_t>(
								this->substrates_, ax_, bx_, cx_, b_scratch_[get_thread_num()],
								get_substrates_layout<3>(), get_scratch_layout<'x'>(), s, z, this->problem_.nx);

						solve_slice_y_3d<index_t>(this->substrates_, ay_, by_, cy_, b_scratch_[get_thread_num()],
												  get_substrates_layout<3>(), get_scratch_layout<'y'>(), s, z,
												  x_tile_size_);
					}
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
sdd_partial_blocking<real_t, aligned_x>::sdd_partial_blocking()
{}

template <typename real_t, bool aligned_x>
sdd_partial_blocking<real_t, aligned_x>::~sdd_partial_blocking()
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

template class sdd_partial_blocking<float, false>;
template class sdd_partial_blocking<double, false>;

template class sdd_partial_blocking<float, true>;
template class sdd_partial_blocking<double, true>;
