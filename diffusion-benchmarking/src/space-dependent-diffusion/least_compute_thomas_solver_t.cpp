#include "least_compute_thomas_solver_t.h"

#include <cstddef>

#include "../perf_utils.h"

template <typename real_t, bool aligned_x>
void sdd_least_compute_thomas_solver_t<real_t, aligned_x>::precompute_values(real_t*& a, real_t*& b, real_t*& c,
																			 index_t shape, index_t n, index_t dims,
																			 char dim)
{
	auto substrates_layout = get_substrates_layout();

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
void sdd_least_compute_thomas_solver_t<real_t, aligned_x>::prepare(const max_problem_t& problem)
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
void sdd_least_compute_thomas_solver_t<real_t, aligned_x>::tune(const nlohmann::json& params)
{
	xs_tile_size_ = params.contains("xs_tile_size") ? (std::size_t)params["xs_tile_size"] : 48;

	alignment_size_ = params.contains("alignment_size") ? (std::size_t)params["alignment_size"] : 64;
}

template <typename real_t, bool aligned_x>
void sdd_least_compute_thomas_solver_t<real_t, aligned_x>::initialize()
{
	precompute_values(ax_, bx_, cx_, this->problem_.dx, this->problem_.nx, this->problem_.dims, 'x');
	precompute_values(ay_, by_, cy_, this->problem_.dy, this->problem_.ny, this->problem_.dims, 'y');
	precompute_values(az_, bz_, cz_, this->problem_.dz, this->problem_.nz, this->problem_.dims, 'z');

	auto diag_l = get_diagonal_layout<'x'>();

	for (int i = 0; i < get_max_threads(); i++)
	{
		if (aligned_x)
			b_scratch_.push_back((real_t*)std::aligned_alloc(alignment_size_, (diag_l | noarr::get_size())));
		else
			b_scratch_.push_back((real_t*)std::malloc((diag_l | noarr::get_size())));
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_x_1d(real_t* __restrict__ densities, const real_t* __restrict__ a, const real_t* __restrict__ b,
							 const real_t* __restrict__ c, real_t* __restrict__ b_scratch,
							 const density_layout_t dens_l, const diagonal_layout_t diag_l)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'x'>();

	auto a_bag = noarr::make_bag(dens_l, a);
	auto b_bag = noarr::make_bag(dens_l, b);
	auto c_bag = noarr::make_bag(dens_l, c);

	auto d = noarr::make_bag(dens_l, densities);

	auto scratch = noarr::make_bag(diag_l, b_scratch);

#pragma omp for schedule(static) nowait
	for (index_t s = 0; s < substrates_count; s++)
	{
		auto idx = noarr::idx<'s', 'x'>(s, 0);
		scratch[idx] = b_bag[idx];
	}

	for (index_t i = 1; i < n; i++)
	{
#pragma omp for schedule(static) nowait
		for (index_t s = 0; s < substrates_count; s++)
		{
			auto idx = noarr::idx<'s', 'x'>(s, i);
			auto prev_idx = noarr::idx<'s', 'x'>(s, i - 1);

			auto r = a_bag[idx] / scratch[prev_idx];

			scratch[idx] = b_bag[idx] - c_bag[prev_idx] * r;

			d[idx] -= r * d[prev_idx];

			// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
		}
	}

#pragma omp for schedule(static) nowait
	for (index_t s = 0; s < substrates_count; s++)
	{
		auto idx = noarr::idx<'s', 'x'>(s, n - 1);
		d[idx] /= scratch[idx];

		// std::cout << "n-1: " << (dens_l | noarr::get_at<'x', 's'>(densities, n - 1, s)) << std::endl;
	}

	for (index_t i = n - 2; i >= 0; i--)
	{
#pragma omp for schedule(static) nowait
		for (index_t s = 0; s < substrates_count; s++)
		{
			auto idx = noarr::idx<'s', 'x'>(s, i);
			auto next_idx = noarr::idx<'s', 'x'>(s, i + 1);

			d[idx] = (d[idx] - c_bag[idx] * d[next_idx]) / scratch[idx];

			// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_x_2d_and_3d(real_t* __restrict__ densities, const real_t* __restrict__ a,
									const real_t* __restrict__ b, const real_t* __restrict__ c,
									real_t* __restrict__ b_scratch, const density_layout_t dens_l,
									const diagonal_layout_t diag_l)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'x'>();
	const index_t m = dens_l | noarr::get_length<'m'>();

	auto a_bag = noarr::make_bag(dens_l, a);
	auto b_bag = noarr::make_bag(dens_l, b);
	auto c_bag = noarr::make_bag(dens_l, c);

	auto d = noarr::make_bag(dens_l, densities);

	auto scratch = noarr::make_bag(diag_l, b_scratch);

#pragma omp for schedule(static) nowait
	for (index_t yz = 0; yz < m; yz++)
	{
		for (index_t s = 0; s < substrates_count; s++)
		{
			auto idx = noarr::idx<'s', 'm', 'x'>(s, yz, 0);
			scratch[idx] = 1 / b_bag[idx];
		}

		for (index_t i = 1; i < n; i++)
		{
			for (index_t s = 0; s < substrates_count; s++)
			{
				auto idx = noarr::idx<'s', 'm', 'x'>(s, yz, i);
				auto prev_idx = noarr::idx<'s', 'm', 'x'>(s, yz, i - 1);

				auto r = a_bag[idx] * scratch[prev_idx];

				scratch[idx] = 1 / (b_bag[idx] - c_bag[prev_idx] * r);

				d[idx] -= r * d[prev_idx];

				// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
			}
		}

		for (index_t s = 0; s < substrates_count; s++)
		{
			auto idx = noarr::idx<'s', 'm', 'x'>(s, yz, n - 1);
			d[idx] *= scratch[idx];

			// std::cout << "n-1: " << (dens_l | noarr::get_at<'x', 's'>(densities, n - 1, s)) << std::endl;
		}

		for (index_t i = n - 2; i >= 0; i--)
		{
			for (index_t s = 0; s < substrates_count; s++)
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
static void solve_slice_y_2d(real_t* __restrict__ densities, const real_t* __restrict__ a, const real_t* __restrict__ b,
							 const real_t* __restrict__ c, real_t* __restrict__ b_scratch,
							 const density_layout_t dens_l, const diagonal_layout_t diag_l, std::size_t xs_tile_size)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'y'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();

	auto blocked_dens_l = dens_l ^ noarr::merge_blocks<'x', 's', 'c'>()
						  ^ noarr::into_blocks_dynamic<'c', 'x', 's', 'b'>(xs_tile_size)
						  ^ noarr::fix<'b'>(noarr::lit<0>);

	const auto remainder = (x_len * substrates_count) % xs_tile_size;
	const auto last_tile = remainder == 0 ? xs_tile_size : remainder;

	const index_t x_block_len = blocked_dens_l | noarr::get_length<'x'>();

	auto a_bag = noarr::make_bag(blocked_dens_l, a);
	auto b_bag = noarr::make_bag(blocked_dens_l, b);
	auto c_bag = noarr::make_bag(blocked_dens_l, c);

	auto d = noarr::make_bag(blocked_dens_l, densities);

	auto scratch = noarr::make_bag(diag_l, b_scratch);

#pragma omp for schedule(static) nowait
	for (index_t x = 0; x < x_block_len; x++)
	{
		const index_t tile_size = (x == x_block_len - 1) ? last_tile : xs_tile_size;

		for (index_t s = 0; s < tile_size; s++)
		{
			auto idx = noarr::idx<'s', 'y', 'x'>(s, 0, x);
			scratch[idx] = 1 / b_bag[idx];
		}

		for (index_t i = 1; i < n; i++)
		{
			for (index_t s = 0; s < tile_size; s++)
			{
				auto idx = noarr::idx<'s', 'y', 'x'>(s, i, x);
				auto prev_idx = noarr::idx<'s', 'y', 'x'>(s, i - 1, x);

				auto r = a_bag[idx] * scratch[prev_idx];

				scratch[idx] = 1 / (b_bag[idx] - c_bag[prev_idx] * r);

				d[idx] -= r * d[prev_idx];

				// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
			}
		}

		for (index_t s = 0; s < tile_size; s++)
		{
			auto idx = noarr::idx<'s', 'y', 'x'>(s, n - 1, x);
			d[idx] *= scratch[idx];

			// std::cout << "n-1: " << (dens_l | noarr::get_at<'x', 's'>(densities, n - 1, s)) << std::endl;
		}

		for (index_t i = n - 2; i >= 0; i--)
		{
			for (index_t s = 0; s < tile_size; s++)
			{
				auto idx = noarr::idx<'s', 'y', 'x'>(s, i, x);
				auto next_idx = noarr::idx<'s', 'y', 'x'>(s, i + 1, x);

				d[idx] = (d[idx] - c_bag[idx] * d[next_idx]) * scratch[idx];

				// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_y_3d(real_t* __restrict__ densities, const real_t* __restrict__ a, const real_t* __restrict__ b,
							 const real_t* __restrict__ c, real_t* __restrict__ b_scratch,
							 const density_layout_t dens_l, const diagonal_layout_t diag_l, std::size_t xs_tile_size)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'y'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t z_len = dens_l | noarr::get_length<'z'>();

	auto blocked_dens_l = dens_l ^ noarr::merge_blocks<'x', 's', 'c'>()
						  ^ noarr::into_blocks_dynamic<'c', 'x', 's', 'b'>(xs_tile_size)
						  ^ noarr::fix<'b'>(noarr::lit<0>);

	const auto remainder = (x_len * substrates_count) % xs_tile_size;
	const auto last_tile = remainder == 0 ? xs_tile_size : remainder;

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
			const index_t tile_size = (x == x_block_len - 1) ? last_tile : xs_tile_size;

			for (index_t s = 0; s < tile_size; s++)
			{
				auto idx = noarr::idx<'s', 'z', 'y', 'x'>(s, z, 0, x);
				scratch[idx] = 1 / b_bag[idx];
			}

			for (index_t i = 1; i < n; i++)
			{
				for (index_t s = 0; s < tile_size; s++)
				{
					auto idx = noarr::idx<'s', 'z', 'y', 'x'>(s, z, i, x);
					auto prev_idx = noarr::idx<'s', 'z', 'y', 'x'>(s, z, i - 1, x);

					auto r = a_bag[idx] * scratch[prev_idx];

					scratch[idx] = 1 / (b_bag[idx] - c_bag[prev_idx] * r);

					d[idx] -= r * d[prev_idx];

					// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
				}
			}

			for (index_t s = 0; s < tile_size; s++)
			{
				auto idx = noarr::idx<'s', 'z', 'y', 'x'>(s, z, n - 1, x);
				d[idx] *= scratch[idx];

				// std::cout << "n-1: " << (dens_l | noarr::get_at<'x', 's'>(densities, n - 1, s)) << std::endl;
			}

			for (index_t i = n - 2; i >= 0; i--)
			{
				for (index_t s = 0; s < tile_size; s++)
				{
					auto idx = noarr::idx<'s', 'z', 'y', 'x'>(s, z, i, x);
					auto next_idx = noarr::idx<'s', 'z', 'y', 'x'>(s, z, i + 1, x);

					d[idx] = (d[idx] - c_bag[idx] * d[next_idx]) * scratch[idx];

					// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
				}
			}
		}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_z_3d(real_t* __restrict__ densities, const real_t* __restrict__ a, const real_t* __restrict__ b,
							 const real_t* __restrict__ c, real_t* __restrict__ b_scratch,
							 const density_layout_t dens_l, const diagonal_layout_t diag_l, std::size_t xs_tile_size)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'z'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();

	auto blocked_dens_l = dens_l ^ noarr::merge_blocks<'x', 's', 'c'>()
						  ^ noarr::into_blocks_dynamic<'c', 'x', 's', 'b'>(xs_tile_size)
						  ^ noarr::fix<'b'>(noarr::lit<0>);

	const auto remainder = (x_len * substrates_count) % xs_tile_size;
	const auto last_tile = remainder == 0 ? xs_tile_size : remainder;

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
			const index_t tile_size = (x == x_block_len - 1) ? last_tile : xs_tile_size;

			for (index_t s = 0; s < tile_size; s++)
			{
				auto idx = noarr::idx<'s', 'z', 'y', 'x'>(s, 0, y, x);
				scratch[idx] = 1 / b_bag[idx];
			}

			for (index_t i = 1; i < n; i++)
			{
				for (index_t s = 0; s < tile_size; s++)
				{
					auto idx = noarr::idx<'s', 'z', 'y', 'x'>(s, i, y, x);
					auto prev_idx = noarr::idx<'s', 'z', 'y', 'x'>(s, i - 1, y, x);

					auto r = a_bag[idx] * scratch[prev_idx];

					scratch[idx] = 1 / (b_bag[idx] - c_bag[prev_idx] * r);

					d[idx] -= r * d[prev_idx];

					// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
				}
			}

			for (index_t s = 0; s < tile_size; s++)
			{
				auto idx = noarr::idx<'s', 'z', 'y', 'x'>(s, n - 1, y, x);
				d[idx] *= scratch[idx];

				// std::cout << "n-1: " << (dens_l | noarr::get_at<'x', 's'>(densities, n - 1, s)) << std::endl;
			}

			for (index_t i = n - 2; i >= 0; i--)
			{
				for (index_t s = 0; s < tile_size; s++)
				{
					auto idx = noarr::idx<'s', 'z', 'y', 'x'>(s, i, y, x);
					auto next_idx = noarr::idx<'s', 'z', 'y', 'x'>(s, i + 1, y, x);

					d[idx] = (d[idx] - c_bag[idx] * d[next_idx]) * scratch[idx];

					// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
				}
			}
		}
}

template <typename real_t, bool aligned_x>
void sdd_least_compute_thomas_solver_t<real_t, aligned_x>::solve_x()
{
	if (this->problem_.dims == 1)
	{
#pragma omp parallel
		solve_slice_x_1d<index_t>(this->substrates_, ax_, bx_, cx_, b_scratch_[get_thread_num()],
								  get_substrates_layout<1>(), get_diagonal_layout<'x'>());
	}
	else if (this->problem_.dims == 2)
	{
#pragma omp parallel
		solve_slice_x_2d_and_3d<index_t>(this->substrates_, ax_, bx_, cx_, b_scratch_[get_thread_num()],
										 get_substrates_layout<2>() ^ noarr::rename<'y', 'm'>(),
										 get_diagonal_layout<'x'>());
	}
	else if (this->problem_.dims == 3)
	{
#pragma omp parallel
		solve_slice_x_2d_and_3d<index_t>(this->substrates_, ax_, bx_, cx_, b_scratch_[get_thread_num()],
										 get_substrates_layout<3>() ^ noarr::merge_blocks<'z', 'y', 'm'>(),
										 get_diagonal_layout<'x'>());
	}
}

template <typename real_t, bool aligned_x>
void sdd_least_compute_thomas_solver_t<real_t, aligned_x>::solve_y()
{
	if (this->problem_.dims == 2)
	{
#pragma omp parallel
		solve_slice_y_2d<index_t>(this->substrates_, ay_, by_, cy_, b_scratch_[get_thread_num()],
								  get_substrates_layout<2>(), get_diagonal_layout<'y'>(), xs_tile_size_);
	}
	else if (this->problem_.dims == 3)
	{
#pragma omp parallel
		solve_slice_y_3d<index_t>(this->substrates_, ay_, by_, cy_, b_scratch_[get_thread_num()],
								  get_substrates_layout<3>(), get_diagonal_layout<'y'>(), xs_tile_size_);
	}
}

template <typename real_t, bool aligned_x>
void sdd_least_compute_thomas_solver_t<real_t, aligned_x>::solve_z()
{
#pragma omp parallel
	solve_slice_z_3d<index_t>(this->substrates_, az_, bz_, cz_, b_scratch_[get_thread_num()],
							  get_substrates_layout<3>(), get_diagonal_layout<'z'>(), xs_tile_size_);
}

template <typename real_t, bool aligned_x>
void sdd_least_compute_thomas_solver_t<real_t, aligned_x>::solve()
{
	if (this->problem_.dims == 1)
	{
#pragma omp parallel
		{
			perf_counter counter("sdd-lstct");

			for (index_t i = 0; i < this->problem_.iterations; i++)
				solve_slice_x_1d<index_t>(this->substrates_, ax_, bx_, cx_, b_scratch_[get_thread_num()],
										  get_substrates_layout<1>(), get_diagonal_layout<'x'>());
		}
	}
	else if (this->problem_.dims == 2)
	{
#pragma omp parallel
		{
			perf_counter counter("sdd-lstct");

			for (index_t i = 0; i < this->problem_.iterations; i++)
			{
				solve_slice_x_2d_and_3d<index_t>(this->substrates_, ax_, bx_, cx_, b_scratch_[get_thread_num()],
												 get_substrates_layout<2>() ^ noarr::rename<'y', 'm'>(),
												 get_diagonal_layout<'x'>());
#pragma omp barrier
				solve_slice_y_2d<index_t>(this->substrates_, ay_, by_, cy_, b_scratch_[get_thread_num()],
										  get_substrates_layout<2>(), get_diagonal_layout<'y'>(), xs_tile_size_);
#pragma omp barrier
			}
		}
	}
	else if (this->problem_.dims == 3)
	{
#pragma omp parallel
		{
			perf_counter counter("sdd-lstct");

			for (index_t i = 0; i < this->problem_.iterations; i++)
			{
				solve_slice_x_2d_and_3d<index_t>(this->substrates_, ax_, bx_, cx_, b_scratch_[get_thread_num()],
												 get_substrates_layout<3>() ^ noarr::merge_blocks<'z', 'y', 'm'>(),
												 get_diagonal_layout<'x'>());
#pragma omp barrier
				solve_slice_y_3d<index_t>(this->substrates_, ay_, by_, cy_, b_scratch_[get_thread_num()],
										  get_substrates_layout<3>(), get_diagonal_layout<'y'>(), xs_tile_size_);
#pragma omp barrier
				solve_slice_z_3d<index_t>(this->substrates_, az_, bz_, cz_, b_scratch_[get_thread_num()],
										  get_substrates_layout<3>(), get_diagonal_layout<'z'>(), xs_tile_size_);
#pragma omp barrier
			}
		}
	}
}


template <typename real_t, bool aligned_x>
sdd_least_compute_thomas_solver_t<real_t, aligned_x>::~sdd_least_compute_thomas_solver_t()
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

template class sdd_least_compute_thomas_solver_t<float, false>;
template class sdd_least_compute_thomas_solver_t<double, false>;

template class sdd_least_compute_thomas_solver_t<float, true>;
template class sdd_least_compute_thomas_solver_t<double, true>;
