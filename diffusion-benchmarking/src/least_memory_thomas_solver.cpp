#include "least_memory_thomas_solver.h"

#include "perf_utils.h"

template <typename real_t>
void least_memory_thomas_solver<real_t>::precompute_values(std::unique_ptr<real_t[]>& a, std::unique_ptr<real_t[]>& b1,
														   std::unique_ptr<real_t[]>& b, index_t shape, index_t dims,
														   index_t n)
{
	a = std::make_unique<real_t[]>(this->problem_.substrates_count);
	b1 = std::make_unique<real_t[]>(this->problem_.substrates_count);
	b = std::make_unique<real_t[]>(n * this->problem_.substrates_count);

	auto layout = get_diagonal_layout(this->problem_, n);

	auto b_diag = noarr::make_bag(layout, b.get());

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

template <typename real_t>
void least_memory_thomas_solver<real_t>::initialize()
{
	if (this->problem_.dims >= 1)
		precompute_values(ax_, b1x_, bx_, this->problem_.dx, this->problem_.dims, this->problem_.nx);
	if (this->problem_.dims >= 2)
		precompute_values(ay_, b1y_, by_, this->problem_.dy, this->problem_.dims, this->problem_.ny);
	if (this->problem_.dims >= 3)
		precompute_values(az_, b1z_, bz_, this->problem_.dz, this->problem_.dims, this->problem_.nz);
}

template <typename real_t>
auto least_memory_thomas_solver<real_t>::get_diagonal_layout(const problem_t<index_t, real_t>& problem, index_t n)
{
	return noarr::scalar<real_t>() ^ noarr::vectors<'i', 's'>(n, problem.substrates_count);
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_x_1d(real_t* __restrict__ densities, const real_t* __restrict__ a,
							 const real_t* __restrict__ b1, const real_t* __restrict__ b, const density_layout_t dens_l,
							 const diagonal_layout_t diag_l)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'x'>();

#pragma omp for schedule(static) nowait
	for (index_t s = 0; s < substrates_count; s++)
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
static void solve_slice_x_2d_and_3d(real_t* __restrict__ densities, const real_t* __restrict__ a,
									const real_t* __restrict__ b1, const real_t* __restrict__ b,
									const density_layout_t dens_l, const diagonal_layout_t diag_l)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'x'>();
	const index_t m = dens_l | noarr::get_length<'m'>();

#pragma omp for schedule(static) collapse(2) nowait
	for (index_t s = 0; s < substrates_count; s++)
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
							 const diagonal_layout_t diag_l)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'y'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();

	for (index_t s = 0; s < substrates_count; s++)
	{
		const real_t a_s = a[s];
		const real_t b1_s = b1[s];
		real_t b_tmp = a_s / (b1_s + a_s);

		for (index_t i = 1; i < n; i++)
		{
#pragma omp for schedule(static) nowait
			for (index_t x = 0; x < x_len; x++)
			{
				(dens_l | noarr::get_at<'x', 'y', 's'>(densities, x, i, s)) -=
					(dens_l | noarr::get_at<'x', 'y', 's'>(densities, x, i - 1, s)) * b_tmp;
			}

			b_tmp = a_s / (b1_s - a_s * b_tmp);
		}

#pragma omp for schedule(static) nowait
		for (index_t x = 0; x < x_len; x++)
		{
			(dens_l | noarr::get_at<'x', 'y', 's'>(densities, x, n - 1, s)) =
				(dens_l | noarr::get_at<'x', 'y', 's'>(densities, x, n - 1, s))
				* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
		}

		for (index_t i = n - 2; i >= 0; i--)
		{
#pragma omp for schedule(static) nowait
			for (index_t x = 0; x < x_len; x++)
			{
				(dens_l | noarr::get_at<'x', 'y', 's'>(densities, x, i, s)) =
					((dens_l | noarr::get_at<'x', 'y', 's'>(densities, x, i, s))
					 - a_s * (dens_l | noarr::get_at<'x', 'y', 's'>(densities, x, i + 1, s)))
					* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_y_3d(real_t* __restrict__ densities, const real_t* __restrict__ a,
							 const real_t* __restrict__ b1, const real_t* __restrict__ b, const density_layout_t dens_l,
							 const diagonal_layout_t diag_l)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'y'>();
	const index_t z_len = dens_l | noarr::get_length<'z'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();

#pragma omp for schedule(static) collapse(2) nowait
	for (index_t s = 0; s < substrates_count; s++)
	{
		for (index_t z = 0; z < z_len; z++)
		{
			const real_t a_s = a[s];
			const real_t b1_s = b1[s];
			real_t b_tmp = a_s / (b1_s + a_s);

			for (index_t i = 1; i < n; i++)
			{
				for (index_t x = 0; x < x_len; x++)
				{
					(dens_l | noarr::get_at<'x', 'z', 'y', 's'>(densities, x, z, i, s)) -=
						(dens_l | noarr::get_at<'x', 'z', 'y', 's'>(densities, x, z, i - 1, s)) * b_tmp;
				}

				b_tmp = a_s / (b1_s - a_s * b_tmp);
			}

			for (index_t x = 0; x < x_len; x++)
			{
				(dens_l | noarr::get_at<'x', 'z', 'y', 's'>(densities, x, z, n - 1, s)) =
					(dens_l | noarr::get_at<'x', 'z', 'y', 's'>(densities, x, z, n - 1, s))
					* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
			}

			for (index_t i = n - 2; i >= 0; i--)
			{
				for (index_t x = 0; x < x_len; x++)
				{
					(dens_l | noarr::get_at<'x', 'z', 'y', 's'>(densities, x, z, i, s)) =
						((dens_l | noarr::get_at<'x', 'z', 'y', 's'>(densities, x, z, i, s))
						 - a_s * (dens_l | noarr::get_at<'x', 'z', 'y', 's'>(densities, x, z, i + 1, s)))
						* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
				}
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_z_3d(real_t* __restrict__ densities, const real_t* __restrict__ a,
							 const real_t* __restrict__ b1, const real_t* __restrict__ b, const density_layout_t dens_l,
							 const diagonal_layout_t diag_l)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'z'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();

	for (index_t s = 0; s < substrates_count; s++)
	{
		const real_t a_s = a[s];
		const real_t b1_s = b1[s];
		real_t b_tmp = a_s / (b1_s + a_s);

		for (index_t i = 1; i < n; i++)
		{
#pragma omp for schedule(static) nowait
			for (index_t y = 0; y < y_len; y++)
			{
				for (index_t x = 0; x < x_len; x++)
				{
					(dens_l | noarr::get_at<'x', 'y', 'z', 's'>(densities, x, y, i, s)) -=
						(dens_l | noarr::get_at<'x', 'y', 'z', 's'>(densities, x, y, i - 1, s)) * b_tmp;
				}
			}

			b_tmp = a_s / (b1_s - a_s * b_tmp);
		}

#pragma omp for schedule(static) nowait
		for (index_t y = 0; y < y_len; y++)
		{
			for (index_t x = 0; x < x_len; x++)
			{
				(dens_l | noarr::get_at<'x', 'y', 'z', 's'>(densities, x, y, n - 1, s)) =
					(dens_l | noarr::get_at<'x', 'y', 'z', 's'>(densities, x, y, n - 1, s))
					* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
			}
		}

		for (index_t i = n - 2; i >= 0; i--)
		{
#pragma omp for schedule(static) nowait
			for (index_t y = 0; y < y_len; y++)
			{
				for (index_t x = 0; x < x_len; x++)
				{
					(dens_l | noarr::get_at<'x', 'y', 'z', 's'>(densities, x, y, i, s)) =
						((dens_l | noarr::get_at<'x', 'y', 'z', 's'>(densities, x, y, i, s))
						 - a_s * (dens_l | noarr::get_at<'x', 'y', 'z', 's'>(densities, x, y, i + 1, s)))
						* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
				}
			}
		}
	}
}

template <typename real_t>
void least_memory_thomas_solver<real_t>::solve_x()
{
	if (this->problem_.dims == 1)
	{
#pragma omp parallel
		solve_slice_x_1d<index_t>(this->substrates_, ax_.get(), b1x_.get(), bx_.get(), get_substrates_layout<1>(),
								  get_diagonal_layout(this->problem_, this->problem_.nx));
	}
	else if (this->problem_.dims == 2)
	{
#pragma omp parallel
		solve_slice_x_2d_and_3d<index_t>(this->substrates_, ax_.get(), b1x_.get(), bx_.get(),
										 get_substrates_layout<2>() ^ noarr::rename<'y', 'm'>(),
										 get_diagonal_layout(this->problem_, this->problem_.nx));
	}
	else if (this->problem_.dims == 3)
	{
#pragma omp parallel
		solve_slice_x_2d_and_3d<index_t>(this->substrates_, ax_.get(), b1x_.get(), bx_.get(),
										 get_substrates_layout<3>() ^ noarr::merge_blocks<'z', 'y', 'm'>(),
										 get_diagonal_layout(this->problem_, this->problem_.nx));
	}
}

template <typename real_t>
void least_memory_thomas_solver<real_t>::solve_y()
{
	if (this->problem_.dims == 2)
	{
#pragma omp parallel
		solve_slice_y_2d<index_t>(this->substrates_, ay_.get(), b1y_.get(), by_.get(), get_substrates_layout<2>(),
								  get_diagonal_layout(this->problem_, this->problem_.ny));
	}
	else if (this->problem_.dims == 3)
	{
#pragma omp parallel
		solve_slice_y_3d<index_t>(this->substrates_, ay_.get(), b1y_.get(), by_.get(), get_substrates_layout<3>(),
								  get_diagonal_layout(this->problem_, this->problem_.ny));
	}
}

template <typename real_t>
void least_memory_thomas_solver<real_t>::solve_z()
{
#pragma omp parallel
	solve_slice_z_3d<index_t>(this->substrates_, az_.get(), b1z_.get(), bz_.get(), get_substrates_layout<3>(),
							  get_diagonal_layout(this->problem_, this->problem_.nz));
}

template <typename real_t>
void least_memory_thomas_solver<real_t>::solve()
{
	if (this->problem_.dims == 1)
	{
#pragma omp parallel
		{
			perf_counter counter("lstm");

			for (index_t i = 0; i < this->problem_.iterations; i++)
				solve_slice_x_1d<index_t>(this->substrates_, ax_.get(), b1x_.get(), bx_.get(),
										  get_substrates_layout<1>(),
										  get_diagonal_layout(this->problem_, this->problem_.nx));
		}
	}
	if (this->problem_.dims == 2)
	{
#pragma omp parallel
		{
			perf_counter counter("lstm");

			for (index_t i = 0; i < this->problem_.iterations; i++)
			{
				solve_slice_x_2d_and_3d<index_t>(this->substrates_, ax_.get(), b1x_.get(), bx_.get(),
												 get_substrates_layout<2>() ^ noarr::rename<'y', 'm'>(),
												 get_diagonal_layout(this->problem_, this->problem_.nx));
#pragma omp barrier
				solve_slice_y_2d<index_t>(this->substrates_, ay_.get(), b1y_.get(), by_.get(),
										  get_substrates_layout<2>(),
										  get_diagonal_layout(this->problem_, this->problem_.ny));
#pragma omp barrier
			}
		}
	}
	if (this->problem_.dims == 3)
	{
#pragma omp parallel
		{
			perf_counter counter("lstm");

			for (index_t i = 0; i < this->problem_.iterations; i++)
			{
				solve_slice_x_2d_and_3d<index_t>(this->substrates_, ax_.get(), b1x_.get(), bx_.get(),
												 get_substrates_layout<3>() ^ noarr::merge_blocks<'z', 'y', 'm'>(),
												 get_diagonal_layout(this->problem_, this->problem_.nx));
#pragma omp barrier
				solve_slice_y_3d<index_t>(this->substrates_, ay_.get(), b1y_.get(), by_.get(),
										  get_substrates_layout<3>(),
										  get_diagonal_layout(this->problem_, this->problem_.ny));
#pragma omp barrier
				solve_slice_z_3d<index_t>(this->substrates_, az_.get(), b1z_.get(), bz_.get(),
										  get_substrates_layout<3>(),
										  get_diagonal_layout(this->problem_, this->problem_.nz));
#pragma omp barrier
			}
		}
	}
}

template class least_memory_thomas_solver<float>;
template class least_memory_thomas_solver<double>;
