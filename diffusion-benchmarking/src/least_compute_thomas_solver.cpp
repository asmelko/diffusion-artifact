#include "least_compute_thomas_solver.h"

#include "perf_utils.h"

template <typename real_t>
void least_compute_thomas_solver<real_t>::precompute_values(std::unique_ptr<real_t[]>& b, std::unique_ptr<real_t[]>& c,
															std::unique_ptr<real_t[]>& e, index_t shape, index_t dims,
															index_t n, index_t copies)
{
	b = std::make_unique<real_t[]>(n * this->problem_.substrates_count * copies);
	e = std::make_unique<real_t[]>((n - 1) * this->problem_.substrates_count * copies);
	c = std::make_unique<real_t[]>(this->problem_.substrates_count * copies);

	auto layout = noarr::scalar<real_t>() ^ noarr::vector<'s'>(this->problem_.substrates_count)
				  ^ noarr::vector<'x'>(copies) ^ noarr::vector<'i'>(n);

	auto b_diag = noarr::make_bag(layout, b.get());
	auto e_diag = noarr::make_bag(layout, e.get());

	// compute c_i
	for (index_t x = 0; x < copies; x++)
		for (index_t s = 0; s < this->problem_.substrates_count; s++)
			c[x * this->problem_.substrates_count + s] =
				-1 * -this->problem_.dt * this->problem_.diffusion_coefficients[s] / (shape * shape);

	// compute b_i
	{
		std::array<index_t, 2> indices = { 0, n - 1 };

		for (index_t i : indices)
			for (index_t x = 0; x < copies; x++)
				for (index_t s = 0; s < this->problem_.substrates_count; s++)
					b_diag.template at<'i', 'x', 's'>(i, x, s) =
						1 + this->problem_.decay_rates[s] * this->problem_.dt / dims
						+ this->problem_.dt * this->problem_.diffusion_coefficients[s] / (shape * shape);

		for (index_t i = 1; i < n - 1; i++)
			for (index_t x = 0; x < copies; x++)
				for (index_t s = 0; s < this->problem_.substrates_count; s++)
					b_diag.template at<'i', 'x', 's'>(i, x, s) =
						1 + this->problem_.decay_rates[s] * this->problem_.dt / dims
						+ 2 * this->problem_.dt * this->problem_.diffusion_coefficients[s] / (shape * shape);
	}

	// compute b_i' and e_i
	{
		for (index_t x = 0; x < copies; x++)
			for (index_t s = 0; s < this->problem_.substrates_count; s++)
				b_diag.template at<'i', 'x', 's'>(0, x, s) = 1 / b_diag.template at<'i', 'x', 's'>(0, x, s);

		for (index_t i = 1; i < n; i++)
			for (index_t x = 0; x < copies; x++)
				for (index_t s = 0; s < this->problem_.substrates_count; s++)
				{
					b_diag.template at<'i', 'x', 's'>(i, x, s) =
						1
						/ (b_diag.template at<'i', 'x', 's'>(i, x, s)
						   - c[x * this->problem_.substrates_count + s] * c[x * this->problem_.substrates_count + s]
								 * b_diag.template at<'i', 'x', 's'>(i - 1, x, s));

					e_diag.template at<'i', 'x', 's'>(i - 1, x, s) =
						c[x * this->problem_.substrates_count + s] * b_diag.template at<'i', 'x', 's'>(i - 1, x, s);
				}
	}
}

template <typename real_t>
void least_compute_thomas_solver<real_t>::initialize()
{
	if (this->problem_.dims >= 1)
		precompute_values(bx_, cx_, ex_, this->problem_.dx, this->problem_.dims, this->problem_.nx, 1);
	if (this->problem_.dims >= 2)
		precompute_values(by_, cy_, ey_, this->problem_.dy, this->problem_.dims, this->problem_.ny, 1);
	if (this->problem_.dims >= 3)
		precompute_values(bz_, cz_, ez_, this->problem_.dz, this->problem_.dims, this->problem_.nz, 1);
}

template <typename real_t>
auto least_compute_thomas_solver<real_t>::get_diagonal_layout(const problem_t<index_t, real_t>& problem, index_t n)
{
	return noarr::scalar<real_t>() ^ noarr::vectors<'s', 'i'>(problem.substrates_count, n);
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_x_1d(real_t* __restrict__ densities, const real_t* __restrict__ b, const real_t* __restrict__ c,
							 const real_t* __restrict__ e, const density_layout_t dens_l,
							 const diagonal_layout_t diag_l)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'x'>();

	for (index_t i = 1; i < n; i++)
	{
#pragma omp for schedule(static) nowait
		for (index_t s = 0; s < substrates_count; s++)
		{
			(dens_l | noarr::get_at<'x', 's'>(densities, i, s)) =
				(dens_l | noarr::get_at<'x', 's'>(densities, i, s))
				+ (diag_l | noarr::get_at<'i', 's'>(e, i - 1, s))
					  * (dens_l | noarr::get_at<'x', 's'>(densities, i - 1, s));

			// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
		}
	}

#pragma omp for schedule(static) nowait
	for (index_t s = 0; s < substrates_count; s++)
	{
		(dens_l | noarr::get_at<'x', 's'>(densities, n - 1, s)) =
			(dens_l | noarr::get_at<'x', 's'>(densities, n - 1, s)) * (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));

		// std::cout << "n-1: " << (dens_l | noarr::get_at<'x', 's'>(densities, n - 1, s)) << std::endl;
	}

	for (index_t i = n - 2; i >= 0; i--)
	{
#pragma omp for schedule(static) nowait
		for (index_t s = 0; s < substrates_count; s++)
		{
			(dens_l | noarr::get_at<'x', 's'>(densities, i, s)) =
				((dens_l | noarr::get_at<'x', 's'>(densities, i, s))
				 + c[s] * (dens_l | noarr::get_at<'x', 's'>(densities, i + 1, s)))
				* (diag_l | noarr::get_at<'i', 's'>(b, i, s));

			// std::cout << i << ": " << (dens_l | noarr::get_at<'x', 's'>(densities, i, s)) << std::endl;
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_x_2d_and_3d(real_t* __restrict__ densities, const real_t* __restrict__ b,
									const real_t* __restrict__ c, const real_t* __restrict__ e,
									const density_layout_t dens_l, const diagonal_layout_t diag_l)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'x'>();
	const index_t m = dens_l | noarr::get_length<'m'>();

#pragma omp for schedule(static) nowait
	for (index_t yz = 0; yz < m; yz++)
	{
		for (index_t i = 1; i < n; i++)
		{
			for (index_t s = 0; s < substrates_count; s++)
			{
				(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, i, s)) =
					(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, i, s))
					+ (diag_l | noarr::get_at<'i', 's'>(e, i - 1, s))
						  * (dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, i - 1, s));
			}
		}

		for (index_t s = 0; s < substrates_count; s++)
		{
			(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, n - 1, s)) =
				(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, n - 1, s))
				* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
		}

		for (index_t i = n - 2; i >= 0; i--)
		{
			for (index_t s = 0; s < substrates_count; s++)
			{
				(dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, i, s)) =
					((dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, i, s))
					 + c[s] * (dens_l | noarr::get_at<'m', 'x', 's'>(densities, yz, i + 1, s)))
					* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_y_2d(real_t* __restrict__ densities, const real_t* __restrict__ b, const real_t* __restrict__ c,
							 const real_t* __restrict__ e, const density_layout_t dens_l,
							 const diagonal_layout_t diag_l)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'y'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();

	for (index_t i = 1; i < n; i++)
	{
#pragma omp for schedule(static) nowait
		for (index_t x = 0; x < x_len; x++)
		{
			for (index_t s = 0; s < substrates_count; s++)
			{
				(dens_l | noarr::get_at<'y', 'x', 's'>(densities, i, x, s)) =
					(dens_l | noarr::get_at<'y', 'x', 's'>(densities, i, x, s))
					+ (diag_l | noarr::get_at<'i', 's'>(e, i - 1, s))
						  * (dens_l | noarr::get_at<'y', 'x', 's'>(densities, i - 1, x, s));
			}
		}
	}

#pragma omp for schedule(static) nowait
	for (index_t x = 0; x < x_len; x++)
	{
		for (index_t s = 0; s < substrates_count; s++)
		{
			(dens_l | noarr::get_at<'y', 'x', 's'>(densities, n - 1, x, s)) =
				(dens_l | noarr::get_at<'y', 'x', 's'>(densities, n - 1, x, s))
				* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
		}
	}

	for (index_t i = n - 2; i >= 0; i--)
	{
#pragma omp for schedule(static) nowait
		for (index_t x = 0; x < x_len; x++)
		{
			for (index_t s = 0; s < substrates_count; s++)
			{
				(dens_l | noarr::get_at<'y', 'x', 's'>(densities, i, x, s)) =
					((dens_l | noarr::get_at<'y', 'x', 's'>(densities, i, x, s))
					 + c[s] * (dens_l | noarr::get_at<'y', 'x', 's'>(densities, i + 1, x, s)))
					* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_y_3d(real_t* __restrict__ densities, const real_t* __restrict__ b, const real_t* __restrict__ c,
							 const real_t* __restrict__ e, const density_layout_t dens_l,
							 const diagonal_layout_t diag_l)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'y'>();
	const index_t z_len = dens_l | noarr::get_length<'z'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();

#pragma omp for schedule(static) nowait
	for (index_t z = 0; z < z_len; z++)
	{
		for (index_t i = 1; i < n; i++)
		{
			for (index_t x = 0; x < x_len; x++)
			{
				for (index_t s = 0; s < substrates_count; s++)
				{
					(dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, z, i, x, s)) =
						(dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, z, i, x, s))
						+ (diag_l | noarr::get_at<'i', 's'>(e, i - 1, s))
							  * (dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, z, i - 1, x, s));
				}
			}
		}

		for (index_t x = 0; x < x_len; x++)
		{
			for (index_t s = 0; s < substrates_count; s++)
			{
				(dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, z, n - 1, x, s)) =
					(dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, z, n - 1, x, s))
					* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
			}
		}

		for (index_t i = n - 2; i >= 0; i--)
		{
			for (index_t x = 0; x < x_len; x++)
			{
				for (index_t s = 0; s < substrates_count; s++)
				{
					(dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, z, i, x, s)) =
						((dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, z, i, x, s))
						 + c[s] * (dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, z, i + 1, x, s)))
						* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
				}
			}
		}
	}
}

template <typename index_t, typename real_t, typename density_layout_t, typename diagonal_layout_t>
static void solve_slice_z_3d(real_t* __restrict__ densities, const real_t* __restrict__ b, const real_t* __restrict__ c,
							 const real_t* __restrict__ e, const density_layout_t dens_l,
							 const diagonal_layout_t diag_l)
{
	const index_t substrates_count = dens_l | noarr::get_length<'s'>();
	const index_t n = dens_l | noarr::get_length<'z'>();
	const index_t y_len = dens_l | noarr::get_length<'y'>();
	const index_t x_len = dens_l | noarr::get_length<'x'>();

	for (index_t i = 1; i < n; i++)
	{
#pragma omp for schedule(static) collapse(2) nowait
		for (index_t y = 0; y < y_len; y++)
		{
			for (index_t x = 0; x < x_len; x++)
			{
				for (index_t s = 0; s < substrates_count; s++)
				{
					(dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, i, y, x, s)) =
						(dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, i, y, x, s))
						+ (diag_l | noarr::get_at<'i', 's'>(e, i - 1, s))
							  * (dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, i - 1, y, x, s));
				}
			}
		}
	}

#pragma omp for schedule(static) collapse(2) nowait
	for (index_t y = 0; y < y_len; y++)
	{
		for (index_t x = 0; x < x_len; x++)
		{
			for (index_t s = 0; s < substrates_count; s++)
			{
				(dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, n - 1, y, x, s)) =
					(dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, n - 1, y, x, s))
					* (diag_l | noarr::get_at<'i', 's'>(b, n - 1, s));
			}
		}
	}

	for (index_t i = n - 2; i >= 0; i--)
	{
#pragma omp for schedule(static) collapse(2) nowait
		for (index_t y = 0; y < y_len; y++)
		{
			for (index_t x = 0; x < x_len; x++)
			{
				for (index_t s = 0; s < substrates_count; s++)
				{
					(dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, i, y, x, s)) =
						((dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, i, y, x, s))
						 + c[s] * (dens_l | noarr::get_at<'z', 'y', 'x', 's'>(densities, i + 1, y, x, s)))
						* (diag_l | noarr::get_at<'i', 's'>(b, i, s));
				}
			}
		}
	}
}

template <typename real_t>
void least_compute_thomas_solver<real_t>::solve_x()
{
	if (this->problem_.dims == 1)
	{
#pragma omp parallel
		solve_slice_x_1d<index_t>(this->substrates_, bx_.get(), cx_.get(), ex_.get(), get_substrates_layout<1>(),
								  get_diagonal_layout(this->problem_, this->problem_.nx));
	}
	else if (this->problem_.dims == 2)
	{
#pragma omp parallel
		solve_slice_x_2d_and_3d<index_t>(this->substrates_, bx_.get(), cx_.get(), ex_.get(),
										 get_substrates_layout<2>() ^ noarr::rename<'y', 'm'>(),
										 get_diagonal_layout(this->problem_, this->problem_.nx));
	}
	else if (this->problem_.dims == 3)
	{
#pragma omp parallel
		solve_slice_x_2d_and_3d<index_t>(this->substrates_, bx_.get(), cx_.get(), ex_.get(),
										 get_substrates_layout<3>() ^ noarr::merge_blocks<'z', 'y', 'm'>(),
										 get_diagonal_layout(this->problem_, this->problem_.nx));
	}
}

template <typename real_t>
void least_compute_thomas_solver<real_t>::solve_y()
{
	if (this->problem_.dims == 2)
	{
#pragma omp parallel
		solve_slice_y_2d<index_t>(this->substrates_, by_.get(), cy_.get(), ey_.get(), get_substrates_layout<2>(),
								  get_diagonal_layout(this->problem_, this->problem_.ny));
	}
	else if (this->problem_.dims == 3)
	{
#pragma omp parallel
		solve_slice_y_3d<index_t>(this->substrates_, by_.get(), cy_.get(), ey_.get(), get_substrates_layout<3>(),
								  get_diagonal_layout(this->problem_, this->problem_.ny));
	}
}

template <typename real_t>
void least_compute_thomas_solver<real_t>::solve_z()
{
#pragma omp parallel
	solve_slice_z_3d<index_t>(this->substrates_, bz_.get(), cz_.get(), ez_.get(), get_substrates_layout<3>(),
							  get_diagonal_layout(this->problem_, this->problem_.nz));
}

template <typename real_t>
void least_compute_thomas_solver<real_t>::solve()
{
	if (this->problem_.dims == 1)
	{
#pragma omp parallel
		{
			perf_counter counter("lstc");

			for (index_t i = 0; i < this->problem_.iterations; i++)
				solve_slice_x_1d<index_t>(this->substrates_, bx_.get(), cx_.get(), ex_.get(),
										  get_substrates_layout<1>(),
										  get_diagonal_layout(this->problem_, this->problem_.nx));
		}
	}
	else if (this->problem_.dims == 2)
	{
#pragma omp parallel
		{
			perf_counter counter("lstc");

			for (index_t i = 0; i < this->problem_.iterations; i++)
			{
				solve_slice_x_2d_and_3d<index_t>(this->substrates_, bx_.get(), cx_.get(), ex_.get(),
												 get_substrates_layout<2>() ^ noarr::rename<'y', 'm'>(),
												 get_diagonal_layout(this->problem_, this->problem_.nx));
#pragma omp barrier
				solve_slice_y_2d<index_t>(this->substrates_, by_.get(), cy_.get(), ey_.get(),
										  get_substrates_layout<2>(),
										  get_diagonal_layout(this->problem_, this->problem_.ny));
#pragma omp barrier
			}
		}
	}
	else if (this->problem_.dims == 3)
	{
#pragma omp parallel
		{
			perf_counter counter("lstc");

			for (index_t i = 0; i < this->problem_.iterations; i++)
			{
				solve_slice_x_2d_and_3d<index_t>(this->substrates_, bx_.get(), cx_.get(), ex_.get(),
												 get_substrates_layout<3>() ^ noarr::merge_blocks<'z', 'y', 'm'>(),
												 get_diagonal_layout(this->problem_, this->problem_.nx));
#pragma omp barrier
				solve_slice_y_3d<index_t>(this->substrates_, by_.get(), cy_.get(), ey_.get(),
										  get_substrates_layout<3>(),
										  get_diagonal_layout(this->problem_, this->problem_.ny));
#pragma omp barrier
				solve_slice_z_3d<index_t>(this->substrates_, bz_.get(), cz_.get(), ez_.get(),
										  get_substrates_layout<3>(),
										  get_diagonal_layout(this->problem_, this->problem_.nz));
#pragma omp barrier
			}
		}
	}
}

template class least_compute_thomas_solver<float>;
template class least_compute_thomas_solver<double>;
