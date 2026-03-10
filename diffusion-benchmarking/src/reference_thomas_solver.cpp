#include "reference_thomas_solver.h"

template <typename real_t>
void reference_thomas_solver<real_t>::precompute_values(std::unique_ptr<real_t[]>& a, std::unique_ptr<real_t[]>& b,
														std::unique_ptr<real_t[]>& b0, index_t shape, index_t dims,
														index_t n)
{
	a = std::make_unique<real_t[]>(this->problem_.substrates_count);
	b = std::make_unique<real_t[]>(this->problem_.substrates_count * n);
	b0 = std::make_unique<real_t[]>(this->problem_.substrates_count);

	auto l = noarr::scalar<real_t>() ^ noarr::vectors<'s', 'i'>(this->problem_.substrates_count, n);

	// compute a_i, b0_i
	for (index_t s = 0; s < this->problem_.substrates_count; s++)
	{
		a[s] = -this->problem_.dt * this->problem_.diffusion_coefficients[s] / (shape * shape);
		b0[s] = 1 + this->problem_.dt * this->problem_.decay_rates[s] / dims
				+ this->problem_.dt * this->problem_.diffusion_coefficients[s] / (shape * shape);
	}

	// compute b_i
	for (index_t i = 0; i < n; i++)
		for (index_t s = 0; s < this->problem_.substrates_count; s++)
			if (i == 0)
				(l | noarr::get_at<'s', 'i'>(b.get(), s, i)) = b0[s];
			else if (i != n - 1)
				(l | noarr::get_at<'s', 'i'>(b.get(), s, i)) =
					(b0[s] - a[s]) - (a[s] * a[s]) / (l | noarr::get_at<'s', 'i'>(b.get(), s, i - 1));
			else
				(l | noarr::get_at<'s', 'i'>(b.get(), s, i)) =
					(b0[s]) - (a[s] * a[s]) / (l | noarr::get_at<'s', 'i'>(b.get(), s, i - 1));
}

template <typename real_t>
void reference_thomas_solver<real_t>::initialize()
{
	if (this->problem_.dims >= 1)
		precompute_values(ax_, bx_, b0x_, this->problem_.dx, this->problem_.dims, this->problem_.nx);
	if (this->problem_.dims >= 2)
		precompute_values(ay_, by_, b0y_, this->problem_.dy, this->problem_.dims, this->problem_.ny);
	if (this->problem_.dims >= 3)
		precompute_values(az_, bz_, b0z_, this->problem_.dz, this->problem_.dims, this->problem_.nz);
}

template <typename real_t>
void reference_thomas_solver<real_t>::solve_x()
{
	auto dens_l = get_substrates_layout();
	auto diag_l =
		noarr::scalar<real_t>() ^ noarr::vectors<'s', 'i'>(this->problem_.substrates_count, this->problem_.nx);

	for (index_t z = 0; z < this->problem_.nz; z++)
	{
		for (index_t y = 0; y < this->problem_.ny; y++)
		{
			for (index_t x = 1; x < this->problem_.nx; x++)
			{
				for (index_t s = 0; s < this->problem_.substrates_count; s++)
				{
					(dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(this->substrates_, s, x, y, z)) -=
						(ax_[s] / (diag_l | noarr::get_at<'s', 'i'>(bx_.get(), s, x - 1)))
						* (dens_l
						   | noarr::get_at<'s', 'x', 'y', 'z'>(this->substrates_, s, x - 1, y,
															   z)); // substrates[x] -= ax/bx[x-1] *substrates[x-1]
				}
			}

			for (index_t s = 0; s < this->problem_.substrates_count; s++)
			{
				(dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(this->substrates_, s, this->problem_.nx - 1, y, z)) /=
					(diag_l | noarr::get_at<'s', 'i'>(bx_.get(), s, this->problem_.nx - 1)); // substrates[x] /= bx[x]
			}

			for (index_t x = this->problem_.nx - 2; x >= 0; x--)
			{
				for (index_t s = 0; s < this->problem_.substrates_count; s++)
				{
					(dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(this->substrates_, s, x, y, z)) =
						((dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(this->substrates_, s, x, y, z))
						 - ax_[s] * (dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(this->substrates_, s, x + 1, y, z)))
						/ (diag_l | noarr::get_at<'s', 'i'>(bx_.get(), s, x));
				}
			}
		}
	}
}

template <typename real_t>
void reference_thomas_solver<real_t>::solve_y()
{
	auto dens_l = get_substrates_layout();
	auto diag_l =
		noarr::scalar<real_t>() ^ noarr::vectors<'s', 'i'>(this->problem_.substrates_count, this->problem_.ny);

	for (index_t z = 0; z < this->problem_.nz; z++)
	{
		for (index_t y = 1; y < this->problem_.ny; y++)
		{
			for (index_t x = 0; x < this->problem_.nx; x++)
			{
				for (index_t s = 0; s < this->problem_.substrates_count; s++)
				{
					(dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(this->substrates_, s, x, y, z)) -=
						(ay_[s] / (diag_l | noarr::get_at<'s', 'i'>(by_.get(), s, y - 1)))
						* (dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(this->substrates_, s, x, y - 1, z));
				}
			}
		}
	}

	for (index_t z = 0; z < this->problem_.nz; z++)
	{
		for (index_t x = 0; x < this->problem_.nx; x++)
		{
			for (index_t s = 0; s < this->problem_.substrates_count; s++)
			{
				(dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(this->substrates_, s, x, this->problem_.ny - 1, z)) /=
					(diag_l | noarr::get_at<'s', 'i'>(by_.get(), s, this->problem_.ny - 1));
			}
		}
	}

	for (index_t z = 0; z < this->problem_.nz; z++)
	{
		for (index_t y = this->problem_.ny - 2; y >= 0; y--)
		{
			for (index_t x = 0; x < this->problem_.nx; x++)
			{
				for (index_t s = 0; s < this->problem_.substrates_count; s++)
				{
					(dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(this->substrates_, s, x, y, z)) =
						((dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(this->substrates_, s, x, y, z))
						 - ay_[s] * (dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(this->substrates_, s, x, y + 1, z)))
						/ (diag_l | noarr::get_at<'s', 'i'>(by_.get(), s, y));
				}
			}
		}
	}
}

template <typename real_t>
void reference_thomas_solver<real_t>::solve_z()
{
	auto dens_l = get_substrates_layout();
	auto diag_l =
		noarr::scalar<real_t>() ^ noarr::vectors<'s', 'i'>(this->problem_.substrates_count, this->problem_.nz);

	for (index_t z = 1; z < this->problem_.nz; z++)
	{
		for (index_t y = 0; y < this->problem_.ny; y++)
		{
			for (index_t x = 0; x < this->problem_.nx; x++)
			{
				for (index_t s = 0; s < this->problem_.substrates_count; s++)
				{
					(dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(this->substrates_, s, x, y, z)) -=
						(az_[s] / (diag_l | noarr::get_at<'s', 'i'>(bz_.get(), s, z - 1)))
						* (dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(this->substrates_, s, x, y, z - 1));
				}
			}
		}
	}

	for (index_t y = 0; y < this->problem_.ny; y++)
	{
		for (index_t x = 0; x < this->problem_.nx; x++)
		{
			for (index_t s = 0; s < this->problem_.substrates_count; s++)
			{
				(dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(this->substrates_, s, x, y, this->problem_.nz - 1)) /=
					(diag_l | noarr::get_at<'s', 'i'>(bz_.get(), s, this->problem_.nz - 1));
			}
		}
	}

	for (index_t z = this->problem_.nz - 2; z >= 0; z--)
	{
		for (index_t y = 0; y < this->problem_.ny; y++)
		{
			for (index_t x = 0; x < this->problem_.nx; x++)
			{
				for (index_t s = 0; s < this->problem_.substrates_count; s++)
				{
					(dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(this->substrates_, s, x, y, z)) =
						((dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(this->substrates_, s, x, y, z))
						 - az_[s] * (dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(this->substrates_, s, x, y, z + 1)))
						/ (diag_l | noarr::get_at<'s', 'i'>(bz_.get(), s, z));
				}
			}
		}
	}
}

template <typename real_t>
void reference_thomas_solver<real_t>::solve()
{
	for (index_t i = 0; i < this->problem_.iterations; i++)
	{
		if (this->problem_.dims == 1)
		{
			solve_x();
		}
		else if (this->problem_.dims == 2)
		{
			solve_x();
			solve_y();
		}
		else if (this->problem_.dims == 3)
		{
			solve_x();
			solve_y();
			solve_z();
		}
	}
}

template class reference_thomas_solver<float>;
template class reference_thomas_solver<double>;
