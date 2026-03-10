#include "lapack_thomas_solver.h"

extern "C"
{
	extern void spttrf_(const int* n, float* d, float* e, int* info);
	extern void spttrs_(const int* n, const int* nrhs, const float* d, const float* e, float* b, const int* ldb,
						int* info);

	extern void dpttrf_(const int* n, double* d, double* e, int* info);
	extern void dpttrs_(const int* n, const int* nrhs, const double* d, const double* e, double* b, const int* ldb,
						int* info);

	extern void sptsv_(const int* n, const int* nrhs, float* d, float* e, float* b, const int* ldb, int* info);
	extern void dptsv_(const int* n, const int* nrhs, double* d, double* e, double* b, const int* ldb, int* info);
}

template <>
void lapack_thomas_solver<float>::pttrf(const int* n, float* d, float* e, int* info)
{
#ifdef HAS_LAPACK
	spttrf_(n, d, e, info);
#else
	(void)n;
	(void)d;
	(void)e;
	(void)info;
	throw std::runtime_error("LAPACK not found. Cannot perform operation.");
#endif
}

template <>
void lapack_thomas_solver<double>::pttrf(const int* n, double* d, double* e, int* info)
{
#ifdef HAS_LAPACK
	dpttrf_(n, d, e, info);
#else
	(void)n;
	(void)d;
	(void)e;
	(void)info;
	throw std::runtime_error("LAPACK not found. Cannot perform operation.");
#endif
}

template <>
void lapack_thomas_solver<float>::pttrs(const int* n, const int* nrhs, const float* d, const float* e, float* b,
										const int* ldb, int* info)
{
#ifdef HAS_LAPACK
	spttrs_(n, nrhs, d, e, b, ldb, info);
#else
	(void)n;
	(void)nrhs;
	(void)d;
	(void)e;
	(void)b;
	(void)ldb;
	(void)info;
	throw std::runtime_error("LAPACK not found. Cannot perform operation.");
#endif
}

template <>
void lapack_thomas_solver<double>::pttrs(const int* n, const int* nrhs, const double* d, const double* e, double* b,
										 const int* ldb, int* info)
{
#ifdef HAS_LAPACK
	dpttrs_(n, nrhs, d, e, b, ldb, info);
#else
	(void)n;
	(void)nrhs;
	(void)d;
	(void)e;
	(void)b;
	(void)ldb;
	(void)info;
	throw std::runtime_error("LAPACK not found. Cannot perform operation.");
#endif
}

template <>
void lapack_thomas_solver<float>::ptsv(const int* n, const int* nrhs, float* d, float* e, float* b, const int* ldb,
									   int* info)
{
#ifdef HAS_LAPACK
	sptsv_(n, nrhs, d, e, b, ldb, info);
#else
	(void)n;
	(void)nrhs;
	(void)d;
	(void)e;
	(void)b;
	(void)ldb;
	(void)info;
	throw std::runtime_error("LAPACK not found. Cannot perform operation.");
#endif
}

template <>
void lapack_thomas_solver<double>::ptsv(const int* n, const int* nrhs, double* d, double* e, double* b, const int* ldb,
										int* info)
{
#ifdef HAS_LAPACK
	dptsv_(n, nrhs, d, e, b, ldb, info);
#else
	(void)n;
	(void)nrhs;
	(void)d;
	(void)e;
	(void)b;
	(void)ldb;
	(void)info;
	throw std::runtime_error("LAPACK not found. Cannot perform operation.");
#endif
}

template <typename real_t>
void lapack_thomas_solver<real_t>::precompute_values(std::vector<std::unique_ptr<real_t[]>>& a,
													 std::vector<std::unique_ptr<real_t[]>>& b, index_t shape,
													 index_t dims, index_t n)
{
	for (index_t s_idx = 0; s_idx < this->problem_.substrates_count; s_idx++)
	{
		auto single_substr_a = std::make_unique<real_t[]>(n - 1);
		auto single_substr_b = std::make_unique<real_t[]>(n);
		for (index_t i = 0; i < n; i++)
		{
			if (i != n - 1)
				single_substr_a[i] =
					-this->problem_.dt * this->problem_.diffusion_coefficients[s_idx] / (shape * shape);

			single_substr_b[i] =
				1 + this->problem_.dt * this->problem_.decay_rates[s_idx] / dims
				+ 2 * this->problem_.dt * this->problem_.diffusion_coefficients[s_idx] / (shape * shape);

			if (i == 0 || i == n - 1)
				single_substr_b[i] -=
					this->problem_.dt * this->problem_.diffusion_coefficients[s_idx] / (shape * shape);
		}

		int info;
		pttrf(&n, single_substr_b.get(), single_substr_a.get(), &info);

		if (info != 0)
			throw std::runtime_error("LAPACK pttrf failed with error code " + std::to_string(info));

		a.emplace_back(std::move(single_substr_a));
		b.emplace_back(std::move(single_substr_b));
	}
}

template <typename real_t>
void lapack_thomas_solver<real_t>::initialize()
{
	if (this->problem_.dims >= 1)
		precompute_values(ax_, bx_, this->problem_.dx, this->problem_.dims, this->problem_.nx);
	if (this->problem_.dims >= 2)
		precompute_values(ay_, by_, this->problem_.dy, this->problem_.dims, this->problem_.ny);
	if (this->problem_.dims >= 3)
		precompute_values(az_, bz_, this->problem_.dz, this->problem_.dims, this->problem_.nz);
}

template <typename real_t>
void lapack_thomas_solver<real_t>::tune(const nlohmann::json& params)
{
	work_items_ = params.contains("work_items") ? (std::size_t)params["work_items"] : 10;
}

template <typename real_t>
void lapack_thomas_solver<real_t>::solve_x()
{
	auto dens_l = get_substrates_layout() ^ noarr::merge_blocks<'z', 'y', 'm'>();
	auto yz_len = dens_l | noarr::get_length<'m'>();

	for (index_t s = 0; s < this->problem_.substrates_count; s++)
	{
#pragma omp for schedule(static) nowait
		for (std::size_t yz = 0; yz < yz_len; yz += work_items_)
		{
			const std::size_t begin_offset = (dens_l | noarr::offset<'x', 'm', 's'>(0, yz, s)) / sizeof(real_t);

			int info;
			int rhs = yz + work_items_ > yz_len ? yz_len - yz : work_items_;
			pttrs(&this->problem_.nx, &rhs, bx_[s].get(), ax_[s].get(), this->substrates_ + begin_offset,
				  &this->problem_.nx, &info);

			if (info != 0)
				throw std::runtime_error("LAPACK pttrs failed with error code " + std::to_string(info));
		}
	}
}

template <typename real_t>
void lapack_thomas_solver<real_t>::solve_y()
{}

template <typename real_t>
void lapack_thomas_solver<real_t>::solve_z()
{}

template <typename real_t>
void lapack_thomas_solver<real_t>::solve()
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

template class lapack_thomas_solver<float>;
template class lapack_thomas_solver<double>;
