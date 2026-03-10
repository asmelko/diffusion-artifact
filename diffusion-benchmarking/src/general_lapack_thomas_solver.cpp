#include "general_lapack_thomas_solver.h"

extern "C"
{
	extern void sgttrf_(const int* n, float* dl, float* d, float* du, float* du2, int* ipiv, int* info);
	extern void sgttrs_(const char* trans, const int* n, const int* nrhs, const float* dl, const float* d,
						const float* du, const float* du2, const int* ipiv, float* b, const int* ldb, int* info);

	extern void dgttrf_(const int* n, double* dl, double* d, double* du, double* du2, int* ipiv, int* info);
	extern void dgttrs_(const char* trans, const int* n, const int* nrhs, const double* dl, const double* d,
						const double* du, const double* du2, const int* ipiv, double* b, const int* ldb, int* info);
}

template <>
void general_lapack_thomas_solver<float>::gttrf(const int* n, float* dl, float* d, float* du, float* du2, int* ipiv,
												int* info)
{
#ifdef HAS_LAPACK
	sgttrf_(n, dl, d, du, du2, ipiv, info);
#else
	(void)n;
	(void)dl;
	(void)d;
	(void)du;
	(void)du2;
	(void)ipiv;
	(void)info;
	throw std::runtime_error("LAPACK not found. Cannot perform operation.");
#endif
}

template <>
void general_lapack_thomas_solver<double>::gttrf(const int* n, double* dl, double* d, double* du, double* du2,
												 int* ipiv, int* info)
{
#ifdef HAS_LAPACK
	dgttrf_(n, dl, d, du, du2, ipiv, info);
#else
	(void)n;
	(void)dl;
	(void)d;
	(void)du;
	(void)du2;
	(void)ipiv;
	(void)info;
	throw std::runtime_error("LAPACK not found. Cannot perform operation.");
#endif
}

template <>
void general_lapack_thomas_solver<float>::gttrs(const char* trans, const int* n, const int* nrhs, const float* dl,
												const float* d, const float* du, const float* du2, const int* ipiv,
												float* b, const int* ldb, int* info)
{
#ifdef HAS_LAPACK
	sgttrs_(trans, n, nrhs, dl, d, du, du2, ipiv, b, ldb, info);
#else
	(void)trans;
	(void)n;
	(void)nrhs;
	(void)dl;
	(void)d;
	(void)du;
	(void)du2;
	(void)ipiv;
	(void)b;
	(void)ldb;
	(void)info;
	throw std::runtime_error("LAPACK not found. Cannot perform operation.");
#endif
};

template <>
void general_lapack_thomas_solver<double>::gttrs(const char* trans, const int* n, const int* nrhs, const double* dl,
												 const double* d, const double* du, const double* du2, const int* ipiv,
												 double* b, const int* ldb, int* info)
{
#ifdef HAS_LAPACK
	dgttrs_(trans, n, nrhs, dl, d, du, du2, ipiv, b, ldb, info);
#else
	(void)trans;
	(void)n;
	(void)nrhs;
	(void)dl;
	(void)d;
	(void)du;
	(void)du2;
	(void)ipiv;
	(void)b;
	(void)ldb;
	(void)info;
	throw std::runtime_error("LAPACK not found. Cannot perform operation.");
#endif
}

template <typename real_t>
void general_lapack_thomas_solver<real_t>::precompute_values(std::vector<std::unique_ptr<real_t[]>>& dls,
															 std::vector<std::unique_ptr<real_t[]>>& ds,
															 std::vector<std::unique_ptr<real_t[]>>& dus,
															 std::vector<std::unique_ptr<real_t[]>>& du2s,
															 std::vector<std::unique_ptr<int[]>>& ipivs, index_t shape,
															 index_t dims, index_t n)
{
	for (index_t s_idx = 0; s_idx < this->problem_.substrates_count; s_idx++)
	{
		auto dl = std::make_unique<real_t[]>(n - 1);
		auto d = std::make_unique<real_t[]>(n);
		auto du = std::make_unique<real_t[]>(n - 1);
		auto du2 = std::make_unique<real_t[]>(n - 2);
		auto ipiv = std::make_unique<int[]>(n);
		for (index_t i = 0; i < n; i++)
		{
			if (i != n - 1)
			{
				dl[i] = -this->problem_.dt * this->problem_.diffusion_coefficients[s_idx] / (shape * shape);
				du[i] = -this->problem_.dt * this->problem_.diffusion_coefficients[s_idx] / (shape * shape);
			}

			d[i] = 1 + this->problem_.dt * this->problem_.decay_rates[s_idx] / dims
				   + 2 * this->problem_.dt * this->problem_.diffusion_coefficients[s_idx] / (shape * shape);

			if (i == 0 || i == n - 1)
				d[i] -= this->problem_.dt * this->problem_.diffusion_coefficients[s_idx] / (shape * shape);
		}

		int info;
		gttrf(&n, dl.get(), d.get(), du.get(), du2.get(), ipiv.get(), &info);

		if (info != 0)
			throw std::runtime_error("LAPACK spttrf failed with error code " + std::to_string(info));

		dls.emplace_back(std::move(dl));
		ds.emplace_back(std::move(d));
		dus.emplace_back(std::move(du));
		du2s.emplace_back(std::move(du2));
		ipivs.emplace_back(std::move(ipiv));
	}
}

template <typename real_t>
void general_lapack_thomas_solver<real_t>::initialize()
{
	if (this->problem_.dims >= 1)
		precompute_values(dlx_, dx_, dux_, du2x_, ipivx_, this->problem_.dx, this->problem_.dims, this->problem_.nx);
	if (this->problem_.dims >= 2)
		precompute_values(dly_, dy_, duy_, du2y_, ipivy_, this->problem_.dy, this->problem_.dims, this->problem_.ny);
	if (this->problem_.dims >= 3)
		precompute_values(dlz_, dz_, duz_, du2z_, ipivz_, this->problem_.dz, this->problem_.dims, this->problem_.nz);
}

template <typename real_t>
void general_lapack_thomas_solver<real_t>::tune(const nlohmann::json& params)
{
	work_items_ = params.contains("work_items") ? (std::size_t)params["work_items"] : 10;
}

template <typename real_t>
void general_lapack_thomas_solver<real_t>::solve_x()
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
			char c = 'N';
			gttrs(&c, &this->problem_.nx, &rhs, dlx_[s].get(), dx_[s].get(), dux_[s].get(), du2x_[s].get(),
				  ipivx_[s].get(), this->substrates_ + begin_offset, &this->problem_.nx, &info);

			if (info != 0)
				throw std::runtime_error("LAPACK spttrs failed with error code " + std::to_string(info));
		}
	}
}

template <typename real_t>
void general_lapack_thomas_solver<real_t>::solve_y()
{}

template <typename real_t>
void general_lapack_thomas_solver<real_t>::solve_z()
{}

template <typename real_t>
void general_lapack_thomas_solver<real_t>::solve()
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

template class general_lapack_thomas_solver<float>;
template class general_lapack_thomas_solver<double>;
