#include "full_lapack_solver.h"

extern "C"
{
	extern void spbtrf_(const char* uplo, const int* n, const int* kd, float* ab, const int* ldab, int* info);
	extern void spbtrs_(const char* uplo, const int* n, const int* kd, const int* nrhs, const float* ab,
						const int* ldab, float* b, const int* ldb, int* info);

	extern void dpbtrf_(const char* uplo, const int* n, const int* kd, double* ab, const int* ldab, int* info);
	extern void dpbtrs_(const char* uplo, const int* n, const int* kd, const int* nrhs, const double* ab,
						const int* ldab, double* b, const int* ldb, int* info);
}

template <>
void full_lapack_solver<float>::pbtrf(const char* uplo, const int* n, const int* kd, float* ab, const int* ldab,
									  int* info)
{
#ifdef HAS_LAPACK
	spbtrf_(uplo, n, kd, ab, ldab, info);
#else
	(void)uplo;
	(void)n;
	(void)kd;
	(void)ab;
	(void)ldab;
	(void)info;
	throw std::runtime_error("LAPACK not found. Cannot perform operation.");
#endif
}

template <>
void full_lapack_solver<double>::pbtrf(const char* uplo, const int* n, const int* kd, double* ab, const int* ldab,
									   int* info)
{
#ifdef HAS_LAPACK
	dpbtrf_(uplo, n, kd, ab, ldab, info);
#else
	(void)uplo;
	(void)n;
	(void)kd;
	(void)ab;
	(void)ldab;
	(void)info;
	throw std::runtime_error("LAPACK not found. Cannot perform operation.");
#endif
}

template <>
void full_lapack_solver<float>::pbtrs(const char* uplo, const int* n, const int* kd, const int* nrhs, const float* ab,
									  const int* ldab, float* b, const int* ldb, int* info)
{
#ifdef HAS_LAPACK
	spbtrs_(uplo, n, kd, nrhs, ab, ldab, b, ldb, info);
#else
	(void)uplo;
	(void)n;
	(void)kd;
	(void)nrhs;
	(void)ab;
	(void)ldab;
	(void)b;
	(void)ldb;
	(void)info;
	throw std::runtime_error("LAPACK not found. Cannot perform operation.");
#endif
}

template <>
void full_lapack_solver<double>::pbtrs(const char* uplo, const int* n, const int* kd, const int* nrhs, const double* ab,
									   const int* ldab, double* b, const int* ldb, int* info)
{
#ifdef HAS_LAPACK
	dpbtrs_(uplo, n, kd, nrhs, ab, ldab, b, ldb, info);
#else
	(void)uplo;
	(void)n;
	(void)kd;
	(void)nrhs;
	(void)ab;
	(void)ldab;
	(void)b;
	(void)ldb;
	(void)info;
	throw std::runtime_error("LAPACK not found. Cannot perform operation.");
#endif
}

template <typename real_t>
void full_lapack_solver<real_t>::precompute_values()
{
	auto substrates_layout = get_substrates_layout();

	int kd =
		1 * (this->problem_.dims >= 2 ? this->problem_.nx : 1) * (this->problem_.dims >= 3 ? this->problem_.ny : 1);

	auto ab_layout = noarr::scalar<real_t>()
					 ^ noarr::vectors<'i', 'j'>(kd + 1, this->problem_.nx * this->problem_.ny * this->problem_.nz);

	for (index_t s_idx = 0; s_idx < this->problem_.substrates_count; s_idx++)
	{
		auto single_substr_ab = std::make_unique<real_t[]>((std::size_t)this->problem_.nx * this->problem_.ny
														   * this->problem_.nz * (kd + 1));

		std::fill(single_substr_ab.get(),
				  single_substr_ab.get() + this->problem_.nx * this->problem_.ny * this->problem_.nz * (kd + 1), 0);

		auto r_x =
			-this->problem_.dt * this->problem_.diffusion_coefficients[s_idx] / (this->problem_.dx * this->problem_.dx);
		auto r_y =
			-this->problem_.dt * this->problem_.diffusion_coefficients[s_idx] / (this->problem_.dy * this->problem_.dy);
		auto r_z =
			-this->problem_.dt * this->problem_.diffusion_coefficients[s_idx] / (this->problem_.dz * this->problem_.dz);

		for (index_t z = 0; z < this->problem_.nz; z++)
			for (index_t y = 0; y < this->problem_.ny; y++)
				for (index_t x = 0; x < this->problem_.nx; x++)
				{
					auto i = (substrates_layout | noarr::offset<'x', 'y', 'z', 's'>(x, y, z, 0)) / sizeof(real_t);

					real_t x_neighbors = 0;
					real_t y_neighbors = 0;
					real_t z_neighbors = 0;

					if (x > 0)
					{
						auto j =
							(substrates_layout | noarr::offset<'x', 'y', 'z', 's'>(x - 1, y, z, 0)) / sizeof(real_t);
						(ab_layout | noarr::get_at<'i', 'j'>(single_substr_ab.get(), i - j, j)) = r_x;
						x_neighbors++;
					}
					if (x < this->problem_.nx - 1)
					{
						x_neighbors++;
					}

					if (y > 0)
					{
						auto j =
							(substrates_layout | noarr::offset<'x', 'y', 'z', 's'>(x, y - 1, z, 0)) / sizeof(real_t);
						(ab_layout | noarr::get_at<'i', 'j'>(single_substr_ab.get(), i - j, j)) = r_y;
						y_neighbors++;
					}
					if (y < this->problem_.ny - 1)
					{
						y_neighbors++;
					}

					if (z > 0)
					{
						auto j =
							(substrates_layout | noarr::offset<'x', 'y', 'z', 's'>(x, y, z - 1, 0)) / sizeof(real_t);
						(ab_layout | noarr::get_at<'i', 'j'>(single_substr_ab.get(), i - j, j)) = r_z;
						z_neighbors++;
					}
					if (z < this->problem_.nz - 1)
					{
						z_neighbors++;
					}

					(ab_layout | noarr::get_at<'i', 'j'>(single_substr_ab.get(), i - i, i)) =
						1 + this->problem_.dt * this->problem_.decay_rates[s_idx]
						+ this->problem_.dt * this->problem_.diffusion_coefficients[s_idx]
							  * (x_neighbors / (this->problem_.dx * this->problem_.dx)
								 + y_neighbors / (this->problem_.dy * this->problem_.dy)
								 + z_neighbors / (this->problem_.dz * this->problem_.dz));
				}

		int info;
		int n = this->problem_.nx * this->problem_.ny * this->problem_.nz;
		int ldab = kd + 1;
		pbtrf("L", &n, &kd, single_substr_ab.get(), &ldab, &info);

		if (info != 0)
			throw std::runtime_error("LAPACK pbtrf failed with error code " + std::to_string(info));

		ab_.emplace_back(std::move(single_substr_ab));
	}
}

template <typename real_t>
void full_lapack_solver<real_t>::initialize()
{
	precompute_values();
}

template <typename real_t>
void full_lapack_solver<real_t>::solve()
{
	auto dens_l = get_substrates_layout();

	for (index_t i = 0; i < this->problem_.iterations; i++)
	{
#pragma omp for schedule(static) nowait
		for (index_t s = 0; s < this->problem_.substrates_count; s++)
		{
			const std::size_t begin_offset = (dens_l | noarr::offset<'x', 'y', 'z', 's'>(0, 0, 0, s)) / sizeof(real_t);

			int info;
			int n = this->problem_.nx * this->problem_.ny * this->problem_.nz;
			int kd = 1 * (this->problem_.dims >= 2 ? this->problem_.nx : 1)
					 * (this->problem_.dims >= 3 ? this->problem_.ny : 1);
			int rhs = 1;
			int ldab = kd + 1;
			pbtrs("L", &n, &kd, &rhs, ab_[s].get(), &ldab, this->substrates_ + begin_offset, &n, &info);

			if (info != 0)
				throw std::runtime_error("LAPACK pbtrs failed with error code " + std::to_string(info));
		}
	}
}

template class full_lapack_solver<float>;
template class full_lapack_solver<double>;
