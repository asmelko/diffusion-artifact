#pragma once

#include "base_solver.h"
#include "substrate_layouts.h"

/*
The diffusion is the problem of solving 3/5/7-diagonal matrix system, depending on the dimensionality of the problem.

This algorithm uses LAPACK pbtrf+pbtrs routines to solve the X-diagonal matrix system.
*/

template <typename real_t>
class full_lapack_solver : public base_solver<real_t, full_lapack_solver<real_t>>
{
	using index_t = std::int32_t;

	std::vector<std::unique_ptr<real_t[]>> ab_;

	void precompute_values();

	static void pbtrf(const char* uplo, const int* n, const int* kd, real_t* ab, const int* ldab, int* info);
	static void pbtrs(const char* uplo, const int* n, const int* kd, const int* nrhs, const real_t* ab, const int* ldab,
					  real_t* b, const int* ldb, int* info);

public:
	auto get_substrates_layout() const { return substrate_layouts::get_xyzs_layout<3>(this->problem_); }

	void initialize() override;

	void solve() override;
};
