#pragma once

#include "base_solver.h"
#include "substrate_layouts.h"
#include "tridiagonal_solver.h"

template <typename real_t>
class reference_thomas_solver : public locally_onedimensional_solver,
								public base_solver<real_t, reference_thomas_solver<real_t>>
{
	friend base_solver<real_t, reference_thomas_solver<real_t>>;

	using index_t = std::int32_t;

	std::unique_ptr<real_t[]> ax_, b0x_, bx_;
	std::unique_ptr<real_t[]> ay_, b0y_, by_;
	std::unique_ptr<real_t[]> az_, b0z_, bz_;

	void precompute_values(std::unique_ptr<real_t[]>& a, std::unique_ptr<real_t[]>& b, std::unique_ptr<real_t[]>& b0,
						   index_t shape, index_t dims, index_t n);

public:
	auto get_substrates_layout() const { return substrate_layouts::get_sxyz_layout<3>(this->problem_); }

	void initialize() override;

	void solve_x() override;
	void solve_y() override;
	void solve_z() override;

	void solve() override;
};
