#pragma once

#include <memory>

#include "tridiagonal_solver.h"

template <typename real_t>
class biofvm : public locally_onedimensional_solver
{
	using index_t = std::int32_t;
	using long_index_t = std::int64_t;

	problem_t<index_t, real_t> problem_;

	std::unique_ptr<real_t[]> substrates_;

	std::vector<std::vector<real_t>> bx_, cx_; // bx = denomx | cx = cx
	std::vector<std::vector<real_t>> by_, cy_;
	std::vector<std::vector<real_t>> bz_, cz_;
    std::vector<real_t> constant1;
    long_index_t thomas_i_jump;
    long_index_t thomas_j_jump;
    long_index_t thomas_k_jump;

	std::size_t work_items_;

	void precompute_values();

	static auto get_substrates_layout(const problem_t<index_t, real_t>& problem);

public:
	void prepare(const max_problem_t& problem) override;

	void initialize() override; //done

	void solve_x() override;
	void solve_y() override;
	void solve_z() override;

	void solve() override; //done

	void save(std::ostream& out) const override; //done

	double access(std::size_t s, std::size_t x, std::size_t y, std::size_t z) const override; //
};