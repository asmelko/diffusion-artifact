#pragma once

#include "../base_solver.h"
#include "../substrate_layouts.h"
#include "../tridiagonal_solver.h"


template <typename real_t, bool aligned_x>
class sdd_least_compute_thomas_solver_t
	: public locally_onedimensional_solver,
	  public base_solver<real_t, sdd_least_compute_thomas_solver_t<real_t, aligned_x>>
{
protected:
	using index_t = std::int32_t;

	real_t *ax_ = nullptr, *bx_ = nullptr, *cx_ = nullptr;
	real_t *ay_ = nullptr, *by_ = nullptr, *cy_ = nullptr;
	real_t *az_ = nullptr, *bz_ = nullptr, *cz_ = nullptr;
	std::vector<real_t*> b_scratch_;

	std::size_t xs_tile_size_;
	std::size_t alignment_size_;

	void precompute_values(real_t*& a, real_t*& b, real_t*& c, index_t shape, index_t n, index_t dims, char dim);

	template <char dim>
	auto get_diagonal_layout()
	{
		const index_t s = std::max((index_t)xs_tile_size_, (index_t)this->problem_.substrates_count);
		const auto n = std::max({ this->problem_.nx, this->problem_.ny, this->problem_.nz });

		if constexpr (aligned_x)
		{
			std::size_t ns_size = n * s * sizeof(real_t);
			std::size_t ns_size_padded = (ns_size + alignment_size_ - 1) / alignment_size_ * alignment_size_;
			ns_size_padded /= sizeof(real_t);

			return noarr::scalar<real_t>() ^ noarr::vectors<'s', dim>(ns_size_padded, n);
		}
		else
			return noarr::scalar<real_t>() ^ noarr::vectors<'s', dim>(s, n);
	}

public:
	template <std::size_t dims = 3>
	auto get_substrates_layout() const
	{
		if constexpr (aligned_x)
			return substrate_layouts::get_sxyz_aligned_layout<dims>(this->problem_, alignment_size_);
		else
			return substrate_layouts::get_sxyz_layout<dims>(this->problem_);
	}

	void prepare(const max_problem_t& problem) override;

	void tune(const nlohmann::json& params) override;

	void initialize() override;

	void solve_x() override;
	void solve_y() override;
	void solve_z() override;

	void solve() override;

	~sdd_least_compute_thomas_solver_t();
};
