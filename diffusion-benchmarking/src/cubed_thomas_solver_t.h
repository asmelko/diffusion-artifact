#pragma once

#include <functional>
#include <iostream>
#include <memory>

#include "base_solver.h"
#include "blocked_thomas_solver.h"
#include "substrate_layouts.h"
#include "tridiagonal_solver.h"

/*

Restrictions:
- dimension sizes must be divisible by block size
- dimension sizes must all be the same
*/

template <typename real_t, bool aligned_x>
class cubed_thomas_solver_t : public locally_onedimensional_solver,
							  public base_solver<real_t, cubed_thomas_solver_t<real_t, aligned_x>>
{
protected:
	using index_t = std::int32_t;

	real_t *ax_, *b1x_;
	real_t *ay_, *b1y_;
	real_t *az_, *b1z_;

	std::unique_ptr<aligned_atomic<long>[]> countersx_, countersy_, countersz_;
	index_t countersx_count_, countersy_count_, countersz_count_;

	real_t *a_scratchx_, *c_scratchx_;
	real_t *a_scratchy_, *c_scratchy_;
	real_t *a_scratchz_, *c_scratchz_;

	std::size_t alignment_size_;
	std::size_t x_tile_size_;

	bool vectorized_x_;

	std::array<index_t, 3> cores_division_;

	std::array<index_t, 3> group_blocks_;
	std::vector<index_t> group_block_lengthsx_;
	std::vector<index_t> group_block_lengthsy_;
	std::vector<index_t> group_block_lengthsz_;

	std::vector<index_t> group_block_offsetsx_;
	std::vector<index_t> group_block_offsetsy_;
	std::vector<index_t> group_block_offsetsz_;

	index_t aligned_block_size_;

	using sync_func_t = std::function<void>;

	void precompute_values(real_t*& a, real_t*& b1, index_t shape, index_t dims, index_t n, index_t counters_count,
						   std::unique_ptr<aligned_atomic<long>[]>& counters, index_t group_size, index_t& block_size,
						   std::vector<index_t>& group_block_lengths, std::vector<index_t>& group_block_offsets,
						   bool aligned);

	auto get_diagonal_layout(const problem_t<index_t, real_t>& problem, index_t n);

public:
	static constexpr index_t min_block_size = 2;

	cubed_thomas_solver_t(bool vectorized_x);

	template <std::size_t dims = 3>
	auto get_substrates_layout() const
	{
		if constexpr (aligned_x)
			return noarr::scalar<real_t>()
				   ^ noarr::vectors<'x', 'y', 'z', 's'>(aligned_block_size_ * (this->problem_.ny / cores_division_[0]),
														this->problem_.ny, this->problem_.nz,
														this->problem_.substrates_count);
		else
			return substrate_layouts::get_xyzs_layout<dims>(this->problem_);
	}

	auto get_scratch_layout(const index_t n, const index_t groups) const
	{
		if constexpr (aligned_x)
		{
			std::size_t size = n * sizeof(real_t);
			std::size_t size_padded = (size + alignment_size_ - 1) / alignment_size_ * alignment_size_;
			size_padded /= sizeof(real_t);
			return noarr::scalar<real_t>()
				   ^ noarr::vectors<'i', 'l', 's'>(size_padded, groups, this->problem_.substrates_count)
				   ^ noarr::slice<'i'>(n);
		}
		else
			return noarr::scalar<real_t>() ^ noarr::vectors<'i', 'l', 's'>(n, groups, this->problem_.substrates_count);
	}

	std::function<void> get_synchronization_function();

	void prepare(const max_problem_t& problem) override;

	void tune(const nlohmann::json& params) override;

	void initialize() override;

	void solve_x() override;
	void solve_y() override;
	void solve_z() override;

	void solve() override;

	virtual double access(std::size_t s, std::size_t x, std::size_t y, std::size_t z) const override;

	real_t& at(std::size_t s, std::size_t x, std::size_t y, std::size_t z) const;


	~cubed_thomas_solver_t();
};
