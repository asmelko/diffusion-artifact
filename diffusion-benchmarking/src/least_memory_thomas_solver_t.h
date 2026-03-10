#pragma once

#include "base_solver.h"
#include "substrate_layouts.h"
#include "tridiagonal_solver.h"

/*
The diffusion is the problem of solving tridiagonal matrix system with these coeficients:
For dimension x:
a_i  == -dt*diffusion_coefs/dx^2                              1 <= i <= n
b_1  == 1 + dt*decay_rates/dims + dt*diffusion_coefs/dx^2
b_i  == 1 + dt*decay_rates/dims + 2*dt*diffusion_coefs/dx^2   1 <  i <  n
b_n  == 1 + dt*decay_rates/dims + dt*diffusion_coefs/dx^2
c_i  == -dt*diffusion_coefs/dx^2                              1 <= i <= n
d_i  == current diffusion rates
For dimension y/z (if they exist):
substitute dx accordingly to dy/dz

Since the matrix is constant for multiple right hand sides, we precompute its values in the following way:
b_1'  == 1/b_1
b_i'  == 1/(b_i - a_i*c_i*b_(i-1)')                           1 <  i <= n
e_i   == a_i*b_(i-1)'                                         1 <  i <= n

Then, the forward substitution is as follows (n multiplications + n subtractions):
d_i'  == d_i - e_i*d_(i-1)                                    1 <  i <= n
The backpropagation (2n multiplications + n subtractions):
d_n'' == d_n'/b_n'
d_i'' == (d_i' - c_i*d_(i+1)'')*b_i'                          n >  i >= 1

Optimizations:
- Precomputed a_i, b_1, b_i'
- Minimized memory accesses by computing e_i on the fly
- Aligned memory for x dimension (tunable by 'alignment_size')
- Better temporal locality of memory accesses - x dimension is divided into smaller tiles (tunable by 'x_tile_size') and
y/z dimensions are solved alongside tiled x dimension
- X dimension is vectorized manually - squares of x*yz plane are loaded into vector registers, then the data is
transposed so the vectorization can be utilized
*/

template <typename real_t, bool aligned_x>
class least_memory_thomas_solver_t : public locally_onedimensional_solver,
									 public base_solver<real_t, least_memory_thomas_solver_t<real_t, aligned_x>>

{
	using index_t = std::int32_t;

	real_t *ax_, *b1x_, *bx_;
	real_t *ay_, *b1y_, *by_;
	real_t *az_, *b1z_, *bz_;

	bool vectorized_x_;
	std::size_t x_tile_size_;
	std::size_t alignment_size_;
	index_t substrate_step_;

	auto get_diagonal_layout(const problem_t<index_t, real_t>& problem_, index_t n);

	void precompute_values(real_t*& a, real_t*& b1, real_t*& b, index_t shape, index_t dims, index_t n);

public:
	least_memory_thomas_solver_t(bool vectorized_x);

	template <std::size_t dims = 3>
	auto get_substrates_layout() const
	{
		if constexpr (aligned_x)
			return substrate_layouts::get_xyzs_aligned_layout<dims>(this->problem_, alignment_size_);
		else
			return substrate_layouts::get_xyzs_layout<dims>(this->problem_);
	}

	void prepare(const max_problem_t& problem) override;

	void tune(const nlohmann::json& params) override;

	void initialize() override;

	void solve_x() override;
	void solve_y() override;
	void solve_z() override;

	void solve() override;

	~least_memory_thomas_solver_t();
};
