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
a_i' = c_i' = -a_i
b_1'  == 1/b_1
b_i'  == 1/(b_i - a_i'*c_i'*b_(i-1)')                         1 <  i <= n
e_i   == a_i'*b_(i-1)'                                        1 <  i <= n

Then, the forward substitution is as follows (n FMAs):
d_i'  == d_i + e_i*d_(i-1)                                    1 <  i <= n
The backpropagation (n multiplications + n FMAs):
d_n'' == d_n'/b_n'
d_i'' == (d_i' + c_i*d_(i+1)'')*b_i'                          n >  i >= 1

Optimizations:
- Precomputed a_i, b_i', e_i
- Aligned memory for x dimension (tunable by 'alignment_size')
- Better temporal locality of memory accesses - sx plane is divided into smaller tiles (tunable by 'xs_tile_size') and
y/z dimensions are solved alongside tiled xs dimension
*/

template <typename real_t, bool aligned_x>
class cyclic_reduction_solver : public locally_onedimensional_solver,
								public base_solver<real_t, cyclic_reduction_solver<real_t, aligned_x>>
{
protected:
	using index_t = std::int32_t;

	real_t *ax_, *b1x_;
	real_t *ay_, *b1y_;
	real_t *az_, *b1z_;

	std::vector<real_t*> a_scratch_, b_scratch_, c_scratch_;

	std::size_t x_tile_size_;
	std::size_t alignment_size_;

	void precompute_values(real_t*& a, real_t*& b1, index_t shape, index_t dims);

	auto get_diagonal_layout(const problem_t<index_t, real_t>& problem, index_t n);

public:
	cyclic_reduction_solver();

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

	~cyclic_reduction_solver();
};
