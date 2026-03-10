#include "biofvm.h"

#include <iostream>
#include <vector>

#include "perf_utils.h"
#include "solver_utils.h"

template <typename real_t>
void biofvm<real_t>::precompute_values()
{
	thomas_i_jump = problem_.substrates_count;
	thomas_j_jump = thomas_i_jump * problem_.nx;
	thomas_k_jump = thomas_j_jump * problem_.ny;

	std::vector<real_t> zero(problem_.substrates_count, 0.0);
	std::vector<real_t> one(problem_.substrates_count, 1.0);
	real_t dt = problem_.dt;

	// Thomas initialization
	bx_.resize(problem_.nx, zero); // sizeof(x_coordinates) = local_x_nodes, denomx is the main diagonal elements
	cx_.resize(problem_.nx, zero); // Both b and c of tridiagonal matrix are equal, hence just one array needed

	by_.resize(problem_.ny, zero);
	cy_.resize(problem_.ny, zero);

	bz_.resize(problem_.nz, zero);
	cz_.resize(problem_.nz, zero);

	constant1 = problem_.diffusion_coefficients;		  // dt*D/dx^2
	std::vector<real_t> constant1a = zero;				  // -dt*D/dx^2;
	std::vector<real_t> constant2 = problem_.decay_rates; // (1/3)* dt*lambda
	std::vector<real_t> constant3 = one;				  // 1 + 2*constant1 + constant2;
	std::vector<real_t> constant3a = one;				  // 1 + constant1 + constant2;

	for (index_t d = 0; d < problem_.substrates_count; d++)
	{
		constant1[d] *= dt;
		constant1[d] /= problem_.dx; // dx
		constant1[d] /= problem_.dx; // dx

		constant1a[d] = constant1[d];
		constant1a[d] *= -1.0;

		constant2[d] *= dt;
		constant2[d] /= 3.0; // for the LOD splitting of the source, division by 3 is for 3-D

		constant3[d] += constant1[d];
		constant3[d] += constant1[d];
		constant3[d] += constant2[d];

		constant3a[d] += constant1[d];
		constant3a[d] += constant2[d];
	}


	// Thomas solver coefficients

	cx_.assign(problem_.nx, constant1a); // Fill b and c elements with -D * dt/dx^2
	bx_.assign(problem_.nx, constant3);	 // Fill diagonal elements with (1 + 1/3 * lambda * dt + 2*D*dt/dx^2)
	bx_[0] = constant3a;
	bx_[problem_.nx - 1] = constant3a;
	if (problem_.nx == 1)
	{
		bx_[0] = one;
		for (index_t d = 0; d < problem_.substrates_count; d++)
			bx_[0][d] += constant2[d];
	}

	for (index_t d = 0; d < problem_.substrates_count; d++)
		cx_[0][d] /= bx_[0][d];
	for (index_t i = 1; i <= problem_.nx - 1; i++)
	{
		for (index_t d = 0; d < problem_.substrates_count; d++)
		{
			bx_[i][d] += constant1[d] * cx_[i - 1][d];
			cx_[i][d] /= bx_[i][d]; // the value at  size-1 is not actually used
		}
	}

	cy_.assign(problem_.ny, constant1a); // Fill b and c elements with -D * dt/dy^2
	by_.assign(problem_.ny, constant3);	 // Fill diagonal elements with (1 + 1/3 * lambda * dt + 2*D*dt/dy^2)
	by_[0] = constant3a;
	by_[problem_.ny - 1] = constant3a;
	if (problem_.ny == 1)
	{
		by_[0] = one;
		for (int d = 0; d < problem_.substrates_count; d++)
			by_[0][d] += constant2[d];
	}

	for (int d = 0; d < problem_.substrates_count; d++)
		cy_[0][d] /= by_[0][d];
	for (index_t i = 1; i <= problem_.ny - 1; i++)
	{
		for (index_t d = 0; d < problem_.substrates_count; d++)
		{
			by_[i][d] += constant1[d] * cy_[i - 1][d];
			cy_[i][d] /= by_[i][d]; // the value at  size-1 is not actually used
		}
	}

	cz_.assign(problem_.nz, constant1a); // Fill b and c elements with -D * dt/dz^2
	bz_.assign(problem_.nz, constant3);	 // Fill diagonal elements with (1 + 1/3 * lambda * dt + 2*D*dt/dz^2)
	bz_[0] = constant3a;
	bz_[problem_.nz - 1] = constant3a;
	if (problem_.nz == 1)
	{
		bz_[0] = one;
		for (index_t d = 0; d < problem_.substrates_count; d++)
			bz_[0][d] += constant2[d];
	}

	for (index_t d = 0; d < problem_.substrates_count; d++)
		cz_[0][d] /= bz_[0][d];
	for (index_t i = 1; i <= problem_.nz - 1; i++)
	{
		for (index_t d = 0; d < problem_.substrates_count; d++)
		{
			bz_[i][d] += constant1[d] * cz_[i - 1][d];
			cz_[i][d] /= bz_[i][d]; // the value at  size-1 is not actually used
		}
	}
}

template <typename real_t>
void biofvm<real_t>::initialize()
{
	precompute_values();
}
template <typename real_t>
auto biofvm<real_t>::get_substrates_layout(const problem_t<index_t, real_t>& problem)
{
	return noarr::scalar<real_t>()
		   ^ noarr::vectors<'s', 'x', 'y', 'z'>(problem.substrates_count, problem.nx, problem.ny, problem.nz);
}


template <typename real_t>
void biofvm<real_t>::prepare(const max_problem_t& problem)
{
	problem_ = problems::cast<std::int32_t, real_t>(problem);
	long long int array_size = static_cast<long long int>(problem_.nx) * static_cast<long long int>(problem_.ny)
							   * static_cast<long long int>(problem_.nz)
							   * static_cast<long long int>(problem_.substrates_count);
	substrates_ = std::make_unique<real_t[]>(array_size);

	// Initialize substrates

	auto substrates_layout = get_substrates_layout(problem_);

	solver_utils::initialize_substrate(substrates_layout, substrates_.get(), problem_);
}

template <typename real_t>
void biofvm<real_t>::solve()
{
#pragma omp parallel
	{
		perf_counter counter("biofvm");

		for (index_t i = 0; i < this->problem_.iterations; i++)
		{
			solve_x();
			solve_y();
			solve_z();
		}
	}
}

template <typename real_t>
void biofvm<real_t>::solve_x()
{
#pragma omp for collapse(2)
	for (index_t k = 0; k < problem_.nz; k++)
	{
		for (index_t j = 0; j < problem_.ny; j++)
		{
			long_index_t index = k * thomas_k_jump + j * thomas_j_jump;
			//(*(*M.p_density_vectors))[n] /= M.thomas_denomz[0];
			for (index_t d = 0; d < problem_.substrates_count; d++)
			{
				substrates_[index + d] /= bx_[0][d];
			}

			// should be an empty loop if mesh.z_coordinates.size() < 2
			for (index_t i = 1; i < problem_.nx; i++)
			{
				long_index_t index_inc = index + thomas_i_jump;
				// axpy(&(*(*M.p_density_vectors))[n], M.thomas_constant1, (*(*M.p_density_vectors))[n -
				// M.thomas_k_jump]);
				for (index_t d = 0; d < problem_.substrates_count; d++)
				{
					substrates_[index_inc + d] += constant1[d] * substrates_[index + d];
				}
				//(*(*M.p_density_vectors))[n] /= M.thomas_denomz[k];
				for (index_t d = 0; d < problem_.substrates_count; d++)
				{
					substrates_[index_inc + d] /= bx_[i][d];
				}

				index = index_inc;
			}

			index = k * thomas_k_jump + j * thomas_j_jump + (thomas_i_jump * (problem_.nx - 1));
			for (index_t i = problem_.nx - 2; i >= 0; i--)
			{
				long_index_t index_dec = index - thomas_i_jump;
				// naxpy(&(*(*M.p_density_vectors))[n], M.thomas_cz[k], (*(*M.p_density_vectors))[n + M.thomas_k_jump]);
				for (index_t d = 0; d < problem_.substrates_count; d++)
				{
					substrates_[index_dec + d] -= cx_[i][d] * substrates_[index + d];
				}
				index = index_dec;
			}
		}
	}
}

template <typename real_t>
void biofvm<real_t>::solve_y()
{
#pragma omp for collapse(2)
	for (index_t k = 0; k < problem_.nz; k++)
	{
		for (index_t i = 0; i < problem_.nx; i++)
		{
			index_t index = k * thomas_k_jump + i * thomas_i_jump;
			//(*(*M.p_density_vectors))[n] /= M.thomas_denomz[0];
			for (index_t d = 0; d < problem_.substrates_count; d++)
			{
				substrates_[index + d] /= by_[0][d];
			}

			// should be an empty loop if mesh.z_coordinates.size() < 2
			for (index_t j = 1; j < problem_.ny; j++)
			{
				index_t index_inc = index + thomas_j_jump;
				// axpy(&(*(*M.p_density_vectors))[n], M.thomas_constant1, (*(*M.p_density_vectors))[n -
				// M.thomas_k_jump]);
				for (index_t d = 0; d < problem_.substrates_count; d++)
				{
					substrates_[index_inc + d] += constant1[d] * substrates_[index + d];
				}
				//(*(*M.p_density_vectors))[n] /= M.thomas_denomz[k];
				for (index_t d = 0; d < problem_.substrates_count; d++)
				{
					substrates_[index_inc + d] /= by_[j][d];
				}

				index = index_inc;
			}

			index = k * thomas_k_jump + i * thomas_i_jump + (thomas_j_jump * (problem_.ny - 1));
			for (index_t j = problem_.ny - 2; j >= 0; j--)
			{
				index_t index_dec = index - thomas_j_jump;
				// naxpy(&(*(*M.p_density_vectors))[n], M.thomas_cz[k], (*(*M.p_density_vectors))[n + M.thomas_k_jump]);
				for (index_t d = 0; d < problem_.substrates_count; d++)
				{
					substrates_[index_dec + d] -= cy_[j][d] * substrates_[index + d];
				}
				index = index_dec;
			}
		}
	}
}

template <typename real_t>
void biofvm<real_t>::solve_z()
{
#pragma omp for collapse(2)
	for (index_t j = 0; j < problem_.ny; j++)
	{
		for (index_t i = 0; i < problem_.nx; i++)
		{
			long_index_t index = j * thomas_j_jump + i * thomas_i_jump;
			//(*(*M.p_density_vectors))[n] /= M.thomas_denomz[0];
			for (index_t d = 0; d < problem_.substrates_count; d++)
			{
				substrates_[index + d] /= bz_[0][d];
			}

			// should be an empty loop if mesh.z_coordinates.size() < 2
			for (index_t k = 1; k < problem_.nz; k++)
			{
				long_index_t index_inc = index + thomas_k_jump;
				// axpy(&(*(*M.p_density_vectors))[n], M.thomas_constant1, (*(*M.p_density_vectors))[n -
				// M.thomas_k_jump]);
				for (index_t d = 0; d < problem_.substrates_count; d++)
				{
					substrates_[index_inc + d] += constant1[d] * substrates_[index + d];
				}
				//(*(*M.p_density_vectors))[n] /= M.thomas_denomz[k];
				for (index_t d = 0; d < problem_.substrates_count; d++)
				{
					substrates_[index_inc + d] /= bz_[k][d];
				}

				index = index_inc;
			}

			index = i * thomas_i_jump + j * thomas_j_jump + (thomas_k_jump * (problem_.nz - 1));
			for (index_t k = problem_.nz - 2; k >= 0; k--)
			{
				long_index_t index_dec = index - thomas_k_jump;
				// naxpy(&(*(*M.p_density_vectors))[n], M.thomas_cz[k], (*(*M.p_density_vectors))[n + M.thomas_k_jump]);
				for (index_t d = 0; d < problem_.substrates_count; d++)
				{
					substrates_[index_dec + d] -= cz_[k][d] * substrates_[index + d];
				}
				index = index_dec;
			}
		}
	}
}

template <typename real_t>
void biofvm<real_t>::save(std::ostream& out) const
{
	auto dens_l = get_substrates_layout(problem_);

	for (index_t z = 0; z < problem_.nz; z++)
		for (index_t y = 0; y < problem_.ny; y++)
			for (index_t x = 0; x < problem_.nx; x++)
			{
				out << "(" << x << ", " << y << ", " << z << ")";
				for (index_t s = 0; s < problem_.substrates_count; s++)
					out << (dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_.get(), s, x, y, z)) << " ";
				out << std::endl;
			}
}

template <typename real_t>
double biofvm<real_t>::access(std::size_t s, std::size_t x, std::size_t y, std::size_t z) const
{
	// auto dens_l = get_substrates_layout(problem_);

	//return (dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_.get(), s, x, y, z));
	return substrates_[z * thomas_k_jump + y * thomas_j_jump + x * thomas_i_jump + s];
}

template class biofvm<float>;
template class biofvm<double>;
