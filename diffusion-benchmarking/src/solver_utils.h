#pragma once

#include <math.h>

#include <noarr/traversers.hpp>

#include "noarr/structures/extra/funcs.hpp"
#include "omp_helper.h"
#include "problem.h"

class solver_utils
{
public:
	template <typename index_t, typename real_t>
	static real_t gaussian_analytical_solution(index_t s, index_t x, index_t y, index_t z,
											   const problem_t<index_t, real_t>& problem)
	{
		constexpr real_t initial_pulse_time = 0.01;

		real_t x_coord = 0, y_coord = 0, z_coord = 0;
		// 1D
		{
			real_t length_x = problem.dx * problem.nx;
			real_t begin_x = -length_x / 2;
			real_t step_x = length_x / (problem.nx - 1);
			x_coord = begin_x + x * step_x;
		}
		// 2D
		if (problem.dims >= 2)
		{
			real_t length_y = problem.dy * problem.ny;
			real_t begin_y = -length_y / 2;
			real_t step_y = length_y / (problem.ny - 1);
			y_coord = begin_y + y * step_y;
		}
		// 3D
		if (problem.dims >= 3)
		{
			real_t length_z = problem.dz * problem.nz;
			real_t begin_z = -length_z / 2;
			real_t step_z = length_z / (problem.nz - 1);
			z_coord = begin_z + z * step_z;
		}

		return std::exp(-(x_coord * x_coord + y_coord * y_coord + z_coord * z_coord)
						/ (4 * problem.diffusion_coefficients[s] * initial_pulse_time))
			   * std::exp(-problem.decay_rates[s] * initial_pulse_time) * problem.initial_conditions[s]
			   / std::pow(4 * M_PI * problem.diffusion_coefficients[s] * initial_pulse_time, problem.dims / 2);
	}

	template <typename index_t, typename real_t>
	static void initialize_substrate(auto substrates_layout, real_t* substrates,
									 const problem_t<index_t, real_t>& problem)
	{
		if (problem.gaussian_pulse)
		{
			initialize_gaussian_pulse(substrates_layout, substrates, problem);
		}
		else
		{
			initialize_substrate_constant(substrates_layout, substrates, problem.initial_conditions.data());
		}
	}

	template <typename real_t>
	static void initialize_substrate_constant(auto substrates_layout, real_t* substrates,
											  const real_t* initial_conditions)
	{
		omp_trav_for_each(noarr::traverser(substrates_layout), [&](auto state) {
			auto s_idx = noarr::get_index<'s'>(state);

			(substrates_layout | noarr::get_at(substrates, state)) = initial_conditions[s_idx];
		});
	}

	template <typename real_t>
	static void initialize_substrate_linear(auto substrates_layout, real_t* substrates)
	{
		omp_trav_for_each(noarr::traverser(substrates_layout), [&](auto state) {
			auto offset = (substrates_layout | noarr::offset(state)) / sizeof(real_t);

			(substrates_layout | noarr::get_at(substrates, state)) = offset;
		});
	}

	template <typename index_t, typename real_t>
	static void initialize_gaussian_pulse(auto substrates_layout, real_t* substrates,
										  const problem_t<index_t, real_t>& problem)
	{
		omp_trav_for_each(noarr::traverser(substrates_layout), [&](auto state) {
			index_t s = noarr::get_index<'s'>(state);
			index_t x = noarr::get_index<'x'>(state);
			index_t y = noarr::get_index<'y'>(state);
			index_t z = noarr::get_index<'z'>(state);

			(substrates_layout | noarr::get_at(substrates, state)) = gaussian_analytical_solution(s, x, y, z, problem);
		});
	}
};
