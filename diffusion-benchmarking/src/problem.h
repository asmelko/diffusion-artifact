#pragma once

#include <string>
#include <vector>

template <typename num_t, typename real_t>
struct problem_t
{
	num_t dims;
	num_t dx, dy, dz;
	num_t nx, ny, nz;
	num_t substrates_count;

	num_t iterations;

	real_t dt;
	std::vector<real_t> diffusion_coefficients;
	std::vector<real_t> decay_rates;
	std::vector<real_t> initial_conditions;

	bool gaussian_pulse;

	problem_t()
		: dims(1),
		  dx(20),
		  dy(20),
		  dz(20),
		  nx(1),
		  ny(1),
		  nz(1),
		  substrates_count(1),
		  iterations(1),
		  dt(0.01),
		  gaussian_pulse(false)
	{}
};

using max_problem_t = problem_t<std::size_t, double>;

class problems
{
public:
	template <typename out_num_t, typename out_real_t, typename in_num_t, typename in_real_t>
	static problem_t<out_num_t, out_real_t> cast(const problem_t<in_num_t, in_real_t>& problem)
	{
		problem_t<out_num_t, out_real_t> other_problem;
		other_problem.dims = static_cast<out_num_t>(problem.dims);
		other_problem.dx = static_cast<out_num_t>(problem.dx);
		other_problem.dy = static_cast<out_num_t>(problem.dy);
		other_problem.dz = static_cast<out_num_t>(problem.dz);
		other_problem.nx = static_cast<out_num_t>(problem.nx);
		other_problem.ny = static_cast<out_num_t>(problem.ny);
		other_problem.nz = static_cast<out_num_t>(problem.nz);
		other_problem.substrates_count = static_cast<out_num_t>(problem.substrates_count);
		other_problem.iterations = static_cast<out_num_t>(problem.iterations);
		other_problem.dt = static_cast<out_real_t>(problem.dt);
		other_problem.diffusion_coefficients =
			std::vector<out_real_t>(problem.diffusion_coefficients.begin(), problem.diffusion_coefficients.end());
		other_problem.decay_rates = std::vector<out_real_t>(problem.decay_rates.begin(), problem.decay_rates.end());
		other_problem.initial_conditions =
			std::vector<out_real_t>(problem.initial_conditions.begin(), problem.initial_conditions.end());
		other_problem.gaussian_pulse = problem.gaussian_pulse;
		return other_problem;
	}

	static max_problem_t read_problem(const std::string& file);
};
