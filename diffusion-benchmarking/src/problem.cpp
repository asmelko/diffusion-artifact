#include "problem.h"

#include <fstream>

#if defined(__clang__)
	#pragma clang diagnostic push
	#pragma clang diagnostic ignored "-Wnan-infinity-disabled"
	#include <nlohmann/json.hpp>
	#pragma clang diagnostic pop
#else
	#include <nlohmann/json.hpp>
#endif

max_problem_t problems::read_problem(const std::string& file)
{
	std::ifstream ifs(file);
	if (!ifs)
		throw std::runtime_error("Cannot open file " + file);

	nlohmann::json j;
	ifs >> j;

	max_problem_t problem;
	problem.dims = j["dims"];
	problem.dx = j["dx"];
	problem.nx = j["nx"];

	if (problem.nx < 2)
		throw std::runtime_error("nx must be at least 2");

	if (problem.dims >= 2)
	{
		problem.dy = j["dy"];
		problem.ny = j["ny"];

		if (problem.ny < 2)
			throw std::runtime_error("ny must be at least 2");
	}

	if (problem.dims >= 3)
	{
		problem.dz = j["dz"];
		problem.nz = j["nz"];

		if (problem.nz < 2)
			throw std::runtime_error("nz must be at least 2");
	}

	problem.substrates_count = j["substrates_count"];
	problem.iterations = j["iterations"];
	problem.dt = j["dt"];

	if (j["diffusion_coefficients"].is_array())
		problem.diffusion_coefficients = j["diffusion_coefficients"].get<std::vector<double>>();
	else
		problem.diffusion_coefficients = std::vector<double>(problem.substrates_count, j["diffusion_coefficients"]);

	if (j["decay_rates"].is_array())
		problem.decay_rates = j["decay_rates"].get<std::vector<double>>();
	else
		problem.decay_rates = std::vector<double>(problem.substrates_count, j["decay_rates"]);

	if (j["initial_conditions"].is_array())
		problem.initial_conditions = j["initial_conditions"].get<std::vector<double>>();
	else
		problem.initial_conditions = std::vector<double>(problem.substrates_count, j["initial_conditions"]);

	if (problem.diffusion_coefficients.size() != problem.substrates_count)
		throw std::runtime_error("diffusion_coefficients size does not match substrates_count");

	if (problem.decay_rates.size() != problem.substrates_count)
		throw std::runtime_error("decay_rates size does not match substrates_count");

	if (problem.initial_conditions.size() != problem.substrates_count)
		throw std::runtime_error("initial_conditions size does not match substrates_count");

	if (problem.dims < 1 || problem.dims > 3)
		throw std::runtime_error("dims must be in range [1, 3]");

	if (j.contains("gaussian_pulse"))
		problem.gaussian_pulse = j["gaussian_pulse"];

	return problem;
}
