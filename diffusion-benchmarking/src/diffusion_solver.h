#pragma once

#if defined(__clang__)
	#pragma clang diagnostic push
	#pragma clang diagnostic ignored "-Wnan-infinity-disabled"
	#include <nlohmann/json.hpp>
	#pragma clang diagnostic pop
#else
	#include <nlohmann/json.hpp>
#endif


#include "problem.h"

class diffusion_solver
{
public:
	// Allocates common resources
	virtual void prepare(const max_problem_t& problem) = 0;

	// Sets the solver specific parameters
	virtual void tune(const nlohmann::json&) {};

	// Allocates solver specific resources
	virtual void initialize() = 0;

	// Solves the diffusion problem
	virtual void solve() = 0;

	// Saves data to a file in human readable format with the following structure:
	// Each line contains a space-separated list of values for a single point in the grid (so all substrates)
	// The points are ordered in x, y, z order
	virtual void save(std::ostream& out) const = 0;

	// Accesses the value at the given coordinates
	virtual double access(std::size_t s, std::size_t x, std::size_t y, std::size_t z) const = 0;

	virtual ~diffusion_solver() = default;
};
