#pragma once

#include <noarr/structures.hpp>

#include "diffusion_solver.h"
#include "solver_utils.h"

template <typename real_t, class derived_solver>
class base_solver : public virtual diffusion_solver
{
protected:
	using index_t = std::int32_t;

	problem_t<index_t, real_t> problem_;

	real_t* substrates_ = nullptr;

	real_t access_internal(std::size_t s, std::size_t x, std::size_t y, std::size_t z) const
	{
		auto dens_l = static_cast<const derived_solver*>(this)->get_substrates_layout();

		return (dens_l | noarr::get_at<'s', 'x', 'y', 'z'>(substrates_, s, x, y, z));
	}

public:
	void prepare(const max_problem_t& problem) override
	{
		problem_ = problems::cast<std::int32_t, real_t>(problem);

		auto substrates_layout = static_cast<derived_solver*>(this)->get_substrates_layout();
		substrates_ = (real_t*)std::malloc((substrates_layout | noarr::get_size()));

		// Initialize substrates
		solver_utils::initialize_substrate(substrates_layout, substrates_, problem_);
	}

	virtual void save(std::ostream& out) const override
	{
		for (index_t z = 0; z < problem_.nz; z++)
			for (index_t y = 0; y < problem_.ny; y++)
				for (index_t x = 0; x < problem_.nx; x++)
				{
					for (index_t s = 0; s < problem_.substrates_count; s++)
						out << access(s, x, y, z) << " ";
					out << std::endl;
				}
	}

	virtual double access(std::size_t s, std::size_t x, std::size_t y, std::size_t z) const override
	{
		return (double)access_internal(s, x, y, z);
	}

	virtual ~base_solver() override
	{
		if (substrates_)
			std::free(substrates_);
	}
};
