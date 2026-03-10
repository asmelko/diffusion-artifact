#pragma once

#include "diffusion_solver.h"

class locally_onedimensional_solver : public virtual diffusion_solver
{
public:
	virtual void solve_x() = 0;
	virtual void solve_y() = 0;
	virtual void solve_z() = 0;
};
