#pragma once

#include <noarr/structures_extended.hpp>

#include "problem.h"

struct substrate_layouts
{
	template <std::size_t dims, typename index_t, typename real_t>
	static auto get_sxyz_layout(const problem_t<index_t, real_t>& problem)
	{
		if constexpr (dims == 1)
			return noarr::scalar<real_t>() ^ noarr::vectors<'s', 'x'>(problem.substrates_count, problem.nx);
		else if constexpr (dims == 2)
			return noarr::scalar<real_t>()
				   ^ noarr::vectors<'s', 'x', 'y'>(problem.substrates_count, problem.nx, problem.ny);
		else if constexpr (dims == 3)
			return noarr::scalar<real_t>()
				   ^ noarr::vectors<'s', 'x', 'y', 'z'>(problem.substrates_count, problem.nx, problem.ny, problem.nz);
	}

	template <std::size_t dims, typename index_t, typename real_t>
	static auto get_sxyz_aligned_layout(const problem_t<index_t, real_t>& problem, std::size_t alignment_size)
	{
		std::size_t xs_size = problem.nx * problem.substrates_count * sizeof(real_t);
		std::size_t xs_size_padded = (xs_size + alignment_size - 1) / alignment_size * alignment_size;
		xs_size_padded /= sizeof(real_t);

		if constexpr (dims == 1)
			return noarr::scalar<real_t>() ^ noarr::vectors<'x'>(xs_size_padded)
				   ^ noarr::into_blocks_static<'x', 'b', 'x', 's'>(problem.substrates_count)
				   ^ noarr::fix<'b'>(noarr::lit<0>) ^ noarr::slice<'x'>(problem.nx);
		else if constexpr (dims == 2)
			return noarr::scalar<real_t>() ^ noarr::vectors<'x', 'y'>(xs_size_padded, problem.ny)
				   ^ noarr::into_blocks_static<'x', 'b', 'x', 's'>(problem.substrates_count)
				   ^ noarr::fix<'b'>(noarr::lit<0>) ^ noarr::slice<'x'>(problem.nx);
		else if constexpr (dims == 3)
			return noarr::scalar<real_t>() ^ noarr::vectors<'x', 'y', 'z'>(xs_size_padded, problem.ny, problem.nz)
				   ^ noarr::into_blocks_static<'x', 'b', 'x', 's'>(problem.substrates_count)
				   ^ noarr::fix<'b'>(noarr::lit<0>) ^ noarr::slice<'x'>(problem.nx);
	}

	template <std::size_t dims, typename index_t, typename real_t>
	static auto get_xyzs_layout(const problem_t<index_t, real_t>& problem)
	{
		if constexpr (dims == 1)
			return noarr::scalar<real_t>() ^ noarr::vectors<'x', 's'>(problem.nx, problem.substrates_count);
		else if constexpr (dims == 2)
			return noarr::scalar<real_t>()
				   ^ noarr::vectors<'x', 'y', 's'>(problem.nx, problem.ny, problem.substrates_count);
		else if constexpr (dims == 3)
			return noarr::scalar<real_t>()
				   ^ noarr::vectors<'x', 'y', 'z', 's'>(problem.nx, problem.ny, problem.nz, problem.substrates_count);
	}

	template <std::size_t dims, typename index_t, typename real_t>
	static auto get_xyzs_aligned_layout(const problem_t<index_t, real_t>& problem, std::size_t alignment_size)
	{
		std::size_t x_size = problem.nx * sizeof(real_t);
		std::size_t x_size_padded = (x_size + alignment_size - 1) / alignment_size * alignment_size;
		x_size_padded /= sizeof(real_t);

		if constexpr (dims == 1)
			return noarr::scalar<real_t>() ^ noarr::vectors<'x', 's'>(x_size_padded, problem.substrates_count)
				   ^ noarr::slice<'x'>(problem.nx);
		else if constexpr (dims == 2)
			return noarr::scalar<real_t>()
				   ^ noarr::vectors<'x', 'y', 's'>(x_size_padded, problem.ny, problem.substrates_count)
				   ^ noarr::slice<'x'>(problem.nx);
		else if constexpr (dims == 3)
			return noarr::scalar<real_t>()
				   ^ noarr::vectors<'x', 'y', 'z', 's'>(x_size_padded, problem.ny, problem.nz, problem.substrates_count)
				   ^ noarr::slice<'x'>(problem.nx);
	}
};
