#pragma once

#include <iostream>

#include "../barrier.h"
#include "../base_solver.h"
#include "../least_memory_thomas_solver_d_f.h"
#include "../substrate_layouts.h"
#include "../tridiagonal_solver.h"
#include "../vector_transpose_helper.h"

template <typename real_t, bool aligned_x>
class sdd_full_blocking : public locally_onedimensional_solver,
						  public base_solver<real_t, sdd_full_blocking<real_t, aligned_x>>

{
	using index_t = std::int32_t;

	std::unique_ptr<std::unique_ptr<aligned_atomic<index_t>>[]> countersx_, countersy_, countersz_;
	std::unique_ptr<std::unique_ptr<std::barrier<>>[]> barriersx_, barriersy_, barriersz_;
	index_t countersx_count_, countersy_count_, countersz_count_;

	std::unique_ptr<real_t*[]> ax_, bx_, cx_;
	std::unique_ptr<real_t*[]> ay_, by_, cy_;
	std::unique_ptr<real_t*[]> az_, bz_, cz_;
	std::unique_ptr<real_t*[]> thread_substrate_array_;
	std::unique_ptr<real_t*[]> a_scratch_, c_scratch_;
	std::unique_ptr<real_t*[]> a_streamx_, c_streamx_, d_streamx_;
	std::unique_ptr<real_t*[]> a_streamy_, c_streamy_, d_streamy_;
	std::unique_ptr<real_t*[]> a_streamz_, c_streamz_, d_streamz_;

	std::unique_ptr<std::vector<barrier_t<true, index_t>>[]> barriers_wrapper_x_;
	std::unique_ptr<std::vector<barrier_t<true, index_t>>[]> barriers_wrapper_y_;
	std::unique_ptr<std::vector<barrier_t<true, index_t>>[]> barriers_wrapper_z_;

	index_t x_tile_size_;
	std::size_t alignment_size_;
	bool fuse_z_;
	bool alt_blocked_;

	std::array<index_t, 3> cores_division_;

	index_t x_sync_step_ = 1, y_sync_step_ = 1, z_sync_step_ = 1;
	index_t streams_count_;

	std::array<index_t, 3> group_blocks_;
	std::vector<index_t> group_block_lengthsx_;
	std::vector<index_t> group_block_lengthsy_;
	std::vector<index_t> group_block_lengthsz_;
	std::vector<index_t> group_block_lengthss_;

	std::vector<index_t> group_block_offsetsx_;
	std::vector<index_t> group_block_offsetsy_;
	std::vector<index_t> group_block_offsetsz_;
	std::vector<index_t> group_block_offsetss_;

	index_t substrate_groups_;

	auto get_stream_layout(const index_t coop_size) const
	{
		using simd_tag = hn::ScalableTag<real_t>;
		simd_tag d;

		auto layout = noarr::scalar<real_t>() ^ noarr::vector<'v'>() ^ noarr::vector<'i'>() ^ noarr::vector<'j'>()
					  ^ noarr::vector<'s'>();

		return layout ^ noarr::set_length<'v', 'i', 's'>(hn::Lanes(d), coop_size * 2, streams_count_);
	}

	template <char dim_to_skip = ' '>
	auto get_blocked_substrate_layout(index_t nx, index_t ny, index_t nz, index_t substrates_count) const
	{
		std::size_t x_size = nx * sizeof(real_t);
		std::size_t x_size_padded = (x_size + alignment_size_ - 1) / alignment_size_ * alignment_size_;
		x_size_padded /= sizeof(real_t);

		auto layout = noarr::scalar<real_t>() ^ noarr::vector<'x'>() ^ noarr::vector<'y'>() ^ noarr::vector<'z'>()
					  ^ noarr::vector<'s'>();

		if constexpr (dim_to_skip == 'x')
			return layout ^ noarr::set_length<'y', 'z', 's'>(ny, nz, substrates_count);
		else if constexpr (dim_to_skip == 'y')
			return layout ^ noarr::set_length<'x', 'z', 's'>(x_size_padded, nz, substrates_count);
		else if constexpr (dim_to_skip == 'z')
			return layout ^ noarr::set_length<'x', 'y', 's'>(x_size_padded, ny, substrates_count);
		else if constexpr (dim_to_skip == '*')
			return layout ^ noarr::set_length<'s'>(substrates_count);
		else
			return layout ^ noarr::set_length<'x', 'y', 'z', 's'>(x_size_padded, ny, nz, substrates_count);
	}

	template <char dim, bool with_len, bool fused_z = true>
	auto get_scratch_layout(index_t nx, index_t ny, index_t nz)
	{
		std::size_t x_size = nx * sizeof(real_t);
		std::size_t x_size_padded = (x_size + alignment_size_ - 1) / alignment_size_ * alignment_size_;
		x_size_padded /= sizeof(real_t);

		if constexpr (dim == 'x')
		{
			using simd_tag = hn::ScalableTag<real_t>;
			simd_tag d;
			std::size_t elements = hn::Lanes(d);

			if constexpr (fused_z)
			{
				std::size_t Y_size = (ny * nz + elements - 1) / elements;

				auto layout = noarr::scalar<real_t>() ^ noarr::vector<'y'>(elements) ^ noarr::vector<'x'>()
							  ^ noarr::vector<'Y'>(Y_size);

				if constexpr (!with_len)
					return layout;
				else
					return layout ^ noarr::set_length<'x'>(nx);
			}
			else
			{
				std::size_t Y_size = (ny + elements - 1) / elements;

				auto layout = noarr::scalar<real_t>() ^ noarr::vector<'y'>(elements) ^ noarr::vector<'x'>()
							  ^ noarr::vector<'Y'>(Y_size) ^ noarr::vector<'z'>(nz);

				if constexpr (!with_len)
					return layout;
				else
					return layout ^ noarr::set_length<'x'>(nx);
			}
		}
		else if constexpr (dim == 'y')
		{
			auto tile_size = std::min<index_t>(x_tile_size_, x_size_padded);
			auto X_len = (x_size_padded + tile_size - 1) / tile_size;

			auto layout = noarr::scalar<real_t>() ^ noarr::vector<'x'>(tile_size) ^ noarr::vector<'y'>()
						  ^ noarr::vector<'X'>(X_len) ^ noarr::vector<'z'>(nz);

			if constexpr (!with_len)
				return layout;
			else
				return layout ^ noarr::set_length<'y'>(ny);
		}
		else if constexpr (dim == 'z')
		{
			auto tile_size = std::min<index_t>(x_tile_size_, x_size_padded);
			auto X_len = (x_size_padded + tile_size - 1) / tile_size;

			auto layout = noarr::scalar<real_t>() ^ noarr::vector<'x'>(tile_size) ^ noarr::vector<'z'>()
						  ^ noarr::vector<'X'>(X_len) ^ noarr::vector<'y'>(ny);

			if constexpr (!with_len)
				return layout;
			else
				return layout ^ noarr::set_length<'z'>(nz);
		}
	}

	template <char dim>
	auto get_non_blocked_scratch_layout(index_t n, index_t s)
	{
		std::size_t ssize = s * sizeof(real_t);
		std::size_t ssize_padded = (ssize + alignment_size_ - 1) / alignment_size_ * alignment_size_;
		ssize_padded /= sizeof(real_t);

		return noarr::scalar<real_t>() ^ noarr::vectors<'v', dim>(ssize_padded, n);
	}

	auto get_thread_distribution_layout() const
	{
		return noarr::vectors<'x', 'y', 'z', 'g'>(cores_division_[0], cores_division_[1], cores_division_[2],
												  substrate_groups_);
	}

	template <char dim, bool fused_z = true>
	auto get_diag_layout(index_t nx, index_t ny, index_t nz, index_t substrates_count, index_t sync_step) const
	{
		using simd_tag = hn::ScalableTag<real_t>;
		simd_tag d;
		std::size_t elements = hn::Lanes(d);

		if constexpr (dim == 'x')
		{
			if constexpr (fused_z)
			{
				sync_step = std::min(sync_step, nz);

				std::size_t Y_size = (ny * sync_step + elements - 1) / elements;

				return noarr::scalar<real_t>()
					   ^ noarr::vectors<'y', 'x', 'Y', 'Z', 's'>(elements, nx, Y_size, (nz + sync_step - 1) / sync_step,
																 substrates_count);
			}
			else
			{
				std::size_t Y_size = (ny + elements - 1) / elements;

				return noarr::scalar<real_t>()
					   ^ noarr::vectors<'y', 'x', 'Y', 'z', 's'>(elements, nx, Y_size, nz, substrates_count);
			}
		}
		else if constexpr (dim == 'y')
		{
			std::size_t x_size = nx * sizeof(real_t);
			std::size_t x_size_padded = (x_size + alignment_size_ - 1) / alignment_size_ * alignment_size_;
			x_size_padded /= sizeof(real_t);

			auto tile_size = std::min<index_t>(x_tile_size_, x_size_padded);
			auto X_len = (x_size_padded + tile_size - 1) / tile_size;

			return noarr::scalar<real_t>()
				   ^ noarr::vectors<'x', 'y', 'X', 'z', 's'>(tile_size, ny, X_len, nz, substrates_count);
		}
		else if constexpr (dim == 'z')
		{
			std::size_t x_size = nx * sizeof(real_t);
			std::size_t x_size_padded = (x_size + alignment_size_ - 1) / alignment_size_ * alignment_size_;
			x_size_padded /= sizeof(real_t);

			auto tile_size = std::min<index_t>(x_tile_size_, x_size_padded);
			auto X_len = (x_size_padded + tile_size - 1) / tile_size;

			return noarr::scalar<real_t>()
				   ^ noarr::vectors<'x', 'z', 'X', 'y', 's'>(tile_size, nz, X_len, ny, substrates_count);
		}
	}

	template <char dim, bool fused_z = true>
	void precompute_values(std::unique_ptr<real_t*[]>& a, std::unique_ptr<real_t*[]>& b, std::unique_ptr<real_t*[]>& c,
						   index_t shape, index_t n, index_t dims);

	void precompute_values(index_t counters_count,
						   std::unique_ptr<std::unique_ptr<aligned_atomic<index_t>>[]>& counters,
						   std::unique_ptr<std::unique_ptr<std::barrier<>>[]>& barriers, index_t group_size, char dim);

	void set_block_bounds(index_t n, index_t group_size, index_t& block_size, std::vector<index_t>& group_block_lengths,
						  std::vector<index_t>& group_block_offsets);

	thread_id_t<index_t> get_thread_id() const;

	index_t get_lane_id(char dim) const
	{
		const auto tid = get_thread_id();

		if (dim == 'x')
			return tid.y * cores_division_[2] + tid.group * cores_division_[1] * cores_division_[2] + tid.z;
		else if (dim == 'y')
			return tid.x * cores_division_[2] + tid.group * cores_division_[0] * cores_division_[2] + tid.z;
		else
			return tid.x * cores_division_[1] + tid.group * cores_division_[0] * cores_division_[1] + tid.y;
	}

	void validate_restrictions();

public:
	sdd_full_blocking();

	template <std::size_t dims = 3>
	auto get_substrates_layout() const
	{
		std::size_t x_size = this->problem_.nx * sizeof(real_t);
		std::size_t x_size_padded = (x_size + alignment_size_ - 1) / alignment_size_ * alignment_size_;
		x_size_padded /= sizeof(real_t);

		if constexpr (dims == 1)
			return noarr::scalar<real_t>() ^ noarr::vectors<'x', 's'>(x_size_padded, this->problem_.substrates_count);
		else if constexpr (dims == 2)
			return noarr::scalar<real_t>()
				   ^ noarr::vectors<'x', 'y', 's'>(x_size_padded, this->problem_.ny, this->problem_.substrates_count);
		else if constexpr (dims == 3)
			return noarr::scalar<real_t>()
				   ^ noarr::vectors<'x', 'y', 'z', 's'>(x_size_padded, this->problem_.ny, this->problem_.nz,
														this->problem_.substrates_count);
	}

	double access(std::size_t s, std::size_t x, std::size_t y, std::size_t z) const override;

	void prepare(const max_problem_t& problem) override;

	void tune(const nlohmann::json& params) override;

	void initialize() override;

	void solve_x() override;
	void solve_y() override;
	void solve_z() override;

	void solve() override;

	void solve_nf();
	void solve_x_nf();

	~sdd_full_blocking();
};
