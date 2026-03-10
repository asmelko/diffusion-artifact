#pragma once

#include <type_traits>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include <hwy/highway.h>
#pragma GCC diagnostic pop

namespace hn = hwy::HWY_NAMESPACE;

template <int First, int Last>
struct static_for
{
	template <typename Fn>
	void operator()(Fn const& fn) const
	{
		if (First < Last)
		{
			fn(std::integral_constant<int, First> {});
			static_for<First + 1, Last>()(fn);
		}
	}
};

template <int N>
struct static_for<N, N>
{
	template <typename Fn>
	void operator()(Fn const&) const
	{}
};

template <int First, int Last>
struct static_rfor
{
	template <typename Fn>
	void operator()(Fn const& fn) const
	{
		if (First >= Last)
		{
			fn(std::integral_constant<int, First> {});
			static_rfor<First - 1, Last>()(fn);
		}
	}
};

template <int N>
struct static_rfor<-1, N>
{
	template <typename Fn>
	void operator()(Fn const&) const
	{}
};

// 16xdouble
template <typename vec_t, std::enable_if_t<std::is_same_v<hn::TFromV<vec_t>, double>, bool> = true>
HWY_INLINE void transpose(vec_t& row0, vec_t& row1, vec_t& row2, vec_t& row3, vec_t& row4, vec_t& row5, vec_t& row6,
						  vec_t& row7, vec_t& row8, vec_t& row9, vec_t& row10, vec_t& row11, vec_t& row12, vec_t& row13,
						  vec_t& row14, vec_t& row15)
{}

// 16xfloat
template <typename vec_t, std::enable_if_t<std::is_same_v<hn::TFromV<vec_t>, float>, bool> = true>
HWY_INLINE void transpose(vec_t& row0, vec_t& row1, vec_t& row2, vec_t& row3, vec_t& row4, vec_t& row5, vec_t& row6,
						  vec_t& row7, vec_t& row8, vec_t& row9, vec_t& row10, vec_t& row11, vec_t& row12, vec_t& row13,
						  vec_t& row14, vec_t& row15)
{
	hn::DFromV<vec_t> d;

	auto t0 = hn::ConcatLowerLower(d, row8, row0);
	auto t1 = hn::ConcatLowerLower(d, row9, row1);
	auto t2 = hn::ConcatLowerLower(d, row10, row2);
	auto t3 = hn::ConcatLowerLower(d, row11, row3);
	auto t4 = hn::ConcatLowerLower(d, row12, row4);
	auto t5 = hn::ConcatLowerLower(d, row13, row5);
	auto t6 = hn::ConcatLowerLower(d, row14, row6);
	auto t7 = hn::ConcatLowerLower(d, row15, row7);
	auto t8 = hn::ConcatUpperUpper(d, row8, row0);
	auto t9 = hn::ConcatUpperUpper(d, row9, row1);
	auto t10 = hn::ConcatUpperUpper(d, row10, row2);
	auto t11 = hn::ConcatUpperUpper(d, row11, row3);
	auto t12 = hn::ConcatUpperUpper(d, row12, row4);
	auto t13 = hn::ConcatUpperUpper(d, row13, row5);
	auto t14 = hn::ConcatUpperUpper(d, row14, row6);
	auto t15 = hn::ConcatUpperUpper(d, row15, row7);

	auto tt0 = hn::InterleaveEvenBlocks(d, t0, t4);
	auto tt1 = hn::InterleaveEvenBlocks(d, t1, t5);
	auto tt2 = hn::InterleaveEvenBlocks(d, t2, t6);
	auto tt3 = hn::InterleaveEvenBlocks(d, t3, t7);
	auto tt4 = hn::InterleaveOddBlocks(d, t0, t4);
	auto tt5 = hn::InterleaveOddBlocks(d, t1, t5);
	auto tt6 = hn::InterleaveOddBlocks(d, t2, t6);
	auto tt7 = hn::InterleaveOddBlocks(d, t3, t7);
	auto tt8 = hn::InterleaveEvenBlocks(d, t8, t12);
	auto tt9 = hn::InterleaveEvenBlocks(d, t9, t13);
	auto tt10 = hn::InterleaveEvenBlocks(d, t10, t14);
	auto tt11 = hn::InterleaveEvenBlocks(d, t11, t15);
	auto tt12 = hn::InterleaveOddBlocks(d, t8, t12);
	auto tt13 = hn::InterleaveOddBlocks(d, t9, t13);
	auto tt14 = hn::InterleaveOddBlocks(d, t10, t14);
	auto tt15 = hn::InterleaveOddBlocks(d, t11, t15);

	auto u0 = hn::InterleaveLower(d, tt0, tt2);
	auto u1 = hn::InterleaveLower(d, tt1, tt3);
	auto u2 = hn::InterleaveUpper(d, tt0, tt2);
	auto u3 = hn::InterleaveUpper(d, tt1, tt3);
	auto u4 = hn::InterleaveLower(d, tt4, tt6);
	auto u5 = hn::InterleaveLower(d, tt5, tt7);
	auto u6 = hn::InterleaveUpper(d, tt4, tt6);
	auto u7 = hn::InterleaveUpper(d, tt5, tt7);
	auto u8 = hn::InterleaveLower(d, tt8, tt10);
	auto u9 = hn::InterleaveLower(d, tt9, tt11);
	auto u10 = hn::InterleaveUpper(d, tt8, tt10);
	auto u11 = hn::InterleaveUpper(d, tt9, tt11);
	auto u12 = hn::InterleaveLower(d, tt12, tt14);
	auto u13 = hn::InterleaveLower(d, tt13, tt15);
	auto u14 = hn::InterleaveUpper(d, tt12, tt14);
	auto u15 = hn::InterleaveUpper(d, tt13, tt15);

	row0 = hn::InterleaveLower(d, u0, u1);
	row1 = hn::InterleaveUpper(d, u0, u1);
	row2 = hn::InterleaveLower(d, u2, u3);
	row3 = hn::InterleaveUpper(d, u2, u3);
	row4 = hn::InterleaveLower(d, u4, u5);
	row5 = hn::InterleaveUpper(d, u4, u5);
	row6 = hn::InterleaveLower(d, u6, u7);
	row7 = hn::InterleaveUpper(d, u6, u7);
	row8 = hn::InterleaveLower(d, u8, u9);
	row9 = hn::InterleaveUpper(d, u8, u9);
	row10 = hn::InterleaveLower(d, u10, u11);
	row11 = hn::InterleaveUpper(d, u10, u11);
	row12 = hn::InterleaveLower(d, u12, u13);
	row13 = hn::InterleaveUpper(d, u12, u13);
	row14 = hn::InterleaveLower(d, u14, u15);
	row15 = hn::InterleaveUpper(d, u14, u15);
}

// 16xfloat
template <typename vec_t, std::enable_if_t<HWY_MAX_LANES_V(vec_t) == 16, bool> = true,
		  std::enable_if_t<std::is_same_v<hn::TFromV<vec_t>, float>, bool> = true>
HWY_INLINE void transpose(vec_t rows[16])
{
	hn::DFromV<vec_t> d;

	auto t0 = hn::ConcatLowerLower(d, rows[8], rows[0]);
	auto t1 = hn::ConcatLowerLower(d, rows[9], rows[1]);
	auto t2 = hn::ConcatLowerLower(d, rows[10], rows[2]);
	auto t3 = hn::ConcatLowerLower(d, rows[11], rows[3]);
	auto t4 = hn::ConcatLowerLower(d, rows[12], rows[4]);
	auto t5 = hn::ConcatLowerLower(d, rows[13], rows[5]);
	auto t6 = hn::ConcatLowerLower(d, rows[14], rows[6]);
	auto t7 = hn::ConcatLowerLower(d, rows[15], rows[7]);
	auto t8 = hn::ConcatUpperUpper(d, rows[8], rows[0]);
	auto t9 = hn::ConcatUpperUpper(d, rows[9], rows[1]);
	auto t10 = hn::ConcatUpperUpper(d, rows[10], rows[2]);
	auto t11 = hn::ConcatUpperUpper(d, rows[11], rows[3]);
	auto t12 = hn::ConcatUpperUpper(d, rows[12], rows[4]);
	auto t13 = hn::ConcatUpperUpper(d, rows[13], rows[5]);
	auto t14 = hn::ConcatUpperUpper(d, rows[14], rows[6]);
	auto t15 = hn::ConcatUpperUpper(d, rows[15], rows[7]);

	auto tt0 = hn::InterleaveEvenBlocks(d, t0, t4);
	auto tt1 = hn::InterleaveEvenBlocks(d, t1, t5);
	auto tt2 = hn::InterleaveEvenBlocks(d, t2, t6);
	auto tt3 = hn::InterleaveEvenBlocks(d, t3, t7);
	auto tt4 = hn::InterleaveOddBlocks(d, t0, t4);
	auto tt5 = hn::InterleaveOddBlocks(d, t1, t5);
	auto tt6 = hn::InterleaveOddBlocks(d, t2, t6);
	auto tt7 = hn::InterleaveOddBlocks(d, t3, t7);
	auto tt8 = hn::InterleaveEvenBlocks(d, t8, t12);
	auto tt9 = hn::InterleaveEvenBlocks(d, t9, t13);
	auto tt10 = hn::InterleaveEvenBlocks(d, t10, t14);
	auto tt11 = hn::InterleaveEvenBlocks(d, t11, t15);
	auto tt12 = hn::InterleaveOddBlocks(d, t8, t12);
	auto tt13 = hn::InterleaveOddBlocks(d, t9, t13);
	auto tt14 = hn::InterleaveOddBlocks(d, t10, t14);
	auto tt15 = hn::InterleaveOddBlocks(d, t11, t15);

	auto u0 = hn::InterleaveLower(d, tt0, tt2);
	auto u1 = hn::InterleaveLower(d, tt1, tt3);
	auto u2 = hn::InterleaveUpper(d, tt0, tt2);
	auto u3 = hn::InterleaveUpper(d, tt1, tt3);
	auto u4 = hn::InterleaveLower(d, tt4, tt6);
	auto u5 = hn::InterleaveLower(d, tt5, tt7);
	auto u6 = hn::InterleaveUpper(d, tt4, tt6);
	auto u7 = hn::InterleaveUpper(d, tt5, tt7);
	auto u8 = hn::InterleaveLower(d, tt8, tt10);
	auto u9 = hn::InterleaveLower(d, tt9, tt11);
	auto u10 = hn::InterleaveUpper(d, tt8, tt10);
	auto u11 = hn::InterleaveUpper(d, tt9, tt11);
	auto u12 = hn::InterleaveLower(d, tt12, tt14);
	auto u13 = hn::InterleaveLower(d, tt13, tt15);
	auto u14 = hn::InterleaveUpper(d, tt12, tt14);
	auto u15 = hn::InterleaveUpper(d, tt13, tt15);

	rows[0] = hn::InterleaveLower(d, u0, u1);
	rows[1] = hn::InterleaveUpper(d, u0, u1);
	rows[2] = hn::InterleaveLower(d, u2, u3);
	rows[3] = hn::InterleaveUpper(d, u2, u3);
	rows[4] = hn::InterleaveLower(d, u4, u5);
	rows[5] = hn::InterleaveUpper(d, u4, u5);
	rows[6] = hn::InterleaveLower(d, u6, u7);
	rows[7] = hn::InterleaveUpper(d, u6, u7);
	rows[8] = hn::InterleaveLower(d, u8, u9);
	rows[9] = hn::InterleaveUpper(d, u8, u9);
	rows[10] = hn::InterleaveLower(d, u10, u11);
	rows[11] = hn::InterleaveUpper(d, u10, u11);
	rows[12] = hn::InterleaveLower(d, u12, u13);
	rows[13] = hn::InterleaveUpper(d, u12, u13);
	rows[14] = hn::InterleaveLower(d, u14, u15);
	rows[15] = hn::InterleaveUpper(d, u14, u15);
}

// 8xdouble
template <typename vec_t, std::enable_if_t<std::is_same_v<hn::TFromV<vec_t>, double>, bool> = true>
HWY_INLINE void transpose(vec_t& row0, vec_t& row1, vec_t& row2, vec_t& row3, vec_t& row4, vec_t& row5, vec_t& row6,
						  vec_t& row7)
{
	hn::DFromV<vec_t> d;

	auto t0 = hn::ConcatLowerLower(d, row4, row0);
	auto t1 = hn::ConcatLowerLower(d, row5, row1);
	auto t2 = hn::ConcatLowerLower(d, row6, row2);
	auto t3 = hn::ConcatLowerLower(d, row7, row3);
	auto t4 = hn::ConcatUpperUpper(d, row4, row0);
	auto t5 = hn::ConcatUpperUpper(d, row5, row1);
	auto t6 = hn::ConcatUpperUpper(d, row6, row2);
	auto t7 = hn::ConcatUpperUpper(d, row7, row3);

	auto u0 = hn::InterleaveEvenBlocks(d, t0, t2);
	auto u1 = hn::InterleaveEvenBlocks(d, t1, t3);
	auto u2 = hn::InterleaveOddBlocks(d, t0, t2);
	auto u3 = hn::InterleaveOddBlocks(d, t1, t3);
	auto u4 = hn::InterleaveEvenBlocks(d, t4, t6);
	auto u5 = hn::InterleaveEvenBlocks(d, t5, t7);
	auto u6 = hn::InterleaveOddBlocks(d, t4, t6);
	auto u7 = hn::InterleaveOddBlocks(d, t5, t7);

	row0 = hn::InterleaveLower(d, u0, u1);
	row1 = hn::InterleaveUpper(d, u0, u1);
	row2 = hn::InterleaveLower(d, u2, u3);
	row3 = hn::InterleaveUpper(d, u2, u3);
	row4 = hn::InterleaveLower(d, u4, u5);
	row5 = hn::InterleaveUpper(d, u4, u5);
	row6 = hn::InterleaveLower(d, u6, u7);
	row7 = hn::InterleaveUpper(d, u6, u7);
}

// 8xdouble
template <typename vec_t, std::enable_if_t<HWY_MAX_LANES_V(vec_t) == 8, bool> = true,
		  std::enable_if_t<std::is_same_v<hn::TFromV<vec_t>, double>, bool> = true>
HWY_INLINE void transpose(vec_t rows[8])
{
	hn::DFromV<vec_t> d;

	auto t0 = hn::ConcatLowerLower(d, rows[4], rows[0]);
	auto t1 = hn::ConcatLowerLower(d, rows[5], rows[1]);
	auto t2 = hn::ConcatLowerLower(d, rows[6], rows[2]);
	auto t3 = hn::ConcatLowerLower(d, rows[7], rows[3]);
	auto t4 = hn::ConcatUpperUpper(d, rows[4], rows[0]);
	auto t5 = hn::ConcatUpperUpper(d, rows[5], rows[1]);
	auto t6 = hn::ConcatUpperUpper(d, rows[6], rows[2]);
	auto t7 = hn::ConcatUpperUpper(d, rows[7], rows[3]);

	auto u0 = hn::InterleaveEvenBlocks(d, t0, t2);
	auto u1 = hn::InterleaveEvenBlocks(d, t1, t3);
	auto u2 = hn::InterleaveOddBlocks(d, t0, t2);
	auto u3 = hn::InterleaveOddBlocks(d, t1, t3);
	auto u4 = hn::InterleaveEvenBlocks(d, t4, t6);
	auto u5 = hn::InterleaveEvenBlocks(d, t5, t7);
	auto u6 = hn::InterleaveOddBlocks(d, t4, t6);
	auto u7 = hn::InterleaveOddBlocks(d, t5, t7);

	rows[0] = hn::InterleaveLower(d, u0, u1);
	rows[1] = hn::InterleaveUpper(d, u0, u1);
	rows[2] = hn::InterleaveLower(d, u2, u3);
	rows[3] = hn::InterleaveUpper(d, u2, u3);
	rows[4] = hn::InterleaveLower(d, u4, u5);
	rows[5] = hn::InterleaveUpper(d, u4, u5);
	rows[6] = hn::InterleaveLower(d, u6, u7);
	rows[7] = hn::InterleaveUpper(d, u6, u7);
}

// 8xfloat
template <typename vec_t, std::enable_if_t<std::is_same_v<hn::TFromV<vec_t>, float>, bool> = true>
HWY_INLINE void transpose(vec_t& row0, vec_t& row1, vec_t& row2, vec_t& row3, vec_t& row4, vec_t& row5, vec_t& row6,
						  vec_t& row7)
{
	hn::DFromV<vec_t> d;

	auto t0 = hn::InterleaveEvenBlocks(d, row0, row4);
	auto t1 = hn::InterleaveEvenBlocks(d, row1, row5);
	auto t2 = hn::InterleaveEvenBlocks(d, row2, row6);
	auto t3 = hn::InterleaveEvenBlocks(d, row3, row7);
	auto t4 = hn::InterleaveOddBlocks(d, row0, row4);
	auto t5 = hn::InterleaveOddBlocks(d, row1, row5);
	auto t6 = hn::InterleaveOddBlocks(d, row2, row6);
	auto t7 = hn::InterleaveOddBlocks(d, row3, row7);

	auto u0 = hn::InterleaveLower(d, t0, t2);
	auto u1 = hn::InterleaveLower(d, t1, t3);
	auto u2 = hn::InterleaveUpper(d, t0, t2);
	auto u3 = hn::InterleaveUpper(d, t1, t3);
	auto u4 = hn::InterleaveLower(d, t4, t6);
	auto u5 = hn::InterleaveLower(d, t5, t7);
	auto u6 = hn::InterleaveUpper(d, t4, t6);
	auto u7 = hn::InterleaveUpper(d, t5, t7);

	row0 = hn::InterleaveLower(d, u0, u1);
	row1 = hn::InterleaveUpper(d, u0, u1);
	row2 = hn::InterleaveLower(d, u2, u3);
	row3 = hn::InterleaveUpper(d, u2, u3);
	row4 = hn::InterleaveLower(d, u4, u5);
	row5 = hn::InterleaveUpper(d, u4, u5);
	row6 = hn::InterleaveLower(d, u6, u7);
	row7 = hn::InterleaveUpper(d, u6, u7);
}

// 8xfloat
template <typename vec_t, std::enable_if_t<HWY_MAX_LANES_V(vec_t) == 8, bool> = true,
		  std::enable_if_t<std::is_same_v<hn::TFromV<vec_t>, float>, bool> = true>
HWY_INLINE void transpose(vec_t rows[8])
{
	hn::DFromV<vec_t> d;

	auto t0 = hn::InterleaveEvenBlocks(d, rows[0], rows[4]);
	auto t1 = hn::InterleaveEvenBlocks(d, rows[1], rows[5]);
	auto t2 = hn::InterleaveEvenBlocks(d, rows[2], rows[6]);
	auto t3 = hn::InterleaveEvenBlocks(d, rows[3], rows[7]);
	auto t4 = hn::InterleaveOddBlocks(d, rows[0], rows[4]);
	auto t5 = hn::InterleaveOddBlocks(d, rows[1], rows[5]);
	auto t6 = hn::InterleaveOddBlocks(d, rows[2], rows[6]);
	auto t7 = hn::InterleaveOddBlocks(d, rows[3], rows[7]);

	auto u0 = hn::InterleaveLower(d, t0, t2);
	auto u1 = hn::InterleaveLower(d, t1, t3);
	auto u2 = hn::InterleaveUpper(d, t0, t2);
	auto u3 = hn::InterleaveUpper(d, t1, t3);
	auto u4 = hn::InterleaveLower(d, t4, t6);
	auto u5 = hn::InterleaveLower(d, t5, t7);
	auto u6 = hn::InterleaveUpper(d, t4, t6);
	auto u7 = hn::InterleaveUpper(d, t5, t7);

	rows[0] = hn::InterleaveLower(d, u0, u1);
	rows[1] = hn::InterleaveUpper(d, u0, u1);
	rows[2] = hn::InterleaveLower(d, u2, u3);
	rows[3] = hn::InterleaveUpper(d, u2, u3);
	rows[4] = hn::InterleaveLower(d, u4, u5);
	rows[5] = hn::InterleaveUpper(d, u4, u5);
	rows[6] = hn::InterleaveLower(d, u6, u7);
	rows[7] = hn::InterleaveUpper(d, u6, u7);

	// TODO - overlap transpose with loads
	// __m256 t0 = _mm256_unpacklo_ps(rows[0].raw, rows[1].raw);
	// __m256 t1 = _mm256_unpackhi_ps(rows[0].raw, rows[1].raw);
	// __m256 t2 = _mm256_unpacklo_ps(rows[2].raw, rows[3].raw);
	// __m256 t3 = _mm256_unpackhi_ps(rows[2].raw, rows[3].raw);
	// __m256 t4 = _mm256_unpacklo_ps(rows[4].raw, rows[5].raw);
	// __m256 t5 = _mm256_unpackhi_ps(rows[4].raw, rows[5].raw);
	// __m256 t6 = _mm256_unpacklo_ps(rows[6].raw, rows[7].raw);
	// __m256 t7 = _mm256_unpackhi_ps(rows[6].raw, rows[7].raw);

	// __m256 tt0 = _mm256_shuffle_ps(t0, t2, 0x44);
	// __m256 tt1 = _mm256_shuffle_ps(t0, t2, 0xEE);
	// __m256 tt2 = _mm256_shuffle_ps(t1, t3, 0x44);
	// __m256 tt3 = _mm256_shuffle_ps(t1, t3, 0xEE);
	// __m256 tt4 = _mm256_shuffle_ps(t4, t6, 0x44);
	// __m256 tt5 = _mm256_shuffle_ps(t4, t6, 0xEE);
	// __m256 tt6 = _mm256_shuffle_ps(t5, t7, 0x44);
	// __m256 tt7 = _mm256_shuffle_ps(t5, t7, 0xEE);

	// rows[0].raw = _mm256_permute2f128_ps(tt0, tt4, 0x20);
	// rows[1].raw = _mm256_permute2f128_ps(tt1, tt5, 0x20);
	// rows[2].raw = _mm256_permute2f128_ps(tt2, tt6, 0x20);
	// rows[3].raw = _mm256_permute2f128_ps(tt3, tt7, 0x20);
	// rows[4].raw = _mm256_permute2f128_ps(tt0, tt4, 0x31);
	// rows[5].raw = _mm256_permute2f128_ps(tt1, tt5, 0x31);
	// rows[6].raw = _mm256_permute2f128_ps(tt2, tt6, 0x31);
	// rows[7].raw = _mm256_permute2f128_ps(tt3, tt7, 0x31);

	//  __m256 t0 = _mm256_permute2f128_ps(rows[0].raw, rows[4].raw, 0x20);
	//  __m256 t1 = _mm256_permute2f128_ps(rows[1].raw, rows[5].raw, 0x20);
	//  __m256 t2 = _mm256_permute2f128_ps(rows[2].raw, rows[6].raw, 0x20);
	//  __m256 t3 = _mm256_permute2f128_ps(rows[3].raw, rows[7].raw, 0x20);
	//  __m256 t4 = _mm256_permute2f128_ps(rows[0].raw, rows[4].raw, 0x31);
	//  __m256 t5 = _mm256_permute2f128_ps(rows[1].raw, rows[5].raw, 0x31);
	//  __m256 t6 = _mm256_permute2f128_ps(rows[2].raw, rows[6].raw, 0x31);
	//  __m256 t7 = _mm256_permute2f128_ps(rows[3].raw, rows[7].raw, 0x31);

	//  __m256 tt0 = _mm256_unpacklo_ps(t0, t2);
	//  __m256 tt1 = _mm256_unpacklo_ps(t1, t3);
	//  __m256 tt2 = _mm256_unpackhi_ps(t0, t2);
	//  __m256 tt3 = _mm256_unpackhi_ps(t1, t3);
	//  __m256 tt4 = _mm256_unpacklo_ps(t4, t6);
	//  __m256 tt5 = _mm256_unpacklo_ps(t5, t7);
	//  __m256 tt6 = _mm256_unpackhi_ps(t4, t6);
	//  __m256 tt7 = _mm256_unpackhi_ps(t5, t7);

	//  rows[0].raw = _mm256_unpacklo_ps(tt0, tt1);
	//  rows[1].raw = _mm256_unpackhi_ps(tt0, tt1);
	//  rows[2].raw = _mm256_unpacklo_ps(tt2, tt3);
	//  rows[3].raw = _mm256_unpackhi_ps(tt2, tt3);
	//  rows[4].raw = _mm256_unpacklo_ps(tt4, tt5);
	//  rows[5].raw = _mm256_unpackhi_ps(tt4, tt5);
	//  rows[6].raw = _mm256_unpacklo_ps(tt6, tt7);
	//  rows[7].raw = _mm256_unpackhi_ps(tt6, tt7);

	// hn::DFromV<vec_t> d;

	// auto t0 = hn::InterleaveLower(d, rows[0], rows[2]);
	// auto t1 = hn::InterleaveUpper(d, rows[0], rows[2]);
	// auto t2 = hn::InterleaveLower(d, rows[1], rows[3]);
	// auto t3 = hn::InterleaveUpper(d, rows[1], rows[3]);
	// auto t4 = hn::InterleaveLower(d, rows[4], rows[6]);
	// auto t5 = hn::InterleaveUpper(d, rows[4], rows[6]);
	// auto t6 = hn::InterleaveLower(d, rows[5], rows[7]);
	// auto t7 = hn::InterleaveUpper(d, rows[5], rows[7]);


	// auto u0 = hn::InterleaveLower(d, t0, t2);
	// auto u1 = hn::InterleaveUpper(d, t0, t2);
	// auto u2 = hn::InterleaveLower(d, t1, t3);
	// auto u3 = hn::InterleaveUpper(d, t1, t3);
	// auto u4 = hn::InterleaveLower(d, t4, t6);
	// auto u5 = hn::InterleaveUpper(d, t4, t6);
	// auto u6 = hn::InterleaveLower(d, t5, t7);
	// auto u7 = hn::InterleaveUpper(d, t5, t7);

	// rows[0] = hn::InterleaveEvenBlocks(d, u0, u4);
	// rows[1] = hn::InterleaveEvenBlocks(d, u1, u5);
	// rows[2] = hn::InterleaveEvenBlocks(d, u2, u6);
	// rows[3] = hn::InterleaveEvenBlocks(d, u3, u7);
	// rows[4] = hn::InterleaveOddBlocks(d, u0, u4);
	// rows[5] = hn::InterleaveOddBlocks(d, u1, u5);
	// rows[6] = hn::InterleaveOddBlocks(d, u2, u6);
	// rows[7] = hn::InterleaveOddBlocks(d, u3, u7);
}

// 4xfloat
template <typename vec_t, std::enable_if_t<std::is_same_v<hn::TFromV<vec_t>, float>, bool> = true>
HWY_INLINE void transpose(vec_t& row0, vec_t& row1, vec_t& row2, vec_t& row3)
{
	hn::DFromV<vec_t> d;

	auto u0 = hn::InterleaveLower(d, row0, row2);
	auto u1 = hn::InterleaveLower(d, row1, row3);
	auto u2 = hn::InterleaveUpper(d, row0, row2);
	auto u3 = hn::InterleaveUpper(d, row1, row3);

	row0 = hn::InterleaveLower(d, u0, u1);
	row1 = hn::InterleaveUpper(d, u0, u1);
	row2 = hn::InterleaveLower(d, u2, u3);
	row3 = hn::InterleaveUpper(d, u2, u3);
}

// 4xfloat
template <typename vec_t, std::enable_if_t<HWY_MAX_LANES_V(vec_t) == 4, bool> = true,
		  std::enable_if_t<std::is_same_v<hn::TFromV<vec_t>, float>, bool> = true>
HWY_INLINE void transpose(vec_t rows[4])
{
	hn::DFromV<vec_t> d;

	auto u0 = hn::InterleaveLower(d, rows[0], rows[2]);
	auto u1 = hn::InterleaveLower(d, rows[1], rows[3]);
	auto u2 = hn::InterleaveUpper(d, rows[0], rows[2]);
	auto u3 = hn::InterleaveUpper(d, rows[1], rows[3]);

	rows[0] = hn::InterleaveLower(d, u0, u1);
	rows[1] = hn::InterleaveUpper(d, u0, u1);
	rows[2] = hn::InterleaveLower(d, u2, u3);
	rows[3] = hn::InterleaveUpper(d, u2, u3);
}

// 4xdouble
template <typename vec_t, std::enable_if_t<std::is_same_v<hn::TFromV<vec_t>, double>, bool> = true>
HWY_INLINE void transpose(vec_t& row0, vec_t& row1, vec_t& row2, vec_t& row3)
{
	hn::DFromV<vec_t> d;

	auto t0 = hn::InterleaveEvenBlocks(d, row0, row2);
	auto t1 = hn::InterleaveEvenBlocks(d, row1, row3);
	auto t2 = hn::InterleaveOddBlocks(d, row0, row2);
	auto t3 = hn::InterleaveOddBlocks(d, row1, row3);

	row0 = hn::InterleaveLower(d, t0, t1);
	row1 = hn::InterleaveUpper(d, t0, t1);
	row2 = hn::InterleaveLower(d, t2, t3);
	row3 = hn::InterleaveUpper(d, t2, t3);
}

// 4xdouble
template <typename vec_t, std::enable_if_t<HWY_MAX_LANES_V(vec_t) == 4, bool> = true,
		  std::enable_if_t<std::is_same_v<hn::TFromV<vec_t>, double>, bool> = true>
HWY_INLINE void transpose(vec_t rows[4])
{
	hn::DFromV<vec_t> d;

	auto t0 = hn::InterleaveEvenBlocks(d, rows[0], rows[2]);
	auto t1 = hn::InterleaveEvenBlocks(d, rows[1], rows[3]);
	auto t2 = hn::InterleaveOddBlocks(d, rows[0], rows[2]);
	auto t3 = hn::InterleaveOddBlocks(d, rows[1], rows[3]);

	rows[0] = hn::InterleaveLower(d, t0, t1);
	rows[1] = hn::InterleaveUpper(d, t0, t1);
	rows[2] = hn::InterleaveLower(d, t2, t3);
	rows[3] = hn::InterleaveUpper(d, t2, t3);
}

// 2xfloat & 2xdouble
template <typename vec_t>
HWY_INLINE void transpose(vec_t& row0, vec_t& row1)
{
	hn::DFromV<vec_t> d;

	auto t0 = hn::InterleaveLower(d, row0, row1);
	auto t1 = hn::InterleaveUpper(d, row0, row1);

	row0 = t0;
	row1 = t1;
}

// 2xfloat & 2xdouble
template <typename vec_t, std::enable_if_t<HWY_MAX_LANES_V(vec_t) == 2, bool> = true>
HWY_INLINE void transpose(vec_t rows[2])
{
	hn::DFromV<vec_t> d;

	auto t0 = hn::InterleaveLower(d, rows[0], rows[1]);
	auto t1 = hn::InterleaveUpper(d, rows[0], rows[1]);

	rows[0] = t0;
	rows[1] = t1;
}

// 1xfloat & 1xdouble
template <typename vec_t, std::enable_if_t<HWY_MAX_LANES_V(vec_t) == 1, bool> = true>
HWY_INLINE void transpose(vec_t[1])
{}
