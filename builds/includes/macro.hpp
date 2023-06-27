
#ifndef __macro_hpp__
#define __macro_hpp__

#include <vector>
#include <cstdint>
#include <cmath>
#include <cstddef>
#include <algorithm>
#include <numeric>
#include <limits>
#include <typedef.hpp>

#if defined(WIN32)	// Windows
#if _MSC_VER >= 1900	// Visual Studio 2015 or later
#define THREAD_LOCAL thread_local
#else	// Visual Studio 2013 or  earlier
#define THREAD_LOCAL
#endif	// _MSC_VER >= 1900
#elif defined(__linux) || defined(__linux__)	// Linux
#define THREAD_LOCAL thread_local
#else	// Mac
#define THREAD_LOCAL
#endif

#if 0
template <class T>
static inline const T& HjiMax(const T& a, const T& b)
{
	return a < b ? b : a;
}

template <class T>
static inline const T& HjiMin(const T& a, const T& b)
{
	return b < a ? b : a;
}
#else
#define HjiMax(a,b) std::max<FLOAT_TYPE>(a,b)
#define HjiMin(a,b) std::min<FLOAT_TYPE>(a,b)
#endif

template <class T>
inline T HjiFabs(const T& a)
{
	return a < 0 ? -a : a;
}
template <>
inline double HjiFabs(const double& a)
{
	return fabs(a);
}
template <>
inline float HjiFabs(const float& a)
{
	return fabsf(a);
}
template <>
inline long double HjiFabs(const long double& a)
{
	return fabsl(a);
}

/*
	@brief Initialize arithmetic progression sequence in same precision of Matlab...
	@param	[in]		min	minimum value
	@param	[in]		delta	difference 
	@param	[in]		max	maximum value
	@return	arithmetic progressoin sequence vector
	*/
template<typename T>
inline
std::vector<T> generateArithmeticSequence(
	const T& min,
	const T& delta,
	const T& max
) {
	std::vector<T> seq((size_t)std::round((max - min) / delta) + 1);
	size_t size = seq.size();
	std::iota(seq.begin(), seq.end(), static_cast<T>(0));
	//! Initialize lower half
	std::transform(seq.cbegin(), seq.cbegin() + size / 2, seq.begin(), ([min, delta](const auto& iota_i) {
		return min + iota_i * delta;
	}));
	//! Initialize upper half
	std::transform(seq.cbegin() + size / 2, seq.cend(), seq.begin() + size / 2, ([max, delta, size](const auto& iota_i) {
		return max - (size - iota_i - 1) * delta;
	}));
	return seq;
}
namespace beacls {
	template<typename T, typename Iterator>
	inline
		std::pair<T, T> minmax_value(const Iterator begin, const Iterator end) {
		const std::pair<T, T> initial(std::numeric_limits<T>::max(), -std::numeric_limits<T>::max());
		return std::accumulate(begin, end, initial, [](auto& lhs, const auto& rhs) {
			if (lhs.first > rhs) lhs.first = rhs;
			if (lhs.second < rhs) lhs.second = rhs;
			return lhs;
		});
	}
	template<typename T, typename Iterator>
	inline
		T min_value(const Iterator begin, const Iterator end) {
		const T initial = std::numeric_limits<T>::max();
		return std::accumulate(begin, end, initial, [](auto& lhs, const auto& rhs) {
			if (lhs > rhs) lhs = rhs;
			return lhs;
		});
	}
	template<typename T, typename Iterator>
	inline
		T max_value(const Iterator begin, const Iterator end) {
		const T initial = -std::numeric_limits<T>::max();
		return std::accumulate(begin, end, initial, [](auto& lhs, const auto& rhs) {
			if (lhs < rhs) lhs = rhs;
			return lhs;
		});
	}
	template<typename T, typename Iterator>
	inline
		T min_value_at_index(const Iterator begin, const Iterator end, const size_t index) {
		const T initial = std::numeric_limits<T>::max();
		return std::accumulate(begin, end, initial, [index](auto& lhs, const auto& rhs) {
			const T val = rhs[index];
			if (lhs > val) lhs = val;
			return lhs;
		});
	}
	template<typename T, typename Iterator>
	inline
		T max_value_at_index(const Iterator begin, const Iterator end, const size_t index) {
		const T initial = -std::numeric_limits<T>::max();
		return std::accumulate(begin, end, initial, [index](auto& lhs, const auto& rhs) {
			const T val = rhs[index];
			if (lhs < val) lhs = val;
			return lhs;
		});
	}

};
#endif	/*__macro_hpp__ */
