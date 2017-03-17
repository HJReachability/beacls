#ifndef __cuda_macro_hpp__
#define __cuda_macro_hpp__

#include <cmath>
#include <vector>
#include <deque>
#include <algorithm>
#include <typedef.hpp>
#if defined(WITH_GPU)
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#else	/* defined(WITH_GPU) */
#define __device__
#define __global__
#define __host__ 
#endif	/* !defined(WITH_GPU) */

namespace beacls {
	class UseTupleArg
	{
		UseTupleArg(){}
		~UseTupleArg(){}
	};
};
#if !defined(__CUDACC__)
template<typename T>
static inline
T max_float_type(const T& a, const T& b) {
	return std::max<T>(a, b);
}
template<typename T>
static inline
T min_float_type(const T& a, const T& b) {
	return std::min<T>(a, b);
}
template<typename T>
static inline
T abs_float_type(const T& a) {
	return std::abs(a);
}
template<typename T>
static inline
void sincos_float_type(const T& a, T& s, T& c) {
	s = std::sin(a);
	c = std::cos(a);
}
template<typename T>
static inline
T sin_float_type(const T& a) {
	return std::sin(a);
}
template<typename T>
static inline
T cos_float_type(const T& a) {
	return std::cos(a);
}
template<typename T>
static inline
T sqrt_float_type(const T& a) {
	return std::sqrt(a);
}
template<typename Tuple, typename S, size_t N>
class Get_Var{
	Get_Var(
		const Tuple&,
		S);
public:
	FLOAT_TYPE operator()() const;
};
template<typename Tuple, size_t N>
class Get_Var<Tuple, const FLOAT_TYPE, N>{
  const FLOAT_TYPE v;
public:
	Get_Var(
		const Tuple&,
		const FLOAT_TYPE v) :
		v(v) {}
	FLOAT_TYPE operator()() const {
		return v;
	}
};
template<typename Tuple, size_t N>
class Get_Var<Tuple, const FLOAT_TYPE*, N>{
const FLOAT_TYPE* v;
public:
	Get_Var(
		const Tuple&,
		const FLOAT_TYPE* v) :
		v(v) {}
	FLOAT_TYPE operator()() const {
		return v[0];
	}
};
#else /* defined(__CUDACC__) */
template<typename T>
__host__ __device__
static inline
T max_float_type(const T& a, const T& b);
template<>
__host__ __device__
float max_float_type(const float& a, const float& b) {
	return fmaxf(a, b);
}
template<>
__host__ __device__
double max_float_type(const double& a, const double& b) {
	return fmax(a, b);
}

template<typename T>
__host__ __device__
static inline
T min_float_type(const T& a, const T& b);
template<>
__host__ __device__
float min_float_type(const float& a, const float& b) {
	return fminf(a, b);
}
template<>
__host__ __device__
double min_float_type(const double& a, const double& b) {
	return fmin(a, b);
}

template<typename T>
__host__ __device__
static inline
T abs_float_type(const T& a);
template<>
__host__ __device__
float abs_float_type(const float& a) {
	return fabsf(a);
}
template<>
__host__ __device__
double abs_float_type(const double& a) {
	return fabs(a);
}

template<typename T>
__host__ __device__
static inline
void sincos_float_type(const T& a, T& s, T& c);
template<>
__host__ __device__
void sincos_float_type(const float& a, float& s, float& c) {
	return sincosf(a, &s, &c);
}
template<>
__host__ __device__
void sincos_float_type(const double& a, double& s, double& c) {
	return sincos(a, &s, &c);
}

template<typename T>
__host__ __device__
static inline
T sin_float_type(const T& a);
template<>
__host__ __device__
float sin_float_type(const float& a) {
	return sinf(a);
};
template<>
__host__ __device__
double sin_float_type(const double& a) {
	return sin(a);
}

template<typename T>
__host__ __device__
static inline
T cos_float_type(const T& a);
template<>
__host__ __device__
float cos_float_type(const float& a) {
	return cosf(a);
}
template<>
__host__ __device__
double cos_float_type(const double& a) {
	return cos(a);
}

template<typename T>
static inline
T sqrt_float_type(const T& a);
template<>
__host__ __device__
float sqrt_float_type(const float& a) {
	return sqrtf(a);
}
template<>
__host__ __device__
double sqrt_float_type(const double& a) {
	return sqrt(a);
}

template<typename Tuple, typename S, size_t N>
class Get_Var{
	Get_Var(
		const Tuple&,
		S);
public:
	FLOAT_TYPE operator()() const;
};
template<typename Tuple, size_t N>
class Get_Var<Tuple, const FLOAT_TYPE, N>{
	const FLOAT_TYPE v;
public:
	__host__ __device__
	Get_Var(
		const Tuple&,
		const FLOAT_TYPE v) :
		v(v) {}
	__host__ __device__
	FLOAT_TYPE operator()() const {
		return v;
	}
};
template<typename Tuple, size_t N>
class Get_Var<Tuple, thrust::device_ptr<const FLOAT_TYPE>, N >{
	thrust::device_ptr<const FLOAT_TYPE> v;
public:
	__host__ __device__
	Get_Var(
		const Tuple&,
		thrust::device_ptr<const FLOAT_TYPE> v) :
		v(v) {}
	__host__ __device__
	FLOAT_TYPE operator()() const {
		return v[0];
	}
};
template<typename Tuple, size_t N>
class Get_Var<Tuple, beacls::UseTupleArg*, N>{
	Tuple t;
public:
	__host__ __device__
	Get_Var(
		const Tuple& t, 
		beacls::UseTupleArg*) : 
		t(t) {}
	__host__ __device__
	FLOAT_TYPE operator()() const {
		return thrust::get<N>(t);
	}
};
namespace beacls {
	typedef enum ArgumentType {
		Argument_Scalar_Cpu,
		Argument_Vector_Cpu,
		Argument_Scalar_Cuda,
		Argument_Vector_Cuda,
	}ArgumentType;
	static inline
	ArgumentType get_ArgumentType(
		const size_t dimension,
		const beacls::IntegerVec& sizes,
		const std::deque<bool>& is_cudas,
		const size_t length
	) {
		if (sizes[dimension] == length) {
			if (is_cudas[dimension]){
				return beacls::Argument_Vector_Cuda;
			} else {
				return beacls::Argument_Vector_Cpu;
			}
		} else {
			if (is_cudas[dimension]){
				return beacls::Argument_Scalar_Cuda;
			} else {
				return beacls::Argument_Scalar_Cpu;
			}
		}
	}
};
#endif	/* defined(__CUDACC__) */

template<typename T>
static __device__ inline
T get_actual_length(
	const T total_length,
	const T thread_length,
	const T loop_index_base
) {
	if (total_length >= (thread_length + loop_index_base))
		return thread_length;
	else if (total_length >= loop_index_base)
		return total_length - loop_index_base;
	else
		return 0;
}

template<typename T>
static inline
void get_cuda_thread_size(
	T& thread_length_z,
	T& thread_length_y,
	T& thread_length_x,
	T& num_of_threads_z,
	T& num_of_threads_y,
	T& num_of_threads_x,
	T& num_of_blocks_y,
	T& num_of_blocks_x,
	const T total_length_z,
	const T total_length_y,
	const T total_length_x,
	const T minimum_thread_length_z,
	const T minimum_thread_length_y,
	const T minimum_thread_length_x,
	const T maximum_num_of_threads
) {
	const T maximum_num_of_blocks_y = 65535;
	const T maximum_num_of_blocks_x = 65535;
	const T minimum_num_of_threads_x = 1;
	const T minimum_num_of_threads_y = 1;
	const T maximum_num_of_threads_x = 1024;
	const T maximum_num_of_threads_y = 1024;
	const T maximum_num_of_threads_z = 64;
	const T num_of_threads_in_warp_z = 4;
	const T num_of_threads_in_warp_x = 4;
	const T num_of_threads_in_warp_y = 2;
	const T num_of_blocks_z = 1;
	const T available_num_of_threads_z = std::min(maximum_num_of_threads_z, std::max((T)1, (T)std::floor((double)maximum_num_of_threads / minimum_num_of_threads_x / minimum_num_of_threads_y)));
	const T unlimited_num_of_threads_z = (T)std::ceil((double)total_length_z / minimum_thread_length_z / num_of_threads_in_warp_z)*num_of_threads_in_warp_z;
	if (unlimited_num_of_threads_z < available_num_of_threads_z) {
		num_of_threads_z = unlimited_num_of_threads_z;
	}
	else {
		num_of_threads_z = available_num_of_threads_z;
	}
	thread_length_z = (T)std::ceil((double)total_length_z / num_of_threads_z / num_of_blocks_z);

	const T available_num_of_threads_x = std::min(maximum_num_of_threads_x, std::max((T)1, (T)std::floor((double)maximum_num_of_threads / num_of_threads_z / minimum_num_of_threads_y)));
	const T unlimited_num_of_threads_x = (T)std::ceil((double)total_length_x / minimum_thread_length_x / num_of_threads_in_warp_x)*num_of_threads_in_warp_x;
	if (unlimited_num_of_threads_x < available_num_of_threads_x) {
		num_of_threads_x = unlimited_num_of_threads_x;
		num_of_blocks_x = 1;
	}
	else if (unlimited_num_of_threads_x < available_num_of_threads_x * maximum_num_of_blocks_x) {
		num_of_threads_x = std::max((T)1, (T)std::floor((double)available_num_of_threads_x / num_of_threads_in_warp_x)*num_of_threads_in_warp_x);
		num_of_blocks_x = (T)std::ceil((double)(T)std::ceil((double)total_length_x / minimum_thread_length_x) / num_of_threads_x);
	}
	else {
		num_of_threads_x = (T)std::ceil((double)available_num_of_threads_x / num_of_threads_in_warp_x)*num_of_threads_in_warp_x;
		num_of_blocks_x = maximum_num_of_blocks_x;
	}
	thread_length_x = (T)std::ceil((double)total_length_x / num_of_threads_x / num_of_blocks_x);

	const T available_num_of_threads_y = std::min(maximum_num_of_threads_y, std::max((T)1, (T)std::floor((double)maximum_num_of_threads / num_of_threads_z / num_of_threads_x)));
	const T unlimited_num_of_threads_y = (T)std::ceil((double)total_length_y / minimum_thread_length_y / num_of_threads_in_warp_y)*num_of_threads_in_warp_y;
	if (unlimited_num_of_threads_y < available_num_of_threads_y) {
		num_of_threads_y = unlimited_num_of_threads_y;
		num_of_blocks_y = 1;
	}
	else if (unlimited_num_of_threads_y < available_num_of_threads_y * maximum_num_of_blocks_y) {
		num_of_threads_y = available_num_of_threads_y;
		num_of_blocks_y = (T)std::ceil((double)(T)std::ceil((double)total_length_y / minimum_thread_length_y) / num_of_threads_y);
	}
	else {
		num_of_threads_y = available_num_of_threads_y;
		num_of_blocks_y = maximum_num_of_blocks_x;
	}
	thread_length_y = (T)std::ceil((double)total_length_y / num_of_threads_y / num_of_blocks_y);
}


#endif	/*__cuda_macro_hpp__ */
