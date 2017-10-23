#ifndef __ArtificialDissipationGLF_cuda_hpp__
#define __ArtificialDissipationGLF_cuda_hpp__
#include <typedef.hpp>
#include <Core/UVec.hpp>
#include <cuda_macro.hpp>

template<typename T>
inline
__device__
void dev_minmax(T& dst_min, T& dst_max, const T& lhs, const T& rhs) {
	if (lhs < rhs) {
		dst_min = lhs;
		dst_max = rhs;
	}
	else {
		dst_min = rhs;
		dst_max = lhs;
	}
}
template<typename T>
inline
__device__
void dev_min(T& dst_min, const T& lhs) {
	if (lhs < dst_min) {
		dst_min = lhs;
	}
}
template<typename T>
inline
__device__
void dev_max(T& dst_max, const T& lhs) {
	if (lhs > dst_max) {
		dst_max = lhs;
	}
}

void ArtificialDissipationGLF_execute_cuda
(
	beacls::UVec& diss_uvec,
	const beacls::UVec& alpha_uvec,
	const beacls::UVec& deriv_l_uvec,
	const beacls::UVec& deriv_r_uvec,
	beacls::UVec &tmp_min_cuda_uvec,
	beacls::UVec &tmp_max_cuda_uvec,
	beacls::UVec &tmp_max_alpha_cuda_uvec,
	const size_t dimension,
	const size_t loop_size,
	const bool updateDerivMinMax
);
void ArtificialDissipationGLF_reduce_cuda
(
	FLOAT_TYPE& step_bound_inv,
	FLOAT_TYPE& derivMin,
	FLOAT_TYPE& derivMax,
	const beacls::UVec& alpha_uvec,
	const beacls::UVec &tmp_min_cuda_uvec,
	const beacls::UVec &tmp_max_cuda_uvec,
	const beacls::UVec &tmp_max_alpha_cuda_uvec,
	beacls::UVec &tmp_min_cpu_uvecs,
	beacls::UVec &tmp_max_cpu_uvecs,
	beacls::UVec &tmp_max_alpha_cpu_uvecs,
	const FLOAT_TYPE dxInv,
	const bool updateDerivMinMax
);
#endif	/* __ArtificialDissipationGLF_cuda_hpp__ */
