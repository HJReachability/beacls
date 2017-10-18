#ifndef __ArtificialDissipationGLF_cuda_hpp__
#define __ArtificialDissipationGLF_cuda_hpp__
#include <typedef.hpp>
#include <Core/UVec.hpp>
void CalculateRangeOfGradient_execute_cuda
(
	const beacls::UVec& deriv_l_uvec,
	const beacls::UVec& deriv_r_uvec,
	beacls::UVec &tmp_min_uvec,
	beacls::UVec &tmp_max_uvec
);
void CalculateRangeOfGradient_reduce_cuda
(
	FLOAT_TYPE& derivMin,
	FLOAT_TYPE& derivMax,
	const beacls::UVec& deriv_l_uvec,
	const beacls::UVec& deriv_r_uvec,
	const beacls::UVec &tmp_min_uvec,
	const beacls::UVec &tmp_max_uvec
);
void ArtificialDissipationGLF_execute_cuda
(
	beacls::UVec& diss_uvec,
	const beacls::UVec& alpha_uvec,
	const beacls::UVec& deriv_l_uvec,
	const beacls::UVec& deriv_r_uvec,
	beacls::UVec &tmp_min_uvec,
	beacls::UVec &tmp_max_uvec,
	beacls::UVec &tmp_alpha_uvec,
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
	const beacls::UVec &tmp_max_cuda_alpha_uvec,
	beacls::UVec &tmp_min_cpu_uvecs,
	beacls::UVec &tmp_max_cpu_uvecs,
	beacls::UVec &tmp_max_alpha_cpu_uvecs,
	const FLOAT_TYPE dxInv,
	const bool updateDerivMinMax
);
#endif	/* __ArtificialDissipationGLF_cuda_hpp__ */
