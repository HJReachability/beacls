#ifndef __ArtificialDissipationGLF_cuda_hpp__
#define __ArtificialDissipationGLF_cuda_hpp__
#include <typedef.hpp>
#include <Core/UVec.hpp>
void CalculateRangeOfGradient_execute_cuda
(
	FLOAT_TYPE& derivMin,
	FLOAT_TYPE& derivMax,
	const beacls::UVec& deriv_l_uvec,
	const beacls::UVec& deriv_r_uvec
);
void ArtificialDissipationGLF_execute_cuda
(
	beacls::UVec& diss_uvec,
	FLOAT_TYPE& step_bound_inv,
	const beacls::UVec& alpha_uvec,
	const beacls::UVec& deriv_l_uvec,
	const beacls::UVec& deriv_r_uvec,
	const FLOAT_TYPE dxInv,
	const size_t dimension,
	const size_t loop_size
);
#endif	/* __ArtificialDissipationGLF_cuda_hpp__ */
