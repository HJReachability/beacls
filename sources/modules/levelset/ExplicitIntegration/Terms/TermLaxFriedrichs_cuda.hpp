#ifndef __TermLaxFriedrichs_cuda_hpp__
#define __TermLaxFriedrichs_cuda_hpp__
#include <typedef.hpp>
#include <cuda_macro.hpp>
#include <Core/UVec.hpp>
void TermLaxFriedrichs_execute_cuda
(
	beacls::UVec& ydot_uvec,
	const beacls::UVec& diss_uvec,
	const beacls::UVec& ham_uvec
);
#endif	/* __TermLaxFriedrichs_cuda_hpp__ */
