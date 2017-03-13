#ifndef __DynSysSchemeData_cuda_hpp__
#define __DynSysSchemeData_cuda_hpp__

#include <vector>
#include <cuda_macro.hpp>
#include <Core/UVec.hpp>
#if defined(USER_DEFINED_GPU_DYNSYS_FUNC)
bool hamFunc_exec_cuda(
	beacls::UVec& hamValue_uvec,
	const std::vector<beacls::UVec>& deriv_uvecs,
	const std::vector<beacls::UVec>& dx_uvecs,
	const std::vector<beacls::UVec>& TIdx_uvecs,
	const FLOAT_TYPE TIderiv,
	const bool TIdim,
	const bool negate
);
bool partialFunc_exec_cuda(
	beacls::UVec& alphas_uvec,
	const beacls::UVec& dxLL_dim,
	const beacls::UVec& dxLU_dim,
	const beacls::UVec& dxUL_dim,
	const beacls::UVec& dxUU_dim
	);
#endif /* defined(USER_DEFINED_GPU_DYNSYS_FUNC) */
#endif /* __DynSysSchemeData_cuda_hpp__ */
