#include <cuda_macro.hpp>
#include <helperOC/DynSys/DynSys/DynSysSchemeData.hpp>
#include <vector>
#include <deque>
#include <macro.hpp>
#include "DynSysSchemeData_cuda.hpp"
#include <Core/UVec.hpp>
#if !defined(WITH_GPU)
#if defined(USER_DEFINED_GPU_DYNSYS_FUNC)
bool hamFunc_exec_cuda(
	beacls::UVec& hamValue_uvec,
	const std::vector<beacls::UVec>& deriv_uvecs,
	const std::vector<beacls::UVec>& dx_uvecs,
	const std::vector<beacls::UVec>& TIdx_uvecs,
	const FLOAT_TYPE TIderiv,
	const bool TIdim,
	const bool negate
	) {
	if (dx_uvecs.empty()) return false;
	beacls::reallocateAsSrc(hamValue_uvec, dx_uvecs[0]);
	FLOAT_TYPE* hamValue = beacls::UVec_<FLOAT_TYPE>(hamValue_uvec).ptr();

	const size_t num_of_dimensions = deriv_uvecs.size();
	for (size_t dimension = 0; dimension < num_of_dimensions; ++dimension) {
		const FLOAT_TYPE* deriv_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[dimension]).ptr();
		const FLOAT_TYPE* dx_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[dimension]).ptr();
		if (dimension == 0) {
			for (size_t index = 0; index < dx_uvecs[dimension].size(); ++index) {
				hamValue[index] = deriv_ptr[index] * dx_ptr[index];
			}
		}
		else {
			for (size_t index = 0; index < dx_uvecs[dimension].size(); ++index) {
				hamValue[index] += deriv_ptr[index] * dx_ptr[index];
			}
		}
	}

	if (TIdim) {
		for (size_t dimension = 0; dimension < num_of_dimensions; ++dimension) {
			const FLOAT_TYPE* TIdx_ptr = beacls::UVec_<FLOAT_TYPE>(TIdx_uvecs[dimension]).ptr();
			for (size_t index = 0; index < TIdx_uvecs[dimension].size(); ++index) {
				hamValue[index] += TIderiv * TIdx_ptr[index];
			}
		}
	}

	//!< Negate hamValue if backward reachable set
	if (negate) {
		for (size_t index = 0; index < hamValue_uvec.size(); ++index) {
			hamValue[index] = -hamValue[index];
		}
	}
	return true;
}
bool partialFunc_exec_cuda(
	beacls::UVec& alphas_uvec,
	const beacls::UVec& dxLL_dim,
	const beacls::UVec& dxLU_dim,
	const beacls::UVec& dxUL_dim,
	const beacls::UVec& dxUU_dim
	) {
	const size_t length = dxUU_dim.size();
	beacls::reallocateAsSrc(alphas_uvec, dxUU_dim);

	FLOAT_TYPE* alphas = beacls::UVec_<FLOAT_TYPE>(alphas_uvec).ptr();
	const FLOAT_TYPE* dxUU_dim_ptr = beacls::UVec_<FLOAT_TYPE>(dxUU_dim).ptr();
	const FLOAT_TYPE* dxUL_dim_ptr = beacls::UVec_<FLOAT_TYPE>(dxUL_dim).ptr();
	const FLOAT_TYPE* dxLL_dim_ptr = beacls::UVec_<FLOAT_TYPE>(dxLL_dim).ptr();
	const FLOAT_TYPE* dxLU_dim_ptr = beacls::UVec_<FLOAT_TYPE>(dxLU_dim).ptr();

	for (size_t i = 0; i < length; ++i) {
		const FLOAT_TYPE max0 = max_float_type<FLOAT_TYPE>(abs_float_type<FLOAT_TYPE>(dxUU_dim_ptr[i]), abs_float_type<FLOAT_TYPE>(dxUL_dim_ptr[i]));
		const FLOAT_TYPE max1 = max_float_type<FLOAT_TYPE>(abs_float_type<FLOAT_TYPE>(dxLL_dim_ptr[i]), abs_float_type<FLOAT_TYPE>(dxLU_dim_ptr[i]));
		alphas[i] = max_float_type<FLOAT_TYPE>(max0, max1);
	}

	return true;
}
#endif /* defined(USER_DEFINED_GPU_DYNSYS_FUNC) */
#endif /* !defined(WITH_GPU) */
