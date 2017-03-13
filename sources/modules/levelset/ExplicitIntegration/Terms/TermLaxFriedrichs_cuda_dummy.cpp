#include <typedef.hpp>
#include <cuda_macro.hpp>
#include <algorithm>
#include "TermLaxFriedrichs_cuda.hpp"

#if !defined(WITH_GPU)

void TermLaxFriedrichs_execute_cuda(
	beacls::UVec& ydot_uvec,
	const beacls::UVec& diss_uvec,
	const beacls::UVec& ham_uvec
) {
	FLOAT_TYPE* dst_ydot_ptr = beacls::UVec_<FLOAT_TYPE>(ydot_uvec).ptr();
	const FLOAT_TYPE* diss_ptr = beacls::UVec_<FLOAT_TYPE>(diss_uvec).ptr();
	const FLOAT_TYPE* ham_ptr = beacls::UVec_<FLOAT_TYPE>(ham_uvec).ptr();
	beacls::synchronizeUVec(ham_uvec);
	beacls::synchronizeUVec(diss_uvec);
	const size_t length = ham_uvec.size();
	for (size_t index = 0; index < length; ++index) {
		dst_ydot_ptr[index] = diss_ptr[index] - ham_ptr[index];
	}
}
#endif /* !defined(WITH_GPU)  */
