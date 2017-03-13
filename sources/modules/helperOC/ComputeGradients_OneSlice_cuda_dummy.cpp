#include <cuda_macro.hpp>
#include <vector>
#include <typedef.hpp>
#include <numeric>
#include <Core/UVec.hpp>
#include <Core/CudaStream.hpp>
#include "ComputeGradients_OneSlice_cuda.hpp"

#if !defined(WITH_GPU)
bool copyBackInfNan_cuda(
	beacls::UVec& deriv_c_uvec,
	beacls::UVec& deriv_l_uvec,
	beacls::UVec& deriv_r_uvec,
	const beacls::UVec& original_data_uvec
) {
	FLOAT_TYPE* derivC_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_c_uvec).ptr();
	FLOAT_TYPE* derivL_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_l_uvec).ptr();
	FLOAT_TYPE* derivR_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_r_uvec).ptr();
	const FLOAT_TYPE* original_data_ptr = beacls::UVec_<FLOAT_TYPE>(original_data_uvec).ptr();
	beacls::synchronizeUVec(deriv_c_uvec);
	beacls::synchronizeUVec(deriv_l_uvec);
	beacls::synchronizeUVec(deriv_r_uvec);
	const size_t length = deriv_c_uvec.size();
	for (size_t i = 0; i < length; ++i) {
		const FLOAT_TYPE original_data = original_data_ptr[i];
		if ((original_data == std::numeric_limits<FLOAT_TYPE>::signaling_NaN()) || (original_data == std::numeric_limits<FLOAT_TYPE>::infinity())) {
			derivC_ptr[i] = original_data;
			derivL_ptr[i] = original_data;
			derivR_ptr[i] = original_data;
		}
	}
	return true;
}
#endif /* !defined(WITH_GPU) */
