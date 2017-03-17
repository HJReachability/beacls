#include <typedef.hpp>
#include <cuda_macro.hpp>
#include <cmath>
#include <numeric>
#include <functional>
#include <utility>
#include "ArtificialDissipationGLF_cuda.hpp"
#include <macro.hpp>
#if !defined(WITH_GPU) 

void CalculateRangeOfGradient_execute_cuda (
	FLOAT_TYPE& derivMin,
	FLOAT_TYPE& derivMax,
	const beacls::UVec& deriv_l_uvec,
	const beacls::UVec& deriv_r_uvec
) {
	const FLOAT_TYPE* deriv_l_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_l_uvec).ptr();
	const FLOAT_TYPE* deriv_r_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_r_uvec).ptr();
	beacls::synchronizeUVec(deriv_l_uvec);
	beacls::synchronizeUVec(deriv_r_uvec);
	const size_t length = deriv_l_uvec.size();
	const auto minMax_l = beacls::minmax_value<FLOAT_TYPE>(deriv_l_ptr, deriv_l_ptr + length);
	const auto minMax_r = beacls::minmax_value<FLOAT_TYPE>(deriv_r_ptr, deriv_r_ptr + length);
	derivMin = min_float_type<FLOAT_TYPE>(minMax_l.first, minMax_r.first);
	derivMax = max_float_type<FLOAT_TYPE>(minMax_l.second, minMax_r.second);
}

void ArtificialDissipationGLF_execute_cuda(
	beacls::UVec& diss_uvec,
	FLOAT_TYPE& step_bound_inv,
	const beacls::UVec& alpha_uvec,
	const beacls::UVec& deriv_l_uvec,
	const beacls::UVec& deriv_r_uvec,
	const FLOAT_TYPE dxInv,
	const size_t dimension,
	const size_t loop_size
) {
	const size_t alpha_size = alpha_uvec.size();
	const FLOAT_TYPE* alphas = beacls::UVec_<FLOAT_TYPE>(alpha_uvec).ptr();
	const FLOAT_TYPE* deriv_l = beacls::UVec_<FLOAT_TYPE>(deriv_l_uvec).ptr();
	const FLOAT_TYPE* deriv_r = beacls::UVec_<FLOAT_TYPE>(deriv_r_uvec).ptr();
	FLOAT_TYPE*  diss = beacls::UVec_<FLOAT_TYPE>(diss_uvec).ptr();
	//!< If partial function doesn't require global deriv min/max and it returns true,
	// calculate dissipations and step bound from alphas.
	if (beacls::is_cuda(alpha_uvec)) {
		const FLOAT_TYPE max_value = beacls::max_value<FLOAT_TYPE>(alphas, alphas + alpha_size);
		step_bound_inv = max_value * dxInv;
		if (dimension == 0) {
			for (size_t index = 0; index < loop_size; ++index) {
				const FLOAT_TYPE alpha = alphas[index];
				const FLOAT_TYPE deriv_r_index = deriv_r[index];
				const FLOAT_TYPE deriv_l_index = deriv_l[index];
				diss[index] = (deriv_r_index - deriv_l_index) * alpha / 2;
			}
		}
		else {
			for (size_t index = 0; index < loop_size; ++index) {
				const FLOAT_TYPE alpha = alphas[index];
				const FLOAT_TYPE deriv_r_index = deriv_r[index];
				const FLOAT_TYPE deriv_l_index = deriv_l[index];
				diss[index] += (deriv_r_index - deriv_l_index) * alpha / 2;
			}
		}
	}
	else {
		const FLOAT_TYPE alpha = alphas[0];
		const FLOAT_TYPE max_value = alpha;
		step_bound_inv = max_value * dxInv;
		if (dimension == 0) {
			for (size_t index = 0; index < loop_size; ++index) {
				const FLOAT_TYPE deriv_r_index = deriv_r[index];
				const FLOAT_TYPE deriv_l_index = deriv_l[index];
				diss[index] = (deriv_r_index - deriv_l_index) * alpha / 2;
			}
		}
		else {
			for (size_t index = 0; index < loop_size; ++index) {
				const FLOAT_TYPE deriv_r_index = deriv_r[index];
				const FLOAT_TYPE deriv_l_index = deriv_l[index];
				diss[index] += (deriv_r_index - deriv_l_index) * alpha / 2;
			}
		}
	}
}
#endif /* !defined(WITH_GPU) */
