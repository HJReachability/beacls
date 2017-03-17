#include <helperOC/DynSys/KinVehicleND/KinVehicleND.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <array>
#include <cuda_macro.hpp>
#include "KinVehicleND_cuda.hpp"

#if !defined(WITH_GPU) 
#if defined(USER_DEFINED_GPU_DYNSYS_FUNC)
namespace KinVehicleND_CUDA {
	bool optCtrl_execute_cuda(
		std::vector<beacls::UVec>& u_uvecs,
		const std::vector<beacls::UVec>& deriv_uvecs,
		const FLOAT_TYPE vMax,
		const helperOC::DynSys_UMode_Type uMode
	){
		bool result = true;
		const size_t length = deriv_uvecs[0].size();
		std::for_each(u_uvecs.begin(), u_uvecs.end(), [&deriv_uvecs](auto& rhs) {
			beacls::reallocateAsSrc(rhs, deriv_uvecs[0]);
		});
		const size_t nu = deriv_uvecs.size();
		if (beacls::is_cuda(deriv_uvecs[0])) {
			FLOAT_TYPE* last_uOpt_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[nu - 1]).ptr();
			if ((uMode == helperOC::DynSys_UMode_Max) || (uMode == helperOC::DynSys_UMode_Min)) {
				const FLOAT_TYPE moded_vMax = (uMode == helperOC::DynSys_UMode_Max) ? vMax : -vMax;
				//!< store denom to uOpts[nu-1]
				const FLOAT_TYPE* deriv0 = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[0]).ptr();
				for (size_t index = 0; index < length; ++index) {
					const FLOAT_TYPE d0 = deriv0[index];
					last_uOpt_ptr[index] = d0 * d0;
				}
				for (size_t dim = 1; dim < nu; ++dim) {
					const FLOAT_TYPE* deriv_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[dim]).ptr();
					for (size_t index = 0; index < length; ++index) {
						const FLOAT_TYPE d = deriv_ptr[index];
						last_uOpt_ptr[index] = last_uOpt_ptr[index] + d * d;
					}
				}
				for (size_t index = 0; index < length; ++index) {
					last_uOpt_ptr[index] = sqrt_float_type<FLOAT_TYPE>(last_uOpt_ptr[index]);
				}
				for (size_t dim = 1; dim < nu; ++dim) {
					FLOAT_TYPE* uOpt_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[dim]).ptr();
					const FLOAT_TYPE* deriv_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[dim]).ptr();
					for (size_t index = 0; index < length; ++index) {
						const FLOAT_TYPE denom = last_uOpt_ptr[index];
						const FLOAT_TYPE d = deriv_ptr[index];
						uOpt_ptr[index] = (denom == 0) ? 0 : moded_vMax * d / denom;
					}
				}
			}
			else {
				std::cerr << "Unknown uMode!: " << uMode << std::endl;
				result = false;
			}
		}
		else {
			if ((uMode == helperOC::DynSys_UMode_Max) || (uMode == helperOC::DynSys_UMode_Min)) {
				const FLOAT_TYPE moded_vMax = (uMode == helperOC::DynSys_UMode_Max) ? vMax : -vMax;
				//!< store denom to uOpts[nu-1]
				const FLOAT_TYPE* deriv0_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[0]).ptr();
				const FLOAT_TYPE d0 = deriv0_ptr[0];
				FLOAT_TYPE last_uOpt = d0 * d0;
				for (size_t dim = 1; dim < nu; ++dim) {
					const FLOAT_TYPE* deriv_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[dim]).ptr();
					const FLOAT_TYPE d = deriv_ptr[0];
					last_uOpt += d * d;
				}
				const FLOAT_TYPE denom = std::sqrt(last_uOpt);
				for (size_t dim = 1; dim < nu; ++dim) {
					const FLOAT_TYPE* deriv_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[dim]).ptr();
					FLOAT_TYPE* uOpt_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[dim]).ptr();
					const FLOAT_TYPE d = deriv_ptr[0];
					const FLOAT_TYPE val = (denom == 0) ? 0 : moded_vMax * d / denom;
					uOpt_ptr[0] = val;
				}
			}
			else {
				std::cerr << "Unknown uMode!: " << uMode << std::endl;
				result = false;
			}

		}
		return result;
	}
};
#endif /* defined(USER_DEFINED_GPU_DYNSYS_FUNC) */
#endif /* !defined(WITH_GPU) */
