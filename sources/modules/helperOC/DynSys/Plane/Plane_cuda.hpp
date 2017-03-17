#ifndef __Plane_cuda_hpp__
#define __Plane_cuda_hpp__

#include <typedef.hpp>
#include <cuda_macro.hpp>
#include <vector>
#include <helperOC/DynSys/DynSys/DynSysTypeDef.hpp>
#include <Core/UVec.hpp>

#if defined(USER_DEFINED_GPU_DYNSYS_FUNC)
namespace Plane_CUDA {
	bool optCtrl_execute_cuda(
		std::vector<beacls::UVec>& u_uvecs,
		const std::vector<beacls::UVec>& x_uvecs,
		const std::vector<beacls::UVec>& deriv_uvecs,
		const FLOAT_TYPE wMax,
		const FLOAT_TYPE vrange_min,
		const FLOAT_TYPE vrange_max,
		const helperOC::DynSys_UMode_Type uMode
	);
	bool optDstb_execute_cuda(
		std::vector<beacls::UVec>& d_uvecs,
		const std::vector<beacls::UVec>& x_uvecs,
		const std::vector<beacls::UVec>& deriv_uvecs,
		const beacls::FloatVec& dMax,
		const helperOC::DynSys_DMode_Type dMode
	);
	bool dynamics_cell_helper_execute_cuda_dimAll(
		std::vector<beacls::UVec>& dx_uvecs,
		const std::vector<beacls::UVec>& x_uvecs,
		const std::vector<beacls::UVec>& u_uvecs,
		const std::vector<beacls::UVec>& d_uvecs
	);
	bool dynamics_cell_helper_execute_cuda(
		beacls::UVec& dx_uvec,
		const std::vector<beacls::UVec>& x_uvecs,
		const std::vector<beacls::UVec>& u_uvecs,
		const std::vector<beacls::UVec>& d_uvecs,
		const size_t dim
	);
	bool optCtrl_execute_cuda(
		std::vector<beacls::UVec>& uL_uvecs,
		std::vector<beacls::UVec>& uU_uvecs,
		const std::vector<beacls::UVec>& x_uvecs,
		const std::vector<beacls::UVec>& derivMin_uvecs,
		const std::vector<beacls::UVec>& derivMax_uvecs,
		const FLOAT_TYPE wMax,
		const FLOAT_TYPE vrange_min,
		const FLOAT_TYPE vrange_max,
		const helperOC::DynSys_UMode_Type uMode
	);
	bool optDstb_execute_cuda(
		std::vector<beacls::UVec>& dL_uvecs,
		std::vector<beacls::UVec>& dU_uvecs,
		const std::vector<beacls::UVec>& x_uvecs,
		const std::vector<beacls::UVec>& derivMin_uvecs,
		const std::vector<beacls::UVec>& derivMax_uvecs,
		const beacls::FloatVec& dMax,
		const helperOC::DynSys_DMode_Type dMode
	);
	bool dynamics_cell_helper_execute_cuda(
		beacls::UVec& alpha_uvec,
		const std::vector<beacls::UVec>& x_uvecs,
		const std::vector<beacls::UVec>& uL_uvecs,
		const std::vector<beacls::UVec>& uU_uvecs,
		const std::vector<beacls::UVec>& dL_uvecs,
		const std::vector<beacls::UVec>& dU_uvecs,
		const size_t dim
	);
	bool HamFunction_cuda(
		beacls::UVec& hamValue_uvec,
		const std::vector<beacls::UVec>& x_uvecs,
		const std::vector<beacls::UVec>& deriv_uvecs,
		const FLOAT_TYPE wMax,
		const FLOAT_TYPE vrange_min,
		const FLOAT_TYPE vrange_max,
		const beacls::FloatVec& dMax,
		const helperOC::DynSys_UMode_Type uMode,
		const helperOC::DynSys_DMode_Type dMode,
		const bool negate);
	bool PartialFunction_cuda(
		beacls::UVec& alpha_uvecs,
		const std::vector<beacls::UVec>& x_uvecs,
		const std::vector<beacls::UVec>& derivMin_uvecs,
		const std::vector<beacls::UVec>& derivMax_uvecs,
		const size_t dim,
		const FLOAT_TYPE wMax,
		const FLOAT_TYPE vrange_min,
		const FLOAT_TYPE vrange_max,
		const beacls::FloatVec& dMax,
		const helperOC::DynSys_UMode_Type uMode,
		const helperOC::DynSys_DMode_Type dMode
	);
	template<typename T>
	__host__ __device__
	static inline
	void get_dxs(
		T& dx0, 
		T& dx1, 
		T& dx2,
		const T y2,
		const T deriv0, 
		const T deriv1,
		const T deriv2,
		const T wMax,
		const T vrange_min, 
		const T vrange_max,
		const T dMax_0, 
		const T dMax_1
		) {
		T cos_y2;
		T sin_y2;
		sincos_float_type<T>(y2, sin_y2, cos_y2);
		const T det1 = deriv0 * cos_y2 + deriv1 * sin_y2;
		const T u0 = (det1 >= 0) ? vrange_max : vrange_min;
		const T normDeriv = sqrt_float_type(deriv0 * deriv0 + deriv1 * deriv1);
		const T d0 = (normDeriv == 0) ? 0 : dMax_0 * deriv0 / normDeriv;
		const T d1 = (normDeriv == 0) ? 0 : dMax_0 * deriv1 / normDeriv;

		dx0 = u0 * cos_y2 + d0;
		dx1 = u0 * sin_y2 + d1;
		dx2 = (deriv2 >= 0) ? (wMax + dMax_1) : -(wMax + dMax_1);
	}
	template<typename T>
	__host__ __device__
	static inline
	T getAlpha(
		const T cos_y2,
		const T sin_y2,
		const T cos_or_sin_y2,
		const T derivMin0,
		const T derivMax0,
		const T derivMin1,
		const T derivMax1,
		const T dL,
		const T dU,
		const T vrange_min, const T vrange_max
		)
	{
		const FLOAT_TYPE detMax1 = derivMax0 * cos_y2 + derivMax1 * sin_y2;
		const FLOAT_TYPE uU0 = (detMax1 >= 0) ? vrange_max : vrange_min;
		const FLOAT_TYPE detMin1 = derivMin0 * cos_y2 + derivMin1 * sin_y2;
		const FLOAT_TYPE uL0 = (detMin1 >= 0) ? vrange_max : vrange_min;

		const FLOAT_TYPE dxUU = uU0 * cos_or_sin_y2 + dU;
		const FLOAT_TYPE dxUL = uU0 * cos_or_sin_y2 + dL;
		const FLOAT_TYPE dxLU = uL0 * cos_or_sin_y2 + dU;
		const FLOAT_TYPE dxLL = uL0 * cos_or_sin_y2 + dL;
		const FLOAT_TYPE max0 = max_float_type<FLOAT_TYPE>(abs_float_type<FLOAT_TYPE>(dxUU), abs_float_type<FLOAT_TYPE>(dxUL));
		const FLOAT_TYPE max1 = max_float_type<FLOAT_TYPE>(abs_float_type<FLOAT_TYPE>(dxLL), abs_float_type<FLOAT_TYPE>(dxLU));
		return max_float_type<FLOAT_TYPE>(max0, max1);
	}
};
#endif /* defined(USER_DEFINED_GPU_DYNSYS_FUNC) */
#endif /*__Plane_cuda_hpp__*/
