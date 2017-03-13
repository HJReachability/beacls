#include <helperOC/DynSys/Quad4D/Quad4D.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <array>
#include <cuda_macro.hpp>
#include "Quad4D_cuda.hpp"

#if !defined(WITH_GPU) 
#if defined(USER_DEFINED_GPU_DYNSYS_FUNC) && 0
namespace Quad4D_CUDA {
	bool optCtrl_execute_cuda(
		std::vector<beacls::UVec>& u_uvecs,
		const std::vector<beacls::UVec>& x_uvecs,
		const std::vector<beacls::UVec>& deriv_uvecs,
		const FLOAT_TYPE wMax,
		const FLOAT_TYPE vrange_min,
		const FLOAT_TYPE vrange_max,
		const DynSys_UMode_Type uMode
	)
	{
		bool result = true;
		if ((uMode == DynSys_UMode_Max) || (uMode == DynSys_UMode_Min)) {
			const FLOAT_TYPE moded_vrange_max = (uMode == DynSys_UMode_Max) ? vrange_max : vrange_min;
			const FLOAT_TYPE moded_vrange_min = (uMode == DynSys_UMode_Max) ? vrange_min : vrange_max;
			const FLOAT_TYPE moded_wMax = (uMode == DynSys_UMode_Max) ? wMax : -wMax;
			beacls::reallocateAsSrc(u_uvecs[0], x_uvecs[2]);
			beacls::reallocateAsSrc(u_uvecs[1], deriv_uvecs[2]);
			FLOAT_TYPE* uOpt0_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[0]).ptr();
			FLOAT_TYPE* uOpt1_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[1]).ptr();
			const FLOAT_TYPE* y2_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[2]).ptr();
			const FLOAT_TYPE* deriv0_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[0]).ptr();
			const FLOAT_TYPE* deriv1_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[1]).ptr();
			const FLOAT_TYPE* deriv2_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[2]).ptr();
			if (is_cuda(deriv_uvecs[2])){
				for (size_t index = 0; index < x_uvecs[2].size(); ++index) {
					const FLOAT_TYPE y2 = y2_ptr[index];
					const FLOAT_TYPE deriv0 = deriv0_ptr[index];
					const FLOAT_TYPE deriv1 = deriv1_ptr[index];
					const FLOAT_TYPE det1 = deriv0 * cos_float_type(y2) + deriv1 * sin_float_type(y2);
					uOpt0_ptr[index] = (det1 >= 0) ? moded_vrange_max : moded_vrange_min;
				}
				for (size_t index = 0; index < deriv_uvecs[2].size(); ++index) {
					uOpt1_ptr[index] = (deriv2_ptr[index] >= 0) ? moded_wMax : -moded_wMax;
				}
			}
			else {
				const FLOAT_TYPE deriv0 = deriv0_ptr[0];
				const FLOAT_TYPE deriv1 = deriv1_ptr[0];
				const FLOAT_TYPE deriv2 = deriv2_ptr[0];
				for (size_t index = 0; index < x_uvecs[2].size(); ++index) {
					const FLOAT_TYPE y2 = y2_ptr[index];
					const FLOAT_TYPE det1 = deriv0 * cos_float_type(y2) + deriv1 * sin_float_type(y2);
					uOpt0_ptr[index] = (det1 >= 0) ? moded_vrange_max : moded_vrange_min;
				}
				uOpt1_ptr[0] = (deriv2 >= 0) ? moded_wMax : -moded_wMax;
			}
		}
		else {
			std::cerr << "Unknown uMode!: " << uMode << std::endl;
			result = false;
		}
		return result;
	}

	bool optDstb_execute_cuda(
		std::vector<beacls::UVec>& d_uvecs,
		const std::vector<beacls::UVec>& x_uvecs,
		const std::vector<beacls::UVec>& deriv_uvecs,
		const beacls::FloatVec& dMax,
		const DynSys_DMode_Type dMode
	)
	{
		bool result = true;
		const FLOAT_TYPE dMax_0 = dMax[0];
		const FLOAT_TYPE dMax_1 = dMax[1];
		if ((dMode == DynSys_DMode_Max) || (dMode == DynSys_DMode_Min)) {
			const FLOAT_TYPE moded_dMax_0 = (dMode == DynSys_DMode_Max) ? dMax_0 : -dMax_0;
			const FLOAT_TYPE moded_dMax_1 = (dMode == DynSys_DMode_Max) ? dMax_1 : -dMax_1;
			beacls::reallocateAsSrc(d_uvecs[0], deriv_uvecs[0]);
			beacls::reallocateAsSrc(d_uvecs[1], deriv_uvecs[0]);
			beacls::reallocateAsSrc(d_uvecs[2], deriv_uvecs[2]);
			FLOAT_TYPE* dOpt0_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[0]).ptr();
			FLOAT_TYPE* dOpt1_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[1]).ptr();
			FLOAT_TYPE* dOpt2_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[2]).ptr();
			const FLOAT_TYPE* deriv0_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[0]).ptr();
			const FLOAT_TYPE* deriv1_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[1]).ptr();
			const FLOAT_TYPE* deriv2_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[2]).ptr();

			if (is_cuda(deriv_uvecs[0]) && is_cuda(deriv_uvecs[1]) && is_cuda(deriv_uvecs[2])) {
				for (size_t index = 0; index < x_uvecs[2].size(); ++index) {
					const FLOAT_TYPE deriv0 = deriv0_ptr[index];
					const FLOAT_TYPE deriv1 = deriv1_ptr[index];
					const FLOAT_TYPE deriv2 = deriv2_ptr[index];
					const FLOAT_TYPE normDeriv01 = std::sqrt(deriv0 * deriv0 + deriv1 * deriv1);
					dOpt0_ptr[index] = (normDeriv01 == 0) ? 0 : moded_dMax_0 * deriv0 / normDeriv01;
					dOpt1_ptr[index] = (normDeriv01 == 0) ? 0 : moded_dMax_0 * deriv1 / normDeriv01;
					dOpt2_ptr[index] = (deriv2 >= 0) ? moded_dMax_1 : -moded_dMax_1;
				}
			}
			else {
				const FLOAT_TYPE deriv0 = deriv0_ptr[0];
				const FLOAT_TYPE deriv1 = deriv1_ptr[0];
				const FLOAT_TYPE deriv2 = deriv2_ptr[0];
				const FLOAT_TYPE normDeriv = sqrt_float_type(deriv0 * deriv0 + deriv1 * deriv1);
				if (normDeriv == 0) {
					dOpt0_ptr[0] = 0;
					dOpt1_ptr[0] = 0;
				}
				else {
					dOpt0_ptr[0] = moded_dMax_0 * deriv0 / normDeriv;
					dOpt1_ptr[0] = moded_dMax_0 * deriv1 / normDeriv;
				}
				dOpt2_ptr[0] = (deriv2 >= 0) ? moded_dMax_1 : -moded_dMax_1;
			}
		}
		else {
			std::cerr << "Unknown dMode!: " << dMode << std::endl;
			result = false;
		}
		return result;
	}

	bool dynamics_cell_helper_execute_cuda_dimAll(
		std::vector<beacls::UVec>& dx_uvecs,
		const std::vector<beacls::UVec>& x_uvecs,
		const std::vector<beacls::UVec>& u_uvecs,
		const std::vector<beacls::UVec>& d_uvecs
	) {
		bool result = true;
		const size_t src_x_dim_index = 2;
		beacls::reallocateAsSrc(dx_uvecs[0], x_uvecs[src_x_dim_index]);
		beacls::reallocateAsSrc(dx_uvecs[1], x_uvecs[src_x_dim_index]);
		beacls::reallocateAsSrc(dx_uvecs[2], u_uvecs[1]);
		FLOAT_TYPE* dx_dim0_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[0]).ptr();
		FLOAT_TYPE* dx_dim1_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[1]).ptr();
		FLOAT_TYPE* dx_dim2_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[2]).ptr();
		const FLOAT_TYPE* us_0_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[0]).ptr();
		const FLOAT_TYPE* us_1_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[1]).ptr();
		const FLOAT_TYPE* ds_0_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[0]).ptr();
		const FLOAT_TYPE* ds_1_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[1]).ptr();
		const FLOAT_TYPE* ds_2_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[2]).ptr();
		const FLOAT_TYPE* x_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[src_x_dim_index]).ptr();
		if (is_cuda(d_uvecs[0]) && is_cuda(d_uvecs[1])&& is_cuda(u_uvecs[1]) && is_cuda(d_uvecs[2])) {
			for (size_t index = 0; index < x_uvecs[src_x_dim_index].size(); ++index) {
				const FLOAT_TYPE u0 = us_0_ptr[index];
				const FLOAT_TYPE d0 = ds_0_ptr[index];
				const FLOAT_TYPE d1 = ds_1_ptr[index];
				const FLOAT_TYPE x = x_ptr[index];
				dx_dim0_ptr[index] = u0 * cos_float_type(x) + d0;
				dx_dim1_ptr[index] = u0 * sin_float_type(x) + d1;
				const FLOAT_TYPE u1 = us_1_ptr[index];
				const FLOAT_TYPE d2 = ds_2_ptr[index];
				dx_dim2_ptr[index] = u1 + d2;
			}
		}
		else if (!is_cuda(d_uvecs[0]) && !is_cuda(d_uvecs[1])&& !is_cuda(u_uvecs[1]) && !is_cuda(d_uvecs[2])) {
			const FLOAT_TYPE d0 = ds_0_ptr[0];
			const FLOAT_TYPE d1 = ds_1_ptr[0];
			for (size_t index = 0; index < x_uvecs[src_x_dim_index].size(); ++index) {
				const FLOAT_TYPE u0 = us_0_ptr[index];
				const FLOAT_TYPE x = x_ptr[index];
				dx_dim0_ptr[index] = u0 * cos_float_type(x) + d0;
				dx_dim1_ptr[index] = u0 * sin_float_type(x) + d1;
				const FLOAT_TYPE u1 = us_1_ptr[0];
				const FLOAT_TYPE d2 = ds_2_ptr[0];
				dx_dim2_ptr[0] = u1 + d2;
			}
		} else {
			std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << " Invalid data size" << std::endl;
			result = false;
		}
		return result;
	}

	bool dynamics_cell_helper_execute_cuda(
		beacls::UVec& dx_uvec,
		const std::vector<beacls::UVec>& x_uvecs,
		const std::vector<beacls::UVec>& u_uvecs,
		const std::vector<beacls::UVec>& d_uvecs,
		const size_t dim
	) {
		bool result = true;
		const size_t src_x_dim_index = 2;
		switch (dim) {
		case 0:
			if (beacls::is_cuda(u_uvecs[0])){
				beacls::reallocateAsSrc(dx_uvec, x_uvecs[src_x_dim_index]);
				FLOAT_TYPE* dx_dim_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvec).ptr();
				const FLOAT_TYPE* x_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[src_x_dim_index]).ptr();
				const FLOAT_TYPE* us_0_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[0]).ptr();
				const FLOAT_TYPE* ds_0_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[0]).ptr();
				if (is_cuda(d_uvecs[0])) {
					for (size_t index = 0; index < x_uvecs[src_x_dim_index].size(); ++index) {
						const FLOAT_TYPE u0 = us_0_ptr[index];
						const FLOAT_TYPE d0 = ds_0_ptr[index];
						const FLOAT_TYPE x = x_ptr[index];
						dx_dim_ptr[index] = u0 * cos_float_type(x) + d0;
					}
				}
				else {	//!< ds_0_size != length
					const FLOAT_TYPE d0 = ds_0_ptr[0];
					for (size_t index = 0; index < x_uvecs[src_x_dim_index].size(); ++index) {
						const FLOAT_TYPE u0 = us_0_ptr[index];
						const FLOAT_TYPE x = x_ptr[index];
						dx_dim_ptr[index] = u0 * cos_float_type(x) + d0;
					}
				}
			}
			else {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << " Invalid data size" << std::endl;
				result = false;
			}
			break;
		case 1:
			if (beacls::is_cuda(u_uvecs[0])) {
				beacls::reallocateAsSrc(dx_uvec, x_uvecs[src_x_dim_index]);
				FLOAT_TYPE* dx_dim_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvec).ptr();
				const FLOAT_TYPE* x_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[src_x_dim_index]).ptr();
				const FLOAT_TYPE* us_0_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[0]).ptr();
				const FLOAT_TYPE* ds_1_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[1]).ptr();
				if (is_cuda(d_uvecs[1])) {
					for (size_t index = 0; index < x_uvecs[src_x_dim_index].size(); ++index) {
						const FLOAT_TYPE u0 = us_0_ptr[index];
						const FLOAT_TYPE d1 = ds_1_ptr[index];
						const FLOAT_TYPE x = x_ptr[index];
						dx_dim_ptr[index] = u0 * sin_float_type(x) + d1;
					}
				}
				else {	//!< ds_1_size != length
					const FLOAT_TYPE d1 = ds_1_ptr[0];
					for (size_t index = 0; index < x_uvecs[src_x_dim_index].size(); ++index) {
						const FLOAT_TYPE u0 = us_0_ptr[index];
						const FLOAT_TYPE x = x_ptr[index];
						dx_dim_ptr[index] = u0 * sin_float_type(x) + d1;
					}
				}
			}
			else {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << " Invalid data size" << std::endl;
				result = false;
			}
			break;
		case 2:
			{
				beacls::reallocateAsSrc(dx_uvec, u_uvecs[1]);
				FLOAT_TYPE* dx_dim_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvec).ptr();
				const FLOAT_TYPE* us_1_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[1]).ptr();
				const FLOAT_TYPE* ds_2_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[2]).ptr();
				if (is_cuda(u_uvecs[1]) && is_cuda(d_uvecs[2])) {
					for (size_t index = 0; index < u_uvecs[1].size(); ++index) {
						const FLOAT_TYPE u1 = us_1_ptr[index];
						const FLOAT_TYPE d2 = ds_2_ptr[index];
						dx_dim_ptr[index] = u1 + d2;
					}
				}
				else if (!is_cuda(u_uvecs[1]) && !is_cuda(d_uvecs[2])) {
					const FLOAT_TYPE u1 = us_1_ptr[0];
					const FLOAT_TYPE d2 = ds_2_ptr[0];
					dx_dim_ptr[0] = u1 + d2;
				}
				else {
					std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << " Invalid data size" << std::endl;
					result = false;
				}
			}
			break;
		default:
			std::cerr << "Only dimension 1-4 are defined for dynamics of Quad4D!" << std::endl;
			result = false;
			break;
		}
		return result;
	}
};
#endif /* defined(USER_DEFINED_GPU_DYNSYS_FUNC) */
#endif /* !defined(WITH_GPU) */
