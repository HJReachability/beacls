#include <helperOC/DynSys/PlaneCAvoid/PlaneCAvoid.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <array>
#include <cuda_macro.hpp>
#include "PlaneCAvoid_cuda.hpp"

#if !defined(WITH_GPU) 
#if defined(USER_DEFINED_GPU_DYNSYS_FUNC)
namespace PlaneCAvoid_CUDA {
	bool optCtrl_execute_cuda(
		std::vector<beacls::UVec>& u_uvecs,
		const std::vector<beacls::UVec>& x_uvecs,
		const std::vector<beacls::UVec>& deriv_uvecs,
		const FLOAT_TYPE wMaxA,
		const beacls::FloatVec& vRangeA,
		const helperOC::DynSys_UMode_Type uMode
	)
	{
		bool result = true;
		beacls::reallocateAsSrc(u_uvecs[0], deriv_uvecs[0]);
		beacls::reallocateAsSrc(u_uvecs[1], x_uvecs[0]);
		FLOAT_TYPE* uOpt0_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[0]).ptr();
		FLOAT_TYPE* uOpt1_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[1]).ptr();
		const FLOAT_TYPE* y0_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[0]).ptr();
		const FLOAT_TYPE* y1_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[1]).ptr();
		const FLOAT_TYPE* deriv0_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[0]).ptr();
		const FLOAT_TYPE* deriv1_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[1]).ptr();
		const FLOAT_TYPE* deriv2_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[2]).ptr();
		const FLOAT_TYPE vRangeA0 = vRangeA[0];
		const FLOAT_TYPE vRangeA1 = vRangeA[1];
		if ((uMode == helperOC::DynSys_UMode_Max) || (uMode == helperOC::DynSys_UMode_Min)) {
			const FLOAT_TYPE moded_wMaxA = (uMode == helperOC::DynSys_UMode_Max) ? wMaxA : -wMaxA;
			const FLOAT_TYPE moded_vRangeA0 = (uMode == helperOC::DynSys_UMode_Max) ? vRangeA0 : vRangeA1;
			const FLOAT_TYPE moded_vRangeA1 = (uMode == helperOC::DynSys_UMode_Max) ? vRangeA1 : vRangeA0;
			if(is_cuda(deriv_uvecs[0]) && is_cuda(deriv_uvecs[1]) && is_cuda(deriv_uvecs[2])){
				for (size_t index = 0; index < x_uvecs[0].size(); ++index) {
					const FLOAT_TYPE y0 = y0_ptr[index];
					const FLOAT_TYPE y1 = y1_ptr[index];
					const FLOAT_TYPE d0 = deriv0_ptr[index];
					const FLOAT_TYPE d1 = deriv1_ptr[index];
					const FLOAT_TYPE d2 = deriv2_ptr[index];
					const FLOAT_TYPE det1 = d0 * y1 - d1 * y0 - d2;
					uOpt1_ptr[index] = (det1 >= 0) ? moded_wMaxA : -moded_wMaxA;
				}
				for (size_t index = 0; index < deriv_uvecs[0].size(); ++index) {
					const FLOAT_TYPE det0 = -deriv0_ptr[index];
					uOpt0_ptr[index] = (det0 >= 0) ? moded_vRangeA1 : moded_vRangeA0;
				}
			}
			else {
				const FLOAT_TYPE d0 = deriv0_ptr[0];
				const FLOAT_TYPE d1 = deriv1_ptr[0];
				const FLOAT_TYPE d2 = deriv2_ptr[0];
				const FLOAT_TYPE det0 = -d0;
				uOpt0_ptr[0] = (det0 >= 0) ? moded_vRangeA1 : moded_vRangeA0;
				for (size_t index = 0; index < x_uvecs[0].size(); ++index) {
					const FLOAT_TYPE y0 = y0_ptr[index];
					const FLOAT_TYPE y1 = y1_ptr[index];
					const FLOAT_TYPE det1 = d0 * y1 - d1 * y0 - d2;
					uOpt1_ptr[index] = (det1 >= 0) ? moded_wMaxA : -moded_wMaxA;
				}
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
		const beacls::FloatVec& dMaxA,
		const beacls::FloatVec& dMaxB,
		const beacls::FloatVec& vRangeB,
		const FLOAT_TYPE wMaxB,
		const helperOC::DynSys_DMode_Type dMode
	)
	{
		bool result = true;
		beacls::reallocateAsSrc(d_uvecs[0], x_uvecs[2]);
		beacls::reallocateAsSrc(d_uvecs[1], deriv_uvecs[0]);
		beacls::reallocateAsSrc(d_uvecs[2], deriv_uvecs[0]);
		beacls::reallocateAsSrc(d_uvecs[3], deriv_uvecs[0]);
		beacls::reallocateAsSrc(d_uvecs[4], deriv_uvecs[0]);
		FLOAT_TYPE* dOpt0_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[0]).ptr();
		FLOAT_TYPE* dOpt1_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[1]).ptr();
		FLOAT_TYPE* dOpt2_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[2]).ptr();
		FLOAT_TYPE* dOpt3_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[3]).ptr();
		FLOAT_TYPE* dOpt4_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[4]).ptr();
		const FLOAT_TYPE* y2_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[2]).ptr();
		const FLOAT_TYPE* deriv0_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[0]).ptr();
		const FLOAT_TYPE* deriv1_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[1]).ptr();
		const FLOAT_TYPE* deriv2_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[2]).ptr();
		const FLOAT_TYPE dMaxA_0 = dMaxA[0];
		const FLOAT_TYPE dMaxA_1 = dMaxA[1];
		const FLOAT_TYPE dMaxB_0 = dMaxB[0];
		const FLOAT_TYPE dMaxB_1 = dMaxB[1];
		const FLOAT_TYPE vRangeB0 = vRangeB[0];
		const FLOAT_TYPE vRangeB1 = vRangeB[1];
		const FLOAT_TYPE dMaxA_0_dMaxB_0 = dMaxA_0 + dMaxB_0;
		const FLOAT_TYPE dMaxA_1_dMaxB_1 = dMaxA_1 + dMaxB_1;
		if ((dMode == helperOC::DynSys_DMode_Max) || (dMode == helperOC::DynSys_DMode_Min)) {
			const FLOAT_TYPE moded_wMaxB = (dMode == helperOC::DynSys_DMode_Max) ? wMaxB : -wMaxB;
			const FLOAT_TYPE moded_vRangeB0 = (dMode == helperOC::DynSys_DMode_Max) ? vRangeB0 : vRangeB1;
			const FLOAT_TYPE moded_vRangeB1 = (dMode == helperOC::DynSys_DMode_Max) ? vRangeB1 : vRangeB0;
			const FLOAT_TYPE moded_dMaxA_0_dMaxB_0 = (dMode == helperOC::DynSys_DMode_Max) ? dMaxA_0_dMaxB_0 : -dMaxA_0_dMaxB_0;
			const FLOAT_TYPE moded_dMaxA_1_dMaxB_1 = (dMode == helperOC::DynSys_DMode_Max) ? dMaxA_1_dMaxB_1 : -dMaxA_1_dMaxB_1;
			if (beacls::is_cuda(deriv_uvecs[0]) && beacls::is_cuda(deriv_uvecs[1]) && beacls::is_cuda(deriv_uvecs[2])){
				for (size_t index = 0; index < x_uvecs[2].size(); ++index) {
					const FLOAT_TYPE d0 = deriv0_ptr[index];
					const FLOAT_TYPE d1 = deriv1_ptr[index];
					const FLOAT_TYPE y2 = y2_ptr[index];
					const FLOAT_TYPE det0 = d0 * cos_float_type<FLOAT_TYPE>(y2) + d1 * sin_float_type<FLOAT_TYPE>(y2);
					dOpt0_ptr[index] = (det0 >= 0) ? moded_vRangeB1 : moded_vRangeB0;
				}
				for (size_t index = 0; index < deriv_uvecs[0].size(); ++index) {
					const FLOAT_TYPE d0 = deriv0_ptr[index];
					const FLOAT_TYPE d1 = deriv1_ptr[index];
					const FLOAT_TYPE d2 = deriv2_ptr[index];
					const FLOAT_TYPE det1 = d2;
					const FLOAT_TYPE det4 = d2;
					dOpt1_ptr[index] = (det1 >= 0) ? moded_wMaxB : -moded_wMaxB;
					dOpt4_ptr[index] = (det4 >= 0) ? moded_dMaxA_1_dMaxB_1 : -moded_dMaxA_1_dMaxB_1;
					const FLOAT_TYPE denom = sqrt_float_type<FLOAT_TYPE>(d0 * d0 + d1 * d1);
					dOpt2_ptr[index] = (denom == 0) ? 0 : moded_dMaxA_0_dMaxB_0 * d0 / denom;
					dOpt3_ptr[index] = (denom == 0) ? 0 : moded_dMaxA_0_dMaxB_0 * d1 / denom;
				}
			}
			else {
				const FLOAT_TYPE d0 = deriv0_ptr[0];
				const FLOAT_TYPE d1 = deriv1_ptr[0];
				const FLOAT_TYPE d2 = deriv2_ptr[0];
				for (size_t index = 0; index < x_uvecs[2].size(); ++index) {
					const FLOAT_TYPE y2 = y2_ptr[index];
					const FLOAT_TYPE det0 = d0 * cos_float_type<FLOAT_TYPE>(y2) + d1 * sin_float_type<FLOAT_TYPE>(y2);
					dOpt0_ptr[index] = (det0 >= 0) ? moded_vRangeB1 : moded_vRangeB0;
				}
				const FLOAT_TYPE det1 = d2;
				const FLOAT_TYPE det4 = d2;
				dOpt1_ptr[0] = (det1 >= 0) ? moded_wMaxB : -moded_wMaxB;
				dOpt4_ptr[0] = (det4 >= 0) ? moded_dMaxA_1_dMaxB_1 : -moded_dMaxA_1_dMaxB_1;
				const FLOAT_TYPE denom = sqrt_float_type<FLOAT_TYPE>(d0 * d0 + d1 * d1);
				dOpt2_ptr[0] = (denom == 0) ? 0 : moded_dMaxA_0_dMaxB_0 * d0 / denom;
				dOpt3_ptr[0] = (denom == 0) ? 0 : moded_dMaxA_0_dMaxB_0 * d1 / denom;
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
		beacls::reallocateAsSrc(dx_uvecs[0], x_uvecs[2]);
		beacls::reallocateAsSrc(dx_uvecs[1], x_uvecs[2]);
		beacls::reallocateAsSrc(dx_uvecs[2], u_uvecs[1]);
		FLOAT_TYPE* dx_dim0_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[0]).ptr();
		FLOAT_TYPE* dx_dim1_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[1]).ptr();
		FLOAT_TYPE* dx_dim2_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[2]).ptr();
		const FLOAT_TYPE* uOpt0_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[0]).ptr();
		const FLOAT_TYPE* uOpt1_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[1]).ptr();
		const FLOAT_TYPE* dOpt0_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[0]).ptr();
		const FLOAT_TYPE* dOpt1_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[1]).ptr();
		const FLOAT_TYPE* dOpt2_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[2]).ptr();
		const FLOAT_TYPE* dOpt3_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[3]).ptr();
		const FLOAT_TYPE* dOpt4_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[4]).ptr();
		const FLOAT_TYPE* x0_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[0]).ptr();
		const FLOAT_TYPE* x1_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[1]).ptr();
		const FLOAT_TYPE* x2_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[2]).ptr();
		if (beacls::is_cuda(u_uvecs[1]) && beacls::is_cuda(d_uvecs[0])) {
			if (beacls::is_cuda(u_uvecs[0]) && beacls::is_cuda(d_uvecs[2]) && beacls::is_cuda(d_uvecs[3]) && beacls::is_cuda(d_uvecs[1]) && beacls::is_cuda(d_uvecs[4])){
				for (size_t index = 0; index < x_uvecs[2].size(); ++index) {
					const FLOAT_TYPE u0 = uOpt0_ptr[index];
					const FLOAT_TYPE u1 = uOpt1_ptr[index];
					const FLOAT_TYPE d0 = dOpt0_ptr[index];
					const FLOAT_TYPE d2 = dOpt2_ptr[index];
					const FLOAT_TYPE d3 = dOpt3_ptr[index];
					const FLOAT_TYPE x0 = x0_ptr[index];
					const FLOAT_TYPE x1 = x1_ptr[index];
					const FLOAT_TYPE x2 = x2_ptr[index];
					dx_dim0_ptr[index] = -u0 + d0 * cos_float_type<FLOAT_TYPE>(x2) + u1 * x1 + d2;
					dx_dim1_ptr[index] = d0 * sin_float_type<FLOAT_TYPE>(x2) - u1 * x0 + d3;
					const FLOAT_TYPE d1 = dOpt1_ptr[index];
					const FLOAT_TYPE d4 = dOpt4_ptr[index];
					dx_dim2_ptr[index] = d1 - u1 + d4;

				}
			}
			else {
				const FLOAT_TYPE u0 = uOpt0_ptr[0];
				const FLOAT_TYPE d1 = dOpt1_ptr[0];
				const FLOAT_TYPE d2 = dOpt2_ptr[0];
				const FLOAT_TYPE d3 = dOpt3_ptr[0];
				const FLOAT_TYPE d4 = dOpt4_ptr[0];
				for (size_t index = 0; index < x_uvecs[2].size(); ++index) {
					const FLOAT_TYPE u1 = uOpt1_ptr[index];
					const FLOAT_TYPE d0 = dOpt0_ptr[index];
					const FLOAT_TYPE x0 = x0_ptr[index];
					const FLOAT_TYPE x1 = x1_ptr[index];
					const FLOAT_TYPE x2 = x2_ptr[index];
					dx_dim0_ptr[index] = -u0 + d0 * cos_float_type<FLOAT_TYPE>(x2) + u1 * x1 + d2;
					dx_dim1_ptr[index] = d0 * sin_float_type<FLOAT_TYPE>(x2) - u1 * x0 + d3;
					dx_dim2_ptr[index] = d1 - u1 + d4;
				}
			}
		}
		else {
			std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << " Invalid data size" << std::endl;
			return false;
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
		switch (dim) {
		case 0:
			if (beacls::is_cuda(u_uvecs[1]) && beacls::is_cuda(d_uvecs[0])) {
				beacls::reallocateAsSrc(dx_uvec, x_uvecs[2]);
				FLOAT_TYPE* dx_dim_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvec).ptr();
				const FLOAT_TYPE* x1_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[1]).ptr();
				const FLOAT_TYPE* x2_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[2]).ptr();
				const FLOAT_TYPE* uOpt0_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[0]).ptr();
				const FLOAT_TYPE* uOpt1_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[1]).ptr();
				const FLOAT_TYPE* dOpt0_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[0]).ptr();
				const FLOAT_TYPE* dOpt2_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[2]).ptr();
				if (beacls::is_cuda(u_uvecs[0]) && beacls::is_cuda(d_uvecs[2])) {
					for (size_t index = 0; index < x_uvecs[2].size(); ++index) {
						const FLOAT_TYPE u0 = uOpt0_ptr[index];
						const FLOAT_TYPE u1 = uOpt1_ptr[index];
						const FLOAT_TYPE d0 = dOpt0_ptr[index];
						const FLOAT_TYPE d2 = dOpt2_ptr[index];
						const FLOAT_TYPE x1 = x1_ptr[index];
						const FLOAT_TYPE x2 = x2_ptr[index];
						dx_dim_ptr[index] = -u0 + d0 * cos_float_type(x2) + u1 * x1 + d2;
					}
				}
				else {
					const FLOAT_TYPE u0 = uOpt0_ptr[0];
					const FLOAT_TYPE d2 = dOpt2_ptr[0];
					for (size_t index = 0; index < x_uvecs[2].size(); ++index) {
						const FLOAT_TYPE u1 = uOpt1_ptr[index];
						const FLOAT_TYPE d0 = dOpt0_ptr[index];
						const FLOAT_TYPE x1 = x1_ptr[index];
						const FLOAT_TYPE x2 = x2_ptr[index];
						dx_dim_ptr[index] = -u0 + d0 * cos_float_type(x2) + u1 * x1 + d2;
					}
				}
			}
			else {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << " Invalid data size" << std::endl;
				result = false;
			}
			break;
		case 1:
			if (beacls::is_cuda(u_uvecs[1]) && beacls::is_cuda(d_uvecs[0])) {
				beacls::reallocateAsSrc(dx_uvec, x_uvecs[2]);
				FLOAT_TYPE* dx_dim_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvec).ptr();
				const FLOAT_TYPE* x0_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[0]).ptr();
				const FLOAT_TYPE* x2_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[2]).ptr();
				const FLOAT_TYPE* uOpt1_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[1]).ptr();
				const FLOAT_TYPE* dOpt0_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[0]).ptr();
				const FLOAT_TYPE* dOpt3_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[3]).ptr();
				if (beacls::is_cuda(d_uvecs[3])) {
					for (size_t index = 0; index < x_uvecs[2].size(); ++index) {
						const FLOAT_TYPE u1 = uOpt1_ptr[index];
						const FLOAT_TYPE d0 = dOpt0_ptr[index];
						const FLOAT_TYPE d3 = dOpt3_ptr[index];
						const FLOAT_TYPE x0 = x0_ptr[index];
						const FLOAT_TYPE x2 = x2_ptr[index];
						dx_dim_ptr[index] = d0 * sin_float_type(x2) - u1 * x0 + d3;
					}
				}
				else {
					const FLOAT_TYPE d3 = dOpt3_ptr[0];
					for (size_t index = 0; index < x_uvecs[2].size(); ++index) {
						const FLOAT_TYPE u1 = uOpt1_ptr[index];
						const FLOAT_TYPE d0 = dOpt0_ptr[index];
						const FLOAT_TYPE x0 = x0_ptr[index];
						const FLOAT_TYPE x2 = x2_ptr[index];
						dx_dim_ptr[index] = d0 * sin_float_type(x2) - u1 * x0 + d3;
					}
				}
			}
			else {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << " Invalid data size" << std::endl;
				result = false;
			}
			break;
		case 2:
			if (beacls::is_cuda(u_uvecs[1])) {
				beacls::reallocateAsSrc(dx_uvec, u_uvecs[1]);
				FLOAT_TYPE* dx_dim_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvec).ptr();
				const FLOAT_TYPE* uOpt1_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[1]).ptr();
				const FLOAT_TYPE* dOpt1_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[1]).ptr();
				const FLOAT_TYPE* dOpt4_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[4]).ptr();
				if (beacls::is_cuda(d_uvecs[1]) && beacls::is_cuda(d_uvecs[4])) {
					for (size_t index = 0; index < u_uvecs[1].size(); ++index) {
						const FLOAT_TYPE u1 = uOpt1_ptr[index];
						const FLOAT_TYPE d1 = dOpt1_ptr[index];
						const FLOAT_TYPE d4 = dOpt4_ptr[index];
						dx_dim_ptr[index] = d1 - u1 + d4;
					}
				}
				else {
					const FLOAT_TYPE d1 = dOpt1_ptr[0];
					const FLOAT_TYPE d4 = dOpt4_ptr[0];
					for (size_t index = 0; index < u_uvecs[1].size(); ++index) {
						const FLOAT_TYPE u1 = uOpt1_ptr[index];
						dx_dim_ptr[index] = d1 - u1 + d4;
					}
				}
			}
			else {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << " Invalid data size" << std::endl;
				result = false;
			}
			break;
		default:
			std::cerr << "Only dimension 1-4 are defined for dynamics of PlaneCAvoid!" << std::endl;
			result = false;
			break;
		}
		return result;
	}
};
#endif /* defined(USER_DEFINED_GPU_DYNSYS_FUNC) */
#endif /* !defined(WITH_GPU) */
