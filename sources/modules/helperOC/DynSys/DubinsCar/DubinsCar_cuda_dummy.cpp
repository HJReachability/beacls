#include <helperOC/DynSys/DubinsCar/DubinsCar.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <array>
#include <cuda_macro.hpp>
#include "DubinsCar_cuda.hpp"

#if !defined(WITH_GPU) 
#if defined(USER_DEFINED_GPU_DYNSYS_FUNC)
namespace DubinsCar_CUDA {
	bool optCtrl_execute_cuda(
		beacls::UVec& u_uvec,
		const beacls::UVec& deriv_uvec,
		const FLOAT_TYPE wMax,
		const helperOC::DynSys_UMode_Type uMode
	)
	{
		bool result = true;
		beacls::reallocateAsSrc(u_uvec, deriv_uvec);
		FLOAT_TYPE* uOpt_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvec).ptr();
		const FLOAT_TYPE* deriv_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvec).ptr();
		if (beacls::is_cuda(deriv_uvec)) {
			switch (uMode) {
			case helperOC::DynSys_UMode_Max:
				for (size_t index = 0; index < deriv_uvec.size(); ++index) {
					uOpt_ptr[index] = (deriv_ptr[index] >= 0) ? wMax : -wMax;
				}
				break;
			case helperOC::DynSys_UMode_Min:
				for (size_t index = 0; index < deriv_uvec.size(); ++index) {
					uOpt_ptr[index] = (deriv_ptr[index] >= 0) ? -wMax : wMax;
				}
				break;
			case helperOC::DynSys_UMode_Invalid:
			default:
				std::cerr << "Unknown uMode!: " << uMode << std::endl;
				result = false;
				break;
			}
		}
		else {
			const FLOAT_TYPE d = deriv_ptr[0];
			switch (uMode) {
			case helperOC::DynSys_UMode_Max:
				uOpt_ptr[0] = (d >= 0) ? wMax : -wMax;
				break;
			case helperOC::DynSys_UMode_Min:
				uOpt_ptr[0] = (d >= 0) ? -wMax : wMax;
				break;
			case helperOC::DynSys_UMode_Invalid:
			default:
				std::cerr << "Unknown uMode!: " << uMode << std::endl;
				result = false;
				break;
			}
		}
		return result;
	}

	bool optDstb_execute_cuda(
		std::vector<beacls::UVec>& d_uvecs,
		const beacls::UVec& deriv_uvec,
		const beacls::FloatVec& dMax,
		const helperOC::DynSys_DMode_Type dMode,
		const beacls::IntegerVec& dims
	)
	{
		bool result = true;
		const FLOAT_TYPE* deriv_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvec).ptr();
		if (beacls::is_cuda(deriv_uvec)) {
			for (size_t dim = 0; dim < 3; ++dim) {
				if (std::find(dims.cbegin(), dims.cend(), dim) != dims.cend()) {
					beacls::reallocateAsSrc(d_uvecs[dim], deriv_uvec);
					FLOAT_TYPE* dOpt_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[dim]).ptr();
					const FLOAT_TYPE dMax_d = dMax[dim];
					switch (dMode) {
					case helperOC::DynSys_DMode_Max:
						for (size_t index = 0; index < deriv_uvec.size(); ++index) {
							dOpt_ptr[index] = deriv_ptr[index] >= 0 ? dMax_d : -dMax_d;
						}
						break;
					case helperOC::DynSys_DMode_Min:
						for (size_t index = 0; index < deriv_uvec.size(); ++index) {
							dOpt_ptr[index] = deriv_ptr[index] >= 0 ? -dMax_d : dMax_d;
						}
						break;
					case helperOC::DynSys_UMode_Invalid:
					default:
						std::cerr << "Unknown dMode!: " << dMode << std::endl;
						result = false;
					}
				}
			}
		}
		else {
			const FLOAT_TYPE d = deriv_ptr[0];
			for (size_t dim = 0; dim < 3; ++dim) {
				if (std::find(dims.cbegin(), dims.cend(), dim) != dims.cend()) {
					beacls::reallocateAsSrc(d_uvecs[dim], deriv_uvec);
					FLOAT_TYPE* dOpt_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[dim]).ptr();
					const FLOAT_TYPE dMax_d = dMax[dim];
					switch (dMode) {
					case helperOC::DynSys_DMode_Max:
						dOpt_ptr[0] = (d >= 0) ? dMax_d : -dMax_d;
						break;
					case helperOC::DynSys_DMode_Min:
						dOpt_ptr[0] = (d >= 0) ? -dMax_d : dMax_d;
						break;
					case helperOC::DynSys_UMode_Invalid:
					default:
						std::cerr << "Unknown dMode!: " << dMode << std::endl;
						result = false;
					}
				}
			}
		}
		return result;
	}

	bool dynamics_cell_helper_execute_cuda_dimAll(
		std::vector<beacls::UVec>& dx_uvecs,
		const std::vector<beacls::UVec>& x_uvecs,
		const std::vector<beacls::UVec>& d_uvecs,
		const FLOAT_TYPE speed,
		const size_t src_dim0_dst_dx_index,
		const size_t src_dim1_dst_dx_index,
		const size_t src_x_dim
		) {
		bool result = true;
		const size_t length = x_uvecs[src_x_dim].size();
		beacls::reallocateAsSrc(dx_uvecs[src_dim0_dst_dx_index], x_uvecs[src_x_dim]);
		beacls::reallocateAsSrc(dx_uvecs[src_dim1_dst_dx_index], x_uvecs[src_x_dim]);
		FLOAT_TYPE* dx_dim0_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[src_dim0_dst_dx_index]).ptr();
		FLOAT_TYPE* dx_dim1_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[src_dim1_dst_dx_index]).ptr();
		const FLOAT_TYPE* ds_0_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[src_dim0_dst_dx_index]).ptr();
		const FLOAT_TYPE* ds_1_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[src_dim1_dst_dx_index]).ptr();
		const FLOAT_TYPE* x_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[src_x_dim]).ptr();

		if (beacls::is_cuda(d_uvecs[0]) && beacls::is_cuda(d_uvecs[1])) {
			for (size_t index = 0; index < length; ++index) {
				const FLOAT_TYPE x = x_ptr[index];
				const FLOAT_TYPE d0 = ds_0_ptr[index];
				const FLOAT_TYPE d1 = ds_1_ptr[index];
				dx_dim0_ptr[index] = speed * cos_float_type(x) + d0;
				dx_dim1_ptr[index] = speed * sin_float_type(x) + d1;
			}
		}
		else if (!beacls::is_cuda(d_uvecs[0]) && !beacls::is_cuda(d_uvecs[1])) {
			const FLOAT_TYPE d0 = ds_0_ptr[0];
			const FLOAT_TYPE d1 = ds_1_ptr[0];
			for (size_t index = 0; index < length; ++index) {
				const FLOAT_TYPE x = x_ptr[index];
				dx_dim0_ptr[index] = speed * cos_float_type(x) + d0;
				dx_dim1_ptr[index] = speed * sin_float_type(x) + d1;
			}
		}
		else {
			std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << " Invalid data size" << std::endl;
			result = false;
		}
		return result;
	}

	bool dynamics_cell_helper_execute_cuda(
		std::vector<beacls::UVec>& dx_uvecs,
		const std::vector<beacls::UVec>& x_uvecs,
		const std::vector<beacls::UVec>& u_uvecs,
		const std::vector<beacls::UVec>& d_uvecs,
		const FLOAT_TYPE speed,
		const size_t dst_target_dim,
		const size_t src_target_dim,
		const size_t src_x_dim
	) {
		bool result = true;
		switch (src_target_dim) {
		case 0:
		{
			beacls::reallocateAsSrc(dx_uvecs[dst_target_dim], x_uvecs[src_x_dim]);
			FLOAT_TYPE* dx_dim_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[dst_target_dim]).ptr();
			const FLOAT_TYPE* x_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[src_x_dim]).ptr();
			const FLOAT_TYPE* ds_0_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[0]).ptr();
			if (beacls::is_cuda(d_uvecs[0])) {
				for (size_t index = 0; index < x_uvecs[src_x_dim].size(); ++index) {
					const FLOAT_TYPE d0 = ds_0_ptr[index];
					dx_dim_ptr[index] = speed * cos_float_type(x_ptr[index]) + d0;
				}
			}
			else {
				const FLOAT_TYPE d0 = ds_0_ptr[0];
				for (size_t index = 0; index < x_uvecs[src_x_dim].size(); ++index) {
					dx_dim_ptr[index] = speed * cos_float_type(x_ptr[index]) + d0;
				}
			}
		}
			break;
		case 1:
		{
			beacls::reallocateAsSrc(dx_uvecs[dst_target_dim], x_uvecs[src_x_dim]);
			FLOAT_TYPE* dx_dim_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[dst_target_dim]).ptr();
			const FLOAT_TYPE* x_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[src_x_dim]).ptr();
			const FLOAT_TYPE* ds_1_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[1]).ptr();
			if (beacls::is_cuda(d_uvecs[1])) {
				for (size_t index = 0; index < x_uvecs[src_x_dim].size(); ++index) {
					const FLOAT_TYPE d1 = ds_1_ptr[index];
					dx_dim_ptr[index] = speed * sin_float_type(x_ptr[index]) + d1;
				}
			}
			else {
				const FLOAT_TYPE d1 = ds_1_ptr[0];
				for (size_t index = 0; index < x_uvecs[src_x_dim].size(); ++index) {
					dx_dim_ptr[index] = speed * sin_float_type(x_ptr[index]) + d1;
				}
			}
			break;
		}
		case 2:
		{
			beacls::reallocateAsSrc(dx_uvecs[dst_target_dim], u_uvecs[0]);
			FLOAT_TYPE* dx_dim_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[dst_target_dim]).ptr();
			const FLOAT_TYPE* us_0_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[0]).ptr();
			const FLOAT_TYPE* ds_2_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[2]).ptr();
			if (beacls::is_cuda(u_uvecs[0]) && beacls::is_cuda(d_uvecs[2])) {
				for (size_t index = 0; index < u_uvecs[0].size(); ++index) {
					const FLOAT_TYPE u0 = us_0_ptr[index];
					const FLOAT_TYPE d2 = ds_2_ptr[index];
					dx_dim_ptr[index] = u0 + d2;
				}
			}
			else if (!beacls::is_cuda(u_uvecs[0]) && !beacls::is_cuda(d_uvecs[2])) {
				const FLOAT_TYPE u0 = us_0_ptr[0];
				const FLOAT_TYPE d2 = ds_2_ptr[0];
				dx_dim_ptr[0] = u0 + d2;
			}
			else {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << " Invalid data size" << std::endl;
				result = false;
			}
		}
			break;
		default:
			std::cerr << "Only dimension 1-4 are defined for dynamics of DubinsCar!" << std::endl;
			result = false;
			break;
		}
		return result;
	}
};
#endif /* defined(USER_DEFINED_GPU_DYNSYS_FUNC) */
#endif /* !defined(WITH_GPU) */
