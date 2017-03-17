#ifndef __PlaneCAvoid_cuda_hpp__
#define __PlaneCAvoid_cuda_hpp__

#include <typedef.hpp>
#include <vector>
#include <helperOC/DynSys/DynSys/DynSysTypeDef.hpp>
#include <cuda_macro.hpp>
#include <Core/UVec.hpp>

#if defined(USER_DEFINED_GPU_DYNSYS_FUNC)
namespace PlaneCAvoid_CUDA {
	bool optCtrl_execute_cuda(
		std::vector<beacls::UVec>& u_uvecs,
		const std::vector<beacls::UVec>& x_uvecs,
		const std::vector<beacls::UVec>& deriv_uvecs,
		const FLOAT_TYPE wMaxA,
		const beacls::FloatVec& vRangeA,
		const helperOC::DynSys_UMode_Type uMode
	);
	bool optDstb_execute_cuda(
		std::vector<beacls::UVec>& d_uvecs,
		const std::vector<beacls::UVec>& x_uvecs,
		const std::vector<beacls::UVec>& deriv_uvecs,
		const beacls::FloatVec& dMaxA,
		const beacls::FloatVec& dMaxB,
		const beacls::FloatVec& vRangeB,
		const FLOAT_TYPE wMaxB,
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
};
#endif /* defined(USER_DEFINED_GPU_DYNSYS_FUNC) */
#endif /*__PlaneCAvoid_cuda_hpp__*/
