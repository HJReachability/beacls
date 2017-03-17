#ifndef __Plane_cuda_hpp__
#define __Plane_cuda_hpp__

#include <typedef.hpp>
#include <cuda_macro.hpp>
#include <vector>
#include <helperOC/DynSys/DynSys/DynSysTypeDef.hpp>
#include <Core/UVec.hpp>

#if defined(USER_DEFINED_GPU_DYNSYS_FUNC)
namespace MyPlane_CUDA {
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
};
#endif /* defined(USER_DEFINED_GPU_DYNSYS_FUNC) */
#endif /*__Plane_cuda_hpp__*/
