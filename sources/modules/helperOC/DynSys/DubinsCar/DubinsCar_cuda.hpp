#ifndef __DubinsCar_cuda_hpp__
#define __DubinsCar_cuda_hpp__

#include <typedef.hpp>
#include <cuda_macro.hpp>
#include <vector>
#include <helperOC/DynSys/DynSys/DynSysTypeDef.hpp>
#include <Core/UVec.hpp>
#if defined(USER_DEFINED_GPU_DYNSYS_FUNC)
namespace DubinsCar_CUDA {
	bool optCtrl_execute_cuda(
		beacls::UVec& u_uvec,
		const beacls::UVec& deriv_uvec,
		const FLOAT_TYPE wMax,
		const DynSys_UMode_Type uMode
	);
	bool optDstb_execute_cuda(
		std::vector<beacls::UVec>& d_uvecs,
		const beacls::UVec& deriv_uvec,
		const beacls::FloatVec& dMax,
		const DynSys_DMode_Type dMode,
		const beacls::IntegerVec& dims
	);
	bool dynamics_cell_helper_execute_cuda_dimAll(
		std::vector<beacls::UVec>& dx_uvecs,
		const std::vector<beacls::UVec>& x_uvecs,
		const std::vector<beacls::UVec>& d_uvecs,
		const FLOAT_TYPE speed,
		const size_t src_dim0_dst_dx_index,
		const size_t src_dim1_dst_dx_index,
		const size_t src_x_dim
	);
	bool dynamics_cell_helper_execute_cuda(
		std::vector<beacls::UVec>& dx_uvecs,
		const std::vector<beacls::UVec>& x_uvecs,
		const std::vector<beacls::UVec>& u_uvecs,
		const std::vector<beacls::UVec>& d_uvecs,
		const FLOAT_TYPE speed,
		const size_t dst_target_dim,
		const size_t src_target_dim,
		const size_t src_x_dim
	);
};
#endif /* defined(USER_DEFINED_GPU_DYNSYS_FUNC) */
#endif /*__DubinsCar_cuda_hpp__*/
