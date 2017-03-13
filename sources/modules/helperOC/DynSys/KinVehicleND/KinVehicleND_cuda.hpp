#ifndef __KinVehicleND_cuda_hpp__
#define __KinVehicleND_cuda_hpp__

#include <typedef.hpp>
#include <cuda_macro.hpp>
#include <vector>
#include <helperOC/DynSys/DynSys/DynSysTypeDef.hpp>
#include <Core/UVec.hpp>
#if defined(USER_DEFINED_GPU_DYNSYS_FUNC)
namespace KinVehicleND_CUDA {
	bool optCtrl_execute_cuda(
		std::vector<beacls::UVec>& u_uvecs,
		const std::vector<beacls::UVec>& deriv_uvecs,
		const FLOAT_TYPE vMax,
		const DynSys_UMode_Type uMode
	);
};
#endif /* defined(USER_DEFINED_GPU_DYNSYS_FUNC) */
#endif /*__KinVehicleND_cuda_hpp__*/
