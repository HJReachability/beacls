// CUDA runtime
#include <cuda_runtime.h>

#include <typedef.hpp>
#include <cuda_macro.hpp>
#include <thrust/device_ptr.h>
#include <thrust/tuple.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/system/cuda/execution_policy.h>
#include "MyPlane_cuda.hpp"
#include <vector>
#include <algorithm>
#include <helperOC/DynSys/DynSys/DynSys.hpp>
#include <Core/UVec.hpp>

#if defined(WITH_GPU) 
#if defined(USER_DEFINED_GPU_DYNSYS_FUNC)
namespace MyPlane_CUDA {

	struct Get_optCtrl_D {
	public:
			/* 
			To Be filled 
			*/
	};

	struct Get_optCtrl_dim0_d {
	public:
			/* 
			To Be filled 
			*/
	};

	bool optCtrl_execute_cuda(
		std::vector<beacls::UVec>& u_uvecs,
		const std::vector<beacls::UVec>& x_uvecs,
		const std::vector<beacls::UVec>& deriv_uvecs,
		const FLOAT_TYPE wMax,
		const FLOAT_TYPE vrange_min,
		const FLOAT_TYPE vrange_max,
		const helperOC::DynSys_UMode_Type uMode
	)
	{
		bool result = true;
			/* 
			To Be filled 
			*/
		return result;
	}

	struct Get_optDstb_dim0dim1dim2 {
	public:
			/* 
			To Be filled 
			*/
	};

	bool optDstb_execute_cuda(
		std::vector<beacls::UVec>& d_uvecs,
		const std::vector<beacls::UVec>& x_uvecs,
		const std::vector<beacls::UVec>& deriv_uvecs,
		const std::vector<FLOAT_TYPE>& dMax,
		const helperOC::DynSys_DMode_Type dMode
	)
	{
		bool result = true;
			/* 
			To Be filled 
			*/
		return result;
	}

	struct Get_dynamics_dim0_U0_D0 {
	public:
			/* 
			To Be filled 
			*/
	};
	struct Get_dynamics_dim0_U0_d0 {
	public:
			/* 
			To Be filled 
			*/
	};
	struct Get_dynamics_dim1_U0_D1 {
	public:
			/* 
			To Be filled 
			*/
	};
	struct Get_dynamics_dim1_U0_d1 {
	public:
			/* 
			To Be filled 
			*/
	};

	bool dynamics_cell_helper_execute_cuda(
		beacls::UVec& dx_uvec,
		const std::vector<beacls::UVec>& x_uvecs,
		const std::vector<beacls::UVec>& u_uvecs,
		const std::vector<beacls::UVec>& d_uvecs,
		const size_t dim
	) {
		bool result = true;
			/* 
			To Be filled 
			*/
		return result;
	}
};
#endif /* defined(USER_DEFINED_GPU_DYNSYS_FUNC) */
#endif /* defined(WITH_GPU) */
