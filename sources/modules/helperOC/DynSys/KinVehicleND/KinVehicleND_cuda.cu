// CUDA runtime
#include <cuda_runtime.h>

#include <typedef.hpp>
#include <cuda_macro.hpp>
#include <thrust/device_ptr.h>
#include <thrust/tuple.h>
#include <thrust/pair.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/system/cuda/execution_policy.h>
#include "KinVehicleND_cuda.hpp"
#include <vector>
#include <algorithm>
#include <helperOC/DynSys/DynSys/DynSys.hpp>
#include <Core/UVec.hpp>
#include <Core/CudaStream.hpp>


#if defined(WITH_GPU) 
#if defined(USER_DEFINED_GPU_DYNSYS_FUNC)
namespace KinVehicleND_CUDA {
template<typename T>
struct Get_opt {
public:
	T vMax;
	Get_opt(const T vMax) : vMax(vMax) {}
	__host__ __device__
	T operator()(const T& deriv, const T& denom) const
	{
		return (denom == 0) ? 0 : vMax * deriv / denom;
	}
};

template<typename T>
struct Get_NormAccumulate {
public:
	__host__ __device__
	T operator()(const T& lhs, const T& rhs) const
	{
		return rhs + lhs * lhs;
	}
};
template<typename T>
struct Get_sqrt {
public:
	__host__ __device__
	T operator()(const T& a) const
	{
		return sqrt_float_type<T>(a);
	}
};

bool optCtrl_execute_cuda(
	std::vector<beacls::UVec>& u_uvecs,
	const std::vector<beacls::UVec>& deriv_uvecs,
	const FLOAT_TYPE vMax,
	const helperOC::DynSys_UMode_Type uMode
)
{
	bool result = true;
	const size_t length = deriv_uvecs[0].size();
	std::for_each(u_uvecs.begin(), u_uvecs.end(), [&deriv_uvecs](beacls::UVec& rhs) {
		beacls::reallocateAsSrc(rhs, deriv_uvecs[0]);
	});
	const size_t nu = deriv_uvecs.size();
	if (beacls::is_cuda(deriv_uvecs[0])) {
		cudaStream_t u_stream = beacls::get_stream(u_uvecs[0]);
		beacls::CudaStream* cudaStream = u_uvecs[0].get_cudaStream();
		std::for_each(u_uvecs.begin()+1, u_uvecs.end(), [cudaStream](beacls::UVec& rhs) {
			rhs.set_cudaStream(cudaStream);
		});
		FLOAT_TYPE* last_uOpt_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[nu - 1]).ptr();
		thrust::device_ptr<FLOAT_TYPE> last_uOpt_dev_ptr = thrust::device_pointer_cast(last_uOpt_ptr);
		if ((uMode == helperOC::DynSys_UMode_Max) || (uMode == helperOC::DynSys_UMode_Min)) {
			const FLOAT_TYPE moded_vMax = (uMode == helperOC::DynSys_UMode_Max) ? vMax : -vMax;
			//!< store denom to uOpts[nu-1]
			const FLOAT_TYPE* deriv0 = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[0]).ptr();
			thrust::device_ptr<const FLOAT_TYPE> deriv0_dev_ptr = thrust::device_pointer_cast(deriv0);
			thrust::transform(thrust::cuda::par.on(u_stream), 
				deriv0_dev_ptr, deriv0_dev_ptr + length, deriv0_dev_ptr, last_uOpt_dev_ptr, thrust::multiplies<FLOAT_TYPE>());
			for (size_t dim = 1; dim < nu; ++dim) {
				const FLOAT_TYPE* deriv_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[dim]).ptr();
				thrust::device_ptr<const FLOAT_TYPE> deriv_dev_ptr = thrust::device_pointer_cast(deriv_ptr);
				thrust::transform(thrust::cuda::par.on(u_stream), 
					deriv_dev_ptr, deriv_dev_ptr + length, last_uOpt_dev_ptr, last_uOpt_dev_ptr, Get_NormAccumulate<FLOAT_TYPE>());
			}
			thrust::transform(last_uOpt_dev_ptr, last_uOpt_dev_ptr + length, last_uOpt_dev_ptr, Get_sqrt<FLOAT_TYPE>());
			for (size_t dim = 1; dim < nu; ++dim) {
				const FLOAT_TYPE* deriv_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[dim]).ptr();
				FLOAT_TYPE* uOpt_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[dim]).ptr();
				thrust::device_ptr<const FLOAT_TYPE> deriv_dev_ptr = thrust::device_pointer_cast(deriv_ptr);
				thrust::device_ptr<FLOAT_TYPE> uOpt_dev_ptr = thrust::device_pointer_cast(uOpt_ptr);
				thrust::transform(thrust::cuda::par.on(u_stream), 
					deriv_dev_ptr, deriv_dev_ptr + length, last_uOpt_dev_ptr, uOpt_dev_ptr, Get_opt<FLOAT_TYPE>(moded_vMax));
			}
		} else {
			printf("Unknown uMode!: %d\n", uMode);
			result = false;
		}
	} 
	else {
		if ((uMode == helperOC::DynSys_UMode_Max) || (uMode == helperOC::DynSys_UMode_Min)) {
			const FLOAT_TYPE moded_vMax = (uMode == helperOC::DynSys_UMode_Max) ? vMax : -vMax;
			//!< store denom to uOpts[nu-1]
			const FLOAT_TYPE* deriv0 = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[0]).ptr();
			const FLOAT_TYPE d0 = deriv0[0];
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
				const FLOAT_TYPE d =  deriv_ptr[0];
				const FLOAT_TYPE val = (denom == 0) ? 0 : moded_vMax * d / denom;
				uOpt_ptr[0] = val;
			}
		} else {
			printf("Unknown uMode!: %d\n", uMode);
			result = false;
		}
	}
	return result;
}
};
#endif /* defined(USER_DEFINED_GPU_DYNSYS_FUNC) */
#endif /* defined(WITH_GPU) */
