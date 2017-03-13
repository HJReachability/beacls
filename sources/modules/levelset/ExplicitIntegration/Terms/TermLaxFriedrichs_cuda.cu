// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_macro.hpp>
#include <typedef.hpp>
#include <thrust/transform.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h>
#include "TermLaxFriedrichs_cuda.hpp"
#include <Core/UVec.hpp>
#include <Core/CudaStream.hpp>

#if defined(WITH_GPU)

void TermLaxFriedrichs_execute_cuda (
	beacls::UVec& ydot_uvec,
	const beacls::UVec& diss_uvec,
	const beacls::UVec& ham_uvec
) {
	FLOAT_TYPE* dst_ydot_ptr = beacls::UVec_<FLOAT_TYPE>(ydot_uvec).ptr();
	const FLOAT_TYPE* diss_ptr = beacls::UVec_<FLOAT_TYPE>(diss_uvec).ptr();
	const FLOAT_TYPE* ham_ptr = beacls::UVec_<FLOAT_TYPE>(ham_uvec).ptr();
	beacls::synchronizeUVec(ham_uvec);
	beacls::synchronizeUVec(diss_uvec);
	thrust::device_ptr<const FLOAT_TYPE> diss_dev_ptr = thrust::device_pointer_cast(diss_ptr);
	thrust::device_ptr<const FLOAT_TYPE> ham_dev_ptr = thrust::device_pointer_cast(ham_ptr);
	thrust::device_ptr<FLOAT_TYPE> dst_ydot_dev_ptr = thrust::device_pointer_cast(dst_ydot_ptr);
	cudaStream_t ydot_stream = beacls::get_stream(ydot_uvec);
	
	thrust::transform(thrust::cuda::par.on(ydot_stream),
		diss_dev_ptr, diss_dev_ptr + diss_uvec.size(), ham_dev_ptr, dst_ydot_dev_ptr,
		thrust::minus<FLOAT_TYPE>());
}
#endif /* defined(WITH_GPU)  */
