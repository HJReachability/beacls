// CUDA runtime
#include <cuda_runtime.h>
#include <typedef.hpp>
#include <cuda_macro.hpp>
#include <thrust/device_ptr.h>
#include <thrust/tuple.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/system/cuda/execution_policy.h>
#include <Core/UVec.hpp>
#include <Core/CudaStream.hpp>
#include "ComputeGradients_OneSlice_cuda.hpp"

#if defined(WITH_GPU) 
struct Get_copyBackInfNan
{
	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple v) const
	{
		const FLOAT_TYPE original_data = thrust::get<0>(v);
		if (isnan(original_data) || isinf(original_data)) {
			thrust::get<1>(v) = original_data;
			thrust::get<2>(v) = original_data;
			thrust::get<3>(v) = original_data;
		}
	}
};

bool copyBackInfNan_cuda(
	beacls::UVec& deriv_c_uvec,
	beacls::UVec& deriv_l_uvec,
	beacls::UVec& deriv_r_uvec,
	const beacls::UVec& original_data_uvec
) {
	FLOAT_TYPE* derivC_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_c_uvec).ptr();
	FLOAT_TYPE* derivL_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_l_uvec).ptr();
	FLOAT_TYPE* derivR_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_r_uvec).ptr();
	const FLOAT_TYPE* original_data_ptr = beacls::UVec_<FLOAT_TYPE>(original_data_uvec).ptr();
	beacls::synchronizeUVec(deriv_c_uvec);
	beacls::synchronizeUVec(deriv_l_uvec);
	beacls::synchronizeUVec(deriv_r_uvec);
	beacls::synchronizeUVec(original_data_uvec);
	cudaStream_t original_data_stream = beacls::get_stream(original_data_uvec);
	deriv_c_uvec.set_cudaStream(original_data_uvec.get_cudaStream());
	deriv_r_uvec.set_cudaStream(original_data_uvec.get_cudaStream());
	deriv_l_uvec.set_cudaStream(original_data_uvec.get_cudaStream());
	const size_t length = deriv_c_uvec.size();
	thrust::device_ptr<FLOAT_TYPE> derivC_dev_ptr = thrust::device_pointer_cast(derivC_ptr);
	thrust::device_ptr<FLOAT_TYPE> derivL_dev_ptr = thrust::device_pointer_cast(derivL_ptr);
	thrust::device_ptr<FLOAT_TYPE> derivR_dev_ptr = thrust::device_pointer_cast(derivR_ptr);
	thrust::device_ptr<const FLOAT_TYPE> original_data_dev_ptr = thrust::device_pointer_cast(original_data_ptr);

	auto dst_src_Tuple = thrust::make_tuple(original_data_dev_ptr, derivC_dev_ptr, derivL_dev_ptr, derivR_dev_ptr);
	auto dst_src_Iterator = thrust::make_zip_iterator(dst_src_Tuple);
	thrust::for_each(thrust::cuda::par.on(original_data_stream),
		dst_src_Iterator, dst_src_Iterator + length, Get_copyBackInfNan());
	return true;
}

#endif /* defined(WITH_GPU) */
