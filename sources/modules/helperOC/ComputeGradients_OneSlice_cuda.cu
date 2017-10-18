// CUDA runtime
#include <cuda_runtime.h>
#include <typedef.hpp>
#include <cuda_macro.hpp>
#include <Core/UVec.hpp>
#include <Core/CudaStream.hpp>
#include "ComputeGradients_OneSlice_cuda.hpp"

#if defined(WITH_GPU) 
__global__ static
void kernel_copyBackInfNan(
	FLOAT_TYPE* g_dst_c_ptr,
	FLOAT_TYPE* g_dst_l_ptr,
	FLOAT_TYPE* g_dst_r_ptr,
	const FLOAT_TYPE* g_src_ptr,
	const size_t loop_length
) {
	const size_t tid = threadIdx.x;
	size_t index = (blockIdx.x * blockDim.x + tid);
	const size_t gridSize = blockDim.x * gridDim.x;
	while (index < loop_length) {
		const FLOAT_TYPE original_data = g_src_ptr[index];
		if (isnan(original_data) || isinf(original_data)) {
			g_dst_c_ptr[index] = original_data;
			g_dst_l_ptr[index] = original_data;
			g_dst_r_ptr[index] = original_data;
		}
		index += gridSize;
	}
}

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
	beacls::synchronizeUVec(original_data_uvec);
	const size_t loop_length = deriv_c_uvec.size();
	size_t num_of_threads_x;
	size_t num_of_blocks_x;
	get_cuda_thread_size_1d<size_t>(
		num_of_threads_x,
		num_of_blocks_x,
		loop_length
		);
	dim3 num_of_blocks(num_of_blocks_x, 1);
	dim3 num_of_threads(num_of_threads_x, 1, 1);
	cudaStream_t deriv_c_stream = beacls::get_stream(deriv_c_uvec);
	kernel_copyBackInfNan << <num_of_blocks, num_of_threads, 0, deriv_c_stream >> >(
		derivC_ptr, derivL_ptr, derivR_ptr, original_data_ptr,
		loop_length
		);
	return true;
}

#endif /* defined(WITH_GPU) */
