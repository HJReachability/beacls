// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_macro.hpp>
#include <typedef.hpp>
#include "TermLaxFriedrichs_cuda.hpp"
#include <Core/UVec.hpp>
#include <Core/CudaStream.hpp>

#if defined(WITH_GPU)

__global__
void kernel_TermLaxFriedrichs(
	FLOAT_TYPE  *g_dst_ydot_ptr,
	const FLOAT_TYPE *g_diss_ptr,
	const FLOAT_TYPE *g_ham_ptr,
	const size_t loop_length) {

	const size_t tid = threadIdx.x;
	size_t i = (blockIdx.x * blockDim.x + tid);
	const size_t gridSize = blockDim.x * gridDim.x;
	while (i < loop_length) {
		const FLOAT_TYPE diss = g_diss_ptr[i];
		const FLOAT_TYPE ham = g_ham_ptr[i];
		g_dst_ydot_ptr[i] = diss - ham;
		i += gridSize;
	}
}


void TermLaxFriedrichs_execute_cuda (
	beacls::UVec& ydot_uvec,
	const beacls::UVec& diss_uvec,
	const beacls::UVec& ham_uvec
) {
	FLOAT_TYPE* dst_ydot_ptr = beacls::UVec_<FLOAT_TYPE>(ydot_uvec).ptr();
	const FLOAT_TYPE* diss_ptr = beacls::UVec_<FLOAT_TYPE>(diss_uvec).ptr();
	const FLOAT_TYPE* ham_ptr = beacls::UVec_<FLOAT_TYPE>(ham_uvec).ptr();
//	beacls::synchronizeUVec(ham_uvec);
//	beacls::synchronizeUVec(diss_uvec);
	cudaStream_t ydot_stream = beacls::get_stream(ydot_uvec);
	const size_t loop_length = diss_uvec.size();
	size_t num_of_threads_x;
	size_t num_of_blocks_x;
	get_cuda_thread_size_1d<size_t>(
		num_of_threads_x,
		num_of_blocks_x,
		loop_length,
		512
		);
	dim3 num_of_blocks((unsigned int)num_of_blocks_x, 1);
	dim3 num_of_threads((unsigned int)num_of_threads_x, 1, 1);
	kernel_TermLaxFriedrichs<< <num_of_blocks, num_of_threads, 0, ydot_stream >> > (
		dst_ydot_ptr, diss_ptr, ham_ptr,
		loop_length
		);
}
#endif /* defined(WITH_GPU)  */
