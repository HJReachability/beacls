// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_macro.hpp>
#include <typedef.hpp>
#include "HJI_Grid_cuda.hpp"
#include <Core/UVec.hpp>
#include <Core/CudaStream.hpp>

#if defined(WITH_GPU)

__global__
void kernel_HJI_Grid_calc_xs(
	FLOAT_TYPE  *g_xs_ptr,
	const FLOAT_TYPE *g_vs_ptr,
	const size_t start,
	const size_t loop_length,
	const size_t stride
	) {

	const size_t tid = threadIdx.x;
	size_t i = (blockIdx.x * blockDim.x + tid);
	const size_t gridSize = blockDim.x * gridDim.x;
	while (i < loop_length) {
		i += gridSize;
	}
}


void HJI_Grid_calc_xs_execute_cuda
(
	beacls::UVec& x_uvec,
	const beacls::UVec& v_uvec,
	const size_t dimension,
	const size_t start_index,
	const size_t loop_length,
	const size_t stride
) {
	beacls::synchronizeUVec(v_uvec);
	cudaStream_t x_stream = beacls::get_stream(x_uvec);
	FLOAT_TYPE* xs_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvec).ptr();
	const FLOAT_TYPE* vs_ptr = beacls::UVec_<const FLOAT_TYPE>(v_uvec).ptr();
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
	kernel_HJI_Grid_calc_xs <<<num_of_blocks, num_of_threads, 0, x_stream >> > (
		xs_ptr, vs_ptr,
		start_index, loop_length, stride
		);
}
#endif /* defined(WITH_GPU)  */
