// CUDA runtime
#include <cuda_runtime.h>

#include <typedef.hpp>
#include <cuda_macro.hpp>
#include <Core/UVec.hpp>
#include <Core/CudaStream.hpp>
#include "UpwindFirstWENO5a_cuda.hpp"

#if defined(WITH_GPU)
__global__ static
void kernel_dim0_EpsilonCalculationMethod_Constant2(
	FLOAT_TYPE* dst_deriv_l_ptr,
	FLOAT_TYPE* dst_deriv_r_ptr,
	const FLOAT_TYPE* boundedSrc_base_ptr,
	const FLOAT_TYPE dxInv,
	const FLOAT_TYPE dxInv_2,
	const FLOAT_TYPE dxInv_3,
	const FLOAT_TYPE dx,
	const FLOAT_TYPE x2_dx_square,
	const FLOAT_TYPE dx_square,
	const FLOAT_TYPE weightL0,
	const FLOAT_TYPE weightL1,
	const FLOAT_TYPE weightL2,
	const FLOAT_TYPE weightR0,
	const FLOAT_TYPE weightR1,
	const FLOAT_TYPE weightR2,
	const size_t num_of_slices,
	const size_t loop_length,
	const size_t src_target_dimension_loop_size,
	const size_t first_dimension_loop_size,
	const size_t slice_length,
	const size_t stencil,
//	const size_t thread_length_z,
	const size_t thread_length_y,
	const size_t thread_length_x
) {
	kernel_dim0_EpsilonCalculationMethod_inline2(
		dst_deriv_l_ptr, dst_deriv_r_ptr,
		boundedSrc_base_ptr, dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
		weightL0, weightL1, weightL2, weightR0, weightR1, weightR2,
		num_of_slices, loop_length,
		src_target_dimension_loop_size, first_dimension_loop_size, slice_length,
		stencil,
#if 0
		thread_length_z, thread_length_y, thread_length_x,
		0, blockIdx.y, blockIdx.x,
		blockDim.z, blockDim.y, blockDim.x,
		threadIdx.z, threadIdx.y, threadIdx.x,
#else
		thread_length_y, thread_length_x,
		blockIdx.y, blockIdx.x,
		blockDim.y, blockDim.x,
		threadIdx.y, threadIdx.x,
#endif
		ECM_Constant_Cuda());
}

__global__ static
void kernel_dim0_EpsilonCalculationMethod_maxOverNeighbor2(
	FLOAT_TYPE* dst_deriv_l_ptr,
	FLOAT_TYPE* dst_deriv_r_ptr,
	const FLOAT_TYPE* boundedSrc_base_ptr,
	const FLOAT_TYPE dxInv,
	const FLOAT_TYPE dxInv_2,
	const FLOAT_TYPE dxInv_3,
	const FLOAT_TYPE dx,
	const FLOAT_TYPE x2_dx_square,
	const FLOAT_TYPE dx_square,
	const FLOAT_TYPE weightL0,
	const FLOAT_TYPE weightL1,
	const FLOAT_TYPE weightL2,
	const FLOAT_TYPE weightR0,
	const FLOAT_TYPE weightR1,
	const FLOAT_TYPE weightR2,
	const size_t num_of_slices,
	const size_t loop_length,
	const size_t src_target_dimension_loop_size,
	const size_t first_dimension_loop_size,
	const size_t slice_length,
	const size_t stencil,
//	const size_t thread_length_z,
	const size_t thread_length_y,
	const size_t thread_length_x
) {
	kernel_dim0_EpsilonCalculationMethod_inline2(
		dst_deriv_l_ptr, dst_deriv_r_ptr,
		boundedSrc_base_ptr, dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
		weightL0, weightL1, weightL2, weightR0, weightR1, weightR2,
		num_of_slices, loop_length, 
		src_target_dimension_loop_size, first_dimension_loop_size, slice_length,
		stencil,
#if 0
		thread_length_z, thread_length_y, thread_length_x,
		0, blockIdx.y, blockIdx.x,
		blockDim.z, blockDim.y, blockDim.x,
		threadIdx.z, threadIdx.y, threadIdx.x,
#else
		thread_length_y, thread_length_x,
		blockIdx.y, blockIdx.x,
		blockDim.y, blockDim.x,
		threadIdx.y, threadIdx.x,
#endif
		ECM_MaxOverNeighbor_Cuda());
}

void UpwindFirstWENO5a_execute_dim0_cuda2 (
	FLOAT_TYPE* dst_deriv_l_ptr,
	FLOAT_TYPE* dst_deriv_r_ptr,
	const FLOAT_TYPE* boundedSrc_base_ptr,
	const FLOAT_TYPE dxInv,
	const FLOAT_TYPE dxInv_2,
	const FLOAT_TYPE dxInv_3,
	const FLOAT_TYPE dx,
	const FLOAT_TYPE x2_dx_square,
	const FLOAT_TYPE dx_square,
	const FLOAT_TYPE weightL0,
	const FLOAT_TYPE weightL1,
	const FLOAT_TYPE weightL2,
	const FLOAT_TYPE weightR0,
	const FLOAT_TYPE weightR1,
	const FLOAT_TYPE weightR2,
	const size_t num_of_slices,
	const size_t loop_length,
	const size_t src_target_dimension_loop_size,
	const size_t first_dimension_loop_size,
	const size_t slice_length,
	const size_t stencil,
	const levelset::EpsilonCalculationMethod_Type epsilonCalculationMethod_Type,
	beacls::CudaStream* cudaStream
) {
	size_t num_of_threads_z;
	size_t thread_length_z;
	size_t num_of_threads_y;
	size_t num_of_blocks_y;
	size_t thread_length_y;
	size_t num_of_threads_x;
	size_t num_of_blocks_x;
	size_t thread_length_x;
#if 1
	get_cuda_thread_size<size_t>(
		thread_length_z, thread_length_y, thread_length_x,
		num_of_threads_z, num_of_threads_y, num_of_threads_x,
		num_of_blocks_y, num_of_blocks_x,
		1, loop_length * num_of_slices, first_dimension_loop_size,
		1, 1, 4, 512
		);
#else
	get_cuda_thread_size<size_t>(
		thread_length_z, thread_length_y, thread_length_x,
		num_of_threads_z, num_of_threads_y, num_of_threads_x,
		num_of_blocks_y, num_of_blocks_x,
		first_dimension_loop_size, loop_length, num_of_slices,
		4, 1, 1, max_num_of_threads
		);
#endif
	dim3 num_of_blocks((unsigned int)num_of_blocks_x, (unsigned int)num_of_blocks_y);
	dim3 num_of_threads((unsigned int)num_of_threads_x, (unsigned int)num_of_threads_y, (unsigned int)num_of_threads_z);

	cudaStream_t stream = cudaStream->get_stream();
	switch (epsilonCalculationMethod_Type) {
	case levelset::EpsilonCalculationMethod_Invalid:
	default:
//		printf("Unknown epsilonCalculationMethod %d\n", epsilonCalculationMethod_Type);
		return;
	case levelset::EpsilonCalculationMethod_Constant:
		kernel_dim0_EpsilonCalculationMethod_Constant2<<<num_of_blocks,num_of_threads, 0, stream>>>(
			dst_deriv_l_ptr, dst_deriv_r_ptr,
			boundedSrc_base_ptr, dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
			weightL0, weightL1, weightL2, weightR0, weightR1, weightR2,
			1, num_of_slices*loop_length,
			src_target_dimension_loop_size, first_dimension_loop_size, slice_length,
			stencil,
#if 0
			thread_length_z, thread_length_y, thread_length_x
#else
			thread_length_y, thread_length_x
#endif
			);
		CUDA_CHECK(cudaPeekAtLastError());
		break;
	case levelset::EpsilonCalculationMethod_maxOverGrid:
//		printf("epsilonCalculationMethod %d is not supported yet\n", epsilonCalculationMethod_Type);
		return;
	case levelset::EpsilonCalculationMethod_maxOverNeighbor:
		kernel_dim0_EpsilonCalculationMethod_maxOverNeighbor2<<<num_of_blocks,num_of_threads, 0, stream>>>(
			dst_deriv_l_ptr, dst_deriv_r_ptr,
			boundedSrc_base_ptr, dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
			weightL0, weightL1, weightL2, weightR0, weightR1, weightR2,
			1, num_of_slices*loop_length,
			src_target_dimension_loop_size, first_dimension_loop_size, slice_length,
			stencil,
#if 0
			thread_length_z, thread_length_y, thread_length_x
#else
			thread_length_y, thread_length_x
#endif
			);
		CUDA_CHECK(cudaPeekAtLastError());
		break;
	}
}
__global__ static 
void kernel_dim1_EpsilonCalculationMethod_Constant2(
	FLOAT_TYPE* dst_deriv_l_ptr,
	FLOAT_TYPE* dst_deriv_r_ptr,
	const FLOAT_TYPE* tmpBoundedSrc_ptr,
	const FLOAT_TYPE dxInv,
	const FLOAT_TYPE dxInv_2,
	const FLOAT_TYPE dxInv_3,
	const FLOAT_TYPE dx,
	const FLOAT_TYPE x2_dx_square,
	const FLOAT_TYPE dx_square,
	const FLOAT_TYPE weightL0,
	const FLOAT_TYPE weightL1,
	const FLOAT_TYPE weightL2,
	const FLOAT_TYPE weightR0,
	const FLOAT_TYPE weightR1,
	const FLOAT_TYPE weightR2,
	const size_t num_of_slices,
	const size_t loop_length,
	const size_t first_dimension_loop_size,
	const size_t slice_length,
	const size_t stencil,
	const size_t thread_length_z,
	const size_t thread_length_y,
	const size_t thread_length_x
) {
	kernel_dim1_EpsilonCalculationMethod_inline2(
		dst_deriv_l_ptr, dst_deriv_r_ptr,
		tmpBoundedSrc_ptr, dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
		weightL0, weightL1, weightL2, weightR0, weightR1, weightR2,
		num_of_slices, loop_length, first_dimension_loop_size, slice_length,
		stencil,
		thread_length_z, thread_length_y, thread_length_x,
		0, blockIdx.y, blockIdx.x,
		blockDim.z, blockDim.y, blockDim.x,
		threadIdx.z, threadIdx.y, threadIdx.x,
		ECM_Constant_Cuda());
}

__global__ static
void kernel_dim1_EpsilonCalculationMethod_maxOverNeighbor2(
	FLOAT_TYPE* dst_deriv_l_ptr,
	FLOAT_TYPE* dst_deriv_r_ptr,
	const FLOAT_TYPE* tmpBoundedSrc_ptr,
	const FLOAT_TYPE dxInv,
	const FLOAT_TYPE dxInv_2,
	const FLOAT_TYPE dxInv_3,
	const FLOAT_TYPE dx,
	const FLOAT_TYPE x2_dx_square,
	const FLOAT_TYPE dx_square,
	const FLOAT_TYPE weightL0,
	const FLOAT_TYPE weightL1,
	const FLOAT_TYPE weightL2,
	const FLOAT_TYPE weightR0,
	const FLOAT_TYPE weightR1,
	const FLOAT_TYPE weightR2,
	const size_t num_of_slices,
	const size_t loop_length,
	const size_t first_dimension_loop_size,
	const size_t slice_length,
	const size_t stencil,
	const size_t thread_length_z,
	const size_t thread_length_y,
	const size_t thread_length_x
) {
	kernel_dim1_EpsilonCalculationMethod_inline2(
		dst_deriv_l_ptr, dst_deriv_r_ptr,
		tmpBoundedSrc_ptr, dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
		weightL0, weightL1, weightL2, weightR0, weightR1, weightR2,
		num_of_slices, loop_length, first_dimension_loop_size, slice_length,
		stencil,
		thread_length_z, thread_length_y, thread_length_x,
		0, blockIdx.y, blockIdx.x, 
		blockDim.z, blockDim.y, blockDim.x,
		threadIdx.z, threadIdx.y, threadIdx.x,
		ECM_MaxOverNeighbor_Cuda());
}

void UpwindFirstWENO5a_execute_dim1_cuda2 (
	FLOAT_TYPE* dst_deriv_l_ptr,
	FLOAT_TYPE* dst_deriv_r_ptr,
	const FLOAT_TYPE* tmpBoundedSrc_ptr,
	const FLOAT_TYPE dxInv,
	const FLOAT_TYPE dxInv_2,
	const FLOAT_TYPE dxInv_3,
	const FLOAT_TYPE dx,
	const FLOAT_TYPE x2_dx_square,
	const FLOAT_TYPE dx_square,
	const FLOAT_TYPE weightL0,
	const FLOAT_TYPE weightL1,
	const FLOAT_TYPE weightL2,
	const FLOAT_TYPE weightR0,
	const FLOAT_TYPE weightR1,
	const FLOAT_TYPE weightR2,
	const size_t num_of_slices,
	const size_t loop_length,
	const size_t first_dimension_loop_size,
	const size_t slice_length,
	const size_t stencil,
	const levelset::EpsilonCalculationMethod_Type epsilonCalculationMethod_Type,
	beacls::CudaStream* cudaStream
) {
	size_t num_of_threads_z;
	size_t thread_length_z;
	size_t num_of_threads_y;
	size_t num_of_blocks_y;
	size_t thread_length_y;
	size_t num_of_threads_x;
	size_t num_of_blocks_x;
	size_t thread_length_x;
	get_cuda_thread_size<size_t>(
		thread_length_z, thread_length_y, thread_length_x,
		num_of_threads_z, num_of_threads_y, num_of_threads_x,
		num_of_blocks_y, num_of_blocks_x,
		first_dimension_loop_size, loop_length, num_of_slices,
		1, 8, 1, 512
		);
	dim3 num_of_blocks((unsigned int)num_of_blocks_x, (unsigned int)num_of_blocks_y);
	dim3 num_of_threads((unsigned int)num_of_threads_x, (unsigned int)num_of_threads_y, (unsigned int)num_of_threads_z);
	cudaStream_t stream = cudaStream->get_stream();

	switch (epsilonCalculationMethod_Type) {
	case levelset::EpsilonCalculationMethod_Invalid:
	default:
//		printf("Unknown epsilonCalculationMethod %d\n", epsilonCalculationMethod_Type);
		return;
	case levelset::EpsilonCalculationMethod_Constant:
		kernel_dim1_EpsilonCalculationMethod_Constant2<<<num_of_blocks,num_of_threads, 0, stream>>>(
		dst_deriv_l_ptr, dst_deriv_r_ptr,
		tmpBoundedSrc_ptr, dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
		weightL0, weightL1, weightL2, weightR0, weightR1, weightR2,
		num_of_slices, loop_length, first_dimension_loop_size, slice_length,
		stencil,
		thread_length_z, thread_length_y, thread_length_x
		);
		CUDA_CHECK(cudaPeekAtLastError());
		break;
	case levelset::EpsilonCalculationMethod_maxOverGrid:
//		printf("epsilonCalculationMethod %d is not supported yet\n", epsilonCalculationMethod_Type);
		return;
	case levelset::EpsilonCalculationMethod_maxOverNeighbor:
		kernel_dim1_EpsilonCalculationMethod_maxOverNeighbor2<<<num_of_blocks,num_of_threads, 0, stream>>>(
		dst_deriv_l_ptr, dst_deriv_r_ptr,
		tmpBoundedSrc_ptr, dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
		weightL0, weightL1, weightL2, weightR0, weightR1, weightR2,
		num_of_slices, loop_length, first_dimension_loop_size, slice_length,
		stencil,
		thread_length_z, thread_length_y, thread_length_x
		);
		CUDA_CHECK(cudaPeekAtLastError());
		break;
	}
}
__global__ static 
void kernel_dimLET2_EpsilonCalculationMethod_Constant2(
	FLOAT_TYPE* dst_deriv_l_ptr,
	FLOAT_TYPE* dst_deriv_r_ptr,
	const FLOAT_TYPE* tmpBoundedSrc_ptr,
const FLOAT_TYPE dxInv,
	const FLOAT_TYPE dxInv_2,
	const FLOAT_TYPE dxInv_3,
	const FLOAT_TYPE dx,
	const FLOAT_TYPE x2_dx_square,
	const FLOAT_TYPE dx_square,
	const FLOAT_TYPE weightL0,
	const FLOAT_TYPE weightL1,
	const FLOAT_TYPE weightL2,
	const FLOAT_TYPE weightR0,
	const FLOAT_TYPE weightR1,
	const FLOAT_TYPE weightR2,
	const size_t num_of_slices,
	const size_t loop_length,
	const size_t first_dimension_loop_size,
	const size_t stride_distance,
	const size_t slice_length,
///	const size_t thread_length_z,
	const size_t thread_length_y,
	const size_t thread_length_x
) {
	kernel_dimLET2_EpsilonCalculationMethod_inline2(
		dst_deriv_l_ptr, dst_deriv_r_ptr,
		tmpBoundedSrc_ptr, 
		dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
		weightL0, weightL1, weightL2, weightR0, weightR1, weightR2,
		num_of_slices, loop_length, first_dimension_loop_size,
		stride_distance,
		slice_length,
		thread_length_y, thread_length_x,
		blockIdx.y, blockIdx.x, 
		blockDim.y, blockDim.x,
		threadIdx.y, threadIdx.x,
		ECM_Constant_Cuda());
}

__global__ static
void kernel_dimLET2_EpsilonCalculationMethod_maxOverNeighbor2(
	FLOAT_TYPE* dst_deriv_l_ptr,
	FLOAT_TYPE* dst_deriv_r_ptr,
	const FLOAT_TYPE* tmpBoundedSrc_ptr,
	const FLOAT_TYPE dxInv,
	const FLOAT_TYPE dxInv_2,
	const FLOAT_TYPE dxInv_3,
	const FLOAT_TYPE dx,
	const FLOAT_TYPE x2_dx_square,
	const FLOAT_TYPE dx_square,
	const FLOAT_TYPE weightL0,
	const FLOAT_TYPE weightL1,
	const FLOAT_TYPE weightL2,
	const FLOAT_TYPE weightR0,
	const FLOAT_TYPE weightR1,
	const FLOAT_TYPE weightR2,
	const size_t num_of_slices,
	const size_t loop_length,
	const size_t first_dimension_loop_size,
	const size_t stride_distance,
	const size_t slice_length,
//	const size_t thread_length_z,
	const size_t thread_length_y,
	const size_t thread_length_x
) {
	kernel_dimLET2_EpsilonCalculationMethod_inline2(
		dst_deriv_l_ptr, dst_deriv_r_ptr,
		tmpBoundedSrc_ptr,
		dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
		weightL0, weightL1, weightL2, weightR0, weightR1, weightR2,
		num_of_slices, loop_length, first_dimension_loop_size,
		stride_distance,
		slice_length,
		thread_length_y, thread_length_x,
		blockIdx.y, blockIdx.x,
		blockDim.y, blockDim.x,
		threadIdx.y, threadIdx.x,
		ECM_MaxOverNeighbor_Cuda());
}


void UpwindFirstWENO5a_execute_dimLET2_cuda2 (
	FLOAT_TYPE* dst_deriv_l_ptr,
	FLOAT_TYPE* dst_deriv_r_ptr,
	const FLOAT_TYPE* tmpBoundedSrc_ptr,
	const FLOAT_TYPE dxInv,
	const FLOAT_TYPE dxInv_2,
	const FLOAT_TYPE dxInv_3,
	const FLOAT_TYPE dx,
	const FLOAT_TYPE x2_dx_square,
	const FLOAT_TYPE dx_square,
	const FLOAT_TYPE weightL0,
	const FLOAT_TYPE weightL1,
	const FLOAT_TYPE weightL2,
	const FLOAT_TYPE weightR0,
	const FLOAT_TYPE weightR1,
	const FLOAT_TYPE weightR2,
	const size_t num_of_slices,
	const size_t loop_length,
	const size_t first_dimension_loop_size,
	const size_t stride_distance,
	const size_t slice_length,
	const levelset::EpsilonCalculationMethod_Type epsilonCalculationMethod_Type,
	beacls::CudaStream* cudaStream
) {
	size_t num_of_threads_z;
	size_t thread_length_z;
	size_t num_of_threads_y;
	size_t num_of_blocks_y;
	size_t thread_length_y;
	size_t num_of_threads_x;
	size_t num_of_blocks_x;
	size_t thread_length_x;
	get_cuda_thread_size<size_t>(
		thread_length_z, thread_length_y, thread_length_x,
		num_of_threads_z, num_of_threads_y, num_of_threads_x,
		num_of_blocks_y, num_of_blocks_x,
		1, num_of_slices, loop_length * first_dimension_loop_size,
		1, 1, 1, 512
		);

	dim3 num_of_blocks((unsigned int)num_of_blocks_x, (unsigned int)num_of_blocks_y);
	dim3 num_of_threads((unsigned int)num_of_threads_x, (unsigned int)num_of_threads_y, (unsigned int)num_of_threads_z);

	cudaStream_t stream = cudaStream->get_stream();
	switch (epsilonCalculationMethod_Type) {
	case levelset::EpsilonCalculationMethod_Invalid:
	default:
//		printf("Unknown epsilonCalculationMethod %d\n", epsilonCalculationMethod_Type);
		return;
	case levelset::EpsilonCalculationMethod_Constant:
		kernel_dimLET2_EpsilonCalculationMethod_Constant2<<<num_of_blocks,num_of_threads, 0, stream>>>(
		dst_deriv_l_ptr, dst_deriv_r_ptr,
		tmpBoundedSrc_ptr,
		dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
		weightL0, weightL1, weightL2, weightR0, weightR1, weightR2,
		1, num_of_slices, loop_length * first_dimension_loop_size,
		stride_distance,
		slice_length,
		thread_length_y, thread_length_x
		);
		CUDA_CHECK(cudaPeekAtLastError());
		break;
	case levelset::EpsilonCalculationMethod_maxOverGrid:
//		printf("epsilonCalculationMethod %d is not supported yet\n", epsilonCalculationMethod_Type);
		return;
	case levelset::EpsilonCalculationMethod_maxOverNeighbor:
		kernel_dimLET2_EpsilonCalculationMethod_maxOverNeighbor2<<<num_of_blocks,num_of_threads, 0, stream>>>(
		dst_deriv_l_ptr, dst_deriv_r_ptr,
		tmpBoundedSrc_ptr,
		dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
		weightL0, weightL1, weightL2, weightR0, weightR1, weightR2,
		1, num_of_slices, loop_length * first_dimension_loop_size,
		stride_distance,
		slice_length,
		thread_length_y, thread_length_x
		);
		CUDA_CHECK(cudaPeekAtLastError());
		break;
	}
}

#endif /* defined(WITH_GPU) */
