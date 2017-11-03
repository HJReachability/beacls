// CUDA runtime
#include <cuda_runtime.h>

#include <typedef.hpp>
#include <cuda_macro.hpp>
#include <Core/UVec.hpp>
#include <Core/CudaStream.hpp>
#include "UpwindFirstWENO5a_cuda.hpp"

#if defined(WITH_GPU)
__global__  static
void kernel_UpwindFirstWENO5aHelper_execute_dim0_cuda_wSaveDD_woApprox4_woStripDD(
	FLOAT_TYPE* dst_dL0_ptr,
	FLOAT_TYPE* dst_dL1_ptr,
	FLOAT_TYPE* dst_dL2_ptr,
	FLOAT_TYPE* dst_dL3_ptr,
	FLOAT_TYPE* dst_dR0_ptr,
	FLOAT_TYPE* dst_dR1_ptr,
	FLOAT_TYPE* dst_dR2_ptr,
	FLOAT_TYPE* dst_dR3_ptr,
	FLOAT_TYPE* dst_DD0_ptr,
	const FLOAT_TYPE* boundedSrc_base_ptr,
	const FLOAT_TYPE dxInv,
	const FLOAT_TYPE dxInv_2,
	const FLOAT_TYPE dxInv_3,
	const FLOAT_TYPE dx,
	const FLOAT_TYPE x2_dx_square,
	const FLOAT_TYPE dx_square,
	const size_t num_of_slices,
	const size_t loop_length,
	const size_t target_dimension_loop_size,
	const size_t first_dimension_loop_size,
	const size_t slice_length,
	const size_t stencil,
	const size_t dst_DD0_line_length,
	const size_t thread_length_z,
	const size_t thread_length_y,
	const size_t thread_length_x
)
{
	calc_D1toD3andDD_dim0_inline<FLOAT_TYPE, SaveDD, noApprox4, noStripDD>(
		dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dL3_ptr,
		dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, dst_dR3_ptr,
		dst_DD0_ptr,
		boundedSrc_base_ptr, dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
		num_of_slices, loop_length,
		target_dimension_loop_size, first_dimension_loop_size, slice_length,
		stencil,
		dst_DD0_line_length,
		thread_length_z, thread_length_y, thread_length_x,
		blockIdx.y, blockIdx.x,
		blockDim.y, blockDim.x,
		threadIdx.z, threadIdx.y, threadIdx.x);
}

void UpwindFirstWENO5aHelper_execute_dim0_cuda
(
	FLOAT_TYPE* dst_dL0_ptr,
	FLOAT_TYPE* dst_dL1_ptr,
	FLOAT_TYPE* dst_dL2_ptr,
	FLOAT_TYPE* dst_dL3_ptr,
	FLOAT_TYPE* dst_dR0_ptr,
	FLOAT_TYPE* dst_dR1_ptr,
	FLOAT_TYPE* dst_dR2_ptr,
	FLOAT_TYPE* dst_dR3_ptr,
	FLOAT_TYPE* dst_DD0_ptr,
	const FLOAT_TYPE* boundedSrc_base_ptr,
	const FLOAT_TYPE dxInv,
	const FLOAT_TYPE dxInv_2,
	const FLOAT_TYPE dxInv_3,
	const FLOAT_TYPE dx,
	const FLOAT_TYPE x2_dx_square,
	const FLOAT_TYPE dx_square,
	const size_t num_of_slices,
	const size_t outer_dimensions_loop_length,
	const size_t target_dimension_loop_size,
	const size_t first_dimension_loop_size,
	const size_t slice_length,
	const size_t stencil,
	const bool saveDD,
	const bool approx4,
	const bool stripDD,
	const size_t dst_DD0_line_length,
	beacls::CudaStream* cudaStream
) {
	const size_t loop_length = outer_dimensions_loop_length;
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
		num_of_slices, loop_length, first_dimension_loop_size,
		1, 1, 8, max_num_of_threads
		);
	dim3 num_of_blocks(num_of_blocks_x, num_of_blocks_y);
	dim3 num_of_threads(num_of_threads_x, num_of_threads_y, num_of_threads_z);

	cudaStream_t stream = cudaStream->get_stream();
	kernel_UpwindFirstWENO5aHelper_execute_dim0_cuda_wSaveDD_woApprox4_woStripDD << <num_of_blocks, num_of_threads, 0, stream >> >(
		dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dL3_ptr,
		dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, dst_dR3_ptr,
		dst_DD0_ptr, 
		boundedSrc_base_ptr, dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
		num_of_slices, loop_length,
		target_dimension_loop_size, first_dimension_loop_size, slice_length,
		stencil,
		dst_DD0_line_length, 
		thread_length_z, thread_length_y, thread_length_x);
}

__global__  static
void kernel_calc_D1toD3andDD_dim1_wSaveDD_woApprox4_woStripDD(
	FLOAT_TYPE* dst_dL0_ptr,
	FLOAT_TYPE* dst_dL1_ptr,
	FLOAT_TYPE* dst_dL2_ptr,
	FLOAT_TYPE* dst_dL3_ptr,
	FLOAT_TYPE* dst_dR0_ptr,
	FLOAT_TYPE* dst_dR1_ptr,
	FLOAT_TYPE* dst_dR2_ptr,
	FLOAT_TYPE* dst_dR3_ptr,
	FLOAT_TYPE* dst_DD0_ptr,
	const FLOAT_TYPE* tmpBoundedSrc_ptr,
	const FLOAT_TYPE dxInv,
	const FLOAT_TYPE dxInv_2,
	const FLOAT_TYPE dxInv_3,
	const FLOAT_TYPE dx,
	const FLOAT_TYPE x2_dx_square,
	const FLOAT_TYPE dx_square,
	const size_t num_of_slices,
	const size_t loop_length,
	const size_t first_dimension_loop_size,
	const size_t slice_length,
	const size_t stencil,
	const size_t dst_DD0_size,
	const size_t thread_length_z,
	const size_t thread_length_y,
	const size_t thread_length_x
)
{
	calc_D1toD3andDD_dim1_inline<FLOAT_TYPE, SaveDD, noApprox4, noStripDD>(
		dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dL3_ptr,
		dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, dst_dR3_ptr,
		dst_DD0_ptr, 
		tmpBoundedSrc_ptr, dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
		num_of_slices, loop_length, first_dimension_loop_size, slice_length,
		stencil,
		dst_DD0_size,
		thread_length_z, thread_length_y, thread_length_x,
		blockIdx.y, blockIdx.x,
		blockDim.y, blockDim.x,
		threadIdx.z, threadIdx.y, threadIdx.x);
}

void UpwindFirstWENO5aHelper_execute_dim1_cuda
(
	FLOAT_TYPE* dst_dL0_ptr,
	FLOAT_TYPE* dst_dL1_ptr,
	FLOAT_TYPE* dst_dL2_ptr,
	FLOAT_TYPE* dst_dL3_ptr,
	FLOAT_TYPE* dst_dR0_ptr,
	FLOAT_TYPE* dst_dR1_ptr,
	FLOAT_TYPE* dst_dR2_ptr,
	FLOAT_TYPE* dst_dR3_ptr,
	FLOAT_TYPE* dst_DD0_ptr,
	const FLOAT_TYPE* tmpBoundedSrc_ptr,
	const FLOAT_TYPE dxInv,
	const FLOAT_TYPE dxInv_2,
	const FLOAT_TYPE dxInv_3,
	const FLOAT_TYPE dx,
	const FLOAT_TYPE x2_dx_square,
	const FLOAT_TYPE dx_square,
	const size_t num_of_slices,
	const size_t loop_length,
	const size_t first_dimension_loop_size,
	const size_t slice_length,
	const size_t stencil,
	const size_t dst_DD0_size,
	const bool saveDD,
	const bool approx4,
	const bool stripDD,
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
		num_of_slices, loop_length, first_dimension_loop_size,
		1, 8, 1, max_num_of_threads
		);
	dim3 num_of_blocks(num_of_blocks_x, num_of_blocks_y);
	dim3 num_of_threads(num_of_threads_x, num_of_threads_y, num_of_threads_z);

	cudaStream_t stream = cudaStream->get_stream();
	kernel_calc_D1toD3andDD_dim1_wSaveDD_woApprox4_woStripDD << <num_of_blocks, num_of_threads, 0, stream >> >(
		dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dL3_ptr,
		dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, dst_dR3_ptr,
		dst_DD0_ptr,
		tmpBoundedSrc_ptr, dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
		num_of_slices, loop_length, first_dimension_loop_size, slice_length,
		stencil,
		dst_DD0_size,
		thread_length_z, thread_length_y, thread_length_x);
}


__global__  static
void kernel_calc_D1toD3andDD_dimLET2_woApprox4_woStripDD(
	FLOAT_TYPE* dst_dL0_ptr,
	FLOAT_TYPE* dst_dL1_ptr,
	FLOAT_TYPE* dst_dL2_ptr,
	FLOAT_TYPE* dst_dL3_ptr,
	FLOAT_TYPE* dst_dR0_ptr,
	FLOAT_TYPE* dst_dR1_ptr,
	FLOAT_TYPE* dst_dR2_ptr,
	FLOAT_TYPE* dst_dR3_ptr,
	FLOAT_TYPE* dst_DD0_ptr,
	const FLOAT_TYPE* tmpBoundedSrc_ptr,
	const FLOAT_TYPE dxInv,
	const FLOAT_TYPE dxInv_2,
	const FLOAT_TYPE dxInv_3,
	const FLOAT_TYPE dx,
	const FLOAT_TYPE x2_dx_square,
	const FLOAT_TYPE dx_square,
	const size_t num_of_slices,
	const size_t loop_length,
	const size_t first_dimension_loop_size,
	const size_t num_of_strides,
	const size_t num_of_dLdR_in_slice,
	const size_t slice_length,
	const size_t thread_length_z,
	const size_t thread_length_y,
	const size_t thread_length_x
)
{
	calc_D1toD3andDD_dimLET2_inline<FLOAT_TYPE, noApprox4, noStripDD>(
		dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dL3_ptr,
		dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, dst_dR3_ptr,
		dst_DD0_ptr, 
		tmpBoundedSrc_ptr, dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
		num_of_slices, loop_length, first_dimension_loop_size,
		num_of_strides, num_of_dLdR_in_slice, slice_length,
		thread_length_z, thread_length_y, thread_length_x,
		blockIdx.y, blockIdx.x,
		blockDim.y, blockDim.x,
		threadIdx.z, threadIdx.y, threadIdx.x);
}

void UpwindFirstWENO5aHelper_execute_dimLET2_cuda
(
	FLOAT_TYPE* dst_dL0_ptr,
	FLOAT_TYPE* dst_dL1_ptr,
	FLOAT_TYPE* dst_dL2_ptr,
	FLOAT_TYPE* dst_dL3_ptr,
	FLOAT_TYPE* dst_dR0_ptr,
	FLOAT_TYPE* dst_dR1_ptr,
	FLOAT_TYPE* dst_dR2_ptr,
	FLOAT_TYPE* dst_dR3_ptr,
	FLOAT_TYPE* dst_DD0_ptr,
	const FLOAT_TYPE* tmpBoundedSrc_ptr,
	const FLOAT_TYPE dxInv,
	const FLOAT_TYPE dxInv_2,
	const FLOAT_TYPE dxInv_3,
	const FLOAT_TYPE dx,
	const FLOAT_TYPE x2_dx_square,
	const FLOAT_TYPE dx_square,
	const size_t num_of_slices,
	const size_t loop_length,
	const size_t first_dimension_loop_size,
	const size_t num_of_strides,
	const size_t num_of_dLdR_in_slice,
	const size_t slice_length,
	const bool saveDD,
	const bool approx4,
	const bool stripDD,
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
		num_of_slices, loop_length, first_dimension_loop_size,
		1, 1, 1, max_num_of_threads
		);
	dim3 num_of_blocks(num_of_blocks_x, num_of_blocks_y);
	dim3 num_of_threads(num_of_threads_x, num_of_threads_y, num_of_threads_z);

	cudaStream_t stream = cudaStream->get_stream();
	kernel_calc_D1toD3andDD_dimLET2_woApprox4_woStripDD << <num_of_blocks, num_of_threads, 0, stream >> >(
		dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dL3_ptr,
		dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, dst_dR3_ptr,
		dst_DD0_ptr,
		tmpBoundedSrc_ptr, dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
		num_of_slices, loop_length, first_dimension_loop_size,
		num_of_strides, num_of_dLdR_in_slice, slice_length,
		thread_length_z, thread_length_y, thread_length_x);
}
__global__ static
void kernel_dim0_EpsilonCalculationMethod_Constant(
	FLOAT_TYPE* dst_deriv_l_ptr,
	FLOAT_TYPE* dst_deriv_r_ptr,
	const FLOAT_TYPE* DD0_0_ptr,
	const FLOAT_TYPE* dL0_ptr,
	const FLOAT_TYPE* dL1_ptr,
	const FLOAT_TYPE* dL2_ptr,
	const FLOAT_TYPE* dR0_ptr,
	const FLOAT_TYPE* dR1_ptr,
	const FLOAT_TYPE* dR2_ptr,
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
	const size_t thread_length_z,
	const size_t thread_length_y,
	const size_t thread_length_x
) {
	kernel_dim0_EpsilonCalculationMethod_Constant_inline(
		dst_deriv_l_ptr, dst_deriv_r_ptr,
		DD0_0_ptr,
		dL0_ptr, dL1_ptr, dL2_ptr,dR0_ptr, dR1_ptr, dR2_ptr,
		weightL0, weightL1, weightL2, weightR0, weightR1, weightR2, 
		num_of_slices, loop_length, src_target_dimension_loop_size, first_dimension_loop_size, slice_length,
		thread_length_z, thread_length_y, thread_length_x,
		blockIdx.y, blockIdx.x, 
		blockDim.y, blockDim.x,
		threadIdx.z, threadIdx.y, threadIdx.x);
}

__global__ static
void kernel_dim0_EpsilonCalculationMethod_maxOverNeighbor(
	FLOAT_TYPE* dst_deriv_l_ptr,
	FLOAT_TYPE* dst_deriv_r_ptr,
	const FLOAT_TYPE* DD0_0_ptr,
	const FLOAT_TYPE* dL0_ptr,
	const FLOAT_TYPE* dL1_ptr,
	const FLOAT_TYPE* dL2_ptr,
	const FLOAT_TYPE* dR0_ptr,
	const FLOAT_TYPE* dR1_ptr,
	const FLOAT_TYPE* dR2_ptr,
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
	const size_t thread_length_z,
	const size_t thread_length_y,
	const size_t thread_length_x
) {
	kernel_dim0_EpsilonCalculationMethod_maxOverNeighbor_inline(
		dst_deriv_l_ptr, dst_deriv_r_ptr,
		DD0_0_ptr,
		dL0_ptr, dL1_ptr, dL2_ptr,dR0_ptr, dR1_ptr, dR2_ptr,
		weightL0, weightL1, weightL2, weightR0, weightR1, weightR2, 
		num_of_slices, loop_length, src_target_dimension_loop_size, first_dimension_loop_size,  slice_length,
		thread_length_z, thread_length_y, thread_length_x,
		blockIdx.y, blockIdx.x, 
		blockDim.y, blockDim.x,
		threadIdx.z, threadIdx.y, threadIdx.x);
}


void UpwindFirstWENO5a_execute_dim0_cuda (
	FLOAT_TYPE* dst_deriv_l_ptr,
	FLOAT_TYPE* dst_deriv_r_ptr,
	const FLOAT_TYPE* DD0_ptr,
	const FLOAT_TYPE* dL0_ptr,
	const FLOAT_TYPE* dL1_ptr,
	const FLOAT_TYPE* dL2_ptr,
	const FLOAT_TYPE* dR0_ptr,
	const FLOAT_TYPE* dR1_ptr,
	const FLOAT_TYPE* dR2_ptr,
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
		num_of_slices, loop_length, first_dimension_loop_size,
		1, 1, 8, max_num_of_threads
		);
	dim3 num_of_blocks(num_of_blocks_x, num_of_blocks_y);
	dim3 num_of_threads(num_of_threads_x, num_of_threads_y, num_of_threads_z);

	cudaStream_t stream = cudaStream->get_stream();
	switch (epsilonCalculationMethod_Type) {
	case levelset::EpsilonCalculationMethod_Invalid:
	default:
//		printf("Unknown epsilonCalculationMethod %d\n", epsilonCalculationMethod_Type);
		return;
	case levelset::EpsilonCalculationMethod_Constant:
		kernel_dim0_EpsilonCalculationMethod_Constant<<<num_of_blocks,num_of_threads, 0, stream>>>(
			dst_deriv_l_ptr, dst_deriv_r_ptr,
			DD0_ptr,
			dL0_ptr, dL1_ptr, dL2_ptr, dR0_ptr, dR1_ptr, dR2_ptr,
			weightL0, weightL1, weightL2, weightR0, weightR1, weightR2,
			num_of_slices, loop_length, src_target_dimension_loop_size, first_dimension_loop_size, slice_length,
			thread_length_z, thread_length_y, thread_length_x
			);
		break;
	case levelset::EpsilonCalculationMethod_maxOverGrid:
//		printf("epsilonCalculationMethod %d is not supported yet\n", epsilonCalculationMethod_Type);
		return;
	case levelset::EpsilonCalculationMethod_maxOverNeighbor:
		kernel_dim0_EpsilonCalculationMethod_maxOverNeighbor<<<num_of_blocks,num_of_threads, 0, stream>>>(
			dst_deriv_l_ptr, dst_deriv_r_ptr,
			DD0_ptr,
			dL0_ptr, dL1_ptr, dL2_ptr, dR0_ptr, dR1_ptr, dR2_ptr,
			weightL0, weightL1, weightL2, weightR0, weightR1, weightR2,
			num_of_slices, loop_length, src_target_dimension_loop_size, first_dimension_loop_size, slice_length,
			thread_length_z, thread_length_y, thread_length_x
			);
		break;
	}
}

__global__ static 
void kernel_dim1_EpsilonCalculationMethod_Constant(
	FLOAT_TYPE* dst_deriv_l_ptr,
	FLOAT_TYPE* dst_deriv_r_ptr,
	const FLOAT_TYPE* DD0_ptr,
	const FLOAT_TYPE* dL0_ptr,
	const FLOAT_TYPE* dL1_ptr,
	const FLOAT_TYPE* dL2_ptr,
	const FLOAT_TYPE* dR0_ptr,
	const FLOAT_TYPE* dR1_ptr,
	const FLOAT_TYPE* dR2_ptr,
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
	const size_t DD0_slice_size,
	const size_t thread_length_z,
	const size_t thread_length_y,
	const size_t thread_length_x
) {
	kernel_dim1_EpsilonCalculationMethod_Constant_inline(
		dst_deriv_l_ptr, dst_deriv_r_ptr,
		DD0_ptr,
		dL0_ptr, dL1_ptr, dL2_ptr,dR0_ptr, dR1_ptr, dR2_ptr,
		weightL0, weightL1, weightL2, weightR0, weightR1, weightR2, 
		num_of_slices, loop_length, first_dimension_loop_size, slice_length, 
		DD0_slice_size,
		thread_length_z, thread_length_y, thread_length_x,
		blockIdx.y, blockIdx.x, 
		blockDim.y, blockDim.x,
		threadIdx.z, threadIdx.y, threadIdx.x);
}

__global__ static
void kernel_dim1_EpsilonCalculationMethod_maxOverNeighbor(
	FLOAT_TYPE* dst_deriv_l_ptr,
	FLOAT_TYPE* dst_deriv_r_ptr,
	const FLOAT_TYPE* DD0_ptr,
	const FLOAT_TYPE* dL0_ptr,
	const FLOAT_TYPE* dL1_ptr,
	const FLOAT_TYPE* dL2_ptr,
	const FLOAT_TYPE* dR0_ptr,
	const FLOAT_TYPE* dR1_ptr,
	const FLOAT_TYPE* dR2_ptr,
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
	const size_t DD0_slice_size,
	const size_t thread_length_z,
	const size_t thread_length_y,
	const size_t thread_length_x
) {
	kernel_dim1_EpsilonCalculationMethod_maxOverNeighbor_inline(
		dst_deriv_l_ptr, dst_deriv_r_ptr,
		DD0_ptr,
		dL0_ptr, dL1_ptr, dL2_ptr,dR0_ptr, dR1_ptr, dR2_ptr,
		weightL0, weightL1, weightL2, weightR0, weightR1, weightR2, 
		num_of_slices, loop_length, first_dimension_loop_size, slice_length, 
		DD0_slice_size,
		thread_length_z, thread_length_y, thread_length_x,
		blockIdx.y, blockIdx.x, 
		blockDim.y, blockDim.x,
		threadIdx.z, threadIdx.y, threadIdx.x);
}


void UpwindFirstWENO5a_execute_dim1_cuda (
	FLOAT_TYPE* dst_deriv_l_ptr,
	FLOAT_TYPE* dst_deriv_r_ptr,
	const FLOAT_TYPE* DD0_ptr,
	const FLOAT_TYPE* dL0_ptr,
	const FLOAT_TYPE* dL1_ptr,
	const FLOAT_TYPE* dL2_ptr,
	const FLOAT_TYPE* dR0_ptr,
	const FLOAT_TYPE* dR1_ptr,
	const FLOAT_TYPE* dR2_ptr,
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
	const size_t DD0_slice_size,
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
		num_of_slices, loop_length, first_dimension_loop_size,
		1, 8, 1, max_num_of_threads
		);
	dim3 num_of_blocks(num_of_blocks_x, num_of_blocks_y);
	dim3 num_of_threads(num_of_threads_x, num_of_threads_y, num_of_threads_z);
	cudaStream_t stream = cudaStream->get_stream();

	switch (epsilonCalculationMethod_Type) {
	case levelset::EpsilonCalculationMethod_Invalid:
	default:
//		printf("Unknown epsilonCalculationMethod %d\n", epsilonCalculationMethod_Type);
		return;
	case levelset::EpsilonCalculationMethod_Constant:
		kernel_dim1_EpsilonCalculationMethod_Constant<<<num_of_blocks,num_of_threads, 0, stream>>>(
		dst_deriv_l_ptr, dst_deriv_r_ptr,
		DD0_ptr,
		dL0_ptr, dL1_ptr, dL2_ptr,dR0_ptr, dR1_ptr, dR2_ptr,
		weightL0, weightL1, weightL2, weightR0, weightR1, weightR2, 
		num_of_slices, loop_length, first_dimension_loop_size, slice_length, 
		DD0_slice_size,
		thread_length_z, thread_length_y, thread_length_x
		);
		break;
	case levelset::EpsilonCalculationMethod_maxOverGrid:
//		printf("epsilonCalculationMethod %d is not supported yet\n", epsilonCalculationMethod_Type);
		return;
	case levelset::EpsilonCalculationMethod_maxOverNeighbor:
		kernel_dim1_EpsilonCalculationMethod_maxOverNeighbor<<<num_of_blocks,num_of_threads, 0, stream>>>(
		dst_deriv_l_ptr, dst_deriv_r_ptr,
		DD0_ptr,
		dL0_ptr, dL1_ptr, dL2_ptr,dR0_ptr, dR1_ptr, dR2_ptr,
		weightL0, weightL1, weightL2, weightR0, weightR1, weightR2, 
		num_of_slices, loop_length, first_dimension_loop_size, slice_length, 
		DD0_slice_size,
		thread_length_z, thread_length_y, thread_length_x
				);
		break;
	}
}

__global__ static 
void kernel_dimLET2_EpsilonCalculationMethod_Constant(
	FLOAT_TYPE* dst_deriv_l_ptr,
	FLOAT_TYPE* dst_deriv_r_ptr,
	const FLOAT_TYPE* DD0_0_ptr,
	const FLOAT_TYPE* DD1_0_ptr,
	const FLOAT_TYPE* DD2_0_ptr,
	const FLOAT_TYPE* DD3_0_ptr,
	const FLOAT_TYPE* DD4_0_ptr,
	const FLOAT_TYPE* DD5_0_ptr,
	const FLOAT_TYPE* dL0_ptr,
	const FLOAT_TYPE* dL1_ptr,
	const FLOAT_TYPE* dL2_ptr,
	const FLOAT_TYPE* dR0_ptr,
	const FLOAT_TYPE* dR1_ptr,
	const FLOAT_TYPE* dR2_ptr,
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
	const size_t thread_length_z,
	const size_t thread_length_y,
	const size_t thread_length_x
) {
	kernel_dimLET2_EpsilonCalculationMethod_Constant_inline(
		dst_deriv_l_ptr, dst_deriv_r_ptr,
		DD0_0_ptr,DD1_0_ptr,DD2_0_ptr, DD3_0_ptr, DD4_0_ptr, DD5_0_ptr,
		dL0_ptr, dL1_ptr, dL2_ptr,dR0_ptr, dR1_ptr, dR2_ptr,
		weightL0, weightL1, weightL2, weightR0, weightR1, weightR2, 
		num_of_slices, loop_length, first_dimension_loop_size, slice_length, 
		thread_length_z, thread_length_y, thread_length_x,
		blockIdx.y, blockIdx.x, 
		blockDim.y, blockDim.x,
		threadIdx.z, threadIdx.y, threadIdx.x);
}

__global__ static
void kernel_dimLET2_EpsilonCalculationMethod_maxOverNeighbor(
	FLOAT_TYPE* dst_deriv_l_ptr,
	FLOAT_TYPE* dst_deriv_r_ptr,
	const FLOAT_TYPE* DD0_0_ptr,
	const FLOAT_TYPE* DD1_0_ptr,
	const FLOAT_TYPE* DD2_0_ptr,
	const FLOAT_TYPE* DD3_0_ptr,
	const FLOAT_TYPE* DD4_0_ptr,
	const FLOAT_TYPE* DD5_0_ptr,
	const FLOAT_TYPE* dL0_ptr,
	const FLOAT_TYPE* dL1_ptr,
	const FLOAT_TYPE* dL2_ptr,
	const FLOAT_TYPE* dR0_ptr,
	const FLOAT_TYPE* dR1_ptr,
	const FLOAT_TYPE* dR2_ptr,
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
	const size_t thread_length_z,
	const size_t thread_length_y,
	const size_t thread_length_x
) {
	kernel_dimLET2_EpsilonCalculationMethod_maxOverNeighbor_inline(
		dst_deriv_l_ptr, dst_deriv_r_ptr,
		DD0_0_ptr,DD1_0_ptr,DD2_0_ptr, DD3_0_ptr, DD4_0_ptr, DD5_0_ptr,
		dL0_ptr, dL1_ptr, dL2_ptr,dR0_ptr, dR1_ptr, dR2_ptr,
		weightL0, weightL1, weightL2, weightR0, weightR1, weightR2, 
		num_of_slices, loop_length, first_dimension_loop_size, slice_length,
		thread_length_z, thread_length_y, thread_length_x,
		blockIdx.y, blockIdx.x, 
		blockDim.y, blockDim.x,
		threadIdx.z, threadIdx.y, threadIdx.x);
}


void UpwindFirstWENO5a_execute_dimLET2_cuda (
	FLOAT_TYPE* dst_deriv_l_ptr,
	FLOAT_TYPE* dst_deriv_r_ptr,
	const FLOAT_TYPE* DD0_0_ptr,
	const FLOAT_TYPE* DD1_0_ptr,
	const FLOAT_TYPE* DD2_0_ptr,
	const FLOAT_TYPE* DD3_0_ptr,
	const FLOAT_TYPE* DD4_0_ptr,
	const FLOAT_TYPE* DD5_0_ptr,
	const FLOAT_TYPE* dL0_ptr,
	const FLOAT_TYPE* dL1_ptr,
	const FLOAT_TYPE* dL2_ptr,
	const FLOAT_TYPE* dR0_ptr,
	const FLOAT_TYPE* dR1_ptr,
	const FLOAT_TYPE* dR2_ptr,
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
		num_of_slices, loop_length, first_dimension_loop_size,
		1, 1, 1, max_num_of_threads
		);
	dim3 num_of_blocks(num_of_blocks_x, num_of_blocks_y);
	dim3 num_of_threads(num_of_threads_x, num_of_threads_y, num_of_threads_z);

	cudaStream_t stream = cudaStream->get_stream();
	switch (epsilonCalculationMethod_Type) {
	case levelset::EpsilonCalculationMethod_Invalid:
	default:
//		printf("Unknown epsilonCalculationMethod %d\n", epsilonCalculationMethod_Type);
		return;
	case levelset::EpsilonCalculationMethod_Constant:
		kernel_dimLET2_EpsilonCalculationMethod_Constant<<<num_of_blocks,num_of_threads, 0, stream>>>(
		dst_deriv_l_ptr, dst_deriv_r_ptr,
		DD0_0_ptr,DD1_0_ptr,DD2_0_ptr, DD3_0_ptr, DD4_0_ptr, DD5_0_ptr,
		dL0_ptr, dL1_ptr, dL2_ptr,dR0_ptr, dR1_ptr, dR2_ptr,
		weightL0, weightL1, weightL2, weightR0, weightR1, weightR2, 
		num_of_slices, loop_length, first_dimension_loop_size, slice_length, 
		thread_length_z, thread_length_y, thread_length_x
		);
		break;
	case levelset::EpsilonCalculationMethod_maxOverGrid:
//		printf("epsilonCalculationMethod %d is not supported yet\n", epsilonCalculationMethod_Type);
		return;
	case levelset::EpsilonCalculationMethod_maxOverNeighbor:
		kernel_dimLET2_EpsilonCalculationMethod_maxOverNeighbor<<<num_of_blocks,num_of_threads, 0, stream>>>(
		dst_deriv_l_ptr, dst_deriv_r_ptr,
		DD0_0_ptr,DD1_0_ptr,DD2_0_ptr, DD3_0_ptr, DD4_0_ptr, DD5_0_ptr,
		dL0_ptr, dL1_ptr, dL2_ptr,dR0_ptr, dR1_ptr, dR2_ptr,
		weightL0, weightL1, weightL2, weightR0, weightR1, weightR2, 
		num_of_slices, loop_length, first_dimension_loop_size, slice_length, 
		thread_length_z, thread_length_y, thread_length_x
		);
		break;
	}
}
#endif /* defined(WITH_GPU) */
