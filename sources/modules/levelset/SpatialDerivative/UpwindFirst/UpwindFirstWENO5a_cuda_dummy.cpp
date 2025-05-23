#include <typedef.hpp>
#include <cuda_macro.hpp>
#include <cmath>
#include "UpwindFirstWENO5a_cuda.hpp"
#include <Core/UVec.hpp>

#if !defined(WITH_GPU) 

void UpwindFirstWENO5a_execute_dim0_cuda(
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
	beacls::synchronizeCuda(cudaStream);
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

	switch (epsilonCalculationMethod_Type) {
	case levelset::EpsilonCalculationMethod_Invalid:
	default:
		//		printf("Unknown epsilonCalculationMethod %d\n", epsilonCalculationMethod_Type);
		return;
	case levelset::EpsilonCalculationMethod_Constant:
		for (size_t blockIdx_y = 0; blockIdx_y < num_of_blocks_y; ++blockIdx_y) {
			for (size_t blockIdx_x = 0; blockIdx_x < num_of_blocks_x; ++blockIdx_x) {
				for (size_t threadIdx_z = 0; threadIdx_z < num_of_threads_z; ++threadIdx_z) {
					for (size_t threadIdx_y = 0; threadIdx_y < num_of_threads_y; ++threadIdx_y) {
						for (size_t threadIdx_x = 0; threadIdx_x < num_of_threads_x; ++threadIdx_x) {
							kernel_dim0_EpsilonCalculationMethod_Constant_inline(
								dst_deriv_l_ptr, dst_deriv_r_ptr,
								DD0_ptr,
								dL0_ptr, dL1_ptr, dL2_ptr, dR0_ptr, dR1_ptr, dR2_ptr,
								weightL0, weightL1, weightL2, weightR0, weightR1, weightR2,
								num_of_slices, loop_length, src_target_dimension_loop_size, first_dimension_loop_size, slice_length,
								thread_length_z, thread_length_y, thread_length_x,
								blockIdx_y, blockIdx_x,
								num_of_threads_y, num_of_threads_x,
								threadIdx_z, threadIdx_y, threadIdx_x);
						}
					}
				}
			}
		}
		break;
	case levelset::EpsilonCalculationMethod_maxOverGrid:
		//		printf("epsilonCalculationMethod %d is not supported yet\n", epsilonCalculationMethod_Type);
		return;
	case levelset::EpsilonCalculationMethod_maxOverNeighbor:
		for (size_t blockIdx_y = 0; blockIdx_y < num_of_blocks_y; ++blockIdx_y) {
			for (size_t blockIdx_x = 0; blockIdx_x < num_of_blocks_x; ++blockIdx_x) {
				for (size_t threadIdx_z = 0; threadIdx_z < num_of_threads_z; ++threadIdx_z) {
					for (size_t threadIdx_y = 0; threadIdx_y < num_of_threads_y; ++threadIdx_y) {
						for (size_t threadIdx_x = 0; threadIdx_x < num_of_threads_x; ++threadIdx_x) {
							kernel_dim0_EpsilonCalculationMethod_maxOverNeighbor_inline(
								dst_deriv_l_ptr, dst_deriv_r_ptr,
								DD0_ptr,
								dL0_ptr, dL1_ptr, dL2_ptr, dR0_ptr, dR1_ptr, dR2_ptr,
								weightL0, weightL1, weightL2, weightR0, weightR1, weightR2,
								num_of_slices, loop_length, src_target_dimension_loop_size, first_dimension_loop_size, slice_length,
								thread_length_z, thread_length_y, thread_length_x,
								blockIdx_y, blockIdx_x,
								num_of_threads_y, num_of_threads_x,
								threadIdx_z, threadIdx_y, threadIdx_x);
						}
					}
				}
			}
		}
		break;
	}
}

void UpwindFirstWENO5a_execute_dim1_cuda(
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
	beacls::synchronizeCuda(cudaStream);
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

	switch (epsilonCalculationMethod_Type) {
	case levelset::EpsilonCalculationMethod_Invalid:
	default:
		//		printf("Unknown epsilonCalculationMethod %d\n", epsilonCalculationMethod_Type);
		return;
	case levelset::EpsilonCalculationMethod_Constant:
		for (size_t blockIdx_y = 0; blockIdx_y < num_of_blocks_y; ++blockIdx_y) {
			for (size_t blockIdx_x = 0; blockIdx_x < num_of_blocks_x; ++blockIdx_x) {
				for (size_t threadIdx_z = 0; threadIdx_z < num_of_threads_z; ++threadIdx_z) {
					for (size_t threadIdx_y = 0; threadIdx_y < num_of_threads_y; ++threadIdx_y) {
						for (size_t threadIdx_x = 0; threadIdx_x < num_of_threads_x; ++threadIdx_x) {
							kernel_dim1_EpsilonCalculationMethod_Constant_inline(
								dst_deriv_l_ptr, dst_deriv_r_ptr,
								DD0_ptr,
								dL0_ptr, dL1_ptr, dL2_ptr, dR0_ptr, dR1_ptr, dR2_ptr,
								weightL0, weightL1, weightL2, weightR0, weightR1, weightR2,
								num_of_slices, loop_length, first_dimension_loop_size, slice_length,
								DD0_slice_size,
								thread_length_z, thread_length_y, thread_length_x,
								blockIdx_y, blockIdx_x,
								num_of_threads_y, num_of_threads_x,
								threadIdx_z, threadIdx_y, threadIdx_x);
						}
					}
				}
			}
		}
		break;
	case levelset::EpsilonCalculationMethod_maxOverGrid:
		//		printf("epsilonCalculationMethod %d is not supported yet\n", epsilonCalculationMethod_Type);
		return;
	case levelset::EpsilonCalculationMethod_maxOverNeighbor:
		for (size_t blockIdx_y = 0; blockIdx_y < num_of_blocks_y; ++blockIdx_y) {
			for (size_t blockIdx_x = 0; blockIdx_x < num_of_blocks_x; ++blockIdx_x) {
				for (size_t threadIdx_z = 0; threadIdx_z < num_of_threads_z; ++threadIdx_z) {
					for (size_t threadIdx_y = 0; threadIdx_y < num_of_threads_y; ++threadIdx_y) {
						for (size_t threadIdx_x = 0; threadIdx_x < num_of_threads_x; ++threadIdx_x) {
							kernel_dim1_EpsilonCalculationMethod_maxOverNeighbor_inline(
								dst_deriv_l_ptr, dst_deriv_r_ptr,
								DD0_ptr,
								dL0_ptr, dL1_ptr, dL2_ptr, dR0_ptr, dR1_ptr, dR2_ptr,
								weightL0, weightL1, weightL2, weightR0, weightR1, weightR2,
								num_of_slices, loop_length, first_dimension_loop_size, slice_length,
								DD0_slice_size,
								thread_length_z, thread_length_y, thread_length_x,
								blockIdx_y, blockIdx_x,
								num_of_threads_y, num_of_threads_x,
								threadIdx_z, threadIdx_y, threadIdx_x);
						}
					}
				}
			}
		}
		break;
	}
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
	beacls::synchronizeCuda(cudaStream);
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

	switch (epsilonCalculationMethod_Type) {
	case levelset::EpsilonCalculationMethod_Invalid:
	default:
//		printf("Unknown epsilonCalculationMethod %d\n", epsilonCalculationMethod_Type);
		return;
	case levelset::EpsilonCalculationMethod_Constant:
		for (size_t blockIdx_y = 0; blockIdx_y < num_of_blocks_y; ++blockIdx_y) {
			for (size_t blockIdx_x = 0; blockIdx_x < num_of_blocks_x; ++blockIdx_x) {
				for (size_t threadIdx_z = 0; threadIdx_z < num_of_threads_z; ++threadIdx_z) {
					for (size_t threadIdx_y = 0; threadIdx_y < num_of_threads_y; ++threadIdx_y) {
						for (size_t threadIdx_x = 0; threadIdx_x < num_of_threads_x; ++threadIdx_x) {
							kernel_dimLET2_EpsilonCalculationMethod_Constant_inline(
								dst_deriv_l_ptr, dst_deriv_r_ptr,
								DD0_0_ptr, DD1_0_ptr, DD2_0_ptr, DD3_0_ptr, DD4_0_ptr, DD5_0_ptr,
								dL0_ptr, dL1_ptr, dL2_ptr, dR0_ptr, dR1_ptr, dR2_ptr,
								weightL0, weightL1, weightL2, weightR0, weightR1, weightR2,
								num_of_slices, loop_length, first_dimension_loop_size, slice_length,
								thread_length_z, thread_length_y, thread_length_x,
								blockIdx_y, blockIdx_x,
								num_of_threads_y, num_of_threads_x,
								threadIdx_z, threadIdx_y, threadIdx_x);
						}
					}
				}
			}
		}
		break;
	case levelset::EpsilonCalculationMethod_maxOverGrid:
//		printf("epsilonCalculationMethod %d is not supported yet\n", epsilonCalculationMethod_Type);
		return;
	case levelset::EpsilonCalculationMethod_maxOverNeighbor:
		for (size_t blockIdx_y = 0; blockIdx_y < num_of_blocks_y; ++blockIdx_y) {
			for (size_t blockIdx_x = 0; blockIdx_x < num_of_blocks_x; ++blockIdx_x) {
				for (size_t threadIdx_z = 0; threadIdx_z < num_of_threads_z; ++threadIdx_z) {
					for (size_t threadIdx_y = 0; threadIdx_y < num_of_threads_y; ++threadIdx_y) {
						for (size_t threadIdx_x = 0; threadIdx_x < num_of_threads_x; ++threadIdx_x) {
							kernel_dimLET2_EpsilonCalculationMethod_maxOverNeighbor_inline(
								dst_deriv_l_ptr, dst_deriv_r_ptr,
								DD0_0_ptr, DD1_0_ptr, DD2_0_ptr, DD3_0_ptr, DD4_0_ptr, DD5_0_ptr,
								dL0_ptr, dL1_ptr, dL2_ptr, dR0_ptr, dR1_ptr, dR2_ptr,
								weightL0, weightL1, weightL2, weightR0, weightR1, weightR2,
								num_of_slices, loop_length, first_dimension_loop_size, slice_length,
								thread_length_z, thread_length_y, thread_length_x,
								blockIdx_y, blockIdx_x,
								num_of_threads_y, num_of_threads_x,
								threadIdx_z, threadIdx_y, threadIdx_x);
						}
					}
				}
			}
		}
		break;
	}
}
#endif /* !defined(WITH_GPU) */
