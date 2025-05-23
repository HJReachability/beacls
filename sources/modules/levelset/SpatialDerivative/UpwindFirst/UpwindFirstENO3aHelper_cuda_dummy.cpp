#include <typedef.hpp>
#include <cuda_macro.hpp>
#include "UpwindFirstENO3aHelper_cuda.hpp"
#include <cmath>
#include <Core/UVec.hpp>
#if !defined(WITH_GPU)

void UpwindFirstENO3aHelper_execute_dim0_cuda
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
	FLOAT_TYPE* dst_DD1_ptr,
	FLOAT_TYPE* dst_DD2_ptr,
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
	const size_t dst_DD1_line_length,
	const size_t dst_DD2_line_length,
	beacls::CudaStream* cudaStream
) {
	beacls::synchronizeCuda(cudaStream);
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

	if (saveDD) {
		if (approx4) {
			if (stripDD) {
				for (size_t blockIdx_y = 0; blockIdx_y < num_of_blocks_y; ++blockIdx_y) {
					for (size_t blockIdx_x = 0; blockIdx_x < num_of_blocks_x; ++blockIdx_x) {
						for (size_t threadIdx_z = 0; threadIdx_z < num_of_threads_z; ++threadIdx_z) {
							for (size_t threadIdx_y = 0; threadIdx_y < num_of_threads_y; ++threadIdx_y) {
								for (size_t threadIdx_x = 0; threadIdx_x < num_of_threads_x; ++threadIdx_x) {
									calc_D1toD3andDD_dim0_inline<FLOAT_TYPE, SaveDD, Approx4, StripDD>(
										dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dL3_ptr,
										dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, dst_dR3_ptr,
										dst_DD0_ptr, dst_DD1_ptr, dst_DD2_ptr,
										boundedSrc_base_ptr, dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
										num_of_slices, loop_length,
										target_dimension_loop_size, first_dimension_loop_size, slice_length,
										stencil,
										dst_DD0_line_length, dst_DD1_line_length, dst_DD2_line_length,
										thread_length_z, thread_length_y, thread_length_x,
										blockIdx_y, blockIdx_x,
										num_of_threads_y, num_of_threads_x,
										threadIdx_z, threadIdx_y, threadIdx_x);
								}
							}
						}
					}
				}
			}
			else {
				for (size_t blockIdx_y = 0; blockIdx_y < num_of_blocks_y; ++blockIdx_y) {
					for (size_t blockIdx_x = 0; blockIdx_x < num_of_blocks_x; ++blockIdx_x) {
						for (size_t threadIdx_z = 0; threadIdx_z < num_of_threads_z; ++threadIdx_z) {
							for (size_t threadIdx_y = 0; threadIdx_y < num_of_threads_y; ++threadIdx_y) {
								for (size_t threadIdx_x = 0; threadIdx_x < num_of_threads_x; ++threadIdx_x) {
									calc_D1toD3andDD_dim0_inline<FLOAT_TYPE, SaveDD, Approx4, noStripDD>(
										dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dL3_ptr,
										dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, dst_dR3_ptr,
										dst_DD0_ptr, dst_DD1_ptr, dst_DD2_ptr,
										boundedSrc_base_ptr, dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
										num_of_slices, loop_length,
										target_dimension_loop_size, first_dimension_loop_size, slice_length,
										stencil,
										dst_DD0_line_length, dst_DD1_line_length, dst_DD2_line_length,
										thread_length_z, thread_length_y, thread_length_x,
										blockIdx_y, blockIdx_x,
										num_of_threads_y, num_of_threads_x,
										threadIdx_z, threadIdx_y, threadIdx_x);
								}
							}
						}
					}
				}
			}
		}
		else {
			if (stripDD) {
				for (size_t blockIdx_y = 0; blockIdx_y < num_of_blocks_y; ++blockIdx_y) {
					for (size_t blockIdx_x = 0; blockIdx_x < num_of_blocks_x; ++blockIdx_x) {
						for (size_t threadIdx_z = 0; threadIdx_z < num_of_threads_z; ++threadIdx_z) {
							for (size_t threadIdx_y = 0; threadIdx_y < num_of_threads_y; ++threadIdx_y) {
								for (size_t threadIdx_x = 0; threadIdx_x < num_of_threads_x; ++threadIdx_x) {
									calc_D1toD3andDD_dim0_inline<FLOAT_TYPE, SaveDD, noApprox4, StripDD>(
										dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dL3_ptr,
										dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, dst_dR3_ptr,
										dst_DD0_ptr, dst_DD1_ptr, dst_DD2_ptr,
										boundedSrc_base_ptr, dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
										num_of_slices, loop_length,
										target_dimension_loop_size, first_dimension_loop_size, slice_length,
										stencil,
										dst_DD0_line_length, dst_DD1_line_length, dst_DD2_line_length,
										thread_length_z, thread_length_y, thread_length_x,
										blockIdx_y, blockIdx_x,
										num_of_threads_y, num_of_threads_x,
										threadIdx_z, threadIdx_y, threadIdx_x);
								}
							}
						}
					}
				}
			}
			else {
				for (size_t blockIdx_y = 0; blockIdx_y < num_of_blocks_y; ++blockIdx_y) {
					for (size_t blockIdx_x = 0; blockIdx_x < num_of_blocks_x; ++blockIdx_x) {
						for (size_t threadIdx_z = 0; threadIdx_z < num_of_threads_z; ++threadIdx_z) {
							for (size_t threadIdx_y = 0; threadIdx_y < num_of_threads_y; ++threadIdx_y) {
								for (size_t threadIdx_x = 0; threadIdx_x < num_of_threads_x; ++threadIdx_x) {
									calc_D1toD3andDD_dim0_inline<FLOAT_TYPE, SaveDD, noApprox4, noStripDD>(
										dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dL3_ptr,
										dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, dst_dR3_ptr,
										dst_DD0_ptr, dst_DD1_ptr, dst_DD2_ptr,
										boundedSrc_base_ptr, dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
										num_of_slices, loop_length,
										target_dimension_loop_size, first_dimension_loop_size, slice_length,
										stencil,
										dst_DD0_line_length, dst_DD1_line_length, dst_DD2_line_length,
										thread_length_z, thread_length_y, thread_length_x,
										blockIdx_y, blockIdx_x,
										num_of_threads_y, num_of_threads_x,
										threadIdx_z, threadIdx_y, threadIdx_x);
								}
							}
						}
					}
				}
			}
		}
	}
	else {
		if (approx4) {
			for (size_t blockIdx_y = 0; blockIdx_y < num_of_blocks_y; ++blockIdx_y) {
				for (size_t blockIdx_x = 0; blockIdx_x < num_of_blocks_x; ++blockIdx_x) {
					for (size_t threadIdx_z = 0; threadIdx_z < num_of_threads_z; ++threadIdx_z) {
						for (size_t threadIdx_y = 0; threadIdx_y < num_of_threads_y; ++threadIdx_y) {
							for (size_t threadIdx_x = 0; threadIdx_x < num_of_threads_x; ++threadIdx_x) {
								calc_D1toD3andDD_dim0_inline<FLOAT_TYPE, noSaveDD, Approx4, noStripDD>(
									dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dL3_ptr,
									dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, dst_dR3_ptr,
									dst_DD0_ptr, dst_DD1_ptr, dst_DD2_ptr,
									boundedSrc_base_ptr, dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
									num_of_slices, loop_length,
									target_dimension_loop_size, first_dimension_loop_size, slice_length,
									stencil,
									dst_DD0_line_length, dst_DD1_line_length, dst_DD2_line_length,
									thread_length_z, thread_length_y, thread_length_x,
									blockIdx_y, blockIdx_x,
									num_of_threads_y, num_of_threads_x,
									threadIdx_z, threadIdx_y, threadIdx_x);
							}
						}
					}
				}
			}
		}
		else {
			for (size_t blockIdx_y = 0; blockIdx_y < num_of_blocks_y; ++blockIdx_y) {
				for (size_t blockIdx_x = 0; blockIdx_x < num_of_blocks_x; ++blockIdx_x) {
					for (size_t threadIdx_z = 0; threadIdx_z < num_of_threads_z; ++threadIdx_z) {
						for (size_t threadIdx_y = 0; threadIdx_y < num_of_threads_y; ++threadIdx_y) {
							for (size_t threadIdx_x = 0; threadIdx_x < num_of_threads_x; ++threadIdx_x) {
								calc_D1toD3andDD_dim0_inline<FLOAT_TYPE, noSaveDD, noApprox4, noStripDD>(
									dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dL3_ptr,
									dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, dst_dR3_ptr,
									dst_DD0_ptr, dst_DD1_ptr, dst_DD2_ptr,
									boundedSrc_base_ptr, dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
									num_of_slices, loop_length,
									target_dimension_loop_size, first_dimension_loop_size, slice_length,
									stencil,
									dst_DD0_line_length, dst_DD1_line_length, dst_DD2_line_length,
									thread_length_z, thread_length_y, thread_length_x,
									blockIdx_y, blockIdx_x,
									num_of_threads_y, num_of_threads_x,
									threadIdx_z, threadIdx_y, threadIdx_x);
							}
						}
					}
				}
			}
		}
	}
}

void UpwindFirstENO3aHelper_execute_dim1_cuda
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
	FLOAT_TYPE* dst_DD1_ptr,
	FLOAT_TYPE* dst_DD2_ptr,
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
	const size_t dst_DD1_size,
	const size_t dst_DD2_size, 
	const bool saveDD,
	const bool approx4,
	const bool stripDD,
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
	if (saveDD) {
		if (approx4) {
			if (stripDD) {
				for (size_t blockIdx_y = 0; blockIdx_y < num_of_blocks_y; ++blockIdx_y) {
					for (size_t blockIdx_x = 0; blockIdx_x < num_of_blocks_x; ++blockIdx_x) {
						for (size_t threadIdx_z = 0; threadIdx_z < num_of_threads_z; ++threadIdx_z) {
							for (size_t threadIdx_y = 0; threadIdx_y < num_of_threads_y; ++threadIdx_y) {
								for (size_t threadIdx_x = 0; threadIdx_x < num_of_threads_x; ++threadIdx_x) {
									calc_D1toD3andDD_dim1_inline<FLOAT_TYPE, SaveDD, Approx4, StripDD>(
										dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dL3_ptr,
										dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, dst_dR3_ptr,
										dst_DD0_ptr, dst_DD1_ptr, dst_DD2_ptr,
										tmpBoundedSrc_ptr, dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
										num_of_slices, loop_length, first_dimension_loop_size, slice_length,
										stencil,
										dst_DD0_size, dst_DD1_size, dst_DD2_size,
										thread_length_z, thread_length_y, thread_length_x,
										blockIdx_y, blockIdx_x,
										num_of_threads_y, num_of_threads_x,
										threadIdx_z, threadIdx_y, threadIdx_x);
								}
							}
						}
					}
				}
			}
			else {
				for (size_t blockIdx_y = 0; blockIdx_y < num_of_blocks_y; ++blockIdx_y) {
					for (size_t blockIdx_x = 0; blockIdx_x < num_of_blocks_x; ++blockIdx_x) {
						for (size_t threadIdx_z = 0; threadIdx_z < num_of_threads_z; ++threadIdx_z) {
							for (size_t threadIdx_y = 0; threadIdx_y < num_of_threads_y; ++threadIdx_y) {
								for (size_t threadIdx_x = 0; threadIdx_x < num_of_threads_x; ++threadIdx_x) {
									calc_D1toD3andDD_dim1_inline<FLOAT_TYPE, SaveDD, Approx4, noStripDD>(
										dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dL3_ptr,
										dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, dst_dR3_ptr,
										dst_DD0_ptr, dst_DD1_ptr, dst_DD2_ptr,
										tmpBoundedSrc_ptr, dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
										num_of_slices, loop_length, first_dimension_loop_size, slice_length,
										stencil,
										dst_DD0_size, dst_DD1_size, dst_DD2_size,
										thread_length_z, thread_length_y, thread_length_x,
										blockIdx_y, blockIdx_x,
										num_of_threads_y, num_of_threads_x,
										threadIdx_z, threadIdx_y, threadIdx_x);
								}
							}
						}
					}
				}
			}
		}
		else {
			if (stripDD) {
				for (size_t blockIdx_y = 0; blockIdx_y < num_of_blocks_y; ++blockIdx_y) {
					for (size_t blockIdx_x = 0; blockIdx_x < num_of_blocks_x; ++blockIdx_x) {
						for (size_t threadIdx_z = 0; threadIdx_z < num_of_threads_z; ++threadIdx_z) {
							for (size_t threadIdx_y = 0; threadIdx_y < num_of_threads_y; ++threadIdx_y) {
								for (size_t threadIdx_x = 0; threadIdx_x < num_of_threads_x; ++threadIdx_x) {
									calc_D1toD3andDD_dim1_inline<FLOAT_TYPE, SaveDD, noApprox4, StripDD>(
										dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dL3_ptr,
										dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, dst_dR3_ptr,
										dst_DD0_ptr, dst_DD1_ptr, dst_DD2_ptr,
										tmpBoundedSrc_ptr, dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
										num_of_slices, loop_length, first_dimension_loop_size, slice_length,
										stencil,
										dst_DD0_size, dst_DD1_size, dst_DD2_size,
										thread_length_z, thread_length_y, thread_length_x,
										blockIdx_y, blockIdx_x,
										num_of_threads_y, num_of_threads_x,
										threadIdx_z, threadIdx_y, threadIdx_x);
								}
							}
						}
					}
				}
			}
			else {
				for (size_t blockIdx_y = 0; blockIdx_y < num_of_blocks_y; ++blockIdx_y) {
					for (size_t blockIdx_x = 0; blockIdx_x < num_of_blocks_x; ++blockIdx_x) {
						for (size_t threadIdx_z = 0; threadIdx_z < num_of_threads_z; ++threadIdx_z) {
							for (size_t threadIdx_y = 0; threadIdx_y < num_of_threads_y; ++threadIdx_y) {
								for (size_t threadIdx_x = 0; threadIdx_x < num_of_threads_x; ++threadIdx_x) {
									calc_D1toD3andDD_dim1_inline<FLOAT_TYPE, SaveDD, noApprox4, noStripDD>(
										dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dL3_ptr,
										dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, dst_dR3_ptr,
										dst_DD0_ptr, dst_DD1_ptr, dst_DD2_ptr,
										tmpBoundedSrc_ptr, dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
										num_of_slices, loop_length, first_dimension_loop_size, slice_length,
										stencil,
										dst_DD0_size, dst_DD1_size, dst_DD2_size,
										thread_length_z, thread_length_y, thread_length_x,
										blockIdx_y, blockIdx_x,
										num_of_threads_y, num_of_threads_x,
										threadIdx_z, threadIdx_y, threadIdx_x);
								}
							}
						}
					}
				}
			}
		}
	}
	else {
		if (approx4) {
			for (size_t blockIdx_y = 0; blockIdx_y < num_of_blocks_y; ++blockIdx_y) {
				for (size_t blockIdx_x = 0; blockIdx_x < num_of_blocks_x; ++blockIdx_x) {
					for (size_t threadIdx_z = 0; threadIdx_z < num_of_threads_z; ++threadIdx_z) {
						for (size_t threadIdx_y = 0; threadIdx_y < num_of_threads_y; ++threadIdx_y) {
							for (size_t threadIdx_x = 0; threadIdx_x < num_of_threads_x; ++threadIdx_x) {
								calc_D1toD3andDD_dim1_inline<FLOAT_TYPE, noSaveDD, Approx4, noStripDD>(
									dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dL3_ptr,
									dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, dst_dR3_ptr,
									dst_DD0_ptr, dst_DD1_ptr, dst_DD2_ptr,
									tmpBoundedSrc_ptr, dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
									num_of_slices, loop_length, first_dimension_loop_size, slice_length,
									stencil,
									dst_DD0_size, dst_DD1_size, dst_DD2_size,
									thread_length_z, thread_length_y, thread_length_x,
									blockIdx_y, blockIdx_x,
									num_of_threads_y, num_of_threads_x,
									threadIdx_z, threadIdx_y, threadIdx_x);
							}
						}
					}
				}
			}
		}
		else {
			for (size_t blockIdx_y = 0; blockIdx_y < num_of_blocks_y; ++blockIdx_y) {
				for (size_t blockIdx_x = 0; blockIdx_x < num_of_blocks_x; ++blockIdx_x) {
					for (size_t threadIdx_z = 0; threadIdx_z < num_of_threads_z; ++threadIdx_z) {
						for (size_t threadIdx_y = 0; threadIdx_y < num_of_threads_y; ++threadIdx_y) {
							for (size_t threadIdx_x = 0; threadIdx_x < num_of_threads_x; ++threadIdx_x) {
								calc_D1toD3andDD_dim1_inline<FLOAT_TYPE, noSaveDD, noApprox4, noStripDD>(
									dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dL3_ptr,
									dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, dst_dR3_ptr,
									dst_DD0_ptr, dst_DD1_ptr, dst_DD2_ptr,
									tmpBoundedSrc_ptr, dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
									num_of_slices, loop_length, first_dimension_loop_size, slice_length,
									stencil,
									dst_DD0_size, dst_DD1_size, dst_DD2_size,
									thread_length_z, thread_length_y, thread_length_x,
									blockIdx_y, blockIdx_x,
									num_of_threads_y, num_of_threads_x,
									threadIdx_z, threadIdx_y, threadIdx_x);
							}
						}
					}
				}
			}
		}
	}
}

void UpwindFirstENO3aHelper_execute_dimLET2_cuda
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
	FLOAT_TYPE* dst_DD1_ptr,
	FLOAT_TYPE* dst_DD2_ptr,
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

	if (saveDD) {
		if (approx4) {
			if (stripDD) {
				for (size_t blockIdx_y = 0; blockIdx_y < num_of_blocks_y; ++blockIdx_y) {
					for (size_t blockIdx_x = 0; blockIdx_x < num_of_blocks_x; ++blockIdx_x) {
						for (size_t threadIdx_z = 0; threadIdx_z < num_of_threads_z; ++threadIdx_z) {
							for (size_t threadIdx_y = 0; threadIdx_y < num_of_threads_y; ++threadIdx_y) {
								for (size_t threadIdx_x = 0; threadIdx_x < num_of_threads_x; ++threadIdx_x) {
									calc_D1toD3andDD_dimLET2_inline<FLOAT_TYPE, Approx4, StripDD>(
										dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dL3_ptr,
										dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, dst_dR3_ptr,
										dst_DD0_ptr, dst_DD1_ptr, dst_DD2_ptr,
										tmpBoundedSrc_ptr, dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
										num_of_slices, loop_length, first_dimension_loop_size,
										num_of_strides, num_of_dLdR_in_slice, slice_length,
										thread_length_z, thread_length_y, thread_length_x,
										blockIdx_y, blockIdx_x,
										num_of_threads_y, num_of_threads_x,
										threadIdx_z, threadIdx_y, threadIdx_x);
								}
							}
						}
					}
				}
			}
			else {
				for (size_t blockIdx_y = 0; blockIdx_y < num_of_blocks_y; ++blockIdx_y) {
					for (size_t blockIdx_x = 0; blockIdx_x < num_of_blocks_x; ++blockIdx_x) {
						for (size_t threadIdx_z = 0; threadIdx_z < num_of_threads_z; ++threadIdx_z) {
							for (size_t threadIdx_y = 0; threadIdx_y < num_of_threads_y; ++threadIdx_y) {
								for (size_t threadIdx_x = 0; threadIdx_x < num_of_threads_x; ++threadIdx_x) {
									calc_D1toD3andDD_dimLET2_inline<FLOAT_TYPE, Approx4, noStripDD>(
										dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dL3_ptr,
										dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, dst_dR3_ptr,
										dst_DD0_ptr, dst_DD1_ptr, dst_DD2_ptr,
										tmpBoundedSrc_ptr, dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
										num_of_slices, loop_length, first_dimension_loop_size,
										num_of_strides, num_of_dLdR_in_slice, slice_length,
										thread_length_z, thread_length_y, thread_length_x,
										blockIdx_y, blockIdx_x,
										num_of_threads_y, num_of_threads_x,
										threadIdx_z, threadIdx_y, threadIdx_x);
								}
							}
						}
					}
				}
			}
		}
		else {
			if (stripDD) {
				for (size_t blockIdx_y = 0; blockIdx_y < num_of_blocks_y; ++blockIdx_y) {
					for (size_t blockIdx_x = 0; blockIdx_x < num_of_blocks_x; ++blockIdx_x) {
						for (size_t threadIdx_z = 0; threadIdx_z < num_of_threads_z; ++threadIdx_z) {
							for (size_t threadIdx_y = 0; threadIdx_y < num_of_threads_y; ++threadIdx_y) {
								for (size_t threadIdx_x = 0; threadIdx_x < num_of_threads_x; ++threadIdx_x) {
									calc_D1toD3andDD_dimLET2_inline<FLOAT_TYPE, noApprox4, StripDD>(
										dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dL3_ptr,
										dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, dst_dR3_ptr,
										dst_DD0_ptr, dst_DD1_ptr, dst_DD2_ptr,
										tmpBoundedSrc_ptr, dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
										num_of_slices, loop_length, first_dimension_loop_size,
										num_of_strides, num_of_dLdR_in_slice, slice_length,
										thread_length_z, thread_length_y, thread_length_x,
										blockIdx_y, blockIdx_x,
										num_of_threads_y, num_of_threads_x,
										threadIdx_z, threadIdx_y, threadIdx_x);
								}
							}
						}
					}
				}
			}
			else {
				for (size_t blockIdx_y = 0; blockIdx_y < num_of_blocks_y; ++blockIdx_y) {
					for (size_t blockIdx_x = 0; blockIdx_x < num_of_blocks_x; ++blockIdx_x) {
						for (size_t threadIdx_z = 0; threadIdx_z < num_of_threads_z; ++threadIdx_z) {
							for (size_t threadIdx_y = 0; threadIdx_y < num_of_threads_y; ++threadIdx_y) {
								for (size_t threadIdx_x = 0; threadIdx_x < num_of_threads_x; ++threadIdx_x) {
									calc_D1toD3andDD_dimLET2_inline<FLOAT_TYPE, noApprox4, noStripDD>(
										dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dL3_ptr,
										dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, dst_dR3_ptr,
										dst_DD0_ptr, dst_DD1_ptr, dst_DD2_ptr,
										tmpBoundedSrc_ptr, dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
										num_of_slices, loop_length, first_dimension_loop_size,
										num_of_strides, num_of_dLdR_in_slice, slice_length,
										thread_length_z, thread_length_y, thread_length_x,
										blockIdx_y, blockIdx_x,
										num_of_threads_y, num_of_threads_x,
										threadIdx_z, threadIdx_y, threadIdx_x);
								}
							}
						}
					}
				}
			}
		}
	}
	else {
		if (approx4) {
			for (size_t blockIdx_y = 0; blockIdx_y < num_of_blocks_y; ++blockIdx_y) {
				for (size_t blockIdx_x = 0; blockIdx_x < num_of_blocks_x; ++blockIdx_x) {
					for (size_t threadIdx_z = 0; threadIdx_z < num_of_threads_z; ++threadIdx_z) {
						for (size_t threadIdx_y = 0; threadIdx_y < num_of_threads_y; ++threadIdx_y) {
							for (size_t threadIdx_x = 0; threadIdx_x < num_of_threads_x; ++threadIdx_x) {
								calc_D1toD3_dimLET2_inline<FLOAT_TYPE, Approx4>(
									dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dL3_ptr,
									dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, dst_dR3_ptr,
									tmpBoundedSrc_ptr, dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
									num_of_slices, loop_length, first_dimension_loop_size,
									slice_length,
									thread_length_z, thread_length_y, thread_length_x,
									blockIdx_y, blockIdx_x,
									num_of_threads_y, num_of_threads_x,
									threadIdx_z, threadIdx_y, threadIdx_x);
							}
						}
					}
				}
			}
		}
		else {
			for (size_t blockIdx_y = 0; blockIdx_y < num_of_blocks_y; ++blockIdx_y) {
				for (size_t blockIdx_x = 0; blockIdx_x < num_of_blocks_x; ++blockIdx_x) {
					for (size_t threadIdx_z = 0; threadIdx_z < num_of_threads_z; ++threadIdx_z) {
						for (size_t threadIdx_y = 0; threadIdx_y < num_of_threads_y; ++threadIdx_y) {
							for (size_t threadIdx_x = 0; threadIdx_x < num_of_threads_x; ++threadIdx_x) {
								calc_D1toD3_dimLET2_inline<FLOAT_TYPE, noApprox4>(
									dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dL3_ptr,
									dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, dst_dR3_ptr,
									tmpBoundedSrc_ptr, dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
									num_of_slices, loop_length, first_dimension_loop_size,
									slice_length,
									thread_length_z, thread_length_y, thread_length_x,
									blockIdx_y, blockIdx_x,
									num_of_threads_y, num_of_threads_x,
									threadIdx_z, threadIdx_y, threadIdx_x);
							}
						}
					}
				}
			}
		}
	}
}

#endif /* !defined(WITH_GPU) */
