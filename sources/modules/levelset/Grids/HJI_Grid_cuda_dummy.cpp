#include <typedef.hpp>
#include <cuda_macro.hpp>
#include <algorithm>
#include "HJI_Grid_cuda.hpp"

#if !defined(WITH_GPU)

void memcpy2d(
	void* dst,
	size_t dpitch,
	const void* src,
	size_t spitch,
	size_t width,
	size_t height
) {
	size_t dst_offset = 0;
	size_t src_offset = 0;
	for (size_t i = 0; i < height; ++i) {
		memcpy((void*)((uint8_t*)dst + dst_offset), (const void*)((uint8_t*)src + src_offset), width);
		dst_offset += dpitch;
		src_offset += spitch;
	}
}

void HJI_Grid_calc_xs_execute_cuda_dim0
(
	FLOAT_TYPE* xs_ptr,
	FLOAT_TYPE* v_ptr,
	const size_t target_dimension_begin_index,
	const size_t loop_length,
	const size_t target_dimension_loop_size,
	const size_t inner_dimensions_loop_size,
	const size_t block_id_y,
	const size_t block_id_x,
	const size_t block_length_y,
	const size_t block_length_x,
	const size_t thread_id_y,
	const size_t thread_id_x,
	const size_t thread_length_y,
	const size_t thread_length_x
) {
	switch (dimension) {
	case 0:
		memcpy2d(xs_ptr, target_dimension_loop_size, v_ptr, target_dimension_loop_size, target_dimension_loop_size, loop_length);
		break;
	case 1:
		memcpy2d(xs_ptr, target_dimension_loop_size, v_ptr, target_dimension_loop_size, target_dimension_loop_size, loop_length);
		break;
	}

	size_t index = thread_id + block_id * block_length;
	const size_t target_dimension_size = target_dimension_loop_size * inner_dimensions_loop_size;
	size_t target_dimension_offset = target_dimension_begin_index * inner_dimensions_loop_size + index;
	while (target_dimension_offset < )
	size_t target_dimension_index = target_dimension_begin_index;
	while (index < loop_length) {
		xs_ptr[index] = v_ptr[target_dimension_index];
		index += thread_length;
		if (index + target_dimension_offset >= target_dimension_size) {
			target_dimension_index = 0;
		}
	}

}


void HJI_Grid_calc_xs_execute_cuda
(
	beacls::UVec& x_uvec,
	const beacls::UVec& v_uvec,
	const size_t dimension,
	const size_t start_index,
	const size_t loop_length,
	const size_t stride,
	const std::vector<size_t>& Ns
) {
	const size_t num_of_dimensions = Ns.size();
	const size_t modified_length = loop_length;
	beacls::synchronizeUVec(v_uvec);
	//! Transposing copy
	size_t inner_dimensions_loop_size = 1;
	size_t target_dimension_loop_size = 1;
	size_t outer_dimensions_loop_size = 1;
	const beacls::FloatVec* v_ptr = beacls::UVec_<FLOAT_TYPE>(v_uvec).vec();
	beacls::FloatVec* xs_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvec).vec();
	for (size_t target_dimension = 0; target_dimension < dimension; ++target_dimension) {
		inner_dimensions_loop_size *= Ns[target_dimension];
	}
	target_dimension_loop_size = Ns[dimension];
	for (size_t target_dimension = dimension + 1; target_dimension < num_of_dimensions; ++target_dimension) {
		outer_dimensions_loop_size *= Ns[target_dimension];
	}
	if (inner_dimensions_loop_size == 1) {
		size_t outer_dimensions_loop_begin = (size_t)(std::ceil((double)start_index / inner_dimensions_loop_size));
		size_t outer_dimensions_loop_end = (size_t)(std::floor((double)(start_index + modified_length) / inner_dimensions_loop_size));

		for (size_t outer_dimensions_loop_index = 0; outer_dimensions_loop_index < (outer_dimensions_loop_end - outer_dimensions_loop_begin); ++outer_dimensions_loop_index) {
			size_t outer_index_term = outer_dimensions_loop_index * target_dimension_loop_size;
			std::copy(v_ptr->cbegin(), v_ptr->cend(), xs_ptr->begin() + outer_index_term);
		}
	}
	else {
		const size_t loop_index_div_inner_size = start_index / inner_dimensions_loop_size;
		const size_t inner_dimensions_loop_index = start_index % inner_dimensions_loop_size;
		const size_t outer_dimensions_loop_begin = loop_index_div_inner_size / target_dimension_loop_size;
		const size_t target_dimension_loop_begin = loop_index_div_inner_size % target_dimension_loop_size;
		const size_t outer_dimensions_loop_end = (size_t)(std::floor((double)(start_index + modified_length) / inner_dimensions_loop_size)) / target_dimension_loop_size;
		const size_t target_dimension_loop_end = (size_t)(std::floor((double)(start_index + modified_length) / inner_dimensions_loop_size)) % target_dimension_loop_size;
		{
			size_t outer_index_term = outer_dimensions_loop_begin * target_dimension_loop_size * inner_dimensions_loop_size;
			for (size_t target_dimension_loop_index = target_dimension_loop_begin; target_dimension_loop_index < target_dimension_loop_size; ++target_dimension_loop_index) {
				const size_t target_index_term = target_dimension_loop_index * inner_dimensions_loop_size;
				const size_t dst_offset = outer_index_term + target_index_term;
				const FLOAT_TYPE vs_val = (*v_ptr)[target_dimension_loop_index];
				std::fill(xs_ptr->begin() + dst_offset, xs_ptr->begin() + dst_offset + inner_dimensions_loop_size, vs_val);
			}
		}
		for (size_t outer_dimensions_loop_index = outer_dimensions_loop_begin + 1; outer_dimensions_loop_index < outer_dimensions_loop_end; ++outer_dimensions_loop_index) {
			size_t outer_index_term = outer_dimensions_loop_index * target_dimension_loop_size * inner_dimensions_loop_size;
			for (size_t target_dimension_loop_index = 0; target_dimension_loop_index < target_dimension_loop_size; ++target_dimension_loop_index) {
				const size_t target_index_term = target_dimension_loop_index * inner_dimensions_loop_size;
				const size_t dst_offset = outer_index_term + target_index_term;
				const FLOAT_TYPE vs_val = (*v_ptr)[target_dimension_loop_index];
				std::fill(xs_ptr->begin() + dst_offset, xs_ptr->begin() + dst_offset + inner_dimensions_loop_size, vs_val);
			}
		}
		{
			size_t outer_index_term = outer_dimensions_loop_end * target_dimension_loop_size * inner_dimensions_loop_size;
			for (size_t target_dimension_loop_index = 0; target_dimension_loop_index < target_dimension_loop_end; ++target_dimension_loop_index) {
				const size_t target_index_term = target_dimension_loop_index * inner_dimensions_loop_size;
				const size_t dst_offset = outer_index_term + target_index_term;
				const FLOAT_TYPE vs_val = (*v_ptr)[target_dimension_loop_index];
				std::fill(xs_ptr->begin() + dst_offset, xs_ptr->begin() + dst_offset + inner_dimensions_loop_size, vs_val);
			}
		}
	}
}
#endif /* !defined(WITH_GPU)  */
