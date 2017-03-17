#include <levelset/BoundaryCondition/AddGhostPeriodic.hpp>
#include <vector>
#include <iterator>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <cstring>
#include <typeinfo>
#include "AddGhostPeriodic_impl.hpp"
//#define PARALLEL_Y
using namespace levelset;


AddGhostPeriodic_impl::AddGhostPeriodic_impl(
){
}
AddGhostPeriodic_impl::~AddGhostPeriodic_impl() {
}
bool AddGhostPeriodic_impl::execute(
	const FLOAT_TYPE* src,
	beacls::UVec& dst_buffer,
	const FLOAT_TYPE*& dst_ptr,
	const size_t width,
	const size_t outer_dimensions_loop_index,
	const size_t target_dimension_loop_size,
	const size_t target_dimension_loop_index,
	const size_t inner_dimensions_loop_size,
	const size_t inner_dimensions_loop_index,
	const size_t first_dimension_loop_size
) const {
	size_t src_target_dimension_loop_size = (target_dimension_loop_size - width * 2);

	size_t src_outer_index_term = outer_dimensions_loop_index * src_target_dimension_loop_size * inner_dimensions_loop_size*first_dimension_loop_size;
	size_t src_inter_index_term = inner_dimensions_loop_index * first_dimension_loop_size;
	size_t src_target_index_term;
	if (target_dimension_loop_index < width) {
		src_target_index_term = (target_dimension_loop_index + src_target_dimension_loop_size - width) * inner_dimensions_loop_size * first_dimension_loop_size;
	}
	else if (target_dimension_loop_index >= (target_dimension_loop_size - width)) {
		src_target_index_term = (target_dimension_loop_index - src_target_dimension_loop_size - width) * inner_dimensions_loop_size * first_dimension_loop_size;
	}
	else {
		src_target_index_term = (target_dimension_loop_index - width) * inner_dimensions_loop_size * first_dimension_loop_size;
	}
	size_t index_term = src_outer_index_term + src_inter_index_term + src_target_index_term;
	if (dst_buffer.type() == beacls::UVecType_Cuda) {
		copyHostPtrToUVecAsync(dst_buffer, src + index_term, dst_buffer.size());
		dst_ptr = beacls::UVec_<FLOAT_TYPE>(dst_buffer).ptr();
	}
	else {
		dst_ptr = src + index_term;
	}

	return true;
}


bool AddGhostPeriodic_impl::execute(
	const FLOAT_TYPE* src,
	beacls::UVec& dst_buffer,
	beacls::UVec& tmp_buffer,
	std::vector<std::vector<std::vector<const FLOAT_TYPE*> > >& dst_ptrsss,
	const size_t width,
	const size_t dim,
	const size_t target_dimension_loop_size,
	const size_t inner_dimensions_loop_size,
	const size_t first_dimension_loop_size,
	const size_t loop_begin,
	const size_t prologue_loop_dst_offset,
	const size_t num_of_slices,
	const size_t loop_length,
	const size_t num_of_strides,
	const size_t stencil

) const {
	FLOAT_TYPE* tmp_buffer_ptr = NULL;
	const size_t src_target_dimension_loop_size = (target_dimension_loop_size - width * 2);
	const size_t total_buffer_size = loop_length*num_of_slices*num_of_strides*first_dimension_loop_size;
	const bool cuda = (dst_buffer.type() == beacls::UVecType_Cuda);
	if (cuda) {
		if (tmp_buffer.depth() != dst_buffer.depth()
			|| tmp_buffer.type() != beacls::UVecType_Vector)
			tmp_buffer = beacls::UVec(dst_buffer.depth(), beacls::UVecType_Vector, total_buffer_size);
		else if (tmp_buffer.size() < total_buffer_size)
			tmp_buffer.resize(total_buffer_size);
		tmp_buffer_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_buffer).ptr();
	}
	size_t loop_index_div_base;
	size_t target_dimension_loop_index_base;
	size_t inner_dimensions_loop_index_base;
	if (dim == 1) {
		loop_index_div_base = loop_begin / src_target_dimension_loop_size;
		target_dimension_loop_index_base = loop_begin % src_target_dimension_loop_size;
		inner_dimensions_loop_index_base = 0;
	}
	else {
		loop_index_div_base = loop_begin / inner_dimensions_loop_size;
		target_dimension_loop_index_base = 0;
		inner_dimensions_loop_index_base = loop_begin % inner_dimensions_loop_size;
	}
#if defined(PARALLEL_Y)
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int stride_index = 0; stride_index < (int)num_of_strides; ++stride_index) {
		size_t center_block_copy_src_begin = std::numeric_limits<size_t>::max();
		size_t center_block_copy_dst_begin = std::numeric_limits<size_t>::max();
		size_t center_block_copy_src_end = 0;
		size_t top_block_copy_src_begin = std::numeric_limits<size_t>::max();
		size_t top_block_copy_dst_begin = std::numeric_limits<size_t>::max();
		size_t top_block_copy_src_end = 0;
		size_t bottom_block_copy_src_begin = std::numeric_limits<size_t>::max();
		size_t bottom_block_copy_dst_begin = std::numeric_limits<size_t>::max();
		size_t bottom_block_copy_src_end = 0;
#else
	size_t center_block_copy_src_begin = std::numeric_limits<size_t>::max();
	size_t center_block_copy_dst_begin = std::numeric_limits<size_t>::max();
	size_t center_block_copy_src_end = 0;
	size_t top_block_copy_src_begin = std::numeric_limits<size_t>::max();
	size_t top_block_copy_dst_begin = std::numeric_limits<size_t>::max();
	size_t top_block_copy_src_end = 0;
	size_t bottom_block_copy_src_begin = std::numeric_limits<size_t>::max();
	size_t bottom_block_copy_dst_begin = std::numeric_limits<size_t>::max();
	size_t bottom_block_copy_src_end = 0;
	for (size_t stride_index = 0; stride_index < num_of_strides; ++stride_index) {
#endif
		for (size_t slice_index = 0; slice_index < num_of_slices; ++slice_index) {
			for (size_t loop_index = 0; loop_index < loop_length; ++loop_index) {
				size_t target_dimension_loop_index;
				size_t inner_dimensions_loop_index;
				size_t outer_dimensions_loop_index;
				if (dim == 1) {
					const size_t loop_index_div_target_size = loop_index_div_base;
					target_dimension_loop_index = target_dimension_loop_index_base + loop_index + prologue_loop_dst_offset;
					inner_dimensions_loop_index = loop_index_div_target_size % inner_dimensions_loop_size;
					outer_dimensions_loop_index = loop_index_div_target_size / inner_dimensions_loop_size + stride_index*num_of_slices + slice_index;
				}
				else if (dim == 2) {
					const size_t loop_index_div_inner_size = loop_index_div_base;
					inner_dimensions_loop_index = inner_dimensions_loop_index_base + loop_index;
					outer_dimensions_loop_index = slice_index + loop_index_div_inner_size / target_dimension_loop_size;
					target_dimension_loop_index = stride_index + loop_index_div_inner_size % target_dimension_loop_size;
				}
				else {
					const size_t loop_index_div_inner_size = loop_index_div_base;
					inner_dimensions_loop_index = slice_index * loop_length + inner_dimensions_loop_index_base + loop_index;
					outer_dimensions_loop_index = loop_index_div_inner_size / target_dimension_loop_size;
					target_dimension_loop_index = stride_index + loop_index_div_inner_size % target_dimension_loop_size;
				}
				if (target_dimension_loop_index < target_dimension_loop_size + stencil) {

					const size_t src_outer_index_term = outer_dimensions_loop_index * src_target_dimension_loop_size * inner_dimensions_loop_size*first_dimension_loop_size;
					const size_t src_inter_index_term = inner_dimensions_loop_index * first_dimension_loop_size;
					size_t index_term;
					if (target_dimension_loop_index < width) {
						const size_t src_target_index_term = (target_dimension_loop_index + src_target_dimension_loop_size - width) * inner_dimensions_loop_size * first_dimension_loop_size;
						index_term = src_outer_index_term + src_inter_index_term + src_target_index_term;
						if (cuda) {
							const size_t dst_offset = ((stride_index*num_of_slices + slice_index)*loop_length + loop_index)*first_dimension_loop_size;
							if (top_block_copy_src_begin > index_term) {
								top_block_copy_dst_begin = dst_offset;
								top_block_copy_src_begin = index_term;
							}
							top_block_copy_src_end = index_term + first_dimension_loop_size;
						}
					}
					else if (target_dimension_loop_index >= (target_dimension_loop_size - width)) {
						const size_t src_target_index_term = (target_dimension_loop_index - src_target_dimension_loop_size - width) * inner_dimensions_loop_size * first_dimension_loop_size;
						index_term = src_outer_index_term + src_inter_index_term + src_target_index_term;
						if (cuda) {
							const size_t dst_offset = ((stride_index*num_of_slices + slice_index)*loop_length + loop_index)*first_dimension_loop_size;
							if (bottom_block_copy_src_begin > index_term) {
								bottom_block_copy_dst_begin = dst_offset;
								bottom_block_copy_src_begin = index_term;
							}
							bottom_block_copy_src_end = index_term + first_dimension_loop_size;
						}
					}
					else {
						const size_t src_target_index_term = (target_dimension_loop_index - width) * inner_dimensions_loop_size * first_dimension_loop_size;
						index_term = src_outer_index_term + src_inter_index_term + src_target_index_term;
						if (cuda) {
							const size_t dst_offset = ((stride_index*num_of_slices + slice_index)*loop_length + loop_index)*first_dimension_loop_size;
							if (center_block_copy_src_begin > index_term) {
								center_block_copy_dst_begin = dst_offset;
								center_block_copy_src_begin = index_term;
							}
							center_block_copy_src_end = index_term + first_dimension_loop_size;
						}
					}
					if (!cuda) {
						dst_ptrsss[slice_index][loop_index][stride_index] = src + index_term;
					}
				}
			}
			if (cuda && dim == 1) {
				if (bottom_block_copy_src_end > bottom_block_copy_src_begin) {
					memcpy(tmp_buffer_ptr + bottom_block_copy_dst_begin, src + bottom_block_copy_src_begin, (bottom_block_copy_src_end - bottom_block_copy_src_begin) * sizeof(FLOAT_TYPE));
					bottom_block_copy_src_begin = std::numeric_limits<size_t>::max();
					bottom_block_copy_dst_begin = std::numeric_limits<size_t>::max();
					bottom_block_copy_src_end = 0;
				}
				if (center_block_copy_src_end > center_block_copy_src_begin) {
					memcpy(tmp_buffer_ptr + center_block_copy_dst_begin, src + center_block_copy_src_begin, (center_block_copy_src_end - center_block_copy_src_begin) * sizeof(FLOAT_TYPE));
					center_block_copy_src_begin = std::numeric_limits<size_t>::max();
					center_block_copy_dst_begin = std::numeric_limits<size_t>::max();
					center_block_copy_src_end = 0;
				}
				if (top_block_copy_src_end > top_block_copy_src_begin) {
					memcpy(tmp_buffer_ptr + top_block_copy_dst_begin, src + top_block_copy_src_begin, (top_block_copy_src_end - top_block_copy_src_begin) * sizeof(FLOAT_TYPE));
					top_block_copy_src_begin = std::numeric_limits<size_t>::max();
					top_block_copy_dst_begin = std::numeric_limits<size_t>::max();
					top_block_copy_src_end = 0;
				}
			}
		}
#if defined(PARALLEL_Y)
		if (cuda && (dim != 1)) {
#else
		if (cuda && (dim != 1) && (dim != 2)) {
#endif
			if (bottom_block_copy_src_end > bottom_block_copy_src_begin) {
				memcpy(tmp_buffer_ptr + bottom_block_copy_dst_begin, src + bottom_block_copy_src_begin, (bottom_block_copy_src_end - bottom_block_copy_src_begin) * sizeof(FLOAT_TYPE));
				bottom_block_copy_src_begin = std::numeric_limits<size_t>::max();
				bottom_block_copy_dst_begin = std::numeric_limits<size_t>::max();
				bottom_block_copy_src_end = 0;
			}
			if (center_block_copy_src_end > center_block_copy_src_begin) {
				memcpy(tmp_buffer_ptr + center_block_copy_dst_begin, src + center_block_copy_src_begin, (center_block_copy_src_end - center_block_copy_src_begin) * sizeof(FLOAT_TYPE));
				center_block_copy_src_begin = std::numeric_limits<size_t>::max();
				center_block_copy_dst_begin = std::numeric_limits<size_t>::max();
				center_block_copy_src_end = 0;
			}
			if (top_block_copy_src_end > top_block_copy_src_begin) {
				memcpy(tmp_buffer_ptr + top_block_copy_dst_begin, src + top_block_copy_src_begin, (top_block_copy_src_end - top_block_copy_src_begin) * sizeof(FLOAT_TYPE));
				top_block_copy_src_begin = std::numeric_limits<size_t>::max();
				top_block_copy_dst_begin = std::numeric_limits<size_t>::max();
				top_block_copy_src_end = 0;
			}
		}
	}
#if !defined(PARALLEL_Y)
	if (cuda && dim == 2) {
		if (bottom_block_copy_src_end > bottom_block_copy_src_begin) {
			memcpy(tmp_buffer_ptr + bottom_block_copy_dst_begin, src + bottom_block_copy_src_begin, (bottom_block_copy_src_end - bottom_block_copy_src_begin) * sizeof(FLOAT_TYPE));
			bottom_block_copy_src_begin = std::numeric_limits<size_t>::max();
			bottom_block_copy_dst_begin = std::numeric_limits<size_t>::max();
			bottom_block_copy_src_end = 0;
		}
		if (center_block_copy_src_end > center_block_copy_src_begin) {
			memcpy(tmp_buffer_ptr + center_block_copy_dst_begin, src + center_block_copy_src_begin, (center_block_copy_src_end - center_block_copy_src_begin) * sizeof(FLOAT_TYPE));
			center_block_copy_src_begin = std::numeric_limits<size_t>::max();
			center_block_copy_dst_begin = std::numeric_limits<size_t>::max();
			center_block_copy_src_end = 0;
		}
		if (top_block_copy_src_end > top_block_copy_src_begin) {
			memcpy(tmp_buffer_ptr + top_block_copy_dst_begin, src + top_block_copy_src_begin, (top_block_copy_src_end - top_block_copy_src_begin) * sizeof(FLOAT_TYPE));
			top_block_copy_src_begin = std::numeric_limits<size_t>::max();
			top_block_copy_dst_begin = std::numeric_limits<size_t>::max();
			top_block_copy_src_end = 0;
		}
	}
#endif
	if (cuda) {
		beacls::copyHostPtrToUVecAsync(dst_buffer, beacls::UVec_<FLOAT_TYPE>(tmp_buffer).ptr(), tmp_buffer.size());
		FLOAT_TYPE* dst_buffer_ptr = beacls::UVec_<FLOAT_TYPE>(dst_buffer).ptr();
		for (size_t slice_index = 0; slice_index < num_of_slices; ++slice_index) {
			for (size_t loop_index = 0; loop_index < loop_length; ++loop_index) {
				for (size_t stride_index = 0; stride_index < num_of_strides; ++stride_index) {
					dst_ptrsss[slice_index][loop_index][stride_index] = dst_buffer_ptr + (stride_index*loop_length*num_of_slices + slice_index*loop_length + loop_index)*first_dimension_loop_size;
				}
			}
		}
	}
	return true;
}

bool AddGhostPeriodic_impl::execute(
	const FLOAT_TYPE* src,
	beacls::UVec& dst,
	const size_t width,
	const size_t outer_dimensions_loop_index,
	const size_t target_dimension_loop_size
) const {
	// create appropriately sized output array
	beacls::UVecDepth depth = beacls::type_to_depth<FLOAT_TYPE>();
	if (dst.size() != target_dimension_loop_size) {
		if (dst.type() == beacls::UVecType_Invalid)
			dst = beacls::UVec(depth, beacls::UVecType_Vector, target_dimension_loop_size);
		else if (dst.size() != target_dimension_loop_size) dst.resize(target_dimension_loop_size);
	}

	size_t src_target_dimension_loop_size = (target_dimension_loop_size - width * 2);
	size_t src_outer_index_term = outer_dimensions_loop_index * src_target_dimension_loop_size;

	beacls::UVec tmp_uvec;
	FLOAT_TYPE* tmp_buffer_ptr;
	if (dst.type() == beacls::UVecType_Cuda) {
		tmp_uvec = beacls::UVec(depth, beacls::UVecType_Vector, target_dimension_loop_size);
		tmp_buffer_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_uvec).ptr();
	}
	else {
		tmp_buffer_ptr = beacls::UVec_<FLOAT_TYPE>(dst).ptr();
	}
	//! Copy intermediate value.
	memcpy(&tmp_buffer_ptr[width], &src[src_outer_index_term], src_target_dimension_loop_size * sizeof(FLOAT_TYPE));

	for (size_t target_dimension_loop_index = 0; target_dimension_loop_index < width; ++target_dimension_loop_index) {
		size_t dst_target_index_term = target_dimension_loop_index;
		size_t src_target_index_term = target_dimension_loop_index + src_target_dimension_loop_size - width;

		size_t src_index = src_outer_index_term + src_target_index_term;
		size_t dst_index = dst_target_index_term;

		tmp_buffer_ptr[dst_index] = src[src_index];
	}
	for (size_t target_dimension_loop_index = target_dimension_loop_size - width; target_dimension_loop_index < target_dimension_loop_size; ++target_dimension_loop_index) {
		size_t dst_target_index_term = target_dimension_loop_index;
		size_t src_target_index_term = target_dimension_loop_index - src_target_dimension_loop_size - width;
		size_t src_index = src_outer_index_term + src_target_index_term;
		size_t dst_index = dst_target_index_term;

		tmp_buffer_ptr[dst_index] = src[src_index];
	}
	if (dst.type() == beacls::UVecType_Cuda) {
		beacls::copyHostPtrToUVecAsync(dst, beacls::UVec_<FLOAT_TYPE>(tmp_uvec).ptr(), tmp_uvec.size());
	}
	return true;
}
bool AddGhostPeriodic_impl::execute(
	const FLOAT_TYPE* src,
	beacls::UVec& dst_buffer,
	beacls::UVec& tmp_buffer,
	const size_t width,
	const size_t outer_dimensions_loop_length,
	const size_t target_dimension_loop_size,
	const size_t first_dimension_loop_size,
	const size_t loop_begin_base,
	const size_t num_of_slices
)const {
	FLOAT_TYPE* tmp_buffer_ptr = NULL;
	bool cuda = (dst_buffer.type() == beacls::UVecType_Cuda);
	const size_t total_buffer_size = target_dimension_loop_size*outer_dimensions_loop_length*num_of_slices;
	const size_t slice_length = target_dimension_loop_size*outer_dimensions_loop_length;
	if (cuda) {
		if (tmp_buffer.depth() != dst_buffer.depth()
			|| tmp_buffer.type() != beacls::UVecType_Vector)
			tmp_buffer = beacls::UVec(dst_buffer.depth(), beacls::UVecType_Vector, total_buffer_size);
		else if (tmp_buffer.size() < total_buffer_size)
			tmp_buffer.resize(total_buffer_size);
		tmp_buffer_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_buffer).ptr();
	}
	else {
		tmp_buffer_ptr = beacls::UVec_<FLOAT_TYPE>(dst_buffer).ptr();
	}
#if defined(PARALLEL_Y)
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int slice_index = 0; slice_index < (int)num_of_slices; ++slice_index) {
#else
	for (size_t slice_index = 0; slice_index < num_of_slices; ++slice_index) {
#endif
		const size_t slice_offset = slice_index * slice_length;
		const size_t slice_loop_offset = slice_index * outer_dimensions_loop_length;

		for (size_t index = 0; index < outer_dimensions_loop_length; ++index) {
			size_t outer_dimensions_loop_index = index + loop_begin_base + slice_loop_offset;
			size_t local_index_term = index * (first_dimension_loop_size + width * 2) + slice_offset;

			size_t src_target_dimension_loop_size = (target_dimension_loop_size - width * 2);
			size_t src_outer_index_term = outer_dimensions_loop_index * src_target_dimension_loop_size;
			//! Copy intermediate value.
			memcpy(&tmp_buffer_ptr[width + local_index_term], &src[src_outer_index_term], src_target_dimension_loop_size * sizeof(FLOAT_TYPE));

			for (size_t target_dimension_loop_index = 0; target_dimension_loop_index < width; ++target_dimension_loop_index) {
				size_t dst_target_index_term = target_dimension_loop_index;
				size_t src_target_index_term = target_dimension_loop_index + src_target_dimension_loop_size - width;

				size_t src_index = src_outer_index_term + src_target_index_term;
				size_t dst_index = dst_target_index_term + local_index_term;

				tmp_buffer_ptr[dst_index] = src[src_index];
			}
			for (size_t target_dimension_loop_index = target_dimension_loop_size - width; target_dimension_loop_index < target_dimension_loop_size; ++target_dimension_loop_index) {
				size_t dst_target_index_term = target_dimension_loop_index;
				size_t src_target_index_term = target_dimension_loop_index - src_target_dimension_loop_size - width;
				size_t src_index = src_outer_index_term + src_target_index_term;
				size_t dst_index = dst_target_index_term + local_index_term;

				tmp_buffer_ptr[dst_index] = src[src_index];
			}
		}
	}
	if (dst_buffer.type() == beacls::UVecType_Cuda) {
		beacls::copyHostPtrToUVecAsync(dst_buffer, beacls::UVec_<FLOAT_TYPE>(tmp_buffer).ptr(), tmp_buffer.size());
	}
	return true;
}
AddGhostPeriodic::AddGhostPeriodic(
	) {
	pimpl = new AddGhostPeriodic_impl();
}
AddGhostPeriodic::~AddGhostPeriodic() {
	if (pimpl) delete pimpl;
}
bool AddGhostPeriodic::execute(
	const FLOAT_TYPE* src,
	beacls::UVec& dst_buffer,
	const FLOAT_TYPE*& dst_ptr,
	const size_t width,
	const size_t outer_dimensions_loop_index,
	const size_t target_dimension_loop_size,
	const size_t target_dimension_loop_index,
	const size_t inner_dimensions_loop_size,
	const size_t inner_dimensions_loop_index,
	const size_t first_dimension_loop_size
)const {
	if (pimpl) return pimpl->execute(
		src,
		dst_buffer, dst_ptr,
		width,
		outer_dimensions_loop_index,
		target_dimension_loop_size, target_dimension_loop_index,
		inner_dimensions_loop_size, inner_dimensions_loop_index,
		first_dimension_loop_size);
	else return false;
}

bool AddGhostPeriodic::execute(
	const FLOAT_TYPE* src,
	beacls::UVec& dst_buffer,
	beacls::UVec& tmp_buffer,
	std::vector<std::vector<std::vector<const FLOAT_TYPE*> > >& dst_ptrsss,
	const size_t width,
	const size_t dim,
	const size_t target_dimension_loop_size,
	const size_t inner_dimensions_loop_size,
	const size_t first_dimension_loop_size,
	const size_t loop_begin,
	const size_t prologue_loop_dst_offset,
	const size_t num_of_slices,
	const size_t loop_length,
	const size_t num_of_strides,
	const size_t stencil

) const {
	if (pimpl) return pimpl->execute(
		src,
		dst_buffer, tmp_buffer, dst_ptrsss,
		width,
		dim,
		target_dimension_loop_size,
		inner_dimensions_loop_size,
		first_dimension_loop_size,
		loop_begin, 
		prologue_loop_dst_offset,
		num_of_slices,
		loop_length, 
		num_of_strides,
		stencil);
	else return false;
}


bool AddGhostPeriodic::execute(
	const FLOAT_TYPE* src,
	beacls::UVec& dst,
	const size_t width,
	const size_t outer_dimensions_loop_index,
	const size_t target_dimension_loop_size
)const {
	if (pimpl) return pimpl->execute(src, dst, width, outer_dimensions_loop_index, target_dimension_loop_size);
	else return false;
}
bool AddGhostPeriodic::execute(
	const FLOAT_TYPE* src,
	beacls::UVec& dst,
	beacls::UVec& tmp_buffer,
	const size_t width,
	const size_t outer_dimensions_loop_length,
	const size_t target_dimension_loop_size,
	const size_t first_dimension_loop_size,
	const size_t loop_begin_base,
	const size_t num_of_slices
)const {
	if (pimpl) return pimpl->execute(src, dst, tmp_buffer, width, outer_dimensions_loop_length, target_dimension_loop_size, first_dimension_loop_size, loop_begin_base, num_of_slices);
	else return false;
}
bool AddGhostPeriodic::valid()const {
	if (pimpl) return pimpl->valid();
	else return false;
}
AddGhostPeriodic::AddGhostPeriodic(const AddGhostPeriodic& rhs) :
	pimpl(rhs.pimpl->clone())
{
}
bool AddGhostPeriodic::operator==(const AddGhostPeriodic& rhs) const {
	if (this == &rhs) return true;
	else if (!pimpl) {
		if (!rhs.pimpl) return true;
		else return false;
	}
	else {
		if (!rhs.pimpl) return false;
		else {
			if (pimpl == rhs.pimpl) return true;
			else if (*pimpl == *rhs.pimpl) return true;
			else return false;
		}
	}
}
bool AddGhostPeriodic::operator==(const BoundaryCondition& rhs) const {
	if (this == &rhs) return true;
	else if (typeid(*this) != typeid(rhs)) return false;
	else return operator==(dynamic_cast<const AddGhostPeriodic&>(rhs));
}
AddGhostPeriodic* AddGhostPeriodic::clone() const {
	return new AddGhostPeriodic(*this);
}
