#include <vector>
#include <cstdint>
#include <cmath>
#include <typeinfo>
#include <macro.hpp>
#include <levelset/BoundaryCondition/BoundaryCondition.hpp>
#include <levelset/Grids/HJI_Grid.hpp>
#include <levelset/SpatialDerivative/UpwindFirst/UpwindFirstENO2.hpp>
#include "UpwindFirstENO2_impl.hpp"
UpwindFirstENO2_impl::UpwindFirstENO2_impl(
	const HJI_Grid *hji_grid,
	const beacls::UVecType type
	) :
	type(type),
	dxs(hji_grid->get_dxs()),
	dxInvs(hji_grid->get_dxInvs()),
	stencil(2)
{
	cache = new UpwindFirstENO2_Cache;

	size_t num_of_dimensions = hji_grid->get_num_of_dimensions();
	dxInv_2s.resize(num_of_dimensions);
	outer_dimensions_loop_sizes.resize(num_of_dimensions);
	target_dimension_loop_sizes.resize(num_of_dimensions);
	inner_dimensions_loop_sizes.resize(num_of_dimensions);
	first_dimension_loop_sizes.resize(num_of_dimensions);
	src_target_dimension_loop_sizes.resize(num_of_dimensions);
	tmpBoundedSrc_uvecssss.resize(num_of_dimensions);
	tmpBoundedSrc_ptrssss.resize(num_of_dimensions);
	for (size_t target_dimension = 0; target_dimension < num_of_dimensions; ++target_dimension){
		beacls::IntegerVec sizeIn = hji_grid->get_Ns();
		beacls::IntegerVec sizeOut = sizeIn;
		sizeOut[target_dimension] += 2 * stencil;
		dxInv_2s[target_dimension] = dxInvs[target_dimension] / 2;

		size_t first_dimension_loop_size = sizeIn[0];
		size_t inner_dimensions_loop_size = 1;
		size_t target_dimension_loop_size = sizeOut[target_dimension];
		size_t outer_dimensions_loop_size = 1;
		for (size_t dimension = 1; dimension < target_dimension; ++dimension) {
			inner_dimensions_loop_size *= sizeOut[dimension];
		}
		for (size_t dimension = target_dimension + 1; dimension < num_of_dimensions; ++dimension) {
			outer_dimensions_loop_size *= sizeOut[dimension];
		}
		size_t src_target_dimension_loop_size = (target_dimension_loop_size - stencil * 2);
	
		first_dimension_loop_sizes[target_dimension] = first_dimension_loop_size;
		inner_dimensions_loop_sizes[target_dimension] = inner_dimensions_loop_size;
		target_dimension_loop_sizes[target_dimension] = target_dimension_loop_size;
		src_target_dimension_loop_sizes[target_dimension] = src_target_dimension_loop_size;
	}
	cache->last_d1ss.resize(2);
	size_t first_dimension_loop_size = first_dimension_loop_sizes[0];
	std::for_each(cache->last_d1ss.begin(), cache->last_d1ss.end(), ([first_dimension_loop_size](auto& rhs) {rhs.resize(first_dimension_loop_size, 0); }));

	cache->last_d2s.resize(hji_grid->get_N(0));
	cache->last_d2s_fabs.resize(hji_grid->get_N(0));
	cache->last_dx_d2_effs.resize(hji_grid->get_N(0));
	cudaStreams.resize(num_of_dimensions, NULL);
	if (type == beacls::UVecType_Cuda) {
		std::for_each(cudaStreams.begin(), cudaStreams.end(), [](auto& rhs) {
			rhs = new beacls::CudaStream();
		});
	}
}
UpwindFirstENO2_impl::~UpwindFirstENO2_impl() {
	if (cache) delete cache;
}
bool UpwindFirstENO2_impl::operator==(const UpwindFirstENO2_impl& rhs) const {
	if (this == &rhs) return true;
	else if (type != rhs.type) return false;
	else if ((dxs.size() != rhs.dxs.size()) || !std::equal(dxs.cbegin(), dxs.cend(), rhs.dxs.cbegin())) return false;
	else if ((dxInvs.size() != rhs.dxInvs.size()) || !std::equal(dxInvs.cbegin(), dxInvs.cend(), rhs.dxInvs.cbegin())) return false;
	else if ((dxInv_2s.size() != rhs.dxInv_2s.size()) || !std::equal(dxInv_2s.cbegin(), dxInv_2s.cend(), rhs.dxInv_2s.cbegin())) return false;

	else if ((outer_dimensions_loop_sizes.size() != rhs.outer_dimensions_loop_sizes.size()) || !std::equal(outer_dimensions_loop_sizes.cbegin(), outer_dimensions_loop_sizes.cend(), rhs.outer_dimensions_loop_sizes.cbegin())) return false;
	else if ((target_dimension_loop_sizes.size() != rhs.target_dimension_loop_sizes.size()) || !std::equal(target_dimension_loop_sizes.cbegin(), target_dimension_loop_sizes.cend(), rhs.target_dimension_loop_sizes.cbegin())) return false;
	else if ((inner_dimensions_loop_sizes.size() != rhs.inner_dimensions_loop_sizes.size()) || !std::equal(inner_dimensions_loop_sizes.cbegin(), inner_dimensions_loop_sizes.cend(), rhs.inner_dimensions_loop_sizes.cbegin())) return false;
	else if ((first_dimension_loop_sizes.size() != rhs.first_dimension_loop_sizes.size()) || !std::equal(first_dimension_loop_sizes.cbegin(), first_dimension_loop_sizes.cend(), rhs.first_dimension_loop_sizes.cbegin())) return false;
	else if ((src_target_dimension_loop_sizes.size() != rhs.src_target_dimension_loop_sizes.size()) || !std::equal(src_target_dimension_loop_sizes.cbegin(), src_target_dimension_loop_sizes.cend(), rhs.src_target_dimension_loop_sizes.cbegin())) return false;
	else return true;
}
bool UpwindFirstENO2_impl::execute_dim0(
	beacls::UVec& dst_deriv_l,
	beacls::UVec& dst_deriv_r,
	const FLOAT_TYPE* src,
	const HJI_Grid *,
	const BoundaryCondition *boundaryCondition,
	const size_t dim,
	const bool ,
	const size_t loop_begin,
	const size_t slice_length,
	const size_t num_of_slices
) {
#if 1	//!< Workaround, following operation doesn't support CUDA memory yet.
	beacls::UVec tmp_deriv_l;
	beacls::UVec tmp_deriv_r;
	if (dst_deriv_l.type() == beacls::UVecType_Cuda) tmp_deriv_l = beacls::UVec(dst_deriv_l.depth(), beacls::UVecType_Vector, dst_deriv_l.size());
	else tmp_deriv_l = dst_deriv_l;
	if (dst_deriv_r.type() == beacls::UVecType_Cuda) tmp_deriv_r = beacls::UVec(dst_deriv_r.depth(), beacls::UVecType_Vector, dst_deriv_r.size());
	else tmp_deriv_r = dst_deriv_r;
	FLOAT_TYPE* dst_deriv_l_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_deriv_l).ptr();
	FLOAT_TYPE* dst_deriv_r_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_deriv_r).ptr();
#else
	FLOAT_TYPE* dst_deriv_l_ptr = beacls::UVec_<FLOAT_TYPE>(dst_deriv_l).ptr();
	FLOAT_TYPE* dst_deriv_r_ptr = beacls::UVec_<FLOAT_TYPE>(dst_deriv_r).ptr();
#endif
	size_t first_dimension_loop_size = first_dimension_loop_sizes[dim];
	size_t outer_dimensions_loop_length = slice_length / first_dimension_loop_size;
	beacls::UVecDepth depth = beacls::type_to_depth<FLOAT_TYPE>();

	if (bounded_first_dimension_line_cache_uvec.type() == beacls::UVecType_Invalid) {
		bounded_first_dimension_line_cache_uvec = beacls::UVec(depth, type, first_dimension_loop_size + 2 * stencil);
	}
	else if (bounded_first_dimension_line_cache_uvec.size() != (first_dimension_loop_size + 2 * stencil)) {
		bounded_first_dimension_line_cache_uvec.resize(first_dimension_loop_size + 2 * stencil);
	}

	const FLOAT_TYPE dxInv = dxInvs[dim];
	const FLOAT_TYPE dxInv_2 = dxInv_2s[dim];
	const FLOAT_TYPE dx = dxs[dim];
	const size_t target_dimension_loop_size = target_dimension_loop_sizes[dim];
	for (size_t slice_index = 0; slice_index < num_of_slices; ++slice_index) {
		const size_t slice_offset = slice_index * slice_length;
		const size_t slice_loop_offset = slice_index * outer_dimensions_loop_length;
		for (size_t index = 0; index < outer_dimensions_loop_length; ++index) {
			size_t outer_dimensions_loop_index = index + loop_begin + slice_loop_offset;
			beacls::UVec& boundedSrc = bounded_first_dimension_line_cache_uvec;

			size_t dst_offset = index * first_dimension_loop_size + slice_offset;

			//! Target dimension is first loop

			// Add ghost cells.
			if (boundedSrc.type() != type) boundedSrc = beacls::UVec(depth, type, first_dimension_loop_size);
			else if (boundedSrc.size() != first_dimension_loop_size) boundedSrc.resize(first_dimension_loop_size);
			boundedSrc.set_cudaStream(cudaStreams[dim]);
			boundaryCondition->execute(
				src,
				boundedSrc,
				stencil,
				outer_dimensions_loop_index,
				target_dimension_loop_size);
			if (is_cuda(boundedSrc)) {
				beacls::synchronizeUVec(boundedSrc);
			}
			if (boundedSrc.type() == beacls::UVecType_Cuda) { //!< Workaround, following operation doesn't support CUDA memory yet.
				boundedSrc.convertTo(boundedSrc, beacls::UVecType_Vector);
			}

			FLOAT_TYPE* boundedSrc_ptr = beacls::UVec_<FLOAT_TYPE>(boundedSrc).ptr();

			FLOAT_TYPE this_src = boundedSrc_ptr[0];
			FLOAT_TYPE d1_m1 = 0;
			FLOAT_TYPE d2_m1 = 0;
			FLOAT_TYPE d2_m1_fabs = 0;

			//! Prologue
			for (size_t target_dimension_loop_index = 0; target_dimension_loop_index < stencil + 1; ++target_dimension_loop_index) {
				size_t src_index = target_dimension_loop_index;
				FLOAT_TYPE next_src = boundedSrc_ptr[src_index + 1];
				FLOAT_TYPE d1 = dxInv * (next_src - this_src);
				this_src = next_src;
				if (target_dimension_loop_index >= 1) {
					FLOAT_TYPE d2 = dxInv_2 * (d1 - d1_m1);
					FLOAT_TYPE d2_fabs = HjiFabs<FLOAT_TYPE>(d2);
					if (target_dimension_loop_index >= 2) {
						FLOAT_TYPE dx_d2_eff = dx * ((d2_m1_fabs < d2_fabs) ? d2_m1 : d2);
						size_t target_dimension_loop_index_stencil = target_dimension_loop_index - stencil;
						if (target_dimension_loop_index < target_dimension_loop_size - 2) {
							size_t dst_deriv_l_index = target_dimension_loop_index_stencil + dst_offset;
							dst_deriv_l_ptr[dst_deriv_l_index] = d1_m1 + dx_d2_eff;
						}
					}
					d2_m1 = d2;
					d2_m1_fabs = d2_fabs;
				}
				d1_m1 = d1;
			}
			//! Body
			for (size_t target_dimension_loop_index = stencil + 1; target_dimension_loop_index < target_dimension_loop_size - stencil; ++target_dimension_loop_index) {
				size_t src_index = target_dimension_loop_index;
				FLOAT_TYPE next_src = boundedSrc_ptr[src_index + 1];
				FLOAT_TYPE d1 = dxInv * (next_src - this_src);
				this_src = next_src;
				FLOAT_TYPE d2 = dxInv_2 * (d1 - d1_m1);
				FLOAT_TYPE d2_fabs = HjiFabs<FLOAT_TYPE>(d2);
				FLOAT_TYPE dx_d2_eff = dx * ((d2_m1_fabs < d2_fabs) ? d2_m1 : d2);
				size_t target_dimension_loop_index_stencil = target_dimension_loop_index - stencil;
				size_t dst_deriv_l_index = target_dimension_loop_index_stencil + dst_offset;
				dst_deriv_l_ptr[dst_deriv_l_index] = d1_m1 + dx_d2_eff;
				size_t dst_deriv_r_index = dst_deriv_l_index - 1;
				dst_deriv_r_ptr[dst_deriv_r_index] = d1_m1 - dx_d2_eff;
				d2_m1 = d2;
				d2_m1_fabs = d2_fabs;
				d1_m1 = d1;
			}
			//! Epilogue
			for (size_t target_dimension_loop_index = target_dimension_loop_size - stencil; target_dimension_loop_index < target_dimension_loop_size - stencil + 1; ++target_dimension_loop_index) {
				size_t src_index = target_dimension_loop_index;
				FLOAT_TYPE next_src = boundedSrc_ptr[src_index + 1];
				FLOAT_TYPE d1 = dxInv * (next_src - this_src);
				this_src = next_src;
				FLOAT_TYPE d2 = dxInv_2 * (d1 - d1_m1);
				FLOAT_TYPE d2_fabs = HjiFabs<FLOAT_TYPE>(d2);
				FLOAT_TYPE dx_d2_eff = dx * ((d2_m1_fabs < d2_fabs) ? d2_m1 : d2);
				size_t target_dimension_loop_index_stencil = target_dimension_loop_index - stencil;
				size_t dst_deriv_r_index = (target_dimension_loop_index_stencil - 1) + dst_offset;
				dst_deriv_r_ptr[dst_deriv_r_index] = d1_m1 - dx_d2_eff;
				d2_m1 = d2;
				d2_m1_fabs = d2_fabs;
				d1_m1 = d1;
			}
		}
	}
#if 1	//!< Workaround, following operation doesn't support CUDA memory yet.
	tmp_deriv_l.convertTo(dst_deriv_l, type);
	tmp_deriv_r.convertTo(dst_deriv_r, type);
#endif
	dst_deriv_l.set_cudaStream(cudaStreams[dim]);
	dst_deriv_r.set_cudaStream(cudaStreams[dim]);
	return true;
}

bool UpwindFirstENO2_impl::execute_dim1(
	beacls::UVec& dst_deriv_l,
	beacls::UVec& dst_deriv_r,
	const FLOAT_TYPE* src,
	const HJI_Grid *,
	const BoundaryCondition *boundaryCondition,
	const size_t dim,
	const bool ,
	const size_t loop_begin,
	const size_t slice_length,
	const size_t num_of_slices
) {
#if 1	//!< Workaround, following operation doesn't support CUDA memory yet.
	beacls::UVec tmp_deriv_l;
	beacls::UVec tmp_deriv_r;
	if (dst_deriv_l.type() == beacls::UVecType_Cuda) tmp_deriv_l = beacls::UVec(dst_deriv_l.depth(), beacls::UVecType_Vector, dst_deriv_l.size());
	else tmp_deriv_l = dst_deriv_l;
	if (dst_deriv_r.type() == beacls::UVecType_Cuda) tmp_deriv_r = beacls::UVec(dst_deriv_r.depth(), beacls::UVecType_Vector, dst_deriv_r.size());
	else tmp_deriv_r = dst_deriv_r;
	FLOAT_TYPE* dst_deriv_l_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_deriv_l).ptr();
	FLOAT_TYPE* dst_deriv_r_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_deriv_r).ptr();
#else
	FLOAT_TYPE* dst_deriv_l_ptr = beacls::UVec_<FLOAT_TYPE>(dst_deriv_l).ptr();
	FLOAT_TYPE* dst_deriv_r_ptr = beacls::UVec_<FLOAT_TYPE>(dst_deriv_r).ptr();
#endif
	std::vector<beacls::UVec > &cachedBoundedSrcs = cache->cachedBoundedSrc_uvecs;
	std::vector<beacls::FloatVec > &last_d1ss = cache->last_d1ss;
	beacls::FloatVec &last_d2s = cache->last_d2s;
	beacls::FloatVec &last_d2s_fabs = cache->last_d2s_fabs;
	beacls::FloatVec &last_dx_d2_effs = cache->last_dx_d2_effs;
	std::vector<const FLOAT_TYPE*> &boundedSrc_ptrs = cache->boundedSrc_ptrs;
	beacls::UVecDepth depth = beacls::type_to_depth<FLOAT_TYPE>();

	const size_t src_target_dimension_loop_size = src_target_dimension_loop_sizes[dim];
	const size_t inner_dimensions_loop_size = inner_dimensions_loop_sizes[dim];

	size_t first_dimension_loop_size = first_dimension_loop_sizes[dim];
	size_t loop_length = slice_length / first_dimension_loop_size;

	const FLOAT_TYPE dxInv = dxInvs[dim];
	const FLOAT_TYPE dxInv_2 = dxInv_2s[dim];
	const FLOAT_TYPE dx = dxs[dim];

	const size_t target_dimension_loop_size = target_dimension_loop_sizes[dim];

	const FLOAT_TYPE* dst_ptr;
	for (size_t slice_index = 0; slice_index < num_of_slices; ++slice_index) {
		const size_t slice_offset = slice_index * slice_length;
		const size_t slice_loop_offset = slice_index * loop_length;
		for (size_t index = 0; index < loop_length; ++index) {
			size_t loop_index = index + loop_begin + slice_loop_offset;
			size_t dst_offset = index * first_dimension_loop_size + slice_offset;
			size_t loop_index_div_target_size = loop_index / src_target_dimension_loop_size;
			size_t target_dimension_loop_index = loop_index % src_target_dimension_loop_size;
			size_t inner_dimensions_loop_index = loop_index_div_target_size % inner_dimensions_loop_size;
			size_t outer_dimensions_loop_index = loop_index_div_target_size / inner_dimensions_loop_size;

			//! Prologue
			if (target_dimension_loop_index == 0) {
				for (size_t prologue_target_dimension_loop_index = 0; prologue_target_dimension_loop_index < stencil * 2; ++prologue_target_dimension_loop_index) {
					size_t boundedSrc_cache_current_index = prologue_target_dimension_loop_index & 0x1;
					size_t boundedSrc_cache_last_index = (boundedSrc_cache_current_index == 0) ? 1 : 0;
					beacls::FloatVec& d1s_m1 = last_d1ss[boundedSrc_cache_current_index];
					beacls::FloatVec& d1s_m2 = last_d1ss[boundedSrc_cache_last_index];
					beacls::UVec& boundedSrc = cachedBoundedSrcs[boundedSrc_cache_current_index];
					const FLOAT_TYPE* &last_boundedSrc_ptr = boundedSrc_ptrs[boundedSrc_cache_last_index];
					const FLOAT_TYPE* &current_boundedSrc_ptr = boundedSrc_ptrs[boundedSrc_cache_current_index];

					if (target_dimension_loop_index < target_dimension_loop_size) {
						// Add ghost cells.
						if (boundedSrc.type() != type) boundedSrc = beacls::UVec(depth, type, first_dimension_loop_size);
						else if (boundedSrc.size() != first_dimension_loop_size) boundedSrc.resize(first_dimension_loop_size);
						boundedSrc.set_cudaStream(cudaStreams[dim]);
						boundaryCondition->execute(
							src,
							boundedSrc, dst_ptr,
							stencil,
							outer_dimensions_loop_index,
							target_dimension_loop_size, prologue_target_dimension_loop_index,
							inner_dimensions_loop_size, inner_dimensions_loop_index,
							first_dimension_loop_size
						);
						if (is_cuda(boundedSrc)) {
							beacls::synchronizeUVec(boundedSrc);
						}
						if (boundedSrc.type() == beacls::UVecType_Cuda) { //!< Workaround, following operation doesn't support CUDA memory yet.
							if (boundedSrc.ptr() == dst_ptr) {
								boundedSrc.convertTo(boundedSrc, beacls::UVecType_Vector);
								dst_ptr = beacls::UVec_<FLOAT_TYPE>(boundedSrc).ptr();
							}
							else {
								boundedSrc = beacls::UVec(depth, beacls::UVecType_Vector, first_dimension_loop_size);
								copyDevicePtrToUVec(boundedSrc, dst_ptr, boundedSrc.size());
								dst_ptr = beacls::UVec_<FLOAT_TYPE>(boundedSrc).ptr();
							}
						}


						current_boundedSrc_ptr = dst_ptr;
					}
					if ((prologue_target_dimension_loop_index >= 1)) {
						for (size_t first_dimension_loop_index = 0; first_dimension_loop_index < first_dimension_loop_size; ++first_dimension_loop_index) {
							FLOAT_TYPE this_src = current_boundedSrc_ptr[first_dimension_loop_index];
							FLOAT_TYPE last_src = last_boundedSrc_ptr[first_dimension_loop_index];
							FLOAT_TYPE d1 = dxInv * (this_src - last_src);
							if (prologue_target_dimension_loop_index >= 2) {
								FLOAT_TYPE d1_m1 = d1s_m1[first_dimension_loop_index];
								FLOAT_TYPE d2 = dxInv_2 * (d1 - d1_m1);
								FLOAT_TYPE d2_fabs = HjiFabs<FLOAT_TYPE>(d2);
								if (prologue_target_dimension_loop_index >= 3) {
									FLOAT_TYPE d2_m1 = last_d2s[first_dimension_loop_index];
									FLOAT_TYPE d2_m1_fabs = last_d2s_fabs[first_dimension_loop_index];
									FLOAT_TYPE d2_eff = (d2_m1_fabs < d2_fabs) ? d2_m1 : d2;
									FLOAT_TYPE dx_d2_eff = dx * d2_eff;
									last_dx_d2_effs[first_dimension_loop_index] = dx_d2_eff;
								}
								last_d2s[first_dimension_loop_index] = d2;
								last_d2s_fabs[first_dimension_loop_index] = d2_fabs;
							}
							d1s_m2[first_dimension_loop_index] = d1;
						}
					}
				}
			}
			if (target_dimension_loop_index < (target_dimension_loop_size - stencil * 2)) {
				//! Body
				size_t shifted_target_dimension_loop_index = target_dimension_loop_index + 2 * stencil;
				size_t boundedSrc_cache_current_index = shifted_target_dimension_loop_index & 0x1;
				size_t boundedSrc_cache_last_index = (boundedSrc_cache_current_index == 0) ? 1 : 0;
				beacls::UVec& boundedSrc = cachedBoundedSrcs[boundedSrc_cache_current_index];
				const FLOAT_TYPE* &last_boundedSrc_ptr = boundedSrc_ptrs[boundedSrc_cache_last_index];
				const FLOAT_TYPE* &current_boundedSrc_ptr = boundedSrc_ptrs[boundedSrc_cache_current_index];
				if (shifted_target_dimension_loop_index < target_dimension_loop_size) {
					// Add ghost cells.
					if (boundedSrc.type() != type) boundedSrc = beacls::UVec(depth, type, first_dimension_loop_size);
					else if (boundedSrc.size() != first_dimension_loop_size) boundedSrc.resize(first_dimension_loop_size);
					boundedSrc.set_cudaStream(cudaStreams[dim]);
					boundaryCondition->execute(
						src,
						boundedSrc, dst_ptr,
						stencil,
						outer_dimensions_loop_index,
						target_dimension_loop_size, shifted_target_dimension_loop_index,
						inner_dimensions_loop_size, inner_dimensions_loop_index,
						first_dimension_loop_size
					);
					if (is_cuda(boundedSrc)) {
						beacls::synchronizeUVec(boundedSrc);
					}
					if (boundedSrc.type() == beacls::UVecType_Cuda) { //!< Workaround, following operation doesn't support CUDA memory yet.
						if (boundedSrc.ptr() == dst_ptr) {
							boundedSrc.convertTo(boundedSrc, beacls::UVecType_Vector);
							dst_ptr = beacls::UVec_<FLOAT_TYPE>(boundedSrc).ptr();
						}
						else {
							boundedSrc = beacls::UVec(depth, beacls::UVecType_Vector, first_dimension_loop_size);
							copyDevicePtrToUVec(boundedSrc, dst_ptr, boundedSrc.size());
							dst_ptr = beacls::UVec_<FLOAT_TYPE>(boundedSrc).ptr();
						}
					}

					current_boundedSrc_ptr = dst_ptr;
				}
				beacls::FloatVec& d1s_m1 = last_d1ss[boundedSrc_cache_current_index];
				beacls::FloatVec& d1s_m2 = last_d1ss[boundedSrc_cache_last_index];
				for (size_t first_dimension_loop_index = 0; first_dimension_loop_index < first_dimension_loop_size; ++first_dimension_loop_index) {
					FLOAT_TYPE this_src = current_boundedSrc_ptr[first_dimension_loop_index];
					FLOAT_TYPE last_src = last_boundedSrc_ptr[first_dimension_loop_index];
					FLOAT_TYPE d1 = dxInv * (this_src - last_src);
					FLOAT_TYPE d1_m1 = d1s_m1[first_dimension_loop_index];
					FLOAT_TYPE d1_m2 = d1s_m2[first_dimension_loop_index];
					FLOAT_TYPE d2 = dxInv_2 * (d1 - d1_m1);
					FLOAT_TYPE d2_fabs = HjiFabs<FLOAT_TYPE>(d2);
					FLOAT_TYPE d2_m1 = last_d2s[first_dimension_loop_index];
					FLOAT_TYPE d2_m1_fabs = last_d2s_fabs[first_dimension_loop_index];
					FLOAT_TYPE d2_eff = (d2_m1_fabs < d2_fabs) ? d2_m1 : d2;
					FLOAT_TYPE dx_d2_eff = dx * d2_eff;
					FLOAT_TYPE dx_d2_eff_m1 = last_dx_d2_effs[first_dimension_loop_index];

					size_t deriv_index = first_dimension_loop_index + dst_offset;

					dst_deriv_l_ptr[deriv_index] = d1_m2 + dx_d2_eff_m1;
					dst_deriv_r_ptr[deriv_index] = d1_m1 - dx_d2_eff;
					last_d2s[first_dimension_loop_index] = d2;
					last_d2s_fabs[first_dimension_loop_index] = d2_fabs;
					d1s_m2[first_dimension_loop_index] = d1;
					last_dx_d2_effs[first_dimension_loop_index] = dx_d2_eff;
				}
			}
		}
	}
#if 1	//!< Workaround, following operation doesn't support CUDA memory yet.
	tmp_deriv_l.convertTo(dst_deriv_l, type);
	tmp_deriv_r.convertTo(dst_deriv_r, type);
#endif
	dst_deriv_l.set_cudaStream(cudaStreams[dim]);
	dst_deriv_r.set_cudaStream(cudaStreams[dim]);
	return true;
}

bool UpwindFirstENO2_impl::execute_dimLET2(
	beacls::UVec& dst_deriv_l,
	beacls::UVec& dst_deriv_r,
	const FLOAT_TYPE* src,
	const HJI_Grid *,
	const BoundaryCondition *boundaryCondition,
	const size_t dim,
	const bool ,
	const size_t loop_begin,
	const size_t slice_length,
	const size_t num_of_slices
) {
#if 1	//!< Workaround, following operation doesn't support CUDA memory yet.
	beacls::UVec tmp_deriv_l;
	beacls::UVec tmp_deriv_r;
	if (dst_deriv_l.type() == beacls::UVecType_Cuda) tmp_deriv_l = beacls::UVec(dst_deriv_l.depth(), beacls::UVecType_Vector, dst_deriv_l.size());
	else tmp_deriv_l = dst_deriv_l;
	if (dst_deriv_r.type() == beacls::UVecType_Cuda) tmp_deriv_r = beacls::UVec(dst_deriv_r.depth(), beacls::UVecType_Vector, dst_deriv_r.size());
	else tmp_deriv_r = dst_deriv_r;
	FLOAT_TYPE* dst_deriv_l_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_deriv_l).ptr();
	FLOAT_TYPE* dst_deriv_r_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_deriv_r).ptr();
#else
	FLOAT_TYPE* dst_deriv_l_ptr = beacls::UVec_<FLOAT_TYPE>(dst_deriv_l).ptr();
	FLOAT_TYPE* dst_deriv_r_ptr = beacls::UVec_<FLOAT_TYPE>(dst_deriv_r).ptr();
#endif
	const size_t src_target_dimension_loop_size = src_target_dimension_loop_sizes[dim];
	const size_t inner_dimensions_loop_size = inner_dimensions_loop_sizes[dim];

	size_t first_dimension_loop_size = first_dimension_loop_sizes[dim];
	size_t loop_length = slice_length / first_dimension_loop_size;
	const FLOAT_TYPE dxInv = dxInvs[dim];
	const FLOAT_TYPE dxInv_2 = dxInv_2s[dim];
	const FLOAT_TYPE dx = dxs[dim];
	beacls::UVecDepth depth = beacls::type_to_depth<FLOAT_TYPE>();
	std::vector<std::vector<std::vector<beacls::UVec > > >& tmpBoundedSrc_uvecsss = tmpBoundedSrc_uvecssss[dim];
	std::vector<std::vector<std::vector<const FLOAT_TYPE*> > >& tmpBoundedSrc_ptrsss = tmpBoundedSrc_ptrssss[dim];
	if (tmpBoundedSrc_ptrsss.size() < num_of_slices) tmpBoundedSrc_ptrsss.resize(num_of_slices);
	for_each(tmpBoundedSrc_ptrsss.begin(), tmpBoundedSrc_ptrsss.end(), ([this, loop_length](auto &rhs) {
		if (rhs.size() != loop_length) rhs.resize(loop_length);
		for_each(rhs.begin(), rhs.end(), ([this, loop_length](auto &rhs) {
			if (rhs.size() != (stencil * 2 + 1)) rhs.resize(stencil * 2 + 1);
		}));
	}));
	if (tmpBoundedSrc_uvecsss.size() < num_of_slices) tmpBoundedSrc_uvecsss.resize(num_of_slices);
	for_each(tmpBoundedSrc_uvecsss.begin(), tmpBoundedSrc_uvecsss.end(), ([this, first_dimension_loop_size, loop_length, depth](auto &rhs) {
		if (rhs.size() < loop_length) rhs.resize(loop_length);
		for_each(rhs.begin(), rhs.end(), ([this, first_dimension_loop_size, depth](auto &rhs) {
			if (rhs.size() != (stencil * 2 + 1)) rhs.resize(stencil * 2 + 1);
			for_each(rhs.begin(), rhs.end(), ([first_dimension_loop_size, depth, this](auto &rhs) {
				if (rhs.type() == beacls::UVecType_Invalid) rhs = beacls::UVec(depth, type, first_dimension_loop_size);
				else if (rhs.size() != first_dimension_loop_size) rhs.resize(first_dimension_loop_size);
			}));
		}));
	}));
	const size_t target_dimension_loop_size = target_dimension_loop_sizes[dim];
	for (size_t slice_index = 0; slice_index < num_of_slices; ++slice_index) {
		const size_t slice_offset = slice_index * slice_length;
		const size_t slice_loop_offset = slice_index * loop_length;
		for (size_t index = 0; index < loop_length; ++index) {
			size_t loop_index = index + loop_begin + slice_loop_offset;

			size_t loop_index_div_inner_size = loop_index / inner_dimensions_loop_size;
			size_t inner_dimensions_loop_index = loop_index % inner_dimensions_loop_size;
			size_t outer_dimensions_loop_index = loop_index_div_inner_size / src_target_dimension_loop_size;
			size_t target_dimension_loop_index = loop_index_div_inner_size % src_target_dimension_loop_size;

			for (size_t sub_loop_index = 0; sub_loop_index <= stencil * 2; ++sub_loop_index) {
				if ((sub_loop_index + target_dimension_loop_index) < target_dimension_loop_size) {
					const FLOAT_TYPE* dst_ptr;
					beacls::UVec& boundedSrc = tmpBoundedSrc_uvecsss[slice_index][index][sub_loop_index];
					if (boundedSrc.type() != type) boundedSrc = beacls::UVec(depth, type, first_dimension_loop_size);
					else if (boundedSrc.size() != first_dimension_loop_size) boundedSrc.resize(first_dimension_loop_size);
					boundedSrc.set_cudaStream(cudaStreams[dim]);
					boundaryCondition->execute(
						src,
						boundedSrc, dst_ptr,
						stencil,
						outer_dimensions_loop_index,
						target_dimension_loop_size, sub_loop_index + target_dimension_loop_index,
						inner_dimensions_loop_size, inner_dimensions_loop_index,
						first_dimension_loop_size
					);
					if (is_cuda(boundedSrc)) {
						beacls::synchronizeUVec(boundedSrc);
					}
					if (boundedSrc.type() == beacls::UVecType_Cuda) { //!< Workaround, following operation doesn't support CUDA memory yet.
						if (boundedSrc.ptr() == dst_ptr) {
							boundedSrc.convertTo(boundedSrc, beacls::UVecType_Vector);
							dst_ptr = beacls::UVec_<FLOAT_TYPE>(boundedSrc).ptr();
						}
						else {
							boundedSrc = beacls::UVec(depth, beacls::UVecType_Vector, first_dimension_loop_size);
							copyDevicePtrToUVec(boundedSrc, dst_ptr, boundedSrc.size());
							dst_ptr = beacls::UVec_<FLOAT_TYPE>(boundedSrc).ptr();
						}
					}
					tmpBoundedSrc_ptrsss[slice_index][index][sub_loop_index] = dst_ptr;
				}
			}
		}
		for (size_t index = 0; index < loop_length; ++index) {
			size_t loop_index = loop_begin + index+ loop_begin;
			size_t dst_offset = index * first_dimension_loop_size + slice_offset;
			size_t loop_index_div_inner_size = loop_index / inner_dimensions_loop_size;
			size_t target_dimension_loop_index = loop_index_div_inner_size % src_target_dimension_loop_size;

			std::vector<const FLOAT_TYPE*>& tmpBoundedSrc_ptrs = tmpBoundedSrc_ptrsss[slice_index][index];
			const FLOAT_TYPE* tmpBoundedSrc_ptrs0 = tmpBoundedSrc_ptrs[0];
			const FLOAT_TYPE* tmpBoundedSrc_ptrs1 = tmpBoundedSrc_ptrs[1];
			const FLOAT_TYPE* tmpBoundedSrc_ptrs2 = tmpBoundedSrc_ptrs[2];
			const FLOAT_TYPE* tmpBoundedSrc_ptrs3 = tmpBoundedSrc_ptrs[3];
			const FLOAT_TYPE* tmpBoundedSrc_ptrs4 = tmpBoundedSrc_ptrs[4];
			size_t shifted_target_dimension_loop_index = target_dimension_loop_index + 2 * stencil;
			if (shifted_target_dimension_loop_index < target_dimension_loop_size) {
				for (size_t first_dimension_loop_index = 0; first_dimension_loop_index < first_dimension_loop_size; ++first_dimension_loop_index) {
					FLOAT_TYPE d0_0 = tmpBoundedSrc_ptrs0[first_dimension_loop_index];
					FLOAT_TYPE d0_1 = tmpBoundedSrc_ptrs1[first_dimension_loop_index];
					FLOAT_TYPE d1_0 = dxInv*(d0_1 - d0_0);

					FLOAT_TYPE d0_2 = tmpBoundedSrc_ptrs2[first_dimension_loop_index];
					FLOAT_TYPE d1_1 = dxInv*(d0_2 - d0_1);
					FLOAT_TYPE d2_0 = dxInv_2 * (d1_1 - d1_0);
					FLOAT_TYPE fabs_d2_0 = HjiFabs<FLOAT_TYPE>(d2_0);

					FLOAT_TYPE d0_3 = tmpBoundedSrc_ptrs3[first_dimension_loop_index];
					FLOAT_TYPE d1_2 = dxInv*(d0_3 - d0_2);
					FLOAT_TYPE d2_1 = dxInv_2 * (d1_2 - d1_1);
					FLOAT_TYPE fabs_d2_1 = HjiFabs<FLOAT_TYPE>(d2_1);
					FLOAT_TYPE d2_eff_0 = (fabs_d2_0 < fabs_d2_1) ? d2_0 : d2_1;
					FLOAT_TYPE dx_d2_eff_0 = dx * d2_eff_0;

					FLOAT_TYPE d0_4 = tmpBoundedSrc_ptrs4[first_dimension_loop_index];
					FLOAT_TYPE d1_3 = dxInv*(d0_4 - d0_3);
					FLOAT_TYPE d2_2 = dxInv_2 * (d1_3 - d1_2);
					FLOAT_TYPE fabs_d2_2 = HjiFabs<FLOAT_TYPE>(d2_2);
					FLOAT_TYPE d2_eff_1 = (fabs_d2_1 < fabs_d2_2) ? d2_1 : d2_2;
					FLOAT_TYPE dx_d2_eff_1 = dx * d2_eff_1;
					size_t dst_index = first_dimension_loop_index + dst_offset;
					dst_deriv_l_ptr[dst_index] = d1_1 + dx_d2_eff_0;
					dst_deriv_r_ptr[dst_index] = d1_2 - dx_d2_eff_1;
				}
			}
		}
	}
#if 1	//!< Workaround, following operation doesn't support CUDA memory yet.
	tmp_deriv_l.convertTo(dst_deriv_l, type);
	tmp_deriv_r.convertTo(dst_deriv_r, type);
#endif
	dst_deriv_l.set_cudaStream(cudaStreams[dim]);
	dst_deriv_r.set_cudaStream(cudaStreams[dim]);
	return true;
}

bool UpwindFirstENO2_impl::execute(
	beacls::UVec& dst_deriv_l,
	beacls::UVec& dst_deriv_r,
	const HJI_Grid *grid,
	const FLOAT_TYPE* src,
	const size_t dim,
	const bool generateAll,
	const size_t loop_begin,
	const size_t slice_length,
	const size_t num_of_slices
) {

	BoundaryCondition* boundaryCondition = grid->get_boundaryCondition(dim);
	switch (dim) {
	case 0:
	{
		execute_dim0(
			dst_deriv_l,
			dst_deriv_r,
			src,
			grid,
			boundaryCondition,
			dim,
			generateAll,
			loop_begin,
			slice_length,
			num_of_slices
		);
	}
	break;
	case 1:
	{
		execute_dim1(
			dst_deriv_l,
			dst_deriv_r,
			src,
			grid,
			boundaryCondition,
			dim,
			generateAll,
			loop_begin,
			slice_length,
			num_of_slices
		);
	}
	break;
	default:
	{
		execute_dimLET2(
			dst_deriv_l,
			dst_deriv_r,
			src,
			grid,
			boundaryCondition,
			dim,
			generateAll,
			loop_begin,
			slice_length,
			num_of_slices
		);
	}
	break;
	}
	return true;
}


UpwindFirstENO2::UpwindFirstENO2(
	const HJI_Grid *hji_grid,
	const beacls::UVecType type
	) {
	pimpl = new UpwindFirstENO2_impl(hji_grid,type);
}
UpwindFirstENO2::~UpwindFirstENO2() {
	if (pimpl) delete pimpl;
}

bool UpwindFirstENO2::execute(
	beacls::UVec& dst_deriv_l,
	beacls::UVec& dst_deriv_r,
	const HJI_Grid *grid,
	const FLOAT_TYPE* src,
	const size_t dim,
	const bool generateAll,
	const size_t loop_begin,
	const size_t slice_length,
	const size_t num_of_slices
)  {
	if (pimpl) return pimpl->execute(dst_deriv_l, dst_deriv_r, grid, src, dim, generateAll, loop_begin, slice_length, num_of_slices);
	else return false;
}
bool UpwindFirstENO2_impl::synchronize(const size_t dim) {
	if ((cudaStreams.size() > dim) && cudaStreams[dim]) {
		beacls::synchronizeCuda(cudaStreams[dim]);
		return true;
	}
	return false;
}
bool UpwindFirstENO2::synchronize(const size_t dim) {
	if (pimpl) return pimpl->synchronize(dim);
	else return false;
}

bool UpwindFirstENO2::operator==(const UpwindFirstENO2& rhs) const {
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
bool UpwindFirstENO2::operator==(const SpatialDerivative& rhs) const {
	if (this == &rhs) return true;
	else if (typeid(*this) != typeid(rhs)) return false;
	else return operator==(dynamic_cast<const UpwindFirstENO2&>(rhs));
}

UpwindFirstENO2::UpwindFirstENO2(const UpwindFirstENO2& rhs) :
	pimpl(rhs.pimpl->clone())
{
}

UpwindFirstENO2* UpwindFirstENO2::clone() const {
	return new UpwindFirstENO2(*this);
}
beacls::UVecType UpwindFirstENO2::get_type() const {
	if (pimpl) return pimpl->get_type();
	else return beacls::UVecType_Invalid;
};
