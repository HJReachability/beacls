#include <vector>
#include <cstdint>
#include <cmath>
#include <array>
#include <typeinfo>
#include <levelset/Grids/HJI_Grid.hpp>
#include <levelset/BoundaryCondition/BoundaryCondition.hpp>
#include <levelset/SpatialDerivative/UpwindFirst/UpwindFirstENO3aHelper.hpp>
#include "UpwindFirstENO3aHelper_impl.hpp"
#include "UpwindFirstENO3aHelper_cuda.hpp"
using namespace levelset;

UpwindFirstENO3aHelper_impl::UpwindFirstENO3aHelper_impl(
	const HJI_Grid *hji_grid,
	const beacls::UVecType type
) :
	type(type),
	dxs(hji_grid->get_dxs()),
	dxInvs(hji_grid->get_dxInvs()),
	stencil(3)
{
	cache = new UpwindFirstENO3aHelper_Cache;

	size_t num_of_dimensions = hji_grid->get_num_of_dimensions();
	dxInv_2s.resize(num_of_dimensions);
	dxInv_3s.resize(num_of_dimensions);
	dx_squares.resize(num_of_dimensions);

	outer_dimensions_loop_sizes.resize(num_of_dimensions);
	target_dimension_loop_sizes.resize(num_of_dimensions);
	inner_dimensions_loop_sizes.resize(num_of_dimensions);
	first_dimension_loop_sizes.resize(num_of_dimensions);
	src_target_dimension_loop_sizes.resize(num_of_dimensions);
	tmpBoundedSrc_uvec_vectors.resize(num_of_dimensions);
	tmpBoundedSrc_uvecs.resize(num_of_dimensions);
	tmpBoundedSrc_ptrssss.resize(num_of_dimensions);
	for (size_t target_dimension = 0; target_dimension < num_of_dimensions; ++target_dimension){
		beacls::IntegerVec sizeIn = hji_grid->get_Ns();
		beacls::IntegerVec sizeOut = sizeIn;
		sizeOut[target_dimension] += 2 * stencil;
		dxInv_2s[target_dimension] = (FLOAT_TYPE)0.5 * dxInvs[target_dimension];
		dxInv_3s[target_dimension] = (FLOAT_TYPE)(1./3.) * dxInvs[target_dimension];
		dx_squares[target_dimension] = std::pow(dxs[target_dimension],2);

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
}
UpwindFirstENO3aHelper_impl::~UpwindFirstENO3aHelper_impl() {
	if (cache) delete cache;
}
bool UpwindFirstENO3aHelper_impl::operator==(const UpwindFirstENO3aHelper_impl& rhs) const {
	if (this == &rhs) return true;
	else if (type != rhs.type) return false;
	else if ((dxs.size() != rhs.dxs.size()) || !std::equal(dxs.cbegin(), dxs.cend(), rhs.dxs.cbegin())) return false;
	else if ((dx_squares.size() != rhs.dx_squares.size()) || !std::equal(dx_squares.cbegin(), dx_squares.cend(), rhs.dx_squares.cbegin())) return false;
	else if ((dxInvs.size() != rhs.dxInvs.size()) || !std::equal(dxInvs.cbegin(), dxInvs.cend(), rhs.dxInvs.cbegin())) return false;
	else if ((dxInv_2s.size() != rhs.dxInv_2s.size()) || !std::equal(dxInv_2s.cbegin(), dxInv_2s.cend(), rhs.dxInv_2s.cbegin())) return false;
	else if ((dxInv_3s.size() != rhs.dxInv_3s.size()) || !std::equal(dxInv_3s.cbegin(), dxInv_3s.cend(), rhs.dxInv_3s.cbegin())) return false;

	else if ((outer_dimensions_loop_sizes.size() != rhs.outer_dimensions_loop_sizes.size()) || !std::equal(outer_dimensions_loop_sizes.cbegin(), outer_dimensions_loop_sizes.cend(), rhs.outer_dimensions_loop_sizes.cbegin())) return false;
	else if ((target_dimension_loop_sizes.size() != rhs.target_dimension_loop_sizes.size()) || !std::equal(target_dimension_loop_sizes.cbegin(), target_dimension_loop_sizes.cend(), rhs.target_dimension_loop_sizes.cbegin())) return false;
	else if ((inner_dimensions_loop_sizes.size() != rhs.inner_dimensions_loop_sizes.size()) || !std::equal(inner_dimensions_loop_sizes.cbegin(), inner_dimensions_loop_sizes.cend(), rhs.inner_dimensions_loop_sizes.cbegin())) return false;
	else if ((first_dimension_loop_sizes.size() != rhs.first_dimension_loop_sizes.size()) || !std::equal(first_dimension_loop_sizes.cbegin(), first_dimension_loop_sizes.cend(), rhs.first_dimension_loop_sizes.cbegin())) return false;
	else if ((src_target_dimension_loop_sizes.size() != rhs.src_target_dimension_loop_sizes.size()) || !std::equal(src_target_dimension_loop_sizes.cbegin(), src_target_dimension_loop_sizes.cend(), rhs.src_target_dimension_loop_sizes.cbegin())) return false;
	else return true;
}

void UpwindFirstENO3aHelper_impl::createCaches(
	const size_t first_dimension_loop_size,
	const size_t num_of_cache_lines)
{
	std::vector<beacls::FloatVec > &last_d1ss = cache->last_d1ss;
	std::vector<beacls::FloatVec > &last_d2ss = cache->last_d2ss;
	std::vector<beacls::FloatVec > &last_d3ss = cache->last_d3ss;
	if (last_d1ss.size() != num_of_cache_lines) last_d1ss.resize(num_of_cache_lines);
	if (last_d2ss.size() != num_of_cache_lines) last_d2ss.resize(num_of_cache_lines);
	if (last_d3ss.size() != num_of_cache_lines) last_d3ss.resize(num_of_cache_lines);
	for_each(last_d1ss.begin(), last_d1ss.end(), ([first_dimension_loop_size](auto& rhs) {if (rhs.size() < first_dimension_loop_size) rhs.resize(first_dimension_loop_size); }));
	for_each(last_d2ss.begin(), last_d2ss.end(), ([first_dimension_loop_size](auto& rhs) {if (rhs.size() < first_dimension_loop_size) rhs.resize(first_dimension_loop_size); }));
	for_each(last_d3ss.begin(), last_d3ss.end(), ([first_dimension_loop_size](auto& rhs) {if (rhs.size() < first_dimension_loop_size) rhs.resize(first_dimension_loop_size); }));
}

void UpwindFirstENO3aHelper_impl::getCachePointers(
	std::vector<FLOAT_TYPE*> &d1s_ms,
	std::vector<FLOAT_TYPE*> &d2s_ms,
	std::vector<FLOAT_TYPE*> &d3s_ms,
	std::vector<beacls::UVec > &dst_DD,
	bool &d1_m0_writeToCache,
	bool &d2_m0_writeToCache,
	bool &d3_m0_writeToCache,
	const size_t slice_index,
	const size_t shifted_target_dimension_loop_index,
	const size_t first_dimension_loop_size,
	const size_t dst_loop_offset,
	const size_t num_of_slices,
	const size_t num_of_cache_lines,
	const bool saveDD,
	const bool stepHead) {
	beacls::IntegerVec &cache_indexes = tmp_cache_indexes;
	if (cache_indexes.size() < num_of_cache_lines) cache_indexes.resize(num_of_cache_lines);
	for (size_t i = 0; i < cache_indexes.size(); ++i) {
		cache_indexes[i] = (((num_of_cache_lines - 2) - (shifted_target_dimension_loop_index % (num_of_cache_lines - 1))) + i) % (num_of_cache_lines - 1);
	}

	std::vector<beacls::FloatVec > &last_d1ss = cache->last_d1ss;
	std::vector<beacls::FloatVec > &last_d2ss = cache->last_d2ss;
	std::vector<beacls::FloatVec > &last_d3ss = cache->last_d3ss;
	for (size_t i = 0; i < cache_indexes.size(); ++i) {
		size_t cache_index = cache_indexes[i];
		d1s_ms[i] = &last_d1ss[cache_index][0];
		d2s_ms[i] = &last_d2ss[cache_index][0];
		d3s_ms[i] = &last_d3ss[cache_index][0];
	}

	d1_m0_writeToCache = d2_m0_writeToCache = d3_m0_writeToCache = true;
	if (saveDD) {
		//! Use DD instead of cache.
		const size_t dst_DD0_slice_size = dst_DD[0].size() / num_of_slices;
		const size_t dst_DD1_slice_size = dst_DD[1].size() / num_of_slices;
		const size_t dst_DD2_slice_size = dst_DD[2].size() / num_of_slices;
		const size_t dst_DD0_slice_offset = slice_index * dst_DD0_slice_size;
		const size_t dst_DD1_slice_offset = slice_index * dst_DD1_slice_size;
		const size_t dst_DD2_slice_offset = slice_index * dst_DD2_slice_size;
		if (!stepHead) {
			for (size_t i = 0; i < d1s_ms.size(); ++i) {
				if (dst_loop_offset >= first_dimension_loop_size * (i + 2)) {
					const size_t dst_dd_base = dst_loop_offset - first_dimension_loop_size * (i + 2);
					if (dst_dd_base < dst_DD0_slice_size) {
						d1s_ms[i] = beacls::UVec_<FLOAT_TYPE>(dst_DD[0]).ptr() + dst_dd_base + dst_DD0_slice_offset;
						if (i == 0) d1_m0_writeToCache = false;
					}
					if (dst_dd_base < dst_DD1_slice_size) {
						d2s_ms[i] = beacls::UVec_<FLOAT_TYPE>(dst_DD[1]).ptr() + dst_dd_base + dst_DD1_slice_offset;
						if (i == 0) d2_m0_writeToCache = false;
					}
					if (dst_dd_base < dst_DD2_slice_size) {
						d3s_ms[i] = beacls::UVec_<FLOAT_TYPE>(dst_DD[2]).ptr() + dst_dd_base + dst_DD2_slice_offset;
						if (i == 0) d3_m0_writeToCache = false;
					}
				}
			}
		}
		else {
			for (size_t i = 0; i < d1s_ms.size(); ++i) {
				if (dst_loop_offset >= first_dimension_loop_size * i) {
					const size_t dst_dd_base = dst_loop_offset - first_dimension_loop_size * i;
					if (dst_dd_base < dst_DD0_slice_size) {
						d1s_ms[i] = beacls::UVec_<FLOAT_TYPE>(dst_DD[0]).ptr() + dst_dd_base + dst_DD0_slice_offset;
						if (i == 0) d1_m0_writeToCache = false;
					}
					if ((dst_dd_base < dst_DD1_slice_size) && (i >= 1)) {
						d2s_ms[i - 1] = beacls::UVec_<FLOAT_TYPE>(dst_DD[1]).ptr() + dst_dd_base + dst_DD1_slice_offset;
						if ((i - 1) == 0) d2_m0_writeToCache = false;
					}
					if ((dst_dd_base < dst_DD2_slice_size) && (i >= 2)) {
						d3s_ms[i - 2] = beacls::UVec_<FLOAT_TYPE>(dst_DD[2]).ptr() + dst_dd_base + dst_DD2_slice_offset;
						if ((i - 2) == 0) d3_m0_writeToCache = false;
					}
				}
			}
		}
	}
}

template<typename T, typename S, typename U> inline
void calc_D1toD3andDD_dimLET2(
	T* dst_dL0_ptr,
	T* dst_dL1_ptr,
	T* dst_dL2_ptr,
	T* dst_dL3_ptr,
	T* dst_dR0_ptr,
	T* dst_dR1_ptr,
	T* dst_dR2_ptr,
	T* dst_dR3_ptr,
	std::vector<T*>& dst_DD0_ptrs,
	std::vector<T*>& dst_DD1_ptrs,
	std::vector<T*>& dst_DD2_ptrs,
	std::vector<std::vector<T> >& d2ss,
	std::vector<std::vector<T> >& d3ss,
	const std::vector<std::vector<std::vector<const T*> > >& dst_ptrsss,
	const T dxInv,
	const T dxInv_2,
	const T dxInv_3,
	const T dx,
	const T x2_dx_square,
	const T dx_square,
	const size_t loop_length,
	const size_t first_dimension_loop_size,
	const size_t num_of_buffer_lines,
	const size_t slice_length,
	const size_t num_of_slices,
	const size_t num_of_strides,
	const size_t num_of_dLdR_in_slice,
	S approx4 = S(),
	U stripDD = U())
{
	for (size_t slice_index = 0; slice_index < num_of_slices; ++slice_index) {
		size_t dst_slice_offset = slice_index * loop_length * first_dimension_loop_size;
		for (size_t loop_index = 0; loop_index < loop_length; ++loop_index) {
			size_t dst_loop_offset = loop_index * first_dimension_loop_size + dst_slice_offset;
			const std::vector<const T*>& tmpBoundedSrc_ptrs = dst_ptrsss[slice_index][loop_index];
			//! Prologue
			{
				size_t pre_loop = 0;
				size_t buffer_index = pre_loop + 1;
				size_t m0_index = ((buffer_index + 3) % num_of_buffer_lines);
				const T* tmpBoundedSrc_ptrs0 = tmpBoundedSrc_ptrs[0 + pre_loop];
				const T* tmpBoundedSrc_ptrs1 = tmpBoundedSrc_ptrs[1 + pre_loop];
				const T* tmpBoundedSrc_ptrs2 = tmpBoundedSrc_ptrs[2 + pre_loop];
				std::vector<T>& d2ss_m0 = d2ss[m0_index];
				for (size_t first_dimension_loop_index = 0; first_dimension_loop_index < first_dimension_loop_size; ++first_dimension_loop_index) {
					T d0_m0 = tmpBoundedSrc_ptrs2[first_dimension_loop_index];
					T d0_m1 = tmpBoundedSrc_ptrs1[first_dimension_loop_index];
					T d0_m2 = tmpBoundedSrc_ptrs0[first_dimension_loop_index];
					T d1_m0 = dxInv * (d0_m0 - d0_m1);
					T d1_m1 = dxInv * (d0_m1 - d0_m2);
					T d2_m0 = dxInv_2 * (d1_m0 - d1_m1);
					d2ss_m0[first_dimension_loop_index] = d2_m0;
				}
			}
			for (size_t stride_index = 0; stride_index < num_of_strides; ++stride_index) {
				const size_t dst_stride_offset = stride_index * loop_length * num_of_slices * first_dimension_loop_size;
				const size_t buffer_index = stride_index + num_of_buffer_lines - 1;
				const size_t m0_index = ((buffer_index + 3) % num_of_buffer_lines);
				const size_t m1_index = ((buffer_index + 2) % num_of_buffer_lines);
				const size_t m2_index = ((buffer_index + 1) % num_of_buffer_lines);

				const T* tmpBoundedSrc_ptrs1 = tmpBoundedSrc_ptrs[0 + stride_index + num_of_buffer_lines - 3];
				const T* tmpBoundedSrc_ptrs2 = tmpBoundedSrc_ptrs[0 + stride_index + num_of_buffer_lines - 2];
				const T* tmpBoundedSrc_ptrs3 = tmpBoundedSrc_ptrs[0 + stride_index + num_of_buffer_lines - 1];
				const T* tmpBoundedSrc_ptrs4 = tmpBoundedSrc_ptrs[0 + stride_index + num_of_buffer_lines];
				std::vector<T>& d2ss_m0 = d2ss[m0_index];
				const std::vector<T>& d2ss_m1 = d2ss[m1_index];
				const std::vector<T>& d2ss_m2 = d2ss[m2_index];
				std::vector<T>& d3ss_m0 = d3ss[m0_index];
				const std::vector<T>& d3ss_m1 = d3ss[m1_index];
				const std::vector<T>& d3ss_m2 = d3ss[m2_index];
				if ((stride_index >= 3) && (stride_index <= 3 + num_of_dLdR_in_slice - 1)) {
					const size_t dst_dLdR_slice_offset = (stride_index - 3) * slice_length;

					const T* tmpBoundedSrc_ptrs0 = tmpBoundedSrc_ptrs[0 + stride_index + num_of_buffer_lines - 4];
					for (size_t first_dimension_loop_index = 0; first_dimension_loop_index < first_dimension_loop_size; ++first_dimension_loop_index) {
						size_t dst_dLdR_index = first_dimension_loop_index + dst_loop_offset + dst_dLdR_slice_offset;
						size_t dst_DD_index = first_dimension_loop_index + dst_loop_offset + dst_stride_offset;
						T d0_m0, d0_m1, d0_m2, d0_m3, d0_m4, d1_m0, d1_m1, d1_m2, d1_m3;
						calc_d0_dimLET2(d0_m4, tmpBoundedSrc_ptrs0, first_dimension_loop_index);
						calc_d0d1_dimLET2<FLOAT_TYPE>(d0_m4, d0_m3, d1_m3, tmpBoundedSrc_ptrs1, dxInv, first_dimension_loop_index);
						calc_d0d1_dimLET2<FLOAT_TYPE>(d0_m3, d0_m2, d1_m2, tmpBoundedSrc_ptrs2, dxInv, first_dimension_loop_index);
						calc_d0d1_dimLET2<FLOAT_TYPE>(d0_m2, d0_m1, d1_m1, tmpBoundedSrc_ptrs3, dxInv, first_dimension_loop_index);
						calc_d0d1_dimLET2<FLOAT_TYPE>(d0_m1, d0_m0, d1_m0, tmpBoundedSrc_ptrs4, dxInv, first_dimension_loop_index);
						T d2_m1 = d2ss_m1[first_dimension_loop_index];
						T d2_m2 = d2ss_m2[first_dimension_loop_index];
						T d2_m3 = d2ss_m0[first_dimension_loop_index];
						T d3_m1 = d3ss_m1[first_dimension_loop_index];
						T d3_m2 = d3ss_m2[first_dimension_loop_index];
						T d3_m3 = d3ss_m0[first_dimension_loop_index];
						T d2_m0 = dxInv_2 * (d1_m0 - d1_m1);
						T d3_m0 = dxInv_3 * (d2_m0 - d2_m1);
						calcApprox1to3<FLOAT_TYPE>(dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr,
							d1_m2, d1_m3, d2_m1, d2_m2, d2_m3, d3_m0, d3_m1, d3_m2, d3_m3, dx, x2_dx_square, dx_square, dst_dLdR_index);
						calcApprox4s<FLOAT_TYPE>(dst_dL3_ptr, dst_dR3_ptr, d1_m3, d1_m2, d2_m2, d2_m1, d3_m2, d3_m1, dx, dx_square, x2_dx_square, dst_dLdR_index, approx4);
						T* dst_DD0_ptr = dst_DD0_ptrs[0];
						T* dst_DD1_ptr = dst_DD1_ptrs[0];
						T* dst_DD2_ptr = dst_DD2_ptrs[0];
						storeDD(dst_DD0_ptr, d1_m0, d1_m2, dst_DD_index, stripDD);
						storeDD(dst_DD1_ptr, d2_m0, d2_m1, dst_DD_index, stripDD);
						storeDD(dst_DD2_ptr, d3_m0, d3_m0, dst_DD_index, stripDD);
						d3ss_m0[first_dimension_loop_index] = d3_m0;
						d2ss_m0[first_dimension_loop_index] = d2_m0;
					}
				}
				else {
					for (size_t first_dimension_loop_index = 0; first_dimension_loop_index < first_dimension_loop_size; ++first_dimension_loop_index) {
						size_t dst_DD_index = first_dimension_loop_index + dst_loop_offset + dst_stride_offset;
						T d0_m0, d0_m1, d0_m2, d0_m3, d1_m0, d1_m1, d1_m2;
						calc_d0_dimLET2<FLOAT_TYPE>(d0_m3, tmpBoundedSrc_ptrs1, first_dimension_loop_index);
						calc_d0d1_dimLET2<FLOAT_TYPE>(d0_m3, d0_m2, d1_m2, tmpBoundedSrc_ptrs2, dxInv, first_dimension_loop_index);
						calc_d0d1_dimLET2<FLOAT_TYPE>(d0_m2, d0_m1, d1_m1, tmpBoundedSrc_ptrs3, dxInv, first_dimension_loop_index);
						calc_d0d1_dimLET2<FLOAT_TYPE>(d0_m1, d0_m0, d1_m0, tmpBoundedSrc_ptrs4, dxInv, first_dimension_loop_index);
						T d2_m1 = d2ss_m1[first_dimension_loop_index];
						T d2_m0 = dxInv_2 * (d1_m0 - d1_m1);
						T d3_m0 = dxInv_3 * (d2_m0 - d2_m1);
						T* dst_DD0_ptr = dst_DD0_ptrs[0];
						T* dst_DD1_ptr = dst_DD1_ptrs[0];
						T* dst_DD2_ptr = dst_DD2_ptrs[0];
						storeDD(dst_DD0_ptr, d1_m0, d1_m2, dst_DD_index, stripDD);
						storeDD(dst_DD1_ptr, d2_m0, d2_m1, dst_DD_index, stripDD);
						storeDD(dst_DD2_ptr, d3_m0, d3_m0, dst_DD_index, stripDD);
						d3ss_m0[first_dimension_loop_index] = d3_m0;
						d2ss_m0[first_dimension_loop_index] = d2_m0;
					}
				}
			}
		}
	}
}

template<typename T,typename S> inline
void calc_D1toD3_dimLET2(
	T* dst_dL0_ptr,
	T* dst_dL1_ptr,
	T* dst_dL2_ptr,
	T* dst_dL3_ptr,
	T* dst_dR0_ptr,
	T* dst_dR1_ptr,
	T* dst_dR2_ptr,
	T* dst_dR3_ptr,
	const std::vector<std::vector<std::vector<const T*> > >& tmpBoundedSrc_ptrsss,
	const T dxInv,
	const T dxInv_2,
	const T dxInv_3,
	const T dx,
	const T x2_dx_square,
	const T dx_square,
	const size_t loop_length,
	const size_t first_dimension_loop_size,
	const size_t slice_length,
	const size_t num_of_slices,
	S approx4 = S()) {
	for (size_t slice_index = 0; slice_index < num_of_slices; ++slice_index) {
		const size_t slice_offset = slice_index * slice_length;
		for (size_t loop_index = 0; loop_index < loop_length; ++loop_index) {
			size_t dst_dLdR_offset = loop_index * first_dimension_loop_size + slice_offset;
			const std::vector<const T*>& tmpBoundedSrc_ptrs = tmpBoundedSrc_ptrsss[slice_index][loop_index];
			{
				size_t plane = 0;
				const T* tmpBoundedSrc_ptrs0 = tmpBoundedSrc_ptrs[0 + plane];
				const T* tmpBoundedSrc_ptrs1 = tmpBoundedSrc_ptrs[1 + plane];
				const T* tmpBoundedSrc_ptrs2 = tmpBoundedSrc_ptrs[2 + plane];
				const T* tmpBoundedSrc_ptrs3 = tmpBoundedSrc_ptrs[3 + plane];
				const T* tmpBoundedSrc_ptrs4 = tmpBoundedSrc_ptrs[4 + plane];
				const T* tmpBoundedSrc_ptrs5 = tmpBoundedSrc_ptrs[5 + plane];
				const T* tmpBoundedSrc_ptrs6 = tmpBoundedSrc_ptrs[6 + plane];
				for (size_t first_dimension_loop_index = 0; first_dimension_loop_index < first_dimension_loop_size; ++first_dimension_loop_index) {
					T d0_0, d0_1, d0_2, d0_3, d0_4, d0_5, d0_6, d1_0, d1_1, d1_2, d1_3, d1_4, d1_5, d2_0, d2_1, d2_2, d2_3, d2_4, d3_0, d3_1, d3_2, d3_3;
					calcD1toD3_dimLET2<FLOAT_TYPE>(
						d0_0, d0_1, d0_2, d0_3, d0_4, d0_5, d0_6, d1_0, d1_1, d1_2, d1_3, d1_4, d1_5, d2_0, d2_1, d2_2, d2_3, d2_4, d3_0, d3_1, d3_2, d3_3,
						tmpBoundedSrc_ptrs0, tmpBoundedSrc_ptrs1, tmpBoundedSrc_ptrs2, tmpBoundedSrc_ptrs3, tmpBoundedSrc_ptrs4, tmpBoundedSrc_ptrs5, tmpBoundedSrc_ptrs6,
						dxInv, dxInv_2, dxInv_3, first_dimension_loop_index
						);
					size_t dst_index = first_dimension_loop_index + dst_dLdR_offset;
					calcApprox1to3<FLOAT_TYPE>(dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, d1_3, d1_2, d2_3, d2_2, d2_1, d3_3, d3_2, d3_1, d3_0, dx, x2_dx_square, dx_square, dst_index);
					calcApprox4s<FLOAT_TYPE>(dst_dL3_ptr, dst_dR3_ptr, d1_2, d1_3, d2_2, d2_3, d3_1, d3_2, dx, dx_square, x2_dx_square, dst_index, approx4);
				}
			}
		}
	}
}


bool UpwindFirstENO3aHelper_impl::execute_dim0(
	std::vector<beacls::UVec > &dst_dL,
	std::vector<beacls::UVec > &dst_dR,
	std::vector<beacls::UVec > &dst_DD,
	const FLOAT_TYPE* src,
	const BoundaryCondition *boundaryCondition,
	const size_t dim,
	const bool approx4,
	const bool stripDD,
	const size_t loop_begin,
	const size_t slice_length,
	const size_t num_of_slices,
	beacls::CudaStream* cudaStream
) {
	const size_t first_dimension_loop_size = first_dimension_loop_sizes[dim];
	const size_t outer_dimensions_loop_length = slice_length / first_dimension_loop_size;
	const size_t total_slices_length = slice_length * num_of_slices;
	size_t num_of_dsts = approx4 ? 4 : 3;
	if (dst_dL.size() != num_of_dsts) dst_dL.resize(num_of_dsts);
	if (dst_dR.size() != num_of_dsts) dst_dR.resize(num_of_dsts);
	beacls::UVecDepth depth = beacls::type_to_depth<FLOAT_TYPE>();
#if 0
	if (bounded_first_dimension_line_cache_uvec.type() == beacls::UVecType_Invalid) {
		bounded_first_dimension_line_cache_uvec = beacls::UVec(depth, type, first_dimension_loop_size + 2 * stencil);
	} else if (bounded_first_dimension_line_cache_uvec.size() < (first_dimension_loop_size + 2 * stencil)) {
		bounded_first_dimension_line_cache_uvec.resize(first_dimension_loop_size + 2 * stencil);
	}
#endif
	for_each(dst_dL.begin(), dst_dL.end(), ([total_slices_length, depth, this](auto &rhs) {
		if (rhs.type() == beacls::UVecType_Invalid) rhs = beacls::UVec(depth, type, total_slices_length);
		else if (rhs.size() < total_slices_length) rhs.resize(total_slices_length);
	}));
	for_each(dst_dR.begin(), dst_dR.end(), ([total_slices_length, depth, this](auto &rhs) {
		if (rhs.type() == beacls::UVecType_Invalid) rhs = beacls::UVec(depth, type, total_slices_length);
		else if (rhs.size() < total_slices_length) rhs.resize(total_slices_length);
	}));

	bool saveDD = (dst_DD.size() == 3);
	const FLOAT_TYPE dxInv = dxInvs[dim];
	const FLOAT_TYPE dxInv_2 = dxInv_2s[dim];
	const FLOAT_TYPE dxInv_3 = dxInv_3s[dim];
	const FLOAT_TYPE dx_square = dx_squares[dim];
	const FLOAT_TYPE x2_dx_square = 2 * dx_square;
	const FLOAT_TYPE dx = dxs[dim];
	const size_t target_dimension_loop_size = target_dimension_loop_sizes[dim];

	if (saveDD) {
		if (stripDD) {
			size_t DD_size_base = (first_dimension_loop_size + 1) * outer_dimensions_loop_length*num_of_slices;
			for (std::vector<beacls::UVec >::iterator ite = dst_DD.begin(); ite != dst_DD.end(); ++ite) {
				if (ite->type() == beacls::UVecType_Invalid) *ite = beacls::UVec(depth, type, DD_size_base);
				else if (ite->size() < DD_size_base) ite->resize(DD_size_base);
				DD_size_base += outer_dimensions_loop_length*num_of_slices;
			}
		}
		else {
			size_t DD_size_base = (first_dimension_loop_size + stencil * 2) * outer_dimensions_loop_length*num_of_slices;
			for (std::vector<beacls::UVec >::iterator ite = dst_DD.begin(); ite != dst_DD.end(); ++ite) {
				if (ite->type() == beacls::UVecType_Invalid) *ite = beacls::UVec(depth, type, DD_size_base);
				else if (ite->size() < DD_size_base) ite->resize(DD_size_base);
				DD_size_base -= outer_dimensions_loop_length*num_of_slices;
			}
		}
	}

	FLOAT_TYPE* dst_dL0_ptr = (dst_dL.size() >= 1) ? beacls::UVec_<FLOAT_TYPE>(dst_dL[0]).ptr() : NULL;
	FLOAT_TYPE* dst_dR0_ptr = (dst_dR.size() >= 1) ? beacls::UVec_<FLOAT_TYPE>(dst_dR[0]).ptr() : NULL;
	FLOAT_TYPE* dst_dL1_ptr = (dst_dL.size() >= 2) ? beacls::UVec_<FLOAT_TYPE>(dst_dL[1]).ptr() : NULL;
	FLOAT_TYPE* dst_dR1_ptr = (dst_dR.size() >= 2) ? beacls::UVec_<FLOAT_TYPE>(dst_dR[1]).ptr() : NULL;
	FLOAT_TYPE* dst_dL2_ptr = (dst_dL.size() >= 3) ? beacls::UVec_<FLOAT_TYPE>(dst_dL[2]).ptr() : NULL;
	FLOAT_TYPE* dst_dR2_ptr = (dst_dR.size() >= 3) ? beacls::UVec_<FLOAT_TYPE>(dst_dR[2]).ptr() : NULL;
	FLOAT_TYPE* dst_dL3_ptr = (dst_dL.size() >= 4) ? beacls::UVec_<FLOAT_TYPE>(dst_dL[3]).ptr() : NULL;
	FLOAT_TYPE* dst_dR3_ptr = (dst_dR.size() >= 4) ? beacls::UVec_<FLOAT_TYPE>(dst_dR[3]).ptr() : NULL;
	FLOAT_TYPE* dst_DD0_ptr = (dst_DD.size() >= 1) ? beacls::UVec_<FLOAT_TYPE>(dst_DD[0]).ptr() : NULL;
	FLOAT_TYPE* dst_DD1_ptr = (dst_DD.size() >= 2) ? beacls::UVec_<FLOAT_TYPE>(dst_DD[1]).ptr() : NULL;
	FLOAT_TYPE* dst_DD2_ptr = (dst_DD.size() >= 3) ? beacls::UVec_<FLOAT_TYPE>(dst_DD[2]).ptr() : NULL;
	beacls::UVec& boundedSrc = tmpBoundedSrc_uvecs[dim];
	size_t total_boundedSrc_size = target_dimension_loop_size * outer_dimensions_loop_length * num_of_slices;
	// Add ghost cells.
	if (boundedSrc.type() != type) boundedSrc = beacls::UVec(depth, type, total_boundedSrc_size);
	else if (boundedSrc.size() != total_boundedSrc_size) boundedSrc.resize(total_boundedSrc_size);
	boundedSrc.set_cudaStream(cudaStream);
	boundaryCondition->execute(
		src,
		boundedSrc,
		tmpBoundedSrc_uvec_vectors[dim],
		stencil,
		outer_dimensions_loop_length,
		target_dimension_loop_size,
		first_dimension_loop_size,
		loop_begin,
		num_of_slices
		);
	FLOAT_TYPE* boundedSrc_base_ptr	= beacls::UVec_<FLOAT_TYPE>(boundedSrc).ptr();
	size_t dst_DD0_slice_size = saveDD ? dst_DD[0].size() / num_of_slices : 0;
	size_t dst_DD1_size = saveDD ? dst_DD[1].size() / num_of_slices : 0;
	size_t dst_DD2_size = saveDD ? dst_DD[2].size() / num_of_slices : 0;

	size_t dst_DD0_line_length = dst_DD0_slice_size / outer_dimensions_loop_length;
	size_t dst_DD1_line_length = dst_DD1_size / outer_dimensions_loop_length;
	size_t dst_DD2_line_length = dst_DD2_size / outer_dimensions_loop_length;
	if (type == beacls::UVecType_Cuda) {
		boundedSrc_base_ptr = beacls::UVec_<FLOAT_TYPE>(boundedSrc).ptr();
		UpwindFirstENO3aHelper_execute_dim0_cuda(
			dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dL3_ptr,
			dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, dst_dR3_ptr,
			dst_DD0_ptr, dst_DD1_ptr, dst_DD2_ptr,
			boundedSrc_base_ptr, dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
			num_of_slices,
			outer_dimensions_loop_length,
			target_dimension_loop_size,
			first_dimension_loop_size,
			slice_length,
			stencil,
			saveDD,
			approx4,
			stripDD,
			dst_DD0_line_length,
			dst_DD1_line_length,
			dst_DD2_line_length, 
			cudaStream);
	}
	else
	{
		for (size_t slice_index = 0; slice_index < num_of_slices; ++slice_index) {
			const size_t slice_loop_offset = slice_index * outer_dimensions_loop_length;
			const size_t dst_dLdR_slice_offset = slice_loop_offset * first_dimension_loop_size;
			for (size_t loop_index = 0; loop_index < outer_dimensions_loop_length; ++loop_index) {
				const size_t loop_index_with_slice = loop_index + slice_loop_offset;
				size_t dst_dLdR_offset = loop_index * first_dimension_loop_size + dst_dLdR_slice_offset;
				//! Target dimension is first loop
				const FLOAT_TYPE* boundedSrc_ptr = boundedSrc_base_ptr + (first_dimension_loop_size + stencil * 2) * loop_index_with_slice;

				FLOAT_TYPE d0_m0 = boundedSrc_ptr[0];
				FLOAT_TYPE d0_m1 = 0;
				FLOAT_TYPE d1_m0 = 0;
				FLOAT_TYPE d1_m1 = 0;
				FLOAT_TYPE d1_m2 = 0;
				FLOAT_TYPE d1_m3 = 0;
				FLOAT_TYPE d2_m0 = 0;
				FLOAT_TYPE d2_m1 = 0;
				FLOAT_TYPE d2_m2 = 0;
				FLOAT_TYPE d2_m3 = 0;
				FLOAT_TYPE d3_m0 = 0;
				FLOAT_TYPE d3_m1 = 0;
				FLOAT_TYPE d3_m2 = 0;
				FLOAT_TYPE d3_m3 = 0;

				size_t dst_DD0_offset = loop_index_with_slice * (first_dimension_loop_size + (stripDD ? 1 : 5));
				size_t dst_DD1_offset = loop_index_with_slice * (first_dimension_loop_size + (stripDD ? 2 : 4));
				size_t dst_DD2_offset = loop_index_with_slice * (first_dimension_loop_size + 3);
				size_t dst_DD0_strip = stripDD ? 2 : 0;
				size_t dst_DD1_strip = stripDD ? 2 : 1;
				size_t dst_DD2_strip = 2;


				//! Prologue
				for (size_t target_dimension_loop_index = 0; target_dimension_loop_index < stencil + 2; ++target_dimension_loop_index) {
					size_t src_index = target_dimension_loop_index;
					d0_m1 = d0_m0;
					d0_m0 = boundedSrc_ptr[src_index + 1];
					d1_m3 = d1_m2;
					d1_m2 = d1_m1;
					d1_m1 = d1_m0;
					d1_m0 = dxInv * (d0_m0 - d0_m1);
					if (target_dimension_loop_index >= 1) {
						d2_m3 = d2_m2;
						d2_m2 = d2_m1;
						d2_m1 = d2_m0;
						d2_m0 = dxInv_2 * (d1_m0 - d1_m1);
						if (target_dimension_loop_index >= 2) {
							d3_m3 = d3_m2;
							d3_m2 = d3_m1;
							d3_m1 = d3_m0;
							d3_m0 = dxInv_3 * (d2_m0 - d2_m1);
							if (saveDD) dst_DD2_ptr[target_dimension_loop_index + dst_DD2_offset - dst_DD2_strip] = d3_m0;
						}
						if (saveDD) {
							if (target_dimension_loop_index >= dst_DD1_strip) dst_DD1_ptr[target_dimension_loop_index + dst_DD1_offset - dst_DD1_strip] = d2_m0;
						}
					}
					if (saveDD) {
						if (target_dimension_loop_index >= dst_DD0_strip) dst_DD0_ptr[target_dimension_loop_index + dst_DD0_offset - dst_DD0_strip] = d1_m0;
					}
				}
				//! Body
				if (approx4) {
					if (saveDD) {
						for (size_t target_dimension_loop_index = stencil + 2; target_dimension_loop_index < target_dimension_loop_size - stencil; ++target_dimension_loop_index) {
							calcD1toD3_dim0(
								d0_m0, d0_m1, d1_m0, d1_m1, d1_m2, d1_m3, d2_m0, d2_m1, d2_m2, d2_m3,
								d3_m0, d3_m1, d3_m2, d3_m3, boundedSrc_ptr, dxInv, dxInv_2, dxInv_3, target_dimension_loop_index);
							dst_DD2_ptr[target_dimension_loop_index + dst_DD2_offset - dst_DD2_strip] = d3_m0;
							dst_DD1_ptr[target_dimension_loop_index + dst_DD1_offset - dst_DD1_strip] = d2_m0;
							dst_DD0_ptr[target_dimension_loop_index + dst_DD0_offset - dst_DD0_strip] = d1_m0;
							size_t target_dimension_loop_index_stencil = target_dimension_loop_index - stencil - 1;
							size_t dst_index = target_dimension_loop_index_stencil + dst_dLdR_offset - 1;
							calcApprox1to3(dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, d1_m2, d1_m3, d2_m1, d2_m2, d2_m3, d3_m0, d3_m1, d3_m2, d3_m3, dx, x2_dx_square, dx_square, dst_index);
							calcApprox4(dst_dL3_ptr, d1_m3, dx * d2_m2, -dx_square * d3_m2, dst_index);
							calcApprox4(dst_dR3_ptr, d1_m2, -dx * d2_m1, x2_dx_square * d3_m1, dst_index);
						}
					}
					else {
						for (size_t target_dimension_loop_index = stencil + 2; target_dimension_loop_index < target_dimension_loop_size - stencil; ++target_dimension_loop_index) {
							calcD1toD3_dim0(
								d0_m0, d0_m1, d1_m0, d1_m1, d1_m2, d1_m3, d2_m0, d2_m1, d2_m2, d2_m3,
								d3_m0, d3_m1, d3_m2, d3_m3, boundedSrc_ptr, dxInv, dxInv_2, dxInv_3, target_dimension_loop_index);
							size_t target_dimension_loop_index_stencil = target_dimension_loop_index - stencil - 1;
							size_t dst_index = target_dimension_loop_index_stencil + dst_dLdR_offset - 1;
							calcApprox1to3(dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, d1_m2, d1_m3, d2_m1, d2_m2, d2_m3, d3_m0, d3_m1, d3_m2, d3_m3, dx, x2_dx_square, dx_square, dst_index);
							calcApprox4(dst_dL3_ptr, d1_m3, dx * d2_m2, -dx_square * d3_m2, dst_index);
							calcApprox4(dst_dR3_ptr, d1_m2, -dx * d2_m1, x2_dx_square * d3_m1, dst_index);
						}
					}
				}
				else {
					if (saveDD) {
						for (size_t target_dimension_loop_index = stencil + 2; target_dimension_loop_index < target_dimension_loop_size - stencil; ++target_dimension_loop_index) {
							calcD1toD3_dim0(
								d0_m0, d0_m1, d1_m0, d1_m1, d1_m2, d1_m3, d2_m0, d2_m1, d2_m2, d2_m3,
								d3_m0, d3_m1, d3_m2, d3_m3, boundedSrc_ptr, dxInv, dxInv_2, dxInv_3, target_dimension_loop_index);
							dst_DD2_ptr[target_dimension_loop_index + dst_DD2_offset - dst_DD2_strip] = d3_m0;
							dst_DD1_ptr[target_dimension_loop_index + dst_DD1_offset - dst_DD1_strip] = d2_m0;
							dst_DD0_ptr[target_dimension_loop_index + dst_DD0_offset - dst_DD0_strip] = d1_m0;
							size_t target_dimension_loop_index_stencil = target_dimension_loop_index - stencil - 1;
							size_t dst_index = target_dimension_loop_index_stencil + dst_dLdR_offset - 1;
							calcApprox1to3(dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, d1_m2, d1_m3, d2_m1, d2_m2, d2_m3, d3_m0, d3_m1, d3_m2, d3_m3, dx, x2_dx_square, dx_square, dst_index);
						}
					}
					else {
						for (size_t target_dimension_loop_index = stencil + 2; target_dimension_loop_index < target_dimension_loop_size - stencil; ++target_dimension_loop_index) {
							calcD1toD3_dim0(
								d0_m0, d0_m1, d1_m0, d1_m1, d1_m2, d1_m3, d2_m0, d2_m1, d2_m2, d2_m3,
								d3_m0, d3_m1, d3_m2, d3_m3, boundedSrc_ptr, dxInv, dxInv_2, dxInv_3, target_dimension_loop_index);
							size_t target_dimension_loop_index_stencil = target_dimension_loop_index - stencil - 1;
							size_t dst_index = target_dimension_loop_index_stencil + dst_dLdR_offset - 1;
							calcApprox1to3(dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, d1_m2, d1_m3, d2_m1, d2_m2, d2_m3, d3_m0, d3_m1, d3_m2, d3_m3, dx, x2_dx_square, dx_square, dst_index);
						}
					}
				}
				//! Epilogue
				for (size_t target_dimension_loop_index = target_dimension_loop_size - stencil; target_dimension_loop_index < target_dimension_loop_size - 1; ++target_dimension_loop_index) {
					calcD1toD3_dim0(
						d0_m0, d0_m1, d1_m0, d1_m1, d1_m2, d1_m3, d2_m0, d2_m1, d2_m2, d2_m3,
						d3_m0, d3_m1, d3_m2, d3_m3, boundedSrc_ptr, dxInv, dxInv_2, dxInv_3, target_dimension_loop_index);

					if (saveDD) {
						if (target_dimension_loop_index < dst_DD2_line_length + dst_DD2_strip)
							dst_DD2_ptr[target_dimension_loop_index + dst_DD2_offset - dst_DD2_strip] = d3_m0;
						if (target_dimension_loop_index < dst_DD1_line_length + dst_DD1_strip)
							dst_DD1_ptr[target_dimension_loop_index + dst_DD1_offset - dst_DD1_strip] = d2_m0;
						if (target_dimension_loop_index < dst_DD0_line_length + dst_DD0_strip)
							dst_DD0_ptr[target_dimension_loop_index + dst_DD0_offset - dst_DD0_strip] = d1_m0;
					}
					size_t target_dimension_loop_index_stencil = target_dimension_loop_index - stencil - 1;
					size_t dst_index = target_dimension_loop_index_stencil + dst_dLdR_offset - 1;
					calcApprox1to3(dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, d1_m2, d1_m3, d2_m1, d2_m2, d2_m3, d3_m0, d3_m1, d3_m2, d3_m3, dx, x2_dx_square, dx_square, dst_index);
					if (approx4) {
						calcApprox4(dst_dL3_ptr, d1_m3, dx * d2_m2, -(dx_square * d3_m2), dst_index);
						calcApprox4(dst_dR3_ptr, d1_m2, -dx * d2_m1, x2_dx_square * d3_m1, dst_index);
					}
				}
			}
		}
	}
	return true;
}

bool UpwindFirstENO3aHelper_impl::execute_dim1(
	std::vector<beacls::UVec > &dst_dL,
	std::vector<beacls::UVec > &dst_dR,
	std::vector<beacls::UVec > &dst_DD,
	const FLOAT_TYPE* src,
	const BoundaryCondition *boundaryCondition,
	const size_t dim,
	const bool approx4,
	const bool stripDD,
	const size_t loop_begin,
	const size_t slice_length,
	const size_t num_of_slices,
	beacls::CudaStream* cudaStream
) {

	const size_t num_of_cache_lines = stripDD ? 4 : 6;
	const size_t src_target_dimension_loop_size = src_target_dimension_loop_sizes[dim];
	const size_t inner_dimensions_loop_size = inner_dimensions_loop_sizes[dim];

	const size_t first_dimension_loop_size = first_dimension_loop_sizes[dim];
	createCaches(first_dimension_loop_size, num_of_cache_lines);
	const size_t loop_length = slice_length / first_dimension_loop_size;
	const size_t target_dimension_loop_size = target_dimension_loop_sizes[dim];
	const size_t total_slices_length = slice_length * num_of_slices;

	size_t num_of_dsts = approx4 ? 4 : 3;
	if (dst_dL.size() != num_of_dsts) dst_dL.resize(num_of_dsts);
	if (dst_dR.size() != num_of_dsts) dst_dR.resize(num_of_dsts);
	beacls::UVecDepth depth = beacls::type_to_depth<FLOAT_TYPE>();
	for_each(dst_dL.begin(), dst_dL.end(), ([total_slices_length, depth, this](auto &rhs) {
		if (rhs.type() == beacls::UVecType_Invalid) rhs = beacls::UVec(depth, type, total_slices_length);
		else if (rhs.size() < total_slices_length) rhs.resize(total_slices_length);
	}));
	for_each(dst_dR.begin(), dst_dR.end(), ([total_slices_length, depth, this](auto &rhs) {
		if (rhs.type() == beacls::UVecType_Invalid) rhs = beacls::UVec(depth, type, total_slices_length);
		else if (rhs.size() < total_slices_length) rhs.resize(total_slices_length);
	}));

	size_t target_dimension_loop_begin = loop_begin % src_target_dimension_loop_size;
	size_t prologue_loop_size;
	size_t prologue_loop_dst_offset;
	size_t prologue_loop_src_offset;
	if (type == beacls::UVecType_Cuda) {
		prologue_loop_size = stencil * 2;
		prologue_loop_dst_offset = 0;
		prologue_loop_src_offset = 0;
	}
	else if (target_dimension_loop_begin < stencil * 2) {
			prologue_loop_size =  stencil * 2 - target_dimension_loop_begin;
			prologue_loop_dst_offset = target_dimension_loop_begin;
			prologue_loop_src_offset = 0;
	}
	else {
		prologue_loop_size = 1;
		prologue_loop_dst_offset = stencil * 2;
		prologue_loop_src_offset = 1;
	}
	std::vector<std::vector<std::vector<const FLOAT_TYPE*> > >& dst_ptrsss = tmpBoundedSrc_ptrssss[dim];
	if (dst_ptrsss.size() < num_of_slices) dst_ptrsss.resize(num_of_slices);
	for_each(dst_ptrsss.begin(), dst_ptrsss.end(), [loop_length, prologue_loop_size](auto &rhs) {
		rhs.resize(loop_length+ prologue_loop_size);
		for_each(rhs.begin(), rhs.end(), [](auto &rhs) {
			rhs.resize(1);
		});
	});
	beacls::UVec& boundedSrc = tmpBoundedSrc_uvecs[dim];
	size_t total_boundedSrc_size = first_dimension_loop_size * num_of_slices*(loop_length + prologue_loop_size);
	if (boundedSrc.type() != type) boundedSrc = beacls::UVec(depth, type, total_boundedSrc_size);
	else if (boundedSrc.size() != total_boundedSrc_size) boundedSrc.resize(total_boundedSrc_size);
	boundedSrc.set_cudaStream(cudaStream);
	bool saveDD = (dst_DD.size() == 3);
	const FLOAT_TYPE dxInv = dxInvs[dim];
	const FLOAT_TYPE dxInv_2 = dxInv_2s[dim];
	const FLOAT_TYPE dxInv_3 = dxInv_3s[dim];
	const FLOAT_TYPE dx_square = dx_squares[dim];
	const FLOAT_TYPE x2_dx_square = 2 * dx_square;
	const FLOAT_TYPE dx = dxs[dim];


	if (tmp_d1s_ms_ites.size() != num_of_cache_lines) tmp_d1s_ms_ites.resize(num_of_cache_lines);
	if (tmp_d2s_ms_ites.size() != num_of_cache_lines) tmp_d2s_ms_ites.resize(num_of_cache_lines);
	if (tmp_d3s_ms_ites.size() != num_of_cache_lines) tmp_d3s_ms_ites.resize(num_of_cache_lines);
	if (tmp_d1s_ms_ites2.size() != num_of_cache_lines) tmp_d1s_ms_ites2.resize(num_of_cache_lines);
	if (tmp_d2s_ms_ites2.size() != num_of_cache_lines) tmp_d2s_ms_ites2.resize(num_of_cache_lines);
	if (tmp_d3s_ms_ites2.size() != num_of_cache_lines) tmp_d3s_ms_ites2.resize(num_of_cache_lines);


	if (saveDD) {
		if (stripDD) {
			size_t DD_size_base = first_dimension_loop_size * (loop_length + 1) * num_of_slices;
			for (std::vector<beacls::UVec >::iterator ite = dst_DD.begin(); ite != dst_DD.end(); ++ite) {
				if (ite->type() == beacls::UVecType_Invalid) *ite = beacls::UVec(depth, type, DD_size_base);
				else if (ite->size() < DD_size_base) ite->resize(DD_size_base);
				DD_size_base += first_dimension_loop_size * num_of_slices;
			}
		}
		else {
			size_t DD_size_base = first_dimension_loop_size * (loop_length + stencil + 2) * num_of_slices;
			for (std::vector<beacls::UVec >::iterator ite = dst_DD.begin(); ite != dst_DD.end(); ++ite) {
				if (ite->type() == beacls::UVecType_Invalid) *ite = beacls::UVec(depth, type, DD_size_base);
				else if (ite->size() < DD_size_base) ite->resize(DD_size_base);
				DD_size_base -= first_dimension_loop_size * num_of_slices;
			}
		}
	}

	std::vector<FLOAT_TYPE*> &d1s_ms = tmp_d1s_ms_ites;
	std::vector<FLOAT_TYPE*> &d2s_ms = tmp_d2s_ms_ites;
	std::vector<FLOAT_TYPE*> &d3s_ms = tmp_d3s_ms_ites;
	size_t dst_DD0_slice_size = saveDD ? dst_DD[0].size() / num_of_slices : 0;
	size_t dst_DD1_slice_size = saveDD ? dst_DD[1].size() / num_of_slices : 0;
	size_t dst_DD2_slice_size = saveDD ? dst_DD[2].size() / num_of_slices : 0;

	bool stepHead;
	if (stripDD) stepHead = false;
	else if ((loop_begin % src_target_dimension_loop_size) == 0) stepHead = true;
	else stepHead = false;

	std::vector<beacls::UVec >& tmp_DD = dst_DD;
	FLOAT_TYPE* dst_dL0_ptr = (dst_dL.size() >= 1) ? beacls::UVec_<FLOAT_TYPE>(dst_dL[0]).ptr() : NULL;
	FLOAT_TYPE* dst_dR0_ptr = (dst_dR.size() >= 1) ? beacls::UVec_<FLOAT_TYPE>(dst_dR[0]).ptr() : NULL;
	FLOAT_TYPE* dst_dL1_ptr = (dst_dL.size() >= 2) ? beacls::UVec_<FLOAT_TYPE>(dst_dL[1]).ptr() : NULL;
	FLOAT_TYPE* dst_dR1_ptr = (dst_dR.size() >= 2) ? beacls::UVec_<FLOAT_TYPE>(dst_dR[1]).ptr() : NULL;
	FLOAT_TYPE* dst_dL2_ptr = (dst_dL.size() >= 3) ? beacls::UVec_<FLOAT_TYPE>(dst_dL[2]).ptr() : NULL;
	FLOAT_TYPE* dst_dR2_ptr = (dst_dR.size() >= 3) ? beacls::UVec_<FLOAT_TYPE>(dst_dR[2]).ptr() : NULL;
	FLOAT_TYPE* dst_dL3_ptr = (dst_dL.size() >= 4) ? beacls::UVec_<FLOAT_TYPE>(dst_dL[3]).ptr() : NULL;
	FLOAT_TYPE* dst_dR3_ptr = (dst_dR.size() >= 4) ? beacls::UVec_<FLOAT_TYPE>(dst_dR[3]).ptr() : NULL;
	FLOAT_TYPE* dst_DD0_ptr = (dst_DD.size() >= 1) ? beacls::UVec_<FLOAT_TYPE>(dst_DD[0]).ptr() : NULL;
	FLOAT_TYPE* dst_DD1_ptr = (dst_DD.size() >= 2) ? beacls::UVec_<FLOAT_TYPE>(dst_DD[1]).ptr() : NULL;
	FLOAT_TYPE* dst_DD2_ptr = (dst_DD.size() >= 3) ? beacls::UVec_<FLOAT_TYPE>(dst_DD[2]).ptr() : NULL;
	boundaryCondition->execute(
		src,
		boundedSrc, tmpBoundedSrc_uvec_vectors[dim], dst_ptrsss,
		stencil,
		dim,
		target_dimension_loop_size,
		inner_dimensions_loop_size,
		first_dimension_loop_size,
		loop_begin - prologue_loop_src_offset,
		prologue_loop_dst_offset,
		num_of_slices,
		loop_length + prologue_loop_size,
		1,
		stencil
	);
	if (type == beacls::UVecType_Cuda) {
		UpwindFirstENO3aHelper_execute_dim1_cuda(
			dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dL3_ptr,
			dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, dst_dR3_ptr,
			dst_DD0_ptr, dst_DD1_ptr, dst_DD2_ptr,
			dst_ptrsss[0][0][0], dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
			num_of_slices, loop_length, first_dimension_loop_size, slice_length,
			stencil,
			dst_DD0_slice_size, dst_DD1_slice_size, dst_DD2_slice_size,
			saveDD, approx4, stripDD,
			cudaStream
		);
	}else
	{
		for (size_t slice_index = 0; slice_index < num_of_slices; ++slice_index) {
			size_t prologue_offset = 0;
			const size_t slice_offset = slice_index * slice_length;
			const size_t slice_loop_offset = slice_index * loop_length;
			const size_t slice_loop_begin = loop_begin + slice_loop_offset;
			for (size_t index = 0; index < loop_length; ++index) {
				const size_t dst_loop_offset = index * first_dimension_loop_size;
				const size_t dst_slice_offset = dst_loop_offset + slice_offset;
				const size_t loop_index = index + slice_loop_begin;
				const size_t target_dimension_loop_index = loop_index % src_target_dimension_loop_size;

				//! Prologue
				if (index == 0) {
					if (target_dimension_loop_index == 0) {
						for (size_t prologue_target_dimension_loop_index = 0; prologue_target_dimension_loop_index < stencil * 2; ++prologue_target_dimension_loop_index) {
							if ((prologue_target_dimension_loop_index >= 1)) {
								const FLOAT_TYPE* current_boundedSrc_ptr = dst_ptrsss[slice_index][prologue_target_dimension_loop_index][0];
								const FLOAT_TYPE* last_boundedSrc_ptr = dst_ptrsss[slice_index][prologue_target_dimension_loop_index - 1][0];

								bool d1_m0_writeToCache, d2_m0_writeToCache, d3_m0_writeToCache;
								getCachePointers(d1s_ms, d2s_ms, d3s_ms, tmp_DD,
									d1_m0_writeToCache, d2_m0_writeToCache, d3_m0_writeToCache,
									slice_index, prologue_target_dimension_loop_index, first_dimension_loop_size,
									dst_loop_offset + prologue_offset, num_of_slices, 
									num_of_cache_lines, saveDD, stepHead);

								for (size_t first_dimension_loop_index = 0; first_dimension_loop_index < first_dimension_loop_size; ++first_dimension_loop_index) {
									FLOAT_TYPE this_src = current_boundedSrc_ptr[first_dimension_loop_index];
									FLOAT_TYPE last_src = last_boundedSrc_ptr[first_dimension_loop_index];
									FLOAT_TYPE d1_m0 = dxInv * (this_src - last_src);
									d1s_ms[0][first_dimension_loop_index] = d1_m0;
									if (prologue_target_dimension_loop_index >= 2) {
										FLOAT_TYPE d1_m1 = d1s_ms[1][first_dimension_loop_index];
										FLOAT_TYPE d2_m0 = dxInv_2 * (d1_m0 - d1_m1);
										d2s_ms[0][first_dimension_loop_index] = d2_m0;
										if (prologue_target_dimension_loop_index >= 3) {
											FLOAT_TYPE d2_m1 = d2s_ms[1][first_dimension_loop_index];
											FLOAT_TYPE d3_m0 = dxInv_3 * (d2_m0 - d2_m1);
											d3s_ms[0][first_dimension_loop_index] = d3_m0;
										}
									}
								}
								prologue_offset += first_dimension_loop_size;
								//! Epilogue (Prologue may overlap Epilogue, when chunk size is short.
								if ((prologue_target_dimension_loop_index + num_of_cache_lines) >= ((stencil * 2 + loop_length) + 1)) {
									if (saveDD) {
										//! Copy back from DD to cache for next chunk
										bool tmp_d1_m0_writeToCache, tmp_d2_m0_writeToCache, tmp_d3_m0_writeToCache;
										getCachePointers(tmp_d1s_ms_ites2, tmp_d2s_ms_ites2, tmp_d3s_ms_ites2, tmp_DD,
											tmp_d1_m0_writeToCache, tmp_d2_m0_writeToCache, tmp_d3_m0_writeToCache,
											slice_index, prologue_target_dimension_loop_index, first_dimension_loop_size,
											dst_loop_offset + prologue_offset, num_of_slices,
											num_of_cache_lines, false, false);
										if (!d1_m0_writeToCache) memcpy(tmp_d1s_ms_ites2[0], d1s_ms[0], first_dimension_loop_size * sizeof(FLOAT_TYPE));
										if (!d2_m0_writeToCache) memcpy(tmp_d2s_ms_ites2[0], d2s_ms[0], first_dimension_loop_size * sizeof(FLOAT_TYPE));
										if (!d3_m0_writeToCache) memcpy(tmp_d3s_ms_ites2[0], d3s_ms[0], first_dimension_loop_size * sizeof(FLOAT_TYPE));
									}
								}
							}
						}
					}
					else {
						size_t shifted_target_dimension_loop_index = target_dimension_loop_index + stencil * 2;
						bool d1_m0_writeToCache, d2_m0_writeToCache, d3_m0_writeToCache;
						getCachePointers(d1s_ms, d2s_ms, d3s_ms, tmp_DD,
							d1_m0_writeToCache, d2_m0_writeToCache, d3_m0_writeToCache,
							slice_index, shifted_target_dimension_loop_index, first_dimension_loop_size,
							dst_loop_offset + prologue_offset, num_of_slices,
							num_of_cache_lines, saveDD, stepHead);
						for (size_t i = 1; i < d1s_ms.size(); i++) {
							size_t cache_offset = dst_slice_offset + first_dimension_loop_size * (num_of_cache_lines - 1 - i);
							if (dst_DD0_slice_size >= (cache_offset + first_dimension_loop_size)) memcpy((beacls::UVec_<FLOAT_TYPE>(tmp_DD[0]).ptr() + cache_offset), d1s_ms[i], first_dimension_loop_size * sizeof(FLOAT_TYPE));
							if (dst_DD1_slice_size >= (cache_offset + first_dimension_loop_size)) memcpy((beacls::UVec_<FLOAT_TYPE>(tmp_DD[1]).ptr() + cache_offset), d2s_ms[i], first_dimension_loop_size * sizeof(FLOAT_TYPE));
							if (dst_DD2_slice_size >= (cache_offset + first_dimension_loop_size)) memcpy((beacls::UVec_<FLOAT_TYPE>(tmp_DD[2]).ptr() + cache_offset), d3s_ms[i], first_dimension_loop_size * sizeof(FLOAT_TYPE));
						}
						prologue_offset = first_dimension_loop_size*(2 + num_of_cache_lines - 1);
					}
				}
				if (target_dimension_loop_index < (target_dimension_loop_size - (stencil + 1))) {
					//! Body
					size_t shifted_target_dimension_loop_index = target_dimension_loop_index + stencil * 2;
					//				size_t boundedSrc_cache_current_index = shifted_target_dimension_loop_index & 0x1;
					//				size_t boundedSrc_cache_last_index = (boundedSrc_cache_current_index == 0) ? 1 : 0;

					const FLOAT_TYPE* last_boundedSrc_ptr = NULL;
					const FLOAT_TYPE* current_boundedSrc_ptr = dst_ptrsss[slice_index][index + prologue_loop_size][0];
					last_boundedSrc_ptr = dst_ptrsss[slice_index][index + prologue_loop_size - 1][0];
					bool d1_m0_writeToCache, d2_m0_writeToCache, d3_m0_writeToCache;
					getCachePointers(d1s_ms, d2s_ms, d3s_ms, tmp_DD,
						d1_m0_writeToCache, d2_m0_writeToCache, d3_m0_writeToCache,
						slice_index, shifted_target_dimension_loop_index, first_dimension_loop_size,
						dst_loop_offset + prologue_offset, num_of_slices,
						num_of_cache_lines, saveDD, stepHead);
					const FLOAT_TYPE* d1s_ms1 = d1s_ms[1];
					const FLOAT_TYPE* d1s_ms2 = d1s_ms[2];
					const FLOAT_TYPE* d1s_ms3 = d1s_ms[3];
					const FLOAT_TYPE* d2s_ms1 = d2s_ms[1];
					const FLOAT_TYPE* d2s_ms2 = d2s_ms[2];
					const FLOAT_TYPE* d2s_ms3 = d2s_ms[3];
					const FLOAT_TYPE* d3s_ms1 = d3s_ms[1];
					const FLOAT_TYPE* d3s_ms2 = d3s_ms[2];
					const FLOAT_TYPE* d3s_ms3 = d3s_ms[3];
					if (approx4) {
						for (size_t first_dimension_loop_index = 0; first_dimension_loop_index < first_dimension_loop_size; ++first_dimension_loop_index) {
							size_t dst_index = first_dimension_loop_index + dst_slice_offset;
							FLOAT_TYPE d1_m0, d1_m1, d1_m2, d1_m3, d2_m0, d2_m1, d2_m2, d2_m3, d3_m0, d3_m1, d3_m2, d3_m3;
							calcD1toD3_dim1(
								d1_m0, d1_m1, d1_m2, d1_m3, d2_m0, d2_m1, d2_m2, d2_m3, d3_m0, d3_m1, d3_m2, d3_m3,
								d1s_ms1, d1s_ms2, d1s_ms3, d2s_ms1, d2s_ms2, d2s_ms3, d3s_ms1, d3s_ms2, d3s_ms3,
								last_boundedSrc_ptr, current_boundedSrc_ptr, dxInv, dxInv_2, dxInv_3, first_dimension_loop_index
							);
							calcApprox1to3(dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, d1_m2, d1_m3, d2_m1, d2_m2, d2_m3, d3_m0, d3_m1, d3_m2, d3_m3, dx, x2_dx_square, dx_square, dst_index);
							calcApprox4(dst_dL3_ptr, d1_m2, dx * d2_m1, -dx_square * d3_m2, dst_index);
							calcApprox4(dst_dR3_ptr, d1_m0, -dx * d2_m0, x2_dx_square * d3_m1, dst_index);
							d1s_ms[0][first_dimension_loop_index] = d1_m0;
							d2s_ms[0][first_dimension_loop_index] = d2_m0;
							d3s_ms[0][first_dimension_loop_index] = d3_m0;
						}
					}
					else {
						for (size_t first_dimension_loop_index = 0; first_dimension_loop_index < first_dimension_loop_size; ++first_dimension_loop_index) {
							size_t dst_index = first_dimension_loop_index + dst_slice_offset;
							FLOAT_TYPE d1_m0, d1_m1, d1_m2, d1_m3, d2_m0, d2_m1, d2_m2, d2_m3, d3_m0, d3_m1, d3_m2, d3_m3;
							calcD1toD3_dim1(
								d1_m0, d1_m1, d1_m2, d1_m3, d2_m0, d2_m1, d2_m2, d2_m3, d3_m0, d3_m1, d3_m2, d3_m3,
								d1s_ms1, d1s_ms2, d1s_ms3, d2s_ms1, d2s_ms2, d2s_ms3, d3s_ms1, d3s_ms2, d3s_ms3,
								last_boundedSrc_ptr, current_boundedSrc_ptr, dxInv, dxInv_2, dxInv_3, first_dimension_loop_index
							);
							calcApprox1to3(dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, d1_m2, d1_m3, d2_m1, d2_m2, d2_m3, d3_m0, d3_m1, d3_m2, d3_m3, dx, x2_dx_square, dx_square, dst_index);
							d1s_ms[0][first_dimension_loop_index] = d1_m0;
							d2s_ms[0][first_dimension_loop_index] = d2_m0;
							d3s_ms[0][first_dimension_loop_index] = d3_m0;
						}
					}
					//! Epilogue
					if ((target_dimension_loop_index + num_of_cache_lines) >= (loop_length + 1)) {
						if (saveDD) {
							//! Copy back from DD to cache for next chunk
							bool tmp_d1_m0_writeToCache, tmp_d2_m0_writeToCache, tmp_d3_m0_writeToCache;
							getCachePointers(tmp_d1s_ms_ites2, tmp_d2s_ms_ites2, tmp_d3s_ms_ites2, tmp_DD,
								tmp_d1_m0_writeToCache, tmp_d2_m0_writeToCache, tmp_d3_m0_writeToCache,
								slice_index, shifted_target_dimension_loop_index, first_dimension_loop_size,
								dst_loop_offset + prologue_offset, num_of_slices,
								num_of_cache_lines, false, false);
							if (!d1_m0_writeToCache) memcpy(tmp_d1s_ms_ites2[0], d1s_ms[0], first_dimension_loop_size * sizeof(FLOAT_TYPE));
							if (!d2_m0_writeToCache) memcpy(tmp_d2s_ms_ites2[0], d2s_ms[0], first_dimension_loop_size * sizeof(FLOAT_TYPE));
							if (!d3_m0_writeToCache) memcpy(tmp_d3s_ms_ites2[0], d3s_ms[0], first_dimension_loop_size * sizeof(FLOAT_TYPE));
						}
					}
				}
			}
		}
	}
	return true;
}

bool UpwindFirstENO3aHelper_impl::execute_dimLET2(
	std::vector<beacls::UVec > &dst_dL,
	std::vector<beacls::UVec > &dst_dR,
	std::vector<beacls::UVec > &dst_DD,
	const FLOAT_TYPE* src,
	const BoundaryCondition *boundaryCondition,
	const size_t dim,
	const bool approx4,
	const bool stripDD,
	const size_t loop_begin,
	const size_t slice_length,
	const size_t num_of_slices,
	const size_t num_of_strides,
	beacls::CudaStream* cudaStream
) {
	const size_t inner_dimensions_loop_size = inner_dimensions_loop_sizes[dim];
	const FLOAT_TYPE dxInv = dxInvs[dim];
	const FLOAT_TYPE dxInv_2 = dxInv_2s[dim];
	const FLOAT_TYPE dxInv_3 = dxInv_3s[dim];
	const FLOAT_TYPE dx_square = dx_squares[dim];
	const FLOAT_TYPE x2_dx_square = 2 * dx_square;
	const FLOAT_TYPE dx = dxs[dim];
	const size_t first_dimension_loop_size = first_dimension_loop_sizes[dim];
	const size_t target_dimension_loop_size = target_dimension_loop_sizes[dim];
	const size_t loop_length = slice_length / first_dimension_loop_size;
	const size_t total_slices_length = slice_length * num_of_slices;

	bool saveDD = (dst_DD.size() == 3);
	const size_t num_of_merged_slices = ((dim == 2) ? 1 : num_of_slices);
	const size_t num_of_merged_strides = ((dim == 2) ? num_of_strides + num_of_slices - 1 : num_of_strides);
	const size_t num_of_boundary_strides = std::max<size_t>(stencil * 2 + 1, stencil + num_of_merged_strides);
	const size_t num_of_dLdR_in_slice = (dim == 2) ? num_of_slices : 1;
	size_t num_of_dsts = approx4 ? 4 : 3;
	beacls::UVecDepth depth = beacls::type_to_depth<FLOAT_TYPE>();
//	beacls::IntegerVec DD_sizes;
	const size_t DD_size = first_dimension_loop_size * loop_length * num_of_merged_slices;
	if (saveDD) {
		const size_t total_DD_size = DD_size * num_of_merged_strides;
		if (dst_DD.size() != num_of_dsts) dst_DD.resize(num_of_dsts);
		for_each(dst_DD.begin(), dst_DD.end(), ([total_DD_size, depth, this](auto &rhs) {
			if (rhs.type() == beacls::UVecType_Invalid) rhs = beacls::UVec(depth, type, total_DD_size);
			else if (rhs.size() < total_DD_size) rhs.resize(total_DD_size);
		}));
	}

	if (dst_dL.size() != num_of_dsts) dst_dL.resize(num_of_dsts);
	if (dst_dR.size() != num_of_dsts) dst_dR.resize(num_of_dsts);

	for_each(dst_dL.begin(), dst_dL.end(), ([total_slices_length, depth, this](auto &rhs) {
		if (rhs.type() == beacls::UVecType_Invalid) rhs = beacls::UVec(depth, type, total_slices_length);
		else if (rhs.size() < total_slices_length) rhs.resize(total_slices_length);
	}));
	for_each(dst_dR.begin(), dst_dR.end(), ([total_slices_length, depth, this](auto &rhs) {
		if (rhs.type() == beacls::UVecType_Invalid) rhs = beacls::UVec(depth, type, total_slices_length);
		else if (rhs.size() < total_slices_length) rhs.resize(total_slices_length);
	}));
	std::vector<std::vector<std::vector<const FLOAT_TYPE*> > >& tmpBoundedSrc_ptrsss = tmpBoundedSrc_ptrssss[dim];

	if (tmpBoundedSrc_ptrsss.size() != num_of_merged_slices) tmpBoundedSrc_ptrsss.resize(num_of_merged_slices);
	for_each(tmpBoundedSrc_ptrsss.begin(), tmpBoundedSrc_ptrsss.end(), ([loop_length, num_of_boundary_strides](auto &rhs) {
		if (rhs.size() < loop_length) rhs.resize(loop_length);
		for_each(rhs.begin(), rhs.end(), ([num_of_boundary_strides](auto &rhs) {
			if (rhs.size() < num_of_boundary_strides) rhs.resize(num_of_boundary_strides);
		}));
	}));
	beacls::UVec& boundedSrc = tmpBoundedSrc_uvecs[dim];
	size_t total_boundedSrc_size = first_dimension_loop_size * num_of_boundary_strides * num_of_merged_slices * loop_length;
	if (boundedSrc.type() != type) boundedSrc = beacls::UVec(depth, type, total_boundedSrc_size);
	else if (boundedSrc.size() != total_boundedSrc_size) boundedSrc.resize(total_boundedSrc_size);
	boundedSrc.set_cudaStream(cudaStream);
	size_t num_of_buffer_lines = 3;
	if (d1ss.size() != num_of_buffer_lines) d1ss.resize(num_of_buffer_lines);
	if (d2ss.size() != num_of_buffer_lines) d2ss.resize(num_of_buffer_lines);
	if (d3ss.size() != num_of_buffer_lines) d3ss.resize(num_of_buffer_lines);
	for_each(d1ss.begin(), d1ss.end(), ([first_dimension_loop_size](auto &rhs) {
		if (rhs.size() < first_dimension_loop_size) rhs.resize(first_dimension_loop_size, 0);
	}));
	for_each(d2ss.begin(), d2ss.end(), ([first_dimension_loop_size](auto &rhs) {
		if (rhs.size() < first_dimension_loop_size) rhs.resize(first_dimension_loop_size, 0);
	}));
	for_each(d3ss.begin(), d3ss.end(), ([first_dimension_loop_size](auto &rhs) {
		if (rhs.size() < first_dimension_loop_size) rhs.resize(first_dimension_loop_size, 0);
	}));

	FLOAT_TYPE* dst_dL0_ptr = (dst_dL.size() >= 1) ? beacls::UVec_<FLOAT_TYPE>(dst_dL[0]).ptr() : NULL;
	FLOAT_TYPE* dst_dR0_ptr = (dst_dR.size() >= 1) ? beacls::UVec_<FLOAT_TYPE>(dst_dR[0]).ptr() : NULL;
	FLOAT_TYPE* dst_dL1_ptr = (dst_dL.size() >= 2) ? beacls::UVec_<FLOAT_TYPE>(dst_dL[1]).ptr() : NULL;
	FLOAT_TYPE* dst_dR1_ptr = (dst_dR.size() >= 2) ? beacls::UVec_<FLOAT_TYPE>(dst_dR[1]).ptr() : NULL;
	FLOAT_TYPE* dst_dL2_ptr = (dst_dL.size() >= 3) ? beacls::UVec_<FLOAT_TYPE>(dst_dL[2]).ptr() : NULL;
	FLOAT_TYPE* dst_dR2_ptr = (dst_dR.size() >= 3) ? beacls::UVec_<FLOAT_TYPE>(dst_dR[2]).ptr() : NULL;
	FLOAT_TYPE* dst_dL3_ptr = (dst_dL.size() >= 4) ? beacls::UVec_<FLOAT_TYPE>(dst_dL[3]).ptr() : NULL;
	FLOAT_TYPE* dst_dR3_ptr = (dst_dR.size() >= 4) ? beacls::UVec_<FLOAT_TYPE>(dst_dR[3]).ptr() : NULL;
	std::vector<FLOAT_TYPE*> dst_DD0_ptrs(num_of_slices * num_of_strides);
	std::vector<FLOAT_TYPE*> dst_DD1_ptrs(num_of_slices * num_of_strides);
	std::vector<FLOAT_TYPE*> dst_DD2_ptrs(num_of_slices * num_of_strides);
	FLOAT_TYPE* dst_DD0_ptr = beacls::UVec_<FLOAT_TYPE>(dst_DD[0]).ptr();
	FLOAT_TYPE* dst_DD1_ptr = beacls::UVec_<FLOAT_TYPE>(dst_DD[1]).ptr();
	FLOAT_TYPE* dst_DD2_ptr = beacls::UVec_<FLOAT_TYPE>(dst_DD[2]).ptr();
	for (size_t stride_index = 0; stride_index < num_of_strides; ++stride_index) {
		for (size_t slice_index = 0; slice_index < num_of_slices; ++slice_index) {
			dst_DD0_ptrs[slice_index + stride_index*num_of_slices] = dst_DD0_ptr;
			dst_DD0_ptr += DD_size;
			dst_DD1_ptrs[slice_index + stride_index*num_of_slices] = dst_DD1_ptr;
			dst_DD1_ptr += DD_size;
			dst_DD2_ptrs[slice_index + stride_index*num_of_slices] = dst_DD2_ptr;
			dst_DD2_ptr += DD_size;
		}
	}
	boundaryCondition->execute(
		src,
		boundedSrc, tmpBoundedSrc_uvec_vectors[dim], tmpBoundedSrc_ptrsss,
		stencil,
		dim,
		target_dimension_loop_size,
		inner_dimensions_loop_size,
		first_dimension_loop_size,
		loop_begin,
		0,
		num_of_merged_slices,
		loop_length,
		num_of_boundary_strides,
		stencil
	);
	if (type == beacls::UVecType_Cuda) {
		UpwindFirstENO3aHelper_execute_dimLET2_cuda(
			dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dL3_ptr,
			dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, dst_dR3_ptr,
			dst_DD0_ptrs[0], dst_DD1_ptrs[0], dst_DD2_ptrs[0],
			tmpBoundedSrc_ptrsss[0][0][0], dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
			num_of_merged_slices, loop_length, first_dimension_loop_size, 
			num_of_merged_strides, num_of_dLdR_in_slice, slice_length, 
			saveDD, approx4, stripDD,
			cudaStream
		);
	} else
	{
		if (saveDD) {
			if (approx4) {
				if (stripDD) {
					calc_D1toD3andDD_dimLET2<FLOAT_TYPE, Approx4, StripDD>(
						dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dL3_ptr, dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, dst_dR3_ptr, dst_DD0_ptrs, dst_DD1_ptrs, dst_DD2_ptrs,
						d2ss, d3ss, tmpBoundedSrc_ptrsss, dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
						loop_length, first_dimension_loop_size, num_of_buffer_lines, slice_length, num_of_merged_slices, num_of_merged_strides, num_of_dLdR_in_slice);
				}
				else {
					calc_D1toD3andDD_dimLET2<FLOAT_TYPE, Approx4, noStripDD>(
						dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dL3_ptr, dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, dst_dR3_ptr, dst_DD0_ptrs, dst_DD1_ptrs, dst_DD2_ptrs,
						d2ss, d3ss, tmpBoundedSrc_ptrsss, dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
						loop_length, first_dimension_loop_size, num_of_buffer_lines, slice_length, num_of_merged_slices, num_of_merged_strides, num_of_dLdR_in_slice);
				}
			}
			else {
				if (stripDD) {
					calc_D1toD3andDD_dimLET2<FLOAT_TYPE, noApprox4, StripDD>(
						dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dL3_ptr, dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, dst_dR3_ptr, dst_DD0_ptrs, dst_DD1_ptrs, dst_DD2_ptrs,
						d2ss, d3ss, tmpBoundedSrc_ptrsss, dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
						loop_length, first_dimension_loop_size, num_of_buffer_lines, slice_length, num_of_merged_slices, num_of_merged_strides, num_of_dLdR_in_slice);
				}
				else {
					calc_D1toD3andDD_dimLET2<FLOAT_TYPE, noApprox4, noStripDD>(
						dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dL3_ptr, dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, dst_dR3_ptr, dst_DD0_ptrs, dst_DD1_ptrs, dst_DD2_ptrs,
						d2ss, d3ss, tmpBoundedSrc_ptrsss, dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
						loop_length, first_dimension_loop_size, num_of_buffer_lines, slice_length, num_of_merged_slices, num_of_merged_strides, num_of_dLdR_in_slice);
				}
			}
		}
		else {
			if (approx4) {
				calc_D1toD3_dimLET2<FLOAT_TYPE, Approx4>(
					dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dL3_ptr, dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, dst_dR3_ptr,
					tmpBoundedSrc_ptrsss, dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
					loop_length, first_dimension_loop_size, slice_length, num_of_slices);
			}
			else {
				calc_D1toD3_dimLET2<FLOAT_TYPE, noApprox4>(
					dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dL3_ptr, dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, dst_dR3_ptr,
					tmpBoundedSrc_ptrsss, dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
					loop_length, first_dimension_loop_size, slice_length, num_of_slices);
			}
		}
	}
	return true;
}

bool UpwindFirstENO3aHelper_impl::execute(
	std::vector<beacls::UVec > &dst_dL,
	std::vector<beacls::UVec > &dst_dR,
	std::vector<beacls::UVec > &dst_DD,
	const HJI_Grid *grid,
	const FLOAT_TYPE* src,
	const size_t dim,
	const bool approx4,
	const bool stripDD,
	const size_t loop_begin,
	const size_t slice_length,
	const size_t num_of_slices,
	const size_t num_of_strides,
	beacls::CudaStream* cudaStream
) {

	BoundaryCondition* boundaryCondition = grid->get_boundaryCondition(dim);
	switch (dim) {
	case 0:
	{
		execute_dim0(
			dst_dL,
			dst_dR,
			dst_DD,
			src,
			boundaryCondition,
			dim,
			approx4,
			stripDD,
			loop_begin,
			slice_length,
			num_of_slices,
			cudaStream
		);
	}
	break;
	case 1:
	{
		execute_dim1(
			dst_dL,
			dst_dR,
			dst_DD,
			src,
			boundaryCondition,
			dim,
			approx4,
			stripDD,
			loop_begin,
			slice_length,
			num_of_slices,
			cudaStream
		);
	}
	break;
	default:
	{
		execute_dimLET2(
			dst_dL,
			dst_dR,
			dst_DD,
			src,
			boundaryCondition,
			dim,
			approx4,
			stripDD,
			loop_begin,
			slice_length,
			num_of_slices,
			num_of_strides,
			cudaStream
		);
	}
	break;
	}
	return true;
}


UpwindFirstENO3aHelper::UpwindFirstENO3aHelper(
	const HJI_Grid *hji_grid,
	const beacls::UVecType type
	) {
	pimpl = new UpwindFirstENO3aHelper_impl(hji_grid,type);
}
UpwindFirstENO3aHelper::~UpwindFirstENO3aHelper() {
	if (pimpl) delete pimpl;
}

bool UpwindFirstENO3aHelper::execute(
	std::vector<beacls::UVec > &dst_dL,
	std::vector<beacls::UVec > &dst_dR,
	std::vector<beacls::UVec > &dst_DD,
	const HJI_Grid *grid,
	const FLOAT_TYPE* src,
	const size_t dim,
	const bool approx4,
	const bool stripDD,
	const size_t loop_begin,
	const size_t slice_length,
	const size_t num_of_slices,
	const size_t num_of_strides,
	beacls::CudaStream* cudaStream
) {
	if (pimpl) return pimpl->execute(dst_dL, dst_dR, dst_DD, grid, src, dim, approx4, stripDD, loop_begin, slice_length, num_of_slices, num_of_strides, cudaStream);
	else return false;
}
size_t UpwindFirstENO3aHelper::get_max_DD_size(const size_t dim) const {
	if (pimpl) return pimpl->get_max_DD_size(dim);
	else return 0;

}
bool UpwindFirstENO3aHelper::operator==(const UpwindFirstENO3aHelper& rhs) const {
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

UpwindFirstENO3aHelper::UpwindFirstENO3aHelper(const UpwindFirstENO3aHelper& rhs) :
	pimpl(rhs.pimpl->clone())
{
}

UpwindFirstENO3aHelper* UpwindFirstENO3aHelper::clone() const {
	return new UpwindFirstENO3aHelper(*this);
}
