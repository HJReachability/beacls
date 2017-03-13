#include <vector>
#include <cstdint>
#include <cmath>
#include <typeinfo>
#include <levelset/Grids/HJI_Grid.hpp>
#include <levelset/BoundaryCondition/BoundaryCondition.hpp>
#include <levelset/SpatialDerivative/UpwindFirst/UpwindFirstENO3bHelper.hpp>
#include "UpwindFirstENO3bHelper_impl.hpp"

UpwindFirstENO3bHelper_impl::UpwindFirstENO3bHelper_impl(
	const HJI_Grid *hji_grid,
	const beacls::UVecType type
) :
	type(type),
	dxs(hji_grid->get_dxs()),
	dxInvs(hji_grid->get_dxInvs()),
	stencil(2),
	tmpBoundedSrcs(stencil * 2 + 1),
	tmpBoundedSrc_ptrs(stencil * 2 + 1)
{
	cache = new UpwindFirstENO3bHelper_Cache;

	size_t num_of_dimensions = hji_grid->get_num_of_dimensions();
	dxInv_2s.resize(num_of_dimensions);
	outer_dimensions_loop_sizes.resize(num_of_dimensions);
	target_dimension_loop_sizes.resize(num_of_dimensions);
	inner_dimensions_loop_sizes.resize(num_of_dimensions);
	first_dimension_loop_sizes.resize(num_of_dimensions);
	src_target_dimension_loop_sizes.resize(num_of_dimensions);


	bounded_first_dimension_line_cache.resize(hji_grid->get_N(0)+ 2 * stencil);

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

}
UpwindFirstENO3bHelper_impl::~UpwindFirstENO3bHelper_impl() {
	if (cache) delete cache;
}
bool UpwindFirstENO3bHelper_impl::operator==(const UpwindFirstENO3bHelper_impl& rhs) const {
	if (this == &rhs) return true;
	else if (type != rhs.type) return false;
	else return true;
}

bool UpwindFirstENO3bHelper_impl::execute_dim0(
	/*
	beacls::FloatVec &dst_deriv,
	beacls::FloatVec &dst_smooth,
	beacls::FloatVec &dst_epsilon,
	const FLOAT_TYPE* src,
	const size_t dim,
	const bool direction,
	const size_t loop_begin,
	const size_t slice_length
	*/
	beacls::UVec &,
	beacls::UVec &,
	beacls::UVec &,
	const FLOAT_TYPE*,
	const size_t,
	const bool,
	const size_t,
	const size_t,
	const size_t,
	beacls::CudaStream*
) {
	/*
	size_t first_dimension_loop_size = first_dimension_loop_sizes[dim];
	size_t outer_dimensions_loop_length = slice_length / first_dimension_loop_size;
	for (size_t outer_dimensions_loop_index = loop_begin;
	outer_dimensions_loop_index < loop_begin+outer_dimensions_loop_length;
	++outer_dimensions_loop_index) {
	beacls::FloatVec& boundedSrc = bounded_first_dimension_line_cache;

	const FLOAT_TYPE dxInv = dxInvs[dim];
	const FLOAT_TYPE dxInv_2 = dxInv_2s[dim];
	const FLOAT_TYPE dx = dxs[dim];
	const size_t target_dimension_loop_size = target_dimension_loop_sizes[dim];
	size_t dst_offset = (outer_dimensions_loop_index - loop_begin) * first_dimension_loop_size;

	//! Target dimension is first loop

	// Add ghost cells.
	boundaryCondition->execute(
	src,
	boundedSrc,
	stencil,
	outer_dimensions_loop_index,
	target_dimension_loop_size);

	FLOAT_TYPE this_src = boundedSrc[0];
	FLOAT_TYPE d1_m1 = 0;
	FLOAT_TYPE d2_m1 = 0;
	FLOAT_TYPE d2_m1_fabs = 0;

	//! Prologue
	for (size_t target_dimension_loop_index = 0; target_dimension_loop_index < stencil + 1; ++target_dimension_loop_index) {
	size_t src_index = target_dimension_loop_index;
	FLOAT_TYPE next_src = boundedSrc[src_index + 1];
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
	dst_deriv_l[dst_deriv_l_index] = d1_m1 + dx_d2_eff;
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
	FLOAT_TYPE next_src = boundedSrc[src_index + 1];
	FLOAT_TYPE d1 = dxInv * (next_src - this_src);
	this_src = next_src;
	FLOAT_TYPE d2 = dxInv_2 * (d1 - d1_m1);
	FLOAT_TYPE d2_fabs = HjiFabs<FLOAT_TYPE>(d2);
	FLOAT_TYPE dx_d2_eff = dx * ((d2_m1_fabs < d2_fabs) ? d2_m1 : d2);
	size_t target_dimension_loop_index_stencil = target_dimension_loop_index - stencil;
	size_t dst_deriv_l_index = target_dimension_loop_index_stencil + dst_offset;
	dst_deriv_l[dst_deriv_l_index] = d1_m1 + dx_d2_eff;
	size_t dst_deriv_r_index = dst_deriv_l_index - 1;
	dst_deriv_r[dst_deriv_r_index] = d1_m1 - dx_d2_eff;
	d2_m1 = d2;
	d2_m1_fabs = d2_fabs;
	d1_m1 = d1;
	}
	//! Epilogue
	for (size_t target_dimension_loop_index = target_dimension_loop_size - stencil; target_dimension_loop_index < target_dimension_loop_size - stencil+1; ++target_dimension_loop_index) {
	size_t src_index = target_dimension_loop_index;
	FLOAT_TYPE next_src = boundedSrc[src_index + 1];
	FLOAT_TYPE d1 = dxInv * (next_src - this_src);
	this_src = next_src;
	FLOAT_TYPE d2 = dxInv_2 * (d1 - d1_m1);
	FLOAT_TYPE d2_fabs = HjiFabs<FLOAT_TYPE>(d2);
	FLOAT_TYPE dx_d2_eff = dx * ((d2_m1_fabs < d2_fabs) ? d2_m1 : d2);
	size_t target_dimension_loop_index_stencil = target_dimension_loop_index - stencil;
	size_t dst_deriv_r_index = (target_dimension_loop_index_stencil - 1) + dst_offset;
	dst_deriv_r[dst_deriv_r_index] = d1_m1 - dx_d2_eff;
	d2_m1 = d2;
	d2_m1_fabs = d2_fabs;
	d1_m1 = d1;
	}
	}
	*/
	return true;
}

bool UpwindFirstENO3bHelper_impl::execute_dim1(
	/*	beacls::FloatVec &dst_deriv,
	beacls::FloatVec &dst_smooth,
	beacls::FloatVec &dst_epsilon,
	const FLOAT_TYPE* src,
	const size_t dim,
	const bool direction,
	const size_t loop_begin,
	const size_t slice_length
	*/
	beacls::UVec &,
	beacls::UVec &,
	beacls::UVec &,
	const FLOAT_TYPE*,
	const size_t,
	const bool,
	const size_t,
	const size_t,
	const size_t,
	beacls::CudaStream*
) {
	/*
	std::vector<beacls::FloatVec > &cachedBoundedSrcs = cache->cachedBoundedSrcs;
	std::vector<beacls::FloatVec > &last_d1ss = cache->last_d1ss;
	beacls::FloatVec &last_d2s = cache->last_d2s;
	beacls::FloatVec &last_d2s_fabs = cache->last_d2s_fabs;
	beacls::FloatVec &last_dx_d2_effs = cache->last_dx_d2_effs;
	std::vector<const FLOAT_TYPE*> &boundedSrc_ptrs = cache->boundedSrc_ptrs;

	const size_t src_target_dimension_loop_size = src_target_dimension_loop_sizes[dim];
	const size_t inner_dimensions_loop_size = inner_dimensions_loop_sizes[dim];

	size_t first_dimension_loop_size = first_dimension_loop_sizes[dim];
	size_t loop_length = slice_length / first_dimension_loop_size;

	for (size_t index = loop_begin; index < loop_begin+ loop_length; ++index) {

	size_t loop_index_div_target_size = index / src_target_dimension_loop_size;
	size_t target_dimension_loop_index = index % src_target_dimension_loop_size;
	size_t inner_dimensions_loop_index = loop_index_div_target_size % inner_dimensions_loop_size;
	size_t outer_dimensions_loop_index = loop_index_div_target_size / inner_dimensions_loop_size;
	size_t dst_offset = (index - loop_begin) * first_dimension_loop_size;


	const FLOAT_TYPE dxInv = dxInvs[dim];
	const FLOAT_TYPE dxInv_2 = dxInv_2s[dim];
	const FLOAT_TYPE dx = dxs[dim];

	const size_t target_dimension_loop_size = target_dimension_loop_sizes[dim];

	const FLOAT_TYPE* dst_ptr;

	//! Prologue
	if (target_dimension_loop_index == 0) {
	for (size_t prologue_target_dimension_loop_index = 0; prologue_target_dimension_loop_index < stencil * 2; ++prologue_target_dimension_loop_index) {
	size_t boundedSrc_cache_current_index = prologue_target_dimension_loop_index & 0x1;
	size_t boundedSrc_cache_last_index = (boundedSrc_cache_current_index == 0) ? 1 : 0;
	beacls::FloatVec& d1s_m1 = last_d1ss[boundedSrc_cache_current_index];
	beacls::FloatVec& d1s_m2 = last_d1ss[boundedSrc_cache_last_index];
	beacls::FloatVec& boundedSrc = cachedBoundedSrcs[boundedSrc_cache_current_index];
	const FLOAT_TYPE* &last_boundedSrc_ptr = boundedSrc_ptrs[boundedSrc_cache_last_index];
	const FLOAT_TYPE* &current_boundedSrc_ptr = boundedSrc_ptrs[boundedSrc_cache_current_index];

	if (target_dimension_loop_index < target_dimension_loop_size) {
	// Add ghost cells.
	boundaryCondition->execute(
	src,
	boundedSrc, dst_ptr,
	stencil,
	outer_dimensions_loop_index,
	target_dimension_loop_size, prologue_target_dimension_loop_index,
	inner_dimensions_loop_size, inner_dimensions_loop_index,
	first_dimension_loop_size
	);
	boundedSrc_begin = dst_ptr;
	}
	if ((prologue_target_dimension_loop_index >= 1)) {
	for (size_t first_dimension_loop_index = 0; first_dimension_loop_index < first_dimension_loop_size; ++first_dimension_loop_index) {
	FLOAT_TYPE this_src = boundedSrc_begin[first_dimension_loop_index];
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
	beacls::FloatVec &deriv_l = cachedBoundedSrcs[boundedSrc_cache_current_index];
	const FLOAT_TYPE* &last_boundedSrc_ptr = boundedSrc_ptrs[boundedSrc_cache_last_index];
	const FLOAT_TYPE* &current_boundedSrc_ptr = boundedSrc_ptrs[boundedSrc_cache_current_index];
	if (shifted_target_dimension_loop_index < target_dimension_loop_size) {
	// Add ghost cells.
	boundaryCondition->execute(
	src,
	boundedSrc, dst_ptr,
	stencil,
	outer_dimensions_loop_index,
	target_dimension_loop_size, shifted_target_dimension_loop_index,
	inner_dimensions_loop_size, inner_dimensions_loop_index,
	first_dimension_loop_size
	);
	boundedSrc_begin = dst_ptr;
	}
	beacls::FloatVec& d1s_m1 = last_d1ss[boundedSrc_cache_current_index];
	beacls::FloatVec& d1s_m2 = last_d1ss[boundedSrc_cache_last_index];
	for (size_t first_dimension_loop_index = 0; first_dimension_loop_index < first_dimension_loop_size; ++first_dimension_loop_index) {
	FLOAT_TYPE this_src = boundedSrc_begin[first_dimension_loop_index];
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

	dst_deriv_l[deriv_index] = d1_m2 + dx_d2_eff_m1;
	dst_deriv_r[deriv_index] = d1_m1 - dx_d2_eff;
	last_d2s[first_dimension_loop_index] = d2;
	last_d2s_fabs[first_dimension_loop_index] = d2_fabs;
	d1s_m2[first_dimension_loop_index] = d1;
	last_dx_d2_effs[first_dimension_loop_index] = dx_d2_eff;
	}
	}
	}
	*/
	return true;
}

bool UpwindFirstENO3bHelper_impl::execute_dimLET2(
	/*	beacls::FloatVec &dst_deriv,
	beacls::FloatVec &dst_smooth,
	beacls::FloatVec &dst_epsilon,
	const FLOAT_TYPE* src,
	const size_t dim,
	const bool direction,
	const size_t loop_begin,
	const size_t slice_length*/
	beacls::UVec &,
	beacls::UVec &,
	beacls::UVec &,
	const FLOAT_TYPE*,
	const size_t,
	const bool,
	const size_t,
	const size_t,
	const size_t,
	beacls::CudaStream*
) {
	/*
	const size_t src_target_dimension_loop_size = src_target_dimension_loop_sizes[dim];
	const size_t inner_dimensions_loop_size = inner_dimensions_loop_sizes[dim];

	size_t first_dimension_loop_size = first_dimension_loop_sizes[dim];
	size_t loop_length = slice_length / first_dimension_loop_size;
	for (size_t index = loop_begin; index < loop_begin+ loop_length; ++index) {

	size_t loop_index_div_inner_size = index / inner_dimensions_loop_size;
	size_t inner_dimensions_loop_index = index % inner_dimensions_loop_size;
	size_t outer_dimensions_loop_index = loop_index_div_inner_size / src_target_dimension_loop_size;
	size_t target_dimension_loop_index = loop_index_div_inner_size % src_target_dimension_loop_size;
	size_t dst_offset = (index - loop_begin) * first_dimension_loop_size;

	const FLOAT_TYPE dxInv = dxInvs[dim];
	const FLOAT_TYPE dxInv_2 = dxInv_2s[dim];
	const FLOAT_TYPE dx = dxs[dim];

	const size_t target_dimension_loop_size = target_dimension_loop_sizes[dim];

	for (size_t loop_index = 0; loop_index <= stencil * 2; ++loop_index) {
	if ((loop_index + target_dimension_loop_index) < target_dimension_loop_size) {
	const FLOAT_TYPE* dst_ptr;
	boundaryCondition->execute(
	src,
	tmpBoundedSrcs[loop_index], dst_ptr,
	stencil,
	outer_dimensions_loop_index,
	target_dimension_loop_size, loop_index + target_dimension_loop_index,
	inner_dimensions_loop_size, inner_dimensions_loop_index,
	first_dimension_loop_size
	);
	tmpBoundedSrc_ptrs[loop_index] = dst_ptr;
	}
	}

	size_t shifted_target_dimension_loop_index = target_dimension_loop_index + 2 * stencil;
	if (shifted_target_dimension_loop_index < target_dimension_loop_size) {
	for (size_t first_dimension_loop_index = 0; first_dimension_loop_index < first_dimension_loop_size; ++first_dimension_loop_index) {
	FLOAT_TYPE d0_0 = tmpBoundedSrc_ptrs[0][first_dimension_loop_index];
	FLOAT_TYPE d0_1 = tmpBoundedSrc_ptrs[1][first_dimension_loop_index];
	FLOAT_TYPE d1_0 = dxInv*(d0_1 - d0_0);

	FLOAT_TYPE d0_2 = tmpBoundedSrc_ptrs[2][first_dimension_loop_index];
	FLOAT_TYPE d1_1 = dxInv*(d0_2 - d0_1);
	FLOAT_TYPE d2_0 = dxInv_2 * (d1_1 - d1_0);
	FLOAT_TYPE fabs_d2_0 = HjiFabs<FLOAT_TYPE>(d2_0);

	FLOAT_TYPE d0_3 = tmpBoundedSrc_ptrs[3][first_dimension_loop_index];
	FLOAT_TYPE d1_2 = dxInv*(d0_3 - d0_2);
	FLOAT_TYPE d2_1 = dxInv_2 * (d1_2 - d1_1);
	FLOAT_TYPE fabs_d2_1 = HjiFabs<FLOAT_TYPE>(d2_1);
	FLOAT_TYPE d2_eff_0 = (fabs_d2_0 < fabs_d2_1) ? d2_0 : d2_1;
	FLOAT_TYPE dx_d2_eff_0 = dx * d2_eff_0;

	FLOAT_TYPE d0_4 = tmpBoundedSrc_ptrs[4][first_dimension_loop_index];
	FLOAT_TYPE d1_3 = dxInv*(d0_4 - d0_3);
	FLOAT_TYPE d2_2 = dxInv_2 * (d1_3 - d1_2);
	FLOAT_TYPE fabs_d2_2 = HjiFabs<FLOAT_TYPE>(d2_2);
	FLOAT_TYPE d2_eff_1 = (fabs_d2_1 < fabs_d2_2) ? d2_1 : d2_2;
	FLOAT_TYPE dx_d2_eff_1 = dx * d2_eff_1;
	size_t dst_index = first_dimension_loop_index + dst_offset;
	dst_deriv_l[dst_index] = d1_1 + dx_d2_eff_0;
	dst_deriv_r[dst_index] = d1_2 - dx_d2_eff_1;
	}
	}
	}
	*/
	return true;
}

bool UpwindFirstENO3bHelper_impl::execute(
	beacls::UVec &dst_deriv,
	beacls::UVec &dst_smooth,
	beacls::UVec &dst_epsilon,
	const HJI_Grid *,
	const FLOAT_TYPE* src,
	const size_t dim,
	const bool direction,
	const size_t loop_begin,
	const size_t slice_length,
	const size_t num_of_slices,
	beacls::CudaStream* cudaStream
) {
	switch (dim) {
	case 0:
	{
		execute_dim0(
			dst_deriv,
			dst_smooth,
			dst_epsilon,
			src,
			dim,
			direction,
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
			dst_deriv,
			dst_smooth,
			dst_epsilon,
			src,
			dim,
			direction,
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
			dst_deriv,
			dst_smooth,
			dst_epsilon,
			src,
			dim,
			direction,
			loop_begin,
			slice_length,
			num_of_slices,
			cudaStream
		);
	}
	break;
	}
	return true;
}



UpwindFirstENO3bHelper::UpwindFirstENO3bHelper(
	const HJI_Grid *hji_grid,
	const beacls::UVecType type
	) {
	pimpl = new UpwindFirstENO3bHelper_impl(hji_grid,type);
}
UpwindFirstENO3bHelper::~UpwindFirstENO3bHelper() {
	if (pimpl) delete pimpl;
}

bool UpwindFirstENO3bHelper::execute(
	beacls::UVec &dst_deriv,
	beacls::UVec &dst_smooth,
	beacls::UVec &dst_epsilon,
	const HJI_Grid *grid,
	const FLOAT_TYPE* src,
	const size_t dim,
	const bool direction,
	const size_t loop_begin,
	const size_t slice_length,
	const size_t num_of_slices,
	beacls::CudaStream* cudaStream
) {
	if (pimpl) return pimpl->execute(dst_deriv, dst_smooth, dst_epsilon, grid, src, dim, direction, loop_begin, slice_length, num_of_slices, cudaStream);
	else return false;
}
UpwindFirstENO3bHelper::UpwindFirstENO3bHelper(const UpwindFirstENO3bHelper& rhs) :
	pimpl(rhs.pimpl->clone())
{
}

UpwindFirstENO3bHelper* UpwindFirstENO3bHelper::clone() const {
	return new UpwindFirstENO3bHelper(*this);
}
