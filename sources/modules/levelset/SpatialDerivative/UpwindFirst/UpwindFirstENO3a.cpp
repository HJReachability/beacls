#include <vector>
#include <cstdint>
#include <cmath>
#include <cfloat>
#include <numeric>
#include <typeinfo>
#include <macro.hpp>
#include <Core/CudaStream.hpp>
#include <levelset/Grids/HJI_Grid.hpp>
#include <Core/CudaStream.hpp>
#include <levelset/BoundaryCondition/BoundaryCondition.hpp>
#include <levelset/SpatialDerivative/UpwindFirst/UpwindFirstENO3a.hpp>
#include "UpwindFirstENO3a_impl.hpp"
#include <levelset/SpatialDerivative/UpwindFirst/UpwindFirstENO3aHelper.hpp>
using namespace levelset;

UpwindFirstENO3a_impl::UpwindFirstENO3a_impl(
	const HJI_Grid *hji_grid,
	const beacls::UVecType type
) :
	type(type),
	stencil(3),
	num_of_strides(4),
	checkEquivalentApproximations(true)
{
	upwindFirstENO3aHelper = new UpwindFirstENO3aHelper(hji_grid, type);

	size_t num_of_dimensions = hji_grid->get_num_of_dimensions();
	inner_dimensions_loop_sizes.resize(num_of_dimensions);
	first_dimension_loop_sizes.resize(num_of_dimensions);
	src_target_dimension_loop_sizes.resize(num_of_dimensions);

	dL_uvecs.resize(num_of_dimensions);
	dR_uvecs.resize(num_of_dimensions);
	DD_uvecs.resize(num_of_dimensions);
	for (size_t target_dimension = 0; target_dimension < num_of_dimensions; ++target_dimension){
		std::vector<beacls::UVec >& dL_uvec = dL_uvecs[target_dimension];
		std::vector<beacls::UVec >& dR_uvec = dR_uvecs[target_dimension];
		std::vector<beacls::UVec >& DD_uvec = DD_uvecs[target_dimension];
		dL_uvec.resize(3);
		dR_uvec.resize(3);
		DD_uvec.resize(3);

		beacls::IntegerVec sizeIn = hji_grid->get_Ns();
		beacls::IntegerVec sizeOut = sizeIn;
		sizeOut[target_dimension] += 2 * stencil;

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
		src_target_dimension_loop_sizes[target_dimension] = src_target_dimension_loop_size;
		
	}
	cudaStreams.resize(num_of_dimensions, NULL);
	if (type == beacls::UVecType_Cuda) {
		std::for_each(cudaStreams.begin(), cudaStreams.end(), [](auto& rhs) {
			rhs = new beacls::CudaStream();
		});
	}
}
UpwindFirstENO3a_impl::~UpwindFirstENO3a_impl() {
	if (type == beacls::UVecType_Cuda) {
		std::for_each(cudaStreams.begin(), cudaStreams.end(), [](auto& rhs) {
			if (rhs) delete rhs;
			rhs = NULL;
		});
	}
	if (upwindFirstENO3aHelper) delete upwindFirstENO3aHelper;
}
bool UpwindFirstENO3a_impl::operator==(const UpwindFirstENO3a_impl& rhs) const {
	if (this == &rhs) return true;
	else if (type != rhs.type) return false;
	else if ((first_dimension_loop_sizes.size() != rhs.first_dimension_loop_sizes.size()) || !std::equal(first_dimension_loop_sizes.cbegin(), first_dimension_loop_sizes.cend(), rhs.first_dimension_loop_sizes.cbegin())) return false;
	else if ((inner_dimensions_loop_sizes.size() != rhs.inner_dimensions_loop_sizes.size()) || !std::equal(inner_dimensions_loop_sizes.cbegin(), inner_dimensions_loop_sizes.cend(), rhs.inner_dimensions_loop_sizes.cbegin())) return false;
	else if ((src_target_dimension_loop_sizes.size() != rhs.src_target_dimension_loop_sizes.size()) || !std::equal(src_target_dimension_loop_sizes.cbegin(), src_target_dimension_loop_sizes.cend(), rhs.src_target_dimension_loop_sizes.cbegin())) return false;
	else if ((upwindFirstENO3aHelper != rhs.upwindFirstENO3aHelper) && (!upwindFirstENO3aHelper || !rhs.upwindFirstENO3aHelper || !upwindFirstENO3aHelper->operator==(*rhs.upwindFirstENO3aHelper))) return false;
	else return true;
}

UpwindFirstENO3a_impl::UpwindFirstENO3a_impl(const UpwindFirstENO3a_impl& rhs) :
	type(rhs.type),
	first_dimension_loop_sizes(rhs.first_dimension_loop_sizes),
	inner_dimensions_loop_sizes(rhs.inner_dimensions_loop_sizes),
	src_target_dimension_loop_sizes(rhs.src_target_dimension_loop_sizes),
	stencil(rhs.stencil),
	num_of_strides(rhs.num_of_strides),
	checkEquivalentApproximations(rhs.checkEquivalentApproximations),
	upwindFirstENO3aHelper(rhs.upwindFirstENO3aHelper->clone())
{
	dL_uvecs.resize(rhs.dL_uvecs.size());
	for (size_t i = 0; i < dL_uvecs.size(); ++i) {
		dL_uvecs[i].resize(rhs.dL_uvecs[i].size());
	}
	dR_uvecs.resize(rhs.dR_uvecs.size());
	for (size_t i = 0; i < dR_uvecs.size(); ++i) {
		dR_uvecs[i].resize(rhs.dR_uvecs[i].size());
	}
	DD_uvecs.resize(rhs.DD_uvecs.size());
	for (size_t i = 0; i < DD_uvecs.size(); ++i) {
		DD_uvecs[i].resize(rhs.DD_uvecs[i].size());
	}
	cudaStreams.resize(rhs.cudaStreams.size());
	if (type == beacls::UVecType_Cuda) {
		std::for_each(cudaStreams.begin(), cudaStreams.end(), [](auto& rhs) {
			rhs = new beacls::CudaStream();
		});
	}
}
bool UpwindFirstENO3a_impl::execute_dim0(
	beacls::UVec& dst_deriv_l,
	beacls::UVec& dst_deriv_r,
	const FLOAT_TYPE* src,
	const HJI_Grid *grid,
	const size_t dim,
	const size_t loop_begin,
	const size_t slice_length,
	const size_t num_of_slices
) {
	const size_t first_dimension_loop_size = first_dimension_loop_sizes[dim];
	const size_t outer_dimensions_loop_length = slice_length / first_dimension_loop_size;
	const size_t total_slices_length = slice_length * num_of_slices;
	std::vector<beacls::UVec >& dL_uvec = dL_uvecs[dim];
	std::vector<beacls::UVec >& dR_uvec = dR_uvecs[dim];
	std::vector<beacls::UVec >& DD_uvec = DD_uvecs[dim];

	const bool stripDD = true;
	const bool approx4 = false;

	if (dL_uvec.size() != 3) dL_uvec.resize(3);
	if (dR_uvec.size() != 3) dR_uvec.resize(3);
	if (DD_uvec.size() != 3) DD_uvec.resize(3);
	beacls::UVecDepth depth = dst_deriv_l.depth();
	for_each(dL_uvec.begin(), dL_uvec.end(), ([total_slices_length, this, depth](auto &rhs) {
		if (rhs.type() != type || rhs.depth() != depth) rhs = beacls::UVec(depth, type, total_slices_length);
		else if (rhs.size() != total_slices_length) rhs.resize(total_slices_length);
	}));
	for_each(dR_uvec.begin(), dR_uvec.end(), ([total_slices_length, this, depth](auto &rhs) {
		if (rhs.type() != type || rhs.depth() != depth) rhs = beacls::UVec(depth, type, total_slices_length);
		else if (rhs.size() != total_slices_length) rhs.resize(total_slices_length);
	}));

	size_t DD_size_base = (first_dimension_loop_size + 1) * outer_dimensions_loop_length*num_of_slices;
	for (std::vector<beacls::UVec>::iterator ite = DD_uvec.begin(); ite != DD_uvec.end(); ++ite) {
		if (ite->type() != type || ite->depth() != depth) *ite = beacls::UVec(depth, type, DD_size_base);
		else if (ite->size() != DD_size_base) ite->resize(DD_size_base);
		DD_size_base += outer_dimensions_loop_length*num_of_slices;
	}

	upwindFirstENO3aHelper->execute(dL_uvec, dR_uvec, DD_uvec, grid, src, dim, approx4, stripDD, loop_begin, slice_length, num_of_slices, 1, cudaStreams[dim]);

#if 1	//!< Workaround, following operation doesn't support CUDA memory yet.
	std::vector<beacls::UVec > tmp_DD_uvec(DD_uvec.size());
	std::vector<beacls::UVec > tmp_dL_uvec(dL_uvec.size());
	std::vector<beacls::UVec > tmp_dR_uvec(dR_uvec.size());
	beacls::UVec tmp_deriv_l;
	beacls::UVec tmp_deriv_r;
	std::transform(dL_uvec.cbegin(), dL_uvec.cend(), tmp_dL_uvec.begin(), ([](const auto& rhs) {
		if (rhs.type() == beacls::UVecType_Cuda) {
			beacls::UVec tmp; rhs.convertTo(tmp, beacls::UVecType_Vector); return tmp;
		}
		else return rhs;
	}));
	std::transform(dR_uvec.cbegin(), dR_uvec.cend(), tmp_dR_uvec.begin(), ([](const auto& rhs) {
		if (rhs.type() == beacls::UVecType_Cuda) {
			beacls::UVec tmp; rhs.convertTo(tmp, beacls::UVecType_Vector); return tmp;
		}
		else return rhs;
	}));
	std::transform(DD_uvec.cbegin(), DD_uvec.cend(), tmp_DD_uvec.begin(), ([](const auto& rhs) {
		if (rhs.type() == beacls::UVecType_Cuda) {
			beacls::UVec tmp; rhs.convertTo(tmp, beacls::UVecType_Vector); return tmp;
		}
		else return rhs;
	}));
	if (dst_deriv_l.type() == beacls::UVecType_Cuda) tmp_deriv_l = beacls::UVec(dst_deriv_l.depth(), beacls::UVecType_Vector, dst_deriv_l.size());
	else tmp_deriv_l = dst_deriv_l;
	if (dst_deriv_r.type() == beacls::UVecType_Cuda) tmp_deriv_r = beacls::UVec(dst_deriv_r.depth(), beacls::UVecType_Vector, dst_deriv_r.size());
	else tmp_deriv_r = dst_deriv_r;

	FLOAT_TYPE* DD1_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_DD_uvec[1]).ptr();
	FLOAT_TYPE* DD2_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_DD_uvec[2]).ptr();
	FLOAT_TYPE* dL0_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_dL_uvec[0]).ptr();
	FLOAT_TYPE* dR0_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_dR_uvec[0]).ptr();
	FLOAT_TYPE* dL1_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_dL_uvec[1]).ptr();
	FLOAT_TYPE* dR1_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_dR_uvec[1]).ptr();
	FLOAT_TYPE* dL2_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_dL_uvec[2]).ptr();
	FLOAT_TYPE* dR2_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_dR_uvec[2]).ptr();
	FLOAT_TYPE* dst_deriv_l_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_deriv_l).ptr();
	FLOAT_TYPE* dst_deriv_r_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_deriv_r).ptr();
#else
	FLOAT_TYPE* DD1_ptr = beacls::UVec_<FLOAT_TYPE>(DD_uvec[1]).ptr();
	FLOAT_TYPE* DD2_ptr = beacls::UVec_<FLOAT_TYPE>(DD_uvec[2]).ptr();
	FLOAT_TYPE* dL0_ptr = beacls::UVec_<FLOAT_TYPE>(dL_uvec[0]).ptr();
	FLOAT_TYPE* dR0_ptr = beacls::UVec_<FLOAT_TYPE>(dR_uvec[0]).ptr();
	FLOAT_TYPE* dL1_ptr = beacls::UVec_<FLOAT_TYPE>(dL_uvec[1]).ptr();
	FLOAT_TYPE* dR1_ptr = beacls::UVec_<FLOAT_TYPE>(dR_uvec[1]).ptr();
	FLOAT_TYPE* dL2_ptr = beacls::UVec_<FLOAT_TYPE>(dL_uvec[2]).ptr();
	FLOAT_TYPE* dR2_ptr = beacls::UVec_<FLOAT_TYPE>(dR_uvec[2]).ptr();
	FLOAT_TYPE* dst_deriv_l_ptr = beacls::UVec_<FLOAT_TYPE>(dst_deriv_l).ptr();
	FLOAT_TYPE* dst_deriv_r_ptr = beacls::UVec_<FLOAT_TYPE>(dst_deriv_r).ptr();
#endif

	// Need to figure out which approximation has the least oscillation.
	// Note that L and R in this section refer to neighboring divided
	// difference entries, not to left and right approximations.
	const size_t src_target_dimension_loop_size = src_target_dimension_loop_sizes[dim];
	for (size_t slice_index = 0; slice_index < num_of_slices; ++slice_index) {
		const size_t slice_offset = slice_index * slice_length;
		const size_t slice_loop_offset = slice_index * outer_dimensions_loop_length;
		for (size_t index = 0; index < outer_dimensions_loop_length; ++index) {
			size_t dst_offset = index * first_dimension_loop_size + slice_offset;
			size_t src_D2_offset = (index + slice_loop_offset) * (first_dimension_loop_size + 2);
			size_t src_D3_offset = (index + slice_loop_offset) * (first_dimension_loop_size + 3);
			FLOAT_TYPE abs_D2_src_m1 = 0;
			FLOAT_TYPE abs_D2_src_m2 = 0;
			FLOAT_TYPE abs_D2_src_m3 = 0;
			FLOAT_TYPE abs_D3_src_m1 = 0;
			FLOAT_TYPE abs_D3_src_m2 = 0;
			FLOAT_TYPE abs_D3_src_m3 = 0;
			//! Prologue
			for (size_t target_dimension_loop_index = 0; target_dimension_loop_index < stencil; ++target_dimension_loop_index) {
				size_t src_D2_index = target_dimension_loop_index + src_D2_offset;
				size_t src_D3_index = target_dimension_loop_index + src_D3_offset;
				FLOAT_TYPE D2_src_m0 = DD1_ptr[src_D2_index];
				FLOAT_TYPE D3_src_m0 = DD2_ptr[src_D3_index];

				FLOAT_TYPE abs_D2_src_m0 = HjiFabs(D2_src_m0);
				abs_D2_src_m3 = abs_D2_src_m2;
				abs_D2_src_m2 = abs_D2_src_m1;
				abs_D2_src_m1 = abs_D2_src_m0;

				FLOAT_TYPE abs_D3_src_m0 = HjiFabs(D3_src_m0);
				abs_D3_src_m3 = abs_D3_src_m2;
				abs_D3_src_m2 = abs_D3_src_m1;
				abs_D3_src_m1 = abs_D3_src_m0;

			}
			//! Body
			for (size_t target_dimension_loop_index = stencil; target_dimension_loop_index < src_target_dimension_loop_size + stencil - 1; ++target_dimension_loop_index) {
				size_t src_D2_index = target_dimension_loop_index + src_D2_offset;
				size_t src_D3_index = target_dimension_loop_index + src_D3_offset;

				size_t src_index = target_dimension_loop_index + dst_offset - stencil;
				FLOAT_TYPE D2_src_m0 = DD1_ptr[src_D2_index];
				FLOAT_TYPE D3_src_m0 = DD2_ptr[src_D3_index];

				FLOAT_TYPE abs_D2_src_m0 = HjiFabs(D2_src_m0);
				FLOAT_TYPE abs_D3_src_m0 = HjiFabs(D3_src_m0);

				// Pick out minimum modulus neighboring D2 term.

				bool smallerL_0 = abs_D2_src_m3 < abs_D2_src_m2;
				bool smallerR_0 = !smallerL_0;
				bool smallerL_1 = abs_D2_src_m2 < abs_D2_src_m1;
				bool smallerR_1 = !smallerL_1;

				// Figure out smallest modulus D3 terms,
				// given choice of smallest modulus D2 terms above.
				bool smallerTempL_0 = abs_D3_src_m3 < abs_D3_src_m2;
				bool smallerTempL_1 = abs_D3_src_m2 < abs_D3_src_m1;
				bool smallerTempL_2 = abs_D3_src_m1 < abs_D3_src_m0;

				abs_D2_src_m3 = abs_D2_src_m2;
				abs_D2_src_m2 = abs_D2_src_m1;
				abs_D2_src_m1 = abs_D2_src_m0;

				abs_D3_src_m3 = abs_D3_src_m2;
				abs_D3_src_m2 = abs_D3_src_m1;
				abs_D3_src_m1 = abs_D3_src_m0;


				bool smallerLL_0 = smallerTempL_0 & smallerL_0;
				bool smallerLL_1 = smallerTempL_1 & smallerL_1;
				bool smallerRL_0 = smallerTempL_1 & smallerR_0;
				bool smallerRL_1 = smallerTempL_2 & smallerR_1;


				bool smallerTempR_0 = !smallerTempL_0;
				bool smallerTempR_1 = !smallerTempL_1;
				bool smallerTempR_2 = !smallerTempL_2;

				bool smallerLR_0 = smallerTempR_0 & smallerL_0;
				bool smallerLR_1 = smallerTempR_1 & smallerL_1;
				bool smallerRR_0 = smallerTempR_1 & smallerR_0;
				bool smallerRR_1 = smallerTempR_2 & smallerR_1;


				bool smallerM_0 = smallerRL_0 | smallerLR_0;
				bool smallerM_1 = smallerRL_1 | smallerLR_1;

				FLOAT_TYPE smallerLL_0_f = smallerLL_0 ? (FLOAT_TYPE)1. : 0;
				FLOAT_TYPE smallerM_0_f = smallerM_0 ? (FLOAT_TYPE)1. : 0;
				FLOAT_TYPE smallerRR_0_f = smallerRR_0 ? (FLOAT_TYPE)1. : 0;

				FLOAT_TYPE smallerLL_1_f = smallerLL_1 ? (FLOAT_TYPE)1. : 0;
				FLOAT_TYPE smallerM_1_f = smallerM_1 ? (FLOAT_TYPE)1. : 0;
				FLOAT_TYPE smallerRR_1_f = smallerRR_1 ? (FLOAT_TYPE)1. : 0;

				// Pick out the best third order approximation
				FLOAT_TYPE deriv_l = dL0_ptr[src_index] * smallerLL_0_f
					+ dL1_ptr[src_index] * smallerM_0_f
					+ dL2_ptr[src_index] * smallerRR_0_f;

				FLOAT_TYPE deriv_r = dR0_ptr[src_index] * smallerLL_1_f
					+ dR1_ptr[src_index] * smallerM_1_f
					+ dR2_ptr[src_index] * smallerRR_1_f;

				size_t dst_index = src_index;

				dst_deriv_l_ptr[dst_index] = deriv_l;
				dst_deriv_r_ptr[dst_index] = deriv_r;
			}
			//! Epilogue
			{
				size_t target_dimension_loop_index = src_target_dimension_loop_size + stencil - 1;
				size_t src_D3_index = target_dimension_loop_index + src_D3_offset;

				size_t src_index = target_dimension_loop_index + dst_offset - stencil;
				FLOAT_TYPE D3_src_m0 = DD2_ptr[src_D3_index];

				FLOAT_TYPE abs_D3_src_m0 = HjiFabs(D3_src_m0);

				// Pick out minimum modulus neighboring D2 term.

				bool smallerL_0 = abs_D2_src_m3 < abs_D2_src_m2;
				bool smallerR_0 = !smallerL_0;
				bool smallerL_1 = abs_D2_src_m2 < abs_D2_src_m1;
				bool smallerR_1 = !smallerL_1;

				// Figure out smallest modulus D3 terms,
				// given choice of smallest modulus D2 terms above.
				bool smallerTempL_0 = abs_D3_src_m3 < abs_D3_src_m2;
				bool smallerTempL_1 = abs_D3_src_m2 < abs_D3_src_m1;
				bool smallerTempL_2 = abs_D3_src_m1 < abs_D3_src_m0;

				bool smallerLL_0 = smallerTempL_0 & smallerL_0;
				bool smallerLL_1 = smallerTempL_1 & smallerL_1;
				bool smallerRL_0 = smallerTempL_1 & smallerR_0;
				bool smallerRL_1 = smallerTempL_2 & smallerR_1;


				bool smallerTempR_0 = !smallerTempL_0;
				bool smallerTempR_1 = !smallerTempL_1;
				bool smallerTempR_2 = !smallerTempL_2;

				bool smallerLR_0 = smallerTempR_0 & smallerL_0;
				bool smallerLR_1 = smallerTempR_1 & smallerL_1;
				bool smallerRR_0 = smallerTempR_1 & smallerR_0;
				bool smallerRR_1 = smallerTempR_2 & smallerR_1;


				bool smallerM_0 = smallerRL_0 | smallerLR_0;
				bool smallerM_1 = smallerRL_1 | smallerLR_1;

				FLOAT_TYPE smallerLL_0_f = smallerLL_0 ? (FLOAT_TYPE)1. : 0;
				FLOAT_TYPE smallerM_0_f = smallerM_0 ? (FLOAT_TYPE)1. : 0;
				FLOAT_TYPE smallerRR_0_f = smallerRR_0 ? (FLOAT_TYPE)1. : 0;

				FLOAT_TYPE smallerLL_1_f = smallerLL_1 ? (FLOAT_TYPE)1. : 0;
				FLOAT_TYPE smallerM_1_f = smallerM_1 ? (FLOAT_TYPE)1. : 0;
				FLOAT_TYPE smallerRR_1_f = smallerRR_1 ? (FLOAT_TYPE)1. : 0;

				// Pick out the best third order approximation
				FLOAT_TYPE deriv_l = dL0_ptr[src_index] * smallerLL_0_f
					+ dL1_ptr[src_index] * smallerM_0_f
					+ dL2_ptr[src_index] * smallerRR_0_f;

				FLOAT_TYPE deriv_r = dR0_ptr[src_index] * smallerLL_1_f
					+ dR1_ptr[src_index] * smallerM_1_f
					+ dR2_ptr[src_index] * smallerRR_1_f;

				size_t dst_index = src_index;

				dst_deriv_l_ptr[dst_index] = deriv_l;
				dst_deriv_r_ptr[dst_index] = deriv_r;
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

bool UpwindFirstENO3a_impl::execute_dim1(
	beacls::UVec& dst_deriv_l,
	beacls::UVec& dst_deriv_r,
	const FLOAT_TYPE* src,
	const HJI_Grid *grid,
	const size_t dim,
	const size_t loop_begin,
	const size_t slice_length,
	const size_t num_of_slices
) {
	const size_t first_dimension_loop_size = first_dimension_loop_sizes[dim];
	const size_t outer_dimensions_loop_length = slice_length / first_dimension_loop_size;
	const size_t total_slices_length = slice_length * num_of_slices;

	std::vector<beacls::UVec >& dL_uvec = dL_uvecs[dim];
	std::vector<beacls::UVec >& dR_uvec = dR_uvecs[dim];
	std::vector<beacls::UVec >& DD_uvec = DD_uvecs[dim];

	const bool stripDD = true;
	const bool approx4 = false;

	if (dL_uvec.size() != 3) dL_uvec.resize(3);
	if (dR_uvec.size() != 3) dR_uvec.resize(3);
	if (DD_uvec.size() != 3) DD_uvec.resize(3);
	beacls::UVecDepth depth = dst_deriv_l.depth();
	for_each(dL_uvec.begin(), dL_uvec.end(), ([total_slices_length, this, depth](auto &rhs) {
		if (rhs.type() != type || rhs.depth() != depth) rhs = beacls::UVec(depth, type, total_slices_length);
		else if (rhs.size() != total_slices_length) rhs.resize(total_slices_length);
	}));
	for_each(dR_uvec.begin(), dR_uvec.end(), ([total_slices_length, this, depth](auto &rhs) {
		if (rhs.type() != type || rhs.depth() != depth) rhs = beacls::UVec(depth, type, total_slices_length);
		else if (rhs.size() != total_slices_length) rhs.resize(total_slices_length);
	}));

	size_t DD_size_base = first_dimension_loop_size * (outer_dimensions_loop_length + 1)*num_of_slices;
	for (std::vector<beacls::UVec>::iterator ite = DD_uvec.begin(); ite != DD_uvec.end(); ++ite) {
		if (ite->type() != type || ite->depth() != depth) *ite = beacls::UVec(depth, type, DD_size_base);
		else if (ite->size() != DD_size_base) ite->resize(DD_size_base);
		DD_size_base -= first_dimension_loop_size*num_of_slices;
	}
	upwindFirstENO3aHelper->execute(dL_uvec, dR_uvec, DD_uvec, grid, src, dim, approx4, stripDD, loop_begin, slice_length,num_of_slices,1, cudaStreams[dim]);

#if 1	//!< Workaround, following operation doesn't support CUDA memory yet.
	std::vector<beacls::UVec > tmp_DD_uvec(DD_uvec.size());
	std::vector<beacls::UVec > tmp_dL_uvec(dL_uvec.size());
	std::vector<beacls::UVec > tmp_dR_uvec(dR_uvec.size());
	beacls::UVec tmp_deriv_l;
	beacls::UVec tmp_deriv_r;
	std::transform(dL_uvec.cbegin(), dL_uvec.cend(), tmp_dL_uvec.begin(), ([](const auto& rhs) {
		if (rhs.type() == beacls::UVecType_Cuda) {
			beacls::UVec tmp; rhs.convertTo(tmp, beacls::UVecType_Vector); return tmp;
		}
		else return rhs;
	}));
	std::transform(dR_uvec.cbegin(), dR_uvec.cend(), tmp_dR_uvec.begin(), ([](const auto& rhs) {
		if (rhs.type() == beacls::UVecType_Cuda) {
			beacls::UVec tmp; rhs.convertTo(tmp, beacls::UVecType_Vector); return tmp;
		}
		else return rhs;
	}));
	std::transform(DD_uvec.cbegin(), DD_uvec.cend(), tmp_DD_uvec.begin(), ([](const auto& rhs) {
		if (rhs.type() == beacls::UVecType_Cuda) {
			beacls::UVec tmp; rhs.convertTo(tmp, beacls::UVecType_Vector); return tmp;
		}
		else return rhs;
	}));
	if (dst_deriv_l.type() == beacls::UVecType_Cuda) tmp_deriv_l = beacls::UVec(dst_deriv_l.depth(), beacls::UVecType_Vector, dst_deriv_l.size());
	else tmp_deriv_l = dst_deriv_l;
	if (dst_deriv_r.type() == beacls::UVecType_Cuda) tmp_deriv_r = beacls::UVec(dst_deriv_r.depth(), beacls::UVecType_Vector, dst_deriv_r.size());
	else tmp_deriv_r = dst_deriv_r;

	FLOAT_TYPE* DD1_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_DD_uvec[1]).ptr();
	FLOAT_TYPE* DD2_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_DD_uvec[2]).ptr();
	FLOAT_TYPE* dL0_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_dL_uvec[0]).ptr();
	FLOAT_TYPE* dR0_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_dR_uvec[0]).ptr();
	FLOAT_TYPE* dL1_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_dL_uvec[1]).ptr();
	FLOAT_TYPE* dR1_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_dR_uvec[1]).ptr();
	FLOAT_TYPE* dL2_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_dL_uvec[2]).ptr();
	FLOAT_TYPE* dR2_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_dR_uvec[2]).ptr();
	FLOAT_TYPE* dst_deriv_l_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_deriv_l).ptr();
	FLOAT_TYPE* dst_deriv_r_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_deriv_r).ptr();
#else
	FLOAT_TYPE* DD1_ptr = beacls::UVec_<FLOAT_TYPE>(DD_uvec[1]).ptr();
	FLOAT_TYPE* DD2_ptr = beacls::UVec_<FLOAT_TYPE>(DD_uvec[2]).ptr();
	FLOAT_TYPE* dL0_ptr = beacls::UVec_<FLOAT_TYPE>(dL_uvec[0]).ptr();
	FLOAT_TYPE* dR0_ptr = beacls::UVec_<FLOAT_TYPE>(dR_uvec[0]).ptr();
	FLOAT_TYPE* dL1_ptr = beacls::UVec_<FLOAT_TYPE>(dL_uvec[1]).ptr();
	FLOAT_TYPE* dR1_ptr = beacls::UVec_<FLOAT_TYPE>(dR_uvec[1]).ptr();
	FLOAT_TYPE* dL2_ptr = beacls::UVec_<FLOAT_TYPE>(dL_uvec[2]).ptr();
	FLOAT_TYPE* dR2_ptr = beacls::UVec_<FLOAT_TYPE>(dR_uvec[2]).ptr();
	FLOAT_TYPE* dst_deriv_l_ptr = beacls::UVec_<FLOAT_TYPE>(dst_deriv_l).ptr();
	FLOAT_TYPE* dst_deriv_r_ptr = beacls::UVec_<FLOAT_TYPE>(dst_deriv_r).ptr();
#endif
	const size_t DD1_slice_length = DD_uvec[1].size() / first_dimension_loop_size / num_of_slices;
	const size_t DD2_slice_length = DD_uvec[2].size() / first_dimension_loop_size / num_of_slices;

	// Need to figure out which approximation has the least oscillation.
	// Note that L and R in this section refer to neighboring divided
	// difference entries, not to left and right approximations.
	size_t loop_length = slice_length / first_dimension_loop_size;
	for (size_t slice_index = 0; slice_index < num_of_slices; ++slice_index) {
		const size_t dst_slice_offset = slice_index * slice_length;
		const size_t src_DD1_slice_offset = slice_index * DD1_slice_length * first_dimension_loop_size;
		const size_t src_DD2_slice_offset = slice_index * DD2_slice_length * first_dimension_loop_size;
		const size_t src_dLdR_slice_offset = dst_slice_offset;
		for (size_t index = 0; index < loop_length; ++index) {
			size_t src_DD1_offset = index * first_dimension_loop_size + src_DD1_slice_offset;
			size_t src_DD2_offset = index * first_dimension_loop_size + src_DD2_slice_offset;
			size_t src_dLdR_offset = index * first_dimension_loop_size + src_dLdR_slice_offset;
			size_t dst_offset = index * first_dimension_loop_size + dst_slice_offset;
			for (size_t first_dimension_loop_index = 0; first_dimension_loop_index < first_dimension_loop_size; ++first_dimension_loop_index) {
				size_t src_DD1_index = first_dimension_loop_index + src_DD1_offset;
				size_t src_DD2_index = first_dimension_loop_index + src_DD2_offset;
				size_t src_dLdR_index = first_dimension_loop_index + src_dLdR_offset;
				size_t dst_index = first_dimension_loop_index + dst_offset;

				FLOAT_TYPE D2_src_0 = DD1_ptr[src_DD1_index];
				FLOAT_TYPE D2_src_1 = DD1_ptr[src_DD1_index + first_dimension_loop_size];
				FLOAT_TYPE D2_src_2 = DD1_ptr[src_DD1_index + first_dimension_loop_size * 2];
				FLOAT_TYPE D3_src_0 = DD2_ptr[src_DD2_index];
				FLOAT_TYPE D3_src_1 = DD2_ptr[src_DD2_index + first_dimension_loop_size];
				FLOAT_TYPE D3_src_2 = DD2_ptr[src_DD2_index + first_dimension_loop_size * 2];
				FLOAT_TYPE D3_src_3 = DD2_ptr[src_DD2_index + first_dimension_loop_size * 3];

				FLOAT_TYPE abs_D2_src_1 = HjiFabs(D2_src_1);
				FLOAT_TYPE abs_D3_src_1 = HjiFabs(D3_src_1);
				FLOAT_TYPE abs_D3_src_2 = HjiFabs(D3_src_2);
				// Pick out minimum modulus neighboring D2 term.
				bool smallerL_0 = HjiFabs(D2_src_0) < abs_D2_src_1;
				bool smallerR_0 = !smallerL_0;
				bool smallerL_1 = abs_D2_src_1 < HjiFabs(D2_src_2);
				bool smallerR_1 = !smallerL_1;

				// Figure out smallest modulus D3 terms,
				// given choice of smallest modulus D2 terms above.
				bool smallerTempL_0 = HjiFabs(D3_src_0) < abs_D3_src_1;
				bool smallerTempL_1 = abs_D3_src_1 < abs_D3_src_2;
				bool smallerTempL_2 = abs_D3_src_2 < HjiFabs(D3_src_3);

				bool smallerLL_0 = smallerTempL_0 & smallerL_0;
				bool smallerLL_1 = smallerTempL_1 & smallerL_1;
				bool smallerRL_0 = smallerTempL_1 & smallerR_0;
				bool smallerRL_1 = smallerTempL_2 & smallerR_1;


				bool smallerTempR_0 = !smallerTempL_0;
				bool smallerTempR_1 = !smallerTempL_1;
				bool smallerTempR_2 = !smallerTempL_2;

				bool smallerLR_0 = smallerTempR_0 & smallerL_0;
				bool smallerLR_1 = smallerTempR_1 & smallerL_1;
				bool smallerRR_0 = smallerTempR_1 & smallerR_0;
				bool smallerRR_1 = smallerTempR_2 & smallerR_1;


				bool smallerM_0 = smallerRL_0 | smallerLR_0;
				bool smallerM_1 = smallerRL_1 | smallerLR_1;

				FLOAT_TYPE smallerLL_0_f = smallerLL_0 ? (FLOAT_TYPE)1. : 0;
				FLOAT_TYPE smallerM_0_f = smallerM_0 ? (FLOAT_TYPE)1. : 0;
				FLOAT_TYPE smallerRR_0_f = smallerRR_0 ? (FLOAT_TYPE)1. : 0;

				FLOAT_TYPE smallerLL_1_f = smallerLL_1 ? (FLOAT_TYPE)1. : 0;
				FLOAT_TYPE smallerM_1_f = smallerM_1 ? (FLOAT_TYPE)1. : 0;
				FLOAT_TYPE smallerRR_1_f = smallerRR_1 ? (FLOAT_TYPE)1. : 0;

				// Pick out the best third order approximation
				FLOAT_TYPE deriv_l = dL0_ptr[src_dLdR_index] * smallerLL_0_f
					+ dL1_ptr[src_dLdR_index] * smallerM_0_f
					+ dL2_ptr[src_dLdR_index] * smallerRR_0_f;

				FLOAT_TYPE deriv_r = dR0_ptr[src_dLdR_index] * smallerLL_1_f
					+ dR1_ptr[src_dLdR_index] * smallerM_1_f
					+ dR2_ptr[src_dLdR_index] * smallerRR_1_f;

				dst_deriv_l_ptr[dst_index] = deriv_l;
				dst_deriv_r_ptr[dst_index] = deriv_r;
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

bool UpwindFirstENO3a_impl::execute_dimLET2(
	beacls::UVec& dst_deriv_l,
	beacls::UVec& dst_deriv_r,
	const FLOAT_TYPE* src,
	const HJI_Grid *grid,
	const size_t dim,
	const size_t loop_begin,
	const size_t slice_length,
	const size_t num_of_slices
) {
	const size_t first_dimension_loop_size = first_dimension_loop_sizes[dim];
	const size_t inner_dimensions_loop_size = inner_dimensions_loop_sizes[dim];
	const size_t outer_dimensions_loop_length = slice_length / first_dimension_loop_size;

	const size_t src_target_dimension_loop_size = src_target_dimension_loop_sizes[dim];
	const size_t loop_index_div_inner_size = loop_begin / inner_dimensions_loop_size;
	const size_t inner_dimensions_loop_index = loop_begin % inner_dimensions_loop_size;
	const size_t outer_dimensions_loop_index = loop_index_div_inner_size / src_target_dimension_loop_size;
	const size_t target_dimension_loop_index = loop_index_div_inner_size % src_target_dimension_loop_size;
	const size_t total_slices_length = slice_length * num_of_slices;


	const bool stripDD = true;
	const bool approx4 = false;

	std::vector<beacls::UVec >& dL_uvec = dL_uvecs[dim];
	std::vector<beacls::UVec >& dR_uvec = dR_uvecs[dim];
	std::vector<beacls::UVec >& DD_uvec = DD_uvecs[dim];
	beacls::UVecDepth depth = dst_deriv_l.depth();
	if (dL_uvec.size() != 3) dL_uvec.resize(3);
	if (dR_uvec.size() != 3) dR_uvec.resize(3);
	for_each(dL_uvec.begin(), dL_uvec.end(), ([total_slices_length, this, depth](auto &rhs) {
		if (rhs.type() != type || rhs.depth() != depth) rhs = beacls::UVec(depth, type, total_slices_length);
		else if (rhs.size() != total_slices_length) rhs.resize(total_slices_length);
	}));
	for_each(dR_uvec.begin(), dR_uvec.end(), ([total_slices_length, this, depth](auto &rhs) {
		if (rhs.type() != type || rhs.depth() != depth) rhs = beacls::UVec(depth, type, total_slices_length);
		else if (rhs.size() != total_slices_length) rhs.resize(total_slices_length);
	}));
	const size_t num_of_merged_slices = ((dim == 2) ? 1 : num_of_slices);
	const size_t num_of_merged_strides = ((dim == 2) ? num_of_strides + num_of_slices - 1 : num_of_strides);
	const size_t DD_size = first_dimension_loop_size * outer_dimensions_loop_length * num_of_merged_slices;
	const size_t total_DD_size = DD_size * num_of_merged_strides;
	if (DD_uvec.size() != 3) DD_uvec.resize(3);
	for_each(DD_uvec.begin(), DD_uvec.end(), ([total_DD_size, this, depth](auto &rhs) {
		if (rhs.type() != type || rhs.depth() != depth) rhs = beacls::UVec(depth, type, total_DD_size);
		else if (rhs.size() != total_DD_size) rhs.resize(total_DD_size);
	}));
	size_t margined_loop_begin = (outer_dimensions_loop_index * (src_target_dimension_loop_size + 3 * 2)
		+ target_dimension_loop_index) * inner_dimensions_loop_size + inner_dimensions_loop_index;
	upwindFirstENO3aHelper->execute(dL_uvec, dR_uvec, DD_uvec, grid, src, dim, approx4, stripDD, margined_loop_begin, slice_length, num_of_slices, num_of_strides, cudaStreams[dim]);

#if 1	//!< Workaround, following operation doesn't support CUDA memory yet.
	std::vector<beacls::UVec > tmp_DD_uvec(DD_uvec.size());
	std::vector<beacls::UVec > tmp_dL_uvec(dL_uvec.size());
	std::vector<beacls::UVec > tmp_dR_uvec(dR_uvec.size());
	beacls::UVec tmp_deriv_l;
	beacls::UVec tmp_deriv_r;
	std::transform(dL_uvec.cbegin(), dL_uvec.cend(), tmp_dL_uvec.begin(), ([](const auto& rhs) {
		if (rhs.type() == beacls::UVecType_Cuda) {
			beacls::UVec tmp; rhs.convertTo(tmp, beacls::UVecType_Vector); return tmp;
		}
		else return rhs;
	}));
	std::transform(dR_uvec.cbegin(), dR_uvec.cend(), tmp_dR_uvec.begin(), ([](const auto& rhs) {
		if (rhs.type() == beacls::UVecType_Cuda) {
			beacls::UVec tmp; rhs.convertTo(tmp, beacls::UVecType_Vector); return tmp;
		}
		else return rhs;
	}));
	std::transform(DD_uvec.cbegin(), DD_uvec.cend(), tmp_DD_uvec.begin(), ([](const auto& rhs) {
		if (rhs.type() == beacls::UVecType_Cuda) {
			beacls::UVec tmp; rhs.convertTo(tmp, beacls::UVecType_Vector); return tmp;
		}
		else return rhs;
	}));
	if (dst_deriv_l.type() == beacls::UVecType_Cuda) tmp_deriv_l = beacls::UVec(dst_deriv_l.depth(), beacls::UVecType_Vector, dst_deriv_l.size());
	else tmp_deriv_l = dst_deriv_l;
	if (dst_deriv_r.type() == beacls::UVecType_Cuda) tmp_deriv_r = beacls::UVec(dst_deriv_r.depth(), beacls::UVecType_Vector, dst_deriv_r.size());
	else tmp_deriv_r = dst_deriv_r;

	FLOAT_TYPE* DD0_1_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_DD_uvec[1]).ptr() + DD_size * 0;
	FLOAT_TYPE* DD1_1_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_DD_uvec[1]).ptr() + DD_size * 1;
	FLOAT_TYPE* DD2_1_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_DD_uvec[1]).ptr() + DD_size * 2;
	FLOAT_TYPE* DD0_2_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_DD_uvec[2]).ptr() + DD_size * 0;
	FLOAT_TYPE* DD1_2_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_DD_uvec[2]).ptr() + DD_size * 1;
	FLOAT_TYPE* DD2_2_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_DD_uvec[2]).ptr() + DD_size * 2;
	FLOAT_TYPE* DD3_2_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_DD_uvec[2]).ptr() + DD_size * 3;
	FLOAT_TYPE* dL0_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_dL_uvec[0]).ptr();
	FLOAT_TYPE* dR0_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_dR_uvec[0]).ptr();
	FLOAT_TYPE* dL1_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_dL_uvec[1]).ptr();
	FLOAT_TYPE* dR1_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_dR_uvec[1]).ptr();
	FLOAT_TYPE* dL2_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_dL_uvec[2]).ptr();
	FLOAT_TYPE* dR2_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_dR_uvec[2]).ptr();
	FLOAT_TYPE* dst_deriv_l_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_deriv_l).ptr();
	FLOAT_TYPE* dst_deriv_r_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_deriv_r).ptr();
#else
	FLOAT_TYPE* DD0_1_ptr = beacls::UVec_<FLOAT_TYPE>(DD_uvecs[0][1]).ptr();
	FLOAT_TYPE* DD1_1_ptr = beacls::UVec_<FLOAT_TYPE>(DD_uvecs[1][1]).ptr();
	FLOAT_TYPE* DD2_1_ptr = beacls::UVec_<FLOAT_TYPE>(DD_uvecs[2][1]).ptr();
	FLOAT_TYPE* DD0_2_ptr = beacls::UVec_<FLOAT_TYPE>(DD_uvecs[0][2]).ptr();
	FLOAT_TYPE* DD1_2_ptr = beacls::UVec_<FLOAT_TYPE>(DD_uvecs[1][2]).ptr();
	FLOAT_TYPE* DD2_2_ptr = beacls::UVec_<FLOAT_TYPE>(DD_uvecs[2][2]).ptr();
	FLOAT_TYPE* DD3_2_ptr = beacls::UVec_<FLOAT_TYPE>(DD_uvecs[3][2]).ptr();
	FLOAT_TYPE* dL0_ptr = beacls::UVec_<FLOAT_TYPE>(dL_uvec[0]).ptr();
	FLOAT_TYPE* dR0_ptr = beacls::UVec_<FLOAT_TYPE>(dR_uvec[0]).ptr();
	FLOAT_TYPE* dL1_ptr = beacls::UVec_<FLOAT_TYPE>(dL_uvec[1]).ptr();
	FLOAT_TYPE* dR1_ptr = beacls::UVec_<FLOAT_TYPE>(dR_uvec[1]).ptr();
	FLOAT_TYPE* dL2_ptr = beacls::UVec_<FLOAT_TYPE>(dL_uvec[2]).ptr();
	FLOAT_TYPE* dR2_ptr = beacls::UVec_<FLOAT_TYPE>(dR_uvec[2]).ptr();
	FLOAT_TYPE* dst_deriv_l_ptr = beacls::UVec_<FLOAT_TYPE>(dst_deriv_l).ptr();
	FLOAT_TYPE* dst_deriv_r_ptr = beacls::UVec_<FLOAT_TYPE>(dst_deriv_r).ptr();
#endif

	// Need to figure out which approximation has the least oscillation.
	// Note that L and R in this section refer to neighboring divided
	// difference entries, not to left and right approximations.

	size_t loop_length = slice_length / first_dimension_loop_size;

	for (size_t slice_index = 0; slice_index < num_of_slices; ++slice_index) {
		const size_t slice_offset = slice_index * slice_length;
		for (size_t index = 0; index < loop_length; ++index) {
			size_t dst_offset = index * first_dimension_loop_size + slice_offset;
			for (size_t first_dimension_loop_index = 0; first_dimension_loop_index < first_dimension_loop_size; ++first_dimension_loop_index) {
				size_t src_index = first_dimension_loop_index + dst_offset;
				FLOAT_TYPE D2_src_0 = DD0_1_ptr[src_index];
				FLOAT_TYPE D2_src_1 = DD1_1_ptr[src_index];
				FLOAT_TYPE D2_src_2 = DD2_1_ptr[src_index];
				FLOAT_TYPE D3_src_0 = DD0_2_ptr[src_index];
				FLOAT_TYPE D3_src_1 = DD1_2_ptr[src_index];
				FLOAT_TYPE D3_src_2 = DD2_2_ptr[src_index];
				FLOAT_TYPE D3_src_3 = DD3_2_ptr[src_index];

				FLOAT_TYPE abs_D2_src_1 = HjiFabs(D2_src_1);
				FLOAT_TYPE abs_D3_src_1 = HjiFabs(D3_src_1);
				FLOAT_TYPE abs_D3_src_2 = HjiFabs(D3_src_2);
				// Pick out minimum modulus neighboring D2 term.
				bool smallerL_0 = HjiFabs(D2_src_0) < abs_D2_src_1;
				bool smallerR_0 = !smallerL_0;
				bool smallerL_1 = abs_D2_src_1 < HjiFabs(D2_src_2);
				bool smallerR_1 = !smallerL_1;

				// Figure out smallest modulus D3 terms,
				// given choice of smallest modulus D2 terms above.
				bool smallerTempL_0 = HjiFabs(D3_src_0) < abs_D3_src_1;
				bool smallerTempL_1 = abs_D3_src_1 < abs_D3_src_2;
				bool smallerTempL_2 = abs_D3_src_2 < HjiFabs(D3_src_3);

				bool smallerLL_0 = smallerTempL_0 & smallerL_0;
				bool smallerLL_1 = smallerTempL_1 & smallerL_1;
				bool smallerRL_0 = smallerTempL_1 & smallerR_0;
				bool smallerRL_1 = smallerTempL_2 & smallerR_1;


				bool smallerTempR_0 = !smallerTempL_0;
				bool smallerTempR_1 = !smallerTempL_1;
				bool smallerTempR_2 = !smallerTempL_2;

				bool smallerLR_0 = smallerTempR_0 & smallerL_0;
				bool smallerLR_1 = smallerTempR_1 & smallerL_1;
				bool smallerRR_0 = smallerTempR_1 & smallerR_0;
				bool smallerRR_1 = smallerTempR_2 & smallerR_1;


				bool smallerM_0 = smallerRL_0 | smallerLR_0;
				bool smallerM_1 = smallerRL_1 | smallerLR_1;

				FLOAT_TYPE smallerLL_0_f = smallerLL_0 ? (FLOAT_TYPE)1. : 0;
				FLOAT_TYPE smallerM_0_f = smallerM_0 ? (FLOAT_TYPE)1. : 0;
				FLOAT_TYPE smallerRR_0_f = smallerRR_0 ? (FLOAT_TYPE)1. : 0;

				FLOAT_TYPE smallerLL_1_f = smallerLL_1 ? (FLOAT_TYPE)1. : 0;
				FLOAT_TYPE smallerM_1_f = smallerM_1 ? (FLOAT_TYPE)1. : 0;
				FLOAT_TYPE smallerRR_1_f = smallerRR_1 ? (FLOAT_TYPE)1. : 0;

				// Pick out the best third order approximation
				FLOAT_TYPE deriv_l = dL0_ptr[src_index] * smallerLL_0_f
					+ dL1_ptr[src_index] * smallerM_0_f
					+ dL2_ptr[src_index] * smallerRR_0_f;

				FLOAT_TYPE deriv_r = dR0_ptr[src_index] * smallerLL_1_f
					+ dR1_ptr[src_index] * smallerM_1_f
					+ dR2_ptr[src_index] * smallerRR_1_f;

				size_t dst_index = src_index;

				dst_deriv_l_ptr[dst_index] = deriv_l;
				dst_deriv_r_ptr[dst_index] = deriv_r;
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

bool UpwindFirstENO3a_impl::execute(
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

	if (generateAll) {
		const FLOAT_TYPE small = (FLOAT_TYPE)(100. * std::numeric_limits<FLOAT_TYPE>::epsilon());

		std::vector<beacls::UVec >& dL_uvec = dL_uvecs[dim];
		std::vector<beacls::UVec >& dR_uvec = dR_uvecs[dim];
		std::vector<beacls::UVec >& DD_uvec = DD_uvecs[dim];

		// We only need the three ENO approximations
		// (plus the fourth if we want to check for equivalence).
		if (dL_uvec.size() != 4) dL_uvec.resize(4);
		if (dR_uvec.size() != 4) dR_uvec.resize(4);
		if (DD_uvec.size() != 3) DD_uvec.resize(3);
		beacls::UVecDepth depth = dst_deriv_l.depth();
		for_each(dL_uvec.begin(), dL_uvec.end(), ([slice_length, this, depth](auto &rhs) {
			if (rhs.type() != type || rhs.depth() != depth) rhs = beacls::UVec(depth, type, slice_length);
			else if (rhs.size() != slice_length) rhs.resize(slice_length);
		}));
		for_each(dR_uvec.begin(), dR_uvec.end(), ([slice_length, this, depth](auto &rhs) {
			if (rhs.type() != type || rhs.depth() != depth) rhs = beacls::UVec(depth, type, slice_length);
			else if (rhs.size() != slice_length) rhs.resize(slice_length);
		}));

		const bool stripDD = true;
		const bool approx4 = checkEquivalentApproximations;
		upwindFirstENO3aHelper->execute(dL_uvec, dR_uvec, DD_uvec, grid, src, dim, approx4, stripDD, loop_begin, slice_length, num_of_slices, 1, cudaStreams[dim]);
		beacls::synchronizeCuda(cudaStreams[dim]);
		if (checkEquivalentApproximations) {
			// Only the LLL and RRR approximations are not equivalent to at least one
			// other approximations, so we have several checks.
			beacls::FloatVec relErrors;
			beacls::FloatVec absErrors;

			// Check corresponding left and right approximations against one another.
			checkEquivalentApprox(relErrors, absErrors, dL_uvec[1], dR_uvec[0], small);
			checkEquivalentApprox(relErrors, absErrors, dL_uvec[2], dR_uvec[1], small);

			// Check the middle approximations.
			if (dL_uvec.size() >= 4)
				checkEquivalentApprox(relErrors, absErrors, dL_uvec[1], dL_uvec[3], small);
			if (dR_uvec.size() >= 4)
				checkEquivalentApprox(relErrors, absErrors, dR_uvec[1], dR_uvec[3], small);
		}
		// Caller requested all approximations in each direction.
		// If we requested all four approximations above, strip off the last one.
		const size_t num_of_dL_uvec = std::accumulate(dL_uvec.cbegin(), dL_uvec.cend(), static_cast<size_t>(0), ([](const auto &lhs, const auto &rhs) {
			return lhs + rhs.size();
		}));
		dst_deriv_l.resize(num_of_dL_uvec);
		size_t dst_deriv_l_offset = 0;
		FLOAT_TYPE* dst_deriv_l_ptr = beacls::UVec_<FLOAT_TYPE>(dst_deriv_l).ptr();
		for (size_t i = 0; i < dL_uvec.size(); ++i) {
			const size_t uvec_size = dL_uvec[i].size();
			memcpy(dst_deriv_l_ptr + dst_deriv_l_offset, beacls::UVec_<FLOAT_TYPE>(dL_uvec[i]).ptr(), uvec_size * sizeof(FLOAT_TYPE));
			dst_deriv_l_offset += uvec_size;
		}
		const size_t num_of_dR_uvec = std::accumulate(dR_uvec.cbegin(), dR_uvec.cend(), static_cast<size_t>(0), ([](const auto &lhs, const auto &rhs) {
			return lhs + rhs.size();
		}));
		dst_deriv_r.resize(num_of_dR_uvec);
		size_t dst_deriv_r_offset = 0;
		FLOAT_TYPE* dst_deriv_r_ptr = beacls::UVec_<FLOAT_TYPE>(dst_deriv_r).ptr();
		for (size_t i = 0; i < dR_uvec.size(); ++i) {
			const size_t uvec_size = dR_uvec[i].size();
			memcpy(dst_deriv_r_ptr + dst_deriv_r_offset, beacls::UVec_<FLOAT_TYPE>(dR_uvec[i]).ptr(), uvec_size * sizeof(FLOAT_TYPE));
			dst_deriv_r_offset += uvec_size;
		}
		dst_deriv_l.set_cudaStream(cudaStreams[dim]);
		dst_deriv_r.set_cudaStream(cudaStreams[dim]);
	}
	else {
		// We need the three ENO approximations
		//  plus the(stripped) divided differences to pick the least oscillatory.
		switch (dim) {
		case 0:
		{
			execute_dim0(
				dst_deriv_l,
				dst_deriv_r,
				src,
				grid,
				dim,
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
				dim,
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
				dim,
				loop_begin,
				slice_length,
				num_of_slices
			);
		}
		break;
		}
	}

	return true;
}


UpwindFirstENO3a::UpwindFirstENO3a(
	const HJI_Grid *hji_grid,
	const beacls::UVecType type
	) {
	pimpl = new UpwindFirstENO3a_impl(hji_grid,type);
}
UpwindFirstENO3a::~UpwindFirstENO3a() {
	if (pimpl) delete pimpl;
}

bool UpwindFirstENO3a::execute(
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
	if (pimpl) return pimpl->execute(dst_deriv_l, dst_deriv_r, grid, src, dim, generateAll, loop_begin, slice_length, num_of_slices);
	else return false;
}
bool UpwindFirstENO3a::execute_local_q(
	beacls::UVec& dst_deriv_l,
	beacls::UVec& dst_deriv_r,
	const HJI_Grid *grid,
	const FLOAT_TYPE* src,
	const size_t dim,
	const bool generateAll,
	const size_t loop_begin,
	const size_t slice_length,
	const size_t num_of_slices, 
	const std::set<size_t> &Q 
) {
	if (pimpl) return pimpl->execute(dst_deriv_l, dst_deriv_r, grid, src, dim, generateAll, loop_begin, slice_length, num_of_slices);
	else return false;
}
bool UpwindFirstENO3a_impl::synchronize(const size_t dim) {
	if ((cudaStreams.size() > dim) && cudaStreams[dim]) {
		beacls::synchronizeCuda(cudaStreams[dim]);
		return true;
	}
	return false;
}
bool UpwindFirstENO3a::synchronize(const size_t dim) {
	if (pimpl) return pimpl->synchronize(dim);
	else return false;
}
bool UpwindFirstENO3a::operator==(const UpwindFirstENO3a& rhs) const {
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
bool UpwindFirstENO3a::operator==(const SpatialDerivative& rhs) const {
	if (this == &rhs) return true;
	else if (typeid(*this) != typeid(rhs)) return false;
	else return operator==(dynamic_cast<const UpwindFirstENO3a&>(rhs));
}

UpwindFirstENO3a::UpwindFirstENO3a(const UpwindFirstENO3a& rhs) :
	pimpl(rhs.pimpl->clone())
{
}

UpwindFirstENO3a* UpwindFirstENO3a::clone() const {
	return new UpwindFirstENO3a(*this);
}
beacls::UVecType UpwindFirstENO3a::get_type() const {
	if (pimpl) return pimpl->get_type();
	else return beacls::UVecType_Invalid;
};
