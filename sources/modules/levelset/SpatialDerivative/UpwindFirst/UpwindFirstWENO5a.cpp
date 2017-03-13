#include <vector>
#include <array>
#include <cstdint>
#include <cmath>
#include <cfloat>
#include <numeric>
#include <typeinfo>
#include <levelset/Grids/HJI_Grid.hpp>
#include <Core/CudaStream.hpp>
#include <levelset/BoundaryCondition/BoundaryCondition.hpp>
#include <levelset/SpatialDerivative/UpwindFirst/UpwindFirstWENO5a.hpp>
#include <levelset/SpatialDerivative/UpwindFirst/UpwindFirstENO3aHelper.hpp>
#include "UpwindFirstWENO5a_impl.hpp"
#include "UpwindFirstWENO5a_cuda.hpp"

UpwindFirstWENO5a_impl::UpwindFirstWENO5a_impl(
	const HJI_Grid *hji_grid,
	const beacls::UVecType type
) :
	type(type),
	stencil(3),
	num_of_strides(6),
	tmpSmooths_m1s(2),
	tmpSmooths(2),
	epsilonCalculationMethod_Type(beacls::EpsilonCalculationMethod_maxOverNeighbor)
{
	upwindFirstENO3aHelper = new UpwindFirstENO3aHelper(hji_grid,type);

	size_t num_of_dimensions = hji_grid->get_num_of_dimensions();
	inner_dimensions_loop_sizes.resize(num_of_dimensions);
	first_dimension_loop_sizes.resize(num_of_dimensions);
	src_target_dimension_loop_sizes.resize(num_of_dimensions);

	dL_uvecs.resize(num_of_dimensions);
	dR_uvecs.resize(num_of_dimensions);
	DD_uvecs.resize(num_of_dimensions);
	for (size_t target_dimension = 0; target_dimension < num_of_dimensions; ++target_dimension) {
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
		if (target_dimension == 0) {
			for_each(tmpSmooths.begin(), tmpSmooths.end(), ([first_dimension_loop_size](auto& tmpSmooth) {
				tmpSmooth.resize(3);
			}));
		}
		if (target_dimension == 1) {
			for_each(tmpSmooths_m1s.begin(), tmpSmooths_m1s.end(), ([first_dimension_loop_size](auto& tmpSmooth_m1s) {
				tmpSmooth_m1s.resize(3);
				for_each(tmpSmooth_m1s.begin(), tmpSmooth_m1s.end(), ([first_dimension_loop_size](auto& rhs) {
					rhs.resize(first_dimension_loop_size);
				}));
			}));
		}
		src_target_dimension_loop_sizes[target_dimension] = src_target_dimension_loop_size;

	}
	cudaStreams.resize(num_of_dimensions, NULL);
	if (type == beacls::UVecType_Cuda) {
		std::for_each(cudaStreams.begin(), cudaStreams.end(), [](auto& rhs) {
			rhs = new beacls::CudaStream();
		});
	}
}
UpwindFirstWENO5a_impl::~UpwindFirstWENO5a_impl() {
	if (type == beacls::UVecType_Cuda) {
		std::for_each(cudaStreams.begin(), cudaStreams.end(), [](auto& rhs) {
			if (rhs) delete rhs;
			rhs = NULL;
		});
	}
	if (upwindFirstENO3aHelper) delete upwindFirstENO3aHelper;
}
bool UpwindFirstWENO5a_impl::operator==(const UpwindFirstWENO5a_impl& rhs) const {
	if (this == &rhs) return true;
	else if (epsilonCalculationMethod_Type != rhs.epsilonCalculationMethod_Type) return false;
	else if ((first_dimension_loop_sizes.size() != rhs.first_dimension_loop_sizes.size()) || !std::equal(first_dimension_loop_sizes.cbegin(), first_dimension_loop_sizes.cend(), rhs.first_dimension_loop_sizes.cbegin())) return false;
	else if ((inner_dimensions_loop_sizes.size() != rhs.inner_dimensions_loop_sizes.size()) || !std::equal(inner_dimensions_loop_sizes.cbegin(), inner_dimensions_loop_sizes.cend(), rhs.inner_dimensions_loop_sizes.cbegin())) return false;
	else if ((src_target_dimension_loop_sizes.size() != rhs.src_target_dimension_loop_sizes.size()) || !std::equal(src_target_dimension_loop_sizes.cbegin(), src_target_dimension_loop_sizes.cend(), rhs.src_target_dimension_loop_sizes.cbegin())) return false;
	else if ((upwindFirstENO3aHelper != rhs.upwindFirstENO3aHelper) && (!upwindFirstENO3aHelper || !rhs.upwindFirstENO3aHelper || !upwindFirstENO3aHelper->operator==(*rhs.upwindFirstENO3aHelper))) return false;
	else return true;
}

UpwindFirstWENO5a_impl::UpwindFirstWENO5a_impl(const UpwindFirstWENO5a_impl& rhs) :
	type(rhs.type),
	first_dimension_loop_sizes(rhs.first_dimension_loop_sizes),
	inner_dimensions_loop_sizes(rhs.inner_dimensions_loop_sizes),
	src_target_dimension_loop_sizes(rhs.src_target_dimension_loop_sizes),
	stencil(rhs.stencil),
	num_of_strides(rhs.num_of_strides),
	tmpSmooths_m1s(rhs.tmpSmooths_m1s),
	tmpSmooths(rhs.tmpSmooths),
	weightL(rhs.weightL),
	weightR(rhs.weightR),
	epsilonCalculationMethod_Type(rhs.epsilonCalculationMethod_Type),
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
bool UpwindFirstWENO5a_impl::execute_dim0(
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

	const bool stripDD = false;
	const bool approx4 = false;

	if (dL_uvec.size() != 3) dL_uvec.resize(3);
	if (dR_uvec.size() != 3) dR_uvec.resize(3);
	if (DD_uvec.size() != 3) DD_uvec.resize(3);
	beacls::UVecDepth depth = dst_deriv_l.depth();
	for_each(dL_uvec.begin(), dL_uvec.end(), ([total_slices_length, this, depth](auto &rhs) {
		if (rhs.type() != type || rhs.depth() != depth) rhs = beacls::UVec(depth, type, total_slices_length);
		else if (rhs.size() < total_slices_length) rhs.resize(total_slices_length);
	}));
	for_each(dR_uvec.begin(), dR_uvec.end(), ([total_slices_length, this, depth](auto &rhs) {
		if (rhs.type() != type || rhs.depth() != depth) rhs = beacls::UVec(depth, type, total_slices_length);
		else if (rhs.size() < total_slices_length) rhs.resize(total_slices_length);
	}));

	size_t DD_size_base = (first_dimension_loop_size + 1) * outer_dimensions_loop_length*num_of_slices;
	for (std::vector<beacls::UVec>::iterator ite = DD_uvec.begin(); ite != DD_uvec.end(); ++ite) {
		if (ite->type() != type || ite->depth() != depth) *ite = beacls::UVec(depth, type, DD_size_base);
		else if (ite->size() < DD_size_base) ite->resize(DD_size_base);
		DD_size_base += outer_dimensions_loop_length*num_of_slices;
	}
	// We need the three ENO approximations 
	// plus the(unstripped) divided differences to pick the least oscillatory.
	upwindFirstENO3aHelper->execute(dL_uvec, dR_uvec, DD_uvec, grid, src, dim, approx4, stripDD, loop_begin, slice_length, num_of_slices, 1, cudaStreams[dim]);
	const size_t src_target_dimension_loop_size = src_target_dimension_loop_sizes[dim];

	FLOAT_TYPE* DD0_ptr = beacls::UVec_<FLOAT_TYPE>(DD_uvec[0]).ptr();
	FLOAT_TYPE* dL0_ptr = beacls::UVec_<FLOAT_TYPE>(dL_uvec[0]).ptr();
	FLOAT_TYPE* dR0_ptr = beacls::UVec_<FLOAT_TYPE>(dR_uvec[0]).ptr();
	FLOAT_TYPE* dL1_ptr = beacls::UVec_<FLOAT_TYPE>(dL_uvec[1]).ptr();
	FLOAT_TYPE* dR1_ptr = beacls::UVec_<FLOAT_TYPE>(dR_uvec[1]).ptr();
	FLOAT_TYPE* dL2_ptr = beacls::UVec_<FLOAT_TYPE>(dL_uvec[2]).ptr();
	FLOAT_TYPE* dR2_ptr = beacls::UVec_<FLOAT_TYPE>(dR_uvec[2]).ptr();
	FLOAT_TYPE* dst_deriv_l_ptr = beacls::UVec_<FLOAT_TYPE>(dst_deriv_l).ptr();
	FLOAT_TYPE* dst_deriv_r_ptr = beacls::UVec_<FLOAT_TYPE>(dst_deriv_r).ptr();
	dst_deriv_l.set_cudaStream(cudaStreams[dim]);
	dst_deriv_r.set_cudaStream(cudaStreams[dim]);

	const FLOAT_TYPE weightL0 = weightL[0];
	const FLOAT_TYPE weightL1 = weightL[1];
	const FLOAT_TYPE weightL2 = weightL[2];
	const FLOAT_TYPE weightR0 = weightR[0];
	const FLOAT_TYPE weightR1 = weightR[1];
	const FLOAT_TYPE weightR2 = weightR[2];

	size_t loop_length = slice_length / first_dimension_loop_size;
	if (type == beacls::UVecType_Cuda) {
		UpwindFirstWENO5a_execute_dim0_cuda(
			dst_deriv_l_ptr, dst_deriv_r_ptr,
			DD0_ptr,
			dL0_ptr, dL1_ptr, dL2_ptr, dR0_ptr, dR1_ptr, dR2_ptr,
			weightL0, weightL1, weightL2, weightR0,weightR1,weightR2,
			num_of_slices, loop_length, src_target_dimension_loop_size,first_dimension_loop_size, slice_length,
			epsilonCalculationMethod_Type, 
			cudaStreams[dim]
		);
	}
	else
	{
		// The smoothness estimates may have some relation to the higher order
		// divided differences, but it isn't obvious from just reading O&F.
		// For now, use only the first order divided differences.
		for (size_t slice_index = 0; slice_index < num_of_slices; ++slice_index) {
			const size_t slice_offset = slice_index * slice_length;
			const size_t slice_loop_offset = slice_index * outer_dimensions_loop_length;
			for (size_t loop_index = 0; loop_index < outer_dimensions_loop_length; ++loop_index) {
				size_t tmpSmooth_current_index = loop_index % 2;
				size_t tmpSmooth_last_index = (tmpSmooth_current_index == 0) ? 1 : 0;

				size_t dst_offset = loop_index * first_dimension_loop_size + slice_offset;
				size_t src_D1_offset = (loop_index + slice_loop_offset) * (first_dimension_loop_size + 5);
				//Prologue
				{
					beacls::FloatVec &smooth_m1 = tmpSmooths[tmpSmooth_last_index];
					size_t target_dimension_loop_index = 0;
					size_t src_D1_index = target_dimension_loop_index + src_D1_offset;
					FLOAT_TYPE D1_src_0 = DD0_ptr[src_D1_index];
					FLOAT_TYPE D1_src_1 = DD0_ptr[src_D1_index + 1];
					FLOAT_TYPE D1_src_2 = DD0_ptr[src_D1_index + 2];
					FLOAT_TYPE D1_src_3 = DD0_ptr[src_D1_index + 3];
					FLOAT_TYPE D1_src_4 = DD0_ptr[src_D1_index + 4];
					smooth_m1[0] = calcSmooth0(D1_src_0, D1_src_1, D1_src_2);
					smooth_m1[1] = calcSmooth1(D1_src_1, D1_src_2, D1_src_3);
					smooth_m1[2] = calcSmooth2(D1_src_2, D1_src_3, D1_src_4);
				}
				// Body
				switch (epsilonCalculationMethod_Type) {
				case beacls::EpsilonCalculationMethod_Invalid:
				default:
					printf("Unknown epsilonCalculationMethod %d\n", epsilonCalculationMethod_Type);
					return false;
				case beacls::EpsilonCalculationMethod_Constant:
					for (size_t target_dimension_loop_index = 0; target_dimension_loop_index < src_target_dimension_loop_size; ++target_dimension_loop_index) {
						size_t src_D1_index = target_dimension_loop_index + src_D1_offset;
						size_t src_index = target_dimension_loop_index + dst_offset;
						FLOAT_TYPE D1_src_1 = DD0_ptr[src_D1_index + 1];
						FLOAT_TYPE D1_src_2 = DD0_ptr[src_D1_index + 2];
						FLOAT_TYPE D1_src_3 = DD0_ptr[src_D1_index + 3];
						FLOAT_TYPE D1_src_4 = DD0_ptr[src_D1_index + 4];
						FLOAT_TYPE D1_src_5 = DD0_ptr[src_D1_index + 5];
						beacls::FloatVec &smooth_m0 = tmpSmooths[tmpSmooth_current_index];
						beacls::FloatVec &smooth_m1 = tmpSmooths[tmpSmooth_last_index];
						FLOAT_TYPE smooth1 = calcSmooth0(D1_src_1, D1_src_2, D1_src_3);
						FLOAT_TYPE smooth2 = calcSmooth1(D1_src_2, D1_src_3, D1_src_4);
						FLOAT_TYPE smooth3 = calcSmooth2(D1_src_3, D1_src_4, D1_src_5);
						smooth_m0[0] = smooth1;
						smooth_m0[1] = smooth2;
						smooth_m0[2] = smooth3;
						beacls::FloatVec &smoothL = smooth_m1;
						FLOAT_TYPE epsilon = (FLOAT_TYPE)1e-6;
						size_t dst_index = src_index;
						dst_deriv_l_ptr[dst_index] = weightWENO(dL0_ptr[src_index], dL1_ptr[src_index], dL2_ptr[src_index], smoothL[0], smoothL[1], smoothL[2], weightL0, weightL1, weightL2, epsilon);
						dst_deriv_r_ptr[dst_index] = weightWENO(dR0_ptr[src_index], dR1_ptr[src_index], dR2_ptr[src_index], smooth1, smooth2, smooth3, weightR0, weightR1, weightR2, epsilon);
						tmpSmooth_last_index = (tmpSmooth_last_index == 0) ? 1 : 0;
						tmpSmooth_current_index = (tmpSmooth_current_index == 0) ? 1 : 0;
					}
					break;
				case beacls::EpsilonCalculationMethod_maxOverGrid:
					printf("epsilonCalculationMethod %d is not supported yet\n", epsilonCalculationMethod_Type);
					return false;
				case beacls::EpsilonCalculationMethod_maxOverNeighbor:
					for (size_t target_dimension_loop_index = 0; target_dimension_loop_index < src_target_dimension_loop_size; ++target_dimension_loop_index) {
						size_t src_D1_index = target_dimension_loop_index + src_D1_offset;
						size_t src_index = target_dimension_loop_index + dst_offset;
						FLOAT_TYPE D1_src_0 = DD0_ptr[src_D1_index];
						FLOAT_TYPE D1_src_1 = DD0_ptr[src_D1_index + 1];
						FLOAT_TYPE D1_src_2 = DD0_ptr[src_D1_index + 2];
						FLOAT_TYPE D1_src_3 = DD0_ptr[src_D1_index + 3];
						FLOAT_TYPE D1_src_4 = DD0_ptr[src_D1_index + 4];
						FLOAT_TYPE D1_src_5 = DD0_ptr[src_D1_index + 5];
						beacls::FloatVec &smooth_m0 = tmpSmooths[tmpSmooth_current_index];
						beacls::FloatVec &smooth_m1 = tmpSmooths[tmpSmooth_last_index];
						FLOAT_TYPE smooth1 = calcSmooth0(D1_src_1, D1_src_2, D1_src_3);
						FLOAT_TYPE smooth2 = calcSmooth1(D1_src_2, D1_src_3, D1_src_4);
						FLOAT_TYPE smooth3 = calcSmooth2(D1_src_3, D1_src_4, D1_src_5);
						smooth_m0[0] = smooth1;
						smooth_m0[1] = smooth2;
						smooth_m0[2] = smooth3;
						beacls::FloatVec &smoothL = smooth_m1;
						FLOAT_TYPE pow_D1_src_0 = D1_src_0 * D1_src_0;
						FLOAT_TYPE pow_D1_src_1 = D1_src_1 * D1_src_1;
						FLOAT_TYPE pow_D1_src_2 = D1_src_2 * D1_src_2;
						FLOAT_TYPE pow_D1_src_3 = D1_src_3 * D1_src_3;
						FLOAT_TYPE pow_D1_src_4 = D1_src_4 * D1_src_4;
						FLOAT_TYPE pow_D1_src_5 = D1_src_5 * D1_src_5;
						FLOAT_TYPE max_1_2 = std::max<FLOAT_TYPE>(pow_D1_src_1, pow_D1_src_2);
						FLOAT_TYPE max_3_4 = std::max<FLOAT_TYPE>(pow_D1_src_3, pow_D1_src_4);
						FLOAT_TYPE max_1_2_3_4 = std::max<FLOAT_TYPE>(max_1_2, max_3_4);
						FLOAT_TYPE maxOverNeighborD1squaredL = std::max<FLOAT_TYPE>(max_1_2_3_4, pow_D1_src_0);
						FLOAT_TYPE maxOverNeighborD1squaredR = std::max<FLOAT_TYPE>(max_1_2_3_4, pow_D1_src_5);
						FLOAT_TYPE epsilonL = (FLOAT_TYPE)(1e-6 * maxOverNeighborD1squaredL + get_epsilon_base<FLOAT_TYPE>());
						FLOAT_TYPE epsilonR = (FLOAT_TYPE)(1e-6 * maxOverNeighborD1squaredR + get_epsilon_base<FLOAT_TYPE>());
						size_t dst_index = src_index;
						dst_deriv_l_ptr[dst_index] = weightWENO(dL0_ptr[src_index], dL1_ptr[src_index], dL2_ptr[src_index], smoothL[0], smoothL[1], smoothL[2], weightL0, weightL1, weightL2, epsilonL);
						dst_deriv_r_ptr[dst_index] = weightWENO(dR0_ptr[src_index], dR1_ptr[src_index], dR2_ptr[src_index], smooth1, smooth2, smooth3, weightR0, weightR1, weightR2, epsilonR);
						tmpSmooth_last_index = (tmpSmooth_last_index == 0) ? 1 : 0;
						tmpSmooth_current_index = (tmpSmooth_current_index == 0) ? 1 : 0;
					}
					break;
				}
			}
		}
	}
	return true;
}

bool UpwindFirstWENO5a_impl::execute_dim1(
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

	const bool stripDD = false;
	const bool approx4 = false;
	if (dL_uvec.size() != 3) dL_uvec.resize(3);
	if (dR_uvec.size() != 3) dR_uvec.resize(3);
	if (DD_uvec.size() != 3) DD_uvec.resize(3);
	beacls::UVecDepth depth = dst_deriv_l.depth();
	for_each(dL_uvec.begin(), dL_uvec.end(), ([total_slices_length, this, depth](auto &rhs) {
		if (rhs.type() != type || rhs.depth() != depth) rhs = beacls::UVec(depth, type, total_slices_length);
		else if (rhs.size() < total_slices_length) rhs.resize(total_slices_length);
	}));
	for_each(dR_uvec.begin(), dR_uvec.end(), ([total_slices_length, this, depth](auto &rhs) {
		if (rhs.type() != type || rhs.depth() != depth) rhs = beacls::UVec(depth, type, total_slices_length);
		else if (rhs.size() < total_slices_length) rhs.resize(total_slices_length);
	}));

	size_t DD_size_base = first_dimension_loop_size * (outer_dimensions_loop_length + stencil + 2)*num_of_slices;
	for (std::vector<beacls::UVec>::iterator ite = DD_uvec.begin(); ite != DD_uvec.end(); ++ite) {
		if (ite->type() != type || ite->depth() != depth) *ite = beacls::UVec(depth, type, DD_size_base);
		else if (ite->size() < DD_size_base) ite->resize(DD_size_base);
		DD_size_base -= first_dimension_loop_size*num_of_slices;
	}
	upwindFirstENO3aHelper->execute(dL_uvec, dR_uvec, DD_uvec, grid, src, dim, approx4, stripDD, loop_begin, slice_length,num_of_slices, 1, cudaStreams[dim]);

	FLOAT_TYPE* DD0_ptr = beacls::UVec_<FLOAT_TYPE>(DD_uvec[0]).ptr();
	FLOAT_TYPE* dL0_ptr = beacls::UVec_<FLOAT_TYPE>(dL_uvec[0]).ptr();
	FLOAT_TYPE* dR0_ptr = beacls::UVec_<FLOAT_TYPE>(dR_uvec[0]).ptr();
	FLOAT_TYPE* dL1_ptr = beacls::UVec_<FLOAT_TYPE>(dL_uvec[1]).ptr();
	FLOAT_TYPE* dR1_ptr = beacls::UVec_<FLOAT_TYPE>(dR_uvec[1]).ptr();
	FLOAT_TYPE* dL2_ptr = beacls::UVec_<FLOAT_TYPE>(dL_uvec[2]).ptr();
	FLOAT_TYPE* dR2_ptr = beacls::UVec_<FLOAT_TYPE>(dR_uvec[2]).ptr();

	const size_t DD0_slice_length = DD_uvec[0].size() / first_dimension_loop_size / num_of_slices;

	FLOAT_TYPE* dst_deriv_l_ptr = beacls::UVec_<FLOAT_TYPE>(dst_deriv_l).ptr();
	FLOAT_TYPE* dst_deriv_r_ptr = beacls::UVec_<FLOAT_TYPE>(dst_deriv_r).ptr();
	dst_deriv_l.set_cudaStream(cudaStreams[dim]);
	dst_deriv_r.set_cudaStream(cudaStreams[dim]);
	size_t loop_length = slice_length / first_dimension_loop_size;
	const FLOAT_TYPE weightL0 = weightL[0];
	const FLOAT_TYPE weightL1 = weightL[1];
	const FLOAT_TYPE weightL2 = weightL[2];
	const FLOAT_TYPE weightR0 = weightR[0];
	const FLOAT_TYPE weightR1 = weightR[1];
	const FLOAT_TYPE weightR2 = weightR[2];
	if (type == beacls::UVecType_Cuda) {
		UpwindFirstWENO5a_execute_dim1_cuda(
			dst_deriv_l_ptr, dst_deriv_r_ptr,
			DD0_ptr, 
			dL0_ptr, dL1_ptr, dL2_ptr, dR0_ptr, dR1_ptr, dR2_ptr,
			weightL0, weightL1, weightL2, weightR0, weightR1, weightR2,
			num_of_slices, loop_length, first_dimension_loop_size,slice_length,
			DD0_slice_length * first_dimension_loop_size,
			epsilonCalculationMethod_Type, 
			cudaStreams[dim]
		);
	}
	else
	{
		for (size_t slice_index = 0; slice_index < num_of_slices; ++slice_index) {
			const size_t dst_slice_offset = slice_index * slice_length;
			const size_t src_DD0_slice_offset = slice_index * DD0_slice_length * first_dimension_loop_size;
			const size_t src_dLdR_slice_offset = dst_slice_offset;
			// Need to figure out which approximation has the least oscillation.
			// Note that L and R in this section refer to neighboring divided
			// difference entries, not to left and right approximations.
			//Prologue
			{
				size_t index = 0;
				size_t loop_index = index + loop_begin + dst_slice_offset;
				size_t tmpSmooth_current_index = loop_index % 2;
				size_t tmpSmooth_last_index = (tmpSmooth_current_index == 0) ? 1 : 0;
				std::vector<beacls::FloatVec > &tmpSmooth_m1 = tmpSmooths_m1s[tmpSmooth_last_index];
				beacls::FloatVec &smooth1_m1 = tmpSmooth_m1[0];
				beacls::FloatVec &smooth2_m1 = tmpSmooth_m1[1];
				beacls::FloatVec &smooth3_m1 = tmpSmooth_m1[2];

				size_t src_DD0_offset = index * first_dimension_loop_size + src_DD0_slice_offset;
				for (size_t fdl_index = 0; fdl_index < first_dimension_loop_size; ++fdl_index) {
					size_t src_DD0_index = fdl_index + src_DD0_offset;

					FLOAT_TYPE D1_src_0 = DD0_ptr[src_DD0_index];
					FLOAT_TYPE D1_src_1 = DD0_ptr[src_DD0_index + first_dimension_loop_size * 1];
					FLOAT_TYPE D1_src_2 = DD0_ptr[src_DD0_index + first_dimension_loop_size * 2];
					FLOAT_TYPE D1_src_3 = DD0_ptr[src_DD0_index + first_dimension_loop_size * 3];
					FLOAT_TYPE D1_src_4 = DD0_ptr[src_DD0_index + first_dimension_loop_size * 4];
					smooth1_m1[fdl_index] = calcSmooth0(D1_src_0, D1_src_1, D1_src_2);
					smooth2_m1[fdl_index] = calcSmooth1(D1_src_1, D1_src_2, D1_src_3);
					smooth3_m1[fdl_index] = calcSmooth2(D1_src_2, D1_src_3, D1_src_4);
				}
			}
			// Body
			switch (epsilonCalculationMethod_Type) {
			case beacls::EpsilonCalculationMethod_Invalid:
			default:
				printf("Unknown epsilonCalculationMethod %d\n", epsilonCalculationMethod_Type);
				return false;
			case beacls::EpsilonCalculationMethod_Constant:
				for (size_t index = 0; index < loop_length; ++index) {
					size_t loop_index = index + loop_begin + dst_slice_offset;
					size_t src_DD0_offset = index * first_dimension_loop_size + src_DD0_slice_offset;
					size_t src_dLdR_offset = index * first_dimension_loop_size + src_dLdR_slice_offset;
					size_t dst_offset = index * first_dimension_loop_size + dst_slice_offset;
					size_t tmpSmooth_current_index = loop_index % 2;
					size_t tmpSmooth_last_index = (tmpSmooth_current_index == 0) ? 1 : 0;
					std::vector<beacls::FloatVec > &tmpSmooth_m0 = tmpSmooths_m1s[tmpSmooth_current_index];
					std::vector<beacls::FloatVec > &tmpSmooth_m1 = tmpSmooths_m1s[tmpSmooth_last_index];
					beacls::FloatVec &smooth1_m1 = tmpSmooth_m1[0];
					beacls::FloatVec &smooth2_m1 = tmpSmooth_m1[1];
					beacls::FloatVec &smooth3_m1 = tmpSmooth_m1[2];
					beacls::FloatVec &smooth1_m0 = tmpSmooth_m0[0];
					beacls::FloatVec &smooth2_m0 = tmpSmooth_m0[1];
					beacls::FloatVec &smooth3_m0 = tmpSmooth_m0[2];
					beacls::FloatVec &smooth1L = smooth1_m1;
					beacls::FloatVec &smooth2L = smooth2_m1;
					beacls::FloatVec &smooth3L = smooth3_m1;
					for (size_t fdl_index = 0; fdl_index < first_dimension_loop_size; ++fdl_index) {
						size_t dst_index = fdl_index + dst_offset;
						size_t src_DD0_index = fdl_index + src_DD0_offset;
						size_t src_dLdR_index = fdl_index + src_dLdR_offset;
						FLOAT_TYPE D1_src_1 = DD0_ptr[src_DD0_index + first_dimension_loop_size * 1];
						FLOAT_TYPE D1_src_2 = DD0_ptr[src_DD0_index + first_dimension_loop_size * 2];
						FLOAT_TYPE D1_src_3 = DD0_ptr[src_DD0_index + first_dimension_loop_size * 3];
						FLOAT_TYPE D1_src_4 = DD0_ptr[src_DD0_index + first_dimension_loop_size * 4];
						FLOAT_TYPE D1_src_5 = DD0_ptr[src_DD0_index + first_dimension_loop_size * 5];
						FLOAT_TYPE smooth1 = calcSmooth0(D1_src_1, D1_src_2, D1_src_3);
						FLOAT_TYPE smooth2 = calcSmooth1(D1_src_2, D1_src_3, D1_src_4);
						FLOAT_TYPE smooth3 = calcSmooth2(D1_src_3, D1_src_4, D1_src_5);
						smooth1_m0[fdl_index] = smooth1;
						smooth2_m0[fdl_index] = smooth2;
						smooth3_m0[fdl_index] = smooth3;
						FLOAT_TYPE epsilon = (FLOAT_TYPE)1e-6;
						dst_deriv_l_ptr[dst_index] = weightWENO(dL0_ptr[src_dLdR_index], dL1_ptr[src_dLdR_index], dL2_ptr[src_dLdR_index], smooth1L[fdl_index], smooth2L[fdl_index], smooth3L[fdl_index], weightL0, weightL1, weightL2, epsilon);
						dst_deriv_r_ptr[dst_index] = weightWENO(dR0_ptr[src_dLdR_index], dR1_ptr[src_dLdR_index], dR2_ptr[src_dLdR_index], smooth1, smooth2, smooth3, weightR0, weightR1, weightR2, epsilon);
					}
				}
				break;
			case beacls::EpsilonCalculationMethod_maxOverGrid:
				printf("epsilonCalculationMethod %d is not supported yet\n", epsilonCalculationMethod_Type);
				return false;
			case beacls::EpsilonCalculationMethod_maxOverNeighbor:
				for (size_t index = 0; index < loop_length; ++index) {
					size_t loop_index = index + loop_begin + dst_slice_offset;
					size_t src_DD0_offset = index * first_dimension_loop_size + src_DD0_slice_offset;
					size_t src_dLdR_offset = index * first_dimension_loop_size + src_dLdR_slice_offset;
					size_t dst_offset = index * first_dimension_loop_size + dst_slice_offset;
					size_t tmpSmooth_current_index = loop_index % 2;
					size_t tmpSmooth_last_index = (tmpSmooth_current_index == 0) ? 1 : 0;
					std::vector<beacls::FloatVec > &tmpSmooth_m0 = tmpSmooths_m1s[tmpSmooth_current_index];
					std::vector<beacls::FloatVec > &tmpSmooth_m1 = tmpSmooths_m1s[tmpSmooth_last_index];
					beacls::FloatVec &smooth1_m1 = tmpSmooth_m1[0];
					beacls::FloatVec &smooth2_m1 = tmpSmooth_m1[1];
					beacls::FloatVec &smooth3_m1 = tmpSmooth_m1[2];
					beacls::FloatVec &smooth1_m0 = tmpSmooth_m0[0];
					beacls::FloatVec &smooth2_m0 = tmpSmooth_m0[1];
					beacls::FloatVec &smooth3_m0 = tmpSmooth_m0[2];
					beacls::FloatVec &smooth1L = smooth1_m1;
					beacls::FloatVec &smooth2L = smooth2_m1;
					beacls::FloatVec &smooth3L = smooth3_m1;
					for (size_t fdl_index = 0; fdl_index < first_dimension_loop_size; ++fdl_index) {
						size_t dst_index = fdl_index + dst_offset;
						size_t src_DD0_index = fdl_index + src_DD0_offset;
						size_t src_dLdR_index = fdl_index + src_dLdR_offset;
						FLOAT_TYPE D1_src_0 = DD0_ptr[src_DD0_index];
						FLOAT_TYPE D1_src_1 = DD0_ptr[src_DD0_index + first_dimension_loop_size * 1];
						FLOAT_TYPE D1_src_2 = DD0_ptr[src_DD0_index + first_dimension_loop_size * 2];
						FLOAT_TYPE D1_src_3 = DD0_ptr[src_DD0_index + first_dimension_loop_size * 3];
						FLOAT_TYPE D1_src_4 = DD0_ptr[src_DD0_index + first_dimension_loop_size * 4];
						FLOAT_TYPE D1_src_5 = DD0_ptr[src_DD0_index + first_dimension_loop_size * 5];
						FLOAT_TYPE smooth1 = calcSmooth0(D1_src_1, D1_src_2, D1_src_3);
						FLOAT_TYPE smooth2 = calcSmooth1(D1_src_2, D1_src_3, D1_src_4);
						FLOAT_TYPE smooth3 = calcSmooth2(D1_src_3, D1_src_4, D1_src_5);
						smooth1_m0[fdl_index] = smooth1;
						smooth2_m0[fdl_index] = smooth2;
						smooth3_m0[fdl_index] = smooth3;
						FLOAT_TYPE pow_D1_src_0 = D1_src_0 * D1_src_0;
						FLOAT_TYPE pow_D1_src_1 = D1_src_1 * D1_src_1;
						FLOAT_TYPE pow_D1_src_2 = D1_src_2 * D1_src_2;
						FLOAT_TYPE pow_D1_src_3 = D1_src_3 * D1_src_3;
						FLOAT_TYPE pow_D1_src_4 = D1_src_4 * D1_src_4;
						FLOAT_TYPE pow_D1_src_5 = D1_src_5 * D1_src_5;
						FLOAT_TYPE max_1_2 = std::max<FLOAT_TYPE>(pow_D1_src_1, pow_D1_src_2);
						FLOAT_TYPE max_3_4 = std::max<FLOAT_TYPE>(pow_D1_src_3, pow_D1_src_4);
						FLOAT_TYPE max_1_2_3_4 = std::max<FLOAT_TYPE>(max_1_2, max_3_4);
						FLOAT_TYPE maxOverNeighborD1squaredL = std::max<FLOAT_TYPE>(max_1_2_3_4, pow_D1_src_0);
						FLOAT_TYPE maxOverNeighborD1squaredR = std::max<FLOAT_TYPE>(max_1_2_3_4, pow_D1_src_5);
						FLOAT_TYPE epsilonL = (FLOAT_TYPE)(1e-6 * maxOverNeighborD1squaredL + get_epsilon_base<FLOAT_TYPE>());
						FLOAT_TYPE epsilonR = (FLOAT_TYPE)(1e-6 * maxOverNeighborD1squaredR + get_epsilon_base<FLOAT_TYPE>());
						dst_deriv_l_ptr[dst_index] = weightWENO(dL0_ptr[src_dLdR_index], dL1_ptr[src_dLdR_index], dL2_ptr[src_dLdR_index], smooth1L[fdl_index], smooth2L[fdl_index], smooth3L[fdl_index], weightL0, weightL1, weightL2, epsilonL);
						dst_deriv_r_ptr[dst_index] = weightWENO(dR0_ptr[src_dLdR_index], dR1_ptr[src_dLdR_index], dR2_ptr[src_dLdR_index], smooth1, smooth2, smooth3, weightR0, weightR1, weightR2, epsilonR);
					}
				}
				break;
			}
		}
	}
	return true;
}

bool UpwindFirstWENO5a_impl::execute_dimLET2(
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
	const bool stripDD = false;
	const bool approx4 = false;

	std::vector<beacls::UVec >& dL_uvec = dL_uvecs[dim];
	std::vector<beacls::UVec >& dR_uvec = dR_uvecs[dim];
	std::vector<beacls::UVec >& DD_uvec = DD_uvecs[dim];
	beacls::UVecDepth depth = dst_deriv_l.depth();
	const size_t num_of_merged_slices = ((dim == 2) ? 1 : num_of_slices);
	const size_t num_of_merged_strides = ((dim == 2) ? num_of_strides + num_of_slices - 1 : num_of_strides);
	const size_t DD_size = first_dimension_loop_size * outer_dimensions_loop_length * num_of_merged_slices;
	const size_t total_DD_size = DD_size * num_of_merged_strides;
	if (dL_uvec.size() != 3) dL_uvec.resize(3);
	if (dR_uvec.size() != 3) dR_uvec.resize(3);
	if (DD_uvec.size() != 3) DD_uvec.resize(3);
	for_each(DD_uvec.begin(), DD_uvec.end(), ([total_DD_size, this, depth](auto &rhs) {
		if (rhs.type() != type || rhs.depth() != depth) rhs = beacls::UVec(depth, type, total_DD_size);
		else if (rhs.size() < total_DD_size) rhs.resize(total_DD_size);
	}));
	for_each(dL_uvec.begin(), dL_uvec.end(), ([total_slices_length, this, depth](auto &rhs) {
		if (rhs.type() != type || rhs.depth() != depth) rhs = beacls::UVec(depth, type, total_slices_length);
		else if (rhs.size() < total_slices_length) rhs.resize(total_slices_length);
	}));
	for_each(dR_uvec.begin(), dR_uvec.end(), ([total_slices_length, this, depth](auto &rhs) {
		if (rhs.type() != type || rhs.depth() != depth) rhs = beacls::UVec(depth, type, total_slices_length);
		else if (rhs.size() < total_slices_length) rhs.resize(total_slices_length);
	}));
	size_t margined_loop_begin = (outer_dimensions_loop_index * (src_target_dimension_loop_size + 3 * 2)
		+ target_dimension_loop_index) * inner_dimensions_loop_size + inner_dimensions_loop_index;
	upwindFirstENO3aHelper->execute(dL_uvec, dR_uvec, DD_uvec, grid, src, dim, approx4, stripDD, margined_loop_begin, slice_length,num_of_slices, num_of_strides, cudaStreams[dim]);

	FLOAT_TYPE* DD0_0_ptr = beacls::UVec_<FLOAT_TYPE>(DD_uvec[0]).ptr() + DD_size * 0;
	FLOAT_TYPE* DD1_0_ptr = beacls::UVec_<FLOAT_TYPE>(DD_uvec[0]).ptr() + DD_size * 1;
	FLOAT_TYPE* DD2_0_ptr = beacls::UVec_<FLOAT_TYPE>(DD_uvec[0]).ptr() + DD_size * 2;
	FLOAT_TYPE* DD3_0_ptr = beacls::UVec_<FLOAT_TYPE>(DD_uvec[0]).ptr() + DD_size * 3;
	FLOAT_TYPE* DD4_0_ptr = beacls::UVec_<FLOAT_TYPE>(DD_uvec[0]).ptr() + DD_size * 4;
	FLOAT_TYPE* DD5_0_ptr = beacls::UVec_<FLOAT_TYPE>(DD_uvec[0]).ptr() + DD_size * 5;
	FLOAT_TYPE* dL0_ptr = beacls::UVec_<FLOAT_TYPE>(dL_uvec[0]).ptr();
	FLOAT_TYPE* dR0_ptr = beacls::UVec_<FLOAT_TYPE>(dR_uvec[0]).ptr();
	FLOAT_TYPE* dL1_ptr = beacls::UVec_<FLOAT_TYPE>(dL_uvec[1]).ptr();
	FLOAT_TYPE* dR1_ptr = beacls::UVec_<FLOAT_TYPE>(dR_uvec[1]).ptr();
	FLOAT_TYPE* dL2_ptr = beacls::UVec_<FLOAT_TYPE>(dL_uvec[2]).ptr();
	FLOAT_TYPE* dR2_ptr = beacls::UVec_<FLOAT_TYPE>(dR_uvec[2]).ptr();
	FLOAT_TYPE* dst_deriv_l_ptr = beacls::UVec_<FLOAT_TYPE>(dst_deriv_l).ptr();
	FLOAT_TYPE* dst_deriv_r_ptr = beacls::UVec_<FLOAT_TYPE>(dst_deriv_r).ptr();
	dst_deriv_l.set_cudaStream(cudaStreams[dim]);
	dst_deriv_r.set_cudaStream(cudaStreams[dim]);
	// Need to figure out which approximation has the least oscillation.
	// Note that L and R in this section refer to neighboring divided
	// difference entries, not to left and right approximations.

	size_t loop_length = slice_length / first_dimension_loop_size;
	const FLOAT_TYPE weightL0 = weightL[0];
	const FLOAT_TYPE weightL1 = weightL[1];
	const FLOAT_TYPE weightL2 = weightL[2];
	const FLOAT_TYPE weightR0 = weightR[0];
	const FLOAT_TYPE weightR1 = weightR[1];
	const FLOAT_TYPE weightR2 = weightR[2];
	if (type == beacls::UVecType_Cuda) {
		UpwindFirstWENO5a_execute_dimLET2_cuda(
			dst_deriv_l_ptr, dst_deriv_r_ptr,
			DD0_0_ptr, DD1_0_ptr, DD2_0_ptr, DD3_0_ptr, DD4_0_ptr, DD5_0_ptr,
			dL0_ptr, dL1_ptr, dL2_ptr, dR0_ptr, dR1_ptr, dR2_ptr,
			weightL0, weightL1, weightL2, weightR0, weightR1, weightR2,
			num_of_slices, loop_length, first_dimension_loop_size, slice_length, 
			epsilonCalculationMethod_Type, 
			cudaStreams[dim]
		);
	}
	else
	{
		for (size_t slice_index = 0; slice_index < num_of_slices; ++slice_index) {
			const size_t slice_offset = slice_index * slice_length;
			switch (epsilonCalculationMethod_Type) {
			case beacls::EpsilonCalculationMethod_Invalid:
			default:
				printf("Unknown epsilonCalculationMethod %d\n", epsilonCalculationMethod_Type);
				return false;
			case beacls::EpsilonCalculationMethod_Constant:
				for (size_t index = 0; index < loop_length; ++index) {
					size_t dst_offset = index * first_dimension_loop_size + slice_offset;
					for (size_t first_dimension_loop_index = 0; first_dimension_loop_index < first_dimension_loop_size; ++first_dimension_loop_index) {
						size_t src_index = first_dimension_loop_index + dst_offset;
						FLOAT_TYPE D1_src_0 = DD0_0_ptr[src_index];
						FLOAT_TYPE D1_src_1 = DD1_0_ptr[src_index];
						FLOAT_TYPE D1_src_2 = DD2_0_ptr[src_index];
						FLOAT_TYPE D1_src_3 = DD3_0_ptr[src_index];
						FLOAT_TYPE D1_src_4 = DD4_0_ptr[src_index];
						FLOAT_TYPE D1_src_5 = DD5_0_ptr[src_index];
						FLOAT_TYPE smoothL_0 = calcSmooth0(D1_src_0, D1_src_1, D1_src_2);
						FLOAT_TYPE smoothL_1 = calcSmooth1(D1_src_1, D1_src_2, D1_src_3);
						FLOAT_TYPE smoothL_2 = calcSmooth2(D1_src_2, D1_src_3, D1_src_4);
						FLOAT_TYPE smoothR_0 = calcSmooth0(D1_src_1, D1_src_2, D1_src_3);
						FLOAT_TYPE smoothR_1 = calcSmooth1(D1_src_2, D1_src_3, D1_src_4);
						FLOAT_TYPE smoothR_2 = calcSmooth2(D1_src_3, D1_src_4, D1_src_5);
						FLOAT_TYPE epsilon = (FLOAT_TYPE)1e-6;
						size_t dst_index = src_index;
						dst_deriv_l_ptr[dst_index] = weightWENO(dL0_ptr[src_index], dL1_ptr[src_index], dL2_ptr[src_index], smoothL_0, smoothL_1, smoothL_2, weightL0, weightL1, weightL2, epsilon);
						dst_deriv_r_ptr[dst_index] = weightWENO(dR0_ptr[src_index], dR1_ptr[src_index], dR2_ptr[src_index], smoothR_0, smoothR_1, smoothR_2, weightR0, weightR1, weightR2, epsilon);
					}
				}
				break;
			case beacls::EpsilonCalculationMethod_maxOverGrid:
				printf("epsilonCalculationMethod %d is not supported yet\n", epsilonCalculationMethod_Type);
				return false;
			case beacls::EpsilonCalculationMethod_maxOverNeighbor:
				for (size_t index = 0; index < loop_length; ++index) {
					size_t dst_offset = index * first_dimension_loop_size + slice_offset;
					for (size_t first_dimension_loop_index = 0; first_dimension_loop_index < first_dimension_loop_size; ++first_dimension_loop_index) {
						size_t src_index = first_dimension_loop_index + dst_offset;
						FLOAT_TYPE D1_src_0 = DD0_0_ptr[src_index];
						FLOAT_TYPE D1_src_1 = DD1_0_ptr[src_index];
						FLOAT_TYPE D1_src_2 = DD2_0_ptr[src_index];
						FLOAT_TYPE D1_src_3 = DD3_0_ptr[src_index];
						FLOAT_TYPE D1_src_4 = DD4_0_ptr[src_index];
						FLOAT_TYPE D1_src_5 = DD5_0_ptr[src_index];
						FLOAT_TYPE smoothL_0 = calcSmooth0(D1_src_0, D1_src_1, D1_src_2);
						FLOAT_TYPE smoothL_1 = calcSmooth1(D1_src_1, D1_src_2, D1_src_3);
						FLOAT_TYPE smoothL_2 = calcSmooth2(D1_src_2, D1_src_3, D1_src_4);
						FLOAT_TYPE smoothR_0 = calcSmooth0(D1_src_1, D1_src_2, D1_src_3);
						FLOAT_TYPE smoothR_1 = calcSmooth1(D1_src_2, D1_src_3, D1_src_4);
						FLOAT_TYPE smoothR_2 = calcSmooth2(D1_src_3, D1_src_4, D1_src_5);
						FLOAT_TYPE pow_D1_src_0 = D1_src_0 * D1_src_0;
						FLOAT_TYPE pow_D1_src_1 = D1_src_1 * D1_src_1;
						FLOAT_TYPE pow_D1_src_2 = D1_src_2 * D1_src_2;
						FLOAT_TYPE pow_D1_src_3 = D1_src_3 * D1_src_3;
						FLOAT_TYPE pow_D1_src_4 = D1_src_4 * D1_src_4;
						FLOAT_TYPE pow_D1_src_5 = D1_src_5 * D1_src_5;
						FLOAT_TYPE max_1_2 = std::max<FLOAT_TYPE>(pow_D1_src_1, pow_D1_src_2);
						FLOAT_TYPE max_3_4 = std::max<FLOAT_TYPE>(pow_D1_src_3, pow_D1_src_4);
						FLOAT_TYPE max_1_2_3_4 = std::max<FLOAT_TYPE>(max_1_2, max_3_4);
						FLOAT_TYPE maxOverNeighborD1squaredL = std::max<FLOAT_TYPE>(max_1_2_3_4, pow_D1_src_0);
						FLOAT_TYPE maxOverNeighborD1squaredR = std::max<FLOAT_TYPE>(max_1_2_3_4, pow_D1_src_5);
						FLOAT_TYPE epsilonL = (FLOAT_TYPE)(1e-6 * maxOverNeighborD1squaredL + get_epsilon_base<FLOAT_TYPE>());
						FLOAT_TYPE epsilonR = (FLOAT_TYPE)(1e-6 * maxOverNeighborD1squaredR + get_epsilon_base<FLOAT_TYPE>());
						size_t dst_index = src_index;
						dst_deriv_l_ptr[dst_index] = weightWENO(dL0_ptr[src_index], dL1_ptr[src_index], dL2_ptr[src_index], smoothL_0, smoothL_1, smoothL_2, weightL0, weightL1, weightL2, epsilonL);
						dst_deriv_r_ptr[dst_index] = weightWENO(dR0_ptr[src_index], dR1_ptr[src_index], dR2_ptr[src_index], smoothR_0, smoothR_1, smoothR_2, weightR0, weightR1, weightR2, epsilonR);
					}
				}
				break;
			}
		}
	}
	return true;
}

bool UpwindFirstWENO5a_impl::execute(
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
		std::vector<beacls::UVec >& dL_uvec = dL_uvecs[dim];
		std::vector<beacls::UVec >& dR_uvec = dR_uvecs[dim];
		std::vector<beacls::UVec >& DD_uvec = DD_uvecs[dim];

		// We only need the three ENO approximations
		// (plus the fourth if we want to check for equivalence).
		if (dL_uvec.size() != 4) dL_uvec.resize(4);
		if (dR_uvec.size() != 4) dR_uvec.resize(4);
		beacls::UVecDepth depth = dst_deriv_l.depth();
		for_each(dL_uvec.begin(), dL_uvec.end(), ([slice_length, this, depth](auto &rhs) {
			if (rhs.type() != type || rhs.depth() != depth) rhs = beacls::UVec(depth, type, slice_length);
			else if (rhs.size() < slice_length) rhs.resize(slice_length);
		}));
		for_each(dR_uvec.begin(), dR_uvec.end(), ([slice_length, this, depth](auto &rhs) {
			if (rhs.type() != type || rhs.depth() != depth) rhs = beacls::UVec(depth, type, slice_length);
			else if (rhs.size() < slice_length) rhs.resize(slice_length);
		}));

		const bool stripDD = false;
		const bool approx4 = false;
		upwindFirstENO3aHelper->execute(dL_uvec, dR_uvec, DD_uvec, grid, src, dim, approx4, stripDD, loop_begin, slice_length, num_of_slices, 1, cudaStreams[dim]);

		// Caller requested all approximations in each direction.
		// If we requested all four approximations above, strip off the last one.
		const size_t num_of_dL = std::accumulate(dL_uvec.cbegin(), dL_uvec.cend(), static_cast<size_t>(0), ([](const auto &lhs, const auto &rhs) {
			return lhs + rhs.size();
		}));
		beacls::FloatVec dst_deriv_l_vec;
		beacls::FloatVec dst_deriv_r_vec;

		dst_deriv_l_vec.resize(0);
		dst_deriv_l_vec.reserve(num_of_dL);
		for_each(dL_uvec.cbegin(), dL_uvec.cend(), ([&dst_deriv_l_vec](const auto &rhs) {
			dst_deriv_l_vec.insert(dst_deriv_l_vec.end(), (FLOAT_TYPE*)rhs.ptr(), (FLOAT_TYPE*)rhs.ptr() + rhs.size());
		}));
		const size_t num_of_dR = std::accumulate(dR_uvec.cbegin(), dR_uvec.cend(), static_cast<size_t>(0), ([](const auto &lhs, const auto &rhs) {
			return lhs + rhs.size();
		}));
		dst_deriv_r_vec.resize(0);
		dst_deriv_r_vec.reserve(num_of_dR);
		for_each(dR_uvec.cbegin(), dR_uvec.cend(), ([&dst_deriv_r_vec](const auto &rhs) {
			dst_deriv_r_vec.insert(dst_deriv_r_vec.end(), (FLOAT_TYPE*)rhs.ptr(), (FLOAT_TYPE*)rhs.ptr() + rhs.size());
		}));
		dst_deriv_l = beacls::UVec(dst_deriv_l_vec, type, true);
		dst_deriv_r = beacls::UVec(dst_deriv_r_vec, type, true);
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

UpwindFirstWENO5a::UpwindFirstWENO5a(
	const HJI_Grid *hji_grid,
	const beacls::UVecType type
) {
	pimpl = new UpwindFirstWENO5a_impl(hji_grid,type);
}
UpwindFirstWENO5a::~UpwindFirstWENO5a() {
	if (pimpl) delete pimpl;
}


bool UpwindFirstWENO5a::execute(
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
bool UpwindFirstWENO5a_impl::synchronize(const size_t dim) {
	if ((cudaStreams.size() > dim) && cudaStreams[dim]) {
		beacls::synchronizeCuda(cudaStreams[dim]);
		return true;
	}
	return false;
}
bool UpwindFirstWENO5a::synchronize(const size_t dim) {
	if (pimpl) return pimpl->synchronize(dim);
	else return false;
}
bool UpwindFirstWENO5a::operator==(const UpwindFirstWENO5a& rhs) const {
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
bool UpwindFirstWENO5a::operator==(const SpatialDerivative& rhs) const {
	if (this == &rhs) return true;
	else if (typeid(*this) != typeid(rhs)) return false;
	else return operator==(dynamic_cast<const UpwindFirstWENO5a&>(rhs));
}

UpwindFirstWENO5a::UpwindFirstWENO5a(const UpwindFirstWENO5a& rhs) :
	pimpl(rhs.pimpl->clone())
{
}

UpwindFirstWENO5a* UpwindFirstWENO5a::clone() const {
	return new UpwindFirstWENO5a(*this);
}
beacls::UVecType UpwindFirstWENO5a::get_type() const {
	if (pimpl) return pimpl->get_type();
	else return beacls::UVecType_Invalid;
};
