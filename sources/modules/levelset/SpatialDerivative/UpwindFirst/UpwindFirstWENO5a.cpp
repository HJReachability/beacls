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
#include <chrono>

using namespace levelset;

UpwindFirstWENO5a_impl::UpwindFirstWENO5a_impl(
	const HJI_Grid *hji_grid,
	const beacls::UVecType type
) :
	type(type),
	dxs(hji_grid->get_dxs()),
	dxInvs(hji_grid->get_dxInvs()),
	stencil(3),
	num_of_strides(6),
	epsilonCalculationMethod_Type(levelset::EpsilonCalculationMethod_maxOverNeighbor)
{
	upwindFirstENO3aHelper = new UpwindFirstENO3aHelper(hji_grid,type);
	cache = new UpwindFirstWENO5a_Cache;
	size_t num_of_dimensions = hji_grid->get_num_of_dimensions();
	dxInv_2s.resize(num_of_dimensions);
	dxInv_3s.resize(num_of_dimensions);
	dx_squares.resize(num_of_dimensions);

	target_dimension_loop_sizes.resize(num_of_dimensions);
	inner_dimensions_loop_sizes.resize(num_of_dimensions);
	first_dimension_loop_sizes.resize(num_of_dimensions);
	src_target_dimension_loop_sizes.resize(num_of_dimensions);

	tmpBoundedSrc_ptrssss.resize(num_of_dimensions);
	//!< Dimension step wise double buffer 
	tmpBoundedSrc_uvecs.resize(2);
	tmpBoundedSrc_uvec_vectors.resize(2);

	dL_uvecs.resize(num_of_dimensions);
	dR_uvecs.resize(num_of_dimensions);
	DD_uvecs.resize(num_of_dimensions);
	for (size_t target_dimension = 0; target_dimension < num_of_dimensions; ++target_dimension) {
		dL_uvecs[target_dimension].resize(3);
		dR_uvecs[target_dimension].resize(3);
		DD_uvecs[target_dimension].resize(3);

		beacls::IntegerVec sizeIn = hji_grid->get_Ns();
		beacls::IntegerVec sizeOut = sizeIn;
		sizeOut[target_dimension] += 2 * stencil;
		dxInv_2s[target_dimension] = (FLOAT_TYPE)0.5 * dxInvs[target_dimension];
		dxInv_3s[target_dimension] = (FLOAT_TYPE)(1. / 3.) * dxInvs[target_dimension];
		dx_squares[target_dimension] = std::pow(dxs[target_dimension], 2);

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
		target_dimension_loop_sizes[target_dimension] = target_dimension_loop_size;
	}
	cudaStreams.resize(num_of_dimensions, NULL);
	if (type == beacls::UVecType_Cuda) {
#if 1
		//!< Streams for dimension step wise double buffer 
		for (size_t target_dimension = 0; target_dimension < num_of_dimensions; ++target_dimension) {
			if (target_dimension < 2)
				cudaStreams[target_dimension] = new beacls::CudaStream();
			else
				cudaStreams[target_dimension] = cudaStreams[target_dimension%2];
		}
#else
		std::for_each(cudaStreams.begin(), cudaStreams.end(), [](auto& rhs) {
			rhs = new beacls::CudaStream();
		});
#endif
	}
}
UpwindFirstWENO5a_impl::~UpwindFirstWENO5a_impl() {
	if (type == beacls::UVecType_Cuda) {
#if 1
		const size_t end_offset = std::min<size_t>(cudaStreams.size(), 2);
		std::for_each(cudaStreams.begin(), cudaStreams.begin() + end_offset, [](auto& rhs) {
			if (rhs) delete rhs;
			rhs = NULL;
		});
#else
		std::for_each(cudaStreams.begin(), cudaStreams.end(), [](auto& rhs) {
			if (rhs) delete rhs;
			rhs = NULL;
		});
#endif
	}
	if (cache) delete cache;
	if (upwindFirstENO3aHelper) delete upwindFirstENO3aHelper;
}
bool UpwindFirstWENO5a_impl::operator==(const UpwindFirstWENO5a_impl& rhs) const {
	if (this == &rhs) return true;
	else if (epsilonCalculationMethod_Type != rhs.epsilonCalculationMethod_Type) return false;
	else if ((dxs.size() != rhs.dxs.size()) || !std::equal(dxs.cbegin(), dxs.cend(), rhs.dxs.cbegin())) return false;
	else if ((dx_squares.size() != rhs.dx_squares.size()) || !std::equal(dx_squares.cbegin(), dx_squares.cend(), rhs.dx_squares.cbegin())) return false;
	else if ((dxInvs.size() != rhs.dxInvs.size()) || !std::equal(dxInvs.cbegin(), dxInvs.cend(), rhs.dxInvs.cbegin())) return false;
	else if ((dxInv_2s.size() != rhs.dxInv_2s.size()) || !std::equal(dxInv_2s.cbegin(), dxInv_2s.cend(), rhs.dxInv_2s.cbegin())) return false;
	else if ((dxInv_3s.size() != rhs.dxInv_3s.size()) || !std::equal(dxInv_3s.cbegin(), dxInv_3s.cend(), rhs.dxInv_3s.cbegin())) return false;
	else if ((target_dimension_loop_sizes.size() != rhs.target_dimension_loop_sizes.size()) || !std::equal(target_dimension_loop_sizes.cbegin(), target_dimension_loop_sizes.cend(), rhs.target_dimension_loop_sizes.cbegin())) return false;
	else if ((inner_dimensions_loop_sizes.size() != rhs.inner_dimensions_loop_sizes.size()) || !std::equal(inner_dimensions_loop_sizes.cbegin(), inner_dimensions_loop_sizes.cend(), rhs.inner_dimensions_loop_sizes.cbegin())) return false;
	else if ((first_dimension_loop_sizes.size() != rhs.first_dimension_loop_sizes.size()) || !std::equal(first_dimension_loop_sizes.cbegin(), first_dimension_loop_sizes.cend(), rhs.first_dimension_loop_sizes.cbegin())) return false;
	else if ((src_target_dimension_loop_sizes.size() != rhs.src_target_dimension_loop_sizes.size()) || !std::equal(src_target_dimension_loop_sizes.cbegin(), src_target_dimension_loop_sizes.cend(), rhs.src_target_dimension_loop_sizes.cbegin())) return false;
	else if ((upwindFirstENO3aHelper != rhs.upwindFirstENO3aHelper) && (!upwindFirstENO3aHelper || !rhs.upwindFirstENO3aHelper || !upwindFirstENO3aHelper->operator==(*rhs.upwindFirstENO3aHelper))) return false;
	else return true;
}

void UpwindFirstWENO5a_impl::createCaches()
{
	const size_t first_dimension_loop_size = first_dimension_loop_sizes[1];
	std::vector<beacls::FloatVec > &last_d1ss = cache->last_d1ss;
	std::vector<beacls::FloatVec > &last_d2ss = cache->last_d2ss;
	std::vector<beacls::FloatVec > &last_d3ss = cache->last_d3ss;
	std::vector<beacls::FloatVec > &last_dx_d2ss = cache->last_dx_d2ss;
	std::vector<beacls::FloatVec >&last_smooths1 = cache->last_smooths[0];
	std::vector<beacls::FloatVec >&last_smooths2 = cache->last_smooths[1];
	std::vector<beacls::FloatVec >&last_smooths3 = cache->last_smooths[2];
	if (last_d1ss.size() != 6) last_d1ss.resize(6);
	if (last_d2ss.size() != 5) last_d2ss.resize(5);
	if (last_d3ss.size() != 4) last_d3ss.resize(4);
	if (last_dx_d2ss.size() != 3) last_dx_d2ss.resize(3);
	if (last_smooths1.size() != 2) last_smooths1.resize(2);
	if (last_smooths2.size() != 2) last_smooths2.resize(2);
	if (last_smooths3.size() != 2) last_smooths3.resize(2);
	for_each(last_d1ss.begin(), last_d1ss.end(), ([first_dimension_loop_size](auto& rhs) {if (rhs.size() < first_dimension_loop_size) rhs.resize(first_dimension_loop_size); }));
	for_each(last_d2ss.begin(), last_d2ss.end(), ([first_dimension_loop_size](auto& rhs) {if (rhs.size() < first_dimension_loop_size) rhs.resize(first_dimension_loop_size); }));
	for_each(last_d3ss.begin(), last_d3ss.end(), ([first_dimension_loop_size](auto& rhs) {if (rhs.size() < first_dimension_loop_size) rhs.resize(first_dimension_loop_size); }));
	for_each(last_dx_d2ss.begin(), last_dx_d2ss.end(), ([first_dimension_loop_size](auto& rhs) {if (rhs.size() < first_dimension_loop_size) rhs.resize(first_dimension_loop_size); }));
	for_each(last_smooths1.begin(), last_smooths1.end(), ([first_dimension_loop_size](auto& rhs) {if (rhs.size() < first_dimension_loop_size) rhs.resize(first_dimension_loop_size); }));
	for_each(last_smooths2.begin(), last_smooths2.end(), ([first_dimension_loop_size](auto& rhs) {if (rhs.size() < first_dimension_loop_size) rhs.resize(first_dimension_loop_size); }));
	for_each(last_smooths3.begin(), last_smooths3.end(), ([first_dimension_loop_size](auto& rhs) {if (rhs.size() < first_dimension_loop_size) rhs.resize(first_dimension_loop_size); }));
}

void UpwindFirstWENO5a_impl::getCachePointer(
	std::vector<FLOAT_TYPE*> &dst_pointers,
	std::vector<beacls::FloatVec > &src_pointers,
	const size_t shifted_target_dimension_loop_index
	){
	const size_t num_of_cache_lines = src_pointers.size();
	if (dst_pointers.size() != src_pointers.size()) dst_pointers.resize(src_pointers.size());
	if (num_of_cache_lines > 2) {
		for (size_t i = 0; i < num_of_cache_lines; ++i) {
			size_t cache_index = (((num_of_cache_lines - 2) - (shifted_target_dimension_loop_index % (num_of_cache_lines - 1))) + i) % (num_of_cache_lines - 1);
			dst_pointers[i] = src_pointers[cache_index].data();
		}
	}
	else {
		dst_pointers[0] = src_pointers[shifted_target_dimension_loop_index % 2].data();
		dst_pointers[1] = src_pointers[(shifted_target_dimension_loop_index+1) % 2].data();
	}
}
void UpwindFirstWENO5a_impl::getCachePointers(
	std::vector<FLOAT_TYPE*> &d1s_ms,
	std::vector<FLOAT_TYPE*> &d2s_ms,
	std::vector<FLOAT_TYPE*> &d3s_ms,
	std::vector<FLOAT_TYPE*> &dx_d2ss,
	std::vector<FLOAT_TYPE*> &smooths1_ms,
	std::vector<FLOAT_TYPE*> &smooths2_ms,
	std::vector<FLOAT_TYPE*> &smooths3_ms,
	const size_t shifted_target_dimension_loop_index) {
	getCachePointer(d1s_ms, cache->last_d1ss, shifted_target_dimension_loop_index);
	getCachePointer(d2s_ms, cache->last_d2ss, shifted_target_dimension_loop_index);
	getCachePointer(d3s_ms, cache->last_d3ss, shifted_target_dimension_loop_index);
	getCachePointer(dx_d2ss, cache->last_dx_d2ss, shifted_target_dimension_loop_index);
	getCachePointer(smooths1_ms, cache->last_smooths[0], shifted_target_dimension_loop_index);
	getCachePointer(smooths2_ms, cache->last_smooths[1], shifted_target_dimension_loop_index);
	getCachePointer(smooths3_ms, cache->last_smooths[2], shifted_target_dimension_loop_index);
}

UpwindFirstWENO5a_impl::UpwindFirstWENO5a_impl(const UpwindFirstWENO5a_impl& rhs) :
	type(rhs.type),
	dxs(rhs.dxs),
	dx_squares(rhs.dx_squares),
	dxInvs(rhs.dxInvs),
	dxInv_2s(rhs.dxInv_2s),
	dxInv_3s(rhs.dxInv_3s),
	target_dimension_loop_sizes(rhs.target_dimension_loop_sizes),
	inner_dimensions_loop_sizes(rhs.inner_dimensions_loop_sizes),
	first_dimension_loop_sizes(rhs.first_dimension_loop_sizes),
	src_target_dimension_loop_sizes(rhs.src_target_dimension_loop_sizes),
	stencil(rhs.stencil),
	tmpBoundedSrc_ptrssss(rhs.tmpBoundedSrc_ptrssss),
	tmp_d1s_ms_ites(rhs.tmp_d1s_ms_ites),
	tmp_d2s_ms_ites(rhs.tmp_d2s_ms_ites),
	tmp_d3s_ms_ites(rhs.tmp_d3s_ms_ites),
	tmp_dx_d2s_ms_ites(rhs.tmp_dx_d2s_ms_ites),
	tmp_smooths1_ms_ites(rhs.tmp_smooths1_ms_ites),
	tmp_smooths2_ms_ites(rhs.tmp_smooths2_ms_ites),
	tmp_smooths3_ms_ites(rhs.tmp_smooths3_ms_ites),
	num_of_strides(rhs.num_of_strides),
	d1ss(rhs.d1ss),
	d2ss(rhs.d2ss),
	d3ss(rhs.d3ss),
	tmp_cache_indexes(rhs.tmp_cache_indexes),
	weightL(rhs.weightL),
	weightR(rhs.weightR),
	epsilonCalculationMethod_Type(rhs.epsilonCalculationMethod_Type),
	upwindFirstENO3aHelper(rhs.upwindFirstENO3aHelper->clone()),
	cache(new UpwindFirstWENO5a_Cache(*rhs.cache))
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
#if 1
		//!< Streams for dimension step wise double buffer 
		for (size_t target_dimension = 0; target_dimension < cudaStreams.size(); ++target_dimension) {
			if (target_dimension < 2)
				cudaStreams[target_dimension] = new beacls::CudaStream();
			else
				cudaStreams[target_dimension] = cudaStreams[target_dimension % 2];
		}
#else
		std::for_each(cudaStreams.begin(), cudaStreams.end(), [](auto& rhs) {
			rhs = new beacls::CudaStream();
		});
#endif
	}
	tmpBoundedSrc_uvec_vectors.resize(rhs.tmpBoundedSrc_uvec_vectors.size());
	tmpBoundedSrc_uvecs.resize(rhs.tmpBoundedSrc_uvecs.size());
}

class ECM_Constant {
};
class ECM_MaxOverNeighbor {
};

template<typename T, typename E> inline
void calc_epsilon_dim0(
	T& epsilonL,
	T& epsilonR,
	T& pow_D1_src_0,
	T& pow_D1_src_1,
	T& pow_D1_src_2,
	T& pow_D1_src_3,
	T& pow_D1_src_4,
	const T D1_src_5,
	E e
);

template<typename T> inline
void calc_epsilon_dim0(
	T& epsilonL,
	T& epsilonR,
	T&,
	T&,
	T&,
	T&,
	T&,
	const T,
	ECM_Constant
) {
	epsilonL = epsilonR = (T)1e-6;
}
template<typename T> inline
void calc_epsilon_dim0(
	T& epsilonL,
	T& epsilonR,
	T& pow_D1_src_0,
	T& pow_D1_src_1,
	T& pow_D1_src_2,
	T& pow_D1_src_3,
	T& pow_D1_src_4,
	const T D1_src_5,
	ECM_MaxOverNeighbor
) {
	const T pow_D1_src_5 = D1_src_5 * D1_src_5;
	const T max_1_2 = std::max<T>(pow_D1_src_1, pow_D1_src_2);
	const T max_3_4 = std::max<T>(pow_D1_src_3, pow_D1_src_4);
	const T max_1_2_3_4 = std::max<T>(max_1_2, max_3_4);
	const T maxOverNeighborD1squaredL = std::max<T>(max_1_2_3_4, pow_D1_src_0);
	const T maxOverNeighborD1squaredR = std::max<T>(max_1_2_3_4, pow_D1_src_5);
	epsilonL = (T)(1e-6 * maxOverNeighborD1squaredL + get_epsilon_base<T>());
	epsilonR = (T)(1e-6 * maxOverNeighborD1squaredR + get_epsilon_base<T>());
	pow_D1_src_0 = pow_D1_src_1;
	pow_D1_src_1 = pow_D1_src_2;
	pow_D1_src_2 = pow_D1_src_3;
	pow_D1_src_3 = pow_D1_src_4;
	pow_D1_src_4 = pow_D1_src_5;

}


template<typename T, typename E>
void calc_deriv_dim0(
	T& deriv_l,
	T& deriv_r,
	T& smooth_m1_0,
	T& smooth_m1_1,
	T& smooth_m1_2,
	T& d0_m0,

	T& d0_m1,
	T& d1_m0,
	T& d1_m1,
	T& d1_m2,
	T& d1_m3,
	T& d1_m4,
	T& d1_m5,
	T& d1_m6,
	T& d2_m0,
	T& d2_m1,
	T& d2_m2,
	T& d2_m3,
	T& d3_m0,
	T& d3_m1,
	T& d3_m2,
	T& d3_m3,
	T& D1_src_1,
	T& D1_src_2,
	T& D1_src_3,
	T& D1_src_4,
	T& pow_D1_src_0,
	T& pow_D1_src_1,
	T& pow_D1_src_2,
	T& pow_D1_src_3,
	T& pow_D1_src_4,
	const T* boundedSrc_ptr,
	const T dx,
	const T dxInv,
	const T dxInv_2,
	const T dxInv_3,
	const T dx_square,
	const T x2_dx_square,
	const T weightL0,
	const T weightL1,
	const T weightL2,
	const T weightR0,
	const T weightR1,
	const T weightR2,
	const size_t target_dimension_loop_index,
	const size_t stencil,
	const E e

) {
	d0_m1 = d0_m0;
	d0_m0 = boundedSrc_ptr[target_dimension_loop_index + stencil + 3];
	d1_m6 = d1_m5;
	d1_m5 = d1_m4;
	d1_m4 = d1_m3;
	d1_m3 = d1_m2;
	d1_m2 = d1_m1;
	d1_m1 = d1_m0;
	d1_m0 = dxInv * (d0_m0 - d0_m1);
	d2_m3 = d2_m2;
	d2_m2 = d2_m1;
	d2_m1 = d2_m0;
	d2_m0 = dxInv_2 * (d1_m0 - d1_m1);
	d3_m3 = d3_m2;
	d3_m2 = d3_m1;
	d3_m1 = d3_m0;
	d3_m0 = dxInv_3 * (d2_m0 - d2_m1);

	const T dx_d2_m3 = dx * d2_m3;
	const T dx_d2_m2 = dx * d2_m2;
	const T dx_d2_m1 = dx * d2_m1;

	const T dL0 = d1_m3 + dx_d2_m3;
	const T dL1 = d1_m3 + dx_d2_m3;
	const T dL2 = d1_m3 + dx_d2_m2;

	const T dR0 = d1_m2 - dx_d2_m2;
	const T dR1 = d1_m2 - dx_d2_m2;
	const T dR2 = d1_m2 - dx_d2_m1;

	const T dLL0 = dL0 + x2_dx_square * d3_m3;
	const T dLL1 = dL1 + x2_dx_square * d3_m2;
	const T dLL2 = dL2 - dx_square * d3_m1;

	const T dRR0 = dR0 - dx_square * d3_m2;
	const T dRR1 = dR1 - dx_square * d3_m1;
	const T dRR2 = dR2 + x2_dx_square * d3_m0;
	const T D1_src_5 = d1_m0;
	const T smooth_m0_0 = calcSmooth0(D1_src_1, D1_src_2, D1_src_3);
	const T smooth_m0_1 = calcSmooth1(D1_src_2, D1_src_3, D1_src_4);
	const T smooth_m0_2 = calcSmooth2(D1_src_3, D1_src_4, D1_src_5);
	T epsilonL, epsilonR;
	calc_epsilon_dim0(epsilonL, epsilonR, pow_D1_src_0, pow_D1_src_1, pow_D1_src_2, pow_D1_src_3, pow_D1_src_4, D1_src_5, e);
	deriv_l = weightWENO(dLL0, dLL1, dLL2, smooth_m1_0, smooth_m1_1, smooth_m1_2, weightL0, weightL1, weightL2, epsilonL);
	deriv_r = weightWENO(dRR0, dRR1, dRR2, smooth_m0_0, smooth_m0_1, smooth_m0_2, weightR0, weightR1, weightR2, epsilonR);

	smooth_m1_0 = smooth_m0_0;
	smooth_m1_1 = smooth_m0_1;
	smooth_m1_2 = smooth_m0_2;
	D1_src_1 = D1_src_2;
	D1_src_2 = D1_src_3;
	D1_src_3 = D1_src_4;
	D1_src_4 = D1_src_5;

}

template<typename T, typename E> inline
void calc_epsilon_dim1(
	T& epsilonL,
	T& epsilonR,
	const T D1_src_0,
	const T D1_src_1,
	const T D1_src_2,
	const T D1_src_3,
	const T D1_src_4,
	const T D1_src_5,
	E e
);

template<typename T> inline
void calc_epsilon_dim1(
	T& epsilonL,
	T& epsilonR,
	const T,
	const T,
	const T,
	const T,
	const T,
	const T,
	ECM_Constant
) {
	epsilonL = epsilonR = (T)1e-6;
}
template<typename T> inline
void calc_epsilon_dim1(
	T& epsilonL,
	T& epsilonR,
	const T D1_src_0,
	const T D1_src_1,
	const T D1_src_2,
	const T D1_src_3,
	const T D1_src_4,
	const T D1_src_5,
	ECM_MaxOverNeighbor
) {
	const T pow_D1_src_0 = D1_src_0 * D1_src_0;
	const T pow_D1_src_1 = D1_src_1 * D1_src_1;
	const T pow_D1_src_2 = D1_src_2 * D1_src_2;
	const T pow_D1_src_3 = D1_src_3 * D1_src_3;
	const T pow_D1_src_4 = D1_src_4 * D1_src_4;
	const T pow_D1_src_5 = D1_src_5 * D1_src_5;
	const T max_1_2 = std::max<T>(pow_D1_src_1, pow_D1_src_2);
	const T max_3_4 = std::max<T>(pow_D1_src_3, pow_D1_src_4);
	const T max_1_2_3_4 = std::max<T>(max_1_2, max_3_4);
	const T maxOverNeighborD1squaredL = std::max<T>(max_1_2_3_4, pow_D1_src_0);
	const T maxOverNeighborD1squaredR = std::max<T>(max_1_2_3_4, pow_D1_src_5);
	epsilonL = (T)(1e-6 * maxOverNeighborD1squaredL + get_epsilon_base<T>());
	epsilonR = (T)(1e-6 * maxOverNeighborD1squaredR + get_epsilon_base<T>());
}



template<typename T, typename E>
void calc_deriv_dim1(
	T& deriv_l,
	T& deriv_r,
	FLOAT_TYPE* d1s_ms0,
	const FLOAT_TYPE* d1s_ms1,
	const FLOAT_TYPE* d1s_ms2,
	const FLOAT_TYPE* d1s_ms3,
	const FLOAT_TYPE* d1s_ms4,
	const FLOAT_TYPE* d1s_ms5,
	FLOAT_TYPE* d2s_ms0,
	const FLOAT_TYPE* d2s_ms1,
	const FLOAT_TYPE* d2s_ms2,
	const FLOAT_TYPE* d2s_ms3,
	const FLOAT_TYPE* d2s_ms4,
	FLOAT_TYPE* d3s_ms0,
	const FLOAT_TYPE* d3s_ms1,
	const FLOAT_TYPE* d3s_ms2,
	const FLOAT_TYPE* d3s_ms3,
	FLOAT_TYPE* dx_d2s_ms1,
	const FLOAT_TYPE* dx_d2s_ms2,
	const FLOAT_TYPE* dx_d2s_ms3,
	const T* current_boundedSrc_ptr,
	const T* last_boundedSrc_ptr,
	T* smooth1_m0,
	T* smooth2_m0,
	T* smooth3_m0,
	T* smooth1_m1,
	T* smooth2_m1,
	T* smooth3_m1,
	const T dx,
	const T dxInv,
	const T dxInv_2,
	const T dxInv_3,
	const T dx_square,
	const T x2_dx_square,
	const T weightL0,
	const T weightL1,
	const T weightL2,
	const T weightR0,
	const T weightR1,
	const T weightR2,
	const size_t first_dimension_loop_index,
	const E e

) {
	const T d0_m0 = current_boundedSrc_ptr[first_dimension_loop_index];
	const T d0_m1 = last_boundedSrc_ptr[first_dimension_loop_index];
	const T d1_m0 = dxInv * (d0_m0 - d0_m1);
	const T d1_m1 = d1s_ms1[first_dimension_loop_index];
	const T d1_m2 = d1s_ms2[first_dimension_loop_index];
	const T d1_m3 = d1s_ms3[first_dimension_loop_index];
	const T d1_m4 = d1s_ms4[first_dimension_loop_index];
	const T d1_m5 = d1s_ms5[first_dimension_loop_index];
	const T d2_m0 = dxInv_2 * (d1_m0 - d1_m1);
	const T d2_m1 = d2s_ms1[first_dimension_loop_index];
	const T d2_m2 = d2s_ms2[first_dimension_loop_index];
	const T d2_m3 = d2s_ms3[first_dimension_loop_index];
	const T d2_m4 = d2s_ms4[first_dimension_loop_index];
	const T d3_m0 = dxInv_3 * (d2_m0 - d2_m1);
	const T d3_m1 = d3s_ms1[first_dimension_loop_index];
	const T d3_m2 = d3s_ms2[first_dimension_loop_index];
	const T d3_m3 = d3s_ms3[first_dimension_loop_index];

	const T D1_src_0 = d1_m5;
	const T D1_src_1 = d1_m4;
	const T D1_src_2 = d1_m3;
	const T D1_src_3 = d1_m2;
	const T D1_src_4 = d1_m1;
	const T D1_src_5 = d1_m0;

	const T dx_d2_m3 = dx_d2s_ms3[first_dimension_loop_index];
	const T dx_d2_m2 = dx_d2s_ms2[first_dimension_loop_index];
	const T dx_d2_m1 = dx * d2_m1;

	const T dL0 = d1_m3 + dx_d2_m3;
	const T dL1 = d1_m3 + dx_d2_m3;
	const T dL2 = d1_m3 + dx_d2_m2;

	const T dR0 = d1_m2 - dx_d2_m2;
	const T dR1 = d1_m2 - dx_d2_m2;
	const T dR2 = d1_m2 - dx_d2_m1;

	const T dLL0 = dL0 + x2_dx_square * d3_m3;
	const T dLL1 = dL1 + x2_dx_square * d3_m2;
	const T dLL2 = dL2 - dx_square * d3_m1;

	const T dRR0 = dR0 - dx_square * d3_m2;
	const T dRR1 = dR1 - dx_square * d3_m1;
	const T dRR2 = dR2 + x2_dx_square * d3_m0;

	d1s_ms0[first_dimension_loop_index] = d1_m0;
	d2s_ms0[first_dimension_loop_index] = d2_m0;
	d3s_ms0[first_dimension_loop_index] = d3_m0;
	dx_d2s_ms1[first_dimension_loop_index] = dx_d2_m1;

	const T smooth_m0_0 = calcSmooth0(D1_src_1, D1_src_2, D1_src_3);
	const T smooth_m0_1 = calcSmooth1(D1_src_2, D1_src_3, D1_src_4);
	const T smooth_m0_2 = calcSmooth2(D1_src_3, D1_src_4, D1_src_5);
	const T smooth_m1_0 = smooth1_m1[first_dimension_loop_index];
	const T smooth_m1_1 = smooth2_m1[first_dimension_loop_index];
	const T smooth_m1_2 = smooth3_m1[first_dimension_loop_index];
	smooth1_m0[first_dimension_loop_index] = smooth_m0_0;
	smooth2_m0[first_dimension_loop_index] = smooth_m0_1;
	smooth3_m0[first_dimension_loop_index] = smooth_m0_2;

	T epsilonL, epsilonR;
	calc_epsilon_dim1(epsilonL, epsilonR, D1_src_0, D1_src_1, D1_src_2, D1_src_3, D1_src_4, D1_src_5, e);
	deriv_l = weightWENO(dLL0, dLL1, dLL2, smooth_m1_0, smooth_m1_1, smooth_m1_2, weightL0, weightL1, weightL2, epsilonL);
	deriv_r = weightWENO(dRR0, dRR1, dRR2, smooth_m0_0, smooth_m0_1, smooth_m0_2, weightR0, weightR1, weightR2, epsilonR);
}

template<typename T, typename E> inline
void calc_epsilon_dimLET2(
	T& epsilonL,
	T& epsilonR,
	T& pow_D1_src_0,
	T& pow_D1_src_1,
	T& pow_D1_src_2,
	T& pow_D1_src_3,
	T& pow_D1_src_4,
	const T D1_src_5,
	E e
);
template<typename T> inline
void calc_epsilon_dimLET2(
	T& epsilonL,
	T& epsilonR,
	const T,
	const T,
	const T,
	const T,
	const T,
	const T,
	ECM_Constant
) {
	epsilonL = epsilonR = (T)1e-6;
}
template<typename T> inline
void calc_epsilon_dimLET2(
	T& epsilonL,
	T& epsilonR,
	const T D1_src_0,
	const T D1_src_1,
	const T D1_src_2,
	const T D1_src_3,
	const T D1_src_4,
	const T D1_src_5,
	ECM_MaxOverNeighbor
) {
	const T pow_D1_src_0 = D1_src_0 * D1_src_0;
	const T pow_D1_src_1 = D1_src_1 * D1_src_1;
	const T pow_D1_src_2 = D1_src_2 * D1_src_2;
	const T pow_D1_src_3 = D1_src_3 * D1_src_3;
	const T pow_D1_src_4 = D1_src_4 * D1_src_4;
	const T pow_D1_src_5 = D1_src_5 * D1_src_5;
	const T max_1_2 = std::max<T>(pow_D1_src_1, pow_D1_src_2);
	const T max_3_4 = std::max<T>(pow_D1_src_3, pow_D1_src_4);
	const T max_1_2_3_4 = std::max<T>(max_1_2, max_3_4);
	const T maxOverNeighborD1squaredL = std::max<T>(max_1_2_3_4, pow_D1_src_0);
	const T maxOverNeighborD1squaredR = std::max<T>(max_1_2_3_4, pow_D1_src_5);
	epsilonL = (T)(1e-6 * maxOverNeighborD1squaredL + get_epsilon_base<T>());
	epsilonR = (T)(1e-6 * maxOverNeighborD1squaredR + get_epsilon_base<T>());
}



template<typename T, typename E>
void calc_deriv_dimLET2(
	T* dst_deriv_l_ptr,
	T* dst_deriv_r_ptr,
	std::vector<std::vector<std::vector<const T*> > >&  tmpBoundedSrc_ptrsss,
	const T dx,
	const T dxInv,
	const T dxInv_2,
	const T dxInv_3,
	const T dx_square,
	const T x2_dx_square,
	const T weightL0,
	const T weightL1,
	const T weightL2,
	const T weightR0,
	const T weightR1,
	const T weightR2,
	const size_t slice_length,
	const size_t num_of_slices,
	const size_t loop_length,
	const size_t first_dimension_loop_size,
	const size_t dim,
	const E e

) {
	for (size_t slice_index = 0; slice_index < num_of_slices; ++slice_index) {
		for (size_t loop_index = 0; loop_index < loop_length; ++loop_index) {
			const size_t slice_offset = slice_index * slice_length;
			size_t dst_offset = loop_index * first_dimension_loop_size + slice_offset;
			const size_t src_slice_index = ((dim == 2) ? 0 : slice_index);
			const size_t src_stride_offset = ((dim == 2) ? slice_index : 0);
			const std::vector<const T*>& tmpBoundedSrc_ptrs = tmpBoundedSrc_ptrsss[src_slice_index][loop_index];
			const T* tmpBoundedSrc_ptrs0 = tmpBoundedSrc_ptrs[0 + src_stride_offset];
			const T* tmpBoundedSrc_ptrs1 = tmpBoundedSrc_ptrs[1 + src_stride_offset];
			const T* tmpBoundedSrc_ptrs2 = tmpBoundedSrc_ptrs[2 + src_stride_offset];
			const T* tmpBoundedSrc_ptrs3 = tmpBoundedSrc_ptrs[3 + src_stride_offset];
			const T* tmpBoundedSrc_ptrs4 = tmpBoundedSrc_ptrs[4 + src_stride_offset];
			const T* tmpBoundedSrc_ptrs5 = tmpBoundedSrc_ptrs[5 + src_stride_offset];
			const T* tmpBoundedSrc_ptrs6 = tmpBoundedSrc_ptrs[6 + src_stride_offset];
			for (size_t first_dimension_loop_index = 0; first_dimension_loop_index < first_dimension_loop_size; ++first_dimension_loop_index) {
				const T d0_m0 = tmpBoundedSrc_ptrs6[first_dimension_loop_index];
				const T d0_m1 = tmpBoundedSrc_ptrs5[first_dimension_loop_index];
				const T d0_m2 = tmpBoundedSrc_ptrs4[first_dimension_loop_index];
				const T d0_m3 = tmpBoundedSrc_ptrs3[first_dimension_loop_index];
				const T d0_m4 = tmpBoundedSrc_ptrs2[first_dimension_loop_index];
				const T d0_m5 = tmpBoundedSrc_ptrs1[first_dimension_loop_index];
				const T d0_m6 = tmpBoundedSrc_ptrs0[first_dimension_loop_index];
				const T d1_m0 = dxInv * (d0_m0 - d0_m1);
				const T d1_m1 = dxInv * (d0_m1 - d0_m2);
				const T d1_m2 = dxInv * (d0_m2 - d0_m3);
				const T d1_m3 = dxInv * (d0_m3 - d0_m4);
				const T d1_m4 = dxInv * (d0_m4 - d0_m5);
				const T d1_m5 = dxInv * (d0_m5 - d0_m6);
				const T d2_m0 = dxInv_2 * (d1_m0 - d1_m1);
				const T d2_m1 = dxInv_2 * (d1_m1 - d1_m2);
				const T d2_m2 = dxInv_2 * (d1_m2 - d1_m3);
				const T d2_m3 = dxInv_2 * (d1_m3 - d1_m4);
				const T d2_m4 = dxInv_2 * (d1_m4 - d1_m5);
				const T d3_m0 = dxInv_3 * (d2_m0 - d2_m1);
				const T d3_m1 = dxInv_3 * (d2_m1 - d2_m2);
				const T d3_m2 = dxInv_3 * (d2_m2 - d2_m3);
				const T d3_m3 = dxInv_3 * (d2_m3 - d2_m4);

				const T D1_src_0 = d1_m5;
				const T D1_src_1 = d1_m4;
				const T D1_src_2 = d1_m3;
				const T D1_src_3 = d1_m2;
				const T D1_src_4 = d1_m1;
				const T D1_src_5 = d1_m0;

				const T dx_d2_m3 = dx * d2_m3;
				const T dx_d2_m2 = dx * d2_m2;
				const T dx_d2_m1 = dx * d2_m1;

				const T dL0 = d1_m3 + dx_d2_m3;
				const T dL1 = d1_m3 + dx_d2_m3;
				const T dL2 = d1_m3 + dx_d2_m2;

				const T dR0 = d1_m2 - dx_d2_m2;
				const T dR1 = d1_m2 - dx_d2_m2;
				const T dR2 = d1_m2 - dx_d2_m1;

				const T dLL0 = dL0 + x2_dx_square * d3_m3;
				const T dLL1 = dL1 + x2_dx_square * d3_m2;
				const T dLL2 = dL2 - dx_square * d3_m1;

				const T dRR0 = dR0 - dx_square * d3_m2;
				const T dRR1 = dR1 - dx_square * d3_m1;
				const T dRR2 = dR2 + x2_dx_square * d3_m0;

				const T smoothL_0 = calcSmooth0(D1_src_0, D1_src_1, D1_src_2);
				const T smoothL_1 = calcSmooth1(D1_src_1, D1_src_2, D1_src_3);
				const T smoothL_2 = calcSmooth2(D1_src_2, D1_src_3, D1_src_4);
				const T smoothR_0 = calcSmooth0(D1_src_1, D1_src_2, D1_src_3);
				const T smoothR_1 = calcSmooth1(D1_src_2, D1_src_3, D1_src_4);
				const T smoothR_2 = calcSmooth2(D1_src_3, D1_src_4, D1_src_5);
				const T epsilon = (T)1e-6;

				const size_t dst_index = first_dimension_loop_index + dst_offset;
				T epsilonL, epsilonR;
				calc_epsilon_dimLET2(epsilonL, epsilonR, D1_src_0, D1_src_1, D1_src_2, D1_src_3, D1_src_4, D1_src_5, e);
				dst_deriv_l_ptr[dst_index] = weightWENO(dLL0, dLL1, dLL2, smoothL_0, smoothL_1, smoothL_2, weightL0, weightL1, weightL2, epsilonL);
				dst_deriv_r_ptr[dst_index] = weightWENO(dRR0, dRR1, dRR2, smoothR_0, smoothR_1, smoothR_2, weightR0, weightR1, weightR2, epsilonR);
			}
		}
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

	beacls::UVecDepth depth = dst_deriv_l.depth();

	// We need the three ENO approximations 
	// plus the(unstripped) divided differences to pick the least oscillatory.
	beacls::CudaStream* cudaStream = cudaStreams[dim];
	BoundaryCondition* boundaryCondition = grid->get_boundaryCondition(dim);
	const FLOAT_TYPE dxInv = dxInvs[dim];
	const FLOAT_TYPE dxInv_2 = dxInv_2s[dim];
	const FLOAT_TYPE dxInv_3 = dxInv_3s[dim];
	const FLOAT_TYPE dx_square = dx_squares[dim];
	const FLOAT_TYPE x2_dx_square = 2 * dx_square;
	const FLOAT_TYPE dx = dxs[dim];
	const size_t target_dimension_loop_size = target_dimension_loop_sizes[dim];

	beacls::UVec& boundedSrc = tmpBoundedSrc_uvecs[dim%2];
	size_t total_boundedSrc_size = target_dimension_loop_size * outer_dimensions_loop_length * num_of_slices;
	// Add ghost cells.
	if (boundedSrc.type() != type) boundedSrc = beacls::UVec(depth, type, total_boundedSrc_size);
	else if (boundedSrc.size() != total_boundedSrc_size) boundedSrc.resize(total_boundedSrc_size);
	boundedSrc.set_cudaStream(cudaStream);
	boundaryCondition->execute(
		src,
		boundedSrc,
		tmpBoundedSrc_uvec_vectors[dim%2],
		stencil,
		outer_dimensions_loop_length,
		target_dimension_loop_size,
		first_dimension_loop_size,
		loop_begin,
		num_of_slices
	);
	FLOAT_TYPE* boundedSrc_base_ptr = beacls::UVec_<FLOAT_TYPE>(boundedSrc).ptr();
	if (type == beacls::UVecType_Cuda) {
		boundedSrc_base_ptr = beacls::UVec_<FLOAT_TYPE>(boundedSrc).ptr();
		const size_t src_target_dimension_loop_size = src_target_dimension_loop_sizes[dim];
		FLOAT_TYPE* dst_deriv_l_ptr = beacls::UVec_<FLOAT_TYPE>(dst_deriv_l).ptr();
		FLOAT_TYPE* dst_deriv_r_ptr = beacls::UVec_<FLOAT_TYPE>(dst_deriv_r).ptr();
		dst_deriv_l.set_cudaStream(cudaStream);
		dst_deriv_r.set_cudaStream(cudaStream);

		const FLOAT_TYPE weightL0 = weightL[0];
		const FLOAT_TYPE weightL1 = weightL[1];
		const FLOAT_TYPE weightL2 = weightL[2];
		const FLOAT_TYPE weightR0 = weightR[0];
		const FLOAT_TYPE weightR1 = weightR[1];
		const FLOAT_TYPE weightR2 = weightR[2];
		size_t loop_length = slice_length / first_dimension_loop_size;
		UpwindFirstWENO5a_execute_dim0_cuda2(
			dst_deriv_l_ptr, dst_deriv_r_ptr,
			boundedSrc_base_ptr, dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
			weightL0, weightL1, weightL2, weightR0, weightR1, weightR2,
			num_of_slices, 
			loop_length, src_target_dimension_loop_size, first_dimension_loop_size, slice_length,
			stencil,
			epsilonCalculationMethod_Type,
			cudaStream
		);
	}
	else
	{
		const size_t src_target_dimension_loop_size = src_target_dimension_loop_sizes[dim];
		FLOAT_TYPE* dst_deriv_l_ptr = beacls::UVec_<FLOAT_TYPE>(dst_deriv_l).ptr();
		FLOAT_TYPE* dst_deriv_r_ptr = beacls::UVec_<FLOAT_TYPE>(dst_deriv_r).ptr();

		const FLOAT_TYPE weightL0 = weightL[0];
		const FLOAT_TYPE weightL1 = weightL[1];
		const FLOAT_TYPE weightL2 = weightL[2];
		const FLOAT_TYPE weightR0 = weightR[0];
		const FLOAT_TYPE weightR1 = weightR[1];
		const FLOAT_TYPE weightR2 = weightR[2];

		for (size_t slice_index = 0; slice_index < num_of_slices; ++slice_index) {
			const size_t slice_loop_offset = slice_index * outer_dimensions_loop_length;
			const size_t slice_offset = slice_index * slice_length;
			for (size_t loop_index = 0; loop_index < outer_dimensions_loop_length; ++loop_index) {
				const size_t loop_index_with_slice = loop_index + slice_loop_offset;
				//! Target dimension is first loop
				const FLOAT_TYPE* boundedSrc_ptr = boundedSrc_base_ptr + (first_dimension_loop_size + stencil * 2) * loop_index_with_slice;

				FLOAT_TYPE d0_m0 = boundedSrc_ptr[0];
				FLOAT_TYPE d0_m1 = 0;
				FLOAT_TYPE d1_m0 = 0;
				FLOAT_TYPE d1_m1 = 0;
				FLOAT_TYPE d1_m2 = 0;
				FLOAT_TYPE d1_m3 = 0;
				FLOAT_TYPE d1_m4 = 0;
				FLOAT_TYPE d1_m5 = 0;
				FLOAT_TYPE d1_m6 = 0;
				FLOAT_TYPE d2_m0 = 0;
				FLOAT_TYPE d2_m1 = 0;
				FLOAT_TYPE d2_m2 = 0;
				FLOAT_TYPE d2_m3 = 0;
				FLOAT_TYPE d3_m0 = 0;
				FLOAT_TYPE d3_m1 = 0;
				FLOAT_TYPE d3_m2 = 0;
				FLOAT_TYPE d3_m3 = 0;

				const size_t dst_offset = loop_index * first_dimension_loop_size + slice_offset;

				FLOAT_TYPE D1_src_0 = 0;
				FLOAT_TYPE D1_src_1 = 0;
				FLOAT_TYPE D1_src_2 = 0;
				FLOAT_TYPE D1_src_3 = 0;
				FLOAT_TYPE D1_src_4 = 0;

				//! Prologue
				FLOAT_TYPE smooth_m1_0 = 0;
				FLOAT_TYPE smooth_m1_1 = 0;
				FLOAT_TYPE smooth_m1_2 = 0;
				FLOAT_TYPE pow_D1_src_0 = 0;
				FLOAT_TYPE pow_D1_src_1 = 0;
				FLOAT_TYPE pow_D1_src_2 = 0;
				FLOAT_TYPE pow_D1_src_3 = 0;
				FLOAT_TYPE pow_D1_src_4 = 0;
				for (size_t target_dimension_loop_index = 0; target_dimension_loop_index < stencil + 2; ++target_dimension_loop_index) {
					size_t src_index = target_dimension_loop_index;
					d0_m1 = d0_m0;
					d0_m0 = boundedSrc_ptr[src_index + 1];
					d1_m6 = d1_m5;
					d1_m5 = d1_m4;
					d1_m4 = d1_m3;
					d1_m3 = d1_m2;
					d1_m2 = d1_m1;
					d1_m1 = d1_m0;
					D1_src_0 = D1_src_1;
					D1_src_1 = D1_src_2;
					D1_src_2 = D1_src_3;
					D1_src_3 = D1_src_4;
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
						}
					}
					D1_src_4 = d1_m0;
					smooth_m1_0 = calcSmooth0(D1_src_0, D1_src_1, D1_src_2);
					smooth_m1_1 = calcSmooth1(D1_src_1, D1_src_2, D1_src_3);
					smooth_m1_2 = calcSmooth2(D1_src_2, D1_src_3, D1_src_4);
					if (epsilonCalculationMethod_Type == levelset::EpsilonCalculationMethod_maxOverNeighbor) {
						pow_D1_src_0 = D1_src_0 * D1_src_0;
						pow_D1_src_1 = D1_src_1 * D1_src_1;
						pow_D1_src_2 = D1_src_2 * D1_src_2;
						pow_D1_src_3 = D1_src_3 * D1_src_3;
						pow_D1_src_4 = D1_src_4 * D1_src_4;
					}
				}
				// The smoothness estimates may have some relation to the higher order
				// divided differences, but it isn't obvious from just reading O&F.
				// For now, use only the first order divided differences.
				//Prologue
				// Body
				switch (epsilonCalculationMethod_Type) {
				case levelset::EpsilonCalculationMethod_Invalid:
				default:
					printf("Unknown epsilonCalculationMethod %d\n", epsilonCalculationMethod_Type);
					return false;
				case levelset::EpsilonCalculationMethod_Constant:
					for (size_t target_dimension_loop_index = 0; target_dimension_loop_index < src_target_dimension_loop_size; ++target_dimension_loop_index) {
						const size_t dst_index = target_dimension_loop_index + dst_offset;
						calc_deriv_dim0(
							dst_deriv_l_ptr[dst_index], dst_deriv_r_ptr[dst_index],
							smooth_m1_0, smooth_m1_1, smooth_m1_2,
							d0_m0, d0_m1,
							d1_m0, d1_m1, d1_m2, d1_m3, d1_m4, d1_m5, d1_m6,
							d2_m0, d2_m1, d2_m2, d2_m3,
							d3_m0, d3_m1, d3_m2, d3_m3,
							D1_src_1, D1_src_2, D1_src_3, D1_src_4,
							pow_D1_src_0, pow_D1_src_1, pow_D1_src_2, pow_D1_src_3, pow_D1_src_4,
							boundedSrc_ptr,
							dx, dxInv, dxInv_2, dxInv_3, dx_square,  x2_dx_square,
							weightL0, weightL1, weightL2, weightR0, weightR1, weightR2,
							target_dimension_loop_index, stencil,
							ECM_Constant()
						);
					}
					break;
				case levelset::EpsilonCalculationMethod_maxOverGrid:
					printf("epsilonCalculationMethod %d is not supported yet\n", epsilonCalculationMethod_Type);
					return false;
				case levelset::EpsilonCalculationMethod_maxOverNeighbor:
					for (size_t target_dimension_loop_index = 0; target_dimension_loop_index < src_target_dimension_loop_size; ++target_dimension_loop_index) {
						const size_t dst_index = target_dimension_loop_index + dst_offset;
						calc_deriv_dim0(
							dst_deriv_l_ptr[dst_index], dst_deriv_r_ptr[dst_index],
							smooth_m1_0, smooth_m1_1, smooth_m1_2,
							d0_m0, d0_m1,
							d1_m0, d1_m1, d1_m2, d1_m3, d1_m4, d1_m5, d1_m6,
							d2_m0, d2_m1, d2_m2, d2_m3,
							d3_m0, d3_m1, d3_m2, d3_m3,
							D1_src_1, D1_src_2, D1_src_3, D1_src_4,
							pow_D1_src_0, pow_D1_src_1, pow_D1_src_2, pow_D1_src_3, pow_D1_src_4,
							boundedSrc_ptr,
							dx, dxInv, dxInv_2, dxInv_3, dx_square, x2_dx_square,
							weightL0, weightL1, weightL2, weightR0, weightR1, weightR2,
							target_dimension_loop_index, stencil,
							ECM_MaxOverNeighbor()
						);
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

	beacls::UVecDepth depth = dst_deriv_l.depth();

	beacls::CudaStream* cudaStream = cudaStreams[dim];
	BoundaryCondition* boundaryCondition = grid->get_boundaryCondition(dim);
	const size_t src_target_dimension_loop_size = src_target_dimension_loop_sizes[dim];
	const size_t inner_dimensions_loop_size = inner_dimensions_loop_sizes[dim];

	const size_t loop_length = slice_length / first_dimension_loop_size;
	const size_t target_dimension_loop_size = target_dimension_loop_sizes[dim];

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
		prologue_loop_size = stencil * 2 - target_dimension_loop_begin;
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
		rhs.resize(loop_length + prologue_loop_size);
		for_each(rhs.begin(), rhs.end(), [](auto &rhs) {
			rhs.resize(1);
		});
	});
	beacls::UVec& boundedSrc = tmpBoundedSrc_uvecs[dim%2];
	size_t total_boundedSrc_size = first_dimension_loop_size * num_of_slices*(loop_length + prologue_loop_size);
	if (boundedSrc.type() != type) boundedSrc = beacls::UVec(depth, type, total_boundedSrc_size);
	else if (boundedSrc.size() != total_boundedSrc_size) boundedSrc.resize(total_boundedSrc_size);
	boundedSrc.set_cudaStream(cudaStream);
	const FLOAT_TYPE dxInv = dxInvs[dim];
	const FLOAT_TYPE dxInv_2 = dxInv_2s[dim];
	const FLOAT_TYPE dxInv_3 = dxInv_3s[dim];
	const FLOAT_TYPE dx_square = dx_squares[dim];
	const FLOAT_TYPE x2_dx_square = 2 * dx_square;
	const FLOAT_TYPE dx = dxs[dim];

	boundaryCondition->execute(
		src,
		boundedSrc, tmpBoundedSrc_uvec_vectors[dim%2], dst_ptrsss,
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
		FLOAT_TYPE* dst_deriv_l_ptr = beacls::UVec_<FLOAT_TYPE>(dst_deriv_l).ptr();
		FLOAT_TYPE* dst_deriv_r_ptr = beacls::UVec_<FLOAT_TYPE>(dst_deriv_r).ptr();
		dst_deriv_l.set_cudaStream(cudaStream);
		dst_deriv_r.set_cudaStream(cudaStream);
		const FLOAT_TYPE weightL0 = weightL[0];
		const FLOAT_TYPE weightL1 = weightL[1];
		const FLOAT_TYPE weightL2 = weightL[2];
		const FLOAT_TYPE weightR0 = weightR[0];
		const FLOAT_TYPE weightR1 = weightR[1];
		const FLOAT_TYPE weightR2 = weightR[2];
		UpwindFirstWENO5a_execute_dim1_cuda2(
			dst_deriv_l_ptr, dst_deriv_r_ptr,
			dst_ptrsss[0][0][0], dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
			weightL0, weightL1, weightL2, weightR0, weightR1, weightR2,
			num_of_slices, loop_length, first_dimension_loop_size, slice_length,
			stencil,
			epsilonCalculationMethod_Type,
			cudaStream
		);
	}
	else
	{
		createCaches();
		std::vector<FLOAT_TYPE*> &d1s_ms = tmp_d1s_ms_ites;
		std::vector<FLOAT_TYPE*> &d2s_ms = tmp_d2s_ms_ites;
		std::vector<FLOAT_TYPE*> &d3s_ms = tmp_d3s_ms_ites;
		std::vector<FLOAT_TYPE*> &dx_d2s_ms = tmp_dx_d2s_ms_ites;
		std::vector<FLOAT_TYPE*> &smooths1_ms = tmp_smooths1_ms_ites;
		std::vector<FLOAT_TYPE*> &smooths2_ms = tmp_smooths2_ms_ites;
		std::vector<FLOAT_TYPE*> &smooths3_ms = tmp_smooths3_ms_ites;
		FLOAT_TYPE* dst_deriv_l_ptr = beacls::UVec_<FLOAT_TYPE>(dst_deriv_l).ptr();
		FLOAT_TYPE* dst_deriv_r_ptr = beacls::UVec_<FLOAT_TYPE>(dst_deriv_r).ptr();
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
					const size_t tmpSmooth_current_index = loop_index % 2;
					const size_t tmpSmooth_last_index = (tmpSmooth_current_index == 0) ? 1 : 0;
					if (target_dimension_loop_index == 0) {
						for (size_t prologue_target_dimension_loop_index = 0; prologue_target_dimension_loop_index < stencil * 2; ++prologue_target_dimension_loop_index) {
							if ((prologue_target_dimension_loop_index >= 1)) {
								const FLOAT_TYPE* current_boundedSrc_ptr = dst_ptrsss[slice_index][prologue_target_dimension_loop_index][0];
								const FLOAT_TYPE* last_boundedSrc_ptr = dst_ptrsss[slice_index][prologue_target_dimension_loop_index - 1][0];

								getCachePointers(d1s_ms, d2s_ms, d3s_ms, dx_d2s_ms,
									smooths1_ms, smooths2_ms, smooths3_ms,
									prologue_target_dimension_loop_index);
								FLOAT_TYPE* d1s_ms0 = d1s_ms[0];
								FLOAT_TYPE* d2s_ms0 = d2s_ms[0];
								FLOAT_TYPE* d3s_ms0 = d3s_ms[0];
								FLOAT_TYPE* dx_d2s_ms1 = dx_d2s_ms[0];
								FLOAT_TYPE* smooth1_m0 = smooths1_ms[0];
								FLOAT_TYPE* smooth2_m0 = smooths2_ms[0];
								FLOAT_TYPE* smooth3_m0 = smooths3_ms[0];

								const FLOAT_TYPE* d1s_ms1 = d1s_ms[1];
								const FLOAT_TYPE* d1s_ms2 = d1s_ms[2];
								const FLOAT_TYPE* d1s_ms3 = d1s_ms[3];
								const FLOAT_TYPE* d1s_ms4 = d1s_ms[4];
								const FLOAT_TYPE* d2s_ms1 = d2s_ms[1];

								for (size_t first_dimension_loop_index = 0; first_dimension_loop_index < first_dimension_loop_size; ++first_dimension_loop_index) {
									const FLOAT_TYPE d0_m0 = current_boundedSrc_ptr[first_dimension_loop_index];
									const FLOAT_TYPE d0_m1 = last_boundedSrc_ptr[first_dimension_loop_index];
									const FLOAT_TYPE d1_m0 = dxInv * (d0_m0 - d0_m1);
									d1s_ms0[first_dimension_loop_index] = d1_m0;
									if (prologue_target_dimension_loop_index >= 2) {
										const FLOAT_TYPE d1_m1 = d1s_ms1[first_dimension_loop_index];
										const FLOAT_TYPE d2_m0 = dxInv_2 * (d1_m0 - d1_m1);
										d2s_ms0[first_dimension_loop_index] = d2_m0;
										if (prologue_target_dimension_loop_index >= 3) {
											const FLOAT_TYPE d2_m1 = d2s_ms1[first_dimension_loop_index];
											const FLOAT_TYPE d3_m0 = dxInv_3 * (d2_m0 - d2_m1);
											d3s_ms0[first_dimension_loop_index] = d3_m0;
											const FLOAT_TYPE dx_d2_m1 = dx * d2_m1;
											dx_d2s_ms1[first_dimension_loop_index] = dx_d2_m1;
											if (prologue_target_dimension_loop_index >= stencil * 2-1) {
												const FLOAT_TYPE d1_m2 = d1s_ms2[first_dimension_loop_index];
												const FLOAT_TYPE d1_m3 = d1s_ms3[first_dimension_loop_index];
												const FLOAT_TYPE d1_m4 = d1s_ms4[first_dimension_loop_index];
												const FLOAT_TYPE D1_src_0 = d1_m4;
												const FLOAT_TYPE D1_src_1 = d1_m3;
												const FLOAT_TYPE D1_src_2 = d1_m2;
												const FLOAT_TYPE D1_src_3 = d1_m1;
												const FLOAT_TYPE D1_src_4 = d1_m0;

												smooth1_m0[first_dimension_loop_index] = calcSmooth0(D1_src_0, D1_src_1, D1_src_2);
												smooth2_m0[first_dimension_loop_index] = calcSmooth1(D1_src_1, D1_src_2, D1_src_3);
												smooth3_m0[first_dimension_loop_index] = calcSmooth2(D1_src_2, D1_src_3, D1_src_4);
											}
										}
									}
								}
								prologue_offset += first_dimension_loop_size;
							}
						}
					}
				}
				if (target_dimension_loop_index < (target_dimension_loop_size - (stencil + 1))) {
					//! Body
					const size_t shifted_target_dimension_loop_index = target_dimension_loop_index + stencil * 2;

					const FLOAT_TYPE* last_boundedSrc_ptr = NULL;
					const FLOAT_TYPE* current_boundedSrc_ptr = dst_ptrsss[slice_index][index + prologue_loop_size][0];
					last_boundedSrc_ptr = dst_ptrsss[slice_index][index + prologue_loop_size - 1][0];
					getCachePointers(d1s_ms, d2s_ms, d3s_ms, dx_d2s_ms,
						smooths1_ms, smooths2_ms, smooths3_ms,
						shifted_target_dimension_loop_index);
					const size_t dst_offset = index * first_dimension_loop_size + dst_slice_offset;
					FLOAT_TYPE* smooth1_m1 = smooths1_ms[1];
					FLOAT_TYPE* smooth2_m1 = smooths2_ms[1];
					FLOAT_TYPE* smooth3_m1 = smooths3_ms[1];
					FLOAT_TYPE* smooth1_m0 = smooths1_ms[0];
					FLOAT_TYPE* smooth2_m0 = smooths2_ms[0];
					FLOAT_TYPE* smooth3_m0 = smooths3_ms[0];
					FLOAT_TYPE* d1s_ms0 = d1s_ms[0];
					FLOAT_TYPE* d2s_ms0 = d2s_ms[0];
					FLOAT_TYPE* d3s_ms0 = d3s_ms[0];
					FLOAT_TYPE* dx_d2s_ms1 = dx_d2s_ms[0];
					const FLOAT_TYPE* d1s_ms1 = d1s_ms[1];
					const FLOAT_TYPE* d1s_ms2 = d1s_ms[2];
					const FLOAT_TYPE* d1s_ms3 = d1s_ms[3];
					const FLOAT_TYPE* d1s_ms4 = d1s_ms[4];
					const FLOAT_TYPE* d1s_ms5 = d1s_ms[5];
					const FLOAT_TYPE* d2s_ms1 = d2s_ms[1];
					const FLOAT_TYPE* d2s_ms2 = d2s_ms[2];
					const FLOAT_TYPE* d2s_ms3 = d2s_ms[3];
					const FLOAT_TYPE* d2s_ms4 = d2s_ms[4];
					const FLOAT_TYPE* d3s_ms1 = d3s_ms[1];
					const FLOAT_TYPE* d3s_ms2 = d3s_ms[2];
					const FLOAT_TYPE* d3s_ms3 = d3s_ms[3];
					const FLOAT_TYPE* dx_d2s_ms2 = dx_d2s_ms[1];
					const FLOAT_TYPE* dx_d2s_ms3 = dx_d2s_ms[2];
					const FLOAT_TYPE weightL0 = weightL[0];
					const FLOAT_TYPE weightL1 = weightL[1];
					const FLOAT_TYPE weightL2 = weightL[2];
					const FLOAT_TYPE weightR0 = weightR[0];
					const FLOAT_TYPE weightR1 = weightR[1];
					const FLOAT_TYPE weightR2 = weightR[2];

					switch (epsilonCalculationMethod_Type) {
					case levelset::EpsilonCalculationMethod_Invalid:
					default:
						printf("Unknown epsilonCalculationMethod %d\n", epsilonCalculationMethod_Type);
						return false;
					case levelset::EpsilonCalculationMethod_Constant:
						for (size_t first_dimension_loop_index = 0; first_dimension_loop_index < first_dimension_loop_size; ++first_dimension_loop_index) {
							const size_t dst_index = first_dimension_loop_index + dst_slice_offset;
							calc_deriv_dim1(
								dst_deriv_l_ptr[dst_index], dst_deriv_r_ptr[dst_index],
								d1s_ms0, d1s_ms1, d1s_ms2, d1s_ms3, d1s_ms4, d1s_ms5,
								d2s_ms0, d2s_ms1, d2s_ms2, d2s_ms3, d2s_ms4,
								d3s_ms0, d3s_ms1, d3s_ms2, d3s_ms3,
								dx_d2s_ms1, dx_d2s_ms2, dx_d2s_ms3,
								current_boundedSrc_ptr, last_boundedSrc_ptr,
								smooth1_m0, smooth2_m0, smooth3_m0, smooth1_m1, smooth2_m1, smooth3_m1,
								dx, dxInv, dxInv_2, dxInv_3, dx_square, x2_dx_square,
								weightL0, weightL1, weightL2, weightR0, weightR1, weightR2, first_dimension_loop_index,
								ECM_Constant()
							);
						}
						break;

					case levelset::EpsilonCalculationMethod_maxOverGrid:
						printf("epsilonCalculationMethod %d is not supported yet\n", epsilonCalculationMethod_Type);
						return false;
					case levelset::EpsilonCalculationMethod_maxOverNeighbor:
						for (size_t first_dimension_loop_index = 0; first_dimension_loop_index < first_dimension_loop_size; ++first_dimension_loop_index) {
							const size_t dst_index = first_dimension_loop_index + dst_slice_offset;
							calc_deriv_dim1(
								dst_deriv_l_ptr[dst_index], dst_deriv_r_ptr[dst_index],
								d1s_ms0, d1s_ms1, d1s_ms2, d1s_ms3, d1s_ms4, d1s_ms5,
								d2s_ms0, d2s_ms1, d2s_ms2, d2s_ms3, d2s_ms4,
								d3s_ms0, d3s_ms1, d3s_ms2, d3s_ms3,
								dx_d2s_ms1, dx_d2s_ms2, dx_d2s_ms3,
								current_boundedSrc_ptr, last_boundedSrc_ptr,
								smooth1_m0, smooth2_m0, smooth3_m0, smooth1_m1, smooth2_m1, smooth3_m1,
								dx, dxInv, dxInv_2, dxInv_3, dx_square, x2_dx_square,
								weightL0, weightL1, weightL2, weightR0, weightR1, weightR2, first_dimension_loop_index,
								ECM_MaxOverNeighbor()
							);
						}
						break;
					}
				}
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

	beacls::UVecDepth depth = dst_deriv_l.depth();
	const size_t num_of_merged_slices = ((dim == 2) ? 1 : num_of_slices);
	const size_t num_of_merged_strides = ((dim == 2) ? num_of_strides + num_of_slices - 1 : num_of_strides);

	size_t margined_loop_begin = (outer_dimensions_loop_index * (src_target_dimension_loop_size + 3 * 2)
		+ target_dimension_loop_index) * inner_dimensions_loop_size + inner_dimensions_loop_index;
	beacls::CudaStream* cudaStream = cudaStreams[dim];
	BoundaryCondition* boundaryCondition = grid->get_boundaryCondition(dim);
	const FLOAT_TYPE dxInv = dxInvs[dim];
	const FLOAT_TYPE dxInv_2 = dxInv_2s[dim];
	const FLOAT_TYPE dxInv_3 = dxInv_3s[dim];
	const FLOAT_TYPE dx_square = dx_squares[dim];
	const FLOAT_TYPE x2_dx_square = 2 * dx_square;
	const FLOAT_TYPE dx = dxs[dim];
	const size_t target_dimension_loop_size = target_dimension_loop_sizes[dim];
	const size_t loop_length = slice_length / first_dimension_loop_size;

	const size_t num_of_boundary_strides = std::max<size_t>(stencil * 2 + 1, stencil + num_of_merged_strides);

	std::vector<std::vector<std::vector<const FLOAT_TYPE*> > >& tmpBoundedSrc_ptrsss = tmpBoundedSrc_ptrssss[dim];

	if (tmpBoundedSrc_ptrsss.size() != num_of_merged_slices) tmpBoundedSrc_ptrsss.resize(num_of_merged_slices);
	for_each(tmpBoundedSrc_ptrsss.begin(), tmpBoundedSrc_ptrsss.end(), ([loop_length, num_of_boundary_strides](auto &rhs) {
		if (rhs.size() < loop_length) rhs.resize(loop_length);
		for_each(rhs.begin(), rhs.end(), ([num_of_boundary_strides](auto &rhs) {
			if (rhs.size() < num_of_boundary_strides) rhs.resize(num_of_boundary_strides);
		}));
	}));
	beacls::UVec& boundedSrc = tmpBoundedSrc_uvecs[dim%2];
	size_t total_boundedSrc_size = first_dimension_loop_size * num_of_boundary_strides * num_of_merged_slices * loop_length;
	if (boundedSrc.type() != type) boundedSrc = beacls::UVec(depth, type, total_boundedSrc_size);
	else if (boundedSrc.size() != total_boundedSrc_size) boundedSrc.resize(total_boundedSrc_size);
	boundedSrc.set_cudaStream(cudaStream);

	boundaryCondition->execute(
		src,
		boundedSrc, tmpBoundedSrc_uvec_vectors[dim%2], tmpBoundedSrc_ptrsss,
		stencil,
		dim,
		target_dimension_loop_size,
		inner_dimensions_loop_size,
		first_dimension_loop_size,
		margined_loop_begin,
		0,
		num_of_merged_slices,
		loop_length,
		num_of_boundary_strides,
		stencil
	);
	if (type == beacls::UVecType_Cuda) {
		FLOAT_TYPE* dst_deriv_l_ptr = beacls::UVec_<FLOAT_TYPE>(dst_deriv_l).ptr();
		FLOAT_TYPE* dst_deriv_r_ptr = beacls::UVec_<FLOAT_TYPE>(dst_deriv_r).ptr();
		dst_deriv_l.set_cudaStream(cudaStream);
		dst_deriv_r.set_cudaStream(cudaStream);
		// Need to figure out which approximation has the least oscillation.
		// Note that L and R in this section refer to neighboring divided
		// difference entries, not to left and right approximations.

		const FLOAT_TYPE weightL0 = weightL[0];
		const FLOAT_TYPE weightL1 = weightL[1];
		const FLOAT_TYPE weightL2 = weightL[2];
		const FLOAT_TYPE weightR0 = weightR[0];
		const FLOAT_TYPE weightR1 = weightR[1];
		const FLOAT_TYPE weightR2 = weightR[2];
		const size_t stride_distance = tmpBoundedSrc_ptrsss[0][0][1] - tmpBoundedSrc_ptrsss[0][0][0];
		UpwindFirstWENO5a_execute_dimLET2_cuda2(
			dst_deriv_l_ptr, dst_deriv_r_ptr,
			tmpBoundedSrc_ptrsss[0][0][0], 
			dxInv, dxInv_2, dxInv_3, dx, x2_dx_square, dx_square,
			weightL0, weightL1, weightL2, weightR0, weightR1, weightR2,
			num_of_slices, loop_length, first_dimension_loop_size, 
			stride_distance,
			slice_length,
			epsilonCalculationMethod_Type,
			cudaStream
		);
	}
	else
	{
		FLOAT_TYPE* dst_deriv_l_ptr = beacls::UVec_<FLOAT_TYPE>(dst_deriv_l).ptr();
		FLOAT_TYPE* dst_deriv_r_ptr = beacls::UVec_<FLOAT_TYPE>(dst_deriv_r).ptr();
		// Need to figure out which approximation has the least oscillation.
		// Note that L and R in this section refer to neighboring divided
		// difference entries, not to left and right approximations.

		const FLOAT_TYPE weightL0 = weightL[0];
		const FLOAT_TYPE weightL1 = weightL[1];
		const FLOAT_TYPE weightL2 = weightL[2];
		const FLOAT_TYPE weightR0 = weightR[0];
		const FLOAT_TYPE weightR1 = weightR[1];
		const FLOAT_TYPE weightR2 = weightR[2];
		switch (epsilonCalculationMethod_Type) {
		case levelset::EpsilonCalculationMethod_Invalid:
		default:
			printf("Unknown epsilonCalculationMethod %d\n", epsilonCalculationMethod_Type);
			return false;
		case levelset::EpsilonCalculationMethod_Constant:
			calc_deriv_dimLET2(
				dst_deriv_l_ptr, dst_deriv_r_ptr,
				tmpBoundedSrc_ptrsss,
				dx, dxInv, dxInv_2, dxInv_3, dx_square, x2_dx_square,
				weightL0, weightL1, weightL2, weightR0, weightR1, weightR2,
				slice_length, num_of_slices, loop_length, first_dimension_loop_size,
				dim, 
				ECM_Constant()
			);
			break;
		case levelset::EpsilonCalculationMethod_maxOverGrid:
			printf("epsilonCalculationMethod %d is not supported yet\n", epsilonCalculationMethod_Type);
			return false;
		case levelset::EpsilonCalculationMethod_maxOverNeighbor:
			calc_deriv_dimLET2(
				dst_deriv_l_ptr, dst_deriv_r_ptr,
				tmpBoundedSrc_ptrsss,
				dx, dxInv, dxInv_2, dxInv_3, dx_square, x2_dx_square,
				weightL0, weightL1, weightL2, weightR0, weightR1, weightR2,
				slice_length, num_of_slices, loop_length, first_dimension_loop_size,
				dim,
				ECM_MaxOverNeighbor()
			);
			break;
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
