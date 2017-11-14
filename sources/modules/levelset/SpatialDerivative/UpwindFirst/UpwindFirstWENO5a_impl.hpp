#ifndef __UpwindFirstWENO5a_impl_hpp__
#define __UpwindFirstWENO5a_impl_hpp__

#include <cstdint>
#include <vector>
#include <typedef.hpp>
#include <Core/UVec.hpp>

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
	T& ,
	T& ,
	T& ,
	T& ,
	T& ,
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


namespace levelset {
	class HJI_Grid;
	class UpwindFirstENO3aHelper;
	class UpwindFirstWENO5a_Cache {
	public:
		std::vector<beacls::FloatVec > last_d1ss;
		std::vector<beacls::FloatVec > last_d2ss;
		std::vector<beacls::FloatVec > last_d3ss;
		std::vector<beacls::FloatVec > last_dx_d2ss;
		std::vector<std::vector<beacls::FloatVec > > last_smooths;
		std::vector<const FLOAT_TYPE*> boundedSrc_ptrs;
		UpwindFirstWENO5a_Cache() : boundedSrc_ptrs(2), last_smooths(3) {}
		/** @overload
		Disable copy constructor
		*/
		UpwindFirstWENO5a_Cache(const UpwindFirstWENO5a_Cache& rhs) :
			last_d1ss(rhs.last_d1ss),
			last_d2ss(rhs.last_d2ss),
			last_d3ss(rhs.last_d3ss),
			last_dx_d2ss(rhs.last_dx_d2ss),
			last_smooths(rhs.last_smooths) {
			boundedSrc_ptrs.resize(rhs.boundedSrc_ptrs.size(), NULL);
		}
	};

	class UpwindFirstWENO5a_impl {
	private:
		beacls::UVecType type;
		std::vector<beacls::UVec> tmpBoundedSrc_uvec_vectors;
		beacls::FloatVec dxs;
		beacls::FloatVec dx_squares;
		beacls::FloatVec dxInvs;
		beacls::FloatVec dxInv_2s;
		beacls::FloatVec dxInv_3s;

		beacls::IntegerVec target_dimension_loop_sizes;
		beacls::IntegerVec inner_dimensions_loop_sizes;
		beacls::IntegerVec first_dimension_loop_sizes;
		beacls::IntegerVec src_target_dimension_loop_sizes;


		const size_t stencil;
		std::vector<beacls::UVec > tmpBoundedSrc_uvecs;
		std::vector<std::vector<std::vector<std::vector<const FLOAT_TYPE*> > > > tmpBoundedSrc_ptrssss;
		std::vector<FLOAT_TYPE*> tmp_d1s_ms_ites;
		std::vector<FLOAT_TYPE*> tmp_d2s_ms_ites;
		std::vector<FLOAT_TYPE*> tmp_d3s_ms_ites;
		std::vector<FLOAT_TYPE*> tmp_dx_d2s_ms_ites;
		std::vector<FLOAT_TYPE*> tmp_smooths1_ms_ites;
		std::vector<FLOAT_TYPE*> tmp_smooths2_ms_ites;
		std::vector<FLOAT_TYPE*> tmp_smooths3_ms_ites;
		size_t num_of_strides;

		std::vector<std::vector<beacls::UVec > > dL_uvecs;
		std::vector<std::vector<beacls::UVec > > dR_uvecs;
		std::vector<std::vector<beacls::UVec > > DD_uvecs;

		std::vector<beacls::FloatVec > d1ss;
		std::vector<beacls::FloatVec > d2ss;
		std::vector<beacls::FloatVec > d3ss;

		beacls::IntegerVec tmp_cache_indexes;

//		std::vector<beacls::FloatVec > tmpSmooths;

		const beacls::FloatVec weightL = { (FLOAT_TYPE)0.1, (FLOAT_TYPE)0.6,(FLOAT_TYPE)0.3 };
		const beacls::FloatVec weightR = { (FLOAT_TYPE)0.3, (FLOAT_TYPE)0.6,(FLOAT_TYPE)0.1 };

		levelset::EpsilonCalculationMethod_Type epsilonCalculationMethod_Type;
		UpwindFirstENO3aHelper* upwindFirstENO3aHelper;
		std::vector<beacls::CudaStream*> cudaStreams;
		UpwindFirstWENO5a_Cache *cache;
	public:
		UpwindFirstWENO5a_impl(
			const HJI_Grid *hji_grid,
			const beacls::UVecType type = beacls::UVecType_Vector
		);
		~UpwindFirstWENO5a_impl();

		bool execute_dim0(
			beacls::UVec& dst_deriv_l,
			beacls::UVec& dst_deriv_r,
			const FLOAT_TYPE* src,
			const HJI_Grid *grid,
			const size_t dim,
			const size_t loop_begin,
			const size_t slice_length,
			const size_t num_of_slices
		);
		bool execute_dim1(
			beacls::UVec& dst_deriv_l,
			beacls::UVec& dst_deriv_r,
			const FLOAT_TYPE* src,
			const HJI_Grid *grid,
			const size_t dim,
			const size_t loop_begin,
			const size_t slice_length,
			const size_t num_of_slices
		);
		bool execute_dimLET2(
			beacls::UVec& dst_deriv_l,
			beacls::UVec& dst_deriv_r,
			const FLOAT_TYPE* src,
			const HJI_Grid *grid,
			const size_t dim,
			const size_t loop_begin,
			const size_t slice_length,
			const size_t num_of_slices
		);
		bool execute(
			beacls::UVec& dst_deriv_l,
			beacls::UVec& dst_deriv_r,
			const HJI_Grid *grid,
			const FLOAT_TYPE* src,
			const size_t dim,
			const bool generateAll,
			const size_t loop_begin,
			const size_t slice_length,
			const size_t num_of_slices
		);
		bool synchronize(const size_t dim);
		bool operator==(const UpwindFirstWENO5a_impl& rhs) const;
		UpwindFirstWENO5a_impl* clone() const {
			return new UpwindFirstWENO5a_impl(*this);
		};
		beacls::UVecType get_type() const {
			return type;
		};
	private:
		void getCachePointer(
			std::vector<FLOAT_TYPE*> &dst_pointers,
			std::vector<beacls::FloatVec > &src_pointers,
			const size_t shifted_target_dimension_loop_index
		);
		void getCachePointers(
			std::vector<FLOAT_TYPE*> &d1s_ms,
			std::vector<FLOAT_TYPE*> &d2s_ms,
			std::vector<FLOAT_TYPE*> &d3s_ms,
			std::vector<FLOAT_TYPE*> &dx_d2ss,
			std::vector<FLOAT_TYPE*> &smooths1_ms,
			std::vector<FLOAT_TYPE*> &smooths2_ms,
			std::vector<FLOAT_TYPE*> &smooths3_ms,
			const size_t shifted_target_dimension_loop_index);
		void createCaches();


		/** @overload
		Disable operator=
		*/
		UpwindFirstWENO5a_impl& operator=(const UpwindFirstWENO5a_impl& rhs);
		/** @overload
		Disable copy constructor
		*/
		UpwindFirstWENO5a_impl(const UpwindFirstWENO5a_impl& rhs);
	};
};
#endif	/* __UpwindFirstWENO5a_impl_hpp__ */
