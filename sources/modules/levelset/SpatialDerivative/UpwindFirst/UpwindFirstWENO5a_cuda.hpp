#ifndef __UpwindFirstWENO5a_cuda_hpp__
#define __UpwindFirstWENO5a_cuda_hpp__
#include <typedef.hpp>
#include <cuda_macro.hpp>
#include <iostream>
namespace beacls {
	class CudaStream;
};


class ECM_Constant_Cuda {
};
class ECM_MaxOverNeighbor_Cuda {
};

template<typename T>
__device__
inline T get_epsilon_base(T a = 0);
template<>
__device__
inline double get_epsilon_base(double) {
	return (double)1e-99;
}
template<>
__device__
inline float get_epsilon_base(float) {
	return (float)1e-15;	//!< T.B.D. It should be larger than sqrt(FLT_MIN)
}

template<typename T, typename E>
__device__ inline
void calc_epsilon_dim0_cuda(
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

template<typename T>
__device__ inline
void calc_epsilon_dim0_cuda(
	T& epsilonL,
	T& epsilonR,
	T&,
	T&,
	T&,
	T&,
	T&,
	const T,
	ECM_Constant_Cuda
) {
	epsilonL = epsilonR = (T)1e-6;
}
template<typename T>
__device__ inline
void calc_epsilon_dim0_cuda(
	T& epsilonL,
	T& epsilonR,
	T& pow_D1_src_0,
	T& pow_D1_src_1,
	T& pow_D1_src_2,
	T& pow_D1_src_3,
	T& pow_D1_src_4,
	const T D1_src_5,
	ECM_MaxOverNeighbor_Cuda
) {
	const T pow_D1_src_5 = D1_src_5 * D1_src_5;
	const T max_1_2 = max_float_type<T>(pow_D1_src_1, pow_D1_src_2);
	const T max_3_4 = max_float_type<T>(pow_D1_src_3, pow_D1_src_4);
	const T max_1_2_3_4 = max_float_type<T>(max_1_2, max_3_4);
	const T maxOverNeighborD1squaredL = max_float_type<T>(max_1_2_3_4, pow_D1_src_0);
	const T maxOverNeighborD1squaredR = max_float_type<T>(max_1_2_3_4, pow_D1_src_5);
	epsilonL = (T)(1e-6 * maxOverNeighborD1squaredL + get_epsilon_base<T>());
	epsilonR = (T)(1e-6 * maxOverNeighborD1squaredR + get_epsilon_base<T>());
	pow_D1_src_0 = pow_D1_src_1;
	pow_D1_src_1 = pow_D1_src_2;
	pow_D1_src_2 = pow_D1_src_3;
	pow_D1_src_3 = pow_D1_src_4;
	pow_D1_src_4 = pow_D1_src_5;

}
template<typename T, typename E>
__device__ inline
void calc_epsilon_dimLET2_cuda(
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
template<typename T>
__device__ inline
void calc_epsilon_dimLET2_cuda(
	T& epsilonL,
	T& epsilonR,
	const T,
	const T,
	const T,
	const T,
	const T,
	const T,
	ECM_Constant_Cuda
) {
	epsilonL = epsilonR = (T)1e-6;
}
template<typename T>
__device__ inline
void calc_epsilon_dimLET2_cuda(
	T& epsilonL,
	T& epsilonR,
	const T D1_src_0,
	const T D1_src_1,
	const T D1_src_2,
	const T D1_src_3,
	const T D1_src_4,
	const T D1_src_5,
	ECM_MaxOverNeighbor_Cuda
) {
	const T pow_D1_src_0 = D1_src_0 * D1_src_0;
	const T pow_D1_src_1 = D1_src_1 * D1_src_1;
	const T pow_D1_src_2 = D1_src_2 * D1_src_2;
	const T pow_D1_src_3 = D1_src_3 * D1_src_3;
	const T pow_D1_src_4 = D1_src_4 * D1_src_4;
	const T pow_D1_src_5 = D1_src_5 * D1_src_5;
	const T max_1_2 = max_float_type<T>(pow_D1_src_1, pow_D1_src_2);
	const T max_3_4 = max_float_type<T>(pow_D1_src_3, pow_D1_src_4);
	const T max_1_2_3_4 = max_float_type<T>(max_1_2, max_3_4);
	const T maxOverNeighborD1squaredL = max_float_type<T>(max_1_2_3_4, pow_D1_src_0);
	const T maxOverNeighborD1squaredR = max_float_type<T>(max_1_2_3_4, pow_D1_src_5);
	epsilonL = (T)(1e-6 * maxOverNeighborD1squaredL + get_epsilon_base<T>());
	epsilonR = (T)(1e-6 * maxOverNeighborD1squaredR + get_epsilon_base<T>());
}



template<typename T> inline
__device__
T calcSmooth(
	const T src_1,
	const T src_2,
	const T src_3,
	const int b1,
	const int b2,
	const int b3
) {
	T a = src_1 - 2 * src_2 + src_3;
	T b = b1 * src_1 + b2 * src_2 + b3 * src_3;
	return (T)((13. / 12.) * (a * a)
		+ (1. / 4.) * (b * b));
}
template<typename T> inline
__device__
T calcSmooth0(
	const T src_1,
	const T src_2,
	const T src_3
) {
	return calcSmooth(src_1, src_2, src_3, 1, -4, 3);
}
template<typename T> inline
__device__
T calcSmooth1(
	const T src_1,
	const T src_2,
	const T src_3
) {
	return calcSmooth(src_1, src_2, src_3, 1, 0, -1);
}
template<typename T> inline
__device__
T calcSmooth2(
	const T src_1,
	const T src_2,
	const T src_3
) {
	return calcSmooth(src_1, src_2, src_3, 3, -4, 1);
}
template<typename T> inline
__device__
T weightWENO(
	const T d0,
	const T d1,
	const T d2,
	const T s0,
	const T s1,
	const T s2,
	const T w0,
	const T w1,
	const T w2,
	const T epsilon
) {
	T s0_epsilon = s0 + epsilon;
	T s1_epsilon = s1 + epsilon;
	T s2_epsilon = s2 + epsilon;
	T alpha1 = w0 / (s0_epsilon * s0_epsilon);
	T alpha2 = w1 / (s1_epsilon * s1_epsilon);
	T alpha3 = w2 / (s2_epsilon * s2_epsilon);
	T sum = alpha1 + alpha2 + alpha3;
	T alpha1_d0 = alpha1 * d0;
	T alpha2_d1 = alpha2 * d1;
	T alpha3_d2 = alpha3 * d2;
	T alpha_d = alpha1_d0 + alpha2_d1 + alpha3_d2;
	T result = alpha_d / sum;
	return result;
}

template<typename T, typename E>
__device__ inline
void kernel_dim0_EpsilonCalculationMethod_inline2(
	T* dst_deriv_l_ptr,
	T* dst_deriv_r_ptr,
	const T* boundedSrc_base_ptr,
	const T dxInv,
	const T dxInv_2,
	const T dxInv_3,
	const T dx,
	const T x2_dx_square,
	const T dx_square,
	const T weightL0,
	const T weightL1,
	const T weightL2,
	const T weightR0,
	const T weightR1,
	const T weightR2,
	const size_t num_of_slices,
	const size_t loop_length,
	const size_t src_target_dimension_loop_size,
	const size_t first_dimension_loop_size,
	const size_t slice_length,
	const size_t stencil,
	const size_t thread_length_x,
	const size_t thread_length_y,
	const size_t thread_length_z,
	const size_t blockIdx_x,
	const size_t blockIdx_y,
	const size_t blockIdx_z,
	const size_t blockDim_x,
	const size_t blockDim_y,
	const size_t blockDim_z,
	const size_t threadIdx_x,
	const size_t threadIdx_y,
	const size_t threadIdx_z,
	const E e) {
	const size_t global_thraed_index_z = blockIdx_z * blockDim_z + threadIdx_z;
	const size_t loop_index_z_base = global_thraed_index_z*thread_length_z;
	const size_t actual_thread_length_z = get_actual_length(num_of_slices, thread_length_z, loop_index_z_base);
	for (size_t loop_index_z = loop_index_z_base; loop_index_z < actual_thread_length_z + loop_index_z_base; ++loop_index_z) {
		const size_t dst_slice_offset = loop_index_z * slice_length;
		const size_t dst_slice_upperlimit = dst_slice_offset + slice_length;
		const size_t global_thraed_index_y = blockIdx_y * blockDim_y + threadIdx_y;
		const size_t loop_index_y_base = global_thraed_index_y*thread_length_y;
		const size_t actual_thread_length_y = get_actual_length(loop_length, thread_length_y, loop_index_y_base);
		for (size_t loop_index_y = loop_index_y_base; loop_index_y < actual_thread_length_y + loop_index_y_base; ++loop_index_y) {
			const size_t global_thread_index_x = blockIdx_x * blockDim_x + threadIdx_x;
			const size_t loop_index_x_base = global_thread_index_x * thread_length_x;
			const size_t actual_thread_length_x = get_actual_length(src_target_dimension_loop_size, thread_length_x, loop_index_x_base);
			if (actual_thread_length_x > 0) {
				const size_t dst_offset = loop_index_y * first_dimension_loop_size + dst_slice_offset;
				T D1_src_0 = 0;
				T D1_src_1 = 0;
				T D1_src_2 = 0;
				T D1_src_3 = 0;
				T D1_src_4 = 0;

				//! Target dimension is first loop
				const size_t loop_index_with_slice = loop_index_y + loop_index_z * loop_length;
				const size_t src_offset = (first_dimension_loop_size + stencil * 2) * loop_index_with_slice;
				const T* boundedSrc_ptr = boundedSrc_base_ptr + src_offset;
#if 0
				std::cout << 
					dst_offset << ", " <<
					src_offset << ", " <<
					loop_index_x_base << ", " <<
					actual_thread_length_x << std::endl;
#endif
				T d0_m0 = boundedSrc_ptr[loop_index_x_base];
				T d0_m1 = 0;
				T d1_m0 = 0;
				T d1_m1 = 0;
				T d1_m2 = 0;
				T d1_m3 = 0;
				T d2_m0 = 0;
				T d2_m1 = 0;
				T d2_m2 = 0;
				T d2_m3 = 0;
				T d3_m0 = 0;
				T d3_m1 = 0;
				T d3_m2 = 0;
				T d3_m3 = 0;
				T smooth_m1_0 = 0;
				T smooth_m1_1 = 0;
				T smooth_m1_2 = 0;
				T pow_D1_src_0 = 0;
				T pow_D1_src_1 = 0;
				T pow_D1_src_2 = 0;
				T pow_D1_src_3 = 0;
				T pow_D1_src_4 = 0;
				//! Prologue
				for (size_t target_dimension_loop_index = loop_index_x_base; target_dimension_loop_index < loop_index_x_base+ stencil + 2; ++target_dimension_loop_index) {
					size_t src_index = target_dimension_loop_index;
					d0_m1 = d0_m0;
					d0_m0 = boundedSrc_ptr[src_index + 1];
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
					pow_D1_src_0 = D1_src_0 * D1_src_0;
					pow_D1_src_1 = D1_src_1 * D1_src_1;
					pow_D1_src_2 = D1_src_2 * D1_src_2;
					pow_D1_src_3 = D1_src_3 * D1_src_3;
					pow_D1_src_4 = D1_src_4 * D1_src_4;
				}
				//! Body
				for (size_t loop_index_x = loop_index_x_base; loop_index_x < actual_thread_length_x + loop_index_x_base; ++loop_index_x) {
					d0_m1 = d0_m0;
					d0_m0 = boundedSrc_ptr[loop_index_x + stencil +3];
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

					const size_t dst_index = loop_index_x + dst_offset;
					T epsilonL, epsilonR;
					calc_epsilon_dim0_cuda(epsilonL, epsilonR, pow_D1_src_0, pow_D1_src_1, pow_D1_src_2, pow_D1_src_3, pow_D1_src_4, D1_src_5, e);
#if 1
					dst_deriv_l_ptr[dst_index] = weightWENO(dLL0, dLL1, dLL2, smooth_m1_0, smooth_m1_1, smooth_m1_2, weightL0, weightL1, weightL2, epsilonL);
					dst_deriv_r_ptr[dst_index] = weightWENO(dRR0, dRR1, dRR2, smooth_m0_0, smooth_m0_1, smooth_m0_2, weightR0, weightR1, weightR2, epsilonR);
#else
					if (dst_index < dst_slice_upperlimit) {
						dst_deriv_l_ptr[dst_index] = weightWENO(dLL0, dLL1, dLL2, smooth_m1_0, smooth_m1_1, smooth_m1_2, weightL0, weightL1, weightL2, epsilonL);
						dst_deriv_r_ptr[dst_index] = weightWENO(dRR0, dRR1, dRR2, smooth_m0_0, smooth_m0_1, smooth_m0_2, weightR0, weightR1, weightR2, epsilonR);
					}
					else {
						std::cout << std::endl;
					}
#endif
					smooth_m1_0 = smooth_m0_0;
					smooth_m1_1 = smooth_m0_1;
					smooth_m1_2 = smooth_m0_2;
					D1_src_1 = D1_src_2;
					D1_src_2 = D1_src_3;
					D1_src_3 = D1_src_4;
					D1_src_4 = D1_src_5;
				}
			}
		}
	}
}

void UpwindFirstWENO5a_execute_dim0_cuda2
(
	FLOAT_TYPE* dst_deriv_l_ptr,
	FLOAT_TYPE* dst_deriv_r_ptr,
	const FLOAT_TYPE* boundedSrc_base_ptr,
	const FLOAT_TYPE dxInv,
	const FLOAT_TYPE dxInv_2,
	const FLOAT_TYPE dxInv_3,
	const FLOAT_TYPE dx,
	const FLOAT_TYPE x2_dx_square,
	const FLOAT_TYPE dx_square,
	const FLOAT_TYPE weightL0,
	const FLOAT_TYPE weightL1,
	const FLOAT_TYPE weightL2,
	const FLOAT_TYPE weightR0,
	const FLOAT_TYPE weightR1,
	const FLOAT_TYPE weightR2,
	const size_t num_of_slices,
	const size_t loop_length,
	const size_t src_target_dimension_loop_size,
	const size_t first_dimension_loop_size,
	const size_t slice_length,
	const size_t stencil,
	const levelset::EpsilonCalculationMethod_Type epsilonCalculationMethod_Type,
	beacls::CudaStream* cudaStream
);


template<typename T, typename E>
__device__ inline
void kernel_dim1_EpsilonCalculationMethod_inline2(
	T* dst_deriv_l_ptr,
	T* dst_deriv_r_ptr,
	const T* boundedSrc_base_ptr,
	const T dxInv,
	const T dxInv_2,
	const T dxInv_3,
	const T dx,
	const T x2_dx_square,
	const T dx_square,
	const T weightL0,
	const T weightL1,
	const T weightL2,
	const T weightR0,
	const T weightR1,
	const T weightR2,
	const size_t num_of_slices,
	const size_t loop_length,
	const size_t first_dimension_loop_size,
	const size_t slice_length,
	const size_t stencil,
	const size_t thread_length_x,
	const size_t thread_length_y,
	const size_t thread_length_z,
	const size_t blockIdx_x,
	const size_t blockIdx_y,
	const size_t blockIdx_z,
	const size_t blockDim_x,
	const size_t blockDim_y,
	const size_t blockDim_z,
	const size_t threadIdx_x,
	const size_t threadIdx_y,
	const size_t threadIdx_z,
	const E e) {
	const size_t global_thraed_index_z = blockIdx_z * blockDim_z + threadIdx_z;
	const size_t loop_index_z_base = global_thraed_index_z*thread_length_z;
	const size_t actual_thread_length_z = get_actual_length(num_of_slices, thread_length_z, loop_index_z_base);
	const size_t prologue_length = stencil * 2;
	const size_t src_loop_length = (loop_length + prologue_length);
	for (size_t loop_index_z = loop_index_z_base; loop_index_z < actual_thread_length_z + loop_index_z_base; ++loop_index_z) {
		const size_t dst_slice_offset = loop_index_z * slice_length;
		const size_t dst_slice_upperlimit = dst_slice_offset + slice_length;
		const size_t global_thraed_index_y = blockIdx_y * blockDim_y + threadIdx_y;
		const size_t loop_index_y_base = global_thraed_index_y*thread_length_y;
		const size_t actual_thread_length_y = get_actual_length(loop_length, thread_length_y, loop_index_y_base);
		if (actual_thread_length_y > 0) {
			const size_t global_thread_index_x = blockIdx_x * blockDim_x + threadIdx_x;
#if 1
			const size_t loop_index_x_base = global_thread_index_x;
			for (size_t fdl_index = loop_index_x_base; fdl_index < first_dimension_loop_size; fdl_index += blockDim_x) {
#else
			const size_t loop_index_x_base = global_thread_index_x * thread_length_x;
			const size_t actual_thread_length_x = get_actual_length(first_dimension_loop_size, thread_length_x, loop_index_x_base);
			for (size_t fdl_index = loop_index_x_base; fdl_index < actual_thread_length_x + loop_index_x_base; ++fdl_index) {
#endif
				T d0_m0 = -1;
				T d0_m1 = -1;
				T d1_m0 = -1;
				T d1_m1 = -1;
				T d1_m2 = -1;
				T d1_m3 = -1;
				T d1_m4 = -1;
				T d2_m0 = -1;
				T d2_m1 = -1;
				T d3_m0 = -1;
				T d3_m1 = -1;
				T d3_m2 = -1;
				T d3_m3 = -1;

				//Prologue
				T smooth_m1_0 = -1;
				T smooth_m1_1 = -1;
				T smooth_m1_2 = -1;
				T pow_D1_src_0 = -1;
				T pow_D1_src_1 = -1;
				T pow_D1_src_2 = -1;
				T pow_D1_src_3 = -1;
				T pow_D1_src_4 = -1;
				T dx_d2_m1 = -1;
				T dx_d2_m2 = -1;
				T dx_d2_m3 = -1;
				if (loop_index_y_base == 0) {
					for (size_t prologue_index = 0; prologue_index < prologue_length; ++prologue_index) {
						const T* tmpBoundedSrc_ptrs0 = boundedSrc_base_ptr +
							(loop_index_z*src_loop_length + prologue_index + loop_index_y_base)*first_dimension_loop_size;
						d0_m1 = d0_m0;
						d0_m0 = tmpBoundedSrc_ptrs0[fdl_index];
						if (prologue_index >= 1) {
							d1_m4 = d1_m3;
							d1_m3 = d1_m2;
							d1_m2 = d1_m1;
							d1_m1 = d1_m0;
							d1_m0 = dxInv * (d0_m0 - d0_m1);
							if (prologue_index >= 2) {
								d2_m1 = d2_m0;
								d2_m0 = dxInv_2 * (d1_m0 - d1_m1);
								if (prologue_index >= 3) {
									d3_m3 = d3_m2;
									d3_m2 = d3_m1;
									d3_m1 = d3_m0;
									d3_m0 = dxInv_3 * (d2_m0 - d2_m1);
									dx_d2_m3 = dx_d2_m2;
									dx_d2_m2 = dx_d2_m1;
									dx_d2_m1 = dx * d2_m1;
									if (prologue_index >= stencil * 2 - 1) {
										const T D1_src_0 = d1_m4;
										const T D1_src_1 = d1_m3;
										const T D1_src_2 = d1_m2;
										const T D1_src_3 = d1_m1;
										const T D1_src_4 = d1_m0;

										smooth_m1_0 = calcSmooth0(D1_src_0, D1_src_1, D1_src_2);
										smooth_m1_1 = calcSmooth1(D1_src_1, D1_src_2, D1_src_3);
										smooth_m1_2 = calcSmooth2(D1_src_2, D1_src_3, D1_src_4);
										pow_D1_src_0 = D1_src_0 * D1_src_0;
										pow_D1_src_1 = D1_src_1 * D1_src_1;
										pow_D1_src_2 = D1_src_2 * D1_src_2;
										pow_D1_src_3 = D1_src_3 * D1_src_3;
										pow_D1_src_4 = D1_src_4 * D1_src_4;
									}
								}
							}
						}
					}
				}
				else {
					for (size_t prologue_index = 0; prologue_index < prologue_length; ++prologue_index) {
						const T* tmpBoundedSrc_ptrs0 = boundedSrc_base_ptr +
							(loop_index_z*src_loop_length + prologue_index + loop_index_y_base)*first_dimension_loop_size;
						d0_m1 = d0_m0;
						d0_m0 = tmpBoundedSrc_ptrs0[fdl_index];
						d1_m4 = d1_m3;
						d1_m3 = d1_m2;
						d1_m2 = d1_m1;
						d1_m1 = d1_m0;
						d1_m0 = dxInv * (d0_m0 - d0_m1);
						d2_m1 = d2_m0;
						d2_m0 = dxInv_2 * (d1_m0 - d1_m1);
						d3_m3 = d3_m2;
						d3_m2 = d3_m1;
						d3_m1 = d3_m0;
						d3_m0 = dxInv_3 * (d2_m0 - d2_m1);
						dx_d2_m3 = dx_d2_m2;
						dx_d2_m2 = dx_d2_m1;
						dx_d2_m1 = dx * d2_m1;
						const T D1_src_0 = d1_m4;
						const T D1_src_1 = d1_m3;
						const T D1_src_2 = d1_m2;
						const T D1_src_3 = d1_m1;
						const T D1_src_4 = d1_m0;

						smooth_m1_0 = calcSmooth0(D1_src_0, D1_src_1, D1_src_2);
						smooth_m1_1 = calcSmooth1(D1_src_1, D1_src_2, D1_src_3);
						smooth_m1_2 = calcSmooth2(D1_src_2, D1_src_3, D1_src_4);
						pow_D1_src_0 = D1_src_0 * D1_src_0;
						pow_D1_src_1 = D1_src_1 * D1_src_1;
						pow_D1_src_2 = D1_src_2 * D1_src_2;
						pow_D1_src_3 = D1_src_3 * D1_src_3;
						pow_D1_src_4 = D1_src_4 * D1_src_4;
					}
				}
				// Need to figure out which approximation has the least oscillation.
				// Note that L and R in this section refer to neighboring divided
				// difference entries, not to left and right approximations.
				for (size_t loop_index_y = loop_index_y_base; loop_index_y < actual_thread_length_y+ loop_index_y_base; ++loop_index_y) {
					const size_t dst_offset = loop_index_y * first_dimension_loop_size + dst_slice_offset;
					const T* tmpBoundedSrc_ptrs0 = boundedSrc_base_ptr +
						(loop_index_z*src_loop_length + prologue_length + loop_index_y)*first_dimension_loop_size;
					d0_m1 = d0_m0;
					d1_m4 = d1_m3;
					d1_m3 = d1_m2;
					d1_m2 = d1_m1;
					d1_m1 = d1_m0;
					d2_m1 = d2_m0;
					d3_m3 = d3_m2;
					d3_m2 = d3_m1;
					d3_m1 = d3_m0;

					d0_m0 = tmpBoundedSrc_ptrs0[fdl_index];
					d1_m0 = dxInv * (d0_m0 - d0_m1);
					d2_m0 = dxInv_2 * (d1_m0 - d1_m1);
					d3_m0 = dxInv_3 * (d2_m0 - d2_m1);

					const T D1_src_1 = d1_m4;
					const T D1_src_2 = d1_m3;
					const T D1_src_3 = d1_m2;
					const T D1_src_4 = d1_m1;
					const T D1_src_5 = d1_m0;

					dx_d2_m3 = dx_d2_m2;
					dx_d2_m2 = dx_d2_m1;
					dx_d2_m1 = dx * d2_m1;

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


					const T smooth_m0_0 = calcSmooth0(D1_src_1, D1_src_2, D1_src_3);
					const T smooth_m0_1 = calcSmooth1(D1_src_2, D1_src_3, D1_src_4);
					const T smooth_m0_2 = calcSmooth2(D1_src_3, D1_src_4, D1_src_5);
					T epsilonL, epsilonR;
					calc_epsilon_dim0_cuda(epsilonL, epsilonR, pow_D1_src_0, pow_D1_src_1, pow_D1_src_2, pow_D1_src_3, pow_D1_src_4, D1_src_5, e);
					const size_t dst_index = fdl_index + dst_offset;
#if 1
					dst_deriv_l_ptr[dst_index] = weightWENO(dLL0, dLL1, dLL2, smooth_m1_0, smooth_m1_1, smooth_m1_2, weightL0, weightL1, weightL2, epsilonL);
					dst_deriv_r_ptr[dst_index] = weightWENO(dRR0, dRR1, dRR2, smooth_m0_0, smooth_m0_1, smooth_m0_2, weightR0, weightR1, weightR2, epsilonR);
#else
					if (dst_index < dst_slice_upperlimit) {
						dst_deriv_l_ptr[dst_index] = weightWENO(dLL0, dLL1, dLL2, smooth_m1_0, smooth_m1_1, smooth_m1_2, weightL0, weightL1, weightL2, epsilonL);
						dst_deriv_r_ptr[dst_index] = weightWENO(dRR0, dRR1, dRR2, smooth_m0_0, smooth_m0_1, smooth_m0_2, weightR0, weightR1, weightR2, epsilonR);
					}
					else {
						std::cout << std::endl;
					}
#endif
					smooth_m1_0 = smooth_m0_0;
					smooth_m1_1 = smooth_m0_1;
					smooth_m1_2 = smooth_m0_2;
				}
			}
		}
	}
}

void UpwindFirstWENO5a_execute_dim1_cuda2
(
	FLOAT_TYPE* dst_deriv_l_ptr,
	FLOAT_TYPE* dst_deriv_r_ptr,
	const FLOAT_TYPE* tmpBoundedSrc_ptr,
	const FLOAT_TYPE dxInv,
	const FLOAT_TYPE dxInv_2,
	const FLOAT_TYPE dxInv_3,
	const FLOAT_TYPE dx,
	const FLOAT_TYPE x2_dx_square,
	const FLOAT_TYPE dx_square,
	const FLOAT_TYPE weightL0,
	const FLOAT_TYPE weightL1,
	const FLOAT_TYPE weightL2,
	const FLOAT_TYPE weightR0,
	const FLOAT_TYPE weightR1,
	const FLOAT_TYPE weightR2,
	const size_t num_of_slices,
	const size_t loop_length,
	const size_t first_dimension_loop_size,
	const size_t slice_length,
	const size_t stencil,
	const levelset::EpsilonCalculationMethod_Type epsilonCalculationMethod_Type,
	beacls::CudaStream* cudaStream

);

template<typename T, typename E>
__device__ inline
void kernel_dimLET2_EpsilonCalculationMethod_inline2(
	T* dst_deriv_l_ptr,
	T* dst_deriv_r_ptr,
	const T* tmpBoundedSrc_ptr,
	const T dxInv,
	const T dxInv_2,
	const T dxInv_3,
	const T dx,
	const T x2_dx_square,
	const T dx_square,
	const T weightL0,
	const T weightL1,
	const T weightL2,
	const T weightR0,
	const T weightR1,
	const T weightR2,
	const size_t num_of_slices,
	const size_t loop_length,
	const size_t first_dimension_loop_size,
	const size_t stride_distance,
	const size_t slice_length,
	const size_t thread_length_z,
	const size_t thread_length_y,
	const size_t thread_length_x,
	const size_t blockIdx_y,
	const size_t blockIdx_x,
	const size_t blockDim_y,
	const size_t blockDim_x,
	const size_t threadIdx_z,
	const size_t threadIdx_y,
	const size_t threadIdx_x,
	const E e
) {
	const size_t global_thraed_index_z = threadIdx_z;
	const size_t loop_index_z_base = global_thraed_index_z*thread_length_z;
	const size_t actual_thread_length_z = get_actual_length(num_of_slices, thread_length_z, loop_index_z_base);
	for (size_t loop_index_z = loop_index_z_base; loop_index_z < actual_thread_length_z + loop_index_z_base; ++loop_index_z) {
		const size_t dst_slice_offset = loop_index_z * slice_length;
		const size_t global_thraed_index_y = blockIdx_y * blockDim_y + threadIdx_y;
		const size_t loop_index_y_base = global_thraed_index_y*thread_length_y;
		const size_t actual_thread_length_y = get_actual_length(loop_length, thread_length_y, loop_index_y_base);
		for (size_t loop_index_y = loop_index_y_base; loop_index_y < actual_thread_length_y + loop_index_y_base; ++loop_index_y) {
			const size_t line_offset = loop_index_y * first_dimension_loop_size;
			const size_t dst_offset = line_offset + dst_slice_offset;
			const size_t global_thread_index_x = blockIdx_x * blockDim_x + threadIdx_x;

			const size_t src_stride_offset = loop_index_z * slice_length + line_offset;
			const T* tmpBoundedSrc_ptrs0 = tmpBoundedSrc_ptr + src_stride_offset;
			const T* tmpBoundedSrc_ptrs1 = tmpBoundedSrc_ptrs0 + stride_distance;
			const T* tmpBoundedSrc_ptrs2 = tmpBoundedSrc_ptrs1 + stride_distance;
			const T* tmpBoundedSrc_ptrs3 = tmpBoundedSrc_ptrs2 + stride_distance;
			const T* tmpBoundedSrc_ptrs4 = tmpBoundedSrc_ptrs3 + stride_distance;
			const T* tmpBoundedSrc_ptrs5 = tmpBoundedSrc_ptrs4 + stride_distance;
			const T* tmpBoundedSrc_ptrs6 = tmpBoundedSrc_ptrs5 + stride_distance;
#if 0
			for (size_t loop_index = global_thread_index_x; loop_index < first_dimension_loop_size; loop_index+= blockDim_x) {
				const size_t first_dimension_loop_index = loop_index;
#else
			const size_t loop_index_x_base = global_thread_index_x * thread_length_x;
			const size_t actual_thread_length_x = get_actual_length(first_dimension_loop_size, thread_length_x, loop_index_x_base);
			for (size_t loop_index = 0; loop_index < actual_thread_length_x; ++loop_index) {
				const size_t first_dimension_loop_index = loop_index + loop_index_x_base;
#endif

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
				T epsilonL, epsilonR;
				calc_epsilon_dimLET2_cuda(epsilonL, epsilonR, D1_src_0, D1_src_1, D1_src_2, D1_src_3, D1_src_4, D1_src_5, e);
				const size_t dst_index = first_dimension_loop_index + dst_offset;
				dst_deriv_l_ptr[dst_index] = weightWENO(dLL0, dLL1, dLL2, smoothL_0, smoothL_1, smoothL_2, weightL0, weightL1, weightL2, epsilonL);
				dst_deriv_r_ptr[dst_index] = weightWENO(dRR0, dRR1, dRR2, smoothR_0, smoothR_1, smoothR_2, weightR0, weightR1, weightR2, epsilonR);
			}
		}
	}
}
void UpwindFirstWENO5a_execute_dimLET2_cuda2
(
	FLOAT_TYPE* dst_deriv_l_ptr,
	FLOAT_TYPE* dst_deriv_r_ptr,
	const FLOAT_TYPE* tmpBoundedSrc_ptr,
	const FLOAT_TYPE dxInv,
	const FLOAT_TYPE dxInv_2,
	const FLOAT_TYPE dxInv_3,
	const FLOAT_TYPE dx,
	const FLOAT_TYPE x2_dx_square,
	const FLOAT_TYPE dx_square,
	const FLOAT_TYPE weightL0,
	const FLOAT_TYPE weightL1,
	const FLOAT_TYPE weightL2,
	const FLOAT_TYPE weightR0,
	const FLOAT_TYPE weightR1,
	const FLOAT_TYPE weightR2,
	const size_t num_of_slices,
	const size_t loop_length,
	const size_t first_dimension_loop_size,
	const size_t stride_distance,
	const size_t slice_length,
	const levelset::EpsilonCalculationMethod_Type epsilonCalculationMethod_Type,
	beacls::CudaStream* cudaStream

);

#endif	/* __UpwindFirstWENO5a_cuda_hpp__ */
