#ifndef __UpwindFirstWENO5a_cuda_hpp__
#define __UpwindFirstWENO5a_cuda_hpp__
#include <typedef.hpp>
#include <cuda_macro.hpp>

static const size_t max_num_of_threads = 1024;

namespace beacls {
	class CudaStream;
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
__device__ inline
void kernel_dim0_EpsilonCalculationMethod_Constant_inline(
	FLOAT_TYPE* dst_deriv_l_ptr,
	FLOAT_TYPE* dst_deriv_r_ptr,
	const FLOAT_TYPE* DD0_ptr,
	const FLOAT_TYPE* dL0_ptr,
	const FLOAT_TYPE* dL1_ptr,
	const FLOAT_TYPE* dL2_ptr,
	const FLOAT_TYPE* dR0_ptr,
	const FLOAT_TYPE* dR1_ptr,
	const FLOAT_TYPE* dR2_ptr,
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
	const size_t thread_length_z,
	const size_t thread_length_y,
	const size_t thread_length_x,
	const size_t blockIdx_y,
	const size_t blockIdx_x,
	const size_t blockDim_y,
	const size_t blockDim_x,
	const size_t threadIdx_z,
	const size_t threadIdx_y,
	const size_t threadIdx_x) {
	const size_t global_thraed_index_z = threadIdx_z;
	const size_t loop_index_z_base = global_thraed_index_z*thread_length_z;
	const size_t actual_thread_length_z = get_actual_length(num_of_slices, thread_length_z, loop_index_z_base);
	for (size_t loop_index_z = loop_index_z_base; loop_index_z < actual_thread_length_z + loop_index_z_base; ++loop_index_z) {
		const size_t dst_slice_offset = loop_index_z * slice_length;
		const size_t dst_slice_upperlimit = dst_slice_offset + slice_length;
		const size_t src_DD0_slice_offset = loop_index_z * loop_length * (first_dimension_loop_size + 5);
		const size_t global_thraed_index_y = blockIdx_y * blockDim_y + threadIdx_y;
		const size_t loop_index_y_base = global_thraed_index_y*thread_length_y;
		const size_t actual_thread_length_y = get_actual_length(loop_length, thread_length_y, loop_index_y_base);
		for (size_t loop_index_y = loop_index_y_base; loop_index_y < actual_thread_length_y + loop_index_y_base; ++loop_index_y) {
			const size_t global_thread_index_x = blockIdx_x * blockDim_x + threadIdx_x;
			const size_t loop_index_x_base = global_thread_index_x * thread_length_x;
			const size_t actual_thread_length_x = get_actual_length(src_target_dimension_loop_size, thread_length_x, loop_index_x_base);
			if (actual_thread_length_x > 0) {
				const size_t dst_offset = loop_index_y * first_dimension_loop_size + dst_slice_offset;
				const size_t src_D1_offset = loop_index_y * (first_dimension_loop_size + 5) + src_DD0_slice_offset;
				FLOAT_TYPE smooth_m1_0;
				FLOAT_TYPE smooth_m1_1;
				FLOAT_TYPE smooth_m1_2;
				FLOAT_TYPE D1_src_1;
				FLOAT_TYPE D1_src_2;
				FLOAT_TYPE D1_src_3;
				FLOAT_TYPE D1_src_4;
				//Prologue
				{
					const size_t loop_index_x = loop_index_x_base;
					const size_t src_D1_index = loop_index_x + src_D1_offset;
					const FLOAT_TYPE D1_src_0 = DD0_ptr[src_D1_index];
					D1_src_1 = DD0_ptr[src_D1_index + 1];
					D1_src_2 = DD0_ptr[src_D1_index + 2];
					D1_src_3 = DD0_ptr[src_D1_index + 3];
					D1_src_4 = DD0_ptr[src_D1_index + 4];
					smooth_m1_0 = calcSmooth0(D1_src_0, D1_src_1, D1_src_2);
					smooth_m1_1 = calcSmooth1(D1_src_1, D1_src_2, D1_src_3);
					smooth_m1_2 = calcSmooth2(D1_src_2, D1_src_3, D1_src_4);
				}
				for (size_t loop_index_x = loop_index_x_base; loop_index_x < actual_thread_length_x + loop_index_x_base; ++loop_index_x) {
					const size_t src_D1_index = loop_index_x + src_D1_offset;
					const size_t dst_index = loop_index_x + dst_offset;
					const FLOAT_TYPE D1_src_5 = DD0_ptr[src_D1_index + 5];
					const FLOAT_TYPE smooth_m0_0 = calcSmooth0(D1_src_1, D1_src_2, D1_src_3);
					const FLOAT_TYPE smooth_m0_1 = calcSmooth1(D1_src_2, D1_src_3, D1_src_4);
					const FLOAT_TYPE smooth_m0_2 = calcSmooth2(D1_src_3, D1_src_4, D1_src_5);
					const FLOAT_TYPE epsilon = (FLOAT_TYPE)1e-6;
					const size_t src_index = dst_index;
					if (dst_index < dst_slice_upperlimit) {
						dst_deriv_l_ptr[dst_index] = weightWENO(dL0_ptr[src_index], dL1_ptr[src_index], dL2_ptr[src_index], smooth_m1_0, smooth_m1_1, smooth_m1_2, weightL0, weightL1, weightL2, epsilon);
						dst_deriv_r_ptr[dst_index] = weightWENO(dR0_ptr[src_index], dR1_ptr[src_index], dR2_ptr[src_index], smooth_m0_0, smooth_m0_1, smooth_m0_2, weightR0, weightR1, weightR2, epsilon);
					}
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

__device__ inline
void kernel_dim0_EpsilonCalculationMethod_maxOverNeighbor_inline(
	FLOAT_TYPE* dst_deriv_l_ptr,
	FLOAT_TYPE* dst_deriv_r_ptr,
	const FLOAT_TYPE* DD0_ptr,
	const FLOAT_TYPE* dL0_ptr,
	const FLOAT_TYPE* dL1_ptr,
	const FLOAT_TYPE* dL2_ptr,
	const FLOAT_TYPE* dR0_ptr,
	const FLOAT_TYPE* dR1_ptr,
	const FLOAT_TYPE* dR2_ptr,
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
	const size_t thread_length_z,
	const size_t thread_length_y,
	const size_t thread_length_x,
	const size_t blockIdx_y,
	const size_t blockIdx_x,
	const size_t blockDim_y,
	const size_t blockDim_x,
	const size_t threadIdx_z,
	const size_t threadIdx_y,
	const size_t threadIdx_x) {
	const size_t global_thraed_index_z = threadIdx_z;
	const size_t loop_index_z_base = global_thraed_index_z*thread_length_z;
	const size_t actual_thread_length_z = get_actual_length(num_of_slices, thread_length_z, loop_index_z_base);
	for (size_t loop_index_z = loop_index_z_base; loop_index_z < actual_thread_length_z + loop_index_z_base; ++loop_index_z) {
		const size_t dst_slice_offset = loop_index_z * slice_length;
		const size_t dst_slice_upperlimit = dst_slice_offset + slice_length;
		const size_t src_DD0_slice_offset = loop_index_z * loop_length * (first_dimension_loop_size+5);
		const size_t global_thraed_index_y = blockIdx_y * blockDim_y + threadIdx_y;
		const size_t loop_index_y_base = global_thraed_index_y*thread_length_y;
		const size_t actual_thread_length_y = get_actual_length(loop_length, thread_length_y, loop_index_y_base);
		for (size_t loop_index_y = loop_index_y_base; loop_index_y < actual_thread_length_y + loop_index_y_base; ++loop_index_y) {
			const size_t global_thread_index_x = blockIdx_x * blockDim_x + threadIdx_x;
			const size_t loop_index_x_base = global_thread_index_x * thread_length_x;
			const size_t actual_thread_length_x = get_actual_length(src_target_dimension_loop_size, thread_length_x, loop_index_x_base);
			if (actual_thread_length_x > 0) {
				const size_t dst_offset = loop_index_y * first_dimension_loop_size + dst_slice_offset;
				const size_t src_D1_offset = loop_index_y * (first_dimension_loop_size + 5) + src_DD0_slice_offset;
				FLOAT_TYPE smooth_m1_0;
				FLOAT_TYPE smooth_m1_1;
				FLOAT_TYPE smooth_m1_2;
				FLOAT_TYPE D1_src_1;
				FLOAT_TYPE D1_src_2;
				FLOAT_TYPE D1_src_3;
				FLOAT_TYPE D1_src_4;
				FLOAT_TYPE pow_D1_src_0;
				FLOAT_TYPE pow_D1_src_1;
				FLOAT_TYPE pow_D1_src_2;
				FLOAT_TYPE pow_D1_src_3;
				FLOAT_TYPE pow_D1_src_4;
				//Prologue
				{
					const size_t loop_index_x = loop_index_x_base;
					const size_t src_D1_index = loop_index_x + src_D1_offset;
					const FLOAT_TYPE D1_src_0 = DD0_ptr[src_D1_index];
					D1_src_1 = DD0_ptr[src_D1_index + 1];
					D1_src_2 = DD0_ptr[src_D1_index + 2];
					D1_src_3 = DD0_ptr[src_D1_index + 3];
					D1_src_4 = DD0_ptr[src_D1_index + 4];
					smooth_m1_0 = calcSmooth0(D1_src_0, D1_src_1, D1_src_2);
					smooth_m1_1 = calcSmooth1(D1_src_1, D1_src_2, D1_src_3);
					smooth_m1_2 = calcSmooth2(D1_src_2, D1_src_3, D1_src_4);
					pow_D1_src_0 = D1_src_0 * D1_src_0;
					pow_D1_src_1 = D1_src_1 * D1_src_1;
					pow_D1_src_2 = D1_src_2 * D1_src_2;
					pow_D1_src_3 = D1_src_3 * D1_src_3;
					pow_D1_src_4 = D1_src_4 * D1_src_4;
				}
				for (size_t loop_index_x = loop_index_x_base; loop_index_x < actual_thread_length_x + loop_index_x_base; ++loop_index_x) {
					const size_t src_D1_index = loop_index_x + src_D1_offset;
					const size_t dst_index = loop_index_x + dst_offset;
					const FLOAT_TYPE D1_src_5 = DD0_ptr[src_D1_index + 5];
					const FLOAT_TYPE smooth_m0_0 = calcSmooth0(D1_src_1, D1_src_2, D1_src_3);
					const FLOAT_TYPE smooth_m0_1 = calcSmooth1(D1_src_2, D1_src_3, D1_src_4);
					const FLOAT_TYPE smooth_m0_2 = calcSmooth2(D1_src_3, D1_src_4, D1_src_5);
					const FLOAT_TYPE pow_D1_src_5 = D1_src_5 * D1_src_5;
					const FLOAT_TYPE max_1_2 = max_float_type(pow_D1_src_1, pow_D1_src_2);
					const FLOAT_TYPE max_3_4 = max_float_type(pow_D1_src_3, pow_D1_src_4);
					const FLOAT_TYPE max_1_2_3_4 = max_float_type(max_1_2, max_3_4);
					const FLOAT_TYPE maxOverNeighborD1squaredL = max_float_type(max_1_2_3_4, pow_D1_src_0);
					const FLOAT_TYPE maxOverNeighborD1squaredR = max_float_type(max_1_2_3_4, pow_D1_src_5);
					const FLOAT_TYPE epsilonL = (FLOAT_TYPE)(1e-6 * maxOverNeighborD1squaredL + get_epsilon_base<FLOAT_TYPE>());
					const FLOAT_TYPE epsilonR = (FLOAT_TYPE)(1e-6 * maxOverNeighborD1squaredR + get_epsilon_base<FLOAT_TYPE>());
					const size_t src_index = dst_index;
					if (dst_index < dst_slice_upperlimit) {
						dst_deriv_l_ptr[dst_index] = weightWENO(dL0_ptr[src_index], dL1_ptr[src_index], dL2_ptr[src_index], smooth_m1_0, smooth_m1_1, smooth_m1_2, weightL0, weightL1, weightL2, epsilonL);
						dst_deriv_r_ptr[dst_index] = weightWENO(dR0_ptr[src_index], dR1_ptr[src_index], dR2_ptr[src_index], smooth_m0_0, smooth_m0_1, smooth_m0_2, weightR0, weightR1, weightR2, epsilonR);
					}
					smooth_m1_0 = smooth_m0_0;
					smooth_m1_1 = smooth_m0_1;
					smooth_m1_2 = smooth_m0_2;
					D1_src_1 = D1_src_2;
					D1_src_2 = D1_src_3;
					D1_src_3 = D1_src_4;
					D1_src_4 = D1_src_5;
					pow_D1_src_0 = pow_D1_src_1;
					pow_D1_src_1 = pow_D1_src_2;
					pow_D1_src_2 = pow_D1_src_3;
					pow_D1_src_3 = pow_D1_src_4;
					pow_D1_src_4 = pow_D1_src_5;
				}
			}
		}
	}
}

void UpwindFirstWENO5a_execute_dim0_cuda
(
	FLOAT_TYPE* dst_deriv_l_ptr,
	FLOAT_TYPE* dst_deriv_r_ptr,
	const FLOAT_TYPE* DD0_0_ptr,
	const FLOAT_TYPE* dL0_ptr,
	const FLOAT_TYPE* dL1_ptr,
	const FLOAT_TYPE* dL2_ptr,
	const FLOAT_TYPE* dR0_ptr,
	const FLOAT_TYPE* dR1_ptr,
	const FLOAT_TYPE* dR2_ptr,
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
	const levelset::EpsilonCalculationMethod_Type epsilonCalculationMethod_Type,
	beacls::CudaStream* cudaStream
);

__device__ inline
void kernel_dim1_EpsilonCalculationMethod_Constant_inline(
	FLOAT_TYPE* dst_deriv_l_ptr,
	FLOAT_TYPE* dst_deriv_r_ptr,
	const FLOAT_TYPE* DD0_ptr,
	const FLOAT_TYPE* dL0_ptr,
	const FLOAT_TYPE* dL1_ptr,
	const FLOAT_TYPE* dL2_ptr,
	const FLOAT_TYPE* dR0_ptr,
	const FLOAT_TYPE* dR1_ptr,
	const FLOAT_TYPE* dR2_ptr,
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
	const size_t DD0_slice_size,
	const size_t thread_length_z,
	const size_t thread_length_y,
	const size_t thread_length_x,
	const size_t blockIdx_y,
	const size_t blockIdx_x,
	const size_t blockDim_y,
	const size_t blockDim_x,
	const size_t threadIdx_z,
	const size_t threadIdx_y,
	const size_t threadIdx_x) {
	const size_t global_thraed_index_z = threadIdx_z;
	const size_t loop_index_z_base = global_thraed_index_z*thread_length_z;
	const size_t actual_thread_length_z = get_actual_length(num_of_slices, thread_length_z, loop_index_z_base);
	for (size_t loop_index_z = loop_index_z_base; loop_index_z < actual_thread_length_z + loop_index_z_base; ++loop_index_z) {
		const size_t dst_slice_offset = loop_index_z * slice_length;
		const size_t src_dLdR_slice_offset = dst_slice_offset;
		const size_t src_DD0_slice_offset = loop_index_z * DD0_slice_size;
		const size_t dst_slice_upperlimit = dst_slice_offset + slice_length;
		const size_t global_thraed_index_y = blockIdx_y * blockDim_y + threadIdx_y;
		const size_t loop_index_y_base = global_thraed_index_y*thread_length_y;
		const size_t actual_thread_length_y = get_actual_length(loop_length, thread_length_y, loop_index_y_base);
		if (actual_thread_length_y > 0) {
			const size_t global_thread_index_x = blockIdx_x * blockDim_x + threadIdx_x;
			const size_t loop_index_x_base = global_thread_index_x * thread_length_x;
			const size_t actual_thread_length_x = get_actual_length(first_dimension_loop_size, thread_length_x, loop_index_x_base);
			const size_t loop_base_offset = loop_index_y_base*first_dimension_loop_size;
			for (size_t fdl_index = loop_index_x_base; fdl_index < actual_thread_length_x+ loop_index_x_base; ++fdl_index) {
				// Need to figure out which approximation has the least oscillation.
				// Note that L and R in this section refer to neighboring divided
				// difference entries, not to left and right approximations.
				FLOAT_TYPE smooth_m1_0;
				FLOAT_TYPE smooth_m1_1;
				FLOAT_TYPE smooth_m1_2;
				FLOAT_TYPE D1_src_1;
				FLOAT_TYPE D1_src_2;
				FLOAT_TYPE D1_src_3;
				FLOAT_TYPE D1_src_4;
				//Prologue
				{
					size_t src_DD0_offset = loop_base_offset + src_DD0_slice_offset;
					size_t src_DD0_index = fdl_index + src_DD0_offset;
					FLOAT_TYPE D1_src_0 = DD0_ptr[src_DD0_index];
					D1_src_1 = DD0_ptr[src_DD0_index + first_dimension_loop_size * 1];
					D1_src_2 = DD0_ptr[src_DD0_index + first_dimension_loop_size * 2];
					D1_src_3 = DD0_ptr[src_DD0_index + first_dimension_loop_size * 3];
					D1_src_4 = DD0_ptr[src_DD0_index + first_dimension_loop_size * 4];
					smooth_m1_0 = calcSmooth0(D1_src_0, D1_src_1, D1_src_2);
					smooth_m1_1 = calcSmooth1(D1_src_1, D1_src_2, D1_src_3);
					smooth_m1_2 = calcSmooth2(D1_src_2, D1_src_3, D1_src_4);
				}
				for (size_t loop_index_y = loop_index_y_base; loop_index_y < actual_thread_length_y + loop_index_y_base; ++loop_index_y) {
					size_t dst_offset = loop_index_y * first_dimension_loop_size + dst_slice_offset;
					size_t src_DD0_offset = loop_index_y * first_dimension_loop_size + src_DD0_slice_offset;
					size_t src_dLdR_offset = loop_index_y * first_dimension_loop_size + src_dLdR_slice_offset;
					size_t src_DD0_index = fdl_index + src_DD0_offset;
					size_t src_dLdR_index = fdl_index + src_dLdR_offset;
					FLOAT_TYPE D1_src_5 = DD0_ptr[src_DD0_index + first_dimension_loop_size * 5];
					FLOAT_TYPE smooth_m0_0 = calcSmooth0(D1_src_1, D1_src_2, D1_src_3);
					FLOAT_TYPE smooth_m0_1 = calcSmooth1(D1_src_2, D1_src_3, D1_src_4);
					FLOAT_TYPE smooth_m0_2 = calcSmooth2(D1_src_3, D1_src_4, D1_src_5);
					FLOAT_TYPE epsilon = (FLOAT_TYPE)1e-6;
					size_t dst_index = fdl_index + dst_offset;
					if (dst_index < dst_slice_upperlimit) {
						dst_deriv_l_ptr[dst_index] = weightWENO(dL0_ptr[src_dLdR_index], dL1_ptr[src_dLdR_index], dL2_ptr[src_dLdR_index], smooth_m1_0, smooth_m1_1, smooth_m1_2, weightL0, weightL1, weightL2, epsilon);
						dst_deriv_r_ptr[dst_index] = weightWENO(dR0_ptr[src_dLdR_index], dR1_ptr[src_dLdR_index], dR2_ptr[src_dLdR_index], smooth_m0_0, smooth_m0_1, smooth_m0_2, weightR0, weightR1, weightR2, epsilon);
					}
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

__device__ inline
void kernel_dim1_EpsilonCalculationMethod_maxOverNeighbor_inline(
	FLOAT_TYPE* dst_deriv_l_ptr,
	FLOAT_TYPE* dst_deriv_r_ptr,
	const FLOAT_TYPE* DD0_ptr,
	const FLOAT_TYPE* dL0_ptr,
	const FLOAT_TYPE* dL1_ptr,
	const FLOAT_TYPE* dL2_ptr,
	const FLOAT_TYPE* dR0_ptr,
	const FLOAT_TYPE* dR1_ptr,
	const FLOAT_TYPE* dR2_ptr,
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
	const size_t DD0_slice_size,
	const size_t thread_length_z,
	const size_t thread_length_y,
	const size_t thread_length_x,
	const size_t blockIdx_y,
	const size_t blockIdx_x,
	const size_t blockDim_y,
	const size_t blockDim_x,
	const size_t threadIdx_z,
	const size_t threadIdx_y,
	const size_t threadIdx_x) {
	const size_t global_thraed_index_z = threadIdx_z;
	const size_t loop_index_z_base = global_thraed_index_z*thread_length_z;
	const size_t actual_thread_length_z = get_actual_length(num_of_slices, thread_length_z, loop_index_z_base);
	for (size_t loop_index_z = loop_index_z_base; loop_index_z < actual_thread_length_z + loop_index_z_base; ++loop_index_z) {
		const size_t dst_slice_offset = loop_index_z * slice_length;
		const size_t src_dLdR_slice_offset = dst_slice_offset;
		const size_t src_DD0_slice_offset = loop_index_z * DD0_slice_size;
		const size_t dst_slice_upperlimit = dst_slice_offset + slice_length;
		const size_t global_thraed_index_y = blockIdx_y * blockDim_y + threadIdx_y;
		const size_t loop_index_y_base = global_thraed_index_y*thread_length_y;
		const size_t actual_thread_length_y = get_actual_length(loop_length, thread_length_y, loop_index_y_base);
		if (actual_thread_length_y > 0) {
			const size_t global_thread_index_x = blockIdx_x * blockDim_x + threadIdx_x;
			const size_t loop_index_x_base = global_thread_index_x * thread_length_x;
			const size_t actual_thread_length_x = get_actual_length(first_dimension_loop_size, thread_length_x, loop_index_x_base);
			const size_t loop_base_offset = loop_index_y_base*first_dimension_loop_size;
			for (size_t fdl_index = loop_index_x_base; fdl_index < actual_thread_length_x + loop_index_x_base; ++fdl_index) {
				// Need to figure out which approximation has the least oscillation.
				// Note that L and R in this section refer to neighboring divided
				// difference entries, not to left and right approximations.
				FLOAT_TYPE smooth_m1_0;
				FLOAT_TYPE smooth_m1_1;
				FLOAT_TYPE smooth_m1_2;
				FLOAT_TYPE D1_src_1;
				FLOAT_TYPE D1_src_2;
				FLOAT_TYPE D1_src_3;
				FLOAT_TYPE D1_src_4;
				FLOAT_TYPE pow_D1_src_0;
				FLOAT_TYPE pow_D1_src_1;
				FLOAT_TYPE pow_D1_src_2;
				FLOAT_TYPE pow_D1_src_3;
				FLOAT_TYPE pow_D1_src_4;
				//Prologue
				{
					size_t src_DD0_offset = loop_base_offset + src_DD0_slice_offset;
					size_t src_DD0_index = fdl_index + src_DD0_offset;
					FLOAT_TYPE D1_src_0 = DD0_ptr[src_DD0_index];
					D1_src_1 = DD0_ptr[src_DD0_index + first_dimension_loop_size * 1];
					D1_src_2 = DD0_ptr[src_DD0_index + first_dimension_loop_size * 2];
					D1_src_3 = DD0_ptr[src_DD0_index + first_dimension_loop_size * 3];
					D1_src_4 = DD0_ptr[src_DD0_index + first_dimension_loop_size * 4];
					smooth_m1_0 = calcSmooth0(D1_src_0, D1_src_1, D1_src_2);
					smooth_m1_1 = calcSmooth1(D1_src_1, D1_src_2, D1_src_3);
					smooth_m1_2 = calcSmooth2(D1_src_2, D1_src_3, D1_src_4);
					pow_D1_src_0 = D1_src_0 * D1_src_0;
					pow_D1_src_1 = D1_src_1 * D1_src_1;
					pow_D1_src_2 = D1_src_2 * D1_src_2;
					pow_D1_src_3 = D1_src_3 * D1_src_3;
					pow_D1_src_4 = D1_src_4 * D1_src_4;
				}
				for (size_t loop_index_y = loop_index_y_base; loop_index_y < actual_thread_length_y+ loop_index_y_base; ++loop_index_y) {
					size_t dst_offset = loop_index_y * first_dimension_loop_size + dst_slice_offset;
					size_t src_DD0_offset = loop_index_y * first_dimension_loop_size + src_DD0_slice_offset;
					size_t src_dLdR_offset = loop_index_y * first_dimension_loop_size + src_dLdR_slice_offset;
					size_t src_DD0_index = fdl_index + src_DD0_offset;
					size_t src_dLdR_index = fdl_index + src_dLdR_offset;
					FLOAT_TYPE D1_src_5 = DD0_ptr[src_DD0_index + first_dimension_loop_size * 5];
					FLOAT_TYPE smooth_m0_0 = calcSmooth0(D1_src_1, D1_src_2, D1_src_3);
					FLOAT_TYPE smooth_m0_1 = calcSmooth1(D1_src_2, D1_src_3, D1_src_4);
					FLOAT_TYPE smooth_m0_2 = calcSmooth2(D1_src_3, D1_src_4, D1_src_5);
					FLOAT_TYPE pow_D1_src_5 = D1_src_5 * D1_src_5;
					FLOAT_TYPE max_1_2 = max_float_type(pow_D1_src_1, pow_D1_src_2);
					FLOAT_TYPE max_3_4 = max_float_type(pow_D1_src_3, pow_D1_src_4);
					FLOAT_TYPE max_1_2_3_4 = max_float_type(max_1_2, max_3_4);
					FLOAT_TYPE maxOverNeighborD1squaredL = max_float_type(max_1_2_3_4, pow_D1_src_0);
					FLOAT_TYPE maxOverNeighborD1squaredR = max_float_type(max_1_2_3_4, pow_D1_src_5);
					FLOAT_TYPE epsilonL = (FLOAT_TYPE)(1e-6 * maxOverNeighborD1squaredL + get_epsilon_base<FLOAT_TYPE>());
					FLOAT_TYPE epsilonR = (FLOAT_TYPE)(1e-6 * maxOverNeighborD1squaredR + get_epsilon_base<FLOAT_TYPE>());
					size_t dst_index = fdl_index + dst_offset;
					if (dst_index < dst_slice_upperlimit) {
						dst_deriv_l_ptr[dst_index] = weightWENO(dL0_ptr[src_dLdR_index], dL1_ptr[src_dLdR_index], dL2_ptr[src_dLdR_index], smooth_m1_0, smooth_m1_1, smooth_m1_2, weightL0, weightL1, weightL2, epsilonL);
						dst_deriv_r_ptr[dst_index] = weightWENO(dR0_ptr[src_dLdR_index], dR1_ptr[src_dLdR_index], dR2_ptr[src_dLdR_index], smooth_m0_0, smooth_m0_1, smooth_m0_2, weightR0, weightR1, weightR2, epsilonR);
					}
					smooth_m1_0 = smooth_m0_0;
					smooth_m1_1 = smooth_m0_1;
					smooth_m1_2 = smooth_m0_2;
					D1_src_1 = D1_src_2;
					D1_src_2 = D1_src_3;
					D1_src_3 = D1_src_4;
					D1_src_4 = D1_src_5;
					pow_D1_src_0 = pow_D1_src_1;
					pow_D1_src_1 = pow_D1_src_2;
					pow_D1_src_2 = pow_D1_src_3;
					pow_D1_src_3 = pow_D1_src_4;
					pow_D1_src_4 = pow_D1_src_5;
				}
			}
		}
	}
}

void UpwindFirstWENO5a_execute_dim1_cuda
(
	FLOAT_TYPE* dst_deriv_l_ptr,
	FLOAT_TYPE* dst_deriv_r_ptr,
	const FLOAT_TYPE* DD0_0_ptr,
	const FLOAT_TYPE* dL0_ptr,
	const FLOAT_TYPE* dL1_ptr,
	const FLOAT_TYPE* dL2_ptr,
	const FLOAT_TYPE* dR0_ptr,
	const FLOAT_TYPE* dR1_ptr,
	const FLOAT_TYPE* dR2_ptr,
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
	const size_t DD0_slice_size,
	const levelset::EpsilonCalculationMethod_Type epsilonCalculationMethod_Type,
	beacls::CudaStream* cudaStream

);

__device__ inline
void kernel_dimLET2_EpsilonCalculationMethod_Constant_inline(
	FLOAT_TYPE* dst_deriv_l_ptr,
	FLOAT_TYPE* dst_deriv_r_ptr,
	const FLOAT_TYPE* DD0_0_ptr,
	const FLOAT_TYPE* DD1_0_ptr,
	const FLOAT_TYPE* DD2_0_ptr,
	const FLOAT_TYPE* DD3_0_ptr,
	const FLOAT_TYPE* DD4_0_ptr,
	const FLOAT_TYPE* DD5_0_ptr,
	const FLOAT_TYPE* dL0_ptr,
	const FLOAT_TYPE* dL1_ptr,
	const FLOAT_TYPE* dL2_ptr,
	const FLOAT_TYPE* dR0_ptr,
	const FLOAT_TYPE* dR1_ptr,
	const FLOAT_TYPE* dR2_ptr,
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
	const size_t thread_length_z,
	const size_t thread_length_y,
	const size_t thread_length_x,
	const size_t blockIdx_y,
	const size_t blockIdx_x,
	const size_t blockDim_y,
	const size_t blockDim_x,
	const size_t threadIdx_z,
	const size_t threadIdx_y,
	const size_t threadIdx_x) {
	const size_t global_thraed_index_z = threadIdx_z;
	const size_t loop_index_z_base = global_thraed_index_z*thread_length_z;
	const size_t actual_thread_length_z = get_actual_length(num_of_slices, thread_length_z, loop_index_z_base);
	for (size_t loop_index_z = loop_index_z_base; loop_index_z < actual_thread_length_z + loop_index_z_base; ++loop_index_z) {
		const size_t dst_slice_offset = loop_index_z * slice_length;
		const size_t global_thraed_index_y = blockIdx_y * blockDim_y + threadIdx_y;
		const size_t loop_index_y_base = global_thraed_index_y*thread_length_y;
		const size_t actual_thread_length_y = get_actual_length(loop_length, thread_length_y, loop_index_y_base);
		for (size_t loop_index_y = loop_index_y_base; loop_index_y < actual_thread_length_y + loop_index_y_base; ++loop_index_y) {
			const size_t dst_offset = loop_index_y * first_dimension_loop_size + dst_slice_offset;
			const size_t global_thread_index_x = blockIdx_x * blockDim_x + threadIdx_x;
			const size_t loop_index_x_base = global_thread_index_x * thread_length_x;
			const size_t actual_thread_length_x = get_actual_length(first_dimension_loop_size, thread_length_x, loop_index_x_base);
			for (size_t loop_index = 0; loop_index < actual_thread_length_x; ++loop_index) {
				const size_t first_dimension_loop_index = loop_index + loop_index_x_base;
				const size_t src_index = first_dimension_loop_index + dst_offset;
				const FLOAT_TYPE D1_src_0 = DD0_0_ptr[src_index];
				const FLOAT_TYPE D1_src_1 = DD1_0_ptr[src_index];
				const FLOAT_TYPE D1_src_2 = DD2_0_ptr[src_index];
				const FLOAT_TYPE D1_src_3 = DD3_0_ptr[src_index];
				const FLOAT_TYPE D1_src_4 = DD4_0_ptr[src_index];
				const FLOAT_TYPE D1_src_5 = DD5_0_ptr[src_index];
				const FLOAT_TYPE smoothL_0 = calcSmooth0(D1_src_0, D1_src_1, D1_src_2);
				const FLOAT_TYPE smoothL_1 = calcSmooth1(D1_src_1, D1_src_2, D1_src_3);
				const FLOAT_TYPE smoothL_2 = calcSmooth2(D1_src_2, D1_src_3, D1_src_4);
				const FLOAT_TYPE smoothR_0 = calcSmooth0(D1_src_1, D1_src_2, D1_src_3);
				const FLOAT_TYPE smoothR_1 = calcSmooth1(D1_src_2, D1_src_3, D1_src_4);
				const FLOAT_TYPE smoothR_2 = calcSmooth2(D1_src_3, D1_src_4, D1_src_5);
				const FLOAT_TYPE epsilon = (FLOAT_TYPE)1e-6;
				const size_t dst_index = src_index;
				dst_deriv_l_ptr[dst_index] = weightWENO(dL0_ptr[src_index], dL1_ptr[src_index], dL2_ptr[src_index], smoothL_0, smoothL_1, smoothL_2, weightL0, weightL1, weightL2, epsilon);
				dst_deriv_r_ptr[dst_index] = weightWENO(dR0_ptr[src_index], dR1_ptr[src_index], dR2_ptr[src_index], smoothR_0, smoothR_1, smoothR_2, weightR0, weightR1, weightR2, epsilon);
			}
		}
	}
}

__device__ inline
void kernel_dimLET2_EpsilonCalculationMethod_maxOverNeighbor_inline(
	FLOAT_TYPE* dst_deriv_l_ptr,
	FLOAT_TYPE* dst_deriv_r_ptr,
	const FLOAT_TYPE* DD0_0_ptr,
	const FLOAT_TYPE* DD1_0_ptr,
	const FLOAT_TYPE* DD2_0_ptr,
	const FLOAT_TYPE* DD3_0_ptr,
	const FLOAT_TYPE* DD4_0_ptr,
	const FLOAT_TYPE* DD5_0_ptr,
	const FLOAT_TYPE* dL0_ptr,
	const FLOAT_TYPE* dL1_ptr,
	const FLOAT_TYPE* dL2_ptr,
	const FLOAT_TYPE* dR0_ptr,
	const FLOAT_TYPE* dR1_ptr,
	const FLOAT_TYPE* dR2_ptr,
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
	const size_t thread_length_z,
	const size_t thread_length_y,
	const size_t thread_length_x,
	const size_t blockIdx_y,
	const size_t blockIdx_x,
	const size_t blockDim_y,
	const size_t blockDim_x,
	const size_t threadIdx_z,
	const size_t threadIdx_y,
	const size_t threadIdx_x) {
	const size_t global_thraed_index_z = threadIdx_z;
	const size_t loop_index_z_base = global_thraed_index_z*thread_length_z;
	const size_t actual_thread_length_z = get_actual_length(num_of_slices, thread_length_z, loop_index_z_base);
	for (size_t loop_index_z = loop_index_z_base; loop_index_z < actual_thread_length_z + loop_index_z_base; ++loop_index_z) {
		const size_t dst_slice_offset = loop_index_z * slice_length;
		const size_t global_thraed_index_y = blockIdx_y * blockDim_y + threadIdx_y;
		const size_t loop_index_y_base = global_thraed_index_y*thread_length_y;
		const size_t actual_thread_length_y = get_actual_length(loop_length, thread_length_y, loop_index_y_base);
		for (size_t loop_index_y = loop_index_y_base; loop_index_y < actual_thread_length_y + loop_index_y_base; ++loop_index_y) {
			const size_t dst_offset = loop_index_y * first_dimension_loop_size + dst_slice_offset;
			const size_t global_thread_index_x = blockIdx_x * blockDim_x + threadIdx_x;
			const size_t loop_index_x_base = global_thread_index_x * thread_length_x;
			const size_t actual_thread_length_x = get_actual_length(first_dimension_loop_size, thread_length_x, loop_index_x_base);
			for (size_t loop_index = 0; loop_index < actual_thread_length_x; ++loop_index) {
				const size_t first_dimension_loop_index = loop_index + loop_index_x_base;
				const size_t src_index = first_dimension_loop_index + dst_offset;
				const FLOAT_TYPE D1_src_0 = DD0_0_ptr[src_index];
				const FLOAT_TYPE D1_src_1 = DD1_0_ptr[src_index];
				const FLOAT_TYPE D1_src_2 = DD2_0_ptr[src_index];
				const FLOAT_TYPE D1_src_3 = DD3_0_ptr[src_index];
				const FLOAT_TYPE D1_src_4 = DD4_0_ptr[src_index];
				const FLOAT_TYPE D1_src_5 = DD5_0_ptr[src_index];
				const FLOAT_TYPE smoothL_0 = calcSmooth0(D1_src_0, D1_src_1, D1_src_2);
				const FLOAT_TYPE smoothL_1 = calcSmooth1(D1_src_1, D1_src_2, D1_src_3);
				const FLOAT_TYPE smoothL_2 = calcSmooth2(D1_src_2, D1_src_3, D1_src_4);
				const FLOAT_TYPE smoothR_0 = calcSmooth0(D1_src_1, D1_src_2, D1_src_3);
				const FLOAT_TYPE smoothR_1 = calcSmooth1(D1_src_2, D1_src_3, D1_src_4);
				const FLOAT_TYPE smoothR_2 = calcSmooth2(D1_src_3, D1_src_4, D1_src_5);
				const FLOAT_TYPE pow_D1_src_0 = D1_src_0 * D1_src_0;
				const FLOAT_TYPE pow_D1_src_1 = D1_src_1 * D1_src_1;
				const FLOAT_TYPE pow_D1_src_2 = D1_src_2 * D1_src_2;
				const FLOAT_TYPE pow_D1_src_3 = D1_src_3 * D1_src_3;
				const FLOAT_TYPE pow_D1_src_4 = D1_src_4 * D1_src_4;
				const FLOAT_TYPE pow_D1_src_5 = D1_src_5 * D1_src_5;
				const FLOAT_TYPE max_1_2 = max_float_type(pow_D1_src_1, pow_D1_src_2);
				const FLOAT_TYPE max_3_4 = max_float_type(pow_D1_src_3, pow_D1_src_4);
				const FLOAT_TYPE max_1_2_3_4 = max_float_type(max_1_2, max_3_4);
				const FLOAT_TYPE maxOverNeighborD1squaredL = max_float_type(max_1_2_3_4, pow_D1_src_0);
				const FLOAT_TYPE maxOverNeighborD1squaredR = max_float_type(max_1_2_3_4, pow_D1_src_5);
				const FLOAT_TYPE epsilonL = (FLOAT_TYPE)(1e-6 * maxOverNeighborD1squaredL + get_epsilon_base<FLOAT_TYPE>());
				const FLOAT_TYPE epsilonR = (FLOAT_TYPE)(1e-6 * maxOverNeighborD1squaredR + get_epsilon_base<FLOAT_TYPE>());
				const size_t dst_index = src_index;
				dst_deriv_l_ptr[dst_index] = weightWENO(dL0_ptr[src_index], dL1_ptr[src_index], dL2_ptr[src_index], smoothL_0, smoothL_1, smoothL_2, weightL0, weightL1, weightL2, epsilonL);
				dst_deriv_r_ptr[dst_index] = weightWENO(dR0_ptr[src_index], dR1_ptr[src_index], dR2_ptr[src_index], smoothR_0, smoothR_1, smoothR_2, weightR0, weightR1, weightR2, epsilonR);
			}
		}
	}
}

void UpwindFirstWENO5a_execute_dimLET2_cuda
(
	FLOAT_TYPE* dst_deriv_l_ptr,
	FLOAT_TYPE* dst_deriv_r_ptr,
	const FLOAT_TYPE* DD0_0_ptr,
	const FLOAT_TYPE* DD1_0_ptr,
	const FLOAT_TYPE* DD2_0_ptr,
	const FLOAT_TYPE* DD3_0_ptr,
	const FLOAT_TYPE* DD4_0_ptr,
	const FLOAT_TYPE* DD5_0_ptr,
	const FLOAT_TYPE* dL0_ptr,
	const FLOAT_TYPE* dL1_ptr,
	const FLOAT_TYPE* dL2_ptr,
	const FLOAT_TYPE* dR0_ptr,
	const FLOAT_TYPE* dR1_ptr,
	const FLOAT_TYPE* dR2_ptr,
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
	const levelset::EpsilonCalculationMethod_Type epsilonCalculationMethod_Type,
	beacls::CudaStream* cudaStream
																		      
     );
#endif	/* __UpwindFirstWENO5a_cuda_hpp__ */
