// CUDA runtime
#include <cuda_runtime.h>

#include <typedef.hpp>
#include <cuda_macro.hpp>
#include <Core/UVec.hpp>
#include <Core/CudaStream.hpp>
#include "ArtificialDissipationGLF_cuda.hpp"
#if defined(WITH_GPU)

template<typename T>
inline
__device__
void dev_minmax(T& dst_min, T& dst_max, const T& lhs, const T& rhs) {
	if (lhs < rhs) {
		dst_min = lhs;
		dst_max = rhs;
	}
	else {
		dst_min = rhs;
		dst_max = lhs;
	}
}
template<typename T>
inline
__device__
void dev_min(T& dst_min, const T& lhs) {
	if (lhs < dst_min) {
		dst_min = lhs;
	}
}
template<typename T>
inline
__device__
void dev_max(T& dst_max, const T& lhs) {
	if (lhs > dst_max) {
		dst_max = lhs;
	}
}

template<size_t blockSize, bool vector_alpha, bool dim0, bool updateDerivMinMax>
__global__
void kernel_Dissipation(
	FLOAT_TYPE  *g_dst_diss_ptr,
	FLOAT_TYPE  *g_dst_min_ptr,
	FLOAT_TYPE  *g_dst_max_ptr,
	FLOAT_TYPE  *g_dst_max_alpha_ptr,
	const FLOAT_TYPE *g_l_ptr,
	const FLOAT_TYPE *g_r_ptr,
	const FLOAT_TYPE *g_a_ptr,
	const FLOAT_TYPE alpha0,
	const FLOAT_TYPE float_max,
	const size_t loop_length) {
	extern __shared__ uint8_t shared_area[];
	FLOAT_TYPE* s_max_a_data = (FLOAT_TYPE*)&shared_area;
	FLOAT_TYPE* s_min_data;
	FLOAT_TYPE* s_max_data;
	if (updateDerivMinMax) {
		s_min_data = (FLOAT_TYPE*)&s_max_a_data[blockSize];
		s_max_data = (FLOAT_TYPE*)&s_min_data[blockSize];
	}
	else {
		s_min_data = NULL;
		s_max_data = NULL;
	}

	const size_t tid = threadIdx.x;
	size_t i = (blockIdx.x*(blockSize)+tid);
	const size_t gridSize = blockSize * gridDim.x;
	FLOAT_TYPE local_min = float_max;
	FLOAT_TYPE local_max = -float_max;
	FLOAT_TYPE local_max_a = -float_max;
	while (i < loop_length) {
		const FLOAT_TYPE l = g_l_ptr[i];
		const FLOAT_TYPE r = g_r_ptr[i];
		FLOAT_TYPE a;
		if (vector_alpha) {
			a = g_a_ptr[i];
			dev_max(local_max_a, a);
		}
		else a = alpha0;
		if (updateDerivMinMax) {
			FLOAT_TYPE min_lr;
			FLOAT_TYPE max_lr;
			dev_minmax(min_lr, max_lr, l, r);
			dev_min(local_min, min_lr);
			dev_max(local_max, max_lr);
		}
		if (dim0) g_dst_diss_ptr[i] = (r - l) * a / 2;
		else g_dst_diss_ptr[i] += (r - l) * a / 2;
		i += gridSize;
	}
	if (updateDerivMinMax || vector_alpha) {
		if (updateDerivMinMax) {
			s_min_data[tid] = local_min;
			s_max_data[tid] = local_max;
		}
		if (vector_alpha) s_max_a_data[tid] = local_max_a;
		__syncthreads();

		if (blockSize >= 1024) {
			if (tid < 512) {
				if (updateDerivMinMax)dev_min(s_min_data[tid], s_min_data[tid + 512]);
				if (updateDerivMinMax)dev_max(s_max_data[tid], s_max_data[tid + 512]);
				if (vector_alpha) dev_max(s_max_a_data[tid], s_max_a_data[tid + 512]);
			} __syncthreads();
		}
		if (blockSize >= 512) {
			if (tid < 256) {
				if (updateDerivMinMax)dev_min(s_min_data[tid], s_min_data[tid + 256]);
				if (updateDerivMinMax)dev_max(s_max_data[tid], s_max_data[tid + 256]);
				if (vector_alpha) dev_max(s_max_a_data[tid], s_max_a_data[tid + 256]);
			} __syncthreads();
		}
		if (blockSize >= 256) {
			if (tid < 128) {
				if (updateDerivMinMax)dev_min(s_min_data[tid], s_min_data[tid + 128]);
				if (updateDerivMinMax)dev_max(s_max_data[tid], s_max_data[tid + 128]);
				if (vector_alpha) dev_max(s_max_a_data[tid], s_max_a_data[tid + 128]);
			} __syncthreads();
		}
		if (blockSize >= 128) {
			if (tid < 64) {
				if (updateDerivMinMax)dev_min(s_min_data[tid], s_min_data[tid + 64]);
				if (updateDerivMinMax)dev_max(s_max_data[tid], s_max_data[tid + 64]);
				if (vector_alpha) dev_max(s_max_a_data[tid], s_max_a_data[tid + 64]);
			}
			__syncthreads();
		}
		if (blockSize >= 64) {
			if (tid < 32) {
				if (updateDerivMinMax)dev_min(s_min_data[tid], s_min_data[tid + 32]);
				if (updateDerivMinMax)dev_max(s_max_data[tid], s_max_data[tid + 32]);
				if (vector_alpha) dev_max(s_max_a_data[tid], s_max_a_data[tid + 32]);
			}
			__syncthreads();
		}
		if (blockSize >= 32) {
			if (tid < 16) {
				if (updateDerivMinMax)dev_min(s_min_data[tid], s_min_data[tid + 16]);
				if (updateDerivMinMax)dev_max(s_max_data[tid], s_max_data[tid + 16]);
				if (vector_alpha) dev_max(s_max_a_data[tid], s_max_a_data[tid + 16]);
			}
			__syncthreads();
		}
		if (blockSize >= 16) {
			if (tid < 8) {
				if (updateDerivMinMax)dev_min(s_min_data[tid], s_min_data[tid + 8]);
				if (updateDerivMinMax)dev_max(s_max_data[tid], s_max_data[tid + 8]);
				if (vector_alpha) dev_max(s_max_a_data[tid], s_max_a_data[tid + 8]);
			}
			__syncthreads();
		}
		if (blockSize >= 8) {
			if (tid < 4) {
				if (updateDerivMinMax)dev_min(s_min_data[tid], s_min_data[tid + 4]);
				if (updateDerivMinMax)dev_max(s_max_data[tid], s_max_data[tid + 4]);
				if (vector_alpha) dev_max(s_max_a_data[tid], s_max_a_data[tid + 4]);
			}
			__syncthreads();
		}
		if (blockSize >= 4) {
			if (tid < 2) {
				if (updateDerivMinMax)dev_min(s_min_data[tid], s_min_data[tid + 2]);
				if (updateDerivMinMax)dev_max(s_max_data[tid], s_max_data[tid + 2]);
				if (vector_alpha) dev_max(s_max_a_data[tid], s_max_a_data[tid + 2]);
			}
			__syncthreads();
		}
		if (blockSize >= 2) {
			if (tid < 1) {
				if (updateDerivMinMax)dev_min(s_min_data[tid], s_min_data[tid + 1]);
				if (updateDerivMinMax)dev_max(s_max_data[tid], s_max_data[tid + 1]);
				if (vector_alpha) dev_max(s_max_a_data[tid], s_max_a_data[tid + 1]);
			}
			__syncthreads();
		}
		if (tid == 0) {
			if (updateDerivMinMax)g_dst_min_ptr[blockIdx.x] = s_min_data[0];
			if (updateDerivMinMax)g_dst_max_ptr[blockIdx.x] = s_max_data[0];
			if (vector_alpha) g_dst_max_alpha_ptr[blockIdx.x] = s_max_a_data[0];
		}
	}
}

template<bool vector_alpha, bool dim0, bool updateDerivMinMax>
void launch_kernel_Dissipation(
	FLOAT_TYPE* diss_ptr,
	FLOAT_TYPE* min_ptr,
	FLOAT_TYPE* max_ptr,
	FLOAT_TYPE* max_alpha_ptr,
	const FLOAT_TYPE* deriv_l_ptr,
	const FLOAT_TYPE* deriv_r_ptr,
	const FLOAT_TYPE* a_ptr,
	const FLOAT_TYPE alpha0,
	const size_t loop_length,
	const dim3 num_of_blocks,
	const dim3 num_of_threads,
	cudaStream_t diss_stream
	
) {
	const size_t num_of_threads_x = num_of_threads.x;
	const int num_of_shared_memory_buffer = (num_of_threads_x <= 32) ? 2 : 1;
	int num_of_shared_memories = 0;
	if (vector_alpha) num_of_shared_memories += 1;
	if (updateDerivMinMax) num_of_shared_memories += 2;

	const int smemSize  = num_of_shared_memory_buffer * num_of_threads_x * sizeof(FLOAT_TYPE) * num_of_shared_memories;
	switch (num_of_threads_x) {
	case 1024:
		kernel_Dissipation<1024, vector_alpha, dim0, updateDerivMinMax> << <num_of_blocks, num_of_threads, smemSize, diss_stream >> > (
			diss_ptr, min_ptr, max_ptr, max_alpha_ptr, deriv_l_ptr, deriv_r_ptr, a_ptr, alpha0, std::numeric_limits<FLOAT_TYPE>::max(),
			loop_length
			);
		break;
	case 512:
		kernel_Dissipation<512, vector_alpha, dim0, updateDerivMinMax> << <num_of_blocks, num_of_threads, smemSize, diss_stream >> > (
			diss_ptr, min_ptr, max_ptr, max_alpha_ptr, deriv_l_ptr, deriv_r_ptr, a_ptr, alpha0, std::numeric_limits<FLOAT_TYPE>::max(),
			loop_length
			);
		break;
	case 256:
		kernel_Dissipation<256, vector_alpha, dim0, updateDerivMinMax> << <num_of_blocks, num_of_threads, smemSize, diss_stream >> > (
			diss_ptr, min_ptr, max_ptr, max_alpha_ptr, deriv_l_ptr, deriv_r_ptr, a_ptr, alpha0, std::numeric_limits<FLOAT_TYPE>::max(),
			loop_length
			);
		break;
	case 128:
		kernel_Dissipation<128, vector_alpha, dim0, updateDerivMinMax> << <num_of_blocks, num_of_threads, smemSize, diss_stream >> > (
			diss_ptr, min_ptr, max_ptr, max_alpha_ptr, deriv_l_ptr, deriv_r_ptr, a_ptr, alpha0, std::numeric_limits<FLOAT_TYPE>::max(),
			loop_length
			);
		break;
	case 64:
		kernel_Dissipation<64, vector_alpha, dim0, updateDerivMinMax> << <num_of_blocks, num_of_threads, smemSize, diss_stream >> > (
			diss_ptr, min_ptr, max_ptr, max_alpha_ptr, deriv_l_ptr, deriv_r_ptr, a_ptr, alpha0, std::numeric_limits<FLOAT_TYPE>::max(),
			loop_length
			);
		break;
	case 32:
		kernel_Dissipation<32, vector_alpha, dim0, updateDerivMinMax> << <num_of_blocks, num_of_threads, smemSize, diss_stream >> > (
			diss_ptr, min_ptr, max_ptr, max_alpha_ptr, deriv_l_ptr, deriv_r_ptr, a_ptr, alpha0, std::numeric_limits<FLOAT_TYPE>::max(),
			loop_length
			);
		break;
	case 16:
		kernel_Dissipation<16, vector_alpha, dim0, updateDerivMinMax> << <num_of_blocks, num_of_threads, smemSize, diss_stream >> > (
			diss_ptr, min_ptr, max_ptr, max_alpha_ptr, deriv_l_ptr, deriv_r_ptr, a_ptr, alpha0, std::numeric_limits<FLOAT_TYPE>::max(),
			loop_length
			);
		break;
	case 8:
		kernel_Dissipation<8, vector_alpha, dim0, updateDerivMinMax> << <num_of_blocks, num_of_threads, smemSize, diss_stream >> > (
			diss_ptr, min_ptr, max_ptr, max_alpha_ptr, deriv_l_ptr, deriv_r_ptr, a_ptr, alpha0, std::numeric_limits<FLOAT_TYPE>::max(),
			loop_length
			);
		break;
	case 4:
		kernel_Dissipation<4, vector_alpha, dim0, updateDerivMinMax> << <num_of_blocks, num_of_threads, smemSize, diss_stream >> > (
			diss_ptr, min_ptr, max_ptr, max_alpha_ptr, deriv_l_ptr, deriv_r_ptr, a_ptr, alpha0, std::numeric_limits<FLOAT_TYPE>::max(),
			loop_length
			);
		break;
	case 2:
		kernel_Dissipation<2, vector_alpha, dim0, updateDerivMinMax> << <num_of_blocks, num_of_threads, smemSize, diss_stream >> > (
			diss_ptr, min_ptr, max_ptr, max_alpha_ptr, deriv_l_ptr, deriv_r_ptr, a_ptr, alpha0, std::numeric_limits<FLOAT_TYPE>::max(),
			loop_length
			);
		break;
	case 1:
		kernel_Dissipation<1, vector_alpha, dim0, updateDerivMinMax> << <num_of_blocks, num_of_threads, smemSize, diss_stream >> > (
			diss_ptr, min_ptr, max_ptr, max_alpha_ptr, deriv_l_ptr, deriv_r_ptr, a_ptr, alpha0, std::numeric_limits<FLOAT_TYPE>::max(),
			loop_length
			);
		break;
	default:
		std::cerr << "Illigal num_of_threads_x: " << num_of_threads_x << std::endl;
		break;
	}
}

void ArtificialDissipationGLF_execute_cuda (
	beacls::UVec& diss_uvec,
	const beacls::UVec& alpha_uvec,
	const beacls::UVec& deriv_l_uvec,
	const beacls::UVec& deriv_r_uvec,
	beacls::UVec &tmp_min_cuda_uvec,
	beacls::UVec &tmp_max_cuda_uvec,
	beacls::UVec &tmp_max_alpha_cuda_uvec,
	const size_t dimension,
	const size_t loop_size,
	const bool updateDerivMinMax
) {
	const size_t loop_length = loop_size;
	size_t num_of_threads_x;
	size_t num_of_blocks_x;
	get_cuda_thread_size_1d<size_t>(
		num_of_threads_x,
		num_of_blocks_x,
		loop_length,
		512
		);
	beacls::UVecType type = deriv_l_uvec.type();
	beacls::UVecDepth depth = deriv_l_uvec.depth();
	if (tmp_min_cuda_uvec.type() != type) tmp_min_cuda_uvec = beacls::UVec(depth, type, num_of_blocks_x);
	else if (tmp_min_cuda_uvec.size() != num_of_blocks_x) tmp_min_cuda_uvec.resize(num_of_blocks_x);
	if (tmp_max_cuda_uvec.type() != type) tmp_max_cuda_uvec = beacls::UVec(depth, type, num_of_blocks_x);
	else if (tmp_max_cuda_uvec.size() != num_of_blocks_x) tmp_max_cuda_uvec.resize(num_of_blocks_x);
	if (tmp_max_alpha_cuda_uvec.type() != type) tmp_max_alpha_cuda_uvec = beacls::UVec(depth, type, num_of_blocks_x);
	else if (tmp_max_alpha_cuda_uvec.size() != num_of_blocks_x) tmp_max_alpha_cuda_uvec.resize(num_of_blocks_x);

	FLOAT_TYPE* diss_ptr = beacls::UVec_<FLOAT_TYPE>(diss_uvec).ptr();
	const FLOAT_TYPE* alpha_ptr = beacls::UVec_<FLOAT_TYPE>(alpha_uvec).ptr();
	const FLOAT_TYPE* deriv_l_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_l_uvec).ptr();
	const FLOAT_TYPE* deriv_r_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_r_uvec).ptr();
	const size_t alpha_size = alpha_uvec.size();

//	beacls::synchronizeUVec(deriv_r_uvec);
	cudaStream_t diss_stream = beacls::get_stream(diss_uvec);

	dim3 num_of_blocks(num_of_blocks_x, 1);
	dim3 num_of_threads(num_of_threads_x, 1, 1);
	FLOAT_TYPE* min_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_min_cuda_uvec).ptr();
	FLOAT_TYPE* max_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_max_cuda_uvec).ptr();
	FLOAT_TYPE* max_alpha_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_max_alpha_cuda_uvec).ptr();

	if (beacls::is_cuda(alpha_uvec)) {
		if (dimension == 0) {
			if (updateDerivMinMax) {
				launch_kernel_Dissipation<true, true, true>(
					diss_ptr, min_ptr, max_ptr, max_alpha_ptr, deriv_l_ptr, deriv_r_ptr, alpha_ptr, 0,
					loop_length, num_of_blocks, num_of_threads, diss_stream);
			}
			else {
				launch_kernel_Dissipation<true, true, false>(
					diss_ptr, min_ptr, max_ptr, max_alpha_ptr, deriv_l_ptr, deriv_r_ptr, alpha_ptr, 0,
					loop_length, num_of_blocks, num_of_threads, diss_stream);
			}
		}
		else {
			if (updateDerivMinMax) {
				launch_kernel_Dissipation<true, false, true>(
					diss_ptr, min_ptr, max_ptr, max_alpha_ptr, deriv_l_ptr, deriv_r_ptr, alpha_ptr, 0,
					loop_length, num_of_blocks, num_of_threads, diss_stream);
			}
			else {
				launch_kernel_Dissipation<true, false, false>(
					diss_ptr, min_ptr, max_ptr, max_alpha_ptr, deriv_l_ptr, deriv_r_ptr, alpha_ptr, 0,
					loop_length, num_of_blocks, num_of_threads, diss_stream);
			}
		}
	}
	else {
		const FLOAT_TYPE alpha = alpha_ptr[0];
		if (dimension == 0) {
			if (updateDerivMinMax) {
				launch_kernel_Dissipation<false, true, true>(
					diss_ptr, min_ptr, max_ptr, max_alpha_ptr, deriv_l_ptr, deriv_r_ptr, alpha_ptr, alpha,
					loop_length, num_of_blocks, num_of_threads, diss_stream);
			}
			else {
				launch_kernel_Dissipation<false, true, false>(
					diss_ptr, min_ptr, max_ptr, max_alpha_ptr, deriv_l_ptr, deriv_r_ptr, alpha_ptr, alpha,
					loop_length, num_of_blocks, num_of_threads, diss_stream);
			}
		}
		else {
			if (updateDerivMinMax) {
				launch_kernel_Dissipation<false, false, true>(
					diss_ptr, min_ptr, max_ptr, max_alpha_ptr, deriv_l_ptr, deriv_r_ptr, alpha_ptr, alpha,
					loop_length, num_of_blocks, num_of_threads, diss_stream);
			}
			else {
				launch_kernel_Dissipation<false, false, false>(
					diss_ptr, min_ptr, max_ptr, max_alpha_ptr, deriv_l_ptr, deriv_r_ptr, alpha_ptr, alpha,
					loop_length, num_of_blocks, num_of_threads, diss_stream);
			}
		}
	}
}
void ArtificialDissipationGLF_reduce_cuda
(
	FLOAT_TYPE& step_bound_inv,
	FLOAT_TYPE& derivMin,
	FLOAT_TYPE& derivMax,
	const beacls::UVec& alpha_uvec,
	const beacls::UVec &tmp_min_cuda_uvec,
	const beacls::UVec &tmp_max_cuda_uvec,
	const beacls::UVec &tmp_max_cuda_alpha_uvec,
	beacls::UVec &tmp_min_cpu_uvec,
	beacls::UVec &tmp_max_cpu_uvec,
	beacls::UVec &tmp_max_alpha_cpu_uvec,
	const FLOAT_TYPE dxInv,
	const bool updateDerivMinMax
) {
	beacls::UVecType type = beacls::UVecType_Vector;
	beacls::UVecDepth depth = tmp_min_cuda_uvec.depth();
	const size_t num_of_blocks_x = tmp_min_cpu_uvec.size();
	if (updateDerivMinMax) {
		tmp_min_cuda_uvec.convertTo(tmp_min_cpu_uvec, beacls::UVecType_Vector);
		tmp_max_cuda_uvec.convertTo(tmp_max_cpu_uvec, beacls::UVecType_Vector);
		derivMin = std::numeric_limits<FLOAT_TYPE>::max();
		derivMax = -std::numeric_limits<FLOAT_TYPE>::max();
		const beacls::FloatVec& tmp_min_cpu_vec = *(beacls::UVec_<FLOAT_TYPE>(tmp_min_cpu_uvec).vec());
		const beacls::FloatVec& tmp_max_cpu_vec = *(beacls::UVec_<FLOAT_TYPE>(tmp_max_cpu_uvec).vec());
		for (size_t index = 0; index < tmp_min_cpu_vec.size(); ++index) {
			if (derivMin > tmp_min_cpu_vec[index]) derivMin = tmp_min_cpu_vec[index];
			if (derivMax < tmp_max_cpu_vec[index]) derivMax = tmp_max_cpu_vec[index];
		}
	}
	if (beacls::is_cuda(alpha_uvec)) {
		const beacls::FloatVec& tmp_max_alpha_cpu_vec = *(beacls::UVec_<FLOAT_TYPE>(tmp_max_alpha_cpu_uvec).vec());
		FLOAT_TYPE max_value = -std::numeric_limits<FLOAT_TYPE>::max();
		for (size_t index = 0; index < tmp_min_cpu_uvec.size(); ++index) {
			if (max_value < tmp_max_alpha_cpu_vec[index]) max_value = tmp_max_alpha_cpu_vec[index];
		}
		step_bound_inv = max_value * dxInv;
	}
	else {
		const FLOAT_TYPE* alpha_ptr = beacls::UVec_<FLOAT_TYPE>(alpha_uvec).ptr();
		const FLOAT_TYPE alpha = alpha_ptr[0];
		const FLOAT_TYPE max_value = alpha;
		step_bound_inv = max_value * dxInv;
	}

}
#endif /* defined(WITH_GPU) */
