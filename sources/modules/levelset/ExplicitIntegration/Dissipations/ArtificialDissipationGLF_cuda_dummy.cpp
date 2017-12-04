#include <typedef.hpp>
#include <cuda_macro.hpp>
#include <cmath>
#include <numeric>
#include <functional>
#include <utility>
#include <Core/UVec.hpp>
#include <Core/CudaStream.hpp>
#include <iostream>
#include "ArtificialDissipationGLF_cuda.hpp"
#include <macro.hpp>
#if !defined(WITH_GPU)
struct dim3 {
public:
	size_t x;
	size_t y;
	size_t z;
	dim3(const size_t x, const size_t y=1, const size_t z=1) : x(x), y(y), z(z) {}
};

template<size_t blockSize, bool vector_alpha, bool dim0, bool updateDerivMinMax>
void kernel_Dissipation_start(
	FLOAT_TYPE  *g_dst_diss_ptr,
	const FLOAT_TYPE *g_l_ptr,
	const FLOAT_TYPE *g_r_ptr,
	const FLOAT_TYPE *g_a_ptr,
	const FLOAT_TYPE alpha0,
	const FLOAT_TYPE float_max,
	const size_t loop_length,
	const dim3& gridDim,
	const dim3& blockIdx,
	const dim3& threadIdx,
	std::vector<uint8_t>& shared_area
	) {
	FLOAT_TYPE* s_max_a_data = (FLOAT_TYPE*)shared_area.data();
	FLOAT_TYPE* s_min_data = get_min_data_ptr<blockSize, FLOAT_TYPE, vector_alpha, updateDerivMinMax>(s_max_a_data);
	FLOAT_TYPE* s_max_data = get_max_data_ptr<blockSize, FLOAT_TYPE, vector_alpha, updateDerivMinMax>(s_max_a_data);

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
	}
}

template<size_t blockSize, bool vector_alpha, bool dim0, bool updateDerivMinMax>
void kernel_Dissipation_step1024(
	const dim3& threadIdx,
	std::vector<uint8_t>& shared_area
) {
	FLOAT_TYPE* s_max_a_data = (FLOAT_TYPE*)shared_area.data();
	FLOAT_TYPE* s_min_data = get_min_data_ptr<blockSize, FLOAT_TYPE, vector_alpha, updateDerivMinMax>(s_max_a_data);
	FLOAT_TYPE* s_max_data = get_max_data_ptr<blockSize, FLOAT_TYPE, vector_alpha, updateDerivMinMax>(s_max_a_data);

	const size_t tid = threadIdx.x;
	if (updateDerivMinMax || vector_alpha) {
		if (blockSize >= 1024) {
			if (tid < 512) {
				if (updateDerivMinMax)dev_min(s_min_data[tid], s_min_data[tid + 512]);
				if (updateDerivMinMax)dev_max(s_max_data[tid], s_max_data[tid + 512]);
				if (vector_alpha) dev_max(s_max_a_data[tid], s_max_a_data[tid + 512]);
			}
		}
	}
}
template<size_t blockSize, bool vector_alpha, bool dim0, bool updateDerivMinMax>
void kernel_Dissipation_step512(
	const dim3& threadIdx,
	std::vector<uint8_t>& shared_area
) {
	FLOAT_TYPE* s_max_a_data = (FLOAT_TYPE*)shared_area.data();
	FLOAT_TYPE* s_min_data = get_min_data_ptr<blockSize, FLOAT_TYPE, vector_alpha, updateDerivMinMax>(s_max_a_data);
	FLOAT_TYPE* s_max_data = get_max_data_ptr<blockSize, FLOAT_TYPE, vector_alpha, updateDerivMinMax>(s_max_a_data);

	const size_t tid = threadIdx.x;
	if (updateDerivMinMax || vector_alpha) {
		if (blockSize >= 512) {
			if (tid < 256) {
				if (updateDerivMinMax)dev_min(s_min_data[tid], s_min_data[tid + 256]);
				if (updateDerivMinMax)dev_max(s_max_data[tid], s_max_data[tid + 256]);
				if (vector_alpha) dev_max(s_max_a_data[tid], s_max_a_data[tid + 256]);
			}
		}
	}
}
template<size_t blockSize, bool vector_alpha, bool dim0, bool updateDerivMinMax>
void kernel_Dissipation_step256(
	const dim3& threadIdx,
	std::vector<uint8_t>& shared_area
) {
	FLOAT_TYPE* s_max_a_data = (FLOAT_TYPE*)shared_area.data();
	FLOAT_TYPE* s_min_data = get_min_data_ptr<blockSize, FLOAT_TYPE, vector_alpha, updateDerivMinMax>(s_max_a_data);
	FLOAT_TYPE* s_max_data = get_max_data_ptr<blockSize, FLOAT_TYPE, vector_alpha, updateDerivMinMax>(s_max_a_data);

	const size_t tid = threadIdx.x;
	if (updateDerivMinMax || vector_alpha) {
		if (blockSize >= 256) {
			if (tid < 128) {
				if (updateDerivMinMax)dev_min(s_min_data[tid], s_min_data[tid + 128]);
				if (updateDerivMinMax)dev_max(s_max_data[tid], s_max_data[tid + 128]);
				if (vector_alpha) dev_max(s_max_a_data[tid], s_max_a_data[tid + 128]);
			}
		}
	}
}
template<size_t blockSize, bool vector_alpha, bool dim0, bool updateDerivMinMax>
void kernel_Dissipation_step128(
	const dim3& threadIdx,
	std::vector<uint8_t>& shared_area
) {
	FLOAT_TYPE* s_max_a_data = (FLOAT_TYPE*)shared_area.data();
	FLOAT_TYPE* s_min_data = get_min_data_ptr<blockSize, FLOAT_TYPE, vector_alpha, updateDerivMinMax>(s_max_a_data);
	FLOAT_TYPE* s_max_data = get_max_data_ptr<blockSize, FLOAT_TYPE, vector_alpha, updateDerivMinMax>(s_max_a_data);

	const size_t tid = threadIdx.x;
	if (updateDerivMinMax || vector_alpha) {
		if (blockSize >= 128) {
			if (tid < 64) {
				if (updateDerivMinMax)dev_min(s_min_data[tid], s_min_data[tid + 64]);
				if (updateDerivMinMax)dev_max(s_max_data[tid], s_max_data[tid + 64]);
				if (vector_alpha) dev_max(s_max_a_data[tid], s_max_a_data[tid + 64]);
			}
		}
	}
}
template<size_t blockSize, bool vector_alpha, bool dim0, bool updateDerivMinMax>
void kernel_Dissipation_step64(
	const dim3& threadIdx,
	std::vector<uint8_t>& shared_area
) {
	FLOAT_TYPE* s_max_a_data = (FLOAT_TYPE*)shared_area.data();
	FLOAT_TYPE* s_min_data = get_min_data_ptr<blockSize, FLOAT_TYPE, vector_alpha, updateDerivMinMax>(s_max_a_data);
	FLOAT_TYPE* s_max_data = get_max_data_ptr<blockSize, FLOAT_TYPE, vector_alpha, updateDerivMinMax>(s_max_a_data);

	const size_t tid = threadIdx.x;
	if (updateDerivMinMax || vector_alpha) {
		if (blockSize >= 64) {
			if (tid < 32) {
				if (updateDerivMinMax)dev_min(s_min_data[tid], s_min_data[tid + 32]);
				if (updateDerivMinMax)dev_max(s_max_data[tid], s_max_data[tid + 32]);
				if (vector_alpha) dev_max(s_max_a_data[tid], s_max_a_data[tid + 32]);
			}
		}
	}
}
template<size_t blockSize, bool vector_alpha, bool dim0, bool updateDerivMinMax>
void kernel_Dissipation_step32(
	const dim3& threadIdx,
	std::vector<uint8_t>& shared_area
) {
	FLOAT_TYPE* s_max_a_data = (FLOAT_TYPE*)shared_area.data();
	FLOAT_TYPE* s_min_data = get_min_data_ptr<blockSize, FLOAT_TYPE, vector_alpha, updateDerivMinMax>(s_max_a_data);
	FLOAT_TYPE* s_max_data = get_max_data_ptr<blockSize, FLOAT_TYPE, vector_alpha, updateDerivMinMax>(s_max_a_data);

	const size_t tid = threadIdx.x;
	if (updateDerivMinMax || vector_alpha) {
		if (blockSize >= 32) {
			if (tid < 16) {
				if (updateDerivMinMax)dev_min(s_min_data[tid], s_min_data[tid + 16]);
				if (updateDerivMinMax)dev_max(s_max_data[tid], s_max_data[tid + 16]);
				if (vector_alpha) dev_max(s_max_a_data[tid], s_max_a_data[tid + 16]);
			}
		}
	}
}
template<size_t blockSize, bool vector_alpha, bool dim0, bool updateDerivMinMax>
void kernel_Dissipation_step16(
	const dim3& threadIdx,
	std::vector<uint8_t>& shared_area
) {
	FLOAT_TYPE* s_max_a_data = (FLOAT_TYPE*)shared_area.data();
	FLOAT_TYPE* s_min_data = get_min_data_ptr<blockSize, FLOAT_TYPE, vector_alpha, updateDerivMinMax>(s_max_a_data);
	FLOAT_TYPE* s_max_data = get_max_data_ptr<blockSize, FLOAT_TYPE, vector_alpha, updateDerivMinMax>(s_max_a_data);

	const size_t tid = threadIdx.x;
	if (updateDerivMinMax || vector_alpha) {
		if (blockSize >= 16) {
			if (tid < 8) {
				if (updateDerivMinMax)dev_min(s_min_data[tid], s_min_data[tid + 8]);
				if (updateDerivMinMax)dev_max(s_max_data[tid], s_max_data[tid + 8]);
				if (vector_alpha) dev_max(s_max_a_data[tid], s_max_a_data[tid + 8]);
			}
		}
	}
}
template<size_t blockSize, bool vector_alpha, bool dim0, bool updateDerivMinMax>
void kernel_Dissipation_step8(
	const dim3& threadIdx,
	std::vector<uint8_t>& shared_area
) {
	FLOAT_TYPE* s_max_a_data = (FLOAT_TYPE*)shared_area.data();
	FLOAT_TYPE* s_min_data = get_min_data_ptr<blockSize, FLOAT_TYPE, vector_alpha, updateDerivMinMax>(s_max_a_data);
	FLOAT_TYPE* s_max_data = get_max_data_ptr<blockSize, FLOAT_TYPE, vector_alpha, updateDerivMinMax>(s_max_a_data);

	const size_t tid = threadIdx.x;
	if (updateDerivMinMax || vector_alpha) {
		if (blockSize >= 8) {
			if (tid < 4) {
				if (updateDerivMinMax)dev_min(s_min_data[tid], s_min_data[tid + 4]);
				if (updateDerivMinMax)dev_max(s_max_data[tid], s_max_data[tid + 4]);
				if (vector_alpha) dev_max(s_max_a_data[tid], s_max_a_data[tid + 4]);
			}
		}
	}
}
template<size_t blockSize, bool vector_alpha, bool dim0, bool updateDerivMinMax>
void kernel_Dissipation_step4(
	const dim3& threadIdx,
	std::vector<uint8_t>& shared_area
) {
	FLOAT_TYPE* s_max_a_data = (FLOAT_TYPE*)shared_area.data();
	FLOAT_TYPE* s_min_data = get_min_data_ptr<blockSize, FLOAT_TYPE, vector_alpha, updateDerivMinMax>(s_max_a_data);
	FLOAT_TYPE* s_max_data = get_max_data_ptr<blockSize, FLOAT_TYPE, vector_alpha, updateDerivMinMax>(s_max_a_data);

	const size_t tid = threadIdx.x;
	if (updateDerivMinMax || vector_alpha) {
		if (blockSize >= 4) {
			if (tid < 2) {
				if (updateDerivMinMax)dev_min(s_min_data[tid], s_min_data[tid + 2]);
				if (updateDerivMinMax)dev_max(s_max_data[tid], s_max_data[tid + 2]);
				if (vector_alpha) dev_max(s_max_a_data[tid], s_max_a_data[tid + 2]);
			}
		}
	}
}
template<size_t blockSize, bool vector_alpha, bool dim0, bool updateDerivMinMax>
void kernel_Dissipation_step2(
	const dim3& threadIdx,
	std::vector<uint8_t>& shared_area
) {
	FLOAT_TYPE* s_max_a_data = (FLOAT_TYPE*)shared_area.data();
	FLOAT_TYPE* s_min_data = get_min_data_ptr<blockSize, FLOAT_TYPE, vector_alpha, updateDerivMinMax>(s_max_a_data);
	FLOAT_TYPE* s_max_data = get_max_data_ptr<blockSize, FLOAT_TYPE, vector_alpha, updateDerivMinMax>(s_max_a_data);

	const size_t tid = threadIdx.x;
	if (updateDerivMinMax || vector_alpha) {
		if (blockSize >= 2) {
			if (tid < 1) {
				if (updateDerivMinMax)dev_min(s_min_data[tid], s_min_data[tid + 1]);
				if (updateDerivMinMax)dev_max(s_max_data[tid], s_max_data[tid + 1]);
				if (vector_alpha) dev_max(s_max_a_data[tid], s_max_a_data[tid + 1]);
			}
		}
	}
}
template<size_t blockSize, bool vector_alpha, bool dim0, bool updateDerivMinMax>
void kernel_Dissipation_step1(
	FLOAT_TYPE  *g_dst_min_ptr,
	FLOAT_TYPE  *g_dst_max_ptr,
	FLOAT_TYPE  *g_dst_max_alpha_ptr,
	const dim3& blockIdx,
	const dim3& threadIdx,
	std::vector<uint8_t>& shared_area
) {
	FLOAT_TYPE* s_max_a_data = (FLOAT_TYPE*)shared_area.data();
	FLOAT_TYPE* s_min_data = get_min_data_ptr<blockSize, FLOAT_TYPE, vector_alpha, updateDerivMinMax>(s_max_a_data);
	FLOAT_TYPE* s_max_data = get_max_data_ptr<blockSize, FLOAT_TYPE, vector_alpha, updateDerivMinMax>(s_max_a_data);

	const size_t tid = threadIdx.x;
	if (updateDerivMinMax || vector_alpha) {
		if (tid == 0) {
			if (updateDerivMinMax)g_dst_min_ptr[blockIdx.x] = s_min_data[0];
			if (updateDerivMinMax)g_dst_max_ptr[blockIdx.x] = s_max_data[0];
			if (vector_alpha) g_dst_max_alpha_ptr[blockIdx.x] = s_max_a_data[0];
		}
	}
}
template<size_t blockSize, bool vector_alpha, bool dim0, bool updateDerivMinMax>
void kernel_Dissipation_launcher(
	FLOAT_TYPE  *diss_ptr,
	FLOAT_TYPE  *min_ptr,
	FLOAT_TYPE  *max_ptr,
	FLOAT_TYPE  *max_alpha_ptr,
	const FLOAT_TYPE *deriv_l_ptr,
	const FLOAT_TYPE *deriv_r_ptr,
	const FLOAT_TYPE *a_ptr,
	const FLOAT_TYPE alpha0,
	const size_t loop_length,
	const dim3& num_of_blocks,
	const dim3& num_of_threads,
	std::vector<uint8_t>& shared_area
) {
	for (size_t blockIdx_x = 0; blockIdx_x < num_of_blocks.x; ++blockIdx_x) {
		for (size_t threadIdx_x = 0; threadIdx_x < num_of_threads.x; ++threadIdx_x) {
			dim3 threadIdx(threadIdx_x, 1, 1);
			dim3 blockIdx(blockIdx_x, 1, 1);
			kernel_Dissipation_start<blockSize, vector_alpha, dim0, updateDerivMinMax>(
				diss_ptr, deriv_l_ptr, deriv_r_ptr, a_ptr, alpha0, std::numeric_limits<FLOAT_TYPE>::max(),
				loop_length, num_of_threads, blockIdx, threadIdx, shared_area
				);
		}
		for (size_t threadIdx_x = 0; threadIdx_x < num_of_threads.x; ++threadIdx_x) {
			dim3 threadIdx(threadIdx_x, 1, 1);
			kernel_Dissipation_step1024<blockSize, vector_alpha, dim0, updateDerivMinMax>(
				threadIdx, shared_area
				);
		}
		for (size_t threadIdx_x = 0; threadIdx_x < num_of_threads.x; ++threadIdx_x) {
			dim3 threadIdx(threadIdx_x, 1, 1);
			kernel_Dissipation_step512<blockSize, vector_alpha, dim0, updateDerivMinMax>(
				threadIdx, shared_area
				);
		}
		for (size_t threadIdx_x = 0; threadIdx_x < num_of_threads.x; ++threadIdx_x) {
			dim3 threadIdx(threadIdx_x, 1, 1);
			kernel_Dissipation_step256<blockSize, vector_alpha, dim0, updateDerivMinMax>(
				threadIdx, shared_area
				);
		}
		for (size_t threadIdx_x = 0; threadIdx_x < num_of_threads.x; ++threadIdx_x) {
			dim3 threadIdx(threadIdx_x, 1, 1);
			dim3 blockIdx(blockIdx_x, 1, 1);
			kernel_Dissipation_step128<blockSize, vector_alpha, dim0, updateDerivMinMax>(
				threadIdx, shared_area
				);
		}
		for (size_t threadIdx_x = 0; threadIdx_x < num_of_threads.x; ++threadIdx_x) {
			dim3 threadIdx(threadIdx_x, 1, 1);
			dim3 blockIdx(blockIdx_x, 1, 1);
			kernel_Dissipation_step64<blockSize, vector_alpha, dim0, updateDerivMinMax>(
				threadIdx, shared_area
				);
		}
		for (size_t threadIdx_x = 0; threadIdx_x < num_of_threads.x; ++threadIdx_x) {
			dim3 threadIdx(threadIdx_x, 1, 1);
			dim3 blockIdx(blockIdx_x, 1, 1);
			kernel_Dissipation_step32<blockSize, vector_alpha, dim0, updateDerivMinMax>(
				threadIdx, shared_area
				);
		}
		for (size_t threadIdx_x = 0; threadIdx_x < num_of_threads.x; ++threadIdx_x) {
			dim3 threadIdx(threadIdx_x, 1, 1);
			dim3 blockIdx(blockIdx_x, 1, 1);
			kernel_Dissipation_step16<blockSize, vector_alpha, dim0, updateDerivMinMax>(
				threadIdx, shared_area
				);
		}
		for (size_t threadIdx_x = 0; threadIdx_x < num_of_threads.x; ++threadIdx_x) {
			dim3 threadIdx(threadIdx_x, 1, 1);
			dim3 blockIdx(blockIdx_x, 1, 1);
			kernel_Dissipation_step8<blockSize, vector_alpha, dim0, updateDerivMinMax>(
				threadIdx, shared_area
				);
		}
		for (size_t threadIdx_x = 0; threadIdx_x < num_of_threads.x; ++threadIdx_x) {
			dim3 threadIdx(threadIdx_x, 1, 1);
			dim3 blockIdx(blockIdx_x, 1, 1);
			kernel_Dissipation_step4<blockSize, vector_alpha, dim0, updateDerivMinMax>(
				threadIdx, shared_area
				);
		}
		for (size_t threadIdx_x = 0; threadIdx_x < num_of_threads.x; ++threadIdx_x) {
			dim3 threadIdx(threadIdx_x, 1, 1);
			dim3 blockIdx(blockIdx_x, 1, 1);
			kernel_Dissipation_step2<blockSize, vector_alpha, dim0, updateDerivMinMax>(
				threadIdx, shared_area
				);
		}
		for (size_t threadIdx_x = 0; threadIdx_x < num_of_threads.x; ++threadIdx_x) {
			dim3 threadIdx(threadIdx_x, 1, 1);
			dim3 blockIdx(blockIdx_x, 1, 1);
			kernel_Dissipation_step1<blockSize, vector_alpha, dim0, updateDerivMinMax>(
				min_ptr, max_ptr, max_alpha_ptr,
				blockIdx, threadIdx, shared_area
				);
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
	const dim3 num_of_threads

) {
	const size_t num_of_threads_x = num_of_threads.x;
	const int num_of_shared_memory_buffer = (num_of_threads_x <= 32) ? 2 : 1;
	size_t num_of_shared_memories = 0;
	if (vector_alpha) num_of_shared_memories += 1;
	if (updateDerivMinMax) num_of_shared_memories += 2;

	const size_t smemSize = num_of_shared_memory_buffer * num_of_threads_x * sizeof(FLOAT_TYPE) * num_of_shared_memories;

	std::vector<uint8_t> shared_area(smemSize);
	dim3 gridDim = num_of_threads;
	switch (num_of_threads_x) {
	case 1024:
		kernel_Dissipation_launcher<1024, vector_alpha, dim0, updateDerivMinMax>(
			diss_ptr, min_ptr, max_ptr, max_alpha_ptr, deriv_l_ptr, deriv_r_ptr, a_ptr, alpha0, 
			loop_length, num_of_blocks, num_of_threads, shared_area
			);
		break;
	case 512:
		kernel_Dissipation_launcher<512, vector_alpha, dim0, updateDerivMinMax>(
			diss_ptr, min_ptr, max_ptr, max_alpha_ptr, deriv_l_ptr, deriv_r_ptr, a_ptr, alpha0, 
			loop_length, num_of_blocks, num_of_threads, shared_area
			);
		break;
	case 256:
		kernel_Dissipation_launcher<256, vector_alpha, dim0, updateDerivMinMax>(
			diss_ptr, min_ptr, max_ptr, max_alpha_ptr, deriv_l_ptr, deriv_r_ptr, a_ptr, alpha0, 
			loop_length, num_of_blocks, num_of_threads, shared_area
			);
		break;
	case 128:
		kernel_Dissipation_launcher<128, vector_alpha, dim0, updateDerivMinMax>(
			diss_ptr, min_ptr, max_ptr, max_alpha_ptr, deriv_l_ptr, deriv_r_ptr, a_ptr, alpha0, 
			loop_length, num_of_blocks, num_of_threads, shared_area
			);
		break;
	case 64:
		kernel_Dissipation_launcher<64, vector_alpha, dim0, updateDerivMinMax>(
			diss_ptr, min_ptr, max_ptr, max_alpha_ptr, deriv_l_ptr, deriv_r_ptr, a_ptr, alpha0, 
			loop_length, num_of_blocks, num_of_threads, shared_area
			);
		break;
	case 32:
		kernel_Dissipation_launcher<32, vector_alpha, dim0, updateDerivMinMax>(
			diss_ptr, min_ptr, max_ptr, max_alpha_ptr, deriv_l_ptr, deriv_r_ptr, a_ptr, alpha0, 
			loop_length, num_of_blocks, num_of_threads, shared_area
			);
		break;
	case 16:
		kernel_Dissipation_launcher<16, vector_alpha, dim0, updateDerivMinMax>(
			diss_ptr, min_ptr, max_ptr, max_alpha_ptr, deriv_l_ptr, deriv_r_ptr, a_ptr, alpha0, 
			loop_length, num_of_blocks, num_of_threads, shared_area
			);
		break;
	case 8:
		kernel_Dissipation_launcher<8, vector_alpha, dim0, updateDerivMinMax>(
			diss_ptr, min_ptr, max_ptr, max_alpha_ptr, deriv_l_ptr, deriv_r_ptr, a_ptr, alpha0, 
			loop_length, num_of_blocks, num_of_threads, shared_area
			);
		break;
	case 4:
		kernel_Dissipation_launcher<4, vector_alpha, dim0, updateDerivMinMax>(
			diss_ptr, min_ptr, max_ptr, max_alpha_ptr, deriv_l_ptr, deriv_r_ptr, a_ptr, alpha0, 
			loop_length, num_of_blocks, num_of_threads, shared_area
			);
		break;
	case 2:
		kernel_Dissipation_launcher<2, vector_alpha, dim0, updateDerivMinMax>(
			diss_ptr, min_ptr, max_ptr, max_alpha_ptr, deriv_l_ptr, deriv_r_ptr, a_ptr, alpha0, 
			loop_length, num_of_blocks, num_of_threads, shared_area
			);
		break;
	case 1:
		kernel_Dissipation_launcher<1, vector_alpha, dim0, updateDerivMinMax>(
			diss_ptr, min_ptr, max_ptr, max_alpha_ptr, deriv_l_ptr, deriv_r_ptr, a_ptr, alpha0, 
			loop_length, num_of_blocks, num_of_threads, shared_area
			);
		break;
	default:
		std::cerr << "Illigal num_of_threads_x: " << num_of_threads_x << std::endl;
		break;
	}
}
void ArtificialDissipationGLF_execute_cuda(
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
					loop_length, num_of_blocks, num_of_threads);
			}
			else {
				launch_kernel_Dissipation<true, true, false>(
					diss_ptr, min_ptr, max_ptr, max_alpha_ptr, deriv_l_ptr, deriv_r_ptr, alpha_ptr, 0,
					loop_length, num_of_blocks, num_of_threads);
			}
		}
		else {
			if (updateDerivMinMax) {
				launch_kernel_Dissipation<true, false, true>(
					diss_ptr, min_ptr, max_ptr, max_alpha_ptr, deriv_l_ptr, deriv_r_ptr, alpha_ptr, 0,
					loop_length, num_of_blocks, num_of_threads);
			}
			else {
				launch_kernel_Dissipation<true, false, false>(
					diss_ptr, min_ptr, max_ptr, max_alpha_ptr, deriv_l_ptr, deriv_r_ptr, alpha_ptr, 0,
					loop_length, num_of_blocks, num_of_threads);
			}
		}
	}
	else {
		const FLOAT_TYPE alpha = alpha_ptr[0];
		if (dimension == 0) {
			if (updateDerivMinMax) {
				launch_kernel_Dissipation<false, true, true>(
					diss_ptr, min_ptr, max_ptr, max_alpha_ptr, deriv_l_ptr, deriv_r_ptr, alpha_ptr, alpha,
					loop_length, num_of_blocks, num_of_threads);
			}
			else {
				launch_kernel_Dissipation<false, true, false>(
					diss_ptr, min_ptr, max_ptr, max_alpha_ptr, deriv_l_ptr, deriv_r_ptr, alpha_ptr, alpha,
					loop_length, num_of_blocks, num_of_threads);
			}
		}
		else {
			if (updateDerivMinMax) {
				launch_kernel_Dissipation<false, false, true>(
					diss_ptr, min_ptr, max_ptr, max_alpha_ptr, deriv_l_ptr, deriv_r_ptr, alpha_ptr, alpha,
					loop_length, num_of_blocks, num_of_threads);
			}
			else {
				launch_kernel_Dissipation<false, false, false>(
					diss_ptr, min_ptr, max_ptr, max_alpha_ptr, deriv_l_ptr, deriv_r_ptr, alpha_ptr, alpha,
					loop_length, num_of_blocks, num_of_threads);
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
	const beacls::UVec &tmp_max_alpha_cuda_uvec,
	beacls::UVec &tmp_min_cpu_uvec,
	beacls::UVec &tmp_max_cpu_uvec,
	beacls::UVec &tmp_max_alpha_cpu_uvec,
	const FLOAT_TYPE dxInv,
	const bool updateDerivMinMax
) {
	beacls::UVecType type = beacls::UVecType_Vector;
	const size_t num_of_blocks_x = tmp_min_cpu_uvec.size();
	if (updateDerivMinMax) {
		tmp_min_cuda_uvec.convertTo(tmp_min_cpu_uvec, type);
		tmp_max_cuda_uvec.convertTo(tmp_max_cpu_uvec, type);
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
		tmp_max_alpha_cuda_uvec.convertTo(tmp_max_alpha_cpu_uvec, type);
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
#endif /* !defined(WITH_GPU) */
