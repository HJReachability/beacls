#ifndef __UVec_impl_cuda_hpp__
#define __UVec_impl_cuda_hpp__

#include <typedef.hpp>
#if defined(WITH_GPU)
#include <cuda_runtime.h>
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef unsigned short uint16_t;
typedef signed short int16_t;
typedef unsigned int uint32_t;
//typedef unsigned long int uint64_t;
#else
#include <thread>
#include <deque>
#endif
namespace beacls
{
	class CudaStream_impl {
#if defined(WITH_GPU)
		cudaStream_t stream;
	public:
		cudaStream_t get_stream();
#else
		std::deque<std::thread*> ths;
	public:
		std::thread* pop_thread();
		void push_thread(std::thread* th);
#endif

		CudaStream_impl();
		~CudaStream_impl();
	private:
		CudaStream_impl(const CudaStream_impl& rhs);
		CudaStream_impl& operator==(const CudaStream_impl& rhs);
	};


	void* allocateCudaMem(const size_t s);
	void freeCudaMem(void* ptr);
	void copyCudaDeviceToHost(void* dst, const void* src, const size_t s);
	void copyCudaDeviceToDevice(void* dst, const void* src, const size_t s);
	void copyCudaHostToDevice(void* dst, const void* src, const size_t s);
	void copyCudaDeviceToHostAsync(void* dst, const void* src, const size_t s, beacls::CudaStream* cudaStream);
	void copyCudaDeviceToDeviceAsync(void* dst, const void* src, const size_t s, beacls::CudaStream* cudaStream);
	void copyCudaHostToDeviceAsync(void* dst, const void* src, const size_t s, beacls::CudaStream* cudaStream);
	void cudaAverage(beacls::UVec& dst_uvec, const beacls::UVec& src1, const beacls::UVec& src2);
	void fillCudaMemory(uint8_t* dst, const uint8_t val, size_t s);
	void fillCudaMemory(int8_t* dst, const int8_t val, size_t s);
	void fillCudaMemory(uint16_t* dst, const uint16_t val, size_t s);
	void fillCudaMemory(int16_t* dst, const int16_t val, size_t s);
	void fillCudaMemory(uint32_t* dst, const uint32_t val, size_t s);
	void fillCudaMemory(int32_t* dst, const int32_t val, size_t s);
	void fillCudaMemory(uint64_t* dst, const uint64_t val, size_t s);
	void fillCudaMemory(int64_t* dst, const int64_t val, size_t s);
	void fillCudaMemory(double* dst, const double val, size_t s);
	void fillCudaMemory(float* dst, const float val, size_t s);
	int get_num_of_gpus_impl();
	void set_gpu_id_impl(const int id);
}	// beacls
#endif	/* __UVec_impl_cuda_hpp__ */

