#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <thread>
#include <Core/UVec.hpp>
#include <Core/CudaStream.hpp>
#include "UVec_impl_cuda.hpp"
//#define DUMMY_WAIT
#if defined(DUMMY_WAIT)
inline
void sleep_microseconds(const size_t us) {
	std::this_thread::sleep_for(std::chrono::microseconds(us));
}
#else
inline
void sleep_microseconds(const size_t) {
}
#endif
#if !defined(WITH_GPU)
namespace beacls
{
void* allocateCudaMem(const size_t s)
{
	sleep_microseconds(30);
	void* ptr = malloc(s);
	if(ptr) memset(ptr, 0xdeadbeaf, s);
	return ptr;
}
void freeCudaMem(void* ptr)
{
	sleep_microseconds(30);
	if (ptr) free(ptr);
}
void copyCudaDeviceToHost(void* dst, const void* src, const size_t s)
{
	sleep_microseconds(30);
	std::memcpy(dst, src, s);
}
void copyCudaDeviceToDevice(void* dst, const void* src, const size_t s)
{
	sleep_microseconds(30);
	std::memcpy(dst, src, s);
}
void copyCudaHostToDevice(void* dst, const void* src, const size_t s)
{
	sleep_microseconds(30);
	std::memcpy(dst, src, s);
}
beacls::CudaStream_impl::CudaStream_impl() {
}
beacls::CudaStream_impl::~CudaStream_impl() {
	std::for_each(ths.begin(), ths.end(), [](auto& rhs) {
		if (rhs) {
			if (rhs->joinable()) rhs->join();
			delete rhs;
		}
	});
}
std::thread* beacls::CudaStream_impl::pop_thread() {
	if (!ths.empty()) {
		std::thread* th = ths.front();
		ths.pop_front();
		return th;
	}
	return NULL;
}
void beacls::CudaStream_impl::push_thread(std::thread* th) {
	ths.push_back(th);
}

beacls::CudaStream::CudaStream() {
	pimpl = new CudaStream_impl();
}
beacls::CudaStream::~CudaStream() {
	delete pimpl;
}
std::thread* beacls::CudaStream::pop_thread() {
	if (pimpl) return pimpl->pop_thread();
	else return NULL;
}
void beacls::CudaStream::push_thread(std::thread* th) {
	if (pimpl) pimpl->push_thread(th);
	else return;
}

void copyCudaDeviceToHostAsync(void* dst, const void* src, const size_t s, beacls::CudaStream* cudaStream) {
	if (cudaStream) {
		std::thread* th = new std::thread(std::memcpy, dst, src, s);
		cudaStream->push_thread(th);
	}
}
void copyCudaDeviceToDeviceAsync(void* dst, const void* src, const size_t s, beacls::CudaStream* cudaStream) {
	if (cudaStream) {
		std::thread* th = new std::thread(std::memcpy, dst, src, s);
		cudaStream->push_thread(th);
	}
}
void copyCudaHostToDeviceAsync(void* dst, const void* src, const size_t s, beacls::CudaStream* cudaStream) {
	if (cudaStream) {
		std::thread* th = new std::thread(std::memcpy, dst, src, s);
		cudaStream->push_thread(th);
	}
}
void synchronizeCuda(beacls::CudaStream* cudaStream) {
	sleep_microseconds(30);
	if (cudaStream) {
		std::thread* th = cudaStream->pop_thread();
		if (th) {
			if (th->joinable()) {
				th->join();
			}
			delete th;
		}
	}
}


void fillCudaMemory(uint8_t* dst, const uint8_t val, size_t s)
{
	sleep_microseconds(30);
	std::fill(dst, dst + s, val);
}
void fillCudaMemory(int8_t* dst, const int8_t val, size_t s)
{
	sleep_microseconds(30);
	std::fill(dst, dst + s, val);
}
void fillCudaMemory(uint16_t* dst, const uint16_t val, size_t s)
{
	sleep_microseconds(30);
	std::fill(dst, dst + s, val);
}
void fillCudaMemory(int16_t* dst, const int16_t val, size_t s)
{
	sleep_microseconds(30);
	std::fill(dst, dst + s, val);
}
void fillCudaMemory(uint32_t* dst, const uint32_t val, size_t s)
{
	sleep_microseconds(30);
	std::fill(dst, dst + s, val);
}
void fillCudaMemory(int32_t* dst, const int32_t val, size_t s)
{
	sleep_microseconds(30);
	std::fill(dst, dst + s, val);
}
void fillCudaMemory(uint64_t* dst, const uint64_t val, size_t s)
{
	sleep_microseconds(30);
	std::fill(dst, dst + s, val);
}
void fillCudaMemory(int64_t* dst, const int64_t val, size_t s)
{
	sleep_microseconds(30);
	std::fill(dst, dst + s, val);
}
void fillCudaMemory(double* dst, const double val, size_t s)
{
	sleep_microseconds(30);
	std::fill(dst, dst+s,val);
}
void fillCudaMemory(float* dst, const float val, size_t s)
{
	sleep_microseconds(30);
	std::fill(dst, dst+s,val);
}

template<typename T>
void cudaAgerage_template(T* dst, const T* src1_ptr, const T* src2_ptr, const size_t length) {
	for (size_t i = 0; i < length; ++i) {
		*dst++ = (*src1_ptr++ + *src2_ptr++) / 2;
	}
}

void cudaAverage(beacls::UVec& dst_uvec, const beacls::UVec& src1, const beacls::UVec& src2) {
	beacls::synchronizeUVec(src1);
	beacls::synchronizeUVec(src2);
	beacls::reallocateAsSrc(dst_uvec, src1);
	dst_uvec.set_cudaStream(src1.get_cudaStream());
	FLOAT_TYPE* dst_ptr = beacls::UVec_<FLOAT_TYPE>(dst_uvec).ptr();
	const FLOAT_TYPE* src1_ptr = beacls::UVec_<FLOAT_TYPE>(src1).ptr();
	const FLOAT_TYPE* src2_ptr = beacls::UVec_<FLOAT_TYPE>(src2).ptr();
	const size_t length = src1.size();
	const UVecDepth d = src1.depth();
	switch (d) {
	case UVecDepth_Invalid:
	case UVecDepth_User:
	default:
		break;
	case UVecDepth_8U:
		cudaAgerage_template((uint8_t*)dst_ptr, (const uint8_t*)src1_ptr, (const uint8_t*)src2_ptr, length);
		break;
	case UVecDepth_8S:
		cudaAgerage_template((int8_t*)dst_ptr, (const int8_t*)src1_ptr, (const int8_t*)src2_ptr, length);
		break;
	case UVecDepth_16U:
		cudaAgerage_template((uint16_t*)dst_ptr, (const uint16_t*)src1_ptr, (const uint16_t*)src2_ptr, length);
		break;
	case UVecDepth_16S:
		cudaAgerage_template((int16_t*)dst_ptr, (const int16_t*)src1_ptr, (const int16_t*)src2_ptr, length);
		break;
	case UVecDepth_32S:
		cudaAgerage_template((int32_t*)dst_ptr, (const int32_t*)src1_ptr, (const int32_t*)src2_ptr, length);
		break;
	case UVecDepth_32F:
		cudaAgerage_template((float*)dst_ptr, (const float*)src1_ptr, (const float*)src2_ptr, length);
		break;
	case UVecDepth_64F:
		cudaAgerage_template((double*)dst_ptr, (const double*)src1_ptr, (const double*)src2_ptr, length);
		break;
	case UVecDepth_32U:
		cudaAgerage_template((uint32_t*)dst_ptr, (const uint32_t*)src1_ptr, (const uint32_t*)src2_ptr, length);
		break;
	case UVecDepth_64U:
		cudaAgerage_template((uint64_t*)dst_ptr, (const uint64_t*)src1_ptr, (const uint64_t*)src2_ptr, length);
		break;
	case UVecDepth_64S:
		cudaAgerage_template((int64_t*)dst_ptr, (const int64_t*)src1_ptr, (const int64_t*)src2_ptr, length);
		break;

	}
}
size_t get_minimum_global_memory_in_devices_impl() {
	return 1024*1024*1024*4;
}
int get_num_of_gpus_impl() {
	return 1;
}
void set_gpu_id_impl(const int) {
}

}	// beacls
#endif /* !defined(WITH_GPU) */

