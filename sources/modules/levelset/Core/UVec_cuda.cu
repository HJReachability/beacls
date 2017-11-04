#include <cuda_runtime.h>
#include <typedef.hpp>
#if defined(WITH_GPU)
typedef unsigned char uint8_t;
typedef	signed char int8_t;
typedef	unsigned short uint16_t;
typedef signed short int16_t;
typedef	unsigned int uint32_t;
//typedef signed long int32_t;
//typedef	unsigned long int uint64_t;
//typedef signed long long int64_t;
#include <Core/UVec.hpp>
#include <Core/CudaStream.hpp>
#include "UVec_impl_cuda.hpp"
#include <cuda_macro.hpp>

namespace beacls
{
void* allocateCudaMem(const size_t s)
{
	void* ptr = NULL;
	cudaMalloc((void**)&ptr,s);
	return ptr;

}
void freeCudaMem(void* ptr)
{
	if(ptr) cudaFree(ptr);
}
void copyCudaDeviceToHost(void* dst, const void* src, size_t s)
{
	cudaMemcpy(dst,src,s,cudaMemcpyDeviceToHost);
}
void copyCudaDeviceToDevice(void* dst, const void* src, size_t s)
{
	cudaMemcpy(dst,src,s,cudaMemcpyDeviceToDevice);
}
void copyCudaHostToDevice(void* dst, const void* src, size_t s)
{
	cudaMemcpy(dst,src,s,cudaMemcpyHostToDevice);
}
beacls::CudaStream_impl::CudaStream_impl() {
	cudaStreamCreate(&stream);
//	cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
}
beacls::CudaStream_impl::~CudaStream_impl() {
	if (stream) {
		cudaStreamDestroy(stream);
	}
}
cudaStream_t beacls::CudaStream_impl::get_stream() {
	return stream;
}
beacls::CudaStream::CudaStream() {
	pimpl = new CudaStream_impl();
}
beacls::CudaStream::~CudaStream() {
	delete pimpl;
}
cudaStream_t beacls::CudaStream::get_stream() {
	if (pimpl) return pimpl->get_stream();
	else return NULL;
}
cudaStream_t get_stream(const beacls::UVec& src) {
	if (beacls::is_cuda(src)) {
		beacls::CudaStream* cudaStream = src.get_cudaStream();
		if (cudaStream) return cudaStream->get_stream();
		else return NULL;
	}
	else return NULL;
}

void copyCudaDeviceToHostAsync(void* dst, const void* src, const size_t s, beacls::CudaStream* cudaStream) {
	if (cudaStream) {
		cudaStream_t stream = cudaStream->get_stream();
		cudaMemcpyAsync(dst, src, s, cudaMemcpyDeviceToHost, stream);
	}
}
void copyCudaDeviceToDeviceAsync(void* dst, const void* src, const size_t s, beacls::CudaStream* cudaStream) {
	if (cudaStream) {
		cudaStream_t stream = cudaStream->get_stream();
		cudaMemcpyAsync(dst, src, s, cudaMemcpyDeviceToDevice, stream);
	}
}
void copyCudaHostToDeviceAsync(void* dst, const void* src, const size_t s, beacls::CudaStream* cudaStream) {
	if (cudaStream) {
		cudaStream_t stream = cudaStream->get_stream();
		cudaMemcpyAsync(dst, src, s, cudaMemcpyHostToDevice, stream);
	}
}
void synchronizeCuda(beacls::CudaStream* cudaStream) {
	if (cudaStream) {
		cudaStream_t stream = cudaStream->get_stream();
		cudaStreamSynchronize(stream);
	}
}
template<typename T>
__global__ static
void kernel_FillCudaMemory(
	T* dst_ptr,
	const T val,
	const size_t loop_length
) {
	const size_t tid = threadIdx.x;
	size_t index = (blockIdx.x * blockDim.x + tid);
	const size_t gridSize = blockDim.x * gridDim.x;
	while (index < loop_length) {
		dst_ptr[index] = val;
		index += gridSize;
	}
}


template <typename T>
void fillCudaMemory_template(T* dst_raw_ptr, const T val, size_t length) {
	size_t num_of_threads_x;
	size_t num_of_blocks_x;
	get_cuda_thread_size_1d<size_t>(
		num_of_threads_x,
		num_of_blocks_x,
		length
		);
	dim3 num_of_blocks((unsigned int)num_of_blocks_x, 1);
	dim3 num_of_threads((unsigned int)num_of_threads_x, 1, 1);
	kernel_FillCudaMemory<T> << <num_of_blocks, num_of_threads, 0 >> >(
		dst_raw_ptr, val, length
		);
}
void fillCudaMemory(uint8_t* dst, const uint8_t val, size_t s)
{
	fillCudaMemory_template<uint8_t>(dst,val,s);
}
void fillCudaMemory(int8_t* dst, const int8_t val, size_t s)
{
	fillCudaMemory_template<int8_t>(dst,val,s);
}
void fillCudaMemory(uint16_t* dst, const uint16_t val, size_t s)
{
	fillCudaMemory_template<uint16_t>(dst,val,s);
}
void fillCudaMemory(int16_t* dst, const int16_t val, size_t s)
{
	fillCudaMemory_template<int16_t>(dst,val,s);
}
void fillCudaMemory(uint32_t* dst, const uint32_t val, size_t s)
{
	fillCudaMemory_template<uint32_t>(dst,val,s);
}
void fillCudaMemory(int32_t* dst, const int32_t val, size_t s)
{
	fillCudaMemory_template<int32_t>(dst,val,s);
}
void fillCudaMemory(uint64_t* dst, const uint64_t val, size_t s)
{
	fillCudaMemory_template<uint64_t>(dst,val,s);
}
void fillCudaMemory(int64_t* dst, const int64_t val, size_t s)
{
	fillCudaMemory_template<int64_t>(dst,val,s);
}

void fillCudaMemory(double* dst, const double val, size_t s)
{
	fillCudaMemory_template<double>(dst,val,s);
}
void fillCudaMemory(float* dst, const float val, size_t s)
{
	fillCudaMemory_template<float>(dst,val,s);
}

template<typename T>
__global__ static
void kernel_Average(
	T* dst_ptr,
	const T* lhs_ptr,
	const T* rhs_ptr,
	const size_t loop_length
) {
	const size_t tid = threadIdx.x;
	size_t index = (blockIdx.x * blockDim.x + tid);
	const size_t gridSize = blockDim.x * gridDim.x;
	while (index < loop_length) {
		const T lhs = lhs_ptr[index];
		const T rhs = rhs_ptr[index];
		dst_ptr[index] = (rhs + lhs) / 2;
		index += gridSize;
	}
}

template <typename T>
void average_template(void* dst_raw_ptr, const void* src1_raw_ptr, const void* src2_raw_ptr, const size_t length, cudaStream_t stream) {
	size_t num_of_threads_x;
	size_t num_of_blocks_x;
	get_cuda_thread_size_1d<size_t>(
		num_of_threads_x,
		num_of_blocks_x,
		length
		);
	dim3 num_of_blocks((unsigned int)num_of_blocks_x, 1);
	dim3 num_of_threads((unsigned int)num_of_threads_x, 1, 1);
	kernel_Average<T><<<num_of_blocks, num_of_threads, 0, stream>>>(
		(T*)dst_raw_ptr, (const T*)src1_raw_ptr, (const T*)src2_raw_ptr, length
		);
}
void cudaAverage(beacls::UVec& dst_uvec, const beacls::UVec& src1, const beacls::UVec& src2) {
	const size_t length = src1.size();
	const UVecDepth d = src1.depth();
	beacls::reallocateAsSrc(dst_uvec, src1);
	dst_uvec.set_cudaStream(src1.get_cudaStream());
	cudaStream_t stream = beacls::get_stream(dst_uvec);
	FLOAT_TYPE* dst_ptr = beacls::UVec_<FLOAT_TYPE>(dst_uvec).ptr();
	const FLOAT_TYPE* src1_ptr = beacls::UVec_<FLOAT_TYPE>(src1).ptr();
	const FLOAT_TYPE* src2_ptr = beacls::UVec_<FLOAT_TYPE>(src2).ptr();

	switch (d) {
	case UVecDepth_Invalid:
	case UVecDepth_User:
	default:
		break;
	case UVecDepth_8U:
		average_template<uint8_t>(dst_ptr, src1_ptr, src2_ptr, length, stream);
		break;
	case UVecDepth_8S:
		average_template<int8_t>(dst_ptr, src1_ptr, src2_ptr, length, stream);
		break;
	case UVecDepth_16U:
		average_template<uint16_t>(dst_ptr, src1_ptr, src2_ptr, length, stream);
		break;
	case UVecDepth_16S:
		average_template<int16_t>(dst_ptr, src1_ptr, src2_ptr, length, stream);	
		break;
	case UVecDepth_32S:
		average_template<int32_t>(dst_ptr, src1_ptr, src2_ptr, length, stream);
		break;
	case UVecDepth_32F:
		average_template<float>(dst_ptr, src1_ptr, src2_ptr, length, stream);
		break;
	case UVecDepth_64F:
		average_template<double>(dst_ptr, src1_ptr, src2_ptr, length, stream);
		break;
	case UVecDepth_32U:
		average_template<uint32_t>(dst_ptr, src1_ptr, src2_ptr, length, stream);
		break;
	case UVecDepth_64U:
		average_template<uint64_t>(dst_ptr, src1_ptr, src2_ptr, length, stream);
		break;
	case UVecDepth_64S:
		average_template<int64_t>(dst_ptr, src1_ptr, src2_ptr, length, stream);
		break;
	}
}
size_t get_minimum_global_memory_in_devices_impl() {
	int device_count = get_num_of_gpus_impl();
	size_t minimum_global_memory_in_devices = std::numeric_limits<size_t>::max();
	for (size_t id = 0; id < device_count; ++id) {
		struct cudaDeviceProp prop;
		cudaError_t err;
		err = cudaGetDeviceProperties(&prop, id);
		if (err) return 0;
		if (minimum_global_memory_in_devices > prop.totalGlobalMem) minimum_global_memory_in_devices = prop.totalGlobalMem;
	}
	return minimum_global_memory_in_devices;
}
int get_num_of_gpus_impl() {
	int device_count=0;
	cudaError_t err;
	err = cudaGetDeviceCount(&device_count);
	if(err) {
		return 0;
	}
	return device_count;
}
void set_gpu_id_impl(const int id) {
	cudaSetDevice(id);
}

}	// beacls
#endif /* defined(WITH_GPU) */
