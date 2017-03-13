#include <cuda_runtime.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/transform.h>
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
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

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

template <typename T>
void fillCudaMemory_template(T* dst_raw_ptr, const T val, size_t length) {
	thrust::device_ptr<T> dst_dev_ptr = thrust::device_pointer_cast((T*)dst_raw_ptr);
	thrust::fill(dst_dev_ptr, dst_dev_ptr + length, val);
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
struct AverageFunctor : public thrust::binary_function<const T, const T, T> {
	__host__ __device__
	T operator()(const T& rhs, const T& lhs) const
	{
		return (rhs + lhs) / 2;
	}
};

template <typename T>
void average_template(void* dst_raw_ptr, const void* src1_raw_ptr, const void* src2_raw_ptr, const size_t length, cudaStream_t stream) {
	thrust::device_ptr<T> dst_dev_ptr = thrust::device_pointer_cast((T*)dst_raw_ptr);
	thrust::device_ptr<const T> src1_dev_ptr = thrust::device_pointer_cast((const T*)src1_raw_ptr);
	thrust::device_ptr<const T> src2_dev_ptr = thrust::device_pointer_cast((const T*)src2_raw_ptr);
	thrust::transform(thrust::cuda::par.on(stream),
		src1_dev_ptr, src1_dev_ptr + length, src2_dev_ptr, dst_dev_ptr, AverageFunctor<T>());
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

}	// bears
#endif /* defined(WITH_GPU) */
