#ifndef __CudaStream_hpp__
#define __CudaStream_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#if defined(WITH_GPU)
#include <cuda_runtime.h>
#else
#include <thread>
#include <deque>
#endif
#include <Core/UVec.hpp>
namespace beacls
{
	class CudaStream_impl;
	class CudaStream {
	private:
		CudaStream_impl* pimpl;
	public:
#if defined(WITH_GPU)
		cudaStream_t get_stream();
#else
		std::thread* pop_thread();
		void push_thread(std::thread* th);
#endif
		CudaStream();
		~CudaStream();
	private:
		CudaStream(const CudaStream& rhs);
		CudaStream& operator==(const CudaStream& rhs);
	};
#if defined(WITH_GPU)
	PREFIX_VC_DLL
		cudaStream_t get_stream(const beacls::UVec& src);
#endif

};	// beacls
#endif	/* __CudaStream_hpp__ */

