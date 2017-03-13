#ifndef __UVec_hpp__
#define __UVec_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#include <typedef.hpp>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <thread>
namespace beacls
{
class UVec_impl;
class CudaStream;
static inline size_t depth_to_byte(const UVecDepth depth)  {
	size_t byte = 0;
	switch (depth) {
	default:
	case UVecDepth_User:
	case UVecDepth_Invalid:
		break;
	case UVecDepth_8U:
		byte = 1;
		break;
	case UVecDepth_8S:
		byte = 1;
		break;
	case UVecDepth_16U:
		byte = 2;
		break;
	case UVecDepth_16S:
		byte = 2;
		break;
	case UVecDepth_32S:
		byte = 4;
		break;
	case UVecDepth_32F:
		byte = 4;
		break;
	case UVecDepth_64F:
		byte = 8;
		break;
	case UVecDepth_32U:
		byte = 4;
		break;
	case UVecDepth_64U:
		byte = 8;
		break;
	case UVecDepth_64S:
		byte = 8;
		break;
	}
	return byte;
}

template<typename T>
UVecDepth type_to_depth(T a = 0);

template<> inline
UVecDepth type_to_depth(uint8_t) { return UVecDepth_8U; }
template<> inline
UVecDepth type_to_depth(int8_t) { return UVecDepth_8S; }
template<> inline
UVecDepth type_to_depth(uint16_t) { return UVecDepth_16U; }
template<> inline
UVecDepth type_to_depth(int16_t) { return UVecDepth_16S; }
template<> inline
UVecDepth type_to_depth(uint32_t) { return UVecDepth_32U; }
template<> inline
UVecDepth type_to_depth(int32_t) { return UVecDepth_32S; }
template<> inline
UVecDepth type_to_depth(float) { return UVecDepth_32F; }
template<> inline
UVecDepth type_to_depth(double) { return UVecDepth_64F; }
template<> inline
UVecDepth type_to_depth(uint64_t) { return UVecDepth_64U; }
template<> inline
UVecDepth type_to_depth(int64_t) { return UVecDepth_64S; }

class UVec;
PREFIX_VC_DLL
void copyDevicePtrToUVec(UVec& dst, const void* src, const size_t s);
PREFIX_VC_DLL
void copyHostPtrToUVec(UVec& dst, const void* src, const size_t s);
PREFIX_VC_DLL
void copyUVecToDevicePtr(void* dst, const UVec& src);
PREFIX_VC_DLL
void copyUVecToHost(void* dst, const UVec& src);
PREFIX_VC_DLL
void copyDevicePtrToUVecAsync(UVec& dst, const void* src, const size_t s);
PREFIX_VC_DLL
void copyHostPtrToUVecAsync(UVec& dst, const void* src, const size_t s);
PREFIX_VC_DLL
void copyUVecToDevicePtrAsync(void* dst, const UVec& src);
PREFIX_VC_DLL
void copyUVecToHostAsync(void* dst, const UVec& src);
PREFIX_VC_DLL
void synchronizeUVec(const UVec& src);

/*
	@brief Unified vector class
*/
class UVec {
	friend void copyDevicePtrToUVec(UVec& dst, const void* src, const size_t s);
	friend void copyHostPtrToUVec(UVec& dst, const void* src, const size_t s);
	friend void copyUVecToDevicePtr(void* dst, const UVec& src);
	friend void copyUVecToHost(void* dst, const UVec& src);
	friend void copyDevicePtrToUVecAsync(UVec& dst, const void* src, const size_t s);
	friend void copyHostPtrToUVecAsync(UVec& dst, const void* src, const size_t s);
	friend void copyUVecToDevicePtrAsync(void* dst, const UVec& src);
	friend void copyUVecToHostAsync(void* dst, const UVec& src);
	friend void synchronizeUVec(const UVec& src);

public:
	PREFIX_VC_DLL
		size_t size() const;
	PREFIX_VC_DLL
		UVecDepth depth() const;
	PREFIX_VC_DLL
		UVecType type() const;
	PREFIX_VC_DLL
		void convertTo(UVec& dst, UVecDepth depth, 
			UVecType type = UVecType_Invalid, 
			const double alpha=1., const double beta=0.) const;
	PREFIX_VC_DLL
		void convertTo(UVec& dst, 
			UVecType type = UVecType_Invalid,
			const double alpha = 1., const double beta = 0.) const;
	PREFIX_VC_DLL
		void resize(const size_t _Newsize);
	PREFIX_VC_DLL
		void resize(const size_t _Newsize, const double value);
	PREFIX_VC_DLL
		void release();
	PREFIX_VC_DLL
		bool empty() const;
	PREFIX_VC_DLL
		bool valid() const;
	PREFIX_VC_DLL
		CudaStream* get_cudaStream() const;
	PREFIX_VC_DLL
		void set_cudaStream(CudaStream* cudaStream);
	PREFIX_VC_DLL
		UVec(
			const UVecDepth depth,
			const UVecType type,
			const size_t size
		);
	PREFIX_VC_DLL
		UVec(
			const UVecDepth depth,
			const UVecType type
		);
#if 0
	PREFIX_VC_DLL
		UVec(
			void* pointer,
			const UVecDepth depth,
			const size_t size,
			const UVecType type = UVecType_Vector,
			const bool copyData = false
		);
#endif
	PREFIX_VC_DLL
		UVec(
			const std::vector<double>& vec,
			const UVecType type = UVecType_Vector,
			const bool copyData=false
		);
	PREFIX_VC_DLL
		UVec(
			const std::vector<float>& vec,
			const UVecType type = UVecType_Vector,
			const bool copyData = false
		);
	PREFIX_VC_DLL
		void copyTo(
			UVec& dst
		) const;
	PREFIX_VC_DLL
		void copyTo(
			std::vector<double>& vec
		) const;
	PREFIX_VC_DLL
		void copyTo(
			std::vector<float>& vec
		) const;
	PREFIX_VC_DLL
		const void* ptr() const;
	PREFIX_VC_DLL
		void* ptr();
	PREFIX_VC_DLL
		const void* vec_ptr() const;
	PREFIX_VC_DLL
		void* vec_ptr();
	PREFIX_VC_DLL
		UVec(
		);
	PREFIX_VC_DLL
		~UVec();
	PREFIX_VC_DLL
		UVec clone() const;
	/** @overload
	Disable operator=
	*/
	PREFIX_VC_DLL
		UVec& operator=(const UVec& rhs);
	/** @overload
	Disable copy constructor
	*/
	PREFIX_VC_DLL
		UVec(const UVec& rhs);
private:
	UVec_impl *pimpl;
};

PREFIX_VC_DLL
void average(const UVec& src1, const UVec& src2, UVec& dst);

/*
	@brief Template matrix class derived from UVec
*/
template<typename T>
class UVec_ : public UVec {
public:
	std::vector<T>* vec();
	const std::vector<T>* vec() const;
	T* ptr();
//	const T* vec() const;

	/** @overload
	Disable operator=
	*/
		UVec_& operator=(const UVec& rhs);
	/** @overload
	Disable copy constructor
	*/
		UVec_(const UVec& rhs);
		/*
		@brief Default constructor
		*/
		UVec_() : UVec() {}
		~UVec_() {}

};

template<typename T> inline
std::vector<T>*  UVec_<T>::vec() {
	if (type() == UVecType_Vector) return static_cast<std::vector<T>*>(UVec::vec_ptr());
	else return NULL;
}
template<typename T> inline
const std::vector<T>*  UVec_<T>::vec() const {
	if (type() == UVecType_Vector) return static_cast<const std::vector<T>*>(UVec::vec_ptr());
	else return NULL;
}
template<typename T> inline
T*  UVec_<T>::ptr() {
	return static_cast<T*>(UVec::ptr());
}
#if 0
template<typename T> inline
const T*  UVec_<T>::ptr() const {
	return static_cast<T*>(ptr());
}
#endif

template<typename T> inline
UVec_<T>& UVec_<T>::operator=(const UVec& rhs) {
	if (this == &rhs) return *this;
	UVec::operator = (rhs);
	return *this;
}

template<typename T> inline
UVec_<T>::UVec_(const UVec& rhs) :
	UVec(rhs) {
}

void synchronizeCuda(beacls::CudaStream* cudaStream);

PREFIX_VC_DLL
int get_num_of_gpus();
PREFIX_VC_DLL
void set_gpu_id(const int id);

PREFIX_VC_DLL
void reallocateAsSrc(beacls::UVec& dst, const::beacls::UVec& src);
PREFIX_VC_DLL
bool is_cuda(const::beacls::UVec& src);

};	// bears
#endif	/* __UVec_hpp__ */

