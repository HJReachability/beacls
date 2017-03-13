#ifndef __UVec_impl_hpp__
#define __UVec_impl_hpp__

#include <cstdint>
#include <vector>
//#include <typedef.hpp>
#include <Core/UVec.hpp>
#include <thread>
#include <mutex>
#include "UVec_impl_cuda.hpp"

namespace beacls
{

class UVec_impl {
public:
	size_t ref_counter;
private:
	std::mutex mtx;
	UVecDepth d;
	UVecType t;
	size_t s;
	size_t buf_b;
	bool declare_outside;
	bool cudaStreamDeclare_outside;
public:
	union {
		std::vector<int8_t> *data8s;
		std::vector<uint8_t> *data8u;
		std::vector<int16_t> *data16s;
		std::vector<uint16_t> *data16u;
		std::vector<int32_t> *data32s;
		std::vector<float> *data32f;
		std::vector<double> *data64f;
		std::vector<uint32_t> *data32u;
		std::vector<int64_t> *data64s;
		std::vector<uint64_t> *data64u;
		void* data_cuda;
		void* p;
	};
	beacls::CudaStream* cudaStream;
	bool is_declare_outside() const { return declare_outside; }
	bool is_cudaStreamDeclare_outside() const { return cudaStreamDeclare_outside; }
	void set_cudaStreamDeclare_outside(const bool v) { cudaStreamDeclare_outside = v; }
	void lock() { mtx.lock(); }
	void unlock() { mtx.unlock(); }
	template <typename T>
	void set_vec(const std::vector<T>& vec);
	template <typename T>
	void copy_vec(const std::vector<T>& vec);
	template<typename T>
	void copyToVec(std::vector<T>& vec) const;
	template<typename T>
	void copyCudaToVec(std::vector<T>& vec) const;
	void* ptr();
	const void* ptr() const;
	void* vec_ptr();
	const void* vec_ptr() const;
	size_t increment_ref_counter() {
		return ++ref_counter;
	}
	size_t size() const {
		return s;
	}
	size_t buf_bytes() const {
		return buf_b;
	}
	void set_buf_bytes(const size_t buf_bytes) {
		buf_b = buf_bytes;
	}
	UVecType type() const {
		return t;
	}
	UVecDepth depth() const {
		return d;
	}
	template<typename T>
	void create(const size_t _Newsize, const T val);
	template<typename T>
	void resize_template(const size_t _Newsize, const T value);
	void resize(const size_t size, const double value);
	void release();

	bool empty() const {
		return s == 0;
	}
	bool valid() const {
		if (t == UVecType_Invalid) return false;
		else return true;
	}
	UVec_impl(
		void* pointer,
		const UVecDepth depth,
		const UVecType type,
		const size_t size
	) : ref_counter(1), 
		d(depth), 
		t(type), 
		s(size),
		buf_b(type==UVecType_Cuda ? size * depth_to_byte(depth) : 0), 
		declare_outside(true), 
		cudaStreamDeclare_outside(true),
		p(pointer),
		cudaStream(NULL)
	{
	}
	UVec_impl(
		const UVecDepth depth,
		const UVecType type,
		const size_t size
	);
	UVec_impl() : 
		ref_counter(1), 
		d(UVecDepth_Invalid), 
		t(UVecType_Invalid), 
		s(0), 
		buf_b(0), 
		declare_outside(true), 
		cudaStreamDeclare_outside(true),
		p(NULL),
		cudaStream(NULL)
	{
	}
	~UVec_impl();
	UVec_impl* clone() const {
		return new UVec_impl(*this);
	};
	/** @overload
	Disable operator=
	*/
	UVec_impl& operator=(const UVec_impl& rhs); 
private:

	/** @overload
	copy constructor
	*/
	UVec_impl(const UVec_impl& rhs);
};
template <>
void UVec_impl::set_vec(const std::vector<double>& vec) {
	s = vec.size();
	data64f = const_cast<std::vector<double>*>(&vec);
	declare_outside = true;
}
template <>
void UVec_impl::set_vec(const std::vector<float>& vec) {
	s = vec.size();
	data32f = const_cast<std::vector<float>*>(&vec);
	declare_outside = true;
}
template <>
void UVec_impl::copy_vec(const std::vector<double>& vec) {
	s = vec.size();
	*data64f = vec;
}
template <>
void UVec_impl::copy_vec(const std::vector<float>& vec) {
	s = vec.size();
	*data32f = vec;
}

template<>
void UVec_impl::copyToVec(std::vector<double>& vec) const {
	switch (t) {
	case UVecType_Invalid:
	default:
		break;
	case UVecType_Vector:
		if(data64f) vec = *data64f;
		break;
	case UVecType_Cuda:
		copyCudaToVec(vec);
		break;
	}
}
template<>
void UVec_impl::copyToVec(std::vector<float>& vec) const {
	switch (t) {
	case UVecType_Invalid:
	default:
		break;
	case UVecType_Vector:
		if (data32f) vec = *data32f;
		break;
	case UVecType_Cuda:
		copyCudaToVec(vec);
		break;
	}
}

void* UVec_impl::ptr() {
	void* ptr = NULL;
	switch (t) {
	case UVecType_Invalid:
	default:
		break;
	case UVecType_Vector:
		switch (d) {
		case UVecDepth_Invalid:
		case UVecDepth_User:
		default:
			break;
		case UVecDepth_8U:
			ptr = static_cast<void*>(data8u->data());
			break;
		case UVecDepth_8S:
			ptr = static_cast<void*>(data8s->data());
			break;
		case UVecDepth_16U:
			ptr = static_cast<void*>(data16u->data());
			break;
		case UVecDepth_16S:
			ptr = static_cast<void*>(data16s->data());
			break;
		case UVecDepth_32S:
			ptr = static_cast<void*>(data32s->data());
			break;
		case UVecDepth_32F:
			ptr = static_cast<void*>(data32f->data());
			break;
		case UVecDepth_64F:
			ptr = static_cast<void*>(data64f->data());
			break;
		case UVecDepth_32U:
			ptr = static_cast<void*>(data32u->data());
			break;
		case UVecDepth_64U:
			ptr = static_cast<void*>(data64u->data());
			break;
		case UVecDepth_64S:
			ptr = static_cast<void*>(data64s->data());
			break;
		}
		break;
	case UVecType_Cuda:
		ptr = static_cast<void*>(data_cuda);
		break;
	}
	return ptr;
}
const void* UVec_impl::ptr() const {
	const void* ptr = NULL;
	switch (t) {
	case UVecType_Invalid:
	default:
		break;
	case UVecType_Vector:
		switch (d) {
		case UVecDepth_Invalid:
		case UVecDepth_User:
		default:
			break;
		case UVecDepth_8U:
			ptr = static_cast<const void*>(&(*data8u)[0]);
			break;
		case UVecDepth_8S:
			ptr = static_cast<const void*>(&(*data8s)[0]);
			break;
		case UVecDepth_16U:
			ptr = static_cast<const void*>(&(*data16u)[0]);
			break;
		case UVecDepth_16S:
			ptr = static_cast<const void*>(&(*data16s)[0]);
			break;
		case UVecDepth_32S:
			ptr = static_cast<const void*>(&(*data32s)[0]);
			break;
		case UVecDepth_32F:
			ptr = static_cast<const void*>(&(*data32f)[0]);
			break;
		case UVecDepth_64F:
			ptr = static_cast<const void*>(&(*data64f)[0]);
			break;
		case UVecDepth_32U:
			ptr = static_cast<const void*>(&(*data32u)[0]);
			break;
		case UVecDepth_64U:
			ptr = static_cast<const void*>(&(*data64u)[0]);
			break;
		case UVecDepth_64S:
			ptr = static_cast<const void*>(&(*data64s)[0]);
			break;
		}
		break;
	case UVecType_Cuda:
		ptr = static_cast<void*>(data_cuda);
		break;
	}
	return ptr;
}

void* UVec_impl::vec_ptr() {
	void* ptr = NULL;
	switch (t) {
	case UVecType_Invalid:
	case UVecType_Cuda:
	default:
		break;
	case UVecType_Vector:
		switch (d) {
		case UVecDepth_Invalid:
		case UVecDepth_User:
		default:
			break;
		case UVecDepth_8U:
		case UVecDepth_8S:
		case UVecDepth_16U:
		case UVecDepth_16S:
		case UVecDepth_32S:
		case UVecDepth_32F:
		case UVecDepth_64F:
		case UVecDepth_32U:
		case UVecDepth_64U:
		case UVecDepth_64S:
			ptr = static_cast<void*>(p);
			break;
		}
		break;
	}
	return ptr;
}
const void* UVec_impl::vec_ptr() const {
	const void* ptr = NULL;
	switch (t) {
	case UVecType_Invalid:
	case UVecType_Cuda:
	default:
		break;
	case UVecType_Vector:
		switch (d) {
		case UVecDepth_Invalid:
		case UVecDepth_User:
		default:
			break;
		case UVecDepth_8U:
		case UVecDepth_8S:
		case UVecDepth_16U:
		case UVecDepth_16S:
		case UVecDepth_32S:
		case UVecDepth_32F:
		case UVecDepth_64F:
		case UVecDepth_32U:
		case UVecDepth_64U:
		case UVecDepth_64S:
			ptr = static_cast<const void*>(p);
			break;
		}
		break;
	}
	return ptr;
}


}	// bears
#endif	/* __UVec_impl_hpp__ */

