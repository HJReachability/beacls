#include <vector>
#include <iterator>
#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include <Core/UVec.hpp>
#include <Core/CudaStream.hpp>
#include "UVec_impl.hpp"
#include "UVec_impl_cuda.hpp"

#include <macro.hpp>

namespace beacls
{

	UVec_impl::UVec_impl(
		const UVecDepth depth,
		const UVecType type,
		const size_t size
	) :
		ref_counter(1),
		d(depth),
		t(type),
		s(0),
		buf_b(0),
		declare_outside(false),
		cudaStreamDeclare_outside(true),
		p(NULL),
		cudaStream(NULL)
	{
		if (size) create(size, 0);
		s = size;
	}
	UVec_impl::~UVec_impl()
	{
		if (!declare_outside) {
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
					if (data8u) delete data8u;
					break;
				case UVecDepth_8S:
					if (data8s) delete data8s;
					break;
				case UVecDepth_16U:
					if (data16u) delete data16u;
					break;
				case UVecDepth_16S:
					if (data16s) delete data16s;
					break;
				case UVecDepth_32S:
					if (data32s) delete data32s;
					break;
				case UVecDepth_32F:
					if (data32f) delete data32f;
					break;
				case UVecDepth_64F:
					if (data64f) delete data64f;
					break;
				case UVecDepth_32U:
					if (data32u) delete data32u;
					break;
				case UVecDepth_64U:
					if (data64u) delete data64u;
					break;
				case UVecDepth_64S:
					if (data64s) delete data64s;
					break;
				}
				break;
			case UVecType_Cuda:
				if (data_cuda) {
					freeCudaMem(data_cuda);
					buf_b = 0;
				}
				if (cudaStream && !cudaStreamDeclare_outside) {
					delete cudaStream;
					cudaStream = NULL;
					cudaStreamDeclare_outside = true;
				}
				break;
			}
		}
	}
	UVec_impl& UVec_impl::operator=(const UVec_impl& rhs)
	{
		if (this == &rhs) return *this;
		if (p) {
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
					delete data8u;
					break;
				case UVecDepth_8S:
					delete data8s;
					break;
				case UVecDepth_16U:
					delete data16u;
					break;
				case UVecDepth_16S:
					delete data16s;
					break;
				case UVecDepth_32S:
					delete data32s;
					break;
				case UVecDepth_32F:
					delete data32f;
					break;
				case UVecDepth_64F:
					delete data64f;
					break;
				case UVecDepth_32U:
					delete data32u;
					break;
				case UVecDepth_64U:
					delete data64u;
					break;
				case UVecDepth_64S:
					delete data64s;
					break;
				}
				break;
			case UVecType_Cuda:
				freeCudaMem(data_cuda);
				buf_b = 0;
				break;
			}
		}
		d = rhs.d;
		t = rhs.t;
		s = rhs.s;
		p = NULL;
		cudaStream = NULL;
		declare_outside = false;
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
				data8u = new std::vector<uint8_t>(*rhs.data8u);
				break;
			case UVecDepth_8S:
				data8s = new std::vector<int8_t>(*rhs.data8s);
				break;
			case UVecDepth_16U:
				data16u = new std::vector<uint16_t>(*rhs.data16u);
				break;
			case UVecDepth_16S:
				data16s = new std::vector<int16_t>(*rhs.data16s);
				break;
			case UVecDepth_32S:
				data32s = new std::vector<int32_t>(*rhs.data32s);
				break;
			case UVecDepth_32F:
				data32f = new std::vector<float>(*rhs.data32f);
				break;
			case UVecDepth_64F:
				data64f = new std::vector<double>(*rhs.data64f);
				break;
			case UVecDepth_32U:
				data32u = new std::vector<uint32_t>(*rhs.data32u);
				break;
			case UVecDepth_64U:
				data64u = new std::vector<uint64_t>(*rhs.data64u);
				break;
			case UVecDepth_64S:
				data64s = new std::vector<int64_t>(*rhs.data64s);
				break;
			}
			break;
		case UVecType_Cuda:
		{
			size_t byte = depth_to_byte(rhs.depth());
			void* new_ptr = allocateCudaMem(rhs.size() * byte);
			if (rhs.data_cuda) {
				size_t copy_size = (rhs.size() > s) ? s : rhs.size();
				copyCudaDeviceToDevice(new_ptr, rhs.data_cuda, copy_size * byte);
			}
			buf_b = rhs.size() * byte;
			data_cuda = new_ptr;
			if (!cudaStream) {
				cudaStream = new CudaStream();
				cudaStreamDeclare_outside = false;
			}
		}
		break;
		}
		return *this;
	}
	UVec_impl::UVec_impl(const UVec_impl& rhs) :
		ref_counter(1)
		, d(rhs.d)
		, t(rhs.t)
		, p(NULL)
		, cudaStream(NULL)
	{
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
				data8u = new std::vector<uint8_t>(*rhs.data8u);
				break;
			case UVecDepth_8S:
				data8s = new std::vector<int8_t>(*rhs.data8s);
				break;
			case UVecDepth_16U:
				data16u = new std::vector<uint16_t>(*rhs.data16u);
				break;
			case UVecDepth_16S:
				data16s = new std::vector<int16_t>(*rhs.data16s);
				break;
			case UVecDepth_32S:
				data32s = new std::vector<int32_t>(*rhs.data32s);
				break;
			case UVecDepth_32F:
				data32f = new std::vector<float>(*rhs.data32f);
				break;
			case UVecDepth_64F:
				data64f = new std::vector<double>(*rhs.data64f);
				break;
			case UVecDepth_32U:
				data32u = new std::vector<uint32_t>(*rhs.data32u);
				break;
			case UVecDepth_64U:
				data64u = new std::vector<uint64_t>(*rhs.data64u);
				break;
			case UVecDepth_64S:
				data64s = new std::vector<int64_t>(*rhs.data64s);
				break;
			}
			break;
		case UVecType_Cuda:
		{
			size_t byte = depth_to_byte(rhs.depth());
			if (buf_b < (rhs.size() * byte)) {
				void* new_ptr = allocateCudaMem(rhs.size() * byte);
				if (data_cuda) freeCudaMem(data_cuda);
				buf_b = rhs.size() * byte;
				data_cuda = new_ptr;
				if (!cudaStream) {
					cudaStream = new CudaStream();
					cudaStreamDeclare_outside = false;
				}
			}
			if (rhs.data_cuda) {
				size_t copy_size = rhs.size();
				copyCudaDeviceToDevice(data_cuda, rhs.data_cuda, copy_size * byte);
			}
		}
		break;
		}
		s = rhs.s;
		declare_outside = false;
	}
	template<typename T>
	void UVec_impl::create(const size_t _Newsize, const T val)
	{
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
				if (!declare_outside && data8u) data8u->resize(_Newsize, (uint8_t)val);
				else {
					data8u = new std::vector<uint8_t>(_Newsize, (uint8_t)val);
					declare_outside = false;
				}
				break;
			case UVecDepth_8S:
				if (!declare_outside && data8s) data8s->resize(_Newsize, (int8_t)val);
				else {
					data8s = new std::vector<int8_t>(_Newsize, (int8_t)val);
					declare_outside = false;
				}
				break;
			case UVecDepth_16U:
				if (!declare_outside && data16u) data16u->resize(_Newsize, (uint16_t)val);
				else {
					data16u = new std::vector<uint16_t>(_Newsize, (uint16_t)val);
					declare_outside = false;
				}
				break;
			case UVecDepth_16S:
				if (!declare_outside && data16s) data16s->resize(_Newsize, (int16_t)val);
				else {
					data16s = new std::vector<int16_t>(_Newsize, (int16_t)val);
					declare_outside = false;
				}
				break;
			case UVecDepth_32S:
				if (!declare_outside && data32s) data32s->resize(_Newsize, (int32_t)val);
				else {
					data32s = new std::vector<int32_t>(_Newsize, (int32_t)val);
					declare_outside = false;
				}
				break;
			case UVecDepth_32F:
				if (!declare_outside && data32f) data32f->resize(_Newsize, (float)val);
				else {
					data32f = new std::vector<float>(_Newsize, (float)val);
					declare_outside = false;
				}
				break;
			case UVecDepth_64F:
				if (!declare_outside && data64f) data64f->resize(_Newsize, (double)val);
				else {
					data64f = new std::vector<double>(_Newsize, (double)val);
					declare_outside = false;
				}
				break;
			case UVecDepth_32U:
				if (!declare_outside && data32u) data32u->resize(_Newsize, (uint32_t)val);
				else {
					data32u = new std::vector<uint32_t>(_Newsize, (uint32_t)val);
					declare_outside = false;
				}
				break;
			case UVecDepth_64U:
				if (!declare_outside && data64u) data64u->resize(_Newsize, (uint64_t)val);
				else {
					data64u = new std::vector<uint64_t>(_Newsize, (uint64_t)val);
					declare_outside = false;
				}
				break;
			case UVecDepth_64S:
				if (!declare_outside && data64s) data64s->resize(_Newsize, (int64_t)val);
				else {
					data64s = new std::vector<int64_t>(_Newsize, (int64_t)val);
					declare_outside = false;
				}
				break;
			}
			break;
		case UVecType_Cuda:
		{
			size_t byte = depth_to_byte(d);
			if (buf_b < (_Newsize * byte)) {
				void* new_ptr = allocateCudaMem(_Newsize * byte);
				if (data_cuda) {
					freeCudaMem(data_cuda);
				}
				buf_b = _Newsize * byte;
				data_cuda = new_ptr;
				if (!cudaStream) {
					cudaStream = new CudaStream();
					cudaStreamDeclare_outside = false;
				}
			}
		}
		break;
		}
		s = _Newsize;
	}
	template<typename T>
	void clear_swap(T& obj) {
		obj.clear();
		T().swap(obj);
	}

	template<typename T>
	void resize_T(std::vector<T>*& obj, const size_t _Newsize, const T value, const size_t s, bool& declare_outside) {
		size_t copy_size = (s < _Newsize) ? s : _Newsize;
		if (!declare_outside)
			obj->resize(_Newsize, static_cast<T>(value));
		else {
			std::vector<T>* new_ptr = new std::vector<T>(_Newsize);
			std::copy(obj->begin(), obj->begin() + copy_size, new_ptr->begin());
			obj = new_ptr;
		}
	}
	template<typename T>
	void UVec_impl::resize_template(const size_t _Newsize, const T value) {
		mtx.lock();
		if (s!=0) {
			switch (t) {
			case UVecType_Invalid:
			default:
				break;
			case UVecType_Vector:
				resize_T<T>((std::vector<T>*&)p, _Newsize, value, s, declare_outside);

				break;
			case UVecType_Cuda:
				{
					size_t byte = depth_to_byte(d);
					if (buf_b < (_Newsize * byte)) {
						void* new_ptr = allocateCudaMem(_Newsize * byte);
						if (data_cuda) {
							size_t copy_size = (_Newsize > s) ? s : _Newsize;
							copyCudaDeviceToDevice(new_ptr, data_cuda, copy_size * byte);
							freeCudaMem(data_cuda);
						}
						fillCudaMemory((T*)new_ptr + s, value, _Newsize - s);
						buf_b = _Newsize * byte;
						data_cuda = new_ptr;
						if (!cudaStream) {
							cudaStream = new CudaStream();
							cudaStreamDeclare_outside = false;
						}
					}
				}
				break;
			}
		}
		else {
			create(_Newsize, value);
		}
		s = _Newsize;
		mtx.unlock();

	}

	void UVec_impl::resize(const size_t _Newsize, const double value) {
		if (s != _Newsize) {
			switch (d) {
			case UVecDepth_Invalid:
			case UVecDepth_User:
			default:
				break;
			case UVecDepth_8U:
				resize_template<uint8_t>(_Newsize, (uint8_t)value);
				break;
			case UVecDepth_8S:
				resize_template<int8_t>(_Newsize, (int8_t)value);
				break;
			case UVecDepth_16U:
				resize_template<uint16_t>(_Newsize, (uint16_t)value);
				break;
			case UVecDepth_16S:
				resize_template<int16_t>(_Newsize, (int16_t)value);
				break;
			case UVecDepth_32S:
				resize_template<int32_t>(_Newsize, (int32_t)value);
				break;
			case UVecDepth_32F:
				resize_template<float>(_Newsize, (float)value);
				break;
			case UVecDepth_64F:
				resize_template<double>(_Newsize, (double)value);
				break;
			case UVecDepth_32U:
				resize_template<uint32_t>(_Newsize, (uint32_t)value);
				break;
			case UVecDepth_64U:
				resize_template<uint64_t>(_Newsize, (uint64_t)value);
				break;
			case UVecDepth_64S:
				resize_template<int64_t>(_Newsize, (int64_t)value);
				break;
			}
		}
	}
	void UVec_impl::release() {
		if (ref_counter == 0) {
			s = 0;
			d = UVecDepth_Invalid;
			t = UVecType_Invalid;
			p = NULL;
			cudaStream = NULL;
			return;
		}
		if (--ref_counter == 0) {
			if (!declare_outside) {
				if (p) {
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
							delete data8u;
							break;
						case UVecDepth_8S:
							delete data8s;
							break;
						case UVecDepth_16U:
							delete data16u;
							break;
						case UVecDepth_16S:
							delete data16s;
							break;
						case UVecDepth_32S:
							delete data32s;
							break;
						case UVecDepth_32F:
							delete data32f;
							break;
						case UVecDepth_64F:
							delete data64f;
							break;
						case UVecDepth_32U:
							delete data32u;
							break;
						case UVecDepth_64U:
							delete data64u;
							break;
						case UVecDepth_64S:
							delete data64s;
							break;
						}
						break;
					case UVecType_Cuda:
						if (data_cuda)
							freeCudaMem(data_cuda);
						if (cudaStream && !cudaStreamDeclare_outside) {
							delete cudaStream;
							cudaStreamDeclare_outside = true;
						}
						buf_b = 0;
						break;
					}
				}
			}
			s = 0;
			d = UVecDepth_Invalid;
			t = UVecType_Invalid;
			p = NULL;
			cudaStream = NULL;
		}
	}

	size_t UVec::size() const
	{
		if (pimpl) return pimpl->size();
		else return false;
	}

	template<typename T, typename S>
	void convertHostToHost_T_S(
		std::vector<T>& dst, const std::vector<S>& src,
		const double alpha, const double beta) {
		std::transform(src.cbegin(), src.cend(), dst.begin(), ([alpha, beta](const auto &rhs) {
			return static_cast<T>(rhs * alpha + beta);
		}));
	}

	template<typename T>
	void convertHostToHost_T(
		std::vector<T>& dst, const UVec_impl* src_pimpl,
		const double alpha, const double beta) {
		switch (src_pimpl->depth()) {
		case UVecDepth_Invalid:
		case UVecDepth_User:
		default:
			break;
		case UVecDepth_8U:
			convertHostToHost_T_S(dst, *src_pimpl->data8u, alpha, beta);
			break;
		case UVecDepth_8S:
			convertHostToHost_T_S(dst, *src_pimpl->data8s, alpha, beta);
			break;
		case UVecDepth_16U:
			convertHostToHost_T_S(dst, *src_pimpl->data16u, alpha, beta);
			break;
		case UVecDepth_16S:
			convertHostToHost_T_S(dst, *src_pimpl->data16s, alpha, beta);
			break;
		case UVecDepth_32S:
			convertHostToHost_T_S(dst, *src_pimpl->data32s, alpha, beta);
			break;
		case UVecDepth_32F:
			convertHostToHost_T_S(dst, *src_pimpl->data32f, alpha, beta);
			break;
		case UVecDepth_64F:
			convertHostToHost_T_S(dst, *src_pimpl->data64f, alpha, beta);
			break;
		case UVecDepth_32U:
			convertHostToHost_T_S(dst, *src_pimpl->data32u, alpha, beta);
			break;
		case UVecDepth_64U:
			convertHostToHost_T_S(dst, *src_pimpl->data64u, alpha, beta);
			break;
		case UVecDepth_64S:
			convertHostToHost_T_S(dst, *src_pimpl->data64s, alpha, beta);
			break;
		}

	}

	void convertHostToHost(
		UVec_impl* dst_pimpl, const UVec_impl* src_pimpl,
		const double alpha, const double beta) {
		if ((dst_pimpl->depth() == src_pimpl->depth()) && (alpha == 1) && (beta == 0)) {
			*dst_pimpl = *src_pimpl;
			return;
		}
		switch (dst_pimpl->depth()) {
		case UVecDepth_Invalid:
		case UVecDepth_User:
		default:
			break;
		case UVecDepth_8U:
			convertHostToHost_T(*dst_pimpl->data8u, src_pimpl, alpha, beta);
			break;
		case UVecDepth_8S:
			convertHostToHost_T(*dst_pimpl->data8s, src_pimpl, alpha, beta);
			break;
		case UVecDepth_16U:
			convertHostToHost_T(*dst_pimpl->data16u, src_pimpl, alpha, beta);
			break;
		case UVecDepth_16S:
			convertHostToHost_T(*dst_pimpl->data16s, src_pimpl, alpha, beta);
			break;
		case UVecDepth_32S:
			convertHostToHost_T(*dst_pimpl->data32s, src_pimpl, alpha, beta);
			break;
		case UVecDepth_32F:
			convertHostToHost_T(*dst_pimpl->data32f, src_pimpl, alpha, beta);
			break;
		case UVecDepth_64F:
			convertHostToHost_T(*dst_pimpl->data64f, src_pimpl, alpha, beta);
			break;
		case UVecDepth_32U:
			convertHostToHost_T(*dst_pimpl->data32u, src_pimpl, alpha, beta);
			break;
		case UVecDepth_64U:
			convertHostToHost_T(*dst_pimpl->data64u, src_pimpl, alpha, beta);
			break;
		case UVecDepth_64S:
			convertHostToHost_T(*dst_pimpl->data64s, src_pimpl, alpha, beta);
			break;
		}

	}
	void convertHostToDevice(
		UVec_impl* dst_pimpl, const UVec_impl* src_pimpl,
		const double alpha, const double beta) {
		bool delete_tmp_pimpl = false;
		const UVec_impl* tmp_pimpl;
		if (
			(dst_pimpl->depth() == src_pimpl->depth())
			&& (alpha ==1)
			&& (beta == 0)
			) {
			tmp_pimpl = src_pimpl;
		}
		else {
			UVec_impl* tmp_pimpl2 = new UVec_impl(src_pimpl->depth(), dst_pimpl->type(), dst_pimpl->size());
			delete_tmp_pimpl = true;
			convertHostToHost(tmp_pimpl2, src_pimpl, alpha, beta);
			tmp_pimpl = tmp_pimpl2;
		}
		size_t byte = depth_to_byte(tmp_pimpl->depth());
		void* new_ptr;
		if (dst_pimpl->data_cuda &&
			(tmp_pimpl->size() * byte <= dst_pimpl->buf_bytes())
		) {
			new_ptr = dst_pimpl->data_cuda;
			dst_pimpl->data_cuda = NULL;
		}
		else {
			new_ptr = allocateCudaMem(tmp_pimpl->size() * byte);
			dst_pimpl->set_buf_bytes(tmp_pimpl->size() * byte);
		}
		switch (tmp_pimpl->depth()) {
		case UVecDepth_Invalid:
		case UVecDepth_User:
		default:
			break;
		case UVecDepth_8U:
			copyCudaHostToDevice(new_ptr, (void*)&(*tmp_pimpl->data8u)[0], dst_pimpl->size() * byte);
			break;
		case UVecDepth_8S:
			copyCudaHostToDevice(new_ptr, (void*)&(*tmp_pimpl->data8s)[0], dst_pimpl->size() * byte);
			break;
		case UVecDepth_16U:
			copyCudaHostToDevice(new_ptr, (void*)&(*tmp_pimpl->data16u)[0], dst_pimpl->size() * byte);
			break;
		case UVecDepth_16S:
			copyCudaHostToDevice(new_ptr, (void*)&(*tmp_pimpl->data16s)[0], dst_pimpl->size() * byte);
			break;
		case UVecDepth_32S:
			copyCudaHostToDevice(new_ptr, (void*)&(*tmp_pimpl->data32s)[0], dst_pimpl->size() * byte);
			break;
		case UVecDepth_32F:
			copyCudaHostToDevice(new_ptr, (void*)&(*tmp_pimpl->data32f)[0], dst_pimpl->size() * byte);
			break;
		case UVecDepth_64F:
			copyCudaHostToDevice(new_ptr, (void*)&(*tmp_pimpl->data64f)[0], dst_pimpl->size() * byte);
			break;
		case UVecDepth_32U:
			copyCudaHostToDevice(new_ptr, (void*)&(*tmp_pimpl->data32u)[0], dst_pimpl->size() * byte);
			break;
		case UVecDepth_64U:
			copyCudaHostToDevice(new_ptr, (void*)&(*tmp_pimpl->data64u)[0], dst_pimpl->size() * byte);
			break;
		case UVecDepth_64S:
			copyCudaHostToDevice(new_ptr, (void*)&(*tmp_pimpl->data64s)[0], dst_pimpl->size() * byte);
			break;
		}
		if(delete_tmp_pimpl) delete tmp_pimpl;
		if (dst_pimpl->data_cuda) {
			freeCudaMem(dst_pimpl->data_cuda);
		}
		dst_pimpl->data_cuda = new_ptr;
		if (!dst_pimpl->cudaStream) {
			dst_pimpl->cudaStream = new CudaStream();
			dst_pimpl->set_cudaStreamDeclare_outside(false);
		}
	}
	void convertDeviceToHost(
		UVec_impl* dst_pimpl, const UVec_impl* src_pimpl,
		const double alpha, const double beta) {
		bool delete_tmp_pimpl = false;
		UVec_impl* tmp_pimpl;
		if (
			dst_pimpl->depth() == src_pimpl->depth()) {
			tmp_pimpl = dst_pimpl;
		}
		else {
			tmp_pimpl = new UVec_impl(src_pimpl->depth(), dst_pimpl->type(), dst_pimpl->size());
			delete_tmp_pimpl = true;
		}


		size_t byte = depth_to_byte(tmp_pimpl->depth());
		switch (tmp_pimpl->depth()) {
		case UVecDepth_Invalid:
		case UVecDepth_User:
		default:
			break;
		case UVecDepth_8U:
			copyCudaDeviceToHost((void*)&(*tmp_pimpl->data8u)[0], src_pimpl->data_cuda, dst_pimpl->size() * byte);
			break;
		case UVecDepth_8S:
			copyCudaDeviceToHost((void*)&(*tmp_pimpl->data8s)[0], src_pimpl->data_cuda, dst_pimpl->size() * byte);
			break;
		case UVecDepth_16U:
			copyCudaDeviceToHost((void*)&(*tmp_pimpl->data16u)[0], src_pimpl->data_cuda, dst_pimpl->size() * byte);
			break;
		case UVecDepth_16S:
			copyCudaDeviceToHost((void*)&(*tmp_pimpl->data16s)[0], src_pimpl->data_cuda, dst_pimpl->size() * byte);
			break;
		case UVecDepth_32S:
			copyCudaDeviceToHost((void*)&(*tmp_pimpl->data32s)[0], src_pimpl->data_cuda, dst_pimpl->size() * byte);
			break;
		case UVecDepth_32F:
			copyCudaDeviceToHost((void*)&(*tmp_pimpl->data32f)[0], src_pimpl->data_cuda, dst_pimpl->size() * byte);
			break;
		case UVecDepth_64F:
			copyCudaDeviceToHost((void*)&(*tmp_pimpl->data64f)[0], src_pimpl->data_cuda, dst_pimpl->size() * byte);
			break;
		case UVecDepth_32U:
			copyCudaDeviceToHost((void*)&(*tmp_pimpl->data32u)[0], src_pimpl->data_cuda, dst_pimpl->size() * byte);
			break;
		case UVecDepth_64U:
			copyCudaDeviceToHost((void*)&(*tmp_pimpl->data64u)[0], src_pimpl->data_cuda, dst_pimpl->size() * byte);
			break;
		case UVecDepth_64S:
			copyCudaDeviceToHost((void*)&(*tmp_pimpl->data64s)[0], src_pimpl->data_cuda, dst_pimpl->size() * byte);
			break;
		}
		convertHostToHost(dst_pimpl, tmp_pimpl, alpha, beta);
		if(delete_tmp_pimpl) delete tmp_pimpl;
	}
	void convertDeviceToDevice(
		UVec_impl* dst_pimpl, const UVec_impl* src_pimpl,
		const double alpha, const double beta) {
		UVec_impl* tmp_pimpl = new UVec_impl(src_pimpl->depth(), dst_pimpl->type(), dst_pimpl->size());

		size_t byte = depth_to_byte(tmp_pimpl->depth());
		switch (tmp_pimpl->depth()) {
		case UVecDepth_Invalid:
		case UVecDepth_User:
		default:
			break;
		case UVecDepth_8U:
			copyCudaDeviceToHost((void*)&(*tmp_pimpl->data8u)[0], src_pimpl->data_cuda, dst_pimpl->size() * byte);
			break;
		case UVecDepth_8S:
			copyCudaDeviceToHost((void*)&(*tmp_pimpl->data8s)[0], src_pimpl->data_cuda, dst_pimpl->size() * byte);
			break;
		case UVecDepth_16U:
			copyCudaDeviceToHost((void*)&(*tmp_pimpl->data16u)[0], src_pimpl->data_cuda, dst_pimpl->size() * byte);
			break;
		case UVecDepth_16S:
			copyCudaDeviceToHost((void*)&(*tmp_pimpl->data16s)[0], src_pimpl->data_cuda, dst_pimpl->size() * byte);
			break;
		case UVecDepth_32S:
			copyCudaDeviceToHost((void*)&(*tmp_pimpl->data32s)[0], src_pimpl->data_cuda, dst_pimpl->size() * byte);
			break;
		case UVecDepth_32F:
			copyCudaDeviceToHost((void*)&(*tmp_pimpl->data32f)[0], src_pimpl->data_cuda, dst_pimpl->size() * byte);
			break;
		case UVecDepth_64F:
			copyCudaDeviceToHost((void*)&(*tmp_pimpl->data64f)[0], src_pimpl->data_cuda, dst_pimpl->size() * byte);
			break;
		case UVecDepth_32U:
			copyCudaDeviceToHost((void*)&(*tmp_pimpl->data32u)[0], src_pimpl->data_cuda, dst_pimpl->size() * byte);
			break;
		case UVecDepth_64U:
			copyCudaDeviceToHost((void*)&(*tmp_pimpl->data64u)[0], src_pimpl->data_cuda, dst_pimpl->size() * byte);
			break;
		case UVecDepth_64S:
			copyCudaDeviceToHost((void*)&(*tmp_pimpl->data64s)[0], src_pimpl->data_cuda, dst_pimpl->size() * byte);
			break;
		}
		UVec_impl* tmp2_pimpl = new UVec_impl(dst_pimpl->depth(), tmp_pimpl->type(), tmp_pimpl->size());
		convertHostToHost(tmp2_pimpl, tmp_pimpl, alpha, beta);
		delete tmp_pimpl;

		void* new_ptr;
		if (dst_pimpl->data_cuda &&
			(tmp2_pimpl->size() * byte <= dst_pimpl->buf_bytes())
			) {
			new_ptr = dst_pimpl->data_cuda;
			dst_pimpl->data_cuda = NULL;
		}
		else {
			new_ptr = allocateCudaMem(tmp2_pimpl->size() * byte);
			dst_pimpl->set_buf_bytes(tmp2_pimpl->size() * byte);
		}
		switch (tmp2_pimpl->depth()) {
		case UVecDepth_Invalid:
		case UVecDepth_User:
		default:
			break;
		case UVecDepth_8U:
			copyCudaHostToDevice(new_ptr, (void*)&(*tmp2_pimpl->data8u)[0], dst_pimpl->size() * byte);
			break;
		case UVecDepth_8S:
			copyCudaHostToDevice(new_ptr, (void*)&(*tmp2_pimpl->data8s)[0], dst_pimpl->size() * byte);
			break;
		case UVecDepth_16U:
			copyCudaHostToDevice(new_ptr, (void*)&(*tmp2_pimpl->data16u)[0], dst_pimpl->size() * byte);
			break;
		case UVecDepth_16S:
			copyCudaHostToDevice(new_ptr, (void*)&(*tmp2_pimpl->data16s)[0], dst_pimpl->size() * byte);
			break;
		case UVecDepth_32S:
			copyCudaHostToDevice(new_ptr, (void*)&(*tmp2_pimpl->data32s)[0], dst_pimpl->size() * byte);
			break;
		case UVecDepth_32F:
			copyCudaHostToDevice(new_ptr, (void*)&(*tmp2_pimpl->data32f)[0], dst_pimpl->size() * byte);
			break;
		case UVecDepth_64F:
			copyCudaHostToDevice(new_ptr, (void*)&(*tmp2_pimpl->data64f)[0], dst_pimpl->size() * byte);
			break;
		case UVecDepth_32U:
			copyCudaHostToDevice(new_ptr, (void*)&(*tmp2_pimpl->data32u)[0], dst_pimpl->size() * byte);
			break;
		case UVecDepth_64U:
			copyCudaHostToDevice(new_ptr, (void*)&(*tmp2_pimpl->data64u)[0], dst_pimpl->size() * byte);
			break;
		case UVecDepth_64S:
			copyCudaHostToDevice(new_ptr, (void*)&(*tmp2_pimpl->data64s)[0], dst_pimpl->size() * byte);
			break;
		}
		delete tmp2_pimpl;
		if (dst_pimpl->data_cuda) {
			freeCudaMem(dst_pimpl->data_cuda);
		}
		dst_pimpl->data_cuda = new_ptr;
		if (!dst_pimpl->cudaStream) {
			dst_pimpl->cudaStream = new CudaStream();
			dst_pimpl->set_cudaStreamDeclare_outside(false);
		}
	}
	void UVec::convertTo(
		UVec& dst, UVecDepth depth,
		UVecType type,
		const double alpha, const double beta) const
	{
		if (!pimpl) return;
		UVec_impl *org_dst_pimpl = dst.pimpl;
		UVec_impl *new_dst_pimpl;
		if (!org_dst_pimpl->is_declare_outside()
			&& (pimpl->depth() == depth)
			&& (pimpl->type() == type)
			&& (alpha == 1) && (beta == 0)) {
			pimpl->lock();
			pimpl->increment_ref_counter();
			new_dst_pimpl = pimpl;
			pimpl->unlock();
		}
		else {
			if ((org_dst_pimpl->depth() == depth)
				&& (org_dst_pimpl->type() == type)
				) {
				org_dst_pimpl->resize((pimpl->size()), 0);
				new_dst_pimpl = org_dst_pimpl;
				org_dst_pimpl = NULL;
			} else {
				new_dst_pimpl = new UVec_impl(depth, type, pimpl->size());
			}
			switch (pimpl->type()) {
			case UVecType_Invalid:
			default:
				break;
			case UVecType_Vector:
				switch (new_dst_pimpl->type()) {
				case UVecType_Invalid:
				default:
					break;
				case UVecType_Vector:
					convertHostToHost(new_dst_pimpl, pimpl, alpha, beta);
					break;
				case UVecType_Cuda:
					convertHostToDevice(new_dst_pimpl, pimpl, alpha, beta);
					break;
				}
				break;
			case UVecType_Cuda:
				switch (new_dst_pimpl->type()) {
				case UVecType_Invalid:
				default:
					break;
				case UVecType_Vector:
					convertDeviceToHost(new_dst_pimpl, pimpl, alpha, beta);
					break;
				case UVecType_Cuda:
					convertDeviceToDevice(new_dst_pimpl, pimpl, alpha, beta);
					break;
				}
				break;
			}
			dst.pimpl = new_dst_pimpl;
			if (org_dst_pimpl) {
				org_dst_pimpl->lock();
				org_dst_pimpl->release();
				org_dst_pimpl->unlock();
				if (org_dst_pimpl->ref_counter == 0) {
					delete org_dst_pimpl;
				}
			}
		}
	}
	void UVec::convertTo(
		UVec& dst,
		UVecType type,
		const double alpha, const double beta) const
	{
		convertTo(dst, pimpl->depth(), type, alpha, beta);
	}

	template<typename T> static inline
		UVec_impl* createPimplFromVec(
			const std::vector<T>& vec,
			const UVecType type,
			const bool copyData
		) {
		UVec_impl* pimpl;
		UVecDepth depth = type_to_depth<T>();
		if (!copyData && (type == UVecType_Vector)) {
			pimpl = new UVec_impl(depth, type, 0);
			pimpl->set_vec(vec);
		}
		else {
			pimpl = new UVec_impl(depth, type, vec.size());
			if (type == UVecType_Vector) {
				pimpl->copy_vec(vec);
			}
			else if (type == UVecType_Cuda) {
				size_t byte = depth_to_byte(pimpl->depth());
				if (pimpl->data_cuda) {
					copyCudaHostToDevice(pimpl->data_cuda, &vec[0], vec.size() * byte);
				}
			}
		}
		return pimpl;
	}
	UVec::UVec(
		const std::vector<double>& vec,
		const UVecType type,
		const bool copyData
	) {
		pimpl = createPimplFromVec(vec, type, copyData);
	}
	UVec::UVec(
		const std::vector<float>& vec,
		const UVecType type,
		const bool copyData
	) {
		pimpl = createPimplFromVec(vec, type, copyData);
	}
	template <typename T>
	void UVec_impl::copyCudaToVec(std::vector<T>& vec) const
	{
		size_t byte = depth_to_byte(d);
		vec.resize(s);
		if (data_cuda) {
			copyCudaDeviceToHost((void*)&vec[0], data_cuda, s * byte);
	}
	}
	void  UVec::copyTo(
		std::vector<double>& vec
	) const {
		if (pimpl) pimpl->copyToVec(vec);
	}

	void UVec::resize(const size_t size)
	{
		if (pimpl) pimpl->resize(size, 0);
	}
	void UVec::resize(const size_t size, const double value)
	{
		if (pimpl) pimpl->resize(size, value);
	}
	UVecDepth UVec::depth() const
	{
		if (pimpl) return pimpl->depth();
		else return UVecDepth_Invalid;
	}
	UVecType UVec::type() const
	{
		if (pimpl) return pimpl->type();
		else return UVecType_Invalid;
	}
	void UVec::release() {
		if (pimpl) {
			pimpl->lock();
			pimpl->release();
			pimpl->unlock();
			if (pimpl->ref_counter == 0) 
				delete pimpl;
			pimpl = NULL;
		}
	}

	bool UVec::empty() const {
		if (pimpl) return pimpl->empty();
		else return true;
	}


	bool UVec::valid() const
	{
		if (pimpl) return pimpl->valid();
		else return false;
	}
	UVec::UVec(
		const UVecDepth depth,
		const UVecType type,
		const size_t size
	)
	{
		pimpl = new UVec_impl(depth, type, size);
	}
	UVec::UVec(
		const UVecDepth depth,
		const UVecType type
	)
	{
		pimpl = new UVec_impl(depth, type, 0);
	}
	UVec::UVec(
	)
	{
		pimpl = new UVec_impl();
	}
	UVec::~UVec()
	{
		if (pimpl) {
			pimpl->lock();
			pimpl->release();
			pimpl->unlock();
			if (pimpl->ref_counter == 0)
				delete pimpl;
		}
	}
	UVec UVec::clone() const
	{
		UVec uvec;
		if (pimpl) {
			*uvec.pimpl = *pimpl;
		}
		return uvec;
	}
	void UVec::copyTo(
		UVec& dst
	) const {
		if (this == &dst) return;
		if (pimpl == dst.pimpl) return;
		if (dst.pimpl->is_declare_outside()) {
			this->convertTo(dst, this->type());
		}
		else {
			dst = clone();
		}
	}

	UVec& UVec::operator=(const UVec& rhs) {
		if (this == &rhs) return *this;
		if (pimpl == rhs.pimpl) return *this;
		if (pimpl) {
			pimpl->lock();
			pimpl->release();
			pimpl->unlock();
			if (pimpl->ref_counter == 0)
				delete pimpl;
			pimpl = NULL;
		}

		if (rhs.pimpl) {
				rhs.pimpl->lock();
				rhs.pimpl->increment_ref_counter();
				pimpl = rhs.pimpl;
				rhs.pimpl->unlock();
		}
		return *this;
	}
	UVec::UVec(const UVec& rhs) {
		pimpl = NULL;
		if (rhs.pimpl) {
			rhs.pimpl->lock();
			rhs.pimpl->increment_ref_counter();
			pimpl = rhs.pimpl;
			rhs.pimpl->unlock();
		}
	}

	const void* UVec::ptr() const {
		if (pimpl) return pimpl->ptr();
		else return NULL;
	}
	void* UVec::ptr() {
		if (pimpl) return pimpl->ptr();
		else return NULL;
	}
	const void* UVec::vec_ptr() const {
		if (pimpl) return pimpl->vec_ptr();
		else return NULL;
	}
	void* UVec::vec_ptr() {
		if (pimpl) return pimpl->vec_ptr();
		else return NULL;
	}
	CudaStream* UVec::get_cudaStream() const {
		if (pimpl) return pimpl->cudaStream;
		else return NULL;
	}
	void UVec::set_cudaStream(CudaStream* cudaStream) {
		if (pimpl) {
			if (is_cuda(*this)) {
				if (pimpl->cudaStream == cudaStream) return;
				if (pimpl->cudaStream && !pimpl->is_cudaStreamDeclare_outside()) {
					delete pimpl->cudaStream;
				}
				pimpl->cudaStream = cudaStream;
				pimpl->set_cudaStreamDeclare_outside(true);
			}
		}
	}

	void average(const UVec& src1, const UVec& src2, UVec& dst) {
		if (src1.depth() != src2.depth()) {
			std::cerr << "Error: " << __func__ << "Depth mismatch: " << src1.depth() << " != " << src2.depth() << std::endl;
		}
		if ((src1.type() == src2.type()) && (src1.type() == UVecType_Cuda)) {
			cudaAverage(dst, src1, src2);
		}
		else {
			UVec src1_vec;
			UVec src2_vec;
			src1.convertTo(src1_vec, UVecType_Vector);
			src2.convertTo(src2_vec, UVecType_Vector);
			UVec dst_vec;
			if (dst.type() == UVecType_Vector) dst_vec = dst;
			else dst_vec = UVec(src1.depth(), UVecType_Vector, src1.size());
			const beacls::FloatVec* src1_vector = beacls::UVec_<FLOAT_TYPE>(src1_vec).vec();
			const beacls::FloatVec* src2_vector = beacls::UVec_<FLOAT_TYPE>(src2_vec).vec();
			beacls::FloatVec* dst_vector = beacls::UVec_<FLOAT_TYPE>(dst_vec).vec();
			std::transform(src1_vector->cbegin(), src1_vector->cend(), src2_vector->cbegin(), dst_vector->begin(), ([](const auto& lhs, const auto& rhs) {
				return (lhs + rhs) / 2;
			}));
		}

	}
	void copyDevicePtrToUVec(UVec& dst, const void* src, const size_t s) {
		void* dst_ptr = dst.ptr();
		if (dst_ptr != src) {
			size_t byte = depth_to_byte(dst.depth());
			switch (dst.type()) {
			case UVecType_Invalid:
			default:
				break;
			case UVecType_Vector:
				copyCudaDeviceToHost(dst_ptr, src, s*byte);
				break;
			case UVecType_Cuda:
				copyCudaDeviceToDevice(dst_ptr, src, s*byte);
				break;
			}
		}
	}
	void copyHostPtrToUVec(UVec& dst, const void* src, const size_t s) {
		void* dst_ptr = dst.ptr();
		if (dst_ptr != src) {
			size_t byte = depth_to_byte(dst.depth());
			switch (dst.type()) {
			case UVecType_Invalid:
			default:
				break;
			case UVecType_Vector:
				memcpy(dst_ptr, src, s*byte);
				break;
			case UVecType_Cuda:
				copyCudaHostToDevice(dst_ptr, src, s*byte);
				break;
			}
		}
	}
	void copyUVecToDevicePtr(void* dst, const UVec& src) {
		const void* src_ptr = src.ptr();
		if (src_ptr != dst) {
			size_t byte = depth_to_byte(src.depth());
			size_t s = src.size();
			switch (src.type()) {
			case UVecType_Invalid:
			default:
				break;
			case UVecType_Vector:
				copyCudaDeviceToHost(dst, src_ptr, s*byte);
				break;
			case UVecType_Cuda:
				copyCudaDeviceToDevice(dst, src_ptr, s*byte);
				break;
			}
		}
	}
	void copyUVecToHost(void* dst, const UVec& src) {
		const void* src_ptr = src.ptr();
		if (src_ptr != dst) {
			size_t byte = depth_to_byte(src.depth());
			size_t s = src.size();
			switch (src.type()) {
			case UVecType_Invalid:
			default:
				break;
			case UVecType_Vector:
				memcpy(dst, src_ptr, s*byte);
				break;
			case UVecType_Cuda:
				copyCudaDeviceToHost(dst, src_ptr, s*byte);
				break;
			}
		}
	}

	void copyDevicePtrToUVecAsync(UVec& dst, const void* src, const size_t s) {
		void* dst_ptr = dst.ptr();
		if (dst_ptr != src) {
			size_t byte = depth_to_byte(dst.depth());
			switch (dst.type()) {
			case UVecType_Invalid:
			default:
				break;
			case UVecType_Vector:
				copyCudaDeviceToHostAsync(dst_ptr, src, s*byte, dst.pimpl->cudaStream);
				break;
			case UVecType_Cuda:
				copyCudaDeviceToDeviceAsync(dst_ptr, src, s*byte, dst.pimpl->cudaStream);
				break;
			}
		}
	}
	void copyHostPtrToUVecAsync(UVec& dst, const void* src, const size_t s) {
		void* dst_ptr = dst.ptr();
		if (dst_ptr != src) {
			size_t byte = depth_to_byte(dst.depth());
			switch (dst.type()) {
			case UVecType_Invalid:
			default:
				break;
			case UVecType_Vector:
				memcpy(dst_ptr, src, s*byte);
				break;
			case UVecType_Cuda:
				copyCudaHostToDeviceAsync(dst_ptr, src, s*byte, dst.pimpl->cudaStream);
				break;
			}
		}
	}
	void copyUVecToDevicePtrAsync(void* dst, const UVec& src) {
		const void* src_ptr = src.ptr();
		if (src_ptr != dst) {
			size_t byte = depth_to_byte(src.depth());
			size_t s = src.size();
			switch (src.type()) {
			case UVecType_Invalid:
			default:
				break;
			case UVecType_Vector:
				copyCudaDeviceToHostAsync(dst, src_ptr, s*byte, src.pimpl->cudaStream);
				break;
			case UVecType_Cuda:
				copyCudaDeviceToDeviceAsync(dst, src_ptr, s*byte, src.pimpl->cudaStream);
				break;
			}
		}
	}
	void copyUVecToHostAsync(void* dst, const UVec& src) {
		const void* src_ptr = src.ptr();
		if (src_ptr != dst) {
			size_t byte = depth_to_byte(src.depth());
			size_t s = src.size();
			switch (src.type()) {
			case UVecType_Invalid:
			default:
				break;
			case UVecType_Vector:
				memcpy(dst, src_ptr, s*byte);
				break;
			case UVecType_Cuda:
				copyCudaDeviceToHostAsync(dst, src_ptr, s*byte, src.pimpl->cudaStream);
				break;
			}
		}
	}
	void synchronizeUVec(const UVec& src) {
		if (is_cuda(src)) {
			synchronizeCuda(src.pimpl->cudaStream);
		}
	}

	int get_num_of_gpus() {
		return get_num_of_gpus_impl();
	}
	void set_gpu_id(const int id) {
		set_gpu_id_impl(id);
	}
	void reallocateAsSrc(beacls::UVec& dst, const::beacls::UVec& src) {
		if (dst.type() != src.type()) dst = beacls::UVec(src.depth(), src.type(), src.size());
		else dst.resize(src.size());
	}
	bool is_cuda(const::beacls::UVec& src) {
		if (src.type() == UVecType_Cuda) return true;
		else return false;
	}

}// bears
