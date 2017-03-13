// CUDA runtime
#include <cuda_runtime.h>

#include <typedef.hpp>
#include <cuda_macro.hpp>
#include <thrust/device_ptr.h>
#include <thrust/tuple.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/system/cuda/execution_policy.h>
#include <vector>
#include <iostream>
#include "DynSysSchemeData_cuda.hpp"
#include <Core/UVec.hpp>
#include <Core/CudaStream.hpp>
#if defined(WITH_GPU) 
#if defined(USER_DEFINED_GPU_DYNSYS_FUNC)

struct GetHamValue3_dim0
{
	template<typename Tuple>
	__host__ __device__
	FLOAT_TYPE operator()(const Tuple v) const
	{
		FLOAT_TYPE b = thrust::get<0>(v);
		FLOAT_TYPE c = thrust::get<1>(v);
		return b * c;
	}
};
struct GetHamValue3_dimNot0
{
	template<typename Tuple>
	__host__ __device__
	void operator()(Tuple v) const
	{
		FLOAT_TYPE b = thrust::get<1>(v);
		FLOAT_TYPE c = thrust::get<2>(v);
		thrust::get<0>(v) += b * c;
	}
};
struct GetHamValue3_dimNot0Neg
{
	template<typename Tuple>
	__host__ __device__
	void operator()(Tuple v) const
	{
		FLOAT_TYPE b = thrust::get<1>(v);
		FLOAT_TYPE c = thrust::get<2>(v);
		const FLOAT_TYPE r = thrust::get<0>(v) + b * c;
		thrust::get<0>(v) = -r;
	}
};
struct GetHamValue2
{
	const FLOAT_TYPE b;
	GetHamValue2(const FLOAT_TYPE b) : b(b) {}
	template<typename Tuple>
	__host__ __device__
	void operator()(Tuple v) const
	{
		FLOAT_TYPE c = thrust::get<1>(v);
		thrust::get<0>(v) += b * c;
	}
};
struct GetHamValue2Neg
{
	const FLOAT_TYPE b;
	GetHamValue2Neg(const FLOAT_TYPE b) : b(b) {}
	template<typename Tuple>
	__host__ __device__
	void operator()(Tuple v) const
	{
		const FLOAT_TYPE c = thrust::get<1>(v);
		const FLOAT_TYPE r = thrust::get<0>(v) + b * c;
		thrust::get<0>(v) = -r;
	}
};

struct GetHamValue_1dim
{
	GetHamValue_1dim() {}
	template<typename Tuple>
	__host__ __device__
	FLOAT_TYPE operator()(const Tuple v) const
	{
		const FLOAT_TYPE deriv0 = thrust::get<0>(v);
		const FLOAT_TYPE dx0 = thrust::get<1>(v);
		FLOAT_TYPE r = deriv0 * dx0;
		return r;
	}
};

struct GetHamValue_1dimNeg
{
	GetHamValue_1dimNeg() {}
	template<typename Tuple>
	__host__ __device__
	FLOAT_TYPE operator()(const Tuple v) const
	{
		const FLOAT_TYPE deriv0 = thrust::get<0>(v);
		const FLOAT_TYPE dx0 = thrust::get<1>(v);
		FLOAT_TYPE r = deriv0 * dx0;
		return -r;
	}
};

struct GetHamValue_2dim
{
	GetHamValue_2dim() {}
	template<typename Tuple>
	__host__ __device__
	FLOAT_TYPE operator()(const Tuple v) const
	{
		const FLOAT_TYPE deriv0 = thrust::get<0>(v);
		const FLOAT_TYPE deriv1 = thrust::get<1>(v);
		const FLOAT_TYPE dx0 = thrust::get<2>(v);
		const FLOAT_TYPE dx1 = thrust::get<3>(v);
		FLOAT_TYPE r = deriv0 * dx0;
		r += deriv1 * dx1;
		return r;
	}
};

struct GetHamValue_2dimNeg
{
	GetHamValue_2dimNeg() {}
	template<typename Tuple>
	__host__ __device__
	FLOAT_TYPE operator()(const Tuple v) const
	{
		const FLOAT_TYPE deriv0 = thrust::get<0>(v);
		const FLOAT_TYPE deriv1 = thrust::get<1>(v);
		const FLOAT_TYPE dx0 = thrust::get<2>(v);
		const FLOAT_TYPE dx1 = thrust::get<3>(v);
		FLOAT_TYPE r = deriv0 * dx0;
		r += deriv1 * dx1;
		return -r;
	}
};

struct GetHamValue_3dim
{
	GetHamValue_3dim() {}
	template<typename Tuple>
	__host__ __device__
	FLOAT_TYPE operator()(const Tuple v) const
	{
		const FLOAT_TYPE deriv0 = thrust::get<0>(v);
		const FLOAT_TYPE deriv1 = thrust::get<1>(v);
		const FLOAT_TYPE deriv2 = thrust::get<2>(v);
		const FLOAT_TYPE dx0 = thrust::get<3>(v);
		const FLOAT_TYPE dx1 = thrust::get<4>(v);
		const FLOAT_TYPE dx2 = thrust::get<5>(v);
		FLOAT_TYPE r = deriv0 * dx0;
		r += deriv1 * dx1;
		r += deriv2 * dx2;
		return r;
	}
};

struct GetHamValue_3dimNeg
{
	GetHamValue_3dimNeg() {}
	template<typename Tuple>
	__host__ __device__
	FLOAT_TYPE operator()(const Tuple v) const
	{
		const FLOAT_TYPE deriv0 = thrust::get<0>(v);
		const FLOAT_TYPE deriv1 = thrust::get<1>(v);
		const FLOAT_TYPE deriv2 = thrust::get<2>(v);
		const FLOAT_TYPE dx0 = thrust::get<3>(v);
		const FLOAT_TYPE dx1 = thrust::get<4>(v);
		const FLOAT_TYPE dx2 = thrust::get<5>(v);
		FLOAT_TYPE r = deriv0 * dx0;
		r += deriv1 * dx1;
		r += deriv2 * dx2;
		return -r;
	}
};

struct GetHamValue_4dim
{
	GetHamValue_4dim() {}
	template<typename Tuple>
	__host__ __device__
	FLOAT_TYPE operator()(const Tuple v) const
	{
		const FLOAT_TYPE deriv0 = thrust::get<0>(v);
		const FLOAT_TYPE deriv1 = thrust::get<1>(v);
		const FLOAT_TYPE deriv2 = thrust::get<2>(v);
		const FLOAT_TYPE deriv3 = thrust::get<3>(v);
		const FLOAT_TYPE dx0 = thrust::get<4>(v);
		const FLOAT_TYPE dx1 = thrust::get<5>(v);
		const FLOAT_TYPE dx2 = thrust::get<6>(v);
		const FLOAT_TYPE dx3 = thrust::get<7>(v);
		FLOAT_TYPE r = deriv0 * dx0;
		r += deriv1 * dx1;
		r += deriv2 * dx2;
		r += deriv3 * dx3;
		return r;
	}
};

struct GetHamValue_4dimNeg
{
	GetHamValue_4dimNeg() {}
	template<typename Tuple>
	__host__ __device__
	FLOAT_TYPE operator()(const Tuple v) const
	{
		const FLOAT_TYPE deriv0 = thrust::get<0>(v);
		const FLOAT_TYPE deriv1 = thrust::get<1>(v);
		const FLOAT_TYPE deriv2 = thrust::get<2>(v);
		const FLOAT_TYPE deriv3 = thrust::get<3>(v);
		const FLOAT_TYPE dx0 = thrust::get<4>(v);
		const FLOAT_TYPE dx1 = thrust::get<5>(v);
		const FLOAT_TYPE dx2 = thrust::get<6>(v);
		const FLOAT_TYPE dx3 = thrust::get<7>(v);
		FLOAT_TYPE r = deriv0 * dx0;
		r += deriv1 * dx1;
		r += deriv2 * dx2;
		r += deriv3 * dx3;
		return -r;
	}
};

struct GetHamValue_5dim
{
	GetHamValue_5dim() {}
	template<typename Tuple>
	__host__ __device__
	FLOAT_TYPE operator()(const Tuple v) const
	{
		const FLOAT_TYPE deriv0 = thrust::get<0>(v);
		const FLOAT_TYPE deriv1 = thrust::get<1>(v);
		const FLOAT_TYPE deriv2 = thrust::get<2>(v);
		const FLOAT_TYPE deriv3 = thrust::get<3>(v);
		const FLOAT_TYPE deriv4 = thrust::get<4>(v);
		const FLOAT_TYPE dx0 = thrust::get<5>(v);
		const FLOAT_TYPE dx1 = thrust::get<6>(v);
		const FLOAT_TYPE dx2 = thrust::get<7>(v);
		const FLOAT_TYPE dx3 = thrust::get<8>(v);
		const FLOAT_TYPE dx4 = thrust::get<9>(v);
		FLOAT_TYPE r = deriv0 * dx0;
		r += deriv1 * dx1;
		r += deriv2 * dx2;
		r += deriv3 * dx3;
		r += deriv4 * dx4;
		return r;
	}
};

struct GetHamValue_5dimNeg
{
	GetHamValue_5dimNeg() {}
	template<typename Tuple>
	__host__ __device__
	FLOAT_TYPE operator()(const Tuple v) const
	{
		const FLOAT_TYPE deriv0 = thrust::get<0>(v);
		const FLOAT_TYPE deriv1 = thrust::get<1>(v);
		const FLOAT_TYPE deriv2 = thrust::get<2>(v);
		const FLOAT_TYPE deriv3 = thrust::get<3>(v);
		const FLOAT_TYPE deriv4 = thrust::get<4>(v);
		const FLOAT_TYPE dx0 = thrust::get<5>(v);
		const FLOAT_TYPE dx1 = thrust::get<6>(v);
		const FLOAT_TYPE dx2 = thrust::get<7>(v);
		const FLOAT_TYPE dx3 = thrust::get<8>(v);
		const FLOAT_TYPE dx4 = thrust::get<9>(v);
		FLOAT_TYPE r = deriv0 * dx0;
		r += deriv1 * dx1;
		r += deriv2 * dx2;
		r += deriv3 * dx3;
		r += deriv4 * dx4;
		return -r;
	}
};


struct GetHamValueTIdx_1dim
{
	const FLOAT_TYPE TIderiv;
	GetHamValueTIdx_1dim(const FLOAT_TYPE TIderiv) : TIderiv(TIderiv) {}
	template<typename Tuple>
	__host__ __device__
	FLOAT_TYPE operator()(const Tuple v) const
	{
		const FLOAT_TYPE deriv0 = thrust::get<0>(v);
		const FLOAT_TYPE dx0 = thrust::get<1>(v);
		const FLOAT_TYPE TIdx0 = thrust::get<2>(v);
		FLOAT_TYPE r = deriv0 * dx0;
		r += TIderiv * TIdx0;
		return r;
	}
};

struct GetHamValueTIdx_1dimNeg
{
	const FLOAT_TYPE TIderiv;
	GetHamValueTIdx_1dimNeg(const FLOAT_TYPE TIderiv) : TIderiv(TIderiv) {}
	template<typename Tuple>
	__host__ __device__
	FLOAT_TYPE operator()(const Tuple v) const
	{
		const FLOAT_TYPE deriv0 = thrust::get<0>(v);
		const FLOAT_TYPE dx0 = thrust::get<1>(v);
		const FLOAT_TYPE TIdx0 = thrust::get<2>(v);
		FLOAT_TYPE r = deriv0 * dx0;
		r += TIderiv * TIdx0;
		return -r;
	}
};


struct GetHamValueTIdx_2dim
{
	const FLOAT_TYPE TIderiv;
	GetHamValueTIdx_2dim(const FLOAT_TYPE TIderiv) : TIderiv(TIderiv) {}
	template<typename Tuple>
	__host__ __device__
	FLOAT_TYPE operator()(const Tuple v) const
	{
		const FLOAT_TYPE deriv0 = thrust::get<0>(v);
		const FLOAT_TYPE deriv1 = thrust::get<1>(v);
		const FLOAT_TYPE dx0 = thrust::get<2>(v);
		const FLOAT_TYPE dx1 = thrust::get<3>(v);
		const FLOAT_TYPE TIdx0 = thrust::get<4>(v);
		const FLOAT_TYPE TIdx1 = thrust::get<5>(v);
		FLOAT_TYPE r = deriv0 * dx0;
		r += deriv1 * dx1;
		r += TIderiv * TIdx0;
		r += TIderiv * TIdx1;
		return r;
	}
};

struct GetHamValueTIdx_2dimNeg
{
	const FLOAT_TYPE TIderiv;
	GetHamValueTIdx_2dimNeg(const FLOAT_TYPE TIderiv) : TIderiv(TIderiv) {}
	template<typename Tuple>
	__host__ __device__
	FLOAT_TYPE operator()(const Tuple v) const
	{
		const FLOAT_TYPE deriv0 = thrust::get<0>(v);
		const FLOAT_TYPE deriv1 = thrust::get<1>(v);
		const FLOAT_TYPE dx0 = thrust::get<2>(v);
		const FLOAT_TYPE dx1 = thrust::get<3>(v);
		const FLOAT_TYPE TIdx0 = thrust::get<4>(v);
		const FLOAT_TYPE TIdx1 = thrust::get<5>(v);
		FLOAT_TYPE r = deriv0 * dx0;
		r += deriv1 * dx1;
		r += TIderiv * TIdx0;
		r += TIderiv * TIdx1;
		return -r;
	}
};


struct GetHamValueTIdx_3dim
{
	const FLOAT_TYPE TIderiv;
	GetHamValueTIdx_3dim(const FLOAT_TYPE TIderiv) : TIderiv(TIderiv) {}
	template<typename Tuple>
	__host__ __device__
	FLOAT_TYPE operator()(const Tuple v) const
	{
		const FLOAT_TYPE deriv0 = thrust::get<0>(v);
		const FLOAT_TYPE deriv1 = thrust::get<1>(v);
		const FLOAT_TYPE deriv2 = thrust::get<2>(v);
		const FLOAT_TYPE dx0 = thrust::get<3>(v);
		const FLOAT_TYPE dx1 = thrust::get<4>(v);
		const FLOAT_TYPE dx2 = thrust::get<5>(v);
		const FLOAT_TYPE TIdx0 = thrust::get<6>(v);
		const FLOAT_TYPE TIdx1 = thrust::get<7>(v);
		const FLOAT_TYPE TIdx2 = thrust::get<8>(v);
		FLOAT_TYPE r = deriv0 * dx0;
		r += deriv1 * dx1;
		r += deriv2 * dx2;
		r += TIderiv * TIdx0;
		r += TIderiv * TIdx1;
		r += TIderiv * TIdx2;
		return r;
	}
};

struct GetHamValueTIdx_3dimNeg
{
	const FLOAT_TYPE TIderiv;
	GetHamValueTIdx_3dimNeg(const FLOAT_TYPE TIderiv) : TIderiv(TIderiv) {}
	template<typename Tuple>
	__host__ __device__
	FLOAT_TYPE operator()(const Tuple v) const
	{
		const FLOAT_TYPE deriv0 = thrust::get<0>(v);
		const FLOAT_TYPE deriv1 = thrust::get<1>(v);
		const FLOAT_TYPE deriv2 = thrust::get<2>(v);
		const FLOAT_TYPE dx0 = thrust::get<3>(v);
		const FLOAT_TYPE dx1 = thrust::get<4>(v);
		const FLOAT_TYPE dx2 = thrust::get<5>(v);
		const FLOAT_TYPE TIdx0 = thrust::get<6>(v);
		const FLOAT_TYPE TIdx1 = thrust::get<7>(v);
		const FLOAT_TYPE TIdx2 = thrust::get<8>(v);
		FLOAT_TYPE r = deriv0 * dx0;
		r += deriv1 * dx1;
		r += deriv2 * dx2;
		r += TIderiv * TIdx0;
		r += TIderiv * TIdx1;
		r += TIderiv * TIdx2;
		return -r;
	}
};

bool hamFunc_exec_cuda(
	beacls::UVec& hamValue_uvec,
	const std::vector<beacls::UVec>& deriv_uvecs,
	const std::vector<beacls::UVec>& dx_uvecs,
	const std::vector<beacls::UVec>& TIdx_uvecs,
	const FLOAT_TYPE TIderiv,
	const bool TIdim,
	const bool negate
	) {
	if (dx_uvecs.empty()) return false;
	beacls::reallocateAsSrc(hamValue_uvec, dx_uvecs[0]);
	FLOAT_TYPE* hamValue = beacls::UVec_<FLOAT_TYPE>(hamValue_uvec).ptr();
	const size_t num_of_dimensions = deriv_uvecs.size();
	if (beacls::is_cuda(hamValue_uvec)) {
		thrust::device_ptr<FLOAT_TYPE> hamValue_dev_ptr = thrust::device_pointer_cast(hamValue);
		cudaStream_t ham_stream = beacls::get_stream(hamValue_uvec);
		if (!TIdim) {
			switch(num_of_dimensions) {
			case 1:
			{
				const FLOAT_TYPE* deriv0_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[0]).ptr();
				beacls::synchronizeUVec(dx_uvecs[0]);
				const FLOAT_TYPE* dx0_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[0]).ptr();
				thrust::device_ptr<const FLOAT_TYPE> deriv0_dev_ptr = thrust::device_pointer_cast(deriv0_ptr);
				thrust::device_ptr<const FLOAT_TYPE> dx0_dev_ptr = thrust::device_pointer_cast(dx0_ptr);

				auto src_Tuple = thrust::make_tuple(deriv0_dev_ptr, 
					dx0_dev_ptr);
				auto src_ite = thrust::make_zip_iterator(src_Tuple);
				if (negate) {
					thrust::transform(thrust::cuda::par.on(ham_stream),
						src_ite, src_ite + dx_uvecs[0].size(), hamValue_dev_ptr, 
						GetHamValue_1dimNeg());
				}
				else {
					thrust::transform(thrust::cuda::par.on(ham_stream),
						src_ite, src_ite + dx_uvecs[0].size(), hamValue_dev_ptr, 
						GetHamValue_1dim());
				}
			}
				break;
			case 2:
			{
				const FLOAT_TYPE* deriv0_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[0]).ptr();
				beacls::synchronizeUVec(dx_uvecs[0]);
				const FLOAT_TYPE* dx0_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[0]).ptr();
				thrust::device_ptr<const FLOAT_TYPE> deriv0_dev_ptr = thrust::device_pointer_cast(deriv0_ptr);
				thrust::device_ptr<const FLOAT_TYPE> dx0_dev_ptr = thrust::device_pointer_cast(dx0_ptr);

				const FLOAT_TYPE* deriv1_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[1]).ptr();
				const FLOAT_TYPE* dx1_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[1]).ptr();
				thrust::device_ptr<const FLOAT_TYPE> deriv1_dev_ptr = thrust::device_pointer_cast(deriv1_ptr);
				thrust::device_ptr<const FLOAT_TYPE> dx1_dev_ptr = thrust::device_pointer_cast(dx1_ptr);
				auto src_Tuple = thrust::make_tuple(deriv0_dev_ptr, deriv1_dev_ptr,
					dx0_dev_ptr, dx1_dev_ptr);
				auto src_ite = thrust::make_zip_iterator(src_Tuple);
				if (negate) {
					thrust::transform(thrust::cuda::par.on(ham_stream),
						src_ite, src_ite + dx_uvecs[0].size(), hamValue_dev_ptr, 
						GetHamValue_2dimNeg());
				}
				else {
					thrust::transform(thrust::cuda::par.on(ham_stream),
						src_ite, src_ite + dx_uvecs[0].size(), hamValue_dev_ptr, 
						GetHamValue_2dim());
				}
			}
				break;
			case 3:
			{
				const FLOAT_TYPE* deriv0_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[0]).ptr();
				beacls::synchronizeUVec(dx_uvecs[0]);
				const FLOAT_TYPE* dx0_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[0]).ptr();
				thrust::device_ptr<const FLOAT_TYPE> deriv0_dev_ptr = thrust::device_pointer_cast(deriv0_ptr);
				thrust::device_ptr<const FLOAT_TYPE> dx0_dev_ptr = thrust::device_pointer_cast(dx0_ptr);

				const FLOAT_TYPE* deriv1_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[1]).ptr();
				const FLOAT_TYPE* dx1_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[1]).ptr();
				thrust::device_ptr<const FLOAT_TYPE> deriv1_dev_ptr = thrust::device_pointer_cast(deriv1_ptr);
				thrust::device_ptr<const FLOAT_TYPE> dx1_dev_ptr = thrust::device_pointer_cast(dx1_ptr);

				const FLOAT_TYPE* deriv2_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[2]).ptr();
				const FLOAT_TYPE* dx2_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[2]).ptr();
				thrust::device_ptr<const FLOAT_TYPE> deriv2_dev_ptr = thrust::device_pointer_cast(deriv2_ptr);
				thrust::device_ptr<const FLOAT_TYPE> dx2_dev_ptr = thrust::device_pointer_cast(dx2_ptr);
				auto src_Tuple = thrust::make_tuple(deriv0_dev_ptr, deriv1_dev_ptr, deriv2_dev_ptr, 
					dx0_dev_ptr, dx1_dev_ptr, dx2_dev_ptr);
				auto src_ite = thrust::make_zip_iterator(src_Tuple);
				if (negate) {
					thrust::transform(thrust::cuda::par.on(ham_stream),
						src_ite, src_ite + dx_uvecs[0].size(), hamValue_dev_ptr, 
						GetHamValue_3dimNeg());
				}
				else {
					thrust::transform(thrust::cuda::par.on(ham_stream),
						src_ite, src_ite + dx_uvecs[0].size(), hamValue_dev_ptr, 
						GetHamValue_3dim());
				}
			}
				break;
			case 4:
			{
				const FLOAT_TYPE* deriv0_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[0]).ptr();
				beacls::synchronizeUVec(dx_uvecs[0]);
				const FLOAT_TYPE* dx0_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[0]).ptr();
				thrust::device_ptr<const FLOAT_TYPE> deriv0_dev_ptr = thrust::device_pointer_cast(deriv0_ptr);
				thrust::device_ptr<const FLOAT_TYPE> dx0_dev_ptr = thrust::device_pointer_cast(dx0_ptr);

				const FLOAT_TYPE* deriv1_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[1]).ptr();
				const FLOAT_TYPE* dx1_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[1]).ptr();
				thrust::device_ptr<const FLOAT_TYPE> deriv1_dev_ptr = thrust::device_pointer_cast(deriv1_ptr);
				thrust::device_ptr<const FLOAT_TYPE> dx1_dev_ptr = thrust::device_pointer_cast(dx1_ptr);

				const FLOAT_TYPE* deriv2_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[2]).ptr();
				const FLOAT_TYPE* dx2_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[2]).ptr();
				thrust::device_ptr<const FLOAT_TYPE> deriv2_dev_ptr = thrust::device_pointer_cast(deriv2_ptr);
				thrust::device_ptr<const FLOAT_TYPE> dx2_dev_ptr = thrust::device_pointer_cast(dx2_ptr);

				const FLOAT_TYPE* deriv3_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[3]).ptr();
				const FLOAT_TYPE* dx3_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[3]).ptr();
				thrust::device_ptr<const FLOAT_TYPE> deriv3_dev_ptr = thrust::device_pointer_cast(deriv3_ptr);
				thrust::device_ptr<const FLOAT_TYPE> dx3_dev_ptr = thrust::device_pointer_cast(dx3_ptr);

				auto src_Tuple = thrust::make_tuple(deriv0_dev_ptr, deriv1_dev_ptr, deriv2_dev_ptr, deriv3_dev_ptr,
					dx0_dev_ptr, dx1_dev_ptr, dx2_dev_ptr, dx3_dev_ptr);
				auto src_ite = thrust::make_zip_iterator(src_Tuple);
				if (negate) {
					thrust::transform(thrust::cuda::par.on(ham_stream),
						src_ite, src_ite + dx_uvecs[0].size(), hamValue_dev_ptr, 
						GetHamValue_4dimNeg());
				}
				else {
					thrust::transform(thrust::cuda::par.on(ham_stream),
						src_ite, src_ite + dx_uvecs[0].size(), hamValue_dev_ptr, 
						GetHamValue_4dim());
				}
			}
				break;
			case 5:
			{
				const FLOAT_TYPE* deriv0_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[0]).ptr();
				beacls::synchronizeUVec(dx_uvecs[0]);
				const FLOAT_TYPE* dx0_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[0]).ptr();
				thrust::device_ptr<const FLOAT_TYPE> deriv0_dev_ptr = thrust::device_pointer_cast(deriv0_ptr);
				thrust::device_ptr<const FLOAT_TYPE> dx0_dev_ptr = thrust::device_pointer_cast(dx0_ptr);

				const FLOAT_TYPE* deriv1_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[1]).ptr();
				const FLOAT_TYPE* dx1_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[1]).ptr();
				thrust::device_ptr<const FLOAT_TYPE> deriv1_dev_ptr = thrust::device_pointer_cast(deriv1_ptr);
				thrust::device_ptr<const FLOAT_TYPE> dx1_dev_ptr = thrust::device_pointer_cast(dx1_ptr);

				const FLOAT_TYPE* deriv2_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[2]).ptr();
				const FLOAT_TYPE* dx2_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[2]).ptr();
				thrust::device_ptr<const FLOAT_TYPE> deriv2_dev_ptr = thrust::device_pointer_cast(deriv2_ptr);
				thrust::device_ptr<const FLOAT_TYPE> dx2_dev_ptr = thrust::device_pointer_cast(dx2_ptr);

				const FLOAT_TYPE* deriv3_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[3]).ptr();
				const FLOAT_TYPE* dx3_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[3]).ptr();
				thrust::device_ptr<const FLOAT_TYPE> deriv3_dev_ptr = thrust::device_pointer_cast(deriv3_ptr);
				thrust::device_ptr<const FLOAT_TYPE> dx3_dev_ptr = thrust::device_pointer_cast(dx3_ptr);

				const FLOAT_TYPE* deriv4_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[4]).ptr();
				const FLOAT_TYPE* dx4_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[4]).ptr();
				thrust::device_ptr<const FLOAT_TYPE> deriv4_dev_ptr = thrust::device_pointer_cast(deriv4_ptr);
				thrust::device_ptr<const FLOAT_TYPE> dx4_dev_ptr = thrust::device_pointer_cast(dx4_ptr);
				auto src_Tuple = thrust::make_tuple(deriv0_dev_ptr, deriv1_dev_ptr, deriv2_dev_ptr, deriv3_dev_ptr, deriv4_dev_ptr,
					dx0_dev_ptr, dx1_dev_ptr, dx2_dev_ptr, dx3_dev_ptr, dx4_dev_ptr);
				auto src_ite = thrust::make_zip_iterator(src_Tuple);
				if (negate) {
					thrust::transform(thrust::cuda::par.on(ham_stream),
						src_ite, src_ite + dx_uvecs[0].size(), hamValue_dev_ptr, 
						GetHamValue_5dimNeg());
				}
				else {
					thrust::transform(thrust::cuda::par.on(ham_stream),
						src_ite, src_ite + dx_uvecs[0].size(), hamValue_dev_ptr, 
						GetHamValue_5dim());
				}
			}
				break;
			default:
				beacls::synchronizeUVec(dx_uvecs[0]);
				for (size_t dimension = 0; dimension < num_of_dimensions; ++dimension) {
					const FLOAT_TYPE* deriv_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[dimension]).ptr();
					const FLOAT_TYPE* dx_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[dimension]).ptr();
					
					thrust::device_ptr<const FLOAT_TYPE> deriv_dev_ptr = thrust::device_pointer_cast(deriv_ptr);
					thrust::device_ptr<const FLOAT_TYPE> dx_dev_ptr = thrust::device_pointer_cast(dx_ptr);
					if (dimension==0){
						auto src_Tuple = thrust::make_tuple(deriv_dev_ptr, dx_dev_ptr);
						auto src_ite = thrust::make_zip_iterator(src_Tuple);
						thrust::transform(thrust::cuda::par.on(ham_stream),
							src_ite, src_ite + dx_uvecs[dimension].size(), hamValue_dev_ptr, 
							GetHamValue3_dim0());
					}
					else {
						auto src_dst_Tuple = thrust::make_tuple(hamValue_dev_ptr, deriv_dev_ptr, dx_dev_ptr);
						auto src_dst_ite = thrust::make_zip_iterator(src_dst_Tuple);
						thrust::for_each(thrust::cuda::par.on(ham_stream),
							src_dst_ite, src_dst_ite + dx_uvecs[dimension].size(), 
							GetHamValue3_dimNot0());
					}
				}
				break;
			}
		} else {
			switch(num_of_dimensions) {
			case 1:
			{
				const FLOAT_TYPE* deriv0_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[0]).ptr();
				beacls::synchronizeUVec(dx_uvecs[0]);
				const FLOAT_TYPE* dx0_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[0]).ptr();
				thrust::device_ptr<const FLOAT_TYPE> deriv0_dev_ptr = thrust::device_pointer_cast(deriv0_ptr);
				thrust::device_ptr<const FLOAT_TYPE> dx0_dev_ptr = thrust::device_pointer_cast(dx0_ptr);
				const FLOAT_TYPE* TIdx0_ptr = beacls::UVec_<FLOAT_TYPE>(TIdx_uvecs[0]).ptr();
				thrust::device_ptr<const FLOAT_TYPE> TIdx0_dev_ptr = thrust::device_pointer_cast(TIdx0_ptr);

				auto src_Tuple = thrust::make_tuple(deriv0_dev_ptr,
					dx0_dev_ptr,
					TIdx0_dev_ptr
					);
				auto src_ite = thrust::make_zip_iterator(src_Tuple);
				if (negate) {
					thrust::transform(thrust::cuda::par.on(ham_stream),
						src_ite, src_ite + dx_uvecs[0].size(), hamValue_dev_ptr, 
						GetHamValueTIdx_1dimNeg(TIderiv));
				}
				else {
					thrust::transform(thrust::cuda::par.on(ham_stream),
						src_ite, src_ite + dx_uvecs[0].size(), hamValue_dev_ptr, 
						GetHamValueTIdx_1dim(TIderiv));
				}
			}
				break;
			case 2:
			{
				const FLOAT_TYPE* deriv0_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[0]).ptr();
				beacls::synchronizeUVec(dx_uvecs[0]);
				const FLOAT_TYPE* dx0_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[0]).ptr();
				thrust::device_ptr<const FLOAT_TYPE> deriv0_dev_ptr = thrust::device_pointer_cast(deriv0_ptr);
				thrust::device_ptr<const FLOAT_TYPE> dx0_dev_ptr = thrust::device_pointer_cast(dx0_ptr);
				beacls::synchronizeUVec(TIdx_uvecs[0]);
				const FLOAT_TYPE* TIdx0_ptr = beacls::UVec_<FLOAT_TYPE>(TIdx_uvecs[0]).ptr();
				thrust::device_ptr<const FLOAT_TYPE> TIdx0_dev_ptr = thrust::device_pointer_cast(TIdx0_ptr);

				const FLOAT_TYPE* deriv1_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[1]).ptr();
				const FLOAT_TYPE* dx1_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[1]).ptr();
				thrust::device_ptr<const FLOAT_TYPE> deriv1_dev_ptr = thrust::device_pointer_cast(deriv1_ptr);
				thrust::device_ptr<const FLOAT_TYPE> dx1_dev_ptr = thrust::device_pointer_cast(dx1_ptr);
				const FLOAT_TYPE* TIdx1_ptr = beacls::UVec_<FLOAT_TYPE>(TIdx_uvecs[1]).ptr();
				thrust::device_ptr<const FLOAT_TYPE> TIdx1_dev_ptr = thrust::device_pointer_cast(TIdx1_ptr);

				auto src_Tuple = thrust::make_tuple(deriv0_dev_ptr, deriv1_dev_ptr,
					dx0_dev_ptr, dx1_dev_ptr,
					TIdx0_dev_ptr, TIdx1_dev_ptr
					);
				auto src_ite = thrust::make_zip_iterator(src_Tuple);
				if (negate) {
					thrust::transform(thrust::cuda::par.on(ham_stream),
						src_ite, src_ite + dx_uvecs[0].size(), hamValue_dev_ptr, 
						GetHamValueTIdx_2dimNeg(TIderiv));
				}
				else {
					thrust::transform(thrust::cuda::par.on(ham_stream),
						src_ite, src_ite + dx_uvecs[0].size(), hamValue_dev_ptr, 
						GetHamValueTIdx_2dim(TIderiv));
				}
			}
				break;
			case 3:
			{
				const FLOAT_TYPE* deriv0_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[0]).ptr();
				beacls::synchronizeUVec(dx_uvecs[0]);
				const FLOAT_TYPE* dx0_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[0]).ptr();
				thrust::device_ptr<const FLOAT_TYPE> deriv0_dev_ptr = thrust::device_pointer_cast(deriv0_ptr);
				thrust::device_ptr<const FLOAT_TYPE> dx0_dev_ptr = thrust::device_pointer_cast(dx0_ptr);
				beacls::synchronizeUVec(TIdx_uvecs[0]);
				const FLOAT_TYPE* TIdx0_ptr = beacls::UVec_<FLOAT_TYPE>(TIdx_uvecs[0]).ptr();
				thrust::device_ptr<const FLOAT_TYPE> TIdx0_dev_ptr = thrust::device_pointer_cast(TIdx0_ptr);

				const FLOAT_TYPE* deriv1_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[1]).ptr();
				const FLOAT_TYPE* dx1_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[1]).ptr();
				thrust::device_ptr<const FLOAT_TYPE> deriv1_dev_ptr = thrust::device_pointer_cast(deriv1_ptr);
				thrust::device_ptr<const FLOAT_TYPE> dx1_dev_ptr = thrust::device_pointer_cast(dx1_ptr);
				const FLOAT_TYPE* TIdx1_ptr = beacls::UVec_<FLOAT_TYPE>(TIdx_uvecs[1]).ptr();
				thrust::device_ptr<const FLOAT_TYPE> TIdx1_dev_ptr = thrust::device_pointer_cast(TIdx1_ptr);

				const FLOAT_TYPE* deriv2_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[2]).ptr();
				const FLOAT_TYPE* dx2_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[2]).ptr();
				thrust::device_ptr<const FLOAT_TYPE> deriv2_dev_ptr = thrust::device_pointer_cast(deriv2_ptr);
				thrust::device_ptr<const FLOAT_TYPE> dx2_dev_ptr = thrust::device_pointer_cast(dx2_ptr);
				const FLOAT_TYPE* TIdx2_ptr = beacls::UVec_<FLOAT_TYPE>(TIdx_uvecs[2]).ptr();
				thrust::device_ptr<const FLOAT_TYPE> TIdx2_dev_ptr = thrust::device_pointer_cast(TIdx2_ptr);

				auto src_Tuple = thrust::make_tuple(deriv0_dev_ptr, deriv1_dev_ptr, deriv2_dev_ptr, 
					dx0_dev_ptr, dx1_dev_ptr, dx2_dev_ptr, 
					TIdx0_dev_ptr, TIdx1_dev_ptr, TIdx2_dev_ptr
					);
				auto src_ite = thrust::make_zip_iterator(src_Tuple);
				if (negate) {
					thrust::transform(thrust::cuda::par.on(ham_stream),
						src_ite, src_ite + dx_uvecs[0].size(), hamValue_dev_ptr, 
						GetHamValueTIdx_3dimNeg(TIderiv));
				}
				else {
					thrust::transform(thrust::cuda::par.on(ham_stream),
						src_ite, src_ite + dx_uvecs[0].size(), hamValue_dev_ptr, 
						GetHamValueTIdx_3dim(TIderiv));
				}
			}
				break;
			default:
					beacls::synchronizeUVec(dx_uvecs[0]);
				for (size_t dimension = 0; dimension < num_of_dimensions; ++dimension) {
					const FLOAT_TYPE* deriv_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[dimension]).ptr();
					const FLOAT_TYPE* dx_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[dimension]).ptr();
					
					thrust::device_ptr<const FLOAT_TYPE> deriv_dev_ptr = thrust::device_pointer_cast(deriv_ptr);
					thrust::device_ptr<const FLOAT_TYPE> dx_dev_ptr = thrust::device_pointer_cast(dx_ptr);
					if (dimension==0){
						auto src_Tuple = thrust::make_tuple(deriv_dev_ptr, dx_dev_ptr);
						auto src_ite = thrust::make_zip_iterator(src_Tuple);
						thrust::transform(thrust::cuda::par.on(ham_stream),
							src_ite, src_ite + dx_uvecs[dimension].size(), hamValue_dev_ptr, 
							GetHamValue3_dim0());
					}
					else if (dimension==num_of_dimensions-1 && negate) {
						auto src_dst_Tuple = thrust::make_tuple(hamValue_dev_ptr, deriv_dev_ptr, dx_dev_ptr);
						auto src_dst_ite = thrust::make_zip_iterator(src_dst_Tuple);
						thrust::for_each(thrust::cuda::par.on(ham_stream),
							src_dst_ite, src_dst_ite + dx_uvecs[dimension].size(), 
							GetHamValue3_dimNot0Neg());
					}else {
						auto src_dst_Tuple = thrust::make_tuple(hamValue_dev_ptr, deriv_dev_ptr, dx_dev_ptr);
						auto src_dst_ite = thrust::make_zip_iterator(src_dst_Tuple);
						thrust::for_each(thrust::cuda::par.on(ham_stream),
							src_dst_ite, src_dst_ite + dx_uvecs[dimension].size(), 
							GetHamValue3_dimNot0());
					}
				}
				beacls::synchronizeUVec(TIdx_uvecs[0]);
				for (size_t dimension = 0; dimension < num_of_dimensions; ++dimension) {
					const FLOAT_TYPE* TIdx_ptr = beacls::UVec_<FLOAT_TYPE>(TIdx_uvecs[dimension]).ptr();
					thrust::device_ptr<const FLOAT_TYPE> TIdx_dev_ptr = thrust::device_pointer_cast(TIdx_ptr);
					auto src_dst_Tuple = thrust::make_tuple(hamValue_dev_ptr, TIdx_dev_ptr);
					auto src_dst_ite = thrust::make_zip_iterator(src_dst_Tuple);
					if (dimension==num_of_dimensions-1 && negate) {
						thrust::for_each(thrust::cuda::par.on(ham_stream),
							src_dst_ite, src_dst_ite + TIdx_uvecs[dimension].size(), 
							GetHamValue2Neg(TIderiv));
					}
					else {
						thrust::for_each(thrust::cuda::par.on(ham_stream),
							src_dst_ite, src_dst_ite + TIdx_uvecs[dimension].size(), 
							GetHamValue2(TIderiv));
					}
				}
			}
		}
	} else {
		std::cerr << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << " Invalid data size" << std::endl;
		return false;
	}
	return true;
}

struct GetMax4
{
	template<typename Tuple>
	__host__ __device__
	FLOAT_TYPE operator()(const Tuple v) const
	{
		const FLOAT_TYPE uu = thrust::get<0>(v);
		const FLOAT_TYPE ul = thrust::get<1>(v);
		const FLOAT_TYPE ll = thrust::get<2>(v);
		const FLOAT_TYPE lu = thrust::get<3>(v);
		const FLOAT_TYPE maxUUUL = max_float_type<FLOAT_TYPE>(abs_float_type<FLOAT_TYPE>(uu), abs_float_type<FLOAT_TYPE>(ul));
		const FLOAT_TYPE maxLULL = max_float_type<FLOAT_TYPE>(abs_float_type<FLOAT_TYPE>(lu), abs_float_type<FLOAT_TYPE>(ll));
		const FLOAT_TYPE maxUUULLULL = max_float_type<FLOAT_TYPE>(maxUUUL, maxLULL);
		return maxUUULLULL;
	}
};


bool partialFunc_exec_cuda(
	beacls::UVec& alphas_uvec,
	const beacls::UVec& dxLL_dim,
	const beacls::UVec& dxLU_dim,
	const beacls::UVec& dxUL_dim,
	const beacls::UVec& dxUU_dim
	) {
	const size_t length = dxUU_dim.size();
	if (alphas_uvec.type() != dxUU_dim.type()) alphas_uvec = beacls::UVec(dxUU_dim.depth(), dxUU_dim.type(), length);
	else alphas_uvec.resize(length);
	const FLOAT_TYPE* dxUU_dim_ptr = beacls::UVec_<FLOAT_TYPE>(dxUU_dim).ptr();
	const FLOAT_TYPE* dxUL_dim_ptr = beacls::UVec_<FLOAT_TYPE>(dxUL_dim).ptr();
	const FLOAT_TYPE* dxLL_dim_ptr = beacls::UVec_<FLOAT_TYPE>(dxLL_dim).ptr();
	const FLOAT_TYPE* dxLU_dim_ptr = beacls::UVec_<FLOAT_TYPE>(dxLU_dim).ptr();
	FLOAT_TYPE* alphas = beacls::UVec_<FLOAT_TYPE>(alphas_uvec).ptr();

	if ((dxUU_dim.type() == beacls::UVecType_Cuda) 
		&& (dxUL_dim.type() == beacls::UVecType_Cuda) 
		&& (dxLU_dim.type() == beacls::UVecType_Cuda) 
		&& (dxLL_dim.type() == beacls::UVecType_Cuda) ){
		cudaStream_t alpha_stream = beacls::get_stream(alphas_uvec);

		thrust::device_ptr<FLOAT_TYPE> alphas_dev_ptr = thrust::device_pointer_cast(alphas);
		beacls::synchronizeUVec(dxUU_dim);
		beacls::synchronizeUVec(dxUL_dim);
		beacls::synchronizeUVec(dxLU_dim);
		beacls::synchronizeUVec(dxLL_dim);
		thrust::device_ptr<const FLOAT_TYPE> dxUU_dim_dev_ptr = thrust::device_pointer_cast(dxUU_dim_ptr);
		thrust::device_ptr<const FLOAT_TYPE> dxUL_dim_dev_ptr = thrust::device_pointer_cast(dxUL_dim_ptr);
		thrust::device_ptr<const FLOAT_TYPE> dxLL_dim_dev_ptr = thrust::device_pointer_cast(dxLL_dim_ptr);
		thrust::device_ptr<const FLOAT_TYPE> dxLU_dim_dev_ptr = thrust::device_pointer_cast(dxLU_dim_ptr);
		
		auto float_type4Tuple = thrust::make_tuple(
			dxUU_dim_dev_ptr, dxUL_dim_dev_ptr, 
			dxLL_dim_dev_ptr, dxLU_dim_dev_ptr);
		auto float_type4Iterator = thrust::make_zip_iterator(float_type4Tuple);

		thrust::transform(thrust::cuda::par.on(alpha_stream),
			float_type4Iterator, float_type4Iterator + length, alphas_dev_ptr, GetMax4());
	} else {
		for (size_t i = 0; i < length; ++i) {
			const FLOAT_TYPE max0 = max_float_type<FLOAT_TYPE>(
				abs_float_type<FLOAT_TYPE>(dxUU_dim_ptr[i]), abs_float_type<FLOAT_TYPE>(dxUL_dim_ptr[i]));
			const FLOAT_TYPE max1 = max_float_type<FLOAT_TYPE>(
				abs_float_type<FLOAT_TYPE>(dxLL_dim_ptr[i]), abs_float_type<FLOAT_TYPE>(dxLU_dim_ptr[i]));
			alphas[i] = max_float_type<FLOAT_TYPE>(max0, max1);
		}
	}
	return true;
}
#endif /* defined(USER_DEFINED_GPU_DYNSYS_FUNC) */
#endif /* defined(WITH_GPU) */
