// CUDA runtime
#include <cuda_runtime.h>

#include <typedef.hpp>
#include <cuda_macro.hpp>
#include <thrust/device_ptr.h>
#include <thrust/tuple.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/system/cuda/execution_policy.h>
#include "PlaneCAvoid_cuda.hpp"
#include <vector>
#include <algorithm>
#include <helperOC/DynSys/DynSys/DynSys.hpp>
#include <Core/UVec.hpp>
#include <Core/CudaStream.hpp>

#if defined(WITH_GPU) 
#if defined(USER_DEFINED_GPU_DYNSYS_FUNC)
namespace PlaneCAvoid_CUDA {

struct Get_optCtrl_dim0dim1_D {
public:
	const FLOAT_TYPE vrangeA0;
	const FLOAT_TYPE vrangeA1;
	const FLOAT_TYPE wMax;
	Get_optCtrl_dim0dim1_D(
		const FLOAT_TYPE vrangeA0, 
		const FLOAT_TYPE vrangeA1, 
		const FLOAT_TYPE wMax) : 
		vrangeA0(vrangeA0), 
		vrangeA1(vrangeA1), 
		wMax(wMax) {}
	template<typename Tuple>
	__host__ __device__
	void operator()(Tuple v) const
	{
		const FLOAT_TYPE y0 = thrust::get<2>(v);
		const FLOAT_TYPE y1 = thrust::get<3>(v);
		const FLOAT_TYPE deriv0 = thrust::get<4>(v);
		const FLOAT_TYPE deriv1 = thrust::get<5>(v);
		const FLOAT_TYPE deriv2 = thrust::get<6>(v);
		const FLOAT_TYPE det0 = -deriv0;
		const FLOAT_TYPE det1 = deriv0 * y1 - deriv1 * y0 - deriv2;
		thrust::get<0>(v) = (det0 >= 0) ? vrangeA1 : vrangeA0;
		thrust::get<1>(v) = (det1 >= 0) ? wMax : -wMax;
	}
};

struct Get_optCtrl_dim1_d {
public:
	const FLOAT_TYPE wMax;
	const FLOAT_TYPE d0;
	const FLOAT_TYPE d1;
	const FLOAT_TYPE d2;
	Get_optCtrl_dim1_d(const FLOAT_TYPE wMax, const FLOAT_TYPE d0, const FLOAT_TYPE d1, const FLOAT_TYPE d2) : 
		wMax(wMax), d0(d0), d1(d1), d2(d2) {}
	__host__ __device__
	FLOAT_TYPE operator()(const FLOAT_TYPE y0, const FLOAT_TYPE y1) const
	{
		const FLOAT_TYPE det1 = d0 * y1 - d1 * y0 - d2;
		return (det1 >= 0) ? wMax : -wMax;
	}
};

bool optCtrl_execute_cuda(
	std::vector<beacls::UVec>& u_uvecs,
	const std::vector<beacls::UVec>& x_uvecs,
	const std::vector<beacls::UVec>& deriv_uvecs,
	const FLOAT_TYPE wMaxA,
	const std::vector<FLOAT_TYPE>& vRangeA,
	const helperOC::DynSys_UMode_Type uMode
)
{
	bool result = true;
	beacls::reallocateAsSrc(u_uvecs[0], deriv_uvecs[0]);
	beacls::reallocateAsSrc(u_uvecs[1], x_uvecs[0]);
	FLOAT_TYPE* uOpt0_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[0]).ptr();
	FLOAT_TYPE* uOpt1_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[1]).ptr();
	const FLOAT_TYPE* y0_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[0]).ptr();
	const FLOAT_TYPE* y1_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[1]).ptr();
	const FLOAT_TYPE* deriv0_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[0]).ptr();
	const FLOAT_TYPE* deriv1_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[1]).ptr();
	const FLOAT_TYPE* deriv2_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[2]).ptr();
	thrust::device_ptr<const FLOAT_TYPE> y0_dev_ptr = thrust::device_pointer_cast(y0_ptr);
	thrust::device_ptr<const FLOAT_TYPE> y1_dev_ptr = thrust::device_pointer_cast(y1_ptr);
	const FLOAT_TYPE vRangeA0 = vRangeA[0];
	const FLOAT_TYPE vRangeA1 = vRangeA[1];
	if ((uMode == helperOC::DynSys_UMode_Max) || (uMode == helperOC::DynSys_UMode_Min)) {
		const FLOAT_TYPE moded_wMaxA = (uMode == helperOC::DynSys_UMode_Max) ? wMaxA : -wMaxA;
		const FLOAT_TYPE moded_vRangeA0 = (uMode == helperOC::DynSys_UMode_Max) ? vRangeA0 : vRangeA1;
		const FLOAT_TYPE moded_vRangeA1 = (uMode == helperOC::DynSys_UMode_Max) ? vRangeA1 : vRangeA0;
		cudaStream_t u_stream = beacls::get_stream(u_uvecs[1]);
		thrust::device_ptr<FLOAT_TYPE> uOpt1_dev_ptr = thrust::device_pointer_cast(uOpt1_ptr);
		if(is_cuda(deriv_uvecs[0]) && is_cuda(deriv_uvecs[1]) && is_cuda(deriv_uvecs[2])){
			thrust::device_ptr<FLOAT_TYPE> uOpt0_dev_ptr = thrust::device_pointer_cast(uOpt0_ptr);
			u_uvecs[0].set_cudaStream(u_uvecs[1].get_cudaStream());
			thrust::device_ptr<const FLOAT_TYPE> deriv0_dev_ptr = thrust::device_pointer_cast(deriv0_ptr);
			thrust::device_ptr<const FLOAT_TYPE> deriv1_dev_ptr = thrust::device_pointer_cast(deriv1_ptr);
			thrust::device_ptr<const FLOAT_TYPE> deriv2_dev_ptr = thrust::device_pointer_cast(deriv2_ptr);
			auto src_dst_Tuple = thrust::make_tuple(uOpt0_dev_ptr, uOpt1_dev_ptr, 
				y0_dev_ptr, y1_dev_ptr, deriv0_dev_ptr, deriv1_dev_ptr, deriv2_dev_ptr);
			auto src_dst_Iterator = thrust::make_zip_iterator(src_dst_Tuple);
			thrust::for_each(thrust::cuda::par.on(u_stream), 
				src_dst_Iterator, src_dst_Iterator + x_uvecs[0].size(),
				Get_optCtrl_dim0dim1_D(moded_vRangeA0, moded_vRangeA1, moded_wMaxA));
		}
		else {
			const FLOAT_TYPE d0 = deriv0_ptr[0];
			const FLOAT_TYPE d1 = deriv1_ptr[0];
			const FLOAT_TYPE d2 = deriv2_ptr[0];
			const FLOAT_TYPE det0 = -d0;
			uOpt0_ptr[0] = (det0 >= 0) ? moded_vRangeA1 : moded_vRangeA0;
			thrust::transform(thrust::cuda::par.on(u_stream), 
				y0_dev_ptr, y0_dev_ptr + x_uvecs[0].size(), y1_dev_ptr, uOpt1_dev_ptr,
				Get_optCtrl_dim1_d(moded_wMaxA, d0, d1, d2));
		}
	}
	else {
		std::cerr << "Unknown uMode!: " << uMode << std::endl;
		result = false;
	}
	return result;
}

struct Get_optDstb_dim0_d {
public:
	const FLOAT_TYPE vrangeB0;
	const FLOAT_TYPE vrangeB1;
	const FLOAT_TYPE d0;
	const FLOAT_TYPE d1;
	Get_optDstb_dim0_d(const FLOAT_TYPE vrangeB0, const FLOAT_TYPE vrangeB1, const FLOAT_TYPE d0, const FLOAT_TYPE d1) : 
		vrangeB0(vrangeB0), vrangeB1(vrangeB1), d0(d0), d1(d1) {}
	__host__ __device__
	FLOAT_TYPE operator()(const FLOAT_TYPE y2) const
	{
		FLOAT_TYPE sin_y2;
		FLOAT_TYPE cos_y2;
		sincos_float_type<FLOAT_TYPE>(y2, sin_y2, cos_y2);
		
		const FLOAT_TYPE det0 = d0 * cos_y2 + d1 * sin_y2;
		return (det0 >= 0) ? vrangeB1 : vrangeB0;
	}
};

struct Get_optDstb_dim0dim1dim2dim3dim4_D {
public:
	const FLOAT_TYPE vrangeB0;
	const FLOAT_TYPE vrangeB1;
	const FLOAT_TYPE wMaxB;
	const FLOAT_TYPE dMaxA_0_dMaxB_0;
	const FLOAT_TYPE dMaxA_1_dMaxB_1;
	Get_optDstb_dim0dim1dim2dim3dim4_D(
		const FLOAT_TYPE vrangeB0, 
		const FLOAT_TYPE vrangeB1,
		const FLOAT_TYPE wMaxB, 
		const FLOAT_TYPE dMaxA_0_dMaxB_0, 
		const FLOAT_TYPE dMaxA_1_dMaxB_1) : 
		vrangeB0(vrangeB0), 
		vrangeB1(vrangeB1),
		wMaxB(wMaxB), 
		dMaxA_0_dMaxB_0(dMaxA_0_dMaxB_0), 
		dMaxA_1_dMaxB_1(dMaxA_1_dMaxB_1) {}
	template<typename Tuple>
	__host__ __device__
	void operator()(Tuple v) const
	{
		const FLOAT_TYPE y2 = thrust::get<5>(v);
		const FLOAT_TYPE deriv0 = thrust::get<6>(v);
		const FLOAT_TYPE deriv1 = thrust::get<7>(v);
		const FLOAT_TYPE deriv2 = thrust::get<8>(v);
		FLOAT_TYPE sin_y2;
		FLOAT_TYPE cos_y2;
		sincos_float_type<FLOAT_TYPE>(y2, sin_y2, cos_y2);
		const FLOAT_TYPE normDeriv = sqrt_float_type<FLOAT_TYPE>(deriv0 * deriv0 + deriv1 * deriv1);
		const FLOAT_TYPE det0 = deriv0 * cos_y2 + deriv1 * sin_y2;
		thrust::get<0>(v) = (det0 >= 0) ? vrangeB1 : vrangeB0;
		if (normDeriv == 0) {
			thrust::get<2>(v) = 0;
			thrust::get<3>(v) = 0;
		} else {
			thrust::get<2>(v) = dMaxA_0_dMaxB_0 * deriv0 / normDeriv;
			thrust::get<3>(v) = dMaxA_0_dMaxB_0 * deriv1 / normDeriv;
		}
		if (deriv2 >= 0) {
			thrust::get<1>(v) = wMaxB;
			thrust::get<4>(v) = dMaxA_1_dMaxB_1;
		} else {
			thrust::get<1>(v) = -wMaxB;
			thrust::get<4>(v) = -dMaxA_1_dMaxB_1;
		}
	}
};
bool optDstb_execute_cuda(
	std::vector<beacls::UVec>& d_uvecs,
	const std::vector<beacls::UVec>& x_uvecs,
	const std::vector<beacls::UVec>& deriv_uvecs,
	const std::vector<FLOAT_TYPE>& dMaxA,
	const std::vector<FLOAT_TYPE>& dMaxB,
	const std::vector<FLOAT_TYPE>& vRangeB,
	const FLOAT_TYPE wMaxB,
	const helperOC::DynSys_DMode_Type dMode
)
{
	bool result = true;
	beacls::reallocateAsSrc(d_uvecs[0], x_uvecs[2]);
	beacls::reallocateAsSrc(d_uvecs[1], deriv_uvecs[0]);
	beacls::reallocateAsSrc(d_uvecs[2], deriv_uvecs[0]);
	beacls::reallocateAsSrc(d_uvecs[3], deriv_uvecs[0]);
	beacls::reallocateAsSrc(d_uvecs[4], deriv_uvecs[0]);
	FLOAT_TYPE* dOpt0_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[0]).ptr();
	FLOAT_TYPE* dOpt1_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[1]).ptr();
	FLOAT_TYPE* dOpt2_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[2]).ptr();
	FLOAT_TYPE* dOpt3_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[3]).ptr();
	FLOAT_TYPE* dOpt4_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[4]).ptr();
	const FLOAT_TYPE* y2_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[2]).ptr();
	const FLOAT_TYPE* deriv0_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[0]).ptr();
	const FLOAT_TYPE* deriv1_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[1]).ptr();
	const FLOAT_TYPE* deriv2_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[2]).ptr();
	const FLOAT_TYPE dMaxA_0 = dMaxA[0];
	const FLOAT_TYPE dMaxA_1 = dMaxA[1];
	const FLOAT_TYPE dMaxB_0 = dMaxB[0];
	const FLOAT_TYPE dMaxB_1 = dMaxB[1];
	const FLOAT_TYPE vRangeB0 = vRangeB[0];
	const FLOAT_TYPE vRangeB1 = vRangeB[1];
	const FLOAT_TYPE dMaxA_0_dMaxB_0 = dMaxA_0 + dMaxB_0;
	const FLOAT_TYPE dMaxA_1_dMaxB_1 = dMaxA_1 + dMaxB_1;
	if ((dMode == helperOC::DynSys_DMode_Max) || (dMode == helperOC::DynSys_DMode_Min)) {
		const FLOAT_TYPE moded_wMaxB = (dMode == helperOC::DynSys_DMode_Max) ? wMaxB : -wMaxB;
		const FLOAT_TYPE moded_vRangeB0 = (dMode == helperOC::DynSys_DMode_Max) ? vRangeB0 : vRangeB1;
		const FLOAT_TYPE moded_vRangeB1 = (dMode == helperOC::DynSys_DMode_Max) ? vRangeB1 : vRangeB0;
		const FLOAT_TYPE moded_dMaxA_0_dMaxB_0 = (dMode == helperOC::DynSys_DMode_Max) ? dMaxA_0_dMaxB_0 : -dMaxA_0_dMaxB_0;
		const FLOAT_TYPE moded_dMaxA_1_dMaxB_1 = (dMode == helperOC::DynSys_DMode_Max) ? dMaxA_1_dMaxB_1 : -dMaxA_1_dMaxB_1;
		thrust::device_ptr<FLOAT_TYPE> dOpt0_dev_ptr = thrust::device_pointer_cast(dOpt0_ptr);
		cudaStream_t d_stream = beacls::get_stream(d_uvecs[0]);
		thrust::device_ptr<const FLOAT_TYPE> y2_dev_ptr = thrust::device_pointer_cast(y2_ptr);
		if (beacls::is_cuda(deriv_uvecs[0]) && beacls::is_cuda(deriv_uvecs[1]) && beacls::is_cuda(deriv_uvecs[2])){
			thrust::device_ptr<FLOAT_TYPE> dOpt1_dev_ptr = thrust::device_pointer_cast(dOpt1_ptr);
			thrust::device_ptr<FLOAT_TYPE> dOpt2_dev_ptr = thrust::device_pointer_cast(dOpt2_ptr);
			thrust::device_ptr<FLOAT_TYPE> dOpt3_dev_ptr = thrust::device_pointer_cast(dOpt3_ptr);
			thrust::device_ptr<FLOAT_TYPE> dOpt4_dev_ptr = thrust::device_pointer_cast(dOpt4_ptr);
			thrust::device_ptr<const FLOAT_TYPE> deriv0_dev_ptr = thrust::device_pointer_cast(deriv0_ptr);
			thrust::device_ptr<const FLOAT_TYPE> deriv1_dev_ptr = thrust::device_pointer_cast(deriv1_ptr);
			thrust::device_ptr<const FLOAT_TYPE> deriv2_dev_ptr = thrust::device_pointer_cast(deriv2_ptr);
			d_uvecs[1].set_cudaStream(d_uvecs[0].get_cudaStream());
			d_uvecs[2].set_cudaStream(d_uvecs[0].get_cudaStream());
			d_uvecs[3].set_cudaStream(d_uvecs[0].get_cudaStream());
			d_uvecs[4].set_cudaStream(d_uvecs[0].get_cudaStream());
			auto dst_src_Tuple = thrust::make_tuple(
				dOpt0_dev_ptr, dOpt1_dev_ptr, dOpt2_dev_ptr, dOpt3_dev_ptr, dOpt4_dev_ptr,
				y2_dev_ptr, deriv0_dev_ptr, deriv1_dev_ptr, deriv2_dev_ptr);
			auto dst_src_Iterator = thrust::make_zip_iterator(dst_src_Tuple);
			thrust::for_each(thrust::cuda::par.on(d_stream),
				dst_src_Iterator, dst_src_Iterator + deriv_uvecs[0].size(), 
				Get_optDstb_dim0dim1dim2dim3dim4_D(moded_vRangeB0, moded_vRangeB1, 
					moded_wMaxB, moded_dMaxA_0_dMaxB_0, moded_dMaxA_1_dMaxB_1));
		}
		else {
			const FLOAT_TYPE d0 = deriv0_ptr[0];
			const FLOAT_TYPE d1 = deriv1_ptr[0];
			const FLOAT_TYPE d2 = deriv2_ptr[0];
			thrust::transform(thrust::cuda::par.on(d_stream),
				y2_dev_ptr, y2_dev_ptr + x_uvecs[2].size(), dOpt0_dev_ptr, 
				Get_optDstb_dim0_d(moded_vRangeB0, moded_vRangeB1, d0, d1));
			const FLOAT_TYPE det1 = d2;
			const FLOAT_TYPE det4 = d2;
			dOpt1_ptr[0] = (det1 >= 0) ? moded_wMaxB : -moded_wMaxB;
			dOpt4_ptr[0] = (det4 >= 0) ? moded_dMaxA_1_dMaxB_1 : -moded_dMaxA_1_dMaxB_1;
			const FLOAT_TYPE denom = sqrt_float_type<FLOAT_TYPE>(d0 * d0 + d1 * d1);
			dOpt2_ptr[0] = (denom == 0) ? 0 : moded_dMaxA_0_dMaxB_0 * d0 / denom;
			dOpt3_ptr[0] = (denom == 0) ? 0 : moded_dMaxA_0_dMaxB_0 * d1 / denom;
		}
	}
	else {
		std::cerr << "Unknown dMode!: " << dMode << std::endl;
		result = false;
	}
	return result;
}

struct Get_dynamics_dimAll_U0_U1_D0_D1_D2_D3_D4
{
	Get_dynamics_dimAll_U0_U1_D0_D1_D2_D3_D4(
	)
	 {}
	template<typename Tuple>
	__host__ __device__
	void operator()(Tuple v) const
	{
		const FLOAT_TYPE x0 = thrust::get<3>(v);
		const FLOAT_TYPE x1 = thrust::get<4>(v);
		const FLOAT_TYPE x2 = thrust::get<5>(v);
		const FLOAT_TYPE u0 = thrust::get<6>(v);
		const FLOAT_TYPE u1 = thrust::get<7>(v);
		const FLOAT_TYPE d0 = thrust::get<8>(v);
		const FLOAT_TYPE d1 = thrust::get<9>(v);
		const FLOAT_TYPE d2 = thrust::get<10>(v);
		const FLOAT_TYPE d3 = thrust::get<11>(v);
		const FLOAT_TYPE d4 = thrust::get<12>(v);
		FLOAT_TYPE sin_x;
		FLOAT_TYPE cos_x;
		sincos_float_type<FLOAT_TYPE>(x2, sin_x, cos_x);
		thrust::get<0>(v) = -u0 + d0 * cos_x + u1 * x1 + d2;
		thrust::get<1>(v) = d0 * sin_x - u1 * x0 + d3;
		thrust::get<2>(v) = d1 - u1 + d4;
	}
};
struct Get_dynamics_dim01_U0_U1_D0_D2_D3
{
	Get_dynamics_dim01_U0_U1_D0_D2_D3(
	)
	 {}
	template<typename Tuple>
	__host__ __device__
	void operator()(Tuple v) const
	{
		const FLOAT_TYPE x0 = thrust::get<2>(v);
		const FLOAT_TYPE x1 = thrust::get<3>(v);
		const FLOAT_TYPE x2 = thrust::get<4>(v);
		const FLOAT_TYPE u0 = thrust::get<5>(v);
		const FLOAT_TYPE u1 = thrust::get<6>(v);
		const FLOAT_TYPE d0 = thrust::get<7>(v);
		const FLOAT_TYPE d2 = thrust::get<8>(v);
		const FLOAT_TYPE d3 = thrust::get<9>(v);
		FLOAT_TYPE sin_x;
		FLOAT_TYPE cos_x;
		sincos_float_type<FLOAT_TYPE>(x2, sin_x, cos_x);
		thrust::get<0>(v) = -u0 + d0 * cos_x + u1 * x1 + d2;
		thrust::get<1>(v) = d0 * sin_x - u1 * x0 + d3;
	}
};
struct Get_dynamics_dimAll_u0_U1_D0_d1_d2_d3_d4
{
	const FLOAT_TYPE u0;
	const FLOAT_TYPE d1;
	const FLOAT_TYPE d2;
	const FLOAT_TYPE d3;
	const FLOAT_TYPE d4;
	Get_dynamics_dimAll_u0_U1_D0_d1_d2_d3_d4(
		const FLOAT_TYPE u0,
		const FLOAT_TYPE d1,
		const FLOAT_TYPE d2,
		const FLOAT_TYPE d3,
		const FLOAT_TYPE d4
	) : u0(u0), d1(d1), d2(d2), d3(d3), d4(d4) {}
	template<typename Tuple>
	__host__ __device__
	void operator()(Tuple v) const
	{
		const FLOAT_TYPE x0 = thrust::get<3>(v);
		const FLOAT_TYPE x1 = thrust::get<4>(v);
		const FLOAT_TYPE x2 = thrust::get<5>(v);
		const FLOAT_TYPE u1 = thrust::get<6>(v);
		const FLOAT_TYPE d0 = thrust::get<7>(v);
		FLOAT_TYPE sin_x;
		FLOAT_TYPE cos_x;
		sincos_float_type<FLOAT_TYPE>(x2, sin_x, cos_x);
		thrust::get<0>(v) = -u0 + d0 * cos_x + u1 * x1 + d2;
		thrust::get<1>(v) = d0 * sin_x - u1 * x0 + d3;
		thrust::get<2>(v) = d1 - u1 + d4;
	}
};

struct Get_dynamics_dim0_U0_U1_D0_D2 {
public:
	template<typename Tuple>
	__host__ __device__
	FLOAT_TYPE operator()(const Tuple &v) const
	{
		const FLOAT_TYPE x1 = thrust::get<0>(v);
		const FLOAT_TYPE x2 = thrust::get<1>(v);
		const FLOAT_TYPE u0 = thrust::get<2>(v);
		const FLOAT_TYPE u1 = thrust::get<3>(v);
		const FLOAT_TYPE d0 = thrust::get<4>(v);
		const FLOAT_TYPE d2 = thrust::get<5>(v);
		return -u0 + d0 * cos_float_type<FLOAT_TYPE>(x2) + u1 * x1 + d2;
	}
};
struct Get_dynamics_dim0_u0_U1_D0_d2 {
public:
	const FLOAT_TYPE u0;
	const FLOAT_TYPE d2;
	Get_dynamics_dim0_u0_U1_D0_d2(const FLOAT_TYPE u0, const FLOAT_TYPE d2) : u0(u0), d2(d2) {}
	template<typename Tuple>
	__host__ __device__
	FLOAT_TYPE operator()(const Tuple &v) const
	{
		const FLOAT_TYPE x1 = thrust::get<0>(v);
		const FLOAT_TYPE x2 = thrust::get<1>(v);
		const FLOAT_TYPE u1 = thrust::get<2>(v);
		const FLOAT_TYPE d0 = thrust::get<3>(v);
		return -u0 + d0 * cos_float_type<FLOAT_TYPE>(x2) + u1 * x1 + d2;
	}
};
struct Get_dynamics_dim1_U1_D0_D3 {
public:
	template<typename Tuple>
	__host__ __device__
	FLOAT_TYPE operator()(const Tuple &v) const
	{
		const FLOAT_TYPE x0 = thrust::get<0>(v);
		const FLOAT_TYPE x2 = thrust::get<1>(v);
		const FLOAT_TYPE u1 = thrust::get<2>(v);
		const FLOAT_TYPE d0 = thrust::get<3>(v);
		const FLOAT_TYPE d3 = thrust::get<4>(v);
		return d0 * sin_float_type<FLOAT_TYPE>(x2) - u1 * x0 + d3;
	}
};
struct Get_dynamics_dim1_U1_D0_d3 {
public:
	const FLOAT_TYPE d3;
	Get_dynamics_dim1_U1_D0_d3(const FLOAT_TYPE d3) : d3(d3) {}
	template<typename Tuple>
	__host__ __device__
	FLOAT_TYPE operator()(const Tuple &v) const
	{
		const FLOAT_TYPE x0 = thrust::get<0>(v);
		const FLOAT_TYPE x2 = thrust::get<1>(v);
		const FLOAT_TYPE u1 = thrust::get<2>(v);
		const FLOAT_TYPE d0 = thrust::get<3>(v);
		return d0 * sin_float_type<FLOAT_TYPE>(x2) - u1 * x0 + d3;
	}
};
struct Get_dynamics_dim2_U1_D1_D4 {
public:
	template<typename Tuple>
	__host__ __device__
	FLOAT_TYPE operator()(const Tuple &v) const
	{
		const FLOAT_TYPE u1 = thrust::get<0>(v);
		const FLOAT_TYPE d1 = thrust::get<1>(v);
		const FLOAT_TYPE d4 = thrust::get<2>(v);
		return d1 - u1 + d4;
	}
};
struct Get_dynamics_dim2_U1_D1_d4 {
public:
	const FLOAT_TYPE d4;
	Get_dynamics_dim2_U1_D1_d4(const FLOAT_TYPE d4) : d4(d4) {}
	__host__ __device__
	FLOAT_TYPE operator()(const FLOAT_TYPE u1, const FLOAT_TYPE d1) const
	{
		return d1 - u1 + d4;
	}
};
struct Get_dynamics_dim2_U1_d1_D4 {
public:
	const FLOAT_TYPE d1;
	Get_dynamics_dim2_U1_d1_D4(const FLOAT_TYPE d1) : d1(d1) {}
	__host__ __device__
	FLOAT_TYPE operator()(const FLOAT_TYPE u1, const FLOAT_TYPE d4) const
	{
		return d1 - u1 + d4;
	}
};
struct Get_dynamics_dim2_U1_d1_d4 {
public:
	const FLOAT_TYPE d1;
	const FLOAT_TYPE d4;
	Get_dynamics_dim2_U1_d1_d4(const FLOAT_TYPE d1, const FLOAT_TYPE d4) : d1(d1), d4(d4) {}
	__host__ __device__
	FLOAT_TYPE operator()(const FLOAT_TYPE u1) const
	{
		return d1 - u1 + d4;
	}
};
bool dynamics_cell_helper_execute_cuda_dimAll(
	std::vector<beacls::UVec>& dx_uvecs,
	const std::vector<beacls::UVec>& x_uvecs,
	const std::vector<beacls::UVec>& u_uvecs,
	const std::vector<beacls::UVec>& d_uvecs
) {
	bool result = true;
	beacls::reallocateAsSrc(dx_uvecs[0], x_uvecs[2]);
	beacls::reallocateAsSrc(dx_uvecs[1], x_uvecs[2]);
	beacls::reallocateAsSrc(dx_uvecs[2], u_uvecs[1]);
	FLOAT_TYPE* dx_dim0_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[0]).ptr();
	FLOAT_TYPE* dx_dim1_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[1]).ptr();
	FLOAT_TYPE* dx_dim2_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[2]).ptr();
	const FLOAT_TYPE* uOpt0_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[0]).ptr();
	const FLOAT_TYPE* uOpt1_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[1]).ptr();
	const FLOAT_TYPE* dOpt0_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[0]).ptr();
	const FLOAT_TYPE* dOpt1_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[1]).ptr();
	const FLOAT_TYPE* dOpt2_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[2]).ptr();
	const FLOAT_TYPE* dOpt3_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[3]).ptr();
	const FLOAT_TYPE* dOpt4_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[4]).ptr();
	const FLOAT_TYPE* x0_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[0]).ptr();
	const FLOAT_TYPE* x1_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[1]).ptr();
	const FLOAT_TYPE* x2_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[2]).ptr();

	thrust::device_ptr<FLOAT_TYPE> dx_dim0_dev_ptr = thrust::device_pointer_cast(dx_dim0_ptr);
	thrust::device_ptr<FLOAT_TYPE> dx_dim1_dev_ptr = thrust::device_pointer_cast(dx_dim1_ptr);
	thrust::device_ptr<FLOAT_TYPE> dx_dim2_dev_ptr = thrust::device_pointer_cast(dx_dim2_ptr);
	cudaStream_t dx_stream = beacls::get_stream(dx_uvecs[0]);
	dx_uvecs[1].set_cudaStream(dx_uvecs[0].get_cudaStream());
	dx_uvecs[2].set_cudaStream(dx_uvecs[0].get_cudaStream());
	thrust::device_ptr<const FLOAT_TYPE> us_1_dev_ptr = thrust::device_pointer_cast(uOpt1_ptr);
	thrust::device_ptr<const FLOAT_TYPE> ds_0_dev_ptr = thrust::device_pointer_cast(dOpt0_ptr);
	beacls::synchronizeUVec(u_uvecs[1]);
	beacls::synchronizeUVec(d_uvecs[0]);
	thrust::device_ptr<const FLOAT_TYPE> x0_dev_ptr = thrust::device_pointer_cast(x0_ptr);
	thrust::device_ptr<const FLOAT_TYPE> x1_dev_ptr = thrust::device_pointer_cast(x1_ptr);
	thrust::device_ptr<const FLOAT_TYPE> x2_dev_ptr = thrust::device_pointer_cast(x2_ptr);

	if (beacls::is_cuda(u_uvecs[1]) && beacls::is_cuda(d_uvecs[0])) {
		if (beacls::is_cuda(u_uvecs[0]) && beacls::is_cuda(d_uvecs[2]) && beacls::is_cuda(d_uvecs[3]) && beacls::is_cuda(d_uvecs[1]) && beacls::is_cuda(d_uvecs[4])){
			thrust::device_ptr<const FLOAT_TYPE> us_0_dev_ptr = thrust::device_pointer_cast(uOpt0_ptr);
			thrust::device_ptr<const FLOAT_TYPE> ds_1_dev_ptr = thrust::device_pointer_cast(dOpt1_ptr);
			thrust::device_ptr<const FLOAT_TYPE> ds_2_dev_ptr = thrust::device_pointer_cast(dOpt2_ptr);
			thrust::device_ptr<const FLOAT_TYPE> ds_3_dev_ptr = thrust::device_pointer_cast(dOpt3_ptr);
			thrust::device_ptr<const FLOAT_TYPE> ds_4_dev_ptr = thrust::device_pointer_cast(dOpt4_ptr);
			auto dst_src_Tuple = thrust::make_tuple(
				dx_dim0_dev_ptr, dx_dim1_dev_ptr, 
				x0_dev_ptr, x1_dev_ptr, x2_dev_ptr, 
				us_0_dev_ptr, us_1_dev_ptr, 
				ds_0_dev_ptr, ds_2_dev_ptr, ds_3_dev_ptr);
			auto dst_src_Iterator = thrust::make_zip_iterator(dst_src_Tuple);
			thrust::for_each(thrust::cuda::par.on(dx_stream),
				dst_src_Iterator, dst_src_Iterator + x_uvecs[0].size(), 
				Get_dynamics_dim01_U0_U1_D0_D2_D3());
			//!< limit of template variables of thrust::tuple is 10, therefore devide 2 thrust calls.
			auto src2_Tuple = thrust::make_tuple(us_1_dev_ptr, ds_1_dev_ptr, ds_4_dev_ptr);
			auto src2_Iterator = thrust::make_zip_iterator(src2_Tuple);
			thrust::transform(thrust::cuda::par.on(dx_stream),
				src2_Iterator, src2_Iterator + u_uvecs[1].size(), dx_dim2_dev_ptr, 
				Get_dynamics_dim2_U1_D1_D4());
		}
		else {
			const FLOAT_TYPE u0 = uOpt0_ptr[0];
			const FLOAT_TYPE d1 = dOpt1_ptr[0];
			const FLOAT_TYPE d2 = dOpt2_ptr[0];
			const FLOAT_TYPE d3 = dOpt3_ptr[0];
			const FLOAT_TYPE d4 = dOpt4_ptr[0];
			auto dst_src_Tuple = thrust::make_tuple(
				dx_dim0_dev_ptr, dx_dim1_dev_ptr, dx_dim2_dev_ptr, 
				x0_dev_ptr, x1_dev_ptr, x2_dev_ptr, 
				us_1_dev_ptr, ds_0_dev_ptr);
			auto dst_src_Iterator = thrust::make_zip_iterator(dst_src_Tuple);
			thrust::for_each(thrust::cuda::par.on(dx_stream),
				dst_src_Iterator, dst_src_Iterator + x_uvecs[0].size(), 
				Get_dynamics_dimAll_u0_U1_D0_d1_d2_d3_d4(u0, d1, d2, d3, d4));
		}
	}
	else {
		std::cerr << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << " Invalid data size" << std::endl;
		return false;
	}
	return result;
}

bool dynamics_cell_helper_execute_cuda(
	beacls::UVec& dx_uvec,
	const std::vector<beacls::UVec>& x_uvecs,
	const std::vector<beacls::UVec>& u_uvecs,
	const std::vector<beacls::UVec>& d_uvecs,
	const size_t dim
) {
	bool result = true;
	switch (dim) {
	case 0:
		if (beacls::is_cuda(u_uvecs[1]) && beacls::is_cuda(d_uvecs[0])) {
			beacls::reallocateAsSrc(dx_uvec, x_uvecs[2]);
			FLOAT_TYPE* dx_dim_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvec).ptr();
			cudaStream_t dx_stream = beacls::get_stream(dx_uvec);
			const FLOAT_TYPE* x1_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[1]).ptr();
			const FLOAT_TYPE* x2_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[2]).ptr();
			const FLOAT_TYPE* uOpt0_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[0]).ptr();
			const FLOAT_TYPE* uOpt1_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[1]).ptr();
			const FLOAT_TYPE* dOpt0_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[0]).ptr();
			const FLOAT_TYPE* dOpt2_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[2]).ptr();

			thrust::device_ptr<FLOAT_TYPE> dx_dim_dev_ptr = thrust::device_pointer_cast(dx_dim_ptr);
			thrust::device_ptr<const FLOAT_TYPE> x1_dev_ptr = thrust::device_pointer_cast(x1_ptr);
			thrust::device_ptr<const FLOAT_TYPE> x2_dev_ptr = thrust::device_pointer_cast(x2_ptr);
			thrust::device_ptr<const FLOAT_TYPE> us_1_dev_ptr = thrust::device_pointer_cast(uOpt1_ptr);
			thrust::device_ptr<const FLOAT_TYPE> ds_0_dev_ptr = thrust::device_pointer_cast(dOpt0_ptr);
			beacls::synchronizeUVec(u_uvecs[1]);
			beacls::synchronizeUVec(d_uvecs[0]);

			if (beacls::is_cuda(u_uvecs[0]) && beacls::is_cuda(d_uvecs[2])) {
				thrust::device_ptr<const FLOAT_TYPE> us_0_dev_ptr = thrust::device_pointer_cast(uOpt0_ptr);
				thrust::device_ptr<const FLOAT_TYPE> ds_2_dev_ptr = thrust::device_pointer_cast(dOpt2_ptr);

				auto src0_Tuple = thrust::make_tuple(
					x1_dev_ptr, x2_dev_ptr, 
					us_0_dev_ptr, us_1_dev_ptr, ds_0_dev_ptr, ds_2_dev_ptr);
				auto src0_Iterator = thrust::make_zip_iterator(src0_Tuple);
				thrust::transform(thrust::cuda::par.on(dx_stream),
					src0_Iterator, src0_Iterator + x_uvecs[1].size(), dx_dim_dev_ptr, 
					Get_dynamics_dim0_U0_U1_D0_D2());
			}
			else {
				const FLOAT_TYPE u0 = uOpt0_ptr[0];
				const FLOAT_TYPE d2 = dOpt2_ptr[0];
				auto src0_Tuple = thrust::make_tuple(
					x1_dev_ptr, x2_dev_ptr, 
					us_1_dev_ptr, ds_0_dev_ptr);
				auto src0_Iterator = thrust::make_zip_iterator(src0_Tuple);
				thrust::transform(thrust::cuda::par.on(dx_stream),
					src0_Iterator, src0_Iterator + x_uvecs[1].size(), dx_dim_dev_ptr,
					Get_dynamics_dim0_u0_U1_D0_d2(u0, d2));
			}
		}
		else {
			std::cerr << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << " Invalid data size" << std::endl;
			result = false;
		}
		break;
	case 1:
		if (beacls::is_cuda(u_uvecs[1]) && beacls::is_cuda(d_uvecs[0])) {
			beacls::reallocateAsSrc(dx_uvec, x_uvecs[2]);
			FLOAT_TYPE* dx_dim_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvec).ptr();
			cudaStream_t dx_stream = beacls::get_stream(dx_uvec);
			const FLOAT_TYPE* x0_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[0]).ptr();
			const FLOAT_TYPE* x2_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[2]).ptr();
			const FLOAT_TYPE* uOpt1_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[1]).ptr();
			const FLOAT_TYPE* dOpt0_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[0]).ptr();
			const FLOAT_TYPE* dOpt3_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[3]).ptr();

			thrust::device_ptr<FLOAT_TYPE> dx_dim_dev_ptr = thrust::device_pointer_cast(dx_dim_ptr);
			thrust::device_ptr<const FLOAT_TYPE> x0_dev_ptr = thrust::device_pointer_cast(x0_ptr);
			thrust::device_ptr<const FLOAT_TYPE> x2_dev_ptr = thrust::device_pointer_cast(x2_ptr);
			thrust::device_ptr<const FLOAT_TYPE> us_1_dev_ptr = thrust::device_pointer_cast(uOpt1_ptr);
			thrust::device_ptr<const FLOAT_TYPE> ds_0_dev_ptr = thrust::device_pointer_cast(dOpt0_ptr);
			beacls::synchronizeUVec(u_uvecs[1]);
			beacls::synchronizeUVec(d_uvecs[0]);

			if (beacls::is_cuda(d_uvecs[3])) {
				thrust::device_ptr<const FLOAT_TYPE> ds_3_dev_ptr = thrust::device_pointer_cast(dOpt3_ptr);
				auto src1_Tuple = thrust::make_tuple(x0_dev_ptr, x2_dev_ptr, us_1_dev_ptr, ds_0_dev_ptr, ds_3_dev_ptr);
				auto src1_Iterator = thrust::make_zip_iterator(src1_Tuple);
				thrust::transform(thrust::cuda::par.on(dx_stream),
					src1_Iterator, src1_Iterator + x_uvecs[0].size(), dx_dim_dev_ptr, 
					Get_dynamics_dim1_U1_D0_D3());
			}
			else {
				const FLOAT_TYPE d3 = dOpt3_ptr[0];
				auto src1_Tuple = thrust::make_tuple(x0_dev_ptr, x2_dev_ptr, us_1_dev_ptr, ds_0_dev_ptr);
				auto src1_Iterator = thrust::make_zip_iterator(src1_Tuple);
				thrust::transform(thrust::cuda::par.on(dx_stream),
					src1_Iterator, src1_Iterator + x_uvecs[0].size(), dx_dim_dev_ptr, 
					Get_dynamics_dim1_U1_D0_d3(d3));
			}
		}
		else {
			std::cerr << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << " Invalid data size" << std::endl;
			result = false;
		}
		break;
	case 2:
		if (beacls::is_cuda(u_uvecs[1])) {
			beacls::reallocateAsSrc(dx_uvec, u_uvecs[1]);
			FLOAT_TYPE* dx_dim_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvec).ptr();
			cudaStream_t dx_stream = beacls::get_stream(dx_uvec);
			const FLOAT_TYPE* uOpt1_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[1]).ptr();
			const FLOAT_TYPE* dOpt1_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[1]).ptr();
			const FLOAT_TYPE* dOpt4_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[4]).ptr();
			thrust::device_ptr<FLOAT_TYPE> dx_dim_dev_ptr = thrust::device_pointer_cast(dx_dim_ptr);
			thrust::device_ptr<const FLOAT_TYPE> us_1_dev_ptr = thrust::device_pointer_cast(uOpt1_ptr);
			beacls::synchronizeUVec(u_uvecs[1]);
			if (beacls::is_cuda(d_uvecs[1]) && beacls::is_cuda(d_uvecs[4])) {
				thrust::device_ptr<const FLOAT_TYPE> ds_1_dev_ptr = thrust::device_pointer_cast(dOpt1_ptr);
				thrust::device_ptr<const FLOAT_TYPE> ds_4_dev_ptr = thrust::device_pointer_cast(dOpt4_ptr);
				beacls::synchronizeUVec(d_uvecs[1]);
				auto src2_Tuple = thrust::make_tuple(us_1_dev_ptr, ds_1_dev_ptr, ds_4_dev_ptr);
				auto src2_Iterator = thrust::make_zip_iterator(src2_Tuple);
				thrust::transform(thrust::cuda::par.on(dx_stream),
					src2_Iterator, src2_Iterator + u_uvecs[1].size(), dx_dim_dev_ptr, 
					Get_dynamics_dim2_U1_D1_D4());
			}
			else {
				const FLOAT_TYPE d1 = dOpt1_ptr[0];
				const FLOAT_TYPE d4 = dOpt4_ptr[0];
				thrust::transform(thrust::cuda::par.on(dx_stream),
					us_1_dev_ptr, us_1_dev_ptr + u_uvecs[1].size(), dx_dim_dev_ptr, 
					Get_dynamics_dim2_U1_d1_d4(d1, d4));
			}
		}
		else {
			std::cerr << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << " Invalid data size" << std::endl;
			result = false;
		}
		break;
	default:
		std::cerr << "Only dimension 1-4 are defined for dynamics of PlaneCAvoid!" << std::endl;
		result = false;
		break;
	}
	return result;
}
};
#endif /* defined(USER_DEFINED_GPU_DYNSYS_FUNC) */
#endif /* defined(WITH_GPU) */
