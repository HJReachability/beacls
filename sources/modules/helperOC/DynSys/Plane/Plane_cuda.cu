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
#include "Plane_cuda.hpp"
#include <vector>
#include <algorithm>
#include <helperOC/DynSys/DynSys/DynSys.hpp>
#include <Core/UVec.hpp>
#include <Core/CudaStream.hpp>

#if defined(WITH_GPU) 
#if defined(USER_DEFINED_GPU_DYNSYS_FUNC)
namespace Plane_CUDA {
template<typename T>
struct Get_opt {
public:
	T a_;
	Get_opt(const T a) : a_(a) {}
	__host__ __device__
	T operator()(const T& v) const
	{
		return (v >= 0) ? a_ : -a_;
	}
};

struct Get_optCtrl_D {
public:
	const FLOAT_TYPE vrange_Min;
	const FLOAT_TYPE vrange_Max;
	const FLOAT_TYPE wMax;
	Get_optCtrl_D(
		const FLOAT_TYPE vrange_Min,
		const FLOAT_TYPE vrange_Max,
		const FLOAT_TYPE wMax) : 
		vrange_Min(vrange_Min),
		vrange_Max(vrange_Max), 
		wMax(wMax) {}
	template<typename Tuple>
	__host__ __device__
	void operator()(Tuple v) const
	{
		const FLOAT_TYPE y2 = thrust::get<2>(v);
		const FLOAT_TYPE deriv0 = thrust::get<3>(v);
		const FLOAT_TYPE deriv1 = thrust::get<4>(v);
		const FLOAT_TYPE deriv2 = thrust::get<5>(v);
		FLOAT_TYPE sin_y2;
		FLOAT_TYPE cos_y2;
		sincos_float_type<FLOAT_TYPE>(y2, sin_y2, cos_y2);
		const FLOAT_TYPE det1 = deriv0 * cos_y2  + deriv1 * sin_y2;
		thrust::get<0>(v) = (det1 >= 0) ? vrange_Max : vrange_Min;
		thrust::get<1>(v) = (deriv2 >= 0) ? wMax : -wMax;
	}
};

struct Get_optCtrl_dim0_d {
public:
	const FLOAT_TYPE vrange_Min;
	const FLOAT_TYPE vrange_Max;
	const FLOAT_TYPE d0;
	const FLOAT_TYPE d1;
	Get_optCtrl_dim0_d(
		const FLOAT_TYPE vrange_Min, 
		const FLOAT_TYPE vrange_Max, 
		const FLOAT_TYPE d0,
		const FLOAT_TYPE d1) : 
			vrange_Min(vrange_Min),
			vrange_Max(vrange_Max), 
			d0(d0), 
			d1(d1) {}
	__host__ __device__
	FLOAT_TYPE operator()(const FLOAT_TYPE y2) const
	{
		FLOAT_TYPE sin_y2;
		FLOAT_TYPE cos_y2;
		sincos_float_type<FLOAT_TYPE>(y2, sin_y2, cos_y2);
		const FLOAT_TYPE det1 = d0 * cos_y2  + d1 * sin_y2;
		return (det1 >= 0) ? vrange_Max : vrange_Min;
	}
};

bool optCtrl_execute_cuda(
	std::vector<beacls::UVec>& u_uvecs,
	const std::vector<beacls::UVec>& x_uvecs,
	const std::vector<beacls::UVec>& deriv_uvecs,
	const FLOAT_TYPE wMax,
	const FLOAT_TYPE vrange_min,
	const FLOAT_TYPE vrange_max,
	const helperOC::DynSys_UMode_Type uMode
)
{
	bool result = true;
	if ((uMode == helperOC::DynSys_UMode_Max) || (uMode == helperOC::DynSys_UMode_Min)) {
		const FLOAT_TYPE moded_vrange_max = (uMode == helperOC::DynSys_UMode_Max) ? vrange_max : vrange_min;
		const FLOAT_TYPE moded_vrange_min = (uMode == helperOC::DynSys_UMode_Max) ? vrange_min : vrange_max;
		const FLOAT_TYPE moded_wMax = (uMode == helperOC::DynSys_UMode_Max) ? wMax : -wMax;
		beacls::reallocateAsSrc(u_uvecs[0], x_uvecs[2]);
		beacls::reallocateAsSrc(u_uvecs[1], deriv_uvecs[2]);
		FLOAT_TYPE* uOpt0_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[0]).ptr();
		FLOAT_TYPE* uOpt1_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[1]).ptr();
		const FLOAT_TYPE* y2_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[2]).ptr();
		const FLOAT_TYPE* deriv0_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[0]).ptr();
		const FLOAT_TYPE* deriv1_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[1]).ptr();
		const FLOAT_TYPE* deriv2_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[2]).ptr();
		thrust::device_ptr<const FLOAT_TYPE> y2_dev_ptr = thrust::device_pointer_cast(y2_ptr);
		if (is_cuda(deriv_uvecs[2])){
			thrust::device_ptr<const FLOAT_TYPE> deriv0_dev_ptr = thrust::device_pointer_cast(deriv0_ptr);
			thrust::device_ptr<const FLOAT_TYPE> deriv1_dev_ptr = thrust::device_pointer_cast(deriv1_ptr);
			thrust::device_ptr<const FLOAT_TYPE> deriv2_dev_ptr = thrust::device_pointer_cast(deriv2_ptr);
			thrust::device_ptr<FLOAT_TYPE> uOpt0_dev_ptr = thrust::device_pointer_cast(uOpt0_ptr);
			thrust::device_ptr<FLOAT_TYPE> uOpt1_dev_ptr = thrust::device_pointer_cast(uOpt1_ptr);
			cudaStream_t u_stream = beacls::get_stream(u_uvecs[0]);
			u_uvecs[1].set_cudaStream(u_uvecs[0].get_cudaStream());
			auto dst_src_Tuple = thrust::make_tuple(uOpt0_dev_ptr, uOpt1_dev_ptr, 
				y2_dev_ptr, deriv0_dev_ptr, deriv1_dev_ptr, deriv2_dev_ptr);
			auto dst_src_Iterator = thrust::make_zip_iterator(dst_src_Tuple);
			thrust::for_each(thrust::cuda::par.on(u_stream), 
				dst_src_Iterator, dst_src_Iterator + x_uvecs[2].size(), 
				Get_optCtrl_D(moded_vrange_min, moded_vrange_max, moded_wMax));
		} else {
			thrust::device_ptr<FLOAT_TYPE> uOpt0_dev_ptr = thrust::device_pointer_cast(uOpt0_ptr);
			cudaStream_t u_stream = beacls::get_stream(u_uvecs[0]);
			const FLOAT_TYPE d0 = deriv0_ptr[0];
			const FLOAT_TYPE d1 = deriv1_ptr[0];
			const FLOAT_TYPE d2 = deriv2_ptr[0];
			thrust::transform(thrust::cuda::par.on(u_stream),
				y2_dev_ptr, y2_dev_ptr + x_uvecs[2].size(), uOpt0_dev_ptr, 
				Get_optCtrl_dim0_d(moded_vrange_min, moded_vrange_max, d0, d1));
			uOpt1_ptr[0] = (d2 >= 0) ? moded_wMax : -moded_wMax;
		}
	}
	else {
		std::cerr << "Unknown uMode!: " << uMode << std::endl;
		result = false;
	}
	return result;
}

struct Get_optDstb_dim0dim1dim2 {
public:
	const FLOAT_TYPE dMax_0;
	const FLOAT_TYPE dMax_1;
	Get_optDstb_dim0dim1dim2(const FLOAT_TYPE dMax_0, const FLOAT_TYPE dMax_1) : dMax_0(dMax_0), dMax_1(dMax_1) {}
	template<typename Tuple>
	__host__ __device__
	void operator()(Tuple v) const
	{
		const FLOAT_TYPE d0 = thrust::get<3>(v);
		const FLOAT_TYPE d1 = thrust::get<4>(v);
		const FLOAT_TYPE d2 = thrust::get<5>(v);
		const FLOAT_TYPE normDeriv = sqrt_float_type(d0 * d0 + d1 * d1);
		if (normDeriv == 0) {
			thrust::get<0>(v) = 0;
			thrust::get<1>(v) = 0;
		} else {
			thrust::get<0>(v) = dMax_0 * d0 / normDeriv;
			thrust::get<1>(v) = dMax_0 * d1 / normDeriv;
		}
		thrust::get<2>(v) = (d2 >= 0) ? dMax_1 : -dMax_1;
	}
};

bool optDstb_execute_cuda(
	std::vector<beacls::UVec>& d_uvecs,
	const std::vector<beacls::UVec>& x_uvecs,
	const std::vector<beacls::UVec>& deriv_uvecs,
	const std::vector<FLOAT_TYPE>& dMax,
	const helperOC::DynSys_DMode_Type dMode
)
{
	bool result = true;
	const FLOAT_TYPE dMax_0 = dMax[0];
	const FLOAT_TYPE dMax_1 = dMax[1];
	if ((dMode == helperOC::DynSys_DMode_Max) || (dMode == helperOC::DynSys_DMode_Min)) {
		const FLOAT_TYPE moded_dMax_0 = (dMode == helperOC::DynSys_DMode_Max) ? dMax_0 : -dMax_0;
		const FLOAT_TYPE moded_dMax_1 = (dMode == helperOC::DynSys_DMode_Max) ? dMax_1 : -dMax_1;
		beacls::reallocateAsSrc(d_uvecs[0], deriv_uvecs[0]);
		beacls::reallocateAsSrc(d_uvecs[1], deriv_uvecs[0]);
		beacls::reallocateAsSrc(d_uvecs[2], deriv_uvecs[2]);
		FLOAT_TYPE* dOpt0_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[0]).ptr();
		FLOAT_TYPE* dOpt1_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[1]).ptr();
		FLOAT_TYPE* dOpt2_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[2]).ptr();
		const FLOAT_TYPE* deriv0_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[0]).ptr();
		const FLOAT_TYPE* deriv1_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[1]).ptr();
		const FLOAT_TYPE* deriv2_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[2]).ptr();

		if (is_cuda(deriv_uvecs[0]) && is_cuda(deriv_uvecs[1]) && is_cuda(deriv_uvecs[2])) {
			thrust::device_ptr<FLOAT_TYPE> dOpt0_dev_ptr = thrust::device_pointer_cast(dOpt0_ptr);
			thrust::device_ptr<FLOAT_TYPE> dOpt1_dev_ptr = thrust::device_pointer_cast(dOpt1_ptr);
			thrust::device_ptr<FLOAT_TYPE> dOpt2_dev_ptr = thrust::device_pointer_cast(dOpt2_ptr);
			thrust::device_ptr<const FLOAT_TYPE> deriv0_dev_ptr = thrust::device_pointer_cast(deriv0_ptr);
			thrust::device_ptr<const FLOAT_TYPE> deriv1_dev_ptr = thrust::device_pointer_cast(deriv1_ptr);
			thrust::device_ptr<const FLOAT_TYPE> deriv2_dev_ptr = thrust::device_pointer_cast(deriv2_ptr);
			cudaStream_t d_stream = beacls::get_stream(d_uvecs[0]);
			d_uvecs[1].set_cudaStream(d_uvecs[0].get_cudaStream());
			d_uvecs[2].set_cudaStream(d_uvecs[0].get_cudaStream());
			auto dst_src_Tuple = thrust::make_tuple(
				dOpt0_dev_ptr, dOpt1_dev_ptr, dOpt2_dev_ptr,
				deriv0_dev_ptr, deriv1_dev_ptr, deriv2_dev_ptr);
			auto dst_src_Iterator = thrust::make_zip_iterator(dst_src_Tuple);

			thrust::for_each(thrust::cuda::par.on(d_stream), 
			dst_src_Iterator, dst_src_Iterator + deriv_uvecs[0].size(), 
				Get_optDstb_dim0dim1dim2(moded_dMax_0, moded_dMax_1));
		} else {
			const FLOAT_TYPE d0 = deriv0_ptr[0];
			const FLOAT_TYPE d1 = deriv1_ptr[0];
			const FLOAT_TYPE d2 = deriv2_ptr[0];
			const FLOAT_TYPE normDeriv = sqrt_float_type(d0 * d0 + d1 * d1);
			if (normDeriv == 0) {
				dOpt0_ptr[0] = 0;
				dOpt1_ptr[0] = 0;
			} else {
				dOpt0_ptr[0] = moded_dMax_0 * d0 / normDeriv;
				dOpt1_ptr[0] = moded_dMax_0 * d1 / normDeriv;
			}
			dOpt2_ptr[0] = (d2 >= 0) ? moded_dMax_1 : -moded_dMax_1;
		}
	}
	else {
		std::cerr << "Unknown dMode!: " << dMode << std::endl;
		result = false;
	}
	return result;
}

struct Get_dynamics_dimAll_U0_U1_D0_D1_U2
{
	Get_dynamics_dimAll_U0_U1_D0_D1_U2() {}
	template<typename Tuple>
	__host__ __device__
	void operator()(Tuple v) const
	{
		const FLOAT_TYPE x = thrust::get<3>(v);
		const FLOAT_TYPE u0 = thrust::get<4>(v);
		const FLOAT_TYPE u1 = thrust::get<5>(v);
		const FLOAT_TYPE d0 = thrust::get<6>(v);
		const FLOAT_TYPE d1 = thrust::get<7>(v);
		const FLOAT_TYPE d2 = thrust::get<8>(v);
		FLOAT_TYPE sin_x;
		FLOAT_TYPE cos_x;
		sincos_float_type<FLOAT_TYPE>(x, sin_x, cos_x);
		thrust::get<0>(v) = u0*cos_x + d0;
		thrust::get<1>(v) = u0*sin_x + d1;
		thrust::get<2>(v) = u1 + d2;
	}
};
struct Get_dynamics_dim01_U0_d0_d1
{
	const FLOAT_TYPE d0;
	const FLOAT_TYPE d1;
	Get_dynamics_dim01_U0_d0_d1(const FLOAT_TYPE d0, const FLOAT_TYPE d1) : d0(d0), d1(d1) {}
	template<typename Tuple>
	__host__ __device__
	void operator()(Tuple v) const
	{
		const FLOAT_TYPE x = thrust::get<2>(v);
		const FLOAT_TYPE u0 = thrust::get<3>(v);
		FLOAT_TYPE sin_x;
		FLOAT_TYPE cos_x;
		sincos_float_type<FLOAT_TYPE>(x, sin_x, cos_x);
		thrust::get<0>(v) = u0*cos_x + d0;
		thrust::get<1>(v) = u0*sin_x + d1;
	}
};

struct Get_dynamics_dim0_U0_D0 {
public:
	template<typename Tuple>
	__host__ __device__
	FLOAT_TYPE operator()(const Tuple &v) const	{
		const FLOAT_TYPE x = thrust::get<0>(v);
		const FLOAT_TYPE u = thrust::get<1>(v);
		const FLOAT_TYPE d = thrust::get<2>(v);
		return u*cos_float_type<FLOAT_TYPE>(x)+d;
	}
};
struct Get_dynamics_dim0_U0_d0 {
public:
	const FLOAT_TYPE d;
	Get_dynamics_dim0_U0_d0(const FLOAT_TYPE d) : d(d) {}
	__host__ __device__
	FLOAT_TYPE operator()(const FLOAT_TYPE& x, const FLOAT_TYPE& u) const
	{
		return u*cos_float_type<FLOAT_TYPE>(x)+d;
	}
};
struct Get_dynamics_dim1_U0_D1 {
public:
	template<typename Tuple>
	__host__ __device__
	FLOAT_TYPE operator()(const Tuple &v) const
	{
		const FLOAT_TYPE x = thrust::get<0>(v);
		const FLOAT_TYPE u = thrust::get<1>(v);
		const FLOAT_TYPE d = thrust::get<2>(v);
		return u*sin_float_type<FLOAT_TYPE>(x)+d;
	}
};
struct Get_dynamics_dim1_U0_d1 {
public:
	const FLOAT_TYPE d;
	Get_dynamics_dim1_U0_d1(const FLOAT_TYPE d) : d(d) {}
	__host__ __device__
	FLOAT_TYPE operator()(const FLOAT_TYPE x, const FLOAT_TYPE u) const
	{
		return u*sin_float_type<FLOAT_TYPE>(x)+d;
	}
};

bool dynamics_cell_helper_execute_cuda_dimAll(
	std::vector<beacls::UVec>& dx_uvecs,
	const std::vector<beacls::UVec>& x_uvecs,
	const std::vector<beacls::UVec>& u_uvecs,
	const std::vector<beacls::UVec>& d_uvecs
) {
	bool result = true;
	const size_t src_x_dim_index = 2;
	beacls::reallocateAsSrc(dx_uvecs[0], x_uvecs[src_x_dim_index]);
	beacls::reallocateAsSrc(dx_uvecs[1], x_uvecs[src_x_dim_index]);
	beacls::reallocateAsSrc(dx_uvecs[2], u_uvecs[1]);
	FLOAT_TYPE* dx_dim0_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[0]).ptr();
	FLOAT_TYPE* dx_dim1_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[1]).ptr();
	FLOAT_TYPE* dx_dim2_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[2]).ptr();
	const FLOAT_TYPE* us_0_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[0]).ptr();
	const FLOAT_TYPE* us_1_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[1]).ptr();
	const FLOAT_TYPE* ds_0_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[0]).ptr();
	const FLOAT_TYPE* ds_1_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[1]).ptr();
	const FLOAT_TYPE* ds_2_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[2]).ptr();
	const FLOAT_TYPE* x_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[src_x_dim_index]).ptr();

	thrust::device_ptr<FLOAT_TYPE> dx_dim0_dev_ptr = thrust::device_pointer_cast(dx_dim0_ptr);
	thrust::device_ptr<FLOAT_TYPE> dx_dim1_dev_ptr = thrust::device_pointer_cast(dx_dim1_ptr);
	thrust::device_ptr<const FLOAT_TYPE> us_0_dev_ptr = thrust::device_pointer_cast(us_0_ptr);
	thrust::device_ptr<const FLOAT_TYPE> x_dev_ptr = thrust::device_pointer_cast(x_ptr);
	cudaStream_t dx_stream = beacls::get_stream(dx_uvecs[0]);
	dx_uvecs[1].set_cudaStream(dx_uvecs[0].get_cudaStream());

	beacls::synchronizeUVec(u_uvecs[0]);
	if (is_cuda(d_uvecs[0]) && is_cuda(d_uvecs[1]) && is_cuda(u_uvecs[1]) && is_cuda(d_uvecs[2])) {
		beacls::synchronizeUVec(d_uvecs[0]);
		thrust::device_ptr<FLOAT_TYPE> dx_dim2_dev_ptr = thrust::device_pointer_cast(dx_dim2_ptr);

		thrust::device_ptr<const FLOAT_TYPE> us_1_dev_ptr = thrust::device_pointer_cast(us_1_ptr);
		thrust::device_ptr<const FLOAT_TYPE> ds_0_dev_ptr = thrust::device_pointer_cast(ds_0_ptr);
		thrust::device_ptr<const FLOAT_TYPE> ds_1_dev_ptr = thrust::device_pointer_cast(ds_1_ptr);
		thrust::device_ptr<const FLOAT_TYPE> ds_2_dev_ptr = thrust::device_pointer_cast(ds_2_ptr);

		auto dst_src_Tuple = thrust::make_tuple(dx_dim0_dev_ptr, dx_dim1_dev_ptr, dx_dim2_dev_ptr, 
			x_dev_ptr, us_0_dev_ptr, us_1_dev_ptr, ds_0_dev_ptr, ds_1_dev_ptr, ds_2_dev_ptr);
		auto dst_src_Iterator = thrust::make_zip_iterator(dst_src_Tuple);
		
		thrust::for_each(thrust::cuda::par.on(dx_stream),
			dst_src_Iterator, dst_src_Iterator + x_uvecs[src_x_dim_index].size(), 
			Get_dynamics_dimAll_U0_U1_D0_D1_U2());
	}
	else if (!is_cuda(d_uvecs[0]) && !is_cuda(d_uvecs[1]) && !is_cuda(u_uvecs[1]) && !is_cuda(d_uvecs[2])) {
		const FLOAT_TYPE d0 = ds_0_ptr[0];
		const FLOAT_TYPE d1 = ds_1_ptr[0];
		auto dst_src_Tuple = thrust::make_tuple(dx_dim0_dev_ptr, dx_dim1_dev_ptr, x_dev_ptr, us_0_dev_ptr);
		auto dst_src_Iterator = thrust::make_zip_iterator(dst_src_Tuple);
		thrust::for_each(thrust::cuda::par.on(dx_stream),
			dst_src_Iterator, dst_src_Iterator + x_uvecs[src_x_dim_index].size(), 
			Get_dynamics_dim01_U0_d0_d1(d0, d1));
		const FLOAT_TYPE u1 = us_1_ptr[0];
		const FLOAT_TYPE d2 = ds_2_ptr[0];
		dx_dim2_ptr[0] = u1 + d2;
	} else {
		std::cerr << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << " Invalid data size" << std::endl;
		result = false;
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
	const size_t src_x_dim_index = 2;

	switch (dim) {
	case 0:
		if (beacls::is_cuda(u_uvecs[0])){
			beacls::synchronizeUVec(u_uvecs[0]);
			beacls::reallocateAsSrc(dx_uvec, x_uvecs[src_x_dim_index]);
			cudaStream_t dx_stream = beacls::get_stream(dx_uvec);
			FLOAT_TYPE* dx_dim_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvec).ptr();
			const FLOAT_TYPE* x_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[src_x_dim_index]).ptr();
			const FLOAT_TYPE* us_0_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[0]).ptr();
			const FLOAT_TYPE* ds_0_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[0]).ptr();

			thrust::device_ptr<FLOAT_TYPE> dx_dim_dev_ptr = thrust::device_pointer_cast(dx_dim_ptr);
			thrust::device_ptr<const FLOAT_TYPE> x_dev_ptr = thrust::device_pointer_cast(x_ptr);
			thrust::device_ptr<const FLOAT_TYPE> us_0_dev_ptr = thrust::device_pointer_cast(us_0_ptr);

			if (is_cuda(d_uvecs[0])){
				beacls::synchronizeUVec(d_uvecs[0]);
				thrust::device_ptr<const FLOAT_TYPE> ds_0_dev_ptr = thrust::device_pointer_cast(ds_0_ptr);
				auto src0_Tuple = thrust::make_tuple(x_dev_ptr, us_0_dev_ptr, ds_0_dev_ptr);
				auto src0_Iterator = thrust::make_zip_iterator(src0_Tuple);
				thrust::transform(thrust::cuda::par.on(dx_stream),
					src0_Iterator, src0_Iterator + x_uvecs[src_x_dim_index].size(), dx_dim_dev_ptr, 
					Get_dynamics_dim0_U0_D0());
			}
			else {	//!< ds_0_size != length
				const FLOAT_TYPE d0 = ds_0_ptr[0];
				thrust::transform(thrust::cuda::par.on(dx_stream),
					x_dev_ptr, x_dev_ptr + x_uvecs[src_x_dim_index].size(), us_0_dev_ptr, dx_dim_dev_ptr,
					Get_dynamics_dim0_U0_d0(d0));
			}
		}
		else {
			std::cerr << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << " Invalid data size" << std::endl;
			result = false;
		}
		break;
	case 1:
		if (beacls::is_cuda(u_uvecs[0])) {
			beacls::synchronizeUVec(u_uvecs[0]);

			beacls::reallocateAsSrc(dx_uvec, x_uvecs[src_x_dim_index]);
			cudaStream_t dx_stream = beacls::get_stream(dx_uvec);
			FLOAT_TYPE* dx_dim_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvec).ptr();
			const FLOAT_TYPE* x_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[src_x_dim_index]).ptr();
			const FLOAT_TYPE* us_0_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[0]).ptr();
			const FLOAT_TYPE* ds_1_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[1]).ptr();

			thrust::device_ptr<FLOAT_TYPE> dx_dim_dev_ptr = thrust::device_pointer_cast(dx_dim_ptr);
			thrust::device_ptr<const FLOAT_TYPE> x_dev_ptr = thrust::device_pointer_cast(x_ptr);
			thrust::device_ptr<const FLOAT_TYPE> us_0_dev_ptr = thrust::device_pointer_cast(us_0_ptr);

			if (is_cuda(d_uvecs[1])) {
				beacls::synchronizeUVec(d_uvecs[1]);
				thrust::device_ptr<const FLOAT_TYPE> ds_1_dev_ptr = thrust::device_pointer_cast(ds_1_ptr);
				auto src0_Tuple = thrust::make_tuple(x_dev_ptr, us_0_dev_ptr, ds_1_dev_ptr);
				auto src0_Iterator = thrust::make_zip_iterator(src0_Tuple);
				thrust::transform(thrust::cuda::par.on(dx_stream),
					src0_Iterator, src0_Iterator + x_uvecs[src_x_dim_index].size(), dx_dim_dev_ptr, 
					Get_dynamics_dim1_U0_D1());
			}
			else {	//!< ds_1_size != length
				const FLOAT_TYPE d1 = ds_1_ptr[0];
				thrust::transform(thrust::cuda::par.on(dx_stream),
					x_dev_ptr, x_dev_ptr + x_uvecs[src_x_dim_index].size(), us_0_dev_ptr, dx_dim_dev_ptr, Get_dynamics_dim1_U0_d1(d1));
			}
		}
		else {
			std::cerr << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << " Invalid data size" << std::endl;
			result = false;
		}
		break;
	case 2:
	{
		beacls::reallocateAsSrc(dx_uvec, u_uvecs[1]);
		FLOAT_TYPE* dx_dim_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvec).ptr();
		const FLOAT_TYPE* us_1_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[1]).ptr();
		const FLOAT_TYPE* ds_2_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[2]).ptr();

		if (is_cuda(u_uvecs[1]) && is_cuda(d_uvecs[2])) {
			beacls::synchronizeUVec(u_uvecs[1]);
			beacls::synchronizeUVec(d_uvecs[2]);
			cudaStream_t dx_stream = beacls::get_stream(dx_uvec);
			thrust::device_ptr<FLOAT_TYPE> dx_dim_dev_ptr = thrust::device_pointer_cast(dx_dim_ptr);
			thrust::device_ptr<const FLOAT_TYPE> us_1_dev_ptr = thrust::device_pointer_cast(us_1_ptr);
			thrust::device_ptr<const FLOAT_TYPE> ds_2_dev_ptr = thrust::device_pointer_cast(ds_2_ptr);
			thrust::transform(thrust::cuda::par.on(dx_stream),
				us_1_dev_ptr, us_1_dev_ptr + u_uvecs[1].size(), ds_2_dev_ptr, dx_dim_dev_ptr,
				thrust::plus<FLOAT_TYPE>());
		} else if(!is_cuda(u_uvecs[1]) && !is_cuda(d_uvecs[2])) {
			const FLOAT_TYPE u1 = us_1_ptr[0];
			const FLOAT_TYPE d2 = ds_2_ptr[0];
			dx_dim_ptr[0] = u1 + d2;
		}
		else {
			std::cerr << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << " Invalid data size" << std::endl;
			result = false;
		}
	}
		break;
	default:
		std::cerr << "Only dimension 1-4 are defined for dynamics of Plane!" << std::endl;
		result = false;
		break;
	}
	return result;
}

struct Get_optCtrlUL_dim0_d {
public:
	const FLOAT_TYPE vrange_Min;
	const FLOAT_TYPE vrange_Max;
	const FLOAT_TYPE derivMin0;
	const FLOAT_TYPE derivMax0;
	const FLOAT_TYPE derivMin1;
	const FLOAT_TYPE derivMax1;
	Get_optCtrlUL_dim0_d(
		const FLOAT_TYPE vrange_Min, 
		const FLOAT_TYPE vrange_Max, 
		const FLOAT_TYPE derivMin0,
		const FLOAT_TYPE derivMax0, 
		const FLOAT_TYPE derivMin1,
		const FLOAT_TYPE derivMax1) : 
			vrange_Min(vrange_Min),
			vrange_Max(vrange_Max), 
			derivMin0(derivMin0), 
			derivMax0(derivMax0),
			derivMin1(derivMin1), 
			derivMax1(derivMax1) {}
	template<typename Tuple>
	__host__ __device__
	void operator()(Tuple v) const
	{
		const FLOAT_TYPE y2 = thrust::get<2>(v);
		FLOAT_TYPE sin_y2;
		FLOAT_TYPE cos_y2;
		sincos_float_type<FLOAT_TYPE>(y2, sin_y2, cos_y2);
		const FLOAT_TYPE detMax1 = derivMax0 * cos_y2 + derivMax1 * sin_y2;
		const FLOAT_TYPE detMin1 = derivMin0 * cos_y2 + derivMin1 * sin_y2;
		thrust::get<0>(v) = (detMin1 >= 0) ? vrange_Max : vrange_Min;
		thrust::get<1>(v) = (detMax1 >= 0) ? vrange_Max : vrange_Min;
	}
};

bool optCtrl_execute_cuda(
	std::vector<beacls::UVec>& uL_uvecs,
	std::vector<beacls::UVec>& uU_uvecs,
	const std::vector<beacls::UVec>& x_uvecs,
	const std::vector<beacls::UVec>& derivMin_uvecs,
	const std::vector<beacls::UVec>& derivMax_uvecs,
	const FLOAT_TYPE wMax,
	const FLOAT_TYPE vrange_min,
	const FLOAT_TYPE vrange_max,
	const helperOC::DynSys_UMode_Type uMode
)
{
	bool result = true;
	if ((uMode == helperOC::DynSys_UMode_Max) || (uMode == helperOC::DynSys_UMode_Min)) {
		const FLOAT_TYPE moded_vrange_max = (uMode == helperOC::DynSys_UMode_Max) ? vrange_max : vrange_min;
		const FLOAT_TYPE moded_vrange_min = (uMode == helperOC::DynSys_UMode_Max) ? vrange_min : vrange_max;
		const FLOAT_TYPE moded_wMax = (uMode == helperOC::DynSys_UMode_Max) ? wMax : -wMax;
		beacls::reallocateAsSrc(uU_uvecs[0], x_uvecs[2]);
		beacls::reallocateAsSrc(uU_uvecs[1], derivMax_uvecs[2]);
		beacls::reallocateAsSrc(uL_uvecs[0], x_uvecs[2]);
		beacls::reallocateAsSrc(uL_uvecs[1], derivMin_uvecs[2]);
		FLOAT_TYPE* uU_0_ptr = beacls::UVec_<FLOAT_TYPE>(uU_uvecs[0]).ptr();
		FLOAT_TYPE* uU_1_ptr = beacls::UVec_<FLOAT_TYPE>(uU_uvecs[1]).ptr();
		FLOAT_TYPE* uL_0_ptr = beacls::UVec_<FLOAT_TYPE>(uL_uvecs[0]).ptr();
		FLOAT_TYPE* uL_1_ptr = beacls::UVec_<FLOAT_TYPE>(uL_uvecs[1]).ptr();
		const FLOAT_TYPE* y2_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[2]).ptr();
		const FLOAT_TYPE* derivMax0_ptr = beacls::UVec_<FLOAT_TYPE>(derivMax_uvecs[0]).ptr();
		const FLOAT_TYPE* derivMax1_ptr = beacls::UVec_<FLOAT_TYPE>(derivMax_uvecs[1]).ptr();
		const FLOAT_TYPE* derivMax2_ptr = beacls::UVec_<FLOAT_TYPE>(derivMax_uvecs[2]).ptr();
		const FLOAT_TYPE* derivMin0_ptr = beacls::UVec_<FLOAT_TYPE>(derivMin_uvecs[0]).ptr();
		const FLOAT_TYPE* derivMin1_ptr = beacls::UVec_<FLOAT_TYPE>(derivMin_uvecs[1]).ptr();
		const FLOAT_TYPE* derivMin2_ptr = beacls::UVec_<FLOAT_TYPE>(derivMin_uvecs[2]).ptr();
		thrust::device_ptr<const FLOAT_TYPE> y2_dev_ptr = thrust::device_pointer_cast(y2_ptr);
		if (is_cuda(derivMax_uvecs[2]) && is_cuda(derivMin_uvecs[2])) {
			std::cerr << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << " Invalid data size" << std::endl;
			result = false;
		}
		else {
			thrust::device_ptr<FLOAT_TYPE> uU_0_dev_ptr = thrust::device_pointer_cast(uU_0_ptr);
			thrust::device_ptr<FLOAT_TYPE> uL_0_dev_ptr = thrust::device_pointer_cast(uL_0_ptr);
			cudaStream_t u_stream = beacls::get_stream(uU_uvecs[0]);
			uL_uvecs[0].set_cudaStream(uU_uvecs[0].get_cudaStream());
			const FLOAT_TYPE dMax0 = derivMax0_ptr[0];
			const FLOAT_TYPE dMax1 = derivMax1_ptr[0];
			const FLOAT_TYPE dMax2 = derivMax2_ptr[0];
			const FLOAT_TYPE dMin0 = derivMin0_ptr[0];
			const FLOAT_TYPE dMin1 = derivMin1_ptr[0];
			const FLOAT_TYPE dMin2 = derivMin2_ptr[0];
			auto src_dst_Tuple = thrust::make_tuple(uL_0_dev_ptr, uU_0_dev_ptr, y2_dev_ptr);
			auto src_dst_Iterator = thrust::make_zip_iterator(src_dst_Tuple);

			thrust::for_each(thrust::cuda::par.on(u_stream),
				src_dst_Iterator, src_dst_Iterator + x_uvecs[2].size(),
				Get_optCtrlUL_dim0_d(moded_vrange_min, moded_vrange_max, dMin0, dMax0, dMin1, dMax1));
			uU_1_ptr[0] = (dMax2 >= 0) ? moded_wMax : -moded_wMax;
			uL_1_ptr[0] = (dMin2 >= 0) ? moded_wMax : -moded_wMax;
		}
	}
	else {
		std::cerr << "Unknown uMode!: " << uMode << std::endl;
		result = false;
	}
	return result;
}

bool optDstb_execute_cuda(
	std::vector<beacls::UVec>& dL_uvecs,
	std::vector<beacls::UVec>& dU_uvecs,
	const std::vector<beacls::UVec>& x_uvecs,
	const std::vector<beacls::UVec>& derivMin_uvecs,
	const std::vector<beacls::UVec>& derivMax_uvecs,
	const std::vector<FLOAT_TYPE>& dMax,
	const helperOC::DynSys_DMode_Type dMode
)
{
	bool result = true;
	const FLOAT_TYPE dMax_0 = dMax[0];
	const FLOAT_TYPE dMax_1 = dMax[1];
	if ((dMode == helperOC::DynSys_DMode_Max) || (dMode == helperOC::DynSys_DMode_Min)) {
		const FLOAT_TYPE moded_dMax_0 = (dMode == helperOC::DynSys_DMode_Max) ? dMax_0 : -dMax_0;
		const FLOAT_TYPE moded_dMax_1 = (dMode == helperOC::DynSys_DMode_Max) ? dMax_1 : -dMax_1;
		beacls::reallocateAsSrc(dU_uvecs[0], derivMax_uvecs[0]);
		beacls::reallocateAsSrc(dU_uvecs[1], derivMax_uvecs[0]);
		beacls::reallocateAsSrc(dU_uvecs[2], derivMax_uvecs[2]);
		beacls::reallocateAsSrc(dL_uvecs[0], derivMin_uvecs[0]);
		beacls::reallocateAsSrc(dL_uvecs[1], derivMin_uvecs[0]);
		beacls::reallocateAsSrc(dL_uvecs[2], derivMin_uvecs[2]);
		FLOAT_TYPE* dU_0_ptr = beacls::UVec_<FLOAT_TYPE>(dU_uvecs[0]).ptr();
		FLOAT_TYPE* dU_1_ptr = beacls::UVec_<FLOAT_TYPE>(dU_uvecs[1]).ptr();
		FLOAT_TYPE* dU_2_ptr = beacls::UVec_<FLOAT_TYPE>(dU_uvecs[2]).ptr();
		FLOAT_TYPE* dL_0_ptr = beacls::UVec_<FLOAT_TYPE>(dL_uvecs[0]).ptr();
		FLOAT_TYPE* dL_1_ptr = beacls::UVec_<FLOAT_TYPE>(dL_uvecs[1]).ptr();
		FLOAT_TYPE* dL_2_ptr = beacls::UVec_<FLOAT_TYPE>(dL_uvecs[2]).ptr();
		const FLOAT_TYPE* derivMax0_ptr = beacls::UVec_<FLOAT_TYPE>(derivMax_uvecs[0]).ptr();
		const FLOAT_TYPE* derivMax1_ptr = beacls::UVec_<FLOAT_TYPE>(derivMax_uvecs[1]).ptr();
		const FLOAT_TYPE* derivMax2_ptr = beacls::UVec_<FLOAT_TYPE>(derivMax_uvecs[2]).ptr();
		const FLOAT_TYPE* derivMin0_ptr = beacls::UVec_<FLOAT_TYPE>(derivMin_uvecs[0]).ptr();
		const FLOAT_TYPE* derivMin1_ptr = beacls::UVec_<FLOAT_TYPE>(derivMin_uvecs[1]).ptr();
		const FLOAT_TYPE* derivMin2_ptr = beacls::UVec_<FLOAT_TYPE>(derivMin_uvecs[2]).ptr();

		if (is_cuda(derivMax_uvecs[0]) && is_cuda(derivMax_uvecs[1]) && is_cuda(derivMax_uvecs[2])) {
			std::cerr << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << " Invalid data size" << std::endl;
			result = false;
		}
		else {
			const FLOAT_TYPE derivMax0 = derivMax0_ptr[0];
			const FLOAT_TYPE derivMax1 = derivMax1_ptr[0];
			const FLOAT_TYPE derivMax2 = derivMax2_ptr[0];
			const FLOAT_TYPE derivMin0 = derivMin0_ptr[0];
			const FLOAT_TYPE derivMin1 = derivMin1_ptr[0];
			const FLOAT_TYPE derivMin2 = derivMin2_ptr[0];
			const FLOAT_TYPE normDerivMax = sqrt_float_type(derivMax0 * derivMax0 + derivMax1 * derivMax1);
			const FLOAT_TYPE normDerivMin = sqrt_float_type(derivMin0 * derivMin0 + derivMin1 * derivMin1);
			if (normDerivMax == 0) {
				dU_0_ptr[0] = 0;
				dU_1_ptr[0] = 0;
			}
			else {
				dU_0_ptr[0] = moded_dMax_0 * derivMax0 / normDerivMax;
				dU_1_ptr[0] = moded_dMax_0 * derivMax1 / normDerivMax;
			}
			if (normDerivMin == 0) {
				dL_0_ptr[0] = 0;
				dL_1_ptr[0] = 0;
			}
			else {
				dL_0_ptr[0] = moded_dMax_0 * derivMin0 / normDerivMin;
				dL_1_ptr[0] = moded_dMax_0 * derivMin1 / normDerivMin;
			}
			dU_2_ptr[0] = (derivMax2 >= 0) ? moded_dMax_1 : -moded_dMax_1;
			dL_2_ptr[0] = (derivMin2 >= 0) ? moded_dMax_1 : -moded_dMax_1;
		}
	}
	else {
		std::cerr << "Unknown dMode!: " << dMode << std::endl;
		result = false;
	}
	return result;
}
struct Get_alpha_dim0_U0_d0 {
public:
	const FLOAT_TYPE dL0;
	const FLOAT_TYPE dU0;
	Get_alpha_dim0_U0_d0(const FLOAT_TYPE dL0, const FLOAT_TYPE dU0) : dL0(dL0), dU0(dU0) {}
	template<typename Tuple>
	__host__ __device__
	void operator()(Tuple v) const
	{
		const FLOAT_TYPE x = thrust::get<1>(v);
		const FLOAT_TYPE uL0 = thrust::get<2>(v);
		const FLOAT_TYPE uU0 = thrust::get<3>(v);
		const FLOAT_TYPE cos_x = cos_float_type<FLOAT_TYPE>(x);
		const FLOAT_TYPE dxUU =  uU0*cos_x + dU0;
		const FLOAT_TYPE dxUL =  uU0*cos_x + dL0;
		const FLOAT_TYPE dxLU =  uL0*cos_x + dU0;
		const FLOAT_TYPE dxLL =  uL0*cos_x + dL0;
		const FLOAT_TYPE max0 = max_float_type<FLOAT_TYPE>(abs_float_type<FLOAT_TYPE>(dxUU), abs_float_type<FLOAT_TYPE>(dxUL));
		const FLOAT_TYPE max1 = max_float_type<FLOAT_TYPE>(abs_float_type<FLOAT_TYPE>(dxLL), abs_float_type<FLOAT_TYPE>(dxLU));
		thrust::get<0>(v) =  max_float_type<FLOAT_TYPE>(max0, max1);
	}
};
struct Get_alpha_dim1_U0_d1 {
public:
	const FLOAT_TYPE dL1;
	const FLOAT_TYPE dU1;
	Get_alpha_dim1_U0_d1(const FLOAT_TYPE dL1, const FLOAT_TYPE dU1) : dL1(dL1), dU1(dU1) {}
	template<typename Tuple>
	__host__ __device__
	void operator()(Tuple v) const
	{
		const FLOAT_TYPE x = thrust::get<1>(v);
		const FLOAT_TYPE uL0 = thrust::get<2>(v);
		const FLOAT_TYPE uU0 = thrust::get<3>(v);
		const FLOAT_TYPE sin_x = sin_float_type<FLOAT_TYPE>(x);
		const FLOAT_TYPE dxUU =  uU0*sin_x + dU1;
		const FLOAT_TYPE dxUL =  uU0*sin_x + dL1;
		const FLOAT_TYPE dxLU =  uL0*sin_x + dU1;
		const FLOAT_TYPE dxLL =  uL0*sin_x + dL1;
		const FLOAT_TYPE max0 = max_float_type<FLOAT_TYPE>(abs_float_type<FLOAT_TYPE>(dxUU), abs_float_type<FLOAT_TYPE>(dxUL));
		const FLOAT_TYPE max1 = max_float_type<FLOAT_TYPE>(abs_float_type<FLOAT_TYPE>(dxLL), abs_float_type<FLOAT_TYPE>(dxLU));
		thrust::get<0>(v) =  max_float_type<FLOAT_TYPE>(max0, max1);
	}
};
bool dynamics_cell_helper_execute_cuda(
	beacls::UVec& alpha_uvec,
	const std::vector<beacls::UVec>& x_uvecs,
	const std::vector<beacls::UVec>& uL_uvecs,
	const std::vector<beacls::UVec>& uU_uvecs,
	const std::vector<beacls::UVec>& dL_uvecs,
	const std::vector<beacls::UVec>& dU_uvecs,
	const size_t dim
) {
	bool result = true;
	const size_t src_x_dim_index = 2;
	switch (dim) {
	case 0:
		if (beacls::is_cuda(uU_uvecs[0]) && beacls::is_cuda(uL_uvecs[0])) {
			beacls::synchronizeUVec(uU_uvecs[0]);
			beacls::synchronizeUVec(uL_uvecs[0]);
			beacls::reallocateAsSrc(alpha_uvec, x_uvecs[src_x_dim_index]);
			cudaStream_t alpha_stream = beacls::get_stream(alpha_uvec);
			FLOAT_TYPE* alpha_ptr = beacls::UVec_<FLOAT_TYPE>(alpha_uvec).ptr();
			const FLOAT_TYPE* x_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[src_x_dim_index]).ptr();
			const FLOAT_TYPE* uUs_0_ptr = beacls::UVec_<FLOAT_TYPE>(uU_uvecs[0]).ptr();
			const FLOAT_TYPE* dUs_0_ptr = beacls::UVec_<FLOAT_TYPE>(dU_uvecs[0]).ptr();
			const FLOAT_TYPE* uLs_0_ptr = beacls::UVec_<FLOAT_TYPE>(uL_uvecs[0]).ptr();
			const FLOAT_TYPE* dLs_0_ptr = beacls::UVec_<FLOAT_TYPE>(dL_uvecs[0]).ptr();
			thrust::device_ptr<FLOAT_TYPE> alpha_dev_ptr = thrust::device_pointer_cast(alpha_ptr);
			thrust::device_ptr<const FLOAT_TYPE> x_dev_ptr = thrust::device_pointer_cast(x_ptr);
			thrust::device_ptr<const FLOAT_TYPE> uUs_0_dev_ptr = thrust::device_pointer_cast(uUs_0_ptr);
			thrust::device_ptr<const FLOAT_TYPE> uLs_0_dev_ptr = thrust::device_pointer_cast(uLs_0_ptr);

			if (is_cuda(dU_uvecs[0]) && is_cuda(dL_uvecs[0])) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << " Invalid data size" << std::endl;
				result = false;
			}
			else {	//!< ds_0_size != length
				const FLOAT_TYPE dU0 = dUs_0_ptr[0];
				const FLOAT_TYPE dL0 = dLs_0_ptr[0];
				auto src_dst_Tuple = thrust::make_tuple(alpha_dev_ptr, x_dev_ptr, uLs_0_dev_ptr, uUs_0_dev_ptr);
				auto src_dst_Iterator = thrust::make_zip_iterator(src_dst_Tuple);

				thrust::for_each(thrust::cuda::par.on(alpha_stream),
					src_dst_Iterator, src_dst_Iterator + x_uvecs[src_x_dim_index].size(),
					Get_alpha_dim0_U0_d0(dL0, dU0));
			}
		}
		else {
			std::cerr << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << " Invalid data size" << std::endl;
			result = false;
		}
		break;
	case 1:
		if (beacls::is_cuda(uU_uvecs[0]) && beacls::is_cuda(uL_uvecs[0])) {
			beacls::reallocateAsSrc(alpha_uvec, x_uvecs[src_x_dim_index]);
			beacls::synchronizeUVec(uU_uvecs[0]);
			beacls::synchronizeUVec(uL_uvecs[0]);
			beacls::reallocateAsSrc(alpha_uvec, x_uvecs[src_x_dim_index]);
			cudaStream_t alpha_stream = beacls::get_stream(alpha_uvec);
			FLOAT_TYPE* alpha_ptr = beacls::UVec_<FLOAT_TYPE>(alpha_uvec).ptr();
			const FLOAT_TYPE* x_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[src_x_dim_index]).ptr();
			const FLOAT_TYPE* uUs_0_ptr = beacls::UVec_<FLOAT_TYPE>(uU_uvecs[0]).ptr();
			const FLOAT_TYPE* dUs_1_ptr = beacls::UVec_<FLOAT_TYPE>(dU_uvecs[1]).ptr();
			const FLOAT_TYPE* uLs_0_ptr = beacls::UVec_<FLOAT_TYPE>(uL_uvecs[0]).ptr();
			const FLOAT_TYPE* dLs_1_ptr = beacls::UVec_<FLOAT_TYPE>(dL_uvecs[1]).ptr();
			thrust::device_ptr<FLOAT_TYPE> alpha_dev_ptr = thrust::device_pointer_cast(alpha_ptr);
			thrust::device_ptr<const FLOAT_TYPE> x_dev_ptr = thrust::device_pointer_cast(x_ptr);
			thrust::device_ptr<const FLOAT_TYPE> uUs_0_dev_ptr = thrust::device_pointer_cast(uUs_0_ptr);
			thrust::device_ptr<const FLOAT_TYPE> uLs_0_dev_ptr = thrust::device_pointer_cast(uLs_0_ptr);

			if (is_cuda(dU_uvecs[1]) && is_cuda(dU_uvecs[1])) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << " Invalid data size" << std::endl;
				result = false;
			}
			else {	//!< ds_1_size != length
				const FLOAT_TYPE dU1 = dUs_1_ptr[0];
				const FLOAT_TYPE dL1 = dLs_1_ptr[0];
				auto src_dst_Tuple = thrust::make_tuple(alpha_dev_ptr, x_dev_ptr, uLs_0_dev_ptr, uUs_0_dev_ptr);
				auto src_dst_Iterator = thrust::make_zip_iterator(src_dst_Tuple);

				thrust::for_each(thrust::cuda::par.on(alpha_stream),
					src_dst_Iterator, src_dst_Iterator + x_uvecs[src_x_dim_index].size(),
					Get_alpha_dim1_U0_d1(dL1, dU1));
			}
		}
		else {
			std::cerr << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << " Invalid data size" << std::endl;
			result = false;
		}
		break;
	case 2:
	{
		beacls::reallocateAsSrc(alpha_uvec, uU_uvecs[1]);
		FLOAT_TYPE* alpha_ptr = beacls::UVec_<FLOAT_TYPE>(alpha_uvec).ptr();
		const FLOAT_TYPE* uUs_1_ptr = beacls::UVec_<FLOAT_TYPE>(uU_uvecs[1]).ptr();
		const FLOAT_TYPE* dUs_2_ptr = beacls::UVec_<FLOAT_TYPE>(dU_uvecs[2]).ptr();
		const FLOAT_TYPE* uLs_1_ptr = beacls::UVec_<FLOAT_TYPE>(uL_uvecs[1]).ptr();
		const FLOAT_TYPE* dLs_2_ptr = beacls::UVec_<FLOAT_TYPE>(dL_uvecs[2]).ptr();
		if (is_cuda(uU_uvecs[1]) && is_cuda(dU_uvecs[2])) {
			std::cerr << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << " Invalid data size" << std::endl;
			result = false;
		}
		else if (!is_cuda(uU_uvecs[1]) && !is_cuda(dU_uvecs[2]) && !is_cuda(uL_uvecs[1]) && !is_cuda(dL_uvecs[2])) {
			const FLOAT_TYPE uU1 = uUs_1_ptr[0];
			const FLOAT_TYPE dU2 = dUs_2_ptr[0];
			const FLOAT_TYPE uL1 = uLs_1_ptr[0];
			const FLOAT_TYPE dL2 = dLs_2_ptr[0];
			const FLOAT_TYPE dxUU = uU1 + dU2;
			const FLOAT_TYPE dxUL = uU1 + dL2;
			const FLOAT_TYPE dxLU = uL1 + dU2;
			const FLOAT_TYPE dxLL = uL1 + dL2;
			const FLOAT_TYPE max0 = max_float_type<FLOAT_TYPE>(abs_float_type<FLOAT_TYPE>(dxUU), abs_float_type<FLOAT_TYPE>(dxUL));
			const FLOAT_TYPE max1 = max_float_type<FLOAT_TYPE>(abs_float_type<FLOAT_TYPE>(dxLL), abs_float_type<FLOAT_TYPE>(dxLU));
			alpha_ptr[0] = max_float_type<FLOAT_TYPE>(max0, max1);
		}
		else {
			std::cerr << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << " Invalid data size" << std::endl;
			result = false;
		}
	}
	break;
	default:
		std::cerr << "Only dimension 1-4 are defined for dynamics of Plane!" << std::endl;
		result = false;
		break;
	}
	return result;
}
	struct HamFunction_Neg {
	public:
		const FLOAT_TYPE wMax;
		const FLOAT_TYPE vrange_min;
		const FLOAT_TYPE vrange_max;
		const FLOAT_TYPE dMax_0;
		const FLOAT_TYPE dMax_1;
		HamFunction_Neg(
			const FLOAT_TYPE wMax,
			const FLOAT_TYPE vrange_min,
			const FLOAT_TYPE vrange_max,
			const FLOAT_TYPE dMax_0,
			const FLOAT_TYPE dMax_1
			) : 
			wMax(wMax),
			vrange_min(vrange_min),
			vrange_max(vrange_max),
			dMax_0(dMax_0),
			dMax_1(dMax_1) {}
		template<typename Tuple>
		__host__ __device__
		FLOAT_TYPE operator()(const Tuple v) const
		{
			const FLOAT_TYPE y2 = thrust::get<0>(v);
			const FLOAT_TYPE deriv0 = thrust::get<1>(v);
			const FLOAT_TYPE deriv1 = thrust::get<2>(v);
			const FLOAT_TYPE deriv2 = thrust::get<3>(v);

			FLOAT_TYPE dx0;
			FLOAT_TYPE dx1;
			FLOAT_TYPE dx2;
			get_dxs(dx0, dx1, dx2, y2, deriv0, deriv1, deriv2, wMax, vrange_min, vrange_max, dMax_0, dMax_1);
			return  - (deriv0 * dx0 + deriv1 * dx1 + deriv2 * dx2);
		}
	};
	struct HamFunction_Pos {
	public:
		const FLOAT_TYPE wMax;
		const FLOAT_TYPE vrange_min;
		const FLOAT_TYPE vrange_max;
		const FLOAT_TYPE dMax_0;
		const FLOAT_TYPE dMax_1;
		HamFunction_Pos(
			const FLOAT_TYPE wMax,
			const FLOAT_TYPE vrange_min,
			const FLOAT_TYPE vrange_max,
			const FLOAT_TYPE dMax_0,
			const FLOAT_TYPE dMax_1
			) : 
			wMax(wMax),
			vrange_min(vrange_min),
			vrange_max(vrange_max),
			dMax_0(dMax_0),
			dMax_1(dMax_1) {}
		template<typename Tuple>
		__host__ __device__
		FLOAT_TYPE operator()(const Tuple v) const
		{
			const FLOAT_TYPE y2 = thrust::get<0>(v);
			const FLOAT_TYPE deriv0 = thrust::get<1>(v);
			const FLOAT_TYPE deriv1 = thrust::get<2>(v);
			const FLOAT_TYPE deriv2 = thrust::get<3>(v);

			FLOAT_TYPE dx0;
			FLOAT_TYPE dx1;
			FLOAT_TYPE dx2;
			get_dxs(dx0, dx1, dx2, y2, deriv0, deriv1, deriv2, wMax, vrange_min, vrange_max, dMax_0, dMax_1);
			return (deriv0 * dx0 + deriv1 * dx1 + deriv2 * dx2);
		}
	};

	bool HamFunction_cuda(
		beacls::UVec& hamValue_uvec,
		const std::vector<beacls::UVec>& x_uvecs,
		const std::vector<beacls::UVec>& deriv_uvecs,
		const FLOAT_TYPE wMax,
		const FLOAT_TYPE vrange_min,
		const FLOAT_TYPE vrange_max,
		const beacls::FloatVec& dMax,
		const helperOC::DynSys_UMode_Type uMode,
		const helperOC::DynSys_DMode_Type dMode,
		const bool negate) {
		if ((uMode != helperOC::DynSys_UMode_Max) && (uMode != helperOC::DynSys_UMode_Min)) return false;
		if ((dMode != helperOC::DynSys_DMode_Max) && (dMode != helperOC::DynSys_DMode_Min)) return false;
		const FLOAT_TYPE moded_vrange_max = (uMode == helperOC::DynSys_UMode_Max) ? vrange_max : vrange_min;
		const FLOAT_TYPE moded_vrange_min = (uMode == helperOC::DynSys_UMode_Max) ? vrange_min : vrange_max;
		const FLOAT_TYPE moded_wMax = (uMode == helperOC::DynSys_UMode_Max) ? wMax : -wMax;
		const FLOAT_TYPE dMax_0 = dMax[0];
		const FLOAT_TYPE dMax_1 = dMax[1];
		const FLOAT_TYPE moded_dMax_0 = (dMode == helperOC::DynSys_DMode_Max) ? dMax_0 : -dMax_0;
		const FLOAT_TYPE moded_dMax_1 = (dMode == helperOC::DynSys_DMode_Max) ? dMax_1 : -dMax_1;
		const size_t src_x_dim_index = 2;
		const FLOAT_TYPE* y2_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[src_x_dim_index]).ptr();
		const FLOAT_TYPE* deriv0_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[0]).ptr();
		const FLOAT_TYPE* deriv1_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[1]).ptr();
		const FLOAT_TYPE* deriv2_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvecs[2]).ptr();

		bool result = true;

		beacls::reallocateAsSrc(hamValue_uvec, x_uvecs[src_x_dim_index]);
		FLOAT_TYPE* hamValue_ptr = beacls::UVec_<FLOAT_TYPE>(hamValue_uvec).ptr();
		cudaStream_t hamValue_stream = beacls::get_stream(hamValue_uvec);

		const FLOAT_TYPE* x_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[src_x_dim_index]).ptr();
		thrust::device_ptr<FLOAT_TYPE> hamValue_dev_ptr = thrust::device_pointer_cast(hamValue_ptr);
		thrust::device_ptr<const FLOAT_TYPE> y2_dev_ptr = thrust::device_pointer_cast(y2_ptr);
		thrust::device_ptr<const FLOAT_TYPE> deriv0_dev_ptr = thrust::device_pointer_cast(deriv0_ptr);
		thrust::device_ptr<const FLOAT_TYPE> deriv1_dev_ptr = thrust::device_pointer_cast(deriv1_ptr);
		thrust::device_ptr<const FLOAT_TYPE> deriv2_dev_ptr = thrust::device_pointer_cast(deriv2_ptr);

		auto src_dst_Tuple = thrust::make_tuple(y2_dev_ptr, deriv0_dev_ptr, deriv1_dev_ptr, deriv2_dev_ptr);
		auto src_dst_Iterator = thrust::make_zip_iterator(src_dst_Tuple);

		//!< Negate hamValue if backward reachable set
		if (negate) {
			thrust::transform(thrust::cuda::par.on(hamValue_stream),
				src_dst_Iterator, src_dst_Iterator + x_uvecs[src_x_dim_index].size(), hamValue_dev_ptr,
				HamFunction_Neg(moded_wMax, moded_vrange_min, moded_vrange_max, moded_dMax_0, moded_dMax_1));
		}
		else {
			thrust::transform(thrust::cuda::par.on(hamValue_stream),
				src_dst_Iterator, src_dst_Iterator + x_uvecs[src_x_dim_index].size(), hamValue_dev_ptr,
				HamFunction_Pos(moded_wMax, moded_vrange_min, moded_vrange_max, moded_dMax_0, moded_dMax_1));
		}
		return result;
	}

	struct PartialFunction_dim0 {
	public:
		const FLOAT_TYPE derivMin0;
		const FLOAT_TYPE derivMax0;
		const FLOAT_TYPE derivMin1;
		const FLOAT_TYPE derivMax1;
		const FLOAT_TYPE dL0;
		const FLOAT_TYPE dU0;
		const FLOAT_TYPE vrange_min;
		const FLOAT_TYPE vrange_max;
			PartialFunction_dim0(
			const FLOAT_TYPE derivMin0,
			const FLOAT_TYPE derivMax0,
			const FLOAT_TYPE derivMin1,
			const FLOAT_TYPE derivMax1,
			const FLOAT_TYPE dL0,
			const FLOAT_TYPE dU0,
			const FLOAT_TYPE vrange_min,
			const FLOAT_TYPE vrange_max
			) : 
			derivMin0(derivMin0),
			derivMax0(derivMax0),
			derivMin1(derivMin1),
			derivMax1(derivMax1), 
			dL0(dL0),
			dU0(dU0),
			vrange_min(vrange_min),
			vrange_max(vrange_max) {}
		__host__ __device__
		FLOAT_TYPE operator()(const FLOAT_TYPE y2) const
		{
			FLOAT_TYPE cos_y2;
			FLOAT_TYPE sin_y2;
			sincos_float_type<FLOAT_TYPE>(y2, sin_y2, cos_y2);
			return getAlpha(cos_y2, sin_y2, cos_y2, derivMin0, derivMax0, derivMin1, derivMax1, dL0, dU0, vrange_min, vrange_max);
		}
	};
	struct PartialFunction_dim1 {
	public:
		const FLOAT_TYPE derivMin0;
		const FLOAT_TYPE derivMax0;
		const FLOAT_TYPE derivMin1;
		const FLOAT_TYPE derivMax1;
		const FLOAT_TYPE dL1;
		const FLOAT_TYPE dU1;
		const FLOAT_TYPE vrange_min;
		const FLOAT_TYPE vrange_max;
		PartialFunction_dim1(
			const FLOAT_TYPE derivMin0,
			const FLOAT_TYPE derivMax0,
			const FLOAT_TYPE derivMin1,
			const FLOAT_TYPE derivMax1,
			const FLOAT_TYPE dL1,
			const FLOAT_TYPE dU1,
			const FLOAT_TYPE vrange_min,
			const FLOAT_TYPE vrange_max
			) : 
			derivMin0(derivMin0),
			derivMax0(derivMax0),
			derivMin1(derivMin1),
			derivMax1(derivMax1), 
			dL1(dL1),
			dU1(dU1),
			vrange_min(vrange_min),
			vrange_max(vrange_max) {}
		__host__ __device__
		FLOAT_TYPE operator()(const FLOAT_TYPE y2) const
		{
			FLOAT_TYPE cos_y2;
			FLOAT_TYPE sin_y2;
			sincos_float_type<FLOAT_TYPE>(y2, sin_y2, cos_y2);
			return getAlpha(cos_y2, sin_y2, sin_y2, derivMin0, derivMax0, derivMin1, derivMax1, dL1, dU1, vrange_min, vrange_max);
		}
	};
	bool PartialFunction_cuda(
		beacls::UVec& alpha_uvec,
		const std::vector<beacls::UVec>& x_uvecs,
		const std::vector<beacls::UVec>& derivMin_uvecs,
		const std::vector<beacls::UVec>& derivMax_uvecs,
		const size_t dim,
		const FLOAT_TYPE wMax,
		const FLOAT_TYPE vrange_min,
		const FLOAT_TYPE vrange_max,
		const beacls::FloatVec& dMax,
		const helperOC::DynSys_UMode_Type uMode,
		const helperOC::DynSys_DMode_Type dMode
	) {
		if ((uMode != helperOC::DynSys_UMode_Max) && (uMode != helperOC::DynSys_UMode_Min)) return false;
		if ((dMode != helperOC::DynSys_DMode_Max) && (dMode != helperOC::DynSys_DMode_Min)) return false;
		const FLOAT_TYPE moded_vrange_max = (uMode == helperOC::DynSys_UMode_Max) ? vrange_max : vrange_min;
		const FLOAT_TYPE moded_vrange_min = (uMode == helperOC::DynSys_UMode_Max) ? vrange_min : vrange_max;
		const FLOAT_TYPE moded_wMax = (uMode == helperOC::DynSys_UMode_Max) ? wMax : -wMax;
		const FLOAT_TYPE dMax_0 = dMax[0];
		const FLOAT_TYPE dMax_1 = dMax[1];
		const FLOAT_TYPE moded_dMax_0 = (dMode == helperOC::DynSys_DMode_Max) ? dMax_0 : -dMax_0;
		const FLOAT_TYPE moded_dMax_1 = (dMode == helperOC::DynSys_DMode_Max) ? dMax_1 : -dMax_1;
		const size_t src_x_dim_index = 2;

		const FLOAT_TYPE* y2_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[src_x_dim_index]).ptr();
		const FLOAT_TYPE* derivMax0_ptr = beacls::UVec_<FLOAT_TYPE>(derivMax_uvecs[0]).ptr();
		const FLOAT_TYPE* derivMax1_ptr = beacls::UVec_<FLOAT_TYPE>(derivMax_uvecs[1]).ptr();
		const FLOAT_TYPE* derivMax2_ptr = beacls::UVec_<FLOAT_TYPE>(derivMax_uvecs[2]).ptr();
		const FLOAT_TYPE* derivMin0_ptr = beacls::UVec_<FLOAT_TYPE>(derivMin_uvecs[0]).ptr();
		const FLOAT_TYPE* derivMin1_ptr = beacls::UVec_<FLOAT_TYPE>(derivMin_uvecs[1]).ptr();
		const FLOAT_TYPE* derivMin2_ptr = beacls::UVec_<FLOAT_TYPE>(derivMin_uvecs[2]).ptr();

		bool result = true;
		switch (dim) {
		case 0:
		{
			beacls::reallocateAsSrc(alpha_uvec, x_uvecs[src_x_dim_index]);
			cudaStream_t alpha_stream = beacls::get_stream(alpha_uvec);
			FLOAT_TYPE* alpha_ptr = beacls::UVec_<FLOAT_TYPE>(alpha_uvec).ptr();
			const FLOAT_TYPE derivMax0 = derivMax0_ptr[0];
			const FLOAT_TYPE derivMax1 = derivMax1_ptr[0];
			const FLOAT_TYPE derivMin0 = derivMin0_ptr[0];
			const FLOAT_TYPE derivMin1 = derivMin1_ptr[0];
			const FLOAT_TYPE normDerivMax = sqrt_float_type(derivMax0 * derivMax0 + derivMax1 * derivMax1);
			const FLOAT_TYPE normDerivMin = sqrt_float_type(derivMin0 * derivMin0 + derivMin1 * derivMin1);
			const FLOAT_TYPE dU0 = (normDerivMax == 0) ? 0 : moded_dMax_0 * derivMax0 / normDerivMax;
			const FLOAT_TYPE dL0 = (normDerivMin == 0) ? 0 : moded_dMax_0 * derivMin0 / normDerivMin;
			thrust::device_ptr<FLOAT_TYPE> alpha_dev_ptr = thrust::device_pointer_cast(alpha_ptr);
			thrust::device_ptr<const FLOAT_TYPE> y2_dev_ptr = thrust::device_pointer_cast(y2_ptr);

			thrust::transform(thrust::cuda::par.on(alpha_stream),
				y2_dev_ptr, y2_dev_ptr + x_uvecs[src_x_dim_index].size(), alpha_dev_ptr,
				PartialFunction_dim0(derivMin0, derivMax0, derivMin1, derivMax1, dL0, dU0, moded_vrange_min, moded_vrange_max));
		}
			break;
		case 1:
		{
			beacls::reallocateAsSrc(alpha_uvec, x_uvecs[src_x_dim_index]);
			cudaStream_t alpha_stream = beacls::get_stream(alpha_uvec);
			FLOAT_TYPE* alpha_ptr = beacls::UVec_<FLOAT_TYPE>(alpha_uvec).ptr();
			const FLOAT_TYPE derivMax0 = derivMax0_ptr[0];
			const FLOAT_TYPE derivMax1 = derivMax1_ptr[0];
			const FLOAT_TYPE derivMin0 = derivMin0_ptr[0];
			const FLOAT_TYPE derivMin1 = derivMin1_ptr[0];
			const FLOAT_TYPE normDerivMax = sqrt_float_type(derivMax0 * derivMax0 + derivMax1 * derivMax1);
			const FLOAT_TYPE normDerivMin = sqrt_float_type(derivMin0 * derivMin0 + derivMin1 * derivMin1);
			const FLOAT_TYPE dU1 = (normDerivMax == 0) ? 0 : moded_dMax_0 * derivMax1 / normDerivMax;
			const FLOAT_TYPE dL1 = (normDerivMin == 0) ? 0 : moded_dMax_0 * derivMin1 / normDerivMin;
			thrust::device_ptr<FLOAT_TYPE> alpha_dev_ptr = thrust::device_pointer_cast(alpha_ptr);
			thrust::device_ptr<const FLOAT_TYPE> y2_dev_ptr = thrust::device_pointer_cast(y2_ptr);

			thrust::transform(thrust::cuda::par.on(alpha_stream),
				y2_dev_ptr, y2_dev_ptr + x_uvecs[src_x_dim_index].size(), alpha_dev_ptr,
				PartialFunction_dim1(derivMin0, derivMax0, derivMin1, derivMax1, dL1, dU1, moded_vrange_min, moded_vrange_max));
		}
			break;
		case 2:
		{
			beacls::reallocateAsSrc(alpha_uvec, derivMax_uvecs[0]);
			FLOAT_TYPE* alpha_ptr = beacls::UVec_<FLOAT_TYPE>(alpha_uvec).ptr();
			const FLOAT_TYPE derivMax2 = derivMax2_ptr[0];
			const FLOAT_TYPE derivMin2 = derivMin2_ptr[0];
			const FLOAT_TYPE uU1 = (derivMax2 >= 0) ? moded_wMax : -moded_wMax;
			const FLOAT_TYPE dU2 = (derivMax2 >= 0) ? moded_dMax_1 : -moded_dMax_1;
			const FLOAT_TYPE uL1 = (derivMin2 >= 0) ? moded_wMax : -moded_wMax;
			const FLOAT_TYPE dL2 = (derivMin2 >= 0) ? moded_dMax_1 : -moded_dMax_1;
			const FLOAT_TYPE dxUU = uU1 + dU2;
			const FLOAT_TYPE dxUL = uU1 + dL2;
			const FLOAT_TYPE dxLU = uL1 + dU2;
			const FLOAT_TYPE dxLL = uL1 + dL2;
			const FLOAT_TYPE max0 = max_float_type<FLOAT_TYPE>(abs_float_type<FLOAT_TYPE>(dxUU), abs_float_type<FLOAT_TYPE>(dxUL));
			const FLOAT_TYPE max1 = max_float_type<FLOAT_TYPE>(abs_float_type<FLOAT_TYPE>(dxLL), abs_float_type<FLOAT_TYPE>(dxLU));
			alpha_ptr[0] = max_float_type<FLOAT_TYPE>(max0, max1);
		}
		break;
		default:
			std::cerr << "Only dimension 1-4 are defined for dynamics of Plane!" << std::endl;
			result = false;
			break;
		}

		return result;
	}
};
#endif /* defined(USER_DEFINED_GPU_DYNSYS_FUNC) */
#endif /* defined(WITH_GPU) */
