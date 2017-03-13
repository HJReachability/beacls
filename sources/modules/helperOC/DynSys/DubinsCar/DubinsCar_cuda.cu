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
#include "DubinsCar_cuda.hpp"
#include <vector>
#include <algorithm>
#include <helperOC/DynSys/DynSys/DynSys.hpp>
#include <Core/UVec.hpp>
#include <Core/CudaStream.hpp>

#if defined(WITH_GPU) 
#if defined(USER_DEFINED_GPU_DYNSYS_FUNC)
namespace DubinsCar_CUDA {
template<typename T>
struct Get_opt : public thrust::unary_function<const T, T> {
public:
	T a_;
	Get_opt(const T a) : a_(a) {}
	__host__ __device__
	T operator()(const T& v) const
	{
		return (v >= 0) ? a_ : -a_;
	}
};

bool optCtrl_execute_cuda(
	beacls::UVec& u_uvec,
	const beacls::UVec& deriv_uvec,
	const FLOAT_TYPE wMax,
	const DynSys_UMode_Type uMode
	)
{
	bool result = true;
	const size_t length = deriv_uvec.size();
	beacls::reallocateAsSrc(u_uvec, deriv_uvec);
	cudaStream_t u_stream = beacls::get_stream(u_uvec);

	FLOAT_TYPE* uOpt_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvec).ptr();
	const FLOAT_TYPE* deriv_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvec).ptr();;

	if (beacls::is_cuda(deriv_uvec)) {
		thrust::device_ptr<FLOAT_TYPE> uOpt_dev_ptr = thrust::device_pointer_cast(uOpt_ptr);
		thrust::device_ptr<const FLOAT_TYPE> deriv_dev_ptr = thrust::device_pointer_cast(deriv_ptr);
		switch (uMode) {
		case DynSys_UMode_Max:
			thrust::transform(thrust::cuda::par.on(u_stream), 
				deriv_dev_ptr, deriv_dev_ptr + length, uOpt_dev_ptr, 
				Get_opt<FLOAT_TYPE>(wMax));
			break;
		case DynSys_UMode_Min:
			thrust::transform(thrust::cuda::par.on(u_stream), 
				deriv_dev_ptr, deriv_dev_ptr + length, uOpt_dev_ptr, 
				Get_opt<FLOAT_TYPE>(-wMax));
			break;
		case DynSys_UMode_Invalid:
		default:
			printf("Unknown uMode!: %d\n", uMode);
			result = false;
			break;
		}
	} else {
		const FLOAT_TYPE d = deriv_ptr[0];
		switch (uMode) {
		case DynSys_UMode_Max:
			uOpt_ptr[0] = Get_opt<FLOAT_TYPE>(wMax)(d);
			break;
		case DynSys_UMode_Min:
			uOpt_ptr[0] = Get_opt<FLOAT_TYPE>(-wMax)(d);
			break;
		case DynSys_UMode_Invalid:
		default:
			printf("Unknown uMode!: %d\n", uMode);
			result = false;
			break;
		}
	}
	return result;
}

bool optDstb_execute_cuda(
	std::vector<beacls::UVec>& d_uvecs,
	const beacls::UVec& deriv_uvec,
	const std::vector<FLOAT_TYPE>& dMax,
	const DynSys_DMode_Type dMode,
	const std::vector<size_t>& dims
)
{
	bool result = true;
	const FLOAT_TYPE* deriv_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_uvec).ptr();
	if (beacls::is_cuda(deriv_uvec)) {
		thrust::device_ptr<const FLOAT_TYPE> deriv_dev_ptr = thrust::device_pointer_cast(deriv_ptr);
		for (size_t dim = 0; dim < 3; ++dim) {
			if (std::find(dims.cbegin(),dims.cend(),dim) != dims.cend()) {
				beacls::reallocateAsSrc(d_uvecs[dim], deriv_uvec);
				cudaStream_t d_stream = beacls::get_stream(d_uvecs[dim]);
				const FLOAT_TYPE dMax_d = dMax[dim];
				FLOAT_TYPE* dOpt_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[dim]).ptr();
				thrust::device_ptr<FLOAT_TYPE> dOpt_dev_ptr = thrust::device_pointer_cast(dOpt_ptr);
				switch (dMode) {
				case DynSys_DMode_Max:
					thrust::transform(thrust::cuda::par.on(d_stream),
						deriv_dev_ptr, deriv_dev_ptr + deriv_uvec.size(), dOpt_dev_ptr,
						Get_opt<FLOAT_TYPE>(dMax_d));
					break;
				case DynSys_DMode_Min:
					thrust::transform(thrust::cuda::par.on(d_stream),
						deriv_dev_ptr, deriv_dev_ptr + deriv_uvec.size(), dOpt_dev_ptr,
						Get_opt<FLOAT_TYPE>(-dMax_d));
					break;
				case DynSys_UMode_Invalid:
				default:
					printf("Unknown dMode!: %d\n", dMode);
					result = false;
				}
			}
		}
	} else {
		FLOAT_TYPE d = deriv_ptr[0];
		for (size_t dim = 0; dim < 3; ++dim) {
			if (std::find(dims.cbegin(),dims.cend(),dim) != dims.cend()) {
				beacls::reallocateAsSrc(d_uvecs[dim], deriv_uvec);
				const FLOAT_TYPE dMax_d = dMax[dim];
				FLOAT_TYPE* dOpt_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[dim]).ptr();
				switch (dMode) {
				case DynSys_DMode_Max:
					dOpt_ptr[0] =  Get_opt<FLOAT_TYPE>(dMax_d)(d);
					break;
				case DynSys_DMode_Min:
					dOpt_ptr[0] =  Get_opt<FLOAT_TYPE>(-dMax_d)(d);
					break;
				case DynSys_UMode_Invalid:
				default:
					printf("Unknown dMode!: %d\n", dMode);
					result = false;
				}
			}
		}
	}
	return result;
}


template<typename T>
struct Get_dynamics_dim0_d0 : public thrust::unary_function<const T, T> {
public:
	T speed;
	T d0;
	Get_dynamics_dim0_d0(const T speed, T d0) : speed(speed), d0(d0) {}
	__host__ __device__
	T operator()(const T x) const
	{
		return speed*cos_float_type<T>(x)+d0;
	}
};
template<typename T>
struct Get_dynamics_dim0_D0 : public thrust::binary_function<const T, const T, T> {
public:
	T speed;
	Get_dynamics_dim0_D0(const T speed) : speed(speed) {}
	__host__ __device__
	T operator()(const T x, const T d0) const
	{
		return speed*cos_float_type<T>(x)+d0;
	}
};
template<typename T>
struct Get_dynamics_dim1_d1 : public thrust::unary_function<const T, T> {
public:
	T speed;
	T d1;
	Get_dynamics_dim1_d1(const T speed, const T d1) : speed(speed), d1(d1) {}
	__host__ __device__
	T operator()(const T v) const
	{
		return speed*sin_float_type<T>(v)+d1;
	}
};
template<typename T>
struct Get_dynamics_dim1_D1 : public thrust::binary_function<const T, const T, T> {
public:
	T speed;
	Get_dynamics_dim1_D1(const T speed) : speed(speed) {}
	__host__ __device__
	T operator()(const T x, const T d1) const
	{
		return speed*sin_float_type<T>(x)+d1;
	}
};

struct Get_dynamics_dim0dim1_D0_D1
{
	FLOAT_TYPE speed;
	Get_dynamics_dim0dim1_D0_D1(const FLOAT_TYPE speed) : speed(speed) {}
	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple v) const
	{
		const FLOAT_TYPE x = thrust::get<2>(v);
		const FLOAT_TYPE d0 = thrust::get<3>(v);
		const FLOAT_TYPE d1 = thrust::get<4>(v);
		FLOAT_TYPE sin_x;
		FLOAT_TYPE cos_x;
		sincos_float_type<FLOAT_TYPE>(x, sin_x, cos_x);
		thrust::get<0>(v) = speed*cos_x + d0;
		thrust::get<1>(v) = speed*sin_x + d1;
	}
};
struct Get_dynamics_dim0dim1_d0_d1
{
	FLOAT_TYPE speed;
	FLOAT_TYPE d0;
	FLOAT_TYPE d1;
	Get_dynamics_dim0dim1_d0_d1(const FLOAT_TYPE speed, const FLOAT_TYPE d0, const FLOAT_TYPE d1) : speed(speed), d0(d0), d1(d1) {}
	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple v) const
	{
		const FLOAT_TYPE x = thrust::get<2>(v);
		FLOAT_TYPE sin_x;
		FLOAT_TYPE cos_x;
		sincos_float_type<FLOAT_TYPE>(x, sin_x, cos_x);
		thrust::get<0>(v) = speed*cos_x + d0;
		thrust::get<1>(v) = speed*sin_x + d1;
	}
};
bool dynamics_cell_helper_execute_cuda_dimAll(
	std::vector<beacls::UVec>& dx_uvecs,
	const std::vector<beacls::UVec>& x_uvecs,
	const std::vector<beacls::UVec>& d_uvecs,
	const FLOAT_TYPE speed,
	const size_t src_dim0_dst_dx_index,
	const size_t src_dim1_dst_dx_index,
	const size_t src_x_dim
	) {
	bool result = true;
	const size_t length = x_uvecs[src_x_dim].size();
	beacls::reallocateAsSrc(dx_uvecs[src_dim0_dst_dx_index], x_uvecs[src_x_dim]);
	beacls::reallocateAsSrc(dx_uvecs[src_dim1_dst_dx_index], x_uvecs[src_x_dim]);
	FLOAT_TYPE* dx_dim0_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[src_dim0_dst_dx_index]).ptr();
	FLOAT_TYPE* dx_dim1_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[src_dim1_dst_dx_index]).ptr();
	const FLOAT_TYPE* ds_0_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[src_dim0_dst_dx_index]).ptr();
	const FLOAT_TYPE* ds_1_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[src_dim1_dst_dx_index]).ptr();
	const FLOAT_TYPE* x_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[src_x_dim]).ptr();
	
	thrust::device_ptr<FLOAT_TYPE> dx_dim0_dev_ptr = thrust::device_pointer_cast(dx_dim0_ptr);
	thrust::device_ptr<FLOAT_TYPE> dx_dim1_dev_ptr = thrust::device_pointer_cast(dx_dim1_ptr);
	cudaStream_t dx_stream = beacls::get_stream(dx_uvecs[0]);
	dx_uvecs[1].set_cudaStream(dx_uvecs[0].get_cudaStream());
	thrust::device_ptr<const FLOAT_TYPE> x_dev_ptr = thrust::device_pointer_cast(x_ptr);
	if (beacls::is_cuda(d_uvecs[0]) && beacls::is_cuda(d_uvecs[1])) {
		thrust::device_ptr<const FLOAT_TYPE> ds_0_dev_ptr = thrust::device_pointer_cast(ds_0_ptr);
		thrust::device_ptr<const FLOAT_TYPE> ds_1_dev_ptr = thrust::device_pointer_cast(ds_1_ptr);
		beacls::synchronizeUVec(d_uvecs[0]);
		beacls::synchronizeUVec(d_uvecs[1]);
		auto dst_src_Tuple = thrust::make_tuple(dx_dim0_dev_ptr, dx_dim1_dev_ptr, x_dev_ptr, ds_0_dev_ptr, ds_1_dev_ptr);
		auto dst_src_Iterator = thrust::make_zip_iterator(dst_src_Tuple);
		thrust::for_each(thrust::cuda::par.on(dx_stream),
			dst_src_Iterator, dst_src_Iterator + length, Get_dynamics_dim0dim1_D0_D1(speed));
	}
	else if (!beacls::is_cuda(d_uvecs[0]) && !beacls::is_cuda(d_uvecs[1])) {
		const FLOAT_TYPE d0 = ds_0_ptr[0];
		const FLOAT_TYPE d1 = ds_1_ptr[0];
		auto dst_src_Tuple = thrust::make_tuple(dx_dim0_dev_ptr, dx_dim1_dev_ptr, x_dev_ptr);
		auto dst_src_Iterator = thrust::make_zip_iterator(dst_src_Tuple);
		thrust::for_each(thrust::cuda::par.on(dx_stream),
			dst_src_Iterator, dst_src_Iterator + length, Get_dynamics_dim0dim1_d0_d1(speed, d0, d1));
	} else {
		std::cerr << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << " Invalid data size" << std::endl;
		result = false;
	}
	return result;
}


bool dynamics_cell_helper_execute_cuda(
	std::vector<beacls::UVec>& dx_uvecs,
	const std::vector<beacls::UVec>& x_uvecs,
	const std::vector<beacls::UVec>& u_uvecs,
	const std::vector<beacls::UVec>& d_uvecs,
	const FLOAT_TYPE speed,
	const size_t dst_target_dim,
	const size_t src_target_dim,
	const size_t src_x_dim
) {
	bool result = true;
	switch (src_target_dim) {
	case 0:
	{
		beacls::reallocateAsSrc(dx_uvecs[dst_target_dim], x_uvecs[src_x_dim]);
		FLOAT_TYPE* dx_dim_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[dst_target_dim]).ptr();
		cudaStream_t dx_stream = beacls::get_stream(dx_uvecs[dst_target_dim]);
		const FLOAT_TYPE* x_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[src_x_dim]).ptr();
		const FLOAT_TYPE* ds_0_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[0]).ptr();
		thrust::device_ptr<FLOAT_TYPE> dx_dim_dev_ptr = thrust::device_pointer_cast(dx_dim_ptr);
		thrust::device_ptr<const FLOAT_TYPE> x_dev_ptr = thrust::device_pointer_cast(x_ptr);
		if (beacls::is_cuda(d_uvecs[0])) {
			thrust::device_ptr<const FLOAT_TYPE> ds_0_dev_ptr = thrust::device_pointer_cast(ds_0_ptr);
			beacls::synchronizeUVec(d_uvecs[0]);
			thrust::transform(x_dev_ptr, x_dev_ptr + x_uvecs[src_x_dim].size(), ds_0_dev_ptr, dx_dim_dev_ptr, 
				Get_dynamics_dim0_D0<FLOAT_TYPE>(speed));
		}
		else {
			const FLOAT_TYPE d0 = ds_0_ptr[0];
			thrust::transform(thrust::cuda::par.on(dx_stream),
				x_dev_ptr, x_dev_ptr + x_uvecs[src_x_dim].size(), dx_dim_dev_ptr, 
				Get_dynamics_dim0_d0<FLOAT_TYPE>(speed, d0));
		}
	}
	break;
	case 1:
	{
		beacls::reallocateAsSrc(dx_uvecs[dst_target_dim], x_uvecs[src_x_dim]);
		FLOAT_TYPE* dx_dim_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[dst_target_dim]).ptr();
		cudaStream_t dx_stream = beacls::get_stream(dx_uvecs[dst_target_dim]);
		const FLOAT_TYPE* x_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[src_x_dim]).ptr();
		const FLOAT_TYPE* ds_1_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[1]).ptr();
		thrust::device_ptr<FLOAT_TYPE> dx_dim_dev_ptr = thrust::device_pointer_cast(dx_dim_ptr);
		thrust::device_ptr<const FLOAT_TYPE> x_dev_ptr = thrust::device_pointer_cast(x_ptr);
		if (beacls::is_cuda(d_uvecs[1])) {
			thrust::device_ptr<const FLOAT_TYPE> ds_1_dev_ptr = thrust::device_pointer_cast(ds_1_ptr);
			beacls::synchronizeUVec(d_uvecs[1]);
			thrust::transform(thrust::cuda::par.on(dx_stream),
				x_dev_ptr, x_dev_ptr + x_uvecs[src_x_dim].size(), ds_1_dev_ptr, dx_dim_dev_ptr, 
				Get_dynamics_dim1_D1<FLOAT_TYPE>(speed));
			}
		else {
			const FLOAT_TYPE d1 = ds_1_ptr[0];
			thrust::transform(thrust::cuda::par.on(dx_stream),
				x_dev_ptr, x_dev_ptr + x_uvecs[src_x_dim].size(), dx_dim_dev_ptr, 
				Get_dynamics_dim1_d1<FLOAT_TYPE>(speed, d1));
		}
	}
	break;
	case 2:
	{
		beacls::reallocateAsSrc(dx_uvecs[dst_target_dim], u_uvecs[0]);
		FLOAT_TYPE* dx_dim_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvecs[dst_target_dim]).ptr();
		const FLOAT_TYPE* us_0_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[0]).ptr();
		const FLOAT_TYPE* ds_2_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[2]).ptr();
		if (beacls::is_cuda(u_uvecs[0]) && beacls::is_cuda(d_uvecs[2])) {
			thrust::device_ptr<FLOAT_TYPE> dx_dim_dev_ptr = thrust::device_pointer_cast(dx_dim_ptr);
			cudaStream_t dx_stream = beacls::get_stream(dx_uvecs[dst_target_dim]);
			thrust::device_ptr<const FLOAT_TYPE> us_0_dev_ptr = thrust::device_pointer_cast(us_0_ptr);
			thrust::device_ptr<const FLOAT_TYPE> ds_2_dev_ptr = thrust::device_pointer_cast(ds_2_ptr);
			beacls::synchronizeUVec(u_uvecs[0]);
			beacls::synchronizeUVec(d_uvecs[2]);
			thrust::transform(thrust::cuda::par.on(dx_stream),
				us_0_dev_ptr, us_0_dev_ptr + u_uvecs[0].size(), ds_2_dev_ptr, dx_dim_dev_ptr, 
				thrust::plus<FLOAT_TYPE>());
		} 
		else if (!beacls::is_cuda(u_uvecs[0]) && !beacls::is_cuda(d_uvecs[2])) {
			const FLOAT_TYPE u0 = us_0_ptr[0];
			const FLOAT_TYPE d2 = ds_2_ptr[0];
			dx_dim_ptr[0] = u0 + d2;
		} 
		else {	//!< us_0_size != length
			std::cerr << __FILE__ << ":" << __LINE__ << ":" << __FUNCTION__ << " Invalid data size" << std::endl;
			result = false;
		}
	}
	break;
	default:
		printf("Only dimension 1-4 are defined for dynamics of DubinsCar!\n");
		result = false;
		break;
	}
	return result;
}

};
#endif /* defined(USER_DEFINED_GPU_DYNSYS_FUNC) */
#endif /* defined(WITH_GPU) */
