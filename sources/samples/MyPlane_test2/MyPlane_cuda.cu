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
#include "MyPlane_cuda.hpp"
#include <vector>
#include <algorithm>
#include <helperOC/DynSys/DynSys/DynSys.hpp>
#include <Core/UVec.hpp>

#if defined(WITH_GPU) 
#if defined(USER_DEFINED_GPU_DYNSYS_FUNC)
namespace MyPlane_CUDA {

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
				auto dst_src_Tuple = thrust::make_tuple(uOpt0_dev_ptr, uOpt1_dev_ptr, 
					y2_dev_ptr, deriv0_dev_ptr, deriv1_dev_ptr, deriv2_dev_ptr);
				auto dst_src_Iterator = thrust::make_zip_iterator(dst_src_Tuple);
				thrust::for_each(
					dst_src_Iterator, dst_src_Iterator + x_uvecs[2].size(), 
					Get_optCtrl_D(moded_vrange_min, moded_vrange_max, moded_wMax));
			} else {
				thrust::device_ptr<FLOAT_TYPE> uOpt0_dev_ptr = thrust::device_pointer_cast(uOpt0_ptr);
				const FLOAT_TYPE d0 = deriv0_ptr[0];
				const FLOAT_TYPE d1 = deriv1_ptr[0];
				const FLOAT_TYPE d2 = deriv2_ptr[0];
				thrust::transform(
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
				auto dst_src_Tuple = thrust::make_tuple(
					dOpt0_dev_ptr, dOpt1_dev_ptr, dOpt2_dev_ptr,
					deriv0_dev_ptr, deriv1_dev_ptr, deriv2_dev_ptr);
				auto dst_src_Iterator = thrust::make_zip_iterator(dst_src_Tuple);

				thrust::for_each(
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
				beacls::reallocateAsSrc(dx_uvec, x_uvecs[src_x_dim_index]);
				FLOAT_TYPE* dx_dim_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvec).ptr();
				const FLOAT_TYPE* x_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[src_x_dim_index]).ptr();
				const FLOAT_TYPE* us_0_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[0]).ptr();
				const FLOAT_TYPE* ds_0_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[0]).ptr();

				thrust::device_ptr<FLOAT_TYPE> dx_dim_dev_ptr = thrust::device_pointer_cast(dx_dim_ptr);
				thrust::device_ptr<const FLOAT_TYPE> x_dev_ptr = thrust::device_pointer_cast(x_ptr);
				thrust::device_ptr<const FLOAT_TYPE> us_0_dev_ptr = thrust::device_pointer_cast(us_0_ptr);

				if (is_cuda(d_uvecs[0])){
					thrust::device_ptr<const FLOAT_TYPE> ds_0_dev_ptr = thrust::device_pointer_cast(ds_0_ptr);
					auto src0_Tuple = thrust::make_tuple(x_dev_ptr, us_0_dev_ptr, ds_0_dev_ptr);
					auto src0_Iterator = thrust::make_zip_iterator(src0_Tuple);
					thrust::transform(
						src0_Iterator, src0_Iterator + x_uvecs[src_x_dim_index].size(), dx_dim_dev_ptr, 
						Get_dynamics_dim0_U0_D0());
				}
				else {	//!< ds_0_size != length
					const FLOAT_TYPE d0 = ds_0_ptr[0];
					thrust::transform(
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

				beacls::reallocateAsSrc(dx_uvec, x_uvecs[src_x_dim_index]);
				FLOAT_TYPE* dx_dim_ptr = beacls::UVec_<FLOAT_TYPE>(dx_uvec).ptr();
				const FLOAT_TYPE* x_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvecs[src_x_dim_index]).ptr();
				const FLOAT_TYPE* us_0_ptr = beacls::UVec_<FLOAT_TYPE>(u_uvecs[0]).ptr();
				const FLOAT_TYPE* ds_1_ptr = beacls::UVec_<FLOAT_TYPE>(d_uvecs[1]).ptr();

				thrust::device_ptr<FLOAT_TYPE> dx_dim_dev_ptr = thrust::device_pointer_cast(dx_dim_ptr);
				thrust::device_ptr<const FLOAT_TYPE> x_dev_ptr = thrust::device_pointer_cast(x_ptr);
				thrust::device_ptr<const FLOAT_TYPE> us_0_dev_ptr = thrust::device_pointer_cast(us_0_ptr);

				if (is_cuda(d_uvecs[1])) {
					thrust::device_ptr<const FLOAT_TYPE> ds_1_dev_ptr = thrust::device_pointer_cast(ds_1_ptr);
					auto src0_Tuple = thrust::make_tuple(x_dev_ptr, us_0_dev_ptr, ds_1_dev_ptr);
					auto src0_Iterator = thrust::make_zip_iterator(src0_Tuple);
					thrust::transform(
						src0_Iterator, src0_Iterator + x_uvecs[src_x_dim_index].size(), dx_dim_dev_ptr, 
						Get_dynamics_dim1_U0_D1());
				}
				else {	//!< ds_1_size != length
					const FLOAT_TYPE d1 = ds_1_ptr[0];
					thrust::transform(
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
				thrust::device_ptr<FLOAT_TYPE> dx_dim_dev_ptr = thrust::device_pointer_cast(dx_dim_ptr);
				thrust::device_ptr<const FLOAT_TYPE> us_1_dev_ptr = thrust::device_pointer_cast(us_1_ptr);
				thrust::device_ptr<const FLOAT_TYPE> ds_2_dev_ptr = thrust::device_pointer_cast(ds_2_ptr);
				thrust::transform(
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
			std::cerr << "Only dimension 1-4 are defined for dynamics of MyPlane!" << std::endl;
			result = false;
			break;
		}
		return result;
	}
};
#endif /* defined(USER_DEFINED_GPU_DYNSYS_FUNC) */
#endif /* defined(WITH_GPU) */
