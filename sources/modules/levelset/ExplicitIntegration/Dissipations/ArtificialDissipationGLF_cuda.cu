// CUDA runtime
#include <cuda_runtime.h>

#include <typedef.hpp>
#include <cuda_macro.hpp>
#include <thrust/for_each.h>
#include <thrust/device_ptr.h>
#include <thrust/tuple.h>
#include <thrust/pair.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/system/cuda/execution_policy.h>
#include <Core/UVec.hpp>
#include <Core/CudaStream.hpp>
#include "ArtificialDissipationGLF_cuda.hpp"
#if defined(WITH_GPU)

struct GetMinMax {
public:
	template<typename Tuple0, typename Tuple1>
	__host__ __device__
	Tuple0 operator()(const Tuple0 lhs, const Tuple1 rhs) const
	{
		const FLOAT_TYPE lhs_min = thrust::get<0>(lhs);
		const FLOAT_TYPE lhs_max = thrust::get<1>(lhs);
		const FLOAT_TYPE rhs_min = thrust::get<0>(rhs);
		const FLOAT_TYPE rhs_max = thrust::get<1>(rhs);
		const FLOAT_TYPE lr_min = min_float_type<FLOAT_TYPE>(lhs_min, rhs_min);
		const FLOAT_TYPE lr_max = max_float_type<FLOAT_TYPE>(lhs_max, rhs_max);
		return Tuple0(lr_min, lr_max);
	}
};

void CalculateRangeOfGradient_execute_cuda (
	FLOAT_TYPE& derivMin,
	FLOAT_TYPE& derivMax,
	const beacls::UVec& deriv_l_uvec,
	const beacls::UVec& deriv_r_uvec
) {
	const FLOAT_TYPE* deriv_l_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_l_uvec).ptr();
	const FLOAT_TYPE* deriv_r_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_r_uvec).ptr();
	thrust::device_ptr<const FLOAT_TYPE> deriv_l_dev_ptr = thrust::device_pointer_cast(deriv_l_ptr);
	thrust::device_ptr<const FLOAT_TYPE> deriv_r_dev_ptr = thrust::device_pointer_cast(deriv_r_ptr);
	beacls::synchronizeUVec(deriv_r_uvec);
	cudaStream_t deriv_stream = beacls::get_stream(deriv_l_uvec);
	const auto init_tuple = thrust::make_tuple(
		std::numeric_limits<FLOAT_TYPE>::max(),
		-std::numeric_limits<FLOAT_TYPE>::max());
	auto src_tuple = thrust::make_tuple(deriv_l_dev_ptr, deriv_r_dev_ptr);
	auto src_ite = thrust::make_zip_iterator(src_tuple);
	const size_t length = deriv_l_uvec.size();
	auto derivMinMax = thrust::reduce(thrust::cuda::par.on(deriv_stream),
		src_ite, src_ite + length, init_tuple, GetMinMax());
	derivMin = thrust::get<0>(derivMinMax);
	derivMax = thrust::get<1>(derivMinMax);
}

struct Get_Dissipation_dim0_A
{
	Get_Dissipation_dim0_A() {}
	template <typename Tuple>
	__host__ __device__
	FLOAT_TYPE operator()(Tuple v) const
	{
		const FLOAT_TYPE l = thrust::get<0>(v);
		const FLOAT_TYPE r = thrust::get<1>(v);
		const FLOAT_TYPE a = thrust::get<2>(v);
		return (r - l) * a / 2;
	}
};
struct Get_Dissipation_dimNot0_A
{
	Get_Dissipation_dimNot0_A() {}
	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple v) const
	{
		const FLOAT_TYPE l = thrust::get<0>(v);
		const FLOAT_TYPE r = thrust::get<1>(v);
		const FLOAT_TYPE a = thrust::get<2>(v);
		thrust::get<3>(v) += (r - l) * a / 2;
	}
};

struct Get_Dissipation_dim0_a
{
	const FLOAT_TYPE a;
	Get_Dissipation_dim0_a(const FLOAT_TYPE a) : a(a) {}
	template <typename Tuple>
	__host__ __device__
	FLOAT_TYPE operator()(Tuple v) const
	{
		const FLOAT_TYPE l = thrust::get<0>(v);
		const FLOAT_TYPE r = thrust::get<1>(v);
		return (r - l) * a / 2;
	}
};
struct Get_Dissipation_dimNot0_a
{
	const FLOAT_TYPE a;
	Get_Dissipation_dimNot0_a(const FLOAT_TYPE a) : a(a) {}
	template <typename Tuple>
	__host__ __device__
	void operator()(Tuple v) const
	{
		const FLOAT_TYPE l = thrust::get<0>(v);
		const FLOAT_TYPE r = thrust::get<1>(v);
		thrust::get<2>(v) += (r - l) * a / 2;
	}
};


void ArtificialDissipationGLF_execute_cuda (
	beacls::UVec& diss_uvec,
	FLOAT_TYPE& step_bound_inv,
	const beacls::UVec& alpha_uvec,
	const beacls::UVec& deriv_l_uvec,
	const beacls::UVec& deriv_r_uvec,
	const FLOAT_TYPE dxInv,
	const size_t dimension,
	const size_t loop_size
) {
	FLOAT_TYPE* diss_ptr = beacls::UVec_<FLOAT_TYPE>(diss_uvec).ptr();
	const FLOAT_TYPE* alpha_ptr = beacls::UVec_<FLOAT_TYPE>(alpha_uvec).ptr();
	const FLOAT_TYPE* deriv_l_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_l_uvec).ptr();
	const FLOAT_TYPE* deriv_r_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_r_uvec).ptr();
	const size_t alpha_size = alpha_uvec.size();

	thrust::device_ptr<FLOAT_TYPE> diss_dev_ptr = thrust::device_pointer_cast(diss_ptr);
	cudaStream_t diss_stream = beacls::get_stream(diss_uvec);
	thrust::device_ptr<const FLOAT_TYPE> deriv_l_dev_ptr = thrust::device_pointer_cast(deriv_l_ptr);
	thrust::device_ptr<const FLOAT_TYPE> deriv_r_dev_ptr = thrust::device_pointer_cast(deriv_r_ptr);
	beacls::synchronizeUVec(deriv_l_uvec);
	beacls::synchronizeUVec(deriv_r_uvec);
	//!< If partial function doesn't require global deriv min/max and it returns true,
	// calculate dissipations and step bound from alphas.
	if (beacls::is_cuda(alpha_uvec)) {
		beacls::synchronizeUVec(alpha_uvec);
		thrust::device_ptr<const FLOAT_TYPE> alpha_dev_ptr = thrust::device_pointer_cast(alpha_ptr);
		if (dimension == 0) {
			auto src_Tuple = thrust::make_tuple(deriv_l_dev_ptr, deriv_r_dev_ptr, alpha_dev_ptr);
			auto src_Ite = thrust::make_zip_iterator(src_Tuple);
			thrust::transform(thrust::cuda::par.on(diss_stream),
				src_Ite, src_Ite + loop_size, diss_dev_ptr, Get_Dissipation_dim0_A());
		} else {
			auto src_Tuple = thrust::make_tuple(deriv_l_dev_ptr, deriv_r_dev_ptr, alpha_dev_ptr, diss_dev_ptr);
			auto src_Ite = thrust::make_zip_iterator(src_Tuple);
			thrust::for_each(thrust::cuda::par.on(diss_stream),
				src_Ite, src_Ite + loop_size, Get_Dissipation_dimNot0_A());
		}
		const FLOAT_TYPE max_value = thrust::reduce(thrust::cuda::par.on(diss_stream),
			alpha_dev_ptr, alpha_dev_ptr + alpha_size,
			-std::numeric_limits<FLOAT_TYPE>::max(), thrust::maximum<FLOAT_TYPE>());
		step_bound_inv = max_value * dxInv;
	}
	else {
		const FLOAT_TYPE alpha = alpha_ptr[0];
		const FLOAT_TYPE max_value = alpha;
		step_bound_inv = max_value * dxInv;
		if (dimension == 0) {
			auto src_Tuple = thrust::make_tuple(deriv_l_dev_ptr, deriv_r_dev_ptr);
			auto src_Ite = thrust::make_zip_iterator(src_Tuple);
			thrust::transform(thrust::cuda::par.on(diss_stream),
				src_Ite, src_Ite + loop_size, diss_dev_ptr, Get_Dissipation_dim0_a(alpha));
		} else {
			auto src_Tuple = thrust::make_tuple(deriv_l_dev_ptr, deriv_r_dev_ptr, diss_dev_ptr);
			auto src_Ite = thrust::make_zip_iterator(src_Tuple);
			thrust::for_each(thrust::cuda::par.on(diss_stream),
				src_Ite, src_Ite + loop_size, Get_Dissipation_dimNot0_a(alpha));
		}
	}
	beacls::synchronizeUVec(diss_uvec);
}
#endif /* defined(WITH_GPU) */
