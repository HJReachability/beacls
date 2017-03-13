#ifndef __UpwindFirstENO3bHelper_impl_hpp__
#define __UpwindFirstENO3bHelper_impl_hpp__

#include <cstdint>
#include <vector>
#include <Core/UVec.hpp>
#include <typedef.hpp>
class HJI_Grid;

class UpwindFirstENO3bHelper_Cache {
public:
	std::vector<beacls::FloatVec > cachedBoundedSrcs;
	std::vector<beacls::FloatVec > last_d1ss;
	beacls::FloatVec last_d2s;
	beacls::FloatVec last_d2s_fabs;
	beacls::FloatVec last_dx_d2_effs;
	std::vector<const FLOAT_TYPE*> boundedSrc_ptrs;
	UpwindFirstENO3bHelper_Cache() : cachedBoundedSrcs(2), boundedSrc_ptrs(2) {}
};

class UpwindFirstENO3bHelper_impl {
private:
	beacls::UVecType type;
	beacls::FloatVec dxs;
	beacls::FloatVec dxInvs;
	beacls::FloatVec dxInv_2s;


	beacls::IntegerVec outer_dimensions_loop_sizes;
	beacls::IntegerVec target_dimension_loop_sizes;
	beacls::IntegerVec inner_dimensions_loop_sizes;
	beacls::IntegerVec first_dimension_loop_sizes;
	beacls::IntegerVec src_target_dimension_loop_sizes;
	beacls::FloatVec bounded_first_dimension_line_cache;

	const size_t stencil;
	std::vector<beacls::FloatVec > tmpBoundedSrcs;
	std::vector<const FLOAT_TYPE*> tmpBoundedSrc_ptrs;

	UpwindFirstENO3bHelper_Cache *cache;
public:
	UpwindFirstENO3bHelper_impl(
		const HJI_Grid *hji_grid,
		const beacls::UVecType type = beacls::UVecType_Vector
		);
	~UpwindFirstENO3bHelper_impl();
	bool execute_dim0(
		beacls::UVec &dst_deriv,
		beacls::UVec &dst_smooth,
		beacls::UVec &dst_epsilon,
		const FLOAT_TYPE* src,
		const size_t dim,
		const bool direction,
		const size_t loop_begin,
		const size_t slice_length,
		const size_t num_of_slices,
		beacls::CudaStream* cudaStream
	);
	bool execute_dim1(
		beacls::UVec &dst_deriv,
		beacls::UVec &dst_smooth,
		beacls::UVec &dst_epsilon,
		const FLOAT_TYPE* src,
		const size_t dim,
		const bool direction,
		const size_t loop_begin,
		const size_t slice_length,
		const size_t num_of_slices,
		beacls::CudaStream* cudaStream
	);
	bool execute_dimLET2(
		beacls::UVec &dst_deriv,
		beacls::UVec &dst_smooth,
		beacls::UVec &dst_epsilon,
		const FLOAT_TYPE* src,
		const size_t dim,
		const bool direction,
		const size_t loop_begin,
		const size_t slice_length,
		const size_t num_of_slices,
		beacls::CudaStream* cudaStream
	);
	bool execute(
		beacls::UVec &dst_deriv,
		beacls::UVec &dst_smooth,
		beacls::UVec &dst_epsilon,
		const HJI_Grid *grid,
		const FLOAT_TYPE* src,
		const size_t dim,
		const bool direction,
		const size_t loop_begin,
		const size_t slice_length,
		const size_t num_of_slices,
		beacls::CudaStream* cudaStream
	);
	bool operator==(const UpwindFirstENO3bHelper_impl& rhs) const;
	UpwindFirstENO3bHelper_impl* clone() const {
		return new UpwindFirstENO3bHelper_impl(*this);
	};
private:
	/** @overload
	Disable operator=
	*/
	UpwindFirstENO3bHelper_impl& operator=(const UpwindFirstENO3bHelper_impl& rhs);
	/** @overload
	Disable copy constructor
	*/
	UpwindFirstENO3bHelper_impl(const UpwindFirstENO3bHelper_impl& rhs) : 
		type(rhs.type),
		dxs(rhs.dxs),
		dxInvs(rhs.dxInvs),
		dxInv_2s(rhs.dxInv_2s),
	
	
		outer_dimensions_loop_sizes(rhs.outer_dimensions_loop_sizes),
		target_dimension_loop_sizes(rhs.target_dimension_loop_sizes),
		inner_dimensions_loop_sizes(rhs.inner_dimensions_loop_sizes),
		first_dimension_loop_sizes(rhs.first_dimension_loop_sizes),
		src_target_dimension_loop_sizes(rhs.src_target_dimension_loop_sizes),
		bounded_first_dimension_line_cache(rhs.bounded_first_dimension_line_cache),

		stencil(rhs.stencil),
		tmpBoundedSrcs(rhs.tmpBoundedSrcs),
		tmpBoundedSrc_ptrs(rhs.tmpBoundedSrc_ptrs),

		cache(new UpwindFirstENO3bHelper_Cache(*rhs.cache))
	{}
};

#endif	/* __UpwindFirstENO3bHelper_impl_hpp__ */

