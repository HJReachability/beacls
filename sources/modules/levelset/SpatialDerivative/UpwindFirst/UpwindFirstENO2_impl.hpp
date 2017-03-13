
#ifndef __UpwindFirstENO2_impl_hpp__
#define __UpwindFirstENO2_impl_hpp__

#include <cstdint>
#include <vector>
#include <algorithm>
#include <typedef.hpp>
#include <Core/CudaStream.hpp>
class HJI_Grid;
class BoundaryCondition;
class UpwindFirstENO2_Cache {
public:
	std::vector<beacls::FloatVec > last_d1ss;
	beacls::FloatVec last_d2s;
	beacls::FloatVec last_d2s_fabs;
	beacls::FloatVec last_dx_d2_effs;
	std::vector<beacls::UVec > cachedBoundedSrc_uvecs;
	std::vector<const FLOAT_TYPE*> boundedSrc_ptrs;
	UpwindFirstENO2_Cache() :
		cachedBoundedSrc_uvecs(2), 
		boundedSrc_ptrs(2)
	
	{}
	/** @overload
	Disable copy constructor
	*/
	UpwindFirstENO2_Cache(const UpwindFirstENO2_Cache& rhs) :
		last_d1ss(rhs.last_d1ss),
		last_d2s(rhs.last_d2s),
		last_d2s_fabs(rhs.last_d2s_fabs),
		last_dx_d2_effs(rhs.last_dx_d2_effs)
		{
		cachedBoundedSrc_uvecs.resize(rhs.cachedBoundedSrc_uvecs.size());
		boundedSrc_ptrs.resize(rhs.boundedSrc_ptrs.size(), NULL);
	}
private:
	/** @overload
	Disable operator=
	*/
	UpwindFirstENO2_Cache& operator=(const UpwindFirstENO2_Cache& rhs);

};

class UpwindFirstENO2_impl {
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
	beacls::UVec bounded_first_dimension_line_cache_uvec;

	const size_t stencil;
	std::vector<std::vector<std::vector<std::vector<beacls::UVec > > > > tmpBoundedSrc_uvecssss;
	std::vector<std::vector<std::vector<std::vector<const FLOAT_TYPE*> > > > tmpBoundedSrc_ptrssss;

	UpwindFirstENO2_Cache *cache;
	std::vector<beacls::CudaStream*> cudaStreams;
public:
	UpwindFirstENO2_impl(
		const HJI_Grid *hji_grid,
		const beacls::UVecType type = beacls::UVecType_Vector
		);
	~UpwindFirstENO2_impl();
	bool execute_dim0(
		beacls::UVec& dst_deriv_l,
		beacls::UVec& dst_deriv_r,
		const FLOAT_TYPE* src,
		const HJI_Grid *grid,
		const BoundaryCondition *boundaryCondition,
		const size_t dim,
		const bool generateAll,
		const size_t loop_begin,
		const size_t slice_length,
		const size_t num_of_slices
	);
	bool execute_dim1(
		beacls::UVec& dst_deriv_l,
		beacls::UVec& dst_deriv_r,
		const FLOAT_TYPE* src,
		const HJI_Grid *grid,
		const BoundaryCondition *boundaryCondition,
		const size_t dim,
		const bool generateAll,
		const size_t loop_begin,
		const size_t slice_length,
		const size_t num_of_slices
	);
	bool execute_dimLET2(
		beacls::UVec& dst_deriv_l,
		beacls::UVec& dst_deriv_r,
		const FLOAT_TYPE* src,
		const HJI_Grid *grid,
		const BoundaryCondition *boundaryCondition,
		const size_t dim,
		const bool generateAll,
		const size_t loop_begin,
		const size_t slice_length,
		const size_t num_of_slices
	);
	bool execute(
		beacls::UVec& dst_deriv_l,
		beacls::UVec& dst_deriv_r,
		const HJI_Grid *grid,
		const FLOAT_TYPE* src,
		const size_t dim,
		const bool generateAll,
		const size_t loop_begin,
		const size_t slice_length,
		const size_t num_of_slices
	);
	bool synchronize(const size_t dim);
	bool operator==(const UpwindFirstENO2_impl& rhs) const;
	UpwindFirstENO2_impl* clone() const {
		return new UpwindFirstENO2_impl(*this);
	};
	beacls::UVecType get_type() const {
		return type;
	};
private:
	/** @overload
	Disable operator=
	*/
	UpwindFirstENO2_impl& operator=(const UpwindFirstENO2_impl& rhs);
	/** @overload
	Disable copy constructor
	*/
	UpwindFirstENO2_impl(const UpwindFirstENO2_impl& rhs) : 
		type(rhs.type),
		dxs(rhs.dxs),
		dxInvs(rhs.dxInvs),
		dxInv_2s(rhs.dxInv_2s),
		outer_dimensions_loop_sizes(rhs.outer_dimensions_loop_sizes),
		target_dimension_loop_sizes(rhs.target_dimension_loop_sizes),
		inner_dimensions_loop_sizes(rhs.inner_dimensions_loop_sizes),
		first_dimension_loop_sizes(rhs.first_dimension_loop_sizes),
		src_target_dimension_loop_sizes(rhs.src_target_dimension_loop_sizes),

		stencil(rhs.stencil),
		tmpBoundedSrc_ptrssss(rhs.tmpBoundedSrc_ptrssss),

		cache(new UpwindFirstENO2_Cache(*rhs.cache))
	{
		tmpBoundedSrc_uvecssss.resize(rhs.tmpBoundedSrc_uvecssss.size());
		std::transform(rhs.tmpBoundedSrc_uvecssss.cbegin(), rhs.tmpBoundedSrc_uvecssss.cend(), tmpBoundedSrc_uvecssss.begin(), tmpBoundedSrc_uvecssss.begin(), [](const auto& lhs, auto& rhs) {
			rhs.resize(lhs.size());
			std::transform(lhs.cbegin(), lhs.cend(), rhs.begin(), rhs.begin(), [](const auto& lhs, auto& rhs) {
				rhs.resize(lhs.size());
				std::transform(lhs.cbegin(), lhs.cend(), rhs.begin(), rhs.begin(), [](const auto& lhs, auto& rhs) {
					rhs.resize(lhs.size());
					return rhs;
				});
				return rhs;
			});
			return rhs;
		});
		cudaStreams.resize(rhs.cudaStreams.size());
		if (type == beacls::UVecType_Cuda) {
			std::for_each(cudaStreams.begin(), cudaStreams.end(), [](auto& rhs) {
				rhs = new beacls::CudaStream();
			});
		}
	}
};

#endif	/* __UpwindFirstENO2_impl_hpp__ */

