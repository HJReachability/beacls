#ifndef __UpwindFirstWENO5b_impl_hpp__
#define __UpwindFirstWENO5b_impl_hpp__

#include <cstdint>
#include <vector>
#include <typedef.hpp>
#include <Core/UVec.hpp>
#include <Core/CudaStream.hpp>
#include <algorithm>
namespace levelset {
	class HJI_Grid;
	class BoundaryCondition;
	class UpwindFirstWENO5b_Cache {
	public:
		std::vector<beacls::FloatVec > cachedBoundedSrcs;
		std::vector<beacls::FloatVec > last_d1ss;
		beacls::FloatVec last_d2s;
		beacls::FloatVec last_d2s_fabs;
		beacls::FloatVec last_dx_d2_effs;
		std::vector<const FLOAT_TYPE*> boundedSrc_ptrs;
		UpwindFirstWENO5b_Cache() : cachedBoundedSrcs(2), boundedSrc_ptrs(2) {}
	};

	class UpwindFirstWENO5b_impl {
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
		UpwindFirstWENO5b_Cache *cache;
		std::vector<beacls::CudaStream*> cudaStreams;
	public:
		UpwindFirstWENO5b_impl(
			const HJI_Grid *hji_grid,
			const beacls::UVecType type = beacls::UVecType_Vector
		);
		~UpwindFirstWENO5b_impl();
		bool execute_dim0(
			beacls::UVec& dst_deriv_l,
			beacls::UVec& dst_deriv_r,
			const FLOAT_TYPE* src,
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
		bool operator==(const UpwindFirstWENO5b_impl& rhs) const;
		UpwindFirstWENO5b_impl* clone() const {
			return new UpwindFirstWENO5b_impl(*this);
		};
		beacls::UVecType get_type() const {
			return type;
		};
	private:
		/** @overload
		Disable operator=
		*/
		UpwindFirstWENO5b_impl& operator=(const UpwindFirstWENO5b_impl& rhs);
		/** @overload
		Disable copy constructor
		*/
		UpwindFirstWENO5b_impl(const UpwindFirstWENO5b_impl& rhs) :
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

			cache(new UpwindFirstWENO5b_Cache(*rhs.cache))
		{
			if (type == beacls::UVecType_Cuda) {
				cudaStreams.resize(rhs.cudaStreams.size());
				std::for_each(cudaStreams.begin(), cudaStreams.end(), [](auto& rhs) {
					rhs = new beacls::CudaStream();
				});
			}
		}
	};
};
#endif	/* __UpwindFirstWENO5b_impl_hpp__ */

