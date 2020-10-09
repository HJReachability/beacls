
#ifndef __UpwindFirstENO3a_impl_hpp__
#define __UpwindFirstENO3a_impl_hpp__

#include <cstdint>
#include <vector>
#include <Core/UVec.hpp>
#include <typedef.hpp>
#include <set> 
namespace levelset {

	class HJI_Grid;
	class UpwindFirstENO3aHelper;
	class UpwindFirstENO3a_Cache {
	public:
		std::vector<beacls::FloatVec > cachedBoundedSrcs;
		std::vector<beacls::FloatVec > last_d1ss;
		beacls::FloatVec last_d2s;
		beacls::FloatVec last_d2s_fabs;
		beacls::FloatVec last_dx_d2_effs;
		std::vector<const FLOAT_TYPE*> boundedSrc_ptrs;
		UpwindFirstENO3a_Cache() : cachedBoundedSrcs(2), boundedSrc_ptrs(2) {}
	};

	class UpwindFirstENO3a_impl {
	private:
		beacls::UVecType type;
		beacls::IntegerVec first_dimension_loop_sizes;
		beacls::IntegerVec inner_dimensions_loop_sizes;
		beacls::IntegerVec src_target_dimension_loop_sizes;

		const size_t stencil;
		size_t num_of_strides;

		std::vector<std::vector<beacls::UVec > > dL_uvecs;
		std::vector<std::vector<beacls::UVec > > dR_uvecs;
		std::vector<std::vector<beacls::UVec > > DD_uvecs;

		bool checkEquivalentApproximations;

		UpwindFirstENO3aHelper* upwindFirstENO3aHelper;
		std::vector<beacls::CudaStream*> cudaStreams;
	public:
		UpwindFirstENO3a_impl(
			const HJI_Grid *hji_grid,
			const beacls::UVecType type = beacls::UVecType_Vector
		);
		~UpwindFirstENO3a_impl();

		bool execute_dim0(
			beacls::UVec& dst_deriv_l,
			beacls::UVec& dst_deriv_r,
			const FLOAT_TYPE* src,
			const HJI_Grid *grid,
			const size_t dim,
			const size_t loop_begin,
			const size_t slice_length,
			const size_t num_of_slices
		);
		bool execute_dim1(
			beacls::UVec& dst_deriv_l,
			beacls::UVec& dst_deriv_r,
			const FLOAT_TYPE* src,
			const HJI_Grid *grid,
			const size_t dim,
			const size_t loop_begin,
			const size_t slice_length,
			const size_t num_of_slices
		);
		bool execute_dimLET2(
			beacls::UVec& dst_deriv_l,
			beacls::UVec& dst_deriv_r,
			const FLOAT_TYPE* src,
			const HJI_Grid *grid,
			const size_t dim,
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
		bool execute_dim0_local_q(
			beacls::UVec& dst_deriv_l,
			beacls::UVec& dst_deriv_r,
			const FLOAT_TYPE* src,
			const HJI_Grid *grid,
			const size_t dim,
			const size_t loop_begin,
			const size_t slice_length,
			const size_t num_of_slices, 
			const std::set<size_t> &Q
		);
		bool execute_dim1_local_q(
			beacls::UVec& dst_deriv_l,
			beacls::UVec& dst_deriv_r,
			const FLOAT_TYPE* src,
			const HJI_Grid *grid,
			const size_t dim,
			const size_t loop_begin,
			const size_t slice_length,
			const size_t num_of_slices, 
			const std::set<size_t> &Q
		);
		bool execute_dimLET2_local_q(
			beacls::UVec& dst_deriv_l,
			beacls::UVec& dst_deriv_r,
			const FLOAT_TYPE* src,
			const HJI_Grid *grid,
			const size_t dim,
			const size_t loop_begin,
			const size_t slice_length,
			const size_t num_of_slices, 
			const std::set<size_t> &Q	
		);
		bool execute_local_q(
			beacls::UVec& dst_deriv_l,
			beacls::UVec& dst_deriv_r,
			const HJI_Grid *grid,
			const FLOAT_TYPE* src,
			const size_t dim,
			const bool generateAll,
			const size_t loop_begin,
			const size_t slice_length,
			const size_t num_of_slices, 
			const std::set<size_t> &Q 
		);
		bool synchronize(const size_t dim);
		bool operator==(const UpwindFirstENO3a_impl& rhs) const;
		UpwindFirstENO3a_impl* clone() const {
			return new UpwindFirstENO3a_impl(*this);
		};
		beacls::UVecType get_type() const {
			return type;
		};
	private:
		/** @overload
		Disable operator=
		*/
		UpwindFirstENO3a_impl& operator=(const UpwindFirstENO3a_impl& rhs);
		/** @overload
		Disable copy constructor
		*/
		UpwindFirstENO3a_impl(const UpwindFirstENO3a_impl& rhs);
	};
};
#endif	/* __UpwindFirstENO3a_impl_hpp__ */

