#ifndef __UpwindFirstENO3_impl_hpp__
#define __UpwindFirstENO3_impl_hpp__

#include <cstdint>
#include <vector>
#include <set> 
#include <typedef.hpp>
namespace levelset {
	class HJI_Grid;
	class UpwindFirstENO3a;
	class UpwindFirstENO3_impl {
	private:
		beacls::UVecType type;
		UpwindFirstENO3a *upwindFirstENO3a;
	public:
		UpwindFirstENO3_impl(
			const HJI_Grid *hji_grid,
			const beacls::UVecType type = beacls::UVecType_Vector
		);
		~UpwindFirstENO3_impl();
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
		bool operator==(const UpwindFirstENO3_impl& rhs) const;
		UpwindFirstENO3_impl* clone() const {
			return new UpwindFirstENO3_impl(*this);
		};
		beacls::UVecType get_type() const {
			return type;
		};
	private:
		/** @overload
		Disable operator=
		*/
		UpwindFirstENO3_impl& operator=(const UpwindFirstENO3_impl& rhs);
		/** @overload
		Disable copy constructor
		*/
		UpwindFirstENO3_impl(const UpwindFirstENO3_impl& rhs);
	};
};
#endif	/* __UpwindFirstENO3_impl_hpp__ */

