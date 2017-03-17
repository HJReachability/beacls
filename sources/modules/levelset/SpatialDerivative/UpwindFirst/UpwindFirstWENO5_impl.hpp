#ifndef __UpwindFirstWENO5_impl_hpp__
#define __UpwindFirstWENO5_impl_hpp__

#include <cstdint>
#include <vector>
#include <typedef.hpp>
#include <Core/UVec.hpp>
namespace levelset {

	class HJI_Grid;
	class UpwindFirstWENO5a;

	class UpwindFirstWENO5_impl {
	private:
		beacls::UVecType type;
		UpwindFirstWENO5a *upwindFirstWENO5a;
	public:
		UpwindFirstWENO5_impl(
			const HJI_Grid *hji_grid,
			const beacls::UVecType type = beacls::UVecType_Vector
		);
		~UpwindFirstWENO5_impl();

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
		bool operator==(const UpwindFirstWENO5_impl& rhs) const;
		UpwindFirstWENO5_impl* clone() const {
			return new UpwindFirstWENO5_impl(*this);
		};
		beacls::UVecType get_type() const {
			return type;
		};
	private:
		/** @overload
		Disable operator=
		*/
		UpwindFirstWENO5_impl& operator=(const UpwindFirstWENO5_impl& rhs);
		/** @overload
		Disable copy constructor
		*/
		UpwindFirstWENO5_impl(const UpwindFirstWENO5_impl& rhs);
	};
};
#endif	/* __UpwindFirstWENO5_impl_hpp__ */

