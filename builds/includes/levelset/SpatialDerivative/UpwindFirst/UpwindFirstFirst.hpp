
#ifndef __UpwindFirstFirst_hpp__
#define __UpwindFirstFirst_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#include <typedef.hpp>
#include <levelset/SpatialDerivative/UpwindFirst/UpwindFirst.hpp>
#include <vector>
#include <cstdint>
#include <utility>
using namespace std::rel_ops;
namespace levelset {

	class HJI_Grid;

	class UpwindFirstFirst_impl;

	class UpwindFirstFirst : public UpwindFirst {
	public:
		PREFIX_VC_DLL
			UpwindFirstFirst(
				const HJI_Grid *hji_grid,
				const beacls::UVecType type = beacls::UVecType_Vector
			);
		~UpwindFirstFirst();
		/**
		@brief Computes spatial derivative
		@param	[out]	dst_deriv_l		Left approximation of first derivative (same size as data).
		@param	[out]	dst_deriv_r		Right approximation of first derivative (same size as data).
		@param	[in]	grid			Grid structure.
		@param	[in]	src				Source data array.
		@param	[in]	dim				Which dimension to compute derivative on.
		@param	[in]	generateAll		Ignored by this function (optional, default = 0).
		@param	[in]	loop_begin	Index of loop to begin this itteration.
		@param	[in]	slice_length	slice_length of loop to begin this itteration.
		@param	[in]	num_of_slices	number of strides to begin this itteration.
		@retval	true	Operation succeeded.
		@retval	false	Operation failed.
		*/
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
		bool operator==(const UpwindFirstFirst& rhs) const;
		bool operator==(const SpatialDerivative& rhs) const;
		UpwindFirstFirst* clone() const;
		beacls::UVecType get_type() const;
	private:
		UpwindFirstFirst_impl *pimpl;

		/** @overload
		Disable operator=
		*/
		UpwindFirstFirst& operator=(const UpwindFirstFirst& rhs);
		/** @overload
		Disable copy constructor
		*/
		UpwindFirstFirst(const UpwindFirstFirst& rhs);
	};
};
#endif	/* __UpwindFirstFirst_hpp__ */

