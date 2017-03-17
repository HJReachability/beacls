
#ifndef __UpwindFirst_hpp__
#define __UpwindFirst_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#include <typedef.hpp>
#include <levelset/SpatialDerivative/SpatialDerivative.hpp>
#include <vector>
#include <cstdint>
#include <utility>
using namespace std::rel_ops;

#include <Core/UVec.hpp>
namespace levelset {

	class HJI_Grid;
	class UpwindFirst : public SpatialDerivative {
	public:
		/**
		@brief Computes spatial derivative
		@param	[out]		dst_deriv_l	Cell vector containing the 3 or 4 left approximations
		of the first derivative (each the same size as data).
		@param	[out]		dst_deriv_r	Cell vector containing the 3 or 4 right approximations
		of the first derivative (each the same size as data).
		@param	[in]	grid	Grid structure.
		@param	[in]	src	Source data array.
		@param	[in]	dim	Which dimension to compute derivative on.
		@param	[in]		generateAll	Return all possible third order upwind approximations.
									If this boolean is true, then derivL and derivR will
									be cell vectors containing all the approximations
									instead of just the ENO approximation.
									(optional, default = 0)
		@param	[in]	loop_begin	Index of loop to begin this itteration.
		@param	[in]	slice_length	slice_length of loop to begin this itteration.
		@param	[in]	num_of_slices	number of strides to begin this itteration.
		@param	[out]	x_uvec		where to copy xs between calculation of spatial derivative
		@retval	true	Operation succeeded.
		@retval	false	Operation failed.
		*/
		virtual bool execute(
			beacls::UVec& dst_deriv_l,
			beacls::UVec& dst_deriv_r,
			const HJI_Grid *grid,
			const FLOAT_TYPE* src,
			const size_t dim,
			const bool generateAll,
			const size_t loop_begin,
			const size_t slice_length,
			const size_t num_of_slices
		) = 0;
		virtual bool synchronize(const size_t dim) = 0;
		virtual ~UpwindFirst() = 0;
		virtual bool operator==(const SpatialDerivative& rhs) const = 0;
		virtual UpwindFirst* clone() const = 0;
		virtual beacls::UVecType get_type() const = 0;
	private:
	};
	inline
		UpwindFirst::~UpwindFirst() {}

	/**
	@brief Checks two derivative approximations for equivalence.
	A warning is generated if either of these conditions holds:
	1) The approximation magnitude > bound
	and the maximum relative error > bound.
	2) The approximation magnitude < bound
	and the maximum absolute error > bound.

	Normally, the return values are ignored
	(the whole point is the warning checks).

	@param	[out]		relErrors	The relative error at each point in the array
								where the magnitude > bound (NaN otherwise).
	@param	[out]		absErrors	The absolute error at each point in the array.
	@param	[in]	approx1s	An array containing one approximation.
	@param	[in]	approx2s	An array containing the other approximation.
	@param	[in]	bound	The bound above which warnings are generated.
	@retval	true			There is no error.
	@retval false			There is some error
	*/
	bool checkEquivalentApprox(
		beacls::FloatVec& relErrors,
		beacls::FloatVec& absErrors,
		const beacls::FloatVec& approx1s,
		const beacls::FloatVec& approx2s,
		const FLOAT_TYPE bound
	);
	bool checkEquivalentApprox(
		beacls::FloatVec& relErrors,
		beacls::FloatVec& absErrors,
		const beacls::UVec& approx1s,
		const beacls::UVec& approx2s,
		const FLOAT_TYPE bound
	);
};

#endif	/* __UpwindFirst_hpp__ */

