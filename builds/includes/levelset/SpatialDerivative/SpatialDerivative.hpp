
#ifndef __SpatialDerivative_hpp__
#define __SpatialDerivative_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#include <typedef.hpp>
#include <vector>
#include <cstdint>
#include <utility>
using namespace std::rel_ops;
#include <Core/UVec.hpp>
namespace levelset {
	class HJI_Grid;
	class SpatialDerivative {
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
		PREFIX_VC_DLL
			virtual bool execute(
				beacls::UVec& dst_deriv_l,
				beacls::UVec& dst_deriv_r,
				const HJI_Grid *grid,
				const FLOAT_TYPE* src,
				const size_t dim,
				const bool generateAll,
				const size_t loop_begin,
				const size_t slice_length,
				const size_t num_of_slices = 1
			) = 0;
		PREFIX_VC_DLL
			virtual bool synchronize(const size_t dim) = 0;
		PREFIX_VC_DLL
			virtual bool operator==(const SpatialDerivative& rhs) const = 0;
		PREFIX_VC_DLL
			virtual SpatialDerivative* clone() const = 0;
		virtual ~SpatialDerivative() = 0;
		PREFIX_VC_DLL
			virtual beacls::UVecType get_type() const = 0;
	};
	inline
		SpatialDerivative::~SpatialDerivative() {}
};
#endif	/* __SpatialDerivative_hpp__ */

