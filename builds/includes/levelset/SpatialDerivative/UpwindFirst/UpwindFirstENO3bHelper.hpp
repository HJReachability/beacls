#ifndef __UpwindFirstENO3bHelper_hpp__
#define __UpwindFirstENO3bHelper_hpp__

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

class HJI_Grid;

class UpwindFirstENO3bHelper_impl;

class UpwindFirstENO3bHelper {
public:
	PREFIX_VC_DLL
	UpwindFirstENO3bHelper(
		const HJI_Grid *hji_grid,
		const beacls::UVecType type = beacls::UVecType_Vector
		);
	~UpwindFirstENO3bHelper();
	/**
	@brief Computes spatial derivative
	@param	[out]		dst_deriv	A three element cell vector containing the three
								ENO approximation arrays phi^i for the first derivative.
	@param	[out]		dst_smooth	A three element cell vector containing the three
								smoothness estimate arrays S_i.
								(Optional, don't request it unless you need it)
	@param	[out]		dst_epsilon	CA single array or scalar containing the small term which
								guards against very small smoothness estimates.
								(Optional, don't request it unless you need it)
	@param	[in]	grid	Grid structure.
	@param	[in]	src	Source data array.
	@param	[in]	dim	Which dimension to compute derivative on.
	@param	[in]		direction   A scalar: false for left, true for right.
	@param	[in]		loop_begin	Index of loop to begin this itteration.
	@param	[in]	slice_length	slice_length of loop to begin this itteration.
	@retval	true	Operation succeeded.
	@retval	false	Operation failed.
	*/
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
	bool operator==(const UpwindFirstENO3bHelper& rhs) const;
	UpwindFirstENO3bHelper* clone() const;

private:
	UpwindFirstENO3bHelper_impl *pimpl;

	/** @overload
	Disable operator=
	*/
	UpwindFirstENO3bHelper& operator=(const UpwindFirstENO3bHelper& rhs);
	/** @overload
	Disable copy constructor
	*/
	UpwindFirstENO3bHelper(const UpwindFirstENO3bHelper& rhs);
};

#endif	/* __UpwindFirstENO3bHelper_hpp__ */

