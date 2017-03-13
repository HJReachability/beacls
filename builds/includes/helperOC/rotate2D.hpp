#ifndef __rotate2D_hpp__
#define __rotate2D_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#include <typedef.hpp>
#include <vector>

namespace helperOC {
	/**
		@brief	Rotates the 2D vector vIn by the angle theta and outputs the result vOut.
				Counterclockwise is positive. 
		@param	[out]	vOut
		@param	[in]	vIn
		@param	[in]	theta
		@retval	true	Succeed
		@retval	false	Failed
	*/
	PREFIX_VC_DLL
	bool rotate2D(
		beacls::FloatVec& vOut,
		const beacls::FloatVec& vIn,
		const FLOAT_TYPE theta
	);
};

#endif	/* __rotate2D_hpp__ */
