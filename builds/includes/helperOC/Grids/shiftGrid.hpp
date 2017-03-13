#ifndef __shiftGrid_hpp__
#define __shiftGrid_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#include <typedef.hpp>
class HJI_Grid;

namespace helperOC {
	/**
		@brief	shiftGrid
				Shifts a grid by the amount shiftAmount. The result may no longer
				 be a valid grid structure, but is still sufficient for plotting
		@param	[in]	gIn
		@param	[in]	shiftAmount
		@retval	gShift
		*/
	PREFIX_VC_DLL
		HJI_Grid* shiftGrid(
		const HJI_Grid* gIn,
		const beacls::FloatVec& shiftAmount
	);

};
#endif	/* __shiftGrid_hpp__ */

