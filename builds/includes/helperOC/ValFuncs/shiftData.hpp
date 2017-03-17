#ifndef __shiftData_hpp__
#define __shiftData_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#include <helperOC/helperOC_type.hpp>
#include <typedef.hpp>
#include <cstddef>
class HJI_Grid;

namespace helperOC {
	/**
	@brief	Shifts about origin
			The grid structure g is common to dataIn and dataOut
	@param	[out]	dataOut			shifted data
	@param	[in]	g				grid
	@param	[in]	dataIn			original data
	@param	[in]	shift			shift amount
	@param	[in]	pdims			position dimensions
	@param	[in]	interp_method	interporation method
	@retval	true					Succeeded
	@retval false					Failed
	*/
	PREFIX_VC_DLL
		bool shiftData(
			beacls::FloatVec& dataOut,
			const levelset::HJI_Grid* g,
			const beacls::FloatVec& dataIn,
			const beacls::FloatVec& shift,
			const beacls::IntegerVec& pdims = beacls::IntegerVec{ 0,1 },
			const beacls::Interpolate_Type interp_method = beacls::Interpolate_linear
		);
};
#endif	/* __shiftData_hpp__ */

