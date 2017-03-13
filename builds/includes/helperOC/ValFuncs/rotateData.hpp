#ifndef __rotateData_hpp__
#define __rotateData_hpp__

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
	@brief	Rotates about origin
	The grid structure g is common to dataIn and dataOut
	@param	[out]	dataOut			shifted data
	@param	[in]	g				grid
	@param	[in]	dataIn			original data
	@param	[in]	shift			shift amount
	@param	[in]	pdims			position dimensions
	@param	[in]	adim			angle dimensions
	@param	[in]	interp_method	interporation method
	@retval	true					Succeeded
	@retval false					Failed
	*/
	PREFIX_VC_DLL
	bool rotateData(
		beacls::FloatVec& dataOut,
		const HJI_Grid* g,
		const beacls::FloatVec& dataIn,
		const FLOAT_TYPE theta,
		const beacls::IntegerVec& pdims = beacls::IntegerVec{ 0,1 },
		const beacls::IntegerVec& adim = beacls::IntegerVec{ 2 },
		const beacls::Interpolate_Type interp_method = beacls::Interpolate_linear
	);
};
#endif	/* __rotateData_hpp__ */

