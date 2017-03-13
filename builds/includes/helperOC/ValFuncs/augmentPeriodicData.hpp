#ifndef __augmentPeriodicData_hpp__
#define __augmentPeriodicData_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#include <helperOC/helperOC_type.hpp>
#include <typedef.hpp>
#include <vector>
class HJI_Grid;
namespace helperOC {
	HJI_Grid* augmentPeriodicData(
		beacls::FloatVec& dataOut,
		const HJI_Grid* gIn,
		const beacls::FloatVec& data);

};
#endif	/* __augmentPeriodicData_hpp__ */

