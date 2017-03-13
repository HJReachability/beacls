#ifndef __interpn_hpp__
#define __interpn_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#include <vector>
#include <cstddef>
#include <typedef.hpp>

namespace beacls {
	PREFIX_VC_DLL
		bool interpn(
			beacls::FloatVec& Vq,
			const std::vector<const beacls::FloatVec*>& X_ptrs,
			const std::vector<beacls::IntegerVec>& Ns,
			const Interpolate_Type interp_method = beacls::Interpolate_linear,
			const std::vector<beacls::Extrapolate_Type>& extrap_methods = std::vector<beacls::Extrapolate_Type>()
		);
};
#endif	/* __interpn_hpp__ */

