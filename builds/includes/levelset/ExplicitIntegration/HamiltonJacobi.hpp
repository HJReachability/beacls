
#ifndef __HamiltonJacobi_hpp__
#define __HamiltonJacobi_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#include <typedef.hpp>
#include <cstddef>
#include <vector>
#include <ExplicitIntegration/SchemeData.hpp>

class HamiltonJacobiFunction {
public:
	virtual bool operator()(
		std::vector<FLOAT_TYPE>& hamValue,
		SchemeData** dst_schemeData,
		const FLOAT_TYPE t,
		const SchemeData* src_schemeData,
		const std::vector<std::vector<FLOAT_TYPE> > & data,
		const size_t begin_index,
		const size_t length
		) const = 0;
};

#endif	/* __HamiltonJacobi_hpp__ */

