#ifndef __PartialFunction_hpp__
#define __PartialFunction_hpp__

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
#include <levelset/ExplicitIntegration/SchemeData.hpp>

class PartialFunction_cuda {
public:
	virtual bool operator()(
		std::vector<FLOAT_TYPE>& alphas,
		const FLOAT_TYPE t,
		const std::vector<FLOAT_TYPE>& data,
		const std::vector<FLOAT_TYPE>& derivMins,
		const std::vector<FLOAT_TYPE>& derivMaxs,
		const levelset::SchemeData * schemeData,
		const size_t dim,
		const size_t begin_index,
		const size_t length
		) const = 0;
};

#endif	/* __PartialFunction_hpp__ */

