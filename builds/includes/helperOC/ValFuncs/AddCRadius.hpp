#ifndef __AddCRadius_hpp__
#define __AddCRadius_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#include <typedef.hpp>
#include <vector>
#include <cstddef>
#include <helperOC/helperOC_type.hpp>

namespace levelset {
	class HJI_Grid;
};

namespace helperOC {
	class AddCRadius_impl;
	class AddCRadius {
	private:
		AddCRadius_impl* pimpl;
	public:
		PREFIX_VC_DLL
			AddCRadius(
				const helperOC::ExecParameters& execParameters = helperOC::ExecParameters()
			);
		PREFIX_VC_DLL
			~AddCRadius();
		/**
		@brief	Expands a set given by gIn and dataIn by radius units all around
		@param	[out]	dataOut			shifted data
		@param	[in]	gIn				grid
		@param	[in]	dataIn			original data
		@param	[in]	radius			radius
		@retval	true					Succeeded
		@retval false					Failed
		*/
		PREFIX_VC_DLL
			bool operator()(
				beacls::FloatVec& dataOut,
				const levelset::HJI_Grid* gIn,
				const beacls::FloatVec& dataIn,
				const FLOAT_TYPE radius
				);
	private:
		AddCRadius& operator=(const AddCRadius& rhs);
		AddCRadius(const AddCRadius& rhs);
	};
};
#endif	/* __AddCRadius_hpp__ */

