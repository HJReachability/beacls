
#ifndef __ExtractCostates_impl_hpp__
#define __ExtractCostates_impl_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#include <cstddef>
#include <vector>
#include <typedef.hpp>

namespace helperOC {
	class DynSys;
	class ComputeGradients;

	class ExtractCostates_impl {
	private:
		helperOC::ApproximationAccuracy_Type accuracy;
		helperOC::ComputeGradients* computeGradients;
	public:
		ExtractCostates_impl(
			const helperOC::ApproximationAccuracy_Type accuracy
		);

		~ExtractCostates_impl();
		bool operator()(
			std::vector<beacls::FloatVec >& derivC,
			std::vector<beacls::FloatVec >& derivL,
			std::vector<beacls::FloatVec >& derivR,
			const levelset::HJI_Grid* grid,
			const beacls::FloatVec& data,
			const size_t data_length,
			const bool upWind,
			const helperOC::ExecParameters& execParameters
			);

	private:
		/** @overload
		Disable operator=
		*/
		ExtractCostates_impl& operator=(const ExtractCostates_impl& rhs);
		/** @overload
		Disable copy constructor
		*/
		ExtractCostates_impl(const ExtractCostates_impl& rhs);
	};
};

#endif	/* __ExtractCostates_impl_hpp__ */

