#ifndef __AddCRadius_impl_hpp__
#define __AddCRadius_impl_hpp__

#include <cstdint>
#include <vector>
#include <cstddef>
#include <levelset/levelset.hpp>
#include <helperOC/helperOC_type.hpp>

class HJI_Grid;
class DynSysSchemeData;
namespace helperOC{
	class HDJPDE;
	class AddCRadius_impl {
	public:
	private:
		HJIPDE* hjipde;
		DynSysSchemeData* schemeData;
		HJIPDE_extraArgs extraArgs;
		HJIPDE_extraOuts extraOuts;
	public:
		AddCRadius_impl(
			const helperOC::ExecParameters& execParameters
		);
		~AddCRadius_impl();

		bool operator()(
			beacls::FloatVec& dataOut,
			const HJI_Grid* gIn,
			const beacls::FloatVec& dataIn,
			const FLOAT_TYPE radius
			);
	private:
		/** @overload
		Disable operator=
		*/
		AddCRadius_impl& operator=(const AddCRadius_impl& rhs);
		/** @overload
		Disable copy constructor
		*/
		AddCRadius_impl(const AddCRadius_impl& rhs);
	};
};

#endif	/* __AddCRadius_impl_hpp__ */

