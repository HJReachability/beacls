#ifndef __DynSysSchemeData_hpp__
#define __DynSysSchemeData_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#include <cstdint>
#include <vector>
#include <cstddef>
#include <utility>
using namespace std::rel_ops;

#include <typedef.hpp>
#include <helperOC/helperOC_type.hpp>
#include <helperOC/DynSys/DynSys/DynSysTypeDef.hpp>
#include <levelset/ExplicitIntegration/SchemeData.hpp>
#include <Core/UVec.hpp>

namespace helperOC {
	class ComputeGradients;
	class DynSys;
	class HJI_Grid;
	class DynSysSchemeData_Workspace;
	class DynSysSchemeDataSide {
	public:
		bool upper;
		bool lower;
		bool valid() const { return (upper | lower); };
		DynSysSchemeDataSide() : upper(false), lower(false) {}
		~DynSysSchemeDataSide() {};
		bool operator==(const DynSysSchemeDataSide& rhs) const {
			if (this == &rhs) return true;
			else if (upper != rhs.upper) return false;
			else if (lower != rhs.lower) return false;
			else return true;
		}
	};

	class DynSysSchemeData : public levelset::SchemeData {
	public:
		DynSysSchemeData_Workspace* ws;
		DynSys* dynSys;
		helperOC::ComputeGradients* computeGradients;
		DynSys_UMode_Type uMode;
		DynSys_DMode_Type dMode;
		DynSys_TMode_Type tMode;
		std::vector<beacls::UVec> MIEderivs;
		beacls::IntegerVec MIEdims;
		beacls::IntegerVec TIdims;
		std::vector<beacls::FloatVec > uIns;
		std::vector<beacls::FloatVec > dIns;
		helperOC::ApproximationAccuracy_Type accuracy;
		DynSysSchemeDataSide side;
		FLOAT_TYPE dc;
		bool dissComp;
		bool trueMIEDeriv;

	public:
		PREFIX_VC_DLL
			DynSysSchemeData(
			);
		~DynSysSchemeData();
		PREFIX_VC_DLL
			bool operator==(const DynSysSchemeData& rhs) const;
		PREFIX_VC_DLL
			bool operator==(const SchemeData& rhs) const;

		DynSysSchemeData* clone() const {
			return new DynSysSchemeData(*this);
		}
		PREFIX_VC_DLL
			bool hamFunc(
				beacls::UVec& hamValue_uvec,
				const FLOAT_TYPE t,
				const beacls::UVec& data,
				const std::vector<beacls::UVec>& derivs,
				const size_t begin_index,
				const size_t length
			)const;
		PREFIX_VC_DLL
			bool partialFunc(
				beacls::UVec& alphas_uvec,
				const FLOAT_TYPE t,
				const beacls::UVec& data,
				const std::vector<beacls::UVec>& derivMins,
				const std::vector<beacls::UVec>& derivMaxs,
				const size_t dim,
				const size_t begin_index,
				const size_t length
			) const;
#if defined(USER_DEFINED_GPU_DYNSYS_FUNC)
		PREFIX_VC_DLL
			bool hamFunc_cuda(
				beacls::UVec& hamValue_uvec,
				const FLOAT_TYPE t,
				const beacls::UVec& data,
				const std::vector<beacls::UVec>& x_uvecs,
				const std::vector<beacls::UVec>& derivs,
				const size_t begin_index,
				const size_t length
			)const;
		PREFIX_VC_DLL
			bool partialFunc_cuda(
				beacls::UVec& alphas_uvec,
				const FLOAT_TYPE t,
				const beacls::UVec& data,
				const std::vector<beacls::UVec>& x_uvecs,
				const std::vector<beacls::UVec>& derivMins,
				const std::vector<beacls::UVec>& derivMaxs,
				const size_t dim,
				const size_t begin_index,
				const size_t length
			) const;
#endif /* defined(USER_DEFINED_GPU_DYNSYS_FUNC) */


		/** @overload
		Disable copy constructor
		*/
		DynSysSchemeData(const DynSysSchemeData& rhs);
	private:

		/** @overload
		Disable operator=
		*/
		DynSysSchemeData& operator=(const DynSysSchemeData& rhs);


	};
};
#endif	/* __DynSysSchemeData_hpp__ */
