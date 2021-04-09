#ifndef __SchemeData_hpp__
#define __SchemeData_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#include <cstddef>
#include <vector>
#include <set> 
#include <utility>
using namespace std::rel_ops;

#include <typedef.hpp>
#include <Core/UVec.hpp>
namespace levelset {

	class HJI_Grid;
	class Term;
	class Dissipation;
	class SpatialDerivative;

	class SchemeData_impl;

	class SchemeData {
	public:
	private:
		SchemeData_impl *pimpl;
	public:
		PREFIX_VC_DLL
			SchemeData(
			);
		PREFIX_VC_DLL
			virtual ~SchemeData();
		PREFIX_VC_DLL
			virtual bool operator==(const SchemeData& rhs) const;

		PREFIX_VC_DLL
			const HJI_Grid* get_grid() const;
		SpatialDerivative* get_spatialDerivative() const;
		PREFIX_VC_DLL
			Dissipation* get_dissipation() const;
		PREFIX_VC_DLL
			const Term* get_innerFunc() const;
		PREFIX_VC_DLL
			const SchemeData* get_innerData() const;
		bool get_positive() const;

		PREFIX_VC_DLL
			void set_grid(const HJI_Grid* grid);

		PREFIX_VC_DLL
			void set_spatialDerivative(SpatialDerivative* spatialDerivative);
		PREFIX_VC_DLL
			void set_dissipation(Dissipation* dissipation);
		PREFIX_VC_DLL
			void set_innerFunc(const Term* innerFunc);
		PREFIX_VC_DLL
			void set_innerData(const SchemeData* innerData);
		PREFIX_VC_DLL
			void set_positive(const bool positive);

		PREFIX_VC_DLL
			virtual SchemeData* clone() const;

		PREFIX_VC_DLL
			virtual bool hamFunc(
				beacls::UVec& hamValue,
				const FLOAT_TYPE t,
				const beacls::UVec& data,
				const std::vector<beacls::UVec>& derivs,
				const size_t begin_index,
				const size_t length
			) const;
		PREFIX_VC_DLL
			virtual bool partialFunc(
				beacls::UVec& alphas_uvec,
				const FLOAT_TYPE t,
				const beacls::UVec& data,
				const std::vector<beacls::UVec>& derivMins,
				const std::vector<beacls::UVec>& derivMaxs,
				const size_t dim,
				const size_t begin_index,
				const size_t length
			) const;
		PREFIX_VC_DLL
			virtual bool hamFuncLocalQ(
				beacls::UVec& hamValue,
				const FLOAT_TYPE t,
				const beacls::UVec& data,
				const std::vector<beacls::UVec>& derivs,
				const size_t begin_index,
				const size_t length, 
				const std::set<size_t> &Q
			) const;
		PREFIX_VC_DLL
			virtual bool partialFuncLocalQ(
				beacls::UVec& alphas_uvec,
				const FLOAT_TYPE t,
				const beacls::UVec& data,
				const std::vector<beacls::UVec>& derivMins,
				const std::vector<beacls::UVec>& derivMaxs,
				const size_t dim,
				const size_t begin_index,
				const size_t length, 
				const std::set<size_t> &Q
			) const;
		PREFIX_VC_DLL
			virtual bool hamFunc_cuda(
				beacls::UVec& hamValue,
				const FLOAT_TYPE t,
				const beacls::UVec& data,
				const std::vector<beacls::UVec>& x_uvecs,
				const std::vector<beacls::UVec>& derivs,
				const size_t begin_index,
				const size_t length
			) const;
		PREFIX_VC_DLL
			virtual bool partialFunc_cuda(
				beacls::UVec& alphas,
				const FLOAT_TYPE t,
				const beacls::UVec& data,
				const std::vector<beacls::UVec>& x_uvecs,
				const std::vector<beacls::UVec>& derivMins,
				const std::vector<beacls::UVec>& derivMaxs,
				const size_t dim,
				const size_t begin_index,
				const size_t length
			) const;

		/** @overload
	Disable copy constructor
	*/
		PREFIX_VC_DLL
			SchemeData(const SchemeData& rhs);

	private:

		/** @overload
		Disable operator=
		*/
		SchemeData& operator=(const SchemeData& rhs);
	};
};
#endif	/* __SchemeData_hpp__ */

