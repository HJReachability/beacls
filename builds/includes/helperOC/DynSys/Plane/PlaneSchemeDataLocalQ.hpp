#ifndef __PlaneSchemeDataLocalQ_hpp__
#define __PlaneSchemeDataLocalQ_hpp__

#include <levelset/levelset.hpp>
#include <cstdint>
#include <vector>
#include <cstddef>
#include <utility>
using namespace std::rel_ops;

#include <Core/UVec.hpp>
#include <helperOC/DynSys/DynSys/DynSysSchemeData.hpp> 
#include <levelset/ExplicitIntegration/SchemeData.hpp>
namespace helperOC {
	
	class PlaneSchemeDataLocalQ : public DynSysSchemeData {
	public: 
		FLOAT_TYPE vMin;
		FLOAT_TYPE vMax; 
		FLOAT_TYPE dMax_x;
		FLOAT_TYPE dMax_y; 
		FLOAT_TYPE wMax;
	public:
		PlaneSchemeDataLocalQ() : DynSysSchemeData() {}
		~PlaneSchemeDataLocalQ() {}
		bool operator==(const PlaneSchemeDataLocalQ& rhs) const;
		bool operator==(const SchemeData& rhs) const;

		PlaneSchemeDataLocalQ* clone() const {
			return new PlaneSchemeDataLocalQ(*this);
		}
		bool hamFunc(
			beacls::UVec& hamValue_uvec,
			const FLOAT_TYPE t,
			const beacls::UVec& data,
			const std::vector<beacls::UVec>& derivs,
			const size_t begin_index,
			const size_t length
		)const;
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
		bool initializeLocalQ(
			const beacls::FloatVec &vRange, 
			const beacls::FloatVec &dMax,
			const FLOAT_TYPE wMax 
		);
		bool hamFuncLocalQ(
			beacls::UVec& hamValue_uvec,
			const FLOAT_TYPE t,
			const beacls::UVec& data,
			const std::vector<beacls::UVec>& derivs,
			const size_t begin_index,
			const size_t length, 
			const std::set<size_t> &Q
		) const;
		bool partialFuncLocalQ(
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
	private:

		/** @overload
		Disable operator=
		*/
		PlaneSchemeDataLocalQ& operator=(const PlaneSchemeDataLocalQ& rhs);
		/** @overload
		Disable copy constructor
		*/
		PlaneSchemeDataLocalQ(const PlaneSchemeDataLocalQ& rhs) :
			DynSysSchemeData(rhs)
		{}

	};
};

#endif	/* __PlaneSchemeDataLocalQ_hpp__ */

