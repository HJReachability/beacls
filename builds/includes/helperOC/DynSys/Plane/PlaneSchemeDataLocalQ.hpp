#ifndef __PlaneSchemeDataLocalQ_hpp__
#define __PlaneSchemeDataLocalQ_hpp__

#include <levelset/levelset.hpp>
#include <helperOC/DynSys/DynSys/DynSysSchemeData.hpp>
#include <helperOC/DynSys/DynSys/DynSys.hpp>
#include <cstdint>
#include <vector>
#include <cstddef>
#include <utility>
using namespace std::rel_ops;

#include <Core/UVec.hpp>
namespace helperOC {
	
	class PlaneSchemeDataLocalQ : public helperOC::DynSysSchemeData {
	public: 
		FLOAT_TYPE vMin_;
		FLOAT_TYPE vMax_; 
		FLOAT_TYPE dMax_x_;
		FLOAT_TYPE dMax_y_;
		FLOAT_TYPE dMax_theta_; 
		FLOAT_TYPE wMax_;
	public:
		PlaneSchemeDataLocalQ() : DynSysSchemeData(), vMin_(0), vMax_(1.0), dMax_x_(0), 
			dMax_y_(0), dMax_theta_(0), wMax_(0) {}
		PlaneSchemeDataLocalQ(FLOAT_TYPE vMin_, FLOAT_TYPE vMax_, FLOAT_TYPE dMax_x_, 
			FLOAT_TYPE dMax_y_, FLOAT_TYPE dMax_theta_, FLOAT_TYPE wMax_) : vMin_(vMin_), 
			vMax_(vMax_), dMax_x_(dMax_x_), dMax_y_(dMax_y_), dMax_theta_(dMax_theta_), wMax_(wMax_) {}
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
			DynSysSchemeData(rhs),
			vMin_(rhs.vMin_),
			vMax_(rhs.vMax_),
			dMax_x_(rhs.dMax_x_),
			dMax_y_(rhs.dMax_y_),
			dMax_theta_(rhs.dMax_theta_),
			wMax_(rhs.wMax_)
		{}

	};
};

#endif	/* __PlaneSchemeDataLocalQ_hpp__ */

