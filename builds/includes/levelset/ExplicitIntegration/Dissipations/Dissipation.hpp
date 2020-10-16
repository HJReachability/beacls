#ifndef __Dissipation_hpp__
#define __Dissipation_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#include <cstdint>
#include <cstddef>
#include <vector>
#include <set> 
#include <utility>
using namespace std::rel_ops;

#include <Core/UVec.hpp>
#include <typedef.hpp>
namespace levelset {

	class SchemeData;

	class Dissipation {
	public:
		PREFIX_VC_DLL
			virtual bool execute(
				beacls::UVec& diss,
				beacls::FloatVec& step_bound_invs,
				std::vector<beacls::UVec>& derivMins,
				std::vector<beacls::UVec>& derivMaxs,
				const FLOAT_TYPE t,
				const beacls::UVec& data,
				const std::vector<beacls::UVec>& x_uvecs,
				const std::vector<beacls::UVec>& deriv_ls,
				const std::vector<beacls::UVec>& deriv_rs,
				const SchemeData *schemeData,
				const size_t begin_index,
				const bool enable_user_defined_dynamics_on_gpu = true,
				const bool updateDerivMinMax = true
			) = 0;
		PREFIX_VC_DLL
			virtual bool execute_local_q(
				beacls::UVec& diss,
				beacls::FloatVec& step_bound_invs,
				std::vector<beacls::UVec>& derivMins,
				std::vector<beacls::UVec>& derivMaxs,
				const FLOAT_TYPE t,
				const beacls::UVec& data,
				const std::vector<beacls::UVec>& x_uvecs,
				const std::vector<beacls::UVec>& deriv_ls,
				const std::vector<beacls::UVec>& deriv_rs,
				const SchemeData *schemeData,
				const size_t begin_index,
				const std::set<size_t> &Q,
				const bool enable_user_defined_dynamics_on_gpu = true,
				const bool updateDerivMinMax = true 
			) { return true; };
		PREFIX_VC_DLL
			virtual bool operator==(const Dissipation& rhs) const = 0;
		PREFIX_VC_DLL
			virtual Dissipation* clone() const = 0;
		virtual ~Dissipation() = 0;
	private:
	};
	inline
		Dissipation::~Dissipation() {}
};
#endif	/* __Dissipation_hpp__ */

