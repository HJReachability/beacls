#ifndef __ArtificialDissipationGLF_hpp__
#define __ArtificialDissipationGLF_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#include <cstdint>
#include <vector>
#include <set> 
#include <utility>
using namespace std::rel_ops;
#include <typedef.hpp>
#include <levelset/ExplicitIntegration/Dissipations/Dissipation.hpp>
namespace levelset {

	class SchemeData;

	class ArtificialDissipationGLF_impl;

	class ArtificialDissipationGLF : public Dissipation {
	public:
		PREFIX_VC_DLL
			ArtificialDissipationGLF(
			);
		~ArtificialDissipationGLF();
		bool execute(
			beacls::UVec& diss,
			beacls::FloatVec& step_bound_invs,
			std::vector<beacls::UVec>& derivMins,
			std::vector<beacls::UVec>& derivMaxs,
			const FLOAT_TYPE t,
			const beacls::UVec& data,
			const std::vector<beacls::UVec>& x_uvecs,
			const std::vector<beacls::UVec >& deriv_ls,
			const std::vector<beacls::UVec >& deriv_rs,
			const SchemeData *schemeData,
			const size_t begin_index,
			const bool enable_user_defined_dynamics_on_gpu,
			const bool updateDerivMinMax
		);
		bool execute_local_q(
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
		);
		bool operator==(const ArtificialDissipationGLF& rhs) const;
		bool operator==(const Dissipation& rhs) const;
		ArtificialDissipationGLF* clone() const;
	private:
		ArtificialDissipationGLF_impl *pimpl;

		/** @overload
		Disable operator=
		*/
		ArtificialDissipationGLF& operator=(const ArtificialDissipationGLF& rhs);
		/** @overload
		Disable copy constructor
		*/
		ArtificialDissipationGLF(const ArtificialDissipationGLF& rhs);
	};
};
#endif	/* __ArtificialDissipationGLF_hpp__ */
