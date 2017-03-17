
#ifndef __Term_hpp__
#define __Term_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#include <cstdint>
#include <vector>
#include <utility>
using namespace std::rel_ops;

#include <Core/UVec.hpp>
#include <typedef.hpp>

namespace levelset {
	class SchemeData;

	class Term {
	public:
		/**
		@brief Compute approximate H(x,p) term in an HJ PDE
		@param	[out]		ydot_ite	Change in the data array, in vector form.
		@param	[out]		step_bound_invs	Reciprocal of CFL bound on timesteps for stability of each dimension.
		@param	[in]		t	Time at beginning of timestep.
		@param	[in]		y	Data array in vector form.
		@param	[in]		grid	The input structure
		@param	[in]		loop_begin	loop begin index of this computation
		@param	[in]		loop_length	loop size of this computation
		@param	[in]		num_of_slices	number of strides to begin this itteration.
		@param	[in]		enable_user_defined_dynamics_on_gpu	Flag for user defined dynamics function on gpu
		@retval	true	Succeeded.
		@retval	false	Failed. Dissipation may be required global derivMins/derivMaxs.
						Reduce partial derivMins/derivMaxs to global derivMins/derivMax, then execute again.
		*/
		PREFIX_VC_DLL
			virtual bool execute(
				beacls::FloatVec::iterator ydot_ite,
				beacls::FloatVec& step_bound_invs,
				const FLOAT_TYPE t,
				const beacls::FloatVec& y,
				std::vector<beacls::FloatVec >& derivMins,
				std::vector<beacls::FloatVec >& derivMaxs,
				const SchemeData *schemeData,
				const size_t loop_begin,
				const size_t loop_length,
				const size_t num_of_slices = 1,
				const bool enable_user_defined_dynamics_on_gpu = true,
				const bool updateDerivMinMax = true
			) const = 0;
		PREFIX_VC_DLL
			virtual bool synchronize(const SchemeData* schemeData = NULL) const = 0;
		PREFIX_VC_DLL
			virtual bool operator==(const Term& rhs) const = 0;

		virtual Term* clone() const = 0;
		/**
		@brief destructor
		*/
		virtual ~Term() = 0;
	private:
	};
	inline
		Term::~Term() {}
};
#endif	/* __Term_hpp__ */

