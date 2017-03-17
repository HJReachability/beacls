#ifndef __OdeCFL1_hpp__
#define __OdeCFL1_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#include <cstdint>
#include <vector>
#include <limits>
#include <typedef.hpp>
#include <levelset/ExplicitIntegration/Integrators/Integrator.hpp>
namespace levelset {
	class Term;
	class SchemeData;

	class OdeCFL1_impl;

	class OdeCFL1 : public Integrator {
	public:
		PREFIX_VC_DLL
			OdeCFL1(
				const Term *schemeFunc,
				const FLOAT_TYPE factor_cfl = 0.5,
				const FLOAT_TYPE max_step = std::numeric_limits<FLOAT_TYPE>::max(),
				const std::vector<levelset::PostTimestep_Exec_Type*> &post_time_steps = std::vector<levelset::PostTimestep_Exec_Type*>(),
				const bool single_step = false,
				const bool stats = false,
				const levelset::TerminalEvent_Exec_Type* terminalEvent = NULL
			);
		~OdeCFL1();
		FLOAT_TYPE execute(
			beacls::FloatVec& y,
			const beacls::FloatVec& tspan,
			const beacls::FloatVec& y0,
			const SchemeData *schemeData,
			const size_t line_length_of_chunk,
			const size_t num_of_threads,
			const size_t num_of_gpus,
			const levelset::DelayedDerivMinMax_Type delayedDerivMinMax,
			const bool enable_user_defined_dynamics_on_gpu
		);
		OdeCFL1* clone() const;

	private:
		OdeCFL1_impl *pimpl;

		/** @overload
		Disable operator=
		*/
		OdeCFL1& operator=(const OdeCFL1& rhs);
		/** @overload
		Disable copy constructor
		*/
		OdeCFL1(const OdeCFL1& rhs);
	};
};
#endif	/* __OdeCFL1_hpp__ */

