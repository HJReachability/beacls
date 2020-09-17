#ifndef __Integrator_hpp__
#define __Integrator_hpp__

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
#include <typedef.hpp>
#include <levelset/ExplicitIntegration/ExplicitIntegration.hpp>
namespace levelset {
	class HJI_Grid;
	class SchemeData;

	class Integrator : public ExplicitIntegration {
	public:
		PREFIX_VC_DLL
			virtual FLOAT_TYPE execute(
				beacls::FloatVec& y,
				const beacls::FloatVec& tspan,
				const beacls::FloatVec& y0,
				const SchemeData *schemeData,
				const size_t line_length_of_chunk = 0,
				const size_t num_of_threads = 0,
				const size_t num_of_gpus = 0,
				const levelset::DelayedDerivMinMax_Type delayedDerivMinMax = levelset::DelayedDerivMinMax_Disable,
				const bool enable_user_defined_dynamics_on_gpu = true
			) = 0;
			#pragma GCC diagnostic push
			#pragma GCC diagnostic ignored "-Wunused-parameter"
			virtual FLOAT_TYPE execute_local_q(
				beacls::FloatVec& y,
				const beacls::FloatVec& tspan,
				const beacls::FloatVec& y0,
				const SchemeData *schemeData,
				const size_t line_length_of_chunk,
				const size_t num_of_threads,
				const size_t num_of_gpus,
				const levelset::DelayedDerivMinMax_Type delayedDerivMinMax,
				const bool enable_user_defined_dynamics_on_gpu,
				const std::set<size_t>& Q
			) {return 0;}
			#pragma GCC diagnostic pop
		PREFIX_VC_DLL
			virtual Integrator* clone() const = 0;
		virtual ~Integrator() = 0;

	private:
	};
	inline
		Integrator::~Integrator() {}
};
#endif	/* __Integrator_hpp__ */

