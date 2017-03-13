#ifndef __ExplicitIntegration_hpp__
#define __ExplicitIntegration_hpp__

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
#include <typedef.hpp>

class SchemeData;

class ExplicitIntegration  {
public:
	/*
	@brief
	@param	[in]	line_length_of_chunk				Line length of each parallel execution chunks (0 means set automatically)
	@param	[in]	num_of_threads						Number of CPU Threads (0 means use all logical threads of CPU)
	@param	[in]	num_of_gpus							Number of GPUs which (0 means use all GPUs)
	@param	[in]	delayedDerivMinMax				Use last step's min/max of derivatives, and skip 2nd pass.
	@param	[in]	enable_user_defined_dynamics_on_gpu	Flag for user defined dynamics function on gpu
	@retval												New time
	*/
	PREFIX_VC_DLL
		virtual FLOAT_TYPE execute(
			beacls::FloatVec& y,
			const beacls::FloatVec& tspan,
			const beacls::FloatVec& y0,
			const SchemeData *schemeData,
			const size_t line_length_of_chunk = 0,
			const size_t num_of_threads = 0,
			const size_t num_of_gpus = 0,
			const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable,
			const bool enable_user_defined_dynamics_on_gpu = true
		) = 0;
	virtual ~ExplicitIntegration() = 0;
};
inline
ExplicitIntegration::~ExplicitIntegration() {}
#endif	/* __ExplicitIntegration_hpp__ */

