#ifndef __OdeCFL3_impl_hpp__
#define __OdeCFL3_impl_hpp__

#include <cstdint>
#include <vector>
#include <typedef.hpp>
#include <set> 
namespace levelset {
	class OdeCFL_CommandQueue;
	class OdeCFL_Worker;
	class Term;
	class SchemeData;
	class OdeCFL3_impl {
	private:
		std::vector<levelset::OdeCFL_Worker*> workers;
		std::vector<levelset::OdeCFL_CommandQueue*> commandQueues;
		const Term* term;
		FLOAT_TYPE factor_cfl;
		FLOAT_TYPE max_step;
		const std::vector<levelset::PostTimestep_Exec_Type*> post_time_steps;
		bool single_step;
		bool stats;
		const levelset::TerminalEvent_Exec_Type* terminalEvent;

		beacls::FloatVec y1;
		beacls::FloatVec ydot;
		beacls::FloatVec yOld;
		std::vector<beacls::FloatVec > step_bound_invss;
		beacls::FloatVec step_bound_invs;
		beacls::FloatVec eventValuesOld;
		std::vector<beacls::FloatVec > derivMins;
		std::vector<beacls::FloatVec > derivMaxs;
		std::vector<beacls::FloatVec > local_derivMinss;
		std::vector<beacls::FloatVec > local_derivMaxss;
		std::vector<beacls::FloatVec > lastDerivMaxs;
		std::vector<beacls::FloatVec > lastDerivMins;
		std::deque<bool> executeAgains;
	public:
		OdeCFL3_impl(
			const Term *schemeFunc,
			const FLOAT_TYPE factor_cfl,
			const FLOAT_TYPE max_step,
			const std::vector<levelset::PostTimestep_Exec_Type*> &post_time_steps,
			const bool single_step,
			const bool stats,
			const levelset::TerminalEvent_Exec_Type* terminalEvent
		);
		OdeCFL3_impl(
			const Term *schemeFunc,
			const FLOAT_TYPE factor_cfl,
			const FLOAT_TYPE max_step,
			const std::vector<levelset::PostTimestep_Exec_Type*> &post_time_steps,
			const bool single_step,
			const bool stats,
			const levelset::TerminalEvent_Exec_Type* terminalEvent,
			const bool execute_local_q
		);
		~OdeCFL3_impl();
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
		FLOAT_TYPE execute_local_q(
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
		);
		OdeCFL3_impl* clone() const {
			return new OdeCFL3_impl(*this);
		};
	private:
		/** @overload
		Disable operator=
		*/
		OdeCFL3_impl& operator=(const OdeCFL3_impl& rhs);
		/** @overload
		Disable copy constructor
		*/
		OdeCFL3_impl(const OdeCFL3_impl& rhs);
	};
};
#endif	/* __OdeCFL3_impl_hpp__ */

