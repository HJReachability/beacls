#ifndef __odeCFL_SubStep_hpp__
#define __odeCFL_SubStep_hpp__

#include <typedef.hpp>
#include <vector>
#include <deque>
#include <set> 
namespace levelset {
	class Term;
	class SchemeData;
	class OdeCFL_CommandQueue;
	void odeCFL_SubStep(
		std::vector<OdeCFL_CommandQueue*> odeCFL_CommandQueues,
		const FLOAT_TYPE t,
		std::vector<beacls::FloatVec >& step_bound_invss,
		std::vector<beacls::FloatVec >& local_derivMinss,
		std::vector<beacls::FloatVec >& local_derivMaxss,
		beacls::FloatVec& y,
		beacls::FloatVec& ydot,
		std::vector<beacls::FloatVec >& derivMins,
		std::vector<beacls::FloatVec >& derivMaxs,
		std::vector<beacls::FloatVec >& lastDerivMins,
		std::vector<beacls::FloatVec >& lastDerivMaxs,
		std::deque<bool>& executeAgains,
		const size_t num_of_dimensions,
		const size_t first_dimension_loop_size,
		const size_t num_of_chunk_lines,
		const size_t num_of_slices,
		const size_t parallel_loop_size,
		const size_t num_of_outer_lines,
		const size_t num_of_inner_lines,
		const size_t second_dimension_loop_size,
		const size_t third_dimension_loop_size,
		const size_t num_of_parallel_loop_lines,
		const size_t actual_num_of_threas,
		const levelset::DelayedDerivMinMax_Type delayedDerivMinMax,
		const bool enable_user_defined_dynamics_on_gpu
	);
};

#endif	/* __odeCFL_SubStep_hpp__ */

