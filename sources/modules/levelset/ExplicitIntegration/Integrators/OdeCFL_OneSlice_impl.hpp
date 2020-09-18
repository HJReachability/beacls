#ifndef __OdeCFL_OneSlice_impl_hpp__
#define __OdeCFL_OneSlice_impl_hpp__

#include <typedef.hpp>
#include <cstddef>
#include <vector>
#include <set> 
namespace levelset {

	class Term;
	class SchemeData;
	class OdeCFL_OneSlice_impl {
		FLOAT_TYPE t;
		beacls::FloatVec& sb_inv_out;
		beacls::FloatVec& local_derivMins;
		beacls::FloatVec& local_derivMaxs;
		const beacls::FloatVec& y;
		beacls::FloatVec& ydot;
		std::vector<beacls::FloatVec > derivMins;
		std::vector<beacls::FloatVec > derivMaxs;
		const size_t num_of_dimensions;
		const size_t first_dimension_loop_size;
		const size_t num_of_chunk_lines;
		const size_t num_of_slices;
		const size_t parallel_loop_size;
		const size_t num_of_outer_lines;
		const size_t num_of_inner_lines;
		const size_t parallel_line_index;
		const size_t second_dimension_loop_size;
		const size_t third_dimension_loop_size;
		const bool enable_user_defined_dynamics_on_gpu;
		const bool updateDerivMinMax;
		bool executeAgain;
		bool finished;
	public:
		bool get_executeAgain() const { return executeAgain; }
		bool is_finished() const { return finished; }
		void set_finished() { finished = true; }
		OdeCFL_OneSlice_impl(
			const FLOAT_TYPE t,
			beacls::FloatVec& sb_inv_out,
			beacls::FloatVec& local_derivMins,
			beacls::FloatVec& local_derivMaxs,
			const beacls::FloatVec& y,
			beacls::FloatVec& ydot,
			std::vector<beacls::FloatVec >& derivMins,
			std::vector<beacls::FloatVec >& derivMaxs,
			const size_t num_of_dimensions,
			const size_t first_dimension_loop_size,
			const size_t num_of_chunk_lines,
			const size_t num_of_slices,
			const size_t parallel_loop_size,
			const size_t num_of_outer_lines,
			const size_t num_of_inner_lines,
			const size_t parallel_line_index,
			const size_t second_dimension_loop_size,
			const size_t third_dimension_loop_size,
			const bool enable_user_defined_dynamics_on_gpu,
			const bool updateDerivMinMax
		);
		void execute(
			const Term* term,
			const SchemeData* schemeData,
			std::vector<beacls::FloatVec>& thread_local_derivMins,
			std::vector<beacls::FloatVec>& thread_local_derivMaxs,
			beacls::FloatVec& new_step_bound_invs
		);
		void execute_local_q(
			const Term* term,
			const SchemeData* schemeData,
			std::vector<beacls::FloatVec>& thread_local_derivMins,
			std::vector<beacls::FloatVec>& thread_local_derivMaxs,
			beacls::FloatVec& new_step_bound_invs,
			const std::set<size_t> &Q 
		);
	private:
		OdeCFL_OneSlice_impl();
		OdeCFL_OneSlice_impl(const OdeCFL_OneSlice_impl& rhs);
		OdeCFL_OneSlice_impl& operator=(const OdeCFL_OneSlice_impl& rhs);
	};
};
#endif	/* __OdeCFL_OneSlice_impl_hpp__ */

