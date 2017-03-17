#ifndef __OdeCFL_OneSlice_hpp__
#define __OdeCFL_OneSlice_hpp__

#include <typedef.hpp>
#include <cstddef>
#include <vector>
namespace levelset {
	class Term;
	class SchemeData;
	class OdeCFL_OneSlice_impl;
	class OdeCFL_OneSlice {
	public:
		OdeCFL_OneSlice_impl* pimpl;
		bool get_executeAgain() const;
		bool is_finished() const;
		void set_finished();
		OdeCFL_OneSlice(
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
			const Term* thread_local_term,
			const SchemeData* thread_local_schemeData,
			std::vector<beacls::FloatVec>& thread_local_derivMins,
			std::vector<beacls::FloatVec>& thread_local_derivMaxs,
			beacls::FloatVec& new_step_bound_invs
		);
		~OdeCFL_OneSlice();
	private:
		OdeCFL_OneSlice();
		OdeCFL_OneSlice(const OdeCFL_OneSlice& rhs);
		OdeCFL_OneSlice& operator=(const OdeCFL_OneSlice& rhs);
	};
};
#endif	/* __OdeCFL_OneSlice_hpp__ */

