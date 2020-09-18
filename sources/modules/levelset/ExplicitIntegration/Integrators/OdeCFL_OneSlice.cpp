#include "OdeCFL_OneSlice.hpp"
#include "OdeCFL_OneSlice_impl.hpp"
#include <functional>
#include <algorithm>
#include <levelset/ExplicitIntegration/Terms/Term.hpp>
#include <macro.hpp>

levelset::OdeCFL_OneSlice_impl::OdeCFL_OneSlice_impl(
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
	const bool updateDerivMinMax):
	t(t),
	sb_inv_out(sb_inv_out),
	local_derivMins(local_derivMins),
	local_derivMaxs(local_derivMaxs),
	y(y),
	ydot(ydot),
	derivMins(derivMins),
	derivMaxs(derivMaxs),
	num_of_dimensions(num_of_dimensions),
	first_dimension_loop_size(first_dimension_loop_size),
	num_of_chunk_lines(num_of_chunk_lines),
	num_of_slices(num_of_slices),
	parallel_loop_size(parallel_loop_size),
	num_of_outer_lines(num_of_outer_lines),
	num_of_inner_lines(num_of_inner_lines),
	parallel_line_index(parallel_line_index),
	second_dimension_loop_size(second_dimension_loop_size),
	third_dimension_loop_size(third_dimension_loop_size),
	enable_user_defined_dynamics_on_gpu(enable_user_defined_dynamics_on_gpu),
	updateDerivMinMax(updateDerivMinMax),
	executeAgain(false),
	finished(false)
{}

void levelset::OdeCFL_OneSlice_impl::execute(
		const Term* term,
		const SchemeData* schemeData,
		std::vector<beacls::FloatVec>& thread_local_derivMins,
		std::vector<beacls::FloatVec>& thread_local_derivMaxs,
		beacls::FloatVec& new_step_bound_invs) {

	size_t loop_local_index = 0;
	new_step_bound_invs.resize(num_of_dimensions*num_of_chunk_lines*num_of_slices);
	const size_t paralleled_out_line_size = std::min(parallel_loop_size, 
		(num_of_outer_lines > parallel_line_index * parallel_loop_size) ? 
		num_of_outer_lines - parallel_line_index * parallel_loop_size : 0);

	size_t actual_chunk_size = num_of_chunk_lines;
	size_t actual_num_of_slices = num_of_slices;

	for (size_t line_index = 0; 
		  line_index < paralleled_out_line_size*num_of_inner_lines; 
		  line_index += actual_chunk_size*actual_num_of_slices) {

		thread_local_derivMaxs = derivMaxs;
		thread_local_derivMins = derivMins;

		const size_t inner_line_index = line_index + 
		  parallel_line_index*parallel_loop_size*num_of_inner_lines;

		const size_t second_line_index 
		  = inner_line_index % second_dimension_loop_size;
		  
		actual_chunk_size = std::min(num_of_chunk_lines, second_dimension_loop_size 
			- second_line_index);

		const size_t third_line_index = 
		  (inner_line_index / second_dimension_loop_size) 
		  % third_dimension_loop_size;

		actual_num_of_slices = std::min(num_of_slices, third_dimension_loop_size 
			- third_line_index);

		const size_t line_begin = inner_line_index;
		beacls::FloatVec::iterator ydot_ite 
		  = ydot.begin() + line_begin * first_dimension_loop_size;

		if (term->execute(ydot_ite, new_step_bound_invs, t, y, 
				thread_local_derivMins, thread_local_derivMaxs, schemeData, line_begin, 
				actual_chunk_size, actual_num_of_slices, 
				enable_user_defined_dynamics_on_gpu, updateDerivMinMax)) {

			std::transform(sb_inv_out.cbegin(), sb_inv_out.cend(), 
				new_step_bound_invs.cbegin(), sb_inv_out.begin(), 
				std::ptr_fun<const FLOAT_TYPE&, 
				const FLOAT_TYPE&>(std::max<FLOAT_TYPE>));

			executeAgain |= false;
		}
		else {
			executeAgain |= true;
		}

		for (int dimension = 0; dimension < (int)num_of_dimensions; ++dimension) {
			beacls::FloatVec& cdMins_d = thread_local_derivMins[dimension];
			beacls::FloatVec& cdMaxs_d = thread_local_derivMaxs[dimension];

			const FLOAT_TYPE chunk_derivMin = 
			  beacls::min_value<FLOAT_TYPE>(cdMins_d.cbegin(), cdMins_d.cend());
			const FLOAT_TYPE chunk_derivMax = 
			  beacls::max_value<FLOAT_TYPE>(cdMaxs_d.cbegin(), cdMaxs_d.cend());

			FLOAT_TYPE& ldMins_d = local_derivMins[dimension];
			FLOAT_TYPE& ldMaxs_d = local_derivMaxs[dimension];

			ldMins_d = (loop_local_index != 0) ? 
			  std::min<FLOAT_TYPE>(ldMins_d, chunk_derivMin) : chunk_derivMin;
			ldMaxs_d = (loop_local_index != 0) ? 
			  std::max<FLOAT_TYPE>(ldMaxs_d, chunk_derivMax) : chunk_derivMax;
		}
		++loop_local_index;
	}
	term->synchronize(schemeData);
}

void levelset::OdeCFL_OneSlice_impl::execute_local_q(
		const Term* term,
		const SchemeData* schemeData,
		std::vector<beacls::FloatVec>& thread_local_derivMins,
		std::vector<beacls::FloatVec>& thread_local_derivMaxs,
		beacls::FloatVec& new_step_bound_invs, 
		const std::set<size_t>& Q) {

	size_t loop_local_index = 0;
	new_step_bound_invs.resize(num_of_dimensions*num_of_chunk_lines*num_of_slices);
	const size_t paralleled_out_line_size = std::min(parallel_loop_size, 
		(num_of_outer_lines > parallel_line_index * parallel_loop_size) ? 
		num_of_outer_lines - parallel_line_index * parallel_loop_size : 0);

	size_t actual_chunk_size = num_of_chunk_lines;
	size_t actual_num_of_slices = num_of_slices;

	for (size_t line_index = 0; 
		  line_index < paralleled_out_line_size*num_of_inner_lines; 
		  line_index += actual_chunk_size*actual_num_of_slices) {

		thread_local_derivMaxs = derivMaxs;
		thread_local_derivMins = derivMins;

		const size_t inner_line_index = line_index + 
		  parallel_line_index*parallel_loop_size*num_of_inner_lines;

		const size_t second_line_index 
		  = inner_line_index % second_dimension_loop_size;
		  
		actual_chunk_size = std::min(num_of_chunk_lines, second_dimension_loop_size 
			- second_line_index);

		const size_t third_line_index = 
		  (inner_line_index / second_dimension_loop_size) 
		  % third_dimension_loop_size;

		actual_num_of_slices = std::min(num_of_slices, third_dimension_loop_size 
			- third_line_index);

		const size_t line_begin = inner_line_index;
		beacls::FloatVec::iterator ydot_ite 
		  = ydot.begin() + line_begin * first_dimension_loop_size;

		if (term->execute_local_q(ydot_ite, new_step_bound_invs, t, y, 
				thread_local_derivMins, thread_local_derivMaxs, schemeData, line_begin, 
				actual_chunk_size, Q, actual_num_of_slices,
				enable_user_defined_dynamics_on_gpu, updateDerivMinMax)) {

			std::transform(sb_inv_out.cbegin(), sb_inv_out.cend(), 
				new_step_bound_invs.cbegin(), sb_inv_out.begin(), 
				std::ptr_fun<const FLOAT_TYPE&, 
				const FLOAT_TYPE&>(std::max<FLOAT_TYPE>));

			executeAgain |= false;
		}
		else {
			executeAgain |= true;
		}

		for (int dimension = 0; dimension < (int)num_of_dimensions; ++dimension) {
			beacls::FloatVec& cdMins_d = thread_local_derivMins[dimension];
			beacls::FloatVec& cdMaxs_d = thread_local_derivMaxs[dimension];

			const FLOAT_TYPE chunk_derivMin = 
			  beacls::min_value<FLOAT_TYPE>(cdMins_d.cbegin(), cdMins_d.cend());
			const FLOAT_TYPE chunk_derivMax = 
			  beacls::max_value<FLOAT_TYPE>(cdMaxs_d.cbegin(), cdMaxs_d.cend());

			FLOAT_TYPE& ldMins_d = local_derivMins[dimension];
			FLOAT_TYPE& ldMaxs_d = local_derivMaxs[dimension];

			ldMins_d = (loop_local_index != 0) ? 
			  std::min<FLOAT_TYPE>(ldMins_d, chunk_derivMin) : chunk_derivMin;
			ldMaxs_d = (loop_local_index != 0) ? 
			  std::max<FLOAT_TYPE>(ldMaxs_d, chunk_derivMax) : chunk_derivMax;
		}
		++loop_local_index;
	}
	term->synchronize(schemeData);
}

void levelset::OdeCFL_OneSlice::execute(
		const Term* term,
		const SchemeData* schemeData,
		std::vector<beacls::FloatVec>& thread_local_derivMins,
		std::vector<beacls::FloatVec>& thread_local_derivMaxs,
		beacls::FloatVec& new_step_bound_invs) {

	if (pimpl) pimpl->execute(term, schemeData, thread_local_derivMins, thread_local_derivMaxs, new_step_bound_invs);
}

void levelset::OdeCFL_OneSlice::execute_local_q(
		const Term* term,
		const SchemeData* schemeData,
		std::vector<beacls::FloatVec>& thread_local_derivMins,
		std::vector<beacls::FloatVec>& thread_local_derivMaxs,
		beacls::FloatVec& new_step_bound_invs, 
		const std::set<size_t> &Q) {
	if (pimpl) pimpl->execute_local_q(term, schemeData, thread_local_derivMins, thread_local_derivMaxs, new_step_bound_invs, Q);
}

bool levelset::OdeCFL_OneSlice::get_executeAgain() const {
	if (pimpl) return pimpl->get_executeAgain();
	else return false;
}
bool levelset::OdeCFL_OneSlice::is_finished() const {
	if (pimpl) return pimpl->is_finished();
	else return false;
}
void levelset::OdeCFL_OneSlice::set_finished() { 
	if (pimpl) pimpl->set_finished();
}

levelset::OdeCFL_OneSlice::OdeCFL_OneSlice(
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
		const bool updateDerivMinMax) {

	pimpl = new OdeCFL_OneSlice_impl(
		t,
		sb_inv_out,
		local_derivMins,
		local_derivMaxs,
		y,
		ydot,
		derivMins,
		derivMaxs,
		num_of_dimensions,
		first_dimension_loop_size,
		num_of_chunk_lines,
		num_of_slices,
		parallel_loop_size,
		num_of_outer_lines,
		num_of_inner_lines,
		parallel_line_index,
		second_dimension_loop_size,
		third_dimension_loop_size,
		enable_user_defined_dynamics_on_gpu,
		updateDerivMinMax);
}
levelset::OdeCFL_OneSlice::~OdeCFL_OneSlice()
{
	if (pimpl) delete pimpl;
}


