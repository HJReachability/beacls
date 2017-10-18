#include <levelset/ExplicitIntegration/Integrators/OdeCFL1.hpp>
#include <levelset/Grids/HJI_Grid.hpp>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <functional>
#include <cfloat>
#include <deque>
#include <thread>
#include <macro.hpp>
#include <levelset/ExplicitIntegration/Terms/Term.hpp>
#include <levelset/ExplicitIntegration/SchemeData.hpp>
#include <levelset/SpatialDerivative/SpatialDerivative.hpp>
#include "OdeCFL1_impl.hpp"

#include "odeCFL_SubStep.hpp"
#include "OdeCFL_CommandQueue.hpp"
#include "OdeCFL_Worker.hpp"
using namespace levelset;
OdeCFL1_impl::OdeCFL1_impl(
	const Term *schemeFunc,
	const FLOAT_TYPE factor_cfl,
	const FLOAT_TYPE max_step,
	const std::vector<levelset::PostTimestep_Exec_Type*> &post_time_steps,
	const bool single_step,
	const bool stats,
	const levelset::TerminalEvent_Exec_Type* terminalEvent) :
	term(schemeFunc->clone()),
	factor_cfl(factor_cfl),
	max_step(max_step),
	post_time_steps(post_time_steps),
	single_step(single_step),
	stats(stats),
	terminalEvent(terminalEvent) {
	levelset::OdeCFL_CommandQueue* commandQueue = new levelset::OdeCFL_CommandQueue;
	commandQueues.push_back(commandQueue);
	levelset::OdeCFL_Worker* worker = new  levelset::OdeCFL_Worker(commandQueue, term, 0);
	workers.push_back(worker);
	worker->run();
}
OdeCFL1_impl::~OdeCFL1_impl() {
	std::for_each(workers.begin(), workers.end(), [](auto& rhs) {
		if (rhs) {
			rhs->terminate();
			delete rhs;
		}
	});
	std::for_each(commandQueues.begin(), commandQueues.end(), [](auto& rhs) {
		if (rhs) delete rhs;
	});
	if (term) delete term;
}
OdeCFL1_impl::OdeCFL1_impl(const OdeCFL1_impl& rhs) :
	term(rhs.term->clone()),
	factor_cfl(rhs.factor_cfl),
	max_step(rhs.max_step),
	post_time_steps(rhs.post_time_steps),
	single_step(rhs.single_step),
	stats(rhs.stats),
	terminalEvent(rhs.terminalEvent),
	ydot(rhs.ydot),
	yOld(rhs.yOld),
	step_bound_invss(rhs.step_bound_invss),
	step_bound_invs(rhs.step_bound_invs),
	eventValuesOld(rhs.eventValuesOld),
	derivMins(rhs.derivMins),
	derivMaxs(rhs.derivMaxs),
	local_derivMinss(rhs.local_derivMinss),
	local_derivMaxss(rhs.local_derivMaxss),
	lastDerivMaxs(rhs.lastDerivMaxs),
	lastDerivMins(rhs.lastDerivMins),
	executeAgains(rhs.executeAgains)
{
	commandQueues.resize(rhs.commandQueues.size());
	std::for_each(commandQueues.begin(), commandQueues.end(), [this](auto& rhs) {
		rhs = new levelset::OdeCFL_CommandQueue;
	});
	workers.resize(rhs.workers.size());
	std::transform(commandQueues.cbegin(), commandQueues.cend(), rhs.workers.cbegin(), workers.begin(), [this](const auto& lhs, const auto& rhs) {
		levelset::OdeCFL_Worker* worker = new levelset::OdeCFL_Worker(lhs, term, rhs->get_gpu_id());
		worker->run();
		return worker;
	});
}
FLOAT_TYPE OdeCFL1_impl::execute(
	beacls::FloatVec& y,
	const beacls::FloatVec& tspan,
	const beacls::FloatVec& y0,
	const SchemeData *schemeData,
	const size_t line_length_of_chunk,
	const size_t num_of_threads,
	const size_t num_of_gpus,
	const levelset::DelayedDerivMinMax_Type delayedDerivMinMax,
	const bool enable_user_defined_dynamics_on_gpu)
{
	const HJI_Grid* grid = schemeData->get_grid();
	if (!grid) return false;

	// How close (relative) do we need to be to the final time?
	const double eps = std::numeric_limits<double>::epsilon();	//!< 
	const double small = 100.0 * eps;

	const size_t num_of_elements = grid->get_sum_of_elems();

	//---------------------------------------------------------------------------
	// Create cell array with array indices.
	ydot.resize(num_of_elements);

	//---------------------------------------------------------------------------
	// This routine includes multiple substeps, and the CFL restricted timestep
	//   size is chosen on the first substep.  Subsequent substeps may violate
	//   CFL slightly; how much should be allowed before generating a warning?

	FLOAT_TYPE t = tspan[0];
	int steps = 0;
	FLOAT_TYPE step_bound_inv = 1.0;

	FLOAT_TYPE deltaT = 0.0;
	const size_t num_of_dimensions = grid->get_num_of_dimensions();
	const beacls::IntegerVec& Ns = grid->get_Ns();
	const size_t first_dimension_loop_size = (num_of_dimensions >= 1) ? Ns[0] : 1;
	const size_t second_dimension_loop_size = (num_of_dimensions >= 2) ? Ns[1] : 1;
	const size_t third_dimension_loop_size = (num_of_dimensions >= 3) ? Ns[2] : 1;
	const size_t num_of_lines = num_of_elements / first_dimension_loop_size;
	const size_t num_of_outer_lines = std::accumulate(Ns.cbegin() + 2, Ns.cend(), (size_t)1,
		[](const auto& lhs, const auto& rhs) {return lhs * rhs; });

	const size_t num_of_activated_gpus = (num_of_gpus == 0) ? beacls::get_num_of_gpus() : num_of_gpus;
	const size_t hardware_concurrency = (std::thread::hardware_concurrency() == 0) ? 1 : std::thread::hardware_concurrency();
	const SpatialDerivative* spatialDerivative = schemeData->get_spatialDerivative();
	const size_t actual_num_of_threads = (num_of_threads == 0) ? ((spatialDerivative->get_type() == beacls::UVecType_Cuda) ? num_of_activated_gpus : hardware_concurrency) : num_of_threads;

	const size_t num_of_parallel_loop_lines = std::min((size_t)actual_num_of_threads, num_of_outer_lines);
	const size_t parallel_loop_size = (size_t)std::ceil((FLOAT_TYPE)num_of_outer_lines / num_of_parallel_loop_lines);
	const size_t num_of_inner_lines = (size_t)std::ceil((FLOAT_TYPE)num_of_lines / num_of_outer_lines);

	const double gpu_memory_ocupancy_ratio = 1.0 / num_of_dimensions / 64;
	const size_t minimum_global_memory_in_devices = beacls::get_minimum_global_memory_in_devices();
	const size_t available_line_length_for_cuda = (size_t)std::floor(minimum_global_memory_in_devices * gpu_memory_ocupancy_ratio / first_dimension_loop_size / sizeof(FLOAT_TYPE));
	const size_t available_line_length_of_chunk_for_cuda = (size_t)(std::ceil(std::floor((FLOAT_TYPE)available_line_length_for_cuda / actual_num_of_threads) / second_dimension_loop_size)*second_dimension_loop_size);
	const size_t prefered_line_length_of_chunk_for_cuda = (size_t)(std::ceil(std::ceil((FLOAT_TYPE)num_of_lines / actual_num_of_threads) / second_dimension_loop_size)*second_dimension_loop_size) * num_of_activated_gpus;
	const size_t line_length_of_chunk_for_cuda = available_line_length_of_chunk_for_cuda < prefered_line_length_of_chunk_for_cuda ? available_line_length_of_chunk_for_cuda : prefered_line_length_of_chunk_for_cuda;
	const size_t prefered_line_length_of_chunk_for_cpu = (size_t)std::ceil((FLOAT_TYPE)1024 / first_dimension_loop_size);
	const size_t prefered_line_length_of_chunk = (spatialDerivative->get_type() == beacls::UVecType_Cuda) ? line_length_of_chunk_for_cuda : prefered_line_length_of_chunk_for_cpu;

	const size_t actual_line_length_of_chunk = (line_length_of_chunk == 0) ? prefered_line_length_of_chunk : line_length_of_chunk;	//!< T.B.D.

	const size_t num_of_chunk_lines = (actual_line_length_of_chunk < second_dimension_loop_size) ? actual_line_length_of_chunk : second_dimension_loop_size;
	const size_t num_of_slices = (actual_line_length_of_chunk < second_dimension_loop_size) ? 1 : (size_t)std::floor((FLOAT_TYPE)actual_line_length_of_chunk / second_dimension_loop_size);

	if (step_bound_invss.size() != num_of_parallel_loop_lines) step_bound_invss.resize(num_of_parallel_loop_lines);
	if (local_derivMinss.size() != num_of_parallel_loop_lines) local_derivMinss.resize(num_of_parallel_loop_lines);
	if (local_derivMaxss.size() != num_of_parallel_loop_lines) local_derivMaxss.resize(num_of_parallel_loop_lines);
	if (derivMins.size() != num_of_dimensions) derivMins.resize(num_of_dimensions);
	if (derivMaxs.size() != num_of_dimensions) derivMaxs.resize(num_of_dimensions);
	if (lastDerivMins.size() != num_of_dimensions) lastDerivMins.resize(num_of_dimensions);
	if (lastDerivMaxs.size() != num_of_dimensions) lastDerivMaxs.resize(num_of_dimensions);
	if (executeAgains.size() != num_of_parallel_loop_lines) executeAgains.resize(num_of_parallel_loop_lines);
	if (workers.size() < actual_num_of_threads) workers.resize(actual_num_of_threads, NULL);
	if (commandQueues.size() < actual_num_of_threads) commandQueues.resize(actual_num_of_threads, NULL);
	for (size_t thread_id = 0; thread_id < actual_num_of_threads; ++thread_id) {
		levelset::OdeCFL_CommandQueue* commandQueue = commandQueues[thread_id];
		if (commandQueue == NULL) {
			commandQueue = new levelset::OdeCFL_CommandQueue();
			commandQueues[thread_id] = commandQueue;
		}
		levelset::OdeCFL_Worker* worker = workers[thread_id];
		if (worker == NULL) {
			worker = new levelset::OdeCFL_Worker(commandQueue, term, (num_of_activated_gpus > 1) ? (int)thread_id % num_of_activated_gpus : 0);
			worker->run();
			workers[thread_id] = worker;
		}
		worker->set_schemeData(schemeData);
	}

	std::for_each(step_bound_invss.begin(), step_bound_invss.end(), ([num_of_dimensions](beacls::FloatVec &i) {i.resize(num_of_dimensions, 0.); }));

	FLOAT_TYPE tOld = t;
	std::for_each(local_derivMinss.begin(), local_derivMinss.end(), ([num_of_dimensions](auto& rhs) {
		if (rhs.size() != num_of_dimensions) rhs.resize(num_of_dimensions);
	}));
	std::for_each(local_derivMaxss.begin(), local_derivMaxss.end(), ([num_of_dimensions](auto& rhs) {
		if (rhs.size() != num_of_dimensions) rhs.resize(num_of_dimensions);
	}));

	y = y0;

	while ((tspan[1] - t) >= (small * HjiFabs<FLOAT_TYPE>(tspan[1]))) {
		//	If there is a terminal event function registered, we need
		// to maintain the info from the last timestep.
		if (terminalEvent) {
			yOld = y;
			tOld = t;
		}
		// -----------------------------------------------------------

		// First substep: Forward Euler from t_n to t_{n+1}.
		// Approximate the derivative and CFL restriction.
		levelset::odeCFL_SubStep(
			commandQueues,
			t,
			step_bound_invss,
			local_derivMinss,
			local_derivMaxss,
			y,
			ydot,
			derivMins,
			derivMaxs,
			lastDerivMins,
			lastDerivMaxs,
			executeAgains,
			num_of_dimensions,
			first_dimension_loop_size,
			num_of_chunk_lines,
			num_of_slices,
			parallel_loop_size,
			num_of_outer_lines,
			num_of_inner_lines,
			second_dimension_loop_size,
			third_dimension_loop_size,
			num_of_parallel_loop_lines,
			actual_num_of_threads,
			delayedDerivMinMax,
			enable_user_defined_dynamics_on_gpu
		);
		step_bound_invs.assign(num_of_dimensions, 0.);
		for_each(step_bound_invss.cbegin(), step_bound_invss.cend(), [this](const auto& rhs) {
			std::transform(rhs.cbegin(), rhs.cend(), step_bound_invs.cbegin(), step_bound_invs.begin(), std::ptr_fun<const FLOAT_TYPE&, const FLOAT_TYPE&>(std::max<FLOAT_TYPE>));
		});
		step_bound_inv = std::accumulate(step_bound_invs.cbegin(), step_bound_invs.cend(), static_cast<FLOAT_TYPE>(0));
		FLOAT_TYPE step_bound = (FLOAT_TYPE)(1. / step_bound_inv);

		// -----------------------------------------------------------
		// Determine CFL bound on timestep, but not beyond the final time.
		//   For vector level sets, use the most restrictive stepBound.
		//   We'll use this fixed timestep for both substeps.
		deltaT = HjiMin(factor_cfl * step_bound, tspan[1] - t);
		deltaT = HjiMin(deltaT, max_step);

		// Take the first substep.
		t = t + deltaT;
		std::transform(y.cbegin(), y.cend(), ydot.cbegin(), y.begin(), ([&deltaT](const FLOAT_TYPE &y_i, const FLOAT_TYPE &ydot_i) {
			return y_i + deltaT * ydot_i;
		}));

		steps++;

		//! If there is one or more post-timestep routines, call them.
		if (!post_time_steps.empty()) {
			for_each(post_time_steps.cbegin(), post_time_steps.cend(), ([&t,&y,&grid](const auto& func) {
				if (func) (*func)(y, t, y, grid);
			}));
		}
		//!< Back up last max/min derivatives
		if (delayedDerivMinMax) {
			lastDerivMins = derivMins;
			lastDerivMaxs = derivMaxs;
		}
		//! If we are in single step mode, then do not repeat.
		if (single_step) break;

		//! If there is a terminal event function, establish initial sign
		if (terminalEvent) {
			beacls::FloatVec eventValues;
			(*terminalEvent)(eventValues, t, y, tOld, yOld, grid);
			if ((steps > 1) && 
				!std::equal(eventValues.cbegin(), eventValues.cend(), eventValuesOld.cbegin())
				) {
				break;
			}
			else {
				eventValuesOld = eventValues;
			}
		}
	}
	if (stats) {
		printf("\t%d steps from %g to %g (deltaT=%g)\n", steps, tspan[0], t, deltaT);
	}
	return t;
}


OdeCFL1::OdeCFL1(
	const Term *schemeFunc,
	const FLOAT_TYPE factor_cfl,
	const FLOAT_TYPE max_step,
	const std::vector<levelset::PostTimestep_Exec_Type*> &post_time_steps,
	const bool single_step,
	const bool stats,
	const levelset::TerminalEvent_Exec_Type* terminalEvent) {
	pimpl = new OdeCFL1_impl(schemeFunc, factor_cfl, max_step, post_time_steps, single_step, stats, terminalEvent);
}
OdeCFL1::~OdeCFL1() {
	if (pimpl) delete pimpl;
}
FLOAT_TYPE OdeCFL1::execute(
	beacls::FloatVec& y,
	const beacls::FloatVec& tspan,
	const beacls::FloatVec& y0,
	const SchemeData *schemeData,
	const size_t line_length_of_chunk,
	const size_t num_of_threads,
	const size_t num_of_gpus,
	const levelset::DelayedDerivMinMax_Type delayedDerivMinMax,
	const bool enable_user_defined_dynamics_on_gpu)
{
	if (pimpl) return pimpl->execute(y, tspan, y0, schemeData, line_length_of_chunk, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu);
	else return 0;
}
OdeCFL1::OdeCFL1(const OdeCFL1& rhs) :
	pimpl(rhs.pimpl->clone())
{
}

OdeCFL1* OdeCFL1::clone() const {
	return new OdeCFL1(*this);
}
