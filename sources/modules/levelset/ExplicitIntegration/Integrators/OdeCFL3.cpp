#include <levelset/ExplicitIntegration/Integrators/OdeCFL3.hpp>
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
#include "OdeCFL3_impl.hpp"

#include "odeCFL_SubStep.hpp"
#include "OdeCFL_CommandQueue.hpp"
#include "OdeCFL_Worker.hpp"
//#define PARALLEL_Y
using namespace levelset;

OdeCFL3_impl::OdeCFL3_impl(
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
OdeCFL3_impl::~OdeCFL3_impl() {
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
OdeCFL3_impl::OdeCFL3_impl(const OdeCFL3_impl& rhs) :
  term(rhs.term->clone()),
  factor_cfl(rhs.factor_cfl),
  max_step(rhs.max_step),
  post_time_steps(rhs.post_time_steps),
  single_step(rhs.single_step),
  stats(rhs.stats),
  terminalEvent(rhs.terminalEvent),
  y1(rhs.y1),
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
FLOAT_TYPE OdeCFL3_impl::execute(
    beacls::FloatVec& y,
    const beacls::FloatVec& tspan,
    const beacls::FloatVec& y0,
    const SchemeData *schemeData,
    const size_t line_length_of_chunk,
    const size_t num_of_threads,
    const size_t num_of_gpus,
    const levelset::DelayedDerivMinMax_Type delayedDerivMinMax,
    const bool enable_user_defined_dynamics_on_gpu) {

  const HJI_Grid* grid = schemeData->get_grid();
  if (!grid) return false;

  // How close (relative) do we need to be to the final time?
  const double eps = std::numeric_limits<double>::epsilon();  //!< 
  const double small = 100.0 * eps;


  const size_t num_of_elements = grid->get_numel();

  //---------------------------------------------------------------------------
  // Create cell array with array indices.
  y1.resize(num_of_elements);
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

  // size_t third_dimension_loop_size = 1;
  // if (num_of_dimensions >= 3) {
  //   for (size_t k = 2; k < num_of_dimensions; ++k) {
  //     third_dimension_loop_size *= Ns[k];
  //   }
  // }
  const size_t third_dimension_loop_size = (num_of_dimensions >= 3) ? Ns[2] : 1;

  const size_t num_of_lines = num_of_elements / first_dimension_loop_size;
  const size_t num_of_outer_lines = std::accumulate(Ns.cbegin() + 2, Ns.cend(), 
    (size_t)1, [](const auto& lhs, const auto& rhs) {return lhs * rhs; });

  const size_t num_of_activated_gpus = (num_of_gpus == 0) ? 
    beacls::get_num_of_gpus() : num_of_gpus;
  const size_t hardware_concurrency = (std::thread::hardware_concurrency() == 0) ? 
    1 : std::thread::hardware_concurrency();
  const SpatialDerivative* spatialDerivative = schemeData->get_spatialDerivative();
  const size_t actual_num_of_threads = (num_of_threads == 0) ? 
    ((spatialDerivative->get_type() == beacls::UVecType_Cuda) ? 
    num_of_activated_gpus : hardware_concurrency ): num_of_threads;

  const size_t num_of_parallel_loop_lines = 
    std::min((size_t)actual_num_of_threads, num_of_outer_lines);
  const size_t parallel_loop_size = 
    (size_t)std::ceil((FLOAT_TYPE)num_of_outer_lines / num_of_parallel_loop_lines);
  const size_t num_of_inner_lines = 
    (size_t)std::ceil((FLOAT_TYPE)num_of_lines / num_of_outer_lines);

  const size_t prefered_line_length_of_chunk_for_cuda = 
    (size_t)(std::ceil(std::ceil((FLOAT_TYPE)num_of_lines / actual_num_of_threads)/ second_dimension_loop_size)*second_dimension_loop_size);
  const size_t prefered_line_length_of_chunk_for_cpu = 
    (size_t)std::ceil((FLOAT_TYPE)1024 / first_dimension_loop_size);
  const size_t prefered_line_length_of_chunk = 
    (spatialDerivative->get_type() == beacls::UVecType_Cuda) ? 
    prefered_line_length_of_chunk_for_cuda : prefered_line_length_of_chunk_for_cpu;

  const size_t actual_line_length_of_chunk = (line_length_of_chunk == 0) ? 
    prefered_line_length_of_chunk : line_length_of_chunk; //!< T.B.D.

  const size_t num_of_chunk_lines = 
    (actual_line_length_of_chunk < second_dimension_loop_size) ? 
    actual_line_length_of_chunk : second_dimension_loop_size;
  const size_t num_of_slices = 
    (actual_line_length_of_chunk < second_dimension_loop_size) ? 
    1 : (size_t)std::floor((FLOAT_TYPE)actual_line_length_of_chunk / 
    second_dimension_loop_size);

  if (step_bound_invss.size() != num_of_parallel_loop_lines) 
    step_bound_invss.resize(num_of_parallel_loop_lines);
  if (local_derivMinss.size() != num_of_parallel_loop_lines) 
    local_derivMinss.resize(num_of_parallel_loop_lines);
  if (local_derivMaxss.size() != num_of_parallel_loop_lines) 
    local_derivMaxss.resize(num_of_parallel_loop_lines);

  if (derivMins.size() != num_of_dimensions) derivMins.resize(num_of_dimensions);
  if (derivMaxs.size() != num_of_dimensions) derivMaxs.resize(num_of_dimensions);
  if (lastDerivMins.size() != num_of_dimensions) 
    lastDerivMins.resize(num_of_dimensions);
  if (lastDerivMaxs.size() != num_of_dimensions) 
    lastDerivMaxs.resize(num_of_dimensions);
  if (executeAgains.size() != num_of_parallel_loop_lines) 
    executeAgains.resize(num_of_parallel_loop_lines);
  if (workers.size() < actual_num_of_threads) 
    workers.resize(actual_num_of_threads, NULL);
  if (commandQueues.size() < actual_num_of_threads) 
    commandQueues.resize(actual_num_of_threads, NULL);

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

  std::for_each(step_bound_invss.begin(), step_bound_invss.end(), 
    ([num_of_dimensions](beacls::FloatVec &i) {i.resize(num_of_dimensions, 0.); }));

  FLOAT_TYPE tOld = t;
  std::for_each(local_derivMinss.begin(), local_derivMinss.end(), 
    ([num_of_dimensions](auto& rhs) { 
    if (rhs.size() != num_of_dimensions) rhs.resize(num_of_dimensions); }));
  std::for_each(local_derivMaxss.begin(), local_derivMaxss.end(), 
    ([num_of_dimensions](auto& rhs) { 
    if (rhs.size() != num_of_dimensions) rhs.resize(num_of_dimensions); }));

  y = y0;

  while ((tspan[1] - t) >= (small * HjiFabs<FLOAT_TYPE>(tspan[1]))) {
    //  If there is a terminal event function registered, we need
    // to maintain the info from the last timestep.
    if (terminalEvent) {
      yOld = y;
      tOld = t;
    }
    // -----------------------------------------------------------
    // First substep: Forward Euler from t_n to t_{n+1}.
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
      enable_user_defined_dynamics_on_gpu);

    step_bound_invs.assign(num_of_dimensions, 0.);
    std::for_each(step_bound_invss.cbegin(), step_bound_invss.cend(), [this](const auto& rhs) {
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
    const FLOAT_TYPE t1 = t + deltaT;
#if defined(PARALLEL_Y)
    const int ydot_size = (int)ydot.size();
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < ydot_size; ++i) {
      y1[i] = y[i] + deltaT * ydot[i];
    }
#else

    std::transform(y.cbegin(), y.cend(), ydot.cbegin(), y1.begin(), 
      ([&deltaT](const FLOAT_TYPE &y_i, const FLOAT_TYPE &ydot_i) {
      return y_i + deltaT * ydot_i; }));
#endif

    // -----------------------------------------------------------
    // Second substep: Forward Euler from t_{n+1} to t_{n+2}.
    // Approximate the derivative.
    levelset::odeCFL_SubStep(
      commandQueues,
      t1,
      step_bound_invss,
      local_derivMinss,
      local_derivMaxss,
      y1,
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
#if defined(PARALLEL_Y)
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < ydot_size; ++i) {
#else
    for (size_t i = 0; i < ydot.size(); ++i) {
#endif
      const FLOAT_TYPE y2_i = y1[i] + deltaT * ydot[i];
      y1[i] = (3 * y[i] + y2_i) / 4;
    }
    step_bound_invs.assign(num_of_dimensions, 0.);
    std::for_each(step_bound_invss.cbegin(), step_bound_invss.cend(), 
      [this](const auto& rhs) {std::transform(rhs.cbegin(), rhs.cend(), 
      step_bound_invs.cbegin(), step_bound_invs.begin(), 
      std::ptr_fun<const FLOAT_TYPE&, const FLOAT_TYPE&>(std::max<FLOAT_TYPE>));
    });
    step_bound_inv = std::accumulate(step_bound_invs.begin(), 
      step_bound_invs.end(), static_cast<FLOAT_TYPE>(0));
    step_bound = (FLOAT_TYPE)(1. / step_bound_inv);


    
    // Check CFL bound on timestep :
    // If the timestep chosen on the first substep violates
    //  the CFL condition by a significant amount, throw a warning.
    // For vector level sets, use the most restrictive stepBound.
    // Occasional failure should not cause too many problems.
    if (deltaT > std::min<FLOAT_TYPE>(step_bound, 
        (FLOAT_TYPE)(1.2 * factor_cfl * step_bound))) {
      const FLOAT_TYPE violation = deltaT / step_bound;
      printf("Second substep violated CFL; effective number %lf\n", violation);
    }

    // Take the second substep.
    const FLOAT_TYPE t2 = t1 + deltaT;
    // Combine t_n and t_{n+2} to get approximation at t_{n+1/2}
    const FLOAT_TYPE tHalf = (FLOAT_TYPE)(0.25 * (3 * t + t2));
    const FLOAT_TYPE one_third = (FLOAT_TYPE)( 1. / 3.);
    // -----------------------------------------------------------
    // Third substep: Forward Euler from t_{n+1/2} to t_{n+3/2}.
    // Approximate the derivative.
    levelset::odeCFL_SubStep(
        commandQueues,
        tHalf,
        step_bound_invss,
        local_derivMinss,
        local_derivMaxss,
        y1,
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
        enable_user_defined_dynamics_on_gpu);
#if defined(PARALLEL_Y)
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < ydot_size; ++i) {
#else
    for (size_t i = 0; i < ydot.size(); ++i) {
#endif
      const FLOAT_TYPE y3_i = y1[i] + deltaT * ydot[i];
      y[i] = one_third * (y[i] + 2 * y3_i);
    }
    step_bound_invs.assign(num_of_dimensions, 0.);
    std::for_each(step_bound_invss.cbegin(), step_bound_invss.cend(), [this](const auto& rhs) {
      std::transform(rhs.cbegin(), rhs.cend(), step_bound_invs.cbegin(), step_bound_invs.begin(), std::ptr_fun<const FLOAT_TYPE&, const FLOAT_TYPE&>(std::max<FLOAT_TYPE>));
    });
    step_bound_inv = std::accumulate(step_bound_invs.begin(), step_bound_invs.end(), static_cast<FLOAT_TYPE>(0));
    step_bound = (FLOAT_TYPE)(1. / step_bound_inv);
    // Check CFL bound on timestep :
    // If the timestep chosen on the first substep violates
    //  the CFL condition by a significant amount, throw a warning.
    // For vector level sets, use the most restrictive stepBound.
    // Occasional failure should not cause too many problems.
    if (deltaT > std::min<FLOAT_TYPE>(step_bound, 
        (FLOAT_TYPE)(1.2 * factor_cfl * step_bound))) {

      const FLOAT_TYPE violation = deltaT / step_bound;
      printf("Third substep violated CFL; effective number %lf\n", violation);
    }

    // Take the third substep.
    const FLOAT_TYPE tThreeHalf = tHalf + deltaT;
    // Average t_n and t_{n+2} to get second order approximation of t_{n+1}.
    t =(t + 2 * tThreeHalf) / 3;

    steps++;
    //! If there is one or more post-timestep routines, call them.
    if (!post_time_steps.empty()) {
      std::for_each(post_time_steps.cbegin(), post_time_steps.cend(), ([&t,&y,&grid](const auto& func) {
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

FLOAT_TYPE OdeCFL3_impl::execute_local_q(
    beacls::FloatVec& y,
    const beacls::FloatVec& tspan,
    const beacls::FloatVec& y0,
    const SchemeData *schemeData,
    const size_t line_length_of_chunk,
    const size_t num_of_threads,
    const size_t num_of_gpus,
    const levelset::DelayedDerivMinMax_Type delayedDerivMinMax,
    const bool enable_user_defined_dynamics_on_gpu, 
    const std::set<size_t>& Q) {

  const HJI_Grid* grid = schemeData->get_grid();
  if (!grid) return false;

  // How close (relative) do we need to be to the final time?
  const double eps = std::numeric_limits<double>::epsilon();  //!< 
  const double small = 100.0 * eps;


  const size_t num_of_elements = grid->get_numel();

  //---------------------------------------------------------------------------
  // Create cell array with array indices.
  y1.resize(num_of_elements);
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

  // size_t third_dimension_loop_size = 1;
  // if (num_of_dimensions >= 3) {
  //   for (size_t k = 2; k < num_of_dimensions; ++k) {
  //     third_dimension_loop_size *= Ns[k];
  //   }
  // }
  const size_t third_dimension_loop_size = (num_of_dimensions >= 3) ? Ns[2] : 1;

  const size_t num_of_lines = num_of_elements / first_dimension_loop_size;
  const size_t num_of_outer_lines = std::accumulate(Ns.cbegin() + 2, Ns.cend(), 
    (size_t)1, [](const auto& lhs, const auto& rhs) {return lhs * rhs; });

  const size_t num_of_activated_gpus = (num_of_gpus == 0) ? 
    beacls::get_num_of_gpus() : num_of_gpus;
  const size_t hardware_concurrency = (std::thread::hardware_concurrency() == 0) ? 
    1 : std::thread::hardware_concurrency();
  const SpatialDerivative* spatialDerivative = schemeData->get_spatialDerivative();
  const size_t actual_num_of_threads = (num_of_threads == 0) ? 
    ((spatialDerivative->get_type() == beacls::UVecType_Cuda) ? 
    num_of_activated_gpus : hardware_concurrency ): num_of_threads;

  const size_t num_of_parallel_loop_lines = 
    std::min((size_t)actual_num_of_threads, num_of_outer_lines);
  const size_t parallel_loop_size = 
    (size_t)std::ceil((FLOAT_TYPE)num_of_outer_lines / num_of_parallel_loop_lines);
  const size_t num_of_inner_lines = 
    (size_t)std::ceil((FLOAT_TYPE)num_of_lines / num_of_outer_lines);

  const size_t prefered_line_length_of_chunk_for_cuda = 
    (size_t)(std::ceil(std::ceil((FLOAT_TYPE)num_of_lines / actual_num_of_threads)/ second_dimension_loop_size)*second_dimension_loop_size);
  const size_t prefered_line_length_of_chunk_for_cpu = 
    (size_t)std::ceil((FLOAT_TYPE)1024 / first_dimension_loop_size);
  const size_t prefered_line_length_of_chunk = 
    (spatialDerivative->get_type() == beacls::UVecType_Cuda) ? 
    prefered_line_length_of_chunk_for_cuda : prefered_line_length_of_chunk_for_cpu;

  const size_t actual_line_length_of_chunk = (line_length_of_chunk == 0) ? 
    prefered_line_length_of_chunk : line_length_of_chunk; //!< T.B.D.

  const size_t num_of_chunk_lines = 
    (actual_line_length_of_chunk < second_dimension_loop_size) ? 
    actual_line_length_of_chunk : second_dimension_loop_size;
  const size_t num_of_slices = 
    (actual_line_length_of_chunk < second_dimension_loop_size) ? 
    1 : (size_t)std::floor((FLOAT_TYPE)actual_line_length_of_chunk / 
    second_dimension_loop_size);

  if (step_bound_invss.size() != num_of_parallel_loop_lines) 
    step_bound_invss.resize(num_of_parallel_loop_lines);
  if (local_derivMinss.size() != num_of_parallel_loop_lines) 
    local_derivMinss.resize(num_of_parallel_loop_lines);
  if (local_derivMaxss.size() != num_of_parallel_loop_lines) 
    local_derivMaxss.resize(num_of_parallel_loop_lines);

  if (derivMins.size() != num_of_dimensions) derivMins.resize(num_of_dimensions);
  if (derivMaxs.size() != num_of_dimensions) derivMaxs.resize(num_of_dimensions);
  if (lastDerivMins.size() != num_of_dimensions) 
    lastDerivMins.resize(num_of_dimensions);
  if (lastDerivMaxs.size() != num_of_dimensions) 
    lastDerivMaxs.resize(num_of_dimensions);
  if (executeAgains.size() != num_of_parallel_loop_lines) 
    executeAgains.resize(num_of_parallel_loop_lines);
  if (workers.size() < actual_num_of_threads) 
    workers.resize(actual_num_of_threads, NULL);
  if (commandQueues.size() < actual_num_of_threads) 
    commandQueues.resize(actual_num_of_threads, NULL);

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

  std::for_each(step_bound_invss.begin(), step_bound_invss.end(), 
    ([num_of_dimensions](beacls::FloatVec &i) {i.resize(num_of_dimensions, 0.); }));

  FLOAT_TYPE tOld = t;
  std::for_each(local_derivMinss.begin(), local_derivMinss.end(), 
    ([num_of_dimensions](auto& rhs) { 
    if (rhs.size() != num_of_dimensions) rhs.resize(num_of_dimensions); }));
  std::for_each(local_derivMaxss.begin(), local_derivMaxss.end(), 
    ([num_of_dimensions](auto& rhs) { 
    if (rhs.size() != num_of_dimensions) rhs.resize(num_of_dimensions); }));

  y = y0;

  while ((tspan[1] - t) >= (small * HjiFabs<FLOAT_TYPE>(tspan[1]))) {
    //  If there is a terminal event function registered, we need
    // to maintain the info from the last timestep.
    if (terminalEvent) {
      yOld = y;
      tOld = t;
    }
    // -----------------------------------------------------------
    // First substep: Forward Euler from t_n to t_{n+1}.
    levelset::odeCFL_LocalQ_SubStep(
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
      enable_user_defined_dynamics_on_gpu, 
      Q);

    step_bound_invs.assign(num_of_dimensions, 0.);
    std::for_each(step_bound_invss.cbegin(), step_bound_invss.cend(), [this](const auto& rhs) {
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
    const FLOAT_TYPE t1 = t + deltaT;
#if defined(PARALLEL_Y)
    const int ydot_size = (int)ydot.size();
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < ydot_size; ++i) {
      y1[i] = y[i] + deltaT * ydot[i];
    }
#else
    // std::transform(y.cbegin(), y.cend(), ydot.cbegin(), y1.begin(), 
    //   ([&deltaT](const FLOAT_TYPE &y_i, const FLOAT_TYPE &ydot_i) {
    //   return y_i + deltaT * ydot_i; }));
    y1 = y;
    for (auto ii: Q)
    {
      y1[ii] = y[ii] + deltaT * ydot[ii];
    }
#endif

    // -----------------------------------------------------------
    // Second substep: Forward Euler from t_{n+1} to t_{n+2}.
    // Approximate the derivative.
    levelset::odeCFL_LocalQ_SubStep(
      commandQueues,
      t1,
      step_bound_invss,
      local_derivMinss,
      local_derivMaxss,
      y1,
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
      enable_user_defined_dynamics_on_gpu, 
      Q 
    );
#if defined(PARALLEL_Y)
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < ydot_size; ++i) {
#else
    for (size_t i = 0; i < ydot.size(); ++i) 
    {
#endif
      FLOAT_TYPE y2_i;
      if (Q.find(i) != Q.end())
      {
        y2_i = y1[i] + deltaT * ydot[i];
      }
      else 
      {
        y2_i = y1[i];
      }
      y1[i] = (3 * y[i] + y2_i) / 4;
    }
    step_bound_invs.assign(num_of_dimensions, 0.);
    std::for_each(step_bound_invss.cbegin(), step_bound_invss.cend(), 
      [this](const auto& rhs) {std::transform(rhs.cbegin(), rhs.cend(), 
      step_bound_invs.cbegin(), step_bound_invs.begin(), 
      std::ptr_fun<const FLOAT_TYPE&, const FLOAT_TYPE&>(std::max<FLOAT_TYPE>));
    });
    step_bound_inv = std::accumulate(step_bound_invs.begin(), 
      step_bound_invs.end(), static_cast<FLOAT_TYPE>(0));
    step_bound = (FLOAT_TYPE)(1. / step_bound_inv);


    
    // Check CFL bound on timestep :
    // If the timestep chosen on the first substep violates
    //  the CFL condition by a significant amount, throw a warning.
    // For vector level sets, use the most restrictive stepBound.
    // Occasional failure should not cause too many problems.
    if (deltaT > std::min<FLOAT_TYPE>(step_bound, 
        (FLOAT_TYPE)(1.2 * factor_cfl * step_bound))) {
      const FLOAT_TYPE violation = deltaT / step_bound;
      printf("Second substep violated CFL; effective number %lf\n", violation);
    }

    // Take the second substep.
    const FLOAT_TYPE t2 = t1 + deltaT;
    // Combine t_n and t_{n+2} to get approximation at t_{n+1/2}
    const FLOAT_TYPE tHalf = (FLOAT_TYPE)(0.25 * (3 * t + t2));
    const FLOAT_TYPE one_third = (FLOAT_TYPE)( 1. / 3.);
    // -----------------------------------------------------------
    // Third substep: Forward Euler from t_{n+1/2} to t_{n+3/2}.
    // Approximate the derivative.
    levelset::odeCFL_LocalQ_SubStep(
        commandQueues,
        tHalf,
        step_bound_invss,
        local_derivMinss,
        local_derivMaxss,
        y1,
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
        enable_user_defined_dynamics_on_gpu, 
        Q);
#if defined(PARALLEL_Y)
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < ydot_size; ++i) {
#else
    for (size_t i = 0; i < ydot.size(); ++i) {
#endif
      FLOAT_TYPE y3_i;
      if (Q.find(i) != Q.end())
      {
        y3_i = y1[i] + deltaT * ydot[i];
      }
      else 
      {
        y3_i = y1[i];
      }
      y[i] = one_third * (y[i] + 2 * y3_i);
    }
    step_bound_invs.assign(num_of_dimensions, 0.);
    std::for_each(step_bound_invss.cbegin(), step_bound_invss.cend(), [this](const auto& rhs) {
      std::transform(rhs.cbegin(), rhs.cend(), step_bound_invs.cbegin(), step_bound_invs.begin(), std::ptr_fun<const FLOAT_TYPE&, const FLOAT_TYPE&>(std::max<FLOAT_TYPE>));
    });
    step_bound_inv = std::accumulate(step_bound_invs.begin(), step_bound_invs.end(), static_cast<FLOAT_TYPE>(0));
    step_bound = (FLOAT_TYPE)(1. / step_bound_inv);
    // Check CFL bound on timestep :
    // If the timestep chosen on the first substep violates
    //  the CFL condition by a significant amount, throw a warning.
    // For vector level sets, use the most restrictive stepBound.
    // Occasional failure should not cause too many problems.
    if (deltaT > std::min<FLOAT_TYPE>(step_bound, 
        (FLOAT_TYPE)(1.2 * factor_cfl * step_bound))) {

      const FLOAT_TYPE violation = deltaT / step_bound;
      printf("Third substep violated CFL; effective number %lf\n", violation);
    }

    // Take the third substep.
    const FLOAT_TYPE tThreeHalf = tHalf + deltaT;
    // Average t_n and t_{n+2} to get second order approximation of t_{n+1}.
    t =(t + 2 * tThreeHalf) / 3;

    steps++;
    //! If there is one or more post-timestep routines, call them.
    if (!post_time_steps.empty()) {
      std::for_each(post_time_steps.cbegin(), post_time_steps.cend(), ([&t,&y,&grid](const auto& func) {
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


OdeCFL3::OdeCFL3(
  const Term *schemeFunc,
  const FLOAT_TYPE factor_cfl,
  const FLOAT_TYPE max_step,
  const std::vector<levelset::PostTimestep_Exec_Type*> &post_time_steps,
  const bool single_step,
  const bool stats,
  const levelset::TerminalEvent_Exec_Type* terminalEvent) {
  pimpl = new OdeCFL3_impl(schemeFunc, factor_cfl, max_step, post_time_steps, single_step, stats, terminalEvent);
}
OdeCFL3::~OdeCFL3() {
  if (pimpl) delete pimpl;
}
FLOAT_TYPE OdeCFL3::execute(
  beacls::FloatVec& y,
  const beacls::FloatVec& tspan,
  const beacls::FloatVec& y0,
  const SchemeData *schemeData,
  const size_t line_length_of_chunk,
  const size_t num_of_threads,
  const size_t num_of_gpus,
  const levelset::DelayedDerivMinMax_Type delayedDerivMinMax,
  const bool enable_user_defined_dynamics_on_gpu) {
  if (pimpl) return pimpl->execute(y, tspan, y0, schemeData, line_length_of_chunk, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu);
  else return 0;
}
FLOAT_TYPE OdeCFL3::execute_local_q(
  beacls::FloatVec& y,
  const beacls::FloatVec& tspan,
  const beacls::FloatVec& y0,
  const SchemeData *schemeData,
  const size_t line_length_of_chunk,
  const size_t num_of_threads,
  const size_t num_of_gpus,
  const levelset::DelayedDerivMinMax_Type delayedDerivMinMax,
  const bool enable_user_defined_dynamics_on_gpu,
  const std::set<size_t>& Q) {
  if (pimpl) return pimpl->execute_local_q(y, tspan, y0, schemeData, line_length_of_chunk, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu, Q);
  else return 0;
}
OdeCFL3::OdeCFL3(const OdeCFL3& rhs) :
  pimpl(rhs.pimpl->clone())
{
}

OdeCFL3* OdeCFL3::clone() const {
  return new OdeCFL3(*this);
}
