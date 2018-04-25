#include <algorithm>
#include <chrono>
#include <thread>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "odeCFL_SubStep.hpp"
#include "OdeCFL_OneSlice.hpp"
#include "OdeCFL_CommandQueue.hpp"
#include <macro.hpp>
void levelset::odeCFL_SubStep(
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
) {
	// std::cout << "Begin execution of odeCFL_SubStep." << std::endl; // JFF DEBUG
	bool executeAgain = false;
	std::vector<beacls::FloatVec > originalDerivMins;
	std::vector<beacls::FloatVec > originalDerivMaxs;
	if (delayedDerivMinMax) {
		derivMins = lastDerivMins;
		derivMaxs = lastDerivMaxs;
		if (delayedDerivMinMax == DelayedDerivMinMax_Adaptive) {
			originalDerivMins = lastDerivMins;
			originalDerivMaxs = lastDerivMaxs;
		}
	}
	else {
		std::for_each(derivMins.begin(), derivMins.end(), ([](auto& rhs) { rhs.clear(); }));
		std::for_each(derivMaxs.begin(), derivMaxs.end(), ([](auto& rhs) { rhs.clear(); }));
	}
	bool updateDerivMinMax = true;
	do {
		//! Parallel Body
		std::vector<OdeCFL_OneSlice*> odeCFL_OneSlices(num_of_parallel_loop_lines);
		// std::cout << "Begin for parallel_line_index loop." << std::endl; // JFF DEBUG
		for (int parallel_line_index = 0; parallel_line_index < (int)num_of_parallel_loop_lines; ++parallel_line_index) {
			OdeCFL_OneSlice* odeCFL_OneSlice = new OdeCFL_OneSlice(
				t,
				step_bound_invss[parallel_line_index],
				local_derivMinss[parallel_line_index],
				local_derivMaxss[parallel_line_index],
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
			odeCFL_OneSlices[parallel_line_index] = odeCFL_OneSlice;
			odeCFL_CommandQueues[parallel_line_index%actual_num_of_threas]->push(odeCFL_OneSlice);
		}
		// std::cout << "End for parallel_line_index loop." << std::endl; // JFF DEBUG
		std::transform(odeCFL_OneSlices.begin(), odeCFL_OneSlices.end(), executeAgains.begin(), [](auto& rhs) {
			// std::cout << "Begin while !rhs->is_finished loop." << std::endl; // JFF DEBUG
			while (!rhs->is_finished()) {
				// std::cout << "Evaluated rhs->is_finished." << std::endl; // JFF DEBUG
				std::this_thread::yield();
				// JFF FOUND ISSUE: INFINITE LOOP UNTIL SEGMENTATION FAULT! RHS->IS_FINISHED ALWAYS FALSE.
			}
			// std::cout << "End while !rhs->is_finished loop." << std::endl; // JFF DEBUG
			bool result;
			result = rhs->get_executeAgain();
			// std::cout << "Done get_executeAgain()." << std::endl; // JFF DEBUG

			if (rhs) delete rhs;
			return result;
		});
		// std::cout << "Done stransform()." << std::endl; // JFF DEBUG
		executeAgain = std::any_of(executeAgains.cbegin(), executeAgains.cend(), [](const auto& rhs) { return rhs; });
		for (auto ite = derivMins.begin(); ite != derivMins.end(); ++ite) {
			if (ite->size() != 1) ite->resize(1);
			const size_t dimension = std::distance(derivMins.begin(), ite);
			if (!local_derivMinss.empty()) {
				(*ite)[0] = beacls::min_value_at_index<FLOAT_TYPE>(local_derivMinss.cbegin(), local_derivMinss.cend(), dimension);
			}
		};
		for (auto ite = derivMaxs.begin(); ite != derivMaxs.end(); ++ite) {
			if (ite->size() != 1) ite->resize(1);
			const size_t dimension = std::distance(derivMaxs.begin(), ite);
			if (!local_derivMaxss.empty()) {
				(*ite)[0] = beacls::max_value_at_index<FLOAT_TYPE>(local_derivMaxss.cbegin(), local_derivMaxss.cend(), dimension);
			}
		};
		static const FLOAT_TYPE small = (FLOAT_TYPE)1e-3;
		if (delayedDerivMinMax== DelayedDerivMinMax_Adaptive) {
			for (size_t dimension = 0; dimension < derivMins.size(); ++dimension) {
				for (size_t i = 0; i < derivMins[dimension].size(); ++i) {
					if (!executeAgain && !originalDerivMins[dimension].empty()){
						if (std::abs((derivMins[dimension][i] - originalDerivMins[dimension][i]) / derivMins[dimension][i]) > small) {
							executeAgain = true;
#if 0
							std::cout << "Min[" << dimension << "][" << i << "]" << std::fixed << std::setprecision(6)
								<< derivMins[dimension][i] << " != "
								<< originalDerivMins[dimension][i] << ", AbsDiff= "
								<< std::abs(derivMins[dimension][i] - originalDerivMins[dimension][i]) << " > "
								<< small << ", ErrorRatio= "
								<< (derivMins[dimension][i] - originalDerivMins[dimension][i]) / derivMins[dimension][i]
								<< std::resetiosflags(std::ios_base::floatfield) << std::endl;
#endif
						}
					}
				}
			}
			for (size_t dimension = 0; dimension < derivMaxs.size(); ++dimension) {
				for (size_t i = 0; i < derivMaxs[dimension].size(); ++i) {
					if (!executeAgain && !originalDerivMaxs[dimension].empty()) {
						if (std::abs((derivMaxs[dimension][i] - originalDerivMaxs[dimension][i]) / derivMaxs[dimension][i])> small) {
							executeAgain = true;
#if 0
							std::cout << "Max[" << dimension << "][" << i << "]" << std::fixed << std::setprecision(6)
								<< derivMaxs[dimension][i] << " != "
								<< originalDerivMaxs[dimension][i] << ", AbsDiff= "
								<< std::abs(derivMaxs[dimension][i] - originalDerivMaxs[dimension][i]) << " > "
								<< small << ", ErrorRatio= "
								<< (derivMaxs[dimension][i] - originalDerivMaxs[dimension][i]) / derivMaxs[dimension][i]
								<< std::resetiosflags(std::ios_base::floatfield) << std::endl;
#endif
						}
					}
				}
			}
			originalDerivMins = derivMins;
			originalDerivMaxs = derivMaxs;
		}
		updateDerivMinMax = false;
	} while (executeAgain);
}
