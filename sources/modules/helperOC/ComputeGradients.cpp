#include <helperOC/ComputeGradients.hpp>
#include <iostream>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <numeric>
#include <Core/UVec.hpp>
#include <thread>
#include "computeGradients_SubStep.hpp"
#include "ComputeGradients_CommandQueue.hpp"
#include "ComputeGradients_Worker.hpp"

#include <levelset/levelset.hpp>

helperOC::ComputeGradients::ComputeGradients(
	const levelset::HJI_Grid* grid,
	helperOC::ApproximationAccuracy_Type accuracy,
	const beacls::UVecType type
) : 
	commandQueue(new helperOC::ComputeGradients_CommandQueue),
	type(type) {
	helperOC::ComputeGradients_Worker* worker = new helperOC::ComputeGradients_Worker(commandQueue, grid, accuracy, type, 0);
	workers.push_back(worker);
	worker->run();
}
helperOC::ComputeGradients::~ComputeGradients() {
	std::for_each(workers.begin(), workers.end(), [](auto& rhs) {
		if (rhs) {
			rhs->terminate();
			delete rhs;
		}
	});
	if (commandQueue) delete commandQueue;
}
bool helperOC::ComputeGradients::operator==(const ComputeGradients& rhs) const {
	if (this == &rhs) return true;
	return type == rhs.type;
}
beacls::UVecType helperOC::ComputeGradients::get_type() const { 
	return type; 
}
bool helperOC::ComputeGradients::operator()(
	std::vector<beacls::FloatVec >& derivC,
	std::vector<beacls::FloatVec >& derivL,
	std::vector<beacls::FloatVec >& derivR,
	const levelset::HJI_Grid* grid,
	const beacls::FloatVec& data,
	const size_t data_length,
	const bool upWind,
	const helperOC::ExecParameters& execParameters
) {
	const size_t line_length_of_chunk = execParameters.line_length_of_chunk;
	const size_t num_of_threads = execParameters.num_of_threads;
	const size_t num_of_gpus = execParameters.num_of_gpus;

	if (upWind) {
		std::cerr << "error: " << __func__ << " Upwinding has not been implemented!" << std::endl;
		return false;
	}
	if (data_length==0) {
		std::cerr << "error: " << __func__ << " data is empty!" << std::endl;
		return false;
	}
	const size_t num_of_dimensions = grid->get_num_of_dimensions();
	//!< Go through each dimension and compute the gradient in each
	if (derivC.size() != num_of_dimensions) derivC.resize(num_of_dimensions);
	if (derivL.size() != num_of_dimensions) derivL.resize(num_of_dimensions);
	if (derivR.size() != num_of_dimensions) derivR.resize(num_of_dimensions);

	const size_t num_of_elements = grid->get_sum_of_elems();
	const size_t tau_length = (num_of_elements == data_length) ? 1 : data_length / num_of_elements;

	//!< Just in case there are NaN values in the data(usually from TTR functions)
	//<! Just in case there are inf values
	const FLOAT_TYPE numInfty = 1e6;
	modified_data.resize(data_length);
	for (size_t i = 0; i < data_length; ++i) {
		const FLOAT_TYPE d = data[i];
		if ((d == std::numeric_limits<FLOAT_TYPE>::signaling_NaN()) || (d == std::numeric_limits<FLOAT_TYPE>::infinity()))
			modified_data[i] = numInfty;
		else
			modified_data[i] = d;
	}
//	size_t chunk_size = line_length_of_chunk;
	for_each(derivC.begin(), derivC.end(), ([data_length](auto& rhs) { if (rhs.size() != data_length) rhs.resize(data_length); }));
	for_each(derivL.begin(), derivL.end(), ([data_length](auto& rhs) { if (rhs.size() != data_length) rhs.resize(data_length); }));
	for_each(derivR.begin(), derivR.end(), ([data_length](auto& rhs) { if (rhs.size() != data_length) rhs.resize(data_length); }));
	beacls::UVecDepth depth = beacls::type_to_depth<FLOAT_TYPE>();

	const beacls::IntegerVec& Ns = grid->get_Ns();
	const size_t first_dimension_loop_size = (num_of_dimensions >= 1) ? Ns[0] : 1;
	const size_t second_dimension_loop_size = (num_of_dimensions >= 2) ? Ns[1] : 1;
	const size_t third_dimension_loop_size = (num_of_dimensions >= 3) ? Ns[2] : 1;
	const size_t num_of_lines = num_of_elements / first_dimension_loop_size;
	const size_t num_of_outer_lines = std::accumulate(Ns.cbegin() + 2, Ns.cend(), (size_t)1,
		[](const auto& lhs, const auto& rhs) {return lhs * rhs; });

	const size_t num_of_activated_gpus = (num_of_gpus == 0) ? beacls::get_num_of_gpus() : num_of_gpus;
	const size_t hardware_concurrency = (std::thread::hardware_concurrency() == 0) ? 1 : std::thread::hardware_concurrency();
	const levelset::SpatialDerivative* spatialDerivative = workers[0]->get_spatialDerivative();
	const size_t actual_num_of_threads = (num_of_threads == 0) ? ((spatialDerivative->get_type() == beacls::UVecType_Cuda) ? num_of_activated_gpus : hardware_concurrency) : num_of_threads;

	const size_t num_of_parallel_loop_lines = std::min((size_t)actual_num_of_threads, num_of_outer_lines);
	const size_t parallel_loop_size = (size_t)std::ceil((FLOAT_TYPE)num_of_outer_lines / num_of_parallel_loop_lines);
	const size_t num_of_inner_lines = (size_t)std::ceil((FLOAT_TYPE)num_of_lines / num_of_outer_lines);

	const double gpu_memory_ocupancy_ratio = 1.0 / (2 + 15 * num_of_dimensions) / 2 * 0.8;
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

	if (workers.size() < actual_num_of_threads) workers.resize(actual_num_of_threads, NULL);
	helperOC::ComputeGradients_Worker* worker0 = workers[0];
	for (auto ite = workers.begin(); ite != workers.end(); ++ite) {
		helperOC::ComputeGradients_Worker* worker = *ite;
		if (worker == NULL) {
			const int thread_id = (int)std::distance(workers.begin(), ite);
			worker = new helperOC::ComputeGradients_Worker(commandQueue, worker0->get_spatialDerivative(), (num_of_activated_gpus>1) ? thread_id % num_of_activated_gpus : 0);
			worker->run();
			*ite = worker;
		}
	};
	helperOC::computeGradients_SubStep(
		commandQueue,
		derivC,
		derivL,
		derivR,
		modified_data,
		data,
		grid,
		type,
		depth,
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
		tau_length,
		num_of_elements);
	return true;
}

