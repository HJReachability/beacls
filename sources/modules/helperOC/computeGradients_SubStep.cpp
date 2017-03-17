#include <algorithm>
#include <chrono>
#include <thread>
#include "computeGradients_SubStep.hpp"
#include "ComputeGradients_OneSlice.hpp"
#include "ComputeGradients_CommandQueue.hpp"
void helperOC::computeGradients_SubStep(
	ComputeGradients_CommandQueue* commandQueue,
	std::vector<beacls::FloatVec>& derivC,
	std::vector<beacls::FloatVec>& derivL,
	std::vector<beacls::FloatVec>& derivR,
	const beacls::FloatVec& modified_data,
	const beacls::FloatVec& original_data,
	const levelset::HJI_Grid* grid,
	const beacls::UVecType type,
	const beacls::UVecDepth depth,
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
	const size_t tau_length,
	const size_t num_of_elements

) {
	//! Parallel Body
	std::vector<ComputeGradients_OneSlice*> computeGradients_OneSlices(num_of_parallel_loop_lines * tau_length);
	//!< data at multiple time stamps
	for (size_t t = 0; t < tau_length; ++t) {
		const size_t time_offset = t * num_of_elements;
		for (int parallel_line_index = 0; parallel_line_index < (int)num_of_parallel_loop_lines; ++parallel_line_index) {
			ComputeGradients_OneSlice* computeGradients_OneSlice = new ComputeGradients_OneSlice(
				derivC,
				derivL,
				derivR,
				modified_data,
				original_data,
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
				parallel_line_index,
				second_dimension_loop_size,
				third_dimension_loop_size,
				time_offset);
			computeGradients_OneSlices[parallel_line_index + t * num_of_parallel_loop_lines] = computeGradients_OneSlice;
			commandQueue->push(computeGradients_OneSlice);
		}
	}
	std::for_each(computeGradients_OneSlices.begin(), computeGradients_OneSlices.end(), [](auto& rhs) {
		if (rhs) {
			while (!rhs->is_finished()) {
				std::this_thread::yield();
			}
			delete rhs;
		}
	});
}
