#ifndef __computeGradients_SubStep_hpp__
#define __computeGradients_SubStep_hpp__

#include <typedef.hpp>
#include <vector>
#include <deque>
#include <cstddef>
class HJI_Grid;

namespace helperOC {
	class ComputeGradients_CommandQueue;
	void computeGradients_SubStep(
		ComputeGradients_CommandQueue* commandQueue,
		std::vector<beacls::FloatVec>& derivC,
		std::vector<beacls::FloatVec>& derivL,
		std::vector<beacls::FloatVec>& derivR,
		const beacls::FloatVec& modified_data,
		const beacls::FloatVec& original_data,
		const HJI_Grid* grid,
		const beacls::UVecType type,
		const beacls::UVecDepth depth,
		const size_t num_of_dimensions,
		const size_t first_dimension_loop_size,
		const size_t num_of_chunk_lines,
		const size_t num_of_slices,
		const size_t num_of_outer_lines,
		const size_t num_of_inner_lines,
		const size_t parallel_line_index,
		const size_t second_dimension_loop_size,
		const size_t third_dimension_loop_size,
		const size_t num_of_parallel_loop_lines,
		const size_t tau_length,
		const size_t num_of_elements
		);
};

#endif	/* __computeGradients_SubStep_hpp__ */

