#ifndef __ComputeGradients_OneSlice_impl_hpp__
#define __ComputeGradients_OneSlice_impl_hpp__

#include <typedef.hpp>
#include <cstddef>
#include <vector>
class HJI_Grid;

namespace helperOC {
	class ComputeGradients_OneSlice_impl {
		std::vector<beacls::FloatVec>& derivC;
		std::vector<beacls::FloatVec>& derivL;
		std::vector<beacls::FloatVec>& derivR;
		const HJI_Grid* grid;
		const beacls::FloatVec& modified_data;
		const beacls::FloatVec& original_data;
		const beacls::UVecType type;
		const beacls::UVecDepth depth;
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
		const size_t time_offset;
		bool finished;
	public:
		bool is_finished() const { return finished; }
		void set_finished() { finished = true; }
		ComputeGradients_OneSlice_impl(
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
			const size_t parallel_loop_size,
			const size_t num_of_outer_lines,
			const size_t num_of_inner_lines,
			const size_t parallel_line_index,
			const size_t second_dimension_loop_size,
			const size_t third_dimension_loop_size,
			const size_t time_offset
			) :
			derivC(derivC),
			derivL(derivL),
			derivR(derivR),
			grid(grid),
			modified_data(modified_data),
			original_data(original_data),
			type(type),
			depth(depth),
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
			time_offset(time_offset),
			finished(false)
		{}
		void execute(
			SpatialDerivative* spatialDerivative,
			beacls::UVec& original_data_line_uvec,
			std::vector<beacls::UVec>& deriv_c_line_uvecs,
			std::vector<beacls::UVec>& deriv_l_line_uvecs,
			std::vector<beacls::UVec>& deriv_r_line_uvecs
		);
	private:
		ComputeGradients_OneSlice_impl();
		ComputeGradients_OneSlice_impl(const ComputeGradients_OneSlice_impl& rhs);
		ComputeGradients_OneSlice_impl& operator=(const ComputeGradients_OneSlice_impl& rhs);
	};
};
#endif	/* __ComputeGradients_OneSlice_impl_hpp__ */

