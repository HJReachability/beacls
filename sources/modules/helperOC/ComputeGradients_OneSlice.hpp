#ifndef __ComputeGradients_OneSlice_hpp__
#define __ComputeGradients_OneSlice_hpp__

#include <typedef.hpp>
#include <cstddef>
#include <vector>
#include <Core/UVec.hpp>
class HJI_Grid;
class SpatialDerivative;
namespace helperOC {
	class ComputeGradients_OneSlice_impl;
	class ComputeGradients_OneSlice {
	public:
		ComputeGradients_OneSlice_impl* pimpl;
		bool is_finished() const;
		void set_finished();
		ComputeGradients_OneSlice(
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
		);
		void execute(
			SpatialDerivative* spatialDerivative,
			beacls::UVec& original_data_line_uvec,
			std::vector<beacls::UVec>& deriv_c_line_uvecs,
			std::vector<beacls::UVec>& deriv_l_line_uvecs,
			std::vector<beacls::UVec>& deriv_r_line_uvecs
		);
		~ComputeGradients_OneSlice();
	private:
		ComputeGradients_OneSlice();
		ComputeGradients_OneSlice(const ComputeGradients_OneSlice& rhs);
		ComputeGradients_OneSlice& operator=(const ComputeGradients_OneSlice& rhs);
	};
};
#endif	/* __ComputeGradients_OneSlice_hpp__ */

