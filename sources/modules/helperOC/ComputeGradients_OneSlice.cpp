#include "ComputeGradients_OneSlice.hpp"
#include "ComputeGradients_OneSlice_impl.hpp"
#include <functional>
#include <algorithm>
#include <levelset/SpatialDerivative/SpatialDerivative.hpp>
#include "ComputeGradients_OneSlice_cuda.hpp"
#include <Core/UVec.hpp>
static
bool copyBackInfNan(
	beacls::FloatVec& derivC_ptr,
	beacls::FloatVec& derivL_ptr,
	beacls::FloatVec& derivR_ptr,
	const beacls::FloatVec& original_data,
	const size_t length
) {
	for (size_t i = 0; i < length; ++i) {
		const FLOAT_TYPE d = original_data[i];
		if ((d == std::numeric_limits<FLOAT_TYPE>::signaling_NaN()) || (d == std::numeric_limits<FLOAT_TYPE>::infinity())) {
			derivC_ptr[i] = d;
			derivL_ptr[i] = d;
			derivR_ptr[i] = d;
		}
	}
	return true;
}


void helperOC::ComputeGradients_OneSlice_impl::execute(
	SpatialDerivative* spatialDerivative,
	beacls::UVec& original_data_line_uvec,
	std::vector<beacls::UVec>& deriv_c_line_uvecs,
	std::vector<beacls::UVec>& deriv_l_line_uvecs,
	std::vector<beacls::UVec>& deriv_r_line_uvecs
) {
	const size_t paralleled_out_line_size = std::min(parallel_loop_size, (num_of_outer_lines > parallel_line_index * parallel_loop_size) ? num_of_outer_lines - parallel_line_index * parallel_loop_size : 0);
	size_t actual_chunk_size = num_of_chunk_lines;
	size_t actual_num_of_slices = num_of_slices;
	deriv_c_line_uvecs.resize(num_of_dimensions);
	deriv_l_line_uvecs.resize(num_of_dimensions);
	deriv_r_line_uvecs.resize(num_of_dimensions);
	for (size_t line_index = 0; line_index < paralleled_out_line_size*num_of_inner_lines; line_index += actual_chunk_size*actual_num_of_slices) {
		const size_t inner_line_index = line_index + parallel_line_index*parallel_loop_size*num_of_inner_lines;
		const size_t second_line_index = inner_line_index % second_dimension_loop_size;
		actual_chunk_size = std::min(num_of_chunk_lines, second_dimension_loop_size - second_line_index);
		const size_t third_line_index = (inner_line_index / second_dimension_loop_size) % third_dimension_loop_size;
		actual_num_of_slices = std::min(num_of_slices, third_dimension_loop_size - third_line_index);
		const size_t line_begin = inner_line_index;
		size_t expected_result_offset = line_begin * first_dimension_loop_size + time_offset;
		size_t slices_result_size = actual_chunk_size * first_dimension_loop_size*actual_num_of_slices;
		size_t chunk_result_size = actual_chunk_size * first_dimension_loop_size;

		if (original_data_line_uvec.type() != type) original_data_line_uvec = beacls::UVec(depth, type, slices_result_size);
		else if (original_data_line_uvec.size() != slices_result_size) original_data_line_uvec.resize(slices_result_size);
		beacls::copyHostPtrToUVecAsync(original_data_line_uvec, original_data.data() + expected_result_offset, slices_result_size);
		std::vector<beacls::UVec> dummys(num_of_dimensions);
		for (size_t index = 0; index < num_of_dimensions; ++index) {
			//!< To optimize asynchronous execution, calculate from heavy dimension (0, 2, 3 ... 1);
			const size_t dim = (index == 0) ? index : (index == num_of_dimensions - 1) ? 1 : index + 1;

			beacls::UVec& deriv_c_line_uvec = deriv_c_line_uvecs[dim];
			beacls::UVec& deriv_l_line_uvec = deriv_l_line_uvecs[dim];
			beacls::UVec& deriv_r_line_uvec = deriv_r_line_uvecs[dim];
			if (deriv_l_line_uvec.type() != type) deriv_l_line_uvec = beacls::UVec(depth, type, slices_result_size);
			else deriv_l_line_uvec.resize(slices_result_size);
			if (deriv_r_line_uvec.type() != type) deriv_r_line_uvec = beacls::UVec(depth, type, slices_result_size);
			else deriv_r_line_uvec.resize(slices_result_size);
			if (deriv_c_line_uvec.type() != type) deriv_c_line_uvec = beacls::UVec(depth, type, slices_result_size);
			else deriv_c_line_uvec.resize(slices_result_size);

			spatialDerivative->execute(
				deriv_l_line_uvec, deriv_r_line_uvec,
				grid, modified_data.data() + time_offset,
				dim, false, line_begin, chunk_result_size, actual_num_of_slices);
		}
		for (size_t dim = 0; dim < num_of_dimensions; ++dim) {
			spatialDerivative->synchronize(dim);
			beacls::average(deriv_l_line_uvecs[dim], deriv_r_line_uvecs[dim], deriv_c_line_uvecs[dim]);
			if (dim == 0) {
				beacls::synchronizeUVec(original_data_line_uvec);
			}
			if (type == beacls::UVecType_Cuda) {
				copyBackInfNan_cuda(
					deriv_c_line_uvecs[dim],
					deriv_l_line_uvecs[dim],
					deriv_r_line_uvecs[dim],
					original_data_line_uvec);
			}
			else {
				copyBackInfNan(
					*(beacls::UVec_<FLOAT_TYPE>(deriv_c_line_uvecs[dim]).vec()),
					*(beacls::UVec_<FLOAT_TYPE>(deriv_l_line_uvecs[dim]).vec()),
					*(beacls::UVec_<FLOAT_TYPE>(deriv_r_line_uvecs[dim]).vec()),
					*(beacls::UVec_<FLOAT_TYPE>(original_data_line_uvec).vec()),
					slices_result_size);
			}
		}
		for (size_t dim = 0; dim < num_of_dimensions; ++dim) {
			beacls::synchronizeUVec(deriv_c_line_uvecs[dim]);
			beacls::synchronizeUVec(deriv_l_line_uvecs[dim]);
			beacls::synchronizeUVec(deriv_r_line_uvecs[dim]);
			beacls::copyUVecToHostAsync(derivC[dim].data() + expected_result_offset, deriv_c_line_uvecs[dim]);
			beacls::copyUVecToHostAsync(derivL[dim].data() + expected_result_offset, deriv_l_line_uvecs[dim]);
			beacls::copyUVecToHostAsync(derivR[dim].data() + expected_result_offset, deriv_r_line_uvecs[dim]);
		}
		for (size_t dim = 0; dim < num_of_dimensions; ++dim) {
			beacls::synchronizeUVec(deriv_c_line_uvecs[dim]);
			beacls::synchronizeUVec(deriv_l_line_uvecs[dim]);
			beacls::synchronizeUVec(deriv_r_line_uvecs[dim]);
		}
	}
}
void helperOC::ComputeGradients_OneSlice::execute(
	SpatialDerivative* spatialDerivative,
	beacls::UVec& original_data_line_uvec,
	std::vector<beacls::UVec>& deriv_c_line_uvecs,
	std::vector<beacls::UVec>& deriv_l_line_uvecs,
	std::vector<beacls::UVec>& deriv_r_line_uvecs
) {
	if (pimpl) pimpl->execute(
		spatialDerivative, 
		original_data_line_uvec,
		deriv_c_line_uvecs,
		deriv_l_line_uvecs,
		deriv_r_line_uvecs);
}
bool helperOC::ComputeGradients_OneSlice::is_finished() const {
	if (pimpl) return pimpl->is_finished();
	else return false;
}
void helperOC::ComputeGradients_OneSlice::set_finished() {
	if (pimpl) pimpl->set_finished();
}

helperOC::ComputeGradients_OneSlice::ComputeGradients_OneSlice(
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
) {

	pimpl = new ComputeGradients_OneSlice_impl(
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
		time_offset
	);
}
helperOC::ComputeGradients_OneSlice::~ComputeGradients_OneSlice()
{
	if (pimpl) delete pimpl;
}


