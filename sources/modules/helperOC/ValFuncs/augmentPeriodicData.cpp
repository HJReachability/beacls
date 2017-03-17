#include <helperOC/helperOC_type.hpp>
#include <helperOC/ValFuncs/augmentPeriodicData.hpp>
#include <levelset/BoundaryCondition/AddGhostPeriodic.hpp>
#include <levelset/Grids/HJI_Grid.hpp>
#include <vector>
#include <algorithm>
#include <numeric>
#include <typeinfo>
#if 0
HJI_Grid* helperOC::augmentPeriodicData(
	beacls::FloatVec& dataOut,
	const levelset::HJI_Grid* gIn,
	const beacls::FloatVec& data) {
	const size_t num_of_dimensions = gIn->get_num_of_dimensions();
	HJI_Grid* gOut = gIn->clone(false);

	std::vector<beacls::FloatVec> vss = gIn->get_vss();
	const beacls::FloatVec* data_ptr = &data;
	beacls::FloatVec tmpdata_buf;
	bool odd = false;
	bool augmented = false;
	beacls::IntegerVec NsOut = gIn->get_Ns();
	//!< Dealing with periodicity
	for (size_t dimension = 0; dimension < num_of_dimensions; ++dimension) {
		const BoundaryCondition* boundaryCondition = gIn->get_boundaryCondition(dimension);
		if (boundaryCondition && (typeid(*boundaryCondition) == typeid(AddGhostPeriodic))) {
			beacls::FloatVec& tmp_dst_data = odd ? tmpdata_buf : dataOut;
			odd = odd ? false : true;
			//!< Grid points
			beacls::FloatVec& new_vs = vss[dimension];
			new_vs.push_back(new_vs[new_vs.size() - 1] + gIn->get_dx(dimension));
			//!< Input data; eg. data = cat(:, data, data(:,:,1))
			const size_t inner_loop_size = std::accumulate(NsOut.cbegin(), NsOut.cbegin() + dimension, static_cast<size_t>(1), [](const auto& lhs, const auto& rhs) {
				return lhs * rhs;
			});
			const size_t loop_size = NsOut[dimension];
			const size_t outer_loop_size = std::accumulate(NsOut.cbegin() + dimension + 1, NsOut.cend(), static_cast<size_t>(1), [](const auto& lhs, const auto& rhs) {
				return lhs * rhs;
			});
			tmp_dst_data.resize(inner_loop_size*(loop_size + 1)*outer_loop_size);
			NsOut[dimension]++;
			for (size_t j = 0; j < outer_loop_size; ++j) {
				const size_t src_outer_loop_offset = j * inner_loop_size * loop_size;
				const size_t dst_outer_loop_offset = j * inner_loop_size * (loop_size + 1);
				for (size_t i = 0; i < loop_size; ++i) {
					const size_t src_begin_offset = inner_loop_size*i + src_outer_loop_offset;
					const size_t src_end_offset = inner_loop_size*(i + 1) + src_outer_loop_offset;
					const size_t dst_begin_offset = inner_loop_size*i + dst_outer_loop_offset;
					std::copy(
						data_ptr->cbegin() + src_begin_offset,
						data_ptr->cbegin() + src_end_offset,
						tmp_dst_data.begin() + dst_begin_offset
					);
				}
				std::copy(
					data_ptr->cbegin() + src_outer_loop_offset,
					data_ptr->cbegin() + src_outer_loop_offset + inner_loop_size,
					tmp_dst_data.begin() + dst_outer_loop_offset + inner_loop_size * loop_size
				);
			}
			data_ptr = &tmp_dst_data;
			augmented = true;
		}
	}
	if (augmented) {
		if (!odd) {
			dataOut = tmpdata_buf;
		}
	}
	else {
		dataOut = data;
	}
	gOut->set_Ns(NsOut);
	gOut->set_vss(vss);
	
	return gOut;
}
#endif