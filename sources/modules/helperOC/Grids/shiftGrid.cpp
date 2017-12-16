#include <helperOC/helperOC_type.hpp>
#include <helperOC/Grids/shiftGrid.hpp>
#include <Core/interpn.hpp>
#include <levelset/Grids/HJI_Grid.hpp>
#include <algorithm>
#include <iostream>
#include <deque>
#include <vector>
#include <numeric>
#include <functional>
#include <macro.hpp>


levelset::HJI_Grid* helperOC::shiftGrid(
	const levelset::HJI_Grid* gIn,
	const beacls::FloatVec& shiftAmount
) {
	//!< Input checks
	const size_t gIn_num_of_dimensions = gIn->get_num_of_dimensions();
	if (shiftAmount.size() != gIn_num_of_dimensions) {
		std::cerr << "Length of shiftAmount must match dimension of the grid!" << std::endl;
		return NULL;
	}
	//!< Make sure shiftAmount is a column vector

	//!<  Dimensionality of grid
	levelset::HJI_Grid* gShift = gIn->clone(false);

	//!< Shift the grid
	std::vector<beacls::FloatVec> shifted_xss(shiftAmount.size());
	std::vector<beacls::FloatVec> shifted_vss(shiftAmount.size());
	beacls::FloatVec shifted_mins(shiftAmount.size());
	beacls::FloatVec shifted_maxs(shiftAmount.size());
	const size_t begin_index = 0;
	const size_t length = 0;
	for (size_t dimension = 0; dimension < shiftAmount.size(); ++dimension) {
		beacls::UVec x_uvec;
		gIn->get_xs(x_uvec, dimension, begin_index, length);
		const beacls::FloatVec* xs_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvec).vec();

		const beacls::FloatVec& xs = gIn->get_xs(dimension);
		beacls::FloatVec& shifted_xs = shifted_xss[dimension];
		beacls::FloatVec& shifted_vs = shifted_vss[dimension];
		shifted_xs.resize(xs.size());
		const FLOAT_TYPE shiftAmount_d = shiftAmount[dimension];
		std::transform(xs.cbegin(), xs.cend(), shifted_xs.begin(), [&shiftAmount_d](const auto& rhs) { return rhs + shiftAmount_d; });
		auto minMax = beacls::minmax_value<FLOAT_TYPE>(shifted_xs.cbegin(), shifted_xs.cend());
		shifted_mins[dimension] = minMax.first;
		shifted_maxs[dimension] = minMax.second;
		shifted_vs = shifted_xs;
		std::sort(shifted_vs.begin(), shifted_vs.end());
		shifted_vs.erase(std::unique(shifted_vs.begin(), shifted_vs.end()), shifted_vs.end());
	}
	gShift->set_xss(shifted_xss);
	gShift->set_vss(shifted_vss);
	gShift->set_mins(shifted_mins);
	gShift->set_maxs(shifted_maxs);
	return gShift;
}
