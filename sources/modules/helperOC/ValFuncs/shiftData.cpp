#include <helperOC/helperOC_type.hpp>
#include <helperOC/ValFuncs/shiftData.hpp>
#include <helperOC/ValFuncs/eval_u.hpp>
#include <levelset/Grids/HJI_Grid.hpp>
#include <vector>
#include <iostream>
#include <algorithm>

bool helperOC::shiftData(
	beacls::FloatVec& dataOut,
	const levelset::HJI_Grid* g,
	const beacls::FloatVec& dataIn,
	const beacls::FloatVec& shift,
	const beacls::IntegerVec& pdims,
	const beacls::Interpolate_Type interp_method
) {
	//!< Get a list of new indices
	//!< Shift grid backwards
	const std::vector<beacls::FloatVec>& xss = g->get_xss();
	const size_t num_of_points = xss[0].size();
	std::vector<beacls::FloatVec> rxss;
	rxss.resize(num_of_points);

	for(size_t point = 0; point< num_of_points; ++point) {
		rxss[point].resize(shift.size());
		for (size_t i = 0; i < shift.size(); ++i) {
			const size_t pdim = pdims[i];
			rxss[point][pdim] = xss[pdim][point] - shift[i];
		}
	}

	//!< Interpolate dataIn to get approximation of shifted data
	helperOC::eval_u(dataOut, g, dataIn, rxss, interp_method);
	FLOAT_TYPE max_val = dataOut[0];
	std::for_each(dataOut.cbegin() + 1, dataOut.cend(), [&max_val](const auto& rhs) {
		if ((max_val < rhs) || std::isnan(max_val)) max_val = rhs;
	});
	std::transform(dataOut.cbegin(), dataOut.cend(), dataOut.begin(), [max_val](const auto& rhs) { return (std::isnan(rhs)) ? max_val : rhs; });
	return true;
}
	
