#include <helperOC/helperOC_type.hpp>
#include <helperOC/ValFuncs/rotateData.hpp>
#include <helperOC/ValFuncs/eval_u.hpp>
#include <levelset/Grids/HJI_Grid.hpp>
#include <vector>
#include <iostream>
#include <algorithm>

bool helperOC::rotateData(
	beacls::FloatVec& dataOut,
	const HJI_Grid* g,
	const beacls::FloatVec& dataIn,
	const FLOAT_TYPE theta,
	const beacls::IntegerVec& pdims,
	const beacls::IntegerVec& adims,
	const beacls::Interpolate_Type interp_method
	)
{
	const beacls::FloatVec& xs_pdims0 = g->get_xs(pdims[0]);
	const beacls::FloatVec& xs_pdims1 = g->get_xs(pdims[1]);
	std::vector<beacls::FloatVec> rxss(xs_pdims0.size());
	rxss[pdims[0]].resize(xs_pdims0.size());
	rxss[pdims[1]].resize(xs_pdims1.size());

	//!< Get a list of new indices
	//!< Multiply by rotation matrix for position dimensions
	std::transform(xs_pdims0.cbegin(), xs_pdims0.cend(), xs_pdims1.cbegin(), rxss.begin(), [theta, pdims, adims](const auto& lhs, const auto& rhs) {
		beacls::FloatVec res(2+adims.size());
		res[pdims[0]] = std::cos(-theta) * lhs - sin(-theta) * rhs;
		res[pdims[1]] = std::sin(-theta) * lhs + cos(-theta) * rhs;
		return res;
	});
	//!< Translate in angle
	if (!adims.empty()) {
		std::for_each(adims.cbegin(), adims.cend(), [theta, &rxss, g](const auto& rhs) {
			const size_t adim = rhs;
			const beacls::FloatVec& xs_adim = g->get_xs(rhs);
			std::transform(xs_adim.cbegin(), xs_adim.cend(), rxss.begin(), rxss.begin(), [theta, adim](const auto& lhs, auto& rhs) {
				rhs[adim] = lhs - theta;
				return rhs;
			});
		});
	}
	//!< Interpolate dataIn to get approximation of rotated data
	helperOC::eval_u(dataOut, g, dataIn, rxss, interp_method);
	FLOAT_TYPE max_val = dataOut[0];
	std::for_each(dataOut.cbegin()+1, dataOut.cend(), [&max_val](const auto& rhs) {
		if ((max_val < rhs) || std::isnan(max_val)) max_val = rhs;
	});
	std::transform(dataOut.cbegin(), dataOut.cend(), dataOut.begin(), [max_val](const auto& rhs) { return (std::isnan(rhs)) ? max_val : rhs; });
	return true;
}
