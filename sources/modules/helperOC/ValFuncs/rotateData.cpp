#include <helperOC/helperOC_type.hpp>
#include <helperOC/ValFuncs/rotateData.hpp>
#include <helperOC/ValFuncs/eval_u.hpp>
#include <levelset/Grids/HJI_Grid.hpp>
#include <vector>
#include <iostream>
#include <algorithm>
#include <Core/UVec.hpp>

bool helperOC::rotateData(
	beacls::FloatVec& dataOut,
	const levelset::HJI_Grid* g,
	const beacls::FloatVec& dataIn,
	const FLOAT_TYPE theta,
	const beacls::IntegerVec& pdims,
	const beacls::IntegerVec& adims,
	const beacls::Interpolate_Type interp_method
	)
{
	const size_t begin_index = 0;
	const size_t length = 0;
	beacls::UVec xs_pdims0_uvec;
	beacls::UVec xs_pdims1_uvec;
	g->get_xs(xs_pdims0_uvec, pdims[0], begin_index, length);
	g->get_xs(xs_pdims1_uvec, pdims[0], begin_index, length);
	const beacls::FloatVec* xs_pdims0_ptr = beacls::UVec_<FLOAT_TYPE>(xs_pdims0_uvec).vec();
	const beacls::FloatVec* xs_pdims1_ptr = beacls::UVec_<FLOAT_TYPE>(xs_pdims1_uvec).vec();

	std::vector<beacls::FloatVec> rxss(xs_pdims0_ptr->size());
	rxss[pdims[0]].resize(xs_pdims0_ptr->size());
	rxss[pdims[1]].resize(xs_pdims1_ptr->size());

	//!< Get a list of new indices
	//!< Multiply by rotation matrix for position dimensions
	std::transform(xs_pdims0_ptr->cbegin(), xs_pdims0_ptr->cend(), xs_pdims1_ptr->cbegin(), rxss.begin(), [theta, pdims, adims](const auto& lhs, const auto& rhs) {
		beacls::FloatVec res(2+adims.size());
		res[pdims[0]] = std::cos(-theta) * lhs - sin(-theta) * rhs;
		res[pdims[1]] = std::sin(-theta) * lhs + cos(-theta) * rhs;
		return res;
	});
	//!< Translate in angle
	if (!adims.empty()) {
		std::for_each(adims.cbegin(), adims.cend(), [theta, &rxss, g, begin_index, length](const auto& rhs) {
			const size_t adim = rhs;

			beacls::UVec xs_adim_uvec;
			g->get_xs(xs_adim_uvec, adim, begin_index, length);
			const beacls::FloatVec* xs_adim_ptr = beacls::UVec_<FLOAT_TYPE>(xs_adim_uvec).vec();
			std::transform(xs_adim_ptr->cbegin(), xs_adim_ptr->cend(), rxss.begin(), rxss.begin(), [theta, adim](const auto& lhs, auto& rhs) {
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
