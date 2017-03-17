#include <helperOC/ValFuncs/eval_u.hpp>
#include <Core/interpn.hpp>
#include <levelset/Grids/HJI_Grid.hpp>
#include <vector>
#include <deque>
#include <iostream>
#include <algorithm>
#include <typeinfo>
#include <levelset/BoundaryCondition/AddGhostPeriodic.hpp>
//#include <helperOC/ValFuncs/augmentPeriodicData.hpp>
#include <macro.hpp>

namespace helperOC {
	/*
	@brief	Computes the interpolated value of a value function data at state x
	@param	[in]	g		grid
	@param	[in]	data	implicit function describing the set
	@param	[in]	x		points to check; each row is a point
	@@aram	[out]	v		value at points x
	@param	[in]	interp_method	interporation method
	@retval	true					Succeeded
	@retval false					Failed
	*/
	static bool eval_u_single(
		beacls::FloatVec& v,
		const levelset::HJI_Grid* g,
		const beacls::FloatVec& data,
		const std::vector<beacls::FloatVec >& xss,
		const beacls::Interpolate_Type interp_method
	);
}
static
bool helperOC::eval_u_single(
	beacls::FloatVec& v,
	const levelset::HJI_Grid* g,
	const beacls::FloatVec& data,
	const std::vector<beacls::FloatVec >& xss,
	const beacls::Interpolate_Type interp_method
) {
	std::vector<beacls::FloatVec> modified_xss = xss;


	//!< Dealing with periodicity
	const size_t num_of_dimensions = g->get_num_of_dimensions();
	std::vector<beacls::Extrapolate_Type> extrapolate_methods;
	extrapolate_methods.resize(num_of_dimensions, beacls::Extrapolate_none);
	std::deque<bool> i_above_bounds;
	std::deque<bool> i_below_bounds;
	for (size_t dimension = 0; dimension < num_of_dimensions; ++dimension) {
		const levelset::BoundaryCondition* boundaryCondition = g->get_boundaryCondition(dimension);
		if (boundaryCondition && (typeid(*boundaryCondition) == typeid(levelset::AddGhostPeriodic))) {
			extrapolate_methods[dimension] = beacls::Extrapolate_periodic;
			//!< Map input points to within grid bounds
			const beacls::FloatVec& vs = g->get_vs(dimension);
			const FLOAT_TYPE dx_d = g->get_dx(dimension);
			if (!vs.empty()) {
				auto minmax_pair = beacls::minmax_value<FLOAT_TYPE>(vs.cbegin(), vs.cend());

				const FLOAT_TYPE min_vs = minmax_pair.first;
				const FLOAT_TYPE max_vs = minmax_pair.second + dx_d;
				const FLOAT_TYPE period = max_vs - min_vs;

				std::for_each(modified_xss.begin(), modified_xss.end(), [&period, &dimension, max_vs, min_vs](auto& rhs) {
					const FLOAT_TYPE val = rhs[dimension];
					const FLOAT_TYPE modulo_val = val - std::floor((val - min_vs) / period) * period;
					rhs[dimension] = modulo_val;
				});
			}
		}
	}

	//!< Interpolate
	//!< Input checking
	//!< eg.v = interpn(g.vs{ 1 }, g.vs{ 2 }, data, x(:, 1), x(:, 2), interp_method)
	std::vector<beacls::FloatVec > interpc_argin_vs(num_of_dimensions);
	std::vector<beacls::FloatVec > interpc_argin_x(num_of_dimensions);
	const std::vector< beacls::FloatVec >& vs = g->get_vss();
	const std::size_t xss_size = modified_xss.size();
	std::copy(vs.cbegin(), vs.cend(), interpc_argin_vs.begin());
	std::for_each(interpc_argin_x.begin(), interpc_argin_x.end(), [&modified_xss, xss_size](auto& rhs) { rhs.resize(xss_size); });
	for (size_t interp_point = 0; interp_point < xss_size; ++interp_point) {
		for (size_t dimension = 0; dimension < num_of_dimensions; ++dimension) {
			interpc_argin_x[dimension][interp_point] = modified_xss[interp_point][dimension];
		}
	}
	std::vector<const beacls::FloatVec*> X_ptrs;
	std::vector<beacls::IntegerVec> Ns;
	//	const std::size_t x_size_for_each_dimension = static_cast<size_t>(std::floor((FLOAT_TYPE)x.size() / num_of_dimensions));

	X_ptrs.reserve(num_of_dimensions * 2 + 1);
	Ns.reserve(num_of_dimensions * 2 + 1);
	std::for_each(interpc_argin_vs.cbegin(), interpc_argin_vs.cend(), [&X_ptrs, &Ns](const auto& rhs) {
		X_ptrs.push_back(&rhs);
		beacls::IntegerVec N{ rhs.size() };
		Ns.push_back(N);
	});
	X_ptrs.push_back(&data);
	Ns.push_back(g->get_Ns());
	std::for_each(interpc_argin_x.cbegin(), interpc_argin_x.cend(), [&X_ptrs, &Ns](const auto& rhs) {
		X_ptrs.push_back(&rhs);
		beacls::IntegerVec N{ rhs.size() };
		Ns.push_back(N);
	});
	beacls::interpn(v, X_ptrs, Ns, interp_method, extrapolate_methods);
	return false;
}

bool helperOC::eval_u(
	beacls::FloatVec& dataOut,
	const levelset::HJI_Grid* g,
	const beacls::FloatVec& data,
	const std::vector<beacls::FloatVec >& xs,
	const beacls::Interpolate_Type interp_method
) {
	return eval_u_single(dataOut, g, data, xs, interp_method);
}
bool helperOC::eval_u(
	std::vector<beacls::FloatVec>& dataOuts,
	const levelset::HJI_Grid* g,
	const std::vector<beacls::FloatVec >& datas,
	const beacls::FloatVec& x,
	const beacls::Interpolate_Type interp_method
) {
	bool result = true;
	if(dataOuts.size()!=datas.size()) dataOuts.resize(datas.size());
	for (size_t i = 0; i < datas.size(); ++i) {
		result &= eval_u_single(dataOuts[i], g, datas[i], std::vector<beacls::FloatVec>{x}, interp_method);
	}
	return result;
}
bool helperOC::eval_u(
	std::vector<beacls::FloatVec>& dataOuts,
	const std::vector<levelset::HJI_Grid*>& gs,
	const std::vector<beacls::FloatVec >& datas,
	const std::vector<beacls::FloatVec >& xs,
	const beacls::Interpolate_Type interp_method
) {
	bool result = true;
	if (dataOuts.size() != datas.size()) dataOuts.resize(datas.size());
	for (size_t i = 0; i < datas.size(); ++i) {
		result &= eval_u_single(dataOuts[i], gs[i], datas[i], std::vector<beacls::FloatVec>{xs[i]}, interp_method);
	}
	return result;
}
