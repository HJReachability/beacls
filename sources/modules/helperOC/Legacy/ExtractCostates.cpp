#include <iostream>
#include <cstring>
#include <cstdint>


#include <helperOC/Legacy/ExtractCostates.hpp>
#include <helperOC/ComputeGradients.hpp>
#include "ExtractCostates_impl.hpp"
#include <helperOC/helperOC_type.hpp>
using namespace helperOC;

ExtractCostates_impl::ExtractCostates_impl(
	helperOC::ApproximationAccuracy_Type accuracy
) : accuracy(accuracy), computeGradients(NULL) {
	
}
ExtractCostates_impl::~ExtractCostates_impl() {
	if (computeGradients) delete computeGradients;
}
bool ExtractCostates_impl::operator()(
	std::vector<beacls::FloatVec >& derivC,
	std::vector<beacls::FloatVec >& derivL,
	std::vector<beacls::FloatVec >& derivR,
	const levelset::HJI_Grid* grid,
	const beacls::FloatVec& data,
	const size_t data_length,
	const bool upWind,
	const helperOC::ExecParameters& execParameters
	) {
	const beacls::UVecType type = (execParameters.useCuda) ? beacls::UVecType_Cuda : beacls::UVecType_Vector;
	if (!computeGradients) computeGradients = new helperOC::ComputeGradients(grid, accuracy, type);
	return computeGradients->operator()(derivC, derivL, derivR, grid, data, data_length, upWind, execParameters);
}

ExtractCostates::ExtractCostates(
	helperOC::ApproximationAccuracy_Type accuracy
) {
	pimpl = new ExtractCostates_impl(accuracy);
}
ExtractCostates::~ExtractCostates() {
	if (pimpl) delete pimpl;
}

bool ExtractCostates::operator()(
	std::vector<beacls::FloatVec >& derivC,
	std::vector<beacls::FloatVec >& derivL,
	std::vector<beacls::FloatVec >& derivR,
	const levelset::HJI_Grid* grid,
	const beacls::FloatVec& data,
	const size_t data_length,
	const bool upWind,
	const helperOC::ExecParameters& execParameters
	) {
	if (pimpl) return pimpl->operator()(derivC, derivL, derivR, grid, data, data_length, upWind, execParameters);
	return false;
}
