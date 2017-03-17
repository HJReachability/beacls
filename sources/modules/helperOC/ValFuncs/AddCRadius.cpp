#include <helperOC/helperOC_type.hpp>
#include <helperOC/ValFuncs/AddCRadius.hpp>
#include <helperOC/DynSys/KinVehicleND/KinVehicleND.hpp>
#include <helperOC/DynSys/DynSys/DynSysSchemeData.hpp>
#include <helperOC/ValFuncs/HJIPDE.hpp>
#include <levelset/Grids/HJI_Grid.hpp>
#include <vector>
#include <iostream>
#include <algorithm>
#include "AddCRadius_impl.hpp"

helperOC::AddCRadius_impl::AddCRadius_impl(
	const helperOC::ExecParameters& execParameters
) :
	extraArgs(HJIPDE_extraArgs()),
	extraOuts(HJIPDE_extraOuts()) {
	extraArgs.execParameters = execParameters;
	extraArgs.execParameters.useCuda = false;
	extraArgs.execParameters.delayedDerivMinMax = levelset::DelayedDerivMinMax_Disable;
	extraArgs.execParameters.enable_user_defined_dynamics_on_gpu = false;
	extraArgs.execParameters.calcTTR = false;
	hjipde = new HJIPDE();
	schemeData = new DynSysSchemeData();
}
helperOC::AddCRadius_impl::~AddCRadius_impl() {
	if (hjipde) delete hjipde;
	if (schemeData) delete schemeData;
}

helperOC::AddCRadius::AddCRadius(
	const helperOC::ExecParameters& execParameters
)
{
	pimpl = new AddCRadius_impl(execParameters);
}
helperOC::AddCRadius::~AddCRadius() {
	if (pimpl) delete pimpl;
}
bool helperOC::AddCRadius_impl::operator()(
	beacls::FloatVec& dataOut,
	const levelset::HJI_Grid* gIn,
	const beacls::FloatVec& dataIn,
	const FLOAT_TYPE radius
) {
	//!< Solve HJI PDE for expanding set
	beacls::FloatVec x(gIn->get_num_of_dimensions());
	std::fill(x.begin(), x.end(), (FLOAT_TYPE)0);
	DynSys* dynSys = new KinVehicleND(x, 1);
	schemeData->dynSys = dynSys;
	schemeData->set_grid(gIn);
	extraArgs.quiet = true;
	extraArgs.keepLast = true;
	beacls::FloatVec stoptau;
	hjipde->solve(stoptau, extraOuts, dataIn, beacls::FloatVec{0, radius}, schemeData, HJIPDE::MinWithType_Zero, extraArgs);

	//!< Discard initial set from output
	hjipde->get_last_data(dataOut);
	if (dynSys) delete dynSys;
	return false;
}
	
bool helperOC::AddCRadius::operator()(
	beacls::FloatVec& dataOut,
	const levelset::HJI_Grid* gIn,
	const beacls::FloatVec& dataIn,
	const FLOAT_TYPE radius
	) {
	if (pimpl) return pimpl->operator()(dataOut, gIn, dataIn, radius);
	else return false;
}