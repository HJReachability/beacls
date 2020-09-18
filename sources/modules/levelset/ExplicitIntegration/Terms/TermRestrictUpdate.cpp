#include <levelset/ExplicitIntegration/Terms/TermRestrictUpdate.hpp>
#include <levelset/Grids/HJI_Grid.hpp>
#include <levelset/ExplicitIntegration/SchemeData.hpp>
#include <cmath>
#include <numeric>
#include <typeinfo>
#include "TermRestrictUpdate_impl.hpp"
using namespace levelset;

TermRestrictUpdate_impl::TermRestrictUpdate_impl(
	const beacls::UVecType type
) : type(type) {
}
TermRestrictUpdate_impl::~TermRestrictUpdate_impl() {
}
bool TermRestrictUpdate_impl::execute(
	beacls::FloatVec::iterator ydot_ite,
	beacls::FloatVec& step_bound_invs,
	const FLOAT_TYPE t,
	const beacls::FloatVec& y,
	std::vector<beacls::FloatVec >& derivMins,
	std::vector<beacls::FloatVec >& derivMaxs,
	const SchemeData *schemeData,
	const size_t loop_begin,
	const size_t loop_length,
	const size_t num_of_slices,
	const bool enable_user_defined_dynamics_on_gpu,
	const bool updateDerivMinMax
) const {
	const HJI_Grid* grid = schemeData->get_grid();
	if (!grid) return false;
	const Term *innerFunc = schemeData->get_innerFunc();
	const SchemeData* innerData = schemeData->get_innerData();
	const bool positive = schemeData->get_positive();
	size_t f_d_l_size = schemeData->get_grid()->get_N(0);
	size_t grid_length = num_of_slices*loop_length*f_d_l_size;

	//---------------------------------------------------------------------------
	// Get the unrestricted update.
	//[ unRestricted, stepBound, innerData ] = ...
	//                            feval(thisSchemeData.innerFunc, t, y, innerData);
	if (innerFunc->execute(ydot_ite, step_bound_invs, t, y, derivMins, derivMaxs, innerData, loop_begin, loop_length, num_of_slices, enable_user_defined_dynamics_on_gpu, updateDerivMinMax)) {
		innerFunc->synchronize(schemeData);
		//---------------------------------------------------------------------------
		// Default to positive (nonnegative) update restriction.

		//---------------------------------------------------------------------------
		// Restrict the update (stepBound is returned unchanged).  
		//   Do not negate for RHS of ODE (that is handled by innerFunc).
		if (positive) {
			std::transform(ydot_ite, ydot_ite + grid_length, ydot_ite, ([](const auto& rhs) {return std::max<FLOAT_TYPE>(rhs, 0.); }));

		}
		else {
			std::transform(ydot_ite, ydot_ite + grid_length, ydot_ite, ([](const auto& rhs) {return std::min<FLOAT_TYPE>(rhs, 0.); }));
		}
	}
	else {
		return false;
	}
	return true;

}
bool TermRestrictUpdate_impl::execute_local_q(
	beacls::FloatVec::iterator ydot_ite,
	beacls::FloatVec& step_bound_invs,
	const FLOAT_TYPE t,
	const beacls::FloatVec& y,
	std::vector<beacls::FloatVec >& derivMins,
	std::vector<beacls::FloatVec >& derivMaxs,
	const SchemeData *schemeData,
	const size_t loop_begin,
	const size_t loop_length,
	const size_t num_of_slices,
	const bool enable_user_defined_dynamics_on_gpu,
	const bool updateDerivMinMax,
	const std::set<size_t> &Q 
) const {
	const HJI_Grid* grid = schemeData->get_grid();
	if (!grid) return false;
	const Term *innerFunc = schemeData->get_innerFunc();
	const SchemeData* innerData = schemeData->get_innerData();
	const bool positive = schemeData->get_positive();
	size_t f_d_l_size = schemeData->get_grid()->get_N(0);
	size_t grid_length = num_of_slices*loop_length*f_d_l_size;

	//---------------------------------------------------------------------------
	// Get the unrestricted update.
	//[ unRestricted, stepBound, innerData ] = ...
	//                            feval(thisSchemeData.innerFunc, t, y, innerData);
	if (innerFunc->execute(ydot_ite, step_bound_invs, t, y, derivMins, derivMaxs, innerData, loop_begin, loop_length, num_of_slices, enable_user_defined_dynamics_on_gpu, updateDerivMinMax)) {
		innerFunc->synchronize(schemeData);
		//---------------------------------------------------------------------------
		// Default to positive (nonnegative) update restriction.
		//---------------------------------------------------------------------------
		// Restrict the update (stepBound is returned unchanged).  
		//   Do not negate for RHS of ODE (that is handled by innerFunc).
		if (positive) {
			std::transform(ydot_ite, ydot_ite + grid_length, ydot_ite, ([](const auto& rhs) {return std::max<FLOAT_TYPE>(rhs, 0.); }));
		}
		else {
			std::transform(ydot_ite, ydot_ite + grid_length, ydot_ite, ([](const auto& rhs) {return std::min<FLOAT_TYPE>(rhs, 0.); }));
		}
	}
	else {
		return false;
	}
	return true;

}
TermRestrictUpdate::TermRestrictUpdate(
	const beacls::UVecType type
) {
	pimpl = new TermRestrictUpdate_impl(type);
}
TermRestrictUpdate::~TermRestrictUpdate() {
	if (pimpl) delete pimpl;
}
bool TermRestrictUpdate::execute(
	beacls::FloatVec::iterator ydot_ite,
	beacls::FloatVec& step_bound_invs,
	const FLOAT_TYPE t,
	const beacls::FloatVec& y,
	std::vector<beacls::FloatVec >& derivMins,
	std::vector<beacls::FloatVec >& derivMaxs,
	const SchemeData *schemeData,
	const size_t loop_begin,
	const size_t loop_length,
	const size_t num_of_slices,
	const bool enable_user_defined_dynamics_on_gpu,
	const bool updateDerivMinMax
) const {
	if (pimpl) return pimpl->execute(ydot_ite, step_bound_invs, t, y, derivMins, derivMaxs, schemeData, loop_begin, loop_length, num_of_slices, enable_user_defined_dynamics_on_gpu, updateDerivMinMax);
	return false;
}
bool TermRestrictUpdate::execute_local_q(
	beacls::FloatVec::iterator ydot_ite,
	beacls::FloatVec& step_bound_invs,
	const FLOAT_TYPE t,
	const beacls::FloatVec& y,
	std::vector<beacls::FloatVec >& derivMins,
	std::vector<beacls::FloatVec >& derivMaxs,
	const SchemeData *schemeData,
	const size_t loop_begin,
	const size_t loop_length,
	const std::set<size_t> &Q, 
	const size_t num_of_slices,
	const bool enable_user_defined_dynamics_on_gpu,
	const bool updateDerivMinMax 
) const {
	if (pimpl) return pimpl->execute_local_q(ydot_ite, step_bound_invs, t, y, derivMins, derivMaxs, schemeData, loop_begin, loop_length, num_of_slices, enable_user_defined_dynamics_on_gpu, updateDerivMinMax, Q);
	return false;
}
bool TermRestrictUpdate_impl::synchronize(
	const SchemeData *schemeData
	) const {
	if (schemeData) {
		const Term *innerFunc = schemeData->get_innerFunc();
		if (innerFunc) {
			return innerFunc->synchronize(schemeData);
		}
	}
	return true;
}
bool TermRestrictUpdate::synchronize(const SchemeData *schemeData) const {
	if (pimpl) return pimpl->synchronize(schemeData);
	else return true;
}

bool TermRestrictUpdate::operator==(const TermRestrictUpdate& rhs) const {
	if (this == &rhs) return true;
	else if (!pimpl) {
		if (!rhs.pimpl) return true;
		else return false;
	}
	else {
		if (!rhs.pimpl) return false;
		else {
			if (pimpl == rhs.pimpl) return true;
			else if (*pimpl == *rhs.pimpl) return true;
			else return false;
		}
	}
}
bool TermRestrictUpdate::operator==(const Term& rhs) const {
	if (this == &rhs) return true;
	else if (typeid(*this) != typeid(rhs)) return false;
	else return operator==(dynamic_cast<const TermRestrictUpdate&>(rhs));
}
TermRestrictUpdate::TermRestrictUpdate(const TermRestrictUpdate& rhs) :
	pimpl(rhs.pimpl->clone())
{
};

TermRestrictUpdate* TermRestrictUpdate::clone() const {
	return new TermRestrictUpdate(*this);
};
