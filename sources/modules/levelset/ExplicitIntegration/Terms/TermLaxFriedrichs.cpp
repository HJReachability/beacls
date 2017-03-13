#include <levelset/ExplicitIntegration/Terms/TermLaxFriedrichs.hpp>
#include <levelset/Grids/HJI_Grid.hpp>
#include "TermLaxFriedrichs_impl.hpp"
#include "TermLaxFriedrichs_cuda.hpp"
#include <algorithm>
#include <functional>
#include <typeinfo>
#include <levelset/ExplicitIntegration/SchemeData.hpp>
#include <levelset/SpatialDerivative/SpatialDerivative.hpp>
#include <levelset/ExplicitIntegration/Dissipations/Dissipation.hpp>
#include <Core/CacheTag.hpp>

TermLaxFriedrichs_impl::TermLaxFriedrichs_impl(
	const SchemeData* schemeData,
	const beacls::UVecType type
	) :
	first_dimension_loop_size(schemeData->get_grid()->get_N(0)),
	num_of_dimensions(schemeData->get_grid()->get_num_of_dimensions()),
	deriv_l_uvecs(num_of_dimensions),
	deriv_r_uvecs(num_of_dimensions),
	deriv_c_uvecs(num_of_dimensions),
	type(type),
	cacheTag(new beacls::CacheTag())
	{
}
TermLaxFriedrichs_impl::~TermLaxFriedrichs_impl() {
	if (cacheTag) delete cacheTag;
}

TermLaxFriedrichs_impl::TermLaxFriedrichs_impl(const TermLaxFriedrichs_impl& rhs) :
	first_dimension_loop_size(rhs.first_dimension_loop_size),
	num_of_dimensions(rhs.num_of_dimensions),
	type(rhs.type),
	cacheTag(new beacls::CacheTag())
{
	deriv_l_uvecs.resize(rhs.deriv_l_uvecs.size());
	deriv_r_uvecs.resize(rhs.deriv_r_uvecs.size());
	deriv_c_uvecs.resize(rhs.deriv_c_uvecs.size());
	x_uvecs.resize(rhs.x_uvecs.size());
}
bool TermLaxFriedrichs_impl::execute(
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
) {
	const HJI_Grid *grid = schemeData->get_grid();
	SpatialDerivative* spatialDerivative = schemeData->get_spatialDerivative();
	Dissipation* dissipation = schemeData->get_dissipation();
	beacls::UVecDepth depth = beacls::type_to_depth<FLOAT_TYPE>();
	beacls::UVec y_uvec(y, beacls::UVecType_Vector, false);

	//	const HamiltonJacobiFunction* hamiltonJacobiFunction = schemeData->get_hamiltonJacobiFunction();
	//---------------------------------------------------------------------------
	// Get upwinded and centered derivative approximations.

	size_t f_d_l_size = first_dimension_loop_size;
	size_t grid_length = num_of_slices*loop_length*f_d_l_size;
	size_t slice_length = loop_length*f_d_l_size;
	if (ham_uvec.type() != type) ham_uvec = beacls::UVec(depth, type, grid_length);
	else if (ham_uvec.size() != grid_length) ham_uvec.resize(grid_length);
	if (diss_uvec.type() != type) diss_uvec = beacls::UVec(depth, type, grid_length);
	else if (diss_uvec.size() != grid_length) diss_uvec.resize(grid_length);
	if (deriv_l_uvecs.size() != num_of_dimensions) deriv_l_uvecs.resize(num_of_dimensions);
	if (deriv_r_uvecs.size() != num_of_dimensions) deriv_r_uvecs.resize(num_of_dimensions);
	if (deriv_c_uvecs.size() != num_of_dimensions) deriv_c_uvecs.resize(num_of_dimensions);

	step_bound_invs.assign(num_of_dimensions,0.0);

	for_each(deriv_l_uvecs.begin(), deriv_l_uvecs.end(), ([depth,grid_length,this](auto &rhs) {
		if (rhs.type() != type) rhs = beacls::UVec(depth, type, grid_length);
		else if (rhs.size() != grid_length) rhs.resize(grid_length);
	}));
	for_each(deriv_r_uvecs.begin(), deriv_r_uvecs.end(), ([depth, grid_length, this](auto &rhs) {
		if (rhs.type() != type) rhs = beacls::UVec(depth, type, grid_length);
		else if (rhs.size() != grid_length) rhs.resize(grid_length);
	}));
	for_each(deriv_c_uvecs.begin(), deriv_c_uvecs.end(), ([depth, grid_length, this](auto &rhs) {
		if (rhs.type() != type) rhs = beacls::UVec(depth, type, grid_length);
		else if (rhs.size() != grid_length) rhs.resize(grid_length);
	}));

	size_t src_index_term = loop_begin * f_d_l_size;
	if (!cacheTag->check_tag(t, loop_begin, slice_length*num_of_slices)) {
		//!< Copy xs to Cuda memory asynchronously in spatial derivative functions
		x_uvecs.resize(num_of_dimensions);
		if (enable_user_defined_dynamics_on_gpu && (type == beacls::UVecType_Cuda)) {
			for (size_t dimension = 0; dimension < num_of_dimensions; ++dimension) {
				if (x_uvecs[dimension].type() != beacls::UVecType_Cuda) x_uvecs[dimension] = beacls::UVec(depth, beacls::UVecType_Cuda, grid_length);
				else x_uvecs[dimension].resize(grid_length);
			}
		}
		for (size_t index = 0; index < num_of_dimensions; ++index) {
			//!< To optimize asynchronous execution, calculate from heavy dimension (0, 2, 3 ... 1);
			const size_t dimension = (index == 0) ? index : (index == num_of_dimensions - 1) ? 1 : index + 1;
			beacls::UVec& deriv_l_uvec = deriv_l_uvecs[dimension];
			beacls::UVec& deriv_r_uvec = deriv_r_uvecs[dimension];

			spatialDerivative->execute(
				deriv_l_uvec,
				deriv_r_uvec,
				grid,
				y.data(),
				dimension,
				false,
				loop_begin,
				slice_length,
				num_of_slices);
			const beacls::FloatVec& xs = grid->get_xs(dimension);
			beacls::copyHostPtrToUVecAsync(x_uvecs[dimension], xs.data() + src_index_term, grid_length);
			beacls::UVec& deriv_c_uvec = deriv_c_uvecs[dimension];
			beacls::average(deriv_l_uvec, deriv_r_uvec, deriv_c_uvec);
		}
		for (size_t dimension = 0; dimension < num_of_dimensions; ++dimension) {
			beacls::UVec& x_uvecs_dim = x_uvecs[dimension];
			synchronizeUVec(x_uvecs_dim);
		}
		cacheTag->set_tag(t, loop_begin, slice_length*num_of_slices);
	}

	deriv_max_uvecs.resize(derivMaxs.size());
	deriv_min_uvecs.resize(derivMins.size());
	std::transform(derivMaxs.cbegin(), derivMaxs.cend(), deriv_max_uvecs.begin(), [](const auto& rhs) { return beacls::UVec(rhs, beacls::UVecType_Vector, false); });
	std::transform(derivMins.cbegin(), derivMins.cend(), deriv_min_uvecs.begin(), [](const auto& rhs) { return beacls::UVec(rhs, beacls::UVecType_Vector, false); });

	if (!enable_user_defined_dynamics_on_gpu 
		|| type != beacls::UVecType_Cuda
		|| !schemeData->hamFunc_cuda(
			ham_uvec,
			t,
			y_uvec,
			x_uvecs,
			deriv_c_uvecs,
			src_index_term, grid_length)) {
		deriv_c_cpu_uvecs.resize(deriv_c_uvecs.size());
		std::transform(deriv_c_uvecs.cbegin(), deriv_c_uvecs.cend(), deriv_c_cpu_uvecs.begin(), ([](const auto& rhs) {
			if (rhs.type() == beacls::UVecType_Cuda) {
				beacls::UVec tmp; rhs.convertTo(tmp, beacls::UVecType_Vector); return tmp;
			}
			else return rhs;
		}));

		schemeData->hamFunc(
			ham_uvec,
			t,
			y_uvec,
			deriv_c_cpu_uvecs,
			src_index_term, grid_length);
	}
	beacls::FloatVec new_step_bound_invs(num_of_dimensions);
	if (!dissipation->execute(
		diss_uvec,
		new_step_bound_invs,
		deriv_min_uvecs,
		deriv_max_uvecs,
		t,
		y_uvec,
		x_uvecs,
		deriv_l_uvecs,
		deriv_r_uvecs,
		schemeData,
		src_index_term,
		enable_user_defined_dynamics_on_gpu,
		updateDerivMinMax)) {
		//!< If partial function called by dissipation requires global deriv min/max and it returns false,
		// fall back to integrator and reduce partial mins/maxs of deriv to global one, then come to this function.
		return false;
	}
	for (size_t dimension = 0; dimension < num_of_dimensions; ++dimension) {
		if (new_step_bound_invs[dimension] > step_bound_invs[dimension]) step_bound_invs[dimension] = new_step_bound_invs[dimension];
	}

	if (beacls::is_cuda(diss_uvec) && beacls::is_cuda(ham_uvec)) {
		//!< Synchronize copy from Device to Host of last call.
		beacls::synchronizeUVec(ydot_cuda_uvec);
		beacls::reallocateAsSrc(ydot_cuda_uvec, ham_uvec);
		TermLaxFriedrichs_execute_cuda(ydot_cuda_uvec, diss_uvec, ham_uvec);
		copyUVecToHostAsync(&ydot_ite[0], ydot_cuda_uvec);
	}
	else
	{
		beacls::UVec diss_cpu_uvec;
		beacls::synchronizeUVec(diss_uvec);
		beacls::synchronizeUVec(ham_uvec);
		if (beacls::is_cuda(diss_uvec)) diss_uvec.convertTo(diss_cpu_uvec, beacls::UVecType_Vector);
		else diss_cpu_uvec = diss_uvec;
		if (beacls::is_cuda(ham_uvec)) ham_uvec.convertTo(ham_cpu_uvec, beacls::UVecType_Vector);
		else ham_cpu_uvec = ham_uvec;

		const beacls::FloatVec* diss_vec_ptr = beacls::UVec_<FLOAT_TYPE>(diss_cpu_uvec).vec();
		const beacls::FloatVec* ham_vec_ptr = beacls::UVec_<FLOAT_TYPE>(ham_cpu_uvec).vec();
		std::transform(diss_vec_ptr->cbegin(), diss_vec_ptr->cend(), ham_vec_ptr->cbegin(), ydot_ite, std::minus<FLOAT_TYPE>());
	}
	return true;
}

TermLaxFriedrichs::TermLaxFriedrichs(
	const SchemeData* schemeData,
	const beacls::UVecType type
) {
	pimpl = new TermLaxFriedrichs_impl(schemeData,type);
}
TermLaxFriedrichs::~TermLaxFriedrichs() {
	if (pimpl) delete pimpl;
}
bool TermLaxFriedrichs::execute(
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
bool TermLaxFriedrichs_impl::synchronize(
	const SchemeData*
) const {
	beacls::synchronizeUVec(ydot_cuda_uvec);
	return true;
}
bool TermLaxFriedrichs::synchronize(const SchemeData *schemeData) const {
	if (pimpl) return pimpl->synchronize(schemeData);
	else return true;
}
bool TermLaxFriedrichs::operator==(const TermLaxFriedrichs& rhs) const {
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
bool TermLaxFriedrichs::operator==(const Term& rhs) const {
	if (this == &rhs) return true;
	else if (typeid(*this) != typeid(rhs)) return false;
	else return operator==(dynamic_cast<const TermLaxFriedrichs&>(rhs));
}
TermLaxFriedrichs::TermLaxFriedrichs(const TermLaxFriedrichs& rhs) :
	pimpl(rhs.pimpl->clone())
{
};

TermLaxFriedrichs* TermLaxFriedrichs::clone() const {
	return new TermLaxFriedrichs(*this);
};
