#include <levelset/ExplicitIntegration/Dissipations/ArtificialDissipationGLF.hpp>
#include <levelset/ExplicitIntegration/SchemeData.hpp>
#include <levelset/ExplicitIntegration/PartialFunction.hpp>
#include <levelset/Grids/HJI_Grid.hpp>
#include "ArtificialDissipationGLF_impl.hpp"
#include "ArtificialDissipationGLF_cuda.hpp"
#include <algorithm>
#include <numeric>
#include <functional>
#include <iomanip>
#include <typeinfo>
#include <cuda_macro.hpp>
#include <macro.hpp>
using namespace levelset;
ArtificialDissipationGLF_impl::ArtificialDissipationGLF_impl(
	) {
}
ArtificialDissipationGLF_impl::~ArtificialDissipationGLF_impl() {

}
bool ArtificialDissipationGLF_impl::execute(
	beacls::UVec& diss,
	beacls::FloatVec& step_bound_invs,
	std::vector<beacls::UVec>& derivMins,
	std::vector<beacls::UVec>& derivMaxs,
	const FLOAT_TYPE t,
	const beacls::UVec& data,
	const std::vector<beacls::UVec>& x_uvecs,
	const std::vector<beacls::UVec >& deriv_ls,
	const std::vector<beacls::UVec >& deriv_rs,
	const SchemeData *schemeData,
	const size_t begin_index,
	const bool enable_user_defined_dynamics_on_gpu,
	const bool updateDerivMinMax
) {
	const HJI_Grid* hji_grid = schemeData->get_grid();
	if (!hji_grid) return false;

	//	const beacls::FloatVec& alphas = hji_grid->get_alphas(dimension);
	// Now calculate the dissipation.  Since alpha is the effective speed of
	//   the flow, it provides the CFL timestep bound too.
	size_t loop_size = (size_t)deriv_ls[0].size();
	beacls::FloatVec data_vec;
	size_t num_of_dimensions = deriv_ls.size();
	if (step_bound_invs.size() != num_of_dimensions) step_bound_invs.resize(num_of_dimensions);
	bool result = true;
	alphas_cpu_uvecs.resize(num_of_dimensions);
	alphas_cuda_uvecs.resize(num_of_dimensions);
	for (size_t dimension = 0; dimension < num_of_dimensions; ++dimension) {
		//!< Call partial function without global deriv min/max at first.
		bool partialFunc_result = false;
		if (enable_user_defined_dynamics_on_gpu
			&& (beacls::is_cuda(deriv_ls[0]))
			&& schemeData->partialFunc_cuda(alphas_cuda_uvecs[dimension], t, data, x_uvecs, derivMins, derivMaxs, dimension, begin_index, loop_size)) {
			partialFunc_result = true;
		} else {
			if (schemeData->partialFunc(alphas_cpu_uvecs[dimension], t, data, derivMins, derivMaxs, dimension, begin_index, loop_size)) {
				partialFunc_result = true;
				if (beacls::is_cuda(deriv_ls[0]) && (alphas_cpu_uvecs[dimension].size() != 1)) {
					alphas_cpu_uvecs[dimension].convertTo(alphas_cuda_uvecs[dimension], beacls::UVecType_Cuda);
				}
				else {
					alphas_cuda_uvecs[dimension] = alphas_cpu_uvecs[dimension];
				}
			}
		}

		if (!partialFunc_result) {
			result = false;
		}
		else {
			if (deriv_ls[dimension].type() == beacls::UVecType_Cuda) {
				ArtificialDissipationGLF_execute_cuda(
					diss,
					step_bound_invs[dimension],
					alphas_cuda_uvecs[dimension],
					deriv_ls[dimension],
					deriv_rs[dimension],
					hji_grid->get_dxInv(dimension),
					dimension,
					loop_size);
			}
			else
			{
				beacls::UVec tmp_deriv_l;
				if (deriv_ls[dimension].type() == beacls::UVecType_Cuda) deriv_ls[dimension].convertTo(tmp_deriv_l, beacls::UVecType_Vector);
				else tmp_deriv_l = deriv_ls[dimension];
				beacls::UVec tmp_deriv_r;
				if (deriv_rs[dimension].type() == beacls::UVecType_Cuda) deriv_rs[dimension].convertTo(tmp_deriv_r, beacls::UVecType_Vector);
				else tmp_deriv_r = deriv_rs[dimension];

				const FLOAT_TYPE* deriv_l = beacls::UVec_<FLOAT_TYPE>(tmp_deriv_l).ptr();
				const FLOAT_TYPE* deriv_r = beacls::UVec_<FLOAT_TYPE>(tmp_deriv_r).ptr();
				FLOAT_TYPE* diss_ptr = beacls::UVec_<FLOAT_TYPE>(diss).ptr();
				const FLOAT_TYPE* alpha_ptr = beacls::UVec_<FLOAT_TYPE>(alphas_cpu_uvecs[dimension]).ptr();
				const size_t alpha_size = alphas_cpu_uvecs[dimension].size();
				//!< If partial function doesn't require global deriv min/max and it returns true,
				// calculate dissipations and step bound from alphas.
				if (alpha_size == loop_size) {
					if (dimension == 0) {
						for (size_t index = 0; index < loop_size; ++index) {
							const FLOAT_TYPE alpha = alpha_ptr[index];
							const FLOAT_TYPE deriv_r_index = deriv_r[index];
							const FLOAT_TYPE deriv_l_index = deriv_l[index];
							diss_ptr[index] = (deriv_r_index - deriv_l_index) * alpha / 2;
						}
					}
					else {
						for (size_t index = 0; index < loop_size; ++index) {
							const FLOAT_TYPE alpha = alpha_ptr[index];
							const FLOAT_TYPE deriv_r_index = deriv_r[index];
							const FLOAT_TYPE deriv_l_index = deriv_l[index];
							diss_ptr[index] += (deriv_r_index - deriv_l_index) * alpha / 2;
						}
					}
					const FLOAT_TYPE max_value = beacls::max_value<FLOAT_TYPE>(alpha_ptr, alpha_ptr + alpha_size);
					step_bound_invs[dimension] = max_value * hji_grid->get_dxInv(dimension);
				}
				else {
					const FLOAT_TYPE alpha = alpha_ptr[0];
					if (dimension == 0) {
						for (size_t index = 0; index < loop_size; ++index) {
							const FLOAT_TYPE deriv_r_index = deriv_r[index];
							const FLOAT_TYPE deriv_l_index = deriv_l[index];
							diss_ptr[index] = (deriv_r_index - deriv_l_index) * alpha / 2;
						}
					}
					else {
						for (size_t index = 0; index < loop_size; ++index) {
							const FLOAT_TYPE deriv_r_index = deriv_r[index];
							const FLOAT_TYPE deriv_l_index = deriv_l[index];
							diss_ptr[index] += (deriv_r_index - deriv_l_index) * alpha / 2;
						}
					}
					const FLOAT_TYPE max_value = alpha;
					step_bound_invs[dimension] = max_value * hji_grid->get_dxInv(dimension);
				}
			}
		}
	}
	if (updateDerivMinMax) {
		for (size_t dimension = 0; dimension < num_of_dimensions; ++dimension) {
			//!< If partial function requires global deriv min/max and it returns false,
			// calculate partial mins/maxs of deriv, and fall back to integrator.
			// integrator should reduce partial mins/maxs to global one, then come back to this function.
			beacls::FloatVec& derivMin = *(beacls::UVec_<FLOAT_TYPE>(derivMins[dimension]).vec());
			beacls::FloatVec& derivMax = *(beacls::UVec_<FLOAT_TYPE>(derivMaxs[dimension]).vec());
			if (derivMin.size() != 1)derivMin.resize(1);
			if (derivMax.size() != 1)derivMax.resize(1);
			if (beacls::is_cuda(deriv_ls[dimension])) {
				CalculateRangeOfGradient_execute_cuda(derivMin[0], derivMax[0], deriv_ls[dimension], deriv_rs[dimension]);
			}
			else
			{
				calculateRangeOfGradient(derivMin[0], derivMax[0], deriv_ls[dimension], deriv_rs[dimension]);
			}
		}
	}

	return result;
}
bool ArtificialDissipationGLF_impl::execute_local_q(
	beacls::UVec& diss,
	beacls::FloatVec& step_bound_invs,
	std::vector<beacls::UVec>& derivMins,
	std::vector<beacls::UVec>& derivMaxs,
	const FLOAT_TYPE t,
	const beacls::UVec& data,
	const std::vector<beacls::UVec>& x_uvecs,
	const std::vector<beacls::UVec >& deriv_ls,
	const std::vector<beacls::UVec >& deriv_rs,
	const SchemeData *schemeData,
	const size_t begin_index,
	const std::set<size_t> &Q, 
	const bool enable_user_defined_dynamics_on_gpu,
	const bool updateDerivMinMax
) {
	const HJI_Grid* hji_grid = schemeData->get_grid();
	if (!hji_grid) return false;

	//	const beacls::FloatVec& alphas = hji_grid->get_alphas(dimension);
	// Now calculate the dissipation.  Since alpha is the effective speed of
	//   the flow, it provides the CFL timestep bound too.
	size_t loop_size = (size_t)deriv_ls[0].size();
	beacls::FloatVec data_vec;
	size_t num_of_dimensions = deriv_ls.size();
	if (step_bound_invs.size() != num_of_dimensions) step_bound_invs.resize(num_of_dimensions);
	bool result = true;
	alphas_cpu_uvecs.resize(num_of_dimensions);
	alphas_cuda_uvecs.resize(num_of_dimensions);
	for (size_t dimension = 0; dimension < num_of_dimensions; ++dimension) {
		//!< Call partial function without global deriv min/max at first.
		bool partialFunc_result = false;
		if (enable_user_defined_dynamics_on_gpu
			&& (beacls::is_cuda(deriv_ls[0]))
			&& schemeData->partialFunc_cuda(alphas_cuda_uvecs[dimension], t, data, x_uvecs, derivMins, derivMaxs, dimension, begin_index, loop_size)) {
			partialFunc_result = true;
		} else {
			if (schemeData->partialFunc(alphas_cpu_uvecs[dimension], t, data, derivMins, derivMaxs, dimension, begin_index, loop_size)) {
				partialFunc_result = true;
				if (beacls::is_cuda(deriv_ls[0]) && (alphas_cpu_uvecs[dimension].size() != 1)) {
					alphas_cpu_uvecs[dimension].convertTo(alphas_cuda_uvecs[dimension], beacls::UVecType_Cuda);
				}
				else {
					alphas_cuda_uvecs[dimension] = alphas_cpu_uvecs[dimension];
				}
			}
		}

		if (!partialFunc_result) {
			result = false;
		}
		else {
			if (deriv_ls[dimension].type() == beacls::UVecType_Cuda) {
				ArtificialDissipationGLF_execute_cuda(
					diss,
					step_bound_invs[dimension],
					alphas_cuda_uvecs[dimension],
					deriv_ls[dimension],
					deriv_rs[dimension],
					hji_grid->get_dxInv(dimension),
					dimension,
					loop_size);
			}
			else
			{
				beacls::UVec tmp_deriv_l;
				if (deriv_ls[dimension].type() == beacls::UVecType_Cuda) deriv_ls[dimension].convertTo(tmp_deriv_l, beacls::UVecType_Vector);
				else tmp_deriv_l = deriv_ls[dimension];
				beacls::UVec tmp_deriv_r;
				if (deriv_rs[dimension].type() == beacls::UVecType_Cuda) deriv_rs[dimension].convertTo(tmp_deriv_r, beacls::UVecType_Vector);
				else tmp_deriv_r = deriv_rs[dimension];

				const FLOAT_TYPE* deriv_l = beacls::UVec_<FLOAT_TYPE>(tmp_deriv_l).ptr();
				const FLOAT_TYPE* deriv_r = beacls::UVec_<FLOAT_TYPE>(tmp_deriv_r).ptr();
				FLOAT_TYPE* diss_ptr = beacls::UVec_<FLOAT_TYPE>(diss).ptr();
				const FLOAT_TYPE* alpha_ptr = beacls::UVec_<FLOAT_TYPE>(alphas_cpu_uvecs[dimension]).ptr();
				const size_t alpha_size = alphas_cpu_uvecs[dimension].size();
				//!< If partial function doesn't require global deriv min/max and it returns true,
				// calculate dissipations and step bound from alphas.
				if (alpha_size == loop_size) {
					if (dimension == 0) {
						for (size_t index = 0; index < loop_size; ++index) {
							const FLOAT_TYPE alpha = alpha_ptr[index];
							const FLOAT_TYPE deriv_r_index = deriv_r[index];
							const FLOAT_TYPE deriv_l_index = deriv_l[index];
							diss_ptr[index] = (deriv_r_index - deriv_l_index) * alpha / 2;
						}
					}
					else {
						for (size_t index = 0; index < loop_size; ++index) {
							const FLOAT_TYPE alpha = alpha_ptr[index];
							const FLOAT_TYPE deriv_r_index = deriv_r[index];
							const FLOAT_TYPE deriv_l_index = deriv_l[index];
							diss_ptr[index] += (deriv_r_index - deriv_l_index) * alpha / 2;
						}
					}
					const FLOAT_TYPE max_value = beacls::max_value<FLOAT_TYPE>(alpha_ptr, alpha_ptr + alpha_size);
					step_bound_invs[dimension] = max_value * hji_grid->get_dxInv(dimension);
				}
				else {
					const FLOAT_TYPE alpha = alpha_ptr[0];
					if (dimension == 0) {
						for (size_t index = 0; index < loop_size; ++index) {
							const FLOAT_TYPE deriv_r_index = deriv_r[index];
							const FLOAT_TYPE deriv_l_index = deriv_l[index];
							diss_ptr[index] = (deriv_r_index - deriv_l_index) * alpha / 2;
						}
					}
					else {
						for (size_t index = 0; index < loop_size; ++index) {
							const FLOAT_TYPE deriv_r_index = deriv_r[index];
							const FLOAT_TYPE deriv_l_index = deriv_l[index];
							diss_ptr[index] += (deriv_r_index - deriv_l_index) * alpha / 2;
						}
					}
					const FLOAT_TYPE max_value = alpha;
					step_bound_invs[dimension] = max_value * hji_grid->get_dxInv(dimension);
				}
			}
		}
	}
	if (updateDerivMinMax) {
		for (size_t dimension = 0; dimension < num_of_dimensions; ++dimension) {
			//!< If partial function requires global deriv min/max and it returns false,
			// calculate partial mins/maxs of deriv, and fall back to integrator.
			// integrator should reduce partial mins/maxs to global one, then come back to this function.
			beacls::FloatVec& derivMin = *(beacls::UVec_<FLOAT_TYPE>(derivMins[dimension]).vec());
			beacls::FloatVec& derivMax = *(beacls::UVec_<FLOAT_TYPE>(derivMaxs[dimension]).vec());
			if (derivMin.size() != 1)derivMin.resize(1);
			if (derivMax.size() != 1)derivMax.resize(1);
			if (beacls::is_cuda(deriv_ls[dimension])) {
				CalculateRangeOfGradient_execute_cuda(derivMin[0], derivMax[0], deriv_ls[dimension], deriv_rs[dimension]);
			}
			else
			{
				calculateRangeOfGradient(derivMin[0], derivMax[0], deriv_ls[dimension], deriv_rs[dimension]);
			}
		}
	}

	return result;
}
bool ArtificialDissipationGLF_impl::calculateRangeOfGradient(
	FLOAT_TYPE& derivMin,
	FLOAT_TYPE& derivMax,
	const beacls::UVec& deriv_l_uvec,
	const beacls::UVec& deriv_r_uvec
) const {
	const FLOAT_TYPE* deriv_l_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_l_uvec).ptr();
	const FLOAT_TYPE* deriv_r_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_r_uvec).ptr();
	const size_t length = deriv_l_uvec.size();
	const auto minMax_l = beacls::minmax_value<FLOAT_TYPE>(deriv_l_ptr, deriv_l_ptr + length);
	const auto minMax_r = beacls::minmax_value<FLOAT_TYPE>(deriv_r_ptr, deriv_r_ptr + length);
	derivMin = min_float_type<FLOAT_TYPE>(minMax_l.first, minMax_r.first);
	derivMax = max_float_type<FLOAT_TYPE>(minMax_l.second, minMax_r.second);
	return true;
}
ArtificialDissipationGLF::ArtificialDissipationGLF() {
	pimpl = new ArtificialDissipationGLF_impl();
}
ArtificialDissipationGLF::~ArtificialDissipationGLF() {
	if (pimpl) delete pimpl;
}
bool ArtificialDissipationGLF::execute(
	beacls::UVec& diss,
	beacls::FloatVec& step_bound_invs,
	std::vector<beacls::UVec>& derivMins,
	std::vector<beacls::UVec>& derivMaxs,
	const FLOAT_TYPE t,
	const beacls::UVec& data,
	const std::vector<beacls::UVec>& x_uvecs,
	const std::vector<beacls::UVec >& deriv_ls,
	const std::vector<beacls::UVec >& deriv_rs,
	const SchemeData *schemeData,
	const size_t begin_index,
	const bool enable_user_defined_dynamics_on_gpu,
	const bool updateDerivMinMax

) {
	if (pimpl) return pimpl->execute(diss, step_bound_invs, derivMins, derivMaxs, t, data, x_uvecs, deriv_ls, deriv_rs,schemeData, begin_index, enable_user_defined_dynamics_on_gpu, updateDerivMinMax);
	else return false;
}
bool ArtificialDissipationGLF::execute_local_q(
	beacls::UVec& diss,
	beacls::FloatVec& step_bound_invs,
	std::vector<beacls::UVec>& derivMins,
	std::vector<beacls::UVec>& derivMaxs,
	const FLOAT_TYPE t,
	const beacls::UVec& data,
	const std::vector<beacls::UVec>& x_uvecs,
	const std::vector<beacls::UVec >& deriv_ls,
	const std::vector<beacls::UVec >& deriv_rs,
	const SchemeData *schemeData,
	const size_t begin_index,
	const std::set<size_t> &Q, 
	const bool enable_user_defined_dynamics_on_gpu,
	const bool updateDerivMinMax

) {
	if (pimpl) return pimpl->execute_local_q(diss, step_bound_invs, derivMins, derivMaxs, t, data, x_uvecs, deriv_ls, deriv_rs,schemeData, begin_index, Q,  enable_user_defined_dynamics_on_gpu, updateDerivMinMax);
	else return false;
}
bool ArtificialDissipationGLF::operator==(const ArtificialDissipationGLF& rhs) const {
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
bool ArtificialDissipationGLF::operator==(const Dissipation& rhs) const {
	if (this == &rhs) return true;
	else if (typeid(*this) != typeid(rhs)) return false;
	else return operator==(dynamic_cast<const ArtificialDissipationGLF&>(rhs));
}
ArtificialDissipationGLF::ArtificialDissipationGLF(const ArtificialDissipationGLF& rhs) :
	pimpl(rhs.pimpl->clone())
{
}

ArtificialDissipationGLF* ArtificialDissipationGLF::clone() const {
	return new ArtificialDissipationGLF(*this);
}
