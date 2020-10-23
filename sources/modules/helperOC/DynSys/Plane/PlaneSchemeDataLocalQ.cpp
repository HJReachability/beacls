#include <algorithm>
#include <functional>
#include <iostream>
#include <numeric>
#include <iomanip>
#include <typeinfo>

#include "helperOC/DynSys/Plane/PlaneSchemeDataLocalQ.hpp"
#include <helperOC/DynSys/DynSys/DynSysSchemeData.hpp>
#include <helperOC/DynSys/DynSys/DynSys.hpp>
#include <helperOC/ComputeGradients.hpp>
#include <levelset/Grids/HJI_Grid.hpp>
#include <macro.hpp>

using namespace helperOC;
namespace helperOC {
	class DynSysSchemeData_Workspace {
	public:
		std::vector<beacls::FloatVec > us;
		std::vector<beacls::FloatVec > ds;
		std::vector<beacls::FloatVec > dxs;
		std::vector<beacls::FloatVec > TIdxs;
		std::vector<std::vector<beacls::FloatVec> > uUss;
		std::vector<std::vector<beacls::FloatVec> > uLss;
		std::vector<std::vector<beacls::FloatVec> > dUss;
		std::vector<std::vector<beacls::FloatVec> > dLss;
		std::vector<std::vector<beacls::FloatVec> > dxUUs;
		std::vector<std::vector<beacls::FloatVec> > dxULs;
		std::vector<std::vector<beacls::FloatVec> > dxLLs;
		std::vector<std::vector<beacls::FloatVec> > dxLUs;
#if defined(USER_DEFINED_GPU_DYNSYS_FUNC)
		std::vector<beacls::UVec > u_uvecs;
		std::vector<beacls::UVec > d_uvecs;
		std::vector<beacls::UVec > dx_uvecs;
		std::vector<beacls::UVec > TIdx_uvecs;
		std::vector<std::vector<beacls::UVec > > uU_uvecss;
		std::vector<std::vector<beacls::UVec > > uL_uvecss;
		std::vector<std::vector<beacls::UVec > > dU_uvecss;
		std::vector<std::vector<beacls::UVec > > dL_uvecss;
		std::vector<std::vector<beacls::UVec > > dxUU_uvecss;
		std::vector<std::vector<beacls::UVec > > dxUL_uvecss;
		std::vector<std::vector<beacls::UVec > > dxLL_uvecss;
		std::vector<std::vector<beacls::UVec > > dxLU_uvecss;
#endif /* defined(USER_DEFINED_GPU_DYNSYS_FUNC) */
	};
};

bool PlaneSchemeDataLocalQ::operator==(const PlaneSchemeDataLocalQ& rhs) const {
	if (this == &rhs) return true;
	else if (!DynSysSchemeData::operator==(rhs)) return false;
	else if (vMin != rhs.vMin) return false;
	else if (vMax != rhs.vMax) return false;
	else if (dMax_x != rhs.dMax_x) return false;
	else if (dMax_y != rhs.dMax_y) return false;
	else if (wMax != rhs.wMax) return false;
	else return true;
}
bool PlaneSchemeDataLocalQ::operator==(const SchemeData& rhs) const {
	if (this == &rhs) return true;
	else if (typeid(*this) != typeid(rhs)) return false;
	else return operator==(dynamic_cast<const PlaneSchemeDataLocalQ&>(rhs));
}
static
bool getOptCtrl(
	std::vector<beacls::FloatVec >& us,
	const DynSys* dynSys,
	const std::vector<beacls::FloatVec >& uIns,
	const FLOAT_TYPE t,
	const std::vector<beacls::FloatVec::const_iterator >& xs_ites,
	const std::vector<beacls::UVec>& deriv_uvecs,
	const beacls::IntegerVec& x_sizes,
	const helperOC::DynSys_UMode_Type uMode
) {
	const size_t nu = dynSys->get_nu();
	us.resize(nu);

	if (!uIns.empty()) {
		if (us.size() < uIns.size()) us.resize(uIns.size());
		std::copy(uIns.cbegin(), uIns.cend(), us.begin());
		if (us.size() > uIns.size()) std::fill(us.begin() + uIns.size(), us.end(), uIns[0]);
	}
	else {
		std::vector<const FLOAT_TYPE*> custom_deriv_ptrs(deriv_uvecs.size());
		beacls::IntegerVec custom_deriv_sizes(deriv_uvecs.size());
		std::transform(deriv_uvecs.cbegin(), deriv_uvecs.cend(), custom_deriv_ptrs.begin(), [](const auto& rhs) { return beacls::UVec_<FLOAT_TYPE>(rhs).ptr(); });
		std::transform(deriv_uvecs.cbegin(), deriv_uvecs.cend(), custom_deriv_sizes.begin(), [](const auto& rhs) { return rhs.size(); });
		if (!dynSys->optCtrl(us, t, xs_ites, custom_deriv_ptrs, x_sizes, custom_deriv_sizes, uMode)) return false;
	}
	return true;
}
static
bool getOptDstb(
	std::vector<beacls::FloatVec>& ds,
	const DynSys* dynSys,
	const std::vector<beacls::FloatVec >& dIns,
	const FLOAT_TYPE t,
	const std::vector<beacls::FloatVec::const_iterator>& xs_ites,
	const std::vector<beacls::UVec>& deriv_uvecs,
	const beacls::IntegerVec& x_sizes,
	const helperOC::DynSys_DMode_Type dMode
) {
	const size_t nd = dynSys->get_nd();
	ds.resize(nd);
	
	if (!dIns.empty()) {
		if (ds.size() < dIns.size()) ds.resize(dIns.size());
		std::copy(dIns.cbegin(), dIns.cend(), ds.begin());
		if (ds.size() > dIns.size()) std::fill(ds.begin() + dIns.size(), ds.end(), dIns[0]);
	}
	else {
		std::vector<const FLOAT_TYPE*> custom_deriv_ptrs(deriv_uvecs.size());
		beacls::IntegerVec custom_deriv_sizes(deriv_uvecs.size());
		std::transform(deriv_uvecs.cbegin(), deriv_uvecs.cend(), custom_deriv_ptrs.begin(), [](const auto& rhs) { return beacls::UVec_<FLOAT_TYPE>(rhs).ptr(); });
		std::transform(deriv_uvecs.cbegin(), deriv_uvecs.cend(), custom_deriv_sizes.begin(), [](const auto& rhs) { return rhs.size(); });
		if (!dynSys->optDstb(ds, t, xs_ites, custom_deriv_ptrs, x_sizes, custom_deriv_sizes, dMode)) return false;
	}
	return true;
}

static
bool getDynamics(std::vector<beacls::FloatVec >& dxs,
		const DynSys* dynSys,
		const FLOAT_TYPE t,
		const std::vector<beacls::FloatVec::const_iterator >& xs_ites,
		std::vector<beacls::FloatVec >& us,
		std::vector<beacls::FloatVec >& ds,
		const beacls::IntegerVec& x_sizes,
		const size_t dim = std::numeric_limits<size_t>::max() ) {
	const size_t nx = dynSys->get_nx();
	dxs.resize(nx);
	if (!dynSys->dynamics(dxs, t, xs_ites, us, ds, x_sizes, dim)) 
		return false;
	return true;
}
bool PlaneSchemeDataLocalQ::initializeLocalQ(
	const beacls::FloatVec &vRange, 
	const beacls::FloatVec &dMax,
	const FLOAT_TYPE wMaximum
) {
	dMax_x = dMax[0];
	dMax_y = dMax[1];
	vMin = vRange[0];
	vMax = vRange[1];
	wMax = wMaximum; 
}

bool PlaneSchemeDataLocalQ::hamFunc(
	beacls::UVec& hamValue_uvec,
	const FLOAT_TYPE t,
	const beacls::UVec&,
	const std::vector<beacls::UVec>& derivs,
	const size_t begin_index,
	const size_t length 
) const {
	const levelset::HJI_Grid *hji_grid = get_grid();
	
	// Custom derivative for MIE
	const std::vector<beacls::UVec>& custom_derivs = (!MIEderivs.empty()) ? 
	    MIEderivs : derivs;
	const std::vector<beacls::FloatVec >& xs = hji_grid->get_xss();
	std::vector<beacls::FloatVec::const_iterator > xs_ites(xs.size());
	std::transform(xs.cbegin(), xs.cend(), xs_ites.begin(), 
		  ([begin_index](const auto& rhs) { return rhs.cbegin() + begin_index; }));
	beacls::IntegerVec x_sizes(xs.size());
	// printf("length = %zd\n", length);
	std::fill(x_sizes.begin(), x_sizes.end(), length);
	// printf("x_size = (%zd, %zd, %zd, %zd)\n", x_sizes[0], x_sizes[1], x_sizes[2], x_sizes[3]);

	std::vector<beacls::FloatVec >& us = ws->us;
	if (!getOptCtrl(us, dynSys, uIns, t, xs_ites, custom_derivs, x_sizes, uMode)) 
		return false;

	std::vector<beacls::FloatVec >& ds = ws->ds;
	if (!getOptDstb(ds, dynSys, dIns, t, xs_ites, custom_derivs, x_sizes, dMode)) 
		return false;

	hamValue_uvec.resize(length);
	FLOAT_TYPE TIderiv = 0;
	std::vector<beacls::UVec> modified_derivs(custom_derivs.size());
	if (side.valid()) {
		if (side.lower) {
//			if (dissComp) {
//				for (size_t i = 0; i < length; ++i) {
//					hamValue[i] -= dc;
//				}
//			}
			TIderiv = -1;
		}
		else if (side.upper) {
//			if (dissComp) {
//				for (size_t i = 0; i < length; ++i) {
//					hamValue[i] += dc;
//				}
//			}
//			modified_derivs[0] = new FLOAT_TYPE[length];
//			for (size_t i = 0; i < length; ++i) {
//				modified_derivs[0][i] = -derivMaxs[0][i];
//			}
//			if (trueMIEDeriv) {
//				for (size_t i = 0; i < length; ++i) {
//					modified_derivs[1][i] = -derivMaxs[1][i];
//				}
//			}
			TIderiv = 1;
		}
		else {
			std::cerr << "Side of an MIE function must be upper or lower!" << std::endl;
		}
//		if (!trueMIEDeriv) {
//			//!< T.B.D.
//			std::vector<beacls::FloatVec > derivC, derivL, derivR;
//			computeGradients->operator()(derivC, derivL, derivR,hji_grid,data,length);
//		}
	}
	//!< Plug optimal control into dynamics to compute Hamiltonian

	std::vector<beacls::FloatVec >& dxs = ws->dxs;

	if (!getDynamics(dxs, dynSys, t, xs_ites, us, ds, x_sizes)) 
		return false;

	if (hamValue_uvec.type() != beacls::UVecType_Vector)
	  hamValue_uvec = beacls::UVec(beacls::type_to_depth<FLOAT_TYPE>(), 
	  	  beacls::UVecType_Vector, length);
	else
		hamValue_uvec.resize(length);

	beacls::FloatVec& hamValue = *(beacls::UVec_<FLOAT_TYPE>(hamValue_uvec).vec());
	for (size_t dimension = 0; dimension < dxs.size(); ++dimension) {
		const beacls::UVec& custom_deriv = 
		    (!modified_derivs[dimension].empty()) ? 
		    modified_derivs[dimension] : custom_derivs[dimension];

		const FLOAT_TYPE* custom_deriv_ptr = 
		    beacls::UVec_<FLOAT_TYPE>(custom_deriv).ptr();

		const beacls::FloatVec& dx = dxs[dimension];

		if (dimension == 0) {
			for (size_t index = 0; index < length; ++index) {
				hamValue[index] = custom_deriv_ptr[index] * dx[index];
			}
		}
		else {
			for (size_t index = 0; index < length; ++index) {
				hamValue[index] += custom_deriv_ptr[index] * dx[index];
			}
		}
	}

	if (!dynSys->get_TIdims().empty()) {
		const size_t nTI = dynSys->get_TIdims().size();
		std::vector<beacls::FloatVec >& TIdxs = ws->TIdxs;
		TIdxs.resize(nTI);
		dynSys->TIdyn(TIdxs, t, xs_ites, us, ds, x_sizes);
		for (size_t dimension = 0; dimension < nTI; ++dimension) {
			const beacls::FloatVec& TIdx = TIdxs[dimension];
			for (size_t index = 0; index < TIdx.size(); ++index) {
				hamValue[index] += TIderiv * TIdx[index];
			}
		}
	}

	//!< Negate hamValue if backward reachable set
	if ((tMode == helperOC::DynSys_TMode_Backward) != (side.valid() && side.upper)) {
		for (size_t index = 0; index < length; ++index) {
			hamValue[index] = -hamValue[index];
		}
	}
	return true;
}

bool PlaneSchemeDataLocalQ::partialFunc(
	beacls::UVec& alphas_uvec,
	const FLOAT_TYPE t,
	const beacls::UVec& data,
	const std::vector<beacls::UVec>& derivMins,
	const std::vector<beacls::UVec>& derivMaxs,
	const size_t dim,
	const size_t begin_index,
	const size_t length
	) const {
	//!< Using partial function from dynamical system 
	if (const helperOC::PartialFunction_cuda* partialFunction = dynSys->get_partialFunction()) {
		return (*partialFunction)(alphas_uvec, this, t, data, derivMins, derivMaxs, dim, begin_index, length);
	}

	// Copy state matrices in case we're doing MIE
	// Dimension information(in case we're doing MIE)
//	if (!MIEdims.empty()) {
//	TIdims = schemeData.TIdims;
//		dims = MIEdims;
//	}
	const levelset::HJI_Grid *hji_grid = get_grid();
	const size_t num_of_dimensions = hji_grid->get_num_of_dimensions();
	const std::vector<beacls::FloatVec >& xs = hji_grid->get_xss();
	std::vector<beacls::FloatVec::const_iterator > xs_ites(xs.size());
	std::transform(xs.cbegin(), xs.cend(), xs_ites.begin(), ([begin_index](const auto& rhs) { return rhs.cbegin() + begin_index; }));
	beacls::IntegerVec x_sizes(xs.size());
	std::fill(x_sizes.begin(), x_sizes.end(), length);

	ws->uUss.resize(num_of_dimensions);
	ws->uLss.resize(num_of_dimensions);
	ws->dUss.resize(num_of_dimensions);
	ws->dLss.resize(num_of_dimensions);
	ws->dxUUs.resize(num_of_dimensions);
	ws->dxULs.resize(num_of_dimensions);
	ws->dxLUs.resize(num_of_dimensions);
	ws->dxLLs.resize(num_of_dimensions);

	std::vector<beacls::FloatVec >& uUs = ws->uUss[dim];
	std::vector<beacls::FloatVec >& uLs = ws->uLss[dim];

	//!< Compute control
	if (!getOptCtrl(uUs, dynSys, uIns, t, xs_ites, derivMaxs, x_sizes, uMode)) return false;
	if (!getOptCtrl(uLs, dynSys, uIns, t, xs_ites, derivMins, x_sizes, uMode)) return false;

	std::vector<beacls::FloatVec >& dUs = ws->dUss[dim];
	std::vector<beacls::FloatVec >& dLs = ws->dLss[dim];
	//!< Compute disturbance
	if (!getOptDstb(dUs, dynSys, dIns, t, xs_ites, derivMaxs, x_sizes, dMode)) return false;
	if (!getOptDstb(dLs, dynSys, dIns, t, xs_ites, derivMins, x_sizes, dMode)) return false;

	//!< Compute alpha
	std::vector<beacls::FloatVec >& dxUU = ws->dxUUs[dim];
	std::vector<beacls::FloatVec >& dxUL = ws->dxULs[dim];
	std::vector<beacls::FloatVec >& dxLL = ws->dxLLs[dim];
	std::vector<beacls::FloatVec >& dxLU = ws->dxLUs[dim];

	if (!getDynamics(dxUU, dynSys, t, xs_ites, uUs, dUs, x_sizes, dim)) return false;
	if (!getDynamics(dxUL, dynSys, t, xs_ites, uUs, dLs, x_sizes, dim)) return false;
	if (!getDynamics(dxLU, dynSys, t, xs_ites, uLs, dUs, x_sizes, dim)) return false;
	if (!getDynamics(dxLL, dynSys, t, xs_ites, uLs, dLs, x_sizes, dim)) return false;

	const size_t alpha_size = dxUU[dim].size();
	if (alphas_uvec.type() != beacls::UVecType_Vector) alphas_uvec = beacls::UVec(beacls::type_to_depth<FLOAT_TYPE>(), beacls::UVecType_Vector, alpha_size);
	else alphas_uvec.resize(alpha_size);
	beacls::FloatVec& alphas = *(beacls::UVec_<FLOAT_TYPE>(alphas_uvec).vec());
	const beacls::FloatVec& dxUU_dim = dxUU[dim];
	const beacls::FloatVec& dxUL_dim = dxUL[dim];
	const beacls::FloatVec& dxLL_dim = dxLL[dim];
	const beacls::FloatVec& dxLU_dim = dxLU[dim];
	for (size_t i = 0; i < alpha_size; ++i) {
		FLOAT_TYPE max0 = HjiMax(HjiFabs(dxUU_dim[i]), HjiFabs(dxUL_dim[i]));
		FLOAT_TYPE max1 = HjiMax(HjiFabs(dxLU_dim[i]), HjiFabs(dxLL_dim[i]));
		alphas[i] = HjiMax(max0, max1);
	}
	return true;
}

bool PlaneSchemeDataLocalQ::hamFuncLocalQ(
	beacls::UVec& hamValue_uvec,
	const FLOAT_TYPE t,
	const beacls::UVec& data,
	const std::vector<beacls::UVec>& derivs,
	const size_t begin_index,
	const size_t length, 
	const std::set<size_t> &Q
) const {
	const levelset::HJI_Grid *hji_grid = get_grid();

	const beacls::FloatVec &thetas = hji_grid->get_xs(2);
	beacls::reallocateAsSrc(hamValue_uvec, derivs[0]);
	FLOAT_TYPE* hamValue = beacls::UVec_<FLOAT_TYPE>(hamValue_uvec).ptr();
	const FLOAT_TYPE* deriv0 = beacls::UVec_<FLOAT_TYPE>(derivs[0]).ptr();
	const FLOAT_TYPE* deriv1 = beacls::UVec_<FLOAT_TYPE>(derivs[1]).ptr();
	const FLOAT_TYPE* deriv2 = beacls::UVec_<FLOAT_TYPE>(derivs[2]).ptr();

	for (size_t i = 0; i < length; ++i) {
		int dst_index = begin_index + i; 
		if (Q.find(dst_index) != Q.end())
		{
			FLOAT_TYPE deriv0_i = deriv0[i];
			FLOAT_TYPE deriv1_i = deriv1[i];
			FLOAT_TYPE theta = thetas[dst_index];
			FLOAT_TYPE speedCtrl = deriv0_i * std::cos(theta) + deriv1_i * std::sin(theta);
			if (speedCtrl >= 0)
			{
				hamValue[i] = speedCtrl * vMin;
			}
			else 
			{
				hamValue[i] = speedCtrl * vMax;
			}
			FLOAT_TYPE wTerm = wMax * std::abs(deriv2[i]);
			hamValue[i] = hamValue[i] + wTerm;
			FLOAT_TYPE dTerm = -dMax_x * std::abs(deriv0_i) -dMax_y * abs(deriv1_i);
			hamValue[i] = hamValue[i] + dTerm;
			if ((tMode == helperOC::DynSys_TMode_Backward)) 
			{
				hamValue[i] = -hamValue[i];
			}
		}
		else 
		{
			hamValue[i] = 0;
		}
	}
		
	return true;
}
bool PlaneSchemeDataLocalQ::partialFuncLocalQ(
	beacls::UVec& alphas_uvec,
	const FLOAT_TYPE t,
	const beacls::UVec& data,
	const std::vector<beacls::UVec>& derivMins,
	const std::vector<beacls::UVec>& derivMaxs,
	const size_t dim,
	const size_t begin_index,
	const size_t length, 
	const std::set<size_t> &Q
) const 
{
	const levelset::HJI_Grid *hji_grid = get_grid();
	const size_t num_of_dimensions = hji_grid->get_num_of_dimensions();
	const std::vector<beacls::FloatVec >& xs = hji_grid->get_xss();
	std::vector<beacls::FloatVec::const_iterator > xs_ites(xs.size());
	std::transform(xs.cbegin(), xs.cend(), xs_ites.begin(), ([begin_index](const auto& rhs) { return rhs.cbegin() + begin_index; }));
	beacls::IntegerVec x_sizes(xs.size());
	std::fill(x_sizes.begin(), x_sizes.end(), length);

	ws->uUss.resize(num_of_dimensions);
	ws->uLss.resize(num_of_dimensions);
	ws->dUss.resize(num_of_dimensions);
	ws->dLss.resize(num_of_dimensions);
	ws->dxUUs.resize(num_of_dimensions);
	ws->dxULs.resize(num_of_dimensions);
	ws->dxLUs.resize(num_of_dimensions);
	ws->dxLLs.resize(num_of_dimensions);

	std::vector<beacls::FloatVec >& uUs = ws->uUss[dim];
	std::vector<beacls::FloatVec >& uLs = ws->uLss[dim];

	//!< Compute control
	if (!getOptCtrl(uUs, dynSys, uIns, t, xs_ites, derivMaxs, x_sizes, uMode)) return false;
	if (!getOptCtrl(uLs, dynSys, uIns, t, xs_ites, derivMins, x_sizes, uMode)) return false;

	std::vector<beacls::FloatVec >& dUs = ws->dUss[dim];
	std::vector<beacls::FloatVec >& dLs = ws->dLss[dim];
	//!< Compute disturbance
	if (!getOptDstb(dUs, dynSys, dIns, t, xs_ites, derivMaxs, x_sizes, dMode)) return false;
	if (!getOptDstb(dLs, dynSys, dIns, t, xs_ites, derivMins, x_sizes, dMode)) return false;

	//!< Compute alpha
	std::vector<beacls::FloatVec >& dxUU = ws->dxUUs[dim];
	std::vector<beacls::FloatVec >& dxUL = ws->dxULs[dim];
	std::vector<beacls::FloatVec >& dxLL = ws->dxLLs[dim];
	std::vector<beacls::FloatVec >& dxLU = ws->dxLUs[dim];

	if (!getDynamics(dxUU, dynSys, t, xs_ites, uUs, dUs, x_sizes, dim)) return false;
	if (!getDynamics(dxUL, dynSys, t, xs_ites, uUs, dLs, x_sizes, dim)) return false;
	if (!getDynamics(dxLU, dynSys, t, xs_ites, uLs, dUs, x_sizes, dim)) return false;
	if (!getDynamics(dxLL, dynSys, t, xs_ites, uLs, dLs, x_sizes, dim)) return false;

	const size_t alpha_size = dxUU[dim].size();
	if (alphas_uvec.type() != beacls::UVecType_Vector) alphas_uvec = beacls::UVec(beacls::type_to_depth<FLOAT_TYPE>(), beacls::UVecType_Vector, alpha_size);
	else alphas_uvec.resize(alpha_size);
	beacls::FloatVec& alphas = *(beacls::UVec_<FLOAT_TYPE>(alphas_uvec).vec());
	const beacls::FloatVec& dxUU_dim = dxUU[dim];
	const beacls::FloatVec& dxUL_dim = dxUL[dim];
	const beacls::FloatVec& dxLL_dim = dxLL[dim];
	const beacls::FloatVec& dxLU_dim = dxLU[dim];
	for (size_t i = 0; i < alpha_size; ++i) {
		FLOAT_TYPE max0 = HjiMax(HjiFabs(dxUU_dim[i]), HjiFabs(dxUL_dim[i]));
		FLOAT_TYPE max1 = HjiMax(HjiFabs(dxLU_dim[i]), HjiFabs(dxLL_dim[i]));
		alphas[i] = HjiMax(max0, max1);
	}
	return true;
}
