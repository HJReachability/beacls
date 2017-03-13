#include <algorithm>
#include <functional>
#include <iostream>
#include <numeric>
#include <iomanip>
#include <typeinfo>

#include <helperOC/DynSys/DynSys/DynSysSchemeData.hpp>
#include <helperOC/DynSys/DynSys/DynSys.hpp>
#include <helperOC/ComputeGradients.hpp>
#include <levelset/Grids/HJI_Grid.hpp>
#include <macro.hpp>

#include "DynSysSchemeData_cuda.hpp"

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

DynSysSchemeData::DynSysSchemeData(
	) : SchemeData(),
	ws(new DynSysSchemeData_Workspace()),
	dynSys(NULL),
	computeGradients(NULL),
	uMode(DynSys_UMode_Min),
	dMode(DynSys_DMode_Min),
	tMode(DynSys_TMode_Backward),
	MIEderivs(std::vector<beacls::UVec>()),
	MIEdims(beacls::IntegerVec()),
	TIdims(beacls::IntegerVec()),
	uIns(std::vector<beacls::FloatVec>()),
	dIns(std::vector<beacls::FloatVec>()),

	accuracy(helperOC::ApproximationAccuracy_Invalid),
	side(DynSysSchemeDataSide()),
	dc(0.),
	dissComp(false),
	trueMIEDeriv(false)
	{
}
DynSysSchemeData::DynSysSchemeData(const DynSysSchemeData& rhs) :
	SchemeData(rhs),
	ws(new DynSysSchemeData_Workspace()),
	dynSys(rhs.dynSys),
	computeGradients(rhs.computeGradients),
	uMode(rhs.uMode),
	dMode(rhs.dMode),
	tMode(rhs.tMode),
	MIEderivs(rhs.MIEderivs),
	MIEdims(rhs.MIEdims),
	TIdims(rhs.TIdims),
	uIns(rhs.uIns),
	dIns(rhs.dIns),
	accuracy(rhs.accuracy),
	side(rhs.side),
	dc(rhs.dc),
	dissComp(rhs.dissComp),
	trueMIEDeriv(rhs.trueMIEDeriv)
{}
DynSysSchemeData::~DynSysSchemeData() {
	if (ws) delete ws;
}
bool DynSysSchemeData::operator==(const DynSysSchemeData& rhs) const {
	if (this == &rhs) return true;
	else if (!SchemeData::operator==(rhs)) return false;
	else if (uMode != rhs.uMode) return false;
	else if (dMode != rhs.dMode) return false;
	else if (tMode != rhs.tMode) return false;
	else if (accuracy != rhs.accuracy) return false;
	else if (!side.operator==(rhs.side)) return false;
	else if (dc != rhs.dc) return false;
	else if (dissComp != rhs.dissComp) return false;
	else if (trueMIEDeriv != rhs.trueMIEDeriv) return false;

	else if ((MIEdims.size() != rhs.MIEdims.size()) || !std::equal(MIEdims.cbegin(), MIEdims.cend(), rhs.MIEdims.cbegin())) return false;
	else if ((TIdims.size() != rhs.TIdims.size()) || !std::equal(TIdims.cbegin(), TIdims.cend(), rhs.TIdims.cbegin())) return false;

	else if ((uIns.size() != rhs.uIns.size()) || !std::equal(uIns.cbegin(), uIns.cend(), rhs.uIns.cbegin(), [](const auto& lhs, const auto& rhs) { 
		return  ((lhs.size() == rhs.size()) && std::equal(lhs.cbegin(), lhs.cend(), rhs.cbegin()));
	})) return false;
	else if ((dIns.size() != rhs.dIns.size()) || !std::equal(dIns.cbegin(), dIns.cend(), rhs.dIns.cbegin(), [](const auto& lhs, const auto& rhs) {
		return  ((lhs.size() == rhs.size()) && std::equal(lhs.cbegin(), lhs.cend(), rhs.cbegin()));
	})) return false;

	else if ((dynSys != rhs.dynSys) && (!dynSys || !rhs.dynSys ||!dynSys->operator==(*rhs.dynSys)))return false;
	else if ((computeGradients != rhs.computeGradients) && (!computeGradients || !rhs.computeGradients || !computeGradients->operator==(*rhs.computeGradients))) return false;
	else if ((MIEderivs.size() != rhs.MIEderivs.size()) || !std::equal(MIEderivs.cbegin(), MIEderivs.cend(), rhs.MIEderivs.cbegin(), [](const auto& lhs, const auto& rhs) {
		if ((lhs.type() != beacls::UVecType_Vector) || (rhs.type() != beacls::UVecType_Vector)) return false;
		const beacls::FloatVec& lhs_vec = *beacls::UVec_<FLOAT_TYPE>(lhs).vec();
		const beacls::FloatVec& rhs_vec = *beacls::UVec_<FLOAT_TYPE>(rhs).vec();
		return std::equal(lhs_vec.cbegin(), lhs_vec.cend(), rhs_vec.cbegin());
	})) return false;
	return true;
}
bool DynSysSchemeData::operator==(const SchemeData& rhs) const {
	if (this == &rhs) return true;
	else if (typeid(*this) != typeid(rhs)) return false;
	else return operator==(dynamic_cast<const DynSysSchemeData&>(rhs));
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
	const DynSys_UMode_Type uMode
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
	std::vector<beacls::FloatVec >& ds,
	const DynSys* dynSys,
	const std::vector<beacls::FloatVec >& dIns,
	const FLOAT_TYPE t,
	const std::vector<beacls::FloatVec::const_iterator >& xs_ites,
	const std::vector<beacls::UVec>& deriv_uvecs,
	const beacls::IntegerVec& x_sizes,
	const DynSys_DMode_Type dMode
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
bool getDynamics(
	std::vector<beacls::FloatVec >& dxs,
	const DynSys* dynSys,
	const FLOAT_TYPE t,
	const std::vector<beacls::FloatVec::const_iterator >& xs_ites,
	std::vector<beacls::FloatVec >& us,
	std::vector<beacls::FloatVec >& ds,
	const beacls::IntegerVec& x_sizes,
	const size_t dim = std::numeric_limits<size_t>::max()
) {
	const size_t nx = dynSys->get_nx();
	dxs.resize(nx);
	if (!dynSys->dynamics(dxs, t, xs_ites, us, ds, x_sizes, dim)) return false;
	return true;
}
bool DynSysSchemeData::hamFunc(
	beacls::UVec& hamValue_uvec,
	const FLOAT_TYPE t,
	const beacls::UVec&,
	const std::vector<beacls::UVec>& derivs,
	const size_t begin_index,
	const size_t length
	) const {
	const HJI_Grid *hji_grid = get_grid();

	// Custom derivative for MIE
	const std::vector<beacls::UVec>& custom_derivs = (!MIEderivs.empty()) ? MIEderivs : derivs;
	const std::vector<beacls::FloatVec >& xs = hji_grid->get_xss();
	std::vector<beacls::FloatVec::const_iterator > xs_ites(xs.size());
	std::transform(xs.cbegin(), xs.cend(), xs_ites.begin(), ([begin_index](const auto& rhs) { return rhs.cbegin() + begin_index; }));
	beacls::IntegerVec x_sizes(xs.size());
	std::fill(x_sizes.begin(), x_sizes.end(), length);

	std::vector<beacls::FloatVec >& us = ws->us;
	if (!getOptCtrl(us, dynSys, uIns, t, xs_ites, custom_derivs, x_sizes, uMode)) return false;

	std::vector<beacls::FloatVec >& ds = ws->ds;
	if (!getOptDstb(ds, dynSys, dIns, t, xs_ites, custom_derivs, x_sizes, dMode)) return false;
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
	if (!getDynamics(dxs, dynSys, t, xs_ites, us, ds, x_sizes)) return false;

	if (hamValue_uvec.type() != beacls::UVecType_Vector) hamValue_uvec = beacls::UVec(beacls::type_to_depth<FLOAT_TYPE>(), beacls::UVecType_Vector, length);
	else hamValue_uvec.resize(length);
	beacls::FloatVec& hamValue = *(beacls::UVec_<FLOAT_TYPE>(hamValue_uvec).vec());
	for (size_t dimension = 0; dimension < dxs.size(); ++dimension) {
		const beacls::UVec& custom_deriv = (!modified_derivs[dimension].empty()) ? modified_derivs[dimension] : custom_derivs[dimension];
		const FLOAT_TYPE* custom_deriv_ptr = beacls::UVec_<FLOAT_TYPE>(custom_deriv).ptr();
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
	if ((tMode == DynSys_TMode_Backward) != (side.valid() && side.upper)) {
		for (size_t index = 0; index < length; ++index) {
			hamValue[index] = -hamValue[index];
		}
	}
	return true;
}

bool DynSysSchemeData::partialFunc(
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
	const HJI_Grid *hji_grid = get_grid();
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
#if defined(USER_DEFINED_GPU_DYNSYS_FUNC)
static
bool getOptCtrl_cuda(
	std::vector<beacls::UVec >& u_uvecs,
	const DynSys* dynSys,
	const std::vector<beacls::FloatVec >& uIns,
	const FLOAT_TYPE t,
	const std::vector<beacls::UVec>& x_uvecs,
	const std::vector<beacls::UVec>& derivs,
	const DynSys_UMode_Type uMode
) {
	const size_t nu = dynSys->get_nu();
	u_uvecs.resize(nu);
	if (!uIns.empty()) {
		if (u_uvecs.size() < uIns.size()) u_uvecs.resize(uIns.size());
		std::transform(uIns.cbegin(), uIns.cend(), u_uvecs.begin(), [](const auto& rhs) {
			if (rhs.size() == 1) return beacls::UVec(rhs, beacls::UVecType_Vector, false);
			else return beacls::UVec(rhs, beacls::UVecType_Cuda, true);
		});
		if (u_uvecs.size() > uIns.size()) {
			if (uIns[0].size() == 1) std::fill(u_uvecs.begin() + uIns.size(), u_uvecs.end(), beacls::UVec(uIns[0], beacls::UVecType_Vector, false));
			else std::fill(u_uvecs.begin() + uIns.size(), u_uvecs.end(), beacls::UVec(uIns[0], beacls::UVecType_Cuda, true));
		}
	}
	else {
		if (!dynSys->optCtrl_cuda(u_uvecs, t, x_uvecs, derivs, uMode)) return false;
	}
	return true;
}
static
bool getOptDstb_cuda(
	std::vector<beacls::UVec >& d_uvecs,
	const DynSys* dynSys,
	const std::vector<beacls::FloatVec >& dIns,
	const FLOAT_TYPE t,
	const std::vector<beacls::UVec>& x_uvecs,
	const std::vector<beacls::UVec>& derivs,
	const DynSys_DMode_Type dMode
) {
	const size_t nd = dynSys->get_nd();
	d_uvecs.resize(nd);
	if (!dIns.empty()) {
		if (d_uvecs.size() < dIns.size()) d_uvecs.resize(dIns.size());
		std::transform(dIns.cbegin(), dIns.cend(), d_uvecs.begin(), [](const auto& rhs) {
			if (rhs.size() == 1) return beacls::UVec(rhs, beacls::UVecType_Vector, false);
			else return beacls::UVec(rhs, beacls::UVecType_Cuda, true);
		});
		if (d_uvecs.size() > dIns.size()) {
			if (dIns[0].size() == 1) std::fill(d_uvecs.begin() + dIns.size(), d_uvecs.end(), beacls::UVec(dIns[0], beacls::UVecType_Vector, false));
			else std::fill(d_uvecs.begin() + dIns.size(), d_uvecs.end(), beacls::UVec(dIns[0], beacls::UVecType_Cuda, true));
		}
	}
	else {
		if (!dynSys->optDstb_cuda(d_uvecs, t, x_uvecs, derivs, dMode)) return false;
	}
	return true;
}

static
bool getDynamics_cuda(
	std::vector<beacls::UVec >& dx_uvecs,
	const DynSys* dynSys,
	const FLOAT_TYPE t,
	const std::vector<beacls::UVec>& x_uvecs,
	const std::vector<beacls::UVec>& u_uvecs,
	const std::vector<beacls::UVec>& d_uvecs,
	const size_t dim = std::numeric_limits<size_t>::max()
) {
	const size_t nx = dynSys->get_nx();
	dx_uvecs.resize(nx);
	if (!dynSys->dynamics_cuda(dx_uvecs, t, x_uvecs, u_uvecs, d_uvecs, dim)) return false;
	return true;
}
static
bool getOptCtrl_cuda(
	std::vector<beacls::UVec >& uL_uvecs,
	std::vector<beacls::UVec >& uU_uvecs,
	const DynSys* dynSys,
	const std::vector<beacls::FloatVec >& uIns,
	const FLOAT_TYPE t,
	const std::vector<beacls::UVec>& x_uvecs,
	const std::vector<beacls::UVec>& derivMins,
	const std::vector<beacls::UVec>& derivMaxs,
	const DynSys_UMode_Type uMode
) {
	const size_t nu = dynSys->get_nu();
	uU_uvecs.resize(nu);
	uL_uvecs.resize(nu);
	if (!uIns.empty()) {
		if (uU_uvecs.size() < uIns.size()) uU_uvecs.resize(uIns.size());
		std::transform(uIns.cbegin(), uIns.cend(), uU_uvecs.begin(), [](const auto& rhs) {
			if (rhs.size() == 1) return beacls::UVec(rhs, beacls::UVecType_Vector, false);
			else return beacls::UVec(rhs, beacls::UVecType_Cuda, true);
		});
		if (uU_uvecs.size() > uIns.size()) {
			if (uIns[0].size() == 1) std::fill(uU_uvecs.begin() + uIns.size(), uU_uvecs.end(), beacls::UVec(uIns[0], beacls::UVecType_Vector, false));
			else std::fill(uU_uvecs.begin() + uIns.size(), uU_uvecs.end(), beacls::UVec(uIns[0], beacls::UVecType_Cuda, true));
		}
		uL_uvecs = uU_uvecs;
	}
	else {
		if (!dynSys->optCtrl_cuda(uL_uvecs, uU_uvecs, t, x_uvecs, derivMins, derivMaxs, uMode)) return false;
	}
	return true;
}
static
bool getOptDstb_cuda(
	std::vector<beacls::UVec >& dL_uvecs,
	std::vector<beacls::UVec >& dU_uvecs,
	const DynSys* dynSys,
	const std::vector<beacls::FloatVec >& dIns,
	const FLOAT_TYPE t,
	const std::vector<beacls::UVec>& x_uvecs,
	const std::vector<beacls::UVec>& derivMins,
	const std::vector<beacls::UVec>& derivMaxs,
	const DynSys_DMode_Type dMode
) {
	const size_t nd = dynSys->get_nd();
	dU_uvecs.resize(nd);
	dL_uvecs.resize(nd);
	if (!dIns.empty()) {
		if (dU_uvecs.size() < dIns.size()) dU_uvecs.resize(dIns.size());
		std::transform(dIns.cbegin(), dIns.cend(), dU_uvecs.begin(), [](const auto& rhs) {
			if (rhs.size() == 1) return beacls::UVec(rhs, beacls::UVecType_Vector, false);
			else return beacls::UVec(rhs, beacls::UVecType_Cuda, true);
		});
		if (dU_uvecs.size() > dIns.size()) {
			if (dIns[0].size() == 1) std::fill(dU_uvecs.begin() + dIns.size(), dU_uvecs.end(), beacls::UVec(dIns[0], beacls::UVecType_Vector, false));
			else std::fill(dU_uvecs.begin() + dIns.size(), dU_uvecs.end(), beacls::UVec(dIns[0], beacls::UVecType_Cuda, true));
		}
		dL_uvecs = dU_uvecs;
	}
	else {
		if (!dynSys->optDstb_cuda(dL_uvecs, dU_uvecs, t, x_uvecs, derivMins, derivMaxs, dMode)) return false;
	}
	return true;
}

static
bool getTIDynamics_cuda(
	std::vector<beacls::UVec >& TIdx_uvecs,
	const DynSys* dynSys,
	const FLOAT_TYPE t,
	const std::vector<beacls::UVec>& x_uvecs,
	const std::vector<beacls::UVec>& u_uvecs,
	const std::vector<beacls::UVec>& d_uvecs,
	const size_t dim = std::numeric_limits<size_t>::max()
) {
	if (!dynSys->get_TIdims().empty()) {
		const size_t nTI = dynSys->get_TIdims().size();
		TIdx_uvecs.resize(nTI);
		if (!dynSys->TIdyn_cuda(TIdx_uvecs, t, x_uvecs, u_uvecs, d_uvecs, dim)) return false;
	}
	return true;
}


bool DynSysSchemeData::hamFunc_cuda(
	beacls::UVec& hamValue_uvec,
	const FLOAT_TYPE t,
	const beacls::UVec& data,
	const std::vector<beacls::UVec>& x_uvecs,
	const std::vector<beacls::UVec>& derivs,
	const size_t begin_index,
	const size_t length
) const {
	// Custom derivative for MIE
	const std::vector<beacls::UVec>& custom_derivs = (!MIEderivs.empty()) ? MIEderivs : derivs;
	std::for_each(custom_derivs.begin(), custom_derivs.end(), ([](const auto& rhs) { beacls::synchronizeUVec(rhs); }));

	//!< Using Hamilton Jacobi function from dynamical system 
	if (dynSys->HamFunction_cuda(hamValue_uvec, this, t, data, x_uvecs, custom_derivs, begin_index, length, (tMode == DynSys_TMode_Backward) != (side.valid() && side.upper))) return true;

	std::vector<beacls::UVec >& u_uvecs = ws->u_uvecs;
	if (!getOptCtrl_cuda(u_uvecs, dynSys, uIns, t, x_uvecs, custom_derivs, uMode)) return false;

	std::vector<beacls::UVec >& d_uvecs = ws->d_uvecs;
	if (!getOptDstb_cuda(d_uvecs, dynSys, dIns, t, x_uvecs, custom_derivs, dMode)) return false;

	if (hamValue_uvec.type() != beacls::UVecType_Cuda) hamValue_uvec = beacls::UVec(beacls::type_to_depth<FLOAT_TYPE>(), beacls::UVecType_Cuda, length);
	else hamValue_uvec.resize(length);
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
	const size_t nx = dynSys->get_nx();
	std::vector<beacls::UVec >& dx_uvecs = ws->dx_uvecs;

	if (!getDynamics_cuda(dx_uvecs, dynSys, t, x_uvecs, u_uvecs, d_uvecs)) return false;

	std::vector<beacls::UVec> modified_derivs2(nx);
	std::transform(modified_derivs.cbegin(), modified_derivs.cend(), custom_derivs.cbegin(), modified_derivs2.begin(), [](const auto& lhs, const auto& rhs) {
		return (!rhs.empty()) ? rhs : lhs;
	});
	std::vector<beacls::UVec >& TIdx_uvecs = ws->TIdx_uvecs;
	if (!getTIDynamics_cuda(TIdx_uvecs, dynSys, t, x_uvecs, u_uvecs, d_uvecs)) return false;
	return hamFunc_exec_cuda(
		hamValue_uvec, modified_derivs2, dx_uvecs, TIdx_uvecs, TIderiv, 
		!dynSys->get_TIdims().empty(), (tMode == DynSys_TMode_Backward) != (side.valid() && side.upper));
}

bool DynSysSchemeData::partialFunc_cuda(
	beacls::UVec& alphas_uvec,
	const FLOAT_TYPE t,
	const beacls::UVec& data,
	const std::vector<beacls::UVec>& x_uvecs,
	const std::vector<beacls::UVec>& derivMins,
	const std::vector<beacls::UVec>& derivMaxs,
	const size_t dim,
	const size_t begin_index,
	const size_t length
) const {
	//!< Using partial function from dynamical system 
	if (dynSys->PartialFunction_cuda(alphas_uvec, this, t, data, x_uvecs, derivMins, derivMaxs, dim, begin_index, length)) return true;

	// Copy state matrices in case we're doing MIE
	// Dimension information(in case we're doing MIE)
	//	if (!MIEdims.empty()) {
	//	TIdims = schemeData.TIdims;
	//		dims = MIEdims;
	//	}
	const HJI_Grid *hji_grid = get_grid();
	const size_t num_of_dimensions = hji_grid->get_num_of_dimensions();
	ws->uL_uvecss.resize(num_of_dimensions);
	ws->uU_uvecss.resize(num_of_dimensions);
	ws->dL_uvecss.resize(num_of_dimensions);
	ws->dU_uvecss.resize(num_of_dimensions);
	ws->dxLL_uvecss.resize(num_of_dimensions);
	ws->dxLU_uvecss.resize(num_of_dimensions);
	ws->dxUL_uvecss.resize(num_of_dimensions);
	ws->dxUU_uvecss.resize(num_of_dimensions);

	std::vector<beacls::UVec >& uU_uvecs = ws->uU_uvecss[dim];
	std::vector<beacls::UVec >& uL_uvecs = ws->uL_uvecss[dim];
	if (!getOptCtrl_cuda(uL_uvecs, uU_uvecs, dynSys, uIns, t, x_uvecs, derivMins, derivMaxs, uMode)) {
		if (!getOptCtrl_cuda(uL_uvecs, dynSys, uIns, t, x_uvecs, derivMins, uMode)) return false;
		if (!getOptCtrl_cuda(uU_uvecs, dynSys, uIns, t, x_uvecs, derivMaxs, uMode)) return false;
	}
	std::vector<beacls::UVec >& dU_uvecs = ws->dU_uvecss[dim];
	std::vector<beacls::UVec >& dL_uvecs = ws->dL_uvecss[dim];
	if (!getOptDstb_cuda(dL_uvecs, dU_uvecs, dynSys, uIns, t, x_uvecs, derivMins, derivMaxs, dMode)) {
		if (!getOptDstb_cuda(dL_uvecs, dynSys, dIns, t, x_uvecs, derivMins, dMode)) return false;
		if (!getOptDstb_cuda(dU_uvecs, dynSys, dIns, t, x_uvecs, derivMaxs, dMode)) return false;
	}
	//!< Compute alpha
	std::vector<beacls::UVec >& dxLL_uvecs = ws->dxLL_uvecss[dim];
	std::vector<beacls::UVec >& dxLU_uvecs = ws->dxLU_uvecss[dim];
	std::vector<beacls::UVec >& dxUL_uvecs = ws->dxUL_uvecss[dim];
	std::vector<beacls::UVec >& dxUU_uvecs = ws->dxUU_uvecss[dim];

	if (!dynSys->dynamics_cuda(alphas_uvec, t, x_uvecs, uL_uvecs, uU_uvecs, dL_uvecs, dU_uvecs, dim)) {
		if (!getDynamics_cuda(dxLL_uvecs, dynSys, t, x_uvecs, uL_uvecs, dL_uvecs, dim)) return false;
		if (!getDynamics_cuda(dxLU_uvecs, dynSys, t, x_uvecs, uL_uvecs, dU_uvecs, dim)) return false;
		if (!getDynamics_cuda(dxUL_uvecs, dynSys, t, x_uvecs, uU_uvecs, dL_uvecs, dim)) return false;
		if (!getDynamics_cuda(dxUU_uvecs, dynSys, t, x_uvecs, uU_uvecs, dU_uvecs, dim)) return false;

		return partialFunc_exec_cuda(alphas_uvec, dxLL_uvecs[dim], dxLU_uvecs[dim], dxUL_uvecs[dim], dxUU_uvecs[dim]  );
	}
	else {
		return true;
	}
}
#endif /* defined(USER_DEFINED_GPU_DYNSYS_FUNC) */
