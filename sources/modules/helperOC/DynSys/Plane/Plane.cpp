#include <helperOC/DynSys/Plane/Plane.hpp>
#include <helperOC/DynSys/DynSys/DynSysSchemeData.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>
#include <utility>
#include <array>
#include <typeinfo>
#include <levelset/Grids/HJI_Grid.hpp>
#include "Plane_cuda.hpp"
#include <random>
#include <macro.hpp>
using namespace helperOC;

Plane::Plane(
	const beacls::FloatVec &x,
	const FLOAT_TYPE wMax,
	const beacls::FloatVec& vrange,
	const beacls::FloatVec& dMax
) : DynSys(3, 2, 3,
	beacls::IntegerVec{0, 1},	//!< Position dimensions
	beacls::IntegerVec{2}),	//!< velocity dimensions
	wMax(wMax), vrange(vrange), dMax(dMax) {
	if (x.size() != 3) {
		std::cerr << "Error: " << __func__ << " : Initial state does not have right dimension!" << std::endl;
	}

	DynSys::set_x(x);
	DynSys::push_back_xhist(x);

}
Plane::Plane(
	beacls::MatFStream* fs,
	beacls::MatVariable* variable_ptr
) :
	DynSys(fs, variable_ptr),
	wMax(0),
	vrange(beacls::FloatVec()),
	dMax(beacls::FloatVec())
{
	beacls::IntegerVec dummy;
	load_value(wMax, std::string("wMax"), true, fs, variable_ptr);
	load_vector(vrange, std::string("vrange"), dummy, true, fs, variable_ptr);
	load_vector(dMax, std::string("dMax"), dummy, true, fs, variable_ptr);
}
Plane::~Plane() {
}
bool Plane::operator==(const Plane& rhs) const {
	if (this == &rhs) return true;
	else if (!DynSys::operator==(rhs)) return false;
	else if (wMax != rhs.wMax) return false;	//!< Angular control bounds
	else if ((vrange.size() != rhs.vrange.size()) || !std::equal(vrange.cbegin(), vrange.cend(), rhs.vrange.cbegin())) return false;	//!< Speed control bounds
	else if ((dMax.size() != rhs.dMax.size()) || !std::equal(dMax.cbegin(), dMax.cend(), rhs.dMax.cbegin())) return false;	//!< Disturbance
	else return true;
}
bool Plane::operator==(const DynSys& rhs) const {
	if (this == &rhs) return true;
	else if (typeid(*this) != typeid(rhs)) return false;
	else return operator==(dynamic_cast<const Plane&>(rhs));
}
bool Plane::save(
	beacls::MatFStream* fs,
	beacls::MatVariable* variable_ptr
) {
	bool result = DynSys::save(fs, variable_ptr);

	result &= save_value(wMax, std::string("wMax"), true, fs, variable_ptr);
	if (!vrange.empty()) result &= save_vector(vrange, std::string("vrange"), beacls::IntegerVec(), true, fs, variable_ptr);
	if (!dMax.empty()) result &= save_vector(dMax, std::string("dMax"), beacls::IntegerVec(), true, fs, variable_ptr);
	return result;
}
bool Plane::getVelocity(beacls::FloatVec& v, std::vector<beacls::FloatVec>& vhist) const {
	DynSys::getVelocity(v, vhist);
	//!< Plane is a special case, since speed is one of the controls
	v.resize(1);
	v[0] = std::accumulate(vrange.cbegin(), vrange.cend(), (FLOAT_TYPE)0) / vrange.size();
	vhist[0] = v;
	std::vector<beacls::FloatVec> u = get_u();
	if (!u.empty()) {
		v = u[0];
		std::vector<std::vector<beacls::FloatVec>> uhist = get_uhist();
		vhist.reserve(vhist.size() + uhist.size());
		std::for_each(uhist.cbegin(), uhist.cend(), [&vhist](const auto& rhs) {
			vhist.push_back(rhs[0]);
		});
	}
	//!< If the velocity is a scalar, and there's a heading dimension, then we need to
	//!< compute the velocity from speed and heading
	if (v.size() == 1 && !get_hdim().empty()) {
		beacls::FloatVec h;
		std::vector<beacls::FloatVec> hhist;
		getHeading(h, hhist);
		const FLOAT_TYPE vs = v[0];
		v.resize(2);
		v[0] = vs * std::cos(h[0]);
		v[1] = vs * std::sin(h[0]);
		std::transform(vhist.cbegin(), vhist.cend(), hhist.cbegin(), hhist.begin(), [](const auto& lhs, const auto& rhs) {
			const FLOAT_TYPE v_i = lhs[0];
			const FLOAT_TYPE h_i = rhs[0];
			return beacls::FloatVec{ v_i * std::cos(h_i), v_i * std::sin(h_i) };
		});
	}
	return true;
}
bool Plane::uniformDstb(std::vector<beacls::FloatVec> &ds) const {
	ds.resize(get_nd());
	std::for_each(ds.begin(), ds.end(), [](auto& rhs) {rhs.resize(1, 0); });
	FLOAT_TYPE d0;
	FLOAT_TYPE d1;
	FLOAT_TYPE norm_d0_d1;
	std::mt19937_64 mt(0);
	const FLOAT_TYPE dMax0 = dMax[0];
	std::uniform_real_distribution<FLOAT_TYPE> randomIntDistribution0(-dMax0, dMax0);
	do  {
		d0 = randomIntDistribution0(mt);
		d1 = randomIntDistribution0(mt);
		norm_d0_d1 = std::sqrt(d0 * d0 + d1 * d1);
	} while (norm_d0_d1 > dMax0);
	ds[0][0] = d0;
	ds[1][0] = d1;
	std::uniform_real_distribution<FLOAT_TYPE> randomIntDistribution1(-dMax[1], dMax[1]);
	ds[2][0] = randomIntDistribution1(mt);
	return true;

}

bool Plane::optCtrl(
	std::vector<beacls::FloatVec >& uOpts,
	const FLOAT_TYPE,
	const std::vector<beacls::FloatVec::const_iterator >& y_ites,
	const std::vector<const FLOAT_TYPE*>& deriv_ptrs,
	const beacls::IntegerVec& y_sizes,
	const beacls::IntegerVec& deriv_sizes,
	const helperOC::DynSys_UMode_Type uMode
) const {
	const helperOC::DynSys_UMode_Type modified_uMode = (uMode == helperOC::DynSys_UMode_Default) ? helperOC::DynSys_UMode_Max : uMode;
	const FLOAT_TYPE* deriv0_ptr = deriv_ptrs[0];
	const FLOAT_TYPE* deriv1_ptr = deriv_ptrs[1];
	const FLOAT_TYPE* deriv2_ptr = deriv_ptrs[2];
	beacls::FloatVec::const_iterator ys2 = y_ites[2];
	const size_t y2_size = y_sizes[2];
	const size_t deriv0_size = deriv_sizes[0];
	if (y2_size == 0 || deriv0_size == 0 || deriv0_ptr == NULL || deriv1_ptr == NULL || deriv2_ptr == NULL) return false;
	uOpts.resize(get_nu());
	uOpts[0].resize(y2_size);
	uOpts[1].resize(deriv0_size);
	beacls::FloatVec& uOpt0 = uOpts[0];
	beacls::FloatVec& uOpt1 = uOpts[1];
	const auto vrange_minmax = beacls::minmax_value<FLOAT_TYPE>(vrange.cbegin(), vrange.cend());
	const FLOAT_TYPE vrange_min = vrange_minmax.first;
	const FLOAT_TYPE vrange_max = vrange_minmax.second;

	if ((modified_uMode == helperOC::DynSys_UMode_Max) || (modified_uMode == helperOC::DynSys_UMode_Min)) {
		const FLOAT_TYPE moded_vrange_max = (modified_uMode == helperOC::DynSys_UMode_Max) ? vrange_max : vrange_min;
		const FLOAT_TYPE moded_vrange_min = (modified_uMode == helperOC::DynSys_UMode_Max) ? vrange_min : vrange_max;
		const FLOAT_TYPE moded_wMax = (modified_uMode == helperOC::DynSys_UMode_Max) ? wMax : -wMax;
		if (deriv0_size != y2_size) {
			const FLOAT_TYPE deriv0 = deriv0_ptr[0];
			const FLOAT_TYPE deriv1 = deriv1_ptr[0];
			std::transform(ys2, ys2 + y2_size, uOpt0.begin(), [deriv0, deriv1, moded_vrange_max, moded_vrange_min](const auto& rhs){
				const FLOAT_TYPE y2 = rhs;
				const FLOAT_TYPE det1 = deriv0 * std::cos(y2) + deriv1 * std::sin(y2);
				return (det1 >= 0) ? moded_vrange_max : moded_vrange_min;
			});
		}
		else {
			for (size_t index = 0; index < y2_size; ++index) {
				const FLOAT_TYPE y2 = ys2[index];
				const FLOAT_TYPE deriv0 = deriv0_ptr[index];
				const FLOAT_TYPE deriv1 = deriv1_ptr[index];
				const FLOAT_TYPE det1 = deriv0 * std::cos(y2) + deriv1 * std::sin(y2);
				uOpt0[index] = (det1 >= 0) ? moded_vrange_max : moded_vrange_min;
			}
		}
		std::transform(deriv2_ptr, deriv2_ptr + deriv0_size, uOpt1.begin(), [moded_wMax](const auto& rhs) { return (rhs >= 0) ? moded_wMax : -moded_wMax;  });
	}
	else {
		std::cerr << "Unknown uMode!: " << uMode << std::endl;
		return false;
	}
	return true;
}
bool Plane::optDstb(
	std::vector<beacls::FloatVec >& dOpts,
	const FLOAT_TYPE,
	const std::vector<beacls::FloatVec::const_iterator >&,
	const std::vector<const FLOAT_TYPE*>& deriv_ptrs,
	const beacls::IntegerVec&,
	const beacls::IntegerVec& deriv_sizes,
	const helperOC::DynSys_DMode_Type dMode
) const {
	const helperOC::DynSys_DMode_Type modified_dMode = (dMode == helperOC::DynSys_DMode_Default) ? helperOC::DynSys_DMode_Min : dMode;
	const FLOAT_TYPE* deriv0_ptr = deriv_ptrs[0];
	const FLOAT_TYPE* deriv1_ptr = deriv_ptrs[1];
	const FLOAT_TYPE* deriv2_ptr = deriv_ptrs[2];

	const size_t deriv0_size = deriv_sizes[0];
	if (deriv0_size == 0 || deriv0_ptr == NULL || deriv1_ptr == NULL || deriv2_ptr == NULL) return false;
	dOpts.resize(get_nd());
	std::for_each(dOpts.begin(), dOpts.end(), [deriv0_size](auto& rhs) { rhs.resize(deriv0_size); });
	beacls::FloatVec& dOpt0 = dOpts[0];
	beacls::FloatVec& dOpt1 = dOpts[1];
	beacls::FloatVec& dOpt2 = dOpts[2];
	const FLOAT_TYPE dMax_0 = dMax[0];
	const FLOAT_TYPE dMax_1 = dMax[1];

	if ((modified_dMode == helperOC::DynSys_DMode_Max) || (modified_dMode == helperOC::DynSys_DMode_Min)) {
		const FLOAT_TYPE moded_dMax_0 = (modified_dMode == helperOC::DynSys_DMode_Max) ? dMax_0 : -dMax_0;
		const FLOAT_TYPE moded_dMax_1 = (modified_dMode == helperOC::DynSys_DMode_Max) ? dMax_1 : -dMax_1;
		for (size_t index = 0; index < deriv0_size; ++index) {
			const FLOAT_TYPE deriv0 = deriv0_ptr[index];
			const FLOAT_TYPE deriv1 = deriv1_ptr[index];
			const FLOAT_TYPE deriv2 = deriv2_ptr[index];
			const FLOAT_TYPE normDeriv01 = std::sqrt(deriv0 * deriv0 + deriv1 * deriv1);
			dOpt0[index] = (normDeriv01 == 0) ? 0 : moded_dMax_0 * deriv0 / normDeriv01;
			dOpt1[index] = (normDeriv01 == 0) ? 0 : moded_dMax_0 * deriv1 / normDeriv01;
			dOpt2[index] = (deriv2 >= 0) ? moded_dMax_1 : -moded_dMax_1;
		}
	}
	else {
		std::cerr << "Unknown dMode!: " << modified_dMode << std::endl;
		return false;
	}
	return true;
}

bool Plane::dynamics_cell_helper(
	beacls::FloatVec& dx,
	const beacls::FloatVec::const_iterator& x_ite,
	const std::vector<beacls::FloatVec >& us,
	const std::vector<beacls::FloatVec >& ds,
	const size_t x_size,
	const size_t dim
) const {
	beacls::FloatVec& dx_dim = dx;
	const size_t dx_dim_size = (dim == 2) ? us[1].size() : x_size;
	dx.resize(dx_dim_size);
	bool result = true;
	switch (dim) {
	case 0:
		{
			const beacls::FloatVec& ds_0 = ds[0];
			const beacls::FloatVec& us_0 = us[0];
			if (ds[0].size() == dx_dim_size) {
				for (size_t index = 0; index < dx_dim_size; ++index) {
					dx_dim[index] = us_0[index] * std::cos(x_ite[index]) + ds_0[index];
				}
			}
			else {	//!< ds_0_size != dx_dim_size
				const FLOAT_TYPE d0 = ds_0[0];
				std::transform(x_ite, x_ite + dx_dim_size, us_0.cbegin(), dx_dim.begin(), [d0](const auto& lhs, const auto& rhs) {
					return  rhs * std::cos(lhs) + d0;
				});
			}
		}
		break;
	case 1:
		{
			const beacls::FloatVec& ds_1 = ds[1];
			const beacls::FloatVec& us_0 = us[0];
			if (ds[1].size() == dx_dim_size) {
				for (size_t index = 0; index < dx_dim_size; ++index) {
					dx_dim[index] = us_0[index] * std::sin(x_ite[index]) + ds_1[index];
				}
			}
			else {
				const FLOAT_TYPE d1 = ds_1[0];
				std::transform(x_ite, x_ite + dx_dim_size, us_0.cbegin(), dx_dim.begin(), [d1](const auto& lhs, const auto& rhs) {
					return  rhs * std::sin(lhs) + d1;
				});
			}
		}
		break;
	case 2:
		{
			const beacls::FloatVec& ds_2 = ds[2];
			const beacls::FloatVec& us_1 = us[1];
			if (ds[2].size() == dx_dim_size) {
				std::transform(us_1.cbegin(), us_1.cbegin() + dx_dim_size, ds_2.cbegin(), dx_dim.begin(), std::plus<FLOAT_TYPE>());
			}
			else {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << " Invalid data size" << std::endl;
				result = false;
			}
		}
		break;
	default:
		std::cerr << "Only dimension 1-4 are defined for dynamics of Plane!" << std::endl;
		result = false;
		break;
	}
	return result;
}

bool Plane::dynamics(
	std::vector<beacls::FloatVec >& dxs,
	const FLOAT_TYPE,
	const std::vector<beacls::FloatVec::const_iterator >& x_ites,
	const std::vector<beacls::FloatVec >& us,
	const std::vector<beacls::FloatVec >& ds,
	const beacls::IntegerVec& x_sizes,
	const size_t dst_target_dim
) const {
	static const std::vector<beacls::FloatVec >& dummy_ds{ beacls::FloatVec{0},beacls::FloatVec{0},beacls::FloatVec{0} };
	const std::vector<beacls::FloatVec >& modified_ds = (ds.empty()) ? dummy_ds : ds;
	const size_t src_x_dim_index = 2;
	const beacls::FloatVec::const_iterator& x_ites_target_dim = x_ites[src_x_dim_index];
	if (dst_target_dim == std::numeric_limits<size_t>::max()) {
		dynamics_cell_helper(dxs[0], x_ites_target_dim, us, modified_ds, x_sizes[src_x_dim_index], 0);
		dynamics_cell_helper(dxs[1], x_ites_target_dim, us, modified_ds, x_sizes[src_x_dim_index], 1);
		dynamics_cell_helper(dxs[2], x_ites_target_dim, us, modified_ds, x_sizes[src_x_dim_index], 2);
	}
	else
	{
		if (dst_target_dim < x_ites.size())
			dynamics_cell_helper(dxs[dst_target_dim], x_ites_target_dim, us, modified_ds, x_sizes[src_x_dim_index], dst_target_dim);
		else
			std::cerr << "Invalid target dimension for dynamics: " << dst_target_dim << std::endl;
	}
	return true;
}

FLOAT_TYPE Plane::get_wMax() const {
	return wMax;
}
const beacls::FloatVec& Plane::get_vrange() const {
	return vrange;
}
const beacls::FloatVec& Plane::get_dMax() const {
	return dMax;
}


#if defined(USER_DEFINED_GPU_DYNSYS_FUNC)
bool Plane::optCtrl_cuda(
	std::vector<beacls::UVec>& u_uvecs,
	const FLOAT_TYPE,
	const std::vector<beacls::UVec>& x_uvecs,
	const std::vector<beacls::UVec>& deriv_uvecs,
	const helperOC::DynSys_UMode_Type uMode
) const {
	if (x_uvecs.size() < 3 || x_uvecs[2].empty() || deriv_uvecs.size() < 3 || deriv_uvecs[0].empty() || deriv_uvecs[1].empty() || deriv_uvecs[2].empty()) return false;
	const helperOC::DynSys_UMode_Type modified_uMode = (uMode == helperOC::DynSys_UMode_Default) ? helperOC::DynSys_UMode_Max : uMode;
	const auto vrange_minmax = beacls::minmax_value<FLOAT_TYPE>(vrange.cbegin(), vrange.cend());
	const FLOAT_TYPE vrange_min = vrange_minmax.first;
	const FLOAT_TYPE vrange_max = vrange_minmax.second;
	return Plane_CUDA::optCtrl_execute_cuda(u_uvecs, x_uvecs, deriv_uvecs, wMax, vrange_max, vrange_min, modified_uMode);
}
bool Plane::optDstb_cuda(
	std::vector<beacls::UVec>& d_uvecs,
	const FLOAT_TYPE,
	const std::vector<beacls::UVec>& x_uvecs,
	const std::vector<beacls::UVec>& deriv_uvecs,
	const helperOC::DynSys_DMode_Type dMode
) const {
	if (deriv_uvecs.size() < 3 || deriv_uvecs[0].empty() || deriv_uvecs[1].empty() || deriv_uvecs[2].empty()) return false;
	const helperOC::DynSys_DMode_Type modified_dMode = (dMode == helperOC::DynSys_DMode_Default) ? helperOC::DynSys_DMode_Min : dMode;
	return Plane_CUDA::optDstb_execute_cuda(d_uvecs, x_uvecs, deriv_uvecs, dMax, modified_dMode);
}
bool Plane::dynamics_cuda(
	std::vector<beacls::UVec>& dx_uvecs,
	const FLOAT_TYPE,
	const std::vector<beacls::UVec>& x_uvecs,
	const std::vector<beacls::UVec>& u_uvecs,
	const std::vector<beacls::UVec>& d_uvecs,
	const size_t dst_target_dim
) const {
	beacls::FloatVec dummy_d_vec{ 0 };
	std::vector<beacls::UVec> dummy_d_uvecs;
	if (d_uvecs.empty()) {
		dummy_d_uvecs.resize(get_nd());
		std::for_each(dummy_d_uvecs.begin(), dummy_d_uvecs.end(), [&dummy_d_vec](auto& rhs) {
			rhs = beacls::UVec(dummy_d_vec, beacls::UVecType_Vector, false);
		});
	}
	const std::vector<beacls::UVec>& modified_d_uvecs = (d_uvecs.empty()) ? dummy_d_uvecs : d_uvecs;
	bool result = true;
	if (dst_target_dim == std::numeric_limits<size_t>::max()) {
		result &= Plane_CUDA::dynamics_cell_helper_execute_cuda_dimAll(dx_uvecs, x_uvecs, u_uvecs, modified_d_uvecs);
	}
	else
	{
		if (dst_target_dim < x_uvecs.size()) {
			Plane_CUDA::dynamics_cell_helper_execute_cuda(dx_uvecs[dst_target_dim], x_uvecs, u_uvecs, modified_d_uvecs, dst_target_dim);
		}
		else {
			std::cerr << "Invalid target dimension for dynamics: " << dst_target_dim << std::endl;
			result = false;
		}
	}
	return result;
}

bool Plane::optCtrl_cuda(
	std::vector<beacls::UVec>& uU_uvecs,
	std::vector<beacls::UVec>& uL_uvecs,
	const FLOAT_TYPE,
	const std::vector<beacls::UVec>& x_uvecs,
	const std::vector<beacls::UVec>& derivMax_uvecs,
	const std::vector<beacls::UVec>& derivMin_uvecs,
	const helperOC::DynSys_UMode_Type uMode
) const {
	if (x_uvecs.size() < 3 || x_uvecs[2].empty() || derivMax_uvecs.size() < 3 || derivMax_uvecs[0].empty() || derivMax_uvecs[1].empty() || derivMax_uvecs[2].empty()) return false;
	const helperOC::DynSys_UMode_Type modified_uMode = (uMode == helperOC::DynSys_UMode_Default) ? helperOC::DynSys_UMode_Max : uMode;
	const auto vrange_minmax = beacls::minmax_value<FLOAT_TYPE>(vrange.cbegin(), vrange.cend());
	const FLOAT_TYPE vrange_min = vrange_minmax.first;
	const FLOAT_TYPE vrange_max = vrange_minmax.second;
	return Plane_CUDA::optCtrl_execute_cuda(uU_uvecs, uL_uvecs, x_uvecs, derivMax_uvecs, derivMin_uvecs, wMax, vrange_max, vrange_min, modified_uMode);
}
bool Plane::optDstb_cuda(
	std::vector<beacls::UVec>& dU_uvecs,
	std::vector<beacls::UVec>& dL_uvecs,
	const FLOAT_TYPE,
	const std::vector<beacls::UVec>& x_uvecs,
	const std::vector<beacls::UVec>& derivMax_uvecs,
	const std::vector<beacls::UVec>& derivMin_uvecs,
	const helperOC::DynSys_DMode_Type dMode
) const {
	if (derivMax_uvecs.size() < 3 || derivMax_uvecs[0].empty() || derivMax_uvecs[1].empty() || derivMax_uvecs[2].empty()) return false;
	const helperOC::DynSys_DMode_Type modified_dMode = (dMode == helperOC::DynSys_DMode_Default) ? helperOC::DynSys_DMode_Min : dMode;
	return Plane_CUDA::optDstb_execute_cuda(dU_uvecs, dL_uvecs, x_uvecs, derivMax_uvecs, derivMin_uvecs, dMax, modified_dMode);
}
bool Plane::dynamics_cuda(
	beacls::UVec& alpha_uvec,
	const FLOAT_TYPE,
	const std::vector<beacls::UVec>& x_uvecs,
	const std::vector<beacls::UVec>& uU_uvecs,
	const std::vector<beacls::UVec>& uL_uvecs,
	const std::vector<beacls::UVec>& dU_uvecs,
	const std::vector<beacls::UVec>& dL_uvecs,
	const size_t dst_target_dim
) const {
	beacls::FloatVec dummy_d_vec{ 0 };
	std::vector<beacls::UVec> dummy_d_uvecs;
	if (dU_uvecs.empty() || dL_uvecs.empty()) {
		dummy_d_uvecs.resize(get_nd());
		std::for_each(dummy_d_uvecs.begin(), dummy_d_uvecs.end(), [&dummy_d_vec](auto& rhs) {
			rhs = beacls::UVec(dummy_d_vec, beacls::UVecType_Vector, false);
		});
	}
	const std::vector<beacls::UVec>& modified_dU_uvecs = (dU_uvecs.empty()) ? dummy_d_uvecs : dU_uvecs;
	const std::vector<beacls::UVec>& modified_dL_uvecs = (dL_uvecs.empty()) ? dummy_d_uvecs : dL_uvecs;
	bool result = true;
	if (dst_target_dim < x_uvecs.size()) {
		Plane_CUDA::dynamics_cell_helper_execute_cuda(alpha_uvec, x_uvecs, uU_uvecs, uL_uvecs, modified_dU_uvecs, modified_dL_uvecs, dst_target_dim);
	}
	else {
		std::cerr << "Invalid target dimension for dynamics: " << dst_target_dim << std::endl;
		result = false;
	}
	return result;
}
bool Plane::HamFunction_cuda(
	beacls::UVec& hamValue_uvec,
	const DynSysSchemeData* schemeData,
	const FLOAT_TYPE,
	const beacls::UVec&,
	const std::vector<beacls::UVec>& x_uvecs,
	const std::vector<beacls::UVec>& deriv_uvecs,
	const size_t,
	const size_t,
	const bool negate
) const {
	if (x_uvecs.size() < 3 || x_uvecs[2].empty() || deriv_uvecs.size() < 3 || deriv_uvecs[0].empty() || deriv_uvecs[1].empty() || deriv_uvecs[2].empty()) return false;
	if (deriv_uvecs.size() < 3 || deriv_uvecs[0].empty() || deriv_uvecs[1].empty() || deriv_uvecs[2].empty()) return false;
	const auto vrange_minmax = beacls::minmax_value<FLOAT_TYPE>(vrange.cbegin(), vrange.cend());
	const FLOAT_TYPE vrange_min = vrange_minmax.first;
	const FLOAT_TYPE vrange_max = vrange_minmax.second;
	const helperOC::DynSys_UMode_Type modified_uMode = (schemeData->uMode == helperOC::DynSys_UMode_Default) ? helperOC::DynSys_UMode_Max : schemeData->uMode;
	const helperOC::DynSys_DMode_Type modified_dMode = (schemeData->dMode == helperOC::DynSys_DMode_Default) ? helperOC::DynSys_DMode_Min : schemeData->dMode;
	return Plane_CUDA::HamFunction_cuda(hamValue_uvec, x_uvecs, deriv_uvecs, wMax, vrange_min, vrange_max, dMax, modified_uMode, modified_dMode, negate);
}


bool Plane::PartialFunction_cuda(
	beacls::UVec& alpha_uvec,
	const DynSysSchemeData* schemeData,
	const FLOAT_TYPE,
	const beacls::UVec&,
	const std::vector<beacls::UVec>& x_uvecs,
	const std::vector<beacls::UVec>& derivMin_uvecs,
	const std::vector<beacls::UVec>& derivMax_uvecs,
	const size_t dim,
	const size_t,
	const size_t
	) const {
	if (x_uvecs.size() < 3 || x_uvecs[2].empty() || derivMax_uvecs.size() < 3 || derivMax_uvecs[0].empty() || derivMax_uvecs[1].empty() || derivMax_uvecs[2].empty()) return false;
	if (derivMax_uvecs.size() < 3 || derivMax_uvecs[0].empty() || derivMax_uvecs[1].empty() || derivMax_uvecs[2].empty()) return false;
	const auto vrange_minmax = beacls::minmax_value<FLOAT_TYPE>(vrange.cbegin(), vrange.cend());
	const FLOAT_TYPE vrange_min = vrange_minmax.first;
	const FLOAT_TYPE vrange_max = vrange_minmax.second;
	const helperOC::DynSys_UMode_Type modified_uMode = (schemeData->uMode == helperOC::DynSys_UMode_Default) ? helperOC::DynSys_UMode_Max : schemeData->uMode;
	const helperOC::DynSys_DMode_Type modified_dMode = (schemeData->dMode == helperOC::DynSys_DMode_Default) ? helperOC::DynSys_DMode_Min : schemeData->dMode;
	return Plane_CUDA::PartialFunction_cuda(alpha_uvec, x_uvecs, derivMin_uvecs, derivMax_uvecs, dim, wMax, vrange_min, vrange_max, dMax, modified_uMode, modified_dMode);
}
#endif /* defined(USER_DEFINED_GPU_DYNSYS_FUNC) */
