#include <helperOC/DynSys/PlaneCAvoid/PlaneCAvoid.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>
#include <array>
#include <typeinfo>
#include <levelset/Grids/HJI_Grid.hpp>
#include "PlaneCAvoid_cuda.hpp"
using namespace helperOC;


PlaneCAvoid::PlaneCAvoid(
	const beacls::FloatVec& x,
	const FLOAT_TYPE wMaxA,
	const beacls::FloatVec& vRangeA,
	const FLOAT_TYPE wMaxB,
	const beacls::FloatVec& vRangeB,
	const beacls::FloatVec& dMaxA,
	const beacls::FloatVec& dMaxB
) : DynSys(3, 2, 5, 
		beacls::IntegerVec{0, 1},  //!< Position dimensions
		beacls::IntegerVec{2}),  //!< Heading dimensions
	wMaxA(wMaxA), vRangeA(vRangeA), wMaxB(wMaxB), vRangeB(vRangeB), dMaxA(dMaxA), dMaxB(dMaxB) {
	if (x.size() != 3) {
		std::cerr << "Error: " << __func__ << " : Initial state does not have right dimension!" << std::endl;
	}
	DynSys::set_x(x);
	DynSys::push_back_xhist(x);
}
PlaneCAvoid::PlaneCAvoid(
	beacls::MatFStream* fs,
	beacls::MatVariable* variable_ptr
) :
	DynSys(fs, variable_ptr),
	wMaxA(0), vRangeA(beacls::FloatVec()),
	wMaxB(0), vRangeB(beacls::FloatVec()),
	dMaxA(beacls::FloatVec()), dMaxB(beacls::FloatVec())
{
	beacls::IntegerVec dummy;
	load_value(wMaxA, std::string("wMaxA"), true, fs, variable_ptr);
	load_vector(vRangeA, std::string("vRangeA"), dummy, true, fs, variable_ptr);
	load_value(wMaxB, std::string("wMaxB"), true, fs, variable_ptr);
	load_vector(vRangeB, std::string("vRangeB"), dummy, true, fs, variable_ptr);
	load_vector(dMaxA, std::string("dMaxA"), dummy, true, fs, variable_ptr);
	load_vector(dMaxB, std::string("dMaxB"), dummy, true, fs, variable_ptr);
}
PlaneCAvoid::~PlaneCAvoid() {
}
bool PlaneCAvoid::operator==(const PlaneCAvoid& rhs) const {
	if (this == &rhs) return true;
	else if (!DynSys::operator==(rhs)) return false;
	else if (wMaxA != rhs.wMaxA) return false;	//!< Angular control bounds
	else if (wMaxB != rhs.wMaxB) return false;	//!< Angular control bounds
	else if ((vRangeA.size() != rhs.vRangeA.size()) || !std::equal(vRangeA.cbegin(), vRangeA.cend(), rhs.vRangeA.cbegin())) return false;	//!< Speed control bounds
	else if ((vRangeB.size() != rhs.vRangeB.size()) || !std::equal(vRangeB.cbegin(), vRangeB.cend(), rhs.vRangeB.cbegin())) return false;	//!< Speed control bounds
	else if ((dMaxA.size() != rhs.dMaxA.size()) || !std::equal(dMaxA.cbegin(), dMaxA.cend(), rhs.dMaxA.cbegin())) return false;	//!< Disturbance
	else if ((dMaxB.size() != rhs.dMaxB.size()) || !std::equal(dMaxB.cbegin(), dMaxB.cend(), rhs.dMaxB.cbegin())) return false;	//!< Disturbance
	return true;
}
bool PlaneCAvoid::operator==(const DynSys& rhs) const {
	if (this == &rhs) return true;
	else if (typeid(*this) != typeid(rhs)) return false;
	else return operator==(dynamic_cast<const PlaneCAvoid&>(rhs));
}
bool PlaneCAvoid::save(
	beacls::MatFStream* fs,
	beacls::MatVariable* variable_ptr
) {
	bool result = DynSys::save(fs, variable_ptr);
	result &= save_value(wMaxA, std::string("wMaxA"), true, fs, variable_ptr);
	if (!vRangeA.empty()) result &= save_vector(vRangeA, std::string("vRangeA"), beacls::IntegerVec(), true, fs, variable_ptr);
	result &= save_value(wMaxB, std::string("wMaxB"), true, fs, variable_ptr);
	if (!vRangeB.empty()) result &= save_vector(vRangeB, std::string("vRangeB"), beacls::IntegerVec(), true, fs, variable_ptr);
	if (!dMaxA.empty()) result &= save_vector(dMaxA, std::string("dMaxA"), beacls::IntegerVec(), true, fs, variable_ptr);
	if (!dMaxB.empty()) result &= save_vector(dMaxB, std::string("dMaxB"), beacls::IntegerVec(), true, fs, variable_ptr);
	return result;
}
bool PlaneCAvoid::optCtrl(
	std::vector<beacls::FloatVec >& uOpts,
	const FLOAT_TYPE,
	const std::vector<beacls::FloatVec::const_iterator >& y_ites,
	const std::vector<const FLOAT_TYPE*>& deriv_ptrs,
	const beacls::IntegerVec& y_sizes,
	const beacls::IntegerVec& deriv_sizes,
	const helperOC::DynSys_UMode_Type uMode
) const {
	const helperOC::DynSys_UMode_Type modified_uMode = (uMode == helperOC::DynSys_UMode_Default) ? helperOC::DynSys_UMode_Max : uMode;
	const FLOAT_TYPE* deriv0 = deriv_ptrs[0];
	const FLOAT_TYPE* deriv1 = deriv_ptrs[1];
	const FLOAT_TYPE* deriv2 = deriv_ptrs[2];
	beacls::FloatVec::const_iterator ys0 = y_ites[0];
	beacls::FloatVec::const_iterator ys1 = y_ites[1];
	const size_t y0_size = y_sizes[0];
	const size_t deriv0_size = deriv_sizes[0];
	uOpts.resize(get_nu());
	uOpts[0].resize(deriv0_size);
	uOpts[1].resize(y0_size);
	if (y0_size == 0 || deriv0_size == 0 || deriv0 == NULL || deriv1 == NULL || deriv2 == NULL) return false;
	beacls::FloatVec& uOpt0 = uOpts[0];
	beacls::FloatVec& uOpt1 = uOpts[1];
	const FLOAT_TYPE vRangeA0 = vRangeA[0];
	const FLOAT_TYPE vRangeA1 = vRangeA[1];
	if ((modified_uMode == helperOC::DynSys_UMode_Max) || (modified_uMode == helperOC::DynSys_UMode_Min)) {
		const FLOAT_TYPE moded_wMaxA = (modified_uMode == helperOC::DynSys_UMode_Max) ? wMaxA : -wMaxA;
		const FLOAT_TYPE moded_vRangeA0 = (modified_uMode == helperOC::DynSys_UMode_Max) ? vRangeA0 : vRangeA1;
		const FLOAT_TYPE moded_vRangeA1 = (modified_uMode == helperOC::DynSys_UMode_Max) ? vRangeA1 : vRangeA0;
		if (deriv0_size != y0_size) {
			const FLOAT_TYPE d0 = deriv0[0];
			const FLOAT_TYPE d1 = deriv1[0];
			const FLOAT_TYPE d2 = deriv2[0];
			std::transform(ys0, ys0 + y0_size, ys1, uOpt1.begin(), [d0,d1,d2,moded_wMaxA](const auto& lhs, const auto& rhs) {
				const FLOAT_TYPE y0 = lhs;
				const FLOAT_TYPE y1 = rhs;
				const FLOAT_TYPE det1 = d0 * y1 - d1 * y0 - d2;
				return (det1 >= 0) ? moded_wMaxA : -moded_wMaxA;
			});
		}
		else {
			for (size_t index = 0; index < y0_size; ++index) {
				const FLOAT_TYPE y0 = ys0[index];
				const FLOAT_TYPE y1 = ys1[index];
				const FLOAT_TYPE d0 = deriv0[index];
				const FLOAT_TYPE d1 = deriv1[index];
				const FLOAT_TYPE d2 = deriv2[index];
				const FLOAT_TYPE det1 = d0 * y1 - d1 * y0 - d2;
				uOpt1[index] = (det1 >= 0) ? moded_wMaxA : -moded_wMaxA;
			}
		}
		for (size_t index = 0; index < deriv0_size; ++index) {
			const FLOAT_TYPE d0 = deriv0[index];
			const FLOAT_TYPE det0 = -d0;
			uOpt0[index] = (det0 >= 0) ? moded_vRangeA1 : moded_vRangeA0;
		}
	}
	else {
		std::cerr << "Unknown uMode!: " << modified_uMode << std::endl;
		return false;
	}
	return true;
}
bool PlaneCAvoid::optDstb(
	std::vector<beacls::FloatVec >& dOpts,
	const FLOAT_TYPE,
	const std::vector<beacls::FloatVec::const_iterator >& y_ites,
	const std::vector<const FLOAT_TYPE*>& deriv_ptrs,
	const beacls::IntegerVec& y_sizes,
	const beacls::IntegerVec& deriv_sizes,
	const helperOC::DynSys_DMode_Type dMode
) const {
	const helperOC::DynSys_DMode_Type modified_dMode = (dMode == helperOC::DynSys_DMode_Default) ? helperOC::DynSys_DMode_Min : dMode;
	const FLOAT_TYPE* deriv0 = deriv_ptrs[0];
	const FLOAT_TYPE* deriv1 = deriv_ptrs[1];
	const FLOAT_TYPE* deriv2 = deriv_ptrs[2];
	beacls::FloatVec::const_iterator ys2 = y_ites[2];
	const size_t y2_size = y_sizes[2];
	const size_t deriv0_size = deriv_sizes[0];
	if (y2_size == 0 || deriv0_size == 0 || deriv0 == NULL || deriv1 == NULL || deriv2 == NULL) return false;
	dOpts.resize(get_nd());
	dOpts[0].resize(y2_size);
	std::for_each(dOpts.begin() + 1, dOpts.end(), [&deriv0_size](auto& rhs) { rhs.resize(deriv0_size); });
	beacls::FloatVec& dOpt0 = dOpts[0];
	beacls::FloatVec& dOpt1 = dOpts[1];
	beacls::FloatVec& dOpt2 = dOpts[2];
	beacls::FloatVec& dOpt3 = dOpts[3];
	beacls::FloatVec& dOpt4 = dOpts[4];
	const FLOAT_TYPE dMaxA_0 = dMaxA[0];
	const FLOAT_TYPE dMaxA_1 = dMaxA[1];
	const FLOAT_TYPE dMaxB_0 = dMaxB[0];
	const FLOAT_TYPE dMaxB_1 = dMaxB[1];
	const FLOAT_TYPE vRangeB0 = vRangeB[0];
	const FLOAT_TYPE vRangeB1 = vRangeB[1];
	const FLOAT_TYPE dMaxA_0_dMaxB_0 = dMaxA_0 + dMaxB_0;
	const FLOAT_TYPE dMaxA_1_dMaxB_1 = dMaxA_1 + dMaxB_1;
	if ((modified_dMode == helperOC::DynSys_DMode_Max) || (modified_dMode == helperOC::DynSys_DMode_Min)) {
		const FLOAT_TYPE moded_wMaxB = (modified_dMode == helperOC::DynSys_DMode_Max) ? wMaxB : -wMaxB;
		const FLOAT_TYPE moded_vRangeB0 = (modified_dMode == helperOC::DynSys_DMode_Max) ? vRangeB0 : vRangeB1;
		const FLOAT_TYPE moded_vRangeB1 = (modified_dMode == helperOC::DynSys_DMode_Max) ? vRangeB1 : vRangeB0;
		const FLOAT_TYPE moded_dMaxA_0_dMaxB_0 = (modified_dMode == helperOC::DynSys_DMode_Max) ? dMaxA_0_dMaxB_0 : -dMaxA_0_dMaxB_0;
		const FLOAT_TYPE moded_dMaxA_1_dMaxB_1 = (modified_dMode == helperOC::DynSys_DMode_Max) ? dMaxA_1_dMaxB_1 : -dMaxA_1_dMaxB_1;

		if (deriv0_size != y2_size) {
			const FLOAT_TYPE d0 = deriv0[0];
			const FLOAT_TYPE d1 = deriv1[0];
			std::transform(ys2, ys2 + y2_size, dOpt0.begin(), [d0, d1, moded_vRangeB0, moded_vRangeB1](const auto& rhs) {
				const FLOAT_TYPE y2 = rhs;
				const FLOAT_TYPE det0 = d0 * std::cos(y2) + d1 * std::sin(y2);
				return (det0 >= 0) ? moded_vRangeB1 : moded_vRangeB0;
			});
		}
		else {
			for (size_t index = 0; index < y2_size; ++index) {
				const FLOAT_TYPE d0 = deriv0[index];
				const FLOAT_TYPE d1 = deriv1[index];
				const FLOAT_TYPE y2 = ys2[index];
				const FLOAT_TYPE det0 = d0 * std::cos(y2) + d1 * std::sin(y2);
				dOpt0[index] = (det0 >= 0) ? moded_vRangeB1 : moded_vRangeB0;
			}
		}
		for (size_t index = 0; index < deriv0_size; ++index) {
			const FLOAT_TYPE d0 = deriv0[index];
			const FLOAT_TYPE d1 = deriv1[index];
			const FLOAT_TYPE d2 = deriv2[index];
			const FLOAT_TYPE det1 = d2;
			const FLOAT_TYPE det4 = d2;
			dOpt1[index] = (det1 >= 0) ? moded_wMaxB : -moded_wMaxB;
			dOpt4[index] = (det4 >= 0) ? moded_dMaxA_1_dMaxB_1 : -moded_dMaxA_1_dMaxB_1;
			const FLOAT_TYPE denom = std::sqrt(d0 * d0 + d1 * d1);
			dOpt2[index] = (denom == 0) ? 0 : moded_dMaxA_0_dMaxB_0 * d0 / denom;
			dOpt3[index] = (denom == 0) ? 0 : moded_dMaxA_0_dMaxB_0 * d1 / denom;
		}
	}
	else {
		std::cerr << "Unknown dMode!: " << modified_dMode << std::endl;
		return false;
	}
	return true;
}

bool PlaneCAvoid::dynamics_cell_helper(
	std::vector<beacls::FloatVec >& dx,
	const std::vector<beacls::FloatVec::const_iterator >& x_ites,
	const std::vector<beacls::FloatVec >& us,
	const std::vector<beacls::FloatVec >& ds,
	const beacls::IntegerVec& x_sizes,
	const size_t dim
) const {
	beacls::FloatVec& dx_dim = dx[dim];
	const size_t dx_dim_size = (dim == 2) ? us[1].size() : x_sizes[0];
	dx[dim].resize(dx_dim_size);
	bool result = true;
	const beacls::FloatVec& us_1 = us[1];
	switch (dim) {
	case 0:
		{
			const beacls::FloatVec& ds_0 = ds[0];
			const beacls::FloatVec& ds_2 = ds[2];
			const beacls::FloatVec::const_iterator& x_ites1 = x_ites[1];
			const beacls::FloatVec::const_iterator& x_ites2 = x_ites[2];
			const beacls::FloatVec& us_0 = us[0];
			if ((us[0].size() == dx_dim_size) && (ds[2].size() == dx_dim_size)) {
				for (size_t index = 0; index < dx_dim_size; ++index) {
					dx_dim[index] = -us_0[index] + ds_0[index] * std::cos(x_ites2[index]) + us_1[index] * x_ites1[index] + ds_2[index];
				}
				return true;
			}
			else if ((us[0].size() != dx_dim_size) && (ds[2].size() != dx_dim_size)) {
				const FLOAT_TYPE u0 = us_0[0];
				const FLOAT_TYPE d2 = ds_2[0];
				for (size_t index = 0; index < dx_dim_size; ++index) {
					dx_dim[index] = -u0 + ds_0[index] * std::cos(x_ites2[index]) + us_1[index] * x_ites1[index] + d2;
				}
				return true;
			}
			std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << " Invalid data size" << std::endl;
			result = false;
		}
		break;
	case 1:
		{
			const beacls::FloatVec& ds_0 = ds[0];
			const beacls::FloatVec& ds_3 = ds[3];
			const beacls::FloatVec::const_iterator& x_ites0 = x_ites[0];
			const beacls::FloatVec::const_iterator& x_ites2 = x_ites[2];
			if (ds[3].size() == dx_dim_size) {
				for (size_t index = 0; index < dx_dim_size; ++index) {
					dx_dim[index] = ds_0[index] * std::sin(x_ites2[index]) - us_1[index] * x_ites0[index] + ds_3[index];
				}
				return true;
			}
			else {//!< ds_3_size != dx_dim_size
				const FLOAT_TYPE d3 = ds_3[0];
				for (size_t index = 0; index < dx_dim_size; ++index) {
					dx_dim[index] = ds_0[index] * std::sin(x_ites2[index]) - us_1[index] * x_ites0[index] + d3;
				}
				return true;
			}
		}
		break;
	case 2:
		{
			const beacls::FloatVec& ds_1 = ds[1];
			const beacls::FloatVec& ds_4 = ds[4];
			if ((ds[1].size() == dx_dim_size) && (ds[4].size() == dx_dim_size)) {
				for (size_t index = 0; index < dx_dim_size; ++index) {
					dx_dim[index] = ds_1[index] - us_1[index] + ds_4[index];
				}
				return true;
			}
			else if ((ds[1].size() != dx_dim_size) && (ds[4].size() != dx_dim_size)) {
				const FLOAT_TYPE d1 = ds_1[0];
				const FLOAT_TYPE d4 = ds_4[0];
				for (size_t index = 0; index < dx_dim_size; ++index) {
					dx_dim[index] = d1 - us_1[index] + d4;
				}
				return true;
			}
			std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << " Invalid data size" << std::endl;
			result = false;
		}
		break;
	default:
		std::cerr << "Only dimension 1-4 are defined for dynamics of PlaneCAvoid!" << std::endl;
		result = false;
		break;
	}
	return result;
}


bool PlaneCAvoid::dynamics(
	std::vector<beacls::FloatVec >& dx,
	const FLOAT_TYPE,
	const std::vector<beacls::FloatVec::const_iterator >& x_ites,
	const std::vector<beacls::FloatVec >& us,
	const std::vector<beacls::FloatVec >& ds,
	const beacls::IntegerVec& x_sizes,
	const size_t dst_target_dim
) const {
	if (us.size() != get_nu()) {
		std::cerr << "Incorrect number of control dimensions!" << std::endl;
	}
	if (ds.size() != get_nd()) {
		std::cerr << "Incorrect number of disturbance dimensions!" << std::endl;
	}
	bool result = true;
	if (dst_target_dim == std::numeric_limits<size_t>::max()) {
		result &= dynamics_cell_helper(dx, x_ites, us, ds, x_sizes, 0);
		result &= dynamics_cell_helper(dx, x_ites, us, ds, x_sizes, 1);
		result &= dynamics_cell_helper(dx, x_ites, us, ds, x_sizes, 2);
	}
	else
	{
		if (dst_target_dim < x_ites.size())
			result &= dynamics_cell_helper(dx, x_ites, us, ds, x_sizes, dst_target_dim);
		else {
			std::cerr << "Invalid target dimension for dynamics: " << dst_target_dim << std::endl;
			result = false;
		}
	}
	return result;
}

#if defined(USER_DEFINED_GPU_DYNSYS_FUNC)
bool PlaneCAvoid::optCtrl_cuda(
	std::vector<beacls::UVec>& u_uvecs,
	const FLOAT_TYPE,
	const std::vector<beacls::UVec>& x_uvecs,
	const std::vector<beacls::UVec>& deriv_uvecs,
	const helperOC::DynSys_UMode_Type uMode
) const {
	const helperOC::DynSys_UMode_Type modified_uMode = (uMode == helperOC::DynSys_UMode_Default) ? helperOC::DynSys_UMode_Max : uMode;
	if (x_uvecs.empty() || x_uvecs[0].empty() || deriv_uvecs.size() < 3 || deriv_uvecs[0].empty() || deriv_uvecs[1].empty() || deriv_uvecs[2].empty()) return false;
	return PlaneCAvoid_CUDA::optCtrl_execute_cuda(
		u_uvecs, x_uvecs, deriv_uvecs, wMaxA, vRangeA, modified_uMode);
}
bool PlaneCAvoid::optDstb_cuda(
	std::vector<beacls::UVec>& d_uvecs,
	const FLOAT_TYPE,
	const std::vector<beacls::UVec>& x_uvecs,
	const std::vector<beacls::UVec>& deriv_uvecs,
	const helperOC::DynSys_DMode_Type dMode
) const {
	const helperOC::DynSys_DMode_Type modified_dMode = (dMode == helperOC::DynSys_DMode_Default) ? helperOC::DynSys_DMode_Min : dMode;
	if (x_uvecs.size() < 3 || x_uvecs[2].empty() || deriv_uvecs.size() < 3 || deriv_uvecs[0].empty() || deriv_uvecs[1].empty() || deriv_uvecs[2].empty()) return false;
	return PlaneCAvoid_CUDA::optDstb_execute_cuda(
		d_uvecs, x_uvecs, deriv_uvecs, dMaxA, dMaxB, vRangeB, wMaxB, modified_dMode);
}
bool PlaneCAvoid::dynamics_cuda(
	std::vector<beacls::UVec>& dx_uvecs,
	const FLOAT_TYPE,
	const std::vector<beacls::UVec>& x_uvecs,
	const std::vector<beacls::UVec>& u_uvecs,
	const std::vector<beacls::UVec>& d_uvecs,
	const size_t dst_target_dim
) const {
	bool result = true;
	if (dst_target_dim == std::numeric_limits<size_t>::max()) {
		result &= PlaneCAvoid_CUDA::dynamics_cell_helper_execute_cuda_dimAll(dx_uvecs, x_uvecs, u_uvecs, d_uvecs);
	}
	else
	{
		if (dst_target_dim < x_uvecs.size()) {
			return PlaneCAvoid_CUDA::dynamics_cell_helper_execute_cuda(
				dx_uvecs[dst_target_dim], x_uvecs, u_uvecs, d_uvecs, dst_target_dim
			);
		}
		else {
			std::cerr << "Invalid target dimension for dynamics: " << dst_target_dim << std::endl;
			result = false;
		}
	}
	return result;
}
#endif /* defined(USER_DEFINED_GPU_DYNSYS_FUNC) */
