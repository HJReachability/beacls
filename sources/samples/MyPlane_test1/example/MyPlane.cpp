#include <helperOC/helperOC.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>
#include "MyPlane.hpp"
MyPlane::MyPlane(
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
MyPlane::~MyPlane() {
}
bool MyPlane::optCtrl(
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
bool MyPlane::optDstb(
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

bool MyPlane::dynamics_cell_helper(
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
		std::cerr << "Only dimension 0-2 are defined for dynamics of MyPlane!" << std::endl;
		result = false;
		break;
	}
	return result;
}

bool MyPlane::dynamics(
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
