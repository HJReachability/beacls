#include <helperOC/DynSys/DubinsCar/DubinsCar.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <array>
#include <typeinfo>
#include <levelset/Grids/HJI_Grid.hpp>

#include "DubinsCar_cuda.hpp"


DubinsCar::DubinsCar(
	const beacls::FloatVec& x,
	const FLOAT_TYPE wMax,
	const FLOAT_TYPE speed,
	const beacls::FloatVec& dMax,
	const beacls::IntegerVec& dims
) : DynSys(dims.size(), 1, 3,
	beacls::IntegerVec{find_val(dims, 0), find_val(dims, 1)},  //!< Position dimensions
	beacls::IntegerVec{find_val(dims, 2)}),  //!< Heading dimensions
	wMax(wMax), speed(speed), dMax(dMax), dims(dims) {
	if (x.size() != DynSys::get_nx()) {
		std::cerr << "Error: " << __func__ << " : Initial state does not have right dimension!" << std::endl;
	}

	DynSys::set_x(x);
	DynSys::push_back_xhist(x);

}
DubinsCar::DubinsCar(
	beacls::MatFStream* fs,
	beacls::MatVariable* variable_ptr
) :
	DynSys(fs, variable_ptr),
	wMax(0),
	speed(0),
	dMax(beacls::FloatVec()),
	dims(beacls::IntegerVec())
{
	beacls::IntegerVec dummy;
	load_value(wMax, std::string("wMax"), true, fs, variable_ptr);
	load_value(speed, std::string("speed"), true, fs, variable_ptr);
	load_vector(dMax, std::string("dMax"), dummy, true, fs, variable_ptr);
	load_vector(dims, std::string("dims"), dummy, true, fs, variable_ptr);
}
DubinsCar::~DubinsCar() {
}
bool DubinsCar::operator==(const DubinsCar& rhs) const {
	if (this == &rhs) return true;
	else if (wMax != rhs.wMax) return false;	//!< Angular control bounds
	else if (speed != rhs.speed) return false;	//!< Constant speed
	else if ((dMax.size() != rhs.dMax.size()) || !std::equal(dMax.cbegin(), dMax.cend(), rhs.dMax.cbegin())) return false;	//!< Disturbance
	else if ((dims.size() != rhs.dims.size()) || !std::equal(dims.cbegin(), dims.cend(), rhs.dims.cbegin())) return false;	//!< Dimensions that are active
	return true;
}
bool DubinsCar::operator==(const DynSys& rhs) const {
	if (this == &rhs) return true;
	else if (typeid(*this) != typeid(rhs)) return false;
	else return operator==(dynamic_cast<const DubinsCar&>(rhs));
}
bool DubinsCar::save(
	beacls::MatFStream* fs,
	beacls::MatVariable* variable_ptr
) {
	bool result = DynSys::save(fs, variable_ptr);

	result &= save_value(wMax, std::string("wMax"), true, fs, variable_ptr);
	result &= save_value(speed, std::string("speed"), true, fs, variable_ptr);
	if (!dMax.empty()) result &= save_vector(dMax, std::string("dMax"), beacls::IntegerVec(), true, fs, variable_ptr);
	if (!dims.empty()) result &= save_vector(dims, std::string("dims"), beacls::IntegerVec(), true, fs, variable_ptr);
	return result;
}
bool DubinsCar::getVelocity(beacls::FloatVec& v, std::vector<beacls::FloatVec>& vhist) const {
	DynSys::getVelocity(v, vhist);
	//!< DubinsCar is a special case, since speed is a constant, not a state
	v.resize(1);
	v[0] = speed;
	vhist.resize(2, v);
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

bool DubinsCar::optCtrl(
	std::vector<beacls::FloatVec >& uOpts,
	const FLOAT_TYPE,
	const std::vector<beacls::FloatVec::const_iterator >&,
	const std::vector<const FLOAT_TYPE*>& deriv_ptrs,
	const beacls::IntegerVec&,
	const beacls::IntegerVec& deriv_sizes,
	const DynSys_UMode_Type uMode
) const {
	const DynSys_UMode_Type modified_uMode = (uMode == DynSys_UMode_Default) ? DynSys_UMode_Max : uMode;
	const size_t src_target_dim_index = find_val(dims, 2);
	if (src_target_dim_index == dims.size()) return false;
	const FLOAT_TYPE* deriv = deriv_ptrs[src_target_dim_index];
	const size_t length = deriv_sizes[src_target_dim_index];
	if (length == 0 || deriv == NULL) return false;
	beacls::FloatVec& uOpt = uOpts[0];
	uOpt.resize(length);
	switch (modified_uMode) {
	case DynSys_UMode_Max:
		std::transform(deriv, deriv + length, uOpt.begin(), [this](const auto& rhs) { return (rhs >= 0) ? wMax : -wMax;  });
		break;
	case DynSys_UMode_Min:
		std::transform(deriv, deriv + length, uOpt.begin(), [this](const auto& rhs) { return (rhs >= 0) ? -wMax : wMax;  });
		break;
	case DynSys_UMode_Invalid:
	default:
		std::cerr << "Unknown uMode!: " << modified_uMode << std::endl;
		return false;
	}
	return true;
}
bool DubinsCar::optDstb(
	std::vector<beacls::FloatVec >& dOpts,
	const FLOAT_TYPE,
	const std::vector<beacls::FloatVec::const_iterator >&,
	const std::vector<const FLOAT_TYPE*>& deriv_ptrs,
	const beacls::IntegerVec&,
	const beacls::IntegerVec& deriv_sizes,
	const DynSys_DMode_Type dMode
) const {
	const DynSys_DMode_Type modified_dMode = (dMode == DynSys_DMode_Default) ? DynSys_DMode_Min : dMode;
	const size_t src_target_dim_index = find_val(dims, 2);
	if (src_target_dim_index == dims.size()) return false;
	const FLOAT_TYPE* deriv = deriv_ptrs[src_target_dim_index];
	const size_t length = deriv_sizes[src_target_dim_index];
	if (length == 0 || deriv == NULL) return false;
	const size_t nd = get_nd();
	for (size_t dim = 0; dim < nd; ++dim) {
		if (find_val(dims, dim) != dims.size()) {
			beacls::FloatVec& dOpt = dOpts[dim];
			const FLOAT_TYPE dMax_d = dMax[dim];
			dOpt.resize(length);
			switch (modified_dMode) {
			case DynSys_DMode_Max:
				std::transform(deriv, deriv + length, dOpt.begin(), [dMax_d](const auto& rhs) { return (rhs >= 0) ? dMax_d : -dMax_d;  });
				break;
			case DynSys_DMode_Min:
				std::transform(deriv, deriv + length, dOpt.begin(), [dMax_d](const auto& rhs) { return (rhs >= 0) ? -dMax_d : dMax_d;  });
				break;
			case DynSys_UMode_Invalid:
			default:
				std::cerr << "Unknown dMode!: " << modified_dMode << std::endl;
				return false;
			}
		}
	}
	return true;
}

bool DubinsCar::dynamics_cell_helper(
	std::vector<beacls::FloatVec >& dxs,
	const beacls::FloatVec::const_iterator& x_ite,
	const std::vector<beacls::FloatVec >& us,
	const std::vector<beacls::FloatVec >& ds,
	const size_t x_size,
	const size_t dim
) const {
	beacls::FloatVec& dx_dim = dxs[dim];
	const size_t dx_dim_size = (dim == 2) ? us[0].size() : x_size;
	dx_dim.resize(dx_dim_size);
	bool result = true;
	switch (dim) {
	case 0:
		{
			const beacls::FloatVec& ds_0 = ds[0];
			if (ds[0].size() == x_size) {
				std::transform(x_ite, x_ite + x_size, ds_0.cbegin(), dx_dim.begin(), [this](const auto& lhs, const auto& rhs) {
					return  speed * std::cos(lhs) + rhs;
				});
			}
			else {	//!< ds_0_size != length
				const FLOAT_TYPE d0 = ds_0[0];
				std::transform(x_ite, x_ite + x_size, dx_dim.begin(), [this, d0](const auto& rhs) {
					return  speed * std::cos(rhs) + d0;
				});
			}
		}
		break;
	case 1:
		{
			const beacls::FloatVec& ds_1 = ds[1];
			if (ds[1].size() == x_size) {
				std::transform(x_ite, x_ite + x_size, ds_1.cbegin(), dx_dim.begin(), [this](const auto& lhs, const auto& rhs) {
					return  speed * std::sin(lhs) + rhs;
				});
			}
			else {	//!< ds_1_size != length
				const FLOAT_TYPE d1 = ds_1[0];
				std::transform(x_ite, x_ite + x_size, dx_dim.begin(), [this, d1](const auto& rhs) {
					return  speed * std::sin(rhs) + d1;
				});
			}
		}
		break;
	case 2:
		{
			const beacls::FloatVec& us_0 = us[0];
			const beacls::FloatVec& ds_2 = ds[2];
			if (us[0].size() == x_size) {
				if (ds[2].size() == x_size) {
					std::transform(us_0.cbegin(), us_0.cbegin() + dx_dim_size, ds_2.cbegin(), dx_dim.begin(), std::plus<FLOAT_TYPE>());
				}
				else {	//!< ds_2_size != length
					const FLOAT_TYPE d2 = ds_2[0];
					std::transform(us_0.cbegin(), us_0.cbegin() + dx_dim_size, dx_dim.begin(), [d2](const auto& rhs) { return rhs + d2; });
				}
			}
			else {	//!< us_2_size != length
				const FLOAT_TYPE u0 = us_0[0];
				if (ds[2].size() == x_size) {
					std::transform(ds_2.cbegin(), ds_2.cbegin() + dx_dim_size, dx_dim.begin(), [u0](const auto& rhs) { return u0 + rhs; });
				}
				else {	//!< ds_2_size != length
					const FLOAT_TYPE d2 = ds_2[0];
					std::fill(dx_dim.begin(), dx_dim.begin() + dx_dim_size, u0+d2);
				}
			}
		}
		break;
	default:
		std::cerr << "Only dimension 1-4 are defined for dynamics of DubinsCar!" << std::endl;
		result = false;
		break;
	}
	return result;
}
bool DubinsCar::dynamics(
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
	const size_t src_x_dim = find_val(dims, 2);
	const beacls::FloatVec::const_iterator& x_ites_target_dim = x_ites[src_x_dim];
	if (dst_target_dim == std::numeric_limits<size_t>::max()) {
		for (size_t dim = 0; dim < dims.size(); ++dim) {
			dynamics_cell_helper(dxs, x_ites_target_dim, us, modified_ds, x_sizes[src_x_dim], dims[dim]);
		}
	}
	else
	{
		if (find_val(dims,dst_target_dim) != dims.size())
			dynamics_cell_helper(dxs, x_ites_target_dim, us, modified_ds, x_sizes[src_x_dim], dst_target_dim);
		else
			std::cerr << "Invalid target dimension for dynamics: " << dst_target_dim << std::endl;
	}
	return true;
}

#if defined(USER_DEFINED_GPU_DYNSYS_FUNC)
bool DubinsCar::optCtrl_cuda(
	std::vector<beacls::UVec>& u_uvecs,
	const FLOAT_TYPE,
	const std::vector<beacls::UVec>&,
	const std::vector<beacls::UVec>& deriv_uvecs,
	const DynSys_UMode_Type uMode
) const {
	const DynSys_UMode_Type modified_uMode = (uMode == DynSys_UMode_Default) ? DynSys_UMode_Max : uMode;
	const size_t src_target_dim_index = find_val(dims, 2);
	if (src_target_dim_index == dims.size()) return false;
	if (deriv_uvecs.empty() || deriv_uvecs[src_target_dim_index].empty()) return false;
	u_uvecs.resize(get_nu());
	return DubinsCar_CUDA::optCtrl_execute_cuda(u_uvecs[0], deriv_uvecs[src_target_dim_index], wMax, modified_uMode);
}
bool DubinsCar::optDstb_cuda(
	std::vector<beacls::UVec>& d_uvecs,
	const FLOAT_TYPE,
	const std::vector<beacls::UVec>&,
	const std::vector<beacls::UVec>& deriv_uvecs,
	const DynSys_DMode_Type dMode
) const {
	const DynSys_DMode_Type modified_dMode = (dMode == DynSys_DMode_Default) ? DynSys_DMode_Min : dMode;
	const size_t src_target_dim_index = find_val(dims, 2);
	if (src_target_dim_index == dims.size()) return false;
	if (deriv_uvecs.empty() || deriv_uvecs[src_target_dim_index].empty()) return false;
	d_uvecs.resize(get_nd());
	return DubinsCar_CUDA::optDstb_execute_cuda(d_uvecs, deriv_uvecs[src_target_dim_index], dMax, modified_dMode, dims);
}

bool DubinsCar::dynamics_cuda(
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
	if (x_uvecs.empty())return false;
	bool result = true;
	dx_uvecs.resize(get_nx());
	if (dst_target_dim == std::numeric_limits<size_t>::max()) {
		const size_t src_x_dim = std::distance(dims.cbegin(), std::find(dims.cbegin(), dims.cend(), 2));
		beacls::IntegerVec::const_iterator dim0_ite = std::find(dims.cbegin(), dims.cend(), 0);
		beacls::IntegerVec::const_iterator dim1_ite = std::find(dims.cbegin(), dims.cend(), 1);
		//!< if dim0 and dim1 are also neccesary, use sincos intrinsics of CUDA.
		if (dim0_ite != dims.cend() && dim1_ite != dims.cend()) {
			result &= DubinsCar_CUDA::dynamics_cell_helper_execute_cuda_dimAll(
				dx_uvecs, x_uvecs, modified_d_uvecs,
				speed,
				std::distance(dims.cbegin(), dim0_ite),
				std::distance(dims.cbegin(), dim1_ite),
				src_x_dim
			);

		}
		for (beacls::IntegerVec::const_iterator ite = dims.cbegin(); ite != dims.cend(); ++ite) {
			if (ite != dim0_ite && ite != dim1_ite) {
				size_t src_target_dim = *ite;
				result &= DubinsCar_CUDA::dynamics_cell_helper_execute_cuda(
					dx_uvecs, x_uvecs, u_uvecs, modified_d_uvecs,
					speed,
					std::distance(dims.cbegin(), ite),
					src_target_dim,
					src_x_dim
				);
			}
		}
	}
	else
	{
		const size_t src_x_dim = std::distance(dims.cbegin(), std::find(dims.cbegin(), dims.cend(), 2));
		const size_t src_target_dim = find_val(dims, dst_target_dim);
		if (src_target_dim != dims.size())
			DubinsCar_CUDA::dynamics_cell_helper_execute_cuda(dx_uvecs, x_uvecs, u_uvecs, modified_d_uvecs, speed, dst_target_dim, src_target_dim, src_x_dim);
		else {
			std::cerr << "Invalid target dimension for dynamics: " << dst_target_dim << std::endl;
			result = false;
		}
	}
	return result;
}
#endif /* defined(USER_DEFINED_GPU_DYNSYS_FUNC) */
