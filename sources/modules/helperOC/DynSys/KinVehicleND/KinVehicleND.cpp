#include <helperOC/DynSys/KinVehicleND/KinVehicleND.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <array>
#include <typeinfo>
#include <levelset/Grids/HJI_Grid.hpp>

#include "KinVehicleND_cuda.hpp"


KinVehicleND::KinVehicleND(
	const beacls::FloatVec& x,
	const FLOAT_TYPE vMax
) : DynSys(x.size(), x.size()), vMax(vMax)
{
	DynSys::set_x(x);
	DynSys::push_back_xhist(x);
}
KinVehicleND::KinVehicleND(
	beacls::MatFStream* fs,
	beacls::MatVariable* variable_ptr
) :
	DynSys(fs, variable_ptr),
	vMax(0) {
	load_value(vMax, std::string("vMax"), true, fs, variable_ptr);
}
KinVehicleND::~KinVehicleND() {
}
bool KinVehicleND::operator==(const KinVehicleND& rhs) const {
	if (this == &rhs) return true;
	else if (vMax != rhs.vMax) return false;	//!< Angular control bounds
	return true;
}
bool KinVehicleND::operator==(const DynSys& rhs) const {
	if (this == &rhs) return true;
	else if (!DynSys::operator==(rhs)) return false;
	else if (typeid(*this) != typeid(rhs)) return false;
	else return operator==(dynamic_cast<const KinVehicleND&>(rhs));
}
bool KinVehicleND::save(
	beacls::MatFStream* fs,
	beacls::MatVariable* variable_ptr
) {
	bool result = DynSys::save(fs, variable_ptr);
	result &= save_value(vMax, std::string("vMax"), true, fs, variable_ptr);
	return result;
}

bool KinVehicleND::optCtrl(
	std::vector<beacls::FloatVec >& uOpts,
	const FLOAT_TYPE,
	const std::vector<beacls::FloatVec::const_iterator >&,
	const std::vector<const FLOAT_TYPE*>& deriv_ptrs,
	const beacls::IntegerVec&,
	const beacls::IntegerVec& deriv_sizes,
	const DynSys_UMode_Type uMode
) const {

	const DynSys_UMode_Type modified_uMode = (uMode == DynSys_UMode_Default) ? DynSys_UMode_Min : uMode;
	const size_t length = deriv_sizes[0];
	if (length == 0) return false;
	const size_t nu = get_nu();
	uOpts.resize(get_nu());
	std::for_each(uOpts.begin(), uOpts.end(), [&length](auto& rhs) { rhs.resize(length); });
	//!< store denom to uOpts[nu-1]
	std::transform(deriv_ptrs[0], deriv_ptrs[0] + length, uOpts[nu - 1].begin(), [](const auto& rhs) { return rhs * rhs; });
	for (size_t dim = 1; dim < nu; ++dim) {
		std::transform(deriv_ptrs[dim], deriv_ptrs[dim] + length, uOpts[nu - 1].cbegin(), uOpts[nu - 1].begin(), [](const auto& lhs, const auto& rhs) { return rhs + lhs * lhs; });
	}
	std::transform(uOpts[nu - 1].cbegin(), uOpts[nu - 1].cend(), uOpts[nu - 1].begin(),[](const auto& rhs) { return std::sqrt(rhs); });
	if ((uMode == DynSys_UMode_Max) || (uMode == DynSys_UMode_Min)) {
		const FLOAT_TYPE moded_vMax = (uMode == DynSys_UMode_Max) ? vMax : -vMax;
		for (size_t dim = 0; dim < nu; ++dim) {
			beacls::FloatVec& uOpt = uOpts[dim];
			const FLOAT_TYPE* deriv = deriv_ptrs[dim];
			std::transform(deriv, deriv + length, uOpts[nu - 1].cbegin(), uOpt.begin(), [&moded_vMax](const auto& lhs, const auto& rhs) {
				return (rhs == 0) ? 0 : moded_vMax * lhs / rhs;
			});
		}
	}
	else {
		std::cerr << "Unknown uMode!: " << modified_uMode << std::endl;
		return false;
	}
	return true;
}
bool KinVehicleND::dynamics(
	std::vector<beacls::FloatVec >& dx,
	const FLOAT_TYPE,
	const std::vector<beacls::FloatVec::const_iterator >&,
	const std::vector<beacls::FloatVec >& us,
	const std::vector<beacls::FloatVec >&,
	const beacls::IntegerVec&,
	const size_t dst_target_dim
) const {
	if (dst_target_dim == std::numeric_limits<size_t>::max()) {
		dx = us;
	}
	else {
		dx[dst_target_dim] = us[dst_target_dim];
	}
	return true;
}

#if defined(USER_DEFINED_GPU_DYNSYS_FUNC)
bool KinVehicleND::optCtrl_cuda(
	std::vector<beacls::UVec>& u_uvecs,
	const FLOAT_TYPE,
	const std::vector<beacls::UVec>&,
	const std::vector<beacls::UVec>& deriv_uvecs,
	const DynSys_UMode_Type uMode
) const {
	const DynSys_UMode_Type modified_uMode = (uMode == DynSys_UMode_Default) ? DynSys_UMode_Max : uMode;
	if (deriv_uvecs.empty() || deriv_uvecs[0].empty()) return false;
	return KinVehicleND_CUDA::optCtrl_execute_cuda(u_uvecs, deriv_uvecs, vMax, modified_uMode);
}
bool KinVehicleND::dynamics_cuda(
	std::vector<beacls::UVec>& dx_uvecs,
	const FLOAT_TYPE,
	const std::vector<beacls::UVec>&,
	const std::vector<beacls::UVec>& u_uvecs,
	const std::vector<beacls::UVec>&,
	const size_t dst_target_dim
) const {
	bool result = true;
	if (dst_target_dim == std::numeric_limits<size_t>::max()) {
		dx_uvecs = u_uvecs;
	}
	else {
		dx_uvecs[dst_target_dim] = u_uvecs[dst_target_dim];
	}
	return result;
}
#endif /* defined(USER_DEFINED_GPU_DYNSYS_FUNC) */
