#include <helperOC/DynSys/Quad4D/Quad4D.hpp>
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
#include "Quad4D_cuda.hpp"
#include <random>
#include <macro.hpp>
Quad4D::Quad4D(
	const beacls::FloatVec& x,
	const FLOAT_TYPE uMin,
	const FLOAT_TYPE uMax,
	const beacls::IntegerVec& dims
) : DynSys(dims.size(), 2, 0,
		beacls::IntegerVec{
			(size_t)std::distance(dims.cbegin(), std::find(dims.cbegin(),dims.cend(),0)), 
			(size_t)std::distance(dims.cbegin(), std::find(dims.cbegin(),dims.cend(),2)) },	//!< Position dimensions
		beacls::IntegerVec{
			(size_t)std::distance(dims.cbegin(), std::find(dims.cbegin(),dims.cend(),1)),
			(size_t)std::distance(dims.cbegin(), std::find(dims.cbegin(),dims.cend(),3)) }),	//!< Velocity dimensions
	uMin(uMin), uMax(uMax), dims(dims) {
	if (x.size() != get_nx()) {
		std::cerr << "Error: " << __func__ << " : Initial state does not have right dimension!" << std::endl;
	}

	DynSys::set_x(x);
	DynSys::push_back_xhist(x);

}
Quad4D::Quad4D(
	beacls::MatFStream* fs,
	beacls::MatVariable* variable_ptr
) :
	DynSys(fs, variable_ptr),
	uMin(-3),	//!< Control bounds
	uMax(3),	//!< Control bounds
	dims(beacls::IntegerVec{ 0,1,2,3 })	//!< Active dimensions
{
	beacls::IntegerVec dummy;
	load_value(uMin, std::string("uMin"), true, fs, variable_ptr);
	load_value(uMax, std::string("uMax"), true, fs, variable_ptr);
	load_vector(dims, std::string("dims"), dummy, true, fs, variable_ptr);
}
Quad4D::~Quad4D() {
}
bool Quad4D::operator==(const Quad4D& rhs) const {
	if (this == &rhs) return true;
	else if (!DynSys::operator==(rhs)) return false;
	else if (uMin != rhs.uMin) return false;	//!< Angular control bounds
	else if (uMax != rhs.uMax) return false;	//!< Angular control bounds
	else if ((dims.size() != rhs.dims.size()) || !std::equal(dims.cbegin(), dims.cend(), rhs.dims.cbegin())) return false;	//!< Speed control bounds
	else return true;
}
bool Quad4D::operator==(const DynSys& rhs) const {
	if (this == &rhs) return true;
	else if (typeid(*this) != typeid(rhs)) return false;
	else return operator==(dynamic_cast<const Quad4D&>(rhs));
}
bool Quad4D::save(
	beacls::MatFStream* fs,
	beacls::MatVariable* variable_ptr
) {
	bool result = DynSys::save(fs, variable_ptr);

	result &= save_value(uMin, std::string("uMin"), true, fs, variable_ptr);
	result &= save_value(uMax, std::string("uMax"), true, fs, variable_ptr);
	if (!dims.empty()) result &= save_vector(dims, std::string("dims"), beacls::IntegerVec(), true, fs, variable_ptr);
	return result;
}

bool Quad4D::optCtrl(
	std::vector<beacls::FloatVec >& uOpts,
	const FLOAT_TYPE,
	const std::vector<beacls::FloatVec::const_iterator >&,
	const std::vector<const FLOAT_TYPE*>& deriv_ptrs,
	const beacls::IntegerVec&,
	const beacls::IntegerVec& deriv_sizes,
	const DynSys_UMode_Type uMode
) const {
	const DynSys_UMode_Type modified_uMode = (uMode == DynSys_UMode_Default) ? DynSys_UMode_Min : uMode;
	const size_t dim1 = (size_t)std::distance(dims.cbegin(), std::find(dims.cbegin(), dims.cend(), 1));
	const size_t dim3 = (size_t)std::distance(dims.cbegin(), std::find(dims.cbegin(), dims.cend(), 3));

	const FLOAT_TYPE* deriv1_ptr = deriv_ptrs[dim1];
	const FLOAT_TYPE* deriv3_ptr = deriv_ptrs[dim3];
	const size_t deriv1_size = deriv_sizes[dim1];
	const size_t deriv3_size = deriv_sizes[dim3];
	if (deriv1_size == 0 || deriv3_size == 0 || deriv1_ptr == NULL || deriv3_ptr == NULL) return false;
	uOpts.resize(get_nu());
	uOpts[0].resize(deriv1_size);
	uOpts[1].resize(deriv3_size);
	beacls::FloatVec& uOpt0 = uOpts[0];
	beacls::FloatVec& uOpt1 = uOpts[1];

	if ((modified_uMode == DynSys_UMode_Max) || (modified_uMode == DynSys_UMode_Min)) {
		const FLOAT_TYPE moded_uMax = (modified_uMode == DynSys_UMode_Max) ? uMax : uMin;
		const FLOAT_TYPE moded_uMin = (modified_uMode == DynSys_UMode_Max) ? uMin : uMax;
		if (std::find(dims.cbegin(), dims.cend(), 1) != dims.cend()) {
			for (size_t index = 0; index < deriv1_size; ++index) {
				uOpt0[index] = (deriv1_ptr[index] >= 0) ? moded_uMax : moded_uMin;
			}
		}
		if (std::find(dims.cbegin(), dims.cend(), 3) != dims.cend()) {
			for (size_t index = 0; index < deriv3_size; ++index) {
				uOpt1[index] = (deriv3_ptr[index] >= 0) ? moded_uMax : moded_uMin;
			}
		}
	}
	else {
		std::cerr << "Unknown uMode!: " << uMode << std::endl;
		return false;
	}
	return true;
}

bool Quad4D::dynamics_cell_helper(
		beacls::FloatVec& dx,
		const std::vector<beacls::FloatVec::const_iterator >& x_ites,
		const std::vector<beacls::FloatVec >& us,
		const beacls::IntegerVec& x_sizes,
		const size_t dim
) const {
	bool result = true;
	switch(dim) {
	case 0:
	{
		const size_t dim1 = (size_t)std::distance(dims.cbegin(), std::find(dims.cbegin(), dims.cend(), 1));
		dx.resize(x_sizes[dim1]);
		std::copy(x_ites[dim1], x_ites[dim1] + x_sizes[dim1], dx.begin());
	}
		break;
	case 1:
		dx = us[0];
		break;
	case 2:
	{
		const size_t dim3 = (size_t)std::distance(dims.cbegin(), std::find(dims.cbegin(), dims.cend(), 3));
		dx.resize(x_sizes[dim3]);
		std::copy(x_ites[dim3], x_ites[dim3] + x_sizes[dim3], dx.begin());
	}
		break;
	case 3:
		dx = us[1];
		break;
	default:
		std::cerr << "Only dimensions 1-4 are defined for dynamics of Quad4DC! " << std::endl;
		result = false;
		break;

	}
	return result;
}

bool Quad4D::dynamics(
	std::vector<beacls::FloatVec >& dxs,
	const FLOAT_TYPE,
	const std::vector<beacls::FloatVec::const_iterator >& x_ites,
	const std::vector<beacls::FloatVec >& us,
	const std::vector<beacls::FloatVec >&,
	const beacls::IntegerVec& x_sizes,
	const size_t dst_target_dim
) const {
	if (dst_target_dim == std::numeric_limits<size_t>::max()) {
		for (size_t dim = 0; dim < dims.size(); ++dim) {
			dynamics_cell_helper(dxs[dim], x_ites, us, x_sizes, dim);
		}
	}
	else
	{
		if (dst_target_dim < x_ites.size())
			dynamics_cell_helper(dxs[dst_target_dim], x_ites, us, x_sizes, dst_target_dim);
		else
			std::cerr << "Invalid target dimension for dynamics: " << dst_target_dim << std::endl;
	}
	return true;
}



#if defined(USER_DEFINED_GPU_DYNSYS_FUNC) && 0
bool Quad4D::optCtrl_cuda(
	std::vector<beacls::UVec>& u_uvecs,
	const FLOAT_TYPE,
	const std::vector<beacls::UVec>& x_uvecs,
	const std::vector<beacls::UVec>& deriv_uvecs,
	const DynSys_UMode_Type uMode
) const {
	const DynSys_UMode_Type modified_uMode = (uMode == DynSys_UMode_Default) ? DynSys_UMode_Min : uMode;
	const size_t dim1 = (size_t)std::distance(dims.cbegin(), std::find(dims.cbegin(), dims.cend(), 1));
	const size_t dim3 = (size_t)std::distance(dims.cbegin(), std::find(dims.cbegin(), dims.cend(), 3));

	const size_t deriv1_size = deriv_uvecs[dim1].size();
	const size_t deriv3_size = deriv_uvecs[dim3].size();
	if (deriv1_size == 0 || deriv3_size == 0) return false;
	return Quad4D_CUDA::optCtrl_execute_cuda(u_uvecs, x_uvecs, deriv_uvecs, wMax, vrange_max, vrange_min, modified_uMode);
}
bool Quad4D::dynamics_cuda(
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
		result &= Quad4D_CUDA::dynamics_cell_helper_execute_cuda_dimAll(dx_uvecs, x_uvecs, u_uvecs, modified_d_uvecs);
	}
	else
	{
		if (dst_target_dim < x_uvecs.size()) {
			Quad4D_CUDA::dynamics_cell_helper_execute_cuda(dx_uvecs[dst_target_dim], x_uvecs, u_uvecs, modified_d_uvecs, dst_target_dim);
		}
		else {
			std::cerr << "Invalid target dimension for dynamics: " << dst_target_dim << std::endl;
			result = false;
		}
	}
	return result;
}
#endif /* defined(USER_DEFINED_GPU_DYNSYS_FUNC) */
