#include <helperOC/DynSys/Air3D/Air3D.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <typeinfo>
#include <levelset/Grids/HJI_Grid.hpp>
using namespace helperOC;


Air3D::Air3D(
	const beacls::FloatVec& x,
	const FLOAT_TYPE uMax,
	const FLOAT_TYPE dMax,
	const FLOAT_TYPE va,
	const FLOAT_TYPE vb,
	const beacls::IntegerVec& dims
) : DynSys(3, 1, 1,
	beacls::IntegerVec{0, 1},  //!< Position dimensions
	beacls::IntegerVec{2}),  //!< Heading dimensions
	uMax(uMax), dMax(dMax), va(va), vb(vb), dims(dims) {
	//!< Process control range

	if (x.size() != DynSys::get_nx()) {
		std::cerr << "Error: " << __func__ << " : Initial state does not have right dimension!" << std::endl;
	}
	//!< Process initial state
	DynSys::set_x(x);
	DynSys::push_back_xhist(x);

}
Air3D::Air3D(
	beacls::MatFStream* fs,
	beacls::MatVariable* variable_ptr
) :
	DynSys(fs, variable_ptr),
	uMax(0),
	dMax(0),
	va(0),
	vb(0),
	dims(beacls::IntegerVec()) {
	beacls::IntegerVec dummy;
	load_value(uMax, std::string("uMax"), true, fs, variable_ptr);
	load_value(dMax, std::string("dMax"), true, fs, variable_ptr);
	load_value(va, std::string("va"), true, fs, variable_ptr);
	load_value(vb, std::string("vb"), true, fs, variable_ptr);
	load_vector(dims, std::string("dims"), dummy, true, fs, variable_ptr);
}
Air3D::~Air3D() {
}
bool Air3D::operator==(const Air3D& rhs) const {
	if (this == &rhs) return true;
	else if (!DynSys::operator==(rhs)) return false;
	else if (uMax != rhs.uMax) return false;	//!< Control bounds
	else if (dMax != rhs.dMax) return false;	//!< Control bounds
	else if (va != rhs.va) return false;	//!< Vehicle Speeds
	else if (vb != rhs.vb) return false;	//!< Vehicle Speeds
	else if ((dims.size() != rhs.dims.size()) || !std::equal(dims.cbegin(), dims.cend(), rhs.dims.cbegin())) return false;	//!< Dimensions that are active
	return true;
}
bool Air3D::operator==(const DynSys& rhs) const {
	if (this == &rhs) return true;
	else if (typeid(*this) != typeid(rhs)) return false;
	else return operator==(dynamic_cast<const Air3D&>(rhs));
}
bool Air3D::save(
	beacls::MatFStream* fs,
	beacls::MatVariable* variable_ptr
) {
	bool result = DynSys::save(fs, variable_ptr);
	result &= save_value(uMax, std::string("uMax"), true, fs, variable_ptr);
	result &= save_value(dMax, std::string("dMax"), true, fs, variable_ptr);
	result &= save_value(va, std::string("va"), true, fs, variable_ptr);
	result &= save_value(vb, std::string("vb"), true, fs, variable_ptr);
	if (!dims.empty()) result &= save_vector(dims, std::string("dims"), beacls::IntegerVec(), true, fs, variable_ptr);

	return result;
}

bool Air3D::optCtrl(
	std::vector<beacls::FloatVec >& uOpts,
	const FLOAT_TYPE,
	const std::vector<beacls::FloatVec::const_iterator >& y_ites,
	const std::vector<const FLOAT_TYPE*>& deriv_ptrs,
	const beacls::IntegerVec& y_sizes,
	const beacls::IntegerVec& deriv_sizes,
	const helperOC::DynSys_UMode_Type uMode
) const {
	for (size_t dimension = 0; dimension < deriv_ptrs.size(); ++dimension) {
		if (deriv_sizes[dimension] == 0 || deriv_ptrs[dimension] == NULL) return false;
	}
	const helperOC::DynSys_UMode_Type modified_uMode = (uMode == helperOC::DynSys_UMode_Default) ? helperOC::DynSys_UMode_Max : uMode;
	uOpts.resize(get_nu());
	beacls::FloatVec& uOpt = uOpts[0];
	const size_t y0_size = y_sizes[0];
	uOpt.resize(y0_size);
	const size_t deriv0_size = deriv_sizes[0];
	const FLOAT_TYPE* derivs0 = deriv_ptrs[0];
	const FLOAT_TYPE* derivs1 = deriv_ptrs[1];
	const FLOAT_TYPE* derivs2 = deriv_ptrs[2];
	const beacls::FloatVec::const_iterator& y_ites0 = y_ites[0];
	const beacls::FloatVec::const_iterator& y_ites1 = y_ites[1];

	if (deriv0_size != y0_size) {
		const FLOAT_TYPE deriv0 = derivs0[0];
		const FLOAT_TYPE deriv1 = derivs1[0];
		const FLOAT_TYPE deriv2 = derivs2[0];
		switch (modified_uMode) {
		case helperOC::DynSys_UMode_Max:
			std::transform(y_ites1, y_ites1 + y0_size, y_ites0, uOpt.begin(), ([deriv0, deriv1, deriv2, this](const auto& lhs, const auto& rhs) {
				const FLOAT_TYPE det = deriv0 * lhs - deriv1 * rhs - deriv2;
				return (det >= 0) ? uMax : -uMax;
			}));
			break;
		case helperOC::DynSys_UMode_Min:
			std::transform(y_ites1, y_ites1 + y0_size, y_ites0, uOpt.begin(), ([deriv0, deriv1, deriv2, this](const auto& lhs, const auto& rhs) {
				FLOAT_TYPE det = deriv0 * lhs - deriv1 * rhs - deriv2;
				return (det >= 0) ? -uMax : uMax;
			}));
			break;
		case helperOC::DynSys_UMode_Invalid:
		default:
			std::cerr << "Unknown uMode!: " << modified_uMode << std::endl;
			return false;
		}
	}
	else {
		switch (modified_uMode) {
		case helperOC::DynSys_UMode_Max:
			for (size_t index = 0; index < deriv0_size; ++index) {
				const FLOAT_TYPE det = derivs0[index] * y_ites1[index] - derivs1[index] * y_ites0[index] - derivs2[index];
				uOpt[index] = (det >= 0) ? uMax : -uMax;
			}
			break;
		case helperOC::DynSys_UMode_Min:
			for (size_t index = 0; index < deriv0_size; ++index) {
				const FLOAT_TYPE det = derivs0[index] * y_ites1[index] - derivs1[index] * y_ites0[index] - derivs2[index];
				uOpt[index] = (det >= 0) ? -uMax : uMax;
			}
			break;
		case helperOC::DynSys_UMode_Invalid:
		default:
			std::cerr << "Unknown uMode!: " << modified_uMode << std::endl;
			return false;
		}
	}
	return true;
}
bool Air3D::optDstb(
	std::vector<beacls::FloatVec >& dOpts,
	const FLOAT_TYPE,
	const std::vector<beacls::FloatVec::const_iterator >&,
	const std::vector<const FLOAT_TYPE*>& deriv_ptrs,
	const beacls::IntegerVec&,
	const beacls::IntegerVec& deriv_sizes,
	const helperOC::DynSys_DMode_Type dMode
) const {
	const helperOC::DynSys_DMode_Type modified_dMode = (dMode == helperOC::DynSys_DMode_Default) ? helperOC::DynSys_DMode_Min : dMode;
	const FLOAT_TYPE* derivs2 = deriv_ptrs[2];
	size_t deriv2_size = deriv_sizes[2];
	if (deriv2_size==0|| derivs2 == NULL) return false;
	dOpts.resize(get_nd());
	beacls::FloatVec& dOpt = dOpts[0];
	dOpt.resize(deriv2_size);
	switch (modified_dMode) {
	case helperOC::DynSys_UMode_Max:
		for (size_t i = 0; i < deriv2_size; ++i) {
			dOpt[i] = (derivs2[i] >= 0) ? dMax : -dMax;
		}
		break;
	case helperOC::DynSys_DMode_Min:
		for (size_t i = 0; i < deriv2_size; ++i) {
			dOpt[i] = (derivs2[i] >= 0) ? -dMax : dMax;
		}
		break;
	case helperOC::DynSys_UMode_Invalid:
	default:
		std::cerr << "Unknown dMode!: " << modified_dMode << std::endl;
		return false;

	}
	return true;
}

bool Air3D::dynamics(
	std::vector<beacls::FloatVec >& dx,
	const FLOAT_TYPE,
	const std::vector<beacls::FloatVec::const_iterator >& x_ites,
	const std::vector<beacls::FloatVec >& us,
	const std::vector<beacls::FloatVec >& ds,
	const beacls::IntegerVec& x_sizes,
	const size_t dst_target_dim
) const {
	size_t xs_length = x_sizes[0];
	const beacls::FloatVec& u0 = us[0];
	const beacls::FloatVec& d0 = ds[0];
	const beacls::FloatVec::const_iterator& x_ites0 = x_ites[0];
	const beacls::FloatVec::const_iterator& x_ites1 = x_ites[1];
	const beacls::FloatVec::const_iterator& x_ites2 = x_ites[2];
	dx.resize(get_nx());
	bool result = true;
	if (dst_target_dim == std::numeric_limits<size_t>::max()) {
		beacls::FloatVec& dx0 = dx[0];
		beacls::FloatVec& dx1 = dx[1];
		beacls::FloatVec& dx2 = dx[2];
		std::for_each(dx.begin(), dx.end(), [&xs_length](auto& rhs) { rhs.resize(xs_length); });
		if (d0.size() == 1) {
			FLOAT_TYPE d_0 = d0[0];
			for (size_t index = 0; index < xs_length; ++index) {
				FLOAT_TYPE x2_i = x_ites2[index];
				FLOAT_TYPE x1_i = x_ites1[index];
				FLOAT_TYPE x0_i = x_ites0[index];
				FLOAT_TYPE u_i = u0[index];
				dx0[index] = -va + vb * std::cos(x2_i) + u_i * x1_i;
				dx1[index] = vb * std::sin(x2_i) - u_i * x0_i;
				dx2[index] = d_0 - u_i;
			}
		}
		else {
			for (size_t index = 0; index < xs_length; ++index) {
				FLOAT_TYPE x2_i = x_ites2[index];
				FLOAT_TYPE x1_i = x_ites1[index];
				FLOAT_TYPE x0_i = x_ites0[index];
				FLOAT_TYPE u_i = u0[index];
				dx0[index] = -va + vb * std::cos(x2_i) + u_i * x1_i;
				dx1[index] = vb * std::sin(x2_i) - u_i * x0_i;
				dx2[index] = d0[index] - u_i;
			}
		}
	}
	else
	{
		switch (dst_target_dim) {
			case 0:
			{
				beacls::FloatVec& dx0 = dx[0];
				dx0.resize(xs_length);
				for (size_t index = 0; index < xs_length; ++index) {
					FLOAT_TYPE x2_i = x_ites2[index];
					FLOAT_TYPE x1_i = x_ites1[index];
					FLOAT_TYPE u_i = u0[index];
					dx0[index] = -va + vb * std::cos(x2_i) + u_i * x1_i;
				}
			}
			break;
		case 1:
			{
				beacls::FloatVec& dx1 = dx[1];
				dx1.resize(xs_length);
				for (size_t index = 0; index < xs_length; ++index) {
					FLOAT_TYPE x2_i = x_ites2[index];
					FLOAT_TYPE x0_i = x_ites0[index];
					FLOAT_TYPE u_i = u0[index];
					dx1[index] = vb * std::sin(x2_i) - u_i * x0_i;
				}
			}
			break;
		case 2:
			{
				beacls::FloatVec& dx2 = dx[2];
				dx2.resize(xs_length);
				if (d0.size() == 1) {
					FLOAT_TYPE d_0 = d0[0];
					for (size_t index = 0; index < xs_length; ++index) {
						FLOAT_TYPE u_i = u0[index];
						dx2[index] = d_0 - u_i;
					}
				}
				else {
					for (size_t index = 0; index < xs_length; ++index) {
						FLOAT_TYPE u_i = u0[index];
						dx2[index] = d0[index] - u_i;
					}
				}
			}
			break;
		default:
			std::cerr << "Invalid target dimension for dynamics: " << dst_target_dim << std::endl;
			result = false;
			break;
		}
	}
	return result;
}
