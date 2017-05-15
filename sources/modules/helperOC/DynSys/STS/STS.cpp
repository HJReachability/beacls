#include <helperOC/DynSys/STS/STS.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <array>
#include <typeinfo>
#include <levelset/Grids/HJI_Grid.hpp>

#include "STS_cuda.hpp"
using namespace helperOC;

STS::STS(
    const beacls::FloatVec& x,
    const FLOAT_TYPE M1, //!<% mass of thighs
    const FLOAT_TYPE M2,    //!<% mass of head-arms-trunk
    const FLOAT_TYPE L0,          //!<% length of segment (shank)
    const FLOAT_TYPE L1,         //!< % length of segment .4(thigh)
    const FLOAT_TYPE R1,         //!< % position of COM along segment (thigh)
    const FLOAT_TYPE R2,
    const FLOAT_TYPE grav,
    const beacls::FloatVec& TMax,
    const beacls::FloatVec& TMin,
    const beacls::FloatVec& dMax = beacls::FloatVec{ 0,0,0,0 },
    const beacls::IntegerVec& dims
) : DynSys(dims.size(), TMax.size(), dMax.size(),
	beacls::IntegerVec{find_val(dims, 0), find_val(dims, 2)},  //!< Position dimensions
	beacls::IntegerVec{find_val(dims, 1), find_val(dims, 3)}),  //!< periodic dimensions??
	M1(M1), M2(M2), L0(L0), L1(L1), R1(R1), R2(R2), grav(grav), TMax(TMax), TMin(TMin), dMax(dMax), dims(dims) {
	if (x.size() != DynSys::get_nx()) {
		std::cerr << "Error: " << __func__ << " : Initial state does not have right dimension!" << std::endl;
	}

	DynSys::set_x(x);
	DynSys::push_back_xhist(x);

}
STS::STS(
	beacls::MatFStream* fs,
	beacls::MatVariable* variable_ptr
) :
	DynSys(fs, variable_ptr),
    M1(0),
    M2(0),
    L0(0),
    L1(0),
    R1(0),
    R2(0),
    grav(0),
    TMax(beaclss::FloatVec()),
    TMin(beacls::FloatVec()),
	dMax(beacls::FloatVec()),
	dims(beacls::IntegerVec())
{
	beacls::IntegerVec dummy;
	load_value(M1, std::string("M1"), true, fs, variable_ptr);
    load_value(M2, std::string("M2"), true, fs, variable_ptr);
    load_value(L0, std::string("L0"), true, fs, variable_ptr);
    load_value(L1, std::string("L1"), true, fs, variable_ptr);
    load_value(R1, std::string("R1"), true, fs, variable_ptr);
    load_value(R2, std::string("R2"), true, fs, variable_ptr);
    load_value(grav, std::string("grav"), true, fs, variable_ptr);
    load_vector(TMax, std::string("TMax"), dummy, true, fs, variable_ptr);
    load_vector(TMin, std::string("TMin"), dummy, true, fs, variable_ptr);
	load_vector(dMax, std::string("dMax"), dummy, true, fs, variable_ptr);
	load_vector(dims, std::string("dims"), dummy, true, fs, variable_ptr);
}
STS::~STS() {
}
bool STS::operator==(const STS& rhs) const {
	if (this == &rhs) return true;
	else if (M1 != rhs.M1) return false;
    else if (M2 != rhs.M2) return false;
    else if (L0 != rhs.L0) return false;
    else if (L1 != rhs.L1) return false;
    else if (R1 != rhs.R1) return false;
    else if (R2 != rhs.R2) return false;
    else if (grav != rhs.grav) return false;
	else if ((TMax.size() != rhs.TMax.size()) || !std::equal(TMax.cbegin(), TMax.cend(), rhs.TMax.cbegin())) return false;	//!< Control max
    else if ((TMin.size() != rhs.TMin.size()) || !std::equal(TMin.cbegin(), TMin.cend(), rhs.TMin.cbegin())) return false;	//!< Control Miin
	else if ((dMax.size() != rhs.dMax.size()) || !std::equal(dMax.cbegin(), dMax.cend(), rhs.dMax.cbegin())) return false;	//!< Disturbance
	else if ((dims.size() != rhs.dims.size()) || !std::equal(dims.cbegin(), dims.cend(), rhs.dims.cbegin())) return false;	//!< Dimensions that are active
	return true;
}
bool STS::operator==(const DynSys& rhs) const {
	if (this == &rhs) return true;
	else if (typeid(*this) != typeid(rhs)) return false;
	else return operator==(dynamic_cast<const STS&>(rhs));
}
bool STS::save(
	beacls::MatFStream* fs,
	beacls::MatVariable* variable_ptr
) {
	bool result = DynSys::save(fs, variable_ptr);

	result &= save_value(M1, std::string("M1"), true, fs, variable_ptr);
    result &= save_value(M2, std::string("M2"), true, fs, variable_ptr);
    result &= save_value(L0, std::string("L0"), true, fs, variable_ptr);
    result &= save_value(L1, std::string("L1"), true, fs, variable_ptr);
    result &= save_value(R1, std::string("R1"), true, fs, variable_ptr);
    result &= save_value(R2, std::string("R2"), true, fs, variable_ptr);
    result &= save_value(grav, std::string("grav"), true, fs, variable_ptr);
    if (!TMax.empty()) result &= save_vector(TMax, std::string("TMax"), beacls::IntegerVec(), true, fs, variable_ptr);
    if (!TMin.empty()) result &= save_vector(TMin, std::string("TMin"), beacls::IntegerVec(), true, fs, variable_ptr);
	if (!dMax.empty()) result &= save_vector(dMax, std::string("dMax"), beacls::IntegerVec(), true, fs, variable_ptr);
	if (!dims.empty()) result &= save_vector(dims, std::string("dims"), beacls::IntegerVec(), true, fs, variable_ptr);
	return result;
}
bool STS::optCtrl(
    //output
	std::vector<beacls::FloatVec >& uOpts,
	
    //inputs
    const FLOAT_TYPE,  //ignore this input for this function
	const std::vector<beacls::FloatVec::const_iterator >& y_ites,
	const std::vector<const FLOAT_TYPE*>& deriv_ptrs,
	const beacls::IntegerVec& y_sizes,
	const beacls::IntegerVec& deriv_sizes,
	const helperOC::DynSys_UMode_Type uMode
) const {
    // if uMode is not specified then default to min
	const helperOC::DynSys_UMode_Type modified_uMode = (uMode == helperOC::DynSys_UMode_Default) ? helperOC::DynSys_UMode_Min : uMode;
    
    //get the spatial derivatives and states that you will need
    const FLOAT_TYPE* deriv1_ptr = deriv_pters[1];
    const FLOAT_TYPE* deriv3_ptr = deriv_pters[3];
    beacls::FloatVec::const_iterator ys0 = y_ites[0];
    beacls::FloatVec::const_iterator ys2 = y_ites[2];
    
    //find the size of the states and the derivatives
    const size_t y0_size = y_sizes[0];
    const size_t y2_size = y_sizes[2];
    const size_t deriv1_size = deriv_sizes[1];
    if (y0_size == 0 || y2_size == 0 || deriv1_size == 0 || deriv1_ptr == NULL || deriv3_ptr == NULL || deriv2_ptr == NULL || deriv3_ptr == NULL) return false;
    
    //set the optimal control size
    uOpts.resize(get_nu());
    uOpts[0].resize(y0_size);
    uOpts[1].resize(y0_size);
    
    //make a pointer to the optimal control
    beacls::FloatVec& uOpt0 = uOpts[0];
    beacls::FloatVec& uOpt1 = uOpts[1];

    
    if ((modified_uMode == helperOC::DynSys_UMode_Max)||(modified_uMode==helperOC::DynSys_UMode_Min)){
        //switch max/min of controls based on uMode
        const FLOAT_TYPE moded_tau1max = (modified_uMode == helperOC::DynSys_UMode_Max) ? TMax[0] : TMin[0];
        const FLOAT_TYPE moded_tau1min = (modified_uMode == helperOC::DynSys_UMode_Max) ? TMin[0] : TMax[0];
        const FLOAT_TYPE moded_tau2max = (modified_uMode == helperOC::DynSys_UMode_Max) ? TMax[1] : TMin[1];
        const FLOAT_TYPE moded_tau2min = (modified_uMode == helperOC::DynSys_UMode_Max) ? TMin[1] : TMax[1];
        
        // if your derivs are scalar, just need their first value.  otherwise, need to index through them
        if (deriv1_size != y0_size){
            const FLOAT_TYPE deriv1 = deriv1_ptr[0];
            const FLOAT_TYPE deriv3 = deriv3_ptr[0];
            
            for (size_t index = 0; index < y0_size; ++index){
                //get your current relevant states and derivs
                const FLOAT_TYPE y0 = ys0[index];
                const FLOAT_Type y2 = ys2[index];
                //solve for the control multipliers
                const FLOAT_TYPE ang = std::cos(y0-y2);
                const FLOAT_TYPE denom1 = M1 * pow(R1,4.0) + pow(L1,2.0)*M2*pow(R1,2.0) - pow(L1,2.0)*M2*pow(R2,2.0)*pow(ang,2.0);
                const FLOAT_TYPE tau1num1 = pow(R1,2.0);
                const FLOAT_TYPE tau2num1 = -ang*L1*R2;
                const FLOAT_TYPE denom2 = M2*M1*pow(R1,4.0) + pow(M2,2.0)*pow(L1,2.0)pow(R1,2.0) - pow(M2,2.0)*pow(L1,2.0)*pow(R2,2.0)*pow(ang,2.0);
                const FLOAT_TYPE tau1num2 = -L1*M2*R2*ang;
                const FLOAT_TYPE tau2num2 = M1*pow(R1,2.0)+pow(L1,2.0)*M2;
                const FLOAT_TYPE det0 = deriv1*tau1num1/denom1 + deriv3*tau1num2/denom2;
                const FLOAT_TYPE det1 = deriv1*tau2num1/denom1 + deriv3*tau2num2/denom2;
                
                //find the optimal control
                uOpt0[index]=(det0 >= 0) ? moded_tau1max : moded_tau1min;
                uOpt1[index]=(det1 >= 0) ? moded_tau2max : moded_tau2min;
            }
        }
        else {
            for (size_t index = 0; index < y0_size; ++index){
                
                //get your current relevant states and derivs
                const FLOAT_TYPE y0 = ys0[index];
                const FLOAT_Type y2 = ys2[index];
                const FLOAT_TYPE deriv1 = deriv1_ptr[index];
                const FLOAT_TYPE deriv3 = deriv3_ptr[index];
                //solve for the control multipliers
                const FLOAT_TYPE ang = std::cos(y0-y2);
                const FLOAT_TYPE denom1 = M1 * pow(R1,4.0) + pow(L1,2.0)*M2*pow(R1,2.0) - pow(L1,2.0)*M2*pow(R2,2.0)*pow(ang,2.0);
                const FLOAT_TYPE tau1num1 = pow(R1,2.0);
                const FLOAT_TYPE tau2num1 = -ang*L1*R2;
                const FLOAT_TYPE denom2 = M2*M1*pow(R1,4.0) + pow(M2,2.0)*pow(L1,2.0)pow(R1,2.0) - pow(M2,2.0)*pow(L1,2.0)*pow(R2,2.0)*pow(ang,2.0);
                const FLOAT_TYPE tau1num2 = -L1*M2*R2*ang;
                const FLOAT_TYPE tau2num2 = M1*pow(R1,2.0)+pow(L1,2.0)*M2;
                const FLOAT_TYPE det0 = deriv1*tau1num1/denom1 + deriv3*tau1num2/denom2;
                const FLOAT_TYPE det1 = deriv1*tau2num1/denom1 + deriv3*tau2num2/denom2;
                
                //find the optimal control
                uOpt0[index]=(det0 >= 0) ? moded_tau1max : moded_tau1min;
                uOpt1[index]=(det1 >= 0) ? moded_tau2max : moded_tau2min;
            }
        }
    else {
        std::cerr << "Uknown uMode!: " << uMode << std::end1;
        return false;
    }
    return true;
    }


bool STS::optDstb(
	std::vector<beacls::FloatVec >& dOpts,
	const FLOAT_TYPE,
	const std::vector<beacls::FloatVec::const_iterator >&,
	const std::vector<const FLOAT_TYPE*>& deriv_ptrs,
	const beacls::IntegerVec&,
	const beacls::IntegerVec& deriv_sizes,
	const helperOC::DynSys_DMode_Type dMode
) const {
	const helperOC::DynSys_DMode_Type modified_dMode = (dMode == helperOC::DynSys_DMode_Default) ? helperOC::DynSys_DMode_Min : dMode;
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
			case helperOC::DynSys_DMode_Max:
				std::transform(deriv, deriv + length, dOpt.begin(), [dMax_d](const auto& rhs) { return (rhs >= 0) ? dMax_d : -dMax_d;  });
				break;
			case helperOC::DynSys_DMode_Min:
				std::transform(deriv, deriv + length, dOpt.begin(), [dMax_d](const auto& rhs) { return (rhs >= 0) ? -dMax_d : dMax_d;  });
				break;
			case helperOC::DynSys_UMode_Invalid:
			default:
				std::cerr << "Unknown dMode!: " << modified_dMode << std::endl;
				return false;
			}
		}
	}
	return true;
}

bool STS::dynamics_cell_helper(
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
		std::cerr << "Only dimension 1-4 are defined for dynamics of STS!" << std::endl;
		result = false;
		break;
	}
	return result;
}
bool STS::dynamics(
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
bool STS::optCtrl_cuda(
	std::vector<beacls::UVec>& u_uvecs,
	const FLOAT_TYPE,
	const std::vector<beacls::UVec>&,
	const std::vector<beacls::UVec>& deriv_uvecs,
	const helperOC::DynSys_UMode_Type uMode
) const {
	const helperOC::DynSys_UMode_Type modified_uMode = (uMode == helperOC::DynSys_UMode_Default) ? helperOC::DynSys_UMode_Max : uMode;
	const size_t src_target_dim_index = find_val(dims, 2);
	if (src_target_dim_index == dims.size()) return false;
	if (deriv_uvecs.empty() || deriv_uvecs[src_target_dim_index].empty()) return false;
	u_uvecs.resize(get_nu());
	return STS_CUDA::optCtrl_execute_cuda(u_uvecs[0], deriv_uvecs[src_target_dim_index], wMax, modified_uMode);
}
bool STS::optDstb_cuda(
	std::vector<beacls::UVec>& d_uvecs,
	const FLOAT_TYPE,
	const std::vector<beacls::UVec>&,
	const std::vector<beacls::UVec>& deriv_uvecs,
	const helperOC::DynSys_DMode_Type dMode
) const {
	const helperOC::DynSys_DMode_Type modified_dMode = (dMode == helperOC::DynSys_DMode_Default) ? helperOC::DynSys_DMode_Min : dMode;
	const size_t src_target_dim_index = find_val(dims, 2);
	if (src_target_dim_index == dims.size()) return false;
	if (deriv_uvecs.empty() || deriv_uvecs[src_target_dim_index].empty()) return false;
	d_uvecs.resize(get_nd());
	return STS_CUDA::optDstb_execute_cuda(d_uvecs, deriv_uvecs[src_target_dim_index], dMax, modified_dMode, dims);
}

bool STS::dynamics_cuda(
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
			result &= STS_CUDA::dynamics_cell_helper_execute_cuda_dimAll(
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
				result &= STS_CUDA::dynamics_cell_helper_execute_cuda(
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
			STS_CUDA::dynamics_cell_helper_execute_cuda(dx_uvecs, x_uvecs, u_uvecs, modified_d_uvecs, speed, dst_target_dim, src_target_dim, src_x_dim);
		else {
			std::cerr << "Invalid target dimension for dynamics: " << dst_target_dim << std::endl;
			result = false;
		}
	}
	return result;
}
#endif /* defined(USER_DEFINED_GPU_DYNSYS_FUNC) */
