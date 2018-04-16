#include <helperOC/DynSys/Plane4D/Plane4D.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <typeinfo>
#include <levelset/Grids/HJI_Grid.hpp>
using namespace helperOC;


Plane4D::Plane4D(
    const beacls::FloatVec& x,
    const FLOAT_TYPE wMax,
    const beacls::FloatVec& aRange,
    const beacls::FloatVec& dMax,
    const beacls::IntegerVec& dims): 
    DynSys(dims.size(), 2, 2,
    beacls::IntegerVec{0, 1}, //!< Position dimensions
    beacls::IntegerVec{2},  //!< Heading dimensions
    beacls::IntegerVec{3}),  //!< velocity dimensions
    wMax(wMax), aRange(aRange), dMax(dMax), dims(dims) {

  if (x.size() != DynSys::get_nx()) {
    std::cerr << "Error: " << __func__ << 
      " : Initial state does not have right dimension!" << std::endl;
  }

  DynSys::set_x(x);
  DynSys::push_back_xhist(x);
}

Plane4D::Plane4D(
    beacls::MatFStream* fs,
    beacls::MatVariable* variable_ptr):
    DynSys(fs, variable_ptr),
    wMax(0),
    aRange(beacls::FloatVec()),
    dMax(beacls::FloatVec()),
    dims(beacls::IntegerVec()) {

  beacls::IntegerVec dummy;
  load_value(wMax, std::string("wMax"), true, fs, variable_ptr);
  load_vector(aRange, std::string("aRange"), dummy, true, fs, variable_ptr);
  load_vector(dMax, std::string("dMax"), dummy, true, fs, variable_ptr);
  load_vector(dims, std::string("dims"), dummy, true, fs, variable_ptr);
}

Plane4D::~Plane4D() {
}

bool Plane4D::operator==(const Plane4D& rhs) const {
  if (this == &rhs) return true;
  else if (!DynSys::operator==(rhs)) return false;
  else if (wMax != rhs.wMax) return false;  //!< Angular control bounds
  else if ((aRange.size() != rhs.aRange.size()) || 
      !std::equal(aRange.cbegin(), aRange.cend(), rhs.aRange.cbegin())) {
    return false; //!< Acceleration control bounds
  }
    
  else if ((dMax.size() != rhs.dMax.size()) || 
      !std::equal(dMax.cbegin(), dMax.cend(), rhs.dMax.cbegin())) {
    return false; //!< Disturbance
  }
      
  else if ((dims.size() != rhs.dims.size()) || 
      !std::equal(dims.cbegin(), dims.cend(), rhs.dims.cbegin())) {
    return false; //!< Dimensions that are active
  }
      
  return true;
}

bool Plane4D::operator==(const DynSys& rhs) const {
  if (this == &rhs) return true;
  else if (typeid(*this) != typeid(rhs)) return false;
  else return operator==(dynamic_cast<const Plane4D&>(rhs));
}

bool Plane4D::save(
    beacls::MatFStream* fs,
    beacls::MatVariable* variable_ptr) {
  bool result = DynSys::save( fs, variable_ptr);
  result &= save_value(wMax, std::string("wMax"), true, fs, variable_ptr);
  if (!aRange.empty()) {
    result &= save_vector(aRange, std::string("aRange"), beacls::IntegerVec(), 
      true, fs, variable_ptr);
  }

  if (!dMax.empty()) {
    result &= save_vector(dMax, std::string("dMax"), beacls::IntegerVec(), 
      true, fs, variable_ptr);
  }

  if (!dims.empty()) {
    result &= save_vector(dims, std::string("dims"), beacls::IntegerVec(), 
      true, fs, variable_ptr);
  }
  return result;
}

bool Plane4D::optCtrl0_cell_helper(
    beacls::FloatVec& uOpt0,
    const std::vector<const FLOAT_TYPE*>& derivs,
    const beacls::IntegerVec& deriv_sizes,
    const helperOC::DynSys_UMode_Type uMode,
    const size_t src_target_dim_index) const {

  if (src_target_dim_index < dims.size()) {
    const FLOAT_TYPE* deriv = derivs[src_target_dim_index];
    const size_t length = deriv_sizes[src_target_dim_index];

    if (length == 0 || deriv == NULL) return false;

    uOpt0.resize(length);
    switch (uMode) {
      case helperOC::DynSys_UMode_Max:
        for (size_t i = 0; i < length; ++i) {
          uOpt0[i] = (deriv[i] >= 0) ? wMax : -wMax;
        }
        break;

      case helperOC::DynSys_UMode_Min:
        for (size_t i = 0; i < length; ++i) {
          uOpt0[i] = (deriv[i] >= 0) ? -wMax : wMax;
        }
        break;

      case helperOC::DynSys_UMode_Invalid:

      default:
        std::cerr << "Unknown uMode!: " << uMode << std::endl;
        return false;
    }
  }
  return true;
} 

bool Plane4D::optCtrl1_cell_helper(
    beacls::FloatVec& uOpt1,
    const std::vector<const FLOAT_TYPE* >& derivs,
    const beacls::IntegerVec& deriv_sizes,
    const helperOC::DynSys_UMode_Type uMode,
    const size_t src_target_dim_index) const {

  if (src_target_dim_index < dims.size()) {
    const FLOAT_TYPE* deriv = derivs[src_target_dim_index];
    const size_t length = deriv_sizes[src_target_dim_index];

    if (length == 0 || deriv == NULL) return false;

    uOpt1.resize(length);
    switch (uMode) {
      case helperOC::DynSys_UMode_Max:
        for (size_t i = 0; i < length; ++i) {
          uOpt1[i] = (deriv[i] >= 0) ? aRange[1] : aRange[0];
        }
        break;

      case helperOC::DynSys_UMode_Min:
        for (size_t i = 0; i < length; ++i) {
          uOpt1[i] = (deriv[i] >= 0) ? aRange[0] : aRange[1];
        }
        break;

      case helperOC::DynSys_UMode_Invalid:

      default:
        std::cerr << "Unknown uMode!: " << uMode << std::endl;
        return false;
    }
  }
  return true;
}

bool Plane4D::optDstb0_cell_helper(
    beacls::FloatVec& dOpt0,
    const std::vector<const FLOAT_TYPE* >& derivs,
    const beacls::IntegerVec& deriv_sizes,
    const helperOC::DynSys_DMode_Type dMode,
    const size_t src_target_dim_index) const {

  if (src_target_dim_index != dims.size()) {
    const FLOAT_TYPE* deriv = derivs[src_target_dim_index];
    const size_t length = deriv_sizes[src_target_dim_index];

    if (length == 0 || deriv == NULL) return false;

    dOpt0.resize(length);
    const FLOAT_TYPE dMax0 = dMax[0];
    switch (dMode) {
      case helperOC::DynSys_DMode_Max:
        for (size_t i = 0; i < length; ++i) {
          dOpt0[i] = (deriv[i] >= 0) ? dMax0 : -dMax0;
        }
        break;

      case helperOC::DynSys_DMode_Min:
        for (size_t i = 0; i < length; ++i) {
          dOpt0[i] = (deriv[i] >= 0) ? -dMax0 : dMax0;
        }
        break;

      case helperOC::DynSys_UMode_Invalid:

      default:
        std::cerr << "Unknown uMode!: " << dMode << std::endl;
        return false;
    }
  }
  return true;

}
bool Plane4D::optDstb1_cell_helper(
    beacls::FloatVec& dOpt1,
    const std::vector<const FLOAT_TYPE* >& derivs,
    const beacls::IntegerVec& deriv_sizes,
    const helperOC::DynSys_DMode_Type dMode,
    const size_t src_target_dim_index) const {

  if (src_target_dim_index != dims.size()) {
    const FLOAT_TYPE* deriv = derivs[src_target_dim_index];
    const size_t length = deriv_sizes[src_target_dim_index];

    if (length == 0 || deriv == NULL) return false;

    dOpt1.resize(length);
    const FLOAT_TYPE dMax1 = dMax[1];
    switch (dMode) {
      case helperOC::DynSys_DMode_Max:
        for (size_t i = 0; i < length; ++i) {
          dOpt1[i] = (deriv[i] >= 0) ? dMax1 : -dMax1;
        }
        break;

      case helperOC::DynSys_DMode_Min:
        for (size_t i = 0; i < length; ++i) {
          dOpt1[i] = (deriv[i] >= 0) ? -dMax1 : dMax1;
        }
        break;

      case helperOC::DynSys_UMode_Invalid:

      default:
        std::cerr << "Unknown uMode!: " << dMode << std::endl;
        return false;
    }
  }
  return true;
}

bool Plane4D::optCtrl(
    std::vector<beacls::FloatVec>& uOpts,
    const FLOAT_TYPE,
    const std::vector<beacls::FloatVec::const_iterator>&,
    const std::vector<const FLOAT_TYPE*>& deriv_ptrs,
    const beacls::IntegerVec&,
    const beacls::IntegerVec& deriv_sizes,
    const helperOC::DynSys_UMode_Type uMode) const {

  const helperOC::DynSys_UMode_Type modified_uMode = 
    (uMode == helperOC::DynSys_UMode_Default) ? 
    helperOC::DynSys_UMode_Max : uMode;
  const size_t src_target_dim2_index = find_val(dims, 2);
  const size_t src_target_dim3_index = find_val(dims, 3);

  uOpts.resize(get_nu());

  bool result = true;
  result &= optCtrl0_cell_helper(uOpts[0], deriv_ptrs, deriv_sizes, 
    modified_uMode, src_target_dim2_index);
  result &= optCtrl1_cell_helper(uOpts[1], deriv_ptrs, deriv_sizes, 
    modified_uMode, src_target_dim3_index);

  return result;
}

bool Plane4D::optDstb(
    std::vector<beacls::FloatVec>& dOpts,
    const FLOAT_TYPE,
    const std::vector<beacls::FloatVec::const_iterator >&,
    const std::vector<const FLOAT_TYPE*>& deriv_ptrs,
    const beacls::IntegerVec&,
    const beacls::IntegerVec& deriv_sizes,
    const helperOC::DynSys_DMode_Type dMode) const {

  const helperOC::DynSys_DMode_Type modified_dMode = 
    (dMode == helperOC::DynSys_DMode_Default) ? 
    helperOC::DynSys_DMode_Min : dMode;
  const size_t src_target_dim0_index = find_val(dims, 0);
  const size_t src_target_dim1_index = find_val(dims, 1);

  dOpts.resize(get_nd());
  bool result = true;
  result &= optDstb0_cell_helper(dOpts[0], deriv_ptrs, deriv_sizes, 
    modified_dMode, src_target_dim0_index);
  result &= optDstb1_cell_helper(dOpts[1], deriv_ptrs, deriv_sizes, 
    modified_dMode, src_target_dim1_index);

  return result;
}
bool Plane4D::dynamics_cell_helper(
    std::vector<beacls::FloatVec>& dxs,
    const beacls::FloatVec::const_iterator& x_ites2,
    const beacls::FloatVec::const_iterator& x_ites3,
    const std::vector<beacls::FloatVec>& us,
    const std::vector<beacls::FloatVec>& ds,
    const size_t x2_size,
    const size_t,
    const size_t dim) const {

  beacls::FloatVec& dx_i = dxs[dim];
  bool result = true;
  
  switch (dims[dim]) { // states are (xpos, ypos, heading, velocity)
      case 0: { // \dot x0 = x3 * cos (x2)
        dx_i.assign(x2_size, 10.);
        // dx_i.resize(x2_size);

        // const beacls::FloatVec& ds_0 = ds[0];
        // for (size_t index = 0; index < x2_size; ++index) {
        //   dx_i[index] = x_ites3[index]*std::cos(x_ites2[index]) + ds_0[index];
        // }
      }
      break;

    case 1: { // \dot x0 = x3 * sin (x2)
        // dx_i.assign(x2_size, 10.);
        dx_i.resize(x2_size);
        const beacls::FloatVec& ds_1 = ds[1];
        for (size_t index = 0; index < x2_size; ++index) {
          dx_i[index] = x_ites3[index]*std::sin(x_ites2[index]) + ds_1[index];
        }
      }
      break;

    case 2: // \dot x3 = u0
      // dx_i.assign(x2_size, 10.);
      dx_i.resize(us[0].size());
      std::copy(us[0].cbegin(), us[0].cend(), dx_i.begin());
      break;

    case 3: // \dot x4 = u1
      // dx_i.assign(x2_size, 10.);
      dx_i.resize(us[1].size());
      std::copy(us[1].cbegin(), us[1].cend(), dx_i.begin());
      break;

    default: {
      std::cerr << "Only dimension 1-4 are defined for dynamics of Plane4D!" 
      << std::endl;
      result = false;
    }

      break;
  }


  return result;
}
bool Plane4D::dynamics(
    std::vector<beacls::FloatVec>& dx,
    const FLOAT_TYPE,
    const std::vector<beacls::FloatVec::const_iterator >& x_ites,
    const std::vector<beacls::FloatVec >& us,
    const std::vector<beacls::FloatVec >& ds,
    const beacls::IntegerVec& x_sizes,
    const size_t dst_target_dim) const {

  // printf("x_size = (%zd, %zd, %zd, %zd)\n", x_sizes[0], x_sizes[1], x_sizes[2], x_sizes[3]);
  const size_t src_target_dim2_index = find_val(dims, 2);
  const size_t src_target_dim3_index = find_val(dims, 3);

  if ((src_target_dim2_index == dims.size()) 
    || (src_target_dim3_index == dims.size())) {
    return false;
  }
  
  beacls::FloatVec::const_iterator x_ites2 = x_ites[src_target_dim2_index];
  beacls::FloatVec::const_iterator x_ites3 = x_ites[src_target_dim3_index];
  bool result = true;
  if (dst_target_dim == std::numeric_limits<size_t>::max()) {
    for (size_t dim = 0; dim < dims.size(); ++dim) {
      result &= dynamics_cell_helper(dx, x_ites2, x_ites3, us, ds, 
        x_sizes[src_target_dim2_index], x_sizes[src_target_dim3_index], dim);
    }
  }
  else {
    if (dst_target_dim < dims.size())
      result &= dynamics_cell_helper(dx, x_ites2, x_ites3, us, ds, 
        x_sizes[src_target_dim2_index], x_sizes[src_target_dim3_index], 
        dst_target_dim);
    else {
      std::cerr << "Invalid target dimension for dynamics: " << dst_target_dim 
        << std::endl;
      result = false;
    }
  }
  return result;
}