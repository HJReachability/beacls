/*
 * Copyright (c) 2017, The Regents of the University of California (Regents).
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *    1. Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *    2. Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 *
 *    3. Neither the name of the copyright holder nor the names of its
 *       contributors may be used to endorse or promote products derived
 *       from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS AS IS
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * Please contact the author(s) of this library if you have any questions.
 * Authors: Jaime F. Fisac   ( jfisac@eecs.berkeley.edu )
 */

///////////////////////////////////////////////////////////////////////////////
//
// DynSys subclass: relative dynamics between 5D dynamic Dubins car (tracker)
// and 3D Dubins car (planner).
//
///////////////////////////////////////////////////////////////////////////////


#include <helperOC/DynSys/P5D_Dubins/P5D_Dubins.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <typeinfo>
#include <levelset/Grids/HJI_Grid.hpp>
using namespace helperOC;


P5D_Dubins::P5D_Dubins(
    const beacls::FloatVec& x,      // state of tracker (relative to planner)
    const beacls::FloatVec& aRange, // tracker's linear acceleration bounds
    const FLOAT_TYPE alphaMax,      // tracker's maximum angular acceleration
    const FLOAT_TYPE vOther,        // planner's speed
    const FLOAT_TYPE wMax,          // planner's maximum angular velocity
    const beacls::FloatVec& dMax,   // maximum disturbance on tracker's dynamics
    const beacls::IntegerVec& dims
) : DynSys(5, // states: [x_rel, y_rel, theta_rel, v, w]
           2, // controls: [a, alpha] (linear and angular acceleration)
           6  // disturbances: all state components + planner w input
        // beacls::IntegerVec{0, 1},  //!< Position dimensions
        // beacls::IntegerVec{2},  //!< Heading dimensions
    ),
    aRange(aRange), alphaMax(alphaMax), vOther(vOther),
    wMax(wMax), dMax(dMax), dims(dims) {
    std::cout << "Constructed P5D_Dubins object with:" << std::endl;
    std::cout << "  " << "N_states: " << x.size() << std::endl;
    std::cout << "  " << "N_disturbances: " << dMax.size() << std::endl;
    //!< Process control range
    if (x.size() != DynSys::get_nx()) {
      std::cerr << "Error: " << __func__ <<
        " : Initial state does not have right dimension!" << std::endl;
    }
    //!< Process initial state
    DynSys::set_x(x);
    DynSys::push_back_xhist(x);
}

P5D_Dubins::P5D_Dubins(
    beacls::MatFStream* fs,
    beacls::MatVariable* variable_ptr
) :
    DynSys(fs, variable_ptr),
    aRange(beacls::FloatVec()),
    alphaMax(0),
    vOther(0),
    wMax(0),
    dMax(beacls::FloatVec()),
    dims(beacls::IntegerVec()) {
    beacls::IntegerVec dummy;
    load_vector(aRange, std::string("aRange"), dummy, true, fs, variable_ptr);  
    load_value(alphaMax, std::string("alphaMax"), true, fs, variable_ptr);  
    load_value(vOther, std::string("vOther"), true, fs, variable_ptr);  
    load_value(wMax, std::string("wMax"), true, fs, variable_ptr);  
    load_vector(dMax, std::string("dMax"), dummy, true, fs, variable_ptr);  
    load_vector(dims, std::string("dims"), dummy, true, fs, variable_ptr);
}


P5D_Dubins::~P5D_Dubins() {}


bool P5D_Dubins::operator==(const P5D_Dubins& rhs) const {
  if (this == &rhs) return true;
  else if (!DynSys::operator==(rhs)) return false;
  else if ((aRange.size() != rhs.aRange.size()) ||
    !std::equal(aRange.cbegin(), aRange.cend(), rhs.aRange.cbegin()))
      return false;
  else if (alphaMax != rhs.alphaMax) return false;   
  else if (vOther != rhs.vOther) return false;    
  else if (wMax != rhs.wMax) return false;   
  else if ((dMax.size() != rhs.dMax.size()) || !std::equal(dMax.cbegin(),
    dMax.cend(), rhs.dMax.cbegin())) return false;
  else if ((dims.size() != rhs.dims.size()) || !std::equal(dims.cbegin(),
    dims.cend(), rhs.dims.cbegin())) return false;
  return true;
}


bool P5D_Dubins::operator==(const DynSys& rhs) const {
    if (this == &rhs) return true;
    else if (typeid(*this) != typeid(rhs)) return false;
    else return operator==(dynamic_cast<const P5D_Dubins&>(rhs));
}


bool P5D_Dubins::save(
    beacls::MatFStream* fs,
    beacls::MatVariable* variable_ptr) {

  bool result = DynSys::save(fs, variable_ptr);
  if (!aRange.empty()) result &= save_vector(aRange, std::string("aRange"), 
    beacls::IntegerVec(), true, fs, variable_ptr);
  result &=
    save_value(alphaMax, std::string("alphaMax"), true, fs, variable_ptr);
  result &= save_value(vOther, std::string("vOther"), true, fs, variable_ptr);
  result &= save_value(wMax, std::string("wMax"), true, fs, variable_ptr);
  if (!dMax.empty()) result &= save_vector(dMax, std::string("dMax"), 
    beacls::IntegerVec(), true, fs, variable_ptr);
  if (!dims.empty()) result &= save_vector(dims, std::string("dims"), 
    beacls::IntegerVec(), true, fs, variable_ptr);
  return result;
}


bool P5D_Dubins::optCtrl_i_cell_helper(
    beacls::FloatVec& uOpt_i, // Relevant component i of the optimal input
    const std::vector<const FLOAT_TYPE*>& derivs,
    const beacls::IntegerVec& deriv_sizes,
    const helperOC::DynSys_UMode_Type uMode,
    const size_t src_target_dim_index, // Relevant state j affected by input i
    const beacls::FloatVec& uExtr_i // [u_minimizer_xj_dot, u_maximizer_xj_dot]
    ) const {

  if (src_target_dim_index < dims.size()) {
    const FLOAT_TYPE* deriv_j = derivs[src_target_dim_index];
    const size_t length = deriv_sizes[src_target_dim_index];

    if (length == 0 || deriv_j == NULL) return false;

    uOpt_i.resize(length);
    
    switch (uMode) {
      case helperOC::DynSys_UMode_Max:
        for (size_t ii = 0; ii < length; ++ii) { // iterate over grid
          uOpt_i[ii] = (deriv_j[ii] >= 0) ? uExtr_i[1] : uExtr_i[0];
        }
        break;
     
      case helperOC::DynSys_UMode_Min:
        for (size_t ii = 0; ii < length; ++ii) {
          uOpt_i[ii] = (deriv_j[ii] >= 0) ? uExtr_i[0] : uExtr_i[1];
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


bool P5D_Dubins::optDstb_i_cell_helper(
    beacls::FloatVec& dOpt_i, // Relevant component i of the optimal input
    const std::vector<const FLOAT_TYPE*>& derivs,
    const beacls::IntegerVec& deriv_sizes,
    const helperOC::DynSys_DMode_Type dMode,
    const size_t src_target_dim_index, // Relevant state j affected by input i
    const beacls::FloatVec& dExtr_i // [u_minimizer_xj_dot, u_maximizer_xj_dot]
    ) const {
  if (src_target_dim_index < dims.size()) {
    const FLOAT_TYPE* deriv_j = derivs[src_target_dim_index];
    const size_t length = deriv_sizes[src_target_dim_index];

    if (length == 0 || deriv_j == NULL) return false;

    dOpt_i.resize(length);

    switch (dMode) {
      case helperOC::DynSys_DMode_Max:
        for (size_t ii = 0; ii < length; ++ii) { // iterate over grid
            dOpt_i[ii] = (deriv_j[ii] >= 0) ? dExtr_i[1] : dExtr_i[0];
        }
        break;
      case helperOC::DynSys_DMode_Min:
        for (size_t ii = 0; ii < length; ++ii) {
            dOpt_i[ii] = (deriv_j[ii] >= 0) ? dExtr_i[0] : dExtr_i[1];
        }
        break;
      case helperOC::DynSys_DMode_Invalid:
      default:
        std::cerr << "Unknown dMode!: " << dMode << std::endl;
        return false;
    }
  }
  return true;
} 


bool P5D_Dubins::optCtrl(
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

  uOpts.resize(get_nu());

  bool result = true;

  for (size_t i = 0; i < 2; ++i) {
    uOpts[i].assign(deriv_sizes[0], 0.);
  }  
  // // Call helper to determine optimal value for each control component
  // // (we feed the relevant state component affected by each input as well as
  // //  the input values that maximize and minimize this state's derivative).
  // result &= optCtrl_i_cell_helper(uOpts[0], deriv_ptrs, deriv_sizes,
  //     modified_uMode, 3, aRange);

  // const beacls::FloatVec& alphaRange{-alphaMax, alphaMax};
  // result &= optCtrl_i_cell_helper(uOpts[1], deriv_ptrs, deriv_sizes,
  //     modified_uMode, 4, alphaRange);
  // return result;
}


bool P5D_Dubins::optDstb(
    std::vector<beacls::FloatVec>& dOpts,
    const FLOAT_TYPE,
    const std::vector<beacls::FloatVec::const_iterator>& x_ites,
    const std::vector<const FLOAT_TYPE*>& deriv_ptrs,
    const beacls::IntegerVec& x_sizes,
    const beacls::IntegerVec& deriv_sizes,
    const helperOC::DynSys_DMode_Type dMode
    ) const {
  const helperOC::DynSys_DMode_Type modified_dMode =
    (dMode == helperOC::DynSys_DMode_Default) ?
    helperOC::DynSys_DMode_Min : dMode;

  if ((modified_dMode != helperOC::DynSys_DMode_Max) && 
      (modified_dMode != helperOC::DynSys_DMode_Min)) {
    std::cerr << "Unknown dMode!: " << modified_dMode << std::endl;
    return false;
  }

  dOpts.resize(get_nd());
  bool result = true;

  for (size_t i = 0; i < 6; ++i) {
    dOpts[i].assign(deriv_sizes[0], 0.);
  }

  // // Call helper to determine optimal value for each disturbance component
  // // (we feed the relevant state component affected by each input as well as
  // //  the input values that maximize and minimize this state's derivative).

  // // Disturbances
  // const beacls::FloatVec& dRange0{-dMax[0],dMax[0]};
  // const beacls::FloatVec& dRange1{-dMax[1],dMax[1]};
  // const beacls::FloatVec& dRange2{-dMax[2],dMax[2]};
  // const beacls::FloatVec& dRange3{-dMax[3],dMax[3]};
  // const beacls::FloatVec& dRange4{-dMax[4],dMax[4]};

  // result &= optDstb_i_cell_helper(dOpts[0], deriv_ptrs, deriv_sizes,
  //     modified_dMode, 0, dRange0);
  // result &= optDstb_i_cell_helper(dOpts[1], deriv_ptrs, deriv_sizes,
  //     modified_dMode, 1, dRange1);
  // result &= optDstb_i_cell_helper(dOpts[2], deriv_ptrs, deriv_sizes,
  //     modified_dMode, 2, dRange2);
  // result &= optDstb_i_cell_helper(dOpts[3], deriv_ptrs, deriv_sizes,
  //     modified_dMode, 3, dRange3);
  // result &= optDstb_i_cell_helper(dOpts[4], deriv_ptrs, deriv_sizes,
  //     modified_dMode, 4, dRange4);

  // Planning control
  // assume state is a matrix (not a single state)
  // const size_t x_size = x_sizes[0]; 
  
  // beacls::FloatVec& dOpt5 = dOpts[5];

  // beacls::FloatVec::const_iterator xs0 = x_ites[0];
  // beacls::FloatVec::const_iterator xs1 = x_ites[1];

  // const FLOAT_TYPE* deriv0_ptr = deriv_ptrs[0];
  // const FLOAT_TYPE* deriv1_ptr = deriv_ptrs[1];
  // const FLOAT_TYPE* deriv2_ptr = deriv_ptrs[2];

  // const FLOAT_TYPE w_if_det5_pos = 
  //   (modified_dMode == helperOC::DynSys_DMode_Max) ? wMax : -wMax;
  // const FLOAT_TYPE w_if_det5_neg =
  //   (modified_dMode == helperOC::DynSys_DMode_Max) ? -wMax : wMax;

  // for (size_t index = 0; index < x_size; ++index) {
  //   const FLOAT_TYPE x0 = xs0[index];
  //   const FLOAT_TYPE x1 = xs1[index];
  //   const FLOAT_TYPE deriv0 = deriv0_ptr[index];
  //   const FLOAT_TYPE deriv1 = deriv1_ptr[index];
  //   const FLOAT_TYPE deriv2 = deriv2_ptr[index];

  //   const FLOAT_TYPE det5 = deriv0*x1 - deriv1*x0 - deriv2;

  //   dOpt5[index] = (det5 >= 0) ? w_if_det5_pos : w_if_det5_neg;
  // }

  return result;
}

bool P5D_Dubins::dynamics_cell_helper(
    std::vector<beacls::FloatVec>& dxs,
    const beacls::FloatVec::const_iterator& state_x_rel,
    const beacls::FloatVec::const_iterator& state_y_rel,
    const beacls::FloatVec::const_iterator& state_theta_rel,
    const beacls::FloatVec::const_iterator& state_v,
    const beacls::FloatVec::const_iterator& state_w,
    const std::vector<beacls::FloatVec>& us,
    const std::vector<beacls::FloatVec>& ds,
    const size_t size_x_rel,
    const size_t size_y_rel,
    const size_t size_theta_rel,
    const size_t size_v,
    const size_t size_w,
    const size_t dim) const {

  beacls::FloatVec& dx_i = dxs[dim];
  bool result = true;
  switch (dims[dim]) {
  case 0: { // x_rel_dot = -vOther + v * cos(theta_rel) + wOther*y_rel + d_x_rel
    dx_i.assign(size_x_rel, 1.);
    // dx_i.resize(size_x_rel);
    // const beacls::FloatVec& d_x_rel = ds[0];
    // const beacls::FloatVec& wOther = ds[5];
    // for (size_t index = 0; index < size_x_rel; ++index) {
    //   dx_i[index] = -vOther +
    //     state_v[index] * std::cos(state_theta_rel[index]) +
    //     wOther[index] * state_y_rel[index] +
    //     d_x_rel[index];
    // } 
  }
      break;
  case 1: { // y_rel_dot = v * sin(theta_rel) - wOther*x_rel + d_y_rel
    dx_i.assign(size_y_rel, 1.);
    // dx_i.resize(size_y_rel);
    // const beacls::FloatVec& d_y_rel = ds[1];
    // const beacls::FloatVec& wOther = ds[5];
    // for (size_t index = 0; index < size_y_rel; ++index) {
    //   dx_i[index] =
    //     state_v[index] * std::sin(state_theta_rel[index]) -
    //     wOther[index] * state_x_rel[index] +
    //     d_y_rel[index];
    // }
  }
      break;
  case 2: {   // theta_rel_dot = w - wOther
      dx_i.assign(size_theta_rel, 1.);
      // dx_i.resize(size_theta_rel);
      // const beacls::FloatVec& d_theta_rel = ds[2];
      // const beacls::FloatVec& wOther = ds[5];
      // for (size_t index = 0; index < size_theta_rel; ++index) {
      //   dx_i[index] = state_w[index] - wOther[index] + d_theta_rel[index];
      // }
    }
    break;
  case 3: {   // v_dot = a + d_v
    dx_i.assign(size_v, 1.);
    // dx_i.resize(size_v);
    // const beacls::FloatVec& d_v = ds[3];
    // const beacls::FloatVec& a = us[0];
    // for (size_t index = 0; index < size_v; ++index) {
    //   dx_i[index] = a[index] + d_v[index];
    // }
    }
    break;
  case 4: { // w_dot = alpha + d_w
    dx_i.assign(size_w, 1.);
  //   dx_i.resize(size_w);
  //   const beacls::FloatVec& d_w = ds[4];
  //   const beacls::FloatVec& alpha = us[1];
  //   for (size_t index = 0; index < size_w; ++index) {
  //     dx_i[index] = alpha[index] + d_w[index];
  //   }
  }
  break;
  default:
    std::cerr <<
      "Only dimension 1-5 are defined for dynamics of P5D_Dubins!" <<
      std::endl;
    result = false;
    break;
  }
  return result;
}


bool P5D_Dubins::dynamics(
    std::vector<beacls::FloatVec>& dx,
    const FLOAT_TYPE,
    const std::vector<beacls::FloatVec::const_iterator>& x_ites,
    const std::vector<beacls::FloatVec>& us,
    const std::vector<beacls::FloatVec>& ds,
    const beacls::IntegerVec& x_sizes,
    const size_t dst_target_dim) const {

    // // Define indices and iterators for each state dimension.
    // const size_t src_target_dim0_index = find_val(dims, 0);
    // const size_t src_target_dim1_index = find_val(dims, 1);
    // const size_t src_target_dim2_index = find_val(dims, 2);
    // const size_t src_target_dim3_index = find_val(dims, 3);
    // const size_t src_target_dim4_index = find_val(dims, 4);
    // if ((src_target_dim0_index == dims.size()) ||
    //     (src_target_dim1_index == dims.size()) ||
    //     (src_target_dim2_index == dims.size()) ||
    //     (src_target_dim3_index == dims.size()) ||
    //     (src_target_dim4_index == dims.size())) {
    //   return false;  
    // }
    
    printf("x_ites size = %d\n", x_ites.size());
    const beacls::FloatVec::const_iterator& x_ites0 = x_ites[0];
    const beacls::FloatVec::const_iterator& x_ites1 = x_ites[1];
    const beacls::FloatVec::const_iterator& x_ites2 = x_ites[2];
    const beacls::FloatVec::const_iterator& x_ites3 = x_ites[3];
    const beacls::FloatVec::const_iterator& x_ites4 = x_ites[4];

    bool result = true;
    // Compute dynamics for all components.
    if (dst_target_dim == std::numeric_limits<size_t>::max()) {
      for (size_t dim = 0; dim < 5; ++dim) {
        result &= dynamics_cell_helper(
          dx, x_ites0, x_ites1, x_ites2,x_ites3, x_ites4, us, ds,
          x_sizes[0], x_sizes[1], x_sizes[2], x_sizes[3], x_sizes[4], dim);
        }
    }
    // Compute dynamics for a single, specified component.
    else
    {
        if (dst_target_dim < dims.size())
          result &= dynamics_cell_helper(
            dx, x_ites0, x_ites1, x_ites2,x_ites3, x_ites4, us, ds,
            x_sizes[0], x_sizes[1], x_sizes[2], x_sizes[3], x_sizes[4], 
            dst_target_dim);
        else {
            std::cerr << "Invalid target dimension for dynamics: " <<
                dst_target_dim << std::endl;
            result = false;
        }
    }
    return result;
}
