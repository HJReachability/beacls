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
	/* 
	To Be filled 
	*/
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
	/* 
	To Be filled 
	*/
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
			/* 
			To Be filled 
			*/
		}
		break;
	case 1:
		{
			/* 
			To Be filled 
			*/
		}
		break;
	case 2:
		{
			/* 
			To Be filled 
			*/
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
