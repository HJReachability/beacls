#include <vector>
#include <cstdint>
#include <cmath>
#include <typeinfo>
#include <levelset/SpatialDerivative/UpwindFirst/UpwindFirstENO3.hpp>
#include "UpwindFirstENO3_impl.hpp"
#include <levelset/SpatialDerivative/UpwindFirst/UpwindFirstENO3a.hpp>
using namespace levelset;

UpwindFirstENO3_impl::UpwindFirstENO3_impl(
	const HJI_Grid *hji_grid,
	const beacls::UVecType type
) :
	type(type)
{
	upwindFirstENO3a = new UpwindFirstENO3a(hji_grid, type);
}
UpwindFirstENO3_impl::~UpwindFirstENO3_impl() {
	if (upwindFirstENO3a) delete upwindFirstENO3a;
}
UpwindFirstENO3_impl::UpwindFirstENO3_impl(const UpwindFirstENO3_impl& rhs) :
	type(rhs.type),
	upwindFirstENO3a(rhs.upwindFirstENO3a->clone()) {}
bool UpwindFirstENO3_impl::operator==(const UpwindFirstENO3_impl& rhs) const {
	if (this == &rhs) return true;
	else if (type != rhs.type) return false;
	else if ((upwindFirstENO3a != rhs.upwindFirstENO3a) && (!upwindFirstENO3a || !rhs.upwindFirstENO3a || !upwindFirstENO3a->operator==(*rhs.upwindFirstENO3a))) return false;
	else return true;
}

bool UpwindFirstENO3_impl::execute(
	beacls::UVec& dst_deriv_l,
	beacls::UVec& dst_deriv_r,
	const HJI_Grid *grid,
	const FLOAT_TYPE* src,
	const size_t dim,
	const bool generateAll,
	const size_t loop_begin,
	const size_t slice_length,
	const size_t num_of_slices
) {
	if (upwindFirstENO3a) return upwindFirstENO3a->execute(dst_deriv_l, dst_deriv_r, grid, src, dim, generateAll, loop_begin, slice_length, num_of_slices);
	else return false;
}
bool UpwindFirstENO3_impl::execute_local_q(
	beacls::UVec& dst_deriv_l,
	beacls::UVec& dst_deriv_r,
	const HJI_Grid *grid,
	const FLOAT_TYPE* src,
	const size_t dim,
	const bool generateAll,
	const size_t loop_begin,
	const size_t slice_length,
	const size_t num_of_slices, 
	const std::set<size_t> &Q 
) {
	if (upwindFirstENO3a) return upwindFirstENO3a->execute_local_q(dst_deriv_l, dst_deriv_r, grid, src, dim, generateAll, loop_begin, slice_length, num_of_slices, Q);
	else return false;
}
UpwindFirstENO3::UpwindFirstENO3(
	const HJI_Grid *hji_grid,
	const beacls::UVecType type
	) {
	pimpl = new UpwindFirstENO3_impl(hji_grid,type);
}
UpwindFirstENO3::~UpwindFirstENO3() {
	if (pimpl) delete pimpl;
}

bool UpwindFirstENO3::execute(
	beacls::UVec& dst_deriv_l,
	beacls::UVec& dst_deriv_r,
	const HJI_Grid *grid,
	const FLOAT_TYPE* src,
	const size_t dim,
	const bool generateAll,
	const size_t loop_begin,
	const size_t slice_length,
	const size_t num_of_slices
) {
	if (pimpl) return pimpl->execute(dst_deriv_l, dst_deriv_r, grid, src, dim, generateAll, loop_begin, slice_length, num_of_slices);
	else return false;
}
bool UpwindFirstENO3::execute_local_q(
	beacls::UVec& dst_deriv_l,
	beacls::UVec& dst_deriv_r,
	const HJI_Grid *grid,
	const FLOAT_TYPE* src,
	const size_t dim,
	const bool generateAll,
	const size_t loop_begin,
	const size_t slice_length,
	const size_t num_of_slices, 
	const std::set<size_t> &Q 
) {
	if (pimpl) return pimpl->execute_local_q(dst_deriv_l, dst_deriv_r, grid, src, dim, generateAll, loop_begin, slice_length, num_of_slices, Q);
	else return false;
}
bool UpwindFirstENO3_impl::synchronize(const size_t dim) {
	if (upwindFirstENO3a) return upwindFirstENO3a->synchronize(dim);
	return false;
}
bool UpwindFirstENO3::synchronize(const size_t dim) {
	if (pimpl) return pimpl->synchronize(dim);
	else return false;
}

bool UpwindFirstENO3::operator==(const UpwindFirstENO3& rhs) const {
	if (this == &rhs) return true;
	else if (!pimpl) {
		if (!rhs.pimpl) return true;
		else return false;
	}
	else {
		if (!rhs.pimpl) return false;
		else {
			if (pimpl == rhs.pimpl) return true;
			else if (*pimpl == *rhs.pimpl) return true;
			else return false;
		}
	}
}
bool UpwindFirstENO3::operator==(const SpatialDerivative& rhs) const {
	if (this == &rhs) return true;
	else if (typeid(*this) != typeid(rhs)) return false;
	else return operator==(dynamic_cast<const UpwindFirstENO3&>(rhs));
}

UpwindFirstENO3::UpwindFirstENO3(const UpwindFirstENO3& rhs) :
	pimpl(rhs.pimpl->clone())
{
}

UpwindFirstENO3* UpwindFirstENO3::clone() const {
	return new UpwindFirstENO3(*this);
}
beacls::UVecType UpwindFirstENO3::get_type() const {
	if (pimpl) return pimpl->get_type();
	else return beacls::UVecType_Invalid;
};
