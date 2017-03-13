#include <vector>
#include <cstdint>
#include <cmath>
#include <typeinfo>
#include <levelset/SpatialDerivative/UpwindFirst/UpwindFirstWENO5.hpp>
#include <levelset/SpatialDerivative/UpwindFirst/UpwindFirstWENO5a.hpp>
#include "UpwindFirstWENO5_impl.hpp"

UpwindFirstWENO5_impl::UpwindFirstWENO5_impl(
	const HJI_Grid *hji_grid,
	const beacls::UVecType type
) :
	type(type)
{
	upwindFirstWENO5a = new UpwindFirstWENO5a(hji_grid, type);
}
UpwindFirstWENO5_impl::~UpwindFirstWENO5_impl() {
	if (upwindFirstWENO5a) delete upwindFirstWENO5a;
}
UpwindFirstWENO5_impl::UpwindFirstWENO5_impl(const UpwindFirstWENO5_impl& rhs) :
	type(rhs.type),
	upwindFirstWENO5a(rhs.upwindFirstWENO5a->clone())
{}
bool UpwindFirstWENO5_impl::operator==(const UpwindFirstWENO5_impl& rhs) const {
	if (this == &rhs) return true;
	else if (type != rhs.type) return false;
	else if (!upwindFirstWENO5a->operator==(*rhs.upwindFirstWENO5a)) return false;
	else if ((upwindFirstWENO5a != rhs.upwindFirstWENO5a) && (!upwindFirstWENO5a || !rhs.upwindFirstWENO5a || !upwindFirstWENO5a->operator==(*rhs.upwindFirstWENO5a))) return false;
	else return true;
}


bool UpwindFirstWENO5_impl::execute(
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
	if (upwindFirstWENO5a) return upwindFirstWENO5a->execute(dst_deriv_l, dst_deriv_r, grid, src, dim, generateAll, loop_begin, slice_length, num_of_slices);
	else return false;
}
UpwindFirstWENO5::UpwindFirstWENO5(
	const HJI_Grid *hji_grid,
	const beacls::UVecType type
	) {
	pimpl = new UpwindFirstWENO5_impl(hji_grid,type);
}
UpwindFirstWENO5::~UpwindFirstWENO5() {
	if (pimpl) delete pimpl;
}
bool UpwindFirstWENO5::execute(
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
bool UpwindFirstWENO5_impl::synchronize(const size_t dim) {
	if (upwindFirstWENO5a) return upwindFirstWENO5a->synchronize(dim);
	return false;
}
bool UpwindFirstWENO5::synchronize(const size_t dim) {
	if (pimpl) return pimpl->synchronize(dim);
	else return false;
}


bool UpwindFirstWENO5::operator==(const UpwindFirstWENO5& rhs) const {
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
bool UpwindFirstWENO5::operator==(const SpatialDerivative& rhs) const {
	if (this == &rhs) return true;
	else if (typeid(*this) != typeid(rhs)) return false;
	else return operator==(dynamic_cast<const UpwindFirstWENO5&>(rhs));
}

UpwindFirstWENO5::UpwindFirstWENO5(const UpwindFirstWENO5& rhs) :
	pimpl(rhs.pimpl->clone())
{
}

UpwindFirstWENO5* UpwindFirstWENO5::clone() const {
	return new UpwindFirstWENO5(*this);
}
beacls::UVecType UpwindFirstWENO5::get_type() const {
	if (pimpl) return pimpl->get_type();
	else return beacls::UVecType_Invalid;
};
