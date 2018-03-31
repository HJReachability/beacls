#include <levelset/Grids/HJI_Grid.hpp>
#include "HJI_Grid_impl.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <fstream>
#include <functional>
#include <cstdio>
#include <matio.h>
#include <sstream>
#include <macro.hpp>
#include <vector>
#include <deque>
#include <typeinfo>
#include <levelset/BoundaryCondition/BoundaryCondition.hpp>
#include <levelset/BoundaryCondition/AddGhostExtrapolate.hpp>
#include <levelset/BoundaryCondition/AddGhostPeriodic.hpp>
using namespace levelset;

template<typename T>
inline
matio_classes
get_matio_classes(T a = 0);
template<>
inline
matio_classes
get_matio_classes(float) {
	return MAT_C_SINGLE;
}
template<>
inline
matio_classes
get_matio_classes(double) {
	return MAT_C_DOUBLE;
}
template<>
inline
matio_classes
get_matio_classes(int8_t) {
	return MAT_C_INT8;
}
template<>
inline
matio_classes
get_matio_classes(uint8_t) {
	return MAT_C_UINT8;
}
template<>
inline
matio_classes
get_matio_classes(int16_t) {
	return MAT_C_INT16;
}
template<>
inline
matio_classes
get_matio_classes(uint16_t) {
	return MAT_C_UINT16;
}
template<>
inline
matio_classes
get_matio_classes(int32_t) {
	return MAT_C_INT32;
}
template<>
inline
matio_classes
get_matio_classes(uint32_t) {
	return MAT_C_UINT32;
}
template<>
inline
matio_classes
get_matio_classes(int64_t) {
	return MAT_C_INT64;
}
template<>
inline
matio_classes
get_matio_classes(uint64_t) {
	return MAT_C_UINT64;
}
#if defined(__clang__)
template<>
inline
matio_classes
get_matio_classes(size_t) {
	switch (sizeof(size_t)) {
	case 1:
		return MAT_C_UINT8;
	case 2:
		return MAT_C_UINT16;
	case 4:
		return MAT_C_UINT32;
	case 8:
		return MAT_C_UINT64;
	default:
		return MAT_C_EMPTY;
	}
}
#endif	/* __clang__ */
template<typename T>
inline
matio_types
get_matio_types(T a = 0);
template<>
inline
matio_types
get_matio_types(float) {
	return MAT_T_SINGLE;
}
template<>
inline
matio_types
get_matio_types(double) {
	return MAT_T_DOUBLE;
}
template<>
inline
matio_types
get_matio_types(int8_t) {
	return MAT_T_INT8;
}
template<>
inline
matio_types
get_matio_types(uint8_t) {
	return MAT_T_UINT8;
}
template<>
inline
matio_types
get_matio_types(int16_t) {
	return MAT_T_INT16;
}
template<>
inline
matio_types
get_matio_types(uint16_t) {
	return MAT_T_UINT16;
}
template<>
inline
matio_types
get_matio_types(int32_t) {
	return MAT_T_INT32;
}
template<>
inline
matio_types
get_matio_types(uint32_t) {
	return MAT_T_UINT32;
}
template<>
inline
matio_types
get_matio_types(int64_t) {
	return MAT_T_INT64;
}
template<>
inline
matio_types
get_matio_types(uint64_t) {
	return MAT_T_UINT64;
}
#if defined(__clang__)
template<>
inline
matio_types
get_matio_types(size_t) {
	switch (sizeof(size_t)) {
	case 1:
		return MAT_T_UINT8;
	case 2:
		return MAT_T_UINT16;
	case 4:
		return MAT_T_UINT32;
	case 8:
		return MAT_T_UINT64;
	default:
		return MAT_T_UNKNOWN;
	}
}
#endif	/* __clang__ */

namespace beacls {
	class MatFStream {
	private:
		mat_t* mat;
	public:
		MatFStream(mat_t* p);
		~MatFStream();
		mat_t* get_mat() const;
	private:
		/** @overload
		Disable operator=
		*/
		MatFStream& operator=(const MatFStream& rhs);
		/** @overload
		Disable copy constructor
		*/
		MatFStream(const MatFStream& rhs);
	};
	class MatVariable {
	private:
		matvar_t* matvar;
		bool should_be_free;
	public:
		MatVariable(matvar_t* p);
		MatVariable(matvar_t* p, const bool s);
		~MatVariable();
		matvar_t* get_matvar() const;
		bool should_it_be_free() const;
		void set_should_be_free(const bool v);
	private:
		/** @overload
		Disable operator=
		*/
		MatVariable& operator=(const MatVariable& rhs);
		/** @overload
		Disable copy constructor
		*/
		MatVariable(const MatVariable& rhs);
	};
};
beacls::MatFStream::MatFStream(mat_t* p) : mat(p) {
}
beacls::MatFStream::~MatFStream() {
}
mat_t* beacls::MatFStream::get_mat() const {
	return mat;
}
beacls::MatVariable::MatVariable(matvar_t* p) : matvar(p), should_be_free(true) {
}
beacls::MatVariable::MatVariable(matvar_t* p, const bool s) : matvar(p), should_be_free(s) {
}
beacls::MatVariable::~MatVariable() {
}
matvar_t* beacls::MatVariable::get_matvar() const {
	return matvar;
}
bool beacls::MatVariable::should_it_be_free() const {
	return should_be_free;
}
void beacls::MatVariable::set_should_be_free(const bool v) {
	should_be_free = v;
}


HJI_Grid_impl::HJI_Grid_impl(
) :
	num_of_dimensions(0),
	mins(std::vector<FLOAT_TYPE>()),
	maxs(std::vector<FLOAT_TYPE>()),
	boundaryConditions(std::vector<BoundaryCondition*>()),
	Ns(std::vector<size_t>()),
	dxs(std::vector<FLOAT_TYPE>()),
	dxInvs(std::vector<FLOAT_TYPE>()),
	vss(std::vector<std::vector<FLOAT_TYPE> >()),
	xss(std::vector<std::vector<FLOAT_TYPE> >()),
	axis(std::vector<FLOAT_TYPE>()),
	shape(std::vector<size_t>())
{
}

HJI_Grid_impl::HJI_Grid_impl(
	const size_t num_of_dimensions
) :
	num_of_dimensions(num_of_dimensions),
	mins(std::vector<FLOAT_TYPE>()),
	maxs(std::vector<FLOAT_TYPE>()),
	boundaryConditions(std::vector<BoundaryCondition*>()),
	Ns(std::vector<size_t>()),
	dxs(std::vector<FLOAT_TYPE>()),
	dxInvs(std::vector<FLOAT_TYPE>()),
	vss(std::vector<std::vector<FLOAT_TYPE> >()),
	xss(std::vector<std::vector<FLOAT_TYPE> >()),
	axis(std::vector<FLOAT_TYPE>()),
	shape(std::vector<size_t>())
{
}
HJI_Grid_impl::HJI_Grid_impl(const HJI_Grid_impl& rhs) :
	num_of_dimensions(rhs.num_of_dimensions),
	mins(rhs.mins),
	maxs(rhs.maxs),
	Ns(rhs.Ns),
	dxs(rhs.dxs),
	dxInvs(rhs.dxInvs),
	vss(rhs.vss),
	xss(rhs.xss),
	axis(rhs.axis),
	shape(rhs.shape)
{
	this->boundaryConditions.resize(rhs.boundaryConditions.size());
	std::transform(rhs.boundaryConditions.cbegin(), rhs.boundaryConditions.cend(), this->boundaryConditions.begin(), ([](auto &ptr) { return ptr->clone(); }));
}

HJI_Grid_impl::~HJI_Grid_impl() {
	for_each(boundaryConditions.cbegin(), boundaryConditions.cend(), ([](auto &ptr) {if (ptr)delete ptr; }));
}

bool HJI_Grid_impl::operator==(const HJI_Grid_impl& rhs) const {
	if (this == &rhs) return true;
	else if (num_of_dimensions != rhs.num_of_dimensions) return false;
	else if ((mins.size() != rhs.mins.size()) || !std::equal(mins.cbegin(), mins.cend(), rhs.mins.cbegin())) return false;
	else if ((maxs.size() != rhs.maxs.size()) || !std::equal(maxs.cbegin(), maxs.cend(), rhs.maxs.cbegin())) return false;
	else if ((boundaryConditions.size() != rhs.boundaryConditions.size()) 
		|| !std::equal(boundaryConditions.cbegin(), boundaryConditions.cend(), rhs.boundaryConditions.cbegin(), [](const auto& lhs, const auto& rhs) {
		if (lhs == rhs) return true;
		else if (!lhs) return false;
		else if (!rhs) return false;
		else if (typeid(*lhs) == typeid(*rhs)) return true;
		else return false;
	})) return false;
	else if ((Ns.size() != rhs.Ns.size()) || !std::equal(Ns.cbegin(), Ns.cend(), rhs.Ns.cbegin())) return false;
	else if ((dxs.size() != rhs.dxs.size()) || !std::equal(dxs.cbegin(), dxs.cend(), rhs.dxs.cbegin())) return false;
	else if ((dxInvs.size() != rhs.dxInvs.size()) || !std::equal(dxInvs.cbegin(), dxInvs.cend(), rhs.dxInvs.cbegin())) return false;
	else if ((axis.size() != rhs.axis.size()) || !std::equal(axis.cbegin(), axis.cend(), rhs.axis.cbegin())) return false;
	else if ((shape.size() != rhs.shape.size()) || !std::equal(shape.cbegin(), shape.cend(), rhs.shape.cbegin())) return false;

	else if ((vss.size() != rhs.vss.size()) || !std::equal(vss.cbegin(), vss.cend(), rhs.vss.cbegin(), [](const auto& lhs, const auto& rhs) {
		return  ((lhs.size() == rhs.size()) && std::equal(lhs.cbegin(), lhs.cend(), rhs.cbegin()));
	})) return false;
	else if ((xss.size() != rhs.xss.size()) || !std::equal(xss.cbegin(), xss.cend(), rhs.xss.cbegin(), [](const auto& lhs, const auto& rhs) {
		return  ((lhs.size() == rhs.size()) && std::equal(lhs.cbegin(), lhs.cend(), rhs.cbegin()));
	})) return false;
	return true;
}
bool HJI_Grid::operator==(const HJI_Grid& rhs) const {
	if (this == &rhs) return true;
	if (!pimpl) {
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

bool HJI_Grid_impl::processGrid(
	const std::vector<FLOAT_TYPE> &data
) {
	//! Now we should have a partially complete structure in gridOut.
	if (num_of_dimensions != 0) {
		if (num_of_dimensions > maxDimension) {
			printf("Error::%s : line %d: dimension > %zu, may be dangerously large\n", __FUNCTION__, __LINE__, maxDimension);
			return false;
		}
	}
	else {
		printf("Error::%s : line %d: grid structure must contain dimension\n", __FUNCTION__, __LINE__);
		return false;
	}

	//! Process grid boundaries.
	if (!mins.empty()) {
		if (mins.size() != num_of_dimensions) {
			if (mins.size() == 1) {
				mins.assign(num_of_dimensions, mins[0]);
			}
			else {
				printf("Error::%s : line %d: min field is not column vector of length dim or a scalar\n", __FUNCTION__, __LINE__);
				return false;
			}
		}
	}
	else {
		mins.assign(num_of_dimensions, defaultMin);
	}

	if (!maxs.empty()) {
		if (maxs.size() != num_of_dimensions) {
			if (maxs.size() == 1) {
				maxs.assign(num_of_dimensions, maxs[0]);
			}
			else {
				printf("Error::%s : line %d: max field is not column vector of length dim or a scalar\n", __FUNCTION__, __LINE__);
				return false;
			}
		}
	}
	else {
		maxs.assign(num_of_dimensions, defaultMax);
	}
	//! (max[0] > min[0]) & (max[1] > min[1]) ...
	if (!std::equal(maxs.cbegin(), maxs.cend(), mins.cbegin(), std::greater<FLOAT_TYPE>())) {
		printf("Error::%s : line %d: max bound must be strictly greater than min bound in all dimensions\n", __FUNCTION__, __LINE__);
		return false;
	}

	//! Check N field if necessary.If N is missing but dx is present, we will
	//! determine N later.
	if (!Ns.empty()) {
		//! (Ns[0] > 0) & (Ns[1] > 0) ...
		if (std::any_of(Ns.cbegin(), Ns.cend(), [](const auto& rhs) { return rhs <= 0; })) {
			printf("Error::%s : line %d: number of grid cells must be strictly positive\n", __FUNCTION__, __LINE__);
			return false;
		}
		if (Ns.size() != num_of_dimensions) {
			if (Ns.size() == 1) {
				Ns.assign(num_of_dimensions, Ns[0]);
			}
			else {
				printf("Error::%s : line %d: N field is not column vector of length dim or a scalar\n", __FUNCTION__, __LINE__);
				return false;
			}
		}
	}

	//! Check dx field if necessary.If dx is missing but N is present, infer
	//! dx.If both are present, we will check for consistency later.If
	//! neither are present, use the defaults.
	if (!dxs.empty()) {
		//! (dxs[0] > 0) & (dxs[1] > 0) ...
		if (std::any_of(dxs.cbegin(), dxs.cend(), [](const auto& rhs) { return rhs <= 0; })) {

			printf("Error::%s : line %d: grid cell size dx must be strictly positive\n", __FUNCTION__, __LINE__);
			return false;
		}
		if (dxs.size() != num_of_dimensions) {
			if (dxs.size() == 1) {
				dxs.assign(num_of_dimensions, dxs[0]);
			}
			else {
				printf("Error::%s : line %d: dx field is not column vector of length dim or a scalar\n", __FUNCTION__, __LINE__);
				return false;
			}
		}
	}
	else if (!Ns.empty()){
		//! Only N field is present, so infer dx.
		dxs.reserve(num_of_dimensions);
		for (size_t dimension = 0; dimension < num_of_dimensions; ++dimension) {
			dxs.push_back((FLOAT_TYPE)((maxs[dimension] - mins[dimension]) / ((FLOAT_TYPE)Ns[dimension] - 1.0)));
		}
	}
	else {
		//! Neither field is present, so use default N and infer dx
		Ns.reserve(num_of_dimensions);
		dxs.reserve(num_of_dimensions);
		for (size_t dimension = 0; dimension < num_of_dimensions; ++dimension) {
			Ns.push_back(defaultN);
			dxs.push_back((FLOAT_TYPE)((maxs[dimension] - mins[dimension]) / ((FLOAT_TYPE)Ns[dimension] - 1.0)));
		}
	}

	dxInvs.resize(num_of_dimensions);
	std::transform(dxs.cbegin(), dxs.cend(), dxInvs.begin(), ([](const auto &rhs) {return 1. / rhs; }));


	if (!vss.empty()) {
		if (vss.size() != num_of_dimensions) {
			printf("Error::%s : line %d: vs field is not column cell vector of length dim\n", __FUNCTION__, __LINE__);
			return false;
		} else {
			//! (dxs[0] > 0) & (dxs[1] > 0) ...
			if (std::any_of(vss.cbegin(), vss.cend(), [](const auto& rhs) { return rhs.empty(); })) {
				printf("Error::%s : line %d: vs field is not a cell vector\n", __FUNCTION__, __LINE__);
				return false;
			}
			else {
				//! (vss[0] == Ns[0]) & (vss[1] == Ns[1]) ...
				if (!std::equal(vss.cbegin(), vss.cend(), Ns.cbegin(), [](const auto& lhs, const auto& rhs) { return lhs.size() == rhs; })) {
					printf("Error::%s : line %d: vs cell entry is not correctly sized vector\n", __FUNCTION__, __LINE__);
					return false;
				}
			}
		}
	}
	else {
		//! Neither field is present, so use default N and infer dx
		vss.resize(num_of_dimensions);
		for (size_t dimension = 0; dimension<num_of_dimensions; ++dimension) {
			std::vector<FLOAT_TYPE>& vs = vss[dimension];
			FLOAT_TYPE dx = dxs[dimension];
			FLOAT_TYPE min = mins[dimension];
			FLOAT_TYPE max = maxs[dimension];
			//! Initialize arithmetic progression in same precision of Matlab...
			vs = generateArithmeticSequence<FLOAT_TYPE>(min, dx, max);
		}
	}

	//! Now we can check for consistency between dx and N, based on the size of
	//! the vectors in vs.Note that if N is present, it will be a vector.If
	//! N is not yet a field, set it to be consistent with the size of vs.
	if (!Ns.empty()) {
		if (std::inner_product(Ns.cbegin(), Ns.cend(), vss.cbegin(), false,
			([](const auto &lhs, const auto &rhs) { return (bool)(lhs | rhs); }),
			([](const auto &lhs, const auto &rhs) {
			if (lhs != rhs.size()) {
					printf("Error::%s : line %d: Inconsistent grid size in dimension %zu != %zu\n", __FUNCTION__, __LINE__,lhs,rhs.size());
					return true;
				}
				else return false;
			})
		)) {
			return false;
		}
	}
	else {
		std::transform(vss.cbegin(), vss.cend(), Ns.begin(), ([](const auto &lhs) { return lhs.size(); }));
	}


	if (!xss.empty()) {
		if (xss.size() != num_of_dimensions) {
			printf("Error::%s : line %d: xs field is not column cell vector of length dim\n", __FUNCTION__, __LINE__);
			return false;
		}
		else {
			//! (xss[0] == Ns[0]) & (xss[1] == Ns[1]) ...
			const size_t sum_of_elements = get_sum_of_elems();
			if (std::any_of(xss.cbegin(), xss.cend(), [sum_of_elements](const auto& rhs) {return rhs.size() != sum_of_elements; })) {
				printf("Error::%s : line %d: vs cell entry is not correctly sized vector\n", __FUNCTION__, __LINE__);
				return false;
			}
		}
	}
	else {
		size_t num_of_elements = get_sum_of_elems();

		xss.resize(num_of_dimensions);
		for_each(xss.begin(), xss.end(), ([num_of_elements](auto &rhs) {rhs.resize(num_of_elements); }));
		//! Transposing copy
		for (size_t dim = 0; dim < num_of_dimensions; ++dim) {
			size_t inner_dimensions_loop_size = 1;
			size_t target_dimension_loop_size = 1;
			size_t outer_dimensions_loop_size = 1;
			for (size_t dimension = 0; dimension < dim; ++dimension) {
				inner_dimensions_loop_size *= Ns[dimension];
			}
			target_dimension_loop_size = Ns[dim];
			for (size_t dimension = dim + 1; dimension < num_of_dimensions; ++dimension) {
				outer_dimensions_loop_size *= Ns[dimension];
			}
			if (inner_dimensions_loop_size == 1) {
				for (size_t outer_dimensions_loop_index = 0; outer_dimensions_loop_index < outer_dimensions_loop_size; ++outer_dimensions_loop_index) {
					size_t outer_index_term = outer_dimensions_loop_index * target_dimension_loop_size;
					std::copy(vss[dim].cbegin(), vss[dim].cend(), xss[dim].begin()+ outer_index_term);
				}
			}
			else {
				for (size_t outer_dimensions_loop_index = 0; outer_dimensions_loop_index < outer_dimensions_loop_size; ++outer_dimensions_loop_index) {
					size_t outer_index_term = outer_dimensions_loop_index * target_dimension_loop_size * inner_dimensions_loop_size;
					for (size_t target_dimension_loop_index = 0; target_dimension_loop_index < target_dimension_loop_size; ++target_dimension_loop_index) {
						const size_t target_index_term = target_dimension_loop_index * inner_dimensions_loop_size;
						const size_t dst_offset = outer_index_term + target_index_term;
						const FLOAT_TYPE vs_val = vss[dim][target_dimension_loop_index];
						std::fill(xss[dim].begin() + dst_offset, xss[dim].begin() + dst_offset + inner_dimensions_loop_size, vs_val);
					}
				}
			}
		}
	}

	//! initialize defaultBdry which is empty
	if (!boundaryConditions.empty()) {
		if (boundaryConditions.size() != num_of_dimensions) {
			if (boundaryConditions.size() == 1) {
				BoundaryCondition* boundaryCondition = boundaryConditions[0];
				boundaryConditions.resize(num_of_dimensions);
				if (boundaryCondition && boundaryCondition->valid()) {
					std::for_each(boundaryConditions.begin(), boundaryConditions.end(), ([boundaryCondition](auto &rhs) {
						rhs = boundaryCondition->clone();
					}));
					delete boundaryCondition;
				} else {
					printf("Error::%s : line %d: bdry field is not column cell vector of length dim\n", __FUNCTION__, __LINE__);
					return false;
				}
			}
			else {
				printf("Error::%s : line %d: bdry field is not a cell vector or a scalar\n", __FUNCTION__, __LINE__);
				return false;
			}
		}
		else {
			//! (boundaryConditions[0].valid()) & (boundaryConditions[1].valid()) ...
			if (std::any_of(boundaryConditions.cbegin(), boundaryConditions.cend(), [](const auto& rhs) { return !(rhs && rhs->valid()); })) {
				printf("Error::%s : line %d: bdry field is not a valid\n", __FUNCTION__, __LINE__);
				return false;
			}
		}
	}
	else {
		boundaryConditions.resize(num_of_dimensions);
		for_each(boundaryConditions.begin(), boundaryConditions.end(), ([](auto &rhs) {
			rhs = new defaultBoundaryCondition();
		}));
	}

	if ((num_of_dimensions == 2) || (num_of_dimensions == 3)) {
		if (!axis.empty()) {
			for (size_t dimension = 0; dimension < num_of_dimensions; ++dimension) {
				if (axis[2 * dimension] != mins[dimension]) {
					printf("Error::%s : line %d: axis and min fields do not agree\n", __FUNCTION__, __LINE__);
					return false;
				}
				if (axis[2 * dimension+1] != maxs[dimension]) {
					printf("Error::%s : line %d: axis and maxs fields do not agree\n", __FUNCTION__, __LINE__);
					return false;
				}
			}
		}
		else {
			axis.resize(2 * num_of_dimensions);
			for (size_t dimension = 0; dimension<num_of_dimensions; ++dimension) {
				axis[2 * dimension] = mins[dimension];
				axis[2 * dimension + 1] = maxs[dimension];
			}
		}
	}

	if (!shape.empty()) {
		if (!std::equal(shape.cbegin(), shape.cend(), Ns.cbegin())) {
			printf("Error::%s : line %d: shape and N fields do not agree\n", __FUNCTION__, __LINE__);
			return false;
		}
	}
	else {
		shape.resize(num_of_dimensions);
		for (size_t dimension = 0; dimension < num_of_dimensions; ++dimension) {
			shape[dimension] = Ns[dimension];
		}
	}

	if (!data.empty()) {
		const size_t num_of_elements_shape = std::accumulate(shape.cbegin(), shape.cend(), static_cast<size_t>(1), ([](const auto& lhs, const auto& rhs) {
			return lhs * rhs;
		}));
		if (num_of_elements_shape != data.size()) {
			printf("Error::%s : line %d: data parameter does not agree in array size with grid\n", __FUNCTION__, __LINE__);
			return false;
		}
	}

	return true;
}
HJI_Grid::HJI_Grid(
) {
	pimpl = new HJI_Grid_impl();
}
HJI_Grid::HJI_Grid(
	const size_t num_of_dimensions
) {
	pimpl = new HJI_Grid_impl(num_of_dimensions);
}
HJI_Grid::HJI_Grid(HJI_Grid_impl* pimpl) : pimpl(pimpl) {
}

HJI_Grid::~HJI_Grid() {
	if (pimpl) delete pimpl;
}

void HJI_Grid::set_num_of_dimensions(size_t a) {
	if (pimpl) pimpl->set_num_of_dimensions(a);
}
void HJI_Grid::set_mins(const std::vector<FLOAT_TYPE>& a) {
	if (pimpl) pimpl->set_mins(a);
}
void HJI_Grid::set_maxs(const std::vector<FLOAT_TYPE>& a) {
	if (pimpl) pimpl->set_maxs(a);
}
void HJI_Grid::set_boundaryConditions(const std::vector<BoundaryCondition*>& a) {
	if (pimpl) pimpl->set_boundaryConditions(a);
}
void HJI_Grid::set_Ns(const std::vector<size_t>& a) {
	if (pimpl) pimpl->set_Ns(a);
}
void HJI_Grid::set_dxs(const std::vector<FLOAT_TYPE>& a) {
	if (pimpl) pimpl->set_dxs(a);
}
void HJI_Grid::set_vss(const std::vector<std::vector<FLOAT_TYPE> >& a) {
	if (pimpl) pimpl->set_vss(a);
}
void HJI_Grid::set_xss(const std::vector<std::vector<FLOAT_TYPE> >& a) {
	if (pimpl) pimpl->set_xss(a);
}
void HJI_Grid::set_axis(const std::vector<FLOAT_TYPE>& a) {
	if (pimpl) pimpl->set_axis(a);
}
void HJI_Grid::set_shape(const std::vector<size_t>& a) {
	if (pimpl) pimpl->set_shape(a);
}
size_t HJI_Grid::get_numel() const {
	if (pimpl) return pimpl->get_numel();
	else return 0;
}
size_t HJI_Grid::get_sum_of_elems() const {
	if (pimpl) return pimpl->get_sum_of_elems();
	else return 0;
}
size_t HJI_Grid::get_num_of_dimensions() const {
	if (pimpl) return pimpl->get_num_of_dimensions();
	else return 0;
}
const std::vector<FLOAT_TYPE>& HJI_Grid::get_mins() const {
	if (pimpl) return pimpl->get_mins();
	else return dummy_float_type_vector;
}
const std::vector<FLOAT_TYPE>& HJI_Grid::get_maxs() const {
	if (pimpl) return pimpl->get_maxs();
	else return dummy_float_type_vector;
}
BoundaryCondition* HJI_Grid::get_boundaryCondition(const size_t dimension) const {
	if (pimpl) return pimpl->get_boundaryCondition(dimension);
	else return NULL;
}

const std::vector<size_t>& HJI_Grid::get_Ns() const {
	if (pimpl) return pimpl->get_Ns();
	else return dummy_size_t_vector;
}
size_t HJI_Grid::get_N(const size_t dimension) const {
	if (pimpl) return pimpl->get_N(dimension);
	else return 0;
}
const std::vector<FLOAT_TYPE>& HJI_Grid::get_dxs() const {
	if (pimpl) return pimpl->get_dxs();
	else return dummy_float_type_vector;
}
const std::vector<FLOAT_TYPE>& HJI_Grid::get_dxInvs() const {
	if (pimpl) return pimpl->get_dxInvs();
	else return dummy_float_type_vector;
}
FLOAT_TYPE HJI_Grid::get_dx(const size_t dimension) const {
	if (pimpl) return pimpl->get_dx(dimension);
	else return 0;
}
FLOAT_TYPE HJI_Grid::get_dxInv(const size_t dimension) const {
	if (pimpl) return pimpl->get_dxInv(dimension);
	else return 0;
}
const std::vector<std::vector<FLOAT_TYPE> >& HJI_Grid::get_vss() const {
	if (pimpl) return pimpl->get_vss();
	else return dummy_float_type_vector_vector;
}
const std::vector<std::vector<FLOAT_TYPE> >& HJI_Grid::get_xss() const {
	if (pimpl) return pimpl->get_xss();
	else return dummy_float_type_vector_vector;
}
const std::vector<FLOAT_TYPE>& HJI_Grid::get_vs(const size_t dimension) const {
	if (pimpl) return pimpl->get_vs(dimension);
	else return dummy_float_type_vector;
}
const std::vector<FLOAT_TYPE>& HJI_Grid::get_xs(const size_t dimension) const {
	if (pimpl) return pimpl->get_xs(dimension);
	else return dummy_float_type_vector;
}
const std::vector<FLOAT_TYPE>& HJI_Grid::get_axis() const {
	if (pimpl) return pimpl->get_axis();
	else return dummy_float_type_vector;
}
const std::vector<size_t>& HJI_Grid::get_shape() const {
	if (pimpl) return pimpl->get_shape();
	else return dummy_size_t_vector;
}

bool HJI_Grid::processGrid(const std::vector<FLOAT_TYPE> &data
) {
	if (pimpl) return pimpl->processGrid(data);
	else return false;
}
bool dump_vector(const std::string & file_name, const std::vector<FLOAT_TYPE> &src_vector)
{
	FILE *debug_fp;
#if defined(__GNUC__)
	debug_fp = fopen(file_name.c_str(), "w");
#else
	fopen_s(&debug_fp, file_name.c_str(), "wb");
#endif
	if (!debug_fp) return false;
	size_t num_of_elements = src_vector.size();
	for (size_t i = 0; i < num_of_elements; ++i) {
		//		fprintf(debug_fp, "%30.15f\n", data[i]);
		fprintf(debug_fp, " %24.16le\t\n", src_vector[i]);
	}
	fclose(debug_fp);
	return true;
}


static bool check_matvar_type(
	const matvar_t *matvar,
	const std::string&,
	const matio_classes class_type,
	const matio_types data_type) {
	if (!matvar) return false;
	if (matvar->class_type != class_type) {
		return false;
	}
	if (matvar->data_type != data_type) {
		return false;
	}
	return true;
}

template<typename T>
bool write_Mat_VarStructField(matvar_t *struct_var, const std::string &field_name, const T& src);


template<>
bool write_Mat_VarStructField(matvar_t *struct_var, const std::string &field_name, const std::vector<size_t>& src) {
	std::vector<size_t> dims(2);
	size_t size = src.size();
	dims[0] = size;
	dims[1] = 1;
	std::vector<double> data(size);
	for (size_t i = 0; i < size; ++i) {
		data[i] = static_cast<double>(src[i]);
	}

	matvar_t *struct_field = Mat_VarCreate(field_name.c_str(), get_matio_classes<double>(), get_matio_types<double>(), static_cast<int>(dims.size()), &dims[0], static_cast<void*>(&data[0]), 0);
	Mat_VarAddStructField(struct_var, field_name.c_str());
	Mat_VarSetStructFieldByName(struct_var, field_name.c_str(), 0, struct_field);
	return true;
}


template<>
bool write_Mat_VarStructField(matvar_t *struct_var, const std::string &field_name, const size_t& src) {
	std::vector<size_t> dims = { 1, 1 };
	size_t size = 1;
	std::vector<double> data(size);
	data[0] = static_cast<double>(src);

	matvar_t *struct_field = Mat_VarCreate(field_name.c_str(), get_matio_classes<double>(), get_matio_types<double>(), static_cast<int>(dims.size()), &dims[0], static_cast<void*>(&data[0]), 0);
	Mat_VarAddStructField(struct_var, field_name.c_str());
	Mat_VarSetStructFieldByName(struct_var, field_name.c_str(), 0, struct_field);
	return true;
}

template<>
bool write_Mat_VarStructField(matvar_t *struct_var, const std::string &field_name, const std::vector<FLOAT_TYPE>& src) {
	std::vector<size_t> dims(2);
	size_t size = src.size();
	dims[0] = size;
	dims[1] = 1;
	std::vector<double> data(size);
	for (size_t i = 0; i < size; ++i) {
		data[i] = static_cast<double>(src[i]);
	}

	matvar_t *struct_field = Mat_VarCreate(field_name.c_str(), get_matio_classes<double>(), get_matio_types<double>(), static_cast<int>(dims.size()), &dims[0], static_cast<void*>(&data[0]), 0);
	Mat_VarAddStructField(struct_var, field_name.c_str());
	Mat_VarSetStructFieldByName(struct_var, field_name.c_str(), 0, struct_field);
	return true;
}
template<>
bool write_Mat_VarStructField(matvar_t *struct_var, const std::string &field_name, const std::vector<std::vector<FLOAT_TYPE> >& src) {
	size_t ncells = src.size();
	std::vector<size_t> dims(2);
	dims[0] = ncells;
	dims[1] = 1;
	matvar_t *struct_field = Mat_VarCreate(field_name.c_str(), MAT_C_CELL, MAT_T_CELL, static_cast<int>(dims.size()), &dims[0], NULL, 0);
	for (size_t j = 0; j < ncells; j++) {
		std::vector<size_t> cell_dims(2);
		size_t cell_size = src[j].size();
		cell_dims[0] = cell_size;
		cell_dims[1] = 1;

		std::vector<double> data(cell_size);
		for (size_t i = 0; i < cell_size; ++i) {
			data[i] = static_cast<double>(src[j][i]);
		}
		matvar_t *cell = Mat_VarCreate(NULL, get_matio_classes<double>(), get_matio_types<double>(), static_cast<int>(cell_dims.size()), &cell_dims[0],static_cast<void*>(&data[0]),0);
		Mat_VarSetCell(struct_field, static_cast<int>(j), cell);
	}
	Mat_VarAddStructField(struct_var, field_name.c_str());
	Mat_VarSetStructFieldByName(struct_var, field_name.c_str(), 0, struct_field);
	return true;
}


bool HJI_Grid_impl::save_grid(
	const std::string &variable_name,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress) const {
	enum matio_compression compress_flag = (compress) ? MAT_COMPRESSION_ZLIB : MAT_COMPRESSION_NONE;
	matvar_t *matvar = NULL;
	mat_t *matfp = NULL;
	bool is_parent_cell = false;
	if (!parent) {
		if (!fs) {
			std::cerr << "Invalid Mat file stream." << std::endl;
			return false;
		}
		if (fs) matfp = fs->get_mat();
		if (variable_name.empty()) {
			std::cerr << "Error: variable_name empty." << std::endl;
			return false;
		}
	}
	else {
		matvar_t* parent_matvar = parent->get_matvar();
		if (check_matvar_type(parent_matvar, variable_name, MAT_C_CELL, MAT_T_CELL)) {
			is_parent_cell = true;
		}
	}
	size_t rank = 2;
	std::vector<size_t> dims = { 1,1 };
	matvar = Mat_VarCreateStruct(is_parent_cell ? NULL : variable_name.c_str(), static_cast<int>(rank), &dims[0], NULL, 0);

	write_Mat_VarStructField(matvar, std::string("dim"), num_of_dimensions);
	if (!mins.empty()) write_Mat_VarStructField(matvar, std::string("min"), mins);
	if (!maxs.empty()) write_Mat_VarStructField(matvar, std::string("max"), maxs);
	std::vector<size_t> boundaryCondition_types(boundaryConditions.size());
	std::transform(boundaryConditions.cbegin(), boundaryConditions.cend(), boundaryCondition_types.begin(), [](const auto& rhs) {
		std::type_info const & rhs_type = typeid(*rhs);
		BoundaryCondition::BoundaryCondition_Type type;
		if (rhs_type == typeid(AddGhostExtrapolate)) type = BoundaryCondition::BoundaryCondition_AddGhostExtrapolate;
		else if (rhs_type == typeid(AddGhostPeriodic)) type = BoundaryCondition::BoundaryCondition_AddGhostPeriodic;
		else type = BoundaryCondition::BoundaryCondition_Invalid;
		return type;
	});
	write_Mat_VarStructField(matvar, std::string("bdry_type"), boundaryCondition_types);
	//	write_Mat_VarStructField(matvar, std::string("bdry"), boundaryConditions);
	if (!Ns.empty()) write_Mat_VarStructField(matvar, std::string("N"), Ns);
	if (!dxs.empty()) write_Mat_VarStructField(matvar, std::string("dx"), dxs);
	if (!vss.empty()) write_Mat_VarStructField(matvar, std::string("vs"), vss);
//	write_Mat_VarStructField(matvar, std::string("xs"), xss);
	if (!axis.empty()) write_Mat_VarStructField(matvar, std::string("axis"), axis);
	if (!shape.empty()) write_Mat_VarStructField(matvar, std::string("shape"), shape);

	if (parent) {
		matvar_t* parent_matvar = parent->get_matvar();
		if (is_parent_cell) {
			Mat_VarSetCell(parent_matvar, static_cast<int>(cell_index), matvar);
		}else{
			matvar_t* struct_var = parent_matvar;
			Mat_VarAddStructField(struct_var, variable_name.c_str());
			Mat_VarSetStructFieldByName(struct_var, variable_name.c_str(), 0, matvar);
		}
	} else{
		int result = Mat_VarWrite(matfp, matvar, compress_flag);
		if (result != 0) {
			std::cerr << "Error: cannot write file: " << std::endl;
			Mat_VarFree(matvar);
			return false;
		}
		Mat_VarFree(matvar);
	}
	return true;
}

template<typename T>
bool read_Mat_VarStructField(
	matvar_t *struct_var, 
	const std::string &field_name,
	T& dst);



template<>
bool read_Mat_VarStructField(
	matvar_t *struct_var,
	const std::string &field_name,
	std::vector<size_t>& dst) {
	matvar_t *struct_field = Mat_VarGetStructFieldByName(struct_var, field_name.c_str(), 0);
	if (!check_matvar_type(struct_field, field_name, get_matio_classes<double>(), get_matio_types<double>())) {
//		if(struct_field) Mat_VarFree(struct_field);
		return false;
	}

	size_t size = struct_field->nbytes / struct_field->data_size;
	dst.resize(size);
	for (size_t i = 0; i < size; ++i) {
		dst[i] = static_cast<size_t>(*(static_cast<double*>(struct_field->data)+i));
	}
//	Mat_VarFree(struct_field);
	return true;
}

template<>
bool read_Mat_VarStructField(
	matvar_t *struct_var,
	const std::string &field_name,
	size_t& dst) {
	matvar_t *struct_field = Mat_VarGetStructFieldByName(struct_var, field_name.c_str(), 0);

	if (!check_matvar_type(struct_field, field_name, get_matio_classes<double>(), get_matio_types<double>())) {
//		if (struct_field) Mat_VarFree(struct_field);
		return false;
	}

	size_t size = struct_field->nbytes / struct_field->data_size;
	if (size >= 1)
		dst = static_cast<size_t>(*(static_cast<double*>(struct_field->data)));
//	Mat_VarFree(struct_field);
	return true;
}

template<>
bool read_Mat_VarStructField(
	matvar_t *struct_var, 
	const std::string &field_name, 
	std::vector<FLOAT_TYPE>& dst) {
	matvar_t *struct_field = Mat_VarGetStructFieldByName(struct_var, field_name.c_str(), 0);
	if (!struct_field) return false;
	if (!check_matvar_type(struct_field, field_name, get_matio_classes<double>(), get_matio_types<double>())) {
//		if (struct_field) Mat_VarFree(struct_field);
		return false;
	}

	size_t size = struct_field->nbytes / struct_field->data_size;
	dst.resize(size);
	for (size_t i = 0; i < size; ++i) {
		dst[i] = static_cast<FLOAT_TYPE>(*(static_cast<double*>(struct_field->data) + i));
	}
//	Mat_VarFree(struct_field);
	return true;
}
template<>
bool read_Mat_VarStructField(
	matvar_t *struct_var,
	const std::string &field_name,
	std::vector<std::vector<FLOAT_TYPE> >& dst) {
	matvar_t *struct_field = Mat_VarGetStructFieldByName(struct_var, field_name.c_str(), 0);

	if (!check_matvar_type(struct_field, field_name, MAT_C_CELL, MAT_T_CELL)) {
		if (struct_field) Mat_VarFree(struct_field);
		return false;
	}
	int ncells = static_cast<int>(struct_field->nbytes / struct_field->data_size);
	dst.resize(ncells);
	for (int j = 0; j < ncells; j++) {
		matvar_t *cell = Mat_VarGetCell(struct_field, j);

		if (!check_matvar_type(cell, field_name, get_matio_classes<double>(), get_matio_types<double>())) {
//			if (cell) Mat_VarFree(cell);
			continue;
		}
		size_t size = cell->nbytes / cell->data_size;
		dst[j].resize(size);
		for (size_t i = 0; i < size; ++i) {
			dst[j][i] = static_cast<FLOAT_TYPE>(*(static_cast<double*>(cell->data) + i));
		}
	}

//	Mat_VarFree(struct_field);
	return true;
}

bool HJI_Grid_impl::load_grid(
	const std::string &variable_name,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index
	) {
	matvar_t *matvar = NULL;
	mat_t *matfp = NULL;
	if (!parent) {
		if (!fs) {
			std::cerr << "Invalid Mat file stream." << std::endl;
			return false;
		}
		if (variable_name.empty()) {
			std::cerr << "Error: variable_name empty." << std::endl;
			return false;
		}
		matfp = fs->get_mat();
		matvar = Mat_VarRead(matfp, variable_name.c_str());
	}
	else {
		matvar_t* parent_matvar = parent->get_matvar();
		if (check_matvar_type(parent_matvar, variable_name, MAT_C_CELL, MAT_T_CELL)) {
			matvar = Mat_VarGetCell(matvar, static_cast<int>(cell_index));
		}
		else {
			matvar = Mat_VarGetStructFieldByName(parent_matvar, variable_name.c_str(), 0);
		}
	}
	if (!matvar) {
		std::cerr << "Error: cannot find data: " << variable_name.c_str() << std::endl;
		return false;
	}
	if (!check_matvar_type(matvar, variable_name, MAT_C_STRUCT, MAT_T_STRUCT)) {
		std::cerr << "Error: Invalid variable type: " << variable_name.c_str() << std::endl;
		if (!parent)
			Mat_VarFree(matvar);
		return false;
	}

	read_Mat_VarStructField(matvar, std::string("dim"), num_of_dimensions);
	read_Mat_VarStructField(matvar, std::string("min"), mins);
	read_Mat_VarStructField(matvar, std::string("max"), maxs);
	std::vector<size_t> boundaryCondition_types;
	read_Mat_VarStructField(matvar, std::string("bdry_type"), boundaryCondition_types);
	boundaryConditions.resize(boundaryCondition_types.size());
	std::transform(boundaryCondition_types.cbegin(), boundaryCondition_types.cend(), boundaryConditions.begin(), [](const auto& rhs) {
		BoundaryCondition* boundaryCondition = NULL;
		switch (rhs) {
		default:
		case BoundaryCondition::BoundaryCondition_Invalid:
			std::cerr << "Error: Invalid BoundaryCondition_Type: " << rhs << std::endl;
			break;
		case BoundaryCondition::BoundaryCondition_AddGhostExtrapolate:
			boundaryCondition = new AddGhostExtrapolate();
			break;
		case BoundaryCondition::BoundaryCondition_AddGhostPeriodic:
			boundaryCondition = new AddGhostPeriodic();
			break;
		case BoundaryCondition::BoundaryCondition_AddGhostExtrapolate2:
		case BoundaryCondition::BoundaryCondition_AddGhostDirichlet:
		case BoundaryCondition::BoundaryCondition_AddGhostNeumann:
			std::cerr << "Error: BoundaryCondition_Type: " << rhs << " is not supported yet." << std::endl;
			break;
		}
		return boundaryCondition;
	});
	//			read_Mat_VarStructField(matvar, std::string("bdry"), boundaryConditions);
	read_Mat_VarStructField(matvar, std::string("N"), Ns);
	if (read_Mat_VarStructField(matvar, std::string("dx"), dxs)) {
		set_dxInvs();
	}
	read_Mat_VarStructField(matvar, std::string("vs"), vss);
	read_Mat_VarStructField(matvar, std::string("xs"), xss);
	read_Mat_VarStructField(matvar, std::string("axis"), axis);
	read_Mat_VarStructField(matvar, std::string("shape"), shape);
	if (!parent)
		Mat_VarFree(matvar);
	return true;
}

bool HJI_Grid::save_grid(
	const std::string &variable_name, 
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress) const {
	if (pimpl) return pimpl->save_grid(variable_name, fs, parent, cell_index, compress);
	return false;
}
bool HJI_Grid::load_grid(
	const std::string &variable_name, 
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index) {
	if (pimpl) return pimpl->load_grid(variable_name, fs, parent, cell_index);
	return false;
}

HJI_Grid::HJI_Grid(const HJI_Grid& rhs) :
	pimpl(rhs.pimpl->clone(true))
{
}

HJI_Grid* HJI_Grid::clone(const bool cloneAll) const {
	if (cloneAll)
		return new HJI_Grid(*this);
	else {
		HJI_Grid* g = new HJI_Grid(this->pimpl->clone(false));
		return g;
	}
}
template<typename T>
bool save_vector_impl(	const std::vector<T> &src_vector,
		const std::string &variable_name,
		const std::vector<size_t>& Ns,
		const bool quiet,
		beacls::MatFStream* fs,
		beacls::MatVariable* parent,
		const size_t cell_index,
		const bool compress) {
	enum matio_compression compress_flag = (compress) ? 
	  MAT_COMPRESSION_ZLIB : MAT_COMPRESSION_NONE;

	if (src_vector.empty()) {
		if (quiet) return true;
		std::cerr << "Error: vector empty." << std::endl;
		return false;
	}

	int rank;
	std::vector<size_t> dims_vec;
	if (!Ns.empty()) {
		rank = static_cast<int>(Ns.size());
		dims_vec = Ns;
	}
	else {
		rank = 1;
		dims_vec.resize(1);
		dims_vec[0] = src_vector.size();
	}
	matvar_t *matvar = Mat_VarCreate(variable_name.c_str(), 
		  get_matio_classes<T>(), get_matio_types<T>(), rank, dims_vec.data(), 
		  const_cast<T*>(src_vector.data()), 0);
	if (!matvar) {
		std::cerr << "Error: cannot create mat variable: " << variable_name.c_str() << std::endl;
		return false;
	}
	if (parent) {
		matvar_t* parent_matvar = parent->get_matvar();
		if (check_matvar_type(parent_matvar, variable_name, MAT_C_CELL, MAT_T_CELL)) {
			Mat_VarSetCell(parent_matvar, static_cast<int>(cell_index), matvar);
		}
		else {
			matvar_t* struct_var = parent_matvar;
			Mat_VarAddStructField(struct_var, variable_name.c_str());
			Mat_VarSetStructFieldByName(struct_var, variable_name.c_str(), 0, matvar);
		}
	}
	else {
		if (variable_name.empty()) {
			std::cerr << "Error: variable_name empty." << std::endl;
			return false;
		}
		if (!fs) {
			std::cerr << "Invalid Mat file stream." << std::endl;
			return false;
		}
		mat_t *matfp = NULL;
		if (fs) matfp = fs->get_mat();
		if (matfp) {
			int result = Mat_VarWrite(matfp, matvar, compress_flag);
			if (result != 0) {
				std::cerr << "Error: cannot write file: " << std::endl;
				Mat_VarFree(matvar);
				return false;
			}
		}
		Mat_VarFree(matvar);
	}
	return true;
}

template<typename T, typename S>
bool load_vector_and_cast(
	std::vector<T>& dst_vector,
	std::vector<S>& tmp_vector,
	const void* data_ptr,
	const size_t size
) {
	tmp_vector.resize(size);
	std::copy((S*)data_ptr, (S*)data_ptr + size, tmp_vector.begin());
	dst_vector.resize(size);
	std::transform(tmp_vector.cbegin(), tmp_vector.cend(), dst_vector.begin(), [](const auto& rhs) { return static_cast<T>(rhs); });
	return true;
}
template<typename T, typename S>
bool load_deque_and_cast(
	std::deque<T>& dst_deque,
	std::vector<S>& tmp_vector,
	const void* data_ptr,
	const size_t size
) {
	tmp_vector.resize(size);
	std::copy((S*)data_ptr, (S*)data_ptr + size, tmp_vector.begin());
	dst_deque.resize(size);
	std::transform(tmp_vector.cbegin(), tmp_vector.cend(), dst_deque.begin(), [](const auto& rhs) { return static_cast<T>(rhs); });
	return true;
}
template<typename S>
bool load_deque_and_cast_to_bool(
	std::deque<bool>& dst_deque,
	std::vector<S>& tmp_vector,
	const void* data_ptr,
	const size_t size
) {
	tmp_vector.resize(size);
	std::copy((S*)data_ptr, (S*)data_ptr + size, tmp_vector.begin());
	dst_deque.resize(size);
	std::transform(tmp_vector.cbegin(), tmp_vector.cend(), dst_deque.begin(), [](const auto& rhs) { return (rhs != 0); });
	return true;
}
template<typename T>
bool load_vector_impl(
	std::vector<T> &dst_vector,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index)
{
	matvar_t *matvar = NULL;
	mat_t *matfp = NULL;
	if (!parent) {
		if (!fs) {
			std::cerr << "Invalid Mat file stream." << std::endl;
			return false;
		}
		matfp = fs->get_mat();
		matvar = Mat_VarRead(matfp, variable_name.c_str());
	}
	else {
		matvar_t* parent_matvar = parent->get_matvar();
		if (check_matvar_type(parent_matvar, variable_name, MAT_C_CELL, MAT_T_CELL)) {
			matvar = Mat_VarGetCell(matvar, static_cast<int>(cell_index));
		}
		else {
			matvar = Mat_VarGetStructFieldByName(parent_matvar, variable_name.c_str(), 0);
		}
	}
	if (!matvar) {
		if (quiet) return true;
		std::cerr << "Error: cannot find data: " << variable_name.c_str() << std::endl;
		return false;
	}

	int rank = matvar->rank;
	Ns.resize(rank);
	std::copy(matvar->dims, matvar->dims + rank, Ns.begin());
	size_t num_of_elements = matvar->nbytes / matvar->data_size;
	dst_vector.resize(num_of_elements);
	if (matvar->class_type == get_matio_classes<T>()) {
		std::copy((T*)matvar->data, (T*)matvar->data + num_of_elements, dst_vector.begin());
	}
	else {
		std::vector<float> tmp_vector_float;
		std::vector<double> tmp_vector_double;
		std::vector<int8_t> tmp_vector_int8_t;
		std::vector<uint8_t> tmp_vector_uint8_t;
		std::vector<int16_t> tmp_vector_int16_t;
		std::vector<uint16_t> tmp_vector_uint16_t;
		std::vector<int32_t> tmp_vector_int32_t;
		std::vector<uint32_t> tmp_vector_uint32_t;
		std::vector<int64_t> tmp_vector_int64_t;
		std::vector<uint64_t> tmp_vector_uint64_t;

		switch(matvar->class_type) {
		case MAT_C_SINGLE:
			load_vector_and_cast<T, float>(dst_vector, tmp_vector_float, matvar->data, num_of_elements);
			break;
		case MAT_C_DOUBLE:
			load_vector_and_cast<T, double>(dst_vector, tmp_vector_double, matvar->data, num_of_elements);
			break;
		case MAT_C_INT8:
			load_vector_and_cast<T, int8_t>(dst_vector, tmp_vector_int8_t, matvar->data, num_of_elements);
			break;
		case MAT_C_UINT8:
			load_vector_and_cast<T, uint8_t>(dst_vector, tmp_vector_uint8_t, matvar->data, num_of_elements);
			break;
		case MAT_C_INT16:
			load_vector_and_cast<T, int16_t>(dst_vector, tmp_vector_int16_t, matvar->data, num_of_elements);
			break;
		case MAT_C_UINT16:
			load_vector_and_cast<T, uint16_t>(dst_vector, tmp_vector_uint16_t, matvar->data, num_of_elements);
			break;
		case MAT_C_INT32:
			load_vector_and_cast<T, int32_t>(dst_vector, tmp_vector_int32_t, matvar->data, num_of_elements);
			break;
		case MAT_C_UINT32:
			load_vector_and_cast<T, uint32_t>(dst_vector, tmp_vector_uint32_t, matvar->data, num_of_elements);
			break;
		case MAT_C_INT64:
			load_vector_and_cast<T, int64_t>(dst_vector, tmp_vector_int64_t, matvar->data, num_of_elements);
			break;
		case MAT_C_UINT64:
			load_vector_and_cast<T, uint64_t>(dst_vector, tmp_vector_uint64_t, matvar->data, num_of_elements);
			break;
		default:
			std::cerr << "Error: Invalid variable type: " << variable_name.c_str() << std::endl;
			if (!parent)
				Mat_VarFree(matvar);
			return false;
		}
	}

	if (!parent)
		Mat_VarFree(matvar);
	return true;
}

template<typename T>
bool save_vector_cast(
	const std::vector<T> &src_vector,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress) {
	std::vector<double> tmp_vector(src_vector.size());
	std::transform(src_vector.cbegin(), src_vector.cend(), tmp_vector.begin(), [](const auto& rhs) { return static_cast<double>(rhs); });
	return save_vector_impl(tmp_vector, variable_name, Ns, quiet, fs, parent, cell_index, compress);
};

template<typename T>
bool load_vector_cast(
	std::vector<T> &dst_vector,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index) {
	std::vector<double> tmp_vector;
	if (load_vector_impl(tmp_vector, variable_name, Ns, quiet, fs, parent, cell_index)) {
		dst_vector.resize(tmp_vector.size());
		std::transform(tmp_vector.cbegin(), tmp_vector.cend(), dst_vector.begin(), [](const auto& rhs) { return static_cast<T>(rhs); });
		return true;
	}
	return false;
}

template<typename T>
bool save_deque_cast(
	const std::deque<T> &src_deque,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress) {
	std::vector<double> tmp_vector(src_deque.size());
	std::transform(src_deque.cbegin(), src_deque.cend(), tmp_vector.begin(), [](const auto& rhs) { return static_cast<double>(rhs); });
	return save_vector_impl(tmp_vector, variable_name, Ns, quiet, fs, parent, cell_index, compress);
};
template<>
bool save_deque_cast(
	const std::deque<bool> &src_deque,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress) {
	std::vector<double> tmp_vector(src_deque.size());
	std::transform(src_deque.cbegin(), src_deque.cend(), tmp_vector.begin(), [](const auto& rhs) { return (rhs != 0); });
	return save_vector_impl(tmp_vector, variable_name, Ns, quiet, fs, parent, cell_index, compress);
};
template<typename T>
bool load_deque_cast(
	std::deque<T> &dst_deque,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index) {
	std::vector<uint8_t> tmp_vector;
	if (load_vector(tmp_vector, variable_name, Ns, quiet,fs, parent, cell_index)) {
		dst_deque.resize(tmp_vector.size());
		std::transform(tmp_vector.cbegin(), tmp_vector.cend(), dst_deque.begin(), [](const auto& rhs) { return static_cast<T>(rhs); });
		return true;
	}
	return false;
}
template<>
bool load_deque_cast(
	std::deque<bool> &dst_deque,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index) {
	std::vector<uint8_t> tmp_vector;
	if (load_vector(tmp_vector, variable_name, Ns, quiet,fs, parent, cell_index)) {
		dst_deque.resize(tmp_vector.size());
		std::transform(tmp_vector.cbegin(), tmp_vector.cend(), dst_deque.begin(), [](const auto& rhs) { return (rhs != 0); });
		return true;
	}
	return false;
}
bool save_vector_double(
	const std::vector<double> &src_vector,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress) {
	return save_vector_impl(src_vector, variable_name, Ns, quiet, fs, parent, cell_index, compress);
}
bool load_vector_double(
	std::vector<double> &dst_vector,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index) {
	return load_vector_impl(dst_vector, variable_name, Ns, quiet, fs, parent, cell_index);
}
bool save_vector_float(
	const std::vector<float> &src_vector,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress) {
	return save_vector_impl(src_vector, variable_name, Ns, quiet, fs, parent, cell_index, compress);
};

bool load_vector_float(
	std::vector<float> &dst_vector,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index) {
	return load_vector_impl(dst_vector, variable_name, Ns, quiet, fs, parent, cell_index);
}

bool save_vector_size_t(
	const std::vector<size_t> &src_vector,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress) {
	return save_vector_impl(src_vector, variable_name, Ns, quiet,fs, parent, cell_index, compress);
};

bool load_vector_size_t(
	std::vector<size_t> &dst_vector,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index) {
	return load_vector_impl(dst_vector, variable_name, Ns, quiet,fs, parent, cell_index);
}
bool load_vector_uint8_t(
	std::vector<uint8_t> &dst_vector,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index) {
	return load_vector_impl(dst_vector, variable_name, Ns, quiet, fs, parent, cell_index);
}
bool save_vector_uint8_t(
	const std::vector<uint8_t> &src_vector,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress) {
	return save_vector_impl(src_vector, variable_name, Ns, quiet, fs, parent, cell_index, compress);
}

bool save_deque_bool(
	const std::deque<bool> &src_deque,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress) {
	return save_deque_cast<bool>(src_deque, variable_name, Ns, quiet, fs, parent, cell_index, compress);
}
bool load_deque_bool(
	std::deque<bool> &dst_deque,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index) {
	return load_deque_cast<bool>(dst_deque, variable_name, Ns, quiet, fs, parent, cell_index);
}

template<typename T>
bool save_vector_of_vectors_impl(
	const std::vector<std::vector<T> > &src_vectors,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress) {
	enum matio_compression compress_flag = (compress) ? MAT_COMPRESSION_ZLIB : MAT_COMPRESSION_NONE;
	if (src_vectors.empty()) {
		if (quiet) return true;
		std::cerr << "Error: vector empty." << std::endl;
		return false;
	}

	size_t ncells = src_vectors.size();
	std::vector<size_t> dims(2);
	dims[0] = ncells;
	dims[1] = 1;
	matvar_t *matvar = Mat_VarCreate(variable_name.c_str(), MAT_C_CELL, MAT_T_CELL, static_cast<int>(dims.size()), &dims[0], NULL, 0);
	if (!matvar) {
		std::cerr << "Error: cannot create mat variable: " << variable_name.c_str() << std::endl;
		return false;
	}
	for (size_t i = 0; i < src_vectors.size();++i) {
		std::vector<size_t> cell_dims;
		int rank;
		if (!Ns.empty()) {
			rank = static_cast<int>(Ns.size());
			cell_dims = Ns;
		}
		else {
			rank = 1;
			size_t cell_size = src_vectors[i].size();
			cell_dims.resize(1);
			cell_dims[0] = cell_size;
		}
		matvar_t *cell = Mat_VarCreate(NULL, get_matio_classes<T>(), get_matio_types<T>(), rank, cell_dims.data(), const_cast<T*>(src_vectors[i].data()), 0);
		Mat_VarSetCell(matvar, static_cast<int>(i), cell);
	};

	if (parent) {
		matvar_t* parent_matvar = parent->get_matvar();
		if (check_matvar_type(parent_matvar, variable_name, MAT_C_CELL, MAT_T_CELL)) {
			Mat_VarSetCell(parent_matvar, static_cast<int>(cell_index), matvar);
		}
		else {
			matvar_t* struct_var = parent_matvar;
			Mat_VarAddStructField(struct_var, variable_name.c_str());
			Mat_VarSetStructFieldByName(struct_var, variable_name.c_str(), 0, matvar);
		}
	}
	else {
		if (variable_name.empty()) {
			std::cerr << "Error: variable_name empty." << std::endl;
			return false;
		}
		if (!fs) {
			std::cerr << "Invalid Mat file stream." << std::endl;
			return false;
		}
		mat_t *matfp = NULL;
		if (fs) matfp = fs->get_mat();
		int result = Mat_VarWrite(matfp, matvar, compress_flag);
		if (result != 0) {
			std::cerr << "Error: cannot write file: " << std::endl;
			Mat_VarFree(matvar);
			return false;
		}
		Mat_VarFree(matvar);
	}
	return true;

}
template<typename T>
bool load_vector_of_vectors_impl(
	std::vector<std::vector<T> > &dst_vectors,
	const std::string &variable_name, 
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index) {
	matvar_t *matvar = NULL;
	mat_t *matfp = NULL;
	if (!parent) {
		if (!fs) {
			std::cerr << "Invalid Mat file stream." << std::endl;
			return false;
		}
		matfp = fs->get_mat();
		matvar = Mat_VarRead(matfp, variable_name.c_str());
	}
	else {
		matvar_t* parent_matvar = parent->get_matvar();
		if (check_matvar_type(parent_matvar, variable_name, MAT_C_CELL, MAT_T_CELL)) {
			matvar = Mat_VarGetCell(matvar, static_cast<int>(cell_index));
		}
		else {
			matvar = Mat_VarGetStructFieldByName(parent_matvar, variable_name.c_str(), 0);
		}
	}
	if (!matvar) {
		if (quiet) return true;
		std::cerr << "Error: cannot find data: " << variable_name.c_str() << std::endl;
		return false;
	}
	if (!check_matvar_type(matvar, variable_name, MAT_C_CELL, MAT_T_CELL)) {
		if (quiet) return true;
		std::cerr << "Error: Invalid variable type: " << variable_name.c_str() << std::endl;
		if (!parent)
			Mat_VarFree(matvar);
		return false;
	}

	int ncells = static_cast<int>(matvar->nbytes / matvar->data_size);
	dst_vectors.resize(ncells);
	bool first = true;
	std::vector<float> tmp_vector_float;
	std::vector<double> tmp_vector_double;
	std::vector<int8_t> tmp_vector_int8_t;
	std::vector<uint8_t> tmp_vector_uint8_t;
	std::vector<int16_t> tmp_vector_int16_t;
	std::vector<uint16_t> tmp_vector_uint16_t;
	std::vector<int32_t> tmp_vector_int32_t;
	std::vector<uint32_t> tmp_vector_uint32_t;
	std::vector<int64_t> tmp_vector_int64_t;
	std::vector<uint64_t> tmp_vector_uint64_t;

	for (int j = 0; j < ncells; j++) {
		matvar_t *cell = Mat_VarGetCell(matvar, j);
		if (first) {
			int rank = cell->rank;
			Ns.resize(rank);
			std::copy(cell->dims, cell->dims + rank, Ns.begin());
			first = false;
		}
		size_t size = cell->nbytes / cell->data_size;
		dst_vectors[j].resize(size);
		if (cell->class_type == get_matio_classes<T>()) {
			std::copy(static_cast<T*>(cell->data), static_cast<T*>(cell->data) + size, dst_vectors[j].begin());
		}
		else {
			switch(cell->class_type) {
			case MAT_C_SINGLE:
				load_vector_and_cast<T, float>(dst_vectors[j], tmp_vector_float, cell->data, size);
				break;
			case MAT_C_DOUBLE:
				load_vector_and_cast<T, double>(dst_vectors[j], tmp_vector_double, cell->data, size);
				break;
			case MAT_C_INT8:
				load_vector_and_cast<T, int8_t>(dst_vectors[j], tmp_vector_int8_t, cell->data, size);
				break;
			case MAT_C_UINT8:
				load_vector_and_cast<T, uint8_t>(dst_vectors[j], tmp_vector_uint8_t, cell->data, size);
				break;
			case MAT_C_INT16:
				load_vector_and_cast<T, int16_t>(dst_vectors[j], tmp_vector_int16_t, cell->data, size);
				break;
			case MAT_C_UINT16:
				load_vector_and_cast<T, uint16_t>(dst_vectors[j], tmp_vector_uint16_t, cell->data, size);
				break;
			case MAT_C_INT32:
				load_vector_and_cast<T, int32_t>(dst_vectors[j], tmp_vector_int32_t, cell->data, size);
				break;
			case MAT_C_UINT32:
				load_vector_and_cast<T, uint32_t>(dst_vectors[j], tmp_vector_uint32_t, cell->data, size);
				break;
			case MAT_C_INT64:
				load_vector_and_cast<T, int64_t>(dst_vectors[j], tmp_vector_int64_t, cell->data, size);
				break;
			case MAT_C_UINT64:
				load_vector_and_cast<T, uint64_t>(dst_vectors[j], tmp_vector_uint64_t, cell->data, size);
				break;
			default:
				std::cerr << "Error: Invalid variable type: " << variable_name.c_str() << std::endl;
				if (!parent)
					Mat_VarFree(matvar);
				return false;
				}
			}

	}
	if (!parent)
		Mat_VarFree(matvar);
	return true;
}

template<typename T>
bool save_vector_of_deques_impl(
	const std::vector<std::deque<T> > &src_deques,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress) {
	enum matio_compression compress_flag = (compress) ? MAT_COMPRESSION_ZLIB : MAT_COMPRESSION_NONE;
	if (src_deques.empty()) {
		if (quiet) return true;
		std::cerr << "Error: vector empty." << std::endl;
		return false;
	}

	size_t ncells = src_deques.size();
	std::vector<size_t> dims(2);
	dims[0] = ncells;
	dims[1] = 1;
	matvar_t *matvar = Mat_VarCreate(variable_name.c_str(), MAT_C_CELL, MAT_T_CELL, static_cast<int>(dims.size()), &dims[0], NULL, 0);
	if (!matvar) {
		std::cerr << "Error: cannot create mat variable: " << variable_name.c_str() << std::endl;
		return false;
	}
	std::vector<T> tmp_vector;
	for (size_t i = 0; i < src_deques.size(); ++i) {
		std::vector<size_t> cell_dims;
		int rank;
		if (!Ns.empty()) {
			rank = static_cast<int>(Ns.size());
			cell_dims = Ns;
		}
		else {
			rank = 1;
			size_t cell_size = src_deques[i].size();
			cell_dims.resize(1);
			cell_dims[0] = cell_size;
		}
		tmp_vector.resize(src_deques[i].size());
		std::transform(src_deques[i].cbegin(), src_deques[i].cend(), tmp_vector.begin(), [](const auto& rhs) { return const_cast<T>(rhs); });
		matvar_t *cell = Mat_VarCreate(NULL, get_matio_classes<T>(), get_matio_types<T>(), rank, cell_dims.data(), const_cast<T*>(tmp_vector.data()), 0);
		Mat_VarSetCell(matvar, static_cast<int>(i), cell);
	};


	if (parent) {
		matvar_t* parent_matvar = parent->get_matvar();
		if (check_matvar_type(parent_matvar, variable_name, MAT_C_CELL, MAT_T_CELL)) {
			Mat_VarSetCell(parent_matvar, static_cast<int>(cell_index), matvar);
		}
		else {
			matvar_t* struct_var = parent_matvar;
			Mat_VarAddStructField(struct_var, variable_name.c_str());
			Mat_VarSetStructFieldByName(struct_var, variable_name.c_str(), 0, matvar);
		}
	}
	else {
		if (variable_name.empty()) {
			std::cerr << "Error: variable_name empty." << std::endl;
			return false;
		}
		if (!fs) {
			std::cerr << "Invalid Mat file stream." << std::endl;
			return false;
		}
		mat_t *matfp = NULL;
		if (fs) matfp = fs->get_mat();
		int result = Mat_VarWrite(matfp, matvar, compress_flag);
		if (result != 0) {
			std::cerr << "Error: cannot write file: " << std::endl;
			Mat_VarFree(matvar);
			return false;
		}
		Mat_VarFree(matvar);
	}
	return true;

}
template<typename T>
bool load_vector_of_deques_impl(
	std::vector<std::deque<T> > &dst_deques,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index) {
	matvar_t *matvar = NULL;
	mat_t *matfp = NULL;
	if (!parent) {
		if (!fs) {
			std::cerr << "Invalid Mat file stream." << std::endl;
			return false;
		}
		matfp = fs->get_mat();
		matvar = Mat_VarRead(matfp, variable_name.c_str());
	}
	else {
		matvar_t* parent_matvar = parent->get_matvar();
		if (check_matvar_type(parent_matvar, variable_name, MAT_C_CELL, MAT_T_CELL)) {
			matvar = Mat_VarGetCell(matvar, static_cast<int>(cell_index));
		}
		else {
			matvar = Mat_VarGetStructFieldByName(parent_matvar, variable_name.c_str(), 0);
		}
	}
	if (!matvar) {
		if (quiet) return true;
		std::cerr << "Error: cannot find data: " << variable_name.c_str() << std::endl;
		return false;
	}
	if (!check_matvar_type(matvar, variable_name, MAT_C_CELL, MAT_T_CELL)) {
		if (quiet) return true;
		std::cerr << "Error: Invalid variable type: " << variable_name.c_str() << std::endl;
		if (!parent)
			Mat_VarFree(matvar);
		return false;
	}

	int ncells = static_cast<int>(matvar->nbytes / matvar->data_size);
	dst_deques.resize(ncells);
	bool first = true;
	std::vector<float> tmp_vector_float;
	std::vector<double> tmp_vector_double;
	std::vector<int8_t> tmp_vector_int8_t;
	std::vector<uint8_t> tmp_vector_uint8_t;
	std::vector<int16_t> tmp_vector_int16_t;
	std::vector<uint16_t> tmp_vector_uint16_t;
	std::vector<int32_t> tmp_vector_int32_t;
	std::vector<uint32_t> tmp_vector_uint32_t;
	std::vector<int64_t> tmp_vector_int64_t;
	std::vector<uint64_t> tmp_vector_uint64_t;

	for (int j = 0; j < ncells; j++) {
		matvar_t *cell = Mat_VarGetCell(matvar, j);
		if (first) {
			int rank = cell->rank;
			Ns.resize(rank);
			std::copy(cell->dims, cell->dims + rank, Ns.begin());
			first = false;
		}
		size_t size = cell->nbytes / cell->data_size;
		dst_deques[j].resize(size);
		if (cell->class_type == get_matio_classes<T>()) {
			std::copy(static_cast<T*>(cell->data), static_cast<T*>(cell->data) + size, dst_deques[j].begin());
		}
		else {
			switch (cell->class_type) {
			case MAT_C_SINGLE:
				load_vector_and_cast<T, float>(dst_deques[j], tmp_vector_float, cell->data, size);
				break;
			case MAT_C_DOUBLE:
				load_vector_and_cast<T, double>(dst_deques[j], tmp_vector_double, cell->data, size);
				break;
			case MAT_C_INT8:
				load_vector_and_cast<T, int8_t>(dst_deques[j], tmp_vector_int8_t, cell->data, size);
				break;
			case MAT_C_UINT8:
				load_vector_and_cast<T, uint8_t>(dst_deques[j], tmp_vector_uint8_t, cell->data, size);
				break;
			case MAT_C_INT16:
				load_vector_and_cast<T, int16_t>(dst_deques[j], tmp_vector_int16_t, cell->data, size);
				break;
			case MAT_C_UINT16:
				load_vector_and_cast<T, uint16_t>(dst_deques[j], tmp_vector_uint16_t, cell->data, size);
				break;
			case MAT_C_INT32:
				load_vector_and_cast<T, int32_t>(dst_deques[j], tmp_vector_int32_t, cell->data, size);
				break;
			case MAT_C_UINT32:
				load_vector_and_cast<T, uint32_t>(dst_deques[j], tmp_vector_uint32_t, cell->data, size);
				break;
			case MAT_C_INT64:
				load_vector_and_cast<T, int64_t>(dst_deques[j], tmp_vector_int64_t, cell->data, size);
				break;
			case MAT_C_UINT64:
				load_vector_and_cast<T, uint64_t>(dst_deques[j], tmp_vector_uint64_t, cell->data, size);
				break;
			default:
				std::cerr << "Error: Invalid variable type: " << variable_name.c_str() << std::endl;
				if (!parent)
					Mat_VarFree(matvar);
				return false;
			}
		}

	}
	if (!parent)
		Mat_VarFree(matvar);
	return true;
}

template<>
bool save_vector_of_deques_impl(
	const std::vector<std::deque<bool> > &src_deques,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress) {
	enum matio_compression compress_flag = (compress) ? MAT_COMPRESSION_ZLIB : MAT_COMPRESSION_NONE;
	if (src_deques.empty()) {
		if (quiet) return true;
		std::cerr << "Error: vector empty." << std::endl;
		return false;
	}

	size_t ncells = src_deques.size();
	std::vector<size_t> dims(2);
	dims[0] = ncells;
	dims[1] = 1;
	matvar_t *matvar = Mat_VarCreate(variable_name.c_str(), MAT_C_CELL, MAT_T_CELL, static_cast<int>(dims.size()), &dims[0], NULL, 0);
	if (!matvar) {
		std::cerr << "Error: cannot create mat variable: " << variable_name.c_str() << std::endl;
		return false;
	}
	std::vector<uint8_t> tmp_vector;
	for (size_t i = 0; i < src_deques.size(); ++i) {
		std::vector<size_t> cell_dims;
		int rank;
		if (!Ns.empty()) {
			rank = static_cast<int>(Ns.size());
			cell_dims = Ns;
		}
		else {
			rank = 1;
			size_t cell_size = src_deques[i].size();
			cell_dims.resize(1);
			cell_dims[0] = cell_size;
		}
		tmp_vector.resize(src_deques[i].size());
		std::transform(src_deques[i].cbegin(), src_deques[i].cend(), tmp_vector.begin(), [](const auto& rhs) { return rhs ? std::numeric_limits<uint8_t>::max() : 0; });
		matvar_t *cell = Mat_VarCreate(NULL, get_matio_classes<uint8_t>(), get_matio_types<uint8_t>(), rank, cell_dims.data(), const_cast<uint8_t*>(tmp_vector.data()), 0);
		Mat_VarSetCell(matvar, static_cast<int>(i), cell);
	};

	if (parent) {
		matvar_t* parent_matvar = parent->get_matvar();
		if (check_matvar_type(parent_matvar, variable_name, MAT_C_CELL, MAT_T_CELL)) {
			Mat_VarSetCell(parent_matvar, static_cast<int>(cell_index), matvar);
		}
		else {
			matvar_t* struct_var = parent_matvar;
			Mat_VarAddStructField(struct_var, variable_name.c_str());
			Mat_VarSetStructFieldByName(struct_var, variable_name.c_str(), 0, matvar);
		}
	}
	else {
		if (variable_name.empty()) {
			std::cerr << "Error: variable_name empty." << std::endl;
			return false;
		}
		if (!fs) {
			std::cerr << "Invalid Mat file stream." << std::endl;
			return false;
		}
		mat_t *matfp = NULL;
		if (fs) matfp = fs->get_mat();
		int result = Mat_VarWrite(matfp, matvar, compress_flag);
		if (result != 0) {
			std::cerr << "Error: cannot write file: " << std::endl;
			Mat_VarFree(matvar);
			return false;
		}
		Mat_VarFree(matvar);
	}
	return true;

}
template<>
bool load_vector_of_deques_impl(
	std::vector<std::deque<bool> > &dst_deques,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index) {
	matvar_t *matvar = NULL;
	mat_t *matfp = NULL;
	if (!parent) {
		if (!fs) {
			std::cerr << "Invalid Mat file stream." << std::endl;
			return false;
		}
		matfp = fs->get_mat();
		matvar = Mat_VarRead(matfp, variable_name.c_str());
	}
	else {
		matvar_t* parent_matvar = parent->get_matvar();
		if (check_matvar_type(parent_matvar, variable_name, MAT_C_CELL, MAT_T_CELL)) {
			matvar = Mat_VarGetCell(matvar, static_cast<int>(cell_index));
		}
		else {
			matvar = Mat_VarGetStructFieldByName(parent_matvar, variable_name.c_str(), 0);
		}
	}
	if (!matvar) {
		if (quiet) return true;
		std::cerr << "Error: cannot find data: " << variable_name.c_str() << std::endl;
		return false;
	}
	if (!check_matvar_type(matvar, variable_name, MAT_C_CELL, MAT_T_CELL)) {
		if (quiet) return true;
		std::cerr << "Error: Invalid variable type: " << variable_name.c_str() << std::endl;
		if (!parent)
			Mat_VarFree(matvar);
		return false;
	}

	int ncells = static_cast<int>(matvar->nbytes / matvar->data_size);
	dst_deques.resize(ncells);
	bool first = true;
	std::vector<float> tmp_vector_float;
	std::vector<double> tmp_vector_double;
	std::vector<int8_t> tmp_vector_int8_t;
	std::vector<uint8_t> tmp_vector_uint8_t;
	std::vector<int16_t> tmp_vector_int16_t;
	std::vector<uint16_t> tmp_vector_uint16_t;
	std::vector<int32_t> tmp_vector_int32_t;
	std::vector<uint32_t> tmp_vector_uint32_t;
	std::vector<int64_t> tmp_vector_int64_t;
	std::vector<uint64_t> tmp_vector_uint64_t;

	for (int j = 0; j < ncells; j++) {
		matvar_t *cell = Mat_VarGetCell(matvar, j);
		if (first) {
			int rank = cell->rank;
			Ns.resize(rank);
			std::copy(cell->dims, cell->dims + rank, Ns.begin());
			first = false;
		}
		size_t size = cell->nbytes / cell->data_size;
		dst_deques[j].resize(size);
		switch (cell->class_type) {
		case MAT_C_SINGLE:
			load_deque_and_cast_to_bool<float>(dst_deques[j], tmp_vector_float, cell->data, size);
			break;
		case MAT_C_DOUBLE:
			load_deque_and_cast_to_bool<double>(dst_deques[j], tmp_vector_double, cell->data, size);
			break;
		case MAT_C_INT8:
			load_deque_and_cast_to_bool<int8_t>(dst_deques[j], tmp_vector_int8_t, cell->data, size);
			break;
		case MAT_C_UINT8:
			load_deque_and_cast_to_bool<uint8_t>(dst_deques[j], tmp_vector_uint8_t, cell->data, size);
			break;
		case MAT_C_INT16:
			load_deque_and_cast_to_bool<int16_t>(dst_deques[j], tmp_vector_int16_t, cell->data, size);
			break;
		case MAT_C_UINT16:
			load_deque_and_cast_to_bool<uint16_t>(dst_deques[j], tmp_vector_uint16_t, cell->data, size);
			break;
		case MAT_C_INT32:
			load_deque_and_cast_to_bool<int32_t>(dst_deques[j], tmp_vector_int32_t, cell->data, size);
			break;
		case MAT_C_UINT32:
			load_deque_and_cast_to_bool<uint32_t>(dst_deques[j], tmp_vector_uint32_t, cell->data, size);
			break;
		case MAT_C_INT64:
			load_deque_and_cast_to_bool<int64_t>(dst_deques[j], tmp_vector_int64_t, cell->data, size);
			break;
		case MAT_C_UINT64:
			load_deque_and_cast_to_bool<uint64_t>(dst_deques[j], tmp_vector_uint64_t, cell->data, size);
			break;
		default:
			std::cerr << "Error: Invalid variable type: " << variable_name.c_str() << std::endl;
			if (!parent)
				Mat_VarFree(matvar);
			return false;
		}

	}
	if (!parent)
		Mat_VarFree(matvar);
	return true;
}


template<typename T>
bool save_vector_of_vectors_cast(
	const std::vector<std::vector<T> > &src_vectors,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress) {
	std::vector<std::vector<double>> tmp_vectors(src_vectors.size());
	std::vector<double> tmp_vector;
	std::transform(src_vectors.cbegin(), src_vectors.cend(), tmp_vectors.begin(), [&tmp_vector](const auto& rhs) {
		tmp_vector.resize(rhs.size());
		std::transform(rhs.cbegin(), rhs.cend(), tmp_vector.begin(), [](const auto& rhs) {
			return static_cast<double>(rhs);
		});
		return tmp_vector;
	});
	return save_vector_of_vectors(tmp_vectors, variable_name, Ns, quiet,fs, parent, cell_index, compress);
}
template<>
bool save_vector_of_vectors_cast(
	const std::vector<std::vector<double> > &src_vectors,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress) {
	return save_vector_of_vectors_impl(src_vectors, variable_name, Ns, quiet,fs, parent, cell_index, compress);
}
template<typename T>
bool load_vector_of_vectors_cast(
	std::vector<std::vector<T> > &dst_vectors,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index) {
	std::vector<std::vector<double>> tmp_vectors;
	if (load_vector_of_vectors(tmp_vectors, variable_name, Ns, quiet,fs, parent, cell_index)) {
		dst_vectors.resize(tmp_vectors.size());
		std::vector<T> tmp_vector;
		std::transform(tmp_vectors.cbegin(), tmp_vectors.cend(), dst_vectors.begin(), [&tmp_vector](const auto& rhs) {
			tmp_vector.resize(rhs.size());
			std::transform(rhs.cbegin(), rhs.cend(), tmp_vector.begin(), [](const auto& rhs) {
				return static_cast<T>(rhs);
			});
			return tmp_vector;
		});

		return true;
	}
	return false;
}
template<>
bool load_vector_of_vectors_cast(
	std::vector<std::vector<double> > &dst_vectors,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index) {
	return load_vector_of_vectors_impl(dst_vectors, variable_name, Ns, quiet,fs, parent, cell_index);
}

bool save_vector_of_vectors_double(
	const std::vector<std::vector<double> > &src_vectors,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress) {
	return save_vector_of_vectors_impl(src_vectors, variable_name, Ns, quiet, fs, parent, cell_index, compress);
}
bool load_vector_of_vectors_double(
	std::vector<std::vector<double> > &dst_vectors,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index) {
	return load_vector_of_vectors_impl(dst_vectors, variable_name, Ns, quiet, fs, parent, cell_index);
}
bool save_vector_of_vectors_float(
	const std::vector<std::vector<float> > &src_vectors,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress) {
	return save_vector_of_vectors_impl(src_vectors, variable_name, Ns, quiet, fs, parent, cell_index, compress);
}
bool load_vector_of_vectors_float(
	std::vector<std::vector<float> > &dst_vectors,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index) {
	return load_vector_of_vectors_impl(dst_vectors, variable_name, Ns, quiet, fs, parent, cell_index);
}
bool save_vector_of_vectors_size_t(
	const std::vector<std::vector<size_t> > &src_vectors,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress) {
	return save_vector_of_vectors_impl(src_vectors, variable_name, Ns, quiet, fs, parent, cell_index, compress);
}
bool load_vector_of_vectors_size_t(
	std::vector<std::vector<size_t> > &dst_vectors,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index) {
	return load_vector_of_vectors_impl(dst_vectors, variable_name, Ns, quiet, fs, parent, cell_index);
}
bool save_vector_of_vectors_uint8_t(
	const std::vector<std::vector<uint8_t> > &src_vectors,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress) {
	return save_vector_of_vectors_impl(src_vectors, variable_name, Ns, quiet, fs, parent, cell_index, compress);
}
bool load_vector_of_vectors_uint8_t(
	std::vector<std::vector<uint8_t> > &dst_vectors,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index) {
	return load_vector_of_vectors_impl(dst_vectors, variable_name, Ns, quiet, fs, parent, cell_index);
}
bool save_vector_of_vectors_int8_t(
	const std::vector<std::vector<int8_t> > &src_vectors,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress) {
	return save_vector_of_vectors_impl(src_vectors, variable_name, Ns, quiet, fs, parent, cell_index, compress);
}
bool load_vector_of_vectors_int8_t(
	std::vector<std::vector<int8_t> > &dst_vectors,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index) {
	return load_vector_of_vectors_impl(dst_vectors, variable_name, Ns, quiet, fs, parent, cell_index);
}
bool save_vector_of_deques_bool(
	const std::vector<std::deque<bool> > &src_deques,
	const std::string &variable_name,
	const std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress) {
	return save_vector_of_deques_impl(src_deques, variable_name, Ns, quiet, fs, parent, cell_index, compress);
}
bool load_vector_of_deques_bool(
	std::vector<std::deque<bool> > &dst_deques,
	const std::string &variable_name,
	std::vector<size_t>& Ns,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index) {
	return load_vector_of_deques_impl(dst_deques, variable_name, Ns, quiet, fs, parent, cell_index);
}

bool save_value(
	const float &src_value,
	const std::string &value_name,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress) {
	std::vector<size_t> Ns;
	std::vector<float> src_vector(1);
	src_vector[0] = src_value;
	return save_vector(src_vector, value_name, Ns, quiet, fs, parent, cell_index, compress);
}
bool load_value(
	float &dst_value,
	const std::string &value_name,	
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index) {
	std::vector<size_t> Ns;
	std::vector<float> dst_vector;
	if (!load_vector(dst_vector, value_name, Ns, quiet, fs, parent, cell_index)) {
		return false;
	}
	if (dst_vector.size() != 1) {
		return false;
	}
	dst_value = dst_vector[0];
	return true;
}
bool save_value(
	const double &src_value,
	const std::string &value_name,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress) {
	std::vector<size_t> Ns;
	std::vector<double> src_vector(1);
	src_vector[0] = src_value;
	return save_vector(src_vector, value_name, Ns, quiet, fs, parent, cell_index, compress);
}
bool load_value(
	double &dst_value,
	const std::string &value_name,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index) {
	std::vector<size_t> Ns;
	std::vector<double> dst_vector;
	if (!load_vector(dst_vector, value_name, Ns, quiet, fs, parent, cell_index)) {
		return false;
	}
	if (dst_vector.size() != 1) {
		return false;
	}
	dst_value = dst_vector[0];
	return true;
}
bool save_value(
	const size_t &src_value, 
	const std::string &value_name, 
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress) {
	std::vector<size_t> Ns;
	std::vector<size_t> src_vector(1);
	src_vector[0] = src_value;
	return save_vector_cast(src_vector, value_name, Ns, quiet, fs, parent, cell_index, compress);
}
bool load_value(
	size_t &dst_value, 
	const std::string &value_name,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index) {
	std::vector<size_t> Ns;
	std::vector<size_t> dst_vector;
	if (!load_vector_cast(dst_vector, value_name, Ns, quiet,fs, parent, cell_index)) {
		return false;
	}
	if (dst_vector.size() != 1) {
		return false;
	}
	dst_value = dst_vector[0];
	return true;
}

bool save_value(
	const bool &src_value,
	const std::string &value_name, 
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index,
	const bool compress) {
	std::vector<size_t> Ns;
	std::deque<bool> src_vector(1);
	src_vector[0] = src_value;
	return save_deque_cast(src_vector, value_name, Ns, quiet, fs, parent, cell_index, compress);
}
bool load_value(
	bool &dst_value, 
	const std::string &value_name,
	const bool quiet,
	beacls::MatFStream* fs,
	beacls::MatVariable* parent,
	const size_t cell_index) {
	std::vector<size_t> Ns;
	std::deque<bool> dst_vector;
	if (!load_deque_cast(dst_vector, value_name, Ns, quiet,fs, parent, cell_index)) {
		return false;
	}
	if (dst_vector.size() != 1) {
		return false;
	}
	dst_value = dst_vector[0];
	return true;
}

beacls::MatFStream* beacls::openMatFStream(
	const std::string& file_name,
	const beacls::MatOpenMode mode
) {
	mat_t *matfp;
	switch (mode) {
	default:
	case MatOpenMode_Read:
		matfp =  Mat_Open(file_name.c_str(), MAT_ACC_RDONLY);
		break;
	case MatOpenMode_Write:
		matfp = Mat_CreateVer(file_name.c_str(), NULL, MAT_FT_MAT73);
		break;
	case MatOpenMode_WriteAppend:
		matfp = Mat_Open(file_name.c_str(), MAT_ACC_RDWR);
		break;
	}
	if (!matfp) {
		std::cerr << "Error: cannot open: " << file_name.c_str() << std::endl;
		return NULL;
	}
	return new beacls::MatFStream(matfp);
}
bool beacls::closeMatFStream(
	beacls::MatFStream* fs) {
	mat_t* matfp = fs->get_mat();
	if (!matfp) {
		std::cerr << "Invalid file staream pointer." << std::endl;
		return false;
	}
	Mat_Close(matfp);
	delete fs;
	return true;
}

beacls::MatVariable* beacls::openMatVariable(
	MatFStream* fs,
	const std::string& variable_name
) {
	matvar_t* matvar = NULL;
	mat_t* matfp = fs->get_mat();
	if (!matfp) {
		std::cerr << "Invalid file staream pointer." << std::endl;
		return NULL;
	}
	matvar = Mat_VarRead(matfp, variable_name.c_str());
	return new beacls::MatVariable(matvar);
}
bool beacls::closeMatVariable(MatVariable* variable) {
	matvar_t* matvar = variable->get_matvar();
	if (!matvar) {
		std::cerr << "Invalid variable pointer." << std::endl;
		return false;
	}
	if (variable->should_it_be_free()) {
		Mat_VarFree(matvar);
	}
	delete variable;
	return true;
}
beacls::MatVariable* beacls::createMatStruct(
	const std::string& variable_name
) {
	size_t rank = 2;
	std::vector<size_t> dims = { 1,1 };
	matvar_t *matvar = Mat_VarCreateStruct(variable_name.c_str(), static_cast<int>(rank), &dims[0], NULL, 0);
	return new beacls::MatVariable(matvar);
}
bool beacls::writeMatVariable(
	MatFStream* fs,
	MatVariable* variable,
	const bool compress) {
	enum matio_compression compress_flag = (compress) ? MAT_COMPRESSION_ZLIB : MAT_COMPRESSION_NONE;

	mat_t* matfp = fs->get_mat();
	if (!matfp) {
		std::cerr << "Invalid file staream pointer." << std::endl;
		return false;
	}
	matvar_t* matvar = variable->get_matvar();
	if (!matvar) {
		std::cerr << "Invalid variable pointer." << std::endl;
		return false;
	}
	int result = Mat_VarWrite(matfp, matvar, compress_flag);
	if (result != 0) {
		std::cerr << "Error: cannot write file" << std::endl;
		return false;
	}
	return true;
}
beacls::MatVariable* beacls::createMatCell(
	const std::string& variable_name,
	const size_t size
) {
	std::vector<size_t> dims(2);
	dims[0] = size;
	dims[1] = 1;
	matvar_t *matvar = Mat_VarCreate(variable_name.c_str(), MAT_C_CELL, MAT_T_CELL, static_cast<int>(dims.size()), &dims[0], NULL, 0);
	if (!matvar) {
		std::cerr << "Error: cannot create mat variable: " << variable_name.c_str() << std::endl;
		return NULL;
	}
	return new beacls::MatVariable(matvar);
}
bool beacls::setVariableToStruct(
	MatVariable* parent,
	MatVariable* child,
	const std::string& field_name) {
	matvar_t* struct_var = parent->get_matvar();
	matvar_t* struct_field = child->get_matvar();
	int result = Mat_VarAddStructField(struct_var, field_name.c_str());
	if (result == 0) {
		child->set_should_be_free(false);
		return (Mat_VarSetStructFieldByName(struct_var, field_name.c_str(), 0, struct_field)!=NULL);
	}
	return false;
}
bool beacls::setVariableToCell(
	MatVariable* parent,
	MatVariable* child,
	const size_t index) {
	matvar_t* matvar = parent->get_matvar();
	matvar_t* cell = child->get_matvar();
	child->set_should_be_free(false);
	return Mat_VarSetCell(matvar, static_cast<int>(index), cell) != NULL;
}
beacls::MatVariable* beacls::getVariableFromStruct(
	MatVariable* parent,
	const std::string& variable_name
) {
	matvar_t* parent_matvar = parent->get_matvar();
	matvar_t* matvar = NULL;
	if (parent_matvar) {
		if (check_matvar_type(parent_matvar, variable_name, MAT_C_STRUCT, MAT_T_STRUCT)) {
			matvar = Mat_VarGetStructFieldByName(parent_matvar, variable_name.c_str(), 0);
		}
	}
	if (!matvar) {
		std::cerr << "Error: cannot find data: " << variable_name.c_str() << std::endl;
	}
	return new beacls::MatVariable(matvar, false);
}
beacls::MatVariable* beacls::getVariableFromCell(
	MatVariable* parent,
	const size_t cell_index) {
	matvar_t* parent_matvar = parent->get_matvar();
	matvar_t* matvar = NULL;
	if (parent_matvar) {
		if (check_matvar_type(parent_matvar, std::string(), MAT_C_CELL, MAT_T_CELL)) {
			int ncells = static_cast<int>(parent_matvar->nbytes / parent_matvar->data_size);
			if (cell_index < (size_t)ncells) {
				matvar = Mat_VarGetCell(parent_matvar, static_cast<int>(cell_index));
			}
		}
	}
	if (!matvar) {
		std::cerr << "Error: cannot find data[ " << cell_index << "]" << std::endl;
	}
	return new beacls::MatVariable(matvar, false);
}
size_t beacls::getCellSize(
	MatVariable* variable_ptr) {
	matvar_t* matvar = variable_ptr->get_matvar();
	if (matvar) {
		if (check_matvar_type(matvar, std::string(), MAT_C_CELL, MAT_T_CELL)) {
			int ncells = static_cast<int>(matvar->nbytes / matvar->data_size);
			return (size_t)ncells;
		}
	}
	return 0;
}
