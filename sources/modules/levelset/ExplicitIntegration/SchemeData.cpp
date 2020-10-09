#include <typeinfo>
#include <levelset/Grids/HJI_Grid.hpp>
#include <levelset/ExplicitIntegration/SchemeData.hpp>
#include "SchemeData_impl.hpp"
using namespace levelset;

SchemeData::SchemeData(
) {
	pimpl = new SchemeData_impl();
}
SchemeData::~SchemeData() {
	if (pimpl) delete pimpl;
}
bool SchemeData_impl::operator==(const SchemeData_impl& rhs)  const {
	if (this == &rhs) return true;
	else if ((grid != rhs.grid) && (!grid || !rhs.grid || !grid->operator==(*rhs.grid))) return false;
	else if ((spatialDerivative != rhs.spatialDerivative) && (!spatialDerivative || !rhs.spatialDerivative || !spatialDerivative->operator==(*rhs.spatialDerivative))) return false;
	else if ((dissipation != rhs.dissipation) && (!dissipation || !rhs.dissipation || !dissipation->operator==(*rhs.dissipation))) return false;
	else if ((innerFunc != rhs.innerFunc) && (!innerFunc || !rhs.innerFunc || !innerFunc->operator==(*rhs.innerFunc))) return false;
	else if ((innerData != rhs.innerData) && (!innerData || !rhs.innerData || !innerData->operator==(*rhs.innerData))) return false;
	else if (positive != rhs.positive) return false;
	else return true;
}
bool SchemeData::operator==(const SchemeData& rhs) const {
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
const HJI_Grid* SchemeData::get_grid() const {
		if (pimpl) return pimpl->get_grid();
		return NULL;
	}
SpatialDerivative* SchemeData::get_spatialDerivative() const {
	if (pimpl) return pimpl->get_spatialDerivative();
	return NULL;
}
Dissipation* SchemeData::get_dissipation() const {
	if (pimpl) return pimpl->get_dissipation();
	return NULL;
}
const Term* SchemeData::get_innerFunc() const {
	if (pimpl) return pimpl->get_innerFunc();
	return NULL;
}
const SchemeData* SchemeData::get_innerData() const {
	if (pimpl) return pimpl->get_innerData();
	return NULL;
}
bool SchemeData::get_positive() const {
	if (pimpl) return pimpl->get_positive();
	return false;
}
void SchemeData::set_grid(const HJI_Grid* grid) {
	if (pimpl) return pimpl->set_grid(grid);
}

void SchemeData::set_spatialDerivative(SpatialDerivative* spatialDerivative) {
	if (pimpl) return pimpl->set_spatialDerivative(spatialDerivative);
}
void SchemeData::set_dissipation(Dissipation* dissipation) {
	if (pimpl) return pimpl->set_dissipation(dissipation);
}
void SchemeData::set_innerFunc(const Term* innerFunc) {
	if (pimpl) return pimpl->set_innerFunc(innerFunc);
}
void SchemeData::set_innerData(const SchemeData* innerData) {
	if (pimpl) return pimpl->set_innerData(innerData);
}
void SchemeData::set_positive(const bool positive) {
	if (pimpl) return pimpl->set_positive(positive);
}
bool SchemeData::hamFunc(
	beacls::UVec&,
	const FLOAT_TYPE,
	const beacls::UVec&,
	const std::vector<beacls::UVec>&,
	const size_t,
	const size_t
	) const {
	return true;
}
bool SchemeData::partialFunc(
	beacls::UVec&,
	const FLOAT_TYPE,
	const beacls::UVec&,
	const std::vector<beacls::UVec>&,
	const std::vector<beacls::UVec>&,
	const size_t,
	const size_t,
	const size_t
) const {
	return true;
}; 
bool SchemeData::initializeLocalQ(
	const beacls::FloatVec &vRange, 
	const beacls::FloatVec &dMax,
	const FLOAT_TYPE wMax 
) {
	return true; 
};
bool SchemeData::hamFuncLocalQ(
	beacls::UVec&,
	const FLOAT_TYPE,
	const beacls::UVec&,
	const std::vector<beacls::UVec>&,
	const size_t,
	const size_t,
	const std::set<size_t> &
) const {
	return true; 
}
bool SchemeData::partialFuncLocalQ(
	beacls::UVec&,
	const FLOAT_TYPE,
	const beacls::UVec&,
	const std::vector<beacls::UVec>&,
	const std::vector<beacls::UVec>&,
	const size_t,
	const size_t,
	const size_t, 
	const std::set<size_t>& 
) const {
	return true;
};
bool SchemeData::hamFunc_cuda(
	beacls::UVec&,
	const FLOAT_TYPE,
	const beacls::UVec&,
	const std::vector<beacls::UVec>&,
	const std::vector<beacls::UVec>&,
	const size_t,
	const size_t
) const {
	return false;
}
bool SchemeData::partialFunc_cuda(
	beacls::UVec&,
	const FLOAT_TYPE,
	const beacls::UVec&,
	const std::vector<beacls::UVec>&,
	const std::vector<beacls::UVec>&,
	const std::vector<beacls::UVec>&,
	const size_t,
	const size_t,
	const size_t
) const {
	return false;
};
SchemeData::SchemeData(const SchemeData& rhs) :
	pimpl(rhs.pimpl->clone())
{
}
SchemeData* SchemeData::clone() const {
	return new SchemeData(*this);
}
