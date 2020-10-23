#include "helperOC/DynSys/Plane/PlaneSchemeDataLocalQ.hpp"
#include <algorithm>
#include <functional>
#include <iostream>
#include <numeric>
#include <iomanip>
#include <typeinfo>
#include <macro.hpp>

using namespace helperOC;

bool PlaneSchemeDataLocalQ::operator==(const PlaneSchemeDataLocalQ& rhs) const {
	if (this == &rhs) return true;
	else if (!SchemeData::operator==(rhs)) return false;
	else if (vMin_ != rhs.vMin_) return false;
	else if (vMax_ != rhs.vMax_) return false;
	else if (dMax_x_ != rhs.dMax_x_) return false;
	else if (dMax_y_ != rhs.dMax_y_) return false;
	else if (dMax_theta_ != rhs.dMax_theta_) return false;
	else if (wMax_ != rhs.wMax_) return false;
	else return true;
}
bool PlaneSchemeDataLocalQ::operator==(const SchemeData& rhs) const {
	if (this == &rhs) return true;
	else if (typeid(*this) != typeid(rhs)) return false;
	else return operator==(dynamic_cast<const PlaneSchemeDataLocalQ&>(rhs));
}
bool PlaneSchemeDataLocalQ::initializeLocalQ(
	const beacls::FloatVec &vRange, 
	const beacls::FloatVec &dMax,
	const FLOAT_TYPE wMaximum
) {
	vMin_ = vRange[0];
	vMax_ = vRange[1];
	dMax_x_ = dMax[0];
	dMax_y_ = dMax[1];
	dMax_theta_ = dMax[2];
	wMax_ = wMaximum; 
}

bool PlaneSchemeDataLocalQ::hamFunc(
	beacls::UVec& hamValue_uvec,
	const FLOAT_TYPE t,
	const beacls::UVec&,
	const std::vector<beacls::UVec>& derivs,
	const size_t begin_index,
	const size_t length 
) const {
	const levelset::HJI_Grid *hji_grid = get_grid();
	const beacls::FloatVec &xs2 = hji_grid->get_xs(2);
	beacls::reallocateAsSrc(hamValue_uvec, derivs[0]);
	FLOAT_TYPE* hamValue = beacls::UVec_<FLOAT_TYPE>(hamValue_uvec).ptr();
	const FLOAT_TYPE* deriv0 = beacls::UVec_<FLOAT_TYPE>(derivs[0]).ptr();
	const FLOAT_TYPE* deriv1 = beacls::UVec_<FLOAT_TYPE>(derivs[1]).ptr();
	const FLOAT_TYPE* deriv2 = beacls::UVec_<FLOAT_TYPE>(derivs[2]).ptr();

	for (size_t i = 0; i < length; ++i) {
		int dst_index = begin_index + i; 
		FLOAT_TYPE deriv0_i = deriv0[i];
		FLOAT_TYPE deriv1_i = deriv1[i];
		FLOAT_TYPE theta = xs2[dst_index];
		FLOAT_TYPE speedCtrl = deriv0_i * std::cos(theta) + deriv1_i * std::sin(theta);
		if ((speedCtrl >= 0 && uMode == helperOC::DynSys_UMode_Max) || 
			(speedCtrl < 0 && uMode == helperOC::DynSys_UMode_Min))
		{
			hamValue[i] = speedCtrl * vMax_;
		}
		else 
		{
			hamValue[i] = speedCtrl * vMin_;
		}
		FLOAT_TYPE wTerm;
		if (uMode == helperOC::DynSys_UMode_Max)
		{
			wTerm = wMax_* std::abs(deriv2[i]);
		}
		else 
		{
			wTerm = -wMax_* std::abs(deriv2[i]);
		}
		hamValue[i] += wTerm;

		FLOAT_TYPE dTerm; 
		if (dMode == helperOC::DynSys_DMode_Min)
		{
			 dTerm = -dMax_x_ * std::abs(deriv0_i) - dMax_y_ * std::abs(deriv1_i);
		}
		else
		{
			dTerm = dMax_x_ * std::abs(deriv0_i) + dMax_y_ * std::abs(deriv1_i);
		}
		hamValue[i] += dTerm;
		//backard reachable set
		if (tMode == helperOC::DynSys_TMode_Backward)
		{
			hamValue[i] = -hamValue[i];
		}
	}
		
	return true;
}

bool PlaneSchemeDataLocalQ::partialFunc(
	beacls::UVec& alphas_uvec,
	const FLOAT_TYPE t,
	const beacls::UVec& data,
	const std::vector<beacls::UVec>& derivMins,
	const std::vector<beacls::UVec>& derivMaxs,
	const size_t dim,
	const size_t begin_index,
	const size_t length
) const {
	if (alphas_uvec.type() != beacls::UVecType_Vector) alphas_uvec = beacls::UVec(beacls::type_to_depth<FLOAT_TYPE>(), beacls::UVecType_Vector, length);
	else alphas_uvec.resize(length);
	FLOAT_TYPE* alphas = beacls::UVec_<FLOAT_TYPE>(alphas_uvec).ptr();
	const levelset::HJI_Grid *hji_grid = get_grid();

	switch (dim) {
	case 0:
	{
		const beacls::FloatVec& xs2 = hji_grid->get_xs(2);
		for (size_t i = 0; i < length; ++i) {
			size_t dst_index = i + begin_index;
			alphas[i] = vMax_ * std::abs(std::cos(xs2[dst_index])) + dMax_x_;
		}
	}
	break;
	case 1:
	{
		const beacls::FloatVec& xs2 = hji_grid->get_xs(2);
		for (size_t i = 0; i < length; ++i) {
			size_t dst_index = i + begin_index;
			alphas[i] = vMax_ * std::abs(std::sin(xs2[dst_index])) + dMax_y_;
		}
	}
	break;
	case 2:
	{
		for (size_t i = 0; i < length; ++i) {
			alphas[i] = wMax_ + dMax_theta_;
		}
	}
	break;
	default:
		printf("error:%s,%d\n", __func__, __LINE__);
		return false;
	}
	return true;
}

bool PlaneSchemeDataLocalQ::hamFuncLocalQ(
	beacls::UVec& hamValue_uvec,
	const FLOAT_TYPE t,
	const beacls::UVec& data,
	const std::vector<beacls::UVec>& derivs,
	const size_t begin_index,
	const size_t length, 
	const std::set<size_t> &Q
) const {
	const levelset::HJI_Grid *hji_grid = get_grid();

	const beacls::FloatVec &xs2 = hji_grid->get_xs(2);
	beacls::reallocateAsSrc(hamValue_uvec, derivs[0]);
	FLOAT_TYPE* hamValue = beacls::UVec_<FLOAT_TYPE>(hamValue_uvec).ptr();
	const FLOAT_TYPE* deriv0 = beacls::UVec_<FLOAT_TYPE>(derivs[0]).ptr();
	const FLOAT_TYPE* deriv1 = beacls::UVec_<FLOAT_TYPE>(derivs[1]).ptr();
	const FLOAT_TYPE* deriv2 = beacls::UVec_<FLOAT_TYPE>(derivs[2]).ptr();

	for (size_t i = 0; i < length; ++i) {
		int dst_index = begin_index + i; 
		if (Q.find(dst_index) != Q.end())
		{
			FLOAT_TYPE deriv0_i = deriv0[i];
			FLOAT_TYPE deriv1_i = deriv1[i];
			FLOAT_TYPE theta = xs2[dst_index];
			FLOAT_TYPE speedCtrl = deriv0_i * std::cos(theta) + deriv1_i * std::sin(theta);
			if ((speedCtrl >= 0 && uMode == helperOC::DynSys_UMode_Max) || 
				(speedCtrl < 0 && uMode == helperOC::DynSys_UMode_Min))
			{
				hamValue[i] = speedCtrl * vMax_;
			}
			else 
			{
				hamValue[i] = speedCtrl * vMin_;
			}
			FLOAT_TYPE wTerm;
			if (uMode == helperOC::DynSys_UMode_Max)
			{
				wTerm = wMax_* std::abs(deriv2[i]);
			}
			else 
			{
				wTerm = -wMax_* std::abs(deriv2[i]);
			}
			hamValue[i] += wTerm;

			FLOAT_TYPE dTerm; 
			if (dMode == helperOC::DynSys_DMode_Min)
			{
				dTerm = -dMax_x_ * std::abs(deriv0_i) - dMax_y_ * std::abs(deriv1_i);
			}
			else
			{
				dTerm = dMax_x_ * std::abs(deriv0_i) + dMax_y_ * std::abs(deriv1_i);
			}
			hamValue[i] += dTerm;
			//backard reachable set
			if (tMode == helperOC::DynSys_TMode_Backward)
			{
				hamValue[i] = -hamValue[i];
			}
		}
		else 
		{
			hamValue[i] = 0;
		}
	}
		
	return true;
}
bool PlaneSchemeDataLocalQ::partialFuncLocalQ(
	beacls::UVec& alphas_uvec,
	const FLOAT_TYPE t,
	const beacls::UVec& data,
	const std::vector<beacls::UVec>& derivMins,
	const std::vector<beacls::UVec>& derivMaxs,
	const size_t dim,
	const size_t begin_index,
	const size_t length, 
	const std::set<size_t> &Q
) const 
{
	if (alphas_uvec.type() != beacls::UVecType_Vector) alphas_uvec = beacls::UVec(beacls::type_to_depth<FLOAT_TYPE>(), beacls::UVecType_Vector, length);
	else alphas_uvec.resize(length);
	FLOAT_TYPE* alphas = beacls::UVec_<FLOAT_TYPE>(alphas_uvec).ptr();
	const levelset::HJI_Grid *hji_grid = get_grid();

	switch (dim) {
	case 0:
	{
		const beacls::FloatVec& xs2 = hji_grid->get_xs(2);
		for (size_t i = 0; i < length; ++i) {
			size_t dst_index = i + begin_index;
			if (Q.find(dst_index) != Q.end())
			{
				alphas[i] = vMax_ * std::abs(std::cos(xs2[dst_index]));
				alphas[i] += dMax_x_;
			}
			else
			{
				alphas[i] = 0;
			}
		}
	}
	break;
	case 1:
	{
		const beacls::FloatVec& xs2 = hji_grid->get_xs(2);
		for (size_t i = 0; i < length; ++i) {
			size_t dst_index = i + begin_index;
			if (Q.find(dst_index) != Q.end())
			{
				alphas[i] = vMax_ * std::abs(std::sin(xs2[dst_index]));
				alphas[i] += dMax_y_;
			}
			else
			{
				alphas[i] = 0;
			}
		}
	}
	break;
	case 2:
	{
		for (size_t i = 0; i < length; ++i) {
			size_t dst_index = i + begin_index;
			if (Q.find(dst_index) != Q.end())
			{
				alphas[i] = wMax_; 
				alphas[i] += dMax_theta_;
			}
			else
			{
				alphas[i] = 0;
			}
		}
	}
	break;
	default:
		printf("error:%s,%d\n", __func__, __LINE__);
		return false;
	}
	return true;
}
