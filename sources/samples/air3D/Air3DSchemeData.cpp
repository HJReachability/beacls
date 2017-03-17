#include "Air3DSchemeData.hpp"
#include <algorithm>
#include <typeinfo>
#include <macro.hpp>
bool Air3DSchemeData::operator==(const Air3DSchemeData& rhs) const {
	if (this == &rhs) return true;
	else if (!SchemeData::operator==(rhs)) return false;
	else if (velocityA != rhs.velocityA) return false;
	else if (velocityB != rhs.velocityB) return false;
	else if (inputA != rhs.inputA) return false;
	else if (inputB != rhs.inputB) return false;
	else return true;
}
bool Air3DSchemeData::operator==(const SchemeData& rhs) const {
	if (this == &rhs) return true;
	else if (typeid(*this) != typeid(rhs)) return false;
	else return operator==(dynamic_cast<const Air3DSchemeData&>(rhs));
}
bool Air3DSchemeData::hamFunc(
	beacls::UVec& hamValue_uvec,
	const FLOAT_TYPE,
	const beacls::UVec&,
	const std::vector<beacls::UVec>& derivs,
	const size_t begin_index,
	const size_t length
	) const {
	const levelset::HJI_Grid *hji_grid = get_grid();
	const beacls::FloatVec &xs0 = hji_grid->get_xs(0);
	const beacls::FloatVec &xs1 = hji_grid->get_xs(1);
	const beacls::FloatVec &xs2 = hji_grid->get_xs(2);
	beacls::reallocateAsSrc(hamValue_uvec, derivs[0]);
	FLOAT_TYPE* hamValue = beacls::UVec_<FLOAT_TYPE>(hamValue_uvec).ptr();
	const FLOAT_TYPE* deriv0 = beacls::UVec_<FLOAT_TYPE>(derivs[0]).ptr();
	const FLOAT_TYPE* deriv1 = beacls::UVec_<FLOAT_TYPE>(derivs[1]).ptr();
	const FLOAT_TYPE* deriv2 = beacls::UVec_<FLOAT_TYPE>(derivs[2]).ptr();
	for (size_t i = 0; i<length; ++i) {
		FLOAT_TYPE deriv0_i = deriv0[i];
		FLOAT_TYPE deriv1_i = deriv1[i];
		FLOAT_TYPE deriv2_i = deriv2[i];
		FLOAT_TYPE xs0_i = xs0[begin_index + i];
		FLOAT_TYPE xs1_i = xs1[begin_index + i];
		FLOAT_TYPE xs2_i = xs2[begin_index + i];
		FLOAT_TYPE cos_xs2_i = std::cos(xs2_i);
		FLOAT_TYPE sin_xs2_i = std::sin(xs2_i);
		hamValue[i] = -(-velocityA * deriv0_i
			+ velocityB * cos_xs2_i * deriv0_i
			+ velocityB * sin_xs2_i * deriv1_i
			+ inputA * HjiFabs(xs1_i * deriv0_i
				- xs0_i * deriv1_i - deriv2_i)
			- inputB * HjiFabs(deriv2_i));
	}
	return true;

}
bool Air3DSchemeData::partialFunc(
	beacls::UVec& alphas_uvec,
	const FLOAT_TYPE,
	const beacls::UVec&,
	const std::vector<beacls::UVec>&,
	const std::vector<beacls::UVec>&,
	const size_t dim,
	const size_t begin_index,
	const size_t length
) const {
	if (alphas_uvec.type() != beacls::UVecType_Vector) alphas_uvec = beacls::UVec(beacls::type_to_depth<FLOAT_TYPE>(), beacls::UVecType_Vector, length);
	else alphas_uvec.resize(length);
	FLOAT_TYPE* alphas = beacls::UVec_<FLOAT_TYPE>(alphas_uvec).ptr();
	const levelset::HJI_Grid *hji_grid = get_grid();
	FLOAT_TYPE iA = this->inputA;
	switch (dim) {
	case 0:
	{
		FLOAT_TYPE vA = this->velocityA;
		FLOAT_TYPE vB = this->velocityB;
		const beacls::FloatVec &xs1 = hji_grid->get_xs(1);
		const beacls::FloatVec &xs2 = hji_grid->get_xs(2);
		for (size_t i = 0; i<length; ++i) {
			alphas[i] = HjiFabs(-vA
				+ vB * std::cos(xs2[i+ begin_index]))
				+ iA * HjiFabs<FLOAT_TYPE>(xs1[i+ begin_index]);
		}
	}
	break;
	case 1:
	{
		FLOAT_TYPE vB = this->velocityB;
		const beacls::FloatVec &xs0 = hji_grid->get_xs(0);
		const beacls::FloatVec &xs2 = hji_grid->get_xs(2);
		for (size_t i = 0; i<length; ++i) {
			alphas[i] = HjiFabs(vB * std::sin(xs2[i+begin_index]))
				+ iA * HjiFabs<FLOAT_TYPE>(xs0[i + begin_index]);
		}
	}
	break;
	case 2:
	{
		FLOAT_TYPE tmp = inputA + inputB;
		for (size_t i = 0; i<length; ++i) {
			alphas[i] = tmp;
		}
	}
	break;
	default:
		printf("error:%s,%dPartials for the game of two identical vehicles only exist in dimensions 1-3\n", __func__, __LINE__);
		return false;
	}
	return true;

}
