#include "Plane4DSchemeData.hpp"
#include <algorithm>
#include <typeinfo>
#include <macro.hpp>
bool Plane4DSchemeData::operator==(const Plane4DSchemeData& rhs) const {
	if (this == &rhs) return true;
	else if (!SchemeData::operator==(rhs)) return false;
	else if (wMax != rhs.wMax) return false;
	else if ((aranges.size() != rhs.aranges.size()) || !std::equal(aranges.cbegin(), aranges.cend(), rhs.aranges.cbegin())) return false;
	else return true;
}
bool Plane4DSchemeData::operator==(const SchemeData& rhs) const {
	if (this == &rhs) return true;
	else if (typeid(*this) != typeid(rhs)) return false;
	else return operator==(dynamic_cast<const Plane4DSchemeData&>(rhs));
}

bool Plane4DSchemeData::hamFunc(
	beacls::UVec& hamValue_uvec,
	const FLOAT_TYPE,
	const beacls::UVec&,
	const std::vector<beacls::UVec>& derivs,
	const size_t begin_index,
	const size_t length
	) const {
	const levelset::HJI_Grid *hji_grid = get_grid();

#if 0
	const beacls::FloatVec &xs2 = hji_grid->get_xs(2);
	const beacls::FloatVec &xs3 = hji_grid->get_xs(3);
#else
	beacls::UVec x2_uvec;
	beacls::UVec x3_uvec;
	hji_grid->get_xs(x2_uvec, 2, begin_index, length);
	hji_grid->get_xs(x3_uvec, 3, begin_index, length);
	const FLOAT_TYPE* xs2_ptr = beacls::UVec_<const FLOAT_TYPE>(x2_uvec).ptr();
	const FLOAT_TYPE* xs3_ptr = beacls::UVec_<const FLOAT_TYPE>(x3_uvec).ptr();
#endif
	beacls::reallocateAsSrc(hamValue_uvec, derivs[0]);
	FLOAT_TYPE* hamValue = beacls::UVec_<FLOAT_TYPE>(hamValue_uvec).ptr();
	const FLOAT_TYPE* deriv0 = beacls::UVec_<FLOAT_TYPE>(derivs[0]).ptr();
	const FLOAT_TYPE* deriv1 = beacls::UVec_<FLOAT_TYPE>(derivs[1]).ptr();
	const FLOAT_TYPE* deriv2 = beacls::UVec_<FLOAT_TYPE>(derivs[2]).ptr();
	const FLOAT_TYPE* deriv3 = beacls::UVec_<FLOAT_TYPE>(derivs[3]).ptr();

	for (size_t i = 0; i<length; ++i) {
		FLOAT_TYPE deriv0_i = deriv0[i];
		FLOAT_TYPE deriv1_i = deriv1[i];
		FLOAT_TYPE deriv2_i = deriv2[i];
		FLOAT_TYPE deriv3_i = deriv3[i];
#if 0
		FLOAT_TYPE xs2_i = xs2[begin_index + i];
		FLOAT_TYPE xs3_i = xs3[begin_index + i];
#else
		FLOAT_TYPE xs2_i = xs2_ptr[i];
		FLOAT_TYPE xs3_i = xs3_ptr[i];
#endif
		FLOAT_TYPE tmp = (deriv3_i >= 0) ? deriv3_i * aranges[0] : deriv3_i * aranges[1];
		hamValue[i] = -(deriv0_i * (xs3_i * cos(xs2_i)) +
			deriv1_i * (xs3_i * sin(xs2_i)) +
			-wMax * HjiFabs(deriv2_i) +
			tmp);
	}
	return true;
}
bool Plane4DSchemeData::partialFunc(
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
	switch (dim) {
	case 0:
	{
#if 0
		const beacls::FloatVec &xs2 = hji_grid->get_xs(2);
		const beacls::FloatVec &xs3 = hji_grid->get_xs(3);
		for (size_t i = 0; i<length; ++i) {
			alphas[i] = HjiFabs(xs3[begin_index + i] * cos(xs2[begin_index + i]));
		}
#else
		beacls::UVec x2_uvec;
		beacls::UVec x3_uvec;
		hji_grid->get_xs(x2_uvec, 2, begin_index, length);
		hji_grid->get_xs(x3_uvec, 3, begin_index, length);
		const FLOAT_TYPE* xs2_ptr = beacls::UVec_<const FLOAT_TYPE>(x2_uvec).ptr();
		const FLOAT_TYPE* xs3_ptr = beacls::UVec_<const FLOAT_TYPE>(x3_uvec).ptr();
		for (size_t i = 0; i<length; ++i) {
			alphas[i] = HjiFabs(xs3_ptr[i] * cos(xs2_ptr[i]));
		}
#endif
	}
	break;
	case 1:
	{
#if 0
		const beacls::FloatVec &xs2 = hji_grid->get_xs(2);
		const beacls::FloatVec &xs3 = hji_grid->get_xs(3);
		for (size_t i = 0; i<length; ++i) {
			alphas[i] = HjiFabs(xs3[begin_index + i] * sin(xs2[begin_index + i]));
		}
#else
		beacls::UVec x2_uvec;
		beacls::UVec x3_uvec;
		hji_grid->get_xs(x2_uvec, 2, begin_index, length);
		hji_grid->get_xs(x3_uvec, 3, begin_index, length);
		const FLOAT_TYPE* xs2_ptr = beacls::UVec_<const FLOAT_TYPE>(x2_uvec).ptr();
		const FLOAT_TYPE* xs3_ptr = beacls::UVec_<const FLOAT_TYPE>(x3_uvec).ptr();
		for (size_t i = 0; i<length; ++i) {
			alphas[i] = HjiFabs(xs3_ptr[i] * sin(xs2_ptr[i]));
		}
#endif
	}
	break;
	case 2:
	{
		for (size_t i = 0; i<length; ++i) {
			alphas[i] = wMax;
		}
	}
	break;
	case 3:
	{
		FLOAT_TYPE tmp = HjiMax(HjiFabs(aranges[0]), HjiFabs(aranges[1]));
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
