#include "IgsolrSchemeData.hpp"
#include <algorithm>
#include <vector>
#include <typeinfo>

IgsolrSchemeData::IgsolrSchemeData(
	const beacls::IntegerVec &gridcounts,
	const size_t u_bound_min,
	const size_t u_bound_max,
	const FLOAT_TYPE hover_counts,
	const FLOAT_TYPE k_p,
	const FLOAT_TYPE disturbance_rate,
	const FLOAT_TYPE default_d_max,
	const FLOAT_TYPE min_x,
	const FLOAT_TYPE max_x,
	const FLOAT_TYPE min_v,
	const FLOAT_TYPE max_v,
	const FLOAT_TYPE w,
	const beacls::FloatVec& lipschitz,
	const beacls::FloatVec& impose_d_bounds
	) : 
	SchemeData(),
	g((FLOAT_TYPE)-9.8),
	k_p(k_p),
	u_bound_min(u_bound_min),
	u_bound_max(u_bound_max),
	disturbance_rate(disturbance_rate),
	default_d_max(default_d_max),
	default_d_min(-default_d_max),
	w(w),
	lipschitz(lipschitz),
	impose_d_bounds(impose_d_bounds)
{

	// For convenience and numerical accuracy, we map U in [0 2200]
	// to a scaled input u in [0 10]
	FLOAT_TYPE ubound2u = (FLOAT_TYPE)(1.0 / u_bound_max * 10.0); // Transforms PWM counts to a 0-10 scale

	u_min = u_bound_min * ubound2u;
	u_max = u_bound_max * ubound2u;

	// Motor thrust gain (u to acceleration)
	//k_T = -g  / (hover_counts*U2u);     // Linear model
	k_t = -g / ((hover_counts*ubound2u)*(hover_counts*ubound2u));   // Quadratic dependence

	// Disturbance in acceleration

	// State space grid
	size_t num_of_elements = 1;
	for_each(gridcounts.cbegin(), gridcounts.cend(), ([&num_of_elements](size_t gridcount) {num_of_elements *= gridcount; }));
	d_min.assign(num_of_elements, default_d_min);
	d_max.assign(num_of_elements, default_d_max);

	FLOAT_TYPE tick_x = (max_x - min_x) / (FLOAT_TYPE)gridcounts[0];

	FLOAT_TYPE tick_v = (max_v - min_v) / (FLOAT_TYPE)gridcounts[1];

	FLOAT_TYPE min_th = (FLOAT_TYPE)(u_min - 0.5);      // thrust
	FLOAT_TYPE max_th = (FLOAT_TYPE)(u_max + 0.5);
	FLOAT_TYPE tick_th = (max_th - min_th) / (FLOAT_TYPE)gridcounts[2];

	// Defines grid per dimension and appends grid size at the end

	size_t xs_size = gridcounts[0] + 1;
	size_t vs_size = gridcounts[1] + 1;
	size_t ths_size = gridcounts[2] + 1;
	xs.resize(xs_size);
	vs.resize(vs_size);
	ths.resize(ths_size);

	// TODO:avoid over flow 
	FLOAT_TYPE tmp;
	tmp = min_x;
	for (size_t i = 0; i<xs_size; ++i) {
		xs[i] = tmp;
		tmp += tick_x;
	}
	tmp = min_v;
	for (size_t i = 0; i<vs_size; ++i) {
		vs[i] = tmp;
		tmp += tick_v;
	}
	tmp = min_th;
	printf("%f\n", min_th);
	for (size_t i = 0; i<ths_size; ++i) {
		ths[i] = tmp;
		tmp += tick_th;
	}


	// JFF - Gaussian Process init
	// Disturbance properties defined as part of model premises
	// Absolute global disturbance bound
	D_max_global = 1 / 2 * (default_d_max - default_d_min);
}
bool IgsolrSchemeData::operator==(const IgsolrSchemeData& rhs) const {
	if (this == &rhs) return true;
	else if (!SchemeData::operator==(rhs)) return false;
	else if (g != rhs.g) return false;
	else if (k_p != rhs.k_p) return false;
	else if (k_t != rhs.k_t) return false;
	else if (u_max != rhs.u_max) return false;
	else if (u_min != rhs.u_min) return false;
	else if (u_bound_min != rhs.u_bound_min) return false;
	else if (u_bound_max != rhs.u_bound_max) return false;
	else if (disturbance_rate != rhs.disturbance_rate) return false;
	else if (default_d_max != rhs.default_d_max) return false;
	else if (default_d_min != rhs.default_d_min) return false;
	else if (w != rhs.w) return false;
	else if (D_max_global != rhs.D_max_global) return false;
	else if ((lipschitz.size() != rhs.lipschitz.size()) || !std::equal(lipschitz.cbegin(), lipschitz.cend(), rhs.lipschitz.cbegin())) return false;
	else if ((impose_d_bounds.size() != rhs.impose_d_bounds.size()) || !std::equal(impose_d_bounds.cbegin(), impose_d_bounds.cend(), rhs.impose_d_bounds.cbegin())) return false;
	else if ((d_min.size() != rhs.d_min.size()) || !std::equal(d_min.cbegin(), d_min.cend(), rhs.d_min.cbegin())) return false;
	else if ((d_max.size() != rhs.d_max.size()) || !std::equal(d_max.cbegin(), d_max.cend(), rhs.d_max.cbegin())) return false;
	else if ((xs.size() != rhs.xs.size()) || !std::equal(xs.cbegin(), xs.cend(), rhs.xs.cbegin())) return false;
	else if ((vs.size() != rhs.vs.size()) || !std::equal(vs.cbegin(), vs.cend(), rhs.vs.cbegin())) return false;
	else if ((ths.size() != rhs.ths.size()) || !std::equal(ths.cbegin(), ths.cend(), rhs.ths.cbegin())) return false;
	else return true;
}
bool IgsolrSchemeData::operator==(const SchemeData& rhs) const {
	if (this == &rhs) return true;
	else if (typeid(*this) != typeid(rhs)) return false;
	else return operator==(dynamic_cast<const IgsolrSchemeData&>(rhs));
}
bool IgsolrSchemeData::hamFunc(
	beacls::UVec& hamValue_uvec,
	const FLOAT_TYPE,
	const beacls::UVec& ,
	const std::vector<beacls::UVec>& derivs,
	const size_t begin_index,
	const size_t length
	) const {
	const levelset::HJI_Grid *hji_grid = get_grid();
	const beacls::FloatVec &xs1 = hji_grid->get_xs(1);
	const beacls::FloatVec &xs2 = hji_grid->get_xs(2);
	beacls::reallocateAsSrc(hamValue_uvec, derivs[0]);
	FLOAT_TYPE* hamValue = beacls::UVec_<FLOAT_TYPE>(hamValue_uvec).ptr();
	const FLOAT_TYPE* deriv0 = beacls::UVec_<FLOAT_TYPE>(derivs[0]).ptr();
	const FLOAT_TYPE* deriv1 = beacls::UVec_<FLOAT_TYPE>(derivs[1]).ptr();
	const FLOAT_TYPE* deriv2 = beacls::UVec_<FLOAT_TYPE>(derivs[2]).ptr();

	for (size_t i = 0; i<length; ++i) {
		FLOAT_TYPE deriv2_i = deriv2[i];
		FLOAT_TYPE kp_deriv2 = k_p * deriv2_i;
		FLOAT_TYPE u_limit_value = (kp_deriv2 >= 0) ? u_max : u_min;
		FLOAT_TYPE deriv1_i = deriv1[i];
		FLOAT_TYPE d_limit_value = (deriv1_i >= 0) ? d_min[begin_index + i] : d_max[begin_index + i];
		FLOAT_TYPE xs2_i = xs2[begin_index + i];

		FLOAT_TYPE deriv0_i = deriv0[i];
		FLOAT_TYPE pre_calc_value1 = k_t * xs2_i * xs2_i + g;
		FLOAT_TYPE pre_calc_value2 = k_p * xs2_i;
		hamValue[i] = -(xs1[begin_index + i] * deriv0_i
			+ pre_calc_value1 * deriv1_i
			+ -pre_calc_value2 * deriv2_i
			+ kp_deriv2 * u_limit_value
			+ deriv1_i * d_limit_value);
	}
	return true;

}
bool IgsolrSchemeData::partialFunc(
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
		const beacls::FloatVec& xs1 = hji_grid->get_xs(1);
		for (size_t i = 0; i<length; ++i) {
			alphas[i] = HjiFabs<FLOAT_TYPE>(xs1[begin_index + i]);
		}
	}
	break;
	case 1:
	{
		const beacls::FloatVec& xs2 = hji_grid->get_xs(2);
		for (size_t i = 0; i<length; ++i) {
			const FLOAT_TYPE tmp = HjiMax(HjiFabs<FLOAT_TYPE>(d_max[begin_index + i]), HjiFabs<FLOAT_TYPE>(d_min[begin_index + i]));
			const FLOAT_TYPE xs2_i = xs2[begin_index + i];
			const FLOAT_TYPE tmp2 = k_t * xs2_i * xs2_i + g;
			alphas[i] = HjiFabs<FLOAT_TYPE>(tmp2) + tmp;
		}
	}
	break;
	case 2:
	{
		const beacls::FloatVec& xs2 = hji_grid->get_xs(2);
		FLOAT_TYPE fabs_kp = HjiFabs<FLOAT_TYPE>(k_p);
		const FLOAT_TYPE tmp = HjiMax(HjiFabs<FLOAT_TYPE>(u_max), HjiFabs<FLOAT_TYPE>(u_min));
		for (size_t i = 0; i<length; ++i) {
			const FLOAT_TYPE tmp2 = k_p * xs2[begin_index + i];
			alphas[i] = HjiFabs<FLOAT_TYPE>(tmp2) + fabs_kp * tmp;
		}
	}
	break;
	default:
		printf("error:%s,%d\n", __func__, __LINE__);
		return false;
	}
	return true;

}

