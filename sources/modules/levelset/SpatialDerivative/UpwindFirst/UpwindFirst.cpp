#include <levelset/SpatialDerivative/UpwindFirst/UpwindFirst.hpp>
#include <vector>
#include <cstdint>
#include <cmath>
#include <numeric>
#include <cstdio>
#include <macro.hpp>

bool levelset::checkEquivalentApprox(
	beacls::FloatVec& relErrors,
	beacls::FloatVec& absErrors,
	const beacls::FloatVec& approx1s,
	const beacls::FloatVec& approx2s,
	const FLOAT_TYPE bound
	) {

	if (approx1s.size() != approx2s.size()) return false;
	size_t approx1s_size = approx1s.size();

	beacls::FloatVec magnitude(approx1s_size);


	std::transform(approx1s.cbegin(), approx1s.cend(), approx2s.cbegin(), magnitude.begin(), ([](const auto& lhs, const auto& rhs) {
		return 0.5 * HjiFabs(lhs + rhs);
	}));

	if (absErrors.size() != approx1s_size) absErrors.resize(approx1s_size);
	std::transform(approx1s.cbegin(), approx1s.cend(), approx2s.cbegin(), absErrors.begin(), ([](const auto& lhs, const auto& rhs) {
		return HjiFabs(lhs - rhs);
	}));

	if (relErrors.size() != approx1s_size) relErrors.resize(approx1s_size);
	std::transform(magnitude.cbegin(), magnitude.cend(), absErrors.cbegin(), relErrors.begin(), ([bound](const auto& lhs, const auto& rhs) {
		return (lhs > bound) ? rhs / lhs : std::numeric_limits<FLOAT_TYPE>::signaling_NaN();
	}));
	const FLOAT_TYPE max_relError = std::inner_product(magnitude.cbegin(), magnitude.cend(), relErrors.cbegin(), (FLOAT_TYPE)0.,
		([](const auto &lhs, const auto &rhs) { return std::max<FLOAT_TYPE>(lhs, rhs); }),
		([bound](const auto &lhs, const auto &rhs) { return (lhs > bound) ? rhs : (FLOAT_TYPE)0. ; })
	);
	if (max_relError > bound) {
		printf("%s\n\t%lf exceeded relative bound %g\n", 
			"Error in supposedly equivalent derivative approximations",
			max_relError, bound);
	}
	const FLOAT_TYPE max_absError = std::inner_product(magnitude.cbegin(), magnitude.cend(), absErrors.cbegin(), (FLOAT_TYPE)0.,
		([](const auto &lhs, const auto &rhs) { return std::max<FLOAT_TYPE>(lhs, rhs); }),
		([bound](const auto &lhs, const auto &rhs) { return (lhs <= bound) ? rhs : (FLOAT_TYPE)0.; })
	);
	if (max_absError > bound) {
		printf("%s\n\t%lf exceeded absolute bound %g\n",
			"Error in supposedly equivalent derivative approximations",
			max_relError, bound);
	}

	return true;
}
bool levelset::checkEquivalentApprox(
	beacls::FloatVec& relErrors,
	beacls::FloatVec& absErrors,
	const beacls::UVec& approx1s,
	const beacls::UVec& approx2s,
	const FLOAT_TYPE bound
) {
	const FLOAT_TYPE* approx1s_ptr = beacls::UVec_<FLOAT_TYPE>(approx1s).ptr();
	const FLOAT_TYPE* approx2s_ptr = beacls::UVec_<FLOAT_TYPE>(approx2s).ptr();
	if (approx1s.size() != approx2s.size()) return false;
	size_t approx1s_size = approx1s.size();

	beacls::FloatVec magnitude(approx1s_size);

	for (size_t i = 0; i < approx1s_size; ++i) {
		magnitude[i] = (FLOAT_TYPE)( 0.5 * HjiFabs(approx1s_ptr[i] + approx2s_ptr[i]));
	}

	if (absErrors.size() != approx1s_size) absErrors.resize(approx1s_size);
	for (size_t i = 0; i < approx1s_size; ++i) {
		absErrors[i] = HjiFabs(approx1s_ptr[i] - approx2s_ptr[i]);
	}

	if (relErrors.size() != approx1s_size) relErrors.resize(approx1s_size);
	std::transform(magnitude.cbegin(), magnitude.cend(), absErrors.cbegin(), relErrors.begin(), ([bound](const auto& lhs, const auto& rhs) {
		return (lhs > bound) ? rhs / lhs : std::numeric_limits<FLOAT_TYPE>::signaling_NaN();
	}));
	const FLOAT_TYPE max_relError = std::inner_product(magnitude.cbegin(), magnitude.cend(), relErrors.cbegin(), (FLOAT_TYPE)0.,
		([](const auto &lhs, const auto &rhs) { return std::max<FLOAT_TYPE>(lhs, rhs); }),
		([bound](const auto &lhs, const auto &rhs) { return (lhs > bound) ? rhs : (FLOAT_TYPE)0.; })
	);
	if (max_relError > bound) {
		printf("%s\n\t%lf exceeded relative bound %g\n",
			"Error in supposedly equivalent derivative approximations",
			max_relError, bound);
	}
	const FLOAT_TYPE max_absError = std::inner_product(magnitude.cbegin(), magnitude.cend(), absErrors.cbegin(), (FLOAT_TYPE) 0.,
		([](const auto &lhs, const auto &rhs) { return std::max<FLOAT_TYPE>(lhs, rhs); }),
		([bound](const auto &lhs, const auto &rhs) { return (lhs <= bound) ? rhs : (FLOAT_TYPE)0.; })
	);
	if (max_absError > bound) {
		printf("%s\n\t%lf exceeded absolute bound %g\n",
			"Error in supposedly equivalent derivative approximations",
			max_relError, bound);
	}

	return true;
}