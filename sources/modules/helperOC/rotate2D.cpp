#include <helperOC/rotate2D.hpp>
#include <cuda_macro.hpp>
#include <iostream>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <numeric>
#include <limits>

bool helperOC::rotate2D(
	beacls::FloatVec& vOut,
	const beacls::FloatVec& vIn,
	const FLOAT_TYPE theta
) {
	beacls::FloatVec tmp;
	beacls::FloatVec& tmpOut = (&vIn == &vOut) ? tmp : vOut;
	//!< Rotate matrix and preserve vector size
	tmpOut.resize(vIn.size(), 0);
	const FLOAT_TYPE cos_theta = cos_float_type(theta);
	const FLOAT_TYPE sin_theta = sin_float_type(theta);
	std::transform(vIn.cbegin(), vIn.cbegin() + vIn.size() / 2, vIn.cbegin() + vIn.size() / 2, tmpOut.begin(), [&cos_theta, &sin_theta](const auto& lhs, const auto& rhs) {
		return cos_theta * lhs - sin_theta * rhs;
	});
	std::transform(vIn.cbegin(), vIn.cbegin() + vIn.size() / 2, vIn.cbegin() + vIn.size() / 2, tmpOut.begin() + tmpOut.size() / 2, [&cos_theta, &sin_theta](const auto& lhs, const auto& rhs) {
		return sin_theta * lhs + cos_theta * rhs;
	});
	vOut = tmpOut;
	return true;
}

