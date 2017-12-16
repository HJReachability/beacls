
#include <levelset/InitialConditions/BasicShapes/ShapeCylinder.hpp>
#include <levelset/Grids/HJI_Grid.hpp>
#include "ShapeCylinder_impl.hpp"
#include <numeric>
#include <cmath>
using namespace levelset;

ShapeCylinder_impl::ShapeCylinder_impl(
	const beacls::IntegerVec& ignoreDims,
	const beacls::FloatVec& center,
	const FLOAT_TYPE radius ):

	ignoreDims(ignoreDims),
	center(center),
	radius(radius) {

}

ShapeCylinder_impl::~ShapeCylinder_impl() {
}


bool ShapeCylinder_impl::execute(const HJI_Grid *grid, beacls::FloatVec& data) 
  const {
	if (!grid) return false;
	beacls::FloatVec modified_center;
	size_t num_of_dimensions = grid->get_num_of_dimensions();

	if (center.empty()) {
		modified_center.resize(num_of_dimensions, 0.);
	}
	else if (center.size() == 1) {
		modified_center.resize(num_of_dimensions, center[0]);
	}
	else if (center.size() != num_of_dimensions) {
		modified_center = center;
		modified_center.resize(num_of_dimensions, center[0]);
	}
	else {
		modified_center = center;
	}

  beacls::IntegerVec N = grid->get_Ns();

	const size_t num_of_elements = grid->get_numel();

	data.assign(num_of_elements,0);
	const size_t begin_index = 0;
	const size_t length = 0;

	for (size_t dimension = 0; dimension < num_of_dimensions; ++dimension) {
		if (std::all_of(ignoreDims.cbegin(), ignoreDims.cend(), 
			[dimension](const auto& rhs) { return rhs != dimension; })) {
			beacls::UVec x_uvec;
			grid->get_xs(x_uvec, dimension, begin_index, length);
			const beacls::FloatVec* xs_ptr = beacls::UVec_<FLOAT_TYPE>(x_uvec).vec();
			FLOAT_TYPE center_d = modified_center[dimension];
			std::transform(xs_ptr->cbegin(), xs_ptr->cend(), data.begin(), data.begin(), ([center_d](const auto &xs_i, const auto &data_i) {
				return data_i + std::pow((xs_i - center_d), 2);
			}));
		}
	}
	std::transform(data.begin(), data.end(), data.begin(), 
		([this](const auto &data_i) {
		return std::sqrt(data_i) - radius;
	}));
	return true;

}

ShapeCylinder::ShapeCylinder(
  const beacls::IntegerVec& ignoreDims,
  const beacls::FloatVec& center,
  const FLOAT_TYPE radius
) {
  pimpl = new ShapeCylinder_impl(ignoreDims, center, radius);
}

ShapeCylinder::~ShapeCylinder() {
  if (pimpl) delete pimpl;
}

bool ShapeCylinder::execute(

	const HJI_Grid *grid, beacls::FloatVec& data) const {
	if (pimpl) {
		return pimpl->execute(grid, data);
	}
	else {
		return false;
	}
}
ShapeCylinder::ShapeCylinder(const ShapeCylinder& rhs): 
  pimpl(rhs.pimpl->clone()) {

};

ShapeCylinder* ShapeCylinder::clone() const {
  return new ShapeCylinder(*this);
};
