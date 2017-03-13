
#include <levelset/InitialConditions/BasicShapes/ShapeSphere.hpp>
#include <levelset/InitialConditions/BasicShapes/ShapeCylinder.hpp>
#include <levelset/Grids/HJI_Grid.hpp>

ShapeSphere::ShapeSphere(
	const beacls::FloatVec& center,
	const FLOAT_TYPE radius
) : ShapeCylinder(beacls::IntegerVec(), center, radius) {
}
ShapeSphere::~ShapeSphere() {
}
bool ShapeSphere::execute(
	const HJI_Grid *grid, beacls::FloatVec& data
	) const {
	return ShapeCylinder::execute(grid, data);
}
ShapeSphere::ShapeSphere(const ShapeSphere& rhs) :
	ShapeCylinder(rhs)
{
};

ShapeSphere* ShapeSphere::clone() const {
	return new ShapeSphere(*this);
};
