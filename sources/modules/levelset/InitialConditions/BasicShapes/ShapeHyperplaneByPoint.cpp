#include <levelset/InitialConditions/BasicShapes/ShapeHyperplaneByPoint.hpp>
#include <levelset/Grids/HJI_Grid.hpp>
#include <Core/interpn.hpp>
#include "ShapeHyperplaneByPoint_impl.hpp"
#include <macro.hpp>
#include <limits>
#include <functional>
using namespace levelset;

ShapeHyperplaneByPoint_impl::ShapeHyperplaneByPoint_impl(
	const std::vector<beacls::FloatVec >& points,
	const beacls::FloatVec& positivePoint) : 
	points(points),
	positivePoint(positivePoint)
	{
}
ShapeHyperplaneByPoint_impl::~ShapeHyperplaneByPoint_impl() {

}
bool ShapeHyperplaneByPoint_impl::execute(const HJI_Grid *grid, beacls::FloatVec& data) const {
	if (!grid) return false;
	//!< For the positivePoint parameter, what is "too close" to the interface?
	const FLOAT_TYPE small = static_cast<FLOAT_TYPE>(1e3 * std::numeric_limits<FLOAT_TYPE>::epsilon());

	const size_t check_positive_point = (positivePoint.empty()) ? 0 : 1;
	//!< Check that we have the correct number of points, and they are linearly independent.
	if (std::any_of(points.cbegin(), points.cend(), [&grid](const auto& rhs) {
		return rhs.size() != grid->get_num_of_dimensions();
	})) {
		std::cerr << "Number of points must be equal to grid dimension" << std::endl;
		return false;
	}

	//!< We single out the first point.  Lines from this point to all the others should lie on the hyperplane.
	beacls::FloatVec point0 = points[0];
	std::vector<beacls::FloatVec> A(points.size()-1);
	std::transform(points.cbegin() + 1, points.cend(), A.begin(), [&point0](const auto& rhs) { 
		beacls::FloatVec res(rhs.size());
		std::transform(rhs.cbegin(), rhs.cend(), point0.cbegin(), res.begin(), [](const auto& lhs, const auto& rhs) { return lhs - rhs; });
		return res;
	});
	
	//!< Extract the normal to the hyperplane.
	beacls::FloatVec normal(2);
	normal[0] = -sqrt(A[0][0] / (A[0][0] + A[0][1]));
	normal[1] = sqrt(A[0][1] / (A[0][0] + A[0][1]));
	//!< T.B.D.

	//!< Check to see that it is well defined.
	//!< T.B.D.

	const std::vector<beacls::FloatVec>& xss = grid->get_xss();
	std::vector<beacls::FloatVec> xss_minus_point0(xss.size());

	std::transform(xss.cbegin(), xss.cend(), point0.cbegin(), xss_minus_point0.begin(), [](const auto& lhs, const auto& rhs) {
		beacls::FloatVec res(lhs.size());
		std::transform(lhs.cbegin(), lhs.cend(), res.begin(), [rhs](const auto& lhs) { return lhs - rhs; });
		return res;
	});
	data.resize(xss[0].size(), 0);
	for (size_t i = 0; i < xss.size(); ++i) {
		const FLOAT_TYPE normal_i = normal[i];
		std::transform(data.cbegin(), data.cend(), xss_minus_point0[i].cbegin(), data.begin(), [normal_i](const auto& lhs, const auto& rhs) { return lhs + normal_i * rhs; });
	}
	//!< The procedure above generates a correct normal assuming that the data
	//!<  points are given in a clockwise fashion.If the user supplies
	//!<  parameter positivePoint, we need to use a different test.
	if (check_positive_point) {
		//!< eg.v = interpn(g.vs{ 1 }, g.vs{ 2 }, data, x(:, 1), x(:, 2), interp_method)

		std::vector<beacls::FloatVec> positivePoints(positivePoint.size());
		std::transform(positivePoint.cbegin(), positivePoint.cend(), positivePoints.begin(), [](const auto& rhs){ 
			return beacls::FloatVec{rhs};
		});

		std::vector<const beacls::FloatVec*> X_ptrs;
		std::vector<beacls::IntegerVec> Ns;

		X_ptrs.reserve(xss.size() * 2 + 1);
		Ns.reserve(xss.size() * 2 + 1);
		std::for_each(xss.cbegin(), xss.cend(), [&X_ptrs, &Ns, grid](const auto& rhs) {
			X_ptrs.push_back(&rhs);
			beacls::IntegerVec N{ rhs.size() };
			Ns.push_back(grid->get_Ns());
		});
		X_ptrs.push_back(&data);
		Ns.push_back(grid->get_Ns());
		std::for_each(positivePoints.cbegin(), positivePoints.cend(), [&X_ptrs, &Ns](const auto& rhs) {
			X_ptrs.push_back(&rhs);
			beacls::IntegerVec N{ rhs.size() };
			Ns.push_back(N);
		});

		beacls::FloatVec positivePointValue;
		beacls::interpn(positivePointValue, X_ptrs, Ns);

		if (std::isnan(positivePointValue[0])) {
			std::cerr <<"positivePoint must be within the bounds of the grid." << std::endl;
		}
		else if (std::abs(positivePointValue[0]) < small) {
			std::cerr << "positivePoint parameter is too close to the hyperplane." << std::endl;
		}
		else if (positivePointValue[0] < 0) {
			std::transform(data.cbegin(),data.cend(),data.begin(), std::negate<FLOAT_TYPE>());
		}

	}

	if (std::all_of(data.cbegin(), data.cend(), [](const auto& rhs) { return rhs < 0; }) ||
		std::all_of(data.cbegin(), data.cend(), [](const auto& rhs) { return rhs > 0; })) {
		std::cerr << "Implicit surface not visible because function has single sign on grid" << std::endl;
	}

	return true;
}


ShapeHyperplaneByPoint::ShapeHyperplaneByPoint(
	const std::vector<beacls::FloatVec >& points,
	const beacls::FloatVec& positivePoint){
	pimpl = new ShapeHyperplaneByPoint_impl(points, positivePoint);
}
ShapeHyperplaneByPoint::~ShapeHyperplaneByPoint() {
	if (pimpl) delete pimpl;
}
bool ShapeHyperplaneByPoint::execute(
	const HJI_Grid *grid, beacls::FloatVec& data
	) const {
	if (pimpl) return pimpl->execute(grid, data);
	else return false;
}
ShapeHyperplaneByPoint::ShapeHyperplaneByPoint(const ShapeHyperplaneByPoint& rhs) :
	pimpl(rhs.pimpl->clone())
{
};

ShapeHyperplaneByPoint* ShapeHyperplaneByPoint::clone() const {
	return new ShapeHyperplaneByPoint(*this);
};
