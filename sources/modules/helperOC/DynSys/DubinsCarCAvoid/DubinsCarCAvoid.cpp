#include <helperOC/DynSys/DubinsCarCAvoid/DubinsCarCAvoid.hpp>
using namespace helperOC;
DubinsCarCAvoid::DubinsCarCAvoid(
	const beacls::FloatVec& x,
	const FLOAT_TYPE uMax,
	const FLOAT_TYPE dMax,
	const FLOAT_TYPE va,
	const FLOAT_TYPE vb,
	const beacls::IntegerVec& dims) : Air3D(x,uMax,dMax,va,vb,dims) {
}
DubinsCarCAvoid::~DubinsCarCAvoid() {
}
