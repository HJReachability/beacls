
#ifndef __ShapeCylinder_impl_hpp__
#define __ShapeCylinder_impl_hpp__

#include <cstdint>
#include <vector>
#include <typedef.hpp>
class HJI_Grid;

class ShapeCylinder_impl {
private:
	beacls::IntegerVec ignoreDims;
	beacls::FloatVec center;
	FLOAT_TYPE radius;
public:
	ShapeCylinder_impl(
		const beacls::IntegerVec& ignoreDims,
		const beacls::FloatVec& center,
		const FLOAT_TYPE radius
	);
	~ShapeCylinder_impl();
	bool execute(
		const HJI_Grid *grid, beacls::FloatVec& data) const;
	ShapeCylinder_impl* clone() const {
		return new ShapeCylinder_impl(*this);
	};
private:
	/** @overload
	Disable operator=
	*/
	ShapeCylinder_impl& operator=(const ShapeCylinder_impl& rhs);
	/** @overload
	Disable copy constructor
	*/
	ShapeCylinder_impl(const ShapeCylinder_impl& rhs) :
		ignoreDims(rhs.ignoreDims),
		center(rhs.center),
		radius(rhs.radius)
	{}
};

#endif	/* __ShapeCylinder_impl_hpp__ */

