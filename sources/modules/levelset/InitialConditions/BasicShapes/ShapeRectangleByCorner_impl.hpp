#ifndef __ShapeRectangleByCorner_impl_hpp__
#define __ShapeRectangleByCorner_impl_hpp__

#include <cstdint>
#include <vector>
#include <typedef.hpp>
class HJI_Grid;
class ShapeRectangleByCorner_impl {
private:
	beacls::FloatVec lower;
	beacls::FloatVec upper;
public:
	ShapeRectangleByCorner_impl(
		const beacls::FloatVec& lower,
		const beacls::FloatVec& upper);
	~ShapeRectangleByCorner_impl();
	bool execute(
		const HJI_Grid *grid, beacls::FloatVec& data) const;
	ShapeRectangleByCorner_impl* clone() const {
		return new ShapeRectangleByCorner_impl(*this);
	};
private:
	/** @overload
	Disable operator=
	*/
	ShapeRectangleByCorner_impl& operator=(const ShapeRectangleByCorner_impl& rhs);
	/** @overload
	Disable copy constructor
	*/
	ShapeRectangleByCorner_impl(const ShapeRectangleByCorner_impl& rhs) :
		lower(rhs.lower),
		upper(rhs.upper)
	{}
};

#endif	/* __ShapeRectangleByCorner_impl_hpp__ */

