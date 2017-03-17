
#ifndef __ShapeRectangleByCenter_impl_hpp__
#define __ShapeRectangleByCenter_impl_hpp__

#include <cstdint>
#include <vector>
#include <typedef.hpp>
namespace levelset {
	class HJI_Grid;

	class ShapeRectangleByCenter_impl {
	private:
		beacls::FloatVec center;
		beacls::FloatVec widths;
	public:
		ShapeRectangleByCenter_impl(
			const beacls::FloatVec& center,
			const beacls::FloatVec& widths);
		~ShapeRectangleByCenter_impl();
		bool execute(
			const HJI_Grid *grid, beacls::FloatVec& data) const;
		ShapeRectangleByCenter_impl* clone() const {
			return new ShapeRectangleByCenter_impl(*this);
		};
	private:
		/** @overload
		Disable operator=
		*/
		ShapeRectangleByCenter_impl& operator=(const ShapeRectangleByCenter_impl& rhs);
		/** @overload
		Disable copy constructor
		*/
		ShapeRectangleByCenter_impl(const ShapeRectangleByCenter_impl& rhs) :
			center(rhs.center),
			widths(rhs.widths)
		{}
	};
};
#endif	/* __ShapeRectangleByCenter_impl_hpp__ */

