
#ifndef __ShapeRectangleByCenter_hpp__
#define __ShapeRectangleByCenter_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#include <cstdint>
#include <vector>
#include <typedef.hpp>
#include <levelset/InitialConditions/BasicShapes/BasicShape.hpp>
class HJI_Grid;

class ShapeRectangleByCenter_impl;

// shapeRectangleByCenter: implicit surface function for a(hyper)rectangle.
//
//   data = shapeRectangleByCenter(grid, center, widths)
//
// Creates an implicit surface function(close to signed distance) for a
// coordinate axis aligned(hyper)rectangle specified by its center and
// widths in each dimension.
//
// Can be used to create intervals and slabs by choosing components of the
// widths as + Inf.
//
// The default parameters for shapeRectangleByCenter and
// shapeRectangleByCorners produce different rectangles.
//
// Input Parameters :
//
//   grid : Grid structure(see processGrid.m for details).
//
//   center : Vector specifying center of rectangle.May be a scalar, in
//   which case the scalar is multiplied by a vector of ones of the
//   appropriate length.Defaults to 0 (eg centered at the origin).
//
//   widths: Vector specifying(full) widths of each side of the rectangle.
//   May be a scalar, in which case all dimensions have the same width.
//   Defaults to 1.
//
// Output Parameters :
//
//   data : Output data array(of size grid.size) containing the implicit
//   surface function.
//
// Copyright 2004 Ian M.Mitchell(mitchell@cs.ubc.ca).
// This software is used, copied and distributed under the licensing
//   agreement contained in the file LICENSE in the top directory of
//   the distribution.
//
// Ian Mitchell, 6 / 23 / 04
// $Date: 2009 - 09 - 03 16 : 34 : 07 - 0700 (Thu, 03 Sep 2009) $
// $Id : shapeRectangleByCenter.m 44 2009 - 09 - 03 23 : 34 : 07Z mitchell $
class ShapeRectangleByCenter : public BasicShape {
public:
	/**
	@brief Constructor
	@param	[in]		center	Vector specifying center of rectangle.  May be a scalar, in
							which case the scalar is multiplied by a vector of ones of the
							appropriate length.  Defaults to 0 (eg centered at the origin).
	@param	[in]		widths	Vector specifying (full) widths of each side of the rectangle.
							May be a scalar, in which case all dimensions have the same width.
							Defaults to 1.
	*/
	PREFIX_VC_DLL
		ShapeRectangleByCenter(
			const beacls::FloatVec& center,
			const beacls::FloatVec& widths
		);
	PREFIX_VC_DLL
		~ShapeRectangleByCenter();
	/**
	@brief	Creates an implicit surface function (close to signed distance) for a
			coordinate axis aligned (hyper)rectangle specified by its center and
			widths in each dimension.
	@param	[in]		grid	Grid structure
	@param	[out]		data	 Output data array (of size grid.size) containing the implicit surface function.
	@retval	true	succeeded.
	@retval false	failed.
	*/
	PREFIX_VC_DLL
		bool execute(
		const HJI_Grid *grid, beacls::FloatVec& data) const;
	ShapeRectangleByCenter* clone() const;
private:
	ShapeRectangleByCenter_impl *pimpl;

	/** @overload
	Disable operator=
	*/
	ShapeRectangleByCenter& operator=(const ShapeRectangleByCenter& rhs);
	/** @overload
	Disable copy constructor
	*/
	ShapeRectangleByCenter(const ShapeRectangleByCenter& rhs);
};

#endif	/* __ShapeRectangleByCenter_hpp__ */

