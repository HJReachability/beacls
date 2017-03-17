
#ifndef __ShapeRectangleByCorner_hpp__
#define __ShapeRectangleByCorner_hpp__

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
namespace levelset {
	class HJI_Grid;

	class ShapeRectangleByCorner_impl;

	// shapeRectangleByCorners: implicit surface function for a(hyper)rectangle.
	//
	//   data = shapeRectangleByCorners(grid, lower, upper)
	//
	// Creates an implicit surface function(close to signed distance)
	// for a coordinate axis aligned(hyper)rectangle specified by its
	//   lower and upper corners.
	//
	// Can be used to create intervals, slabs and other unbounded shapes
	//   by choosing components of the corners as + -Inf.
	//
	// The default parameters for shapeRectangleByCenter and
	//   shapeRectangleByCorners produce different rectangles.
	//
	// Input Parameters :
	//
	//   grid : Grid structure(see processGrid.m for details).
	//
	//   lower : Vector specifying the lower corner.May be a scalar, in
	//   which case the scalar is multiplied by a vector of ones of the
	//   appropriate length.Defaults to 0.
	//
	//   upper: Vector specifying the upper corner.May be a scalar, in which
	//   case the scalar is multiplied by a vector of ones of the appropriate
	//   length.Defaults to 1.  Note that all(lower < upper) must hold,
	//   otherwise the implicit surface will be empty.
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
	// $Id : shapeRectangleByCorners.m 44 2009 - 09 - 03 23 : 34 : 07Z mitchell $
	class ShapeRectangleByCorner : public BasicShape {
	public:
		/**
		@brief Constructor
		@param	[in]		lower	Vector specifying the lower corner.  May be a scalar, in
								which case the scalar is multiplied by a vector of ones of the
								appropriate length.  Defaults to 0.
		@param	[in]		upper	Vector specifying the upper corner.  May be a scalar, in which
								case the scalar is multiplied by a vector of ones of the appropriate
								length.  Defaults to 1.  Note that all(lower < upper) must hold,
								otherwise the implicit surface will be empty.
								Defaults to 1.
		*/
		PREFIX_VC_DLL
			ShapeRectangleByCorner(
				const beacls::FloatVec& lower,
				const beacls::FloatVec& upper
			);
		PREFIX_VC_DLL
			~ShapeRectangleByCorner();
		/**
		@brief	Creates an implicit surface function (close to signed distance)
				for a coordinate axis aligned (hyper)rectangle specified by its
				lower and upper corners.
		@param	[in]		grid	Grid structure
		@param	[out]		data	 Output data array (of size grid.size) containing the implicit surface function.
		@retval	true	succeeded.
		@retval false	failed.
		*/
		PREFIX_VC_DLL
			bool execute(
				const HJI_Grid *grid, beacls::FloatVec& data) const;
		ShapeRectangleByCorner* clone() const;
	private:
		ShapeRectangleByCorner_impl *pimpl;

		/** @overload
		Disable operator=
		*/
		ShapeRectangleByCorner& operator=(const ShapeRectangleByCorner& rhs);
		/** @overload
		Disable copy constructor
		*/
		ShapeRectangleByCorner(const ShapeRectangleByCorner& rhs);
	};
};
#endif	/* __ShapeRectangleByCorner_hpp__ */

