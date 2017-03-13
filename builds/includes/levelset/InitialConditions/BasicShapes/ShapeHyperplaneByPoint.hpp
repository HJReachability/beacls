#ifndef __ShapeHyperplaneByPoint_hpp__
#define __ShapeHyperplaneByPoint_hpp__

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

class ShapeHyperplaneByPoint_impl;

// shapeHyperplaneByPoints: implicit surface function for a hyperplane.
//
//   data = shapeHyperplaneByPoints(grid, points, positivePoint)
//
// Creates a signed distance function for a hyperplane.  Unlike
// shapeHyperplane, this version accepts a list of grid.dim points which lie
// on the hyperplane.
//
// The direction of the normal (which determines which side of the hyperplane
// has positive values) is determined by one of two methods:
//
//   1) If the parameter positivePoint is provided, then the normal
//   direction is chosen so that the value at this point is positive.
//
//   2) If the parameter positivePoint is not provided, then it is assumed
//   that the points defining the hyperplane are given in "clockwise" order
//   if the normal points out of the "clock".  This method does not work
//   in 2D.
//
// Input Parameters:
//
//   grid: Grid structure (see processGrid.m for details).
//
//   points: Matrix specifying the points through which the hyperplane should
//   pass.  Each row is one point.  This matrix must be square of dimension
//   grid.dim.
//
//   positivePoint: Vector of length grid.dim specifying a point which lies
//   on the positive side of the interface.  This point should be within the
//   bounds of the grid.  Optional.  The method for determining the normal
//   direction to the hyperplane depends on whether this parameter is
//   supplied; see the discussion above for more details.  It is an error if
//   this point lies on the hyperplane defined by the other points.
//
// Output Parameters:
//
//   data: Output data array (of size grid.size) containing the implicit
//   surface function for the hyperplane.

// Copyright 2007 Ian M. Mitchell (mitchell@cs.ubc.ca).
// This software is used, copied and distributed under the licensing 
//   agreement contained in the file LICENSE in the top directory of 
//   the distribution.
//
// Ian Mitchell, 3/29/05
// Modified to add positivePoint option, Ian Mitchell 5/26/07
// $Date: 2009-09-03 16:34:07 -0700 (Thu, 03 Sep 2009) $
// $Id: shapeHyperplaneByPoints.m 44 2009-09-03 23:34:07Z mitchell $

class ShapeHyperplaneByPoint : public BasicShape {
public:
	/**
	@brief Constructor
	@param	[in]		points			Matrix specifying the points through which the hyperplane should
										pass.  Each row is one point.  This matrix must be square of dimension
										grid.dim.
	@param	[in]		positivePoint	Vector of length grid.dim specifying a point which lies
										on the positive side of the interface.  This point should be within the
										bounds of the grid.  Optional.  The method for determining the normal
										direction to the hyperplane depends on whether this parameter is
										supplied; see the discussion above for more details.  It is an error if
										this point lies on the hyperplane defined by the other points.
	*/
	PREFIX_VC_DLL
		ShapeHyperplaneByPoint(
			const std::vector<beacls::FloatVec >& points,
			const beacls::FloatVec& positivePoint
		);
	PREFIX_VC_DLL
		~ShapeHyperplaneByPoint();
	/**
	@brief	Creates an implicit surface function (close to signed distance) 
			for a coordinate axis aligned (hyper)rectangle specified by its
			lower and upper corners.
	@param	[in]		grid	Grid structure (see processGrid.m for details).
	@param	[out]		data	Output data array (of size grid.size) containing the implicit
								surface function for the hyperplane.
	@retval	true	succeeded.
	@retval false	failed.
	*/
	PREFIX_VC_DLL
		bool execute(
		const HJI_Grid *grid, beacls::FloatVec& data) const;
	ShapeHyperplaneByPoint* clone() const;
private:
	ShapeHyperplaneByPoint_impl *pimpl;

	/** @overload
	Disable operator=
	*/
	ShapeHyperplaneByPoint& operator=(const ShapeHyperplaneByPoint& rhs);
	/** @overload
	Disable copy constructor
	*/
	ShapeHyperplaneByPoint(const ShapeHyperplaneByPoint& rhs);
};

#endif	/* __ShapeHyperplaneByPoint_hpp__ */

