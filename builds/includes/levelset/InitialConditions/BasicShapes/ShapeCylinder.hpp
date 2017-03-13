
#ifndef __ShapeCylinder_hpp__
#define __ShapeCylinder_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#include <cstdint>
#include <cstddef>
#include <vector>
#include <typedef.hpp>
#include <levelset/InitialConditions/BasicShapes/BasicShape.hpp>
class HJI_Grid;

class ShapeCylinder_impl;

// shapeCylinder: implicit surface function for a cylinder.
//
//   data = shapeCylinder(grid, ignoreDims, center, radius)
//
// Creates an implicit surface function(actually signed distance) for a
//   coordinate axis aligned cylinder whose axis runs parallel to the
//   coordinate dimensions specified in ignoreDims.
//
// Can be used to create :
//   Intervals, circles and spheres(if ignoreDims is empty).
//   Slabs(if ignoreDims contains all dimensions except one).
//
// parameters :
// Input Parameters :
//
//   grid : Grid structure(see processGrid.m for details).
//
//   ignoreDims : Vector specifying indices of coordinate axes with which the
//   cylinder is aligned.Defaults to the empty vector(eg : the cylinder is
//   actually a sphere).
//
//   center : Vector specifying a point at the center of the cylinder.
//   Entries in the ignored dimensions are ignored.May be a scalar, in
//   which case the scalar is multiplied by a vector of ones of the
//   appropriate length.Defaults to 0 (eg centered at the origin).
//
//   radius: Scalar specifying the radius of the cylinder.Defaults to 1.
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
// $Date: 2011 - 03 - 18 16 : 56 : 16 - 0700 (Fri, 18 Mar 2011) $
// $Id : shapeCylinder.m 60 2011 - 03 - 18 23 : 56 : 16Z mitchell $

class ShapeCylinder : public BasicShape {
public:
	/**
	@brief Constructor
	@param	[in]		ignoreDims	Vector specifying indices of coordinate axes with which the
								cylinder is aligned.Defaults to the empty vector(eg : the cylinder is
								actually a sphere).
	@param	[in]		center	Vector specifying a point at the center of the cylinder.
							Entries in the ignored dimensions are ignored.May be a scalar, in 
							which case the scalar is multiplied by a vector of ones of the
							appropriate length.Defaults to 0 (eg centered at the origin).
	@param	[in]		radius	Scalar specifying the radius of the cylinder.Defaults to 1.
	*/
	PREFIX_VC_DLL
		ShapeCylinder(
			const beacls::IntegerVec& ignoreDims = beacls::IntegerVec(),
			const beacls::FloatVec& center = beacls::FloatVec(),
			const FLOAT_TYPE radius = 1
		);
	PREFIX_VC_DLL
		~ShapeCylinder();
	/**
	@brief	Creates an implicit surface function (actually signed distance) for a
			coordinate axis aligned cylinder whose axis runs parallel to the
			coordinate dimensions specified in ignoreDims.
	@param	[in]		grid	Grid structure
	@param	[out]		data	 Output data array (of size grid.size) containing the implicit surface function.
	@retval	true	succeeded.
	@retval false	failed.
	*/
	PREFIX_VC_DLL
		bool execute(
		const HJI_Grid *grid, beacls::FloatVec& data) const;
	ShapeCylinder* clone() const;
	/** @overload
	Disable copy constructor
	*/
	ShapeCylinder(const ShapeCylinder& rhs);
private:
	ShapeCylinder_impl *pimpl;

	/** @overload
	Disable operator=
	*/
	ShapeCylinder& operator=(const ShapeCylinder& rhs);
};

#endif	/* __ShapeCylinder_hpp__ */

