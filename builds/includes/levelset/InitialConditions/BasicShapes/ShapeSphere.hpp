
#ifndef __ShapeSphere_hpp__
#define __ShapeSphere_hpp__

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
#include <levelset/InitialConditions/BasicShapes/ShapeCylinder.hpp>
namespace levelset {
	class HJI_Grid;
	class ShapeSphere_impl;

	// shapeSphere: implicit surface function for a sphere.
	//
	//   data = shapeSphere(grid, center, radius)
	//
	// Creates an implicit surface function (actually signed distance) 
	//   for a sphere.
	//
	// Can be used to create circles in 2D or intervals in 1D.
	//
	//
	// Input Parameters:
	//
	//   grid: Grid structure (see processGrid.m for details).
	//
	//   center: Vector specifying center of sphere.  May be a scalar, in
	//   which case the scalar is multiplied by a vector of ones of the
	//   appropriate length.  Defaults to 0 (eg centered at the origin).
	//
	//   radius: Scalar specifying the radius of the sphere. Defaults to 1.
	//
	// Output Parameters:
	//
	//   data: Output data array (of size grid.size) containing the implicit
	//   surface function.

	// Copyright 2004 Ian M. Mitchell (mitchell@cs.ubc.ca).
	// This software is used, copied and distributed under the licensing 
	//   agreement contained in the file LICENSE in the top directory of 
	//   the distribution.
	//
	// Ian Mitchell, 6/23/04
	// $Date: 2009-09-03 16:34:07 -0700 (Thu, 03 Sep 2009) $
	// $Id: shapeSphere.m 44 2009-09-03 23:34:07Z mitchell $


	class ShapeSphere : public ShapeCylinder {
	public:
		/**
		@brief Constructor
		@param	[in]	center	Vector specifying center of sphere.  May be a scalar, in
								which case the scalar is multiplied by a vector of ones of the
								appropriate length.  Defaults to 0 (eg centered at the origin).
		@param	[in]	radius	Scalar specifying the radius of the sphere. Defaults to 1.
		*/
		PREFIX_VC_DLL
			ShapeSphere(
				const beacls::FloatVec& center = beacls::FloatVec(),
				const FLOAT_TYPE radius = 1.
			);
		PREFIX_VC_DLL
			~ShapeSphere();
		/**
		@brief	Creates an implicit surface function (actually signed distance)
				for a sphere.
				Can be used to create circles in 2D or intervals in 1D.
		@param	[in]	grid	Grid structure
		@param	[out]	data	 Output data array (of size grid.size) containing the implicit surface function.
		@retval	true	succeeded.
		@retval false	failed.
		*/
		PREFIX_VC_DLL
			bool execute(
				const HJI_Grid *grid, beacls::FloatVec& data) const;
		ShapeSphere* clone() const;
	private:
		ShapeSphere_impl *pimpl;

		/** @overload
		Disable operator=
		*/
		ShapeSphere& operator=(const ShapeSphere& rhs);
		/** @overload
		Disable copy constructor
		*/
		ShapeSphere(const ShapeSphere& rhs);
	};
};
#endif	/* __ShapeSphere_hpp__ */

