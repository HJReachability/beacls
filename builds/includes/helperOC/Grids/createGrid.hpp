#ifndef __CREATEGRID_hpp__
#define __CREATEGRID_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#include <typedef.hpp>
#include <cstddef>
#include <vector>

class HJI_Grid;

// Mo Chen, 2016-04-18
// Ken Tanabe, 2016-08-05

/**
	@brief	createGrid
			Thin wrapper around processGrid to create a grid compatible with the
			level set toolbox
	@param	[in]		grid_mins	minimum bounds on computation domain
	@param	[in]		grid_maxs	maximum bounds on computation domain
	@param	[in]	Ns	number of grid points in each dimension
	@param	[in]	pdDims	periodic dimensions(eg.pdDims = [2 3] if 2nd and
								3rd dimensions are periodic)
	@param	[in]	process	specifies whether to call processGrid to generate
								grid points
	@return						grid structure
*/
PREFIX_VC_DLL
HJI_Grid* createGrid(
	const beacls::FloatVec& grid_mins,
	const beacls::FloatVec& grid_maxs,
	const beacls::IntegerVec& Ns,
	const beacls::IntegerVec& pdDims = beacls::IntegerVec(),
	const bool process = true
	);
#endif	/* __CREATEGRID_hpp__ */

