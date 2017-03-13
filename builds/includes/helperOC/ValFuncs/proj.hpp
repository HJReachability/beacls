#ifndef __proj_hpp__
#define __proj_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#include <helperOC/helperOC_type.hpp>
#include <typedef.hpp>
#include <cstddef>
class HJI_Grid;
namespace helperOC {
	/**
	@brief	Projects data corresponding to the grid g in g.dim dimensions, removing
			dimensions specified in dims. If a point is specified, a slice of the
			full-dimensional data at the point xs is taken.
	@param	[out]	dataOut			shifted data
	@param	[in]	g				grid
	@param	[in]	dataIn			original data
	@param	[in]	dims			vector of length g.dim specifying dimensions to project
									For example, if g.dim = 4, then dims = [0 0 1 1] would
									project the last two dimensions
	@param	[in]	xs				Type of projection (defaults to 'min')
									'min':    takes the union across the projected dimensions
									'max':    takes the intersection across the projected dimensions
									a vector: takes a slice of the data at the point xs
	@param	[in]	NOut			number of grid points in output grid (defaults to the same
									number of grid points of the original grid in the unprojected
									dimensions)
	@param	[in]	process			specifies whether to call processGrid to generate
									grid points
	@return							grid corresponding to projected data
	*/
	PREFIX_VC_DLL
		HJI_Grid* proj(
			std::vector<beacls::FloatVec >& dataOut,
			const HJI_Grid* g,
			const std::vector<beacls::FloatVec >& dataIn,
			const beacls::IntegerVec& dims,
			const std::vector<Projection_Type>& x_types = std::vector<Projection_Type>(),
			const beacls::FloatVec& xs = beacls::FloatVec(),
			const beacls::IntegerVec& NOut = beacls::IntegerVec(),
			const bool process = true
		);
	/**
	@brief	Projects data corresponding to the grid g in g.dim dimensions, removing
	dimensions specified in dims. If a point is specified, a slice of the
	full-dimensional data at the point xs is taken.
	@param	[out]	dataOut			shifted data
	@param	[in]	g				grid
	@param	[in]	dataIn			original data
	@param	[in]	dims			vector of length g.dim specifying dimensions to project
	For example, if g.dim = 4, then dims = [0 0 1 1] would
	project the last two dimensions
	@param	[in]	xs				Type of projection (defaults to 'min')
	'min':    takes the union across the projected dimensions
	'max':    takes the intersection across the projected dimensions
	a vector: takes a slice of the data at the point xs
	@param	[in]	NOut			number of grid points in output grid (defaults to the same
	number of grid points of the original grid in the unprojected
	dimensions)
	@param	[in]	process			specifies whether to call processGrid to generate
	grid points
	@return							grid corresponding to projected data
	*/
	PREFIX_VC_DLL
		HJI_Grid* proj(
			std::vector<beacls::FloatVec >& dataOut,
			const HJI_Grid* g,
			const std::vector<const beacls::FloatVec* >& dataIn,
			const beacls::IntegerVec& dims,
			const std::vector<Projection_Type>& x_types = std::vector<Projection_Type>(),
			const beacls::FloatVec& xs = beacls::FloatVec(),
			const beacls::IntegerVec& NOut = beacls::IntegerVec(),
			const bool process = true
		);
	/**
	@brief	Projects data corresponding to the grid g in g.dim dimensions, removing
	dimensions specified in dims. If a point is specified, a slice of the
	full-dimensional data at the point xs is taken.
	@param	[out]	dataOut			shifted data
	@param	[in]	g				grid
	@param	[in]	dataIn			original data
	@param	[in]	dims			vector of length g.dim specifying dimensions to project
	For example, if g.dim = 4, then dims = [0 0 1 1] would
	project the last two dimensions
	@param	[in]	xs				Type of projection (defaults to 'min')
	'min':    takes the union across the projected dimensions
	'max':    takes the intersection across the projected dimensions
	a vector: takes a slice of the data at the point xs
	@param	[in]	NOut			number of grid points in output grid (defaults to the same
	number of grid points of the original grid in the unprojected
	dimensions)
	@param	[in]	process			specifies whether to call processGrid to generate
	grid points
	@return							grid corresponding to projected data
	*/
	PREFIX_VC_DLL
		HJI_Grid* proj(
			beacls::FloatVec& dataOut,
			const HJI_Grid* g,
			const beacls::FloatVec& dataIn,
			const beacls::IntegerVec& dims,
			const std::vector<Projection_Type>& x_types = std::vector<Projection_Type>(),
			const beacls::FloatVec& xs = beacls::FloatVec(),
			const beacls::IntegerVec& NOut = beacls::IntegerVec(),
			const bool process = true
		);
};
#endif	/* __proj_hpp__ */

