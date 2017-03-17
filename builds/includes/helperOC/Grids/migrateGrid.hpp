#ifndef __migrateGrid_hpp__
#define __migrateGrid_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#include <vector>
#include <typedef.hpp>
namespace levelset {
	class HJI_Grid;
};

namespace helperOC {
	/**
		@brief	migrateGrid
				Transfers dataOld onto a from the grid gOld to the grid gNew
		@param	[out]	dataNew	equivalent data corresponding to new grid structure
		@param	[in]	gOld	old grid structures
		@param	[in]	dataOld	data corresponding to old grid structure
		@param	[in]	gNew	new grid structures
		@param	[in]	process	specifies whether to call processGrid to generate
									grid points
		@retval	true			Succeeded
		@retval	false			Failed
	*/
	PREFIX_VC_DLL
	bool migrateGrid(
		std::vector<beacls::FloatVec>& dataNew,
		const levelset::HJI_Grid* gOld,
		const std::vector<beacls::FloatVec>& dataOld,
		const levelset::HJI_Grid* gNew
		);
	/**
	@brief	migrateGrid
	Transfers dataOld onto a from the grid gOld to the grid gNew
	@param	[out]	dataNew	equivalent data corresponding to new grid structure
	@param	[in]	gOld	old grid structures
	@param	[in]	dataOld	data corresponding to old grid structure
	@param	[in]	gNew	new grid structures
	@param	[in]	process	specifies whether to call processGrid to generate
	grid points
	@retval	true			Succeeded
	@retval	false			Failed
	*/
	PREFIX_VC_DLL
		bool migrateGrid(
			beacls::FloatVec& dataNew,
			const levelset::HJI_Grid* gOld,
			const beacls::FloatVec& dataOld,
			const levelset::HJI_Grid* gNew
		);
};
#endif	/* __migrateGrid_hpp__ */

