#ifndef __DESTROYGRID_hpp__
#define __DESTROYGRID_hpp__

// Mo Chen, 2016-04-18
// Ken Tanabe, 2016-08-05

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

class HJI_Grid;


/**
	@brief	destroyGrid
	@param	[in]		grid	grid structure
*/
PREFIX_VC_DLL
void destroyGrid(
	HJI_Grid* grid
	);
#endif	/* __DESTROYGRID_hpp__ */

