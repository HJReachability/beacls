#ifndef __find_earliest_BRS_ind_hpp__
#define __find_earliest_BRS_ind_hpp__

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
#include <vector>
class HJI_Grid;
namespace helperOC {
	/**
	@brief	Determine the earliest time that the current state is in the reachable set
	@param	[in]	g		grid and value function representing reachable set
	@param	[in]	data	grid and value function representing reachable set
	@param	[in]	x		state of interest
	@param	[in]	upper	upper indices of the search range
	@param	[in]	lower	lower indices of the search range
	@return	earliest time index that x is in the reachable set
	*/
	PREFIX_VC_DLL
		size_t find_earliest_BRS_ind(
		const HJI_Grid* g,
		const std::vector<beacls::FloatVec >& data,
		const std::vector<beacls::FloatVec >& x,
		const size_t upper =-1,
		const size_t lower =0
	);
};
#endif	/* __find_earliest_BRS_ind_hpp__ */

