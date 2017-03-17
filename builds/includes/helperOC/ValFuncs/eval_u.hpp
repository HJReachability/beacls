#ifndef __eval_u_hpp__
#define __eval_u_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#include <typedef.hpp>
namespace levelset {
	class HJI_Grid;
};
namespace helperOC {
	/**
	@brief	Computes the interpolated value of the value functions datas at the states xs
	Option 1: Single grid, single value function, multiple states
	@param	[out]	dataOut			shifted data
	@param	[in]	gs				a single grid structure
	@param	[in]	data			a single matrix (look-up stable) representing the value function
	@param	[in]	x				set of states; each row is a state
	@param	[in]	interp_method	interporation method
	@retval	true					Succeeded
	@retval false					Failed
	*/
	PREFIX_VC_DLL
		bool eval_u(
			beacls::FloatVec& dataOut,
			const levelset::HJI_Grid* g,
			const beacls::FloatVec& data,
			const std::vector<beacls::FloatVec >& xs,
			const beacls::Interpolate_Type interp_method = beacls::Interpolate_linear
		);
	/**
	@brief	Computes the interpolated value of the value functions datas at the states xs
	Option 1: Single grid, multiple value functions, single state
	@param	[out]	dataOuts		a vector of shifted data
	@param	[in]	gs				a single grid structure
	@param	[in]	datas			a cell structure of matrices representing the value function
	@param	[in]	xs				a single state
	@param	[in]	interp_method	interporation method
	@retval	true					Succeeded
	@retval false					Failed
	*/
	PREFIX_VC_DLL
		bool eval_u(
			std::vector<beacls::FloatVec>& dataOuts,
			const levelset::HJI_Grid* g,
			const std::vector<beacls::FloatVec >& datas,
			const beacls::FloatVec& x,
			const beacls::Interpolate_Type interp_method = beacls::Interpolate_linear
		);
	/**
	@brief	Computes the interpolated value of the value functions datas at the states xs
	Option 1: Single grid, single value function, multiple states
	@param	[out]	dataOuts		a vector of shifted data
	@param	[in]	gs				a cell structure of grid structures
	@param	[in]	data			a cell structure of matrices representing value functions
	@param	[in]	xs				a cell structure of states
	@param	[in]	interp_method	interporation method
	@retval	true					Succeeded
	@retval false					Failed
	*/
	PREFIX_VC_DLL
		bool eval_u(
			std::vector<beacls::FloatVec>& dataOuts,
			const std::vector<levelset::HJI_Grid*>& gs,
			const std::vector<beacls::FloatVec >& datas,
			const std::vector<beacls::FloatVec >& xs,
			const beacls::Interpolate_Type interp_method = beacls::Interpolate_linear
		);
};
#endif	/* __eval_u_hpp__ */

