#ifndef __ExtractCostates_hpp__
#define __ExtractCostates_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#include <vector>
#include <cstddef>
#include <typedef.hpp>
#include <helperOC/helperOC_type.hpp>
class HJI_Grid;

class ExtractCostates_impl;

class ExtractCostates {
private:
	ExtractCostates_impl* pimpl;
public:
	/*
	@brief	Constructor
	@param	[in]	grid		grid structure
	@param	[in]	accuracy	derivative approximation function (from level set
	toolbox)
	*/
	PREFIX_VC_DLL
		ExtractCostates(
			const helperOC::ApproximationAccuracy_Type accuracy = helperOC::ApproximationAccuracy_veryHigh
		);
	PREFIX_VC_DLL
		~ExtractCostates();

	/*
	@brief	Estimates the costate p at position x for cost function data on grid g by
	numerically taking partial derivatives along each grid direction.
	Numerical derivatives are taken using the levelset toolbox
	@param	[out]	derivC	(central) gradient in a g.dim by 1 vector
	@param	[out]	derivL	left gradient
	@param	[out]	derivR	right gradient
	@param	[in]	grid		grid structure
	@param	[in]	data		array of g.dim dimensions containing function values
	@param	[in]	data_length	array data length
	@param	[in]	upWind		whether to use upwinding (ignored; to be implemented in
	the future
	@retval	true	Succeed
	@retval	false	Failed
	*/
	PREFIX_VC_DLL
		bool operator()(
		std::vector<beacls::FloatVec >& derivC,
		std::vector<beacls::FloatVec >& derivL,
		std::vector<beacls::FloatVec >& derivR,
		const HJI_Grid* grid,
		const beacls::FloatVec& data,
		const size_t data_length,
		const bool upWind = false,
		const helperOC::ExecParameters& execParameters = helperOC::ExecParameters()
		);
};
#endif	/* __ExtractCostates_hpp__ */

