
#ifndef __HJIPDE_hpp__
#define __HJIPDE_hpp__

#include <iostream>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <typedef.hpp>
#include <helperOC/helperOC_type.hpp>
//#include <levelset/levelset.hpp>

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif
namespace helperOC {
	class HJIPDE_impl;
	class DynSysSchemeData;


	class HJIPDE {
	private:
		HJIPDE_impl *pimpl;
	public:
		typedef enum MinWithType {
			MinWithType_Invalid = -1,
			MinWithType_Zero,	//!<	set to 'zero' to do min with zero
			MinWithType_None,	//!<	set to 'none' to compute reachable set (not tube)
		} MinWithType;
		typedef enum ObsModeType {
			ObsModeType_Invalid = -1,
			ObsModeType_None,
			ObsModeType_Static,
			ObsModeType_TimeVarying,
		} ObsModeType;
		/*
		@brief Constructor
		@param	[in]	tmp_filename	File name under which temporary datas are saved instead of on memory.
										When it is empty, don't use temporary file.
		*/
		PREFIX_VC_DLL
			HJIPDE(const std::string& tmp_filename);
		PREFIX_VC_DLL
			HJIPDE();

		PREFIX_VC_DLL
			~HJIPDE();
		/*
		@brief Solves HJIPDE with initial conditions data0, at times tau, and with
		parameters schemeData and extraArgs
		@param	[out]	dst_datas	solution corresponding to grid g and time vector tau
		@param	[out]	dst_tau		list of computation times (redundant)
		@param	[out]	extraOuts	extra outputs
		@param	[in]	src_datas		initial value function
		@param	[in]	src_tau			list of computation times
		@param	[in]	SchemeData	problem parameters passed into the Hamiltonian function
		grid: grid (required!)
		@param	[in]	minWith
		@arg	MinWithType_Zero	:	set to 'zero' to do min with zero
		@arg	MinWithType_None	:	set to 'none' to compute reachable set (not tube)
		@param	[in]	extraArgs	this structure can be used to leverage other additional
		functionalities within this functionouts
		@param	[in]	quiet
		@arg	true	:	quiet mode
		@arg	false	:	verbose mode (default)
		*/
		PREFIX_VC_DLL
			bool solve(
				std::vector<beacls::FloatVec >& dst_datas,
				beacls::FloatVec& dst_tau,
				helperOC::HJIPDE_extraOuts& extraOuts,
				const std::vector<beacls::FloatVec >& src_datas,
				const beacls::FloatVec& src_tau,
				const DynSysSchemeData* schemeData,
				const MinWithType minWith = MinWithType_Zero,
				const helperOC::HJIPDE_extraArgs& extraArgs = helperOC::HJIPDE_extraArgs()
			);
		/*
		@brief Solves HJIPDE with initial conditions data0, at times tau, and with
		parameters schemeData and extraArgs
		@param	[out]	dst_datas	solution corresponding to grid g and time vector tau
		@param	[out]	dst_tau		list of computation times (redundant)
		@param	[out]	extraOuts	extra outputs
		@param	[in]	src_data		initial value function
		@param	[in]	src_tau			list of computation times
		@param	[in]	SchemeData	problem parameters passed into the Hamiltonian function
		grid: grid (required!)
		@param	[in]	minWith
		@arg	MinWithType_Zero	:	set to 'zero' to do min with zero
		@arg	MinWithType_None	:	set to 'none' to compute reachable set (not tube)
		@param	[in]	extraArgs	this structure can be used to leverage other additional
		functionalities within this functionouts
		@param	[in]	quiet
		@arg	true	:	quiet mode
		@arg	false	:	verbose mode (default)
		*/
		PREFIX_VC_DLL
			bool solve(
				std::vector<beacls::FloatVec >& dst_datas,
				beacls::FloatVec& dst_tau,
				helperOC::HJIPDE_extraOuts& extraOuts,
				const beacls::FloatVec& src_data,
				const beacls::FloatVec& src_tau,
				const DynSysSchemeData* schemeData,
				const MinWithType minWith = MinWithType_Zero,
				const helperOC::HJIPDE_extraArgs& extraArgs = helperOC::HJIPDE_extraArgs()
			);
		/*
		@brief Solves HJIPDE with initial conditions data0, at times tau, and with
		parameters schemeData and extraArgs
		@param	[out]	dst_tau		list of computation times (redundant)
		@param	[out]	extraOuts	extra outputs
		@param	[in]	src_datas		initial value function
		@param	[in]	src_tau			list of computation times
		@param	[in]	SchemeData	problem parameters passed into the Hamiltonian function
									grid: grid (required!)
		@param	[in]	minWith
		@arg	MinWithType_Zero	:	set to 'zero' to do min with zero
		@arg	MinWithType_None	:	set to 'none' to compute reachable set (not tube)
		@param	[in]	extraArgs	this structure can be used to leverage other additional
									functionalities within this functionouts
		@param	[in]	quiet
		@arg	true	:	quiet mode
		@arg	false	:	verbose mode (default)
									*/
		PREFIX_VC_DLL
			bool solve(
				beacls::FloatVec& dst_tau,
				helperOC::HJIPDE_extraOuts& extraOuts,
				const std::vector<beacls::FloatVec >& src_datas,
				const beacls::FloatVec& src_tau,
				const DynSysSchemeData* schemeData,
				const MinWithType minWith = MinWithType_Zero,
				const helperOC::HJIPDE_extraArgs& extraArgs = helperOC::HJIPDE_extraArgs()
			);
		/*
		@brief Solves HJIPDE with initial conditions data0, at times tau, and with
		parameters schemeData and extraArgs
		@param	[out]	dst_tau		list of computation times (redundant)
		@param	[out]	extraOuts	extra outputs
		@param	[in]	src_data		initial value function
		@param	[in]	src_tau			list of computation times
		@param	[in]	SchemeData	problem parameters passed into the Hamiltonian function
		grid: grid (required!)
		@param	[in]	minWith
		@arg	MinWithType_Zero	:	set to 'zero' to do min with zero
		@arg	MinWithType_None	:	set to 'none' to compute reachable set (not tube)
		@param	[in]	extraArgs	this structure can be used to leverage other additional
		functionalities within this functionouts
		@param	[in]	quiet
		@arg	true	:	quiet mode
		@arg	false	:	verbose mode (default)
		*/
		PREFIX_VC_DLL
			bool solve(
				beacls::FloatVec& dst_tau,
				helperOC::HJIPDE_extraOuts& extraOuts,
				const beacls::FloatVec& src_data,
				const beacls::FloatVec& src_tau,
				const DynSysSchemeData* schemeData,
				const MinWithType minWith = MinWithType_Zero,
				const helperOC::HJIPDE_extraArgs& extraArgs = helperOC::HJIPDE_extraArgs()
			);
		/*
		@brief Read data from temporary file.
		@param	[out]	dst_datas	solution corresponding to grid g and time vector tau
		@param	[in]	src_tau		list of computation times
		functionalities within this functionouts
		*/
		PREFIX_VC_DLL
			bool get_datas(
				std::vector<beacls::FloatVec >& dst_datas,
				const beacls::FloatVec& src_tau,
				const DynSysSchemeData* schemeData
			)const;
		/*
		@brief Get last data
		@param	[out]	dst_data	last data
		@retval	true	Succeded
		@retval	false	Failed
		*/
		PREFIX_VC_DLL
			bool get_last_data(
				beacls::FloatVec& dst_data
			)const;
		/*
		@brief	 Converts a time-dependent value function to a time-to-reach value function
		@param	[out]	TTR	time-to-reach value function
		@param	[in]	g	grid structure
		@param	[in]	tau	time stamps associated with TD
		*/
		PREFIX_VC_DLL
			bool TD2TTR(
				beacls::FloatVec& TTR,
				const levelset::HJI_Grid* g,
				const beacls::FloatVec& tau
			)const;

	private:
		/** @overload
		Disable operator=
		*/
		HJIPDE& operator=(const HJIPDE& rhs);
		/** @overload
		Disable copy constructor
		*/
		HJIPDE(const HJIPDE& rhs);
	};
};
#endif	/* __HJIPDE_hpp__ */

