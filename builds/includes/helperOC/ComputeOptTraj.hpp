#ifndef __computeOptTraj_hpp__
#define __computeOptTraj_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#include <levelset/levelset.hpp>
#include <vector>
#include <iostream>
#include <helperOC/ValFuncs/HJIPDE.hpp>
#include <helperOC/DynSys/DynSys/DynSysTypeDef.hpp>
namespace levelset {
	class HJI_Grid;
};

namespace helperOC {
	class DynSys;
	class ComputeOptTraj_impl;
	class ComputeOptTraj {
	private:
		ComputeOptTraj_impl* pimpl;
	public:
		/**
			@brief	Computes the optimal trajectories given the optimal value function
					represented by (g, data), associated time stamps tau, dynamics given in
					dynSys.
			@param	[out]	traj		(central) gradient in a g.dim by 1 vector
			@param	[out]	traj_tau	left gradient
			@param	[in]	grid		grid structure
			@param	[in]	data		array of g.dim dimensions containing function values
			@param	[in]	tau			time stamp (must be the same length as size of last dimension of data)
			@param	[in]	dynSys		dynamical system object for which the optimal path is to be computed
			@param	[in]	extraArgs	HJIPDE_extraArgs
											uMode			specifies whether the control u aims to minimize or maximize the value function
											visualize		set to true to visualize results
											projDim			set the dimensions that should be projected away when visualizing
											fig_filename	specifies the file name for saving the visualizations
			@retval	true	Succeed
			@retval	false	Failed
		*/
		PREFIX_VC_DLL
			bool operator()(
				std::vector<beacls::FloatVec >& traj,
				beacls::FloatVec& traj_tau,
				const levelset::HJI_Grid* grid,
				const std::vector<beacls::FloatVec >& data,
				const beacls::FloatVec& tau,
				DynSys* dynSys,
				const HJIPDE_extraArgs& extraArgs = HJIPDE_extraArgs(),
				const helperOC::DynSys_UMode_Type uMode = helperOC::DynSys_UMode_Min,
				const size_t subSamples = 4
			);
		PREFIX_VC_DLL
			ComputeOptTraj();
		PREFIX_VC_DLL
			~ComputeOptTraj();
	private:
		ComputeOptTraj& operator=(const ComputeOptTraj& rhs);
		ComputeOptTraj(const ComputeOptTraj& rhs);
	};
};

#endif	/* __computeOptTraj_hpp__ */
