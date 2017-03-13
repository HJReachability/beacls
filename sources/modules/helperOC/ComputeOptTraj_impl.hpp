#ifndef __ComputeOptTraj_impl_hpp__
#define __ComputeOptTraj_impl_hpp__

#include <typedef.hpp>
#include <cstddef>
#include <vector>
class HJI_Grid;

namespace helperOC {
	class ComputeGradients;
	class ComputeOptTraj_impl {
	private:
		ComputeGradients* computeGradients;
	public:
		ComputeOptTraj_impl();
		~ComputeOptTraj_impl();
		bool operator()(
			std::vector<beacls::FloatVec >& traj,
			beacls::FloatVec& traj_tau,
			const HJI_Grid* grid,
			const std::vector<beacls::FloatVec >& data,
			const beacls::FloatVec& tau,
			DynSys* dynSys,
			const HJIPDE_extraArgs& extraArgs,
			const DynSys_UMode_Type uMode,
			const size_t subSamples
		);
	private:
		ComputeOptTraj_impl(const ComputeOptTraj_impl& rhs);
		ComputeOptTraj_impl& operator=(const ComputeOptTraj_impl& rhs);
	};
};
#endif	/* __ComputeOptTraj_impl_hpp__ */

