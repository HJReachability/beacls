#ifndef __PlaneCAvoid_hpp__
#define __PlaneCAvoid_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#include <helperOC/DynSys/DynSys/DynSys.hpp>
#include <typedef.hpp>
#include <cstddef>
#include <vector>
#include <iostream>
#include <cstring>
#include <utility>
using namespace std::rel_ops;
namespace helperOC {
	/*
		@note: Since plane is a "handle class", we can pass on
		handles/pointers to other plane objects
		e.g. a.platoon.leader = b (passes b by reference, does not create a copy)
		Also see constructor
	*/
	class PlaneCAvoid : public DynSys {
	protected:
		FLOAT_TYPE wMaxA;	//!< Angular control bounds
		beacls::FloatVec vRangeA;	//!< Speed control bounds
		FLOAT_TYPE wMaxB;	//!< Angular control bounds
		beacls::FloatVec vRangeB;	//!< Speed control bounds
		beacls::FloatVec dMaxA;	//!< Disturbance
		beacls::FloatVec dMaxB;	//!< Disturbance
	public:
		/*
		@brief Constructor. Creates a plane object with a unique ID,
		state x, and reachable set information reachInfo
		Dynamics of each plane:
			\dot{x}_1 = v * cos(x_3) + d1
			\dot{x}_2 = v * sin(x_3) + d2
			\dot{x}_3 = u            + d3
				v in [vrange(1),  vrange(2)]
				u in [-wMax, wMax]
				norm(d1, d2) <= dMax(1)
				abs(d3) <= dMax(2)
		Dynamics relative to Plane A (evader in air3D.m):
			\dot{x}_1 = -vA + vB*cos(x_3) + wA*x_2 + d1
			\dot{x}_2 = vB*sin(x_3) - wA*x_1 + d2
			\dot{x}_3 = wB - wA + d3
				vA in vRangeA, vB in vRangeB
				wA in [-wMaxA, wMaxA], wB in [-wMaxB, wMaxB]
				norm(d1, d2) <= dMaxA(1) + dMaxB(1)
				abs(d3) <= dMaxA(2) + dMaxB(2)

		@param	[in]		x		state [xpos; ypos; theta]
									Alternatively, x can be a cell of two Plane objects. The first Plane object is Plane A
		@param	[in]		wMaxA	maximum turn rate of vehicle A
		@param	[in]		vRangeA	speed range
		@param	[in]		wMaxB	maximum turn rate of vehicle B
		@param	[in]		vRangeB	speed range
		@param	[in]		dMaxA	disturbance bounds
		@param	[in]		dMaxB	disturbance bounds
		@return	a Plane object
		*/
		PREFIX_VC_DLL
			PlaneCAvoid(
				const beacls::FloatVec& x,
				const FLOAT_TYPE wMaxA,
				const beacls::FloatVec& vRangeA,
				const FLOAT_TYPE wMaxB,
				const beacls::FloatVec& vRRngeB,
				const beacls::FloatVec& dMaxA,
				const beacls::FloatVec& dMaxB
			);
		PREFIX_VC_DLL
			PlaneCAvoid(
				beacls::MatFStream* fs,
				beacls::MatVariable* variable_ptr = NULL
			);
		PREFIX_VC_DLL
			virtual ~PlaneCAvoid();
		PREFIX_VC_DLL
			virtual bool operator==(const PlaneCAvoid& rhs) const;
		PREFIX_VC_DLL
			virtual bool operator==(const DynSys& rhs) const;
		virtual PlaneCAvoid* clone() const {
			return new PlaneCAvoid(*this);
		}
		PREFIX_VC_DLL
			virtual bool save(
				beacls::MatFStream* fs,
				beacls::MatVariable* variable_ptr = NULL);
		PREFIX_VC_DLL
			FLOAT_TYPE get_wMaxA() const { return wMaxA; };
		PREFIX_VC_DLL
			const beacls::FloatVec& get_vRangeA() const { return vRangeA; };
		PREFIX_VC_DLL
			FLOAT_TYPE get_wMaxB() const { return wMaxB; };
		PREFIX_VC_DLL
			const beacls::FloatVec& get_vRangeB() const { return vRangeB; };
		PREFIX_VC_DLL
			const beacls::FloatVec& get_dMaxA() const { return dMaxA; };
		PREFIX_VC_DLL
			const beacls::FloatVec& get_dMaxB() const { return dMaxB; };

		/*
		@brief Control of PlaneCAvoid
		*/
		PREFIX_VC_DLL
			bool optCtrl(
				std::vector<beacls::FloatVec >& uOpts,
				const FLOAT_TYPE t,
				const std::vector<beacls::FloatVec::const_iterator >& y_ites,
				const std::vector<const FLOAT_TYPE*>& deriv_ptrs,
				const beacls::IntegerVec& y_sizes,
				const beacls::IntegerVec& deriv_sizes,
				const helperOC::DynSys_UMode_Type uMode
			) const;
		/*
		@brief disturbance of PlaneCAvoid
			\dot{x}_1 = v * cos(x_3) + d_1
			\dot{x}_2 = v * sin(x_3) + d_2
			\dot{x}_3 = u            + d_3
		*/
		PREFIX_VC_DLL
			bool optDstb(
				std::vector<beacls::FloatVec >& uOpts,
				const FLOAT_TYPE t,
				const std::vector<beacls::FloatVec::const_iterator >& y_ites,
				const std::vector<const FLOAT_TYPE*>& deriv_ptrs,
				const beacls::IntegerVec& y_sizes,
				const beacls::IntegerVec& deriv_sizes,
				const helperOC::DynSys_DMode_Type dMode
			) const;
		/*
		@brief Dynamics of the PlaneCAvoid
		Dynamics relative to Plane A (evader in air3D.m):
			\dot{x}_1 = -vA + vB*cos(x_3) + wA*x_2 + d1
			\dot{x}_2 = vB*sin(x_3) - wA*x_1 + d2
			\dot{x}_3 = wB - wA + d3
				vA in vRangeA, vB in vRangeB
				wA in [-wMaxA, wMaxA], wB in [-wMaxB, wMaxB]
				norm(d1, d2) <= dMaxA(1) + dMaxB(1)
				abs(d3) <= dMaxA(2) + dMaxB(2)

			u = (vA, wA)
			d = (vB, wB, d1, d2, d3)
		*/
		PREFIX_VC_DLL
			bool dynamics(
				std::vector<beacls::FloatVec >& dx,
				const FLOAT_TYPE t,
				const std::vector<beacls::FloatVec::const_iterator >& x_ites,
				const std::vector<beacls::FloatVec >& us,
				const std::vector<beacls::FloatVec >& ds,
				const beacls::IntegerVec& x_sizes,
				const size_t dst_target_dim
			) const;
#if defined(USER_DEFINED_GPU_DYNSYS_FUNC)
		/*
		@brief Control of Dubins Car
		*/
		PREFIX_VC_DLL
			bool optCtrl_cuda(
				std::vector<beacls::UVec>& u_uvecs,
				const FLOAT_TYPE t,
				const std::vector<beacls::UVec>& x_uvecs,
				const std::vector<beacls::UVec>& deriv_uvecs,
				const helperOC::DynSys_UMode_Type uMode
			) const;
		/*
		@brief disturbance of Dubins Car
		\dot{x}_1 = v * cos(x_3) + d_1
		\dot{x}_2 = v * sin(x_3) + d_2
		\dot{x}_3 = u
		*/
		PREFIX_VC_DLL
			bool optDstb_cuda(
				std::vector<beacls::UVec>& d_uvecs,
				const FLOAT_TYPE t,
				const std::vector<beacls::UVec>& x_uvecs,
				const std::vector<beacls::UVec>& deriv_uvecs,
				const helperOC::DynSys_DMode_Type dMode
			) const;
		/*
		@brief Dynamics of the Dubins Car
		\dot{x}_1 = v * cos(x_3)
		\dot{x}_2 = v * sin(x_3)
		\dot{x}_3 = w
		Control: u = w;
		@author Mo Chen, 2016-06-08
		*/
		PREFIX_VC_DLL
			bool dynamics_cuda(
				std::vector<beacls::UVec>& dx_uvecs,
				const FLOAT_TYPE t,
				const std::vector<beacls::UVec>& x_uvecs,
				const std::vector<beacls::UVec>& u_uvecs,
				const std::vector<beacls::UVec>& d_uvecs,
				const size_t dst_target_dim
			) const;
#endif /* defined(USER_DEFINED_GPU_DYNSYS_FUNC) */

	private:
		/** @overload
		Disable operator=
		*/
		PlaneCAvoid& operator=(const PlaneCAvoid& rhs);
		/** @overload
		Disable copy constructor
		*/
		PlaneCAvoid(const PlaneCAvoid& rhs) :
			DynSys(rhs),
			wMaxA(rhs.wMaxA),	//!< Angular control bounds
			vRangeA(rhs.vRangeA),	//!< Speed control bounds
			wMaxB(rhs.wMaxB),	//!< Angular control bounds
			vRangeB(rhs.vRangeB),	//!< Speed control bounds
			dMaxA(rhs.dMaxA),	//!< Disturbance
			dMaxB(rhs.dMaxB)	//!< Disturbance
		{}
		bool dynamics_cell_helper(
			std::vector<beacls::FloatVec >& dx,
			const std::vector<beacls::FloatVec::const_iterator >& x_ites,
			const std::vector<beacls::FloatVec >& us,
			const std::vector<beacls::FloatVec >& ds,
			const beacls::IntegerVec& x_sizes,
			const size_t dim
		) const;
	};
};
#endif	/* __PlaneCAvoid_hpp__ */
