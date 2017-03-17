#ifndef __KinVehicleND_hpp__
#define __KinVehicleND_hpp__

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
		@note:  the 2D kinematic vehicle
	*/
	class KinVehicleND : public DynSys {
	public:
	protected:
		FLOAT_TYPE vMax;	//!< Angular control bounds
	public:
		/*
		@brief Constructor. Creates a plane object with a unique ID,
			state x, and reachable set information reachInfo
		Dynamics:(2D example)
				\dot{x}_1 = v_x
				\dot{x}_2 = v_y
					v_x^2 + v_y^2 <= vMax^2
		@param	[in]		x	state [xpos; ypos; theta]
		@param	[in]		wMax	maximum turn rate
		@param	[in]		vrange	speed range
		@param	[in]		dMax	disturbance bounds
		@return	a Plane object
		*/
		PREFIX_VC_DLL
			KinVehicleND(
				const beacls::FloatVec& x,
				const FLOAT_TYPE vMax = 1
			);
		PREFIX_VC_DLL
			KinVehicleND(
				beacls::MatFStream* fs,
				beacls::MatVariable* variable_ptr = NULL
			);
		PREFIX_VC_DLL
			virtual ~KinVehicleND();
		PREFIX_VC_DLL
			virtual bool operator==(const KinVehicleND& rhs) const;
		PREFIX_VC_DLL
			virtual bool operator==(const DynSys& rhs) const;
		PREFIX_VC_DLL
			virtual bool save(
				beacls::MatFStream* fs,
				beacls::MatVariable* variable_ptr = NULL);
		virtual KinVehicleND* clone() const {
			return new KinVehicleND(*this);
		}
		/*
		@brief Control of the 2D kinematic vehicle
		*/
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
		@brief Dynamics of the 2D kinematic vehicle
		\dot{x}_1 = v_x
		\dot{x}_2 = v_y
			u = (v_x, v_y)
		*/
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
		@brief Control of  the 2D kinematic vehicle
		*/
		bool optCtrl_cuda(
			std::vector<beacls::UVec>& u_uvecs,
			const FLOAT_TYPE t,
			const std::vector<beacls::UVec>& x_uvecs,
			const std::vector<beacls::UVec>& deriv_uvecs,
			const helperOC::DynSys_UMode_Type uMode
		) const;
		/*
		@brief Dynamics of the 2D kinematic vehicle
			\dot{x}_1 = v_x
			\dot{x}_2 = v_y
				u = (v_x, v_y)
		*/
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
		KinVehicleND& operator=(const KinVehicleND& rhs);
		/** @overload
		Disable copy constructor
		*/
		KinVehicleND(const KinVehicleND& rhs) :
			DynSys(rhs),
			vMax(rhs.vMax)	//!< Angular control bounds
		{}
	};
};
#endif	/* __KinVehicleND_hpp__ */
