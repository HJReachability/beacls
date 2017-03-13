#ifndef __DubinsCar_hpp__
#define __DubinsCar_hpp__

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

/*
	@note: Since plane is a "handle class", we can pass on
	handles/pointers to other plane objects
	e.g. a.platoon.leader = b (passes b by reference, does not create a copy)
	Also see constructor
*/
class DubinsCar : public DynSys {
public:
protected:
	FLOAT_TYPE wMax;	//!< Angular control bounds
	FLOAT_TYPE speed;	//!< Constant speed
	beacls::FloatVec dMax;	//!< Disturbance
	beacls::IntegerVec dims;	//!< Dimensions that are active
public:
	/*
	@brief Constructor. Creates a plane object with a unique ID,
		state x, and reachable set information reachInfo
	Dynamics:
		\dot{x}_1 = v * cos(x_3) + d1
		\dot{x}_2 = v * sin(x_3) + d2
		\dot{x}_3 = u
			v \in [vrange(1), vrange(2)]
			u \in [-wMax, wMax]
			d \in [-dMax, dMax]

	@param	[in]		x	state [xpos; ypos; theta]
	@param	[in]		wMax	maximum turn rate
	@param	[in]		vrange	speed range
	@param	[in]		dMax	disturbance bounds
	@return	a Plane object
	*/
	PREFIX_VC_DLL
	DubinsCar(
		const beacls::FloatVec& x,
		const FLOAT_TYPE wMax = 1,
		const FLOAT_TYPE speed = 5,
		const beacls::FloatVec& dMax = beacls::FloatVec{0,0,0},
		const beacls::IntegerVec& dims = beacls::IntegerVec{0,1,2}
	);
	PREFIX_VC_DLL
		DubinsCar(
			beacls::MatFStream* fs,
			beacls::MatVariable* variable_ptr = NULL
		);
			PREFIX_VC_DLL
	virtual ~DubinsCar();
	PREFIX_VC_DLL
		virtual bool operator==(const DubinsCar& rhs) const;
	PREFIX_VC_DLL
		virtual bool operator==(const DynSys& rhs) const;
	PREFIX_VC_DLL
	virtual bool save(
		beacls::MatFStream* fs,
		beacls::MatVariable* variable_ptr = NULL);
	virtual DubinsCar* clone() const {
		return new DubinsCar(*this);
	}
	bool getVelocity(beacls::FloatVec& v, std::vector<beacls::FloatVec>& vhist) const;
	/*
	@brief Control of Dubins Car
	*/
	PREFIX_VC_DLL
	bool optCtrl(
		std::vector<beacls::FloatVec >& uOpts,
		const FLOAT_TYPE t,
		const std::vector<beacls::FloatVec::const_iterator >& y_ites,
		const std::vector<const FLOAT_TYPE*>& deriv_ptrs,
		const beacls::IntegerVec& y_sizes,
		const beacls::IntegerVec& deriv_sizes,
		const DynSys_UMode_Type uMode
	) const;
	/*
	@brief disturbance of Dubins Car
		\dot{x}_1 = v * cos(x_3) + d_1
		\dot{x}_2 = v * sin(x_3) + d_2
		\dot{x}_3 = u
	*/
	PREFIX_VC_DLL
	bool optDstb(
		std::vector<beacls::FloatVec >& uOpts,
		const FLOAT_TYPE t,
		const std::vector<beacls::FloatVec::const_iterator >& y_ites,
		const std::vector<const FLOAT_TYPE*>& deriv_ptrs,
		const beacls::IntegerVec& y_sizes,
		const beacls::IntegerVec& deriv_sizes,
		const DynSys_DMode_Type dMode
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
	bool dynamics(
		std::vector<beacls::FloatVec >& dxs,
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
		const DynSys_UMode_Type uMode
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
		const DynSys_DMode_Type dMode
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
	DubinsCar& operator=(const DubinsCar& rhs);
	/** @overload
	Disable copy constructor
	*/
	DubinsCar(const DubinsCar& rhs) :
		DynSys(rhs),
		wMax(rhs.wMax),	//!< Angular control bounds
		speed(rhs.speed),	//!< Constant speed
		dMax(rhs.dMax),	//!< Disturbance
		dims(rhs.dims)	//!< Dimensions that are active
		{}
	bool dynamics_cell_helper(
		std::vector<beacls::FloatVec >& dxs,
		const beacls::FloatVec::const_iterator& x_ite,
		const std::vector<beacls::FloatVec >& us,
		const std::vector<beacls::FloatVec >& ds,
		const size_t x_size,
		const size_t dim
	) const;
};
#endif	/* __DubinsCar_hpp__ */
