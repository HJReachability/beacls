#ifndef __MyPlane_hpp__
#define __MyPlane_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#include <helperOC/helperOC.hpp>
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
class MyPlane : public DynSys {
public:
protected:
	FLOAT_TYPE wMax;	//!< Angular control bounds
	beacls::FloatVec vrange;	//!< Speed control bounds
	beacls::FloatVec dMax;	//!< Disturbance
public:
	/*
	@brief Constructor. Creates a plane object with a unique ID,
		state x, and reachable set information reachInfo
	Dynamics:
		\dot{x}_1 = v * cos(x_3) + d1
		\dot{x}_2 = v * sin(x_3) + d2
		\dot{x}_3 = u            + d3
			v \in [vrange(1), vrange(2)]
			u \in [-wMax, wMax]

	@param	[in]		x	state [xpos; ypos; theta]
	@param	[in]		wMax	maximum turn rate
	@param	[in]		vrange	speed range
	@param	[in]		dMax	disturbance bounds
	@return	a MyPlane object
	*/
	PREFIX_VC_DLL
	MyPlane(
		const beacls::FloatVec& x,
		const FLOAT_TYPE wMax = 1,
		const beacls::FloatVec& vrange = beacls::FloatVec{ 5 },
		const beacls::FloatVec& dMax = beacls::FloatVec{0,0}
	);
	PREFIX_VC_DLL
		MyPlane(
			beacls::MatFStream* fs,
			beacls::MatVariable* variable_ptr = NULL
		);
	PREFIX_VC_DLL
		virtual ~MyPlane();
	PREFIX_VC_DLL
		virtual bool save(
			beacls::MatFStream* fs,
			beacls::MatVariable* variable_ptr = NULL);
	PREFIX_VC_DLL
		virtual bool operator==(const MyPlane& rhs) const;
	PREFIX_VC_DLL
		virtual bool operator==(const DynSys& rhs) const;

	virtual MyPlane* clone() const {
		return new MyPlane(*this);
	}
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
		std::vector<beacls::FloatVec >& dx,
		const FLOAT_TYPE t,
		const std::vector<beacls::FloatVec::const_iterator >& x_ites,
		const std::vector<beacls::FloatVec >& us,
		const std::vector<beacls::FloatVec >& ds,
		const beacls::IntegerVec& x_sizes,
		const size_t dst_target_dim
	) const;
#if defined(USER_DEFINED_GPU_DYNSYS_FUNC) && defined(WIGH_GPU)
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
#endif /* defined(USER_DEFINED_GPU_DYNSYS_FUNC) && defined(WIGH_GPU) */

	PREFIX_VC_DLL
		FLOAT_TYPE get_wMax() const;
	PREFIX_VC_DLL
		const beacls::FloatVec& get_vrange() const;
	PREFIX_VC_DLL
		const beacls::FloatVec& get_dMax() const;
protected:
	/** @overload
	Disable copy constructor
	*/
	PREFIX_VC_DLL
		MyPlane(const MyPlane& rhs) :
		DynSys(rhs),
		wMax(rhs.wMax),	//!< Angular control bounds
		vrange(rhs.vrange),	//!< Speed control bounds
		dMax(rhs.dMax)	//!< Disturbance
	{}
private:
	/** @overload
	Disable operator=
	*/
	MyPlane& operator=(const MyPlane& rhs);
	bool dynamics_cell_helper(
		beacls::FloatVec& dx,
		const beacls::FloatVec::const_iterator& x_ite,
		const std::vector<beacls::FloatVec >& us,
		const std::vector<beacls::FloatVec >& ds,
		const size_t x_size,
		const size_t dim
	) const;
};
#endif	/* __MyPlane_hpp__ */
