#ifndef __Air3D_hpp__
#define __Air3D_hpp__

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
class Air3D : public DynSys {
public:
protected:
	FLOAT_TYPE uMax;	//!< Control bounds
	FLOAT_TYPE dMax;	//!< Control bounds
	FLOAT_TYPE va;	//!< Vehicle Speeds
	FLOAT_TYPE vb;	//!< Vehicle Speeds
	beacls::IntegerVec dims;	//!< Dimensions that are active
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
	@param	[in]		uMax	maximum turn rate
	@param	[in]		dMax	maximum turn rate
	@param	[in]		va	Vehicle A Speeds
	@param	[in]		vd	Vehicle B Speeds
	@return	a Plane object
	*/
	PREFIX_VC_DLL
	Air3D(
		const beacls::FloatVec& x,
		const FLOAT_TYPE uMax = 1.,
		const FLOAT_TYPE dMax = 1.,
		const FLOAT_TYPE va = 5.,
		const FLOAT_TYPE vb = 5.,
		const beacls::IntegerVec& dims = beacls::IntegerVec{ 0,1,2 }
	);
	PREFIX_VC_DLL
		Air3D(
			beacls::MatFStream* fs,
			beacls::MatVariable* variable_ptr = NULL
		);
	PREFIX_VC_DLL
		virtual ~Air3D();
	PREFIX_VC_DLL
		virtual bool operator==(const Air3D& rhs) const;
	PREFIX_VC_DLL
		virtual bool operator==(const DynSys& rhs) const;
	PREFIX_VC_DLL
		virtual bool save(
			beacls::MatFStream* fs,
			beacls::MatVariable* variable_ptr = NULL
		);
	virtual Air3D* clone() const {
		return new Air3D(*this);
	}
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
	@brief Optimal disturbance function
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
	@brief Dynamics of the Air3D system
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
#if defined(USER_DEFINED_GPU_DYNSYS_FUNC) && 0
	PREFIX_VC_DLL
	bool optCtrl_cuda(
		std::vector<beacls::UVec>& u_uvecs,
		const FLOAT_TYPE t,
		const std::vector<beacls::UVec>& x_uvecs,
		const std::vector<beacls::UVec>& deriv_uvecs,
		const DynSys_UMode_Type uMode
	) const;
	/*
	@brief Optimal disturbance function
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
	@brief Dynamics of the Air3D system
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
	Air3D& operator=(const Air3D& rhs);
	/** @overload
	Disable copy constructor
	*/
	Air3D(const Air3D& rhs) :
		DynSys(rhs),
		uMax(rhs.uMax),	//!< Control bounds
		dMax(rhs.dMax),	//!< Control bounds
		va(rhs.va),	//!< Vehicle Speeds
		vb(rhs.vb),	//!< Vehicle Speeds
		dims(rhs.dims)	//!< Dimensions that are active
	{}
};

#endif	/* __Air3D_hpp__ */
