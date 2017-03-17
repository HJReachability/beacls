#ifndef __DubinsCarCAvoid_hpp__
#define __DubinsCarCAvoid_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#include <helperOC/DynSys/Air3D/Air3D.hpp>
#include <typedef.hpp>
#include <cstddef>
#include <vector>
namespace helperOC {
	/*
		@note: Since plane is a "handle class", we can pass on
		handles/pointers to other plane objects
		e.g. a.platoon.leader = b (passes b by reference, does not create a copy)
		Also see constructor
	*/
	class DubinsCarCAvoid : public Air3D {
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
			DubinsCarCAvoid(
				const beacls::FloatVec& x,
				const FLOAT_TYPE uMax = 1.,
				const FLOAT_TYPE dMax = 1.,
				const FLOAT_TYPE va = 5.,
				const FLOAT_TYPE vb = 5.,
				const beacls::IntegerVec& dims = beacls::IntegerVec{ 0,1,2 }
		);
		PREFIX_VC_DLL
			virtual ~DubinsCarCAvoid();
	private:
	};
};
#endif	/* __DubinsCarCAvoid_hpp__ */
