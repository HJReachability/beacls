#ifndef __STS_hpp__
#define __STS_hpp__

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
namespace helperOC {
	class STS : public DynSys {
	public:
	protected:
		beacls::FloatVec TMax;	//!< Max Torque
        beacls::FloatVec TMin;	//!< Min Torque
        //beacls::FloatVec TArange;	//!< Torque Ankle
		FLOAT_TYPE grav;	//!< gravity
        FLOAT_TYPE R1;	//!<
        FLOAT_TYPE R2;	//!<
        FLOAT_TYPE M1;	//!<
        FLOAT_TYPE M2;	//!<
        FLOAT_TYPE L1;	//!<
        FLOAT_TYPE L0;	//!<
		beacls::FloatVec dMax;	//!< Disturbance
		beacls::IntegerVec dims;	//!< Dimensions that are active
	public:
		/*
		@brief Constructor. Creates a plane object with a unique ID,
			state x, and reachable set information reachInfo
		Dynamics:
			\dot{x}_1 = x2
			\dot{x}_2 = stuff
			\dot{x}_3 = x4
            \dot{x}_4 = stuff
				T1 \in [T1range(1), T1range(2)]
                T2 \in [T2range(1), T2range(2)]
                TA \in [TArange(1), TArange(2)]
				d \in [-dMax, dMax]

		@param	[in]		x	state [Apos; Avel; Hpos; Hvel]
		@param	[in]		R1, R2, M1, M2, L1, L0	parameters
		@param	[in]		grav	gravity
        @param	[in]		T1range, T2range, TArange	torque ranges
		@param	[in]		dMax	disturbance bounds
		@return	a STS object
		*/
		PREFIX_VC_DLL
			STS(
				const beacls::FloatVec& x,
                const FLOAT_TYPE M1 = 2*(0.1416*62), //!<% mass of thighs
                const FLOAT_TYPE M2 = (.0694 +.4346)*62,    //!<% mass of head-arms-trunk
                const FLOAT_TYPE L0 = .25*1.72,          //!<% length of segment (shank)
                const FLOAT_TYPE L1 = .26*1.72,         //!< % length of segment .4(thigh)
                const FLOAT_TYPE R1 = .43*L1,         //!< % position of COM along segment (thigh)
                const FLOAT_TYPE R2 = .6*.4*1.72,
                const FLOAT_TYPE grav = 9.81,
                const beacls::FloatVec& TMax = beacls::FloatVec{107 87},
                const beacls::FloatVec& TMin = beacls::FloatVec{-107 -60}, //!< see if this is actually how he uses it
                //const beacls::FloatVec& T1range = beacls::FloatVec{-107 107};
                //const beacls::FloatVec& T2range = beacls::FloatVec{-60 87}; //!< see if this is actually how he uses it
                //const beacls::FloatVec& TArange = beacls::FloatVec{-50 68};
				const beacls::FloatVec& dMax = beacls::FloatVec{ 0,0,0,0 },
                const beacls::IntegerVec& dims = beacls::IntegerVec{ 0,1,2,3 }
		);
		PREFIX_VC_DLL
			STS(
				beacls::MatFStream* fs,
				beacls::MatVariable* variable_ptr = NULL
			);
		PREFIX_VC_DLL
			virtual ~STS();
		PREFIX_VC_DLL
			virtual bool operator==(const STS& rhs) const;
		PREFIX_VC_DLL
			virtual bool operator==(const DynSys& rhs) const;
		PREFIX_VC_DLL
			virtual bool save(
				beacls::MatFStream* fs,
				beacls::MatVariable* variable_ptr = NULL);
		virtual STS* clone() const {
			return new STS(*this);
		}
		//bool getVelocity(beacls::FloatVec& v, std::vector<beacls::FloatVec>& vhist) const;
		/*
		@brief Control
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
		@brief disturbance
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
		@brief Dynamics
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
				const helperOC::DynSys_UMode_Type uMode
			) const;
		/*
		@brief disturbance
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
		@brief Dynamics
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
		STS& operator=(const STS& rhs);
		/** @overload
		Disable copy constructor
		*/
		STS(const STS& rhs) :
			DynSys(rhs),
			grav(rhs.grav),	//!< gravity
			TMax(rhs.TMax),	//!< max torque
            TMin(rhs.TMin),	//!< min torque
            R1(rhs.R1),
            R2(rhs.R2),
            M1(rhs.M1),
            M2(rhs.M2),
            L1(rhs.L1),
            L2(rhs.L0),
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
};
#endif	/* __STS_hpp__ */
