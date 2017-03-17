#ifndef __Plane4D_hpp__
#define __Plane4D_hpp__

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
		Dynamics:
		Dynamics of the Plane4D
		\dot{x}_1 = x_4 * cos(x_3) + d_1
		\dot{x}_2 = x_4 * sin(x_3) + d_2
		\dot{x}_3 = u_1 = w
		\dot{x}_4 = u_2 = a
		wMin <= w <= wMax
		aMin <= a <= aMax
	*/
	class Plane4D : public DynSys {
	public:
	protected:
		FLOAT_TYPE wMax;	//!< Angular control bounds
		beacls::FloatVec aRange;	//!< Acceleration control bounds
		beacls::FloatVec dMax;	//!< Disturbance
		beacls::IntegerVec dims;	//!< Dimensions that are active
	public:
		/*
		@brief Constructor. Creates a plane object with a unique ID,
			state x, and reachable set information reachInfo
		@param	[in]		x	state [xpos; ypos; theta]
		@param	[in]		wMax	maximum turn rate
		@param	[in]		vrange	speed range
		@param	[in]		dMax	disturbance bounds
		@return	a Plane object
		*/
		PREFIX_VC_DLL
			Plane4D(
				const beacls::FloatVec& x,
				const FLOAT_TYPE wMax,
				const beacls::FloatVec& aRange,
				const beacls::FloatVec& dMax = beacls::FloatVec{ 0,0 },
				const beacls::IntegerVec& dims = beacls::IntegerVec{ 0,1,2,3 }
		);
		PREFIX_VC_DLL
			Plane4D(
				beacls::MatFStream* fs,
				beacls::MatVariable* variable_ptr = NULL
			);
		PREFIX_VC_DLL
			virtual ~Plane4D();
		PREFIX_VC_DLL
			virtual bool operator==(const Plane4D& rhs) const;
		PREFIX_VC_DLL
			virtual bool operator==(const DynSys& rhs) const;
		virtual Plane4D* clone() const {
			return new Plane4D(*this);
		}
		PREFIX_VC_DLL
			virtual bool save(
				beacls::MatFStream* fs,
				beacls::MatVariable* variable_ptr = NULL
			);
		/*
		@brief Optimal control function
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
				const helperOC::DynSys_DMode_Type dMode
			) const;
		/*
		@brief 	Dynamics of the Plane4D
		\dot{x}_1 = x_4 * cos(x_3) + d_1
		\dot{x}_2 = x_4 * sin(x_3) + d_2
		\dot{x}_3 = u_1 = u_1
		\dot{x}_4 = u_2 = u_2
		wMin <= w <= wMax
		aMin <= a <= aMax
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
		/*
		@brief Optimal control function
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
		@brief Optimal disturbance function
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
		@brief 	Dynamics of the Plane4D
		\dot{x}_1 = x_4 * cos(x_3) + d_1
		\dot{x}_2 = x_4 * sin(x_3) + d_2
		\dot{x}_3 = u_1 = u_1
		\dot{x}_4 = u_2 = u_2
		wMin <= w <= wMax
		aMin <= a <= aMax
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
		Plane4D& operator=(const Plane4D& rhs);
		/** @overload
		Disable copy constructor
		*/
		Plane4D(const Plane4D& rhs) :
			DynSys(rhs),
			wMax(rhs.wMax),	//!< Angular control bounds
			aRange(rhs.aRange),	//!< Acceleration control bounds
			dMax(rhs.dMax),	//!< Disturbance
			dims(rhs.dims)	//!< Dimensions that are active
		{}
		bool dynamics_cell_helper(
			std::vector<beacls::FloatVec >& dxs,
			const beacls::FloatVec::const_iterator& x_ites2,
			const beacls::FloatVec::const_iterator& x_ites3,
			const std::vector<beacls::FloatVec >& us,
			const std::vector<beacls::FloatVec >& ds,
			const size_t x2_size,
			const size_t x3_size,
			const size_t dim
		) const;
		bool optCtrl0_cell_helper(
			beacls::FloatVec& uOpt0,
			const std::vector<const FLOAT_TYPE* >& derivs,
			const beacls::IntegerVec& deriv_sizes,
			const helperOC::DynSys_UMode_Type uMode,
			const size_t src_target_dim_index
		) const;
		bool optCtrl1_cell_helper(
			beacls::FloatVec& uOpt1,
			const std::vector<const FLOAT_TYPE* >& derivs,
			const beacls::IntegerVec& deriv_sizes,
			const helperOC::DynSys_UMode_Type uMode,
			const size_t src_target_dim_index
		) const;
		bool optDstb0_cell_helper(
			beacls::FloatVec& dOpt0,
			const std::vector<const FLOAT_TYPE* >& derivs,
			const beacls::IntegerVec& deriv_sizes,
			const helperOC::DynSys_DMode_Type dMode,
			const size_t src_target_dim_index
		) const;
		bool optDstb1_cell_helper(
			beacls::FloatVec& dOpt1,
			const std::vector<const FLOAT_TYPE* >& derivs,
			const beacls::IntegerVec& deriv_sizes,
			const helperOC::DynSys_DMode_Type dMode,
			const size_t src_target_dim_index
		) const;
	};
};
#endif	/* __Plane4D_hpp__ */
