#ifndef __Quad4D_hpp__
#define __Quad4D_hpp__

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
		Dynamics of the Quad4D
			\dot{p}_x = v_x
			\dot{v}_x = u_x
			\dot{p}_y = v_y
			\dot{v}_y = u_y
			uMin <= u_x <= uMax
		*/
	class Quad4D : public DynSys {
	public:
	protected:
		FLOAT_TYPE uMin;	//!< Control bounds
		FLOAT_TYPE uMax;	//!< Control bounds
		beacls::IntegerVec dims;	//!< Active Dimensions
	public:
		/*
		@brief Constructor. Creates a plane object with a unique ID,
			state x, and reachable set information reachInfo
		@param	[in]		x	state: [xpos; xvel; ypos; yvel]
		@param	[in]		uMin	Control bounds
		@param	[in]		uMin	Control bounds
		@param	[in]		dims	Active Dimensions
		@return	a Quad4D object
		*/
		PREFIX_VC_DLL
			Quad4D(
				const beacls::FloatVec& x,
				const FLOAT_TYPE uMin = -3,
				const FLOAT_TYPE uMax = 3,
				const beacls::IntegerVec& dims = beacls::IntegerVec{ 0,1,2,3 }
		);
		PREFIX_VC_DLL
			Quad4D(
				beacls::MatFStream* fs,
				beacls::MatVariable* variable_ptr = NULL
			);
		PREFIX_VC_DLL
			virtual ~Quad4D();
		PREFIX_VC_DLL
			virtual bool operator==(const Quad4D& rhs) const;
		PREFIX_VC_DLL
			virtual bool operator==(const DynSys& rhs) const;
		virtual Quad4D* clone() const {
			return new Quad4D(*this);
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
		@brief 	Dynamics of the Quad4D
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
		@brief 	Dynamics of the Quad4D
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
		Quad4D& operator=(const Quad4D& rhs);
		/** @overload
		Disable copy constructor
		*/
		Quad4D(const Quad4D& rhs) :
			DynSys(rhs),
			uMin(rhs.uMin),	//!< Control bounds
			uMax(rhs.uMax),	//!< Control bounds
			dims(rhs.dims)	//!< Active dimensions
		{}
		bool dynamics_cell_helper(
			beacls::FloatVec& dx,
			const std::vector<beacls::FloatVec::const_iterator >& x_ites,
			const std::vector<beacls::FloatVec >& us,
			const beacls::IntegerVec& x_sizes,
			const size_t dim
		) const;
	};
};
#endif	/* __Quad4D_hpp__ */
