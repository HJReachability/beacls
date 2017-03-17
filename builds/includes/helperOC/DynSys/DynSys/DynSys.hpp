#ifndef __DynSys_hpp__
#define __DynSys_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#include <vector>
#include <deque>
#include <cstddef>
#include <limits>
#include <utility>
using namespace std::rel_ops;

#include <typedef.hpp>
#include <helperOC/DynSys/DynSys/DynSysTypeDef.hpp>
#include <helperOC/helperOC_type.hpp>
#include <iostream>
#include <cstring>
namespace helperOC {
	class DynSys_impl;
	class Platoon;
	class Quadrotor;

	class HPxPy {
	public:
		beacls::FloatVec XData;
		beacls::FloatVec YData;
		FLOAT_TYPE UData;
		FLOAT_TYPE VData;
		std::string Marker;
		beacls::FloatVec Color;
		beacls::FloatVec MarkerFaceColor;
		FLOAT_TYPE MarkerSize;
		FLOAT_TYPE MaxHeadSize;
		FLOAT_TYPE LineWidth;
		bool save(
			beacls::MatFStream* fs,
			beacls::MatVariable* variable_ptr = NULL
		);
		HPxPy(
			beacls::MatFStream* fs,
			beacls::MatVariable* variable_ptr = NULL);
		HPxPy();
		bool operator==(const HPxPy& rhs) const;
	};

	/*
	@brief Dynamical Systems class
	Subclasses : quadrotor, Dubins vehicle(under construction)
	*/
	class DynSys {
	public:

	private:
		DynSys_impl* pimpl;
	public:
		PREFIX_VC_DLL
			DynSys();
		PREFIX_VC_DLL
			DynSys(
				const size_t nx,
				const size_t nu,
				const size_t nd = 0,
				const beacls::IntegerVec& pdim = beacls::IntegerVec(),
				const beacls::IntegerVec& hdim = beacls::IntegerVec(),
				const beacls::IntegerVec& vdim = beacls::IntegerVec(),
				const beacls::IntegerVec& TIdims = beacls::IntegerVec());
		PREFIX_VC_DLL
			DynSys(
				beacls::MatFStream* fs,
				beacls::MatVariable* variable_ptr);
		PREFIX_VC_DLL
			virtual ~DynSys() = 0;
		PREFIX_VC_DLL
			virtual bool operator==(const DynSys& rhs) const;
		PREFIX_VC_DLL
			virtual bool save(
				beacls::MatFStream* fs,
				beacls::MatVariable* variable_ptr);
		/*
		@brief	returns the position and optionally position history of the vehicle
		*/
		PREFIX_VC_DLL
			bool getPosition(beacls::FloatVec& p, std::vector<beacls::FloatVec>& phist) const;
		/*
		@brief	returns the heading and optionally heading history of the vehicle
		*/
		PREFIX_VC_DLL
			bool getHeading(beacls::FloatVec& h, std::vector<beacls::FloatVec>& hhist) const;
		/*
		@brief	returns the velocity and optinally the velocity history of the vehicle
		*/
		PREFIX_VC_DLL
			virtual bool getVelocity(beacls::FloatVec& v, std::vector<beacls::FloatVec>& vhist) const;

		virtual DynSys* clone() const = 0;
		/*
		@brief	Plots the current state and the trajectory of the quadrotor
		@param	[in]	extraArgs	color for plotting
		*/
		PREFIX_VC_DLL
			bool plotPosition(helperOC::PlotExtraArgs& extraArgs);
		/*
		@brief	Updates state based on control
		@param	[in]	u	control (defaults to previous control)
		@param	[in]	T	duration to hold control
		@param	[in]	x0	initial state (defaults to current state if set to [])
		@param	[in]	d	disturbance (defaults to [])
		@param	[out]	x1	final state
		*/
		PREFIX_VC_DLL
			bool updateState(
				beacls::FloatVec& x1,
				const std::vector<beacls::FloatVec >& u,
				const FLOAT_TYPE T,
				const beacls::FloatVec& x0 = beacls::FloatVec(),
				const std::vector<beacls::FloatVec >& d = std::vector<beacls::FloatVec>()
			);
		/*
		@brief Default optimal control function for systems with no control
		*/
		PREFIX_VC_DLL
			virtual bool optCtrl(
				std::vector<beacls::FloatVec >& uOpts,
				const FLOAT_TYPE t,
				const std::vector<beacls::FloatVec::const_iterator >& y_ites,
				const std::vector<const FLOAT_TYPE*>& deriv_ptrs,
				const beacls::IntegerVec& y_sizes,
				const beacls::IntegerVec& deriv_sizes,
				const helperOC::DynSys_UMode_Type uMode = helperOC::DynSys_UMode_Default
			) const;
		/*
		@brief Default optimal disturbance function for systems with no disturbance
		*/
		PREFIX_VC_DLL
			virtual bool optDstb(
				std::vector<beacls::FloatVec >& uOpts,
				const FLOAT_TYPE t,
				const std::vector<beacls::FloatVec::const_iterator >& y_ites,
				const std::vector<const FLOAT_TYPE*>& deriv_ptrs,
				const beacls::IntegerVec& y_sizes,
				const beacls::IntegerVec& deriv_sizes,
				const helperOC::DynSys_DMode_Type dMode = helperOC::DynSys_DMode_Default
			) const;
		/*
		@brief % Pure virtual function of Dynamics, declared to define interface.
		*/
		virtual bool dynamics(
			std::vector<beacls::FloatVec >& dx,
			const FLOAT_TYPE t,
			const std::vector<beacls::FloatVec::const_iterator >& x_ites,
			const std::vector<beacls::FloatVec >& us,
			const std::vector<beacls::FloatVec >& ds,
			const beacls::IntegerVec& x_sizes,
			const size_t dst_target_dim = std::numeric_limits<size_t>::max()
		) const = 0;
		/*
		@brief % Default function of TI Dynamics, declared to define interface.
		*/
		PREFIX_VC_DLL
			virtual bool TIdyn(
				std::vector<beacls::FloatVec >& TIdx,
				const FLOAT_TYPE t,
				const std::vector<beacls::FloatVec::const_iterator >& x_ites,
				const std::vector<beacls::FloatVec >& us,
				const std::vector<beacls::FloatVec >& ds,
				const beacls::IntegerVec& x_sizes,
				const size_t dst_target_dim = std::numeric_limits<size_t>::max()
			) const;
#if defined(USER_DEFINED_GPU_DYNSYS_FUNC)
		/*
		@brief Default optimal control function for systems with no control
		*/
		PREFIX_VC_DLL
			virtual bool optCtrl_cuda(
				std::vector<beacls::UVec>& u_uvecs,
				const FLOAT_TYPE t,
				const std::vector<beacls::UVec>& x_uvecs,
				const std::vector<beacls::UVec>& deriv_uvecs,
				const helperOC::DynSys_UMode_Type uMode = helperOC::DynSys_UMode_Default
			) const;
		/*
		@brief Default optimal disturbance function for systems with no disturbance
		*/
		PREFIX_VC_DLL
			virtual bool optDstb_cuda(
				std::vector<beacls::UVec>& d_uvecs,
				const FLOAT_TYPE t,
				const std::vector<beacls::UVec>& x_uvecs,
				const std::vector<beacls::UVec>& deriv_uvecs,
				const helperOC::DynSys_DMode_Type dMode = helperOC::DynSys_DMode_Default
			) const;
		/*
		@brief % Pure virtual function of Dynamics, declared to define interface.
		*/
		PREFIX_VC_DLL
			virtual bool dynamics_cuda(
				std::vector<beacls::UVec>& dx_uvecs,
				const FLOAT_TYPE t,
				const std::vector<beacls::UVec>& x_uvecs,
				const std::vector<beacls::UVec>& u_uvecs,
				const std::vector<beacls::UVec>& d_uvecs,
				const size_t dst_target_dim = std::numeric_limits<size_t>::max()
			) const;
		/*
		@brief Default optimal control function for systems with no control
		*/
		PREFIX_VC_DLL
			virtual bool optCtrl_cuda(
				std::vector<beacls::UVec>& uL_uvecs,
				std::vector<beacls::UVec>& uU_uvecs,
				const FLOAT_TYPE t,
				const std::vector<beacls::UVec>& x_uvecs,
				const std::vector<beacls::UVec>& derivMin_uvecs,
				const std::vector<beacls::UVec>& derivMax_uvecs,
				const helperOC::DynSys_UMode_Type uMode = helperOC::DynSys_UMode_Default
			) const;
		/*
		@brief Default optimal disturbance function for systems with no disturbance
		*/
		PREFIX_VC_DLL
			virtual bool optDstb_cuda(
				std::vector<beacls::UVec>& dL_uvecs,
				std::vector<beacls::UVec>& dU_uvecs,
				const FLOAT_TYPE t,
				const std::vector<beacls::UVec>& x_uvecs,
				const std::vector<beacls::UVec>& derivMin_uvecs,
				const std::vector<beacls::UVec>& derivMax_uvecs,
				const helperOC::DynSys_DMode_Type dMode = helperOC::DynSys_DMode_Default
			) const;
		/*
		@brief % Pure virtual function of Dynamics, declared to define interface.
		*/
		PREFIX_VC_DLL
			virtual bool dynamics_cuda(
				beacls::UVec& alpha_uvecs,
				const FLOAT_TYPE t,
				const std::vector<beacls::UVec>& x_uvecs,
				const std::vector<beacls::UVec>& uL_uvecs,
				const std::vector<beacls::UVec>& uU_uvecs,
				const std::vector<beacls::UVec>& dL_uvecs,
				const std::vector<beacls::UVec>& dU_uvecs,
				const size_t dst_target_dim = std::numeric_limits<size_t>::max()
			) const;
		/*
		@brief % Default function of TI Dynamics, declared to define interface.
		*/
		PREFIX_VC_DLL
			virtual bool TIdyn_cuda(
				std::vector<beacls::UVec>& TIdx_uvecs,
				const FLOAT_TYPE t,
				const std::vector<beacls::UVec>& x_uvecs,
				const std::vector<beacls::UVec>& u_uvecs,
				const std::vector<beacls::UVec>& d_uvecs,
				const size_t dst_target_dim = std::numeric_limits<size_t>::max()
			) const;
		PREFIX_VC_DLL
			virtual bool HamFunction_cuda(
				beacls::UVec& hamValue_uvec,
				const DynSysSchemeData* schemeData,
				const FLOAT_TYPE t,
				const beacls::UVec& data,
				const std::vector<beacls::UVec>& x_uvecs,
				const std::vector<beacls::UVec>& derivs,
				const size_t begin_index,
				const size_t length,
				const bool negate
			) const;
		PREFIX_VC_DLL
			virtual bool PartialFunction_cuda(
				beacls::UVec& alpha_uvec,
				const DynSysSchemeData* schemeData,
				const FLOAT_TYPE t,
				const beacls::UVec& data,
				const std::vector<beacls::UVec>& x_uvecs,
				const std::vector<beacls::UVec>& derivMin_uvecs,
				const std::vector<beacls::UVec>& derivMax_uvecs,
				const size_t dim,
				const size_t begin_index,
				const size_t length
			) const;
#endif /* defined(USER_DEFINED_GPU_DYNSYS_FUNC) */
		size_t find_val(const beacls::IntegerVec& vec, const size_t value) const;
		PREFIX_VC_DLL
			size_t get_nx() const;
		PREFIX_VC_DLL
			size_t get_nu() const;
		PREFIX_VC_DLL
			size_t get_nd() const;
		PREFIX_VC_DLL
			beacls::FloatVec get_x() const;
		PREFIX_VC_DLL
			std::vector<beacls::FloatVec> get_u() const;

		PREFIX_VC_DLL
			std::vector<beacls::FloatVec> get_xhist() const;	//!< History of state
		PREFIX_VC_DLL
			std::vector<std::vector<beacls::FloatVec>> get_uhist() const;	//!< History of control

		beacls::IntegerVec get_pdim() const;	//!< position dimensions
		beacls::IntegerVec get_vdim() const;	//!< velocity dimensions
		beacls::IntegerVec get_hdim() const;	//!< heading dimensions

		beacls::IntegerVec get_TIdims() const;	//!< TI dimensions

									//! Figure handles
		HPxPy get_hpxpy() const;	//!< Position
		std::vector<HPxPy> get_hpxpyhist() const;	//!< Position history
		beacls::FloatVec get_hvxvy() const;	//!< Velocity
		std::vector<beacls::FloatVec> get_hvxvyhist() const;	//!< Velocity history

														//!< Position velocity(so far only used in DoubleInt)
		beacls::FloatVec get_hpv() const;
		std::vector<beacls::FloatVec> get_hpvhist() const;

		beacls::FloatVec get_data() const;
		const helperOC::PartialFunction_cuda* get_partialFunction() const;

		//	void set_nx(const size_t nx);	//!< Number of state dimensions
		//	void set_nu(const size_t nu);	//!< Number of control inputs
		//	void set_nd(const size_t nd);	//!< Number of disturbance dimensions

		PREFIX_VC_DLL
			void set_x(const beacls::FloatVec& x);	//!< State
		PREFIX_VC_DLL
			void set_u(const std::vector<beacls::FloatVec>& u);	//!< Recent control signal

		PREFIX_VC_DLL
			void set_xhist(const std::vector<beacls::FloatVec>& xhist);	//!< History of state
		PREFIX_VC_DLL
			void set_uhist(const std::vector<std::vector<beacls::FloatVec>>& uhist);	//!< History of control

		//	void set_pdim(const beacls::IntegerVec& pdim);	//!< position dimensions
		//	void set_vdim(const beacls::IntegerVec& vdim);	//!< velocity dimensions
		//	void set_hdim(const beacls::IntegerVec& hdim);	//!< heading dimensions

		//	void set_TIdims(const beacls::IntegerVec& TIdims);	//!< TI dimensions

			//! Platoon - related properties

			//! Figure handles
		void set_hpxpy(const HPxPy& hpxpy);	//!< Position
		void set_hpxpyhist(const std::vector<HPxPy>& hpxpyhist);	//!< Position history
		void set_hvxvy(const beacls::FloatVec& hvxvy);	//!< Velocity
		void set_hvxvyhist(const std::vector<beacls::FloatVec>& hvxvyhist);	//!< Velocity history

		//!< Position velocity(so far only used in DoubleInt)
		void set_hpv(const beacls::FloatVec& hpv);
		void set_hpvhist(const std::vector<beacls::FloatVec>& hpvhist);

		//! Data(any data that you may want to store for convenience)
		void set_data(const beacls::FloatVec& data);
		void set_partialFunction(const helperOC::PartialFunction_cuda* partialFunction);

		PREFIX_VC_DLL
			void clear_x();
		PREFIX_VC_DLL
			void clear_u();
		PREFIX_VC_DLL
			void clear_xhist();
		PREFIX_VC_DLL
			void clear_uhist();

		PREFIX_VC_DLL
			void push_back_xhist(const beacls::FloatVec& val);
		PREFIX_VC_DLL
			void push_back_uhist(const std::vector<beacls::FloatVec>& val);

	protected:
		/** @overload
		Disable copy constructor
		*/
		PREFIX_VC_DLL
			DynSys(const DynSys& rhs);
	private:
		/** @overload
		Disable operator=
		*/
		DynSys& operator=(const DynSys& rhs);
	};
};
#endif	/* __DynSys_hpp__ */
