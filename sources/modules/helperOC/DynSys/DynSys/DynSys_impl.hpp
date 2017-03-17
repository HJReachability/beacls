#ifndef __DynSys_impl_hpp__
#define __DynSys_impl_hpp__

#include <vector>
#include <cstddef>
#include <typedef.hpp>
#include <helperOC/helperOC_type.hpp>
#include <iostream>
#include <cstring>
namespace helperOC {

	class Platoon;
	class Quadrotor;

	class DynSys;
	class DynSys_impl {
	public:

	private:
		const size_t nx;	//!< Number of state dimensions
		const size_t nu;	//!< Number of control inputs
		const size_t nd;	//!< Number of disturbance dimensions

		beacls::FloatVec x;	//!< State
		std::vector<beacls::FloatVec> u;	//!< Recent control signal

		std::vector<beacls::FloatVec> xhist;	//!< History of state
		std::vector<std::vector<beacls::FloatVec>> uhist;	//!< History of control

		const beacls::IntegerVec pdim;	//!< position dimensions
		const beacls::IntegerVec vdim;	//!< velocity dimensions
		const beacls::IntegerVec hdim;	//!< heading dimensions
		const beacls::IntegerVec TIdims;	//!< TI dimensions

		//! Figure handles
		HPxPy hpxpy;	//!< Position
		std::vector<HPxPy> hpxpyhist;	//!< Position history
		beacls::FloatVec hvxvy;	//!< Velocity
		std::vector<beacls::FloatVec> hvxvyhist;	//!< Velocity history

		//!< Position velocity(so far only used in DoubleInt)
		beacls::FloatVec hpv;
		std::vector<beacls::FloatVec> hpvhist;

		//! Data(any data that you may want to store for convenience)
		beacls::FloatVec data;

		const helperOC::PartialFunction_cuda* partialFunction;
	public:
		DynSys_impl();
		DynSys_impl(
			const size_t nx,
			const size_t nu,
			const size_t nd,
			const beacls::IntegerVec& pdim,
			const beacls::IntegerVec& hdim,
			const beacls::IntegerVec& vdim = beacls::IntegerVec(),
			const beacls::IntegerVec& Tdims = beacls::IntegerVec());
		DynSys_impl(
			const size_t nx,
			const size_t nu,
			const size_t nd,
			const beacls::IntegerVec& pdim,
			const beacls::IntegerVec& hdim,
			const beacls::IntegerVec& vdim,
			const beacls::IntegerVec& Tdims,
			beacls::MatFStream* fs,
			beacls::MatVariable* variable_ptr);
		~DynSys_impl();
		bool operator==(const helperOC::DynSys_impl& rhs) const;
		bool save(
			beacls::MatFStream* fs,
			beacls::MatVariable* variable_ptr);

		bool ode113(
			beacls::FloatVec& dst_x,
			const DynSys* dynSys,
			const std::vector<beacls::FloatVec >& src_u,
			const std::vector<beacls::FloatVec >& src_d,
			const beacls::FloatVec& tspan,
			const beacls::FloatVec& x0
		);
		bool plotPosition(const DynSys* dynSys, helperOC::PlotExtraArgs& extraArgs);
		bool updateState(
			beacls::FloatVec& x1,
			const DynSys* dynSys,
			const std::vector<beacls::FloatVec >& src_u,
			const FLOAT_TYPE T,
			const beacls::FloatVec& x0,
			const std::vector<beacls::FloatVec >& d
		);
		//	void set_nx(const size_t val) { this->nx = val; }
		//	void set_nu(const size_t val) { this->nu = val; }
		//	void set_nd(const size_t val) { this->nd = val; }

		void set_x(const beacls::FloatVec& val) { this->x = val; }
		void set_u(const std::vector<beacls::FloatVec>& val) { this->u = val; }

		void set_xhist(const std::vector<beacls::FloatVec>& val) { this->xhist = val; }
		void set_uhist(const std::vector<std::vector<beacls::FloatVec>>& val) { this->uhist = val; }

		void clear_x() { this->x.clear(); }
		void clear_u() { this->u.clear(); }

		void clear_xhist() { this->xhist.clear(); }
		void clear_uhist() { this->uhist.clear(); }

		void push_back_xhist(const beacls::FloatVec& val) { this->xhist.push_back(val); }
		void push_back_uhist(const std::vector<beacls::FloatVec>& val) { this->uhist.push_back(val); }

		//	void set_pdim(const beacls::IntegerVec& val) { this->pdim = val; }
		//	void set_vdim(const beacls::IntegerVec& val) { this->vdim = val; }
		//	void set_hdim(const beacls::IntegerVec& val) { this->hdim = val; }
		//	void set_TIdims(const beacls::IntegerVec& val) { this->TIdims = val; }

			//! Figure handles
		void set_hpxpy(const HPxPy &val) { this->hpxpy = val; }	//!< Position
		void set_hvxvy(const beacls::FloatVec &val) { this->hvxvy = val; }	//!< Velocity
		void set_hpxpyhist(const std::vector<HPxPy> &val) { this->hpxpyhist = val; }	//!< Position history
		void set_hvxvyhist(const std::vector<beacls::FloatVec> &val) { this->hvxvyhist = val; }	//!< Velocity history

		void clear_hpxpyhist() { this->hpxpyhist.clear(); }
		void clear_hvxvyhist() { this->hvxvyhist.clear(); }

		void push_back_hpxpyhist(const HPxPy &val) { this->hpxpyhist.push_back(val); }	//!< Position history
		void push_back_hvxvyhist(const beacls::FloatVec &val) { this->hvxvyhist.push_back(val); }	//!< Velocity history

		//!< Position velocity(so far only used in DoubleInt)
		void set_hpv(const beacls::FloatVec &val) { this->hpv = val; }
		void set_hpvhist(const std::vector<beacls::FloatVec> &val) { this->hpvhist = val; }

		void clear_hpvhist() { this->hpvhist.clear(); }

		void push_back_hpvhist(const beacls::FloatVec &val) { this->hpvhist.push_back(val); }

		//! Data(any data that you may want to store for convenience)
		void set_data(const beacls::FloatVec &val) { this->data = val; }

		void set_partialFunction(const helperOC::PartialFunction_cuda* val) { this->partialFunction = val; }

		size_t get_nx() const { return nx; }
		size_t get_nu() const { return nu; }
		size_t get_nd() const { return nd; }

		beacls::FloatVec get_x() const { return x; }
		std::vector<beacls::FloatVec> get_u() const { return u; }

		std::vector<beacls::FloatVec> get_xhist() const { return xhist; }
		std::vector<std::vector<beacls::FloatVec>> get_uhist() const { return uhist; }

		beacls::IntegerVec get_pdim() const { return pdim; }
		beacls::IntegerVec get_vdim() const { return vdim; }
		beacls::IntegerVec get_hdim() const { return hdim; }
		beacls::IntegerVec get_TIdims() const { return TIdims; }

		//! Figure handles
		HPxPy get_hpxpy() const { return hpxpy; }
		beacls::FloatVec get_hvxvy() const { return hvxvy; }
		std::vector<HPxPy> get_hpxpyhist() const { return hpxpyhist; }
		std::vector<beacls::FloatVec> get_hvxvyhist() const { return hvxvyhist; }

		//!< Position velocity(so far only used in DoubleInt)
		beacls::FloatVec get_hpv() const { return hpv; }
		std::vector<beacls::FloatVec> get_hpvhist() const { return hpvhist; }

		//! Data(any data that you may want to store for convenience)
		beacls::FloatVec get_data() const { return data; }

		const helperOC::PartialFunction_cuda* get_partialFunction() const { return partialFunction; }
		DynSys_impl* clone() const;
	private:
		/** @overload
		Disable operator=
		*/
		DynSys_impl& operator=(const helperOC::DynSys_impl& rhs);
		/** @overload
		Disable copy constructor
		*/
		DynSys_impl(const helperOC::DynSys_impl& rhs) :
			nx(rhs.nx),
			nu(rhs.nu),
			nd(rhs.nd),
			x(rhs.x),
			u(rhs.u),
			xhist(rhs.xhist),
			uhist(rhs.uhist),
			pdim(rhs.pdim),
			vdim(rhs.vdim),
			hdim(rhs.hdim),
			TIdims(rhs.TIdims),
			hpxpy(rhs.hpxpy),
			hpxpyhist(rhs.hpxpyhist),
			hvxvy(rhs.hvxvy),
			hvxvyhist(rhs.hvxvyhist),
			hpv(rhs.hpv),
			hpvhist(rhs.hpvhist),
			data(rhs.data),
			partialFunction(rhs.partialFunction)
		{}
	};
};
#endif	/* __DynSys_impl_hpp__ */
