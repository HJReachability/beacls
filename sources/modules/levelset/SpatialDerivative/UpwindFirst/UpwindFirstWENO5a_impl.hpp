#ifndef __UpwindFirstWENO5a_impl_hpp__
#define __UpwindFirstWENO5a_impl_hpp__

#include <cstdint>
#include <vector>
#include <typedef.hpp>
#include <Core/UVec.hpp>

namespace levelset {
	class HJI_Grid;
	class UpwindFirstENO3aHelper;
	class UpwindFirstWENO5a_Cache {
	public:
		std::vector<beacls::FloatVec > last_d1ss;
		std::vector<beacls::FloatVec > last_d2ss;
		std::vector<beacls::FloatVec > last_d3ss;
		std::vector<beacls::FloatVec > last_dx_d2ss;
		//		beacls::FloatVec last_d2s;
		//beacls::FloatVec last_d2s_fabs;
		//beacls::FloatVec last_dx_d2_effs;
		std::vector<const FLOAT_TYPE*> boundedSrc_ptrs;
		UpwindFirstWENO5a_Cache() : boundedSrc_ptrs(2) {}
		/** @overload
		Disable copy constructor
		*/
		UpwindFirstWENO5a_Cache(const UpwindFirstWENO5a_Cache& rhs) :
			last_d1ss(rhs.last_d1ss),
			last_d2ss(rhs.last_d2ss),
			last_d3ss(rhs.last_d3ss),
			last_dx_d2ss(rhs.last_dx_d2ss) {
			boundedSrc_ptrs.resize(rhs.boundedSrc_ptrs.size(), NULL);
		}
	};

	class UpwindFirstWENO5a_impl {
	private:
		beacls::UVecType type;
		std::vector<beacls::UVec> tmpBoundedSrc_uvec_vectors;
		beacls::FloatVec dxs;
		beacls::FloatVec dx_squares;
		beacls::FloatVec dxInvs;
		beacls::FloatVec dxInv_2s;
		beacls::FloatVec dxInv_3s;

		beacls::IntegerVec target_dimension_loop_sizes;
		beacls::IntegerVec inner_dimensions_loop_sizes;
		beacls::IntegerVec first_dimension_loop_sizes;
		beacls::IntegerVec src_target_dimension_loop_sizes;


		const size_t stencil;
		std::vector<beacls::UVec > tmpBoundedSrc_uvecs;
		std::vector<std::vector<std::vector<std::vector<const FLOAT_TYPE*> > > > tmpBoundedSrc_ptrssss;
		std::vector<FLOAT_TYPE*> tmp_d1s_ms_ites;
		std::vector<FLOAT_TYPE*> tmp_d2s_ms_ites;
		std::vector<FLOAT_TYPE*> tmp_d3s_ms_ites;
		std::vector<FLOAT_TYPE*> tmp_dx_d2s_ms_ites;
		size_t num_of_strides;

		std::vector<std::vector<beacls::UVec > > dL_uvecs;
		std::vector<std::vector<beacls::UVec > > dR_uvecs;
		std::vector<std::vector<beacls::UVec > > DD_uvecs;

		std::vector<beacls::FloatVec > d1ss;
		std::vector<beacls::FloatVec > d2ss;
		std::vector<beacls::FloatVec > d3ss;

		beacls::IntegerVec tmp_cache_indexes;

		std::vector<std::vector<beacls::FloatVec > > tmpSmooths_m1s;
		std::vector<beacls::FloatVec > tmpSmooths;

		const beacls::FloatVec weightL = { (FLOAT_TYPE)0.1, (FLOAT_TYPE)0.6,(FLOAT_TYPE)0.3 };
		const beacls::FloatVec weightR = { (FLOAT_TYPE)0.3, (FLOAT_TYPE)0.6,(FLOAT_TYPE)0.1 };

		levelset::EpsilonCalculationMethod_Type epsilonCalculationMethod_Type;
		UpwindFirstENO3aHelper* upwindFirstENO3aHelper;
		std::vector<beacls::CudaStream*> cudaStreams;
		UpwindFirstWENO5a_Cache *cache;
	public:
		UpwindFirstWENO5a_impl(
			const HJI_Grid *hji_grid,
			const beacls::UVecType type = beacls::UVecType_Vector
		);
		~UpwindFirstWENO5a_impl();

		bool execute_dim0(
			beacls::UVec& dst_deriv_l,
			beacls::UVec& dst_deriv_r,
			const FLOAT_TYPE* src,
			const HJI_Grid *grid,
			const size_t dim,
			const size_t loop_begin,
			const size_t slice_length,
			const size_t num_of_slices
		);
		bool execute_dim1(
			beacls::UVec& dst_deriv_l,
			beacls::UVec& dst_deriv_r,
			const FLOAT_TYPE* src,
			const HJI_Grid *grid,
			const size_t dim,
			const size_t loop_begin,
			const size_t slice_length,
			const size_t num_of_slices
		);
		bool execute_dimLET2(
			beacls::UVec& dst_deriv_l,
			beacls::UVec& dst_deriv_r,
			const FLOAT_TYPE* src,
			const HJI_Grid *grid,
			const size_t dim,
			const size_t loop_begin,
			const size_t slice_length,
			const size_t num_of_slices
		);
		bool execute(
			beacls::UVec& dst_deriv_l,
			beacls::UVec& dst_deriv_r,
			const HJI_Grid *grid,
			const FLOAT_TYPE* src,
			const size_t dim,
			const bool generateAll,
			const size_t loop_begin,
			const size_t slice_length,
			const size_t num_of_slices
		);
		bool synchronize(const size_t dim);
		bool operator==(const UpwindFirstWENO5a_impl& rhs) const;
		UpwindFirstWENO5a_impl* clone() const {
			return new UpwindFirstWENO5a_impl(*this);
		};
		beacls::UVecType get_type() const {
			return type;
		};
	private:
		void getCachePointers(
			std::vector<FLOAT_TYPE*> &d1s_ms,
			std::vector<FLOAT_TYPE*> &d2s_ms,
			std::vector<FLOAT_TYPE*> &d3s_ms,
			std::vector<FLOAT_TYPE*> &dx_d2ss,
			const size_t shifted_target_dimension_loop_index,
			const size_t num_of_cache_lines);
		void createCaches(
			const size_t first_dimension_loop_size,
			const size_t num_of_cache_lines);


		/** @overload
		Disable operator=
		*/
		UpwindFirstWENO5a_impl& operator=(const UpwindFirstWENO5a_impl& rhs);
		/** @overload
		Disable copy constructor
		*/
		UpwindFirstWENO5a_impl(const UpwindFirstWENO5a_impl& rhs);
	};
};
#endif	/* __UpwindFirstWENO5a_impl_hpp__ */
