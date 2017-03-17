#ifndef __UpwindFirstENO3aHelper_impl_hpp__
#define __UpwindFirstENO3aHelper_impl_hpp__

#include <cstdint>
#include <vector>
#include <typedef.hpp>
#include <Core/UVec.hpp>
namespace levelset {

	class BoundaryCondition;
	class HJI_Grid;

	class UpwindFirstENO3aHelper_Cache {
	public:
		std::vector<beacls::FloatVec > last_d1ss;
		std::vector<beacls::FloatVec > last_d2ss;
		std::vector<beacls::FloatVec > last_d3ss;
		std::vector<const FLOAT_TYPE*> boundedSrc_ptrs;
		UpwindFirstENO3aHelper_Cache() :
			boundedSrc_ptrs(2) {}
		/** @overload
		Disable copy constructor
		*/
		UpwindFirstENO3aHelper_Cache(const UpwindFirstENO3aHelper_Cache& rhs) :
			last_d1ss(rhs.last_d1ss),
			last_d2ss(rhs.last_d2ss),
			last_d3ss(rhs.last_d3ss) {
			boundedSrc_ptrs.resize(rhs.boundedSrc_ptrs.size(), NULL);
		}
	private:
		/** @overload
		Disable operator=
		*/
		UpwindFirstENO3aHelper_Cache& operator=(const UpwindFirstENO3aHelper_Cache& rhs);
	};

	class UpwindFirstENO3aHelper_impl {
	private:
		beacls::UVecType type;
		std::vector<beacls::UVec> tmpBoundedSrc_uvec_vectors;
		beacls::FloatVec dxs;
		beacls::FloatVec dx_squares;
		beacls::FloatVec dxInvs;
		beacls::FloatVec dxInv_2s;
		beacls::FloatVec dxInv_3s;

		beacls::IntegerVec outer_dimensions_loop_sizes;
		beacls::IntegerVec target_dimension_loop_sizes;
		beacls::IntegerVec inner_dimensions_loop_sizes;
		beacls::IntegerVec first_dimension_loop_sizes;
		beacls::IntegerVec src_target_dimension_loop_sizes;
		beacls::UVec bounded_first_dimension_line_cache_uvec;

		const size_t stencil;
		std::vector<beacls::UVec > tmpBoundedSrc_uvecs;
		std::vector<std::vector<std::vector<std::vector<const FLOAT_TYPE*> > > > tmpBoundedSrc_ptrssss;
		std::vector<FLOAT_TYPE*> tmp_d1s_ms_ites;
		std::vector<FLOAT_TYPE*> tmp_d2s_ms_ites;
		std::vector<FLOAT_TYPE*> tmp_d3s_ms_ites;

		std::vector<FLOAT_TYPE*> tmp_d1s_ms_ites2;
		std::vector<FLOAT_TYPE*> tmp_d2s_ms_ites2;
		std::vector<FLOAT_TYPE*> tmp_d3s_ms_ites2;
		std::vector<beacls::FloatVec > d1ss;
		std::vector<beacls::FloatVec > d2ss;
		std::vector<beacls::FloatVec > d3ss;

		beacls::IntegerVec tmp_cache_indexes;

		UpwindFirstENO3aHelper_Cache *cache;
	public:
		UpwindFirstENO3aHelper_impl(
			const HJI_Grid *hji_grid,
			const beacls::UVecType type = beacls::UVecType_Vector
		);
		~UpwindFirstENO3aHelper_impl();

		bool execute_dim0(
			std::vector<beacls::UVec > &dst_dL,
			std::vector<beacls::UVec > &dst_dR,
			std::vector<beacls::UVec > &dst_DD,
			const FLOAT_TYPE* src,
			const BoundaryCondition *boundaryCondition,
			const size_t dim,
			const bool approx4,
			const bool stripDD,
			const size_t loop_begin,
			const size_t slice_length,
			const size_t num_of_slices,
			beacls::CudaStream* cudaStream
		);
		bool execute_dim1(
			std::vector<beacls::UVec > &dst_dL,
			std::vector<beacls::UVec > &dst_dR,
			std::vector<beacls::UVec > &dst_DD,
			const FLOAT_TYPE* src,
			const BoundaryCondition *boundaryCondition,
			const size_t dim,
			const bool approx4,
			const bool stripDD,
			const size_t loop_begin,
			const size_t slice_length,
			const size_t num_of_slices,
			beacls::CudaStream* cudaStream
		);
		bool execute_dimLET2(
			std::vector<beacls::UVec > &dst_dL,
			std::vector<beacls::UVec > &dst_dR,
			std::vector<beacls::UVec > &dst_DD,
			const FLOAT_TYPE* src,
			const BoundaryCondition *boundaryCondition,
			const size_t dim,
			const bool approx4,
			const bool stripDD,
			const size_t loop_begin,
			const size_t slice_length,
			const size_t num_of_slices,
			const size_t num_of_strides,
			beacls::CudaStream* cudaStream
		);
		bool execute(
			std::vector<beacls::UVec > &dst_dL,
			std::vector<beacls::UVec > &dst_dR,
			std::vector<beacls::UVec > &dst_DD,
			const HJI_Grid *grid,
			const FLOAT_TYPE* src,
			const size_t dim,
			const bool approx4,
			const bool stripDD,
			const size_t loop_begin,
			const size_t slice_length,
			const size_t num_of_slices,
			const size_t num_of_strides,
			beacls::CudaStream* cudaStream
		);
		size_t get_max_DD_size(const size_t dim) const {
			return target_dimension_loop_sizes[dim] - 1;
		}
		bool operator==(const UpwindFirstENO3aHelper_impl& rhs) const;
		UpwindFirstENO3aHelper_impl* clone() const {
			return new UpwindFirstENO3aHelper_impl(*this);
		};
		beacls::UVecType get_type() const {
			return type;
		};
	private:


		void getCachePointers(
			std::vector<FLOAT_TYPE*> &d1s_ms,
			std::vector<FLOAT_TYPE*> &d2s_ms,
			std::vector<FLOAT_TYPE*> &d3s_ms,
			std::vector<beacls::UVec > &dst_DD,
			bool &d1_m0_writeToCache,
			bool &d2_m0_writeToCache,
			bool &d3_m0_writeToCache,
			const size_t slice_index,
			const size_t shifted_target_dimension_loop_index,
			const size_t first_dimension_loop_size,
			const size_t dst_loop_offset,
			const size_t num_of_slices,
			const size_t num_of_cache_lines,
			const bool saveDD,
			const bool stepHead);
		void createCaches(
			const size_t first_dimension_loop_size,
			const size_t num_of_cache_lines);
		/** @overload
		Disable operator=
		*/
		UpwindFirstENO3aHelper_impl& operator=(const UpwindFirstENO3aHelper_impl& rhs);
		/** @overload
		Disable copy constructor
		*/
		UpwindFirstENO3aHelper_impl(const UpwindFirstENO3aHelper_impl& rhs) :
			type(rhs.type),
			dxs(rhs.dxs),
			dx_squares(rhs.dx_squares),
			dxInvs(rhs.dxInvs),
			dxInv_2s(rhs.dxInv_2s),
			dxInv_3s(rhs.dxInv_3s),
			outer_dimensions_loop_sizes(rhs.outer_dimensions_loop_sizes),
			target_dimension_loop_sizes(rhs.target_dimension_loop_sizes),
			inner_dimensions_loop_sizes(rhs.inner_dimensions_loop_sizes),
			first_dimension_loop_sizes(rhs.first_dimension_loop_sizes),
			src_target_dimension_loop_sizes(rhs.src_target_dimension_loop_sizes),
			stencil(rhs.stencil),
			tmpBoundedSrc_ptrssss(rhs.tmpBoundedSrc_ptrssss),
			tmp_d1s_ms_ites(rhs.tmp_d1s_ms_ites),
			tmp_d2s_ms_ites(rhs.tmp_d2s_ms_ites),
			tmp_d3s_ms_ites(rhs.tmp_d3s_ms_ites),
			tmp_d1s_ms_ites2(rhs.tmp_d1s_ms_ites2),
			tmp_d2s_ms_ites2(rhs.tmp_d2s_ms_ites2),
			tmp_d3s_ms_ites2(rhs.tmp_d3s_ms_ites2),
			d1ss(rhs.d1ss),
			d2ss(rhs.d2ss),
			d3ss(rhs.d3ss),
			tmp_cache_indexes(rhs.tmp_cache_indexes),
			cache(new UpwindFirstENO3aHelper_Cache(*rhs.cache))
		{
			tmpBoundedSrc_uvec_vectors.resize(rhs.tmpBoundedSrc_uvec_vectors.size());
			tmpBoundedSrc_uvecs.resize(rhs.tmpBoundedSrc_uvecs.size());
		}
	};
};

#endif	/* __UpwindFirstENO3aHelper_impl_hpp__ */

