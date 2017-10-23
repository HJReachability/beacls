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
template<typename T, typename S, typename U> inline
void calc_D1toD3andDD_dimLET2(
	T* dst_dL0_ptr,
	T* dst_dL1_ptr,
	T* dst_dL2_ptr,
	T* dst_dL3_ptr,
	T* dst_dR0_ptr,
	T* dst_dR1_ptr,
	T* dst_dR2_ptr,
	T* dst_dR3_ptr,
	std::vector<T*>& dst_DD0_ptrs,
	std::vector<T*>& dst_DD1_ptrs,
	std::vector<T*>& dst_DD2_ptrs,
	std::vector<std::vector<T> >& d2ss,
	std::vector<std::vector<T> >& d3ss,
	const std::vector<std::vector<std::vector<const T*> > >& dst_ptrsss,
	const T dxInv,
	const T dxInv_2,
	const T dxInv_3,
	const T dx,
	const T x2_dx_square,
	const T dx_square,
	const size_t loop_length,
	const size_t first_dimension_loop_size,
	const size_t num_of_buffer_lines,
	const size_t slice_length,
	const size_t num_of_slices,
	const size_t num_of_strides,
	const size_t num_of_dLdR_in_slice,
	S approx4 = S(),
	U stripDD = U())
{
	for (size_t slice_index = 0; slice_index < num_of_slices; ++slice_index) {
		size_t dst_slice_offset = slice_index * loop_length * first_dimension_loop_size;
		for (size_t loop_index = 0; loop_index < loop_length; ++loop_index) {
			size_t dst_loop_offset = loop_index * first_dimension_loop_size + dst_slice_offset;
			const std::vector<const T*>& tmpBoundedSrc_ptrs = dst_ptrsss[slice_index][loop_index];
			//! Prologue
			{
				size_t pre_loop = 0;
				size_t buffer_index = pre_loop + 1;
				size_t m0_index = ((buffer_index + 3) % num_of_buffer_lines);
				const T* tmpBoundedSrc_ptrs0 = tmpBoundedSrc_ptrs[0 + pre_loop];
				const T* tmpBoundedSrc_ptrs1 = tmpBoundedSrc_ptrs[1 + pre_loop];
				const T* tmpBoundedSrc_ptrs2 = tmpBoundedSrc_ptrs[2 + pre_loop];
				std::vector<T>& d2ss_m0 = d2ss[m0_index];
				for (size_t first_dimension_loop_index = 0; first_dimension_loop_index < first_dimension_loop_size; ++first_dimension_loop_index) {
					T d0_m0 = tmpBoundedSrc_ptrs2[first_dimension_loop_index];
					T d0_m1 = tmpBoundedSrc_ptrs1[first_dimension_loop_index];
					T d0_m2 = tmpBoundedSrc_ptrs0[first_dimension_loop_index];
					T d1_m0 = dxInv * (d0_m0 - d0_m1);
					T d1_m1 = dxInv * (d0_m1 - d0_m2);
					T d2_m0 = dxInv_2 * (d1_m0 - d1_m1);
					d2ss_m0[first_dimension_loop_index] = d2_m0;
				}
			}
			for (size_t stride_index = 0; stride_index < num_of_strides; ++stride_index) {
				const size_t dst_stride_offset = stride_index * loop_length * num_of_slices * first_dimension_loop_size;
				const size_t buffer_index = stride_index + num_of_buffer_lines - 1;
				const size_t m0_index = ((buffer_index + 3) % num_of_buffer_lines);
				const size_t m1_index = ((buffer_index + 2) % num_of_buffer_lines);
				const size_t m2_index = ((buffer_index + 1) % num_of_buffer_lines);

				const T* tmpBoundedSrc_ptrs1 = tmpBoundedSrc_ptrs[0 + stride_index + num_of_buffer_lines - 3];
				const T* tmpBoundedSrc_ptrs2 = tmpBoundedSrc_ptrs[0 + stride_index + num_of_buffer_lines - 2];
				const T* tmpBoundedSrc_ptrs3 = tmpBoundedSrc_ptrs[0 + stride_index + num_of_buffer_lines - 1];
				const T* tmpBoundedSrc_ptrs4 = tmpBoundedSrc_ptrs[0 + stride_index + num_of_buffer_lines];
				std::vector<T>& d2ss_m0 = d2ss[m0_index];
				const std::vector<T>& d2ss_m1 = d2ss[m1_index];
				const std::vector<T>& d2ss_m2 = d2ss[m2_index];
				std::vector<T>& d3ss_m0 = d3ss[m0_index];
				const std::vector<T>& d3ss_m1 = d3ss[m1_index];
				const std::vector<T>& d3ss_m2 = d3ss[m2_index];
				if ((stride_index >= 3) && (stride_index <= 3 + num_of_dLdR_in_slice - 1)) {
					const size_t dst_dLdR_slice_offset = (stride_index - 3) * slice_length;

					const T* tmpBoundedSrc_ptrs0 = tmpBoundedSrc_ptrs[0 + stride_index + num_of_buffer_lines - 4];
					for (size_t first_dimension_loop_index = 0; first_dimension_loop_index < first_dimension_loop_size; ++first_dimension_loop_index) {
						size_t dst_dLdR_index = first_dimension_loop_index + dst_loop_offset + dst_dLdR_slice_offset;
						size_t dst_DD_index = first_dimension_loop_index + dst_loop_offset + dst_stride_offset;
						T d0_m0, d0_m1, d0_m2, d0_m3, d0_m4, d1_m0, d1_m1, d1_m2, d1_m3;
						calc_d0_dimLET2(d0_m4, tmpBoundedSrc_ptrs0, first_dimension_loop_index);
						calc_d0d1_dimLET2<FLOAT_TYPE>(d0_m4, d0_m3, d1_m3, tmpBoundedSrc_ptrs1, dxInv, first_dimension_loop_index);
						calc_d0d1_dimLET2<FLOAT_TYPE>(d0_m3, d0_m2, d1_m2, tmpBoundedSrc_ptrs2, dxInv, first_dimension_loop_index);
						calc_d0d1_dimLET2<FLOAT_TYPE>(d0_m2, d0_m1, d1_m1, tmpBoundedSrc_ptrs3, dxInv, first_dimension_loop_index);
						calc_d0d1_dimLET2<FLOAT_TYPE>(d0_m1, d0_m0, d1_m0, tmpBoundedSrc_ptrs4, dxInv, first_dimension_loop_index);
						T d2_m1 = d2ss_m1[first_dimension_loop_index];
						T d2_m2 = d2ss_m2[first_dimension_loop_index];
						T d2_m3 = d2ss_m0[first_dimension_loop_index];
						T d3_m1 = d3ss_m1[first_dimension_loop_index];
						T d3_m2 = d3ss_m2[first_dimension_loop_index];
						T d3_m3 = d3ss_m0[first_dimension_loop_index];
						T d2_m0 = dxInv_2 * (d1_m0 - d1_m1);
						T d3_m0 = dxInv_3 * (d2_m0 - d2_m1);
						calcApprox1to3<FLOAT_TYPE>(dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr,
							d1_m2, d1_m3, d2_m1, d2_m2, d2_m3, d3_m0, d3_m1, d3_m2, d3_m3, dx, x2_dx_square, dx_square, dst_dLdR_index);
						calcApprox4s<FLOAT_TYPE>(dst_dL3_ptr, dst_dR3_ptr, d1_m3, d1_m2, d2_m2, d2_m1, d3_m2, d3_m1, dx, dx_square, x2_dx_square, dst_dLdR_index, approx4);
						T* dst_DD0_ptr = dst_DD0_ptrs[0];
						T* dst_DD1_ptr = dst_DD1_ptrs[0];
						T* dst_DD2_ptr = dst_DD2_ptrs[0];
						storeDD(dst_DD0_ptr, d1_m0, d1_m2, dst_DD_index, stripDD);
						storeDD(dst_DD1_ptr, d2_m0, d2_m1, dst_DD_index, stripDD);
						storeDD(dst_DD2_ptr, d3_m0, d3_m0, dst_DD_index, stripDD);
						d3ss_m0[first_dimension_loop_index] = d3_m0;
						d2ss_m0[first_dimension_loop_index] = d2_m0;
					}
				}
				else {
					for (size_t first_dimension_loop_index = 0; first_dimension_loop_index < first_dimension_loop_size; ++first_dimension_loop_index) {
						size_t dst_DD_index = first_dimension_loop_index + dst_loop_offset + dst_stride_offset;
						T d0_m0, d0_m1, d0_m2, d0_m3, d1_m0, d1_m1, d1_m2;
						calc_d0_dimLET2<FLOAT_TYPE>(d0_m3, tmpBoundedSrc_ptrs1, first_dimension_loop_index);
						calc_d0d1_dimLET2<FLOAT_TYPE>(d0_m3, d0_m2, d1_m2, tmpBoundedSrc_ptrs2, dxInv, first_dimension_loop_index);
						calc_d0d1_dimLET2<FLOAT_TYPE>(d0_m2, d0_m1, d1_m1, tmpBoundedSrc_ptrs3, dxInv, first_dimension_loop_index);
						calc_d0d1_dimLET2<FLOAT_TYPE>(d0_m1, d0_m0, d1_m0, tmpBoundedSrc_ptrs4, dxInv, first_dimension_loop_index);
						T d2_m1 = d2ss_m1[first_dimension_loop_index];
						T d2_m0 = dxInv_2 * (d1_m0 - d1_m1);
						T d3_m0 = dxInv_3 * (d2_m0 - d2_m1);
						T* dst_DD0_ptr = dst_DD0_ptrs[0];
						T* dst_DD1_ptr = dst_DD1_ptrs[0];
						T* dst_DD2_ptr = dst_DD2_ptrs[0];
						storeDD(dst_DD0_ptr, d1_m0, d1_m2, dst_DD_index, stripDD);
						storeDD(dst_DD1_ptr, d2_m0, d2_m1, dst_DD_index, stripDD);
						storeDD(dst_DD2_ptr, d3_m0, d3_m0, dst_DD_index, stripDD);
						d3ss_m0[first_dimension_loop_index] = d3_m0;
						d2ss_m0[first_dimension_loop_index] = d2_m0;
					}
				}
			}
		}
	}
}

template<typename T, typename S> inline
void calc_D1toD3_dimLET2(
	T* dst_dL0_ptr,
	T* dst_dL1_ptr,
	T* dst_dL2_ptr,
	T* dst_dL3_ptr,
	T* dst_dR0_ptr,
	T* dst_dR1_ptr,
	T* dst_dR2_ptr,
	T* dst_dR3_ptr,
	const std::vector<std::vector<std::vector<const T*> > >& tmpBoundedSrc_ptrsss,
	const T dxInv,
	const T dxInv_2,
	const T dxInv_3,
	const T dx,
	const T x2_dx_square,
	const T dx_square,
	const size_t loop_length,
	const size_t first_dimension_loop_size,
	const size_t slice_length,
	const size_t num_of_slices,
	S approx4 = S()) {
	for (size_t slice_index = 0; slice_index < num_of_slices; ++slice_index) {
		const size_t slice_offset = slice_index * slice_length;
		for (size_t loop_index = 0; loop_index < loop_length; ++loop_index) {
			size_t dst_dLdR_offset = loop_index * first_dimension_loop_size + slice_offset;
			const std::vector<const T*>& tmpBoundedSrc_ptrs = tmpBoundedSrc_ptrsss[slice_index][loop_index];
			{
				size_t plane = 0;
				const T* tmpBoundedSrc_ptrs0 = tmpBoundedSrc_ptrs[0 + plane];
				const T* tmpBoundedSrc_ptrs1 = tmpBoundedSrc_ptrs[1 + plane];
				const T* tmpBoundedSrc_ptrs2 = tmpBoundedSrc_ptrs[2 + plane];
				const T* tmpBoundedSrc_ptrs3 = tmpBoundedSrc_ptrs[3 + plane];
				const T* tmpBoundedSrc_ptrs4 = tmpBoundedSrc_ptrs[4 + plane];
				const T* tmpBoundedSrc_ptrs5 = tmpBoundedSrc_ptrs[5 + plane];
				const T* tmpBoundedSrc_ptrs6 = tmpBoundedSrc_ptrs[6 + plane];
				for (size_t first_dimension_loop_index = 0; first_dimension_loop_index < first_dimension_loop_size; ++first_dimension_loop_index) {
					T d0_0, d0_1, d0_2, d0_3, d0_4, d0_5, d0_6, d1_0, d1_1, d1_2, d1_3, d1_4, d1_5, d2_0, d2_1, d2_2, d2_3, d2_4, d3_0, d3_1, d3_2, d3_3;
					calcD1toD3_dimLET2<FLOAT_TYPE>(
						d0_0, d0_1, d0_2, d0_3, d0_4, d0_5, d0_6, d1_0, d1_1, d1_2, d1_3, d1_4, d1_5, d2_0, d2_1, d2_2, d2_3, d2_4, d3_0, d3_1, d3_2, d3_3,
						tmpBoundedSrc_ptrs0, tmpBoundedSrc_ptrs1, tmpBoundedSrc_ptrs2, tmpBoundedSrc_ptrs3, tmpBoundedSrc_ptrs4, tmpBoundedSrc_ptrs5, tmpBoundedSrc_ptrs6,
						dxInv, dxInv_2, dxInv_3, first_dimension_loop_index
						);
					size_t dst_index = first_dimension_loop_index + dst_dLdR_offset;
					calcApprox1to3<FLOAT_TYPE>(dst_dL0_ptr, dst_dL1_ptr, dst_dL2_ptr, dst_dR0_ptr, dst_dR1_ptr, dst_dR2_ptr, d1_3, d1_2, d2_3, d2_2, d2_1, d3_3, d3_2, d3_1, d3_0, dx, x2_dx_square, dx_square, dst_index);
					calcApprox4s<FLOAT_TYPE>(dst_dL3_ptr, dst_dR3_ptr, d1_2, d1_3, d2_2, d2_3, d3_1, d3_2, dx, dx_square, x2_dx_square, dst_index, approx4);
				}
			}
		}
	}
}


#endif	/* __UpwindFirstENO3aHelper_impl_hpp__ */

