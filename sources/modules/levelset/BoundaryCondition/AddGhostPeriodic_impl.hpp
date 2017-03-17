
#ifndef __AddGhostPeriodic_impl_hpp__
#define __AddGhostPeriodic_impl_hpp__

#include <cstdint>
#include <vector>
#include <typedef.hpp>
#include <levelset/BoundaryCondition/BoundaryCondition.hpp>
namespace levelset {
	class AddGhostPeriodic_impl {
	private:
	public:
		AddGhostPeriodic_impl(
		);
		~AddGhostPeriodic_impl();
		bool execute(
			const FLOAT_TYPE* src,
			beacls::UVec& dst_buffer,
			const FLOAT_TYPE*& dst_ptr,
			const size_t width,
			const size_t outer_dimensions_loop_index,
			const size_t target_dimension_loop_size,
			const size_t target_dimension_loop_index,
			const size_t inner_dimensions_loop_size,
			const size_t inner_dimensions_loop_index,
			const size_t first_dimension_loop_size
		) const;
		bool execute(
			const FLOAT_TYPE* src,
			beacls::UVec& dst_buffer,
			beacls::UVec& tmp_buffer,
			std::vector<std::vector<std::vector<const FLOAT_TYPE*> > >& dst_ptrsss,
			const size_t width,
			const size_t dim,
			const size_t target_dimension_loop_size,
			const size_t inner_dimensions_loop_size,
			const size_t first_dimension_loop_size,
			const size_t loop_begin,
			const size_t prologue_loop_dst_offset,
			const size_t num_of_slices,
			const size_t loop_length,
			const size_t num_of_strides,
			const size_t stencil

		) const;
		bool execute(
			const FLOAT_TYPE* src,
			beacls::UVec& dst,
			const size_t width,
			const size_t outer_dimensions_loop_index,
			const size_t target_dimension_loop_size) const;
		bool execute(
			const FLOAT_TYPE* src,
			beacls::UVec& dst,
			beacls::UVec& tmp_buffer,
			const size_t width,
			const size_t outer_dimensions_loop_length,
			const size_t target_dimension_loop_size,
			const size_t first_dimension_loop_size,
			const size_t loop_begin_base,
			const size_t num_of_slices
		)const;
		bool operator==(const AddGhostPeriodic_impl&) const {
			return true;
		}
		bool valid()const {
			return true;
		};
		AddGhostPeriodic_impl* clone() const {
			return new AddGhostPeriodic_impl(*this);
		};
	private:
		/** @overload
		Disable operator=
		*/
		AddGhostPeriodic_impl& operator=(const AddGhostPeriodic_impl& rhs);
		/** @overload
		copy constructor
		*/
		AddGhostPeriodic_impl(const AddGhostPeriodic_impl&)
		{
		}
	};
};
#endif	/* __AddGhostPeriodic_impl_hpp__ */

