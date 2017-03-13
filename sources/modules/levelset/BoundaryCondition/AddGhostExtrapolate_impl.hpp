#ifndef __AddGhostExtrapolate_impl_hpp__
#define __AddGhostExtrapolate_impl_hpp__

#include <cstdint>
#include <vector>
#include <typedef.hpp>
#include <levelset/BoundaryCondition/BoundaryCondition.hpp>

class AddGhostExtrapolate_impl {
private:
	bool towardZero;
public:
	AddGhostExtrapolate_impl(
		const bool towardZero
		);
	~AddGhostExtrapolate_impl();
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
	bool operator==(const AddGhostExtrapolate_impl& rhs) const {
		if (towardZero != rhs.towardZero) return false;
		else return true;
	}
	bool valid()const {
		return true;
	};
	AddGhostExtrapolate_impl* clone() const {
		return new AddGhostExtrapolate_impl(*this);
	};
private:
	/** @overload
	Disable operator=
	*/
	AddGhostExtrapolate_impl& operator=(const AddGhostExtrapolate_impl& rhs);
	/** @overload
	copy constructor
	*/
	AddGhostExtrapolate_impl(const AddGhostExtrapolate_impl& rhs) : 
		towardZero(rhs.towardZero){
	}
};

#endif	/* __AddGhostExtrapolate_impl_hpp__ */

