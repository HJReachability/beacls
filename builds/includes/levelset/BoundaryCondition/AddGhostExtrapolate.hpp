#ifndef __AddGhostExtrapolate_hpp__
#define __AddGhostExtrapolate_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#include <typedef.hpp>
#include <levelset/BoundaryCondition/BoundaryCondition.hpp>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <utility>
using namespace std::rel_ops;
namespace levelset {

	class AddGhostExtrapolate_impl;

	// addGhostExtrapolate: add ghost cells, values extrapolated from bdry nodes.
	//
	//   dataOut = addGhostExtrapolate(dataIn, dim, width, ghostData)
	//
	// Creates ghost cells to manage the boundary conditions for the array dataIn.
	//
	// This m - file fills the ghost cells with data linearly extrapolated
	//   from the grid edge, where the sign of the slope is chosen to make sure the
	//   extrapolation goes away from or towards the zero level set.
	//
	// For implicit surfaces, the extrapolation will typically be away from zero
	// (the extrapolation should not imply the presence of an implicit surface
	//    beyond the array bounds).
	//
	// Notice that the indexing is shifted by the ghost cell width in output array.
	//   So in 2D with dim == 1, the first data in the original array will be at
	//          dataOut(width + 1, 1) == dataIn(1, 1)
	//
	// parameters:
	//   dataIn	Input data array.
	//   dim		Dimension in which to add ghost cells.
	//   width	Number of ghost cells to add on each side(default = 1).
	//   ghostData	A structure(see below).
	//
	//   dataOut	Output data array.
	//
	// ghostData is a structure containing data specific to this type of
	//   ghost cell.For this function it contains the field(s)
	//
	//.towardZero Boolean indicating whether sign of extrapolation should
	//                 be towards or away from the zero level set(default = 0).
	//
	//
	// Copyright 2004 Ian M.Mitchell(mitchell@cs.ubc.ca).
	// This software is used, copied and distributed under the licensing
	//   agreement contained in the file LICENSE in the top directory of
	//   the distribution.
	//
	// Ian Mitchell, 5 / 12 / 03
	// modified to allow choice of dimension, Ian Mitchell, 5 / 27 / 03
	// modified to allow ghostData input structure & renamed, Ian Mitchell, 1 / 13 / 04
	class AddGhostExtrapolate : public BoundaryCondition {
	public:
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
		)const;
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
			const size_t target_dimension_loop_size)const;
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
		bool operator==(const AddGhostExtrapolate& rhs) const;
		bool operator==(const BoundaryCondition& rhs) const;
		bool valid()const;

		/**
		@brief Constructor
		@param	[in]		towardZero	Boolean indicating whether sign of extrapolation should
									be towards or away from the zero level set (default=false)
		*/
		PREFIX_VC_DLL
			AddGhostExtrapolate(
				const bool towardZero = false
			);
		~AddGhostExtrapolate();
		AddGhostExtrapolate* clone() const;
	private:
		AddGhostExtrapolate_impl *pimpl;
		/** @overload
		Disable operator=
		*/
		AddGhostExtrapolate& operator=(const AddGhostExtrapolate& rhs);
		/** @overload
		Disable copy constructor
		*/
		AddGhostExtrapolate(const AddGhostExtrapolate& rhs);
	};
};
#endif	/* __AddGhostExtrapolate_hpp__ */

