#ifndef __AddGhostPeriodic_hpp__
#define __AddGhostPeriodic_hpp__

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
#include <utility>
using namespace std::rel_ops;

namespace levelset {

	class AddGhostPeriodic_impl;

	// addGhostPeriodic: add ghost cells with periodic boundary conditions.
	//
	//   dataOut = addGhostPeriodic(dataIn, dim, width, ghostData)
	//
	// creates ghost cells to manage the boundary conditions for the array dataIn
	//
	// this m - file fills the ghost cells with periodic data
	//   data from the top of the array is put in the bottom ghost cells
	//   data from the bottom of the array is put in the top ghost cells
	//   in 2D for dim == 1
	// dataOut(1, 1) == dataIn(end + 1 - width, 1)
	// dataOut(end, 1) == dataIn(width, 1)
	//
	// notice that the indexing is shifted by the ghost cell width in output array
	//   so in 2D for dim == 1, the first data in the original array will be at
	//          dataOut(width + 1, 1) == dataIn(1, 1)
	//
	// parameters:
	//   dataIn	input data array
	//   dim		dimension in which to add ghost cells
	//   width	number of ghost cells to add on each side(default = 1)
	// ghostData	A structure(see below).
	//
	//   dataOut	Output data array.
	//
	// ghostData is a structure containing data specific to this type of
	//   ghost cell.For this function it is entirely ignored.
	//
	//
	// Copyright 2004 Ian M.Mitchell(mitchell@cs.ubc.ca).
	// This software is used, copied and distributed under the licensing
	//   agreement contained in the file LICENSE in the top directory of
	//   the distribution.
	//
	// created Ian Mitchell, 5 / 12 / 03
	// modified to allow choice of dimension, Ian Mitchell, 5 / 27 / 03
	// modified to allow ghostData input structure, Ian Mitchell, 1 / 13 / 04
	class AddGhostPeriodic : public BoundaryCondition {
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
			const size_t target_dimension_loop_size
		)const;
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
		bool operator==(const AddGhostPeriodic& rhs) const;
		bool operator==(const BoundaryCondition& rhs) const;
		bool valid()const;
		/**
		@brief Constructor
		*/
		PREFIX_VC_DLL
			AddGhostPeriodic(
			);
		~AddGhostPeriodic();
		AddGhostPeriodic* clone() const;
	private:
		AddGhostPeriodic_impl *pimpl;
		/** @overload
		Disable operator=
		*/
		AddGhostPeriodic& operator=(const AddGhostPeriodic& rhs);
		/** @overload
		Disable copy constructor
		*/
		AddGhostPeriodic(const AddGhostPeriodic& rhs);
	};
};
#endif	/* __AddGhostPeriodic_hpp__ */

