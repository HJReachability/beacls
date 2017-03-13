#ifndef __BoundaryCondition_hpp__
#define __BoundaryCondition_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#include <typedef.hpp>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <utility>
using namespace std::rel_ops;
#include <Core/UVec.hpp>

class BoundaryCondition {
public:
	typedef enum BoundaryCondition_Type {
		BoundaryCondition_Invalid,
		BoundaryCondition_AddGhostPeriodic,
		BoundaryCondition_AddGhostExtrapolate,
		BoundaryCondition_AddGhostExtrapolate2,
		BoundaryCondition_AddGhostDirichlet,
		BoundaryCondition_AddGhostNeumann,

	}BoundaryCondition_Type;
	PREFIX_VC_DLL
		virtual bool execute(
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
		)const = 0;
	PREFIX_VC_DLL
		virtual bool execute(
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
		) const = 0;
	PREFIX_VC_DLL
		virtual bool execute(
			const FLOAT_TYPE* src,
			beacls::UVec& dst,
			const size_t width,
			const size_t outer_dimensions_loop_index,
			const size_t target_dimension_loop_size)const = 0;
	PREFIX_VC_DLL
		virtual bool execute(
			const FLOAT_TYPE* src,
			beacls::UVec& dst,
			beacls::UVec& tmp_buffer,
			const size_t width,
			const size_t outer_dimensions_loop_length,
			const size_t target_dimension_loop_size,
			const size_t first_dimension_loop_size,
			const size_t loop_begin_base,
			const size_t num_of_slices
			)const = 0;
	PREFIX_VC_DLL
		virtual bool operator==(const BoundaryCondition& rhs) const = 0;
	PREFIX_VC_DLL
		virtual bool valid()const = 0;
	PREFIX_VC_DLL
		virtual BoundaryCondition* clone() const = 0;
	virtual ~BoundaryCondition() = 0;
private:
};
inline
BoundaryCondition::~BoundaryCondition() {}
#endif	/* __BoundaryCondition_hpp__ */

