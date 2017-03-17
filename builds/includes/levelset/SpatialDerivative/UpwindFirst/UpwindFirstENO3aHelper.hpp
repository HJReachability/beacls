#ifndef __UpwindFirstENO3aHelper_hpp__
#define __UpwindFirstENO3aHelper_hpp__

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
#include <utility>
using namespace std::rel_ops;

#include <Core/UVec.hpp>
namespace levelset {
	class HJI_Grid;

	class UpwindFirstENO3aHelper_impl;

	class UpwindFirstENO3aHelper {
	public:
		PREFIX_VC_DLL
			UpwindFirstENO3aHelper(
				const HJI_Grid *hji_grid,
				const beacls::UVecType type = beacls::UVecType_Vector
			);
		~UpwindFirstENO3aHelper();
		/**
		@brief Computes spatial derivative
		@param	[out]	dst_dL	Cell vector containing the 3 or 4 left approximations
									of the first derivative (each the same size as data).
		@param	[out]	dst_dR	Cell vector containing the 3 or 4 right approximations
									of the first derivative (each the same size as data).
		@param	[out]	dst_DD	Cell vector containing the divided difference tables
									(optional).
		@param	[in]	grid	Grid structure.
		@param	[in]	src	Source data array.
		@param	[in]	dim	Which dimension to compute derivative on.
		@param	[in]	approx4   Generate two copies of middle approximation using
									both left/right and right/left traversal of divided
									difference tree.  The extra copy is placed in the
									fourth element of derivL and derivR, and is equivalent
									to the version in the second element of those cell vectors.
		@param	[in]	stripDD	Strip the divided difference tables down to their
									appropriate size, otherwise they will contain entries
									(at the D1 and D2 levels) that correspond entirely
									to ghost cells.
		@param	[in]	loop_begin	Index of loop to begin this itteration.
		@param	[in]	slice_length	slice_length of loop to begin this itteration.
		@param	[in]	num_of_slices	number of strides to begin this itteration.
		@param	[in]	num_of_strides	number of DD strides to begin this itteration.
		@retval	true	Operation succeeded.
		@retval	false	Operation failed.
		*/
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
		size_t get_max_DD_size(const size_t dim) const;
		bool operator==(const UpwindFirstENO3aHelper& rhs) const;
		UpwindFirstENO3aHelper* clone() const;

	private:
		UpwindFirstENO3aHelper_impl *pimpl;

		/** @overload
		Disable operator=
		*/
		UpwindFirstENO3aHelper& operator=(const UpwindFirstENO3aHelper& rhs);
		/** @overload
		Disable copy constructor
		*/
		UpwindFirstENO3aHelper(const UpwindFirstENO3aHelper& rhs);
	};
};
#endif	/* __UpwindFirstENO3aHelper_hpp__ */

