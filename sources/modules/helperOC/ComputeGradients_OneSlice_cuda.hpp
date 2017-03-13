#ifndef __ComputeGradients_OneSlice_cuda_hpp__
#define __ComputeGradients_OneSlice_cuda_hpp__

#include <vector>
#include <cuda_macro.hpp>
#include <typedef.hpp>

bool copyBackInfNan_cuda(
	beacls::UVec& deriv_c_uvec,
	beacls::UVec& deriv_l_uvec,
	beacls::UVec& deriv_r_uvec,
	const beacls::UVec& original_data_uvec
);
#endif /* __ComputeGradients_OneSlice_cuda_hpp__ */
