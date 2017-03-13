
#ifndef __UTest_HamFunc_hpp__
#define __UTest_HamFunc_hpp__
#include <cstring>
#include <vector>
#include <typedef.hpp>
#include <levelset/ExplicitIntegration/SchemeData.hpp>

bool run_UTest_HamFunc(
	std::string &message,
	const std::string& src_data_filename,
	const std::vector<std::string>& src_deriv_c_filenames,
	const std::string& expects_filename,
	const beacls::FloatVec &maxs,
	const beacls::FloatVec &mins,
	SchemeData *schemeData,
	const beacls::UVecType type,
	const FLOAT_TYPE small_diff,
	const size_t line_length_of_chunk,
	const int num_of_threads,
	const int num_of_gpus,
	const bool enable_user_defined_dynamics_on_gpu
	);

#endif	/* __UTest_HamFunc_hpp__ */

