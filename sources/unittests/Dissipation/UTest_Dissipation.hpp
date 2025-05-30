
#ifndef __UTest_Dissipation_hpp__
#define __UTest_Dissipation_hpp__
#include <cstring>
#include <vector>
#include <typedef.hpp>
#include <levelset/ExplicitIntegration/SchemeData.hpp>

typedef enum Dissipation_Class {
	Dissipation_Class_Invalid,
	Dissipation_Class_ArtificialDissipationGLF,
	Dissipation_Class_ArtificialDissipationLLF,
	Dissipation_Class_ArtificialDissipationLLLF,
}Dissipation_Class;

bool run_UTest_Dissipation(
	std::string &message,
	const std::string& src_data_filename,
	const std::vector<std::string>& src_deriv_l_filenames,
	const std::vector<std::string>& src_deriv_r_filenames,
	const std::string& expects_filename,
	const beacls::FloatVec &maxs,
	const beacls::FloatVec &mins,
	levelset::SchemeData *schemeData,
	const Dissipation_Class& dissipation_class,
	const beacls::UVecType type,
	const FLOAT_TYPE small_diff,
	const size_t line_length_of_chunk,
	const int num_of_threads,
	const int num_of_gpus,
	const bool enable_user_defined_dynamics_on_gpu
	);

#endif	/* __UTest_Dissipation_hpp__ */

