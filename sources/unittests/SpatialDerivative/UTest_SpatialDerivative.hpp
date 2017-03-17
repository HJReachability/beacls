
#ifndef __UTest_SpatialDerivative_hpp__
#define __UTest_SpatialDerivative_hpp__
#include <cstring>
#include <vector>
#include <typedef.hpp>

typedef enum SpatialDerivative_Class {
	SpatialDerivative_Class_Invalid,
	SpatialDerivative_Class_UpwindFirstFirst,
	SpatialDerivative_Class_UpwindFirstENO2,
	SpatialDerivative_Class_UpwindFirstENO3,
	SpatialDerivative_Class_UpwindFirstWENO5,
}SpatialDerivative_Class;

typedef enum BoundaryCondition_Class {
	BoundaryCondition_Class_Invalid,
	BoundaryCondition_Class_AddGhostExtrapolate,
	BoundaryCondition_Class_AddGhostPeriodic,
}BoundaryCondition_Class;

bool run_UTest_SpatialDerivative(
	std::string &message,
	const std::string &src_filename,
	const std::vector<std::string>& expects_l_filenames,
	const std::vector<std::string>& expects_r_filenames,
	const beacls::FloatVec &maxs,
	const beacls::FloatVec &mins,
	const SpatialDerivative_Class spatialDerivative_Class,
	const std::vector<BoundaryCondition_Class> &boundaryCondition_Classes,
	const beacls::UVecType type,
	const FLOAT_TYPE small_diff,
	const size_t line_length_of_chunk,
	const int num_of_threads,
	const int num_of_gpus
	);

#endif	/* __UTest_SpatialDerivative_hpp__ */

