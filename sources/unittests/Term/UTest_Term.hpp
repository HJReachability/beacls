
#ifndef __UTest_Term_hpp__
#define __UTest_Term_hpp__
#include <cstring>
#include <vector>
#include <typedef.hpp>
#include <levelset/ExplicitIntegration/SchemeData.hpp>

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

typedef enum Dissipation_Class {
	Dissipation_Class_Invalid,
	Dissipation_Class_ArtificialDissipationGLF,
	Dissipation_Class_ArtificialDissipationLLF,
	Dissipation_Class_ArtificialDissipationLLLF,
}Dissipation_Class;

typedef enum Term_Class {
	Term_Class_Invalid,
	Term_Class_TermLaxFriedrichs,
	Term_Class_TermRestrictUpdate,
}Term_Class;

bool run_UTest_Term(
	std::string &message,
	const std::string& src_data_filename,
	const std::string& expects_filename,
	const beacls::FloatVec &maxs,
	const beacls::FloatVec &mins,
	SchemeData *schemeData,
	const SpatialDerivative_Class spatialDerivative_Class,
	const std::vector<BoundaryCondition_Class> &boundaryCondition_Classes,
	const Dissipation_Class& dissipation_class, 
	const Term_Class& term_Class,
	const beacls::UVecType type,
	const FLOAT_TYPE small_diff,
	const size_t line_length_of_chunk,
	const int num_of_threads,
	const int num_of_gpus,
	const bool enable_user_defined_dynamics_on_gpu
	);

#endif	/* __UTest_Term_hpp__ */

