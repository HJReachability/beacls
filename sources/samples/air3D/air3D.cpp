#define _USE_MATH_DEFINES
#include <levelset/levelset.hpp>
#include <cmath>
#include <numeric>
#include <functional>
#include <cfloat>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cstring>
#include "Air3DSchemeData.hpp"

typedef enum ApproximationAccuracy_Type{
	ApproximationAccuracy_Invalid,
	ApproximationAccuracy_low,
	ApproximationAccuracy_medium,
	ApproximationAccuracy_high,
	ApproximationAccuracy_veryHigh,

}ApproximationAccuracy_Type;

int main(int argc, char *argv[])
{
	bool debug_dump_file = false;
//	bool debug_dump_file = true;
	bool dump_file = false;
	if (argc >= 2) {
		dump_file = (atoi(argv[1]) == 0) ? false : true;
	}
	size_t line_length_of_chunk = 1;
	if (argc >= 3) {
		line_length_of_chunk = atoi(argv[2]);
	}
	bool useCuda = false;
	if (argc >= 4) {
		useCuda = (atoi(argv[3]) == 0) ? false : true;
	}
	int num_of_threads = 0;
	if (argc >= 5) {
		num_of_threads = atoi(argv[4]);
	}
	int num_of_gpus = 0;
	if (argc >= 6) {
		num_of_gpus = atoi(argv[5]);
	}
	beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
	if (argc >= 7) {
		switch (atoi(argv[6])) {
		default:
		case 0:
			delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
			break;
		case 1:
			delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
			break;
		case 2:
			delayedDerivMinMax = beacls::DelayedDerivMinMax_Adaptive;
			break;
		}
	}
	bool enable_user_defined_dynamics_on_gpu = true;
	if (argc >= 8) {
		enable_user_defined_dynamics_on_gpu = (atoi(argv[7]) == 0) ? false : true;
	}
	const FLOAT_TYPE tMax = (FLOAT_TYPE) 2.8;	//!< End time.
//	const FLOAT_TYPE tMax = 0.35;	//!< End time.
//	const FLOAT_TYPE tMax = 0.1;	//!< End time.
	const int plotSteps = 9;	//!< How many intermediate plots to produce?
//	const int plotSteps = 2;	//!< How many intermediate plots to produce?
	const FLOAT_TYPE t0 = 0.0;	//!< Start time.
	FLOAT_TYPE tPlot = (tMax - t0) / (plotSteps - 1);

	//! How close (relative) do we need to get to tMax to be considered finished?
	const FLOAT_TYPE small_ratio = 100.;	//!< 

	const FLOAT_TYPE eps = std::numeric_limits<FLOAT_TYPE>::epsilon();	//!< 

	FLOAT_TYPE small = small_ratio * eps;	//!< 

	//! Problem Parameters.
	//! Radius of target circle(positive).
	const FLOAT_TYPE targetRadius = 5;
	//! Speed of the evader(positive constant).
	const FLOAT_TYPE velocityA = 5;
	//! Speed of the pursuer(positive constant).
	const FLOAT_TYPE velocityB = 5;
	//! Maximum turn rate of the evader(positive).
	const FLOAT_TYPE inputA = 1;
	//! Maximum turn rate of the pursuer(positive).
	const FLOAT_TYPE inputB = 1;

	size_t num_of_dimensions = 3;

	beacls::FloatVec mins{ -6,-10,0 };
	beacls::FloatVec maxs{ +20,+10,(FLOAT_TYPE)(+2*M_PI) };

	size_t Nx = 51;
	beacls::IntegerVec Ns(num_of_dimensions);
	Ns = {Nx, (size_t)std::ceil(Nx * (maxs[1] - mins[1])/(maxs[0]-mins[0])), Nx-1};
	maxs[2] = (FLOAT_TYPE)(maxs[2] * (1 - 1. / Ns[2]));

	ShapeCylinder *shape = new ShapeCylinder(
		beacls::IntegerVec{2},
		beacls::FloatVec{0.,0.,0},
		targetRadius);

	AddGhostExtrapolate *addGhostExtrapolate = new AddGhostExtrapolate();
	AddGhostPeriodic *addGhostPeriodic = new AddGhostPeriodic();
	std::vector<BoundaryCondition*> boundaryConditions(3);
	boundaryConditions[0] = addGhostExtrapolate;
	boundaryConditions[1] = addGhostExtrapolate;
	boundaryConditions[2] = addGhostPeriodic;

	HJI_Grid *hJI_Grid = new HJI_Grid(
		num_of_dimensions);
	hJI_Grid->set_mins(mins);
	hJI_Grid->set_maxs(maxs);
	hJI_Grid->set_boundaryConditions(boundaryConditions);
	hJI_Grid->set_Ns(Ns);

	if (!hJI_Grid->processGrid()) {
		return -1;
	}

	const beacls::UVecType type = useCuda ? beacls::UVecType_Cuda : beacls::UVecType_Vector;

	ApproximationAccuracy_Type accuracy = ApproximationAccuracy_medium;
//	ApproximationAccuracy_Type accuracy = ApproximationAccuracy_high;
//	ApproximationAccuracy_Type accuracy = ApproximationAccuracy_veryHigh;

	// Choose spatial derivative approimation at appropriate level of accuracy.
	SpatialDerivative *spatialDerivative = NULL;
	switch (accuracy) {
	case ApproximationAccuracy_low:
		spatialDerivative = new UpwindFirstFirst(hJI_Grid, type);
		break;
	case ApproximationAccuracy_medium:
		spatialDerivative = new UpwindFirstENO2(hJI_Grid, type);
		break;
	case ApproximationAccuracy_high:
		spatialDerivative = new UpwindFirstENO3(hJI_Grid, type);
		break;
	case ApproximationAccuracy_veryHigh:
		spatialDerivative = new UpwindFirstWENO5(hJI_Grid, type);
		break;
	default:
		printf("Unkown accuracy level %d\n", accuracy);
		return -1;
	}

	ArtificialDissipationGLF *dissipation = new ArtificialDissipationGLF();

	std::vector<beacls::PostTimestep_Exec_Type*> postTimestep_Execs;
	Integrator *integrator;
	FLOAT_TYPE factor_cfl = (FLOAT_TYPE)0.75;
	FLOAT_TYPE max_step = (FLOAT_TYPE)8.0e16;
	bool single_step = false;
//	bool single_step = true;
	bool stats = false;
	beacls::TerminalEvent_Exec_Type *terminalEvent_Exec_Type = NULL;

	Air3DSchemeData *innerData = new Air3DSchemeData();
	innerData->set_spatialDerivative(spatialDerivative);
	innerData->set_dissipation(dissipation);

	innerData->set_grid(hJI_Grid);
//	innerData->set_hamiltonJacobiFunction(hamiltonJacobiFunction);
//	innerData->set_partialFunction(partialFunction);
	innerData->velocityA = velocityA;
	innerData->velocityB = velocityB;
	innerData->inputA = inputA;
	innerData->inputB = inputB;

	TermLaxFriedrichs *innerFunc = new TermLaxFriedrichs(innerData, type);
	TermRestrictUpdate *schemeFunc = new TermRestrictUpdate();
	Air3DSchemeData *schemeData = new Air3DSchemeData();

	schemeData->set_grid(hJI_Grid);
	schemeData->set_innerFunc(innerFunc);
	schemeData->set_innerData(innerData);
	schemeData->set_positive(false);

	// Choose integration approimation at appropriate level of accuracy.
	switch (accuracy) {
	case ApproximationAccuracy_low:
		integrator = new OdeCFL1(schemeFunc, factor_cfl, max_step, postTimestep_Execs, single_step, stats, terminalEvent_Exec_Type);
		break;
	case ApproximationAccuracy_medium:
		integrator = new OdeCFL2(schemeFunc, factor_cfl, max_step, postTimestep_Execs, single_step, stats, terminalEvent_Exec_Type);
		break;
	case ApproximationAccuracy_high:
		integrator = new OdeCFL3(schemeFunc, factor_cfl, max_step, postTimestep_Execs, single_step, stats, terminalEvent_Exec_Type);
		break;
	case ApproximationAccuracy_veryHigh:
		integrator = new OdeCFL3(schemeFunc, factor_cfl, max_step, postTimestep_Execs, single_step, stats, terminalEvent_Exec_Type);
		break;
	default:
		printf("Unkown accuracy level %d\n", accuracy);
		return -1;
	}


	beacls::FloatVec data;
	shape->execute(hJI_Grid, data);
	if (dump_file) {
		beacls::MatFStream* fs = beacls::openMatFStream(std::string("initial.mat"), beacls::MatOpenMode_Write);
		save_vector(data, std::string("y0"), Ns, false, fs);
		beacls::closeMatFStream(fs);
		//		beacls::FloatVec reload_data;
//		load_vector(std::string("initial.mat"), reload_data);

//		HJI_Grid* tmp_grid = new HJI_Grid();
//		tmp_grid->load_grid(std::string("air3D_0_g_v7_3.mat"));
//		tmp_grid->save_grid(std::string("air3D_0_g_v7_3_new.mat"),std::string("grid"));
//		delete tmp_grid;
	}

	// Loop until tMax (subject to a little roundoff).
	FLOAT_TYPE tNow = t0;
	beacls::FloatVec y;
	beacls::FloatVec y0;
	while ((tMax - tNow) > small * tMax) {
		y0 = data;
		beacls::FloatVec tspan(2);
		tspan[0] = tNow;
		tspan[1] = HjiMin(tMax, tNow + tPlot);
		if (debug_dump_file) {
			std::stringstream ss;
			ss << std::setprecision(5) << tNow << std::resetiosflags(std::ios_base::floatfield);
			std::string filename = "air3D_" + ss.str() + ".txt";
			dump_vector(filename.c_str(), data);
		}
		tNow = integrator->execute(
			y, tspan, y0, schemeData, 
			line_length_of_chunk, num_of_threads, num_of_gpus, 
			delayedDerivMinMax, enable_user_defined_dynamics_on_gpu);
		data = y;
		printf("tNow = %f\n", tNow);

	}
	if (dump_file) {
		dump_vector(std::string("all_loop.csv"),data);
		beacls::MatFStream* fs = beacls::openMatFStream(std::string("all_loop.mat"), beacls::MatOpenMode_Write);
		save_vector(data, std::string("y0"), Ns, false, fs);
		beacls::closeMatFStream(fs);
	}

	if (innerFunc) delete innerFunc;
	if (innerData) delete innerData;
	if (schemeFunc) delete schemeFunc;
	if (dissipation) delete dissipation;
	if (schemeData) delete schemeData;
	if (integrator) delete integrator;
	if (hJI_Grid) delete hJI_Grid;
	if (shape) delete shape;
	if (addGhostExtrapolate) delete addGhostExtrapolate;
	if (addGhostPeriodic) delete addGhostPeriodic;
	if (spatialDerivative) delete spatialDerivative;


	return 0;
}

