#define _USE_MATH_DEFINES
#include <levelset/levelset.hpp>
#include <helperOC/helperOC.hpp>
#include <cmath>
#include <numeric>
#include <functional>
#include <cfloat>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cstring>
#include "Plane4DSchemeData.hpp"

typedef enum ApproximationAccuracy_Type {
	ApproximationAccuracy_Invalid,
	ApproximationAccuracy_low,
	ApproximationAccuracy_medium,
	ApproximationAccuracy_high,
	ApproximationAccuracy_veryHigh,

}ApproximationAccuracy_Type;

int main(int argc, char *argv[])
{
	bool debug_dump_file = false;
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
	levelset::DelayedDerivMinMax_Type delayedDerivMinMax = levelset::DelayedDerivMinMax_Disable;
	if (argc >= 7) {
		switch (atoi(argv[6])) {
		default:
		case 0:
			delayedDerivMinMax = levelset::DelayedDerivMinMax_Disable;
			break;
		case 1:
			delayedDerivMinMax = levelset::DelayedDerivMinMax_Always;
			break;
		case 2:
			delayedDerivMinMax = levelset::DelayedDerivMinMax_Adaptive;
			break;
		}
	}
	bool enable_user_defined_dynamics_on_gpu = true;
	if (argc >= 8) {
		enable_user_defined_dynamics_on_gpu = (atoi(argv[7]) == 0) ? false : true;
	}


	const FLOAT_TYPE tMax = 1;	//!< End time.
	const FLOAT_TYPE dt = (FLOAT_TYPE)0.1;
	const FLOAT_TYPE t0 = 0.0;	//!< Start time.
	const size_t plotSteps = (size_t)std::ceil((tMax-t0)/dt)+1;	//!< How many intermediate plots to produce?

#if 0
	const bool singleStep = false;	//!< Prot at each timestep (overrides tPlot).
									//! Preriod at which intermediate plots should be produced.
#endif
	FLOAT_TYPE tPlot = (tMax - t0) / (plotSteps - 1);

	//! How close (relative) do we need to get to tMax to be considered finished?
	const FLOAT_TYPE small_ratio = 100.;	//!< 

	const FLOAT_TYPE eps = std::numeric_limits<FLOAT_TYPE>::epsilon();	//!< 

	FLOAT_TYPE small = small_ratio * eps;	//!< 

	//! Problem Parameters.

	size_t num_of_dimensions = 4;

	beacls::FloatVec mins{ (FLOAT_TYPE)-5, (FLOAT_TYPE)-10, (FLOAT_TYPE)0,      (FLOAT_TYPE)6 };
	beacls::FloatVec maxs{ (FLOAT_TYPE)+15,(FLOAT_TYPE)+10,(FLOAT_TYPE)(+2*M_PI),(FLOAT_TYPE)12 };

	beacls::FloatVec center{ (FLOAT_TYPE)0, (FLOAT_TYPE)0,  (FLOAT_TYPE)M_PI, (FLOAT_TYPE)9 };
	beacls::FloatVec widths{ (FLOAT_TYPE)2, (FLOAT_TYPE)2,  (FLOAT_TYPE)M_PI/2, (FLOAT_TYPE)1 };


	size_t Nx = 31;
	beacls::IntegerVec Ns(num_of_dimensions);
	Ns.assign(num_of_dimensions, Nx);
	maxs[2] = (FLOAT_TYPE)(maxs[2] * (1 - 1. / Ns[2]));

	FLOAT_TYPE wMax = 1;
	beacls::FloatVec aranges{ (FLOAT_TYPE)0.5, 1 };

	levelset::ShapeRectangleByCenter *shape = new levelset::ShapeRectangleByCenter(center, widths);

	levelset::AddGhostExtrapolate *addGhostExtrapolate = new levelset::AddGhostExtrapolate();
	levelset::AddGhostPeriodic *addGhostPeriodic = new levelset::AddGhostPeriodic();
	std::vector<levelset::BoundaryCondition*> boundaryConditions(num_of_dimensions);
	boundaryConditions[0] = addGhostExtrapolate;
	boundaryConditions[1] = addGhostExtrapolate;
	boundaryConditions[2] = addGhostPeriodic;
	boundaryConditions[3] = addGhostExtrapolate;

	levelset::HJI_Grid *hJI_Grid = new levelset::HJI_Grid(
		num_of_dimensions);
	hJI_Grid->set_mins(mins);
	hJI_Grid->set_maxs(maxs);
	hJI_Grid->set_boundaryConditions(boundaryConditions);
	hJI_Grid->set_Ns(Ns);

	if (!hJI_Grid->processGrid()) {
		return -1;
	}

	ApproximationAccuracy_Type accuracy = ApproximationAccuracy_veryHigh;
//	ApproximationAccuracy_Type accuracy = ApproximationAccuracy_high;

	const beacls::UVecType type = useCuda ? beacls::UVecType_Cuda : beacls::UVecType_Vector;

	// Choose spatial derivative approimation at appropriate level of accuracy.
	levelset::SpatialDerivative *spatialDerivative = NULL;
	switch (accuracy) {
	case ApproximationAccuracy_low:
		spatialDerivative = new levelset::UpwindFirstFirst(hJI_Grid, type);
		break;
	case ApproximationAccuracy_medium:
		spatialDerivative = new levelset::UpwindFirstENO2(hJI_Grid, type);
		break;
	case ApproximationAccuracy_high:
		spatialDerivative = new levelset::UpwindFirstENO3(hJI_Grid, type);
		break;
	case ApproximationAccuracy_veryHigh:
		spatialDerivative = new levelset::UpwindFirstWENO5(hJI_Grid, type);
		break;
	default:
		printf("Unkown accuracy level %d\n", accuracy);
		return -1;
	}


	levelset::ArtificialDissipationGLF *dissipation = new levelset::ArtificialDissipationGLF();

	std::vector<levelset::PostTimestep_Exec_Type*> postTimestep_Execs;
	levelset::Integrator *integrator;
	FLOAT_TYPE factor_cfl = (FLOAT_TYPE)0.8;
	FLOAT_TYPE max_step = (FLOAT_TYPE)8.0e16;
	bool single_step = true;
	bool stats = false;
	levelset::TerminalEvent_Exec_Type *terminalEvent_Exec_Type = NULL;

	Plane4DSchemeData *schemeData = new Plane4DSchemeData();
	schemeData->set_spatialDerivative(spatialDerivative);
	schemeData->set_dissipation(dissipation);

	schemeData->set_grid(hJI_Grid);
	//	innerData->set_hamiltonJacobiFunction(hamiltonJacobiFunction);
	//	innerData->set_partialFunction(partialFunction);
	schemeData->wMax = wMax;
	schemeData->aranges = aranges;

	levelset::TermLaxFriedrichs *schemeFunc = new levelset::TermLaxFriedrichs(schemeData, type);


	// Choose integration approimation at appropriate level of accuracy.
	switch (accuracy) {
	case ApproximationAccuracy_low:
		integrator = new levelset::OdeCFL1(schemeFunc, factor_cfl, max_step, postTimestep_Execs, single_step, stats, terminalEvent_Exec_Type);
		break;
	case ApproximationAccuracy_medium:
		integrator = new levelset::OdeCFL2(schemeFunc, factor_cfl, max_step, postTimestep_Execs, single_step, stats, terminalEvent_Exec_Type);
		break;
	case ApproximationAccuracy_high:
		integrator = new levelset::OdeCFL3(schemeFunc, factor_cfl, max_step, postTimestep_Execs, single_step, stats, terminalEvent_Exec_Type);
		break;
	case ApproximationAccuracy_veryHigh:
		integrator = new levelset::OdeCFL3(schemeFunc, factor_cfl, max_step, postTimestep_Execs, single_step, stats, terminalEvent_Exec_Type);
		break;
	default:
		printf("Unkown accuracy level %d\n", accuracy);
		return -1;
	}

	beacls::FloatVec data;
	shape->execute(hJI_Grid, data);

	beacls::FloatVec tau = generateArithmeticSequence<FLOAT_TYPE>(0., dt, tMax);

	bool minWith = true;

	// Loop until tMax (subject to a little roundoff).
	std::vector<beacls::FloatVec > datas;
	beacls::FloatVec y;
	beacls::FloatVec y0;
	for (beacls::FloatVec::const_iterator ite = tau.cbegin()+1;
		ite != tau.cend(); ++ite) {
		FLOAT_TYPE tNow = *(ite - 1);
		FLOAT_TYPE tau_i = *ite;
		while ((tau_i - tNow) > small *tau_i)
		{
			y0 = data;
			beacls::FloatVec tspan(2);
			tspan[0] = tNow;
			tspan[1] = HjiMin(tau_i, tNow + tPlot);
			beacls::FloatVec yLast;
			if (minWith) {
				yLast = data;
			}
			if (debug_dump_file) {
				std::stringstream ss;
				ss << std::setprecision(5) << tNow << std::resetiosflags(std::ios_base::floatfield);
				std::string filename = "air3D_" + ss.str() + ".txt";
				dump_vector(filename,data);
			}
			tNow = integrator->execute(
				y, tspan, y0, schemeData,
				line_length_of_chunk, num_of_threads, num_of_gpus,
				delayedDerivMinMax, enable_user_defined_dynamics_on_gpu);
			data = y;

			// Min with zero
			if (minWith) {
				std::transform(data.cbegin(),data.cend(),yLast.cbegin(),data.begin(), std::ptr_fun<const FLOAT_TYPE&, const FLOAT_TYPE&>(std::min<FLOAT_TYPE>));
			}
			printf("tNow = %f\n", tNow);

		}
		datas.push_back(data);
	}
	if (dump_file) {
		dump_vector(std::string("all_loop.csv"),data);

		beacls::MatFStream* fs = beacls::openMatFStream(std::string("all_loop.mat"), beacls::MatOpenMode_Write);
		save_vector(data, std::string("y0"), Ns, false, fs);
		beacls::closeMatFStream(fs);
	}


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

