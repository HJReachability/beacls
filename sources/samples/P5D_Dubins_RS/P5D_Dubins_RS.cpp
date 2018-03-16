#define _USE_MATH_DEFINES
#include <levelset/levelset.hpp>
#include <helperOC/helperOC.hpp>
#include <helperOC/Legacy/ExtractCostates.hpp>
#include <helperOC/DynSys/P5D_Dubins/P5D_Dubins.hpp>
#include <helperOC/DynSys/DynSys/DynSysSchemeData.hpp>
#include <cmath>
#include <numeric>
#include <functional>
#include <cfloat>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cstring>

int main(int argc, char *argv[])
{
	bool dump_file = false;
	if (argc >= 2) {
		dump_file = (atoi(argv[1]) == 0) ? false : true;
	}
	size_t line_length_of_chunk = 1;
	if (argc >= 3) {
		line_length_of_chunk = atoi(argv[2]);
	}
	size_t model_size = 2;
	if (argc >= 4) {
		model_size = atoi(argv[3]);
	}
	// Time stamps
	FLOAT_TYPE tMax = 15.;
	if (argc >= 5) {
		tMax = static_cast<FLOAT_TYPE>(atof(argv[4]));
	}
	FLOAT_TYPE dt = (FLOAT_TYPE).05;
	if (argc >= 6) {
		dt = static_cast<FLOAT_TYPE>(atof(argv[5]));
	}
	bool useTempFile = false;
	if (argc >= 7) {
		useTempFile = (atoi(argv[6]) == 0) ? false : true;
	}
	bool keepLast = true;
	if (argc >= 8) {
		keepLast = (atoi(argv[7]) == 0) ? true : false;
	}
	bool calculateTTRduringSolving = false;
	if (argc >= 9) {
		calculateTTRduringSolving = (atoi(argv[8]) == 0) ? false : true;
	}
	bool useCuda = false;
	if (argc >= 10) {
		useCuda = (atoi(argv[9]) == 0) ? false : true;
	}
	int num_of_threads = 0;
	if (argc >= 11) {
		num_of_threads = atoi(argv[10]);
	}
	int num_of_gpus = 0;
	if (argc >= 12) {
		num_of_gpus = atoi(argv[11]);
	}
	levelset::DelayedDerivMinMax_Type delayedDerivMinMax = levelset::DelayedDerivMinMax_Disable;
	if (argc >= 13) {
		switch (atoi(argv[12])) {
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
	if (argc >= 14) {
		enable_user_defined_dynamics_on_gpu = (atoi(argv[13]) == 0) ? false : true;
	}
	// Create grid
	beacls::IntegerVec	Ns;
	switch (model_size) {
	case 0:
		Ns = beacls::IntegerVec{ 51, 51, 31, 31, 31};
		break;
	case 1:
		Ns = beacls::IntegerVec{ 151, 151, 101, 101, 101};
		break;
	case 2:
	default:
		Ns = beacls::IntegerVec{ 501, 501, 301, 301, 301 };
		break;
	case 3:
		Ns = beacls::IntegerVec{ 1501, 1501, 1001, 1001, 1001 };
		break;
	}
	levelset::HJI_Grid* g = helperOC::createGrid( // Grid limits (HARD-CODED)
		beacls::FloatVec{
				(FLOAT_TYPE)-21, (FLOAT_TYPE)-18, (FLOAT_TYPE)-M_PI,
				 (FLOAT_TYPE)1, (FLOAT_TYPE)-M_PI}, 
		beacls::FloatVec{
				(FLOAT_TYPE)15, (FLOAT_TYPE)18, (FLOAT_TYPE)M_PI,
				 (FLOAT_TYPE)5, (FLOAT_TYPE)M_PI}, 
		Ns, beacls::IntegerVec{2}); // Only periodic dimension is theta_rel

	beacls::FloatVec tau = generateArithmeticSequence<FLOAT_TYPE>(0., dt, tMax);
	// Generate P5D_Dubins object (take default parameter values in .hpp)
	helperOC::P5D_Dubins* p5D_Dubins = new helperOC::P5D_Dubins(
		beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0,
			(FLOAT_TYPE)0, (FLOAT_TYPE)0});
	
	// Dynamical system parameters
	helperOC::DynSysSchemeData* schemeData = new helperOC::DynSysSchemeData;
	schemeData->set_grid(g);
	schemeData->dynSys = p5D_Dubins;
	schemeData->accuracy = helperOC::ApproximationAccuracy_veryHigh;
	schemeData->uMode = helperOC::DynSys_UMode_Max;
	schemeData->dMode = helperOC::DynSys_DMode_Min;

	// Target set and visualization
	beacls::FloatVec data0;
	levelset::ShapeSphere* shape = new levelset::ShapeSphere(
		beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0,
			(FLOAT_TYPE)0, (FLOAT_TYPE)0}, (FLOAT_TYPE)0.1);
	shape->execute(g, data0);
	for(size_t i = 0; i < data0.size(); i++) data0[i] = -data0[i]; // Positive inside

	helperOC::HJIPDE_extraArgs extraArgs;
	helperOC::HJIPDE_extraOuts extraOuts;
	extraArgs.visualize = false;
	extraArgs.execParameters.line_length_of_chunk = line_length_of_chunk;
	extraArgs.execParameters.calcTTR = calculateTTRduringSolving;
	extraArgs.keepLast = keepLast;
	extraArgs.execParameters.useCuda = useCuda;
	extraArgs.execParameters.num_of_gpus = num_of_gpus;
	extraArgs.execParameters.num_of_threads = num_of_threads;
	extraArgs.execParameters.delayedDerivMinMax = delayedDerivMinMax;
	extraArgs.execParameters.enable_user_defined_dynamics_on_gpu = enable_user_defined_dynamics_on_gpu;

	helperOC::HJIPDE* hjipde;
	if (useTempFile) hjipde = new helperOC::HJIPDE(std::string("tmp.mat"));
	else hjipde = new helperOC::HJIPDE();

	beacls::FloatVec stoptau;
	std::vector<beacls::FloatVec > data;
	hjipde->solve(data, stoptau, extraOuts, data0, tau, schemeData, helperOC::HJIPDE::MinWithType_Zero, extraArgs);
	beacls::FloatVec TTR;
	std::vector<beacls::FloatVec > P, derivL, derivR;

	// Return signs to standard negative-inside convention
	for(size_t i = 0; i < data0.size(); i++) data0[i] = -data0[i];
	for(size_t t = 0; t < data.size(); t++)
		for(size_t i = 0; i < data[t].size(); i++) data[t][i] = -data[t][i];


	// Convert to TTR
	// hjipde->TD2TTR(TTR, g, tau); // JFF: Not needed for Tracking Error Bound.

	// Compute gradient
	if (!TTR.empty()) {
		helperOC::ExtractCostates* extractCostates = new helperOC::ExtractCostates(schemeData->accuracy);
		extractCostates->operator()(P, derivL, derivR, g, TTR, TTR.size(), false, extraArgs.execParameters);
		if (extractCostates) delete extractCostates;
	}

	if (dump_file) {
		std::string P5D_Dubins_RS_RS_filename("P5D_Dubins_RS_RS.mat");
		beacls::MatFStream* P5D_Dubins_RS_RS_fs = beacls::openMatFStream(P5D_Dubins_RS_RS_filename, beacls::MatOpenMode_Write);
		g->save_grid(std::string("g"), P5D_Dubins_RS_RS_fs);
		if (!data.empty()) save_vector_of_vectors(data, std::string("data"), Ns, false, P5D_Dubins_RS_RS_fs);
		if(!TTR.empty()) save_vector(TTR, std::string("TTR"), Ns, false, P5D_Dubins_RS_RS_fs);
		if (!tau.empty()) save_vector(tau, std::string("tau"), beacls::IntegerVec(), false, P5D_Dubins_RS_RS_fs);
		if (!P.empty()) save_vector_of_vectors(P, std::string("P"), Ns, false, P5D_Dubins_RS_RS_fs);
		beacls::closeMatFStream(P5D_Dubins_RS_RS_fs);

		std::string P5D_Dubins_RS_RS_smaller_filename("P5D_Dubins_RS_RS_smaller.mat");
		beacls::MatFStream* P5D_Dubins_RS_RS_smaller_fs = beacls::openMatFStream(P5D_Dubins_RS_RS_smaller_filename, beacls::MatOpenMode_Write);
		g->save_grid(std::string("g"), P5D_Dubins_RS_RS_smaller_fs);
		if (!TTR.empty()) save_vector(TTR, std::string("TTR"), Ns, false, P5D_Dubins_RS_RS_smaller_fs);
		if (!P.empty()) save_vector_of_vectors(P, std::string("P"), Ns, false, P5D_Dubins_RS_RS_smaller_fs);
		if (!tau.empty()) save_vector(tau, std::string("tau"), beacls::IntegerVec(), false, P5D_Dubins_RS_RS_smaller_fs);
		beacls::closeMatFStream(P5D_Dubins_RS_RS_smaller_fs);
	}

	if (hjipde) delete hjipde;
	if (shape) delete shape;
	if (schemeData) delete schemeData;
	if (p5D_Dubins) delete p5D_Dubins;
	if (g) delete g;
	return 0;
}