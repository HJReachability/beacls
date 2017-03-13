#define _USE_MATH_DEFINES
#include <levelset/levelset.hpp>
#include <helperOC/helperOC.hpp>
#include <helperOC/DynSys/DynSys/DynSysSchemeData.hpp>
#include <helperOC/DynSys/PlaneCAvoid/PlaneCAvoid.hpp>
#include <helperOC/ComputeGradients.hpp>
#include <cmath>
#include <numeric>
#include <functional>
#include <cfloat>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cstring>

/**
	@brief Tests the Plane class by computing a reachable set and then computing the optimal trajectory from the reachable set.
	*/
int main(int argc, char *argv[])
{
	// Time stamps
	bool which_test = true;
	if (argc >= 2) {
		which_test = (atoi(argv[1]) == 0) ? false : true;
	}
	size_t gN_size = 41;
	if (argc >= 3) {
		const size_t new_gN_size = atoi(argv[2]);
		if (new_gN_size != 0) gN_size = new_gN_size;
	}

	bool dump_file = false;
	if (argc >= 4) {
		dump_file = (atoi(argv[3]) == 0) ? false : true;
	}
	bool useTempFile = false;
	if (argc >= 5) {
		useTempFile = (atoi(argv[4]) == 0) ? false : true;
	}
	const bool keepLast = false;
	const bool calculateTTRduringSolving = false;
	beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
	if (argc >= 6) {
		switch (atoi(argv[5])) {
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

	bool useCuda = false;
	if (argc >= 7) {
		useCuda = (atoi(argv[6]) == 0) ? false : true;
	}
	int num_of_threads = 0;
	if (argc >= 8) {
		num_of_threads = atoi(argv[7]);
	}
	int num_of_gpus = 0;
	if (argc >= 9) {
		num_of_gpus = atoi(argv[8]);
	}
	size_t line_length_of_chunk = 1;
	if (argc >= 10) {
		line_length_of_chunk = atoi(argv[9]);
	}

	bool enable_user_defined_dynamics_on_gpu = true;
	if (argc >= 11) {
		enable_user_defined_dynamics_on_gpu = (atoi(argv[10]) == 0) ? false : true;
	}

	//!< Grid
	//!< Choose this to be just big enough to cover the reachable set
	const beacls::FloatVec gMin = beacls::FloatVec{ (FLOAT_TYPE)-10, (FLOAT_TYPE)-15, (FLOAT_TYPE)0 };
	const beacls::FloatVec gMax = beacls::FloatVec{ (FLOAT_TYPE)25, (FLOAT_TYPE)15, (FLOAT_TYPE)(2*M_PI) };
	beacls::IntegerVec gN;
	gN.resize(3, gN_size);
	HJI_Grid* g = createGrid(gMin, gMax, gN);

	//!< Time
	//!< Choose tMax to be large enough for the set to converge
	const FLOAT_TYPE tMax = 3;
	const FLOAT_TYPE dt = (FLOAT_TYPE)0.1;
	const beacls::FloatVec tau = generateArithmeticSequence<FLOAT_TYPE>(0., dt, tMax);

	//!< Vehicle parameters
	//!< Maximum turn rate(rad / s)
	const FLOAT_TYPE wMaxA = 1;
	const FLOAT_TYPE wMaxB = (which_test) ? (FLOAT_TYPE)0 : (FLOAT_TYPE)1;

	//!< Speed range(m / s)
	const beacls::FloatVec vRangeA = beacls::FloatVec{ (FLOAT_TYPE)5, (FLOAT_TYPE)5 };
	const beacls::FloatVec vRangeB = beacls::FloatVec{ (FLOAT_TYPE)5, (FLOAT_TYPE)5 };

	//!< Disturbance(see PlaneCAvoid class)
	const beacls::FloatVec dMaxA = beacls::FloatVec{ (FLOAT_TYPE)0, (FLOAT_TYPE)0 };
	const beacls::FloatVec dMaxB = beacls::FloatVec{ (FLOAT_TYPE)0, (FLOAT_TYPE)0 };

	//!< Initial conditions
	const FLOAT_TYPE targetR = 5; //!< collision radius
	beacls::FloatVec data0;
	ShapeCylinder(beacls::IntegerVec{ 2 }, beacls::FloatVec{ (FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0 }, targetR).execute(g, data0);

	//!< Additional solver parameters
	DynSysSchemeData* sD = new DynSysSchemeData;
	sD->set_grid(g);
	sD->dynSys = new PlaneCAvoid(beacls::FloatVec{ (FLOAT_TYPE)0.,(FLOAT_TYPE)0.,(FLOAT_TYPE)0. }, wMaxA, vRangeA, wMaxB, vRangeB, dMaxA, dMaxB);
	sD->uMode = DynSys_UMode_Max;

	// Target set and visualization
	helperOC::HJIPDE_extraArgs extraArgs;
	helperOC::HJIPDE_extraOuts extraOuts;
	extraArgs.visualize = true;
	extraArgs.deleteLastPlot = true;
	extraArgs.keepLast = true;

	extraArgs.execParameters.line_length_of_chunk = line_length_of_chunk;
	extraArgs.execParameters.calcTTR = calculateTTRduringSolving;
	extraArgs.keepLast = keepLast;
	extraArgs.execParameters.useCuda = useCuda;
	extraArgs.execParameters.num_of_gpus = num_of_gpus;
	extraArgs.execParameters.num_of_threads = num_of_threads;
	extraArgs.execParameters.delayedDerivMinMax = delayedDerivMinMax;
	extraArgs.execParameters.enable_user_defined_dynamics_on_gpu = enable_user_defined_dynamics_on_gpu;

	HJIPDE* hjipde = new HJIPDE();

	//!< Call solver and save

	beacls::FloatVec tau2;
	hjipde->solve(tau2, extraOuts, data0, tau, sD, HJIPDE::MinWithType_Zero, extraArgs);
	std::vector<beacls::FloatVec > datas;
	hjipde->get_datas(datas, tau, sD);
	if (hjipde) delete hjipde;

	beacls::UVecType execType = useCuda ? beacls::UVecType_Cuda : beacls::UVecType_Vector;
	std::vector<beacls::FloatVec> derivC;
	std::vector<beacls::FloatVec> derivL;
	std::vector<beacls::FloatVec> derivR;
	beacls::FloatVec data;
	data.reserve(datas[0].size() * datas.size());
	std::for_each(datas.cbegin(), datas.cend(), [&data](const auto& rhs) { 
		data.insert(data.end(), rhs.cbegin(), rhs.cend());
	});
	helperOC::ComputeGradients(g, helperOC::ApproximationAccuracy_veryHigh, execType)(
		derivC, derivL, derivR, g, data, data.size(), false, extraArgs.execParameters);

	std::string test_PlaneCAvoid_filename("test_PlaneCAvoid");
	if (which_test) test_PlaneCAvoid_filename.append("_straight");
	else  test_PlaneCAvoid_filename.append("_cooperative");
	test_PlaneCAvoid_filename.append(".mat");

	beacls::MatFStream* fs = beacls::openMatFStream(test_PlaneCAvoid_filename, beacls::MatOpenMode_Write);
	beacls::MatVariable* struct_var = beacls::createMatStruct("safety_set");

	beacls::IntegerVec Ns = g->get_Ns();
	g->save_grid(std::string("g"), fs, struct_var);
	save_vector_of_vectors(derivC, std::string("deriv"), Ns, false, fs, struct_var);
	save_vector_of_vectors(datas, std::string("data"), Ns, false, fs, struct_var);
	beacls::writeMatVariable(fs, struct_var);
	beacls::closeMatVariable(struct_var);
	beacls::closeMatFStream(fs);

	if (sD) delete sD;
	if (g) delete g;
	return 0;
}

