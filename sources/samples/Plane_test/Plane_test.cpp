#define _USE_MATH_DEFINES
#include <levelset/levelset.hpp>
#include <helperOC/helperOC.hpp>
#include <helperOC/DynSys/DynSys/DynSysSchemeData.hpp>
#include <helperOC/DynSys/Plane/Plane.hpp>
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
	bool dump_file = false;
	if (argc >= 2) {
		dump_file = (atoi(argv[1]) == 0) ? false : true;
	}
	bool useTempFile = false;
	if (argc >= 3) {
		useTempFile = (atoi(argv[2]) == 0) ? false : true;
	}
	const bool keepLast = false;
	const bool calculateTTRduringSolving = false;
	levelset::DelayedDerivMinMax_Type delayedDerivMinMax = levelset::DelayedDerivMinMax_Disable;
	if (argc >= 4) {
		switch (atoi(argv[3])) {
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

	bool useCuda = false;
	if (argc >= 5) {
		useCuda = (atoi(argv[4]) == 0) ? false : true;
	}
	int num_of_threads = 0;
	if (argc >= 6) {
		num_of_threads = atoi(argv[5]);
	}
	int num_of_gpus = 0;
	if (argc >= 7) {
		num_of_gpus = atoi(argv[6]);
	}
	size_t line_length_of_chunk = 1;
	if (argc >= 8) {
		line_length_of_chunk = atoi(argv[7]);
	}

	bool enable_user_defined_dynamics_on_gpu = true;
	if (argc >= 9) {
		enable_user_defined_dynamics_on_gpu = (atoi(argv[8]) == 0) ? false : true;
	}
	//!< Plane parameters
	const beacls::FloatVec initState{ (FLOAT_TYPE)100, (FLOAT_TYPE)75, (FLOAT_TYPE)(220 * M_PI / 180) };
	const FLOAT_TYPE wMax = (FLOAT_TYPE)1.2;
	const beacls::FloatVec vrange{ (FLOAT_TYPE)1.1, (FLOAT_TYPE)1.3 };
	const beacls::FloatVec dMax{ (FLOAT_TYPE)0, (FLOAT_TYPE)0 };
	helperOC::Plane* pl = new helperOC::Plane(initState, wMax, vrange, dMax);

	const FLOAT_TYPE inf = std::numeric_limits<FLOAT_TYPE>::infinity();
	//!< Target and obstacle
	levelset::HJI_Grid* g = helperOC::createGrid(
		beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 
		beacls::FloatVec{(FLOAT_TYPE)150, (FLOAT_TYPE)150, (FLOAT_TYPE)(2*M_PI)}, 
		beacls::IntegerVec{ 41,41,11 });
//	beacls::IntegerVec{ 401,401,201 });
	std::vector<beacls::FloatVec > targets(1);
	levelset::ShapeCylinder(beacls::IntegerVec{ 2 }, beacls::FloatVec{ 75., 50., 0. }, (FLOAT_TYPE)10).execute(g, targets[0]);
	beacls::FloatVec obs1, obs2;
	levelset::ShapeRectangleByCorner(beacls::FloatVec{ 300, 250, -inf }, beacls::FloatVec{ 350, 300, inf }).execute(g, obs1);
	levelset::ShapeRectangleByCorner(beacls::FloatVec{ 5, 5, -inf }, beacls::FloatVec{ 145, 145, inf }).execute(g, obs2);
	std::transform(obs2.cbegin(), obs2.cend(), obs2.begin(), std::negate<FLOAT_TYPE>());
	std::vector<beacls::FloatVec > obstacles(1);
	obstacles[0].resize(obs1.size());
	std::transform(obs1.cbegin(), obs1.cend(), obs2.cbegin(), obstacles[0].begin(), std::ptr_fun<const FLOAT_TYPE&, const FLOAT_TYPE&>(std::min<FLOAT_TYPE>));

	//!< Compute reachable set
	const FLOAT_TYPE tMax = 500;
	const FLOAT_TYPE dt = 0.25;
	beacls::FloatVec tau = generateArithmeticSequence<FLOAT_TYPE>(0., dt, tMax);

	// Dynamical system parameters
	helperOC::DynSysSchemeData* schemeData = new helperOC::DynSysSchemeData;
	schemeData->set_grid(g);
	schemeData->dynSys = pl;
	schemeData->uMode = helperOC::DynSys_UMode_Min;
	schemeData->dMode = helperOC::DynSys_DMode_Max;

	// Target set and visualization
	helperOC::HJIPDE_extraArgs extraArgs;
	helperOC::HJIPDE_extraOuts extraOuts;
	extraArgs.targets = targets;
	extraArgs.obstacles = obstacles;
	extraArgs.stopInit = pl->get_x();
	extraArgs.visualize = true;
	extraArgs.plotData.plotDims = beacls::IntegerVec{ 1, 1, 0 };
	extraArgs.plotData.projpt = beacls::FloatVec{ pl->get_x()[2] };
	extraArgs.deleteLastPlot = true;
	extraArgs.fig_filename = "Plane_test_BRS";

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

	beacls::FloatVec tau2;
	std::vector<beacls::FloatVec > datas;
	hjipde->solve(datas, tau2, extraOuts, targets, tau, schemeData, helperOC::HJIPDE::MinWithType_None, extraArgs);

	std::string Plane_test_filename("Plane_test.mat");
	beacls::MatFStream* fs = beacls::openMatFStream(Plane_test_filename, beacls::MatOpenMode_Write);
	if (dump_file) {
		beacls::IntegerVec Ns = g->get_Ns();

		g->save_grid(std::string("g"), fs);
		if (!datas.empty()) save_vector_of_vectors(datas, std::string("data"), Ns, false, fs);
		if (!tau2.empty()) save_vector(tau2, std::string("tau2"), beacls::IntegerVec(), false, fs);
	}
	//!< Compute optimal trajectory
	extraArgs.projDim = beacls::IntegerVec{ 1,1,0 };
	extraArgs.fig_filename = "Plane_test_Traj";
	std::vector<beacls::FloatVec > traj;
	beacls::FloatVec traj_tau;
	std::vector<beacls::FloatVec > fliped_data(datas.size());
	std::copy(datas.crbegin(), datas.crend(), fliped_data.begin());
	helperOC::ComputeOptTraj* computeOptTraj = new helperOC::ComputeOptTraj();
	computeOptTraj->operator()(traj, traj_tau, g, fliped_data, tau2, pl, extraArgs);
	if (computeOptTraj) delete computeOptTraj;
	if (dump_file) {
		if (!traj.empty()) save_vector_of_vectors(traj, std::string("traj"), beacls::IntegerVec(), false, fs);
		if (!traj_tau.empty()) save_vector(traj_tau, std::string("traj_tau"), beacls::IntegerVec(), false, fs);
	}
	beacls::closeMatFStream(fs);


	if (hjipde) delete hjipde;
	if (schemeData) delete schemeData;
	if (pl) delete pl;
	if (g) delete g;
	return 0;
}

