#define _USE_MATH_DEFINES
#include <levelset/levelset.hpp>
#include <helperOC/helperOC.hpp>
#include <cmath>
#include <numeric>
#include <functional>
#include <cfloat>
#include <cstring>
#include "MyPlane.hpp"

/**
	@brief Tests the Plane class by computing a reachable set and then computing the optimal trajectory from the reachable set.
	*/
int main(int argc, char *argv[])
{
	bool dump_file = false;
	if (argc >= 2) {
		dump_file = (atoi(argv[1]) == 0) ? false : true;
	}

	//!< Plane parameters
	const beacls::FloatVec initState{ 100, 75, (FLOAT_TYPE)(220 * M_PI / 180) };
	const FLOAT_TYPE wMax = (FLOAT_TYPE)1.2;
	const beacls::FloatVec vrange{ (FLOAT_TYPE)1.1, (FLOAT_TYPE)1.3 };
	const beacls::FloatVec dMax{ 0, 0 };
	MyPlane* pl = new MyPlane(initState, wMax, vrange, dMax);

	const FLOAT_TYPE inf = std::numeric_limits<FLOAT_TYPE>::infinity();
	//!< Target and obstacle
	levelset::HJI_Grid* g = helperOC::createGrid(
		beacls::FloatVec{0, 0, 0}, 
		beacls::FloatVec{150, 150, (FLOAT_TYPE)(2*M_PI)},
		beacls::IntegerVec{41,41,11});
	std::vector<beacls::FloatVec > targets(1);
	levelset::ShapeCylinder(beacls::IntegerVec{ 2 }, beacls::FloatVec{ 75, 50, 0 }, 10).execute(g, targets[0]);
	beacls::FloatVec obs1, obs2;
	levelset::ShapeRectangleByCorner(beacls::FloatVec{ 300, 250, -inf }, beacls::FloatVec{ 350, 300, inf }).execute(g, obs1);
	levelset::ShapeRectangleByCorner(beacls::FloatVec{ 5, 5, -inf }, beacls::FloatVec{ 145, 145, inf }).execute(g, obs2);
	std::transform(obs2.cbegin(), obs2.cend(), obs2.begin(), std::negate<FLOAT_TYPE>());
	std::vector<beacls::FloatVec > obstacles(1);
	obstacles[0].resize(obs1.size());
	std::transform(obs1.cbegin(), obs1.cend(), obs2.cbegin(), obstacles[0].begin(),
		std::ptr_fun<const FLOAT_TYPE&, const FLOAT_TYPE&>(std::min<FLOAT_TYPE>));

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
	if (dump_file) {
		extraArgs.fig_filename = "MyPlane_test2_BRS";
	}
	
	extraArgs.execParameters.useCuda = true;

	helperOC::HJIPDE* hjipde = new helperOC::HJIPDE();

	beacls::FloatVec tau2;
	std::vector<beacls::FloatVec > datas;
	hjipde->solve(datas, tau2, extraOuts, targets, tau, schemeData, helperOC::HJIPDE::MinWithType_None, extraArgs);

	std::string MyPlane_test2_filename("MyPlane_test2.mat");
	beacls::MatFStream* fs = beacls::openMatFStream(MyPlane_test2_filename, beacls::MatOpenMode_Write);
	if (dump_file) {
		beacls::IntegerVec Ns = g->get_Ns();

		g->save_grid(std::string("g"), fs);
		if (!datas.empty()) save_vector_of_vectors(datas, std::string("data"), Ns, false, fs);
		if (!tau2.empty()) save_vector(tau2, std::string("tau2"), beacls::IntegerVec(), false, fs);
	}
	//!< Compute optimal trajectory
	extraArgs.projDim = beacls::IntegerVec{ 1,1,0 };
	if (dump_file) {
		extraArgs.fig_filename = "MyPlane_test2_Traj";
	}
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

