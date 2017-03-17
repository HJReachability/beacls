#define _USE_MATH_DEFINES
#include <levelset/levelset.hpp>
#include <helperOC/helperOC.hpp>
#include <helperOC/DynSys/DynSys/DynSysSchemeData.hpp>
#include <helperOC/DynSys/Quad4D/Quad4D.hpp>
#include <helperOC/ValFuncs/proj.hpp>
#include <helperOC/ValFuncs/visSetIm.hpp>
#include <cmath>
#include <numeric>
#include <functional>
#include <cfloat>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cstring>
#if defined(VISUALIZE_BY_OPENCV)
#include <opencv2/opencv.hpp>
#endif	/* defined(VISUALIZE_BY_OPENCV) */

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

	const FLOAT_TYPE inf = std::numeric_limits<FLOAT_TYPE>::infinity();
	//!< Common parameters
	beacls::FloatVec targetLower{ (FLOAT_TYPE)-0.5, (FLOAT_TYPE)-inf, (FLOAT_TYPE)-0.5, (FLOAT_TYPE)-inf };
	beacls::FloatVec targetUpper{ (FLOAT_TYPE)0.5, (FLOAT_TYPE)inf, (FLOAT_TYPE)0.5, (FLOAT_TYPE)inf };

	//!< Grid
	beacls::FloatVec gMin{ (FLOAT_TYPE)-4, (FLOAT_TYPE)-3, (FLOAT_TYPE)-4, (FLOAT_TYPE)-3 };
	beacls::FloatVec gMax{ (FLOAT_TYPE)4,  (FLOAT_TYPE)3,  (FLOAT_TYPE)4, (FLOAT_TYPE)3 };
	
	helperOC::DynSys_UMode_Type uMode = helperOC::DynSys_UMode_Max;

	beacls::IntegerVec gN{ 65, 61, 65, 61 };

	//!< Time
	const FLOAT_TYPE tMax = 3;
	const FLOAT_TYPE dt = 0.1;
	beacls::FloatVec tau = generateArithmeticSequence<FLOAT_TYPE>(0., dt, tMax);

	//!< Vehicle
	const FLOAT_TYPE uMin = -1;
	const FLOAT_TYPE uMax = 1;

	//!< Grids and initial conditions
	levelset::HJI_Grid* g = helperOC::createGrid(gMin, gMax, gN);
	beacls::FloatVec data0;
	levelset::ShapeRectangleByCorner(targetLower, targetUpper).execute(g, data0);

	//!< Additional solver parameters
	helperOC::DynSysSchemeData* sD = new helperOC::DynSysSchemeData();
	sD->set_grid(g);
	sD->dynSys = new helperOC::Quad4D(beacls::FloatVec{ (FLOAT_TYPE)0, (FLOAT_TYPE)0,(FLOAT_TYPE)0,(FLOAT_TYPE)0 }, uMin, uMax);
	sD->uMode = uMode;

	beacls::FloatVec vslice{ (FLOAT_TYPE)2.5, (FLOAT_TYPE)2.5 };

	// Target set and visualization
	helperOC::HJIPDE_extraArgs extraArgs;
	helperOC::HJIPDE_extraOuts extraOuts;
	extraArgs.visualize = true;
	extraArgs.plotData.plotDims = beacls::IntegerVec{ 1, 0, 1, 0 };
	extraArgs.plotData.projpt = vslice;
	extraArgs.keepLast = true;
	extraArgs.fig_filename = "Quad4D_test_BRS";

	extraArgs.execParameters.line_length_of_chunk = line_length_of_chunk;
	extraArgs.execParameters.calcTTR = calculateTTRduringSolving;
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
	hjipde->solve(datas, tau2, extraOuts, data0, tau, sD, helperOC::HJIPDE::MinWithType_Zero, extraArgs);

	if (dump_file) {
		std::string Quad4D_test_filename("Quad4D_test.mat");
		beacls::MatFStream* fs = beacls::openMatFStream(Quad4D_test_filename, beacls::MatOpenMode_Write);
		beacls::IntegerVec Ns = g->get_Ns();

		g->save_grid(std::string("g"), fs);
		if (!datas.empty()) save_vector_of_vectors(datas, std::string("data"), Ns, false, fs);
		if (!tau2.empty()) save_vector(tau2, std::string("tau"), Ns, false, fs);

		beacls::closeMatFStream(fs);
	}
#if defined(VISUALIZE_BY_OPENCV)

	//!< Visualize
	levelset::HJI_Grid* g2Dp;
	beacls::FloatVec data2Dp;
	g2Dp = helperOC::proj(data2Dp, sD->get_grid(), datas[0], beacls::IntegerVec{ 0,1,0,1 });

	cv::Mat image_mat;
	helperOC::visSetIm(image_mat, image_mat, g2Dp, data2Dp);

	cv::namedWindow(__func__, 0);
	cv::imshow(__func__, image_mat);
	cv::waitKey(1);

	beacls::FloatVec vx = generateArithmeticSequence((FLOAT_TYPE)-2.5, (FLOAT_TYPE)0.5, (FLOAT_TYPE)2.5);
	beacls::FloatVec vy = generateArithmeticSequence((FLOAT_TYPE)-2.5, (FLOAT_TYPE)0.5, (FLOAT_TYPE)2.5);
	std::for_each(vx.cbegin(), vx.cend(), [&image_mat, &datas, &sD, &vy](const auto& x) {
		std::for_each(vy.cbegin(), vy.cend(), [&image_mat, &datas, &sD, x](const auto& y) {
			beacls::FloatVec vslice{ x, y };
			levelset::HJI_Grid* g2Dp;
			beacls::FloatVec data2Dp;
			std::vector<helperOC::Projection_Type> proj_types(vslice.size());
			std::fill(proj_types.begin(), proj_types.end(), helperOC::Projection_Vector);
			g2Dp = helperOC::proj(data2Dp, sD->get_grid(), datas[0], beacls::IntegerVec{ 0,1,0,1 }, proj_types, vslice);
			helperOC::visSetIm(image_mat, image_mat, g2Dp, data2Dp);
			cv::imshow(__func__, image_mat);
			cv::waitKey(16);
		});
	});
	cv::waitKey(0);
#endif	/* defined(VISUALIZE_BY_OPENCV) */

	if (hjipde) delete hjipde;
	if (sD->dynSys) delete sD->dynSys;
	if (sD) delete sD;
	if (g) delete g;
	return 0;
}

