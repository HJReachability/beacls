#include <helperOC/ValFuncs/HJIPDE.hpp>
#include "HJIPDE_impl.hpp"
#include <helperOC/ValFuncs/eval_u.hpp>
#include <helperOC/ValFuncs/proj.hpp>
#include <helperOC/ValFuncs/visSetIm.hpp>
#include <algorithm>
#include <functional>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <levelset/levelset.hpp>
#include <helperOC/DynSys/DynSys/DynSysSchemeData.hpp>
#if defined(VISUALIZE_BY_OPENCV)
#include <opencv2/opencv.hpp>
#endif  /* defined(VISUALIZE_BY_OPENCV) */
using namespace helperOC;

HJIPDE_impl::HJIPDE_impl(
	const std::string& tmp_filename
) : tmp_filename(tmp_filename) {
}
HJIPDE_impl::~HJIPDE_impl() {
}

bool helperOC::ExecParameters::operator==(const ExecParameters& rhs) const {
	if (this == &rhs) return true;
	else if (line_length_of_chunk != rhs.line_length_of_chunk) return false;  //!< Line length of each parallel execution chunks  (0 means set automatically)
	else if (num_of_threads != rhs.num_of_threads) return false;  //!< Number of CPU Threads (0 means use all logical threads of CPU)
	else if (num_of_gpus != rhs.num_of_gpus) return false;  //!<  Number of GPUs which (0 means use all GPUs)
	else if (calcTTR != rhs.calcTTR) return false; //!< calculate TTR during solving
	else if (delayedDerivMinMax != rhs.delayedDerivMinMax) return false;  //!< Use last step's min/max of derivatives, and skip 2nd pass.
	else if (useCuda != rhs.useCuda) return false;//!< Execute type CPU Vector or GPU.
	else if (enable_user_defined_dynamics_on_gpu != rhs.enable_user_defined_dynamics_on_gpu) return false; //!< Flag for user defined dynamics function on gpu
	else return true;
}


bool HJIPDE_impl::getNumericalFuncs(
	levelset::Dissipation*& dissFunc,
	levelset::Integrator*& integratorFunc,
	levelset::SpatialDerivative*& derivFunc,
	const levelset::HJI_Grid* grid,
	const levelset::Term* schemeFunc,
	const helperOC::Dissipation_Type dissType,
	const helperOC::ApproximationAccuracy_Type accuracy,
	const FLOAT_TYPE factorCFL,
	const bool stats,
	const bool single_step,
	const beacls::UVecType type
) const {
	//! Dissipation
	switch (dissType) {
	case helperOC::Dissipation_global:
		dissFunc = new levelset::ArtificialDissipationGLF();
		break;
	case helperOC::Dissipation_local:
#if 0
		///! ToDo
		dissFunc = new levelset::ArtificialDissipationLLF();
		break;
#else
		std::cerr << "Dissipation function " << dissType << " is not implemented yet." << std::endl;
		dissFunc = NULL;
		return false;
#endif
	case helperOC::Dissipation_locallocal:
#if 0
		///! ToDo
		dissFunc = new levelset::ArtificialDissipationLLLF();
		break;
#else
		std::cerr << "Dissipation function " << dissType << " is not implemented yet." << std::endl;
		dissFunc = NULL;
		return false;
#endif
	case helperOC::Dissipation_Invalid:
	default:
		std::cerr << "Unknown dissipation function " << dissType << std::endl;
		dissFunc = NULL;
		return false;
	}

	const FLOAT_TYPE max_step = std::numeric_limits<FLOAT_TYPE>::max();
	std::vector<levelset::PostTimestep_Exec_Type* > postTimestep_Execs;

	//! accuracy
	switch (accuracy) {
	case helperOC::ApproximationAccuracy_low:
		derivFunc = new levelset::UpwindFirstFirst(grid, type);
		integratorFunc = new levelset::OdeCFL1(schemeFunc, factorCFL, max_step, postTimestep_Execs, single_step, stats, NULL);
		break;
	case helperOC::ApproximationAccuracy_medium:
		derivFunc = new levelset::UpwindFirstENO2(grid, type);
		integratorFunc = new levelset::OdeCFL2(schemeFunc, factorCFL, max_step, postTimestep_Execs, single_step, stats, NULL);
		break;
	case helperOC::ApproximationAccuracy_high:
		derivFunc = new levelset::UpwindFirstENO3(grid, type);
		integratorFunc = new levelset::OdeCFL3(schemeFunc, factorCFL, max_step, postTimestep_Execs, single_step, stats, NULL);
		break;
	case helperOC::ApproximationAccuracy_veryHigh:
		derivFunc = new levelset::UpwindFirstWENO5(grid, type);
		integratorFunc = new levelset::OdeCFL3(schemeFunc, factorCFL, max_step, postTimestep_Execs, single_step, stats, NULL);
		break;
	case helperOC::ApproximationAccuracy_Invalid:
	default:
		std::cerr << "Unknown accuracy level " << accuracy << std::endl;
		derivFunc = NULL;
		integratorFunc = NULL;
		return false;
	}
	return true;
}
static bool calcChange(
	FLOAT_TYPE& change,
	const beacls::FloatVec& y,
	const beacls::FloatVec& y0
) {
	change = std::inner_product(y.cbegin(), y.cend(), y0.cbegin(), std::fabs(y[0] - y0[0]),
		([](const auto &lhs, const auto &rhs) { return std::max<FLOAT_TYPE>(lhs, rhs); }),
		([](const auto &lhs, const auto &rhs) { return std::fabs(lhs - rhs); })
	);
	return true;
}

bool HJIPDE_impl::solve(beacls::FloatVec& dst_tau,
	helperOC::HJIPDE_extraOuts& extraOuts,
	const std::vector<beacls::FloatVec >& src_datas,
	const beacls::FloatVec& src_tau,
	const DynSysSchemeData* schemeData,
	const HJIPDE::MinWithType minWith,
	const helperOC::HJIPDE_extraArgs& extraArgs) {
	const bool quiet = extraArgs.quiet;
	const helperOC::ExecParameters execParameters = extraArgs.execParameters;
	const beacls::UVecType execType = (execParameters.useCuda) ? beacls::UVecType_Cuda : beacls::UVecType_Vector;

	const FLOAT_TYPE large = (FLOAT_TYPE)1e6;
	const FLOAT_TYPE small = (FLOAT_TYPE)1e-4;
	const levelset::HJI_Grid* grid = schemeData->get_grid();
	const size_t num_of_dimensions = grid->get_num_of_dimensions();
	// Extract the information from extraargs
	if (quiet) {
		std::cout << "HJIPDE_solve running in quiet mode..." << std::endl;
	}

	//!< Low memory mode
	if (extraArgs.low_memory) {
		std::cout << "HJIPDE_solve running in low memory mode..." << std::endl;
	}
	const bool low_memory = extraArgs.low_memory;
	//!< Save the output in reverse order
	const bool flip_output = extraArgs.low_memory && extraArgs.flip_output;

	const bool keepLast = extraArgs.keepLast;

	// Extract the information about obstacles_ptrs
	HJIPDE::ObsModeType obsMode = HJIPDE::ObsModeType_None;
	std::vector<const beacls::FloatVec* > modified_obstacles_ptrs;
	std::vector<const std::vector<int8_t>* > modified_obstacles_s8_ptrs;
	if (extraArgs.obstacles_ptrs.empty()) {
		modified_obstacles_ptrs.resize(extraArgs.obstacles.size());
		for (size_t i = 0; i < extraArgs.obstacles.size(); ++i) {
			modified_obstacles_ptrs[i] = &extraArgs.obstacles[i];
		}
	}
	if (extraArgs.obstacles_s8_ptrs.empty()) {
		modified_obstacles_s8_ptrs.resize(extraArgs.obstacles_s8.size());
		for (size_t i = 0; i < extraArgs.obstacles_s8.size(); ++i) {
			modified_obstacles_s8_ptrs[i] = &extraArgs.obstacles_s8[i];
		}
	}
	const std::vector<const beacls::FloatVec* >& obstacles_ptrs = !extraArgs.obstacles_ptrs.empty() ? extraArgs.obstacles_ptrs : modified_obstacles_ptrs;
	const std::vector<const std::vector<int8_t>* >& obstacles_s8_ptrs = !extraArgs.obstacles_s8_ptrs.empty() ? extraArgs.obstacles_s8_ptrs : modified_obstacles_s8_ptrs;
	const beacls::FloatVec* obstacle_i = NULL;
	const std::vector<int8_t>* obstacle_s8_i = NULL;
	if (!obstacles_ptrs.empty()) {
		obstacle_i = obstacles_ptrs[0];
		if (obstacles_ptrs.size() == 1) {
			obsMode = HJIPDE::ObsModeType_Static;
		}
		else if (obstacles_ptrs.size() > 1) {
			obsMode = HJIPDE::ObsModeType_TimeVarying;
		}
		else {
			std::cerr << "Iconsistent obstacle dimensions!" << std::endl;
			return false;
		}
	}
	if (!obstacles_s8_ptrs.empty()) {
		obstacle_s8_i = obstacles_s8_ptrs[0];
		if (obstacles_s8_ptrs.size() == 1) {
			obsMode = HJIPDE::ObsModeType_Static;
		}
		else if (obstacles_s8_ptrs.size() > 1) {
			obsMode = HJIPDE::ObsModeType_TimeVarying;
		}
		else {
			std::cerr << "Iconsistent obstacle dimensions!" << std::endl;
			return false;
		}
	}
	// Extract the information about targets
	std::vector<const beacls::FloatVec* > modified_targets_ptrs;
	std::vector<const std::vector<int8_t>* > modified_targets_s8_ptrs;
	if (extraArgs.targets_ptrs.empty()) {
		modified_targets_ptrs.resize(extraArgs.targets.size());
		for (size_t i = 0; i < extraArgs.targets.size(); ++i) {
			modified_targets_ptrs[i] = &extraArgs.targets[i];
		}
	}
	if (extraArgs.targets_s8_ptrs.empty()) {
		modified_targets_s8_ptrs.resize(extraArgs.targets_s8.size());
		for (size_t i = 0; i < extraArgs.targets_s8.size(); ++i) {
			modified_targets_s8_ptrs[i] = &extraArgs.targets_s8[i];
		}
	}
	const std::vector<const beacls::FloatVec* >& targets_ptrs = !extraArgs.targets_ptrs.empty() ? extraArgs.targets_ptrs : modified_targets_ptrs;
	const std::vector<const std::vector<int8_t>* >& targets_s8_ptrs = !extraArgs.targets_s8_ptrs.empty() ? extraArgs.targets_s8_ptrs : modified_targets_s8_ptrs;

	// Check validity of stopInit if needed
	if (!extraArgs.stopInit.empty()) {
		if (extraArgs.stopInit.size() != num_of_dimensions) {
			std::cerr << "stopInit must be a vector of length g.dim!" << std::endl;
			return false;
		}
	}

	beacls::IntegerVec setInds;
	beacls::FloatVec stopSet;
	size_t stopLevel = 0;
	// Check validity of stopSet if needed
	if (!extraArgs.stopSetInclude.empty() || !extraArgs.stopSetIntersect.empty()) {
		if (!extraArgs.stopSetInclude.empty())
			stopSet = extraArgs.stopSetInclude;
		else
			stopSet = extraArgs.stopSetIntersect;

		if (stopSet.size() != grid->get_sum_of_elems()) {
			std::cerr << "Inconsistent stopSet dimensions!" << std::endl;
			return false;
		}
		// Extract set of indices at which stopSet is negative
		for (size_t index = 0; index < stopSet.size(); ++index) {
			if (stopSet[index] < 0) setInds.push_back(index);
		}

		// Check validity of stopLevel if needed
		if (extraArgs.stopLevel != 0)
			stopLevel = extraArgs.stopLevel;
	}


	beacls::IntegerVec plotDims;
	beacls::FloatVec projpt;
	bool deleteLastPlot = false;
	bool need_light = false;
#if defined(VISUALIZE_BY_OPENCV)
	cv::Mat HJIPDE_img;
	cv::Mat HJIPDE_initial_img;
	if (extraArgs.visualize) {
		const cv::Size dsize = extraArgs.visualize_size.size() == 2 ? cv::Size((int)extraArgs.visualize_size[0], (int)extraArgs.visualize_size[1]) : cv::Size();
		const double fx = extraArgs.fx;
		const double fy = extraArgs.fy;
		beacls::IntegerVec plotDimsIdx;
		if (!extraArgs.plotData.plotDims.empty()) {
			// Dimensions to visualize
			//  It will be an array of 1s and 0s with 1s means that dimension should
			//  be plotted.
			plotDims = extraArgs.plotData.plotDims;
			//  Points to project other dimensions at.There should be an entry point
			//  corresponding to each 0 in plotDims.
			projpt = extraArgs.plotData.projpt;

			for (size_t dimension = 0; dimension < plotDims.size(); ++dimension) {
				if (plotDims[dimension] != 0)plotDimsIdx.push_back(dimension);
			}
		}
		else {
			plotDims.assign(num_of_dimensions, 1);
		}
		deleteLastPlot = extraArgs.deleteLastPlot;
		//!<   % Initialize the figure for visualization  
#if defined(VISUALIZE_WITH_GUI)
		cv::namedWindow("HJIPDE", 0);
#endif
		need_light = true;
		if (obsMode == HJIPDE::ObsModeType_Static) {
			beacls::FloatVec tmp_obstacle;
			if (obstacle_s8_i) {
				tmp_obstacle.resize(obstacle_s8_i->size());
				std::transform(obstacle_s8_i->cbegin(), obstacle_s8_i->cend(), tmp_obstacle.begin(), [large](const auto& rhs) {
					return rhs * fix_point_ratio_inv;
				});
			}
			const beacls::FloatVec* obstacle_ptr = obstacle_i ? obstacle_i : &tmp_obstacle;
			if (all_of(plotDims.cbegin(), plotDims.cend(), [](const auto& rhs) { return rhs != 0; })) {
				helperOC::visSetIm(HJIPDE_initial_img, HJIPDE_initial_img, grid, *obstacle_ptr, std::vector<float>{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, beacls::FloatVec(), true, std::string(), dsize, fx, fy);
			}
			else {
				levelset::HJI_Grid* gPlot;
				beacls::FloatVec obsPlot;
				beacls::IntegerVec negatedPlotDims(plotDims.size());
				std::transform(plotDims.cbegin(), plotDims.cend(), negatedPlotDims.begin(), [](const auto& rhs) { return (rhs == 0) ? 1 : 0; });
				std::vector<helperOC::Projection_Type> proj_types(projpt.size());
				std::fill(proj_types.begin(), proj_types.end(), helperOC::Projection_Vector);
				gPlot = helperOC::proj(obsPlot, grid, *obstacle_ptr, negatedPlotDims, proj_types, projpt);
				// visSetIm
				helperOC::visSetIm(HJIPDE_initial_img, HJIPDE_initial_img, gPlot, obsPlot, std::vector<float>{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, beacls::FloatVec(), true, std::string(), dsize, fx, fy);
				if (gPlot) delete gPlot;
			}
		}
		if (!extraArgs.stopInit.empty()) {
			beacls::FloatVec projectedInit;
			const int left_margin = 25;
			const int right_margin = 25;
			const int top_margin = 25;
			const int bottom_margin = 25;
			for (size_t dimension = 0; dimension < plotDims.size(); ++dimension) {
				if (plotDims[dimension] != 0) {
					projectedInit.push_back(extraArgs.stopInit[dimension]);
				}
			}

			const beacls::FloatVec&xs0 = grid->get_xs(plotDimsIdx[0]);
			const beacls::FloatVec&xs1 = grid->get_xs(plotDimsIdx[1]);
			const auto x0MinMax = beacls::minmax_value<FLOAT_TYPE>(xs0.cbegin(), xs0.cend());
			const auto x1MinMax = beacls::minmax_value<FLOAT_TYPE>(xs1.cbegin(), xs1.cend());
			const FLOAT_TYPE x0_range = x0MinMax.second - x0MinMax.first;
			const FLOAT_TYPE x1_range = x1MinMax.second - x1MinMax.first;

			const double org_width = x0_range;
			const double org_height = x1_range;
			double actual_fx = 1.;
			double actual_fy = 1.;
			cv::Size size;
			if ((dsize.height != 0) && (dsize.width != 0)) {
				actual_fx = (double)dsize.width / org_width;
				actual_fy = (double)dsize.height / org_height;
				size = dsize;
			}
			else {
				actual_fx = fx != 0 ? fx : 1.;
				actual_fy = fy != 0 ? fy : 1.;
				size = cv::Size((int)std::ceil(org_width * actual_fx), (int)std::ceil(org_height * actual_fy));
			}

			const int width = size.width;
			const int height = size.height;
			cv::Size margined_size(width + left_margin + right_margin, height + top_margin + bottom_margin);
			const FLOAT_TYPE left_offset = x0MinMax.first;
			const FLOAT_TYPE top_offset = x1MinMax.first;
			if (HJIPDE_initial_img.empty()) {
				HJIPDE_initial_img = cv::Mat(margined_size, CV_8UC3, cv::Scalar(255, 255, 255));
			}
			const double fontScale = 0.5;
			const int thickness = 1;
			if (projectedInit.size() == 2) {
				std::string star_str("*");
				int baseline = 0;
				cv::Size textSize = cv::getTextSize(star_str, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
				cv::putText(HJIPDE_initial_img, star_str,
					cv::Point((int32_t)std::round(projectedInit[0] * actual_fx - textSize.width / 2. + left_margin - left_offset),
					(int32_t)::std::round(height - projectedInit[1] * actual_fy + textSize.height / 2. + top_margin - top_offset)),
					cv::FONT_HERSHEY_SIMPLEX, fontScale,
					cv::Scalar{ 255,0,0 }, thickness, cv::LINE_AA);
			}
			else if (projectedInit.size() == 3) {
				// T.B.D.
				std::cerr << "Warning: " << __func__ << " : visualize is not supported yet." << std::endl;
			}

		}
#if defined(VISUALIZE_WITH_GUI)
		if (!HJIPDE_initial_img.empty()) {
			cv::imshow("HJIPDE", HJIPDE_initial_img);
			cv::waitKey(1);
		}
#endif
	}
#endif  /* defined(VISUALIZE_BY_OPENCV) */

	// Extract dynamical system if needed

	bool stopConverge = extraArgs.stopConverge;
	FLOAT_TYPE convergeThreshold = extraArgs.convergeThreshold;

	//// SchemeFunc and SchemeData
	levelset::Term* schemeFunc = new levelset::TermLaxFriedrichs(schemeData, execType);
	// Extract accuracy parameter o/w set default accuracy
	helperOC::ApproximationAccuracy_Type accuracy = helperOC::ApproximationAccuracy_veryHigh;
	if (schemeData->accuracy != helperOC::ApproximationAccuracy_Invalid) {
		accuracy = schemeData->accuracy;
	}

	DynSysSchemeData* modified_schemeData = schemeData->clone();

	// Time integration
	FLOAT_TYPE factor_cfl = (FLOAT_TYPE)0.8;
	bool single_step = true;
	bool stats = false;

	// Numerical approximation functions
	helperOC::Dissipation_Type dissType = helperOC::Dissipation_global;
	levelset::Dissipation* dissFunc;
	levelset::Integrator* integratorFunc;
	levelset::SpatialDerivative* derivFunc;
	getNumericalFuncs(dissFunc, integratorFunc, derivFunc, grid, schemeFunc, dissType, accuracy, factor_cfl, stats, single_step, execType);
	modified_schemeData->set_spatialDerivative(derivFunc);
	modified_schemeData->set_dissipation(dissFunc);

	auto startTime = std::chrono::system_clock::now();

	// Initialize PDE solution
	const size_t num_of_elements = grid->get_sum_of_elems();
	//  for_each(dst_datas.begin(), dst_datas.end(), ([num_of_elements](auto& rhs) { rhs.resize(num_of_elements,0); }));

	size_t istart;
	if (src_datas.size() == 1) {
		//!< New computation
		istart = 1;
	}
	else {
		//!< Continue an old computation
		istart = extraArgs.istart;
	}
	beacls::MatFStream* tmp_file_fs = NULL;
	beacls::MatVariable* tmp_datas_variable = NULL;
	beacls::IntegerVec Ns = grid->get_Ns();
	datas.clear();
	if (!tmp_filename.empty()) {
		tmp_file_fs = beacls::openMatFStream(tmp_filename, beacls::MatOpenMode_WriteAppend);
		tmp_datas_variable = beacls::createMatCell(std::string("tmp"), src_datas.size());
		for (size_t i = 0; i < istart; ++i) {
			save_vector(src_datas[i], std::string(), Ns, false, tmp_file_fs, tmp_datas_variable, i);
		}
		beacls::closeMatVariable(tmp_datas_variable);
	}
	else if (!keepLast) {
		if (src_datas.size() >= istart) {
			datas.resize(istart);
			std::copy(src_datas.cbegin(), src_datas.cbegin() + istart, datas.begin());
		}
	}

	beacls::FloatVec y;
	if (src_datas.size() != 1) {
		if (flip_output) {
			y = src_datas[0];
		}
		else {
			y = src_datas[istart - 1];
		}
	}
	else if (!tmp_filename.empty()) {
		load_vector(y, std::string(), Ns, false, tmp_file_fs, tmp_datas_variable, istart - 1);
	}
	else if (!keepLast) {
		if (flip_output) {
			y = datas[0];
		}
		else {
			y = datas[istart - 1];
		}
	}
	else {
		y = src_datas[0];
	}
#if !defined(FLOAT_TYPE_32F)
	if (low_memory) {
		std::transform(y.cbegin(), y.cend(), y.begin(), [](const auto& rhs) {
			return static_cast<FLOAT_TYPE>(static_cast<float>(rhs));
		});
	}
#endif
	if (!obstacles_ptrs.empty() && obstacle_i) {
		if (y.size() == obstacle_i->size()) {
			std::transform(y.cbegin(), y.cend(), obstacle_i->cbegin(), y.begin(), ([](const auto& lhs, const auto& rhs) {
				return std::max<FLOAT_TYPE>(lhs, -rhs);
			}));
		}
	}
	if (!obstacles_s8_ptrs.empty() && obstacle_s8_i) {
		if (y.size() == obstacle_s8_i->size()) {
			std::transform(y.cbegin(), y.cend(), obstacle_s8_i->cbegin(), y.begin(), ([large](const auto& lhs, const auto& rhs) {
				return std::max<FLOAT_TYPE>(lhs, -rhs * fix_point_ratio_inv);
			}));
		}
	}

	//!< Calculate TTR during solving
	if (execParameters.calcTTR) {
		if (calculatedTTR.size() != num_of_elements) calculatedTTR.resize(num_of_elements);
		std::transform(y.cbegin(), y.cend(), calculatedTTR.begin(), ([large](const auto& rhs) {
			if (rhs <= 0) return (FLOAT_TYPE)0.;
			else return large;
		}));
	}
	for (size_t i = istart; i < src_tau.size(); ++i) {
		if (!quiet) {
			std::cout << "tau(i) = " << std::fixed << std::setprecision(6) << src_tau[i] << std::resetiosflags(std::ios_base::floatfield) << std::endl;
		}
		beacls::FloatVec yLastTau;
		if (stopConverge &&
			(
			(src_datas.size() <= (i - 1)) &&
				tmp_filename.empty() &&
				keepLast
				)
			) {
			yLastTau = y;
		}

		// Variable SchemeData
		if (extraArgs.sdModFunctor) {
			//!< Load dst_datas from file.
			if (!tmp_filename.empty()) {
				std::vector<beacls::FloatVec> datas_vec;
				get_datas(datas_vec, src_tau, modified_schemeData);
				if (!obstacles_s8_ptrs.empty())
					extraArgs.sdModFunctor->operator()(modified_schemeData, i, src_tau, datas_vec, obstacles_s8_ptrs, extraArgs.sdModParams);
				else
					extraArgs.sdModFunctor->operator()(modified_schemeData, i, src_tau, datas_vec, obstacles_ptrs, extraArgs.sdModParams);
				datas.resize(datas_vec.size());
				std::copy(datas_vec.cbegin(), datas_vec.cend(), datas.begin());
				//!< Put away dst_datas from file.
				if (!tmp_filename.empty()) {
					for_each(datas.begin(), datas.end(), ([num_of_elements](auto& rhs) {
						rhs.clear();
						beacls::FloatVec().swap(rhs);
					}));
					datas.clear();
					std::deque<beacls::FloatVec >().swap(datas);
				}
			}
			else if (!keepLast) {
				std::vector<beacls::FloatVec> datas_vec(datas.size());
				std::copy(datas.cbegin(), datas.cend(), datas_vec.begin());
				if (!obstacles_s8_ptrs.empty())
					extraArgs.sdModFunctor->operator()(modified_schemeData, i, src_tau, datas_vec, obstacles_s8_ptrs, extraArgs.sdModParams);
				else
					extraArgs.sdModFunctor->operator()(modified_schemeData, i, src_tau, datas_vec, obstacles_ptrs, extraArgs.sdModParams);
			}
			else {
				std::cerr << "error : " << __func__ << "SchemeData mod fucntor needs to keep time dependent values." << std::endl;
				if (tmp_datas_variable) {
					beacls::closeMatVariable(tmp_datas_variable);
					tmp_datas_variable = NULL;
				}
				if (tmp_file_fs) {
					beacls::closeMatFStream(tmp_file_fs);
					tmp_file_fs = NULL;
				}
				return false;
			}
		}


		FLOAT_TYPE tNow = src_tau[i - 1];
		if (!obstacles_ptrs.empty()) {
			obstacle_i = (obstacles_ptrs.size() == 1) ? obstacles_ptrs[0] : obstacles_ptrs[i];
		}
		if (!obstacles_s8_ptrs.empty()) {
			obstacle_s8_i = (obstacles_s8_ptrs.size() == 1) ? obstacles_s8_ptrs[0] : obstacles_s8_ptrs[i];
		}

		// Main integration loop to ge to the next tau(i)
		while (tNow < (src_tau[i] - small)) {
			beacls::FloatVec yLast;
			if (minWith == HJIPDE::MinWithType_Zero) {
				yLast = y;
			}

			if (!quiet) {
				std::cout << std::fixed << std::setprecision(6) << "  Computing [" << tNow << " " << src_tau[i] << "]..." << std::resetiosflags(std::ios_base::floatfield) << std::endl;
			}

			beacls::FloatVec tspan{ tNow,src_tau[i] };
			tNow = integratorFunc->execute(
				y, tspan, y, modified_schemeData,
				execParameters.line_length_of_chunk, execParameters.num_of_threads, execParameters.num_of_gpus,
				execParameters.delayedDerivMinMax, execParameters.enable_user_defined_dynamics_on_gpu);

			if (std::any_of(y.cbegin(), y.cend(), [](const auto& rhs) { return std::isnan(rhs); })) {
				char num;
				std::cout << "Nan value found. Enter any key to continue." << std::endl;
				std::cin.get(num);
			}

			// Min with zero
			if (minWith == HJIPDE::MinWithType_Zero) {
				std::transform(y.cbegin(), y.cend(), yLast.cbegin(), y.begin(), std::ptr_fun<const FLOAT_TYPE&, const FLOAT_TYPE&>(std::min<FLOAT_TYPE>));
			}

			// Min with targets
			if (!targets_ptrs.empty()) {
				const beacls::FloatVec* target_i = (targets_ptrs.size() == 1) ? targets_ptrs[0] : targets_ptrs[i];
				std::transform(y.cbegin(), y.cend(), target_i->cbegin(), y.begin(), std::ptr_fun<const FLOAT_TYPE&, const FLOAT_TYPE&>(std::min<FLOAT_TYPE>));
			}
			if (!targets_s8_ptrs.empty()) {
				const std::vector<int8_t>* target_i = (targets_s8_ptrs.size() == 1) ? targets_s8_ptrs[0] : targets_s8_ptrs[i];
				std::transform(y.cbegin(), y.cend(), target_i->cbegin(), y.begin(), ([large](const auto& lhs, const auto& rhs) {
					return std::min<FLOAT_TYPE>(lhs, rhs * fix_point_ratio_inv);
				}));
			}

			// "Mask" using obstales
			if (obstacle_i) {
				if (obstacle_i->size() == y.size()) {
					std::transform(y.cbegin(), y.cend(), obstacle_i->cbegin(), y.begin(), ([](const auto& lhs, const auto& rhs) {
						return std::max<FLOAT_TYPE>(lhs, -rhs);
					}));
				}
			}
			if (obstacle_s8_i) {
				if (obstacle_s8_i->size() == y.size()) {
					std::transform(y.cbegin(), y.cend(), obstacle_s8_i->cbegin(), y.begin(), ([large](const auto& lhs, const auto& rhs) {
						return std::max<FLOAT_TYPE>(lhs, -rhs * fix_point_ratio_inv);
					}));
				}

			}
		}

		FLOAT_TYPE change = 0;
		if (stopConverge) {
			if (src_datas.size() > (i - 1)) {
				calcChange(change, y, src_datas[i - 1]);
			}
			else if (!tmp_filename.empty()) {
				load_vector(y, std::string(), Ns, false, tmp_file_fs, tmp_datas_variable, i - 1);
				beacls::FloatVec y0;
				calcChange(change, y, y0);
			}
			else if (!keepLast) {
				calcChange(change, y, datas[i - 1]);
			}
			else {
				calcChange(change, y, yLastTau);
			}
			if (!quiet) {
				std::cout << "Max change since last iteration : " << change << std::endl;
			}
		}

		// Reshape value function

		if (!tmp_filename.empty()) {
			save_vector(y, std::string(), Ns, false, tmp_file_fs, tmp_datas_variable, i);
		}
		else if (!keepLast) {
			if (flip_output) {
				datas.push_front(y);
			}
			else {
				datas.push_back(y);
			}
		}

		//!< Calculate TTR during solving
		if (extraArgs.execParameters.calcTTR) {
			if (calculatedTTR.size() != num_of_elements) calculatedTTR.resize(num_of_elements);
			const FLOAT_TYPE tau_i = src_tau[i];
			std::transform(calculatedTTR.cbegin(), calculatedTTR.cend(), y.cbegin(), calculatedTTR.begin(), ([tau_i](const auto& lhs, const auto& rhs) {
				if (rhs <= 0) return std::min<FLOAT_TYPE>(tau_i, lhs);
				else return lhs;
			}));
		}

		if (&dst_tau != &src_tau) dst_tau = src_tau;

		// If commanded, stop the reachable set computation once it contains the initial state.
		if (!extraArgs.stopInit.empty()) {
			beacls::FloatVec initValue;
			helperOC::eval_u(initValue, grid, y, std::vector<beacls::FloatVec>{extraArgs.stopInit});
			if (initValue[0] != std::numeric_limits<FLOAT_TYPE>::signaling_NaN()
				&& (initValue[0] <= 0)) {
				extraOuts.stoptau = src_tau[i];
				if (!low_memory && !keepLast) {
					datas.resize(i + 1);
				}
				dst_tau.resize(i + 1);
				break;
			}
		}
		if (!stopSet.empty()) {
			beacls::FloatVec temp = y;
			beacls::IntegerVec dataInds;
			for (size_t index = 0; index < temp.size(); ++index) {
				if (temp[index] <= stopLevel) dataInds.push_back(index);
			}

			bool stopSetResult = false;
			if (!extraArgs.stopSetInclude.empty()) {
				stopSetResult = std::all_of(setInds.cbegin(), setInds.cend(), [&dataInds](const auto& rhs) {
					return (std::find(dataInds.cbegin(), dataInds.cend(), rhs) != dataInds.cend());
				});
			}
			else {
				stopSetResult = std::any_of(setInds.cbegin(), setInds.cend(), [&dataInds](const auto& rhs) {
					return (std::find(dataInds.cbegin(), dataInds.cend(), rhs) != dataInds.cend());
				});
			}
			if (stopSetResult) {
				extraOuts.stoptau = src_tau[i];
				if (!low_memory && !keepLast) {
					datas.resize(i + 1);
				}
				dst_tau.resize(i + 1);
				break;
			}
		}

		if (stopConverge && (change < convergeThreshold)) {
			extraOuts.stoptau = src_tau[i];
			if (!low_memory && !keepLast) {
				datas.resize(i + 1);
			}
			dst_tau.resize(i + 1);
			break;
		}

		if (extraArgs.visualize) {
			//<! If commanded, visualize the level set.
#if defined(VISUALIZE_BY_OPENCV)
			beacls::FloatVec RS_level = extraArgs.RS_level;
			beacls::IntegerVec pDims;
			beacls::IntegerVec plotDims_inv(plotDims.size());
			for (size_t dimension = 0; dimension < plotDims.size(); ++dimension) {
				if (plotDims[dimension] != 0) pDims.push_back(dimension);
			}
			std::transform(plotDims.cbegin(), plotDims.cend(), plotDims_inv.begin(), [](const auto& rhs) { return (rhs == 0) ? 1 : 0; });
			const size_t projDims = projpt.size();
			// Basic Checks
			if ((plotDims.size() != schemeData->get_grid()->get_num_of_dimensions())
				//        || (projDims != (schemeData->get_grid()->get_num_of_dimensions() - pDims))
				) {
				std::cerr << "Mismatch between plot and grid dimensions!" << std::endl;
			}
			if (pDims.size() >= 4 || schemeData->get_grid()->get_num_of_dimensions() > 4) {
				std::cerr << "Mismatch between plot and grid dimensions!" << std::endl;
			}

			//!< Delete last plot
			if (deleteLastPlot) {
				HJIPDE_img = HJIPDE_initial_img;
			}

			beacls::FloatVec tmp_obstacle;
			if (obstacle_s8_i) {
				tmp_obstacle.resize(obstacle_s8_i->size());
				std::transform(obstacle_s8_i->cbegin(), obstacle_s8_i->cend(), tmp_obstacle.begin(), [large](const auto& rhs) {
					return rhs * fix_point_ratio_inv;
				});
			}
			const beacls::FloatVec* obstacle_ptr = obstacle_i ? obstacle_i : &tmp_obstacle;
			// Project
			const levelset::HJI_Grid* gPlot;
			beacls::FloatVec tmp_dataPlot;
			beacls::FloatVec tmp_obsPlot;
			if (projDims == 0) {
				gPlot = grid;
			}
			else {
				std::vector<helperOC::Projection_Type> projtypes;
				projtypes.resize(plotDims.size(), helperOC::Projection_Vector);
				gPlot = helperOC::proj(tmp_dataPlot, grid, y, plotDims_inv, projtypes, projpt);
				if (obsMode == HJIPDE::ObsModeType_TimeVarying) {
					if (!obstacle_ptr->empty()) {
						levelset::HJI_Grid* gObsPlot = helperOC::proj(tmp_obsPlot, grid, *obstacle_ptr, plotDims_inv, projtypes, projpt);
						if (gObsPlot) delete gObsPlot;
					}
				}
			}
			const beacls::FloatVec& dataPlot = (projDims == 0) ? y : tmp_dataPlot;
			const beacls::FloatVec& obsPlot = (projDims == 0) ? *obstacle_ptr : tmp_obsPlot;
			const cv::Size dsize = extraArgs.visualize_size.size() == 2 ? cv::Size((int)extraArgs.visualize_size[0], (int)extraArgs.visualize_size[1]) : cv::Size();
			const double fx = extraArgs.fx;
			const double fy = extraArgs.fy;
			helperOC::visSetIm(HJIPDE_img, HJIPDE_initial_img, gPlot, dataPlot, std::vector<float>{0, 0, 255}, RS_level, false, std::string(), dsize, fx, fy);
			if (!HJIPDE_img.empty()) {

				if (obsMode == HJIPDE::ObsModeType_TimeVarying) {
					helperOC::visSetIm(HJIPDE_img, HJIPDE_img, gPlot, obsPlot, std::vector<float>{0, 0, 0}, beacls::FloatVec{ 0 }, false, std::string(), dsize, fx, fy);
				}
				std::stringstream now_string;
				now_string << tNow;

				std::string title_string = std::string("t = ") + now_string.str();
				const double fontScale = 0.5;
				const int thickness = 1;
				int baseline = 0;

				cv::Size title_stringTextSize = cv::getTextSize(title_string, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
				cv::putText(HJIPDE_img, title_string, cv::Point(5, 5 + title_stringTextSize.height),
					cv::FONT_HERSHEY_SIMPLEX, fontScale,
					cv::Scalar{ 0,0,0 }, thickness, cv::LINE_AA);

#if defined(VISUALIZE_WITH_GUI)
				cv::imshow("HJIPDE", HJIPDE_img);
				cv::waitKey(1);
#endif
				if (!extraArgs.fig_filename.empty()) {
					std::stringstream i_ss;
					i_ss << i;
					std::string filename = extraArgs.fig_filename + i_ss.str() + ".png";
					cv::imwrite(filename, HJIPDE_img);
				}
			}

#else /* defined(VISUALIZE_BY_OPENCV) */
			std::cerr << "Warning: " << __func__ << " : visualize is not supported yet." << std::endl;
#endif  /* defined(VISUALIZE_BY_OPENCV) */
		}
		if (!extraArgs.save_filename.empty()) {
			if ((extraArgs.saveFrequency != 0) && ((i % extraArgs.saveFrequency) == 0)) {
				size_t ilast = i;
				//!< Load dst_datas from file.
				beacls::MatFStream* save_filename_fs = beacls::openMatFStream(extraArgs.save_filename, beacls::MatOpenMode_Write);
				if (!tmp_filename.empty()) {
					std::vector<beacls::FloatVec> datas_vec(datas.size());
					get_datas(datas_vec, src_tau, modified_schemeData);
					save_vector_of_vectors(datas_vec, std::string("data"), Ns, false, save_filename_fs);
					beacls::closeMatFStream(save_filename_fs);
					std::copy(datas_vec.cbegin(), datas_vec.cend(), datas.begin());
					//!< Put away dst_datas from file.
					for_each(datas.begin(), datas.end(), ([num_of_elements](auto& rhs) {
						rhs.clear();
						beacls::FloatVec().swap(rhs);
					}));
					datas.clear();
					std::deque<beacls::FloatVec >().swap(datas);
				}
				else if (!keepLast) {
					std::vector<beacls::FloatVec> datas_vec(datas.size());
					std::copy(datas.cbegin(), datas.cend(), datas_vec.begin());
					save_vector_of_vectors(datas_vec, std::string("data"), Ns, false,
						save_filename_fs);
				}
				else {
					save_vector(y, std::string("y"), Ns, false, save_filename_fs);
				}
				save_vector(dst_tau, std::string("tau"), beacls::IntegerVec(), true,
					save_filename_fs);
				save_value(static_cast<FLOAT_TYPE>(ilast), std::string("ilast"), true,
					save_filename_fs);
				beacls::closeMatFStream(save_filename_fs);
			}
		}
	}
	if (tmp_datas_variable) {
		beacls::closeMatVariable(tmp_datas_variable);
		tmp_datas_variable = NULL;
	}
	if (tmp_file_fs) {
		beacls::closeMatFStream(tmp_file_fs);
		tmp_file_fs = NULL;
	}
	y.swap(last_data);
	y.clear();
	//!< Load dst_datas from file.
	if (derivFunc) delete derivFunc;
	if (integratorFunc) delete integratorFunc;
	if (dissFunc) delete dissFunc;
	if (modified_schemeData) delete modified_schemeData;
	if (schemeFunc) delete schemeFunc;

	auto endTime = std::chrono::system_clock::now();
	if (!quiet) {
		auto diff = endTime - startTime;
		std::cout << "Total execution time "
			<< std::chrono::duration_cast<std::chrono::seconds>(diff).count()
			<< " seconds" << std::endl;
	}

	return true;
}
bool HJIPDE_impl::get_datas(std::vector<beacls::FloatVec >& dst_datas,
	const beacls::FloatVec& src_tau,
	const DynSysSchemeData* schemeData) const {
	//!< Load dst_datas from file.
	if (!tmp_filename.empty()) {
		const levelset::HJI_Grid* grid = schemeData->get_grid();
		const size_t num_of_elements = grid->get_sum_of_elems();
		dst_datas.resize(src_tau.size());
		beacls::IntegerVec dummy;
		beacls::MatFStream* tmp_file_fs =
			beacls::openMatFStream(tmp_filename, beacls::MatOpenMode_Read);
		if (tmp_file_fs) {
			beacls::MatVariable* tmp_datas_variable =
				beacls::openMatVariable(tmp_file_fs, std::string("tmp"));
			if (tmp_datas_variable) {
				for (size_t i = 0; i < src_tau.size(); ++i) {
					if (!load_vector(dst_datas[i], std::string(), dummy, false,
						tmp_file_fs, tmp_datas_variable, i)) {
						dst_datas[i].resize(num_of_elements, 0);
					}
				}
				beacls::closeMatVariable(tmp_datas_variable);
			}
			beacls::closeMatFStream(tmp_file_fs);
		}
		return true;
	}
	else {
		dst_datas.resize(datas.size());
		std::copy(datas.cbegin(), datas.cend(), dst_datas.begin());
	}
	return false;
}
bool HJIPDE_impl::TD2TTR(
	beacls::FloatVec& TTR,
	const levelset::HJI_Grid* g,
	const beacls::FloatVec& tau
) const {
	size_t num_of_elements = g->get_sum_of_elems();
	//!< Compute TTR
	const FLOAT_TYPE large = 1e6;
	if (!calculatedTTR.empty()) {
		TTR = calculatedTTR;
	}
	else if (tmp_filename.empty()) {
		//!< Input checking
		if (tau.size() != datas.size()) {
			std::cerr << "error : " << __func__ << "Grid dimensions must be one less than dimension of time-dependent value function!" << std::endl;
			return false;
		}
		if (TTR.size() != num_of_elements) TTR.resize(num_of_elements);
		std::transform(datas[0].cbegin(), datas[0].cend(), TTR.begin(), ([large](const auto& rhs) {
			if (rhs <= 0) return (FLOAT_TYPE)0.;
			else return large;
		}));

		for (size_t i = 1; i < tau.size(); ++i) {
			FLOAT_TYPE tau_i = tau[i];
			std::transform(TTR.cbegin(), TTR.cend(), datas[i].cbegin(), TTR.begin(), ([tau_i](const auto& lhs, const auto& rhs) {
				if (rhs <= 0) return std::min<FLOAT_TYPE>(tau_i, lhs);
				else return lhs;
			}));
		}
	}
	else {
		if (TTR.size() != num_of_elements) TTR.resize(num_of_elements);
		beacls::FloatVec data_i;
		beacls::MatFStream* tmp_file_fs = beacls::openMatFStream(tmp_filename, beacls::MatOpenMode_Read);
		if (tmp_file_fs) {
			beacls::MatVariable* tmp_datas_variable = beacls::openMatVariable(tmp_file_fs, std::string("tmp"));
			if (tmp_datas_variable) {
				beacls::IntegerVec dummy;
				load_vector(data_i, std::string(), dummy, false, tmp_file_fs, tmp_datas_variable, 0);
				std::transform(data_i.cbegin(), data_i.cend(), TTR.begin(), ([large](const auto& rhs) {
					if (rhs <= 0) return (FLOAT_TYPE)0.;
					else return large;
				}));
				for (size_t i = 1; i < tau.size(); ++i) {
					const FLOAT_TYPE tau_i = tau[i];
					load_vector(data_i, std::string(), dummy, false, tmp_file_fs, tmp_datas_variable, i);
					std::transform(TTR.cbegin(), TTR.cend(), data_i.cbegin(), TTR.begin(), ([tau_i](const auto& lhs, const auto& rhs) {
						if (rhs <= 0) return std::min<FLOAT_TYPE>(tau_i, lhs);
						else return lhs;
					}));
				}
				beacls::closeMatVariable(tmp_datas_variable);
			}
			beacls::closeMatFStream(tmp_file_fs);
		}
	}
	return true;
}


HJIPDE::HJIPDE(const std::string& tmp_filename) {
	pimpl = new HJIPDE_impl(tmp_filename);
}
HJIPDE::HJIPDE() {
	pimpl = new HJIPDE_impl(std::string());
}
HJIPDE::~HJIPDE() {
	if (pimpl) delete pimpl;

}
bool HJIPDE::solve(
	beacls::FloatVec& stoptau,
	helperOC::HJIPDE_extraOuts& extraOuts,
	const std::vector<beacls::FloatVec>& datas,
	const beacls::FloatVec& tau,
	const DynSysSchemeData* schemeData,
	const HJIPDE::MinWithType minWith,
	const helperOC::HJIPDE_extraArgs& extraArgs
) {
	if (pimpl) return pimpl->solve(stoptau, extraOuts, datas, tau, schemeData, minWith, extraArgs);
	return false;
}
bool HJIPDE::solve(
	beacls::FloatVec& stoptau,
	helperOC::HJIPDE_extraOuts& extraOuts,
	const beacls::FloatVec& data,
	const beacls::FloatVec& tau,
	const DynSysSchemeData* schemeData,
	const HJIPDE::MinWithType minWith,
	const helperOC::HJIPDE_extraArgs& extraArgs
) {
	if (pimpl) return pimpl->solve(stoptau, extraOuts, std::vector<beacls::FloatVec>{data}, tau, schemeData, minWith, extraArgs);
	return false;
}
bool HJIPDE::solve(
	std::vector<beacls::FloatVec >& dst_datas,
	beacls::FloatVec& stoptau,
	helperOC::HJIPDE_extraOuts& extraOuts,
	const std::vector<beacls::FloatVec>& datas,
	const beacls::FloatVec& tau,
	const DynSysSchemeData* schemeData,
	const HJIPDE::MinWithType minWith,
	const helperOC::HJIPDE_extraArgs& extraArgs
) {
	if (pimpl) {
		if (pimpl->solve(stoptau, extraOuts, datas, tau, schemeData, minWith, extraArgs)) {
			return get_datas(dst_datas, tau, schemeData);
		}
	}
	return false;
}
bool HJIPDE::solve(std::vector<beacls::FloatVec >& dst_datas,
	beacls::FloatVec& stoptau,
	helperOC::HJIPDE_extraOuts& extraOuts,
	const beacls::FloatVec& data,
	const beacls::FloatVec& tau,
	const DynSysSchemeData* schemeData,
	const HJIPDE::MinWithType minWith,
	const helperOC::HJIPDE_extraArgs& extraArgs) {
	if (pimpl) {
		if (pimpl->solve(stoptau, extraOuts, std::vector<beacls::FloatVec>{data}, tau, schemeData, minWith, extraArgs)) {
			return get_datas(dst_datas, tau, schemeData);
		}
	}
	return false;
}
bool HJIPDE::get_datas(
	std::vector<beacls::FloatVec >& dst_datas,
	const beacls::FloatVec& src_tau,
	const DynSysSchemeData* schemeData
) const {
	if (pimpl) return pimpl->get_datas(dst_datas, src_tau, schemeData);
	return false;

}
bool HJIPDE_impl::set_last_data(
	const beacls::FloatVec& src_data
) {
	last_data = src_data;
	return true;
}
bool HJIPDE_impl::get_last_data(
	beacls::FloatVec& dst_data
)const {
	dst_data = last_data;
	return true;
}
bool HJIPDE::get_last_data(
	beacls::FloatVec& dst_data
)const {
	if (pimpl) return pimpl->get_last_data(dst_data);
	return false;
}
bool HJIPDE::TD2TTR(
	beacls::FloatVec& TTR,
	const levelset::HJI_Grid* g,
	const beacls::FloatVec& tau
) const {
	if (pimpl) return pimpl->TD2TTR(TTR, g, tau);
	return false;

}
