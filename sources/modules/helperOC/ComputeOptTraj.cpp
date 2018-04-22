#include <helperOC/ComputeOptTraj.hpp>
#include <helperOC/ComputeGradients.hpp>
#include <iostream>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <numeric>
#include <limits>
#include <macro.hpp>
#include <Core/UVec.hpp>
#include <helperOC/DynSys/DynSys/DynSys.hpp>
#include <levelset/Grids/HJI_Grid.hpp>
#include <helperOC/ValFuncs/eval_u.hpp>
#include <helperOC//ValFuncs/proj.hpp>
#include <helperOC/ValFuncs/visSetIm.hpp>
#include <helperOC/ValFuncs/find_earliest_BRS_ind.hpp>
#include "ComputeOptTraj_impl.hpp"
#if defined(FILESYSTEM)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#if defined(VISUALIZE_BY_OPENCV)
#include <opencv2/opencv.hpp>
#endif	/* defined(VISUALIZE_BY_OPENCV) */

helperOC::ComputeOptTraj::ComputeOptTraj() :
	pimpl(new ComputeOptTraj_impl())
{
}

helperOC::ComputeOptTraj::~ComputeOptTraj()
{
	if (pimpl) delete pimpl;
}

helperOC::ComputeOptTraj_impl::ComputeOptTraj_impl() :
	computeGradients(NULL)
{
}

helperOC::ComputeOptTraj_impl::~ComputeOptTraj_impl()
{
	if (computeGradients) delete computeGradients;
}

bool helperOC::ComputeOptTraj_impl::operator()(
	std::vector<beacls::FloatVec >& traj,
	beacls::FloatVec& traj_tau,
	const levelset::HJI_Grid* grid,
	const std::vector<beacls::FloatVec >& data,
	const beacls::FloatVec& tau,
	DynSys* dynSys,
	const HJIPDE_extraArgs& extraArgs,
	const helperOC::DynSys_UMode_Type uMode,
	const size_t subSamples
	) {
	const bool visualize = extraArgs.visualize;

	const helperOC::ExecParameters execParameters = extraArgs.execParameters;

	//!< Visualization
#if defined(VISUALIZE_BY_OPENCV)
	cv::Mat BRSplot;
#if defined(VISUALIZE_WITH_GUI)
	if (visualize)
		cv::namedWindow("BRS1", 0);
#endif
#endif	/* defined(VISUALIZE_BY_OPENCV) */
	beacls::IntegerVec showDims;
	beacls::IntegerVec hideDims;
	if (visualize) {
		for (size_t dimension = 0; dimension < extraArgs.projDim.size(); ++dimension) {
			if (extraArgs.projDim[dimension] != 0) showDims.push_back(dimension);
		}
		hideDims.resize(extraArgs.projDim.size());
		std::transform(extraArgs.projDim.cbegin(), extraArgs.projDim.cend(), hideDims.begin(), [](const auto& rhs) { return (rhs == 0) ? 1 : 0; });
	}


	const beacls::UVecType execType = (execParameters.useCuda) ? beacls::UVecType_Cuda : beacls::UVecType_Vector;
	if (std::adjacent_find(tau.cbegin(), tau.cend(), [](const auto& lhs, const auto&rhs) { return lhs >= rhs; }) != tau.cend()) {
		std::cerr << "Time stamps must be in ascending order!" << std::endl;
		return false;
	}
	//!< Time parameters
	size_t iter = 0;
	const size_t tauLength = tau.size();
	if (tauLength < 2) return false;
	FLOAT_TYPE dtSmall = (tau[1] - tau[0]) / subSamples;
	//!< maxIter = 1.25*tauLength;

	//!< Initialize trajectory;
	traj.resize(tauLength);
	std::for_each(traj.begin(), traj.end(), [](auto& rhs) { rhs.resize(3, std::numeric_limits<FLOAT_TYPE>::signaling_NaN()); });
	traj[0] = dynSys->get_x();
	size_t tEarliest = 0;
	std::vector<beacls::FloatVec> xs;
	if (computeGradients && (computeGradients->get_type() != execType)) {
		delete computeGradients;
		computeGradients = NULL;
	}
	if (!computeGradients) computeGradients = new ComputeGradients(grid, helperOC::ApproximationAccuracy_veryHigh, execType);
	std::vector<beacls::FloatVec> derivC;
	std::vector<beacls::FloatVec> derivL;
	std::vector<beacls::FloatVec> derivR;
	std::vector<beacls::FloatVec> deriv;
	std::vector<beacls::FloatVec> u(dynSys->get_nu());
	beacls::IntegerVec u_sizes(dynSys->get_nu());
	beacls::FloatVec x1;
	std::vector<beacls::FloatVec::const_iterator> x_ites;
	beacls::IntegerVec x_sizes;
	std::vector<const FLOAT_TYPE*> derivs;
	beacls::IntegerVec deriv_sizes;
	beacls::FloatVec x;
	while (iter < tauLength) {
		size_t upper = tauLength - 1;
		size_t lower = tEarliest;

		tEarliest = find_earliest_BRS_ind(grid, data, std::vector<beacls::FloatVec>{dynSys->get_x()}, upper, lower);

		//!< BRS at current time
		const beacls::FloatVec& BRS_at_t = data[tEarliest];

		//!< Visualize BRS corresponding to current trajectory point
		if (visualize) {
			const cv::Size dsize = extraArgs.visualize_size.size() == 2 ? cv::Size((int)extraArgs.visualize_size[0], (int)extraArgs.visualize_size[1]) : cv::Size();
			const double fx = extraArgs.fx;
			const double fy = extraArgs.fy;
			beacls::FloatVec data2D;
			beacls::FloatVec trajHide_at_t;
			std::vector<helperOC::Projection_Type> x_types;
			for (size_t dimension = 0; dimension < traj[iter].size(); ++dimension) {
				if (hideDims[dimension]) {
					trajHide_at_t.push_back(traj[iter][dimension]);
					x_types.push_back(helperOC::Projection_Vector);
				}
			}
			levelset::HJI_Grid* g2D = proj(data2D, grid, BRS_at_t, hideDims, x_types, trajHide_at_t);
#if defined(VISUALIZE_BY_OPENCV)
			const int left_margin = 25;
			const int right_margin = 25;
			const int top_margin = 25;
			const int bottom_margin = 25;
			const int thickness = 1;
			const beacls::FloatVec& vs0 = grid->get_vs(0);
			const beacls::FloatVec& vs1 = grid->get_vs(1);
			const auto x0MinMax = beacls::minmax_value<FLOAT_TYPE>(vs0.cbegin(), vs0.cend());
			const auto x1MinMax = beacls::minmax_value<FLOAT_TYPE>(vs1.cbegin(), vs1.cend());
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
			const FLOAT_TYPE left_offset = x0MinMax.first;
			const FLOAT_TYPE top_offset = x1MinMax.first;

			visSetIm(BRSplot, BRSplot, g2D, data2D, std::vector<float>{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, beacls::FloatVec(), true, std::string(), dsize, fx, fy);
			if (!BRSplot.empty()) {
				std::vector<cv::Point> traj_points(iter + 1);
				std::transform(traj.cbegin(), traj.cbegin() + iter + 1, traj_points.begin(), [&showDims, &left_offset, &top_offset, &height, &actual_fx, &actual_fy](const auto& rhs) {
					return cv::Point((int32_t)(rhs[showDims[0]] * actual_fx - left_offset), (int32_t)(height - rhs[showDims[1]] * actual_fy - top_offset));
				});
				cv::Rect roi_rect(left_margin, top_margin, BRSplot.size().width - left_margin - right_margin, BRSplot.size().height - top_margin - bottom_margin);

				cv::polylines(BRSplot(roi_rect), traj_points, false, cv::Scalar(0, 0, 0), thickness, cv::LINE_AA);


				std::stringstream tau_ss, tau_tEarliest_ss;
				tau_ss << tau[iter];
				tau_tEarliest_ss << tau[tEarliest];
				std::string tStr = std::string("t = ") + tau_ss.str();
				std::string tEarliestStr = std::string("tEarliest = ") + tau_tEarliest_ss.str();
				const double fontScale = 0.5;
				int baseline = 0;
				cv::Size tStrTextSize = cv::getTextSize(tStr, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);
				cv::Size tEarliestStrTextSize = cv::getTextSize(tEarliestStr, cv::FONT_HERSHEY_SIMPLEX, fontScale, thickness, &baseline);

				cv::putText(BRSplot, tStr, cv::Point(5, 5 + tStrTextSize.height),
					cv::FONT_HERSHEY_SIMPLEX, fontScale,
					cv::Scalar{ 0,0,0 }, thickness, cv::LINE_AA);
				cv::putText(BRSplot, tEarliestStr, cv::Point(5, 5 + tStrTextSize.height + 5 + tEarliestStrTextSize.height),
					cv::FONT_HERSHEY_SIMPLEX, fontScale,
					cv::Scalar{ 0,0,0 }, thickness, cv::LINE_AA);

#if defined(VISUALIZE_WITH_GUI)
				cv::imshow("BRS1", BRSplot);
				cv::waitKey(1);
#endif
				if (!extraArgs.fig_filename.empty()) {
					std::stringstream i_ss;
					i_ss << iter;
					std::string filename = extraArgs.fig_filename + i_ss.str() + ".png";
					cv::imwrite(filename, BRSplot);
				}
			}
#endif	/* defined(VISUALIZE_BY_OPENCV) */
			if (g2D) delete g2D;
		}

		if (tEarliest == (tauLength - 1)) {
			//!< Trajectory has entered the target
			break;
		}
		//!<  Update trajectory
		if (!BRS_at_t.empty()) {
			computeGradients->operator()(derivC, derivL, derivR, grid, BRS_at_t, BRS_at_t.size(), false, execParameters);
		}
		else {
			derivC.clear();
			derivL.clear();
			derivR.clear();
		}
		for (size_t j = 0; j < subSamples; ++j) {
			x = dynSys->get_x();
			eval_u(deriv, grid, derivC, x);
			std::for_each(u.begin(), u.end(), [](auto& rhs) { rhs.resize(1); });
			if (xs.size() != x.size()) xs.resize(x.size());
			if (x_ites.size() != x.size()) x_ites.resize(x.size());
			if (x_sizes.size() != x.size()) x_sizes.resize(x.size());
			for (size_t dimension = 0; dimension < x.size(); ++dimension) {
				xs[dimension].resize(1);
				xs[dimension][0] = x[dimension];
				x_ites[dimension] = xs[dimension].cbegin();
				x_sizes[dimension] = 1;
			}
			if (derivs.size() != deriv.size()) derivs.resize(deriv.size());
			for (size_t dimension = 0; dimension < deriv.size(); ++dimension) {
				derivs[dimension] = deriv[dimension].data();
			};
			if (deriv_sizes.size() != derivs.size()) deriv_sizes.resize(derivs.size());
			std::fill(deriv_sizes.begin(), deriv_sizes.end(), 1);

			dynSys->optCtrl(u, tau[tEarliest], x_ites, derivs, x_sizes, deriv_sizes, uMode);
			dynSys->updateState(x1, u, dtSmall, x);
		}

		//!< Record new point on nominal trajectory
		++iter;
		if (iter < traj.size()) traj[iter] = dynSys->get_x();
	}
	//!< Delete unused indices
	const size_t traj_size = iter < tau.size() ? iter : tau.size();
	traj.resize(traj_size);
	traj_tau.resize(traj_size);
	std::copy(tau.cbegin(), tau.cbegin() + traj_size, traj_tau.begin());
	return true;
}

bool helperOC::ComputeOptTraj::operator()(
	std::vector<beacls::FloatVec >& traj,
	beacls::FloatVec& traj_tau,
	const levelset::HJI_Grid* grid,
	const std::vector<beacls::FloatVec >& data,
	const beacls::FloatVec& tau,
	DynSys* dynSys,
	const HJIPDE_extraArgs& extraArgs,
	const helperOC::DynSys_UMode_Type uMode,
	const size_t subSamples
	) {
	if (pimpl) pimpl->operator()(traj, traj_tau, grid, data, tau, dynSys, extraArgs, uMode, subSamples);
	return false;
}