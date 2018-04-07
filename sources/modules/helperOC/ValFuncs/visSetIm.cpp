#include <helperOC/helperOC_type.hpp>
#include <helperOC/ValFuncs/visSetIm.hpp>
//#include <helperOC/ValFuncs/augmentPeriodicData.hpp>
#include <levelset/BoundaryCondition/AddGhostPeriodic.hpp>
#include <Core/interpn.hpp>
#include <levelset/Grids/HJI_Grid.hpp>
#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <functional>
#include <typeinfo>
#include <sstream>
#include <macro.hpp>
#if defined(VISUALIZE_BY_OPENCV)

namespace helperOC {

	static
		bool visSetIm_single(
			cv::Mat& dst_img,
			const cv::Mat& src_img,
			const levelset::HJI_Grid* g,
			const beacls::FloatVec& data,
			const std::vector<float>& color,
			const beacls::FloatVec& level,
			const bool applyLight = true,
			const size_t sliceDim = std::numeric_limits<size_t>::max(),
			const cv::Size dsize = cv::Size(),
			const double fx = 0.,
			const double fy = 0.
		);

	static bool plot(
		cv::Mat& dst_img,
		const cv::Mat& src_img,
		const beacls::FloatVec& v0,
		const beacls::FloatVec& data,
		const std::vector<float>& color,
		const cv::Size dsize = cv::Size(),
		const double fx = 0.,
		const double fy = 0.
	);
	static bool contour(
		cv::Mat& dst_img,
		const cv::Mat& src_img,
		const beacls::FloatVec& v0,
		const beacls::FloatVec& v1,
		const beacls::FloatVec& data,
		const beacls::IntegerVec& Ns,
		const beacls::FloatVec& level = beacls::FloatVec(),
		const std::vector<float>& color = std::vector<float>{ 0,0,255 },
		const cv::Size dsize = cv::Size(),
		const double fx = 0.,
		const double fy = 0.
	);
};
static bool helperOC::plot(
	cv::Mat& dst_img,
	const cv::Mat& src_img,
	const beacls::FloatVec& v0,
	const beacls::FloatVec& data,
	const std::vector<float>& color,
	const cv::Size dsize,
	const double fx,
	const double fy
) {
	const int left_margin = 50;
	const int right_margin = 50;
	const int top_margin = 50;
	const int bottom_margin = 50;
	auto minMax = beacls::minmax_value<FLOAT_TYPE>(data.cbegin(), data.cend());
	const FLOAT_TYPE range = minMax.second - minMax.first;
	const FLOAT_TYPE min_value = minMax.first;
	cv::Mat tmp_img;
	const double org_width = (double)data.size();
	const double org_height = (double)std::ceil(range);
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
	int type = CV_8UC3;
	if (!src_img.empty() && (src_img.size() == size) && src_img.type() == type) {
		tmp_img = src_img.clone();
	}
	else {
		tmp_img = cv::Mat(size, type, cv::Scalar(255, 255, 255));
	}
	cv::Scalar color_s(color[0], color[1], color[2]);

	const int thickness = 1;
	std::vector<cv::Point> nodes(data.size());
	for (size_t x = 0; x < data.size(); ++x) {
		nodes[x] = cv::Point((int)x * actual_fx, (int)std::round((data[x] - min_value) * actual_fy));
	}
	cv::polylines(tmp_img, nodes, true, color_s, thickness, cv::LINE_AA);

	auto v0MinMax = beacls::minmax_value<FLOAT_TYPE>(v0.cbegin(), v0.cend());
	const FLOAT_TYPE v0_range = v0MinMax.second - v0MinMax.first;
	const int height = (int)std::round(v0_range);
	cv::resize(tmp_img, dst_img, cv::Size(tmp_img.size().width, height));
	cv::copyMakeBorder(tmp_img, tmp_img, top_margin, bottom_margin, left_margin, right_margin, cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
	dst_img.convertTo(dst_img, CV_8UC3, 255);
	return true;
}
static bool helperOC::contour(
	cv::Mat& dst_img,
	const cv::Mat& src_img,
	const beacls::FloatVec& v0,
	const beacls::FloatVec& v1,
	const beacls::FloatVec& data,
	const beacls::IntegerVec& Ns,
	const beacls::FloatVec& level,
	const std::vector<float>& color,
	const cv::Size dsize,
	const double fx,
	const double fy
) {
	const int left_margin = 25;
	const int right_margin = 25;
	const int top_margin = 25;
	const int bottom_margin = 25;

	cv::Mat data_img(data);
	data_img.convertTo(data_img, CV_32FC1);
	data_img = data_img.reshape(1, (int)Ns[1]);

	auto v0MinMax = beacls::minmax_value<FLOAT_TYPE>(v0.cbegin(), v0.cend());
	auto v1MinMax = beacls::minmax_value<FLOAT_TYPE>(v1.cbegin(), v1.cend());
	const FLOAT_TYPE v0_range = v0MinMax.second - v0MinMax.first;
	const FLOAT_TYPE v1_range = v1MinMax.second - v1MinMax.first;
	const int org_width = (int)std::ceil(v0_range);
	const int org_height = (int)std::ceil(v1_range);
	const FLOAT_TYPE left_offset = v0MinMax.first;
	const FLOAT_TYPE top_offset = v1MinMax.first;
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


	cv::Mat tmp_img;
	cv::Size margined_size(width + left_margin + right_margin, height + top_margin + bottom_margin);
	int type = CV_8UC3;
	if (!src_img.empty() && (src_img.size() == margined_size) && src_img.type() == type) {
		tmp_img = src_img.clone();
	}
	else {
		tmp_img = cv::Mat(margined_size, type, cv::Scalar(255, 255, 255));
	}
	cv::Mat scaled_data_img;
	cv::resize(data_img, scaled_data_img, size);

	const int thickness = 1;
	cv::Scalar color_s(color[0], color[1], color[2]);
	cv::Rect roi_rect(left_margin, top_margin, scaled_data_img.size().width, scaled_data_img.size().height);
	std::for_each(level.cbegin(), level.cend(), [&scaled_data_img, &tmp_img, &thickness, &color_s, &roi_rect, &left_offset, &top_offset, &actual_fx, &actual_fy](const auto& rhs) {
		std::vector<std::vector<cv::Point>> contours;
		cv::Mat lowprecision_img;
		cv::threshold(scaled_data_img, lowprecision_img, (double)rhs, 255, cv::THRESH_BINARY);
		lowprecision_img.convertTo(lowprecision_img, CV_8UC1);
		cv::findContours(lowprecision_img, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
		std::for_each(contours.cbegin(), contours.cend(), [&color_s, &tmp_img, &thickness, &roi_rect, &left_offset, &top_offset, &actual_fx, &actual_fy](const auto contour) {
			std::vector<cv::Point> offseted_contour(contour.size());
			std::transform(contour.cbegin(), contour.cend(), offseted_contour.begin(), [&left_offset, &top_offset, &actual_fx, &actual_fy](const auto& rhs) {
				return cv::Point(rhs.x, rhs.y);
			});
			cv::polylines(tmp_img(roi_rect), offseted_contour, true, color_s, thickness, cv::LINE_AA);
		});
	});

	dst_img = tmp_img;
	return true;
}
static
bool helperOC::visSetIm_single(
	cv::Mat& dst_img,
	const cv::Mat& src_img,
	const levelset::HJI_Grid* g,
	const beacls::FloatVec& data,
	const std::vector<float>& color,
	const beacls::FloatVec& level,
	const bool applyLight,
	const size_t sliceDim,
	const cv::Size dsize,
	const double fx,
	const double fy
) {
	//!<  Slice last dimension by default
	const size_t gDim = g->get_num_of_dimensions();
	size_t modifiedSliceDim = (sliceDim != std::numeric_limits<size_t>::max()) ? sliceDim : gDim - 1;
	cv::Mat fliped_src;
	if (!src_img.empty()) {
		cv::flip(src_img, fliped_src, 0);
	}

	switch (gDim) {
	case 1:
		plot(dst_img, fliped_src, g->get_vs(0), data, color, dsize, fx, fy);
		break;
	case 2:
		if (level.empty()) {
			contour(dst_img, fliped_src, g->get_vs(0), g->get_vs(1), data, g->get_Ns(), beacls::FloatVec{(FLOAT_TYPE)0}, color, dsize, fx, fy);
		}
		else {
			contour(dst_img, fliped_src, g->get_vs(0), g->get_vs(1), data, g->get_Ns(), level, color, dsize, fx, fy);
		}
		break;
	case 3:
		break;
	case 4:
		break;
	default:
		break;

	}
	cv::flip(dst_img, dst_img, 0);
	return true;
}

bool  helperOC::visSetIm(
	std::vector<cv::Mat>& dst_imgs,
	const cv::Mat& src_img,
	const levelset::HJI_Grid* g,
	const beacls::FloatVec& data,
	const std::vector<float>& color,
	const beacls::FloatVec& level,
	const bool deleteLastPlot,
	const std::string& fig_filename,
	const cv::Size dsize,
	const double fx,
	const double fy
) {
	//!< Default parameters and input check
	if (!g) {
		return false;
	}
	bool save_png = false;
	if (!fig_filename.empty()) {
		save_png = true;
	}

	const size_t g_sum_of_elements = g->get_sum_of_elems();
	if (g_sum_of_elements == data.size()) {
		//!< Visualize a single set
		dst_imgs.resize(1);
		visSetIm_single(dst_imgs[0], src_img, g, data, color, level, deleteLastPlot, std::numeric_limits<size_t>::max(), dsize, fx, fy);
		if (save_png) {
			cv::imwrite(fig_filename, dst_imgs[0]);
		}
	}
	else {
		size_t numSets = data.size() / g_sum_of_elements;
		if (!deleteLastPlot) dst_imgs.resize(numSets);
		for (size_t i = 0; i < numSets; ++i) {
			bool applyLight = true;
			if (i > 0) {
				applyLight = false;
			}
			if (deleteLastPlot) {
				dst_imgs.resize(1);
				beacls::FloatVec data_i(data.cbegin() + g_sum_of_elements*i, data.cbegin() + g_sum_of_elements*(i + 1));
				dst_imgs[0] = cv::Mat(src_img.size(), src_img.type(), cv::Scalar(255, 255, 255));
				visSetIm_single(dst_imgs[0], dst_imgs[0], g, data_i, color, level, deleteLastPlot, applyLight, dsize, fx, fy);
			}
			else {
				beacls::FloatVec data_i(data.cbegin() + g_sum_of_elements*i, data.cbegin() + g_sum_of_elements*(i + 1));
				visSetIm_single(dst_imgs[i], src_img, g, data_i, color, level, deleteLastPlot, applyLight, dsize, fx, fy);
			}
#if defined(VISUALIZE_WITH_GUI)
			cv::imshow(fig_filename, dst_imgs[i]);
			cv::waitKey(1);
#endif
			if (save_png) {
				std::stringstream i_ss;
				i_ss << i;
				std::string filename = fig_filename + "_" + i_ss.str() + ".png";
				cv::imwrite(filename, dst_imgs[i]);
			}
		}
	}

	return true;
}
bool  helperOC::visSetIm(
	cv::Mat& dst_img,
	const cv::Mat& src_img,
	const levelset::HJI_Grid* g,
	const beacls::FloatVec& data,
	const std::vector<float>& color,
	const beacls::FloatVec& level,
	const bool deleteLastPlot,
	const std::string& fig_filename,
	const cv::Size dsize,
	const double fx,
	const double fy
) {
	//!< Default parameters and input check
	if (!g) {
		return false;
	}
	bool save_png = false;
	if (!fig_filename.empty()) {
		save_png = true;
	}
	const size_t g_sum_of_elements = g->get_sum_of_elems();
	if (g_sum_of_elements == data.size()) {
		//!< Visualize a single set
		if (deleteLastPlot) {
			dst_img = cv::Mat(src_img.size(), src_img.type(), cv::Scalar(255, 255, 255));
			visSetIm_single(dst_img, dst_img, g, data, color, level, deleteLastPlot, std::numeric_limits<size_t>::max(), dsize, fx, fy);
		}
		else {
			visSetIm_single(dst_img, src_img, g, data, color, level, deleteLastPlot, std::numeric_limits<size_t>::max(), dsize, fx, fy);
		}
		if (save_png) {
			cv::imwrite(fig_filename, dst_img);
		}
	}
	else {
		size_t numSets = data.size() / g_sum_of_elements;
		for (size_t i = 0; i < numSets; ++i) {
			bool applyLight = true;
			if (i > 0) {
				applyLight = false;
			}
			if (deleteLastPlot) {
				dst_img = cv::Mat(src_img.size(), src_img.type(), cv::Scalar(255, 255, 255));
				beacls::FloatVec data_i(data.cbegin() + g_sum_of_elements*i, data.cbegin() + g_sum_of_elements*(i + 1));
				visSetIm_single(dst_img, dst_img, g, data_i, color, level, deleteLastPlot, applyLight, dsize, fx, fy);
			}
			else {
				beacls::FloatVec data_i(data.cbegin() + g_sum_of_elements*i, data.cbegin() + g_sum_of_elements*(i + 1));
				visSetIm_single(dst_img, src_img, g, data_i, color, level, deleteLastPlot, applyLight, dsize, fx, fy);
			}
#if defined(VISUALIZE_WITH_GUI)
			cv::imshow(fig_filename, dst_img);
			cv::waitKey(1);
#endif
			if (save_png) {
				std::stringstream i_ss;
				i_ss << i;
				std::string filename = fig_filename + "_" + i_ss.str() + ".png";
				cv::imwrite(filename, dst_img);
			}
		}
	}

	return true;
}

#endif	/* defined(VISUALIZE_BY_OPENCV) */
