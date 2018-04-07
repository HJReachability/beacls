#ifndef __visSetIm_hpp__
#define __visSetIm_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#include <helperOC/helperOC_type.hpp>
#include <typedef.hpp>
#include <cstddef>
#include <limits>

#if defined(VISUALIZE_BY_OPENCV)
#include <opencv2/opencv.hpp>
#if !defined(CV_VERSION_EPOCH) && (CV_VERSION_MAJOR == 3)	/* OpenCV 3.0 */
#define HELPEROC_OPNECV_3_X
#endif /* OpenCV 3.0 */
#if !defined(HELPEROC_OPNECV_3_X)
namespace cv {
	static const int LINE_AA = CV_AA;
}
#endif	/* HELPEROC_OPNECV_3_X */
#endif	/* defined(VISUALIZE_BY_OPENCV) */


class HJI_Grid;
namespace helperOC {
#if defined(VISUALIZE_BY_OPENCV)
	/**
	@brief	Code for quickly visualizing level sets
	@param	[out]	dst_imgs	vector of output images; these have the size dsize (when it is non-zero) or the size computed from src.size(), fx, and fy; the type of dst is the same as of src. 
	@param	[in]	src_img		input image. 
	@param	[in]	g			grid structure
	@param	[in]	data		value function corresponding to grid g
	@param	[in]	color		(defaults to red)
	@param	[in]	level		level set to display (defaults to 0)
	@param	[in]	deleteLastPlot	delete last plot, or not
	@param	[in]	fig_filename	filename to save
	@param	[in]	dsize		output image size; if it equals zero, it is computed with grid's axis range
	@param	[in]	fx			scale factor along the horizontal axis; when it equals 0, it is computed as (double)dsize.width/(grid axis range)
	@param	[in]	fy			scale factor along the vertical axis; when it equals 0, it is computed as (double)dsize.height/(grid axis range)
	@retval						figure handle
	*/
	PREFIX_VC_DLL
		bool visSetIm(
			std::vector<cv::Mat>& dst_imgs,
			const cv::Mat& src_img,
			const levelset::HJI_Grid* g,
			const beacls::FloatVec& data,
			const std::vector<float>& color = std::vector<float>{ 0,0,255 },
			const beacls::FloatVec& level = beacls::FloatVec(),
			const bool deleteLastPlot = true,
			const std::string& fig_filename = std::string(),
			const cv::Size dsize = cv::Size(),
			const double fx = 0.,
			const double fy = 0.
		);
	/**
	@brief	Code for quickly visualizing level sets
	@param	[out]	dst_img		output image; it has the size dsize (when it is non-zero) or the size computed from src.size(), fx, and fy; the type of dst is the same as of src. 
	@param	[in]	src_img		input image. 
	@param	[in]	g			grid structure
	@param	[in]	data		value function corresponding to grid g
	@param	[in]	color		(defaults to red)
	@param	[in]	level		level set to display (defaults to 0)
	@param	[in]	deleteLastPlot	delete last plot, or not
	@param	[in]	fig_filename	filename to save
	@param	[in]	dsize		output image size; if it equals zero, it is computed with grid's axis range
	@param	[in]	fx			scale factor along the horizontal axis; when it equals 0, it is computed as (double)dsize.width/(grid axis range)
	@param	[in]	fy			scale factor along the vertical axis; when it equals 0, it is computed as (double)dsize.height/(grid axis range)
	@retval						figure handle
	*/
	PREFIX_VC_DLL
		bool visSetIm(
			cv::Mat& dst_imgs,
			const cv::Mat& src_img,
			const levelset::HJI_Grid* g,
			const beacls::FloatVec& data,
			const std::vector<float>& color = std::vector<float>{ 0,0,1 },
			const beacls::FloatVec& level = beacls::FloatVec(),
			const bool deleteLastPlot = true,
			const std::string& fig_filename = std::string(),
			const cv::Size dsize = cv::Size(),
			const double fx = 0.,
			const double fy = 0.
		);
#endif	/* defined(VISUALIZE_BY_OPENCV) */

};
#endif	/* __visSetIm_hpp__ */

