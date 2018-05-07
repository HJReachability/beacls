#ifndef __helperOC_type_hpp__
#define __helperOC_type_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

//#define VISUALIZE_BY_OPENCV
//#define VISUALIZE_WITH_GUI
//#define FILESYSTEM


#include <vector>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <utility>
using namespace std::rel_ops;

#include <typedef.hpp>
#include <Core/UVec.hpp>
namespace levelset {
	class SchemeData;
};
namespace helperOC {
	static const size_t fix_point_depth = 2;
	static const FLOAT_TYPE fix_point_ratio = (FLOAT_TYPE)(1 << fix_point_depth);
	static const FLOAT_TYPE fix_point_ratio_inv = (FLOAT_TYPE)(1. / fix_point_ratio);

	class DynSysSchemeData;

	typedef enum LineStyleType {
		LineStyle_invalid,
		LineStyle_none,

	}LineStyleType;
	class PlotExtraArgs {
	public:
		beacls::FloatVec colors;
		FLOAT_TYPE MarkerSize;
		FLOAT_TYPE arrowLength;
		LineStyleType LineStyle;
		FLOAT_TYPE LineWidth;
		PlotExtraArgs() :
			colors(beacls::FloatVec{ (FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0 }),
			MarkerSize((FLOAT_TYPE)20),
			arrowLength((FLOAT_TYPE)10),
			LineStyle(LineStyle_none),
			LineWidth((FLOAT_TYPE)0.5)
		{
		}
	};
	typedef enum Projection_Type {
		Projection_Invalid = -1,
		Projection_Min,
		Projection_Max,
		Projection_Vector,
	} Projection_Type;
	typedef enum Dissipation_Type {
		Dissipation_Invalid = -1,
		Dissipation_global,
		Dissipation_local,
		Dissipation_locallocal
	} Dissipation_Type;
	typedef enum ApproximationAccuracy_Type {
		ApproximationAccuracy_Invalid = -1,
		ApproximationAccuracy_low,
		ApproximationAccuracy_medium,
		ApproximationAccuracy_high,
		ApproximationAccuracy_veryHigh,
	} ApproximationAccuracy_Type;
	typedef enum DynSysMode_Type {
		DynSysMode_Invalid = -1,
		DynSysMode_Free,
		DynSysMode_Follower,
		DynSysMode_Leader,
		DynSysMode_Faulty,
	} DynSysMode_Type;
	typedef enum DynSysStatus_Type {
		DynSysStatus_Invalid = -1,
		DynSysStatus_idle,
		DynSysStatus_busy,
	} DynSysStatus_Type;
	typedef bool(*PartialFunction_cuda)(
		beacls::UVec&,
		const levelset::SchemeData*,
		const FLOAT_TYPE,
		const beacls::UVec&,
		const std::vector<beacls::UVec>&,
		const std::vector<beacls::UVec>&,
		const size_t,
		const size_t,
		const size_t
		);
	class ExecParameters {
	public:
		size_t line_length_of_chunk;	//!< Line length of each parallel execution chunks  (0 means set automatically)
		size_t num_of_threads;	//!< Number of CPU Threads (0 means use all logical threads of CPU)
		size_t num_of_gpus;	//!<	Number of GPUs which (0 means use all GPUs)
		bool calcTTR; //!< calculate TTR during solving
		levelset::DelayedDerivMinMax_Type delayedDerivMinMax;	//!< Use last step's min/max of derivatives, and skip 2nd pass.
		bool useCuda;//!< Execute type CPU Vector or GPU.
		bool enable_user_defined_dynamics_on_gpu; //!< Flag for user defined dynamics function on gpu
		ExecParameters() :
			line_length_of_chunk(0),
			num_of_threads(0),
			num_of_gpus(0),
			calcTTR(false),
			delayedDerivMinMax(levelset::DelayedDerivMinMax_Disable),
			useCuda(false),
			enable_user_defined_dynamics_on_gpu(true)
		{}
		PREFIX_VC_DLL
			bool operator==(const ExecParameters& rhs) const;
	};
	class SDModParams;

	class SDModFunctor {
	public:
		virtual bool operator()(
			DynSysSchemeData* schemeData,
			const size_t i,
			const beacls::FloatVec& tau,
			const std::vector<beacls::FloatVec >& data,
			const std::vector<const beacls::FloatVec* > obstacles_ptrs,
			const SDModParams* params
			) const = 0;
		virtual bool operator()(
			DynSysSchemeData* schemeData,
			const size_t i,
			const beacls::FloatVec& tau,
			const std::vector<beacls::FloatVec >& data,
			const std::vector<const std::vector<int8_t>* > obstacles_s8_ptrs,
			const SDModParams* params
			) const = 0;
	};

	class PlotData {
	public:
		beacls::IntegerVec plotDims;
		beacls::FloatVec projpt;
	public:
		PlotData(
			const beacls::IntegerVec& plotDims,
			const beacls::FloatVec& projpt
		) : plotDims(plotDims), projpt(projpt) {}
	};
	class HJIPDE_extraArgs {
	public:
		std::vector<beacls::FloatVec> obstacles;	//!< a single obstacle or a list of obstacles_ptrs with time stamps tau(obstacles_ptrs must have same time stamp as the solution)
		std::vector<std::vector<int8_t>> obstacles_s8;	//!< a single obstacle or a list of obstacles_ptrs in binary format with time stamps tau(obstacles_ptrs must have same time stamp as the solution)
		std::vector<const beacls::FloatVec* > obstacles_ptrs;	//!< pointers for a single obstacle or a list of obstacles_ptrs with time stamps tau(obstacles_ptrs must have same time stamp as the solution)
		std::vector<const std::vector<int8_t>* > obstacles_s8_ptrs;	//!< pointers for a single obstacle or a list of obstacles_ptrs in binary format with time stamps tau(obstacles_ptrs must have same time stamp as the solution)
		size_t compRegion;	//!< unused for now(meant to limit computation region)
		bool visualize;	//!< set to true to visualize reachable set
		beacls::IntegerVec visualize_size;	//!< visualize image size; if it equals zero, it is computed with grid's axis range
		double fx;	//!< scale factor along the horizontal axis; when it equals 0, it is computed as (double)dsize.width/(grid axis range)
		double fy;	//!< scale factor along the vertical axis; when it equals 0, it is computed as (double)dsize.height/(grid axis range)
		beacls::FloatVec RS_level;	//!<level set of reachable set to visualize (defaults to 0)
		size_t istart;
		PlotData plotData;	//!< information required to plot the data(need to fill in)
		bool deleteLastPlot; //!< set to true to delete previous plot before displaying next one
		size_t fig_num; //!< List if you want to plot on a specific figure number
		std::string fig_filename;//!< provide this to save the figures(requires export_fig package)
		beacls::FloatVec stopInit;	//!< stop the computation once the reachable set includes the initial state
		beacls::FloatVec stopSetInclude;	//!< stops computation when reachable set includes this set
		beacls::FloatVec stopSetIntersect;	//!< stops computation when reachable set intersects this set
		size_t stopLevel;	//!< level of the stopSet to check the inclusion for.Default level is zero.
		std::vector<beacls::FloatVec > targets;	//!< a single target or a list of targets with time stamps tau(targets must have same time stamp as the solution).This functionality is mainly useful when the targets are time - varying, in case of variational inequality for example; data0 can be used to specify the target otherwise.
		std::vector<std::vector<int8_t> > targets_s8;	//!< a single target or a list of targets with time stamps tau(targets must have same time stamp as the solution).This functionality is mainly useful when the targets are time - varying, in case of variational inequality for example; data0 can be used to specify the target otherwise.
		std::vector<const beacls::FloatVec* > targets_ptrs;	//!< pointers for a single target or a list of targets with time stamps tau(targets must have same time stamp as the solution).This functionality is mainly useful when the targets are time - varying, in case of variational inequality for example; data0 can be used to specify the target otherwise.
		std::vector<const std::vector<int8_t>* > targets_s8_ptrs;	//!< pointers for a single target or a list of targets with time stamps tau(targets must have same time stamp as the solution).This functionality is mainly useful when the targets are time - varying, in case of variational inequality for example; data0 can be used to specify the target otherwise.
		bool stopConverge;	//!< set to true to stop the computation when it converges
		FLOAT_TYPE convergeThreshold;	//!< Max change in each iteration allowed when checking convergence
		SDModFunctor* sdModFunctor;	//!< Function for modifying scheme data every time step given by tau. 		Currently this is only used to switch between using optimal control at every grid point and using maximal control for the SPP project when computing FRS using centralized controller
		SDModParams* sdModParams;	//!< Parameters for modifying scheme data every time step given by tau. 		Currently this is only used to switch between using optimal control at every grid point and using maximal control for the SPP project when computing FRS using centralized controller
		std::string save_filename;	//!< file name under which temporary data is saved at some frequency in terms of the number of time steps
		size_t saveFrequency;	//!< file name under which temporary data is saved at some frequency in terms of the number of time steps
		bool keepLast;	//!< Only keep data from latest time stamp and delete previous datas
		beacls::IntegerVec projDim;	//!< set the dimensions that should be projected away when visualizing
		bool applyLight;
		bool quiet;
		bool low_memory;
		bool flip_output;
		ExecParameters execParameters;
	private:
	public:
		HJIPDE_extraArgs() :
			obstacles(std::vector<beacls::FloatVec >()),
			obstacles_s8(std::vector<std::vector<int8_t> >()),
			obstacles_ptrs(std::vector<const beacls::FloatVec* >()),
			obstacles_s8_ptrs(std::vector<const std::vector<int8_t>*>()),
			compRegion(0),
			visualize(false),
			visualize_size(beacls::IntegerVec{ 0, 0 }),
			fx(0.),
			fy(0.),
			RS_level(beacls::FloatVec()),
			istart(1),
			plotData(PlotData(beacls::IntegerVec(), beacls::FloatVec())),
			deleteLastPlot(false),
			fig_num(0),
			fig_filename(std::string()),
			stopInit(beacls::FloatVec()),
			stopSetInclude(beacls::FloatVec()),
			stopSetIntersect(beacls::FloatVec()),
			stopLevel(0),
			targets(std::vector<beacls::FloatVec >()),
			targets_s8(std::vector<std::vector<int8_t> >()),
			targets_ptrs(std::vector<const beacls::FloatVec* >()),
			targets_s8_ptrs(std::vector<const std::vector<int8_t>* >()),
			stopConverge(false),
			convergeThreshold((FLOAT_TYPE)1e-5),
			sdModFunctor(NULL),
			sdModParams(NULL),
			save_filename(std::string()),
			saveFrequency(1),
			keepLast(false),
			projDim(beacls::IntegerVec()),
			applyLight(true),
			quiet(false),
			low_memory(false),
			flip_output(false),
			execParameters(ExecParameters())
		{};
	private:

	};
	/*
	@brief This structure can be used to pass on extra outputs.
	*/
	class HJIPDE_extraOuts {
	public:
		FLOAT_TYPE stoptau;	//!< time at which the reachable set contains the initial state; tau and data vectors only contain the data till stoptau time.
		size_t hT;			//!< figure handle
	private:
	public:
		HJIPDE_extraOuts() :
			stoptau(0.),
			hT(0)
		{};
	private:

	};
};
#endif	/* __helperOC_type_hpp__ */
