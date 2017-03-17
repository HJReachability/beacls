#ifndef __ComputeGradients_hpp__
#define __ComputeGradients_hpp__


//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#include <typedef.hpp>
#include <vector>
#include <cstddef>
#include <utility>
using namespace std::rel_ops;

#include <helperOC/helperOC_type.hpp>

namespace levelset {
	class HJI_Grid;
	class SpatialDerivative;
};

namespace helperOC {
	class ComputeGradients_CommandQueue;
	class ComputeGradients_Worker;
	class ComputeGradients {
	private:
		std::vector<ComputeGradients_Worker*> workers;
		ComputeGradients_CommandQueue* commandQueue;

		const beacls::UVecType type;
		beacls::FloatVec modified_data;
	public:
		/*
		@brief	Constructor
		@param	[in]	grid		grid structure
		@param	[in]	accuracy	derivative approximation function (from level set
		toolbox)
		*/
		PREFIX_VC_DLL
		ComputeGradients(
			const levelset::HJI_Grid* grid,
			helperOC::ApproximationAccuracy_Type accuracy = helperOC::ApproximationAccuracy_veryHigh,
			const beacls::UVecType type = beacls::UVecType_Vector
		);
		PREFIX_VC_DLL
		~ComputeGradients();
		PREFIX_VC_DLL
			bool operator==(const ComputeGradients& rhs) const;
		PREFIX_VC_DLL
		beacls::UVecType get_type() const;
		/*
			@brief	Estimates the costate p at position x for cost function data on grid g by
					numerically taking partial derivatives along each grid direction.
					Numerical derivatives are taken using the levelset toolbox
			@param	[out]	derivC	(central) gradient in a g.dim by 1 vector
			@param	[out]	derivL	left gradient
			@param	[out]	derivR	right gradient
			@param	[in]	grid		grid structure
			@param	[in]	data		array of g.dim dimensions containing function values
			@param	[in]	data_length	array data length
			@param	[in]	upWind		whether to use upwinding (ignored; to be implemented in
										the future
			@param [in] line_length_of_chunk	Line length of each parallel execution chunks (0 means set automatically)
			@param [in] num_of_threads			Number of CPU Threads (0 means use all logical threads of CPU)
			@param [in] num_of_gpus				Number of GPUs which (0 means use all GPUs)
			@retval	true	Succeed
			@retval	false	Failed
		*/
		PREFIX_VC_DLL
		bool operator()(
			std::vector<beacls::FloatVec >& derivC,
			std::vector<beacls::FloatVec >& derivL,
			std::vector<beacls::FloatVec >& derivR,
			const levelset::HJI_Grid* grid,
			const beacls::FloatVec& data,
			const size_t data_length,
			const bool upWind = false,
			const helperOC::ExecParameters& execParameters = helperOC::ExecParameters()
			);
	private:
		ComputeGradients& operator=(const ComputeGradients& rhs);
		ComputeGradients(const ComputeGradients& rhs);
	};

}
#endif	/* __ComputeGradients_hpp__ */
