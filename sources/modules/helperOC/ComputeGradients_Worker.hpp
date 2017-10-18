#ifndef __ComputeGradients_Worker_hpp__
#define __ComputeGradients_Worker_hpp__

#include <cstddef>
#include <helperOC/helperOC_type.hpp>
namespace levelset {
	class SpatialDerivative;
	class HJI_Grid;
};
namespace helperOC {
	class ComputeGradients_Worker_impl;
	class ComputeGradients_CommandQueue;
	
	class ComputeGradients_Worker {
	private:
		ComputeGradients_Worker_impl* pimpl;
	public:
		void run();
		void terminate();
		ComputeGradients_Worker(
			ComputeGradients_CommandQueue* commandQueue,
			const levelset::HJI_Grid* grid,
			const helperOC::ApproximationAccuracy_Type accuracy,
			const beacls::UVecType type,
			const int gpu_id
		);
		ComputeGradients_Worker(
			ComputeGradients_CommandQueue* commandQueue,
			const levelset::SpatialDerivative* spatialDerivative,
			const int gpu_id
		);
		int get_gpu_id() const;
		const levelset::SpatialDerivative* get_spatialDerivative() const;
		~ComputeGradients_Worker();
		ComputeGradients_Worker* clone() const;
	private:
		ComputeGradients_Worker();
		ComputeGradients_Worker(const ComputeGradients_Worker& rhs);
		ComputeGradients_Worker& operator=(const ComputeGradients_Worker& rhs);
	};
};
#endif	/* __ComputeGradients_Worker_hpp__ */

