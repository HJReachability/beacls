#ifndef __ComputeGradients_Worker_impl_hpp__
#define __ComputeGradients_Worker_impl_hpp__

#include <mutex>
#include <cstddef>
#include <thread>
#include <helperOC/helperOC_type.hpp>
#include <Core/UVec.hpp>
namespace levelset {
	class SpatialDerivative;
};
namespace helperOC {
	class ComputeGradients_CommandQueue;
	class ComputeGradients_Worker_impl {
	private:
		ComputeGradients_CommandQueue* commandQueue;
		std::mutex mtx;
		std::thread th;
		int gpu_id;
		bool exitFlag;
		const levelset::SpatialDerivative* spatialDerivative;
		beacls::UVec original_data_line_uvec;
		std::vector<beacls::UVec> deriv_c_line_uvecs;
		std::vector<beacls::UVec> deriv_l_line_uvecs;
		std::vector<beacls::UVec> deriv_r_line_uvecs;
	public:
		void ComputeGradients_Worker_proc();
		void run();
		void terminate();
		const levelset::SpatialDerivative* get_spatialDerivative() const {
			return spatialDerivative;
		}
		int get_gpu_id() const {
			return gpu_id;
		}
		ComputeGradients_Worker_impl(
			ComputeGradients_CommandQueue* commandQueue,
			const levelset::SpatialDerivative* spatialDerivative,
			const int gpu_id
		);
		ComputeGradients_Worker_impl(
			ComputeGradients_CommandQueue* commandQueue,
			const levelset::HJI_Grid* grid,
			const helperOC::ApproximationAccuracy_Type accuracy,
			const beacls::UVecType type,
			const int gpu_id
		);
		~ComputeGradients_Worker_impl();
		ComputeGradients_Worker_impl* clone() const {
			return new ComputeGradients_Worker_impl(*this);
		}
	private:
		ComputeGradients_Worker_impl();
		ComputeGradients_Worker_impl(const ComputeGradients_Worker_impl& rhs);
		ComputeGradients_Worker_impl& operator=(const ComputeGradients_Worker_impl& rhs);
	};
};
#endif	/* __ComputeGradients_Worker_impl_hpp__ */

