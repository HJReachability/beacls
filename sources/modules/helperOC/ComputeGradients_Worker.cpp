#include "ComputeGradients_Worker.hpp"
#include "ComputeGradients_Worker_impl.hpp"
#include "ComputeGradients_OneSlice.hpp"
#include "ComputeGradients_CommandQueue.hpp"
#include <helperOC/helperOC_type.hpp>
#include <levelset/SpatialDerivative/SpatialDerivative.hpp>
#include <levelset/levelset.hpp>

helperOC::ComputeGradients_Worker_impl::ComputeGradients_Worker_impl(
	ComputeGradients_CommandQueue* commandQueue,
	const levelset::HJI_Grid* grid,
	const helperOC::ApproximationAccuracy_Type accuracy,
	const beacls::UVecType type,
	const int gpu_id
) : commandQueue(commandQueue),
	gpu_id(gpu_id),
	exitFlag(true) {
	//! accuracy
	switch (accuracy) {
	case helperOC::ApproximationAccuracy_low:
		spatialDerivative = new levelset::UpwindFirstFirst(grid, type);
		break;
	case helperOC::ApproximationAccuracy_medium:
		spatialDerivative = new levelset::UpwindFirstENO2(grid, type);
		break;
	case helperOC::ApproximationAccuracy_high:
		spatialDerivative = new levelset::UpwindFirstENO3(grid, type);
		break;
	case helperOC::ApproximationAccuracy_veryHigh:
		spatialDerivative = new levelset::UpwindFirstWENO5(grid, type);
		break;
	case helperOC::ApproximationAccuracy_Invalid:
	default:
		std::cerr << "Unknown accuracy level " << accuracy << std::endl;
		spatialDerivative = NULL;
		break;
	}
}
helperOC::ComputeGradients_Worker_impl::ComputeGradients_Worker_impl(
	ComputeGradients_CommandQueue* commandQueue,
	const levelset::SpatialDerivative* spatialDerivative,
	const int gpu_id
) :
	commandQueue(commandQueue),
	spatialDerivative(spatialDerivative),
	gpu_id(gpu_id),
	exitFlag(true) {
}
void helperOC::ComputeGradients_Worker_impl::ComputeGradients_Worker_proc() {
	bool nextExitFlag;
	mtx.lock();
	nextExitFlag = exitFlag;
	mtx.unlock();
	beacls::set_gpu_id(gpu_id);
	levelset::SpatialDerivative* thread_local_spatialDerivative = spatialDerivative->clone();
	while (!nextExitFlag) {
		if (commandQueue) {
			ComputeGradients_OneSlice* command = commandQueue->pop();
			if (command) {
				command->execute(
					thread_local_spatialDerivative,
					original_data_line_uvec,
					deriv_c_line_uvecs,
					deriv_l_line_uvecs,
					deriv_r_line_uvecs
				);
				command->set_finished();
			}
			std::this_thread::yield();
		}
		mtx.lock();
		nextExitFlag = exitFlag;
		mtx.unlock();
	}
	if (thread_local_spatialDerivative) delete thread_local_spatialDerivative;
}
helperOC::ComputeGradients_Worker_impl::~ComputeGradients_Worker_impl() {
	terminate();
	if (spatialDerivative) delete spatialDerivative;
}
helperOC::ComputeGradients_Worker_impl::ComputeGradients_Worker_impl(const ComputeGradients_Worker_impl& rhs) :
	commandQueue(rhs.commandQueue),
	gpu_id(rhs.gpu_id),
	exitFlag(true),
	spatialDerivative(rhs.spatialDerivative->clone())
{}

void helperOC::ComputeGradients_Worker_impl::terminate() {
	mtx.lock();
	if (!exitFlag) {
		exitFlag = true;
		mtx.unlock();
		th.join();
	}
	else {
		mtx.unlock();
	}
}
void helperOC::ComputeGradients_Worker_impl::run() {
	mtx.lock();
	if (exitFlag) {
		exitFlag = false;
		th = std::thread(&helperOC::ComputeGradients_Worker_impl::ComputeGradients_Worker_proc, this);
	}
	mtx.unlock();
}

int helperOC::ComputeGradients_Worker::get_gpu_id() const {
	if (pimpl) return pimpl->get_gpu_id();
	else return 0;
}
void helperOC::ComputeGradients_Worker::set_gpu_id(const int id) {
	if (pimpl) pimpl->set_gpu_id(id);
}

void helperOC::ComputeGradients_Worker::run() {
	if (pimpl) pimpl->run();
}
void helperOC::ComputeGradients_Worker::terminate() {
	if (pimpl) pimpl->terminate();
}
const levelset::SpatialDerivative* helperOC::ComputeGradients_Worker::get_spatialDerivative() const {
	if (pimpl) return pimpl->get_spatialDerivative();
	else return NULL;
}
helperOC::ComputeGradients_Worker::ComputeGradients_Worker(const helperOC::ComputeGradients_Worker& rhs) :
	pimpl(rhs.pimpl->clone())
{
}
helperOC::ComputeGradients_Worker* helperOC::ComputeGradients_Worker::clone() const {
	return new ComputeGradients_Worker(*this);
}
helperOC::ComputeGradients_Worker::ComputeGradients_Worker(
	ComputeGradients_CommandQueue* commandQueue, 
	const levelset::HJI_Grid* grid,
	const helperOC::ApproximationAccuracy_Type accuracy,
	const beacls::UVecType type,
	const int gpu_id) {
	pimpl = new ComputeGradients_Worker_impl(commandQueue, grid, accuracy, type, gpu_id);
}
helperOC::ComputeGradients_Worker::ComputeGradients_Worker(
	ComputeGradients_CommandQueue* commandQueue,
	const levelset::SpatialDerivative* spatialDerivative,
	const int gpu_id
) {
	pimpl = new ComputeGradients_Worker_impl(commandQueue, spatialDerivative, gpu_id);

}
helperOC::ComputeGradients_Worker::~ComputeGradients_Worker() {
	if (pimpl) delete pimpl;
}
