#include "ComputeGradients_Worker.hpp"
#include "ComputeGradients_Worker_impl.hpp"
#include "ComputeGradients_OneSlice.hpp"
#include "ComputeGradients_CommandQueue.hpp"
#include <levelset/SpatialDerivative/SpatialDerivative.hpp>

helperOC::ComputeGradients_Worker_impl::ComputeGradients_Worker_impl(
	ComputeGradients_CommandQueue* commandQueue,
	const SpatialDerivative* spatialDerivative,
	const int gpu_id
) : commandQueue(commandQueue),
	gpu_id(gpu_id),
	exitFlag(true),
	spatialDerivative(spatialDerivative->clone()) {
}

void helperOC::ComputeGradients_Worker_impl::ComputeGradients_Worker_proc() {
	bool nextExitFlag;
	mtx.lock();
	nextExitFlag = exitFlag;
	mtx.unlock();
	beacls::set_gpu_id(gpu_id);
	SpatialDerivative* thread_local_spatialDerivative = spatialDerivative->clone();
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

void helperOC::ComputeGradients_Worker::run() {
	if (pimpl) pimpl->run();
}
void helperOC::ComputeGradients_Worker::terminate() {
	if (pimpl) pimpl->terminate();
}
const SpatialDerivative* helperOC::ComputeGradients_Worker::get_spatialDerivative() const {
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
	const SpatialDerivative* derivFunc,
	const int gpu_id) {
	pimpl = new ComputeGradients_Worker_impl(commandQueue, derivFunc, gpu_id);
}
helperOC::ComputeGradients_Worker::~ComputeGradients_Worker() {
	if (pimpl) delete pimpl;
}
