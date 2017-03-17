#include "ComputeGradients_CommandQueue.hpp"
#include "ComputeGradients_CommandQueue_impl.hpp"
#include "ComputeGradients_OneSlice.hpp"

#include <thread>
#include <chrono>
#include <algorithm>

helperOC::ComputeGradients_OneSlice* helperOC::ComputeGradients_CommandQueue_impl::pop() {
	mtx.lock();
	ComputeGradients_OneSlice* command;
	if (commandQueue.size() > 0) {
		command = commandQueue.front();
		commandQueue.pop_front();
	}
	else {
		command = NULL;
	}
	mtx.unlock();
	return command;
}
void helperOC::ComputeGradients_CommandQueue_impl::push(ComputeGradients_OneSlice* command) {
	mtx.lock();
	commandQueue.push_back(command);
	mtx.unlock();
}
helperOC::ComputeGradients_CommandQueue_impl::ComputeGradients_CommandQueue_impl() {}
helperOC::ComputeGradients_CommandQueue_impl::~ComputeGradients_CommandQueue_impl() {
	mtx.lock();
	std::for_each(commandQueue.begin(), commandQueue.end(), [](auto& rhs) {
		if (rhs) delete rhs;
	});
	commandQueue.clear();
	mtx.unlock();
}

helperOC::ComputeGradients_OneSlice* helperOC::ComputeGradients_CommandQueue::pop() {
	if (pimpl) return pimpl->pop();
	else return NULL;
}
void helperOC::ComputeGradients_CommandQueue::push(ComputeGradients_OneSlice* command) {
	if (pimpl) pimpl->push(command);
}

helperOC::ComputeGradients_CommandQueue::ComputeGradients_CommandQueue() {
	pimpl = new ComputeGradients_CommandQueue_impl;
}
helperOC::ComputeGradients_CommandQueue::~ComputeGradients_CommandQueue() {
	if (pimpl) delete pimpl;
}

