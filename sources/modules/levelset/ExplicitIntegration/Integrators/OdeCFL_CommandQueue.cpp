#include "OdeCFL_CommandQueue.hpp"
#include "OdeCFL_CommandQueue_impl.hpp"
#include "OdeCFL_OneSlice.hpp"

#include <thread>
#include <chrono>
#include <algorithm>

levelset::OdeCFL_OneSlice* levelset::OdeCFL_CommandQueue_impl::pop() {
	mtx.lock();
	OdeCFL_OneSlice* command;
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
void levelset::OdeCFL_CommandQueue_impl::push(OdeCFL_OneSlice* command) {
	mtx.lock();
	commandQueue.push_back(command);
	mtx.unlock();
}
levelset::OdeCFL_CommandQueue_impl::OdeCFL_CommandQueue_impl() {}
levelset::OdeCFL_CommandQueue_impl::~OdeCFL_CommandQueue_impl() {
	mtx.lock();
	std::for_each(commandQueue.begin(), commandQueue.end(), [](auto& rhs) {
		if (rhs) delete rhs;
	});
	commandQueue.clear();
	mtx.unlock();
}

levelset::OdeCFL_OneSlice* levelset::OdeCFL_CommandQueue::pop() {
	if (pimpl) return pimpl->pop();
	else return NULL;
}
void levelset::OdeCFL_CommandQueue::push(OdeCFL_OneSlice* command) {
	if (pimpl) pimpl->push(command);
}

levelset::OdeCFL_CommandQueue::OdeCFL_CommandQueue() {
	pimpl = new OdeCFL_CommandQueue_impl;
}
levelset::OdeCFL_CommandQueue::~OdeCFL_CommandQueue() {
	if (pimpl) delete pimpl;
}

