#include "OdeCFL_CommandQueue.hpp"
#include "OdeCFL_CommandQueue_impl.hpp"
#include "OdeCFL_OneSlice.hpp"

#include <thread>
#include <chrono>
#include <algorithm>

beacls::OdeCFL_OneSlice* beacls::OdeCFL_CommandQueue_impl::pop() {
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
void beacls::OdeCFL_CommandQueue_impl::push(OdeCFL_OneSlice* command) {
	mtx.lock();
	commandQueue.push_back(command);
	mtx.unlock();
}
beacls::OdeCFL_CommandQueue_impl::OdeCFL_CommandQueue_impl() {}
beacls::OdeCFL_CommandQueue_impl::~OdeCFL_CommandQueue_impl() {
	mtx.lock();
	std::for_each(commandQueue.begin(), commandQueue.end(), [](auto& rhs) {
		if (rhs) delete rhs;
	});
	commandQueue.clear();
	mtx.unlock();
}

beacls::OdeCFL_OneSlice* beacls::OdeCFL_CommandQueue::pop() {
	if (pimpl) return pimpl->pop();
	else return NULL;
}
void beacls::OdeCFL_CommandQueue::push(OdeCFL_OneSlice* command) {
	if (pimpl) pimpl->push(command);
}

beacls::OdeCFL_CommandQueue::OdeCFL_CommandQueue() {
	pimpl = new OdeCFL_CommandQueue_impl;
}
beacls::OdeCFL_CommandQueue::~OdeCFL_CommandQueue() {
	if (pimpl) delete pimpl;
}

