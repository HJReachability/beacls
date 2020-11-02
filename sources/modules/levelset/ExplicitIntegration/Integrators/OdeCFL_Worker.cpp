#include "OdeCFL_Worker.hpp"
#include "OdeCFL_Worker_impl.hpp"
#include "OdeCFL_OneSlice.hpp"
#include "OdeCFL_CommandQueue.hpp"
#include <levelset/ExplicitIntegration/Terms/Term.hpp>
#include <levelset/ExplicitIntegration/SchemeData.hpp>
levelset::OdeCFL_Worker_impl::OdeCFL_Worker_impl(
	OdeCFL_CommandQueue* commandQueue,
	const Term* term,
	const int gpu_id
) : commandQueue(commandQueue),
	gpu_id(gpu_id),
	exitFlag(true),
	term(term->clone()),
	schemeData(NULL),
	schemeDataModified(false) {
}

void levelset::OdeCFL_Worker_impl::set_schemeData(const SchemeData* sd) {
	mtx.lock();
	if (schemeData) {
		if (*schemeData != *sd) {
			delete schemeData;
			schemeData = sd->clone();
			schemeDataModified = true;
		}
	}
	else {
		schemeData = sd->clone();
		schemeDataModified = true;
	}
	mtx.unlock();
}

void levelset::OdeCFL_Worker_impl::OdeCFL_Worker_proc() {
	bool nextExitFlag;
	mtx.lock();
	nextExitFlag = exitFlag;
	beacls::set_gpu_id(get_gpu_id());
	SchemeData* thread_local_schemeData = schemeData ? schemeData->clone() : NULL;
	schemeDataModified = false;
	mtx.unlock();
	const Term* thread_local_term = term->clone();
	while (!nextExitFlag) {
		if (commandQueue) {
			OdeCFL_OneSlice* command = commandQueue->pop();
			if (command) {
				mtx.lock();
				if (schemeDataModified) {	//!< Update thread local scheme data, if it is necessary.
					if (thread_local_schemeData) 
						delete thread_local_schemeData;
					thread_local_schemeData = schemeData->clone();
					schemeDataModified = false;
				}
				mtx.unlock();
				command->execute(
					thread_local_term,
					thread_local_schemeData,
					thread_local_derivMins,
					thread_local_derivMaxs,
					new_step_bound_invs
				);
			  command->set_finished();
			}
			std::this_thread::yield();
		}
		mtx.lock();
		nextExitFlag = exitFlag;
		mtx.unlock();
	}
	if (thread_local_schemeData) delete thread_local_schemeData;
	if (thread_local_term) delete thread_local_term;
}

void levelset::OdeCFL_Worker_impl::OdeCFL_Worker_proc_local_q(const std::set<size_t> &Q) {
	bool nextExitFlag;
	mtx.lock();
	nextExitFlag = exitFlag;
	beacls::set_gpu_id(get_gpu_id());
	SchemeData* thread_local_schemeData = schemeData ? schemeData->clone() : NULL;
	schemeDataModified = false;
	mtx.unlock();
	const Term* thread_local_term = term->clone();
	while (!nextExitFlag) {
		if (commandQueue) {
			OdeCFL_OneSlice* command = commandQueue->pop();
			if (command) {
				mtx.lock();
				if (schemeDataModified) {	//!< Update thread local scheme data, if it is necessary.
					if (thread_local_schemeData) 
						delete thread_local_schemeData;
					thread_local_schemeData = schemeData->clone();
					schemeDataModified = false;
				}
				mtx.unlock();
				command->execute_local_q(
					thread_local_term,
					thread_local_schemeData,
					thread_local_derivMins,
					thread_local_derivMaxs,
					new_step_bound_invs, 
					Q 
				);
			  command->set_finished();
			}
			std::this_thread::yield();
		}
		mtx.lock();
		nextExitFlag = exitFlag;
		mtx.unlock();
	}
	if (thread_local_schemeData) delete thread_local_schemeData;
	if (thread_local_term) delete thread_local_term;
}
levelset::OdeCFL_Worker_impl::~OdeCFL_Worker_impl() {
	terminate();
	if(schemeData) delete schemeData;
	if (term) delete term;
}
void levelset::OdeCFL_Worker_impl::terminate() {
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
void levelset::OdeCFL_Worker_impl::run() {
	mtx.lock();
	if (exitFlag) {
		exitFlag = false;
		th = std::thread(&levelset::OdeCFL_Worker_impl::OdeCFL_Worker_proc, this);
	}
	mtx.unlock();
}

void levelset::OdeCFL_Worker_impl::run_local_q(const std::set<size_t> &Q) {
	mtx.lock();
	if (exitFlag) {
		exitFlag = false;
		th = std::thread(&levelset::OdeCFL_Worker_impl::OdeCFL_Worker_proc_local_q, this, std::ref(Q));
	}
	mtx.unlock();
}
int levelset::OdeCFL_Worker::get_gpu_id() const {
	if (pimpl) return pimpl->get_gpu_id();
	else return 0;
}
void levelset::OdeCFL_Worker::set_schemeData(const SchemeData* sd) {
	if (pimpl) pimpl->set_schemeData(sd);
}

void levelset::OdeCFL_Worker::run() {
	if (pimpl) pimpl->run();
}
void levelset::OdeCFL_Worker::run_local_q(const std::set<size_t>& Q) {
	if (pimpl) pimpl->run_local_q(Q);
}
void levelset::OdeCFL_Worker::terminate() {
	if (pimpl) pimpl->terminate();
}
levelset::OdeCFL_Worker::OdeCFL_Worker(OdeCFL_CommandQueue* commandQueue, const Term* term, const int gpu_id) {
	pimpl = new OdeCFL_Worker_impl(commandQueue, term, gpu_id);
}
levelset::OdeCFL_Worker::~OdeCFL_Worker() {
	if (pimpl) delete pimpl;
}
