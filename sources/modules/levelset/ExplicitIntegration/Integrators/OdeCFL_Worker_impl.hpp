#ifndef __OdeCFL_Worker_impl_hpp__
#define __OdeCFL_Worker_impl_hpp__

#include <mutex>
#include <cstddef>
#include <thread>
#include <Core/UVec.hpp>
class OdeCFL_CommandQueue;
class Term;
class SchemeData;
namespace beacls {
	class OdeCFL_Worker_impl {
	private:
		OdeCFL_CommandQueue* commandQueue;
		std::mutex mtx;
		std::thread th;
		int gpu_id;
		bool exitFlag;
		const Term* term;
		const SchemeData* schemeData;
		std::vector<beacls::FloatVec> thread_local_derivMins;
		std::vector<beacls::FloatVec> thread_local_derivMaxs;
		beacls::FloatVec new_step_bound_invs;
		bool schemeDataModified;

	public:
		void OdeCFL_Worker_proc();
		void run();
		void terminate();
		int get_gpu_id() const {
			return gpu_id;
		}
		OdeCFL_Worker_impl(
			OdeCFL_CommandQueue* commandQueue,
			const Term* term,
			const int gpu_id
		);
		void set_schemeData(const SchemeData* sd);
		~OdeCFL_Worker_impl();
	private:
		OdeCFL_Worker_impl();
		OdeCFL_Worker_impl(const OdeCFL_Worker_impl& rhs);
		OdeCFL_Worker_impl& operator=(const OdeCFL_Worker_impl& rhs);
	};
};
#endif	/* __OdeCFL_Worker_impl_hpp__ */

