#ifndef __OdeCFL_CommandQueue_impl_hpp__
#define __OdeCFL_CommandQueue_impl_hpp__

#include <mutex>
#include <deque>

namespace levelset {
	class OdeCFL_OneSlice;
	class OdeCFL_CommandQueue_impl {
	private:
		std::mutex mtx;
		std::deque<OdeCFL_OneSlice*> commandQueue;
	public:
		OdeCFL_OneSlice* pop();
		void push(OdeCFL_OneSlice* command);
		OdeCFL_CommandQueue_impl();
		~OdeCFL_CommandQueue_impl();
	private:
		OdeCFL_CommandQueue_impl(const OdeCFL_CommandQueue_impl& rhs);
		OdeCFL_CommandQueue_impl& operator=(const OdeCFL_CommandQueue_impl& rhs);
	};
};
#endif	/* __OdeCFL_CommandQueue_impl_hpp__ */

