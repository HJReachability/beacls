#ifndef __ComputeGradients_CommandQueue_impl_hpp__
#define __ComputeGradients_CommandQueue_impl_hpp__

#include <mutex>
#include <deque>

namespace helperOC {
	class ComputeGradients_OneSlice;
	class ComputeGradients_CommandQueue_impl {
	private:
		std::mutex mtx;
		std::deque<ComputeGradients_OneSlice*> commandQueue;
	public:
		ComputeGradients_OneSlice* pop();
		void push(ComputeGradients_OneSlice* command);
		ComputeGradients_CommandQueue_impl();
		~ComputeGradients_CommandQueue_impl();
	private:
		ComputeGradients_CommandQueue_impl(const ComputeGradients_CommandQueue_impl& rhs);
		ComputeGradients_CommandQueue_impl& operator=(const ComputeGradients_CommandQueue_impl& rhs);
	};
};
#endif	/* __ComputeGradients_CommandQueue_impl_hpp__ */

