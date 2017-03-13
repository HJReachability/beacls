#ifndef __ComputeGradients_CommandQueue_hpp__
#define __ComputeGradients_CommandQueue_hpp__

namespace helperOC {
	class ComputeGradients_OneSlice;
	class ComputeGradients_CommandQueue_impl;
	class ComputeGradients_CommandQueue {
	private:
		ComputeGradients_CommandQueue_impl* pimpl;
	public:
		ComputeGradients_OneSlice* pop();
		void push(ComputeGradients_OneSlice* command);
		ComputeGradients_CommandQueue();
		~ComputeGradients_CommandQueue();
	private:
		ComputeGradients_CommandQueue& operator=(const ComputeGradients_CommandQueue& rhs);
	};
};
#endif	/* __ComputeGradients_CommandQueue_hpp__ */

