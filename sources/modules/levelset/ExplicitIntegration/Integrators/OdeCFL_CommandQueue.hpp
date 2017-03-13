#ifndef __OdeCFL_CommandQueue_hpp__
#define __OdeCFL_CommandQueue_hpp__

namespace beacls {
	class OdeCFL_OneSlice;
	class OdeCFL_CommandQueue_impl;
	class OdeCFL_CommandQueue {
	private:
		OdeCFL_CommandQueue_impl* pimpl;
	public:
		OdeCFL_OneSlice* pop();
		void push(OdeCFL_OneSlice* command);
		OdeCFL_CommandQueue();
		~OdeCFL_CommandQueue();
	private:
		OdeCFL_CommandQueue(const OdeCFL_CommandQueue& rhs);
		OdeCFL_CommandQueue& operator=(const OdeCFL_CommandQueue& rhs);
	};
};
#endif	/* __OdeCFL_CommandQueue_hpp__ */

