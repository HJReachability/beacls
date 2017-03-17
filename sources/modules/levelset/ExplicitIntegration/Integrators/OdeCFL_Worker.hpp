#ifndef __OdeCFL_Worker_hpp__
#define __OdeCFL_Worker_hpp__

#include <cstddef>
namespace levelset {
	class Term;
	class SchemeData;
	class OdeCFL_Worker_impl;
	class OdeCFL_CommandQueue;

	class OdeCFL_Worker {
	private:
		OdeCFL_Worker_impl* pimpl;
	public:
		void run();
		void terminate();
		OdeCFL_Worker(
			OdeCFL_CommandQueue* commandQueue,
			const Term* term, 
			const int gpu_id
		);
		void set_schemeData(const SchemeData* sd);
		int get_gpu_id() const;
		~OdeCFL_Worker();
	private:
		OdeCFL_Worker();
		OdeCFL_Worker(const OdeCFL_Worker& rhs);
		OdeCFL_Worker& operator=(const OdeCFL_Worker& rhs);
	};
};
#endif	/* __OdeCFL_Worker_hpp__ */

