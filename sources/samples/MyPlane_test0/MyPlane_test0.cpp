#define _USE_MATH_DEFINES
#include <levelset/levelset.hpp>
#include <helperOC/helperOC.hpp>
#include <cmath>
#include <numeric>
#include <functional>
#include <cfloat>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <helperOC/DynSys/Plane/Plane.hpp>

/**
	@brief Tests the Plane class by computing a reachable set and then computing the optimal trajectory from the reachable set.
	*/
int main(int argc, char *argv[])
{
	bool dump_file = false;
	if (argc >= 2) {
		dump_file = (atoi(argv[1]) == 0) ? false : true;
	}

	bool enable_user_defined_dynamics_on_gpu = true;
	if (argc >= 9) {
		enable_user_defined_dynamics_on_gpu = (atoi(argv[8]) == 0) ? false : true;
	}
	//!< Plane parameters
	/* 
	To Be filled 
	*/

	const FLOAT_TYPE inf = std::numeric_limits<FLOAT_TYPE>::infinity();
	//!< Target and obstacle
	/* 
	To Be filled 
	*/

	//!< Compute reachable set
	/* 
	To Be filled 
	*/

	// Dynamical system parameters
	/* 
	To Be filled 
	*/

	// Target set and visualization
	/* 
	To Be filled 
	*/

	/* 
	To Be filled 
	*/

	/* 
	To Be filled 
	*/
	//!< Compute optimal trajectory
	/* 
	To Be filled 
	*/

	/* 
	To Be filled 
	*/

	/* 
	To Be filled 
	*/
	return 0;
}

