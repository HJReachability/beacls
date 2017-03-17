
#ifndef __UTest_HJIPDE_solve_hpp__
#define __UTest_HJIPDE_solve_hpp__
#include <cstring>
#include <vector>
#include <typedef.hpp>

typedef enum HJIPDE_solve_WhatTest {
	HJIPDE_solve_WhatTest_Invalid,
	HJIPDE_solve_WhatTest_minWith,	//!<Test the minWith functionality
	HJIPDE_solve_WhatTest_tvTargets,	//!<Test the time - varying targets
	HJIPDE_solve_WhatTest_singleObs,	//!<Test with a single static obstacle
	HJIPDE_solve_WhatTest_tvObs,	//!<Test with time - varying obstacles
	HJIPDE_solve_WhatTest_obs_stau,	//!<single obstacle over a few time steps
	HJIPDE_solve_WhatTest_stopInit,	//!<Test the functionality of stopping reachable set computation once it includes the initial state
	HJIPDE_solve_WhatTest_stopSetInclude,	//!<Test the functionality of stopping reacahble set computation once it contains some set
	HJIPDE_solve_WhatTest_stopSetIntersect,	//!<Test the functionality of stopping reacahble set computation once it intersects some set
	HJIPDE_solve_WhatTest_plotData,	//!<Test the functionality of plotting reachable sets as they are being computed
	HJIPDE_solve_WhatTest_savedData,	//!<Test starting from saved data (where data0 has dimension g.dim + 1)
	HJIPDE_solve_WhatTest_stopConvergeSmallDubinsCar,	//!<Test the functionality of plotting reachable sets as they are being computed, with small size DubionsCar model
    HJIPDE_solve_WhatTest_stopConvergeSmallDubinsCarCAvoid,	//!<Test the functionality of plotting reachable sets as they are being computed, with small size DubionsCarCAvoid model
    HJIPDE_solve_WhatTest_stopConvergeMiddleDubinsCar,	//!<Test the functionality of plotting reachable sets as they are being computed, with middle size DubionsCar model
    HJIPDE_solve_WhatTest_stopConvergeMiddleDubinsCarCAvoid,	//!<Test the functionality of plotting reachable sets as they are being computed, with middle size DubionsCarCAvoid model
} HJIPDE_solve_WhatTest;

typedef enum HJIPDE_solve_Shape {
	HJIPDE_solve_Shape_Invalid,
	HJIPDE_solve_Shape_Cylinder,
	HJIPDE_solve_Shape_RectangleByCorner,
	HJIPDE_solve_Shape_RectangleByCenter,
	HJIPDE_solve_Shape_Sphere,
} HJIPDE_solve_Shape;


/*
	@brief Tests the HJIPDE_solve function as well as provide an example of how to use it.
	@param	[out]	message	Output message.
	@param	[in]	expects_filenames	Expected result filenames.
	@param	[in]	whatTest	Argument that can be used to test a particular feature
*/
bool run_UTest_HJIPDE_solve(
	std::string &message,
	const std::vector<std::string>& expects_filenames,
	const HJIPDE_solve_WhatTest whatTest,
	const HJIPDE_solve_Shape shapeType,
	const beacls::UVecType type,
	const FLOAT_TYPE small_diff,
	const size_t chunk_size,
	const int num_of_threads,
	const int num_of_gpus,
	const levelset::DelayedDerivMinMax_Type delayedDerivMinMax,
	const bool enable_user_defined_dynamics_on_gpu
	);

#endif	/* __UTest_HJIPDE_solve_hpp__ */

