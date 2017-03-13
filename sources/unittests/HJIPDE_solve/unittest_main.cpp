//#include "stdafx.h"
#define _USE_MATH_DEFINES
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include "UTest_HJIPDE_solve.hpp"

namespace UTest_CPU_wLastMinMax_HJIPDE_solve
{		
	class UnitTest_CPU_wLastMinMax_HJIPDE_solve_minWith
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_minWith;
		bool Test_CPU_wLastMinMax_HJIPDE_solve_minWith_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_minWith_Cylinder_1.mat"),
				std::string("./inputs/HJIPDE_solve_test_minWith_Cylinder_2.mat")
			};
			std::string message;
			if(!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_CPU_wLastMinMax_HJIPDE_solve_tvTargets
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_tvTargets;
		bool Test_CPU_wLastMinMax_HJIPDE_solve_tvTargets_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_tvTargets_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_CPU_wLastMinMax_HJIPDE_solve_singleObs
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_singleObs;
		bool Test_CPU_wLastMinMax_HJIPDE_solve_singleObs_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_singleObs_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_CPU_wLastMinMax_HJIPDE_solve_tvObs
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_tvObs;
		bool Test_CPU_wLastMinMax_HJIPDE_solve_tvObs_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_tvObs_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_CPU_wLastMinMax_HJIPDE_solve_obs_stau
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_obs_stau;
		bool Test_CPU_wLastMinMax_HJIPDE_solve_obs_stau_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_obs_stau_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_CPU_wLastMinMax_HJIPDE_solve_stopInit
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopInit;
		bool Test_CPU_wLastMinMax_HJIPDE_solve_stopInit_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_stopInit_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_CPU_wLastMinMax_HJIPDE_solve_stopSetInclude
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopSetInclude;
		bool Test_CPU_wLastMinMax_HJIPDE_solve_stopSetInclude_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_stopSetInclude_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_CPU_wLastMinMax_HJIPDE_solve_stopSetIntersect
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopSetIntersect;
		bool Test_CPU_wLastMinMax_HJIPDE_solve_stopSetIntersect_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_stopSetIntersect_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_CPU_wLastMinMax_HJIPDE_solve_plotData
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_plotData;
		bool Test_CPU_wLastMinMax_HJIPDE_solve_plotData_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_plotData_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_CPU_wLastMinMax_HJIPDE_solve_savedData
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_savedData;
		bool Test_CPU_wLastMinMax_HJIPDE_solve_savedData_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_savedData_Cylinder_1.mat"),
				std::string("./inputs/HJIPDE_solve_test_savedData_Cylinder_2.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_CPU_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar
    {
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopConvergeSmallDubinsCar;
		bool Test_CPU_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_stopConvergeSmallDubinsCar_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_CPU_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopConvergeSmallDubinsCarCAvoid;
		bool Test_CPU_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_stopConvergeSmallDubinsCarCAvoid_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
}

namespace UTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve
{		
	class UnitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_minWith
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_minWith;
		bool Test_GPU_PHASE1_wLastMinMax_HJIPDE_solve_minWith_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_minWith_Cylinder_1.mat"),
				std::string("./inputs/HJIPDE_solve_test_minWith_Cylinder_2.mat")
			};
			std::string message;
			if(!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_tvTargets
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_tvTargets;
		bool Test_GPU_PHASE1_wLastMinMax_HJIPDE_solve_tvTargets_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_tvTargets_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_singleObs
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_singleObs;
		bool Test_GPU_PHASE1_wLastMinMax_HJIPDE_solve_singleObs_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_singleObs_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_tvObs
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_tvObs;
		bool Test_GPU_PHASE1_wLastMinMax_HJIPDE_solve_tvObs_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_tvObs_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_obs_stau
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_obs_stau;
		bool Test_GPU_PHASE1_wLastMinMax_HJIPDE_solve_obs_stau_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_obs_stau_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_stopInit
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopInit;
		bool Test_GPU_PHASE1_wLastMinMax_HJIPDE_solve_stopInit_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_stopInit_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_stopSetInclude
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopSetInclude;
		bool Test_GPU_PHASE1_wLastMinMax_HJIPDE_solve_stopSetInclude_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_stopSetInclude_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_stopSetIntersect
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopSetIntersect;
	  bool Test_GPU_PHASE1_wLastMinMax_HJIPDE_solve_stopSetIntersect_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_stopSetIntersect_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_plotData
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_plotData;
		bool Test_GPU_PHASE1_wLastMinMax_HJIPDE_solve_plotData_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_plotData_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_savedData
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_savedData;
		bool Test_GPU_PHASE1_wLastMinMax_HJIPDE_solve_savedData_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_savedData_Cylinder_1.mat"),
				std::string("./inputs/HJIPDE_solve_test_savedData_Cylinder_2.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar
    {
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopConvergeSmallDubinsCar;
		bool Test_GPU_PHASE1_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_stopConvergeSmallDubinsCar_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopConvergeSmallDubinsCarCAvoid;
		bool Test_GPU_PHASE1_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_stopConvergeSmallDubinsCarCAvoid_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
}


namespace UTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve
{		
	class UnitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_minWith
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_minWith;
		bool Test_GPU_PHASE2_wLastMinMax_HJIPDE_solve_minWith_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_minWith_Cylinder_1.mat"),
				std::string("./inputs/HJIPDE_solve_test_minWith_Cylinder_2.mat")
			};
			std::string message;
			if(!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_tvTargets
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_tvTargets;
		bool Test_GPU_PHASE2_wLastMinMax_HJIPDE_solve_tvTargets_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_tvTargets_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_singleObs
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_singleObs;
		bool Test_GPU_PHASE2_wLastMinMax_HJIPDE_solve_singleObs_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_singleObs_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_tvObs
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_tvObs;
		bool Test_GPU_PHASE2_wLastMinMax_HJIPDE_solve_tvObs_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_tvObs_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_obs_stau
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_obs_stau;
		bool Test_GPU_PHASE2_wLastMinMax_HJIPDE_solve_obs_stau_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_obs_stau_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_stopInit
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopInit;
		bool Test_GPU_PHASE2_wLastMinMax_HJIPDE_solve_stopInit_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_stopInit_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_stopSetInclude
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopSetInclude;
		bool Test_GPU_PHASE2_wLastMinMax_HJIPDE_solve_stopSetInclude_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_stopSetInclude_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_stopSetIntersect
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopSetIntersect;
	  bool Test_GPU_PHASE2_wLastMinMax_HJIPDE_solve_stopSetIntersect_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_stopSetIntersect_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_plotData
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_plotData;
		bool Test_GPU_PHASE2_wLastMinMax_HJIPDE_solve_plotData_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_plotData_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_savedData
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_savedData;
		bool Test_GPU_PHASE2_wLastMinMax_HJIPDE_solve_savedData_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_savedData_Cylinder_1.mat"),
				std::string("./inputs/HJIPDE_solve_test_savedData_Cylinder_2.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar
    {
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopConvergeSmallDubinsCar;
		bool Test_GPU_PHASE2_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_stopConvergeSmallDubinsCar_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopConvergeSmallDubinsCarCAvoid;
		bool Test_GPU_PHASE2_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_stopConvergeSmallDubinsCarCAvoid_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
}

namespace UTest_CPU_woLastMinMax_HJIPDE_solve
{		
	class UnitTest_CPU_woLastMinMax_HJIPDE_solve_minWith
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_minWith;
		bool Test_CPU_woLastMinMax_HJIPDE_solve_minWith_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_minWith_Cylinder_1.mat"),
				std::string("./inputs/HJIPDE_solve_test_minWith_Cylinder_2.mat")
			};
			std::string message;
			if(!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_CPU_woLastMinMax_HJIPDE_solve_tvTargets
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_tvTargets;
		bool Test_CPU_woLastMinMax_HJIPDE_solve_tvTargets_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_tvTargets_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_CPU_woLastMinMax_HJIPDE_solve_singleObs
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_singleObs;
		bool Test_CPU_woLastMinMax_HJIPDE_solve_singleObs_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_singleObs_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_CPU_woLastMinMax_HJIPDE_solve_tvObs
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_tvObs;
		bool Test_CPU_woLastMinMax_HJIPDE_solve_tvObs_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_tvObs_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_CPU_woLastMinMax_HJIPDE_solve_obs_stau
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_obs_stau;
		bool Test_CPU_woLastMinMax_HJIPDE_solve_obs_stau_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_obs_stau_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_CPU_woLastMinMax_HJIPDE_solve_stopInit
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopInit;
		bool Test_CPU_woLastMinMax_HJIPDE_solve_stopInit_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_stopInit_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_CPU_woLastMinMax_HJIPDE_solve_stopSetInclude
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopSetInclude;
		bool Test_CPU_woLastMinMax_HJIPDE_solve_stopSetInclude_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_stopSetInclude_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_CPU_woLastMinMax_HJIPDE_solve_stopSetIntersect
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopSetIntersect;
		bool Test_CPU_woLastMinMax_HJIPDE_solve_stopSetIntersect_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_stopSetIntersect_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_CPU_woLastMinMax_HJIPDE_solve_plotData
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_plotData;
		bool Test_CPU_woLastMinMax_HJIPDE_solve_plotData_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_plotData_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_CPU_woLastMinMax_HJIPDE_solve_savedData
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_savedData;
		bool Test_CPU_woLastMinMax_HJIPDE_solve_savedData_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_savedData_Cylinder_1.mat"),
				std::string("./inputs/HJIPDE_solve_test_savedData_Cylinder_2.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_CPU_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar
    {
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopConvergeSmallDubinsCar;
		bool Test_CPU_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_stopConvergeSmallDubinsCar_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_CPU_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopConvergeSmallDubinsCarCAvoid;
		bool Test_CPU_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_stopConvergeSmallDubinsCarCAvoid_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
}

namespace UTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve
{		
	class UnitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_minWith
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_minWith;
		bool Test_GPU_PHASE1_woLastMinMax_HJIPDE_solve_minWith_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_minWith_Cylinder_1.mat"),
				std::string("./inputs/HJIPDE_solve_test_minWith_Cylinder_2.mat")
			};
			std::string message;
			if(!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_tvTargets
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_tvTargets;
		bool Test_GPU_PHASE1_woLastMinMax_HJIPDE_solve_tvTargets_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_tvTargets_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_singleObs
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_singleObs;
		bool Test_GPU_PHASE1_woLastMinMax_HJIPDE_solve_singleObs_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_singleObs_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_tvObs
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_tvObs;
		bool Test_GPU_PHASE1_woLastMinMax_HJIPDE_solve_tvObs_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_tvObs_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_obs_stau
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_obs_stau;
		bool Test_GPU_PHASE1_woLastMinMax_HJIPDE_solve_obs_stau_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_obs_stau_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_stopInit
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopInit;
		bool Test_GPU_PHASE1_woLastMinMax_HJIPDE_solve_stopInit_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_stopInit_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_stopSetInclude
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopSetInclude;
		bool Test_GPU_PHASE1_woLastMinMax_HJIPDE_solve_stopSetInclude_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_stopSetInclude_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_stopSetIntersect
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopSetIntersect;
	  bool Test_GPU_PHASE1_woLastMinMax_HJIPDE_solve_stopSetIntersect_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_stopSetIntersect_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_plotData
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_plotData;
		bool Test_GPU_PHASE1_woLastMinMax_HJIPDE_solve_plotData_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_plotData_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_savedData
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_savedData;
		bool Test_GPU_PHASE1_woLastMinMax_HJIPDE_solve_savedData_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_savedData_Cylinder_1.mat"),
				std::string("./inputs/HJIPDE_solve_test_savedData_Cylinder_2.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar
    {
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopConvergeSmallDubinsCar;
		bool Test_GPU_PHASE1_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_stopConvergeSmallDubinsCar_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopConvergeSmallDubinsCarCAvoid;
		bool Test_GPU_PHASE1_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_stopConvergeSmallDubinsCarCAvoid_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
}


namespace UTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve
{		
	class UnitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_minWith
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_minWith;
		bool Test_GPU_PHASE2_woLastMinMax_HJIPDE_solve_minWith_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_minWith_Cylinder_1.mat"),
				std::string("./inputs/HJIPDE_solve_test_minWith_Cylinder_2.mat")
			};
			std::string message;
			if(!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_tvTargets
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_tvTargets;
		bool Test_GPU_PHASE2_woLastMinMax_HJIPDE_solve_tvTargets_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_tvTargets_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_singleObs
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_singleObs;
		bool Test_GPU_PHASE2_woLastMinMax_HJIPDE_solve_singleObs_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_singleObs_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_tvObs
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_tvObs;
		bool Test_GPU_PHASE2_woLastMinMax_HJIPDE_solve_tvObs_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_tvObs_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_obs_stau
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_obs_stau;
		bool Test_GPU_PHASE2_woLastMinMax_HJIPDE_solve_obs_stau_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_obs_stau_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_stopInit
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopInit;
		bool Test_GPU_PHASE2_woLastMinMax_HJIPDE_solve_stopInit_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_stopInit_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_stopSetInclude
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopSetInclude;
		bool Test_GPU_PHASE2_woLastMinMax_HJIPDE_solve_stopSetInclude_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_stopSetInclude_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_stopSetIntersect
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopSetIntersect;
	  bool Test_GPU_PHASE2_woLastMinMax_HJIPDE_solve_stopSetIntersect_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_stopSetIntersect_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_plotData
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_plotData;
		bool Test_GPU_PHASE2_woLastMinMax_HJIPDE_solve_plotData_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_plotData_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_savedData
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_savedData;
		bool Test_GPU_PHASE2_woLastMinMax_HJIPDE_solve_savedData_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_savedData_Cylinder_1.mat"),
				std::string("./inputs/HJIPDE_solve_test_savedData_Cylinder_2.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar
    {
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopConvergeSmallDubinsCar;
		bool Test_GPU_PHASE2_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_stopConvergeSmallDubinsCar_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
	class UnitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopConvergeSmallDubinsCarCAvoid;
		bool Test_GPU_PHASE2_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid_Cylinder(const FLOAT_TYPE small_diff, const size_t chunk_size, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." << std::endl;
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("./inputs/HJIPDE_solve_test_stopConvergeSmallDubinsCarCAvoid_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
}

int main(int argc, char *argv[])
{
	FLOAT_TYPE small_diff = 1e-13;
	if (argc >= 2) {
		small_diff = static_cast<FLOAT_TYPE>(atof(argv[1]));
	}
	size_t chunk_size = 8;
	if (argc >= 3) {
		chunk_size = atoi(argv[2]);
	}
	int num_of_threads = 0;
	if (argc >= 4) {
		num_of_threads = atoi(argv[3]);
	}
	int num_of_gpus = 0;
	if (argc >= 5) {
		num_of_gpus = atoi(argv[4]);
	}

	UTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_minWith unitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_minWith;
	UTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_tvTargets unitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_tvTargets;
	UTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_singleObs unitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_singleObs;
	UTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_tvObs unitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_tvObs;
	UTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_obs_stau unitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_obs_stau;
	UTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_stopInit unitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_stopInit;
	UTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_stopSetInclude unitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_stopSetInclude;
	UTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_stopSetIntersect unitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_stopSetIntersect;
	UTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_plotData unitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_plotData;
	UTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_savedData unitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_savedData;
	UTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar unitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar;
	UTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid unitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid;

	UTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_minWith unitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_minWith;
	UTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_tvTargets unitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_tvTargets;
	UTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_singleObs unitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_singleObs;
	UTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_tvObs unitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_tvObs;
	UTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_obs_stau unitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_obs_stau;
	UTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_stopInit unitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_stopInit;
	UTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_stopSetInclude unitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_stopSetInclude;
	UTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_stopSetIntersect unitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_stopSetIntersect;
	UTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_plotData unitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_plotData;
	UTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_savedData unitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_savedData;
	UTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar unitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar;
	UTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid unitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid;

	UTest_CPU_woLastMinMax_HJIPDE_solve::UnitTest_CPU_woLastMinMax_HJIPDE_solve_minWith unitTest_CPU_woLastMinMax_HJIPDE_solve_minWith;
	UTest_CPU_woLastMinMax_HJIPDE_solve::UnitTest_CPU_woLastMinMax_HJIPDE_solve_tvTargets unitTest_CPU_woLastMinMax_HJIPDE_solve_tvTargets;
	UTest_CPU_woLastMinMax_HJIPDE_solve::UnitTest_CPU_woLastMinMax_HJIPDE_solve_singleObs unitTest_CPU_woLastMinMax_HJIPDE_solve_singleObs;
	UTest_CPU_woLastMinMax_HJIPDE_solve::UnitTest_CPU_woLastMinMax_HJIPDE_solve_tvObs unitTest_CPU_woLastMinMax_HJIPDE_solve_tvObs;
	UTest_CPU_woLastMinMax_HJIPDE_solve::UnitTest_CPU_woLastMinMax_HJIPDE_solve_obs_stau unitTest_CPU_woLastMinMax_HJIPDE_solve_obs_stau;
	UTest_CPU_woLastMinMax_HJIPDE_solve::UnitTest_CPU_woLastMinMax_HJIPDE_solve_stopInit unitTest_CPU_woLastMinMax_HJIPDE_solve_stopInit;
	UTest_CPU_woLastMinMax_HJIPDE_solve::UnitTest_CPU_woLastMinMax_HJIPDE_solve_stopSetInclude unitTest_CPU_woLastMinMax_HJIPDE_solve_stopSetInclude;
	UTest_CPU_woLastMinMax_HJIPDE_solve::UnitTest_CPU_woLastMinMax_HJIPDE_solve_stopSetIntersect unitTest_CPU_woLastMinMax_HJIPDE_solve_stopSetIntersect;
	UTest_CPU_woLastMinMax_HJIPDE_solve::UnitTest_CPU_woLastMinMax_HJIPDE_solve_plotData unitTest_CPU_woLastMinMax_HJIPDE_solve_plotData;
	UTest_CPU_woLastMinMax_HJIPDE_solve::UnitTest_CPU_woLastMinMax_HJIPDE_solve_savedData unitTest_CPU_woLastMinMax_HJIPDE_solve_savedData;
	UTest_CPU_woLastMinMax_HJIPDE_solve::UnitTest_CPU_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar unitTest_CPU_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar;
	UTest_CPU_woLastMinMax_HJIPDE_solve::UnitTest_CPU_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid unitTest_CPU_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid;

	UTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_minWith unitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_minWith;
	UTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_tvTargets unitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_tvTargets;
	UTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_singleObs unitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_singleObs;
	UTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_tvObs unitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_tvObs;
	UTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_obs_stau unitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_obs_stau;
	UTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_stopInit unitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_stopInit;
	UTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_stopSetInclude unitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_stopSetInclude;
	UTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_stopSetIntersect unitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_stopSetIntersect;
	UTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_plotData unitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_plotData;
	UTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_savedData unitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_savedData;
	UTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar unitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar;
	UTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid unitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid;

	UTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_minWith unitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_minWith;
	UTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_tvTargets unitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_tvTargets;
	UTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_singleObs unitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_singleObs;
	UTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_tvObs unitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_tvObs;
	UTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_obs_stau unitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_obs_stau;
	UTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_stopInit unitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_stopInit;
	UTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_stopSetInclude unitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_stopSetInclude;
	UTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_stopSetIntersect unitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_stopSetIntersect;
	UTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_plotData unitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_plotData;
	UTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_savedData unitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_savedData;
	UTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar unitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar;
	UTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve::UnitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid unitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid;

	UTest_CPU_wLastMinMax_HJIPDE_solve::UnitTest_CPU_wLastMinMax_HJIPDE_solve_minWith unitTest_CPU_wLastMinMax_HJIPDE_solve_minWith;
	UTest_CPU_wLastMinMax_HJIPDE_solve::UnitTest_CPU_wLastMinMax_HJIPDE_solve_tvTargets unitTest_CPU_wLastMinMax_HJIPDE_solve_tvTargets;
	UTest_CPU_wLastMinMax_HJIPDE_solve::UnitTest_CPU_wLastMinMax_HJIPDE_solve_singleObs unitTest_CPU_wLastMinMax_HJIPDE_solve_singleObs;
	UTest_CPU_wLastMinMax_HJIPDE_solve::UnitTest_CPU_wLastMinMax_HJIPDE_solve_tvObs unitTest_CPU_wLastMinMax_HJIPDE_solve_tvObs;
	UTest_CPU_wLastMinMax_HJIPDE_solve::UnitTest_CPU_wLastMinMax_HJIPDE_solve_obs_stau unitTest_CPU_wLastMinMax_HJIPDE_solve_obs_stau;
	UTest_CPU_wLastMinMax_HJIPDE_solve::UnitTest_CPU_wLastMinMax_HJIPDE_solve_stopInit unitTest_CPU_wLastMinMax_HJIPDE_solve_stopInit;
	UTest_CPU_wLastMinMax_HJIPDE_solve::UnitTest_CPU_wLastMinMax_HJIPDE_solve_stopSetInclude unitTest_CPU_wLastMinMax_HJIPDE_solve_stopSetInclude;
	UTest_CPU_wLastMinMax_HJIPDE_solve::UnitTest_CPU_wLastMinMax_HJIPDE_solve_stopSetIntersect unitTest_CPU_wLastMinMax_HJIPDE_solve_stopSetIntersect;
	UTest_CPU_wLastMinMax_HJIPDE_solve::UnitTest_CPU_wLastMinMax_HJIPDE_solve_plotData unitTest_CPU_wLastMinMax_HJIPDE_solve_plotData;
	UTest_CPU_wLastMinMax_HJIPDE_solve::UnitTest_CPU_wLastMinMax_HJIPDE_solve_savedData unitTest_CPU_wLastMinMax_HJIPDE_solve_savedData;
	UTest_CPU_wLastMinMax_HJIPDE_solve::UnitTest_CPU_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar unitTest_CPU_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar;
	UTest_CPU_wLastMinMax_HJIPDE_solve::UnitTest_CPU_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid unitTest_CPU_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid;


	size_t num_of_tests = 0;
	size_t num_of_succeeded = 0;

	++num_of_tests;
	if(unitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_minWith.Test_GPU_PHASE2_wLastMinMax_HJIPDE_solve_minWith_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_tvTargets.Test_GPU_PHASE2_wLastMinMax_HJIPDE_solve_tvTargets_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_singleObs.Test_GPU_PHASE2_wLastMinMax_HJIPDE_solve_singleObs_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_tvObs.Test_GPU_PHASE2_wLastMinMax_HJIPDE_solve_tvObs_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_obs_stau.Test_GPU_PHASE2_wLastMinMax_HJIPDE_solve_obs_stau_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_stopInit.Test_GPU_PHASE2_wLastMinMax_HJIPDE_solve_stopInit_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_stopSetInclude.Test_GPU_PHASE2_wLastMinMax_HJIPDE_solve_stopSetInclude_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_plotData.Test_GPU_PHASE2_wLastMinMax_HJIPDE_solve_plotData_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_savedData.Test_GPU_PHASE2_wLastMinMax_HJIPDE_solve_savedData_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar.Test_GPU_PHASE2_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid.Test_GPU_PHASE2_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	 ++num_of_tests;

	++num_of_tests;
	if(unitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_minWith.Test_GPU_PHASE1_wLastMinMax_HJIPDE_solve_minWith_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_tvTargets.Test_GPU_PHASE1_wLastMinMax_HJIPDE_solve_tvTargets_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_singleObs.Test_GPU_PHASE1_wLastMinMax_HJIPDE_solve_singleObs_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_tvObs.Test_GPU_PHASE1_wLastMinMax_HJIPDE_solve_tvObs_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_obs_stau.Test_GPU_PHASE1_wLastMinMax_HJIPDE_solve_obs_stau_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_stopInit.Test_GPU_PHASE1_wLastMinMax_HJIPDE_solve_stopInit_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_stopSetInclude.Test_GPU_PHASE1_wLastMinMax_HJIPDE_solve_stopSetInclude_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_plotData.Test_GPU_PHASE1_wLastMinMax_HJIPDE_solve_plotData_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_savedData.Test_GPU_PHASE1_wLastMinMax_HJIPDE_solve_savedData_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar.Test_GPU_PHASE1_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid.Test_GPU_PHASE1_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	 ++num_of_tests;

	++num_of_tests;
	if(unitTest_CPU_wLastMinMax_HJIPDE_solve_minWith.Test_CPU_wLastMinMax_HJIPDE_solve_minWith_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_wLastMinMax_HJIPDE_solve_tvTargets.Test_CPU_wLastMinMax_HJIPDE_solve_tvTargets_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_wLastMinMax_HJIPDE_solve_singleObs.Test_CPU_wLastMinMax_HJIPDE_solve_singleObs_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_wLastMinMax_HJIPDE_solve_tvObs.Test_CPU_wLastMinMax_HJIPDE_solve_tvObs_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_wLastMinMax_HJIPDE_solve_obs_stau.Test_CPU_wLastMinMax_HJIPDE_solve_obs_stau_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_wLastMinMax_HJIPDE_solve_stopInit.Test_CPU_wLastMinMax_HJIPDE_solve_stopInit_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_wLastMinMax_HJIPDE_solve_stopSetInclude.Test_CPU_wLastMinMax_HJIPDE_solve_stopSetInclude_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_wLastMinMax_HJIPDE_solve_plotData.Test_CPU_wLastMinMax_HJIPDE_solve_plotData_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_wLastMinMax_HJIPDE_solve_savedData.Test_CPU_wLastMinMax_HJIPDE_solve_savedData_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar.Test_CPU_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid.Test_CPU_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	 ++num_of_tests;

	++num_of_tests;
	if(unitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_minWith.Test_GPU_PHASE2_woLastMinMax_HJIPDE_solve_minWith_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_tvTargets.Test_GPU_PHASE2_woLastMinMax_HJIPDE_solve_tvTargets_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_singleObs.Test_GPU_PHASE2_woLastMinMax_HJIPDE_solve_singleObs_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_tvObs.Test_GPU_PHASE2_woLastMinMax_HJIPDE_solve_tvObs_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_obs_stau.Test_GPU_PHASE2_woLastMinMax_HJIPDE_solve_obs_stau_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_stopInit.Test_GPU_PHASE2_woLastMinMax_HJIPDE_solve_stopInit_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_stopSetInclude.Test_GPU_PHASE2_woLastMinMax_HJIPDE_solve_stopSetInclude_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_plotData.Test_GPU_PHASE2_woLastMinMax_HJIPDE_solve_plotData_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_savedData.Test_GPU_PHASE2_woLastMinMax_HJIPDE_solve_savedData_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar.Test_GPU_PHASE2_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid.Test_GPU_PHASE2_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	 ++num_of_tests;

	++num_of_tests;
	if(unitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_minWith.Test_GPU_PHASE1_woLastMinMax_HJIPDE_solve_minWith_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_tvTargets.Test_GPU_PHASE1_woLastMinMax_HJIPDE_solve_tvTargets_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_singleObs.Test_GPU_PHASE1_woLastMinMax_HJIPDE_solve_singleObs_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_tvObs.Test_GPU_PHASE1_woLastMinMax_HJIPDE_solve_tvObs_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_obs_stau.Test_GPU_PHASE1_woLastMinMax_HJIPDE_solve_obs_stau_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_stopInit.Test_GPU_PHASE1_woLastMinMax_HJIPDE_solve_stopInit_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_stopSetInclude.Test_GPU_PHASE1_woLastMinMax_HJIPDE_solve_stopSetInclude_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_plotData.Test_GPU_PHASE1_woLastMinMax_HJIPDE_solve_plotData_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_savedData.Test_GPU_PHASE1_woLastMinMax_HJIPDE_solve_savedData_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar.Test_GPU_PHASE1_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid.Test_GPU_PHASE1_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	 ++num_of_tests;

	++num_of_tests;
	if(unitTest_CPU_woLastMinMax_HJIPDE_solve_minWith.Test_CPU_woLastMinMax_HJIPDE_solve_minWith_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_woLastMinMax_HJIPDE_solve_tvTargets.Test_CPU_woLastMinMax_HJIPDE_solve_tvTargets_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_woLastMinMax_HJIPDE_solve_singleObs.Test_CPU_woLastMinMax_HJIPDE_solve_singleObs_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_woLastMinMax_HJIPDE_solve_tvObs.Test_CPU_woLastMinMax_HJIPDE_solve_tvObs_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_woLastMinMax_HJIPDE_solve_obs_stau.Test_CPU_woLastMinMax_HJIPDE_solve_obs_stau_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_woLastMinMax_HJIPDE_solve_stopInit.Test_CPU_woLastMinMax_HJIPDE_solve_stopInit_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_woLastMinMax_HJIPDE_solve_stopSetInclude.Test_CPU_woLastMinMax_HJIPDE_solve_stopSetInclude_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_woLastMinMax_HJIPDE_solve_plotData.Test_CPU_woLastMinMax_HJIPDE_solve_plotData_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_woLastMinMax_HJIPDE_solve_savedData.Test_CPU_woLastMinMax_HJIPDE_solve_savedData_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar.Test_CPU_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid.Test_CPU_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid_Cylinder(small_diff, chunk_size, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	 ++num_of_tests;
    
    if (num_of_tests == num_of_succeeded) {
		std::cout << "All test passed " << num_of_succeeded << " / " << num_of_tests << std::endl;
	} else {
		std::cout << "Passed " << num_of_succeeded << " / " << num_of_tests << std::endl;
	}

	return 0;

}

