//#include "stdafx.h"
#define _USE_MATH_DEFINES
#include <cmath>
#include "CppUnitTest.h"
#include "UTest_HJIPDE_solve.hpp"
#include <locale> 
#include <codecvt> 

static const FLOAT_TYPE small_diff = (FLOAT_TYPE)1e-13;
static const size_t chunk_size = 8;
static const int num_of_threads = 0;
static const int num_of_gpus = 0;

using namespace Microsoft::VisualStudio::CppUnitTestFramework;


namespace UTest_CPU_wLastMinMax_HJIPDE_solve
{		
	TEST_CLASS(UnitTest_CPU_wLastMinMax_HJIPDE_solve_minWith)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_minWith;
		TEST_METHOD(Test_CPU_wLastMinMax_HJIPDE_solve_minWith_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_minWith_Cylinder_1.mat"),
				std::string("..\\..\\inputs\\HJIPDE_solve_test_minWith_Cylinder_2.mat")
			};
			std::string message;
			if(!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_CPU_wLastMinMax_HJIPDE_solve_tvTargets)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_tvTargets;
		TEST_METHOD(Test_CPU_wLastMinMax_HJIPDE_solve_tvTargets_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_tvTargets_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_CPU_wLastMinMax_HJIPDE_solve_singleObs)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_singleObs;
		TEST_METHOD(Test_CPU_wLastMinMax_HJIPDE_solve_singleObs_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_singleObs_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_CPU_wLastMinMax_HJIPDE_solve_tvObs)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_tvObs;
		TEST_METHOD(Test_CPU_wLastMinMax_HJIPDE_solve_tvObs_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_tvObs_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_CPU_wLastMinMax_HJIPDE_solve_obs_stau)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_obs_stau;
		TEST_METHOD(Test_CPU_wLastMinMax_HJIPDE_solve_obs_stau_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_obs_stau_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_CPU_wLastMinMax_HJIPDE_solve_stopInit)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopInit;
		TEST_METHOD(Test_CPU_wLastMinMax_HJIPDE_solve_stopInit_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_stopInit_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_CPU_wLastMinMax_HJIPDE_solve_stopSetInclude)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopSetInclude;
		TEST_METHOD(Test_CPU_wLastMinMax_HJIPDE_solve_stopSetInclude_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_stopSetInclude_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_CPU_wLastMinMax_HJIPDE_solve_stopSetIntersect)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopSetIntersect;
		TEST_METHOD(Test_CPU_wLastMinMax_HJIPDE_solve_stopSetIntersect_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_stopSetIntersect_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_CPU_wLastMinMax_HJIPDE_solve_plotData)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_plotData;
		TEST_METHOD(Test_CPU_wLastMinMax_HJIPDE_solve_plotData_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_plotData_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_CPU_wLastMinMax_HJIPDE_solve_savedData)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_savedData;
		TEST_METHOD(Test_CPU_wLastMinMax_HJIPDE_solve_savedData_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_savedData_Cylinder_1.mat"),
				std::string("..\\..\\inputs\\HJIPDE_solve_test_savedData_Cylinder_2.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_CPU_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarC)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopConvergeSmallDubinsCar;
		TEST_METHOD(Test_CPU_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_stopConvergeSmallDubinsCar_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_CPU_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopConvergeSmallDubinsCarCAvoid;
		TEST_METHOD(Test_CPU_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_stopConvergeSmallDubinsCarCAvoid_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
}

namespace UTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve
{
	TEST_CLASS(UnitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_minWith)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_minWith;
		TEST_METHOD(Test_GPU_PHASE1_wLastMinMax_HJIPDE_solve_minWith_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_minWith_Cylinder_1.mat"),
				std::string("..\\..\\inputs\\HJIPDE_solve_test_minWith_Cylinder_2.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_tvTargets)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_tvTargets;
		TEST_METHOD(Test_GPU_PHASE1_wLastMinMax_HJIPDE_solve_tvTargets_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_tvTargets_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_singleObs)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_singleObs;
		TEST_METHOD(Test_GPU_PHASE1_wLastMinMax_HJIPDE_solve_singleObs_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_singleObs_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_tvObs)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_tvObs;
		TEST_METHOD(Test_GPU_PHASE1_wLastMinMax_HJIPDE_solve_tvObs_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_tvObs_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_obs_stau)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_obs_stau;
		TEST_METHOD(Test_GPU_PHASE1_wLastMinMax_HJIPDE_solve_obs_stau_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_obs_stau_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_stopInit)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopInit;
		TEST_METHOD(Test_GPU_PHASE1_wLastMinMax_HJIPDE_solve_stopInit_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_stopInit_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_stopSetInclude)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopSetInclude;
		TEST_METHOD(Test_GPU_PHASE1_wLastMinMax_HJIPDE_solve_stopSetInclude_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_stopSetInclude_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_stopSetIntersect)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopSetIntersect;
		TEST_METHOD(Test_GPU_PHASE1_wLastMinMax_HJIPDE_solve_stopSetIntersect_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_stopSetIntersect_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_plotData)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_plotData;
		TEST_METHOD(Test_GPU_PHASE1_wLastMinMax_HJIPDE_solve_plotData_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_plotData_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_savedData)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_savedData;
		TEST_METHOD(Test_GPU_PHASE1_wLastMinMax_HJIPDE_solve_savedData_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_savedData_Cylinder_1.mat"),
				std::string("..\\..\\inputs\\HJIPDE_solve_test_savedData_Cylinder_2.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarC)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopConvergeSmallDubinsCar;
		TEST_METHOD(Test_GPU_PHASE1_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_stopConvergeSmallDubinsCar_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE1_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopConvergeSmallDubinsCarCAvoid;
		TEST_METHOD(Test_GPU_PHASE1_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_stopConvergeSmallDubinsCarCAvoid_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
}


namespace UTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve
{
	TEST_CLASS(UnitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_minWith)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_minWith;
		TEST_METHOD(Test_GPU_PHASE2_wLastMinMax_HJIPDE_solve_minWith_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_minWith_Cylinder_1.mat"),
				std::string("..\\..\\inputs\\HJIPDE_solve_test_minWith_Cylinder_2.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_tvTargets)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_tvTargets;
		TEST_METHOD(Test_GPU_PHASE2_wLastMinMax_HJIPDE_solve_tvTargets_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_tvTargets_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_singleObs)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_singleObs;
		TEST_METHOD(Test_GPU_PHASE2_wLastMinMax_HJIPDE_solve_singleObs_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_singleObs_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_tvObs)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_tvObs;
		TEST_METHOD(Test_GPU_PHASE2_wLastMinMax_HJIPDE_solve_tvObs_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_tvObs_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_obs_stau)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_obs_stau;
		TEST_METHOD(Test_GPU_PHASE2_wLastMinMax_HJIPDE_solve_obs_stau_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_obs_stau_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_stopInit)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopInit;
		TEST_METHOD(Test_GPU_PHASE2_wLastMinMax_HJIPDE_solve_stopInit_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_stopInit_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_stopSetInclude)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopSetInclude;
		TEST_METHOD(Test_GPU_PHASE2_wLastMinMax_HJIPDE_solve_stopSetInclude_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_stopSetInclude_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_stopSetIntersect)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopSetIntersect;
		TEST_METHOD(Test_GPU_PHASE2_wLastMinMax_HJIPDE_solve_stopSetIntersect_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_stopSetIntersect_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_plotData)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_plotData;
		TEST_METHOD(Test_GPU_PHASE2_wLastMinMax_HJIPDE_solve_plotData_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_plotData_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_savedData)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_savedData;
		TEST_METHOD(Test_GPU_PHASE2_wLastMinMax_HJIPDE_solve_savedData_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_savedData_Cylinder_1.mat"),
				std::string("..\\..\\inputs\\HJIPDE_solve_test_savedData_Cylinder_2.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarC)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopConvergeSmallDubinsCar;
		TEST_METHOD(Test_GPU_PHASE2_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_stopConvergeSmallDubinsCar_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE2_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Always;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopConvergeSmallDubinsCarCAvoid;
		TEST_METHOD(Test_GPU_PHASE2_wLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_stopConvergeSmallDubinsCarCAvoid_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
}



namespace UTest_CPU_woLastMinMax_HJIPDE_solve
{		
	TEST_CLASS(UnitTest_CPU_woLastMinMax_HJIPDE_solve_minWith)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_minWith;
		TEST_METHOD(Test_CPU_woLastMinMax_HJIPDE_solve_minWith_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_minWith_Cylinder_1.mat"),
				std::string("..\\..\\inputs\\HJIPDE_solve_test_minWith_Cylinder_2.mat")
			};
			std::string message;
			if(!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_CPU_woLastMinMax_HJIPDE_solve_tvTargets)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_tvTargets;
		TEST_METHOD(Test_CPU_woLastMinMax_HJIPDE_solve_tvTargets_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_tvTargets_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_CPU_woLastMinMax_HJIPDE_solve_singleObs)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_singleObs;
		TEST_METHOD(Test_CPU_woLastMinMax_HJIPDE_solve_singleObs_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_singleObs_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_CPU_woLastMinMax_HJIPDE_solve_tvObs)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_tvObs;
		TEST_METHOD(Test_CPU_woLastMinMax_HJIPDE_solve_tvObs_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_tvObs_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_CPU_woLastMinMax_HJIPDE_solve_obs_stau)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_obs_stau;
		TEST_METHOD(Test_CPU_woLastMinMax_HJIPDE_solve_obs_stau_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_obs_stau_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_CPU_woLastMinMax_HJIPDE_solve_stopInit)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopInit;
		TEST_METHOD(Test_CPU_woLastMinMax_HJIPDE_solve_stopInit_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_stopInit_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_CPU_woLastMinMax_HJIPDE_solve_stopSetInclude)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopSetInclude;
		TEST_METHOD(Test_CPU_woLastMinMax_HJIPDE_solve_stopSetInclude_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_stopSetInclude_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_CPU_woLastMinMax_HJIPDE_solve_stopSetIntersect)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopSetIntersect;
		TEST_METHOD(Test_CPU_woLastMinMax_HJIPDE_solve_stopSetIntersect_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_stopSetIntersect_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_CPU_woLastMinMax_HJIPDE_solve_plotData)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_plotData;
		TEST_METHOD(Test_CPU_woLastMinMax_HJIPDE_solve_plotData_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_plotData_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_CPU_woLastMinMax_HJIPDE_solve_savedData)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_savedData;
		TEST_METHOD(Test_CPU_woLastMinMax_HJIPDE_solve_savedData_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_savedData_Cylinder_1.mat"),
				std::string("..\\..\\inputs\\HJIPDE_solve_test_savedData_Cylinder_2.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_CPU_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarC)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopConvergeSmallDubinsCar;
		TEST_METHOD(Test_CPU_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_stopConvergeSmallDubinsCar_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_CPU_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopConvergeSmallDubinsCarCAvoid;
		TEST_METHOD(Test_CPU_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_stopConvergeSmallDubinsCarCAvoid_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
}

namespace UTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve
{
	TEST_CLASS(UnitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_minWith)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_minWith;
		TEST_METHOD(Test_GPU_PHASE1_woLastMinMax_HJIPDE_solve_minWith_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_minWith_Cylinder_1.mat"),
				std::string("..\\..\\inputs\\HJIPDE_solve_test_minWith_Cylinder_2.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_tvTargets)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_tvTargets;
		TEST_METHOD(Test_GPU_PHASE1_woLastMinMax_HJIPDE_solve_tvTargets_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_tvTargets_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_singleObs)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_singleObs;
		TEST_METHOD(Test_GPU_PHASE1_woLastMinMax_HJIPDE_solve_singleObs_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_singleObs_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_tvObs)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_tvObs;
		TEST_METHOD(Test_GPU_PHASE1_woLastMinMax_HJIPDE_solve_tvObs_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_tvObs_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_obs_stau)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_obs_stau;
		TEST_METHOD(Test_GPU_PHASE1_woLastMinMax_HJIPDE_solve_obs_stau_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_obs_stau_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_stopInit)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopInit;
		TEST_METHOD(Test_GPU_PHASE1_woLastMinMax_HJIPDE_solve_stopInit_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_stopInit_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_stopSetInclude)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopSetInclude;
		TEST_METHOD(Test_GPU_PHASE1_woLastMinMax_HJIPDE_solve_stopSetInclude_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_stopSetInclude_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_stopSetIntersect)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopSetIntersect;
		TEST_METHOD(Test_GPU_PHASE1_woLastMinMax_HJIPDE_solve_stopSetIntersect_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_stopSetIntersect_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_plotData)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_plotData;
		TEST_METHOD(Test_GPU_PHASE1_woLastMinMax_HJIPDE_solve_plotData_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_plotData_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_savedData)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_savedData;
		TEST_METHOD(Test_GPU_PHASE1_woLastMinMax_HJIPDE_solve_savedData_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_savedData_Cylinder_1.mat"),
				std::string("..\\..\\inputs\\HJIPDE_solve_test_savedData_Cylinder_2.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarC)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopConvergeSmallDubinsCar;
		TEST_METHOD(Test_GPU_PHASE1_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_stopConvergeSmallDubinsCar_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE1_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopConvergeSmallDubinsCarCAvoid;
		TEST_METHOD(Test_GPU_PHASE1_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_stopConvergeSmallDubinsCarCAvoid_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
}


namespace UTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve
{
	TEST_CLASS(UnitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_minWith)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_minWith;
		TEST_METHOD(Test_GPU_PHASE2_woLastMinMax_HJIPDE_solve_minWith_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_minWith_Cylinder_1.mat"),
				std::string("..\\..\\inputs\\HJIPDE_solve_test_minWith_Cylinder_2.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_tvTargets)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_tvTargets;
		TEST_METHOD(Test_GPU_PHASE2_woLastMinMax_HJIPDE_solve_tvTargets_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_tvTargets_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_singleObs)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_singleObs;
		TEST_METHOD(Test_GPU_PHASE2_woLastMinMax_HJIPDE_solve_singleObs_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_singleObs_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_tvObs)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_tvObs;
		TEST_METHOD(Test_GPU_PHASE2_woLastMinMax_HJIPDE_solve_tvObs_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_tvObs_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_obs_stau)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_obs_stau;
		TEST_METHOD(Test_GPU_PHASE2_woLastMinMax_HJIPDE_solve_obs_stau_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_obs_stau_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_stopInit)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopInit;
		TEST_METHOD(Test_GPU_PHASE2_woLastMinMax_HJIPDE_solve_stopInit_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_stopInit_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_stopSetInclude)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopSetInclude;
		TEST_METHOD(Test_GPU_PHASE2_woLastMinMax_HJIPDE_solve_stopSetInclude_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_stopSetInclude_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_stopSetIntersect)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopSetIntersect;
		TEST_METHOD(Test_GPU_PHASE2_woLastMinMax_HJIPDE_solve_stopSetIntersect_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_stopSetIntersect_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_plotData)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_plotData;
		TEST_METHOD(Test_GPU_PHASE2_woLastMinMax_HJIPDE_solve_plotData_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_plotData_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_savedData)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_savedData;
		TEST_METHOD(Test_GPU_PHASE2_woLastMinMax_HJIPDE_solve_savedData_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_savedData_Cylinder_1.mat"),
				std::string("..\\..\\inputs\\HJIPDE_solve_test_savedData_Cylinder_2.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarC)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopConvergeSmallDubinsCar;
		TEST_METHOD(Test_GPU_PHASE2_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCar_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_stopConvergeSmallDubinsCar_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE2_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid)
	{
	public:
		const beacls::DelayedDerivMinMax_Type delayedDerivMinMax = beacls::DelayedDerivMinMax_Disable;
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		HJIPDE_solve_WhatTest HJIPDE_solve_whatTest = HJIPDE_solve_WhatTest_stopConvergeSmallDubinsCarCAvoid;
		TEST_METHOD(Test_GPU_PHASE2_woLastMinMax_HJIPDE_solve_stopConvergeSmallDubinsCarCAvoid_Cylinder)
		{
			HJIPDE_solve_Shape HJIPDE_solve_shape = HJIPDE_solve_Shape_Cylinder;
			std::vector<std::string> expects_filenames{
				std::string("..\\..\\inputs\\HJIPDE_solve_test_stopConvergeSmallDubinsCarCAvoid_Cylinder.mat")
			};
			std::string message;
			if (!run_UTest_HJIPDE_solve(message, expects_filenames, HJIPDE_solve_whatTest, HJIPDE_solve_shape, type, small_diff, chunk_size, num_of_threads, num_of_gpus, delayedDerivMinMax, enable_user_defined_dynamics_on_gpu)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
}
