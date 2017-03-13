//#include "stdafx.h"
#define _USE_MATH_DEFINES
#include <cmath>
#include "CppUnitTest.h"
#include "UTest_SpatialDerivative.hpp"
#include <locale> 
#include <codecvt> 

static const FLOAT_TYPE small_diff = 0;
static const int num_of_threads = 0;
static const int num_of_gpus = 0;

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace UTest_CPU_SpatialDerivative
{		
	TEST_CLASS(UnitTest_CPU_Air3D_UpwindFirstFirst)
	{
	public:
		const beacls::UVecType type = beacls::UVecType_Vector;
		const std::string src_filename = std::string("..\\..\\inputs\\air3D\\derivSrc_upwindFirstFirst_2.7991.mat");
		const std::vector<std::string> expects_l_filenames{
			std::string("..\\..\\inputs\\air3D\\derivL_upwindFirstFirst_2.7991_1.mat"),
			std::string("..\\..\\inputs\\air3D\\derivL_upwindFirstFirst_2.7991_2.mat"),
			std::string("..\\..\\inputs\\air3D\\derivL_upwindFirstFirst_2.7991_3.mat")
		};
		const std::vector<std::string> expects_r_filenames{
			std::string("..\\..\\inputs\\air3D\\derivR_upwindFirstFirst_2.7991_1.mat"),
			std::string("..\\..\\inputs\\air3D\\derivR_upwindFirstFirst_2.7991_2.mat"),
			std::string("..\\..\\inputs\\air3D\\derivR_upwindFirstFirst_2.7991_3.mat")
		};
		const beacls::FloatVec maxs{ (FLOAT_TYPE)+20,(FLOAT_TYPE)+10,(FLOAT_TYPE)(+2 * M_PI) };
		const beacls::FloatVec mins{ (FLOAT_TYPE)-6,(FLOAT_TYPE)-10,(FLOAT_TYPE)0 };
		const SpatialDerivative_Class spatialDerivative_Class = SpatialDerivative_Class_UpwindFirstFirst;
		const std::vector<BoundaryCondition_Class> boundaryCondition_Classes{
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostPeriodic
		};
		TEST_METHOD(Test_CPU_Air3D_upwindFirstFirst_chunksize1)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstFirst_chunksize2)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstFirst_chunksize3)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 3, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstFirst_chunksize4)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 4, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstFirst_chunksize5)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 5, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstFirst_chunksize6)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 6, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstFirst_chunksize7)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 7, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstFirst_chunksize8)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 8, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstFirst_chunksize39)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 39, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstFirst_chunksize40)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 40, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstFirst_chunksize41)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 41, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstFirst_chunksize79)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 79, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstFirst_chunksize80)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 80, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstFirst_chunksize81)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 81, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstFirst_chunksize1999)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1999, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstFirst_chunksize2000)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2000, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstFirst_chunksize2001)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2001, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}

	};
	TEST_CLASS(UnitTest_CPU_Air3D_UpwindFirstENO2)
	{
	public:
		const beacls::UVecType type = beacls::UVecType_Vector;
		const std::string src_filename = std::string("..\\..\\inputs\\air3D\\derivSrc_upwindFirstENO2_2.7991.mat");
		const std::vector<std::string> expects_l_filenames{
			std::string("..\\..\\inputs\\air3D\\derivL_upwindFirstENO2_2.7991_1.mat"),
			std::string("..\\..\\inputs\\air3D\\derivL_upwindFirstENO2_2.7991_2.mat"),
			std::string("..\\..\\inputs\\air3D\\derivL_upwindFirstENO2_2.7991_3.mat")
		};
		const std::vector<std::string> expects_r_filenames{
			std::string("..\\..\\inputs\\air3D\\derivR_upwindFirstENO2_2.7991_1.mat"),
			std::string("..\\..\\inputs\\air3D\\derivR_upwindFirstENO2_2.7991_2.mat"),
			std::string("..\\..\\inputs\\air3D\\derivR_upwindFirstENO2_2.7991_3.mat")
		};
		const beacls::FloatVec maxs{ (FLOAT_TYPE)+20,(FLOAT_TYPE)+10,(FLOAT_TYPE)(+2 * M_PI) };
		const beacls::FloatVec mins{ (FLOAT_TYPE)-6,(FLOAT_TYPE)-10,(FLOAT_TYPE)0 };
		const SpatialDerivative_Class spatialDerivative_Class = SpatialDerivative_Class_UpwindFirstENO2;
		const std::vector<BoundaryCondition_Class> boundaryCondition_Classes{
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostPeriodic
		};
		TEST_METHOD(Test_CPU_Air3D_upwindFirstENO2_chunksize1)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstENO2_chunksize2)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstENO2_chunksize3)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 3, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstENO2_chunksize4)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 4, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstENO2_chunksize5)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 5, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstENO2_chunksize6)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 6, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstENO2_chunksize7)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 7, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstENO2_chunksize8)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 8, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstENO2_chunksize39)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 39, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstENO2_chunksize40)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 40, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstENO2_chunksize41)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 41, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstENO2_chunksize79)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 79, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstENO2_chunksize80)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 80, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstENO2_chunksize81)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 81, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstENO2_chunksize1999)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1999, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstENO2_chunksize2000)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2000, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstENO2_chunksize2001)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2001, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}

	};
	TEST_CLASS(UnitTest_CPU_Air3D_UpwindFirstENO3)
	{
	public:
		const beacls::UVecType type = beacls::UVecType_Vector;
		const std::string src_filename = std::string("..\\..\\inputs\\air3D\\derivSrc_upwindFirstENO3_2.7996.mat");
		const std::vector<std::string> expects_l_filenames{
			std::string("..\\..\\inputs\\air3D\\derivL_upwindFirstENO3_2.7996_1.mat"),
			std::string("..\\..\\inputs\\air3D\\derivL_upwindFirstENO3_2.7996_2.mat"),
			std::string("..\\..\\inputs\\air3D\\derivL_upwindFirstENO3_2.7996_3.mat")
		};
		const std::vector<std::string> expects_r_filenames{
			std::string("..\\..\\inputs\\air3D\\derivR_upwindFirstENO3_2.7996_1.mat"),
			std::string("..\\..\\inputs\\air3D\\derivR_upwindFirstENO3_2.7996_2.mat"),
			std::string("..\\..\\inputs\\air3D\\derivR_upwindFirstENO3_2.7996_3.mat")
		};
		const beacls::FloatVec maxs{ (FLOAT_TYPE)+20,(FLOAT_TYPE)+10,(FLOAT_TYPE)(+2 * M_PI) };
		const beacls::FloatVec mins{ (FLOAT_TYPE)-6,(FLOAT_TYPE)-10,(FLOAT_TYPE)0 };
		const SpatialDerivative_Class spatialDerivative_Class = SpatialDerivative_Class_UpwindFirstENO3;
		const std::vector<BoundaryCondition_Class> boundaryCondition_Classes{
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostPeriodic
		};
		TEST_METHOD(Test_CPU_Air3D_upwindFirstENO3_chunksize1)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstENO3_chunksize2)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstENO3_chunksize3)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 3, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstENO3_chunksize4)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 4, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstENO3_chunksize5)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 5, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstENO3_chunksize6)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 6, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstENO3_chunksize7)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 7, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstENO3_chunksize8)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 8, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstENO3_chunksize39)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 39, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstENO3_chunksize40)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 40, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstENO3_chunksize41)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 41, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstENO3_chunksize79)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 79, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstENO3_chunksize80)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 80, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstENO3_chunksize81)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 81, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstENO3_chunksize1999)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1999, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstENO3_chunksize2000)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2000, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstENO3_chunksize2001)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2001, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}

	};
	TEST_CLASS(UnitTest_CPU_Air3D_UpwindFirstWENO5)
	{
	public:
		const beacls::UVecType type = beacls::UVecType_Vector;
		const std::string src_filename = std::string("..\\..\\inputs\\air3D\\derivSrc_upwindFirstWENO5_2.7996.mat");
		const std::vector<std::string> expects_l_filenames{
			std::string("..\\..\\inputs\\air3D\\derivL_upwindFirstWENO5_2.7996_1.mat"),
			std::string("..\\..\\inputs\\air3D\\derivL_upwindFirstWENO5_2.7996_2.mat"),
			std::string("..\\..\\inputs\\air3D\\derivL_upwindFirstWENO5_2.7996_3.mat")
		};
		const std::vector<std::string> expects_r_filenames{
			std::string("..\\..\\inputs\\air3D\\derivR_upwindFirstWENO5_2.7996_1.mat"),
			std::string("..\\..\\inputs\\air3D\\derivR_upwindFirstWENO5_2.7996_2.mat"),
			std::string("..\\..\\inputs\\air3D\\derivR_upwindFirstWENO5_2.7996_3.mat")
		};
		const beacls::FloatVec maxs{ (FLOAT_TYPE)+20,(FLOAT_TYPE)+10,(FLOAT_TYPE)(+2 * M_PI) };
		const beacls::FloatVec mins{ (FLOAT_TYPE)-6,(FLOAT_TYPE)-10,(FLOAT_TYPE)0 };
		const SpatialDerivative_Class spatialDerivative_Class = SpatialDerivative_Class_UpwindFirstWENO5;
		const std::vector<BoundaryCondition_Class> boundaryCondition_Classes{
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostPeriodic
		};
		TEST_METHOD(Test_CPU_Air3D_upwindFirstWENO5_chunksize1)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstWENO5_chunksize2)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstWENO5_chunksize3)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 3, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstWENO5_chunksize4)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 4, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstWENO5_chunksize5)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 5, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstWENO5_chunksize6)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 6, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstWENO5_chunksize7)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 7, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstWENO5_chunksize8)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 8, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstWENO5_chunksize39)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 39, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstWENO5_chunksize40)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 40, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstWENO5_chunksize41)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 41, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstWENO5_chunksize79)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 79, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstWENO5_chunksize80)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 80, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstWENO5_chunksize81)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 81, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstWENO5_chunksize1999)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1999, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstWENO5_chunksize2000)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2000, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_upwindFirstWENO5_chunksize2001)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2001, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}

	};
	TEST_CLASS(UnitTest_CPU_Plane4D_UpwindFirstFirst)
	{
	public:
		const beacls::UVecType type = beacls::UVecType_Vector;
		const std::string src_filename = std::string("..\\..\\inputs\\plane4D\\derivSrc_upwindFirstFirst_0.98713.mat");
		const std::vector<std::string> expects_l_filenames{
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstFirst_0.98713_1.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstFirst_0.98713_2.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstFirst_0.98713_3.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstFirst_0.98713_4.mat")
		};
		const std::vector<std::string> expects_r_filenames{
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstFirst_0.98713_1.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstFirst_0.98713_2.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstFirst_0.98713_3.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstFirst_0.98713_4.mat")
		};
		const beacls::FloatVec maxs{ (FLOAT_TYPE)+15,(FLOAT_TYPE)+10,(FLOAT_TYPE)(+2 * M_PI),(FLOAT_TYPE)12 };
		const beacls::FloatVec mins{ (FLOAT_TYPE)-5,(FLOAT_TYPE)-10, (FLOAT_TYPE)0,     (FLOAT_TYPE)6 };
		const SpatialDerivative_Class spatialDerivative_Class = SpatialDerivative_Class_UpwindFirstFirst;
		const std::vector<BoundaryCondition_Class> boundaryCondition_Classes{
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostPeriodic,
			BoundaryCondition_Class_AddGhostExtrapolate
		};
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstFirst_chunksize1)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstFirst_chunksize2)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstFirst_chunksize3)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 3, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstFirst_chunksize4)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 4, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstFirst_chunksize5)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 5, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstFirst_chunksize6)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 6, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstFirst_chunksize7)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 7, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstFirst_chunksize8)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 8, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstFirst_chunksize30)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 30, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstFirst_chunksize31)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 31, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstFirst_chunksize32)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 32, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstFirst_chunksize61)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 61, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstFirst_chunksize62)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 62, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstFirst_chunksize63)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 63, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstFirst_chunksize960)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 960, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstFirst_chunksize961)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 961, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstFirst_chunksize962)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 962, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstFirst_chunksize1921)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1921, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstFirst_chunksize1922)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1922, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstFirst_chunksize1923)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1923, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstFirst_chunksize29790)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29790, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstFirst_chunksize29791)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29791, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstFirst_chunksize29792)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29792, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}

	};
	TEST_CLASS(UnitTest_CPU_Plane4D_UpwindFirstENO2)
	{
	public:
		const beacls::UVecType type = beacls::UVecType_Vector;
		const std::string src_filename = std::string("..\\..\\inputs\\plane4D\\derivSrc_upwindFirstENO2_1.mat");
		const std::vector<std::string> expects_l_filenames{
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstENO2_1_1.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstENO2_1_2.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstENO2_1_3.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstENO2_1_4.mat")
		};
		const std::vector<std::string> expects_r_filenames{
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstENO2_1_1.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstENO2_1_2.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstENO2_1_3.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstENO2_1_4.mat")
		};
		const beacls::FloatVec maxs{ (FLOAT_TYPE)+15,(FLOAT_TYPE)+10,(FLOAT_TYPE)(+2 * M_PI),(FLOAT_TYPE)12 };
		const beacls::FloatVec mins{ (FLOAT_TYPE)-5,(FLOAT_TYPE)-10, (FLOAT_TYPE)0,     (FLOAT_TYPE)6 };
		const SpatialDerivative_Class spatialDerivative_Class = SpatialDerivative_Class_UpwindFirstENO2;
		const std::vector<BoundaryCondition_Class> boundaryCondition_Classes{
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostPeriodic,
			BoundaryCondition_Class_AddGhostExtrapolate
		};
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO2_chunksize1)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO2_chunksize2)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO2_chunksize3)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 3, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO2_chunksize4)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 4, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO2_chunksize5)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 5, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO2_chunksize6)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 6, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO2_chunksize7)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 7, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO2_chunksize8)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 8, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO2_chunksize30)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 30, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO2_chunksize31)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 31, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO2_chunksize32)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 32, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO2_chunksize61)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 61, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO2_chunksize62)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 62, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO2_chunksize63)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 63, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO2_chunksize960)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 960, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO2_chunksize961)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 961, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO2_chunksize962)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 962, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO2_chunksize1921)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1921, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO2_chunksize1922)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1922, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO2_chunksize1923)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1923, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO2_chunksize29790)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29790, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO2_chunksize29791)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29791, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO2_chunksize29792)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29792, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}

	};
	TEST_CLASS(UnitTest_CPU_Plane4D_UpwindFirstENO3)
	{
	public:
		const beacls::UVecType type = beacls::UVecType_Vector;
		const std::string src_filename = std::string("..\\..\\inputs\\plane4D\\derivSrc_upwindFirstENO3_1.mat");
		const std::vector<std::string> expects_l_filenames{
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstENO3_1_1.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstENO3_1_2.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstENO3_1_3.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstENO3_1_4.mat")
		};
		const std::vector<std::string> expects_r_filenames{
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstENO3_1_1.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstENO3_1_2.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstENO3_1_3.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstENO3_1_4.mat")
		};
		const beacls::FloatVec maxs{ (FLOAT_TYPE)+15,(FLOAT_TYPE)+10,(FLOAT_TYPE)(+2 * M_PI),(FLOAT_TYPE)12 };
		const beacls::FloatVec mins{ (FLOAT_TYPE)-5,(FLOAT_TYPE)-10, (FLOAT_TYPE)0,     (FLOAT_TYPE)6 };
		const SpatialDerivative_Class spatialDerivative_Class = SpatialDerivative_Class_UpwindFirstENO3;
		const std::vector<BoundaryCondition_Class> boundaryCondition_Classes{
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostPeriodic,
			BoundaryCondition_Class_AddGhostExtrapolate
		};
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO3_chunksize1)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO3_chunksize2)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO3_chunksize3)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 3, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO3_chunksize4)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 4, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO3_chunksize5)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 5, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO3_chunksize6)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 6, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO3_chunksize7)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 7, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO3_chunksize8)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 8, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO3_chunksize30)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 30, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO3_chunksize31)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 31, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO3_chunksize32)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 32, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO3_chunksize61)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 61, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO3_chunksize62)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 62, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO3_chunksize63)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 63, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO3_chunksize960)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 960, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO3_chunksize961)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 961, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO3_chunksize962)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 962, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO3_chunksize1921)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1921, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO3_chunksize1922)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1922, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO3_chunksize1923)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1923, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO3_chunksize29790)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29790, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO3_chunksize29791)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29791, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstENO3_chunksize29792)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29792, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}

	};
	TEST_CLASS(UnitTest_CPU_Plane4D_UpwindFirstWENO5)
	{
	public:
		const beacls::UVecType type = beacls::UVecType_Vector;
		const std::string src_filename = std::string("..\\..\\inputs\\plane4D\\derivSrc_upwindFirstWENO5_1.mat");
		const std::vector<std::string> expects_l_filenames{
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstWENO5_1_1.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstWENO5_1_2.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstWENO5_1_3.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstWENO5_1_4.mat")
		};
		const std::vector<std::string> expects_r_filenames{
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstWENO5_1_1.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstWENO5_1_2.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstWENO5_1_3.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstWENO5_1_4.mat")
		};
		const beacls::FloatVec maxs{ (FLOAT_TYPE)+15,(FLOAT_TYPE)+10,(FLOAT_TYPE)(+2 * M_PI),(FLOAT_TYPE)12 };
		const beacls::FloatVec mins{ (FLOAT_TYPE)-5,(FLOAT_TYPE)-10, (FLOAT_TYPE)0,     (FLOAT_TYPE)6 };
		const SpatialDerivative_Class spatialDerivative_Class = SpatialDerivative_Class_UpwindFirstWENO5;
		const std::vector<BoundaryCondition_Class> boundaryCondition_Classes{
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostPeriodic,
			BoundaryCondition_Class_AddGhostExtrapolate
		};
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstWENO5_chunksize1)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstWENO5_chunksize2)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstWENO5_chunksize3)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 3, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstWENO5_chunksize4)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 4, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstWENO5_chunksize5)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 5, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstWENO5_chunksize6)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 6, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstWENO5_chunksize7)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 7, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstWENO5_chunksize8)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 8, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstWENO5_chunksize30)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 30, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstWENO5_chunksize31)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 31, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstWENO5_chunksize32)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 32, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstWENO5_chunksize61)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 61, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstWENO5_chunksize62)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 62, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstWENO5_chunksize63)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 63, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstWENO5_chunksize960)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 960, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstWENO5_chunksize961)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 961, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstWENO5_chunksize962)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 962, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstWENO5_chunksize1921)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1921, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstWENO5_chunksize1922)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1922, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstWENO5_chunksize1923)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1923, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstWENO5_chunksize29790)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29790, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstWENO5_chunksize29791)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29791, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_upwindFirstWENO5_chunksize29792)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29792, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}

	};
}

namespace UTest_GPU_SpatialDerivative
{		
	TEST_CLASS(UnitTest_GPU_Air3D_UpwindFirstFirst)
	{
	public:
		const beacls::UVecType type = beacls::UVecType_Cuda;
		const std::string src_filename = std::string("..\\..\\inputs\\air3D\\derivSrc_upwindFirstFirst_2.7991.mat");
		const std::vector<std::string> expects_l_filenames{
			std::string("..\\..\\inputs\\air3D\\derivL_upwindFirstFirst_2.7991_1.mat"),
			std::string("..\\..\\inputs\\air3D\\derivL_upwindFirstFirst_2.7991_2.mat"),
			std::string("..\\..\\inputs\\air3D\\derivL_upwindFirstFirst_2.7991_3.mat")
		};
		const std::vector<std::string> expects_r_filenames{
			std::string("..\\..\\inputs\\air3D\\derivR_upwindFirstFirst_2.7991_1.mat"),
			std::string("..\\..\\inputs\\air3D\\derivR_upwindFirstFirst_2.7991_2.mat"),
			std::string("..\\..\\inputs\\air3D\\derivR_upwindFirstFirst_2.7991_3.mat")
		};
		const beacls::FloatVec maxs{ (FLOAT_TYPE)+20,(FLOAT_TYPE)+10,(FLOAT_TYPE)(+2 * M_PI) };
		const beacls::FloatVec mins{ (FLOAT_TYPE)-6,(FLOAT_TYPE)-10,(FLOAT_TYPE)0 };
		const SpatialDerivative_Class spatialDerivative_Class = SpatialDerivative_Class_UpwindFirstFirst;
		const std::vector<BoundaryCondition_Class> boundaryCondition_Classes{
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostPeriodic
		};
		TEST_METHOD(Test_GPU_Air3D_upwindFirstFirst_chunksize1)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstFirst_chunksize2)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstFirst_chunksize3)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 3, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstFirst_chunksize4)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 4, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstFirst_chunksize5)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 5, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstFirst_chunksize6)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 6, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstFirst_chunksize7)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 7, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstFirst_chunksize8)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 8, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstFirst_chunksize39)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 39, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstFirst_chunksize40)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 40, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstFirst_chunksize41)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 41, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstFirst_chunksize79)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 79, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstFirst_chunksize80)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 80, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstFirst_chunksize81)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 81, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstFirst_chunksize1999)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1999, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstFirst_chunksize2000)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2000, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstFirst_chunksize2001)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2001, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}

	};
	TEST_CLASS(UnitTest_GPU_Air3D_UpwindFirstENO2)
	{
	public:
		const beacls::UVecType type = beacls::UVecType_Cuda;
		const std::string src_filename = std::string("..\\..\\inputs\\air3D\\derivSrc_upwindFirstENO2_2.7991.mat");
		const std::vector<std::string> expects_l_filenames{
			std::string("..\\..\\inputs\\air3D\\derivL_upwindFirstENO2_2.7991_1.mat"),
			std::string("..\\..\\inputs\\air3D\\derivL_upwindFirstENO2_2.7991_2.mat"),
			std::string("..\\..\\inputs\\air3D\\derivL_upwindFirstENO2_2.7991_3.mat")
		};
		const std::vector<std::string> expects_r_filenames{
			std::string("..\\..\\inputs\\air3D\\derivR_upwindFirstENO2_2.7991_1.mat"),
			std::string("..\\..\\inputs\\air3D\\derivR_upwindFirstENO2_2.7991_2.mat"),
			std::string("..\\..\\inputs\\air3D\\derivR_upwindFirstENO2_2.7991_3.mat")
		};
		const beacls::FloatVec maxs{ (FLOAT_TYPE)+20,(FLOAT_TYPE)+10,(FLOAT_TYPE)(+2 * M_PI) };
		const beacls::FloatVec mins{ (FLOAT_TYPE)-6,(FLOAT_TYPE)-10,(FLOAT_TYPE)0 };
		const SpatialDerivative_Class spatialDerivative_Class = SpatialDerivative_Class_UpwindFirstENO2;
		const std::vector<BoundaryCondition_Class> boundaryCondition_Classes{
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostPeriodic
		};
		TEST_METHOD(Test_GPU_Air3D_upwindFirstENO2_chunksize1)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstENO2_chunksize2)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstENO2_chunksize3)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 3, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstENO2_chunksize4)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 4, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstENO2_chunksize5)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 5, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstENO2_chunksize6)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 6, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstENO2_chunksize7)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 7, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstENO2_chunksize8)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 8, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstENO2_chunksize39)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 39, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstENO2_chunksize40)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 40, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstENO2_chunksize41)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 41, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstENO2_chunksize79)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 79, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstENO2_chunksize80)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 80, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstENO2_chunksize81)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 81, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstENO2_chunksize1999)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1999, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstENO2_chunksize2000)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2000, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstENO2_chunksize2001)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2001, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}

	};
	TEST_CLASS(UnitTest_GPU_Air3D_UpwindFirstENO3)
	{
	public:
		const beacls::UVecType type = beacls::UVecType_Cuda;
		const std::string src_filename = std::string("..\\..\\inputs\\air3D\\derivSrc_upwindFirstENO3_2.7996.mat");
		const std::vector<std::string> expects_l_filenames{
			std::string("..\\..\\inputs\\air3D\\derivL_upwindFirstENO3_2.7996_1.mat"),
			std::string("..\\..\\inputs\\air3D\\derivL_upwindFirstENO3_2.7996_2.mat"),
			std::string("..\\..\\inputs\\air3D\\derivL_upwindFirstENO3_2.7996_3.mat")
		};
		const std::vector<std::string> expects_r_filenames{
			std::string("..\\..\\inputs\\air3D\\derivR_upwindFirstENO3_2.7996_1.mat"),
			std::string("..\\..\\inputs\\air3D\\derivR_upwindFirstENO3_2.7996_2.mat"),
			std::string("..\\..\\inputs\\air3D\\derivR_upwindFirstENO3_2.7996_3.mat")
		};
		const beacls::FloatVec maxs{ (FLOAT_TYPE)+20,(FLOAT_TYPE)+10,(FLOAT_TYPE)(+2 * M_PI) };
		const beacls::FloatVec mins{ (FLOAT_TYPE)-6,(FLOAT_TYPE)-10,(FLOAT_TYPE)0 };
		const SpatialDerivative_Class spatialDerivative_Class = SpatialDerivative_Class_UpwindFirstENO3;
		const std::vector<BoundaryCondition_Class> boundaryCondition_Classes{
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostPeriodic
		};
		TEST_METHOD(Test_GPU_Air3D_upwindFirstENO3_chunksize1)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstENO3_chunksize2)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstENO3_chunksize3)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 3, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstENO3_chunksize4)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 4, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstENO3_chunksize5)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 5, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstENO3_chunksize6)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 6, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstENO3_chunksize7)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 7, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstENO3_chunksize8)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 8, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstENO3_chunksize39)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 39, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstENO3_chunksize40)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 40, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstENO3_chunksize41)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 41, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstENO3_chunksize79)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 79, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstENO3_chunksize80)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 80, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstENO3_chunksize81)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 81, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstENO3_chunksize1999)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1999, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstENO3_chunksize2000)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2000, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstENO3_chunksize2001)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2001, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}

	};
	TEST_CLASS(UnitTest_GPU_Air3D_UpwindFirstWENO5)
	{
	public:
		const beacls::UVecType type = beacls::UVecType_Cuda;
		const std::string src_filename = std::string("..\\..\\inputs\\air3D\\derivSrc_upwindFirstWENO5_2.7996.mat");
		const std::vector<std::string> expects_l_filenames{
			std::string("..\\..\\inputs\\air3D\\derivL_upwindFirstWENO5_2.7996_1.mat"),
			std::string("..\\..\\inputs\\air3D\\derivL_upwindFirstWENO5_2.7996_2.mat"),
			std::string("..\\..\\inputs\\air3D\\derivL_upwindFirstWENO5_2.7996_3.mat")
		};
		const std::vector<std::string> expects_r_filenames{
			std::string("..\\..\\inputs\\air3D\\derivR_upwindFirstWENO5_2.7996_1.mat"),
			std::string("..\\..\\inputs\\air3D\\derivR_upwindFirstWENO5_2.7996_2.mat"),
			std::string("..\\..\\inputs\\air3D\\derivR_upwindFirstWENO5_2.7996_3.mat")
		};
		const beacls::FloatVec maxs{ (FLOAT_TYPE)+20,(FLOAT_TYPE)+10,(FLOAT_TYPE)(+2 * M_PI) };
		const beacls::FloatVec mins{ (FLOAT_TYPE)-6,(FLOAT_TYPE)-10,(FLOAT_TYPE)0 };
		const SpatialDerivative_Class spatialDerivative_Class = SpatialDerivative_Class_UpwindFirstWENO5;
		const std::vector<BoundaryCondition_Class> boundaryCondition_Classes{
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostPeriodic
		};
		TEST_METHOD(Test_GPU_Air3D_upwindFirstWENO5_chunksize1)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstWENO5_chunksize2)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstWENO5_chunksize3)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 3, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstWENO5_chunksize4)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 4, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstWENO5_chunksize5)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 5, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstWENO5_chunksize6)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 6, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstWENO5_chunksize7)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 7, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstWENO5_chunksize8)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 8, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstWENO5_chunksize39)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 39, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstWENO5_chunksize40)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 40, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstWENO5_chunksize41)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 41, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstWENO5_chunksize79)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 79, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstWENO5_chunksize80)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 80, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstWENO5_chunksize81)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 81, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstWENO5_chunksize1999)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1999, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstWENO5_chunksize2000)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2000, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Air3D_upwindFirstWENO5_chunksize2001)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2001, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}

	};
	TEST_CLASS(UnitTest_GPU_Plane4D_UpwindFirstFirst)
	{
	public:
		const beacls::UVecType type = beacls::UVecType_Cuda;
		const std::string src_filename = std::string("..\\..\\inputs\\plane4D\\derivSrc_upwindFirstFirst_0.98713.mat");
		const std::vector<std::string> expects_l_filenames{
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstFirst_0.98713_1.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstFirst_0.98713_2.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstFirst_0.98713_3.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstFirst_0.98713_4.mat")
		};
		const std::vector<std::string> expects_r_filenames{
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstFirst_0.98713_1.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstFirst_0.98713_2.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstFirst_0.98713_3.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstFirst_0.98713_4.mat")
		};
		const beacls::FloatVec maxs{ (FLOAT_TYPE)+15,(FLOAT_TYPE)+10,(FLOAT_TYPE)(+2 * M_PI),(FLOAT_TYPE)12 };
		const beacls::FloatVec mins{ (FLOAT_TYPE)-5,(FLOAT_TYPE)-10, (FLOAT_TYPE)0,     (FLOAT_TYPE)6 };
		const SpatialDerivative_Class spatialDerivative_Class = SpatialDerivative_Class_UpwindFirstFirst;
		const std::vector<BoundaryCondition_Class> boundaryCondition_Classes{
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostPeriodic,
			BoundaryCondition_Class_AddGhostExtrapolate
		};
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstFirst_chunksize1)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstFirst_chunksize2)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstFirst_chunksize3)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 3, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstFirst_chunksize4)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 4, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstFirst_chunksize5)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 5, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstFirst_chunksize6)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 6, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstFirst_chunksize7)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 7, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstFirst_chunksize8)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 8, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstFirst_chunksize30)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 30, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstFirst_chunksize31)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 31, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstFirst_chunksize32)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 32, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstFirst_chunksize61)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 61, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstFirst_chunksize62)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 62, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstFirst_chunksize63)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 63, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstFirst_chunksize960)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 960, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstFirst_chunksize961)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 961, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstFirst_chunksize962)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 962, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstFirst_chunksize1921)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1921, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstFirst_chunksize1922)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1922, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstFirst_chunksize1923)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1923, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstFirst_chunksize29790)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29790, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstFirst_chunksize29791)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29791, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstFirst_chunksize29792)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29792, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}

	};
	TEST_CLASS(UnitTest_GPU_Plane4D_UpwindFirstENO2)
	{
	public:
		const beacls::UVecType type = beacls::UVecType_Cuda;
		const std::string src_filename = std::string("..\\..\\inputs\\plane4D\\derivSrc_upwindFirstENO2_1.mat");
		const std::vector<std::string> expects_l_filenames{
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstENO2_1_1.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstENO2_1_2.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstENO2_1_3.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstENO2_1_4.mat")
		};
		const std::vector<std::string> expects_r_filenames{
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstENO2_1_1.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstENO2_1_2.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstENO2_1_3.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstENO2_1_4.mat")
		};
		const beacls::FloatVec maxs{ (FLOAT_TYPE)+15,(FLOAT_TYPE)+10,(FLOAT_TYPE)(+2 * M_PI),(FLOAT_TYPE)12 };
		const beacls::FloatVec mins{ (FLOAT_TYPE)-5,(FLOAT_TYPE)-10, (FLOAT_TYPE)0,     (FLOAT_TYPE)6 };
		const SpatialDerivative_Class spatialDerivative_Class = SpatialDerivative_Class_UpwindFirstENO2;
		const std::vector<BoundaryCondition_Class> boundaryCondition_Classes{
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostPeriodic,
			BoundaryCondition_Class_AddGhostExtrapolate
		};
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO2_chunksize1)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO2_chunksize2)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO2_chunksize3)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 3, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO2_chunksize4)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 4, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO2_chunksize5)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 5, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO2_chunksize6)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 6, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO2_chunksize7)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 7, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO2_chunksize8)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 8, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO2_chunksize30)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 30, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO2_chunksize31)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 31, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO2_chunksize32)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 32, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO2_chunksize61)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 61, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO2_chunksize62)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 62, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO2_chunksize63)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 63, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO2_chunksize960)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 960, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO2_chunksize961)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 961, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO2_chunksize962)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 962, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO2_chunksize1921)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1921, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO2_chunksize1922)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1922, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO2_chunksize1923)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1923, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO2_chunksize29790)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29790, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO2_chunksize29791)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29791, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO2_chunksize29792)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29792, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}

	};
	TEST_CLASS(UnitTest_GPU_Plane4D_UpwindFirstENO3)
	{
	public:
		const beacls::UVecType type = beacls::UVecType_Cuda;
		const std::string src_filename = std::string("..\\..\\inputs\\plane4D\\derivSrc_upwindFirstENO3_1.mat");
		const std::vector<std::string> expects_l_filenames{
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstENO3_1_1.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstENO3_1_2.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstENO3_1_3.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstENO3_1_4.mat")
		};
		const std::vector<std::string> expects_r_filenames{
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstENO3_1_1.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstENO3_1_2.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstENO3_1_3.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstENO3_1_4.mat")
		};
		const beacls::FloatVec maxs{ (FLOAT_TYPE)+15,(FLOAT_TYPE)+10,(FLOAT_TYPE)(+2 * M_PI),(FLOAT_TYPE)12 };
		const beacls::FloatVec mins{ (FLOAT_TYPE)-5,(FLOAT_TYPE)-10, (FLOAT_TYPE)0,     (FLOAT_TYPE)6 };
		const SpatialDerivative_Class spatialDerivative_Class = SpatialDerivative_Class_UpwindFirstENO3;
		const std::vector<BoundaryCondition_Class> boundaryCondition_Classes{
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostPeriodic,
			BoundaryCondition_Class_AddGhostExtrapolate
		};
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO3_chunksize1)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO3_chunksize2)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO3_chunksize3)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 3, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO3_chunksize4)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 4, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO3_chunksize5)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 5, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO3_chunksize6)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 6, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO3_chunksize7)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 7, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO3_chunksize8)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 8, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO3_chunksize30)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 30, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO3_chunksize31)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 31, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO3_chunksize32)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 32, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO3_chunksize61)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 61, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO3_chunksize62)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 62, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO3_chunksize63)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 63, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO3_chunksize960)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 960, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO3_chunksize961)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 961, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO3_chunksize962)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 962, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO3_chunksize1921)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1921, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO3_chunksize1922)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1922, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO3_chunksize1923)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1923, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO3_chunksize29790)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29790, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO3_chunksize29791)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29791, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstENO3_chunksize29792)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29792, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}

	};
	TEST_CLASS(UnitTest_GPU_Plane4D_UpwindFirstWENO5)
	{
	public:
		const beacls::UVecType type = beacls::UVecType_Cuda;
		const std::string src_filename = std::string("..\\..\\inputs\\plane4D\\derivSrc_upwindFirstWENO5_1.mat");
		const std::vector<std::string> expects_l_filenames{
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstWENO5_1_1.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstWENO5_1_2.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstWENO5_1_3.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstWENO5_1_4.mat")
		};
		const std::vector<std::string> expects_r_filenames{
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstWENO5_1_1.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstWENO5_1_2.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstWENO5_1_3.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstWENO5_1_4.mat")
		};
		const beacls::FloatVec maxs{ (FLOAT_TYPE)+15,(FLOAT_TYPE)+10,(FLOAT_TYPE)(+2 * M_PI),(FLOAT_TYPE)12 };
		const beacls::FloatVec mins{ (FLOAT_TYPE)-5,(FLOAT_TYPE)-10, (FLOAT_TYPE)0,     (FLOAT_TYPE)6 };
		const SpatialDerivative_Class spatialDerivative_Class = SpatialDerivative_Class_UpwindFirstWENO5;
		const std::vector<BoundaryCondition_Class> boundaryCondition_Classes{
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostPeriodic,
			BoundaryCondition_Class_AddGhostExtrapolate
		};
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstWENO5_chunksize1)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstWENO5_chunksize2)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstWENO5_chunksize3)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 3, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstWENO5_chunksize4)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 4, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstWENO5_chunksize5)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 5, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstWENO5_chunksize6)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 6, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstWENO5_chunksize7)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 7, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstWENO5_chunksize8)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 8, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstWENO5_chunksize30)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 30, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstWENO5_chunksize31)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 31, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstWENO5_chunksize32)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 32, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstWENO5_chunksize61)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 61, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstWENO5_chunksize62)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 62, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstWENO5_chunksize63)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 63, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstWENO5_chunksize960)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 960, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstWENO5_chunksize961)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 961, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstWENO5_chunksize962)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 962, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstWENO5_chunksize1921)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1921, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstWENO5_chunksize1922)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1922, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstWENO5_chunksize1923)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1923, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstWENO5_chunksize29790)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29790, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstWENO5_chunksize29791)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29791, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_Plane4D_upwindFirstWENO5_chunksize29792)
		{
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29792, num_of_threads, num_of_gpus)) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}

	};
}
