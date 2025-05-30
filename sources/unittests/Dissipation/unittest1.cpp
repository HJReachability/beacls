//#include "stdafx.h"
#define _USE_MATH_DEFINES
#include <cmath>
#include "CppUnitTest.h"
#include "UTest_Dissipation.hpp"
#include <locale> 
#include <codecvt> 
#include <helperOC/DynSys/DynSys/DynSysSchemeData.hpp>
#include <helperOC/DynSys/DubinsCar/DubinsCar.hpp>
#include "../../samples/air3D/Air3DSchemeData.hpp"
#include "../../samples/plane4D/Plane4DSchemeData.hpp"

static const FLOAT_TYPE small_diff = (FLOAT_TYPE)1e-14;
static const int num_of_threads = 0;
static const int num_of_gpus = 0;

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace helperOC;
namespace UTest_CPU_Dissipation
{
	TEST_CLASS(UnitTest_CPU_DubinsCar_RS_GLF)
	{
	public:
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		const std::string src_data_filename = std::string("..\\..\\inputs\\DubinsCar_RS\\derivSrc_upwindFirstWENO5_15.mat");
		const std::vector<std::string> src_deriv_l_filenames{
			std::string("..\\..\\inputs\\DubinsCar_RS\\derivL_upwindFirstWENO5_15_1.mat"),
			std::string("..\\..\\inputs\\DubinsCar_RS\\derivL_upwindFirstWENO5_15_2.mat"),
			std::string("..\\..\\inputs\\DubinsCar_RS\\derivL_upwindFirstWENO5_15_3.mat")
		};
		const std::vector<std::string> src_deriv_r_filenames{
			std::string("..\\..\\inputs\\DubinsCar_RS\\derivR_upwindFirstWENO5_15_1.mat"),
			std::string("..\\..\\inputs\\DubinsCar_RS\\derivR_upwindFirstWENO5_15_2.mat"),
			std::string("..\\..\\inputs\\DubinsCar_RS\\derivR_upwindFirstWENO5_15_3.mat")
		};
		const std::string expects_filename = std::string("..\\..\\inputs\\DubinsCar_RS\\artificialDissipationGLF_upwindFirstWENO5_15.mat");
		const Dissipation_Class dissipation_Class = Dissipation_Class_ArtificialDissipationGLF;
		const beacls::FloatVec maxs{ (FLOAT_TYPE)15, (FLOAT_TYPE)18, (FLOAT_TYPE)M_PI };
		const beacls::FloatVec mins{ (FLOAT_TYPE)-21, (FLOAT_TYPE)-18, (FLOAT_TYPE)-M_PI };
		TEST_METHOD(Test_CPU_DubinsCar_RS_GLF_chunksize1)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 1, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_DubinsCar_RS_GLF_chunksize2)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 2, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_DubinsCar_RS_GLF_chunksize3)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 3, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_DubinsCar_RS_GLF_chunksize4)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 4, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_DubinsCar_RS_GLF_chunksize5)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 5, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_DubinsCar_RS_GLF_chunksize6)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 6, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_DubinsCar_RS_GLF_chunksize7)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 7, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_DubinsCar_RS_GLF_chunksize8)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 8, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_DubinsCar_RS_GLF_chunksize50)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 50, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_DubinsCar_RS_GLF_chunksize51)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 51, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_DubinsCar_RS_GLF_chunksize52)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 52, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_DubinsCar_RS_GLF_chunksize101)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 101, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_DubinsCar_RS_GLF_chunksize102)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 102, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_DubinsCar_RS_GLF_chunksize103)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 103, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_DubinsCar_RS_GLF_chunksize1580)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 1580, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_DubinsCar_RS_GLF_chunksize1581)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 1581, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_DubinsCar_RS_GLF_chunksize1582)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 1582, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_CPU_Air3D_GLF)
	{
	public:
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		const std::string src_data_filename = std::string("..\\..\\inputs\\air3D\\derivSrc_upwindFirstWENO5_2.7996.mat");
		const std::vector<std::string> src_deriv_l_filenames{
			std::string("..\\..\\inputs\\air3D\\derivL_upwindFirstWENO5_2.7996_1.mat"),
			std::string("..\\..\\inputs\\air3D\\derivL_upwindFirstWENO5_2.7996_2.mat"),
			std::string("..\\..\\inputs\\air3D\\derivL_upwindFirstWENO5_2.7996_3.mat")
		};
		const std::vector<std::string> src_deriv_r_filenames{
			std::string("..\\..\\inputs\\air3D\\derivR_upwindFirstWENO5_2.7996_1.mat"),
			std::string("..\\..\\inputs\\air3D\\derivR_upwindFirstWENO5_2.7996_2.mat"),
			std::string("..\\..\\inputs\\air3D\\derivR_upwindFirstWENO5_2.7996_3.mat")
		};
		const std::string expects_filename = std::string("..\\..\\inputs\\air3D\\artificialDissipationGLF_upwindFirstWENO5_2.7996.mat");
		const beacls::FloatVec maxs{ (FLOAT_TYPE)+20,(FLOAT_TYPE)+10,(FLOAT_TYPE)(+2 * M_PI) };
		const beacls::FloatVec mins{ (FLOAT_TYPE)-6,(FLOAT_TYPE)-10,(FLOAT_TYPE)0 };
		//! Speed of the evader(positive constant).
		const FLOAT_TYPE velocityA = 5;
		//! Speed of the pursuer(positive constant).
		const FLOAT_TYPE velocityB = 5;
		//! Maximum turn rate of the evader(positive).
		const FLOAT_TYPE inputA = 1;
		//! Maximum turn rate of the pursuer(positive).
		const FLOAT_TYPE inputB = 1;
		const Dissipation_Class dissipation_Class = Dissipation_Class_ArtificialDissipationGLF;
		TEST_METHOD(Test_CPU_Air3D_GLF_chunksize1)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 1, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_GLF_chunksize2)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 2, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_GLF_chunksize3)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 3, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_GLF_chunksize4)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 4, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_GLF_chunksize5)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 5, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_GLF_chunksize6)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 6, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_GLF_chunksize7)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 7, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_GLF_chunksize8)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 8, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_GLF_chunksize39)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 39, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_GLF_chunksize40)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 40, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_GLF_chunksize41)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 41, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_GLF_chunksize79)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 79, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_GLF_chunksize80)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 80, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_GLF_chunksize81)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 81, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_GLF_chunksize1999)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 1999, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_GLF_chunksize2000)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 2000, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Air3D_GLF_chunksize2001)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 2001, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};

	TEST_CLASS(UnitTest_CPU_Plane4D_GLF)
	{
	public:
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Vector;
		const std::string src_data_filename = std::string("..\\..\\inputs\\plane4D\\derivSrc_upwindFirstWENO5_1.mat");
		const std::vector<std::string> src_deriv_l_filenames{
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstWENO5_1_1.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstWENO5_1_2.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstWENO5_1_3.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstWENO5_1_4.mat"),
		};
		const std::vector<std::string> src_deriv_r_filenames{
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstWENO5_1_1.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstWENO5_1_2.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstWENO5_1_3.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstWENO5_1_4.mat"),
		};
		const std::string expects_filename = std::string("..\\..\\inputs\\plane4D\\artificialDissipationGLF_upwindFirstWENO5_1.mat");
		const beacls::FloatVec maxs{ (FLOAT_TYPE)+15,(FLOAT_TYPE)+10,(FLOAT_TYPE)(+2 * M_PI),(FLOAT_TYPE)12 };
		const beacls::FloatVec mins{ (FLOAT_TYPE)-5, (FLOAT_TYPE)-10, (FLOAT_TYPE)0,     (FLOAT_TYPE)6 };
		const FLOAT_TYPE wMax = 1;
		const beacls::FloatVec aranges{ (FLOAT_TYPE)0.5, 1 };
		const Dissipation_Class dissipation_Class = Dissipation_Class_ArtificialDissipationGLF;
		TEST_METHOD(Test_CPU_Plane4D_GLF_chunksize1)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 1, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_GLF_chunksize2)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 2, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_GLF_chunksize3)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 3, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_GLF_chunksize4)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 4, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_GLF_chunksize5)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 5, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_GLF_chunksize6)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 6, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_GLF_chunksize7)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 7, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_GLF_chunksize8)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 8, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_GLF_chunksize30)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 30, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_GLF_chunksize31)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 31, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_GLF_chunksize32)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 32, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_GLF_chunksize61)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 61, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_GLF_chunksize62)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 62, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_GLF_chunksize63)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 63, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_GLF_chunksize960)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 960, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_GLF_chunksize961)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 961, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_GLF_chunksize962)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 962, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_GLF_chunksize1921)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 1921, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_GLF_chunksize1922)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 1922, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_GLF_chunksize1923)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 1923, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_GLF_chunksize29790)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 29790, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_GLF_chunksize29791)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 29791, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_CPU_Plane4D_GLF_chunksize29792)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 29792, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
}

namespace UTest_GPU_PHASE1_Dissipation
{
	TEST_CLASS(UnitTest_GPU_PHASE1_DubinsCar_RS_GLF)
	{
	public:
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		const std::string src_data_filename = std::string("..\\..\\inputs\\DubinsCar_RS\\derivSrc_upwindFirstWENO5_15.mat");
		const std::vector<std::string> src_deriv_l_filenames{
			std::string("..\\..\\inputs\\DubinsCar_RS\\derivL_upwindFirstWENO5_15_1.mat"),
			std::string("..\\..\\inputs\\DubinsCar_RS\\derivL_upwindFirstWENO5_15_2.mat"),
			std::string("..\\..\\inputs\\DubinsCar_RS\\derivL_upwindFirstWENO5_15_3.mat")
		};
		const std::vector<std::string> src_deriv_r_filenames{
			std::string("..\\..\\inputs\\DubinsCar_RS\\derivR_upwindFirstWENO5_15_1.mat"),
			std::string("..\\..\\inputs\\DubinsCar_RS\\derivR_upwindFirstWENO5_15_2.mat"),
			std::string("..\\..\\inputs\\DubinsCar_RS\\derivR_upwindFirstWENO5_15_3.mat")
		};
		const std::string expects_filename = std::string("..\\..\\inputs\\DubinsCar_RS\\artificialDissipationGLF_upwindFirstWENO5_15.mat");
		const Dissipation_Class dissipation_Class = Dissipation_Class_ArtificialDissipationGLF;
		const beacls::FloatVec maxs{ (FLOAT_TYPE)15, (FLOAT_TYPE)18, (FLOAT_TYPE)M_PI };
		const beacls::FloatVec mins{ (FLOAT_TYPE)-21, (FLOAT_TYPE)-18, (FLOAT_TYPE)-M_PI };
		TEST_METHOD(Test_GPU_PHASE1_DubinsCar_RS_GLF_chunksize1)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 1, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_DubinsCar_RS_GLF_chunksize2)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 2, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_DubinsCar_RS_GLF_chunksize3)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 3, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_DubinsCar_RS_GLF_chunksize4)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 4, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_DubinsCar_RS_GLF_chunksize5)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 5, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_DubinsCar_RS_GLF_chunksize6)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 6, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_DubinsCar_RS_GLF_chunksize7)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 7, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_DubinsCar_RS_GLF_chunksize8)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 8, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_DubinsCar_RS_GLF_chunksize50)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 50, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_DubinsCar_RS_GLF_chunksize51)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 51, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_DubinsCar_RS_GLF_chunksize52)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 52, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_DubinsCar_RS_GLF_chunksize101)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 101, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_DubinsCar_RS_GLF_chunksize102)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 102, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_DubinsCar_RS_GLF_chunksize103)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 103, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_DubinsCar_RS_GLF_chunksize1580)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 1580, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_DubinsCar_RS_GLF_chunksize1581)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 1581, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_DubinsCar_RS_GLF_chunksize1582)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 1582, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE1_Air3D_GLF)
	{
	public:
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		const std::string src_data_filename = std::string("..\\..\\inputs\\air3D\\derivSrc_upwindFirstWENO5_2.7996.mat");
		const std::vector<std::string> src_deriv_l_filenames{
			std::string("..\\..\\inputs\\air3D\\derivL_upwindFirstWENO5_2.7996_1.mat"),
			std::string("..\\..\\inputs\\air3D\\derivL_upwindFirstWENO5_2.7996_2.mat"),
			std::string("..\\..\\inputs\\air3D\\derivL_upwindFirstWENO5_2.7996_3.mat")
		};
		const std::vector<std::string> src_deriv_r_filenames{
			std::string("..\\..\\inputs\\air3D\\derivR_upwindFirstWENO5_2.7996_1.mat"),
			std::string("..\\..\\inputs\\air3D\\derivR_upwindFirstWENO5_2.7996_2.mat"),
			std::string("..\\..\\inputs\\air3D\\derivR_upwindFirstWENO5_2.7996_3.mat")
		};
		const std::string expects_filename = std::string("..\\..\\inputs\\air3D\\artificialDissipationGLF_upwindFirstWENO5_2.7996.mat");
		const beacls::FloatVec maxs{ (FLOAT_TYPE)+20,(FLOAT_TYPE)+10,(FLOAT_TYPE)(+2 * M_PI) };
		const beacls::FloatVec mins{ (FLOAT_TYPE)-6,(FLOAT_TYPE)-10,(FLOAT_TYPE)0 };
		//! Speed of the evader(positive constant).
		const FLOAT_TYPE velocityA = 5;
		//! Speed of the pursuer(positive constant).
		const FLOAT_TYPE velocityB = 5;
		//! Maximum turn rate of the evader(positive).
		const FLOAT_TYPE inputA = 1;
		//! Maximum turn rate of the pursuer(positive).
		const FLOAT_TYPE inputB = 1;
		const Dissipation_Class dissipation_Class = Dissipation_Class_ArtificialDissipationGLF;
		TEST_METHOD(Test_GPU_PHASE1_Air3D_GLF_chunksize1)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 1, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_Air3D_GLF_chunksize2)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 2, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_Air3D_GLF_chunksize3)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 3, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_Air3D_GLF_chunksize4)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 4, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_Air3D_GLF_chunksize5)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 5, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_Air3D_GLF_chunksize6)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 6, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_Air3D_GLF_chunksize7)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 7, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_Air3D_GLF_chunksize8)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 8, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_Air3D_GLF_chunksize39)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 39, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_Air3D_GLF_chunksize40)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 40, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_Air3D_GLF_chunksize41)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 41, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_Air3D_GLF_chunksize79)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 79, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_Air3D_GLF_chunksize80)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 80, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_Air3D_GLF_chunksize81)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 81, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_Air3D_GLF_chunksize1999)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 1999, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_Air3D_GLF_chunksize2000)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 2000, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_Air3D_GLF_chunksize2001)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 2001, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};

	TEST_CLASS(UnitTest_GPU_PHASE1_Plane4D_GLF)
	{
	public:
		const bool enable_user_defined_dynamics_on_gpu = false;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		const std::string src_data_filename = std::string("..\\..\\inputs\\plane4D\\derivSrc_upwindFirstWENO5_1.mat");
		const std::vector<std::string> src_deriv_l_filenames{
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstWENO5_1_1.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstWENO5_1_2.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstWENO5_1_3.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstWENO5_1_4.mat"),
		};
		const std::vector<std::string> src_deriv_r_filenames{
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstWENO5_1_1.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstWENO5_1_2.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstWENO5_1_3.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstWENO5_1_4.mat"),
		};
		const std::string expects_filename = std::string("..\\..\\inputs\\plane4D\\artificialDissipationGLF_upwindFirstWENO5_1.mat");
		const beacls::FloatVec maxs{ (FLOAT_TYPE)+15,(FLOAT_TYPE)+10,(FLOAT_TYPE)(+2 * M_PI),(FLOAT_TYPE)12 };
		const beacls::FloatVec mins{ (FLOAT_TYPE)-5, (FLOAT_TYPE)-10, (FLOAT_TYPE)0,     (FLOAT_TYPE)6 };
		const FLOAT_TYPE wMax = 1;
		const beacls::FloatVec aranges{ (FLOAT_TYPE)0.5, 1 };
		const Dissipation_Class dissipation_Class = Dissipation_Class_ArtificialDissipationGLF;
		TEST_METHOD(Test_GPU_PHASE1_Plane4D_GLF_chunksize1)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 1, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_Plane4D_GLF_chunksize2)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 2, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_Plane4D_GLF_chunksize3)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 3, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_Plane4D_GLF_chunksize4)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 4, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_Plane4D_GLF_chunksize5)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 5, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_Plane4D_GLF_chunksize6)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 6, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_Plane4D_GLF_chunksize7)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 7, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_Plane4D_GLF_chunksize8)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 8, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_Plane4D_GLF_chunksize30)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 30, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_Plane4D_GLF_chunksize31)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 31, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_Plane4D_GLF_chunksize32)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 32, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_Plane4D_GLF_chunksize61)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 61, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_Plane4D_GLF_chunksize62)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 62, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_Plane4D_GLF_chunksize63)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 63, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_Plane4D_GLF_chunksize960)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 960, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_Plane4D_GLF_chunksize961)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 961, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_Plane4D_GLF_chunksize962)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 962, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_Plane4D_GLF_chunksize1921)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 1921, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_Plane4D_GLF_chunksize1922)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 1922, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_Plane4D_GLF_chunksize1923)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 1923, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_Plane4D_GLF_chunksize29790)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 29790, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_Plane4D_GLF_chunksize29791)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 29791, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE1_Plane4D_GLF_chunksize29792)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 29792, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
}

namespace UTest_GPU_PHASE2_Dissipation
{
	TEST_CLASS(UnitTest_GPU_PHASE2_DubinsCar_RS_GLF)
	{
	public:
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		const std::string src_data_filename = std::string("..\\..\\inputs\\DubinsCar_RS\\derivSrc_upwindFirstWENO5_15.mat");
		const std::vector<std::string> src_deriv_l_filenames{
			std::string("..\\..\\inputs\\DubinsCar_RS\\derivL_upwindFirstWENO5_15_1.mat"),
			std::string("..\\..\\inputs\\DubinsCar_RS\\derivL_upwindFirstWENO5_15_2.mat"),
			std::string("..\\..\\inputs\\DubinsCar_RS\\derivL_upwindFirstWENO5_15_3.mat")
		};
		const std::vector<std::string> src_deriv_r_filenames{
			std::string("..\\..\\inputs\\DubinsCar_RS\\derivR_upwindFirstWENO5_15_1.mat"),
			std::string("..\\..\\inputs\\DubinsCar_RS\\derivR_upwindFirstWENO5_15_2.mat"),
			std::string("..\\..\\inputs\\DubinsCar_RS\\derivR_upwindFirstWENO5_15_3.mat")
		};
		const std::string expects_filename = std::string("..\\..\\inputs\\DubinsCar_RS\\artificialDissipationGLF_upwindFirstWENO5_15.mat");
		const Dissipation_Class dissipation_Class = Dissipation_Class_ArtificialDissipationGLF;
		const beacls::FloatVec maxs{ (FLOAT_TYPE)15, (FLOAT_TYPE)18, (FLOAT_TYPE)M_PI };
		const beacls::FloatVec mins{ (FLOAT_TYPE)-21, (FLOAT_TYPE)-18, (FLOAT_TYPE)-M_PI };
		TEST_METHOD(Test_GPU_PHASE2_DubinsCar_RS_GLF_chunksize1)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 1, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_DubinsCar_RS_GLF_chunksize2)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 2, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_DubinsCar_RS_GLF_chunksize3)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 3, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_DubinsCar_RS_GLF_chunksize4)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 4, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_DubinsCar_RS_GLF_chunksize5)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 5, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_DubinsCar_RS_GLF_chunksize6)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 6, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_DubinsCar_RS_GLF_chunksize7)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 7, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_DubinsCar_RS_GLF_chunksize8)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 8, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_DubinsCar_RS_GLF_chunksize50)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 50, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_DubinsCar_RS_GLF_chunksize51)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 51, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_DubinsCar_RS_GLF_chunksize52)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 52, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_DubinsCar_RS_GLF_chunksize101)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 101, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_DubinsCar_RS_GLF_chunksize102)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 102, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_DubinsCar_RS_GLF_chunksize103)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 103, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_DubinsCar_RS_GLF_chunksize1580)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 1580, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_DubinsCar_RS_GLF_chunksize1581)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 1581, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_DubinsCar_RS_GLF_chunksize1582)
		{
			DynSysSchemeData* schemeData = new DynSysSchemeData();
			DubinsCar* dubinsCar = new DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, 1., 1.);
			schemeData->dynSys = dubinsCar;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 1582, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (dubinsCar) delete dubinsCar;
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
	TEST_CLASS(UnitTest_GPU_PHASE2_Air3D_GLF)
	{
	public:
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		const std::string src_data_filename = std::string("..\\..\\inputs\\air3D\\derivSrc_upwindFirstWENO5_2.7996.mat");
		const std::vector<std::string> src_deriv_l_filenames{
			std::string("..\\..\\inputs\\air3D\\derivL_upwindFirstWENO5_2.7996_1.mat"),
			std::string("..\\..\\inputs\\air3D\\derivL_upwindFirstWENO5_2.7996_2.mat"),
			std::string("..\\..\\inputs\\air3D\\derivL_upwindFirstWENO5_2.7996_3.mat")
		};
		const std::vector<std::string> src_deriv_r_filenames{
			std::string("..\\..\\inputs\\air3D\\derivR_upwindFirstWENO5_2.7996_1.mat"),
			std::string("..\\..\\inputs\\air3D\\derivR_upwindFirstWENO5_2.7996_2.mat"),
			std::string("..\\..\\inputs\\air3D\\derivR_upwindFirstWENO5_2.7996_3.mat")
		};
		const std::string expects_filename = std::string("..\\..\\inputs\\air3D\\artificialDissipationGLF_upwindFirstWENO5_2.7996.mat");
		const beacls::FloatVec maxs{ (FLOAT_TYPE)+20,(FLOAT_TYPE)+10,(FLOAT_TYPE)(+2 * M_PI) };
		const beacls::FloatVec mins{ (FLOAT_TYPE)-6,(FLOAT_TYPE)-10,(FLOAT_TYPE)0 };
		//! Speed of the evader(positive constant).
		const FLOAT_TYPE velocityA = 5;
		//! Speed of the pursuer(positive constant).
		const FLOAT_TYPE velocityB = 5;
		//! Maximum turn rate of the evader(positive).
		const FLOAT_TYPE inputA = 1;
		//! Maximum turn rate of the pursuer(positive).
		const FLOAT_TYPE inputB = 1;
		const Dissipation_Class dissipation_Class = Dissipation_Class_ArtificialDissipationGLF;
		TEST_METHOD(Test_GPU_PHASE2_Air3D_GLF_chunksize1)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 1, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_Air3D_GLF_chunksize2)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 2, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_Air3D_GLF_chunksize3)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 3, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_Air3D_GLF_chunksize4)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 4, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_Air3D_GLF_chunksize5)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 5, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_Air3D_GLF_chunksize6)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 6, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_Air3D_GLF_chunksize7)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 7, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_Air3D_GLF_chunksize8)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 8, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_Air3D_GLF_chunksize39)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 39, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_Air3D_GLF_chunksize40)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 40, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_Air3D_GLF_chunksize41)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 41, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_Air3D_GLF_chunksize79)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 79, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_Air3D_GLF_chunksize80)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 80, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_Air3D_GLF_chunksize81)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 81, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_Air3D_GLF_chunksize1999)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 1999, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_Air3D_GLF_chunksize2000)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 2000, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_Air3D_GLF_chunksize2001)
		{
			Air3DSchemeData* schemeData = new Air3DSchemeData();
			schemeData->velocityA = velocityA;
			schemeData->velocityB = velocityB;
			schemeData->inputA = inputA;
			schemeData->inputB = inputB;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 2001, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};

	TEST_CLASS(UnitTest_GPU_PHASE2_Plane4D_GLF)
	{
	public:
		const bool enable_user_defined_dynamics_on_gpu = true;
		const beacls::UVecType type = beacls::UVecType_Cuda;
		const std::string src_data_filename = std::string("..\\..\\inputs\\plane4D\\derivSrc_upwindFirstWENO5_1.mat");
		const std::vector<std::string> src_deriv_l_filenames{
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstWENO5_1_1.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstWENO5_1_2.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstWENO5_1_3.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivL_upwindFirstWENO5_1_4.mat"),
		};
		const std::vector<std::string> src_deriv_r_filenames{
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstWENO5_1_1.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstWENO5_1_2.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstWENO5_1_3.mat"),
			std::string("..\\..\\inputs\\plane4D\\derivR_upwindFirstWENO5_1_4.mat"),
		};
		const std::string expects_filename = std::string("..\\..\\inputs\\plane4D\\artificialDissipationGLF_upwindFirstWENO5_1.mat");
		const beacls::FloatVec maxs{ (FLOAT_TYPE)+15,(FLOAT_TYPE)+10,(FLOAT_TYPE)(+2 * M_PI),(FLOAT_TYPE)12 };
		const beacls::FloatVec mins{ (FLOAT_TYPE)-5, (FLOAT_TYPE)-10, (FLOAT_TYPE)0,     (FLOAT_TYPE)6 };
		const FLOAT_TYPE wMax = 1;
		const beacls::FloatVec aranges{ (FLOAT_TYPE)0.5, 1 };
		const Dissipation_Class dissipation_Class = Dissipation_Class_ArtificialDissipationGLF;
		TEST_METHOD(Test_GPU_PHASE2_Plane4D_GLF_chunksize1)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 1, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_Plane4D_GLF_chunksize2)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 2, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_Plane4D_GLF_chunksize3)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 3, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_Plane4D_GLF_chunksize4)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 4, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_Plane4D_GLF_chunksize5)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 5, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_Plane4D_GLF_chunksize6)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 6, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_Plane4D_GLF_chunksize7)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 7, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_Plane4D_GLF_chunksize8)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 8, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_Plane4D_GLF_chunksize30)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 30, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_Plane4D_GLF_chunksize31)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 31, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_Plane4D_GLF_chunksize32)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 32, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_Plane4D_GLF_chunksize61)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 61, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_Plane4D_GLF_chunksize62)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 62, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_Plane4D_GLF_chunksize63)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 63, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_Plane4D_GLF_chunksize960)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 960, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_Plane4D_GLF_chunksize961)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 961, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_Plane4D_GLF_chunksize962)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 962, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_Plane4D_GLF_chunksize1921)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 1921, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_Plane4D_GLF_chunksize1922)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 1922, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_Plane4D_GLF_chunksize1923)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 1923, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_Plane4D_GLF_chunksize29790)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 29790, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_Plane4D_GLF_chunksize29791)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 29791, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
		TEST_METHOD(Test_GPU_PHASE2_Plane4D_GLF_chunksize29792)
		{
			Plane4DSchemeData* schemeData = new Plane4DSchemeData();
			schemeData->wMax = wMax;
			schemeData->aranges = aranges;
			std::string message;
			bool result = true;
			if (!run_UTest_Dissipation(
				message,
				src_data_filename, src_deriv_l_filenames, src_deriv_r_filenames, expects_filename,
				maxs, mins, schemeData, dissipation_Class, type, small_diff, 29792, num_of_threads, num_of_gpus, enable_user_defined_dynamics_on_gpu)) {
				result = false;
			}
			if (schemeData) delete schemeData;
			if (!result) {
				std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> cv;
				std::wstring wsmessage = cv.from_bytes(message);
				__LineInfo lineInfo(__WFILE__, __func__, __LINE__);
				Assert::Fail(wsmessage.c_str(), &lineInfo);
			}
		}
	};
}


