//#include "stdafx.h"
#define _USE_MATH_DEFINES
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include "UTest_SpatialDerivative.hpp"
using namespace helperOC;

namespace UTest_CPU_SpatialDerivative
{		
	class UnitTest_CPU_Air3D_UpwindFirstFirst
	{
	public:
		const beacls::UVecType type = beacls::UVecType_Vector;
		const std::string src_filename = std::string("./inputs/air3D/derivSrc_upwindFirstFirst_2.7991.mat");
		const std::vector<std::string> expects_l_filenames{
			std::string("./inputs/air3D/derivL_upwindFirstFirst_2.7991_1.mat"),
			std::string("./inputs/air3D/derivL_upwindFirstFirst_2.7991_2.mat"),
			std::string("./inputs/air3D/derivL_upwindFirstFirst_2.7991_3.mat")
		};
		const std::vector<std::string> expects_r_filenames{
			std::string("./inputs/air3D/derivR_upwindFirstFirst_2.7991_1.mat"),
			std::string("./inputs/air3D/derivR_upwindFirstFirst_2.7991_2.mat"),
			std::string("./inputs/air3D/derivR_upwindFirstFirst_2.7991_3.mat")
		};
		const std::vector <FLOAT_TYPE> maxs{ +20,+10,+2 * M_PI };
		const std::vector <FLOAT_TYPE> mins{ -6,-10,0 };
		SpatialDerivative_Class spatialDerivative_Class = SpatialDerivative_Class_UpwindFirstFirst;
		const std::vector<BoundaryCondition_Class> boundaryCondition_Classes{
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostPeriodic
		};
		bool Test_CPU_Air3D_upwindFirstFirst_chunksize1(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if(!run_UTest_SpatialDerivative(message,src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstFirst_chunksize2(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstFirst_chunksize3(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 3, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstFirst_chunksize4(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 4, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstFirst_chunksize5(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 5, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstFirst_chunksize6(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 6, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstFirst_chunksize7(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 7, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstFirst_chunksize39(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 39, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstFirst_chunksize40(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 40, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstFirst_chunksize41(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 41, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstFirst_chunksize79(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 79, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstFirst_chunksize80(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 80, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstFirst_chunksize81(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 81, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstFirst_chunksize1999(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1999, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstFirst_chunksize2000(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2000, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstFirst_chunksize2001(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2001, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstFirst_chunksize8(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 8, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}

	};
	class UnitTest_CPU_Air3D_UpwindFirstENO2
	{
	public:
		const beacls::UVecType type = beacls::UVecType_Vector;
		const std::string src_filename = std::string("./inputs/air3D/derivSrc_upwindFirstENO2_2.7991.mat");
		const std::vector<std::string> expects_l_filenames{
			std::string("./inputs/air3D/derivL_upwindFirstENO2_2.7991_1.mat"),
			std::string("./inputs/air3D/derivL_upwindFirstENO2_2.7991_2.mat"),
			std::string("./inputs/air3D/derivL_upwindFirstENO2_2.7991_3.mat")
		};
		const std::vector<std::string> expects_r_filenames{
			std::string("./inputs/air3D/derivR_upwindFirstENO2_2.7991_1.mat"),
			std::string("./inputs/air3D/derivR_upwindFirstENO2_2.7991_2.mat"),
			std::string("./inputs/air3D/derivR_upwindFirstENO2_2.7991_3.mat")
		};
		const std::vector <FLOAT_TYPE> maxs{ +20,+10,+2 * M_PI };
		const std::vector <FLOAT_TYPE> mins{ -6,-10,0 };
		SpatialDerivative_Class spatialDerivative_Class = SpatialDerivative_Class_UpwindFirstENO2;
		const std::vector<BoundaryCondition_Class> boundaryCondition_Classes{
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostPeriodic
		};
		bool Test_CPU_Air3D_upwindFirstENO2_chunksize1(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstENO2_chunksize2(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstENO2_chunksize3(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 3, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstENO2_chunksize4(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 4, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstENO2_chunksize5(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 5, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstENO2_chunksize6(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 6, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstENO2_chunksize7(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 7, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstENO2_chunksize39(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 39, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstENO2_chunksize40(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 40, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstENO2_chunksize41(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 41, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstENO2_chunksize79(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 79, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstENO2_chunksize80(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 80, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstENO2_chunksize81(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 81, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstENO2_chunksize1999(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1999, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstENO2_chunksize2000(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2000, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstENO2_chunksize2001(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2001, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstENO2_chunksize8(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 8, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}

	};
	class UnitTest_CPU_Air3D_UpwindFirstENO3
	{
	public:
		const beacls::UVecType type = beacls::UVecType_Vector;
		const std::string src_filename = std::string("./inputs/air3D/derivSrc_upwindFirstENO3_2.7996.mat");
		const std::vector<std::string> expects_l_filenames{
			std::string("./inputs/air3D/derivL_upwindFirstENO3_2.7996_1.mat"),
			std::string("./inputs/air3D/derivL_upwindFirstENO3_2.7996_2.mat"),
			std::string("./inputs/air3D/derivL_upwindFirstENO3_2.7996_3.mat")
		};
		const std::vector<std::string> expects_r_filenames{
			std::string("./inputs/air3D/derivR_upwindFirstENO3_2.7996_1.mat"),
			std::string("./inputs/air3D/derivR_upwindFirstENO3_2.7996_2.mat"),
			std::string("./inputs/air3D/derivR_upwindFirstENO3_2.7996_3.mat")
		};
		const std::vector <FLOAT_TYPE> maxs{ +20,+10,+2 * M_PI };
		const std::vector <FLOAT_TYPE> mins{ -6,-10,0 };
		SpatialDerivative_Class spatialDerivative_Class = SpatialDerivative_Class_UpwindFirstENO3;
		const std::vector<BoundaryCondition_Class> boundaryCondition_Classes{
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostPeriodic
		};
		bool Test_CPU_Air3D_upwindFirstENO3_chunksize1(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstENO3_chunksize2(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstENO3_chunksize3(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 3, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstENO3_chunksize4(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 4, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstENO3_chunksize5(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 5, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstENO3_chunksize6(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 6, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstENO3_chunksize7(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 7, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstENO3_chunksize39(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 39, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstENO3_chunksize40(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 40, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstENO3_chunksize41(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 41, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstENO3_chunksize79(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 79, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstENO3_chunksize80(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 80, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstENO3_chunksize81(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 81, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstENO3_chunksize1999(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1999, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstENO3_chunksize2000(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2000, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstENO3_chunksize2001(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2001, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstENO3_chunksize8(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 8, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}

	};
	class UnitTest_CPU_Air3D_UpwindFirstWENO5
	{
	public:
		const beacls::UVecType type = beacls::UVecType_Vector;
		const std::string src_filename = std::string("./inputs/air3D/derivSrc_upwindFirstWENO5_2.7996.mat");
		const std::vector<std::string> expects_l_filenames{
			std::string("./inputs/air3D/derivL_upwindFirstWENO5_2.7996_1.mat"),
			std::string("./inputs/air3D/derivL_upwindFirstWENO5_2.7996_2.mat"),
			std::string("./inputs/air3D/derivL_upwindFirstWENO5_2.7996_3.mat")
		};
		const std::vector<std::string> expects_r_filenames{
			std::string("./inputs/air3D/derivR_upwindFirstWENO5_2.7996_1.mat"),
			std::string("./inputs/air3D/derivR_upwindFirstWENO5_2.7996_2.mat"),
			std::string("./inputs/air3D/derivR_upwindFirstWENO5_2.7996_3.mat")
		};
		const std::vector <FLOAT_TYPE> maxs{ +20,+10,+2 * M_PI };
		const std::vector <FLOAT_TYPE> mins{ -6,-10,0 };
		SpatialDerivative_Class spatialDerivative_Class = SpatialDerivative_Class_UpwindFirstWENO5;
		const std::vector<BoundaryCondition_Class> boundaryCondition_Classes{
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostPeriodic
		};
		bool Test_CPU_Air3D_upwindFirstWENO5_chunksize1(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstWENO5_chunksize2(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstWENO5_chunksize3(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 3, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstWENO5_chunksize4(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 4, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstWENO5_chunksize5(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 5, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstWENO5_chunksize6(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 6, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstWENO5_chunksize7(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 7, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstWENO5_chunksize39(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 39, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstWENO5_chunksize40(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 40, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstWENO5_chunksize41(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 41, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstWENO5_chunksize79(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 79, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstWENO5_chunksize80(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 80, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstWENO5_chunksize81(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 81, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstWENO5_chunksize1999(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1999, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstWENO5_chunksize2000(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2000, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstWENO5_chunksize2001(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2001, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Air3D_upwindFirstWENO5_chunksize8(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 8, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}

	};
	class UnitTest_CPU_Plane4D_UpwindFirstFirst
	{
	public:
		const beacls::UVecType type = beacls::UVecType_Vector;
		const std::string src_filename = std::string("./inputs/plane4D/derivSrc_upwindFirstFirst_0.98713.mat");
		const std::vector<std::string> expects_l_filenames{
			std::string("./inputs/plane4D/derivL_upwindFirstFirst_0.98713_1.mat"),
			std::string("./inputs/plane4D/derivL_upwindFirstFirst_0.98713_2.mat"),
			std::string("./inputs/plane4D/derivL_upwindFirstFirst_0.98713_3.mat"),
			std::string("./inputs/plane4D/derivL_upwindFirstFirst_0.98713_4.mat")
		};
		const std::vector<std::string> expects_r_filenames{
			std::string("./inputs/plane4D/derivR_upwindFirstFirst_0.98713_1.mat"),
			std::string("./inputs/plane4D/derivR_upwindFirstFirst_0.98713_2.mat"),
			std::string("./inputs/plane4D/derivR_upwindFirstFirst_0.98713_3.mat"),
			std::string("./inputs/plane4D/derivR_upwindFirstFirst_0.98713_4.mat")
		};
		const std::vector <FLOAT_TYPE> maxs{ +15,+10,+2 * M_PI,12 };
		const std::vector <FLOAT_TYPE> mins{ -5, -10, 0,      6 };
		SpatialDerivative_Class spatialDerivative_Class = SpatialDerivative_Class_UpwindFirstFirst;
		const std::vector<BoundaryCondition_Class> boundaryCondition_Classes{
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostPeriodic,
			BoundaryCondition_Class_AddGhostExtrapolate
		};
		bool Test_CPU_Plane4D_upwindFirstFirst_chunksize1(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstFirst_chunksize2(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstFirst_chunksize3(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 3, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstFirst_chunksize4(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 4, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstFirst_chunksize5(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 5, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstFirst_chunksize6(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 6, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstFirst_chunksize7(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 7, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstFirst_chunksize30(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 30, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstFirst_chunksize31(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 31, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstFirst_chunksize32(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 32, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstFirst_chunksize61(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 61, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstFirst_chunksize62(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 62, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstFirst_chunksize63(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 63, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstFirst_chunksize960(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 960, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstFirst_chunksize961(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 961, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstFirst_chunksize962(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 962, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstFirst_chunksize1921(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1921, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstFirst_chunksize1922(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1922, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstFirst_chunksize1923(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1923, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstFirst_chunksize29790(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29790, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstFirst_chunksize29791(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29791, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstFirst_chunksize29792(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29792, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstFirst_chunksize8(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 8, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}

	};
	class UnitTest_CPU_Plane4D_UpwindFirstENO2
	{
	public:
		const beacls::UVecType type = beacls::UVecType_Vector;
		const std::string src_filename = std::string("./inputs/plane4D/derivSrc_upwindFirstENO2_1.mat");
		const std::vector<std::string> expects_l_filenames{
			std::string("./inputs/plane4D/derivL_upwindFirstENO2_1_1.mat"),
			std::string("./inputs/plane4D/derivL_upwindFirstENO2_1_2.mat"),
			std::string("./inputs/plane4D/derivL_upwindFirstENO2_1_3.mat"),
			std::string("./inputs/plane4D/derivL_upwindFirstENO2_1_4.mat")
		};
		const std::vector<std::string> expects_r_filenames{
			std::string("./inputs/plane4D/derivR_upwindFirstENO2_1_1.mat"),
			std::string("./inputs/plane4D/derivR_upwindFirstENO2_1_2.mat"),
			std::string("./inputs/plane4D/derivR_upwindFirstENO2_1_3.mat"),
			std::string("./inputs/plane4D/derivR_upwindFirstENO2_1_4.mat")
		};
		const std::vector <FLOAT_TYPE> maxs{ +15,+10,+2 * M_PI,12 };
		const std::vector <FLOAT_TYPE> mins{ -5, -10, 0,      6 };
		SpatialDerivative_Class spatialDerivative_Class = SpatialDerivative_Class_UpwindFirstENO2;
		const std::vector<BoundaryCondition_Class> boundaryCondition_Classes{
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostPeriodic,
			BoundaryCondition_Class_AddGhostExtrapolate
		};
		bool Test_CPU_Plane4D_upwindFirstENO2_chunksize1(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO2_chunksize2(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO2_chunksize3(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 3, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO2_chunksize4(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 4, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO2_chunksize5(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 5, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO2_chunksize6(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 6, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO2_chunksize7(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 7, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO2_chunksize30(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 30, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO2_chunksize31(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 31, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO2_chunksize32(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 32, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO2_chunksize61(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 61, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO2_chunksize62(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 62, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO2_chunksize63(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 63, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO2_chunksize960(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 960, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO2_chunksize961(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 961, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO2_chunksize962(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 962, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO2_chunksize1921(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1921, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO2_chunksize1922(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1922, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO2_chunksize1923(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1923, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO2_chunksize29790(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29790, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO2_chunksize29791(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29791, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO2_chunksize29792(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29792, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO2_chunksize8(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 8, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}

	};
	class UnitTest_CPU_Plane4D_UpwindFirstENO3
	{
	public:
		const beacls::UVecType type = beacls::UVecType_Vector;
		const std::string src_filename = std::string("./inputs/plane4D/derivSrc_upwindFirstENO3_1.mat");
		const std::vector<std::string> expects_l_filenames{
			std::string("./inputs/plane4D/derivL_upwindFirstENO3_1_1.mat"),
			std::string("./inputs/plane4D/derivL_upwindFirstENO3_1_2.mat"),
			std::string("./inputs/plane4D/derivL_upwindFirstENO3_1_3.mat"),
			std::string("./inputs/plane4D/derivL_upwindFirstENO3_1_4.mat")
		};
		const std::vector<std::string> expects_r_filenames{
			std::string("./inputs/plane4D/derivR_upwindFirstENO3_1_1.mat"),
			std::string("./inputs/plane4D/derivR_upwindFirstENO3_1_2.mat"),
			std::string("./inputs/plane4D/derivR_upwindFirstENO3_1_3.mat"),
			std::string("./inputs/plane4D/derivR_upwindFirstENO3_1_4.mat")
		};
		const std::vector <FLOAT_TYPE> maxs{ +15,+10,+2 * M_PI,12 };
		const std::vector <FLOAT_TYPE> mins{ -5, -10, 0,      6 };
		SpatialDerivative_Class spatialDerivative_Class = SpatialDerivative_Class_UpwindFirstENO3;
		const std::vector<BoundaryCondition_Class> boundaryCondition_Classes{
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostPeriodic,
			BoundaryCondition_Class_AddGhostExtrapolate
		};
		bool Test_CPU_Plane4D_upwindFirstENO3_chunksize1(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO3_chunksize2(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO3_chunksize3(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 3, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO3_chunksize4(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 4, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO3_chunksize5(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 5, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO3_chunksize6(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 6, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO3_chunksize7(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 7, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO3_chunksize30(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 30, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO3_chunksize31(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 31, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO3_chunksize32(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 32, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO3_chunksize61(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 61, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO3_chunksize62(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 62, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO3_chunksize63(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 63, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO3_chunksize960(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 960, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO3_chunksize961(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 961, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO3_chunksize962(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 962, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO3_chunksize1921(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1921, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO3_chunksize1922(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1922, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO3_chunksize1923(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1923, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO3_chunksize29790(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29790, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO3_chunksize29791(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29791, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO3_chunksize29792(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29792, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstENO3_chunksize8(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 8, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}

	};
	class UnitTest_CPU_Plane4D_UpwindFirstWENO5
	{
	public:
		const beacls::UVecType type = beacls::UVecType_Vector;
		const std::string src_filename = std::string("./inputs/plane4D/derivSrc_upwindFirstWENO5_1.mat");
		const std::vector<std::string> expects_l_filenames{
			std::string("./inputs/plane4D/derivL_upwindFirstWENO5_1_1.mat"),
			std::string("./inputs/plane4D/derivL_upwindFirstWENO5_1_2.mat"),
			std::string("./inputs/plane4D/derivL_upwindFirstWENO5_1_3.mat"),
			std::string("./inputs/plane4D/derivL_upwindFirstWENO5_1_4.mat")
		};
		const std::vector<std::string> expects_r_filenames{
			std::string("./inputs/plane4D/derivR_upwindFirstWENO5_1_1.mat"),
			std::string("./inputs/plane4D/derivR_upwindFirstWENO5_1_2.mat"),
			std::string("./inputs/plane4D/derivR_upwindFirstWENO5_1_3.mat"),
			std::string("./inputs/plane4D/derivR_upwindFirstWENO5_1_4.mat")
		};
		const std::vector <FLOAT_TYPE> maxs{ +15,+10,+2 * M_PI,12 };
		const std::vector <FLOAT_TYPE> mins{ -5, -10, 0,      6 };
		SpatialDerivative_Class spatialDerivative_Class = SpatialDerivative_Class_UpwindFirstWENO5;
		const std::vector<BoundaryCondition_Class> boundaryCondition_Classes{
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostPeriodic,
			BoundaryCondition_Class_AddGhostExtrapolate
		};
		bool Test_CPU_Plane4D_upwindFirstWENO5_chunksize1(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstWENO5_chunksize2(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstWENO5_chunksize3(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 3, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstWENO5_chunksize4(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 4, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstWENO5_chunksize5(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 5, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstWENO5_chunksize6(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 6, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstWENO5_chunksize7(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 7, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstWENO5_chunksize30(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 30, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstWENO5_chunksize31(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 31, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstWENO5_chunksize32(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 32, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstWENO5_chunksize61(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 61, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstWENO5_chunksize62(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 62, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstWENO5_chunksize63(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 63, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstWENO5_chunksize960(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 960, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstWENO5_chunksize961(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 961, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstWENO5_chunksize962(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 962, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstWENO5_chunksize1921(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1921, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstWENO5_chunksize1922(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1922, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstWENO5_chunksize1923(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1923, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstWENO5_chunksize29790(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29790, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstWENO5_chunksize29791(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29791, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstWENO5_chunksize29792(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29792, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_CPU_Plane4D_upwindFirstWENO5_chunksize8(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 8, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
	};
}
namespace UTest_GPU_SpatialDerivative
{
	class UnitTest_GPU_Air3D_UpwindFirstFirst
	{
	public:
		const beacls::UVecType type = beacls::UVecType_Cuda;
		const std::string src_filename = std::string("./inputs/air3D/derivSrc_upwindFirstFirst_2.7991.mat");
		const std::vector<std::string> expects_l_filenames{
			std::string("./inputs/air3D/derivL_upwindFirstFirst_2.7991_1.mat"),
			std::string("./inputs/air3D/derivL_upwindFirstFirst_2.7991_2.mat"),
			std::string("./inputs/air3D/derivL_upwindFirstFirst_2.7991_3.mat")
		};
		const std::vector<std::string> expects_r_filenames{
			std::string("./inputs/air3D/derivR_upwindFirstFirst_2.7991_1.mat"),
			std::string("./inputs/air3D/derivR_upwindFirstFirst_2.7991_2.mat"),
			std::string("./inputs/air3D/derivR_upwindFirstFirst_2.7991_3.mat")
		};
		const std::vector <FLOAT_TYPE> maxs{ +20,+10,+2 * M_PI };
		const std::vector <FLOAT_TYPE> mins{ -6,-10,0 };
		SpatialDerivative_Class spatialDerivative_Class = SpatialDerivative_Class_UpwindFirstFirst;
		const std::vector<BoundaryCondition_Class> boundaryCondition_Classes{
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostPeriodic
		};
		bool Test_GPU_Air3D_upwindFirstFirst_chunksize1(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if(!run_UTest_SpatialDerivative(message,src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstFirst_chunksize2(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstFirst_chunksize3(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 3, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstFirst_chunksize4(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 4, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstFirst_chunksize5(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 5, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstFirst_chunksize6(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 6, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstFirst_chunksize7(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 7, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstFirst_chunksize39(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 39, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstFirst_chunksize40(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 40, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstFirst_chunksize41(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 41, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstFirst_chunksize79(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 79, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstFirst_chunksize80(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 80, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstFirst_chunksize81(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 81, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstFirst_chunksize1999(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1999, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstFirst_chunksize2000(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2000, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstFirst_chunksize2001(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2001, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstFirst_chunksize8(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 8, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}

	};
	class UnitTest_GPU_Air3D_UpwindFirstENO2
	{
	public:
		const beacls::UVecType type = beacls::UVecType_Cuda;
		const std::string src_filename = std::string("./inputs/air3D/derivSrc_upwindFirstENO2_2.7991.mat");
		const std::vector<std::string> expects_l_filenames{
			std::string("./inputs/air3D/derivL_upwindFirstENO2_2.7991_1.mat"),
			std::string("./inputs/air3D/derivL_upwindFirstENO2_2.7991_2.mat"),
			std::string("./inputs/air3D/derivL_upwindFirstENO2_2.7991_3.mat")
		};
		const std::vector<std::string> expects_r_filenames{
			std::string("./inputs/air3D/derivR_upwindFirstENO2_2.7991_1.mat"),
			std::string("./inputs/air3D/derivR_upwindFirstENO2_2.7991_2.mat"),
			std::string("./inputs/air3D/derivR_upwindFirstENO2_2.7991_3.mat")
		};
		const std::vector <FLOAT_TYPE> maxs{ +20,+10,+2 * M_PI };
		const std::vector <FLOAT_TYPE> mins{ -6,-10,0 };
		SpatialDerivative_Class spatialDerivative_Class = SpatialDerivative_Class_UpwindFirstENO2;
		const std::vector<BoundaryCondition_Class> boundaryCondition_Classes{
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostPeriodic
		};
		bool Test_GPU_Air3D_upwindFirstENO2_chunksize1(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstENO2_chunksize2(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstENO2_chunksize3(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 3, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstENO2_chunksize4(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 4, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstENO2_chunksize5(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 5, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstENO2_chunksize6(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 6, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstENO2_chunksize7(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 7, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstENO2_chunksize39(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 39, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstENO2_chunksize40(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 40, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstENO2_chunksize41(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 41, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstENO2_chunksize79(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 79, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstENO2_chunksize80(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 80, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstENO2_chunksize81(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 81, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstENO2_chunksize1999(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1999, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstENO2_chunksize2000(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2000, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstENO2_chunksize2001(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2001, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstENO2_chunksize8(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 8, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}

	};
	class UnitTest_GPU_Air3D_UpwindFirstENO3
	{
	public:
		const beacls::UVecType type = beacls::UVecType_Cuda;
		const std::string src_filename = std::string("./inputs/air3D/derivSrc_upwindFirstENO3_2.7996.mat");
		const std::vector<std::string> expects_l_filenames{
			std::string("./inputs/air3D/derivL_upwindFirstENO3_2.7996_1.mat"),
			std::string("./inputs/air3D/derivL_upwindFirstENO3_2.7996_2.mat"),
			std::string("./inputs/air3D/derivL_upwindFirstENO3_2.7996_3.mat")
		};
		const std::vector<std::string> expects_r_filenames{
			std::string("./inputs/air3D/derivR_upwindFirstENO3_2.7996_1.mat"),
			std::string("./inputs/air3D/derivR_upwindFirstENO3_2.7996_2.mat"),
			std::string("./inputs/air3D/derivR_upwindFirstENO3_2.7996_3.mat")
		};
		const std::vector <FLOAT_TYPE> maxs{ +20,+10,+2 * M_PI };
		const std::vector <FLOAT_TYPE> mins{ -6,-10,0 };
		SpatialDerivative_Class spatialDerivative_Class = SpatialDerivative_Class_UpwindFirstENO3;
		const std::vector<BoundaryCondition_Class> boundaryCondition_Classes{
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostPeriodic
		};
		bool Test_GPU_Air3D_upwindFirstENO3_chunksize1(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstENO3_chunksize2(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstENO3_chunksize3(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 3, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstENO3_chunksize4(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 4, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstENO3_chunksize5(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 5, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstENO3_chunksize6(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 6, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstENO3_chunksize7(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 7, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstENO3_chunksize39(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 39, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstENO3_chunksize40(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 40, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstENO3_chunksize41(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 41, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstENO3_chunksize79(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 79, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstENO3_chunksize80(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 80, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstENO3_chunksize81(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 81, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstENO3_chunksize1999(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1999, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstENO3_chunksize2000(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2000, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstENO3_chunksize2001(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2001, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstENO3_chunksize8(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 8, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}

	};
	class UnitTest_GPU_Air3D_UpwindFirstWENO5
	{
	public:
		const beacls::UVecType type = beacls::UVecType_Cuda;
		const std::string src_filename = std::string("./inputs/air3D/derivSrc_upwindFirstWENO5_2.7996.mat");
		const std::vector<std::string> expects_l_filenames{
			std::string("./inputs/air3D/derivL_upwindFirstWENO5_2.7996_1.mat"),
			std::string("./inputs/air3D/derivL_upwindFirstWENO5_2.7996_2.mat"),
			std::string("./inputs/air3D/derivL_upwindFirstWENO5_2.7996_3.mat")
		};
		const std::vector<std::string> expects_r_filenames{
			std::string("./inputs/air3D/derivR_upwindFirstWENO5_2.7996_1.mat"),
			std::string("./inputs/air3D/derivR_upwindFirstWENO5_2.7996_2.mat"),
			std::string("./inputs/air3D/derivR_upwindFirstWENO5_2.7996_3.mat")
		};
		const std::vector <FLOAT_TYPE> maxs{ +20,+10,+2 * M_PI };
		const std::vector <FLOAT_TYPE> mins{ -6,-10,0 };
		SpatialDerivative_Class spatialDerivative_Class = SpatialDerivative_Class_UpwindFirstWENO5;
		const std::vector<BoundaryCondition_Class> boundaryCondition_Classes{
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostPeriodic
		};
		bool Test_GPU_Air3D_upwindFirstWENO5_chunksize1(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstWENO5_chunksize2(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstWENO5_chunksize3(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 3, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstWENO5_chunksize4(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 4, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstWENO5_chunksize5(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 5, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstWENO5_chunksize6(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 6, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstWENO5_chunksize7(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 7, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstWENO5_chunksize39(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 39, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstWENO5_chunksize40(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 40, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstWENO5_chunksize41(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 41, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstWENO5_chunksize79(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 79, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstWENO5_chunksize80(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 80, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstWENO5_chunksize81(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 81, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstWENO5_chunksize1999(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1999, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstWENO5_chunksize2000(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2000, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstWENO5_chunksize2001(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2001, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Air3D_upwindFirstWENO5_chunksize8(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 8, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}

	};
	class UnitTest_GPU_Plane4D_UpwindFirstFirst
	{
	public:
		const beacls::UVecType type = beacls::UVecType_Cuda;
		const std::string src_filename = std::string("./inputs/plane4D/derivSrc_upwindFirstFirst_0.98713.mat");
		const std::vector<std::string> expects_l_filenames{
			std::string("./inputs/plane4D/derivL_upwindFirstFirst_0.98713_1.mat"),
			std::string("./inputs/plane4D/derivL_upwindFirstFirst_0.98713_2.mat"),
			std::string("./inputs/plane4D/derivL_upwindFirstFirst_0.98713_3.mat"),
			std::string("./inputs/plane4D/derivL_upwindFirstFirst_0.98713_4.mat")
		};
		const std::vector<std::string> expects_r_filenames{
			std::string("./inputs/plane4D/derivR_upwindFirstFirst_0.98713_1.mat"),
			std::string("./inputs/plane4D/derivR_upwindFirstFirst_0.98713_2.mat"),
			std::string("./inputs/plane4D/derivR_upwindFirstFirst_0.98713_3.mat"),
			std::string("./inputs/plane4D/derivR_upwindFirstFirst_0.98713_4.mat")
		};
		const std::vector <FLOAT_TYPE> maxs{ +15,+10,+2 * M_PI,12 };
		const std::vector <FLOAT_TYPE> mins{ -5, -10, 0,      6 };
		SpatialDerivative_Class spatialDerivative_Class = SpatialDerivative_Class_UpwindFirstFirst;
		const std::vector<BoundaryCondition_Class> boundaryCondition_Classes{
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostPeriodic,
			BoundaryCondition_Class_AddGhostExtrapolate
		};
		bool Test_GPU_Plane4D_upwindFirstFirst_chunksize1(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstFirst_chunksize2(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstFirst_chunksize3(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 3, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstFirst_chunksize4(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 4, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstFirst_chunksize5(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 5, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstFirst_chunksize6(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 6, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstFirst_chunksize7(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 7, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstFirst_chunksize30(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 30, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstFirst_chunksize31(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 31, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstFirst_chunksize32(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 32, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstFirst_chunksize61(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 61, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstFirst_chunksize62(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 62, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstFirst_chunksize63(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 63, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstFirst_chunksize960(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 960, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstFirst_chunksize961(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 961, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstFirst_chunksize962(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 962, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstFirst_chunksize1921(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1921, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstFirst_chunksize1922(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1922, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstFirst_chunksize1923(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1923, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstFirst_chunksize29790(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29790, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstFirst_chunksize29791(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29791, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstFirst_chunksize29792(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29792, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstFirst_chunksize8(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 8, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}

	};
	class UnitTest_GPU_Plane4D_UpwindFirstENO2
	{
	public:
		const beacls::UVecType type = beacls::UVecType_Cuda;
		const std::string src_filename = std::string("./inputs/plane4D/derivSrc_upwindFirstENO2_1.mat");
		const std::vector<std::string> expects_l_filenames{
			std::string("./inputs/plane4D/derivL_upwindFirstENO2_1_1.mat"),
			std::string("./inputs/plane4D/derivL_upwindFirstENO2_1_2.mat"),
			std::string("./inputs/plane4D/derivL_upwindFirstENO2_1_3.mat"),
			std::string("./inputs/plane4D/derivL_upwindFirstENO2_1_4.mat")
		};
		const std::vector<std::string> expects_r_filenames{
			std::string("./inputs/plane4D/derivR_upwindFirstENO2_1_1.mat"),
			std::string("./inputs/plane4D/derivR_upwindFirstENO2_1_2.mat"),
			std::string("./inputs/plane4D/derivR_upwindFirstENO2_1_3.mat"),
			std::string("./inputs/plane4D/derivR_upwindFirstENO2_1_4.mat")
		};
		const std::vector <FLOAT_TYPE> maxs{ +15,+10,+2 * M_PI,12 };
		const std::vector <FLOAT_TYPE> mins{ -5, -10, 0,      6 };
		SpatialDerivative_Class spatialDerivative_Class = SpatialDerivative_Class_UpwindFirstENO2;
		const std::vector<BoundaryCondition_Class> boundaryCondition_Classes{
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostPeriodic,
			BoundaryCondition_Class_AddGhostExtrapolate
		};
		bool Test_GPU_Plane4D_upwindFirstENO2_chunksize1(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO2_chunksize2(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO2_chunksize3(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 3, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO2_chunksize4(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 4, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO2_chunksize5(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 5, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO2_chunksize6(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 6, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO2_chunksize7(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 7, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO2_chunksize30(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 30, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO2_chunksize31(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 31, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO2_chunksize32(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 32, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO2_chunksize61(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 61, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO2_chunksize62(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 62, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO2_chunksize63(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 63, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO2_chunksize960(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 960, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO2_chunksize961(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 961, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO2_chunksize962(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 962, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO2_chunksize1921(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1921, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO2_chunksize1922(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1922, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO2_chunksize1923(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1923, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO2_chunksize29790(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29790, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO2_chunksize29791(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29791, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO2_chunksize29792(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29792, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO2_chunksize8(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 8, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}

	};
	class UnitTest_GPU_Plane4D_UpwindFirstENO3
	{
	public:
		const beacls::UVecType type = beacls::UVecType_Cuda;
		const std::string src_filename = std::string("./inputs/plane4D/derivSrc_upwindFirstENO3_1.mat");
		const std::vector<std::string> expects_l_filenames{
			std::string("./inputs/plane4D/derivL_upwindFirstENO3_1_1.mat"),
			std::string("./inputs/plane4D/derivL_upwindFirstENO3_1_2.mat"),
			std::string("./inputs/plane4D/derivL_upwindFirstENO3_1_3.mat"),
			std::string("./inputs/plane4D/derivL_upwindFirstENO3_1_4.mat")
		};
		const std::vector<std::string> expects_r_filenames{
			std::string("./inputs/plane4D/derivR_upwindFirstENO3_1_1.mat"),
			std::string("./inputs/plane4D/derivR_upwindFirstENO3_1_2.mat"),
			std::string("./inputs/plane4D/derivR_upwindFirstENO3_1_3.mat"),
			std::string("./inputs/plane4D/derivR_upwindFirstENO3_1_4.mat")
		};
		const std::vector <FLOAT_TYPE> maxs{ +15,+10,+2 * M_PI,12 };
		const std::vector <FLOAT_TYPE> mins{ -5, -10, 0,      6 };
		SpatialDerivative_Class spatialDerivative_Class = SpatialDerivative_Class_UpwindFirstENO3;
		const std::vector<BoundaryCondition_Class> boundaryCondition_Classes{
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostPeriodic,
			BoundaryCondition_Class_AddGhostExtrapolate
		};
		bool Test_GPU_Plane4D_upwindFirstENO3_chunksize1(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO3_chunksize2(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO3_chunksize3(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 3, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO3_chunksize4(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 4, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO3_chunksize5(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 5, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO3_chunksize6(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 6, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO3_chunksize7(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 7, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO3_chunksize30(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 30, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO3_chunksize31(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 31, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO3_chunksize32(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 32, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO3_chunksize61(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 61, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO3_chunksize62(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 62, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO3_chunksize63(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 63, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO3_chunksize960(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 960, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO3_chunksize961(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 961, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO3_chunksize962(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 962, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO3_chunksize1921(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1921, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO3_chunksize1922(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1922, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO3_chunksize1923(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1923, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO3_chunksize29790(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29790, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO3_chunksize29791(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29791, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO3_chunksize29792(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29792, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstENO3_chunksize8(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 8, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}

	};
	class UnitTest_GPU_Plane4D_UpwindFirstWENO5
	{
	public:
		const beacls::UVecType type = beacls::UVecType_Cuda;
		const std::string src_filename = std::string("./inputs/plane4D/derivSrc_upwindFirstWENO5_1.mat");
		const std::vector<std::string> expects_l_filenames{
			std::string("./inputs/plane4D/derivL_upwindFirstWENO5_1_1.mat"),
			std::string("./inputs/plane4D/derivL_upwindFirstWENO5_1_2.mat"),
			std::string("./inputs/plane4D/derivL_upwindFirstWENO5_1_3.mat"),
			std::string("./inputs/plane4D/derivL_upwindFirstWENO5_1_4.mat")
		};
		const std::vector<std::string> expects_r_filenames{
			std::string("./inputs/plane4D/derivR_upwindFirstWENO5_1_1.mat"),
			std::string("./inputs/plane4D/derivR_upwindFirstWENO5_1_2.mat"),
			std::string("./inputs/plane4D/derivR_upwindFirstWENO5_1_3.mat"),
			std::string("./inputs/plane4D/derivR_upwindFirstWENO5_1_4.mat")
		};
		const std::vector <FLOAT_TYPE> maxs{ +15,+10,+2 * M_PI,12 };
		const std::vector <FLOAT_TYPE> mins{ -5, -10, 0,      6 };
		SpatialDerivative_Class spatialDerivative_Class = SpatialDerivative_Class_UpwindFirstWENO5;
		const std::vector<BoundaryCondition_Class> boundaryCondition_Classes{
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostExtrapolate,
			BoundaryCondition_Class_AddGhostPeriodic,
			BoundaryCondition_Class_AddGhostExtrapolate
		};
		bool Test_GPU_Plane4D_upwindFirstWENO5_chunksize1(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstWENO5_chunksize2(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 2, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstWENO5_chunksize3(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 3, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstWENO5_chunksize4(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 4, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstWENO5_chunksize5(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 5, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstWENO5_chunksize6(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 6, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstWENO5_chunksize7(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 7, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstWENO5_chunksize30(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 30, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstWENO5_chunksize31(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 31, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstWENO5_chunksize32(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 32, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstWENO5_chunksize61(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 61, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstWENO5_chunksize62(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 62, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstWENO5_chunksize63(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 63, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstWENO5_chunksize960(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 960, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstWENO5_chunksize961(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 961, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstWENO5_chunksize962(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 962, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstWENO5_chunksize1921(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1921, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstWENO5_chunksize1922(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1922, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstWENO5_chunksize1923(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 1923, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstWENO5_chunksize29790(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29790, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstWENO5_chunksize29791(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29791, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstWENO5_chunksize29792(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 29792, num_of_threads, num_of_gpus)) {
				std::cerr << __FILE__ << ":" << __LINE__ << ":" << __func__ << std::endl;
				std::cerr << message.c_str() << std::endl;
				return false;
			}
			std::cout << "OK" << std::endl;
			return true;
		}
		bool Test_GPU_Plane4D_upwindFirstWENO5_chunksize8(const FLOAT_TYPE small_diff, const int num_of_threads, const int num_of_gpus) const
		{
			std::cout << __func__ << " ..." <<std::endl;
			std::string message;
			if (!run_UTest_SpatialDerivative(message, src_filename, expects_l_filenames, expects_r_filenames, maxs, mins, spatialDerivative_Class, boundaryCondition_Classes, type, small_diff, 8, num_of_threads, num_of_gpus)) {
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
	FLOAT_TYPE small_diff = 0;
	if (argc >= 2) {
	  small_diff = static_cast<FLOAT_TYPE>(atof(argv[1]));
	}
	int num_of_threads = 0;
	if (argc >= 3) {
		num_of_threads = atoi(argv[2]);
	}
	int num_of_gpus = 0;
	if (argc >= 4) {
		num_of_gpus = atoi(argv[3]);
	}

	UTest_CPU_SpatialDerivative::UnitTest_CPU_Air3D_UpwindFirstFirst unitTest_CPU_Air3D_UpwindFirstFirst;
	UTest_CPU_SpatialDerivative::UnitTest_CPU_Air3D_UpwindFirstENO2 unitTest_CPU_Air3D_UpwindFirstENO2;
	UTest_CPU_SpatialDerivative::UnitTest_CPU_Air3D_UpwindFirstENO3 unitTest_CPU_Air3D_UpwindFirstENO3;
	UTest_CPU_SpatialDerivative::UnitTest_CPU_Air3D_UpwindFirstWENO5 unitTest_CPU_Air3D_UpwindFirstWENO5;

	UTest_CPU_SpatialDerivative::UnitTest_CPU_Plane4D_UpwindFirstFirst unitTest_CPU_Plane4D_UpwindFirstFirst;
	UTest_CPU_SpatialDerivative::UnitTest_CPU_Plane4D_UpwindFirstENO2 unitTest_CPU_Plane4D_UpwindFirstENO2;
	UTest_CPU_SpatialDerivative::UnitTest_CPU_Plane4D_UpwindFirstENO3 unitTest_CPU_Plane4D_UpwindFirstENO3;
	UTest_CPU_SpatialDerivative::UnitTest_CPU_Plane4D_UpwindFirstWENO5 unitTest_CPU_Plane4D_UpwindFirstWENO5;
	
	UTest_GPU_SpatialDerivative::UnitTest_GPU_Air3D_UpwindFirstFirst unitTest_GPU_Air3D_UpwindFirstFirst;
	UTest_GPU_SpatialDerivative::UnitTest_GPU_Air3D_UpwindFirstENO2 unitTest_GPU_Air3D_UpwindFirstENO2;
	UTest_GPU_SpatialDerivative::UnitTest_GPU_Air3D_UpwindFirstENO3 unitTest_GPU_Air3D_UpwindFirstENO3;
	UTest_GPU_SpatialDerivative::UnitTest_GPU_Air3D_UpwindFirstWENO5 unitTest_GPU_Air3D_UpwindFirstWENO5;

	UTest_GPU_SpatialDerivative::UnitTest_GPU_Plane4D_UpwindFirstFirst unitTest_GPU_Plane4D_UpwindFirstFirst;
	UTest_GPU_SpatialDerivative::UnitTest_GPU_Plane4D_UpwindFirstENO2 unitTest_GPU_Plane4D_UpwindFirstENO2;
	UTest_GPU_SpatialDerivative::UnitTest_GPU_Plane4D_UpwindFirstENO3 unitTest_GPU_Plane4D_UpwindFirstENO3;
	UTest_GPU_SpatialDerivative::UnitTest_GPU_Plane4D_UpwindFirstWENO5 unitTest_GPU_Plane4D_UpwindFirstWENO5;
	
	size_t num_of_tests = 0;
	size_t num_of_succeeded = 0;

	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstFirst.Test_GPU_Air3D_upwindFirstFirst_chunksize1(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstFirst.Test_GPU_Air3D_upwindFirstFirst_chunksize2(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstFirst.Test_GPU_Air3D_upwindFirstFirst_chunksize3(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstFirst.Test_GPU_Air3D_upwindFirstFirst_chunksize4(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstFirst.Test_GPU_Air3D_upwindFirstFirst_chunksize5(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstFirst.Test_GPU_Air3D_upwindFirstFirst_chunksize6(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstFirst.Test_GPU_Air3D_upwindFirstFirst_chunksize7(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstFirst.Test_GPU_Air3D_upwindFirstFirst_chunksize8(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstFirst.Test_GPU_Air3D_upwindFirstFirst_chunksize39(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstFirst.Test_GPU_Air3D_upwindFirstFirst_chunksize40(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstFirst.Test_GPU_Air3D_upwindFirstFirst_chunksize41(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstFirst.Test_GPU_Air3D_upwindFirstFirst_chunksize79(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstFirst.Test_GPU_Air3D_upwindFirstFirst_chunksize80(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstFirst.Test_GPU_Air3D_upwindFirstFirst_chunksize81(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstFirst.Test_GPU_Air3D_upwindFirstFirst_chunksize1999(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstFirst.Test_GPU_Air3D_upwindFirstFirst_chunksize2000(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstFirst.Test_GPU_Air3D_upwindFirstFirst_chunksize2001(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;

	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstENO2.Test_GPU_Air3D_upwindFirstENO2_chunksize1(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstENO2.Test_GPU_Air3D_upwindFirstENO2_chunksize2(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstENO2.Test_GPU_Air3D_upwindFirstENO2_chunksize3(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstENO2.Test_GPU_Air3D_upwindFirstENO2_chunksize4(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstENO2.Test_GPU_Air3D_upwindFirstENO2_chunksize5(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstENO2.Test_GPU_Air3D_upwindFirstENO2_chunksize6(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstENO2.Test_GPU_Air3D_upwindFirstENO2_chunksize7(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstENO2.Test_GPU_Air3D_upwindFirstENO2_chunksize8(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstENO2.Test_GPU_Air3D_upwindFirstENO2_chunksize39(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstENO2.Test_GPU_Air3D_upwindFirstENO2_chunksize40(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstENO2.Test_GPU_Air3D_upwindFirstENO2_chunksize41(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstENO2.Test_GPU_Air3D_upwindFirstENO2_chunksize79(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstENO2.Test_GPU_Air3D_upwindFirstENO2_chunksize80(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstENO2.Test_GPU_Air3D_upwindFirstENO2_chunksize81(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstENO2.Test_GPU_Air3D_upwindFirstENO2_chunksize1999(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstENO2.Test_GPU_Air3D_upwindFirstENO2_chunksize2000(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstENO2.Test_GPU_Air3D_upwindFirstENO2_chunksize2001(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;

	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstENO3.Test_GPU_Air3D_upwindFirstENO3_chunksize1(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstENO3.Test_GPU_Air3D_upwindFirstENO3_chunksize2(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstENO3.Test_GPU_Air3D_upwindFirstENO3_chunksize3(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstENO3.Test_GPU_Air3D_upwindFirstENO3_chunksize4(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstENO3.Test_GPU_Air3D_upwindFirstENO3_chunksize5(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstENO3.Test_GPU_Air3D_upwindFirstENO3_chunksize6(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstENO3.Test_GPU_Air3D_upwindFirstENO3_chunksize7(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstENO3.Test_GPU_Air3D_upwindFirstENO3_chunksize8(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstENO3.Test_GPU_Air3D_upwindFirstENO3_chunksize39(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstENO3.Test_GPU_Air3D_upwindFirstENO3_chunksize40(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstENO3.Test_GPU_Air3D_upwindFirstENO3_chunksize41(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstENO3.Test_GPU_Air3D_upwindFirstENO3_chunksize79(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstENO3.Test_GPU_Air3D_upwindFirstENO3_chunksize80(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstENO3.Test_GPU_Air3D_upwindFirstENO3_chunksize81(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstENO3.Test_GPU_Air3D_upwindFirstENO3_chunksize1999(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstENO3.Test_GPU_Air3D_upwindFirstENO3_chunksize2000(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstENO3.Test_GPU_Air3D_upwindFirstENO3_chunksize2001(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;

	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstWENO5.Test_GPU_Air3D_upwindFirstWENO5_chunksize1(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstWENO5.Test_GPU_Air3D_upwindFirstWENO5_chunksize2(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstWENO5.Test_GPU_Air3D_upwindFirstWENO5_chunksize3(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstWENO5.Test_GPU_Air3D_upwindFirstWENO5_chunksize4(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstWENO5.Test_GPU_Air3D_upwindFirstWENO5_chunksize5(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstWENO5.Test_GPU_Air3D_upwindFirstWENO5_chunksize6(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstWENO5.Test_GPU_Air3D_upwindFirstWENO5_chunksize7(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstWENO5.Test_GPU_Air3D_upwindFirstWENO5_chunksize8(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstWENO5.Test_GPU_Air3D_upwindFirstWENO5_chunksize39(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstWENO5.Test_GPU_Air3D_upwindFirstWENO5_chunksize40(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstWENO5.Test_GPU_Air3D_upwindFirstWENO5_chunksize41(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstWENO5.Test_GPU_Air3D_upwindFirstWENO5_chunksize79(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstWENO5.Test_GPU_Air3D_upwindFirstWENO5_chunksize80(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstWENO5.Test_GPU_Air3D_upwindFirstWENO5_chunksize81(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstWENO5.Test_GPU_Air3D_upwindFirstWENO5_chunksize1999(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstWENO5.Test_GPU_Air3D_upwindFirstWENO5_chunksize2000(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Air3D_UpwindFirstWENO5.Test_GPU_Air3D_upwindFirstWENO5_chunksize2001(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;


	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstFirst.Test_GPU_Plane4D_upwindFirstFirst_chunksize1(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstFirst.Test_GPU_Plane4D_upwindFirstFirst_chunksize2(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstFirst.Test_GPU_Plane4D_upwindFirstFirst_chunksize3(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstFirst.Test_GPU_Plane4D_upwindFirstFirst_chunksize4(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstFirst.Test_GPU_Plane4D_upwindFirstFirst_chunksize5(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstFirst.Test_GPU_Plane4D_upwindFirstFirst_chunksize6(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstFirst.Test_GPU_Plane4D_upwindFirstFirst_chunksize7(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstFirst.Test_GPU_Plane4D_upwindFirstFirst_chunksize8(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstFirst.Test_GPU_Plane4D_upwindFirstFirst_chunksize30(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstFirst.Test_GPU_Plane4D_upwindFirstFirst_chunksize31(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstFirst.Test_GPU_Plane4D_upwindFirstFirst_chunksize32(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstFirst.Test_GPU_Plane4D_upwindFirstFirst_chunksize61(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstFirst.Test_GPU_Plane4D_upwindFirstFirst_chunksize62(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstFirst.Test_GPU_Plane4D_upwindFirstFirst_chunksize63(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstFirst.Test_GPU_Plane4D_upwindFirstFirst_chunksize960(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstFirst.Test_GPU_Plane4D_upwindFirstFirst_chunksize961(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstFirst.Test_GPU_Plane4D_upwindFirstFirst_chunksize962(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstFirst.Test_GPU_Plane4D_upwindFirstFirst_chunksize1921(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstFirst.Test_GPU_Plane4D_upwindFirstFirst_chunksize1922(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstFirst.Test_GPU_Plane4D_upwindFirstFirst_chunksize1923(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstFirst.Test_GPU_Plane4D_upwindFirstFirst_chunksize29790(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstFirst.Test_GPU_Plane4D_upwindFirstFirst_chunksize29791(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstFirst.Test_GPU_Plane4D_upwindFirstFirst_chunksize29792(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;

	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO2.Test_GPU_Plane4D_upwindFirstENO2_chunksize1(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO2.Test_GPU_Plane4D_upwindFirstENO2_chunksize2(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO2.Test_GPU_Plane4D_upwindFirstENO2_chunksize3(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO2.Test_GPU_Plane4D_upwindFirstENO2_chunksize4(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO2.Test_GPU_Plane4D_upwindFirstENO2_chunksize5(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO2.Test_GPU_Plane4D_upwindFirstENO2_chunksize6(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO2.Test_GPU_Plane4D_upwindFirstENO2_chunksize7(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO2.Test_GPU_Plane4D_upwindFirstENO2_chunksize8(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO2.Test_GPU_Plane4D_upwindFirstENO2_chunksize30(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO2.Test_GPU_Plane4D_upwindFirstENO2_chunksize31(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO2.Test_GPU_Plane4D_upwindFirstENO2_chunksize32(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO2.Test_GPU_Plane4D_upwindFirstENO2_chunksize61(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO2.Test_GPU_Plane4D_upwindFirstENO2_chunksize62(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO2.Test_GPU_Plane4D_upwindFirstENO2_chunksize63(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO2.Test_GPU_Plane4D_upwindFirstENO2_chunksize960(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO2.Test_GPU_Plane4D_upwindFirstENO2_chunksize961(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO2.Test_GPU_Plane4D_upwindFirstENO2_chunksize962(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO2.Test_GPU_Plane4D_upwindFirstENO2_chunksize1921(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO2.Test_GPU_Plane4D_upwindFirstENO2_chunksize1922(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO2.Test_GPU_Plane4D_upwindFirstENO2_chunksize1923(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO2.Test_GPU_Plane4D_upwindFirstENO2_chunksize29790(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO2.Test_GPU_Plane4D_upwindFirstENO2_chunksize29791(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO2.Test_GPU_Plane4D_upwindFirstENO2_chunksize29792(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;

	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO3.Test_GPU_Plane4D_upwindFirstENO3_chunksize1(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO3.Test_GPU_Plane4D_upwindFirstENO3_chunksize2(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO3.Test_GPU_Plane4D_upwindFirstENO3_chunksize3(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO3.Test_GPU_Plane4D_upwindFirstENO3_chunksize4(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO3.Test_GPU_Plane4D_upwindFirstENO3_chunksize5(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO3.Test_GPU_Plane4D_upwindFirstENO3_chunksize6(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO3.Test_GPU_Plane4D_upwindFirstENO3_chunksize7(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO3.Test_GPU_Plane4D_upwindFirstENO3_chunksize8(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO3.Test_GPU_Plane4D_upwindFirstENO3_chunksize30(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO3.Test_GPU_Plane4D_upwindFirstENO3_chunksize31(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO3.Test_GPU_Plane4D_upwindFirstENO3_chunksize32(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO3.Test_GPU_Plane4D_upwindFirstENO3_chunksize61(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO3.Test_GPU_Plane4D_upwindFirstENO3_chunksize62(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO3.Test_GPU_Plane4D_upwindFirstENO3_chunksize63(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO3.Test_GPU_Plane4D_upwindFirstENO3_chunksize960(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO3.Test_GPU_Plane4D_upwindFirstENO3_chunksize961(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO3.Test_GPU_Plane4D_upwindFirstENO3_chunksize962(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO3.Test_GPU_Plane4D_upwindFirstENO3_chunksize1921(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO3.Test_GPU_Plane4D_upwindFirstENO3_chunksize1922(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO3.Test_GPU_Plane4D_upwindFirstENO3_chunksize1923(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO3.Test_GPU_Plane4D_upwindFirstENO3_chunksize29790(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO3.Test_GPU_Plane4D_upwindFirstENO3_chunksize29791(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstENO3.Test_GPU_Plane4D_upwindFirstENO3_chunksize29792(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;

	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstWENO5.Test_GPU_Plane4D_upwindFirstWENO5_chunksize1(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstWENO5.Test_GPU_Plane4D_upwindFirstWENO5_chunksize2(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstWENO5.Test_GPU_Plane4D_upwindFirstWENO5_chunksize3(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstWENO5.Test_GPU_Plane4D_upwindFirstWENO5_chunksize4(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstWENO5.Test_GPU_Plane4D_upwindFirstWENO5_chunksize5(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstWENO5.Test_GPU_Plane4D_upwindFirstWENO5_chunksize6(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstWENO5.Test_GPU_Plane4D_upwindFirstWENO5_chunksize7(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstWENO5.Test_GPU_Plane4D_upwindFirstWENO5_chunksize8(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstWENO5.Test_GPU_Plane4D_upwindFirstWENO5_chunksize30(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstWENO5.Test_GPU_Plane4D_upwindFirstWENO5_chunksize31(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstWENO5.Test_GPU_Plane4D_upwindFirstWENO5_chunksize32(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstWENO5.Test_GPU_Plane4D_upwindFirstWENO5_chunksize61(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstWENO5.Test_GPU_Plane4D_upwindFirstWENO5_chunksize62(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstWENO5.Test_GPU_Plane4D_upwindFirstWENO5_chunksize63(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstWENO5.Test_GPU_Plane4D_upwindFirstWENO5_chunksize960(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstWENO5.Test_GPU_Plane4D_upwindFirstWENO5_chunksize961(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstWENO5.Test_GPU_Plane4D_upwindFirstWENO5_chunksize962(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstWENO5.Test_GPU_Plane4D_upwindFirstWENO5_chunksize1921(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstWENO5.Test_GPU_Plane4D_upwindFirstWENO5_chunksize1922(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstWENO5.Test_GPU_Plane4D_upwindFirstWENO5_chunksize1923(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstWENO5.Test_GPU_Plane4D_upwindFirstWENO5_chunksize29790(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstWENO5.Test_GPU_Plane4D_upwindFirstWENO5_chunksize29791(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_GPU_Plane4D_UpwindFirstWENO5.Test_GPU_Plane4D_upwindFirstWENO5_chunksize29792(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;


	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstFirst.Test_CPU_Air3D_upwindFirstFirst_chunksize1(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstFirst.Test_CPU_Air3D_upwindFirstFirst_chunksize2(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstFirst.Test_CPU_Air3D_upwindFirstFirst_chunksize3(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstFirst.Test_CPU_Air3D_upwindFirstFirst_chunksize4(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstFirst.Test_CPU_Air3D_upwindFirstFirst_chunksize5(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstFirst.Test_CPU_Air3D_upwindFirstFirst_chunksize6(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstFirst.Test_CPU_Air3D_upwindFirstFirst_chunksize7(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstFirst.Test_CPU_Air3D_upwindFirstFirst_chunksize8(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstFirst.Test_CPU_Air3D_upwindFirstFirst_chunksize39(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstFirst.Test_CPU_Air3D_upwindFirstFirst_chunksize40(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstFirst.Test_CPU_Air3D_upwindFirstFirst_chunksize41(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstFirst.Test_CPU_Air3D_upwindFirstFirst_chunksize79(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstFirst.Test_CPU_Air3D_upwindFirstFirst_chunksize80(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstFirst.Test_CPU_Air3D_upwindFirstFirst_chunksize81(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstFirst.Test_CPU_Air3D_upwindFirstFirst_chunksize1999(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstFirst.Test_CPU_Air3D_upwindFirstFirst_chunksize2000(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstFirst.Test_CPU_Air3D_upwindFirstFirst_chunksize2001(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;

	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstENO2.Test_CPU_Air3D_upwindFirstENO2_chunksize1(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstENO2.Test_CPU_Air3D_upwindFirstENO2_chunksize2(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstENO2.Test_CPU_Air3D_upwindFirstENO2_chunksize3(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstENO2.Test_CPU_Air3D_upwindFirstENO2_chunksize4(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstENO2.Test_CPU_Air3D_upwindFirstENO2_chunksize5(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstENO2.Test_CPU_Air3D_upwindFirstENO2_chunksize6(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstENO2.Test_CPU_Air3D_upwindFirstENO2_chunksize7(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstENO2.Test_CPU_Air3D_upwindFirstENO2_chunksize8(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstENO2.Test_CPU_Air3D_upwindFirstENO2_chunksize39(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstENO2.Test_CPU_Air3D_upwindFirstENO2_chunksize40(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstENO2.Test_CPU_Air3D_upwindFirstENO2_chunksize41(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstENO2.Test_CPU_Air3D_upwindFirstENO2_chunksize79(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstENO2.Test_CPU_Air3D_upwindFirstENO2_chunksize80(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstENO2.Test_CPU_Air3D_upwindFirstENO2_chunksize81(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstENO2.Test_CPU_Air3D_upwindFirstENO2_chunksize1999(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstENO2.Test_CPU_Air3D_upwindFirstENO2_chunksize2000(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstENO2.Test_CPU_Air3D_upwindFirstENO2_chunksize2001(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;

	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstENO3.Test_CPU_Air3D_upwindFirstENO3_chunksize1(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstENO3.Test_CPU_Air3D_upwindFirstENO3_chunksize2(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstENO3.Test_CPU_Air3D_upwindFirstENO3_chunksize3(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstENO3.Test_CPU_Air3D_upwindFirstENO3_chunksize4(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstENO3.Test_CPU_Air3D_upwindFirstENO3_chunksize5(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstENO3.Test_CPU_Air3D_upwindFirstENO3_chunksize6(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstENO3.Test_CPU_Air3D_upwindFirstENO3_chunksize7(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstENO3.Test_CPU_Air3D_upwindFirstENO3_chunksize8(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstENO3.Test_CPU_Air3D_upwindFirstENO3_chunksize39(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstENO3.Test_CPU_Air3D_upwindFirstENO3_chunksize40(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstENO3.Test_CPU_Air3D_upwindFirstENO3_chunksize41(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstENO3.Test_CPU_Air3D_upwindFirstENO3_chunksize79(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstENO3.Test_CPU_Air3D_upwindFirstENO3_chunksize80(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstENO3.Test_CPU_Air3D_upwindFirstENO3_chunksize81(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstENO3.Test_CPU_Air3D_upwindFirstENO3_chunksize1999(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstENO3.Test_CPU_Air3D_upwindFirstENO3_chunksize2000(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstENO3.Test_CPU_Air3D_upwindFirstENO3_chunksize2001(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;

	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstWENO5.Test_CPU_Air3D_upwindFirstWENO5_chunksize1(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstWENO5.Test_CPU_Air3D_upwindFirstWENO5_chunksize2(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstWENO5.Test_CPU_Air3D_upwindFirstWENO5_chunksize3(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstWENO5.Test_CPU_Air3D_upwindFirstWENO5_chunksize4(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstWENO5.Test_CPU_Air3D_upwindFirstWENO5_chunksize5(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstWENO5.Test_CPU_Air3D_upwindFirstWENO5_chunksize6(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstWENO5.Test_CPU_Air3D_upwindFirstWENO5_chunksize7(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstWENO5.Test_CPU_Air3D_upwindFirstWENO5_chunksize8(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstWENO5.Test_CPU_Air3D_upwindFirstWENO5_chunksize39(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstWENO5.Test_CPU_Air3D_upwindFirstWENO5_chunksize40(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstWENO5.Test_CPU_Air3D_upwindFirstWENO5_chunksize41(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstWENO5.Test_CPU_Air3D_upwindFirstWENO5_chunksize79(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstWENO5.Test_CPU_Air3D_upwindFirstWENO5_chunksize80(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstWENO5.Test_CPU_Air3D_upwindFirstWENO5_chunksize81(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstWENO5.Test_CPU_Air3D_upwindFirstWENO5_chunksize1999(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstWENO5.Test_CPU_Air3D_upwindFirstWENO5_chunksize2000(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Air3D_UpwindFirstWENO5.Test_CPU_Air3D_upwindFirstWENO5_chunksize2001(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;


	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstFirst.Test_CPU_Plane4D_upwindFirstFirst_chunksize1(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstFirst.Test_CPU_Plane4D_upwindFirstFirst_chunksize2(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstFirst.Test_CPU_Plane4D_upwindFirstFirst_chunksize3(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstFirst.Test_CPU_Plane4D_upwindFirstFirst_chunksize4(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstFirst.Test_CPU_Plane4D_upwindFirstFirst_chunksize5(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstFirst.Test_CPU_Plane4D_upwindFirstFirst_chunksize6(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstFirst.Test_CPU_Plane4D_upwindFirstFirst_chunksize7(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstFirst.Test_CPU_Plane4D_upwindFirstFirst_chunksize8(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstFirst.Test_CPU_Plane4D_upwindFirstFirst_chunksize30(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstFirst.Test_CPU_Plane4D_upwindFirstFirst_chunksize31(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstFirst.Test_CPU_Plane4D_upwindFirstFirst_chunksize32(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstFirst.Test_CPU_Plane4D_upwindFirstFirst_chunksize61(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstFirst.Test_CPU_Plane4D_upwindFirstFirst_chunksize62(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstFirst.Test_CPU_Plane4D_upwindFirstFirst_chunksize63(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstFirst.Test_CPU_Plane4D_upwindFirstFirst_chunksize960(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstFirst.Test_CPU_Plane4D_upwindFirstFirst_chunksize961(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstFirst.Test_CPU_Plane4D_upwindFirstFirst_chunksize962(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstFirst.Test_CPU_Plane4D_upwindFirstFirst_chunksize1921(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstFirst.Test_CPU_Plane4D_upwindFirstFirst_chunksize1922(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstFirst.Test_CPU_Plane4D_upwindFirstFirst_chunksize1923(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstFirst.Test_CPU_Plane4D_upwindFirstFirst_chunksize29790(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstFirst.Test_CPU_Plane4D_upwindFirstFirst_chunksize29791(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstFirst.Test_CPU_Plane4D_upwindFirstFirst_chunksize29792(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;

	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO2.Test_CPU_Plane4D_upwindFirstENO2_chunksize1(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO2.Test_CPU_Plane4D_upwindFirstENO2_chunksize2(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO2.Test_CPU_Plane4D_upwindFirstENO2_chunksize3(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO2.Test_CPU_Plane4D_upwindFirstENO2_chunksize4(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO2.Test_CPU_Plane4D_upwindFirstENO2_chunksize5(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO2.Test_CPU_Plane4D_upwindFirstENO2_chunksize6(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO2.Test_CPU_Plane4D_upwindFirstENO2_chunksize7(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO2.Test_CPU_Plane4D_upwindFirstENO2_chunksize8(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO2.Test_CPU_Plane4D_upwindFirstENO2_chunksize30(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO2.Test_CPU_Plane4D_upwindFirstENO2_chunksize31(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO2.Test_CPU_Plane4D_upwindFirstENO2_chunksize32(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO2.Test_CPU_Plane4D_upwindFirstENO2_chunksize61(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO2.Test_CPU_Plane4D_upwindFirstENO2_chunksize62(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO2.Test_CPU_Plane4D_upwindFirstENO2_chunksize63(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO2.Test_CPU_Plane4D_upwindFirstENO2_chunksize960(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO2.Test_CPU_Plane4D_upwindFirstENO2_chunksize961(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO2.Test_CPU_Plane4D_upwindFirstENO2_chunksize962(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO2.Test_CPU_Plane4D_upwindFirstENO2_chunksize1921(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO2.Test_CPU_Plane4D_upwindFirstENO2_chunksize1922(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO2.Test_CPU_Plane4D_upwindFirstENO2_chunksize1923(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO2.Test_CPU_Plane4D_upwindFirstENO2_chunksize29790(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO2.Test_CPU_Plane4D_upwindFirstENO2_chunksize29791(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO2.Test_CPU_Plane4D_upwindFirstENO2_chunksize29792(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;

	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO3.Test_CPU_Plane4D_upwindFirstENO3_chunksize1(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO3.Test_CPU_Plane4D_upwindFirstENO3_chunksize2(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO3.Test_CPU_Plane4D_upwindFirstENO3_chunksize3(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO3.Test_CPU_Plane4D_upwindFirstENO3_chunksize4(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO3.Test_CPU_Plane4D_upwindFirstENO3_chunksize5(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO3.Test_CPU_Plane4D_upwindFirstENO3_chunksize6(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO3.Test_CPU_Plane4D_upwindFirstENO3_chunksize7(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO3.Test_CPU_Plane4D_upwindFirstENO3_chunksize8(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO3.Test_CPU_Plane4D_upwindFirstENO3_chunksize30(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO3.Test_CPU_Plane4D_upwindFirstENO3_chunksize31(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO3.Test_CPU_Plane4D_upwindFirstENO3_chunksize32(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO3.Test_CPU_Plane4D_upwindFirstENO3_chunksize61(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO3.Test_CPU_Plane4D_upwindFirstENO3_chunksize62(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO3.Test_CPU_Plane4D_upwindFirstENO3_chunksize63(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO3.Test_CPU_Plane4D_upwindFirstENO3_chunksize960(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO3.Test_CPU_Plane4D_upwindFirstENO3_chunksize961(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO3.Test_CPU_Plane4D_upwindFirstENO3_chunksize962(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO3.Test_CPU_Plane4D_upwindFirstENO3_chunksize1921(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO3.Test_CPU_Plane4D_upwindFirstENO3_chunksize1922(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO3.Test_CPU_Plane4D_upwindFirstENO3_chunksize1923(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO3.Test_CPU_Plane4D_upwindFirstENO3_chunksize29790(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO3.Test_CPU_Plane4D_upwindFirstENO3_chunksize29791(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstENO3.Test_CPU_Plane4D_upwindFirstENO3_chunksize29792(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;

	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstWENO5.Test_CPU_Plane4D_upwindFirstWENO5_chunksize1(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstWENO5.Test_CPU_Plane4D_upwindFirstWENO5_chunksize2(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstWENO5.Test_CPU_Plane4D_upwindFirstWENO5_chunksize3(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstWENO5.Test_CPU_Plane4D_upwindFirstWENO5_chunksize4(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstWENO5.Test_CPU_Plane4D_upwindFirstWENO5_chunksize5(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstWENO5.Test_CPU_Plane4D_upwindFirstWENO5_chunksize6(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstWENO5.Test_CPU_Plane4D_upwindFirstWENO5_chunksize7(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstWENO5.Test_CPU_Plane4D_upwindFirstWENO5_chunksize8(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstWENO5.Test_CPU_Plane4D_upwindFirstWENO5_chunksize30(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstWENO5.Test_CPU_Plane4D_upwindFirstWENO5_chunksize31(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstWENO5.Test_CPU_Plane4D_upwindFirstWENO5_chunksize32(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstWENO5.Test_CPU_Plane4D_upwindFirstWENO5_chunksize61(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstWENO5.Test_CPU_Plane4D_upwindFirstWENO5_chunksize62(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstWENO5.Test_CPU_Plane4D_upwindFirstWENO5_chunksize63(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstWENO5.Test_CPU_Plane4D_upwindFirstWENO5_chunksize960(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstWENO5.Test_CPU_Plane4D_upwindFirstWENO5_chunksize961(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstWENO5.Test_CPU_Plane4D_upwindFirstWENO5_chunksize962(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstWENO5.Test_CPU_Plane4D_upwindFirstWENO5_chunksize1921(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstWENO5.Test_CPU_Plane4D_upwindFirstWENO5_chunksize1922(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstWENO5.Test_CPU_Plane4D_upwindFirstWENO5_chunksize1923(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstWENO5.Test_CPU_Plane4D_upwindFirstWENO5_chunksize29790(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstWENO5.Test_CPU_Plane4D_upwindFirstWENO5_chunksize29791(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;
	++num_of_tests;
	if(unitTest_CPU_Plane4D_UpwindFirstWENO5.Test_CPU_Plane4D_upwindFirstWENO5_chunksize29792(small_diff, num_of_threads, num_of_gpus)) ++num_of_succeeded;

	if (num_of_tests == num_of_succeeded) {
		std::cout << "All test passed " << num_of_succeeded << " / " << num_of_tests << std::endl;
	} else {
		std::cout << "Passed " << num_of_succeeded << " / " << num_of_tests << std::endl;
	}


	return 0;

}

