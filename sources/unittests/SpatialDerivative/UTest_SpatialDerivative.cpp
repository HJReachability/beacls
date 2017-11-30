#define _USE_MATH_DEFINES
#include <levelset/levelset.hpp>
#include <cmath>
#include <numeric>
#include <functional>
#include <cfloat>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cstring>
#include <iostream>
#include <thread>
#include <cstdlib>
#include "UTest_SpatialDerivative.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

bool run_UTest_SpatialDerivative(
	std::string &message,
	const std::string &src_data_filename,
	const std::vector<std::string>& expects_deriv_l_filenames,
	const std::vector<std::string>& expects_deriv_r_filenames,
	const beacls::FloatVec &maxs,
	const beacls::FloatVec &mins,
	const SpatialDerivative_Class spatialDerivative_Class,
	const std::vector<BoundaryCondition_Class> &boundaryCondition_Classes,
	const beacls::UVecType type,
	const FLOAT_TYPE small_diff,
	const size_t line_length_of_chunk,
	const int num_of_threads,
	const int num_of_gpus
) {
	if (src_data_filename.empty()) {
		std::stringstream ss;
		ss << "Error source file name is empty" << std::endl;
		message.append(ss.str());
		return false;
	}
	beacls::MatFStream* src_data_filename_fs = beacls::openMatFStream(src_data_filename, beacls::MatOpenMode_Read);
	beacls::FloatVec data;
	beacls::IntegerVec Ns;
	if (!src_data_filename_fs || !load_vector(data, std::string("data"), Ns, false, src_data_filename_fs)) {
		std::stringstream ss;
		ss << "Cannot open source file: " << src_data_filename.c_str() << std::endl;
		message.append(ss.str());
		if (src_data_filename_fs)
			beacls::closeMatFStream(src_data_filename_fs);
		return false;
	}
	beacls::closeMatFStream(src_data_filename_fs);

	levelset::HJI_Grid *hJI_Grid = new levelset::HJI_Grid(Ns.size());
	if (!hJI_Grid) {
		std::stringstream ss;
		ss << "Cannot create grid" << std::endl;
		message.append(ss.str());
		return false;
	}
	hJI_Grid->set_Ns(Ns);
	size_t num_of_dimensions = hJI_Grid->get_num_of_dimensions();
	if (num_of_dimensions != boundaryCondition_Classes.size()) {
		std::stringstream ss;
		ss << "Number of BoundaryCondisions doesn't match with number of dimensions: " << boundaryCondition_Classes.size() << " != " << num_of_dimensions << std::endl;
		message.append(ss.str());
		delete hJI_Grid;
		return false;
	}

	if (num_of_dimensions != maxs.size()) {
		std::stringstream ss;
		ss << "Number of maxs doesn't match with number of dimensions: " << maxs.size() << " != " << num_of_dimensions << std::endl;
		message.append(ss.str());
		delete hJI_Grid;
		return false;
	}

	if (num_of_dimensions != mins.size()) {
		std::stringstream ss;
		ss << "Number of mins doesn't match with number of dimensions: " << mins.size() << " != " << num_of_dimensions << std::endl;
		message.append(ss.str());
		delete hJI_Grid;
		return false;
	}

	beacls::FloatVec modified_maxs = maxs;
	modified_maxs[2] = (FLOAT_TYPE)(maxs[2] * (1 - 1. / Ns[2]));
	hJI_Grid->set_maxs(modified_maxs);
	hJI_Grid->set_mins(mins);


	if (num_of_dimensions != expects_deriv_l_filenames.size()) {
		std::stringstream ss;
		ss << "Number of expects_deriv_l_filenames doesn't match with number of dimensions: " << expects_deriv_l_filenames.size() << " != " << num_of_dimensions << std::endl;
		message.append(ss.str());
		delete hJI_Grid;
		return false;
	}

	if (num_of_dimensions != expects_deriv_r_filenames.size()) {
		std::stringstream ss;
		ss << "Number of expects_deriv_r_filenames doesn't match with number of dimensions: " << expects_deriv_r_filenames.size() << " != " << num_of_dimensions << std::endl;
		message.append(ss.str());
		delete hJI_Grid;
		return false;
	}

	std::vector<beacls::FloatVec > expected_deriv_ls(num_of_dimensions);
	std::transform(expects_deriv_l_filenames.cbegin(), expects_deriv_l_filenames.cend(), expected_deriv_ls.begin(),([&message](const auto& rhs) {
		beacls::FloatVec data;
		beacls::MatFStream* rhs_fs = beacls::openMatFStream(rhs, beacls::MatOpenMode_Read);
		beacls::IntegerVec dummy;
		if (!load_vector(data, std::string("derivL_i"), dummy, false, rhs_fs)) {
			std::stringstream ss;
			ss << "Cannot open expected result file: " << rhs.c_str() << std::endl;
			message.append(ss.str());
			beacls::closeMatFStream(rhs_fs);
			return beacls::FloatVec();
		}
		beacls::closeMatFStream(rhs_fs);
		return data;
	}));
	if (std::any_of(expected_deriv_ls.cbegin(), expected_deriv_ls.cend(), [](const auto& rhs) { return rhs.empty(); })) {
		delete hJI_Grid;
		return false;
	}

	std::vector<beacls::FloatVec > expected_deriv_rs(num_of_dimensions);
	std::transform(expects_deriv_r_filenames.cbegin(), expects_deriv_r_filenames.cend(), expected_deriv_rs.begin(), ([&message](const auto& rhs) {
		beacls::FloatVec data;
		beacls::MatFStream* rhs_fs = beacls::openMatFStream(rhs, beacls::MatOpenMode_Read);
		beacls::IntegerVec dummy;
		if (!load_vector(data, std::string("derivR_i"), dummy, false, rhs_fs)) {
			std::stringstream ss;
			ss << "Cannot open expected result file: " << rhs.c_str() << std::endl;
			message.append(ss.str());
			beacls::closeMatFStream(rhs_fs);
			return beacls::FloatVec();
		}
		beacls::closeMatFStream(rhs_fs);
		return data;
	}));
	if (std::any_of(expected_deriv_rs.cbegin(), expected_deriv_rs.cend(), [](const auto& rhs) { return rhs.empty(); })) {
		delete hJI_Grid;
		return false;
	}


	std::vector<levelset::BoundaryCondition*> boundaryConditions(num_of_dimensions);
	std::transform(boundaryCondition_Classes.cbegin(), boundaryCondition_Classes.cend(), boundaryConditions.begin(), ([&message](auto& rhs) {
		levelset::BoundaryCondition* boundaryCondition = NULL;
		switch (rhs) {
		case BoundaryCondition_Class_AddGhostExtrapolate:
			boundaryCondition = new levelset::AddGhostExtrapolate();
			break;
		case BoundaryCondition_Class_AddGhostPeriodic:
			boundaryCondition = new levelset::AddGhostPeriodic();
			break;
		default:
			std::stringstream ss;
			ss << "Invalid BoundaryCondition_Class: " << rhs << std::endl;
			message.append(ss.str());
			break;
		}
		return boundaryCondition;
	}));
	if (std::any_of(boundaryConditions.cbegin(), boundaryConditions.cend(), [](const auto& rhs) { return !(rhs && rhs->valid()); })) {
		for_each(boundaryConditions.cbegin(), boundaryConditions.cend(), ([](const auto& rhs) {
			if (rhs) delete rhs;
		}));
		delete hJI_Grid;
		return false;
	}

	hJI_Grid->set_boundaryConditions(boundaryConditions);

	if (!hJI_Grid->processGrid()) {
		std::stringstream ss;
		ss << "Cannot process grid" << std::endl;
		message.append(ss.str());
		for_each(boundaryConditions.cbegin(), boundaryConditions.cend(), ([](const auto& rhs) {
			if (rhs) delete rhs;
		}));
		delete hJI_Grid;
		return false;
	}


	levelset::SpatialDerivative *spatialDerivative = NULL;
	switch (spatialDerivative_Class) {
	case SpatialDerivative_Class_UpwindFirstFirst:
		spatialDerivative = new levelset::UpwindFirstFirst(hJI_Grid, type);
		break;
	case SpatialDerivative_Class_UpwindFirstENO2:
		spatialDerivative = new levelset::UpwindFirstENO2(hJI_Grid, type);
		break;
	case SpatialDerivative_Class_UpwindFirstENO3:
		spatialDerivative = new levelset::UpwindFirstENO3(hJI_Grid, type);
		break;
	case SpatialDerivative_Class_UpwindFirstWENO5:
		spatialDerivative = new levelset::UpwindFirstWENO5(hJI_Grid, type);
		break;
	default:
		std::stringstream ss;
		ss << "Invalid SpatialDerivative_Class: " << spatialDerivative_Class << std::endl;
		message.append(ss.str());

		for_each(boundaryConditions.cbegin(), boundaryConditions.cend(), ([](const auto& rhs) {
			if (rhs) delete rhs;
		}));
		delete hJI_Grid;
		return false;
	}

	bool generateAll = false;

	const size_t num_of_elements = hJI_Grid->get_sum_of_elems();
	const size_t first_dimension_loop_size = (num_of_dimensions >= 1) ? Ns[0] : 1;
	const size_t second_dimension_loop_size = (num_of_dimensions >= 2) ? Ns[1] : 1;
	const size_t third_dimension_loop_size = (num_of_dimensions >= 3) ? Ns[2] : 1;
	const size_t num_of_lines = num_of_elements / first_dimension_loop_size;
	const size_t num_of_outer_lines = std::accumulate(Ns.cbegin() + 2, Ns.cend(), (size_t)1,
		[](const auto& lhs, const auto& rhs) {return lhs * rhs; });

	const size_t num_of_activated_gpus = (num_of_gpus == 0) ? beacls::get_num_of_gpus() : num_of_gpus;
	const size_t hardware_concurrency = (std::thread::hardware_concurrency() == 0) ? 1 : std::thread::hardware_concurrency();
	const size_t actual_num_of_threads = (num_of_threads == 0) ? ((spatialDerivative->get_type() == beacls::UVecType_Cuda) ? num_of_activated_gpus : hardware_concurrency) : num_of_threads;

	const size_t num_of_parallel_loop_lines = std::min((size_t)actual_num_of_threads, num_of_outer_lines);
	const size_t parallel_loop_size = (size_t)std::ceil((FLOAT_TYPE)num_of_outer_lines / num_of_parallel_loop_lines);
	const size_t num_of_inner_lines = (size_t)std::ceil((FLOAT_TYPE)num_of_lines / num_of_outer_lines);

	const double gpu_memory_ocupancy_ratio = 1.0 / (4 + 5 * num_of_dimensions) / 2 * 0.8;
	const size_t minimum_global_memory_in_devices = beacls::get_minimum_global_memory_in_devices();
	const size_t available_line_length_for_cuda = (size_t)std::floor(minimum_global_memory_in_devices * gpu_memory_ocupancy_ratio / first_dimension_loop_size / sizeof(FLOAT_TYPE));
	const size_t available_line_length_of_chunk_for_cuda = (size_t)(std::ceil(std::floor((FLOAT_TYPE)available_line_length_for_cuda / actual_num_of_threads) / second_dimension_loop_size)*second_dimension_loop_size);
	const size_t prefered_line_length_of_chunk_for_cuda = (size_t)(std::ceil(std::ceil((FLOAT_TYPE)num_of_lines / actual_num_of_threads) / second_dimension_loop_size)*second_dimension_loop_size) * num_of_activated_gpus;
	const size_t line_length_of_chunk_for_cuda = available_line_length_of_chunk_for_cuda < prefered_line_length_of_chunk_for_cuda ? available_line_length_of_chunk_for_cuda : prefered_line_length_of_chunk_for_cuda;
	const size_t prefered_line_length_of_chunk_for_cpu = (size_t)std::ceil((FLOAT_TYPE)1024 / first_dimension_loop_size);
	const size_t prefered_line_length_of_chunk = (spatialDerivative->get_type() == beacls::UVecType_Cuda) ? line_length_of_chunk_for_cuda : prefered_line_length_of_chunk_for_cpu;

	const size_t actual_line_length_of_chunk = (line_length_of_chunk == 0) ? prefered_line_length_of_chunk : line_length_of_chunk;	//!< T.B.D.

	const size_t num_of_chunk_lines = (actual_line_length_of_chunk < second_dimension_loop_size) ? actual_line_length_of_chunk : second_dimension_loop_size;
	const size_t num_of_slices = (actual_line_length_of_chunk < second_dimension_loop_size) ? 1 : (size_t)std::floor((FLOAT_TYPE)actual_line_length_of_chunk / second_dimension_loop_size);

	beacls::UVecDepth depth = beacls::type_to_depth<FLOAT_TYPE>();

	std::vector<beacls::UVec> deriv_ls(num_of_dimensions);
	std::transform(expected_deriv_ls.cbegin(), expected_deriv_ls.cend(), deriv_ls.begin(), [depth](const auto& rhs) { 
		return beacls::UVec(depth, beacls::UVecType_Vector, rhs.size());
	});
	std::vector<beacls::UVec> deriv_rs(num_of_dimensions);
	std::transform(expected_deriv_rs.cbegin(), expected_deriv_rs.cend(), deriv_rs.begin(), [depth](const auto& rhs) {
		return beacls::UVec(depth, beacls::UVecType_Vector, rhs.size());
	});


	bool allSucceed = true;
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int parallel_line_index = 0; parallel_line_index < num_of_parallel_loop_lines; ++parallel_line_index) {
		if (num_of_activated_gpus > 1) {
			beacls::set_gpu_id(parallel_line_index%num_of_activated_gpus);
		}
		levelset::SpatialDerivative* loop_local_spatialDerivative = spatialDerivative->clone();
		std::string local_message;
		beacls::UVec deriv_l_line_uvec(depth, type, first_dimension_loop_size*num_of_chunk_lines*num_of_slices);
		beacls::UVec deriv_r_line_uvec(depth, type, first_dimension_loop_size*num_of_chunk_lines*num_of_slices);
		const size_t paralleled_out_line_size = std::min(parallel_loop_size, (num_of_outer_lines > parallel_line_index * parallel_loop_size) ? num_of_outer_lines - parallel_line_index * parallel_loop_size : 0);
		size_t actual_chunk_size = num_of_chunk_lines;
		size_t actual_num_of_slices = num_of_slices;
		for (size_t line_index = 0; line_index < paralleled_out_line_size*num_of_inner_lines; line_index += actual_chunk_size*actual_num_of_slices) {
			const size_t inner_line_index = line_index + parallel_line_index*parallel_loop_size*num_of_inner_lines;
			const size_t second_line_index = inner_line_index % second_dimension_loop_size;
			actual_chunk_size = std::min(num_of_chunk_lines, second_dimension_loop_size - second_line_index);
			const size_t third_line_index = (inner_line_index / second_dimension_loop_size) % third_dimension_loop_size;
			actual_num_of_slices = std::min(num_of_slices, third_dimension_loop_size - third_line_index);
			const size_t line_begin = inner_line_index;
			size_t expected_result_offset = line_begin * first_dimension_loop_size;
			size_t slices_result_size = actual_chunk_size * first_dimension_loop_size*actual_num_of_slices;
			size_t chunk_result_size = actual_chunk_size * first_dimension_loop_size;
			if (deriv_l_line_uvec.size() != slices_result_size) deriv_l_line_uvec.resize(slices_result_size);
			if (deriv_r_line_uvec.size() != slices_result_size) deriv_r_line_uvec.resize(slices_result_size);
			for (size_t dim = 0; dim < num_of_dimensions; ++dim) {
				loop_local_spatialDerivative->execute(deriv_l_line_uvec, deriv_r_line_uvec, hJI_Grid, data.data(), dim, generateAll, line_begin, chunk_result_size, actual_num_of_slices);
				loop_local_spatialDerivative->synchronize(dim);
				beacls::UVec tmp_deriv_l;
				beacls::UVec tmp_deriv_r;

				if (deriv_l_line_uvec.type() == beacls::UVecType_Cuda) deriv_l_line_uvec.convertTo(tmp_deriv_l, beacls::UVecType_Vector);
				else tmp_deriv_l = deriv_l_line_uvec;
				if (deriv_r_line_uvec.type() == beacls::UVecType_Cuda) deriv_r_line_uvec.convertTo(tmp_deriv_r, beacls::UVecType_Vector);
				else tmp_deriv_r = deriv_r_line_uvec;

				FLOAT_TYPE* deriv_l_line_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_deriv_l).ptr();
				FLOAT_TYPE* deriv_r_line_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_deriv_r).ptr();
				FLOAT_TYPE* deriv_l_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_ls[dim]).ptr();
				FLOAT_TYPE* deriv_r_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_rs[dim]).ptr();
				memcpy(deriv_l_ptr + expected_result_offset, deriv_l_line_ptr, slices_result_size * sizeof(FLOAT_TYPE));
				memcpy(deriv_r_ptr + expected_result_offset, deriv_r_line_ptr, slices_result_size * sizeof(FLOAT_TYPE));
			}
		}
#ifdef _OPENMP
#pragma omp critical 
		{
#if defined(WIN32)	// Windows
//			double random_wait = 1000 * 100;
			std::this_thread::sleep_for(std::chrono::microseconds(100*1000));	//!< Workaround for test error which occurs in OPENMP mode.
#endif	/* Windows */
		}
#endif
		if(loop_local_spatialDerivative) delete loop_local_spatialDerivative;
	}
	for (size_t dim = 0; dim < num_of_dimensions; ++dim) {
		const beacls::FloatVec& expected_deriv_l = expected_deriv_ls[dim];
		const beacls::FloatVec& expected_deriv_r = expected_deriv_rs[dim];
		const FLOAT_TYPE* deriv_l_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_ls[dim]).ptr();
		const FLOAT_TYPE* deriv_r_ptr = beacls::UVec_<FLOAT_TYPE>(deriv_rs[dim]).ptr();

		const size_t num_of_lines_expected = expected_deriv_l.size() / first_dimension_loop_size;
		{
			FLOAT_TYPE min_diff = std::numeric_limits<FLOAT_TYPE>::max();
			FLOAT_TYPE max_diff = 0;
			FLOAT_TYPE first_diff = 0;
			FLOAT_TYPE sum_of_square = 0;
			size_t min_diff_line = 0;
			size_t max_diff_line = 0;
			size_t first_diff_line = 0;
			size_t min_diff_index = 0;
			size_t max_diff_index = 0;
			size_t first_diff_index = 0;
			size_t num_of_diffs = 0;
			size_t num_of_datas = 0;
			for (size_t line = 0; line < num_of_lines_expected; ++line) {
				for (size_t index = 0; index < first_dimension_loop_size; ++index) {
					FLOAT_TYPE expected_result = expected_deriv_l[index + line*first_dimension_loop_size];
					FLOAT_TYPE result = deriv_l_ptr[index + line*first_dimension_loop_size];
					++num_of_datas;
					const FLOAT_TYPE diff = std::abs(expected_result - result);
					if (diff > small_diff) {
						allSucceed = false;
						if (min_diff > diff) {
							min_diff = diff;
							min_diff_line = line;
							min_diff_index = index;
						}
						if (max_diff < diff) {
							max_diff = diff;
							max_diff_line = line;
							max_diff_index = index;
						}
						if (first_diff == 0) {
							first_diff = diff;
							first_diff_line = line;
							first_diff_index = index;
						}
						sum_of_square += diff * diff;
						++num_of_diffs;
					}
				}
			}
			if (!allSucceed) {
				const FLOAT_TYPE rms = std::sqrt(sum_of_square / num_of_datas);
				std::stringstream ss;
				ss << "Error: # of L-Diffs = " << num_of_diffs << ", L-RMS = " << std::setprecision(16) << rms << std::resetiosflags(std::ios_base::floatfield)
					<< ", First L-Diff " << std::setprecision(16) << first_diff << std::resetiosflags(std::ios_base::floatfield) << "@(" << first_diff_line << "," << first_diff_index
					<< "), Max L-Diff " << std::setprecision(16) << max_diff << std::resetiosflags(std::ios_base::floatfield) << "@(" << max_diff_line << "," << max_diff_index
					<< "), Min L-Diff " << std::setprecision(16) << min_diff << std::resetiosflags(std::ios_base::floatfield) << "@(" << min_diff_line << "," << min_diff_index
					<< ")" << std::endl;
				message.append(ss.str());
			}
		}
		{
			FLOAT_TYPE min_diff = std::numeric_limits<FLOAT_TYPE>::max();
			FLOAT_TYPE max_diff = 0;
			FLOAT_TYPE first_diff = 0;
			FLOAT_TYPE sum_of_square = 0;
			size_t min_diff_line = 0;
			size_t max_diff_line = 0;
			size_t first_diff_line = 0;
			size_t min_diff_index = 0;
			size_t max_diff_index = 0;
			size_t first_diff_index = 0;
			size_t num_of_diffs = 0;
			size_t num_of_datas = 0;
			for (size_t line = 0; line < num_of_lines_expected; ++line) {
				for (size_t index = 0; index < first_dimension_loop_size; ++index) {
					FLOAT_TYPE expected_result = expected_deriv_r[index + line*first_dimension_loop_size];
					FLOAT_TYPE result = deriv_r_ptr[index + line*first_dimension_loop_size];
					++num_of_datas;
					const FLOAT_TYPE diff = std::abs(expected_result - result);
					if (diff > small_diff) {
						allSucceed = false;
						if (min_diff > diff) {
							min_diff = diff;
							min_diff_line = line;
							min_diff_index = index;
						}
						if (max_diff < diff) {
							max_diff = diff;
							max_diff_line = line;
							max_diff_index = index;
						}
						if (first_diff == 0) {
							first_diff = diff;
							first_diff_line = line;
							first_diff_index = index;
						}
						sum_of_square += diff * diff;
						++num_of_diffs;
					}
				}
			}
			if (!allSucceed) {
				const FLOAT_TYPE rms = std::sqrt(sum_of_square / num_of_datas);
				std::stringstream ss;
				ss << "Error: # of R-Diffs = " << num_of_diffs << ", R-RMS = " << std::setprecision(16) << rms << std::resetiosflags(std::ios_base::floatfield)
					<< ", First R-Diff " << std::setprecision(16) << first_diff << std::resetiosflags(std::ios_base::floatfield) << "@(" << first_diff_line << "," << first_diff_index
					<< "), Max R-Diff " << std::setprecision(16) << max_diff << std::resetiosflags(std::ios_base::floatfield) << "@(" << max_diff_line << "," << max_diff_index
					<< "), Min R-Diff " << std::setprecision(16) << min_diff << std::resetiosflags(std::ios_base::floatfield) << "@(" << min_diff_line << "," << min_diff_index
					<< ")" << std::endl;
				message.append(ss.str());
			}
		}
	}

	if (spatialDerivative) delete spatialDerivative;
	for_each(boundaryConditions.cbegin(), boundaryConditions.cend(), ([](const auto& rhs) {
		if (rhs) delete rhs;
	}));

	if (hJI_Grid) delete hJI_Grid;

	return allSucceed;
}

