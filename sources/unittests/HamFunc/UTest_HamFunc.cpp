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
#include "UTest_HamFunc.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

bool run_UTest_HamFunc(
	std::string &message,
	const std::string& src_data_filename,
	const std::vector<std::string>& src_deriv_c_filenames,
	const std::string& expects_filename,
	const beacls::FloatVec &maxs,
	const beacls::FloatVec &mins,
	levelset::SchemeData *schemeData,
	const beacls::UVecType type,
	const FLOAT_TYPE small_diff,
	const size_t line_length_of_chunk,
	const int num_of_threads,
	const int num_of_gpus,
	const bool enable_user_defined_dynamics_on_gpu
) {
	if (src_data_filename.empty()) {
		std::stringstream ss;
		ss << "Error source file name is empty" << std::endl;
		message.append(ss.str());
		return false;
	}
	FLOAT_TYPE t = 0;
	beacls::MatFStream* src_data_filename_fs = beacls::openMatFStream(src_data_filename, beacls::MatOpenMode_Read);
	if (!src_data_filename_fs || !load_value(t, std::string("t"), false, src_data_filename_fs)) {
		std::stringstream ss;
		ss << "Cannot open source file: " << src_data_filename.c_str() << std::endl;
		message.append(ss.str());
		if (src_data_filename_fs)
			beacls::closeMatFStream(src_data_filename_fs);
		return false;
	}
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
	schemeData->set_grid(hJI_Grid);


	size_t num_of_dimensions = hJI_Grid->get_num_of_dimensions();

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


	if (num_of_dimensions != src_deriv_c_filenames.size()) {
		std::stringstream ss;
		ss << "Number of src_deriv_c_filenames doesn't match with number of dimensions: " << src_deriv_c_filenames.size() << " != " << num_of_dimensions << std::endl;
		message.append(ss.str());
		delete hJI_Grid;
		return false;
	}

	if (expects_filename.empty()) {
		std::stringstream ss;
		ss << "Error source file name is empty" << std::endl;
		message.append(ss.str());
		return false;
	}
	
	std::vector<beacls::FloatVec > src_deriv_cs(num_of_dimensions);
	std::transform(src_deriv_c_filenames.cbegin(), src_deriv_c_filenames.cend(), src_deriv_cs.begin(),([&message](const auto& rhs) {
		beacls::FloatVec data;
		beacls::MatFStream* rhs_fs = beacls::openMatFStream(rhs, beacls::MatOpenMode_Read);
		beacls::IntegerVec dummy;
		if (!load_vector(data, std::string("derivC_i"), dummy, false, rhs_fs)) {
			std::stringstream ss;
			ss << "Cannot open source derivative file: " << rhs.c_str() << std::endl;
			message.append(ss.str());
			return beacls::FloatVec();
		}
		beacls::closeMatFStream(rhs_fs);
		return data;
	}));
	if (std::any_of(src_deriv_cs.cbegin(), src_deriv_cs.cend(), [](const auto& rhs) { return rhs.empty(); })) {
		delete hJI_Grid;
		return false;
	}

	beacls::FloatVec expected_data;
	beacls::MatFStream* expects_filename_fs = beacls::openMatFStream(expects_filename, beacls::MatOpenMode_Read);
	beacls::IntegerVec dummy;
	if (!load_vector(expected_data, std::string("ham"), dummy, false, expects_filename_fs)) {
		std::stringstream ss;
		ss << "Cannot open expected ham file: " << expects_filename.c_str() << std::endl;
		message.append(ss.str());
		delete hJI_Grid;
		return false;
	}
	beacls::closeMatFStream(expects_filename_fs);

	if (!hJI_Grid->processGrid()) {
		std::stringstream ss;
		ss << "Cannot process grid" << std::endl;
		message.append(ss.str());
		delete hJI_Grid;
		return false;
	}

	const size_t num_of_elements = hJI_Grid->get_sum_of_elems();
	const size_t first_dimension_loop_size = (num_of_dimensions >= 1) ? Ns[0] : 1;
	const size_t second_dimension_loop_size = (num_of_dimensions >= 2) ? Ns[1] : 1;
	const size_t third_dimension_loop_size = (num_of_dimensions >= 3) ? Ns[2] : 1;
	const size_t num_of_lines = num_of_elements / first_dimension_loop_size;
	const size_t num_of_outer_lines = std::accumulate(Ns.cbegin() + 2, Ns.cend(), (size_t)1,
		[](const auto& lhs, const auto& rhs) {return lhs * rhs; });

	const size_t num_of_activated_gpus = (num_of_gpus == 0) ? beacls::get_num_of_gpus() : num_of_gpus;
	const size_t hardware_concurrency = (std::thread::hardware_concurrency() == 0) ? 1 : std::thread::hardware_concurrency();
	const size_t actual_num_of_threads = (num_of_threads == 0) ? ((type == beacls::UVecType_Cuda) ? num_of_activated_gpus : hardware_concurrency) : num_of_threads;

	const size_t num_of_parallel_loop_lines = std::min((size_t)actual_num_of_threads, num_of_outer_lines);
	const size_t parallel_loop_size = (size_t)std::ceil((FLOAT_TYPE)num_of_outer_lines / num_of_parallel_loop_lines);
	const size_t num_of_inner_lines = (size_t)std::ceil((FLOAT_TYPE)num_of_lines / num_of_outer_lines);

	const double gpu_memory_ocupancy_ratio = 1.0 / (2 + 6 * num_of_dimensions) / 2 * 0.8;
	const size_t minimum_global_memory_in_devices = beacls::get_minimum_global_memory_in_devices();
	const size_t available_line_length_for_cuda = (size_t)std::floor(minimum_global_memory_in_devices * gpu_memory_ocupancy_ratio / first_dimension_loop_size / sizeof(FLOAT_TYPE));
	const size_t available_line_length_of_chunk_for_cuda = (size_t)(std::ceil(std::floor((FLOAT_TYPE)available_line_length_for_cuda / actual_num_of_threads) / second_dimension_loop_size)*second_dimension_loop_size);
	const size_t prefered_line_length_of_chunk_for_cuda = (size_t)(std::ceil(std::ceil((FLOAT_TYPE)num_of_lines / actual_num_of_threads) / second_dimension_loop_size)*second_dimension_loop_size) * num_of_activated_gpus;
	const size_t line_length_of_chunk_for_cuda = available_line_length_of_chunk_for_cuda < prefered_line_length_of_chunk_for_cuda ? available_line_length_of_chunk_for_cuda : prefered_line_length_of_chunk_for_cuda;
	const size_t prefered_line_length_of_chunk_for_cpu = (size_t)std::ceil((FLOAT_TYPE)1024 / first_dimension_loop_size);
	const size_t prefered_line_length_of_chunk = (type == beacls::UVecType_Cuda) ? line_length_of_chunk_for_cuda : prefered_line_length_of_chunk_for_cpu;

	const size_t actual_line_length_of_chunk = (line_length_of_chunk == 0) ? prefered_line_length_of_chunk : line_length_of_chunk;	//!< T.B.D.

	const size_t num_of_chunk_lines = (actual_line_length_of_chunk < second_dimension_loop_size) ? actual_line_length_of_chunk : second_dimension_loop_size;
	const size_t num_of_slices = (actual_line_length_of_chunk < second_dimension_loop_size) ? 1 : (size_t)std::floor((FLOAT_TYPE)actual_line_length_of_chunk / second_dimension_loop_size);

	beacls::UVecDepth depth = beacls::type_to_depth<FLOAT_TYPE>();
	beacls::UVec src_data_uvec(data, beacls::UVecType_Vector, true);
	bool allSucceed = true;
	std::vector<beacls::FloatVec > derivMins;
	std::vector<beacls::FloatVec > derivMaxs;

	beacls::UVec results(depth, beacls::UVecType_Vector, expected_data.size());
	std::vector<levelset::SchemeData*> thread_local_schemeDatas(num_of_parallel_loop_lines);
	std::for_each(thread_local_schemeDatas.begin(), thread_local_schemeDatas.end(), [schemeData](auto& rhs) {
		rhs = schemeData->clone();
	});

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int parallel_line_index = 0; parallel_line_index < num_of_parallel_loop_lines; ++parallel_line_index) {
		if (num_of_activated_gpus > 1) {
			beacls::set_gpu_id(parallel_line_index%num_of_activated_gpus);
		}
		levelset::SchemeData* thread_local_schemeData = thread_local_schemeDatas[parallel_line_index];
		std::string local_message;
		beacls::UVec ham_line_uvec(depth, type, first_dimension_loop_size*num_of_chunk_lines*num_of_slices);
		std::vector<beacls::UVec> x_uvecs;
		beacls::FloatVec new_step_bound_invs(num_of_dimensions*num_of_chunk_lines*num_of_slices);
		const size_t paralleled_out_line_size = std::min(parallel_loop_size, (num_of_outer_lines > parallel_line_index * parallel_loop_size) ? num_of_outer_lines - parallel_line_index * parallel_loop_size : 0);
		size_t actual_chunk_size = num_of_chunk_lines;
		size_t actual_num_of_slices = num_of_slices;
		for (size_t line_index = 0; line_index < paralleled_out_line_size*num_of_inner_lines; line_index += actual_chunk_size*actual_num_of_slices) {
			const size_t inner_line_index = line_index + parallel_line_index*parallel_loop_size*num_of_inner_lines;
			const size_t second_line_index = inner_line_index % second_dimension_loop_size;
			actual_chunk_size = std::min(num_of_chunk_lines, second_dimension_loop_size - second_line_index);
			const size_t third_line_index = (inner_line_index / second_dimension_loop_size) % third_dimension_loop_size;
			actual_num_of_slices = std::min(num_of_slices, third_dimension_loop_size - third_line_index);
			size_t line_begin = inner_line_index;
			size_t expected_result_offset = line_begin * first_dimension_loop_size;
			size_t slices_result_size = actual_chunk_size * first_dimension_loop_size*actual_num_of_slices;
			if (ham_line_uvec.size() != slices_result_size) ham_line_uvec.resize(slices_result_size);
			std::vector<beacls::UVec> src_deriv_c_uvecs(num_of_dimensions);
			std::vector<beacls::UVec> src_deriv_c_cpu_uvecs(num_of_dimensions);
			for (size_t dim = 0; dim < num_of_dimensions; ++dim) {
				beacls::UVec src_deriv_c_line_uvec(beacls::type_to_depth<FLOAT_TYPE>(), beacls::UVecType_Vector, slices_result_size);

				const beacls::FloatVec& src_deriv_c = src_deriv_cs[dim];
				std::copy(src_deriv_c.cbegin() + expected_result_offset, src_deriv_c.cbegin() + expected_result_offset + slices_result_size, (beacls::UVec_<FLOAT_TYPE>(src_deriv_c_line_uvec)).vec()->begin());
				src_deriv_c_cpu_uvecs[dim] = src_deriv_c_line_uvec;

				beacls::UVec tmp_deriv_c;
				src_deriv_c_line_uvec.convertTo(tmp_deriv_c, type);
				src_deriv_c_uvecs[dim] = tmp_deriv_c;
			}
			if (enable_user_defined_dynamics_on_gpu && (type == beacls::UVecType_Cuda)) {
				const std::vector<beacls::FloatVec >& xs = hJI_Grid->get_xss();
				if (x_uvecs.size() != xs.size()) x_uvecs.resize(xs.size());
				for (size_t dim = 0; dim < xs.size(); ++dim) {
					const beacls::FloatVec& xs_dim = xs[dim];
					beacls::UVec& x_uvecs_dim = x_uvecs[dim];
					if (x_uvecs_dim.type() != type) x_uvecs_dim = beacls::UVec(depth, beacls::UVecType_Cuda, slices_result_size);
					else if (x_uvecs_dim.size() != slices_result_size) x_uvecs_dim.resize(slices_result_size);
					copyHostPtrToUVec(x_uvecs_dim, xs_dim.data() + expected_result_offset, slices_result_size);
				}
			}

			if (!enable_user_defined_dynamics_on_gpu 
				|| (type != beacls::UVecType_Cuda)
				|| !thread_local_schemeData->hamFunc_cuda(
				ham_line_uvec,
				t,
				data,
				src_deriv_c_uvecs,
				x_uvecs,
				expected_result_offset, slices_result_size)) {

				thread_local_schemeData->hamFunc(
					ham_line_uvec,
					t,
					data,
					src_deriv_c_cpu_uvecs,
				expected_result_offset, slices_result_size);
			}
			beacls::UVec tmp_ham;
			if (ham_line_uvec.type() == beacls::UVecType_Cuda) ham_line_uvec.convertTo(tmp_ham, beacls::UVecType_Vector);
			else tmp_ham = ham_line_uvec;

			FLOAT_TYPE* ham_line_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_ham).ptr();
			FLOAT_TYPE* results_ptr = beacls::UVec_<FLOAT_TYPE>(results).ptr();
			memcpy(results_ptr + expected_result_offset, ham_line_ptr, slices_result_size * sizeof(FLOAT_TYPE));
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
	}
	const size_t num_of_lines_expected = expected_data.size() / first_dimension_loop_size;
	const FLOAT_TYPE* results_ptr = beacls::UVec_<FLOAT_TYPE>(results).ptr();
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
			FLOAT_TYPE expected_result = expected_data[index + line*first_dimension_loop_size];
			FLOAT_TYPE result = results_ptr[index + line*first_dimension_loop_size];
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
		ss << "Error: # of Diffs = " << num_of_diffs << ", RMS = " << std::setprecision(16) << rms << std::resetiosflags(std::ios_base::floatfield)
			<< ", First Diff " << std::setprecision(16) <<  first_diff << std::resetiosflags(std::ios_base::floatfield) << "@(" << first_diff_line << "," << first_diff_index
			<< "), Max Diff " << std::setprecision(16) << max_diff << std::resetiosflags(std::ios_base::floatfield) << "@(" << max_diff_line << "," << max_diff_index
			<< "), Min Diff " << std::setprecision(16) << min_diff << std::resetiosflags(std::ios_base::floatfield) << "@(" << min_diff_line << "," << min_diff_index
			<< ")" << std::endl;
		message.append(ss.str());
	}
	std::for_each(thread_local_schemeDatas.begin(), thread_local_schemeDatas.end(), [](auto& rhs) {
		if (rhs) delete rhs;
		rhs = NULL;
	});
	if (hJI_Grid) delete hJI_Grid;

	return allSucceed;
}

