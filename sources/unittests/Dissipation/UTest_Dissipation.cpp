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
#include <deque>
#include <thread>
#include <cstdlib>
#include <macro.hpp>
#include "UTest_Dissipation.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

bool run_UTest_Dissipation(
	std::string &message,
	const std::string& src_data_filename,
	const std::vector<std::string>& src_deriv_l_filenames,
	const std::vector<std::string>& src_deriv_r_filenames,
	const std::string& expects_filename,
	const beacls::FloatVec &maxs,
	const beacls::FloatVec &mins,
	levelset::SchemeData *schemeData,
	const Dissipation_Class& dissipation_class,
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


	if (num_of_dimensions != src_deriv_l_filenames.size()) {
		std::stringstream ss;
		ss << "Number of src_deriv_l_filenames doesn't match with number of dimensions: " << src_deriv_l_filenames.size() << " != " << num_of_dimensions << std::endl;
		message.append(ss.str());
		delete hJI_Grid;
		return false;
	}

	if (num_of_dimensions != src_deriv_r_filenames.size()) {
		std::stringstream ss;
		ss << "Number of src_deriv_r_filenames doesn't match with number of dimensions: " << src_deriv_r_filenames.size() << " != " << num_of_dimensions << std::endl;
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
	
	std::vector<beacls::FloatVec > src_deriv_ls(num_of_dimensions);
	std::transform(src_deriv_l_filenames.cbegin(), src_deriv_l_filenames.cend(), src_deriv_ls.begin(),([&message](const auto& rhs) {
		beacls::FloatVec data;
		beacls::MatFStream* rhs_fs = beacls::openMatFStream(rhs, beacls::MatOpenMode_Read);
		beacls::IntegerVec dummy;
		if (!load_vector(data, std::string("derivL_i"), dummy, false, rhs_fs)) {
			std::stringstream ss;
			ss << "Cannot open source derivL file: " << rhs.c_str() << std::endl;
			message.append(ss.str());
			beacls::closeMatFStream(rhs_fs);
			return beacls::FloatVec();
		}
		beacls::closeMatFStream(rhs_fs);
		return data;
	}));
	if (std::any_of(src_deriv_ls.cbegin(), src_deriv_ls.cend(), [](const auto& rhs) { return rhs.empty(); })) {
		delete hJI_Grid;
		return false;
	}

	std::vector<beacls::FloatVec > src_deriv_rs(num_of_dimensions);
	std::transform(src_deriv_r_filenames.cbegin(), src_deriv_r_filenames.cend(), src_deriv_rs.begin(), ([&message](const auto& rhs) {
		beacls::FloatVec data;
		beacls::MatFStream* rhs_fs = beacls::openMatFStream(rhs, beacls::MatOpenMode_Read);
		beacls::IntegerVec dummy;
		if (!load_vector(data, std::string("derivR_i"), dummy, false, rhs_fs)) {
			std::stringstream ss;
			ss << "Cannot open source derivR file: " << rhs.c_str() << std::endl;
			message.append(ss.str());
			beacls::closeMatFStream(rhs_fs);
			return beacls::FloatVec();
		}
		beacls::closeMatFStream(rhs_fs);
		return data;
	}));
	if (std::any_of(src_deriv_rs.cbegin(), src_deriv_rs.cend(), [](const auto& rhs) { return rhs.empty(); })) {
		delete hJI_Grid;
		return false;
	}

	beacls::FloatVec expected_data;
	beacls::MatFStream* expects_filename_fs = beacls::openMatFStream(expects_filename, beacls::MatOpenMode_Read);
	beacls::IntegerVec dummy;
	if (!load_vector(expected_data, std::string("diss"),dummy,false, expects_filename_fs)) {
		std::stringstream ss;
		ss << "Cannot open expected diss file: " << expects_filename.c_str() << std::endl;
		message.append(ss.str());
		delete hJI_Grid;
		beacls::closeMatFStream(expects_filename_fs);
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

	levelset::Dissipation* dissipation;
	switch (dissipation_class) {
	case Dissipation_Class_ArtificialDissipationGLF:
		dissipation = new levelset::ArtificialDissipationGLF();
		break;
	case Dissipation_Class_ArtificialDissipationLLF:
		//		dissipation = new ArtificialDissipationLLF();
		//		break;
	case Dissipation_Class_ArtificialDissipationLLLF:
		//		dissipation = new ArtificialDissipationLLLF();
		//		break;
	default:
	{
		std::stringstream ss;
		ss << "Invalid Dissipation_class: " << dissipation_class << std::endl;
		message.append(ss.str());
	}
	delete hJI_Grid;
	return false;
	}
	schemeData->set_dissipation(dissipation);

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

	std::vector<beacls::FloatVec > derivMins(num_of_dimensions);
	std::vector<beacls::FloatVec > derivMaxs(num_of_dimensions);
	std::vector<beacls::FloatVec > local_derivMinss(num_of_parallel_loop_lines);
	std::vector<beacls::FloatVec > local_derivMaxss(num_of_parallel_loop_lines);
	for_each(local_derivMinss.begin(), local_derivMinss.end(), ([num_of_dimensions](auto& rhs) {
		if (rhs.size() != num_of_dimensions) rhs.resize(num_of_dimensions);
	}));
	for_each(local_derivMaxss.begin(), local_derivMaxss.end(), ([num_of_dimensions](auto& rhs) {
		if (rhs.size() != num_of_dimensions) rhs.resize(num_of_dimensions);
	}));
	beacls::UVec results(depth, beacls::UVecType_Vector, expected_data.size());
	std::vector<levelset::SchemeData*> thread_local_schemeDatas(num_of_parallel_loop_lines);
	std::for_each(thread_local_schemeDatas.begin(), thread_local_schemeDatas.end(), [schemeData](auto& rhs) {
		rhs = schemeData->clone();
	});
	std::deque<bool> executeAgains(num_of_parallel_loop_lines);
	bool executeAgain;
	for_each(derivMins.begin(), derivMins.end(), ([](auto& rhs) { rhs.clear(); }));
	for_each(derivMaxs.begin(), derivMaxs.end(), ([](auto& rhs) { rhs.clear(); }));
	do {
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int parallel_line_index = 0; parallel_line_index < num_of_parallel_loop_lines; ++parallel_line_index) {
			if (num_of_activated_gpus > 1) {
				beacls::set_gpu_id(parallel_line_index%num_of_activated_gpus);
			}
			levelset::SchemeData* thread_local_schemeData = thread_local_schemeDatas[parallel_line_index];
			size_t loop_local_index = 0;
			levelset::Dissipation* loop_local_dissipation = thread_local_schemeData->get_dissipation();
			beacls::FloatVec& local_derivMins = local_derivMinss[parallel_line_index];
			beacls::FloatVec& local_derivMaxs = local_derivMaxss[parallel_line_index];
			beacls::UVec diss_line_uvec(depth, type, first_dimension_loop_size*num_of_chunk_lines*num_of_slices);
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
				const size_t line_begin = inner_line_index;
				size_t expected_result_offset = line_begin * first_dimension_loop_size;
				size_t slices_result_size = actual_chunk_size * first_dimension_loop_size*actual_num_of_slices;
				if (diss_line_uvec.size() != slices_result_size) diss_line_uvec.resize(slices_result_size);


				std::vector<beacls::UVec> src_deriv_l_uvecs(num_of_dimensions);
				std::vector<beacls::UVec> src_deriv_r_uvecs(num_of_dimensions);
				for (size_t dim = 0; dim < num_of_dimensions; ++dim) {
					beacls::UVec src_deriv_l_line_uvec(beacls::type_to_depth<FLOAT_TYPE>(), beacls::UVecType_Vector, slices_result_size);
					beacls::UVec src_deriv_r_line_uvec(beacls::type_to_depth<FLOAT_TYPE>(), beacls::UVecType_Vector, slices_result_size);

					const beacls::FloatVec& src_deriv_l = src_deriv_ls[dim];
					const beacls::FloatVec& src_deriv_r = src_deriv_rs[dim];
					std::copy(src_deriv_l.cbegin() + expected_result_offset, src_deriv_l.cbegin() + expected_result_offset + slices_result_size, (beacls::UVec_<FLOAT_TYPE>(src_deriv_l_line_uvec)).vec()->begin());
					std::copy(src_deriv_r.cbegin() + expected_result_offset, src_deriv_r.cbegin() + expected_result_offset + slices_result_size, (beacls::UVec_<FLOAT_TYPE>(src_deriv_r_line_uvec)).vec()->begin());
					beacls::UVec tmp_deriv_l;
					beacls::UVec tmp_deriv_r;
					src_deriv_l_line_uvec.convertTo(tmp_deriv_l, type);
					src_deriv_r_line_uvec.convertTo(tmp_deriv_r, type);

					src_deriv_l_uvecs[dim] = tmp_deriv_l;
					src_deriv_r_uvecs[dim] = tmp_deriv_r;
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

				std::vector<beacls::UVec> deriv_max_uvecs(derivMaxs.size());
				std::vector<beacls::UVec> deriv_min_uvecs(derivMins.size());
				deriv_max_uvecs.resize(derivMaxs.size());
				deriv_min_uvecs.resize(derivMins.size());
				std::transform(derivMaxs.cbegin(), derivMaxs.cend(), deriv_max_uvecs.begin(), [](const auto& rhs) { return beacls::UVec(rhs, beacls::UVecType_Vector, false); });
				std::transform(derivMins.cbegin(), derivMins.cend(), deriv_min_uvecs.begin(), [](const auto& rhs) { return beacls::UVec(rhs, beacls::UVecType_Vector, false); });

				bool getReductionLater = false;
				if (loop_local_dissipation->execute(
					diss_line_uvec, new_step_bound_invs, deriv_min_uvecs, deriv_max_uvecs,
					getReductionLater,
					t, src_data_uvec, src_deriv_l_uvecs, src_deriv_r_uvecs,  x_uvecs,
					thread_local_schemeData, expected_result_offset, enable_user_defined_dynamics_on_gpu)) {
					beacls::UVec tmp_diss;

					if (getReductionLater) {
						dissipation->get_reduction(new_step_bound_invs, deriv_min_uvecs, deriv_max_uvecs, diss_line_uvec, schemeData, true);
					}

					if (diss_line_uvec.type() == beacls::UVecType_Cuda) diss_line_uvec.convertTo(tmp_diss, beacls::UVecType_Vector);
					else tmp_diss = diss_line_uvec;

					FLOAT_TYPE* diss_line_ptr = beacls::UVec_<FLOAT_TYPE>(tmp_diss).ptr();
					FLOAT_TYPE* results_ptr = beacls::UVec_<FLOAT_TYPE>(results).ptr();
					memcpy(results_ptr + expected_result_offset, diss_line_ptr, slices_result_size * sizeof(FLOAT_TYPE));
					executeAgains[parallel_line_index] = false;
				}
				else {
					executeAgains[parallel_line_index] = true;
				}
				std::vector<beacls::FloatVec > chunk_derivMins = derivMins;
				std::vector<beacls::FloatVec > chunk_derivMaxs = derivMaxs;
				for (int dimension = 0; dimension < (int)num_of_dimensions; ++dimension) {
					beacls::FloatVec& cdMins_d = chunk_derivMins[dimension];
					beacls::FloatVec& cdMaxs_d = chunk_derivMaxs[dimension];
					const FLOAT_TYPE chunk_derivMin = beacls::min_value<FLOAT_TYPE>(cdMins_d.cbegin(), cdMins_d.cend());
					const FLOAT_TYPE chunk_derivMax = beacls::max_value<FLOAT_TYPE>(cdMaxs_d.cbegin(), cdMaxs_d.cend());
					FLOAT_TYPE& ldMins_d = local_derivMins[dimension];
					FLOAT_TYPE& ldMaxs_d = local_derivMaxs[dimension];
					ldMins_d = (loop_local_index != 0) ? std::min<FLOAT_TYPE>(ldMins_d, chunk_derivMin) : chunk_derivMin;
					ldMaxs_d = (loop_local_index != 0) ? std::max<FLOAT_TYPE>(ldMaxs_d, chunk_derivMax) : chunk_derivMax;
				}
				++loop_local_index;
			}
#ifdef _OPENMP
#pragma omp critical 
			{
#if defined(WIN32)	// Windows
				//			double random_wait = 1000 * 100;
				std::this_thread::sleep_for(std::chrono::microseconds(100 * 1000));	//!< Workaround for test error which occurs in OPENMP mode.
#endif	/* Windows */
			}
#endif
		}
		if (std::any_of(executeAgains.cbegin(), executeAgains.cend(), [](const auto& rhs) { return rhs; }))
			executeAgain = true;
		else
			executeAgain = false;
		for (int dimension = 0; dimension < (int)num_of_dimensions; ++dimension) {
			beacls::FloatVec& dMins_d = derivMins[dimension];
			beacls::FloatVec& dMaxs_d = derivMaxs[dimension];
			if (dMins_d.size() != 1) dMins_d.resize(1);
			if (dMaxs_d.size() != 1) dMaxs_d.resize(1);
			for (int parallel_line_index = 0; parallel_line_index < (int)num_of_parallel_loop_lines; ++parallel_line_index) {
				FLOAT_TYPE& ldMins_d = local_derivMinss[parallel_line_index][dimension];
				FLOAT_TYPE& ldMaxs_d = local_derivMaxss[parallel_line_index][dimension];
				dMins_d[0] = (parallel_line_index != 0) ? std::min<FLOAT_TYPE>(dMins_d[0], ldMins_d) : ldMins_d;
				dMaxs_d[0] = (parallel_line_index != 0) ? std::max<FLOAT_TYPE>(dMaxs_d[0], ldMaxs_d) : ldMaxs_d;
			}
		}

	} while (executeAgain);

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
			<< ", First Diff " << std::setprecision(16) << first_diff << std::resetiosflags(std::ios_base::floatfield) << "@(" << first_diff_line << "," << first_diff_index
			<< "), Max Diff " << std::setprecision(16) << max_diff << std::resetiosflags(std::ios_base::floatfield) << "@(" << max_diff_line << "," << max_diff_index
			<< "), Min Diff " << std::setprecision(16) << min_diff << std::resetiosflags(std::ios_base::floatfield) << "@(" << min_diff_line << "," << min_diff_index
			<< ")" << std::endl;
		message.append(ss.str());
	}
	std::for_each(thread_local_schemeDatas.begin(), thread_local_schemeDatas.end(), [](auto& rhs) {
		if (rhs) delete rhs;
		rhs = NULL;
	});
	if (dissipation) delete dissipation;
	if (hJI_Grid) delete hJI_Grid;

	return allSucceed;
}

