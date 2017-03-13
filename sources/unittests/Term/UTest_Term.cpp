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
#include "UTest_Term.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

bool run_UTest_Term(
	std::string &message,
	const std::string& src_data_filename,
	const std::string& expects_filename,
	const beacls::FloatVec &maxs,
	const beacls::FloatVec &mins,
	SchemeData *schemeData,
	const SpatialDerivative_Class spatialDerivative_Class,
	const std::vector<BoundaryCondition_Class> &boundaryCondition_Classes,
	const Dissipation_Class& dissipation_class,
	const Term_Class& term_Class,
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

	HJI_Grid *hJI_Grid = new HJI_Grid(Ns.size());
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


	if (expects_filename.empty()) {
		std::stringstream ss;
		ss << "Error source file name is empty" << std::endl;
		message.append(ss.str());
		return false;
	}
	
	beacls::FloatVec expected_data;
	beacls::MatFStream* expects_filename_fs = beacls::openMatFStream(expects_filename, beacls::MatOpenMode_Read);
	beacls::IntegerVec dummy;
	if (!load_vector(expected_data, std::string("ydot"), dummy, false, expects_filename_fs)) {
		std::stringstream ss;
		ss << "Cannot open expected term file: " << expects_filename.c_str() << std::endl;
		message.append(ss.str());
		delete hJI_Grid;
		beacls::closeMatFStream(expects_filename_fs);
		return false;
	}
	beacls::closeMatFStream(expects_filename_fs);

	Dissipation* dissipation;
	switch (dissipation_class) {
	case Dissipation_Class_ArtificialDissipationGLF:
		dissipation = new ArtificialDissipationGLF();
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

	std::vector<BoundaryCondition*> boundaryConditions(num_of_dimensions);
	std::transform(boundaryCondition_Classes.cbegin(), boundaryCondition_Classes.cend(), boundaryConditions.begin(), ([&message](auto& rhs) {
		BoundaryCondition* boundaryCondition = NULL;
		switch (rhs) {
		case BoundaryCondition_Class_AddGhostExtrapolate:
			boundaryCondition = new AddGhostExtrapolate();
			break;
		case BoundaryCondition_Class_AddGhostPeriodic:
			boundaryCondition = new AddGhostPeriodic();
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
		delete dissipation;
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
		delete dissipation;
		delete hJI_Grid;
		return false;
	}


	SpatialDerivative *spatialDerivative = NULL;
	switch (spatialDerivative_Class) {
	case SpatialDerivative_Class_UpwindFirstFirst:
		spatialDerivative = new UpwindFirstFirst(hJI_Grid, type);
		break;
	case SpatialDerivative_Class_UpwindFirstENO2:
		spatialDerivative = new UpwindFirstENO2(hJI_Grid, type);
		break;
	case SpatialDerivative_Class_UpwindFirstENO3:
		spatialDerivative = new UpwindFirstENO3(hJI_Grid, type);
		break;
	case SpatialDerivative_Class_UpwindFirstWENO5:
		spatialDerivative = new UpwindFirstWENO5(hJI_Grid, type);
		break;
	default:
		std::stringstream ss;
		ss << "Invalid SpatialDerivative_Class: " << spatialDerivative_Class << std::endl;
		message.append(ss.str());

		for_each(boundaryConditions.cbegin(), boundaryConditions.cend(), ([](const auto& rhs) {
			if (rhs) delete rhs;
		}));
		delete dissipation;
		delete hJI_Grid;
		return false;
	}
	schemeData->set_spatialDerivative(spatialDerivative);

	Term *schemeFunc = NULL;
	switch (term_Class) {
	case Term_Class_TermLaxFriedrichs:
		schemeFunc = new TermLaxFriedrichs(schemeData,type);
		break;
	case Term_Class_TermRestrictUpdate:
		{
		SchemeData* innerData = schemeData->clone();
		schemeData->set_innerData(innerData);
		Term *innerTerm = new TermLaxFriedrichs(innerData, type);
		schemeData->set_innerFunc(innerTerm);
		schemeFunc = new TermRestrictUpdate(type);
		schemeData->set_innerFunc(schemeFunc);
	}
		break;
	default:
		std::stringstream ss;
		ss << "Invalid Term_Class: " << term_Class << std::endl;
		message.append(ss.str());

		for_each(boundaryConditions.cbegin(), boundaryConditions.cend(), ([](const auto& rhs) {
			if (rhs) delete rhs;
		}));
		delete dissipation;
		delete spatialDerivative;
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

	const size_t prefered_line_length_of_chunk_for_cuda = (size_t)(std::ceil(std::ceil((FLOAT_TYPE)num_of_lines / actual_num_of_threads) / second_dimension_loop_size)*second_dimension_loop_size);
	const size_t prefered_line_length_of_chunk_for_cpu = (size_t)std::ceil((FLOAT_TYPE)1024 / first_dimension_loop_size);
	const size_t prefered_line_length_of_chunk = (type == beacls::UVecType_Cuda) ? prefered_line_length_of_chunk_for_cuda : prefered_line_length_of_chunk_for_cpu;

	const size_t actual_line_length_of_chunk = (line_length_of_chunk == 0) ? prefered_line_length_of_chunk : line_length_of_chunk;	//!< T.B.D.

	const size_t num_of_chunk_lines = (actual_line_length_of_chunk < second_dimension_loop_size) ? actual_line_length_of_chunk : second_dimension_loop_size;
	const size_t num_of_slices = (actual_line_length_of_chunk < second_dimension_loop_size) ? 1 : (size_t)std::floor((FLOAT_TYPE)actual_line_length_of_chunk / second_dimension_loop_size);

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
	std::vector<Term*> thread_local_terms(num_of_parallel_loop_lines);
	std::vector<SchemeData*> thread_local_schemeDatas(num_of_parallel_loop_lines);
	std::for_each(thread_local_terms.begin(), thread_local_terms.end(), [schemeFunc](auto& rhs) {
		if (!rhs) rhs = schemeFunc->clone();
	});
	std::for_each(thread_local_schemeDatas.begin(), thread_local_schemeDatas.end(), [schemeData](auto& rhs) {
		rhs = schemeData->clone();
	});
	std::deque<bool> executeAgains(num_of_parallel_loop_lines);
	std::vector<beacls::FloatVec> new_step_bound_invss(num_of_parallel_loop_lines);

	std::vector<beacls::FloatVec > step_bound_invss(num_of_outer_lines);
	for_each(step_bound_invss.begin(), step_bound_invss.end(), ([num_of_dimensions](beacls::FloatVec &i) {i.resize(num_of_dimensions, 0.); }));

	beacls::UVecDepth depth = beacls::type_to_depth<FLOAT_TYPE>();
	beacls::UVec results(depth, beacls::UVecType_Vector, expected_data.size());
	beacls::FloatVec* result_vec_ptr = beacls::UVec_<FLOAT_TYPE>(results).vec();
	bool executeAgain;
	for_each(derivMins.begin(), derivMins.end(), ([](auto& rhs) { rhs.clear(); }));
	for_each(derivMaxs.begin(), derivMaxs.end(), ([](auto& rhs) { rhs.clear(); }));
	bool updateDerivMinMax = true;
	do {
		//! Parallel Body
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for (int parallel_line_index = 0; parallel_line_index < (int)num_of_parallel_loop_lines; ++parallel_line_index) {
			if (num_of_activated_gpus > 1) {
				beacls::set_gpu_id(parallel_line_index%num_of_activated_gpus);
			}
			Term* thread_local_term = thread_local_terms[parallel_line_index];
			std::string local_message;
			SchemeData* thread_local_schemeData = thread_local_schemeDatas[parallel_line_index];
			beacls::FloatVec& sb_inv_out = step_bound_invss[parallel_line_index];
			size_t loop_local_index = 0;
			beacls::FloatVec& local_derivMins = local_derivMinss[parallel_line_index];
			beacls::FloatVec& local_derivMaxs = local_derivMaxss[parallel_line_index];
			beacls::FloatVec& new_step_bound_invs = new_step_bound_invss[parallel_line_index];
			new_step_bound_invs.resize(num_of_dimensions*num_of_chunk_lines*num_of_slices);
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
				std::vector<beacls::FloatVec > chunk_derivMins = derivMins;
				std::vector<beacls::FloatVec > chunk_derivMaxs = derivMaxs;
				beacls::FloatVec::iterator ydot_ite = result_vec_ptr->begin() + expected_result_offset;
				if (thread_local_term->execute(ydot_ite, new_step_bound_invs, t, data, chunk_derivMins, chunk_derivMaxs, thread_local_schemeData, line_begin, actual_chunk_size, actual_num_of_slices, enable_user_defined_dynamics_on_gpu, updateDerivMinMax)) {
					// Take the second substep(y2[i] = y1[i] + deltaT * ydot[i]).
					std::transform(sb_inv_out.cbegin(), sb_inv_out.cend(), new_step_bound_invs.cbegin(), sb_inv_out.begin(), std::ptr_fun<const FLOAT_TYPE&, const FLOAT_TYPE&>(std::max<FLOAT_TYPE>));
					executeAgains[parallel_line_index] = false;
				}
				else {
					executeAgains[parallel_line_index] = true;
				}
				for (int dimension = 0; dimension < (int)num_of_dimensions; ++dimension) {
					beacls::FloatVec& cdMins_d = chunk_derivMins[dimension];
					beacls::FloatVec& cdMaxs_d = chunk_derivMaxs[dimension];
					const FLOAT_TYPE chunk_derivMin = beacls::min_value<FLOAT_TYPE>(cdMins_d.cbegin(), cdMins_d.cend());
					const FLOAT_TYPE chunk_derivMax = beacls::max_value<FLOAT_TYPE>(cdMaxs_d.cbegin(), cdMaxs_d.cend());
					FLOAT_TYPE& ldMins_d = local_derivMins[dimension];
					FLOAT_TYPE& ldMaxs_d = local_derivMaxs[dimension];
					ldMins_d = (inner_line_index != 0) ? std::min<FLOAT_TYPE>(ldMins_d, chunk_derivMin) : chunk_derivMin;
					ldMaxs_d = (inner_line_index != 0) ? std::max<FLOAT_TYPE>(ldMaxs_d, chunk_derivMax) : chunk_derivMax;
				}
				++loop_local_index;
			}
			thread_local_term->synchronize(thread_local_schemeData);
#ifdef _OPENMP
#pragma omp critical 
			{
#if defined(WIN32)	// Windows
				//			double random_wait = 1000 * 100;
				std::this_thread::sleep_for(std::chrono::microseconds(100 * 1000));	//!< Workaround for test error which occurs in OPENMP mode.
#endif	/* Windows */
			}
#endif
			updateDerivMinMax = false;
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
	std::for_each(thread_local_terms.begin(), thread_local_terms.end(), [](auto& rhs) {
		if (rhs) delete rhs;
		rhs = NULL;
	});
	std::for_each(thread_local_schemeDatas.begin(), thread_local_schemeDatas.end(), [](auto& rhs) {
		if (rhs) delete rhs;
		rhs = NULL;
	});
	if (spatialDerivative) delete spatialDerivative;
	for_each(boundaryConditions.cbegin(), boundaryConditions.cend(), ([](const auto& rhs) {
		if (rhs) delete rhs;
	}));
	if (schemeData->get_innerData()) {
		delete schemeData->get_innerData();
		schemeData->set_innerData(NULL);
	}
	if (schemeData->get_innerFunc()) {
		delete schemeData->get_innerFunc();
		schemeData->set_innerFunc(NULL);
	}
	if (dissipation) delete dissipation;
	if (schemeFunc) delete schemeFunc;
	if (hJI_Grid) delete hJI_Grid;

	return allSucceed;
}

