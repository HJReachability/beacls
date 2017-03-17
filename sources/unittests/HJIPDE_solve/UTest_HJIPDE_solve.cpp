#define _USE_MATH_DEFINES
#include <levelset/levelset.hpp>
#include <helperOC/helperOC.hpp>
#include <helperOC/DynSys/DynSys/DynSysSchemeData.hpp>
#include <helperOC/DynSys/DubinsCar/DubinsCar.hpp>
#include <helperOC/DynSys/DubinsCarCAvoid/DubinsCarCAvoid.hpp>
#include <helperOC/DynSys/Air3D/Air3D.hpp>
#include <sstream>
#include <iomanip>

#include "UTest_HJIPDE_solve.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

bool check_and_make_message(
	std::string& message,
	const std::vector<beacls::FloatVec>& expected_datas,
	const std::vector<beacls::FloatVec>& result_datas,
	const FLOAT_TYPE small
) {
	FLOAT_TYPE min_diff = std::numeric_limits<FLOAT_TYPE>::max();
	FLOAT_TYPE max_diff = 0;
	FLOAT_TYPE first_diff = 0;
	FLOAT_TYPE sum_of_square = 0;
	size_t min_diff_t = 0;
	size_t max_diff_t = 0;
	size_t first_diff_t = 0;
	size_t min_diff_index = 0;
	size_t max_diff_index = 0;
	size_t first_diff_index = 0;
	size_t num_of_diffs = 0;
	size_t num_of_datas = 0;
	bool allSucceed = true;
	for (size_t t = 0; t < result_datas.size(); ++t) {
		const beacls::FloatVec& expected_data = expected_datas[t];
		const beacls::FloatVec& result_data = result_datas[t];
		for (size_t index = 0; index < result_data.size(); ++index) {
			FLOAT_TYPE expected_result = expected_data[index];
			FLOAT_TYPE result = result_data[index];
			++num_of_datas;
			const FLOAT_TYPE diff = std::abs(expected_result - result);
			if (diff > small) {
				allSucceed = false;
				if (min_diff > diff) {
					min_diff = diff;
					min_diff_t = t;
					min_diff_index = index;
				}
				if (max_diff < diff) {
					max_diff = diff;
					max_diff_t = t;
					max_diff_index = index;
				}
				if (first_diff == 0) {
					first_diff = diff;
					first_diff_t = t;
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
			<< ", First Diff " << std::setprecision(16) << first_diff << std::resetiosflags(std::ios_base::floatfield) << "@(" << first_diff_t << "," << first_diff_index
			<< "), Max Diff " << std::setprecision(16) << max_diff << std::resetiosflags(std::ios_base::floatfield) << "@(" << max_diff_t << "," << max_diff_index
			<< "), Min Diff " << std::setprecision(16) << min_diff << std::resetiosflags(std::ios_base::floatfield) << "@(" << min_diff_t << "," << min_diff_index
			<< ")" << std::endl;
		message.append(ss.str());
	}
	return allSucceed;
}

bool run_UTest_HJIPDE_solve_minWith(
	std::string &message,
	const std::vector<std::vector<beacls::FloatVec > >& expected_datas,
	helperOC::DynSysSchemeData* schemeData,
	beacls::FloatVec& tau,
	beacls::FloatVec& data0,
	const FLOAT_TYPE small,
	const helperOC::ExecParameters& execParameters
	) {
	/*!< selecting 'zero' computes reachable tube(usually, choose this option)
		selecting 'none' computes reachable set
		selecting 'data0' computes reachable tube, but only use this if there are
		obstacles(constraint / avoid sets) in the state space
		*/
	std::vector<helperOC::HJIPDE::MinWithType> minWiths{helperOC::HJIPDE::MinWithType_None, helperOC::HJIPDE::MinWithType_Zero};
	bool result = true;
	for (size_t i = 0; i < minWiths.size(); ++i) {
		helperOC::HJIPDE_extraArgs extraArgs;
		helperOC::HJIPDE_extraOuts extraOuts;

		extraArgs.keepLast = false;
		extraArgs.execParameters = execParameters;
		helperOC::HJIPDE* hjipde = new helperOC::HJIPDE();

		beacls::FloatVec stoptau;
		std::vector<beacls::FloatVec > datas;
		const std::vector<beacls::FloatVec >& expected_datas_i = expected_datas[i];
		hjipde->solve(stoptau, extraOuts, data0, tau, schemeData, minWiths[i], extraArgs);
		hjipde->get_datas(datas, tau, schemeData);

		if (datas.empty()) {
			std::stringstream ss;
			ss << "Error result is empty: " << i << std::endl;
			message.append(ss.str());
			return false;
		}
		if (datas.size() != expected_datas_i.size()) {
			std::stringstream ss;
			ss << "Error time length of results is different: " << i << " : " << datas.size() <<  "!= " << expected_datas_i.size() << std::endl;
			message.append(ss.str());
			return false;
		}
		result &= check_and_make_message(
			message,
			expected_datas_i,
			datas,
			small
		);
		if(hjipde) delete hjipde;
	}

	return result;
}

/*
	@brief Test using time-varying targets
*/
bool run_UTest_HJIPDE_solve_tvTarget(
	std::string &message,
	const std::vector<std::vector<beacls::FloatVec > >& expected_datas,
	helperOC::DynSysSchemeData* schemeData,
	beacls::FloatVec& tau,
	beacls::FloatVec& data0,
	const FLOAT_TYPE radius,
	const FLOAT_TYPE small,
	const helperOC::ExecParameters& execParameters
) {
	const levelset::HJI_Grid *g = schemeData->get_grid();
	std::vector<beacls::FloatVec> targets(tau.size());
	beacls::FloatVec center{ 1.5,1.5,0. };
	beacls::IntegerVec pdDims{ 2 };	//!< 2nd diemension is periodic
	for (size_t i = 0; i < targets.size(); ++i) {
		levelset::BasicShape* shape = new levelset::ShapeCylinder(pdDims, center, ((FLOAT_TYPE)i+1)/tau.size()*radius);
		shape->execute(g, targets[i]);
		delete shape;
	}

	helperOC::HJIPDE::MinWithType minWith = helperOC::HJIPDE::MinWithType_None;
	bool result = true;
	helperOC::HJIPDE_extraArgs extraArgs;
	helperOC::HJIPDE_extraOuts extraOuts;
	extraArgs.targets = targets;

	extraArgs.keepLast = false;
	extraArgs.execParameters = execParameters;
	helperOC::HJIPDE* hjipde = new helperOC::HJIPDE();

	beacls::FloatVec stoptau;
	std::vector<beacls::FloatVec > datas;
	const std::vector<beacls::FloatVec >& expected_datas_0 = expected_datas[0];
	hjipde->solve(stoptau, extraOuts, data0, tau, schemeData, minWith, extraArgs);
	hjipde->get_datas(datas, tau, schemeData);

	if (datas.empty()) {
		std::stringstream ss;
		ss << "Error result is empty" << std::endl;
		message.append(ss.str());
		return false;
	}
	if (datas.size() != expected_datas_0.size()) {
		std::stringstream ss;
		ss << "Error time length of results is different: " << datas.size() << "!= " << expected_datas_0.size() << std::endl;
		message.append(ss.str());
		return false;
	}
	result &= check_and_make_message(
		message,
		expected_datas_0,
		datas,
		small
	);
	if (hjipde) delete hjipde;

	return result;
}
/*
@brief Test using single obstacle
*/
bool run_UTest_HJIPDE_solve_singleObs(
	std::string &message,
	const std::vector<std::vector<beacls::FloatVec > >& expected_datas,
	helperOC::DynSysSchemeData* schemeData,
	beacls::FloatVec& tau,
	beacls::FloatVec& data0,
	const FLOAT_TYPE radius,
	const FLOAT_TYPE small,
	const helperOC::ExecParameters& execParameters
) {
	const levelset::HJI_Grid *g = schemeData->get_grid();
	std::vector<const beacls::FloatVec*> obstacles(1);
	beacls::FloatVec center{ 1.5,1.5,0. };
	beacls::IntegerVec pdDims{ 2 };	//!< 2nd diemension is periodic

	std::vector<beacls::FloatVec> targets(1);
	targets[0] = data0;

	levelset::BasicShape* shape = new levelset::ShapeCylinder(pdDims, center, (FLOAT_TYPE)(0.75*radius));
	beacls::FloatVec obstacle;
	shape->execute(g, obstacle);
	obstacles[0] = &obstacle;
	delete shape;

	helperOC::HJIPDE::MinWithType minWith = helperOC::HJIPDE::MinWithType_None;
	bool result = true;
	helperOC::HJIPDE_extraArgs extraArgs;
	helperOC::HJIPDE_extraOuts extraOuts;
	extraArgs.targets = targets;
	extraArgs.obstacles = obstacles;

	extraArgs.keepLast = false;
	extraArgs.execParameters = execParameters;
	helperOC::HJIPDE* hjipde = new helperOC::HJIPDE();

	beacls::FloatVec stoptau;
	std::vector<beacls::FloatVec > datas;
	const std::vector<beacls::FloatVec >& expected_datas_0 = expected_datas[0];
	hjipde->solve(stoptau, extraOuts, data0, tau, schemeData, minWith, extraArgs);
	hjipde->get_datas(datas, tau, schemeData);

	if (datas.empty()) {
		std::stringstream ss;
		ss << "Error result is empty" << std::endl;
		message.append(ss.str());
		return false;
	}
	if (datas.size() != expected_datas_0.size()) {
		std::stringstream ss;
		ss << "Error time length of results is different: " << datas.size() << "!= " << expected_datas_0.size() << std::endl;
		message.append(ss.str());
		return false;
	}
	result &= check_and_make_message(
		message,
		expected_datas_0,
		datas,
		small
	);

	if (hjipde) delete hjipde;

	return result;
}
/*
@brief Test using time-varying obstacle
*/
bool run_UTest_HJIPDE_solve_tvObs(
	std::string &message,
	const std::vector<std::vector<beacls::FloatVec > >& expected_datas,
	helperOC::DynSysSchemeData* schemeData,
	beacls::FloatVec& tau,
	beacls::FloatVec& data0,
	const FLOAT_TYPE radius,
	const FLOAT_TYPE small,
	const helperOC::ExecParameters& execParameters
) {
	const levelset::HJI_Grid *g = schemeData->get_grid();
	std::vector<const beacls::FloatVec*> obstacles(tau.size());
	std::vector<beacls::FloatVec> obstacles_vec(tau.size());
	beacls::FloatVec center{ 1.5,1.5,0. };
	beacls::IntegerVec pdDims{ 2 };	//!< 2nd diemension is periodic

	std::vector<beacls::FloatVec> targets(1);
	targets[0] = data0;

	for (size_t i = 0; i < obstacles.size(); ++i) {
		levelset::BasicShape* shape = new levelset::ShapeCylinder(pdDims, center, ((FLOAT_TYPE)i+1) / obstacles.size()*radius);
		shape->execute(g, obstacles_vec[i]);
		obstacles[i] = &obstacles_vec[i];
		delete shape;
	}

	helperOC::HJIPDE::MinWithType minWith = helperOC::HJIPDE::MinWithType_None;
	bool result = true;
	helperOC::HJIPDE_extraArgs extraArgs;
	helperOC::HJIPDE_extraOuts extraOuts;
	extraArgs.targets = targets;
	extraArgs.obstacles = obstacles;

	extraArgs.keepLast = false;
	extraArgs.execParameters = execParameters;
	helperOC::HJIPDE* hjipde = new helperOC::HJIPDE();

	beacls::FloatVec stoptau;
	std::vector<beacls::FloatVec > datas;
	const std::vector<beacls::FloatVec >& expected_datas_0 = expected_datas[0];
	hjipde->solve(stoptau, extraOuts, data0, tau, schemeData, minWith, extraArgs);
	hjipde->get_datas(datas, tau, schemeData);

	if (datas.empty()) {
		std::stringstream ss;
		ss << "Error result is empty" << std::endl;
		message.append(ss.str());
		return false;
	}
	if (datas.size() != expected_datas_0.size()) {
		std::stringstream ss;
		ss << "Error time length of results is different: " << datas.size() << "!= " << expected_datas_0.size() << std::endl;
		message.append(ss.str());
		return false;
	}
	result &= check_and_make_message(
		message,
		expected_datas_0,
		datas,
		small
	);

	if (hjipde) delete hjipde;

	return result;
}
/*
@brief Test using single obstacle but few time steps
*/
bool run_UTest_HJIPDE_solve_obs_stau(
	std::string &message,
	const std::vector<std::vector<beacls::FloatVec > >& expected_datas,
	helperOC::DynSysSchemeData* schemeData,
	beacls::FloatVec& data0,
	const FLOAT_TYPE radius,
	const FLOAT_TYPE small,
	const helperOC::ExecParameters& execParameters
) {
	const levelset::HJI_Grid *g = schemeData->get_grid();
	std::vector<const beacls::FloatVec*> obstacles(1);
	beacls::FloatVec center{ 1.5,1.5,0. };
	beacls::IntegerVec pdDims{ 2 };	//!< 2nd diemension is periodic

	std::vector<beacls::FloatVec> targets(1);
	targets[0] = data0;
	beacls::FloatVec obstacle;
	levelset::BasicShape* shape = new levelset::ShapeCylinder(pdDims, center, (FLOAT_TYPE)(0.75*radius));
	shape->execute(g, obstacle);
	obstacles[0] = &obstacle;
	delete shape;

	FLOAT_TYPE local_tau_bottom = 0.;
	FLOAT_TYPE local_tau_top = 2.;
	size_t local_tau_num = 5;
	beacls::FloatVec local_tau(local_tau_num);
	for (size_t i = 0; i < local_tau_num; ++i) {
		local_tau[i] = local_tau_bottom + i * (local_tau_top - local_tau_bottom) / (local_tau_num - 1);
	}

	helperOC::HJIPDE::MinWithType minWith = helperOC::HJIPDE::MinWithType_None;
	bool result = true;
	helperOC::HJIPDE_extraArgs extraArgs;
	helperOC::HJIPDE_extraOuts extraOuts;
	extraArgs.targets = targets;
	extraArgs.obstacles = obstacles;

	extraArgs.keepLast = false;
	extraArgs.execParameters = execParameters;
	helperOC::HJIPDE* hjipde = new helperOC::HJIPDE();

	beacls::FloatVec stoptau;
	std::vector<beacls::FloatVec > datas;
	const std::vector<beacls::FloatVec >& expected_datas_0 = expected_datas[0];
	hjipde->solve(stoptau, extraOuts, data0, local_tau, schemeData, minWith, extraArgs);
	hjipde->get_datas(datas, local_tau, schemeData);

	if (datas.empty()) {
		std::stringstream ss;
		ss << "Error result is empty" << std::endl;
		message.append(ss.str());
		return false;
	}
	if (datas.size() != expected_datas_0.size()) {
		std::stringstream ss;
		ss << "Error time length of results is different: " << datas.size() << "!= " << expected_datas_0.size() << std::endl;
		message.append(ss.str());
		return false;
	}
	result &= check_and_make_message(
		message,
		expected_datas_0,
		datas,
		small
	);

	if (hjipde) delete hjipde;

	return result;
}
/*
@brief Test the inclusion of initial state
*/
bool run_UTest_HJIPDE_solve_stopInit(
	std::string &message,
	const std::vector<std::vector<beacls::FloatVec > >& expected_datas,
	helperOC::DynSysSchemeData* schemeData,
	beacls::FloatVec& data0,
	const FLOAT_TYPE small,
	const helperOC::ExecParameters& execParameters
) {
	FLOAT_TYPE local_tau_bottom = 0.;
	FLOAT_TYPE local_tau_top = 2.;
	size_t local_tau_num = 50;
	beacls::FloatVec local_tau(local_tau_num);
	for (size_t i = 0; i < local_tau_num; ++i) {
		local_tau[i] = local_tau_bottom + i * (local_tau_top - local_tau_bottom) / (local_tau_num - 1);
	}

	helperOC::HJIPDE::MinWithType minWith = helperOC::HJIPDE::MinWithType_None;
	bool result = true;
	helperOC::HJIPDE_extraArgs extraArgs;
	helperOC::HJIPDE_extraOuts extraOuts;
	extraArgs.stopInit = beacls::FloatVec{ (FLOAT_TYPE)-1.1, (FLOAT_TYPE)-1.1,(FLOAT_TYPE)0 };

	extraArgs.keepLast = false;
	extraArgs.execParameters = execParameters;
	helperOC::HJIPDE* hjipde = new helperOC::HJIPDE();

	beacls::FloatVec stoptau;
	std::vector<beacls::FloatVec > datas;
	const std::vector<beacls::FloatVec >& expected_datas_0 = expected_datas[0];
	hjipde->solve(stoptau, extraOuts, data0, local_tau, schemeData, minWith, extraArgs);
	hjipde->get_datas(datas, local_tau, schemeData);

	if (datas.empty()) {
		std::stringstream ss;
		ss << "Error result is empty" << std::endl;
		message.append(ss.str());
		return false;
	}
	if (datas.size() != expected_datas_0.size()) {
		std::stringstream ss;
		ss << "Error time length of results is different: " << datas.size() << "!= " << expected_datas_0.size() << std::endl;
		message.append(ss.str());
		return false;
	}
	result &= check_and_make_message(
		message,
		expected_datas_0,
		datas,
		small
	);

	if (hjipde) delete hjipde;

	return result;
}
/*
@brief Test the inclusion of initial state
*/
bool run_UTest_HJIPDE_solve_stopSetInclude(
	std::string &message,
	const std::vector<std::vector<beacls::FloatVec > >& expected_datas,
	helperOC::DynSysSchemeData* schemeData,
	beacls::FloatVec& data0,
	const FLOAT_TYPE small,
	const helperOC::ExecParameters& execParameters
) {
	const levelset::HJI_Grid *g = schemeData->get_grid();

	beacls::FloatVec stopSetInclude;
	levelset::BasicShape* shape = new levelset::ShapeSphere(beacls::FloatVec{(FLOAT_TYPE)-1.1, (FLOAT_TYPE)1.1, (FLOAT_TYPE)0}, (FLOAT_TYPE)0.5);
	shape->execute(g, stopSetInclude);
	delete shape;

	FLOAT_TYPE local_tau_bottom = 0.;
	FLOAT_TYPE local_tau_top = 2.;
	size_t local_tau_num = 5;
	beacls::FloatVec local_tau(local_tau_num);
	for (size_t i = 0; i < local_tau_num; ++i) {
		local_tau[i] = local_tau_bottom + i * (local_tau_top - local_tau_bottom) / (local_tau_num - 1);
	}

	helperOC::HJIPDE::MinWithType minWith = helperOC::HJIPDE::MinWithType_None;
	bool result = true;
	helperOC::HJIPDE_extraArgs extraArgs;
	helperOC::HJIPDE_extraOuts extraOuts;
	extraArgs.stopSetInclude = stopSetInclude;

	extraArgs.keepLast = false;
	extraArgs.execParameters = execParameters;
	helperOC::HJIPDE* hjipde = new helperOC::HJIPDE();

	beacls::FloatVec stoptau;
	std::vector<beacls::FloatVec > datas;
	const std::vector<beacls::FloatVec >& expected_datas_0 = expected_datas[0];
	hjipde->solve(stoptau, extraOuts, data0, local_tau, schemeData, minWith, extraArgs);
	hjipde->get_datas(datas, local_tau, schemeData);

	if (datas.empty()) {
		std::stringstream ss;
		ss << "Error result is empty" << std::endl;
		message.append(ss.str());
		return false;
	}
	if (datas.size() != expected_datas_0.size()) {
		std::stringstream ss;
		ss << "Error time length of results is different: " << datas.size() << "!= " << expected_datas_0.size() << std::endl;
		message.append(ss.str());
		return false;
	}
	result &= check_and_make_message(
		message,
		expected_datas_0,
		datas,
		small
	);

	if (hjipde) delete hjipde;

	return result;
}
/*
@brief Test intersection of some set
*/
bool run_UTest_HJIPDE_solve_stopSetIntersect(
	std::string &message,
	const std::vector<std::vector<beacls::FloatVec > >& expected_datas,
	helperOC::DynSysSchemeData* schemeData,
	beacls::FloatVec& data0,
	const FLOAT_TYPE small,
	const helperOC::ExecParameters& execParameters
) {
	const levelset::HJI_Grid *g = schemeData->get_grid();

	beacls::FloatVec stopSetIntersect;
	levelset::BasicShape* shape = new levelset::ShapeSphere(beacls::FloatVec{-1.25, 1.25, 0}, 0.5);
	shape->execute(g, stopSetIntersect);
	delete shape;

	FLOAT_TYPE local_tau_bottom = 0.;
	FLOAT_TYPE local_tau_top = 1.;
	size_t local_tau_num = 11;
	beacls::FloatVec local_tau(local_tau_num);
	for (size_t i = 0; i < local_tau_num; ++i) {
		local_tau[i] = local_tau_bottom + i * (local_tau_top - local_tau_bottom) / (local_tau_num - 1);
	}

	helperOC::HJIPDE::MinWithType minWith = helperOC::HJIPDE::MinWithType_None;
	bool result = true;
	helperOC::HJIPDE_extraArgs extraArgs;
	helperOC::HJIPDE_extraOuts extraOuts;
	extraArgs.stopSetIntersect = stopSetIntersect;

	extraArgs.keepLast = false;
	extraArgs.execParameters = execParameters;
	helperOC::HJIPDE* hjipde = new helperOC::HJIPDE();

	beacls::FloatVec stoptau;
	std::vector<beacls::FloatVec > datas;
	const std::vector<beacls::FloatVec >& expected_datas_0 = expected_datas[0];
	hjipde->solve(stoptau, extraOuts, data0, local_tau, schemeData, minWith, extraArgs);
	hjipde->get_datas(datas, local_tau, schemeData);

	if (datas.empty()) {
		std::stringstream ss;
		ss << "Error result is empty" << std::endl;
		message.append(ss.str());
		return false;
	}
	if (datas.size() != expected_datas_0.size()) {
		std::stringstream ss;
		ss << "Error time length of results is different: " << datas.size() << "!= " << expected_datas_0.size() << std::endl;
		message.append(ss.str());
		return false;
	}
	result &= check_and_make_message(
		message,
		expected_datas_0,
		datas,
		small
	);

	if (hjipde) delete hjipde;

	return result;
}
/*
@brief Test the intermediate plotting
*/
bool run_UTest_HJIPDE_solve_plotData(
	std::string &message,
	const std::vector<std::vector<beacls::FloatVec > >& expected_datas,
	helperOC::DynSysSchemeData* schemeData,
	beacls::FloatVec& data0,
	const FLOAT_TYPE radius,
	const FLOAT_TYPE small,
	const helperOC::ExecParameters& execParameters
) {
	const levelset::HJI_Grid *g = schemeData->get_grid();

	FLOAT_TYPE local_tau_bottom = 0.;
	FLOAT_TYPE local_tau_top = 2.;
	size_t local_tau_num = 51;
	beacls::FloatVec local_tau(local_tau_num);
	for (size_t i = 0; i < local_tau_num; ++i) {
		local_tau[i] = local_tau_bottom + i * (local_tau_top - local_tau_bottom) / (local_tau_num - 1);
	}

	std::vector<const beacls::FloatVec*> obstacles(local_tau.size());
	std::vector<beacls::FloatVec> obstacles_vec(local_tau.size());
	beacls::FloatVec center{ 1.5,1.5,0. };
	beacls::IntegerVec pdDims{ 2 };	//!< 2nd diemension is periodic

	for (size_t i = 0; i < obstacles.size(); ++i) {
		levelset::BasicShape* shape = new levelset::ShapeCylinder(pdDims, center, ((FLOAT_TYPE)i + 1) / obstacles.size()*radius);
		shape->execute(g, obstacles_vec[i]);
		obstacles[i] = &obstacles_vec[i];
		delete shape;
	}


	helperOC::HJIPDE::MinWithType minWith = helperOC::HJIPDE::MinWithType_None;
	bool result = true;
	helperOC::HJIPDE_extraArgs extraArgs;
	helperOC::HJIPDE_extraOuts extraOuts;
	extraArgs.obstacles = obstacles;

	extraArgs.keepLast = false;
	extraArgs.execParameters = execParameters;
	helperOC::HJIPDE* hjipde = new helperOC::HJIPDE();

	beacls::FloatVec stoptau;
	std::vector<beacls::FloatVec > datas;
	const std::vector<beacls::FloatVec >& expected_datas_0 = expected_datas[0];
	hjipde->solve(stoptau, extraOuts, data0, local_tau, schemeData, minWith, extraArgs);
	hjipde->get_datas(datas, local_tau, schemeData);

	if (datas.empty()) {
		std::stringstream ss;
		ss << "Error result is empty" << std::endl;
		message.append(ss.str());
		return false;
	}
	if (datas.size() != expected_datas_0.size()) {
		std::stringstream ss;
		ss << "Error time length of results is different: " << datas.size() << "!= " << expected_datas_0.size() << std::endl;
		message.append(ss.str());
		return false;
	}
	result &= check_and_make_message(
		message,
		expected_datas_0,
		datas,
		small
	);

	if (hjipde) delete hjipde;

	return result;
}
/*
@brief Test starting from saved data (where data0 has dimension g.dim + 1)
*/
bool run_UTest_HJIPDE_solve_savedData(
	std::string &message,
	const std::vector<std::vector<beacls::FloatVec > >& expected_datas,
	helperOC::DynSysSchemeData* schemeData,
	beacls::FloatVec& tau,
	beacls::FloatVec& data0,
	const FLOAT_TYPE small,
	const helperOC::ExecParameters& execParameters
) {
	helperOC::HJIPDE::MinWithType minWith = helperOC::HJIPDE::MinWithType_Zero;
	bool result = true;
	helperOC::HJIPDE_extraArgs extraArgs;
	helperOC::HJIPDE_extraOuts extraOuts;

	extraArgs.keepLast = false;
	extraArgs.execParameters = execParameters;
	helperOC::HJIPDE* hjipde = new helperOC::HJIPDE();

	beacls::FloatVec stoptau;
	std::vector<beacls::FloatVec > datas1;
	const std::vector<beacls::FloatVec >& expected_datas_0 = expected_datas[0];
	hjipde->solve(stoptau, extraOuts, data0, tau, schemeData, minWith, extraArgs);
	hjipde->get_datas(datas1, tau, schemeData);

	if (datas1.empty()) {
		std::stringstream ss;
		ss << "Error result1 is empty" << std::endl;
		message.append(ss.str());
		return false;
	}
	if (datas1.size() != expected_datas_0.size()) {
		std::stringstream ss;
		ss << "Error time length of results is different: " << datas1.size() << "!= " << expected_datas_0.size() << std::endl;
		message.append(ss.str());
		return false;
	}

	//!< Cut off data 1
	FLOAT_TYPE tcutoff = 0.5;

	size_t istart = 1;
	for (size_t i = 0; i < tau.size(); ++i) {
		if (tau[i] > tcutoff) {
			istart = i+1;
			break;
		}
	}
	extraArgs.istart = istart;
	std::vector<beacls::FloatVec > dataSaved(tau.size());
	std::copy(datas1.cbegin(), datas1.cbegin() + istart, dataSaved.begin());

	std::vector<beacls::FloatVec > datas2;
	hjipde->solve(stoptau, extraOuts, dataSaved, tau, schemeData, minWith, extraArgs);
	hjipde->get_datas(datas2, tau, schemeData);

	if (datas2.empty()) {
		std::stringstream ss;
		ss << "Error result2 is empty" << std::endl;
		message.append(ss.str());
		return false;
	}

	if (datas2.size() != expected_datas_0.size()) {
		std::stringstream ss;
		ss << "Error time length of results is different: " << datas2.size() << "!= " << expected_datas_0.size() << std::endl;
		message.append(ss.str());
		return false;
	}
	std::string tmp_message;
	result &= check_and_make_message(
		tmp_message,
		expected_datas_0,
		datas1,
		small
	);
	if (!tmp_message.empty()) {
		message = std::string("Data1: ") + tmp_message;
		tmp_message.clear();
	}
	result &= check_and_make_message(
		tmp_message,
		expected_datas_0,
		datas2,
		small
	);
	if (!tmp_message.empty()) {
		message = std::string("Data2: ") + tmp_message;
		tmp_message.clear();
	}
	result &= check_and_make_message(
		tmp_message,
		datas1,
		datas2,
		small
	);
	if (!tmp_message.empty()) {
		message = std::string("Data1-Data2: ") + tmp_message;
		tmp_message.clear();
	}
	if (hjipde) delete hjipde;

	return result;
}
/*
@brief Test the intermediate plotting
*/
bool run_UTest_HJIPDE_solve_stopConverge(
	std::string &message,
	const std::vector<std::vector<beacls::FloatVec > >& expected_datas,
	const bool isMiddleModel,
	const bool isDubinsCarCAvoidModel,
	const FLOAT_TYPE small,
	const helperOC::ExecParameters& execParameters
) {

	// Grid
    beacls::IntegerVec	Ns;
    beacls::FloatVec grid_min;
    beacls::FloatVec grid_max;
    FLOAT_TYPE captureRadius;
    if(!isMiddleModel){
        Ns = beacls::IntegerVec{ 41, 41, 41 };	//!< Number of grid points per dimension
        grid_min = beacls::FloatVec{ -5, -5, (FLOAT_TYPE)-M_PI };	//!< Lower corner of computation domain
        grid_max = beacls::FloatVec{ 5, 5, (FLOAT_TYPE)M_PI };	//!< Upper corner of computation domain
        captureRadius = 1;
    }else{
        Ns = beacls::IntegerVec{ 61, 61, 61 };	//!< Number of grid points per dimension
        grid_min = beacls::FloatVec{ -25, -20, 0 };	//!< Lower corner of computation domain
        grid_max = beacls::FloatVec{ 25, 20, (FLOAT_TYPE)(2*M_PI) };	//!< Upper corner of computation domain
        captureRadius = 5;
    }
        beacls::IntegerVec pdDims{ 2 };	//!< 2nd diemension is periodic

	FLOAT_TYPE va = 5;
	FLOAT_TYPE vb = 5;
	FLOAT_TYPE uMax = 1;
	FLOAT_TYPE dMax = 1;
    //!< problem parameters
    FLOAT_TYPE speed = 1;
    FLOAT_TYPE wMax = 1;



	levelset::HJI_Grid* g = helperOC::createGrid(grid_min, grid_max, Ns, pdDims);
	levelset::BasicShape* shape;
	beacls::FloatVec center{ 0.,0.,0. };	//!< Center coordinate
	shape = new levelset::ShapeCylinder(pdDims, center, captureRadius);
	beacls::FloatVec data0;
	shape->execute(g, data0);
	if (shape) delete shape;

    helperOC::DynSys* dynSys;
    if(isDubinsCarCAvoidModel)
        dynSys = new helperOC::DubinsCarCAvoid(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, uMax, dMax, va, vb);
    else
        dynSys = new helperOC::DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, wMax, speed);
    
    //!< time vector
	FLOAT_TYPE t0 = 0;
	FLOAT_TYPE tMax = 5;
	FLOAT_TYPE dt = (FLOAT_TYPE)0.01;
	beacls::FloatVec tau = generateArithmeticSequence<FLOAT_TYPE>(t0, dt, tMax);

	helperOC::DynSysSchemeData* schemeData = new helperOC::DynSysSchemeData;
	schemeData->set_grid(g);	//!< Grid MUST be specified!
								//!<Dynamical system parameters
	schemeData->dynSys = dynSys;
	schemeData->uMode = helperOC::DynSys_UMode_Max;
	schemeData->dMode = helperOC::DynSys_DMode_Min;

	helperOC::HJIPDE::MinWithType minWith = helperOC::HJIPDE::MinWithType_Zero;
	bool result = true;
	helperOC::HJIPDE_extraArgs extraArgs;
	helperOC::HJIPDE_extraOuts extraOuts;

	extraArgs.keepLast = false;
	extraArgs.execParameters = execParameters;

	extraArgs.stopConverge = true;
	extraArgs.convergeThreshold = (FLOAT_TYPE)1e-3;
	helperOC::HJIPDE* hjipde = new helperOC::HJIPDE();

	beacls::FloatVec stoptau;
	std::vector<beacls::FloatVec > datas;
	const std::vector<beacls::FloatVec >& expected_datas_0 = expected_datas[0];

	hjipde->solve(stoptau, extraOuts, data0, tau, schemeData, minWith, extraArgs);
	hjipde->get_datas(datas, tau, schemeData);

	if (datas.empty()) {
		std::stringstream ss;
		ss << "Error result is empty" << std::endl;
		message.append(ss.str());
		return false;
	}
	if (datas.size() != expected_datas_0.size()) {
		std::stringstream ss;
		ss << "Error time length of results is different: " << datas.size() << "!= " << expected_datas_0.size() << std::endl;
		message.append(ss.str());
		return false;
	}
	result &= check_and_make_message(
		message,
		expected_datas_0,
		datas,
		small
	);
	if (hjipde) delete hjipde;
	if (dynSys) delete dynSys;

	if (schemeData) delete schemeData;


	return result;
}



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
) {
	helperOC::ExecParameters execParameters;
	execParameters.useCuda = (type == beacls::UVecType_Cuda) ? true : false;
	execParameters.line_length_of_chunk = chunk_size;
	execParameters.num_of_threads = num_of_threads;
	execParameters.num_of_gpus = num_of_gpus;
	execParameters.delayedDerivMinMax = delayedDerivMinMax;
	execParameters.enable_user_defined_dynamics_on_gpu = enable_user_defined_dynamics_on_gpu;

	const size_t line_length_of_chunk = chunk_size;
	const FLOAT_TYPE small = small_diff;
	std::vector<std::vector<beacls::FloatVec > > expected_datas(expects_filenames.size());
	std::transform(expects_filenames.cbegin(), expects_filenames.cend(), expected_datas.begin(), ([&message](const auto& rhs) {
		std::vector<beacls::FloatVec > datas;
		beacls::FloatVec data;
		beacls::MatFStream* rhs_fs = beacls::openMatFStream(rhs, beacls::MatOpenMode_Read);
		beacls::IntegerVec read_Ns;
		if (!load_vector(data, std::string("data"), read_Ns, false, rhs_fs)) {
			std::stringstream ss;
			ss << "Cannot open expected result file: " << rhs.c_str() << std::endl;
			message.append(ss.str());
			beacls::closeMatFStream(rhs_fs);
			return std::vector<beacls::FloatVec >();
		}
		datas.resize(read_Ns[read_Ns.size() - 1]);
		size_t num_of_elements = std::accumulate(read_Ns.cbegin(), read_Ns.cbegin() + read_Ns.size() - 1, (size_t)1, [](const auto& lhs, const auto& rhs) { return lhs * rhs; });
		for (size_t t = 0; t < datas.size(); ++t){
			datas[t].resize(num_of_elements);
			std::copy(data.cbegin() + t*num_of_elements, data.cbegin() + (t + 1)*num_of_elements, datas[t].begin());
		}
		beacls::closeMatFStream(rhs_fs);
		return datas;
	}));
	if (std::any_of(expected_datas.cbegin(), expected_datas.cend(), [](const auto& rhs) { return rhs.empty();  })) {
		return false;
	}

	// Grid
	beacls::FloatVec grid_min{ -5, -5, (FLOAT_TYPE)-M_PI };	//!< Lower corner of computation domain
	beacls::FloatVec grid_max{ 5, 5, (FLOAT_TYPE)M_PI };	//!< Upper corner of computation domain
	beacls::IntegerVec	Ns = beacls::IntegerVec{ 41, 41, 41 };	//!< Number of grid points per dimension
	beacls::IntegerVec pdDims{ 2 };	//!< 2nd diemension is periodic

	levelset::HJI_Grid* g = helperOC::createGrid(grid_min, grid_max, Ns, pdDims);
	// state space dimensions

	// target set
	levelset::BasicShape* shape;
	FLOAT_TYPE radius = 1;
	beacls::FloatVec center{ 0.,0.,0.};	//!< Center coordinate
	switch (shapeType) {
	default:
	case HJIPDE_solve_Shape_Invalid:
		{
			std::stringstream ss;
			ss << "Error Invalid Shape type: " << shapeType << std::endl;
			message.append(ss.str());
		}
		return false;
	case HJIPDE_solve_Shape_Cylinder:
		shape = new levelset::ShapeCylinder(pdDims, center, radius);
		break;
	case HJIPDE_solve_Shape_Sphere:
		shape = new levelset::ShapeSphere(center, radius);
		break;
	case HJIPDE_solve_Shape_RectangleByCorner:
		shape = new levelset::ShapeRectangleByCorner(beacls::FloatVec{-radius, -radius, -radius}, beacls::FloatVec{radius, radius, radius});
		break;
	case HJIPDE_solve_Shape_RectangleByCenter:
		shape = new levelset::ShapeRectangleByCenter(center, beacls::FloatVec{radius,radius,radius});
		break;
	}
	beacls::FloatVec data0;
	shape->execute(g, data0);

	//!< time vector
	FLOAT_TYPE t0 = 0;
	FLOAT_TYPE tMax = 2;
	FLOAT_TYPE dt = (FLOAT_TYPE)0.025;
	beacls::FloatVec tau = generateArithmeticSequence<FLOAT_TYPE>(t0, dt, tMax);
	// If intermediate results are not needed, use 
	// beacls::FloatVec tau = generateArithmeticSequence(t0, tMax, tMax);

	//!< problem parameters
	FLOAT_TYPE speed = 1;
	FLOAT_TYPE wMax = 1;

	//!< Pack problem parameters
	helperOC::DynSysSchemeData* schemeData = new helperOC::DynSysSchemeData;
	schemeData->set_grid(g);	//!< Grid MUST be specified!
	//!<Dynamical system parameters
	helperOC::DubinsCar* dCar = new helperOC::DubinsCar(beacls::FloatVec{(FLOAT_TYPE)0, (FLOAT_TYPE)0, (FLOAT_TYPE)0}, wMax, speed);
	schemeData->dynSys = dCar;

	bool result = true;
	switch (whatTest) {
	default:
	case HJIPDE_solve_WhatTest_Invalid:
	{
		std::stringstream ss;
		ss << "Error Invalid test type: " << whatTest << std::endl;
		message.append(ss.str());
	}
		result = false;
		break;
	case HJIPDE_solve_WhatTest_minWith:
		result = run_UTest_HJIPDE_solve_minWith(message, expected_datas, schemeData, tau, data0,small, execParameters);
		break;
	case HJIPDE_solve_WhatTest_tvTargets:
		result = run_UTest_HJIPDE_solve_tvTarget(message, expected_datas, schemeData, tau, data0, radius,small, execParameters);
		break;
	case HJIPDE_solve_WhatTest_singleObs:
		result = run_UTest_HJIPDE_solve_singleObs(message, expected_datas, schemeData, tau, data0, radius,small, execParameters);
		break;
	case HJIPDE_solve_WhatTest_tvObs:
		result = run_UTest_HJIPDE_solve_tvObs(message, expected_datas, schemeData, tau, data0, radius,small, execParameters);
		break;
	case HJIPDE_solve_WhatTest_obs_stau:
		result = run_UTest_HJIPDE_solve_obs_stau(message, expected_datas, schemeData, data0, radius,small, execParameters);
		break;
	case HJIPDE_solve_WhatTest_stopInit:
		result = run_UTest_HJIPDE_solve_stopInit(message, expected_datas, schemeData, data0,small, execParameters);
		break;
	case HJIPDE_solve_WhatTest_stopSetInclude:
		result = run_UTest_HJIPDE_solve_stopSetInclude(message, expected_datas, schemeData, data0,small, execParameters);
		break;
	case HJIPDE_solve_WhatTest_stopSetIntersect:
		result = run_UTest_HJIPDE_solve_stopSetIntersect(message, expected_datas, schemeData, data0,small, execParameters);
		break;
	case HJIPDE_solve_WhatTest_plotData:
		result = run_UTest_HJIPDE_solve_plotData(message, expected_datas, schemeData, data0, radius,small, execParameters);
		break;
	case HJIPDE_solve_WhatTest_savedData:
		result = run_UTest_HJIPDE_solve_savedData(message, expected_datas, schemeData, tau, data0,small, execParameters);
		break;
	case HJIPDE_solve_WhatTest_stopConvergeSmallDubinsCar:
		result = run_UTest_HJIPDE_solve_stopConverge(message, expected_datas, false, false,small, execParameters);
		break;
	case HJIPDE_solve_WhatTest_stopConvergeSmallDubinsCarCAvoid:
		result = run_UTest_HJIPDE_solve_stopConverge(message, expected_datas, false, true,small, execParameters);
		break;
	}

	if (dCar) delete dCar;
	if (shape) delete shape;
	if (schemeData) delete schemeData;
	if (g) delete g;


	return result;
}

