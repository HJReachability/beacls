
#ifndef __HJIPDE_impl_hpp__
#define __HJIPDE_impl_hpp__

#include <cstdint>
#include <vector>
#include <deque>
#include <cstddef>
#include <levelset/levelset.hpp>
#include <helperOC/DynSys/DynSys/DynSys.hpp>
#include <helperOC/DynSys/DynSys/DynSysSchemeData.hpp>
#include <helperOC/ValFuncs/HJIPDE.hpp>
namespace helperOC {
	class HJIPDE_impl {
	public:
	private:
		std::string tmp_filename;
		std::string windowName;
		std::deque<beacls::FloatVec > datas;
		beacls::FloatVec calculatedTTR;
		beacls::FloatVec last_data;

	public:
		HJIPDE_impl(
			const std::string& tmp_filename);
		~HJIPDE_impl();
		bool solve(
			beacls::FloatVec& stoptau,
			helperOC::HJIPDE_extraOuts& extraOuts,
			const std::vector<beacls::FloatVec>& src_datas,
			const beacls::FloatVec& tau,
			const DynSysSchemeData* schemeData,
			const HJIPDE::MinWithType minWith,
			const helperOC::HJIPDE_extraArgs& extraArgs
		);
		bool solve_local_q(
			beacls::FloatVec& dst_tau,
			helperOC::HJIPDE_extraOuts& extraOuts,
			const std::vector<beacls::FloatVec>& src_datas,
			const beacls::IntegerVec& qIndexes,
			const beacls::FloatVec& src_tau,
			const FLOAT_TYPE updateEpsilon,
			const DynSysSchemeData* schemeData,
			const HJIPDE::MinWithType minWith,
			const helperOC::HJIPDE_extraArgs& extraArgs
		);
		bool getNeighbors(
			std::set<size_t> &neighbors, 
			const std::set<size_t> &Q, 
			const int num_neighbors,
			const levelset::HJI_Grid *g,
			const size_t periodic_dim
		);
		bool getLocalQNumericalFuncs(
			levelset::Dissipation *&dissFunc,
			levelset::Integrator *&integratorFunc,
			levelset::SpatialDerivative *&derivFunc,
			const levelset::HJI_Grid *grid,
			const levelset::Term *schemeFunc,
			const helperOC::Dissipation_Type dissType,
			const helperOC::ApproximationAccuracy_Type accuracy,
			const FLOAT_TYPE factorCFL,
			const bool stats,
			const bool single_step,
			const beacls::UVecType type
		) const;
		bool valid_Q_values(
			const std::set<size_t> &Q, 
			const std::set<size_t> &QOld
		);
		bool get_datas(
			std::vector<beacls::FloatVec >& dst_datas,
			const beacls::FloatVec& src_tau,
			const DynSysSchemeData* schemeData
		) const;
		bool TD2TTR(
			beacls::FloatVec& TTR,
			const levelset::HJI_Grid* g,
			const beacls::FloatVec& tau
		) const;
		bool getNumericalFuncs(
			levelset::Dissipation*& dissFunc,
			levelset::Integrator*& integratorFunc,
			levelset::SpatialDerivative*& derivFunc,
			const levelset::HJI_Grid* grid,
			const levelset::Term* schemeFunc,
			const helperOC::Dissipation_Type dissType,
			const helperOC::ApproximationAccuracy_Type accuracy,
			const FLOAT_TYPE factorCFL,
			const bool stats,
			const bool single_step,
			const beacls::UVecType type
		) const;
		bool get_last_data(
			beacls::FloatVec& dst_data
		)const;
		bool set_last_data(
			const beacls::FloatVec& src_data
		);
	private:
		/** @overload
		Disable operator=
		*/
		HJIPDE_impl& operator=(const HJIPDE_impl& rhs);
		/** @overload
		Disable copy constructor
		*/
		HJIPDE_impl(const HJIPDE_impl& rhs);
	};
};
#endif	/* __HJIPDE_impl_hpp__ */

