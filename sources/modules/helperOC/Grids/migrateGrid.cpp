#include <helperOC/helperOC_type.hpp>
#include <helperOC/Grids/migrateGrid.hpp>
#include <helperOC/ValFuncs/eval_u.hpp>
#include <Core/interpn.hpp>
#include <levelset/Grids/HJI_Grid.hpp>
#include <algorithm>
#include <iostream>
#include <deque>
#include <vector>
#include <numeric>
#include <functional>
namespace helperOC {
	/**
	@brief	migrateGridSingle
	Transfers dataOld onto a from the grid gOld to the grid gNew
	@param	[out]	dataNew	equivalent data corresponding to new grid structure
	@param	[in]	gOld	old grid structures
	@param	[in]	dataOld	data corresponding to old grid structure
	@param	[in]	gNew	new grid structures
	@param	[in]	process	specifies whether to call processGrid to generate
	grid points
	@retval	true			Succeeded
	@retval	false			Failed
	*/
	bool migrateGridSingle(
		beacls::FloatVec& dataNew,
		const levelset::HJI_Grid* gOld,
		const beacls::FloatVec& dataOld,
		const levelset::HJI_Grid* gNew
	);
};

bool helperOC::migrateGridSingle(
	beacls::FloatVec& dataNew,
	const levelset::HJI_Grid* gOld,
	const beacls::FloatVec& dataOld,
	const levelset::HJI_Grid* gNew
) {
#if defined(ADHOCK_XS)
	std::vector<beacls::UVec> gNew_x_uvecs;
	gNew->get_xss(gNew_x_uvecs);
	if (gNew_x_uvecs.empty()) return false;
	std::vector<beacls::FloatVec> gNew_xsVec(gNew_x_uvecs[0].size());
	const size_t gNew_xs_size = gNew_x_uvecs.size();
	for (size_t i = 0; i < gNew_xsVec.size(); ++i) {
		gNew_xsVec[i].resize(gNew_xs_size);
		std::transform(gNew_x_uvecs.cbegin(), gNew_x_uvecs.cend(), gNew_xsVec[i].begin(), [&i](const auto& rhs) {
			const beacls::FloatVec* xs_ptr = beacls::UVec_<FLOAT_TYPE>(rhs).vec();
			return (*xs_ptr)[i];
		});
	}
#else
	const std::vector<beacls::FloatVec>& gNew_xs = gNew->get_xss();
	if (gNew_xs.empty()) return false;
	std::vector<beacls::FloatVec> gNew_xsVec(gNew_xs[0].size());
	const size_t gNew_xs_size = gNew_xs.size();
	for (size_t i = 0; i < gNew_xsVec.size(); ++i) {
		gNew_xsVec[i].resize(gNew_xs_size);
		std::transform(gNew_xs.cbegin(), gNew_xs.cend(), gNew_xsVec[i].begin(), [&i](const auto& rhs) {
			return rhs[i];
		});
	}
#endif
	helperOC::eval_u(dataNew, gOld, dataOld, gNew_xsVec);
	FLOAT_TYPE max_val = std::numeric_limits<FLOAT_TYPE>::signaling_NaN();
	std::for_each(dataNew.cbegin() + 1, dataNew.cend(), [&max_val](const auto& rhs) {
		if ((max_val < rhs) || std::isnan(max_val)) max_val = rhs;
	});
	std::transform(dataNew.cbegin(), dataNew.cend(), dataNew.begin(), [max_val](const auto& rhs) { return (std::isnan(rhs)) ? max_val : rhs; });
	return true;
}

bool helperOC::migrateGrid(
	beacls::FloatVec& dataNew,
	const levelset::HJI_Grid* gOld,
	const beacls::FloatVec& dataOld,
	const levelset::HJI_Grid* gNew
) {
	return migrateGridSingle(dataNew, gOld, dataOld, gNew);
}
bool helperOC::migrateGrid(
	std::vector<beacls::FloatVec>& dataNew,
	const levelset::HJI_Grid* gOld,
	const std::vector<beacls::FloatVec>& dataOld,
	const levelset::HJI_Grid* gNew
) {
	const size_t dataOld_size = dataOld.size();
	bool result = true;
	const size_t newTimeSteps = dataOld_size;
	dataNew.resize(newTimeSteps);
	for (size_t i = 0; i < newTimeSteps; ++i) {
		result &= migrateGridSingle(dataNew[i], gOld, dataOld[i], gNew);
	}
	return result;
}
