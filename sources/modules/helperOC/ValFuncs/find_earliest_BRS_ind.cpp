#include <helperOC/helperOC_type.hpp>
#include <helperOC/ValFuncs/find_earliest_BRS_ind.hpp>
#include <helperOC/ValFuncs/eval_u.hpp>
#include <levelset/Grids/HJI_Grid.hpp>
#include <vector>

size_t helperOC::find_earliest_BRS_ind(
	const HJI_Grid* g,
	const std::vector<beacls::FloatVec >& data,
	const std::vector<beacls::FloatVec >& x,
	const size_t org_upper,
	const size_t org_lower
)
{
	static const FLOAT_TYPE small = static_cast<FLOAT_TYPE>(1e-4);
	size_t upper, lower;
	if (org_upper < org_lower) {
		upper = data.size()-1;
		lower = 0;
	}
	else {
		upper = org_upper;
		lower = org_lower;
	}
	size_t tEarliest;
	while (upper > lower) {
		tEarliest = static_cast<size_t>(std::ceil(((FLOAT_TYPE)upper + lower) / 2));
		beacls::FloatVec valueAtX;
		if (data[tEarliest].empty()) {
			valueAtX.resize(x.size(), 0);
		}
		else {
			helperOC::eval_u(valueAtX, g, data[tEarliest], x);
		}
		if (valueAtX[0] < small) {
			//!< point is in reachable set; eliminate all lower indices
			lower = tEarliest;
		}
		else {
			//!< too late
			upper = tEarliest - 1;
		}
	}
	tEarliest = upper;
	return tEarliest;
}
