#include <levelset/levelset.hpp>
#include <helperOC/Grids/createGrid.hpp>
#include <iostream>
levelset::HJI_Grid* helperOC::createGrid(
	const beacls::FloatVec& grid_mins,
	const beacls::FloatVec& grid_maxs,
	const beacls::IntegerVec& Ns,
	const beacls::IntegerVec& pdDims,
	const bool process
) {
	//! Input checks
	if (grid_mins.empty() || grid_maxs.empty()) {
		std::cerr << "Error : " << __func__ << " : grid_min and grid_max must both be vectors!" << std::endl;
	}
	if (grid_mins.size() != grid_maxs.size()) {
		std::cerr << "Error : " << __func__ << " : grid min and grid_max must have the same number of elements!" << std::endl;
	}
	size_t num_of_dimensions = grid_mins.size();

	//! Create the grid
	levelset::HJI_Grid* g = new levelset::HJI_Grid();
	if (g) {
		std::vector<levelset::BoundaryCondition*> boundaryConditions(num_of_dimensions);
		beacls::FloatVec modified_maxs(grid_maxs);
		for (size_t i = 0; i < boundaryConditions.size(); ++i) {
			if (std::find(pdDims.cbegin(), pdDims.cend(), i) != pdDims.cend()) {
				boundaryConditions[i] = new levelset::AddGhostPeriodic();
				modified_maxs[i] = (FLOAT_TYPE)(grid_mins[i] + (grid_maxs[i]- grid_mins[i])  * (1. - 1. / Ns[i]));
			}
			else {
				boundaryConditions[i] = new levelset::AddGhostExtrapolate();
			}
		}
		g->set_boundaryConditions(boundaryConditions);
		for_each(boundaryConditions.cbegin(), boundaryConditions.cend(), ([](auto &ptr) {if (ptr)delete ptr; }));

		g->set_num_of_dimensions(num_of_dimensions);
		g->set_mins(grid_mins);
		g->set_maxs(modified_maxs);
		g->set_Ns(Ns);
		if(process)
			g->processGrid();
	}
	return g;
}
	