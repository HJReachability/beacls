#include <levelset/levelset.hpp>
#include <helperOC/Grids/destroyGrid.hpp>
void helperOC::destroyGrid(
	levelset::HJI_Grid* grid
) {
	if (grid) {
		delete grid;
	}
}
	