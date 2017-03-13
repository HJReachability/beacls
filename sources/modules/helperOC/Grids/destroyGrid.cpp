#include <levelset/levelset.hpp>
#include <helperOC/Grids/destroyGrid.hpp>
void destroyGrid(
	HJI_Grid* grid
) {
	if (grid) {
		delete grid;
	}
}
	