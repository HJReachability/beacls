#include <levelset/levelset.hpp>
#include <cmath>
#include <numeric>
#include <functional>
#include <cfloat>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cstring>
#include "IgsolrSchemeData.hpp"
/*
* sequence
*/
template<typename T>
class sequence {
private:
	T val_;
	std::function<T(T)> succ_;
public:
	sequence(T val, const std::function<T(T)>& succ) : val_(val), succ_(succ) {}

	operator T() const { return val_; }

	sequence& operator++() { val_ = succ_(val_); return *this; }
	sequence  operator++(int) { sequence r(*this); ++(*this); return r; }

};

int main(int argc, char *argv[])
{
	bool debug_dump_file = false;
//	bool debug_dump_file = true;
	bool dump_file = false;
	if (argc >= 2) {
		dump_file = (atoi(argv[1])==0) ? false : true;
	}
	size_t line_length_of_chunk = 1;
	if (argc >= 3) {
		line_length_of_chunk = atoi(argv[2]);
	}
	bool useCuda = false;
	if (argc >= 4) {
		useCuda = (atoi(argv[3]) == 0) ? false : true;
	}
	int num_of_threads = 0;
	if (argc >= 5) {
		num_of_threads = atoi(argv[4]);
	}
	int num_of_gpus = 0;
	if (argc >= 6) {
		num_of_gpus = atoi(argv[5]);
	}
	levelset::DelayedDerivMinMax_Type delayedDerivMinMax = levelset::DelayedDerivMinMax_Disable;
	if (argc >= 7) {
		switch (atoi(argv[6])) {
		default:
		case 0:
			delayedDerivMinMax = levelset::DelayedDerivMinMax_Disable;
			break;
		case 1:
			delayedDerivMinMax = levelset::DelayedDerivMinMax_Always;
			break;
		case 2:
			delayedDerivMinMax = levelset::DelayedDerivMinMax_Adaptive;
			break;
		}
	}
	bool enable_user_defined_dynamics_on_gpu = true;
	if (argc >= 8) {
		enable_user_defined_dynamics_on_gpu = (atoi(argv[7]) == 0) ? false : true;
	}

	size_t num_of_dimensions = 3;
	size_t num_of_elements = 41;
	beacls::IntegerVec Ns(num_of_dimensions, num_of_elements);

	IgsolrSchemeData *innerData = new IgsolrSchemeData(
		Ns,
		1500, 2200,
		(FLOAT_TYPE)1972.5,
		(FLOAT_TYPE)7.0,
		(FLOAT_TYPE)2.0,
		(FLOAT_TYPE)1.5,
		(FLOAT_TYPE)-0.5, (FLOAT_TYPE)3.0, // altitude
		(FLOAT_TYPE)-4.0, (FLOAT_TYPE)4.0, // velocity
		(FLOAT_TYPE)0.01,
		beacls::FloatVec{(FLOAT_TYPE)0.5, (FLOAT_TYPE)0.1616, (FLOAT_TYPE)0.2},//!< From GP inference on wind-less quadrotor tests (3rd made up)
		beacls::FloatVec{(FLOAT_TYPE)0.0, (FLOAT_TYPE)0.0, (FLOAT_TYPE)0.0} //!< Indicate whether or not these properties are known and should be fixed
	);

	levelset::ShapeRectangleByCorner *shape = new levelset::ShapeRectangleByCorner(
		beacls::FloatVec{(FLOAT_TYPE)0.5, (FLOAT_TYPE)-3.5, (FLOAT_TYPE)-6.0e16},
		beacls::FloatVec{(FLOAT_TYPE)2.8, (FLOAT_TYPE) 3.5, (FLOAT_TYPE)6.0e16});

	levelset::AddGhostExtrapolate *addGhostExtrapolate = new levelset::AddGhostExtrapolate();
	std::vector<levelset::BoundaryCondition*> boundaryConditions(3);
	boundaryConditions[0] = addGhostExtrapolate;
	boundaryConditions[1] = addGhostExtrapolate;
	boundaryConditions[2] = addGhostExtrapolate;

	const beacls::FloatVec& xs = innerData->xs;
	const beacls::FloatVec& vs = innerData->vs;
	const beacls::FloatVec& ths = innerData->ths;
	beacls::FloatVec mins{ xs[0],vs[0],ths[0] };
	beacls::FloatVec maxs{ xs[xs.size() - 1],vs[vs.size() - 1],ths[ths.size() - 1] };
	levelset::HJI_Grid *hJI_Grid = new levelset::HJI_Grid(
		num_of_dimensions);
	std::vector<beacls::FloatVec > vss(num_of_dimensions);
	for (size_t dimension = 0; dimension < num_of_dimensions; ++dimension) {
		vss[dimension].resize((Ns[dimension]));
		FLOAT_TYPE dx = (FLOAT_TYPE)((maxs[dimension] - mins[dimension]) / ((FLOAT_TYPE)Ns[dimension] - 1.0));
		std::iota(vss[dimension].begin(), vss[dimension].end(), sequence<FLOAT_TYPE>(mins[dimension], [dx](auto x) {return x + dx; }));
	}
	hJI_Grid->set_vss(vss);
	hJI_Grid->set_mins(mins);
	hJI_Grid->set_maxs(maxs);
	hJI_Grid->set_boundaryConditions(boundaryConditions);
	hJI_Grid->set_Ns(Ns);

	if (!hJI_Grid->processGrid()) {
		return -1;
	}

	const beacls::UVecType type = useCuda ? beacls::UVecType_Cuda : beacls::UVecType_Vector;

	levelset::UpwindFirstENO2 *spatialDerivative = new levelset::UpwindFirstENO2(hJI_Grid, type);


	levelset::ArtificialDissipationGLF *dissipation = new levelset::ArtificialDissipationGLF();

	std::vector<levelset::PostTimestep_Exec_Type*> postTimestep_Execs;

	innerData->set_spatialDerivative(spatialDerivative);
	innerData->set_dissipation(dissipation);

	innerData->set_grid(hJI_Grid);
	//	innerData->set_hamiltonJacobiFunction(hamiltonJacobiFunction);
	//	innerData->set_partialFunction(partialFunction);

	levelset::TermLaxFriedrichs *innerFunc = new levelset::TermLaxFriedrichs(innerData, type);
	levelset::TermRestrictUpdate *schemeFunc = new levelset::TermRestrictUpdate();
	IgsolrSchemeData *schemeData = new IgsolrSchemeData();

	schemeData->set_grid(hJI_Grid);
	schemeData->set_innerFunc(innerFunc);
	schemeData->set_innerData(innerData);
	schemeData->set_positive(false);

	levelset::Integrator *integrator = new levelset::OdeCFL2(schemeFunc, (FLOAT_TYPE)0.75, (FLOAT_TYPE) 8.0e16, postTimestep_Execs, false, false, NULL);

	beacls::FloatVec data;
	shape->execute(hJI_Grid, data);
	std::transform(data.begin(), data.end(), data.begin(), std::negate<>());

	const FLOAT_TYPE tMax = 5.0;
	const FLOAT_TYPE t0 = 0.0;
	const int plotSteps = 5;

	const FLOAT_TYPE eps = std::numeric_limits<FLOAT_TYPE>::epsilon();	//!< 
	const FLOAT_TYPE small_ratio = 100.;

	FLOAT_TYPE small = small_ratio * eps;
	FLOAT_TYPE tPlot = (tMax - t0) / (plotSteps - 1);

	// Partial Derivative can be calculate if grid and model parameter were known

	// step_bound can also be calculate if grid and model parameter were known


	// Loop until tMax (subject to a little roundoff).
	FLOAT_TYPE tNow = t0;
	beacls::FloatVec y;
	beacls::FloatVec y0;
	while ((tMax - tNow) > small * tMax) {
		y0 = data;
		beacls::FloatVec tspan(2);
		tspan[0] = tNow;
		tspan[1] = HjiMin(tMax, tNow + tPlot);
		if (debug_dump_file) {
			std::stringstream ss;
			ss << std::setprecision(5) << tNow << std::resetiosflags(std::ios_base::floatfield);
			std::string filename = "test_" + ss.str() + ".txt";
			dump_vector(filename.c_str(), data);
		}
		tNow = integrator->execute(y, tspan, y0, schemeData,
			line_length_of_chunk, num_of_threads, num_of_gpus,
			delayedDerivMinMax, enable_user_defined_dynamics_on_gpu);
		data = y;
		printf("tNow = %f\n", tNow);

	}
	if (dump_file) {
		dump_vector(std::string("all_loop.csv"),data);
	}

	if (innerFunc) delete innerFunc;
	if (innerData) delete innerData;
	if (schemeFunc) delete schemeFunc;
	if (dissipation) delete dissipation;
	if (schemeData) delete schemeData;
	if (integrator) delete integrator;
	if (hJI_Grid) delete hJI_Grid;
	if (shape) delete shape;
	if (addGhostExtrapolate) delete addGhostExtrapolate;
	if (spatialDerivative) delete spatialDerivative;


	return 0;
}

