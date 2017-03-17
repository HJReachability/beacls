#include <helperOC/helperOC_type.hpp>
#include <helperOC/ValFuncs/proj.hpp>
//#include <helperOC/ValFuncs/augmentPeriodicData.hpp>
#include <levelset/BoundaryCondition/AddGhostPeriodic.hpp>
#include <Core/interpn.hpp>
#include <levelset/Grids/HJI_Grid.hpp>
#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <functional>
#include <typeinfo>
#include <macro.hpp>
namespace helperOC {
	/**
	@brief	Projects data corresponding to the grid g in g.dim dimensions, removing
	dimensions specified in dims. If a point is specified, a slice of the
	full-dimensional data at the point xs is taken.
	@param	[out]	dataOut			shifted data
	@param	[in]	g				grid
	@param	[in]	dataIn			original data
	@param	[in]	dims			vector of length g.dim specifying dimensions to project
	For example, if g.dim = 4, then dims = [0 0 1 1] would
	project the last two dimensions
	@param	[in]	xs				Type of projection (defaults to 'min')
	'min':    takes the union across the projected dimensions
	'max':    takes the intersection across the projected dimensions
	a vector: takes a slice of the data at the point xs
	@param	[in]	NOut			number of grid points in output grid (defaults to the same
	number of grid points of the original grid in the unprojected
	dimensions)
	@param	[in]	process			specifies whether to call processGrid to generate
	grid points
	@return							grid corresponding to projected data
	*/
	static levelset::HJI_Grid* projSingle(
			beacls::FloatVec& dataOut,
			const levelset::HJI_Grid* g,
			const beacls::FloatVec& dataIn,
			const beacls::IntegerVec& dims,
			const std::vector<Projection_Type>& x_types,
			const beacls::FloatVec& xs,
			const beacls::IntegerVec& NOut,
			const bool process
		);
	static void gen_interpn_param(
		std::vector<const beacls::FloatVec*>& X_ptrs,
		std::vector<beacls::IntegerVec>& Nss,
		const beacls::IntegerVec& dataNs,
		const beacls::FloatVec& data,
		const std::vector<beacls::FloatVec>& Xs,
		const std::vector<beacls::FloatVec>& Xqs

	);
};
static void helperOC::gen_interpn_param(
	std::vector<const beacls::FloatVec*>& X_ptrs,
	std::vector<beacls::IntegerVec>& Nss,
	const beacls::IntegerVec& dataNs,
	const beacls::FloatVec& data,
	const std::vector<beacls::FloatVec>& Xs,
	const std::vector<beacls::FloatVec>& Xqs

) {
	size_t num_of_dimensions_out = Xs.size();
	X_ptrs.reserve(num_of_dimensions_out * 2 + 1);
	Nss.reserve(num_of_dimensions_out * 2 + 1);
	std::for_each(Xs.cbegin(), Xs.cend(), [&X_ptrs, &Nss](const auto& rhs) {
		X_ptrs.push_back(&rhs);
		beacls::IntegerVec N{ rhs.size() };
		Nss.push_back(N);
	});
	X_ptrs.push_back(&data);
	Nss.push_back(dataNs);
	std::for_each(Xqs.cbegin(), Xqs.cend(), [&X_ptrs, &Nss](const auto& rhs) {
		X_ptrs.push_back(&rhs);
		beacls::IntegerVec N{ rhs.size() };
		Nss.push_back(N);
	});
}
static
levelset::HJI_Grid* helperOC::projSingle(
	beacls::FloatVec& dataOut,
	const levelset::HJI_Grid* g,
	const beacls::FloatVec& dataIn,
	const beacls::IntegerVec& dims,
	const std::vector<Projection_Type>& x_types,
	const beacls::FloatVec& xs,
	const beacls::IntegerVec& NOut,
	const bool process
) {

	//!< Create ouptut grid by keeping dimensions that we are not collapsing
	const size_t dims_count_zero = std::count_if(dims.cbegin(), dims.cend(), [](const auto& rhs) { return (rhs == 0); });
	levelset::HJI_Grid* gOut = new levelset::HJI_Grid(dims_count_zero);

	beacls::FloatVec gOut_mins;
	beacls::FloatVec gOut_maxs;
	std::vector<levelset::BoundaryCondition*> gOut_boundaryConditions;
	gOut_mins.reserve(dims_count_zero);
	gOut_maxs.reserve(dims_count_zero);
	gOut_boundaryConditions.reserve(dims_count_zero);
	const beacls::FloatVec& gIn_mins = g->get_mins();
	const beacls::FloatVec& gIn_maxs = g->get_maxs();
	for (size_t dim = 0; dim < dims.size(); ++dim) {
		if (dims[dim] == 0) {
			gOut_mins.push_back(gIn_mins[dim]);
			gOut_maxs.push_back(gIn_maxs[dim]);
			gOut_boundaryConditions.push_back(g->get_boundaryCondition(dim));
		}
	}
	gOut->set_mins(gOut_mins);
	gOut->set_maxs(gOut_maxs);
	gOut->set_boundaryConditions(gOut_boundaryConditions);

	gOut->set_Ns(NOut);

	//!< Process the grid to populate the remaining fields if necessary
	if (process)
		gOut->processGrid();

	//!< Only compute the grid if value function is not requested
	if (dataIn.empty())
		return gOut;

	//!< min or max
	if (xs.empty()) {
		dataOut = dataIn;
		const beacls::IntegerVec& Ns = g->get_Ns();
		for (size_t dim = 0; dim < dims.size(); ++dim) {
			if (dims[dim] != 0) {
				const size_t inner_dim_size = std::inner_product(Ns.cbegin(), Ns.cbegin() + dim, dims.cbegin(), (size_t)1,
					std::multiplies<size_t>(),
					[](const auto& lhs, const auto& rhs) {
					return (rhs != 0) ? 1 : lhs;
				});
				const size_t target_dim_size = Ns[dim];
				const size_t currente_data_size = dataOut.size();
				const size_t next_data_size = currente_data_size / target_dim_size;
				beacls::FloatVec next_data(next_data_size);
				const size_t outer_dim_step = inner_dim_size * target_dim_size;
				const size_t outer_dim_size = currente_data_size / outer_dim_step;
				const size_t target_dim_step = inner_dim_size;
				switch (x_types[dim]) {
				case Projection_Min:
					for (size_t i = 0; i < outer_dim_size; ++i) {
						const size_t outer_dim_offset = i*outer_dim_step;
						std::copy(dataOut.cbegin() + outer_dim_offset, dataOut.cbegin() + outer_dim_offset + inner_dim_size, next_data.begin() + outer_dim_offset);
						for (size_t j = 1; j < target_dim_size; ++j) {
							const size_t target_dim_offset = outer_dim_offset + j*target_dim_step;
							std::transform(dataOut.cbegin() + target_dim_offset, dataOut.cbegin() + target_dim_offset + inner_dim_size, next_data.cbegin() + outer_dim_offset, next_data.begin() + outer_dim_offset, [](const auto& lhs, const auto& rhs) {
								return (lhs > rhs) ? rhs : lhs;
							});
						}
					}
					break;
				case Projection_Max:
					for (size_t i = 0; i < outer_dim_size; ++i) {
						const size_t outer_dim_offset = i*outer_dim_step;
						std::copy(dataOut.cbegin() + outer_dim_offset, dataOut.cbegin() + outer_dim_offset + inner_dim_size, next_data.begin() + outer_dim_offset);
						for (size_t j = 1; j < target_dim_size; ++j) {
							const size_t target_dim_offset = outer_dim_offset + j*target_dim_step;
							std::transform(dataOut.cbegin() + target_dim_offset, dataOut.cbegin() + target_dim_offset + inner_dim_size, next_data.cbegin() + outer_dim_offset, next_data.begin() + outer_dim_offset, [](const auto& lhs, const auto& rhs) {
								return (lhs < rhs) ? rhs : lhs;
							});
						}
					}
					break;
				default:
					std::cerr << "xs must be a vector, ''min'', or ''max''!" << std::endl;
					delete gOut;
					return NULL;
				}
				dataOut = next_data;
			}
		}
		return gOut;
	}
	//!< Take a slice
	const size_t num_of_dimensions = g->get_num_of_dimensions();

	std::vector<beacls::FloatVec> eval_pt(num_of_dimensions);
	beacls::FloatVec modified_xs = xs;
	std::vector<beacls::Extrapolate_Type> extrapolate_methods;
	extrapolate_methods.resize(num_of_dimensions, beacls::Extrapolate_none);
	size_t target_dimension = 0;
	for (size_t dimension = 0; dimension < num_of_dimensions; ++dimension) {
		const beacls::FloatVec& vs = g->get_vs(dimension);
		if (dims[dimension] != 0) {
			//!< If this dimension is periodic, wrap the input point to the correct period
			const levelset::BoundaryCondition* boundaryCondition = g->get_boundaryCondition(dimension);
			if (boundaryCondition && (typeid(*boundaryCondition) == typeid(levelset::AddGhostPeriodic))) {
				extrapolate_methods[dimension] = beacls::Extrapolate_periodic;
				if (!vs.empty()) {
					const FLOAT_TYPE dx_d = g->get_dx(dimension);
					FLOAT_TYPE val = modified_xs[target_dimension];

					auto minmax_pair = beacls::minmax_value<FLOAT_TYPE>(vs.cbegin(), vs.cend());

					const FLOAT_TYPE min_vs = minmax_pair.first;
					const FLOAT_TYPE max_vs = minmax_pair.second + dx_d;
					const FLOAT_TYPE period = max_vs - min_vs;
					const FLOAT_TYPE modulo_val = val - std::floor((val - min_vs) / period) * period;
					modified_xs[target_dimension] = modulo_val;
				}
			}
			eval_pt[dimension].resize(1);
			eval_pt[dimension][0] = xs[target_dimension];
			++target_dimension;
		}
		else {
			eval_pt[dimension] = vs;
		}
	}
	beacls::FloatVec temp;

	std::vector<const beacls::FloatVec*> eval_pt_ptrs;
	std::vector<beacls::IntegerVec> eval_pt_Ns;

	gen_interpn_param(eval_pt_ptrs, eval_pt_Ns, g->get_Ns(), dataIn, g->get_vss(), eval_pt);
	beacls::interpn(temp, eval_pt_ptrs, eval_pt_Ns, beacls::Interpolate_linear, extrapolate_methods);

	dataOut = temp;
	std::vector<beacls::FloatVec> tmp_vss;
	for (size_t dimension = 0; dimension < num_of_dimensions; ++dimension) {
		if (dims[dimension] == 0) {
			tmp_vss.push_back(g->get_vs(dimension));
		}
	}

	std::vector<const beacls::FloatVec*> X0_ptrs;
	std::vector<beacls::IntegerVec> N0s;
	gen_interpn_param(X0_ptrs, N0s, g->get_Ns(), dataOut, tmp_vss, gOut->get_xss());

	beacls::interpn(dataOut, X0_ptrs, N0s, beacls::Interpolate_linear, extrapolate_methods);
	return gOut;
}

levelset::HJI_Grid*  helperOC::proj(
	beacls::FloatVec& dataOut,
	const levelset::HJI_Grid* g,
	const beacls::FloatVec& dataIn,
	const beacls::IntegerVec& dims,
	const std::vector<Projection_Type>& x_types,
	const beacls::FloatVec& xs,
	const beacls::IntegerVec& NOut,
	const bool process
) {
	if (dims.size() != g->get_num_of_dimensions()) {
		std::cerr << "Dimensions are inconsistent!" << std::endl;
		return NULL;
	}

	const size_t dims_count_nonzero = std::count_if(dims.cbegin(), dims.cend(), [](const auto& rhs) { return (rhs != 0); });
	if (dims_count_nonzero == g->get_num_of_dimensions()) {
		levelset::HJI_Grid* gOut = g->clone();
		dataOut = dataIn;
		std::cerr << "Input and output dimensions are the same!" << std::endl;
		return gOut;
	}

	std::vector<Projection_Type> modified_x_types;
	if (x_types.empty()) {
		modified_x_types.resize(dims.size());
		std::fill(modified_x_types.begin(), modified_x_types.end(), Projection_Min);
	}
	else {
		modified_x_types = x_types;
	}
	//!< % If a slice is requested, make sure the specified point has the correct
	//! dimension
	if (std::any_of(x_types.cbegin(), x_types.cend(), [](const auto& rhs) {return rhs == Projection_Vector; }))
	{
		if (xs.size() != dims_count_nonzero) {
			std::cerr << "Dimension of xs and dims do not match!" << std::endl;
			return NULL;
		}
	}

	beacls::IntegerVec modified_NOut;
	if (NOut.empty()) {
		modified_NOut.reserve(NOut.size());
		for (size_t dim = 0; dim < dims.size(); ++dim) {
			if (dims[dim] == 0) modified_NOut.push_back(g->get_N(dim));
		}
	}
	else {
		modified_NOut = NOut;
	}
	return projSingle(dataOut, g, dataIn, dims, modified_x_types, xs, modified_NOut, process);
}
levelset::HJI_Grid*  helperOC::proj(
	std::vector<beacls::FloatVec>& dataOut,
	const levelset::HJI_Grid* g,
	const std::vector<beacls::FloatVec>& dataIn,
	const beacls::IntegerVec& dims,
	const std::vector<Projection_Type>& x_types,
	const beacls::FloatVec& xs,
	const beacls::IntegerVec& NOut,
	const bool process
) {
	if (dims.size() != g->get_num_of_dimensions()) {
		std::cerr << "Dimensions are inconsistent!" << std::endl;
		return NULL;
	}
	const size_t dims_count_nonzero = std::count_if(dims.cbegin(), dims.cend(), [](const auto& rhs) { return (rhs != 0); });
	if (dims_count_nonzero == g->get_num_of_dimensions()) {
		levelset::HJI_Grid* gOut = g->clone();
		dataOut = dataIn;
		std::cerr << "Input and output dimensions are the same!" << std::endl;
		return gOut;
	}

	std::vector<Projection_Type> modified_x_types;
	if (x_types.empty()) {
		modified_x_types.resize(dims.size());
		std::fill(modified_x_types.begin(), modified_x_types.end(), Projection_Min);
	}
	else {
		modified_x_types = x_types;
	}

	//!< % If a slice is requested, make sure the specified point has the correct
	//! dimension
	if (std::any_of(x_types.cbegin(), x_types.cend(), [](const auto& rhs) {return rhs == Projection_Vector; }))
	{
		if (xs.size() != dims_count_nonzero) {
			std::cerr << "Dimension of xs and dims do not match!" << std::endl;
			return NULL;
		}
	}


	beacls::IntegerVec modified_NOut;
	if (NOut.empty()) {
		modified_NOut.reserve(NOut.size());
		for (size_t dim = 0; dim < dims.size(); ++dim) {
			if (dims[dim] == 0) modified_NOut.push_back(g->get_N(dim));
		}
	}
	else {
		modified_NOut = NOut;
	}

	const size_t dataOld_size = dataIn.size();
	levelset::HJI_Grid* gOut = NULL;
	if (dataOld_size == 1) {
		dataOut.resize(1);
		gOut = projSingle(dataOut[0], g, dataIn[0], dims, x_types, xs, modified_NOut, process);
	}
	else {	//!= dataDims == gOld.dim + 1
		const size_t newTimeSteps = dataOld_size;
		dataOut.resize(newTimeSteps);
		gOut = projSingle(dataOut[0], g, beacls::FloatVec(), dims, x_types, xs, NOut, process);
		for (size_t i = 0; i < newTimeSteps; ++i) {
			levelset::HJI_Grid* tmpG = projSingle(dataOut[i], g, dataIn[i], dims, x_types, xs, modified_NOut, process);
			if (tmpG) delete tmpG;
		}
	}
	return gOut;
}
levelset::HJI_Grid*  helperOC::proj(
	std::vector<beacls::FloatVec>& dataOut,
	const levelset::HJI_Grid* g,
	const std::vector<const beacls::FloatVec*>& dataIn,
	const beacls::IntegerVec& dims,
	const std::vector<Projection_Type>& x_types,
	const beacls::FloatVec& xs,
	const beacls::IntegerVec& NOut,
	const bool process
) {
	if (dims.size() != g->get_num_of_dimensions()) {
		std::cerr << "Dimensions are inconsistent!" << std::endl;
		return NULL;
	}
	const size_t dims_count_nonzero = std::count_if(dims.cbegin(), dims.cend(), [](const auto& rhs) { return (rhs != 0); });
	if (dims_count_nonzero == g->get_num_of_dimensions()) {
		levelset::HJI_Grid* gOut = g->clone();
		dataOut.resize(dataIn.size());
		std::transform(dataIn.cbegin(), dataIn.cend(), dataOut.begin(), [](const auto& rhs) {
			return beacls::FloatVec(rhs->cbegin(), rhs->cend());
		});
		std::cerr << "Input and output dimensions are the same!" << std::endl;
		return gOut;
	}

	std::vector<Projection_Type> modified_x_types;
	if (x_types.empty()) {
		modified_x_types.resize(dims.size());
		std::fill(modified_x_types.begin(), modified_x_types.end(), Projection_Min);
	}
	else {
		modified_x_types = x_types;
	}

	//!< % If a slice is requested, make sure the specified point has the correct
	//! dimension
	if (std::any_of(x_types.cbegin(), x_types.cend(), [](const auto& rhs) {return rhs == Projection_Vector; }))
	{
		if (xs.size() != dims_count_nonzero) {
			std::cerr << "Dimension of xs and dims do not match!" << std::endl;
			return NULL;
		}
	}


	beacls::IntegerVec modified_NOut;
	if (NOut.empty()) {
		modified_NOut.reserve(NOut.size());
		for (size_t dim = 0; dim < dims.size(); ++dim) {
			if (dims[dim] == 0) modified_NOut.push_back(g->get_N(dim));
		}
	}
	else {
		modified_NOut = NOut;
	}

	const size_t dataOld_size = dataIn.size();
	levelset::HJI_Grid* gOut = NULL;
	if (dataOld_size == 1) {
		dataOut.resize(1);
		gOut = projSingle(dataOut[0], g, *dataIn[0], dims, x_types, xs, modified_NOut, process);
	}
	else {	//!= dataDims == gOld.dim + 1
		const size_t newTimeSteps = dataOld_size;
		dataOut.resize(newTimeSteps);
		gOut = projSingle(dataOut[0], g, beacls::FloatVec(), dims, x_types, xs, NOut, process);
		for (size_t i = 0; i < newTimeSteps; ++i) {
			levelset::HJI_Grid* tmpG = projSingle(dataOut[i], g, *dataIn[i], dims, x_types, xs, modified_NOut, process);
			if (tmpG) delete tmpG;
		}
	}
	return gOut;
}

