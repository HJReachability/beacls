#ifndef __UpwindFirstWENO5a_impl_hpp__
#define __UpwindFirstWENO5a_impl_hpp__

#include <cstdint>
#include <vector>
#include <typedef.hpp>
#include <Core/UVec.hpp>
class HJI_Grid;
class UpwindFirstENO3aHelper;
class UpwindFirstWENO5a_Cache {
public:
	std::vector<beacls::FloatVec > cachedBoundedSrcs;
	std::vector<beacls::FloatVec > last_d1ss;
	beacls::FloatVec last_d2s;
	beacls::FloatVec last_d2s_fabs;
	beacls::FloatVec last_dx_d2_effs;
	std::vector<const FLOAT_TYPE*> boundedSrc_ptrs;
	UpwindFirstWENO5a_Cache() : cachedBoundedSrcs(2), boundedSrc_ptrs(2) {}
};

class UpwindFirstWENO5a_impl {
private:
	beacls::UVecType type;
	beacls::IntegerVec first_dimension_loop_sizes;
	beacls::IntegerVec inner_dimensions_loop_sizes;
	beacls::IntegerVec src_target_dimension_loop_sizes;

	const size_t stencil;
	size_t num_of_strides;

	std::vector<std::vector<beacls::UVec > > dL_uvecs;
	std::vector<std::vector<beacls::UVec > > dR_uvecs;
	std::vector<std::vector<beacls::UVec > > DD_uvecs;

	std::vector<std::vector<beacls::FloatVec > > tmpSmooths_m1s;
	std::vector<beacls::FloatVec > tmpSmooths;

	const beacls::FloatVec weightL = { (FLOAT_TYPE)0.1, (FLOAT_TYPE)0.6,(FLOAT_TYPE)0.3 };
	const beacls::FloatVec weightR = { (FLOAT_TYPE)0.3, (FLOAT_TYPE)0.6,(FLOAT_TYPE)0.1 };

	beacls::EpsilonCalculationMethod_Type epsilonCalculationMethod_Type;
	UpwindFirstENO3aHelper* upwindFirstENO3aHelper;
	std::vector<beacls::CudaStream*> cudaStreams;
public:
	UpwindFirstWENO5a_impl(
		const HJI_Grid *hji_grid,
		const beacls::UVecType type = beacls::UVecType_Vector
	);
	~UpwindFirstWENO5a_impl();

	bool execute_dim0(
		beacls::UVec& dst_deriv_l,
		beacls::UVec& dst_deriv_r,
		const FLOAT_TYPE* src,
		const HJI_Grid *grid,
		const size_t dim,
		const size_t loop_begin,
		const size_t slice_length,
		const size_t num_of_slices
	);
	bool execute_dim1(
		beacls::UVec& dst_deriv_l,
		beacls::UVec& dst_deriv_r,
		const FLOAT_TYPE* src,
		const HJI_Grid *grid,
		const size_t dim,
		const size_t loop_begin,
		const size_t slice_length,
		const size_t num_of_slices
	);
	bool execute_dimLET2(
		beacls::UVec& dst_deriv_l,
		beacls::UVec& dst_deriv_r,
		const FLOAT_TYPE* src,
		const HJI_Grid *grid,
		const size_t dim,
		const size_t loop_begin,
		const size_t slice_length,
		const size_t num_of_slices
	);
	bool execute(
		beacls::UVec& dst_deriv_l,
		beacls::UVec& dst_deriv_r,
		const HJI_Grid *grid,
		const FLOAT_TYPE* src,
		const size_t dim,
		const bool generateAll,
		const size_t loop_begin,
		const size_t slice_length,
		const size_t num_of_slices
	);
	bool synchronize(const size_t dim);
	bool operator==(const UpwindFirstWENO5a_impl& rhs) const;
	UpwindFirstWENO5a_impl* clone() const {
		return new UpwindFirstWENO5a_impl(*this);
	};
	beacls::UVecType get_type() const {
		return type;
	};
private:


	/** @overload
	Disable operator=
	*/
	UpwindFirstWENO5a_impl& operator=(const UpwindFirstWENO5a_impl& rhs);
	/** @overload
	Disable copy constructor
	*/
	UpwindFirstWENO5a_impl(const UpwindFirstWENO5a_impl& rhs);
};

#endif	/* __UpwindFirstWENO5a_impl_hpp__ */

