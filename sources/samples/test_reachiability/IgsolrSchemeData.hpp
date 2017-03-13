#ifndef __IgsolrSchemeData_hpp__
#define __IgsolrSchemeData_hpp__

#include <levelset/levelset.hpp>
#include <cstdint>
#include <vector>
#include <cstddef>
#include <utility>
#include <Core/UVec.hpp>
using namespace std::rel_ops;

class IgsolrSchemeData : public SchemeData {
public:
	FLOAT_TYPE g;
	FLOAT_TYPE k_p;
	FLOAT_TYPE k_t;
	FLOAT_TYPE u_max;
	FLOAT_TYPE u_min;
	size_t u_bound_min;
	size_t u_bound_max;
	FLOAT_TYPE disturbance_rate;
	FLOAT_TYPE default_d_max;
	FLOAT_TYPE default_d_min;
	FLOAT_TYPE w;
	beacls::FloatVec lipschitz;
	beacls::FloatVec impose_d_bounds;
	beacls::FloatVec d_min;
	beacls::FloatVec d_max;
	beacls::FloatVec xs;
	beacls::FloatVec vs;
	beacls::FloatVec ths;
	FLOAT_TYPE D_max_global;
public:
	IgsolrSchemeData() :
		SchemeData() {};
	IgsolrSchemeData(
		const beacls::IntegerVec &gridcounts,
		const size_t u_bound_min,
		const size_t u_bound_max,
		const FLOAT_TYPE hover_counts,
		const FLOAT_TYPE k_p,
		const FLOAT_TYPE disturbance_rate,
		const FLOAT_TYPE default_d_max,
		const FLOAT_TYPE min_x,
		const FLOAT_TYPE max_x,
		const FLOAT_TYPE min_v,
		const FLOAT_TYPE max_v,
		const FLOAT_TYPE w,
		const beacls::FloatVec& lipschitz,
		const beacls::FloatVec& impose_d_bounds
	);
	~IgsolrSchemeData() {}
	bool operator==(const IgsolrSchemeData& rhs) const;
	bool operator==(const SchemeData& rhs) const;

	IgsolrSchemeData* clone() const {
		return new IgsolrSchemeData(*this);
	}
	bool hamFunc(
		beacls::UVec& hamValue_uvec,
		const FLOAT_TYPE t,
		const beacls::UVec& data,
		const std::vector<beacls::UVec>& derivs,
		const size_t begin_index,
		const size_t length
	)const;
	bool partialFunc(
		beacls::UVec& alphas_uvec,
		const FLOAT_TYPE t,
		const beacls::UVec& data,
		const std::vector<beacls::UVec>& derivMins,
		const std::vector<beacls::UVec>& derivMaxs,
		const size_t dim,
		const size_t begin_index,
		const size_t length
	) const;

private:

	/** @overload
	Disable operator=
	*/
	IgsolrSchemeData& operator=(const IgsolrSchemeData& rhs);
	/** @overload
	Disable copy constructor
	*/
	IgsolrSchemeData(const IgsolrSchemeData& rhs) :
		SchemeData(rhs),
		g(rhs.g),
		k_p(rhs.k_p),
		k_t(rhs.k_t),
		u_max(rhs.u_max),
		u_min(rhs.u_min),
		u_bound_min(rhs.u_bound_min),
		u_bound_max(rhs.u_bound_max),
		disturbance_rate(rhs.disturbance_rate),
		default_d_max(rhs.default_d_max),
		default_d_min(rhs.default_d_min),
		w(rhs.w),
		lipschitz(rhs.lipschitz),
		impose_d_bounds(rhs.impose_d_bounds),
		d_min(rhs.d_min),
		d_max(rhs.d_max),
		xs(rhs.xs),
		vs(rhs.vs),
		ths(rhs.ths),
		D_max_global(rhs.D_max_global)
	{}

};

class IgsolrModel_impl;

class IgsolrModel {
public:
		IgsolrModel(
		const size_t num_of_dimensions,
		const beacls::IntegerVec &gridcounts,
		const size_t u_bound_min,
		const size_t u_bound_max,
		const FLOAT_TYPE hover_counts, //!< Hover motor counts for Pelican1
		const FLOAT_TYPE k_p, //!< Motor dynamics. This is chosen as 1/tau of the identified motor dynamics (first order)
		const FLOAT_TYPE disturbance_rate, //!< m/s^2 per meter
		const FLOAT_TYPE default_d_max, //!< m/s^2
		const FLOAT_TYPE min_x, //!< minimum altitude
		const FLOAT_TYPE max_x, //!< maximum altitude
		const FLOAT_TYPE min_v, //!< minimum velocity
		const FLOAT_TYPE max_v, //!< maximum velocity
		const FLOAT_TYPE w, //!< Distrbance nose bound
		const beacls::FloatVec& lipschitz, //!< From GP inference on wind-less quadrotor tests (3rd made up)
		const beacls::FloatVec& impose_d_bounds //!< Indicate whether or not these properties are known and should be fixed
		);
		~IgsolrModel();
	size_t get_num_of_dimensions() const;
		FLOAT_TYPE get_g() const;
	size_t get_u_bound_min() const;
	size_t get_u_bound_max() const;
	FLOAT_TYPE get_ubound2u() const;
		FLOAT_TYPE get_u_min() const;
		FLOAT_TYPE get_u_max() const;
		FLOAT_TYPE get_k_t() const;
		FLOAT_TYPE get_k_p() const;
	FLOAT_TYPE get_disturbance_rate() const;
	FLOAT_TYPE get_default_d_max() const;
	FLOAT_TYPE get_default_d_min() const;
		const beacls::FloatVec& get_d_min() const;
		const beacls::FloatVec& get_d_max() const;
	const beacls::FloatVec& get_sum_points() const;
		const beacls::FloatVec& get_xss() const;
		const beacls::FloatVec& get_vss() const;
		const beacls::FloatVec& get_ths() const;
		size_t get_xss_size() const;
		size_t get_vss_size() const;
		size_t get_ths_size() const;
	FLOAT_TYPE get_D_max_global() const;
	const beacls::FloatVec& get_lipschitz() const;
	FLOAT_TYPE get_w() const;
	const beacls::FloatVec& get_impose_d_bounds() const;
private:
	IgsolrModel_impl *pimpl;
	const beacls::FloatVec dummy_float_type_vector;	//! dummy result for if pimpl is invalid;
	/** @overload
	Disable operator=
	*/
	IgsolrModel& operator=(const IgsolrModel& rhs);
	/** @overload
	Disable copy constructor
	*/
	IgsolrModel(const IgsolrModel& rhs);
};

#endif	/* __IgsolrSchemeData_hpp__ */

