
#ifndef __TermLaxFriedrichs_impl_hpp__
#define __TermLaxFriedrichs_impl_hpp__

#include <cstdint>
#include <vector>
#include <Core/UVec.hpp>
#include <typedef.hpp>
class SchemeData;
namespace beacls {
	class CacheTag;
};

class TermLaxFriedrichs_impl {
private:
	size_t first_dimension_loop_size;
	size_t num_of_dimensions;
	std::vector<beacls::UVec> deriv_l_uvecs;
	std::vector<beacls::UVec> deriv_r_uvecs;
	std::vector<beacls::UVec> deriv_c_uvecs;
	std::vector<beacls::UVec> deriv_c_cpu_uvecs;

	std::vector<beacls::UVec> x_uvecs;
	std::vector<beacls::UVec> deriv_max_uvecs;
	std::vector<beacls::UVec> deriv_min_uvecs;
	beacls::UVec ydot_cuda_uvec;
	beacls::UVec ham_uvec;
	beacls::UVec ham_cpu_uvec;
	beacls::UVec diss_uvec;
	beacls::UVecType type;
	beacls::CacheTag* cacheTag;
public:
	TermLaxFriedrichs_impl(
		const SchemeData* schemeData,
		const beacls::UVecType type
	);
	~TermLaxFriedrichs_impl();
	bool execute(
		beacls::FloatVec::iterator ydot_ite,
		beacls::FloatVec& step_bound_invs,
		const FLOAT_TYPE t,
		const beacls::FloatVec& y,
		std::vector<beacls::FloatVec >& derivMins,
		std::vector<beacls::FloatVec >& derivMaxs,
		const SchemeData *schemeData,
		const size_t loop_begin,
		const size_t loop_length,
		const size_t num_of_slices,
		const bool enable_user_defined_dynamics_on_gpu,
		const bool updateDerivMinMax
	);
	bool synchronize(const SchemeData* schemeData) const;
	bool operator==(const TermLaxFriedrichs_impl& rhs) const {
		if (this == &rhs) return true;
		else if (type != rhs.type)return false;
		else if (num_of_dimensions != rhs.num_of_dimensions)return false;
		return true;
	}
	TermLaxFriedrichs_impl* clone() const {
		return new TermLaxFriedrichs_impl(*this);
	};
private:
	/** @overload
	Disable operator=
	*/
	TermLaxFriedrichs_impl& operator=(const TermLaxFriedrichs_impl& rhs);
	/** @overload
	Disable copy constructor
	*/
	TermLaxFriedrichs_impl(const TermLaxFriedrichs_impl& rhs);
};

#endif	/* __TermLaxFriedrichs_impl_hpp__ */

