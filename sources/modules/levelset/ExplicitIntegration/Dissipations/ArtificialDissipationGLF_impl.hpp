#ifndef __ArtificialDissipationGLF_impl_hpp__
#define __ArtificialDissipationGLF_impl_hpp__

#include <cstdint>
#include <vector>
#include <typedef.hpp>
#include <Core/UVec.hpp>
class SchemeData;

class ArtificialDissipationGLF_impl {
private:
	std::vector<beacls::UVec> alphas_cpu_uvecs;
	std::vector<beacls::UVec> alphas_cuda_uvecs;
public:
	ArtificialDissipationGLF_impl(
		);
	~ArtificialDissipationGLF_impl();
	bool execute(
		beacls::UVec& diss,
		beacls::FloatVec& step_bound_invs,
		std::vector<beacls::UVec>& derivMins,
		std::vector<beacls::UVec>& derivMaxs,
		const FLOAT_TYPE t,
		const beacls::UVec& data,
		const std::vector<beacls::UVec>& x_uvecs,
		const std::vector<beacls::UVec >& deriv_ls,
		const std::vector<beacls::UVec >& deriv_rs,
		const SchemeData *schemeData,
		const size_t begin_index,
		const bool enable_user_defined_dynamics_on_gpu,
		const bool updateDerivMinMax
	);
	bool calculateRangeOfGradient(
		FLOAT_TYPE& derivMin,
		FLOAT_TYPE& derivMax,
		const beacls::UVec& deriv_l_uvec,
		const beacls::UVec& deriv_r_uvec
	) const; 
	bool operator==(const ArtificialDissipationGLF_impl&) const {
		return true;
	}
	ArtificialDissipationGLF_impl* clone() const {
		return new ArtificialDissipationGLF_impl(*this);
	};
	/** @overload
	Disable copy constructor
	*/
	ArtificialDissipationGLF_impl(const ArtificialDissipationGLF_impl&)	{}
private:
	/** @overload
	Disable operator=
	*/
	ArtificialDissipationGLF_impl& operator=(const ArtificialDissipationGLF_impl& rhs);
};

#endif	/* __ArtificialDissipationGLF_impl_hpp__ */

