#ifndef __TermRestrictUpdate_impl_hpp__
#define __TermRestrictUpdate_impl_hpp__

#include <cstdint>
#include <vector>
#include <typedef.hpp>
namespace levelset {

	class SchemeData;

	class TermRestrictUpdate_impl {
	private:
		const beacls::UVecType type;
	public:
		TermRestrictUpdate_impl(
			const beacls::UVecType type
		);
		~TermRestrictUpdate_impl();
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
		) const;
		bool synchronize(const SchemeData* schemeData) const;
		bool operator==(const TermRestrictUpdate_impl& rhs) const {
			if (this == &rhs) return true;
			else if (type != rhs.type) return false;
			return true;
		}
		TermRestrictUpdate_impl* clone() const {
			return new TermRestrictUpdate_impl(*this);
		};
	private:
		/** @overload
		Disable operator=
		*/
		TermRestrictUpdate_impl& operator=(const TermRestrictUpdate_impl& rhs);
		/** @overload
		Disable copy constructor
		*/
		TermRestrictUpdate_impl(const TermRestrictUpdate_impl& rhs) :
			type(rhs.type)
		{}
	};
};
#endif	/* __TermRestrictUpdate_impl_hpp__ */

