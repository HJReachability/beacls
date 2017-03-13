#ifndef __Air3DSchemeData_hpp__
#define __Air3DSchemeData_hpp__

#include <levelset/levelset.hpp>
#include <cstdint>
#include <vector>
#include <cstddef>
#include <utility>
#include <Core/UVec.hpp>
using namespace std::rel_ops;

class Air3DSchemeData : public SchemeData {
public:
	FLOAT_TYPE velocityA;
	FLOAT_TYPE velocityB;
	FLOAT_TYPE inputA;
	FLOAT_TYPE inputB;
public:
	Air3DSchemeData(
	) : SchemeData(), velocityA(0), velocityB(0), inputA(0), inputB(0) {}
	~Air3DSchemeData() {}
	bool operator==(const Air3DSchemeData& rhs) const;
	bool operator==(const SchemeData& rhs) const;
	Air3DSchemeData* clone() const {
		return new Air3DSchemeData(*this);
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
	Air3DSchemeData& operator=(const Air3DSchemeData& rhs);
	/** @overload
	Disable copy constructor
	*/
	Air3DSchemeData(const Air3DSchemeData& rhs) :
		SchemeData(rhs),
		velocityA(rhs.velocityA),
		velocityB(rhs.velocityB),
		inputA(rhs.inputA),
		inputB(rhs.inputB)
	{}

};

#endif	/* __Air3DSchemeData_hpp__ */

