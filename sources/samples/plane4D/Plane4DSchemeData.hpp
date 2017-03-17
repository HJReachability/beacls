#ifndef __Plane4DSchemeData_hpp__
#define __Plane4DSchemeData_hpp__

#include <levelset/levelset.hpp>
#include <cstdint>
#include <vector>
#include <cstddef>
#include <utility>
#include <Core/UVec.hpp>
using namespace std::rel_ops;

class Plane4DSchemeData : public levelset::SchemeData {
public:
	FLOAT_TYPE wMax;
	beacls::FloatVec aranges;
public:
	Plane4DSchemeData(
	) : SchemeData(), wMax(0), aranges(beacls::FloatVec()){}
	~Plane4DSchemeData() {}
	bool operator==(const Plane4DSchemeData& rhs) const;
	bool operator==(const SchemeData& rhs) const;

	Plane4DSchemeData* clone() const {
		return new Plane4DSchemeData(*this);
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
	Plane4DSchemeData& operator=(const Plane4DSchemeData& rhs);
	/** @overload
	Disable copy constructor
	*/
	Plane4DSchemeData(const Plane4DSchemeData& rhs) :
		SchemeData(rhs),
		wMax(rhs.wMax),
		aranges(rhs.aranges)
	{}

};

#endif	/* __Plane4DSchemeData_hpp__ */

