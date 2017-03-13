
#ifndef __BasicShape_hpp__
#define __BasicShape_hpp__

//! Prefix to generate Visual C++ DLL
#ifdef _MAKE_VC_DLL
#define PREFIX_VC_DLL __declspec(dllexport)

//! Don't add prefix, except dll generating
#else              
#define PREFIX_VC_DLL
#endif

#include <cstdint>
#include <vector>
#include <typedef.hpp>
#include <levelset/InitialConditions/InitialCondition.hpp>
class HJI_Grid;

class BasicShape : public InitialCondition {
public:
	virtual bool execute(
		const HJI_Grid *grid, 
		beacls::FloatVec& data
		) const = 0;
	virtual BasicShape* clone() const = 0;
	virtual ~BasicShape() = 0;
private:
};
inline
BasicShape::~BasicShape() {}
#endif	/* __BasicShape_hpp__ */

