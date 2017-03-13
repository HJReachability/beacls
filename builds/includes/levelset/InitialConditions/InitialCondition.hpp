
#ifndef __InitialCondition_hpp__
#define __InitialCondition_hpp__

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

class HJI_Grid;

class InitialCondition  {
public:
	PREFIX_VC_DLL
	virtual bool execute(
		const HJI_Grid *grid, 
		beacls::FloatVec& data
		) const = 0;
	PREFIX_VC_DLL
	virtual InitialCondition* clone() const = 0;
	virtual ~InitialCondition() = 0;

};
inline
InitialCondition::~InitialCondition() {}
#endif	/* __InitialCondition_hpp__ */

