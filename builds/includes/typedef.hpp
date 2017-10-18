#ifndef __typedef_hpp__
#define __typedef_hpp__
#include <vector>
#include <cstddef>
#define SINGLE_PRECISION
#if defined(SINGLE_PRECISION)
#define FLOAT_TYPE_32F
#else	/* defined(SINGLE_PRECISION) */
#define FLOAT_TYPE_64F
#endif	/* defined(SINGLE_PRECISION) */

#if defined(FLOAT_TYPE_64F)
typedef double FLOAT_TYPE;
#else	/* defined(FLOAT_TYPE_64F) */
typedef float FLOAT_TYPE;
#endif	/* defined(FLOAT_TYPE_64F) */

namespace beacls {
	typedef std::vector<FLOAT_TYPE> FloatVec;
	typedef std::vector<size_t> IntegerVec;
	class CudaStream;
	typedef enum Interpolate_Type {
		Interpolate_Invalid = -1,
		Interpolate_nearest,
		Interpolate_linear,
		Interpolate_pchip,
		Interpolate_cubic,
		Interpolate_spline,
		Interpolate_none,
	} Interpolate_Type;
	typedef enum Extrapolate_Type {
		Extrapolate_Invalid = -1,
		Extrapolate_nearest,
		Extrapolate_linear,
		Extrapolate_pchip,
		Extrapolate_cubic,
		Extrapolate_spline,
		Extrapolate_none,
		Extrapolate_periodic,
	} Extrapolate_Type;	


	typedef enum UVecType {
		UVecType_Invalid = -1,
		UVecType_Vector,
		UVecType_Cuda,
	} UVecType;

	typedef enum UVecDepth {
		UVecDepth_Invalid = -1,
		UVecDepth_8U = 0,
		UVecDepth_8S,
		UVecDepth_16U,
		UVecDepth_16S,
		UVecDepth_32S,
		UVecDepth_32F,
		UVecDepth_64F,
		UVecDepth_User,
		UVecDepth_32U,
		UVecDepth_64U,
		UVecDepth_64S,
	} UVecDepth;
	class MatFStream;
	class MatVariable;
	typedef enum MatOpenMode {
		MatOpenMode_Read,
		MatOpenMode_Write,
		MatOpenMode_WriteAppend,
	}MatOpenMode;
};	// beacls
namespace levelset {
	class HJI_Grid;
	typedef bool(*PostTimestep_Exec_Type)(
		beacls::FloatVec&,
		const FLOAT_TYPE,
		const beacls::FloatVec&,
		const levelset::HJI_Grid *
		);

	typedef bool(*TerminalEvent_Exec_Type)(
		beacls::FloatVec&,
		const FLOAT_TYPE,
		const beacls::FloatVec&,
		const FLOAT_TYPE,
		const beacls::FloatVec&,
		const levelset::HJI_Grid *
		);
	typedef enum EpsilonCalculationMethod_Type {
		EpsilonCalculationMethod_Invalid,
		EpsilonCalculationMethod_Constant,
		EpsilonCalculationMethod_maxOverGrid,
		EpsilonCalculationMethod_maxOverNeighbor,
	} EpsilonCalculationMethod_Type;
	typedef enum DelayedDerivMinMax_Type {
		DelayedDerivMinMax_Invalid = -1,
		DelayedDerivMinMax_Disable,
		DelayedDerivMinMax_Always,
		DelayedDerivMinMax_Adaptive,
	} DelayedDerivMinMax_Type;
};
#endif	/* __typedef_hpp__ */
