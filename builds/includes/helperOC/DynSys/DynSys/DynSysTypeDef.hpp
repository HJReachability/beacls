#ifndef __DynSysTypeDef_hpp__
#define __DynSysTypeDef_hpp__
namespace helperOC {
	typedef enum DynSys_UMode_Type {
		DynSys_UMode_Invalid = -1,
		DynSys_UMode_Default,
		DynSys_UMode_Min,
		DynSys_UMode_Max,
	} DynSys_UMode_Type;
	typedef enum DynSys_DMode_Type {
		DynSys_DMode_Invalid = -1,
		DynSys_DMode_Default,
		DynSys_DMode_Min,
		DynSys_DMode_Max,
	} DynSys_DMode_Type;
	typedef enum DynSys_TMode_Type {
		DynSys_TMode_Invalid = -1,
		DynSys_TMode_Default,
		DynSys_TMode_Backward,
		DynSys_TMode_Forward,
	} DynSys_TMode_Type;
};
#endif /* __DynSysTypeDef_hpp__ */
