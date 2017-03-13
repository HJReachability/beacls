@echo off
set HDF5_DIR_X64=C:\Program Files\HDF_Group\HDF5\1.8.17
set HDF5_DIR_X86=C:\Program Files (x86)\HDF_Group\HDF5\1.8.17

echo Opening "matio.sln" on Visua Studio 2015
"C:\Program Files (x86)\Microsoft Visual Studio 14.0\Common7\IDE\devenv.exe" "3rdparty\matio\visual_studio\matio.sln"

timeout 5
