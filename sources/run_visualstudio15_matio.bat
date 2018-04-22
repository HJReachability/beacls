@echo off
set HDF5_DIR_X64=C:\Program Files\HDF_Group\HDF5\1.8.18
set HDF5_DIR_X86=C:\Program Files (x86)\HDF_Group\HDF5\1.8.18
set SOLUTION_FILE=3rdparty\matio\visual_studio\matio.sln

if exist C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\Common7\IDE\devenv.exe (
  set VS_EDITION=Enterprise
) else (
  if exist C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\Common7\IDE\devenv.exe (
    set VS_EDITION=Professional
  ) else (
    set VS_EDITION=Community
  )
)
echo Opening "%SOLUTION_FILE%" on Visua Studio 2017 %VS_EDITION%
"C:\Program Files (x86)\Microsoft Visual Studio\2017\%VS_EDITION%\Common7\IDE\devenv.exe" "%SOLUTION_FILE%"

timeout 5
