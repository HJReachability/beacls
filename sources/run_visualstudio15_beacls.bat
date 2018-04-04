@echo off
set HDF5_DIR_X64=C:\Program Files\HDF_Group\HDF5\1.8.18
set HDF5_DIR_X86=C:\Program Files (x86)\HDF_Group\HDF5\1.8.18
set MATIO_DIR=%~dp03rdparty\matio\visual_studio
set BOOST_DIR=C:\Boost\boost_1_63_0
set OPENCV_DIR=C:\OpenCV3\opencv3.4.1\opencv\build
set OPENCV_VER=341
set OPENCV_VC_TOOLSET=vc15
set OPENCV_DEBUG_LIBS=opencv_world%OPENCV_VER%d.lib
set OPENCV_RELEASE_LIBS=opencv_world%OPENCV_VER%.lib
set OPENCV_DEBUG_BINS=opencv_world%OPENCV_VER%d.dll
set OPENCV_RELEASE_BINS=opencv_world%OPENCV_VER%.dll
set SOLUTION_FILE=beacls.sln

if exsit C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\Common7\IDE\devenv.exe (
  set VS_EDITION=Enterprise
) else (
  if exsit C:\Program Files (x86)\Microsoft Visual Studio\2017\Professional\Common7\IDE\devenv.exe (
    set VS_EDITION=Professional
  ) else (
    set VS_EDITION=Community
  )
)
echo Opening "%SOLUTION_FILE%" on Visua Studio 2017 %VS_EDITION%
"C:\Program Files (x86)\Microsoft Visual Studio\2017\%VS_EDITION%\Common7\IDE\devenv.exe" "%SOLUTION_FILE%"

timeout 5
