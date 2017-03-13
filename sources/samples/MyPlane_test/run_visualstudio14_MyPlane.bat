@echo off
set HDF5_DIR_X64=C:\Program Files\HDF_Group\HDF5\1.8.17
set HDF5_DIR_X86=C:\Program Files (x86)\HDF_Group\HDF5\1.8.17
set MATIO_DIR=%~dp03rdparty\matio\visual_studio
set BOOST_DIR=C:\Boost\boost_1_62_0
set OPENCV_DIR=C:\OpenCV3\opencv3.2.0\opencv\build
set OPENCV_VER=320
set OPENCV_VC_TOOLSET=vc14
set OPENCV_DEBUG_LIBS=opencv_world%OPENCV_VER%d.lib
set OPENCV_RELEASE_LIBS=opencv_world%OPENCV_VER%.lib
set OPENCV_DEBUG_BINS=opencv_world%OPENCV_VER%d.dll
set OPENCV_RELEASE_BINS=opencv_world%OPENCV_VER%.dll
set BEACLS_DIR=..\..\..\builds

echo Opening "MyPlane_test.sln" on Visua Studio 2015
"C:\Program Files (x86)\Microsoft Visual Studio 14.0\Common7\IDE\devenv.exe" "MyPlane_test.sln"

timeout 5
