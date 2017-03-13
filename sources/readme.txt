* How to build library and execute sample

** Linix

1. Install zlib and hdf5

$ sudo apt-get update
$ sudo apt-get install libhdf5-dev

2. Build and install matio (Matlab file I/O library)

$ pushd sources/3rdparty
$ make
$ popd

3. Build BEARS library

$ pushd sources/modules
$ make OPENMP_ON=Y
$ popd

4. Build and execute samples

$ pushd sources/samples
$ make OPENMP_ON=Y all
$ make OPENMP_ON=Y test
$ popd

** Mac OS El Capitan

1. Install OpenMP

brew install clang-omp

2. Build and install matio (Matlab file I/O library)

$ pushd sources/3rdparty
$ make
$ popd

3. Build BEARS library

$ pushd sources/modules
$ make OPENMP_ON=Y
$ popd

4. Build and execute samples

$ pushd sources/samples
$ make OPENMP_ON=Y all
$ make OPENMP_ON=Y test
$ popd

** Visual Studio 2015

1. Download and install HDF5 Pre-built Binary Distributions

1.1 Download pre-build binary distributions from HDF group site

https://www.hdfgroup.org/HDF5/release/obtain5.html

*** 1.8.17-win64-vs2015
	http://www.hdfgroup.org/ftp/HDF5/current/bin/windows/extra/hdf5-1.8.17-win64-vs2015-shared.zip
*** 1.8.17-win32-vs2015
	http://www.hdfgroup.org/ftp/HDF5/current/bin/windows/extra/hdf5-1.8.17-win32-vs2015-shared.zip

2. Build and install matio (Matlab file I/O library)

2.1 Open Visual Studio solution file with the batch file which sets environmental variables for HDF5 path.

- Double click sources\run_visualstudio14_matio.bat

2.2 Build project of matio

3. Build BEARS library

3.1 Open Visual Studio solution file with the batch file which sets environmental variables for HDF5 path and matio path.

- Double click sources\run_visualstudio14_bears.bat

3.2 Build project of bears


4. Build and execute samples


