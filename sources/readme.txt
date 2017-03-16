* System requirement

** Necessary

*** OS

- Ubuntu Linux 16.04 LTS (x86_64)
- Mac OS X Sierra
- Windows 7/8.1/10 (64bit)

*** Hardware

**** CPU

- Intel Core Processors

** Recommended

*** Hardware

**** CPU

- 4th Generation Intel Core Processors (Haswell arch.), or later 

**** GPU

- NVIDIA GeForce 900 Series (Maxwell arch), or later

* How to build library and execute sample

** Linix (w/o GPU)

1. Install zlib, boost, OpenCV and hdf5

$ sudo apt-get update
$ sudo apt-get upgrade
$ sudo apt-get install zlib libhdf5-dev libboost-dev libopencv-dev

2. Download BEACLS

$ mkdir ~/BEACLS; cd ~/BEACLS
$ git clone https://github.com/HJReachability/beacls
$ cd beacls

3. Build BEACLS

$ cd beacls/sources
$ make all

4. Test BEACLS

$ cd samples/Plane_test
$ make test

** Linix (w/ GPU)

1. Install zlib, boost, OpenCV and hdf5

$ sudo apt-get update
$ sudo apt-get upgrade
$ sudo apt-get install zlib libhdf5-dev libboost-dev libopencv-dev

2. Download and install CUDA 8.0

https://developer.nvidia.com/cuda-downloads

3. Download BEACLS

$ mkdir ~/BEACLS; cd ~/BEACLS
$ git clone https://github.com/HJReachability/beacls
$ cd beacls

4. Build BEACLS

$ cd beacls/sources
$ make WITH_GPU=Y NVCC=/usr/local/cuda-8.0/bin/nvcc all

5. Test BEACLS

$ cd samples/Plane_test
$ make test

** Mac OS El Capitan

1. Install Homeberw

$ export PATH=/usr/local:$PATH 
$ sudo mkdir -p /usr/local
$ ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

2. Install OpenMP, boost, OpenCV and hdf5

$ brew update; brew upgrade
$ brew install clang-omp boost hdf5
$ brew install -with-ffmpeg -with-tbb opencv3
$ brew link opencv3 --force

3. Download BEACLS

$ mkdir ~/BEACLS; cd ~/BEACLS
$ git clone https://github.com/HJReachability/beacls
$ cd beacls

4. Build BEACLS

$ cd beacls/sources
$ make all

5. Test BEACLS

$ cd samples/Plane_test
$ make test

** Windows 7/8.1/10 (w/o GPU)

1. Download and install HDF5 Pre-built Binary Distributions

1.1 Download binary distribution from HDF group site

	https://www.hdfgroup.org/HDF5/release/obtain5.html
	1.8.17-win64-vs2015: http://www.hdfgroup.org/ftp/HDF5/current/bin/windows/extra/hdf5-1.8.17-win64-vs2015-shared.zip

1.2 Extract a zip file and install it.

2. Download and install Boost

2.1 Download binary distribution from the site: http://www.boost.org/users/download/

	1.62.0: https://sourceforge.net/projects/boost/files/boost/1.62.0/boost_1_62_0.zip/download

2.2 Extract a zip file to c:\Boost\Boost_1_62_0

3. Download and install OpenCV

3.1 Download binary distribution from the site: http://opencv.org/

	3.2.0: https://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.2.0/opencv-3.2.0-vc14.exe/download

3.2 Execute installer and extract files to c:\OpenCV3\opencv3.2.0

4. Download and install git for Windows

4.1 Download binary distribution from the site : https://git-for-windows.github.io/

	2.12.0-64bit: https://github.com/git-for-windows/git/releases/download/v2.12.0.windows.1/Git-2.12.0-64-bit.exe

4.2 Execute installer

5. Download and install tortoisegit

5.1 Download Boost from Boost site: https://tortoisegit.org/

	2.4.0.2-64bit: https://download.tortoisegit.org/tgit/2.4.0.0/TortoiseGit-2.4.0.2-64bit.msi

5.2 Execute installer

6. Download and install Visual Studio 2015 Community

	https://www.visualstudio.com/vs/older-downloads/

7. Download BEACLS

7.1 Open Documents folder by explorer

7.2 Choose "Git cloneÅc" from cotext memu.

7.3 Set repository information and push OK

	URL: https://github.com/HJReachability/beacls

7.4 Open beacls folder

8. Build matio (Matlab file I/O library)

8.1 Run Visual Studio solution from the batch file

	sources\run_visualstudio14_matio.bat
		It sets environmental variables for some libraries paths.

8.2 Choose "Release" as Solution Configuration and "x64" as Solution Platform

8.3 Build matio by pushing "F7" key.

9. Build BEACLS

9.1 Run Visual Studio solution from the batch file

	sources\run_visualstudio14_beacls.bat
		It sets environmental variables for some libraries paths.

9.2 Choose "Release" as Solution Configuration and "x64" as Solution Platform

9.3 Build all projects of beacls solution by pushing "F7" key.

10. Execute Plane_test

10.1 Click "Set as StartUp Project" from a context menu of Plane_test in Solution Explorer

10.2 Execute Plane_test by pushing "F5" key.

** Windows 7/8.1/10 (w/ GPU)

Install from step 1 to step 8 of Windows 7/8.1/10 (without GPU)

9. Download and install CUDA 8.0

	https://developer.nvidia.com/cuda-downloads

10. Build BEACLS

10.1 Run Visual Studio solution from the batch file

	sources\run_visualstudio14_beacls_cuda.bat
		It sets environmental variables for some libraries paths.

10.2 Run Visual Studio solution from the batch file

10.3 Enable CUDA build for levelset project.

10.3.1 Choose "levelset" in Solution Explorer

10.3.1 click "Builld CustomizationsÅc" from Project tab of tool bar.

10.3.1 Enable "CUDA 8.0(.targets, .props)"

10.4 Enable CUDA build for helperOC project.

10.5 Choose "Release" as Solution Configuration and "x64" as Solution Platform

10.6 Build all projects of beacls solution by pushing "F7" key.

11. Execute Plane_test

11.1 Click "Set as StartUp Project" from a context menu of Plane_test in Solution Explorer

11.2 Execute Plane_test by pushing "F5" key.


