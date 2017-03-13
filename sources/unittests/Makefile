GPU_ON = N
OPENMP_ON = N
PROFILE_ON = N
SINGLE_PRECISION_ON = N
MAKE_TABLE_ON = N


all:
	cd SpatialDerivative; make all
	cd Dissipation; make all
	cd HamFunc; make all
	cd Term; make all
	cd HJIPDE_solve; make all

time: all
	cd SpatialDerivative; make time
	cd Dissipation; make time
	cd HamFunc; make time
	cd Term; make time
	cd HJIPDE_solve; make time

iprofiler: all
	cd SpatialDerivative; make iprofiler
	cd Dissipation; make iprofiler
	cd HamFunc; make iprofiler
	cd Term; make iprofiler
	cd HJIPDE_solve; make iprofiler

valgrind: all
	cd SpatialDerivative; make valgrind
	cd Dissipation; make valgrind
	cd HamFunc; make valgrind
	cd Term; make valgrind
	cd HJIPDE_solve; make valgrind

massif: all
	cd SpatialDerivative; make massif
	cd Dissipation; make massif
	cd HamFunc; make massif
	cd Term; make massif
	cd HJIPDE_solve; make massif

helgrind: all
	cd SpatialDerivative; make helgrind
	cd Dissipation; make helgrind
	cd HamFunc; make helgrind
	cd Term; make helgrind
	cd HJIPDE_solve; make helgrind

test: all
	cd SpatialDerivative; make test
	cd Dissipation; make test
	cd HamFunc; make test
	cd Term; make test
	cd HJIPDE_solve; make test

clean:
	cd SpatialDerivative; make clean
	cd Dissipation; make clean
	cd HamFunc; make clean
	cd Term; make clean
	cd HJIPDE_solve; make clean
