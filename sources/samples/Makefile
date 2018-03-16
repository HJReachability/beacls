all:
	cd test_reachiability && make all
	cd air3D && make all
	cd plane4D && make all
	cd DubinsCar_RS && make all
	cd Plane_test && make all
	cd PlaneCAvoid_test && make all
	cd Quad4D_test && make all
	cd P5D_Dubins_RS && make all

time: all
	cd test_reachiability && make time
	cd air3D && make time
	cd plane4D && make time
	cd DubinsCar_RS && make time
	cd Plane_test && make time
	cd PlaneCAvoid_test && make time
	cd Quad4D_test && make time

time_chunk_sizes: all
	cd test_reachiability && make time_chunk_sizes
	cd air3D && make time_chunk_sizes
	cd plane4D && make time_chunk_sizes
	cd DubinsCar_RS && make time_chunk_sizes
	cd Plane_test && make time_chunk_sizes
	cd PlaneCAvoid_test && make time_chunk_sizes
	cd Quad4D_test && make time_chunk_sizes

iprofiler: all
	cd test_reachiability && make iprofiler
	cd air3D && make iprofiler
	cd plane4D && make iprofiler
	cd DubinsCar_RS && make iprofiler
	cd Plane_test && make iprofiler
	cd PlaneCAvoid_test && make iprofiler
	cd Quad4D_test && make iprofiler

valgrind: all
	cd test_reachiability && make valgrind
	cd air3D && make valgrind
	cd plane4D && make valgrind
	cd DubinsCar_RS && make valgrind
	cd Plane_test && make valgrind
	cd PlaneCAvoid_test && make valgrind
	cd Quad4D_test && make valgrind

massif: all
	cd test_reachiability && make massif
	cd air3D && make massif
	cd plane4D && make massif
	cd DubinsCar_RS && make massif
	cd Plane_test && make massif
	cd PlaneCAvoid_test && make massif
	cd Quad4D_test && make massif

massif_chunk_sizes: all
	cd test_reachiability && make massif_chunk_sizes
	cd air3D && make massif_chunk_sizes
	cd plane4D && make massif_chunk_sizes
	cd DubinsCar_RS && make massif_chunk_sizes
	cd Plane_test && make massif_chunk_sizes
	cd PlaneCAvoid_test && make massif_chunk_sizes
	cd Quad4D_test && make massif_chunk_sizes

helgrind: all
	cd test_reachiability && make helgrind
	cd air3D && make helgrind
	cd plane4D && make helgrind
	cd DubinsCar_RS && make helgrind
	cd Plane_test && make helgrind
	cd PlaneCAvoid_test && make helgrind
	cd Quad4D_test && make helgrind

test:
	cd test_reachiability && make test
	cd air3D && make test
	cd plane4D && make test
	cd DubinsCar_RS && make test
	cd Plane_test && make test
	cd PlaneCAvoid_test && make test
	cd Quad4D_test && make test

clean:
	cd test_reachiability && make clean
	cd air3D && make clean
	cd plane4D && make clean
	cd DubinsCar_RS && make clean
	cd Plane_test && make clean
	cd PlaneCAvoid_test && make clean
	cd Quad4D_test && make clean

-include $(DEPENDS)
