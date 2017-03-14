.PHONY: all 3rdparty modules SeqPP time time_chunk_sizes \
	iprofiler valgrind massif massif_chunk_sizes helgrind \
	unittest sampletest test clean cleanall

all: 3rdparty modules
	cd samples && make all
	cd unittests && make all

3rdparty:
	cd 3rdparty && make all

modules: 3rdparty
	cd modules && make all

time: modules
	cd samples && make time
	cd unittests && make time

time_chunk_sizes: modules
	cd samples && make time_chunk_sizes

iprofiler: modules
	cd samples && make iprofiler
	cd unittests && make iprofiler

valgrind: modules
	cd samples && make valgrind
	cd unittests && make valgrind

massif: modules
	cd samples && make massif
	cd unittests && make massif

massif_chunk_sizes: modules
	cd samples && make massif_chunk_sizes

helgrind: modules
	cd samples && make helgrind
	cd unittests && make helgrind

unittest: modules
	cd unittests && make test

sampletest: modules
	cd samples && make test

test: modules unittest sampletest

clean:
	cd modules && make clean
	cd samples && make clean
	cd unittests && make clean

cleanall:
	cd 3rdparty && make cleanall
	cd modules && make clean
	cd samples && make clean
	cd unittests && make clean
