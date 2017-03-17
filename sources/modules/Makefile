.PHONY: all levelset helperOC clean cleanall

all: levelset helperOC

levelset:
	cd levelset && make all

helperOC: levelset
	cd helperOC && make all

clean:
	cd levelset && make clean
	cd helperOC && make clean

cleanall:
	cd levelset && make clean
	cd helperOC && make clean
