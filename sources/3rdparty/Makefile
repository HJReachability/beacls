ARCH		:= $(shell uname -m)
UNAME		:= $(shell uname)
PWD		:= $(shell pwd)
INSTALL_DIR	:= $(PWD)/../../builds
MATIO_LIB	:= $(INSTALL_DIR)/$(ARCH)/lib/libmatio.a
ifeq ($(UNAME),Darwin)
	HDF5_DIR        := /usr/local
	CONFIGURE_OPT   += --with-hdf5=$(HDF5_DIR)
endif
ifeq ($(UNAME),Linux)
	HDF5_DIR        := /usr/lib/x86_64-linux-gnu/hdf5/serial
	CONFIGURE_OPT	+= --with-hdf5=$(HDF5_DIR) --enable-mat73=true
endif

.PHONY: all matio test clean
vpath %.a matio/lib

all: install

matio: libmatio.a

libmatio.a:
	cd matio; ./configure --prefix=`pwd` $(CONFIGURE_OPT) ; make install

install: matio
	mkdir -p $(INSTALL_DIR)/$(ARCH)/lib
	cp -rp matio/lib/libmatio* $(INSTALL_DIR)/$(ARCH)/lib

test: matio
	cd matio; make test

clean:
	cd matio; make clean
	rm -rf $(INSTALL_DIR)/$(ARCH)/lib/libmatio*

cleanall: clean
	rm -rf matio/lib matio/include

-include $(DEPENDS)
