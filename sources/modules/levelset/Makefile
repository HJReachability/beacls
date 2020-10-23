GPU_ON = N
VISUALIZE_ON = Y
GUI_ON = Y
FILESYSTEM_ON = N
VERBOSE_ON = N
STATIC_ON = N
PROFILE_ON = N
FMA_ON = Y
USER_DEFINED_GPU_DYNSYS_FUNC_ON = Y
OPTIMIZE = 0
WARNING_ON = Y
SINGLE_PRECISION_ON = Y
ARCH = $(shell uname -m)
PWD = $(shell pwd)
NVCC = /usr/bin/nvcc
INSTALL_DIR = $(PWD)/../../../builds
MATIO_DIR = $(PWD)/../../3rdparty/matio
MATIO_LIB_DIR = $(INSTALL_DIR)/$(ARCH)/lib
MATIO_INC_DIR = $(MATIO_DIR)/include

UNAME = $(shell uname)
ifeq ($(UNAME),Darwin)
	HOSTCC_DIR = /usr
	HOSTCC  = $(HOSTCC_DIR)/bin/clang++
	OPENMP_ON = N
else
	HOSTCC  = g++
	OPENMP_ON = Y
endif


ifeq ($(GPU_ON), Y)
	CPP = $(NVCC)
	LINKER = $(NVCC)
	AR = ar
	HOST_COMPILER	?= $(HOSTCC)
	NVCCFLAGS += -arch=sm_52 -maxrregcount=64
	NVLDFLAGS += -arch=sm_52 -maxrregcount=64
	NVCCFLAGS += -ccbin $(HOST_COMPILER)
	NVCCFLAGS += -lineinfo
	NVCCFLAGS += -DWITH_GPU
	NVCCFLAGS += -L /usr/lib/x86_64-linux-gnu -L/usr/lib/nvidia-361
	NVCCFLAGS += -D_FORCE_INLINES
	NVCCFLAGS += -D_MWAITXINTRIN_H_INCLUDED
	ifeq ($(VERBOSE_ON), Y)
		NVCCFLAGS += --ptxas-options=-v 
	endif
else
	CPP = $(HOSTCC)
	AR = ar
	LINKER = $(HOSTCC)
endif

CFLAGS   += -mfpmath=sse -msse2 -g -march=native
CPPFLAGS += -std=c++14
NVCCFLAGS += -std=c++11

ifeq ($(FMA_ON), N)
	CFLAGS	+= -mno-fma
endif
ifeq ($(UNAME),Darwin)
	CFLAGS += -fno-common
	CFLAGS	+= -I/usr/local/include
	LDFLAGS += -L/usr/local/lib -lstdc++
endif
ifeq ($(UNAME),Linux)
	CFLAGS	+= -fPIC
endif

LDFLAGS  += -g

ifeq ($(UNAME),Darwin)
	LDFLAGS += -L$(MATIO_LIB_DIR) -lmatio -lz -lhdf5
endif
ifeq ($(UNAME),Linux)
	LDFLAGS += -L$(MATIO_LIB_DIR) -lmatio -lz -lhdf5_cpp
endif
CFLAGS += -I$(MATIO_INC_DIR)

ifeq ($(SINGLE_PRECISION_ON), Y)
	CFLAGS += -DSINGLE_PRECISION
endif

ifeq ($(USER_DEFINED_GPU_DYNSYS_FUNC_ON), Y)
	CFLAGS += -DUSER_DEFINED_GPU_DYNSYS_FUNC
endif

ifeq ($(PROFILE_ON), Y)
ifeq ($(GPU_ON), N)
	CFLAGS += -pg
	LDFLAGS += -pg 
endif
endif
ifeq ($(OPTIMIZE), 0)
	CFLAGS += -O0
endif
ifeq ($(OPTIMIZE), 1)
	CFLAGS += -O1
endif
ifeq ($(OPTIMIZE), 2)
	CFLAGS += -O2
endif
ifeq ($(OPTIMIZE), 3)
	CFLAGS += -O3
endif
ifeq ($(WARNING_ON), Y)
	CPPFLAGS += -Wall -W
endif
ifeq ($(FILESYSTEM_ON), Y)
	CFLAGS	+= -DFILESYSTEM
	LDFLAGS	+= -lstdc++fs
endif
ifeq ($(VISUALIZE_ON), Y)
ifeq ($(GUI_ON), Y)
	CFLAGS	+= -DVISUALIZE_WITH_GUI
endif
	CFLAGS	+= -DVISUALIZE_BY_OPENCV
	CFLAGS 	+= $(shell pkg-config --cflags opencv)
	LDFLAGS += $(shell pkg-config --libs opencv)
endif
ifeq ($(OPENMP_ON), Y)
ifeq ($(UNAME),Darwin)
	CFLAGS += -fopenmp
	CFLAGS	+= -I$(HOSTCC_DIR)/include
	LDFLAGS += -L$(HOSTCC_DIR)/lib
	LDFLAGS += -liomp5
	LLVM_LIBRARY_PATH	= $(HOSTCC_DIR)/lib
else
	CFLAGS += -fopenmp
	LDFLAGS += -lgomp
endif
endif
LDFLAGS	+= -lpthread

ALL_CFLAGS :=
ALL_LDFLAGS :=
ifeq ($(GPU_ON), Y)
	ALL_CFLAGS +=	$(NVCCFLAGS)
	ALL_CFLAGS += $(EXTRA_NVCCFLAGS)
	ALL_CFLAGS +=	$(addprefix -Xcompiler ,$(CFLAGS))
	ALL_CPPFLAGS += $(addprefix -Xcompiler ,$(CPPFLAGS))
	ALL_CUFLAGS += $(CUFLAGS)
	ALL_LDFLAGS += $(NVLDFLAGS)
	ALL_LDFLAGS += $(EXTRA_NVLDFLAGS)
	ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
else
	ALL_CFLAGS +=  $(CFLAGS)
	ALL_CPPFLAGS +=  $(CPPFLAGS)
	ALL_LDFLAGS += $(LDFLAGS)
endif

vpath %.cpp .:BoundaryCondition:Core:ExplicitIntegration:ExplicitIntegration/Dissipations:ExplicitIntegration/Integrators:ExplicitIntegration/Terms:Grids:InitialConditions:InitialConditions/BasicShapes:SpatialDerivative:SpatialDerivative/UpwindFirst
vpath %.cu  .:BoundaryCondition:Core:ExplicitIntegration:ExplicitIntegration/Dissipations:ExplicitIntegration/Integrators:ExplicitIntegration/Terms:Grids:InitialConditions:InitialConditions/BasicShapes:SpatialDerivative:SpatialDerivative/UpwindFirst

LIBS     = 
INCLUDE  = -I$(INSTALL_DIR)/includes
TARGET   = liblevelset.so
STATIC_TARGET = liblevelset.a
OBJDIR   = ./obj
ifeq "$(strip $(OBJDIR))" ""
  OBJDIR = .
endif

SOURCES  = AddGhostExtrapolate.cpp \
	AddGhostPeriodic.cpp \
	ArtificialDissipationGLF.cpp \
	interpn.cpp \
	CacheTag.cpp \
	UVec.cpp \
	UVec_cuda_dummy.cpp \
	SchemeData.cpp \
	odeCFL_SubStep.cpp \
	OdeCFL_CommandQueue.cpp \
	OdeCFL_OneSlice.cpp \
	OdeCFL_Worker.cpp \
	OdeCFL1.cpp \
	OdeCFL2.cpp \
	OdeCFL3.cpp \
	TermLaxFriedrichs.cpp \
	TermRestrictUpdate.cpp \
	HJI_Grid.cpp \
	ShapeCylinder.cpp \
	ShapeHyperplaneByPoint.cpp \
	ShapeRectangleByCenter.cpp \
	ShapeRectangleByCorner.cpp \
	ShapeSphere.cpp \
	UpwindFirst.cpp \
	UpwindFirstFirst.cpp \
	UpwindFirstENO2.cpp \
	UpwindFirstENO3.cpp \
	UpwindFirstENO3a.cpp \
	UpwindFirstENO3b.cpp \
	UpwindFirstENO3aHelper.cpp \
	UpwindFirstENO3bHelper.cpp \
	UpwindFirstWENO5.cpp \
	UpwindFirstWENO5a.cpp \
	UpwindFirstWENO5b.cpp \
	ArtificialDissipationGLF_cuda_dummy.cpp \
	TermLaxFriedrichs_cuda_dummy.cpp \
	UpwindFirstENO3aHelper_cuda_dummy.cpp \
	UpwindFirstWENO5a_cuda_dummy.cpp \


CUDA_SOURCES = 	UVec_cuda.cu \
	ArtificialDissipationGLF_cuda.cu \
	TermLaxFriedrichs_cuda.cu \
	UpwindFirstENO3aHelper_cuda.cu \
	UpwindFirstWENO5a_cuda.cu

OBJECTS  = $(SOURCES:.cpp=.o)

ifeq ($(GPU_ON), Y)
	OBJECTS  += $(CUDA_SOURCES:.cu=.o)
endif

DEPENDS  = $(OBJECTS:.o=.d)

$(TARGET): $(OBJECTS) $(LIBS)
	$(LINKER) -shared -o $@ $^ $(ALL_LDFLAGS)

$(STATIC_TARGET): $(OBJECTS) $(LIBS)
	$(AR) rcs $@ $^

%.o: %.cpp
	$(CPP) $(ALL_CFLAGS) $(ALL_CPPFLAGS) $(INCLUDE) -o $@ -c $<

%.o: %.cu
	$(CPP) $(ALL_CFLAGS) $(ALL_CUFLAGS) $(INCLUDE) -o $@ -c $<

all: $(TARGET) $(STATIC_TARGET) install

install: $(TARGET) $(STATIC_TARGET)
	cp -rp $^ $(INSTALL_DIR)/$(ARCH)/lib

clean:
	rm -rf $(OBJDIR)
	rm -f $(OBJECTS)
	rm -f dump_theta.csv
	rm -f gmon.out
	rm -f $(TARGET)
	rm -f $(STATIC_TARGET)

-include $(DEPENDS)
