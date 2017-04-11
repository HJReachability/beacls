GPU_ON = N
VISUALIZE_ON = Y
GUI_ON = Y
FILESYSTEM_ON = N
STATIC_ON = N
PROFILE_ON = N
FMA_ON = Y
USER_DEFINED_GPU_DYNSYS_FUNC_ON = Y
OPTIMIZE = 3
WARNING_ON = Y
SINGLE_PRECISION_ON = Y
FILE_DUMP = 1
CHUNK_SIZE = 0
USE_CUDA = 0
NUM_OF_THREADS = 0
NUM_OF_GPUS = 0
USE_LAST_DERIV_MIN_MAX = 0
USE_USER_DEFINED_GPU_DYNSYS_FUNC = 1
NVCC = /usr/bin/nvcc
ARCH = $(shell uname -m)
PWD = $(shell pwd)
INSTALL_DIR = $(PWD)/../../../builds
MATIO_DIR = $(INSTALL_DIR)
MATIO_LIB_DIR = $(MATIO_DIR)/$(ARCH)/lib
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


BEACLS_DIR = $(INSTALL_DIR)
BEACLS_INC_DIR = $(BEACLS_DIR)/include
BEACLS_LIB_DIR = $(BEACLS_DIR)/$(ARCH)/lib
MODULE_DIR = $(MATIO_LIB_DIR):$(BEACLS_LIB_DIR)

ifeq ($(UNAME),Darwin)
	TIME = /usr/bin/time
endif
ifeq ($(UNAME),Linux)
	TIME = /usr/bin/time -v
endif

ifeq ($(UNAME),Linux)
	GOOGLE_PERF_LIB = libprofiler.so.0
	CPUPROFILE_FREQUENCY = 100
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
else
	CPP = $(HOSTCC)
	LINKER = $(HOSTCC)
endif

CFLAGS   += -std=c++14 -mfpmath=sse -msse2 -g -march=native

ifeq ($(FMA_ON), N)
	CFLAGS	+= -mno-fma
endif
ifeq ($(UNAME),Darwin)
	CFLAGS += -fno-common
	CFLAGS	+= -I/usr/local/include
	LDFLAGS += -L/usr/local/lib -lstdc++
endif

LDFLAGS  += -g

ifeq ($(STATIC_ON), Y)
	LDFLAGS	+= $(BEACLS_LIB_DIR)/libhelperOC.a
	LDFLAGS	+= $(BEACLS_LIB_DIR)/liblevelset.a
	LIBS	+= $(BEACLS_LIB_DIR)/libhelperOC.a
	LIBS	+= $(BEACLS_LIB_DIR)/liblevelset.a
else
	LDFLAGS	+= -L$(BEACLS_LIB_DIR) -lhelperOC -llevelset
	LIBS	+= $(BEACLS_LIB_DIR)/libhelperOC.so
	LIBS	+= $(BEACLS_LIB_DIR)/liblevelset.so
endif

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
	OPENCV_LIB_DIR	+= $(shell pkg-config --libs opencv | sed s/"-lopencv_.*"/""/g | sed s/"libopencv.*"/""/g )
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
	ALL_CFLAGS +=  $(NVCCFLAGS)
	ALL_CFLAGS +=  $(addprefix -Xcompiler ,$(CFLAGS))
	ALL_LDFLAGS += $(NVLDFLAGS)
	ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
else
	ALL_CFLAGS +=  $(CFLAGS)
	ALL_LDFLAGS += $(LDFLAGS)
endif


vpath %.cpp .

INCLUDE  = -I$(INSTALL_DIR)/includes
TARGET   = air3D
OBJDIR   = ./obj
ifeq "$(strip $(OBJDIR))" ""
  OBJDIR = .
endif

SOURCES  = $(TARGET:=.cpp) Air3DSchemeData.cpp
OBJECTS  = $(SOURCES:.cpp=.o)

ifeq ($(UNAME),Darwin)
	TEST_ENV_NAME = DYLD_LIBRARY_PATH
endif
ifeq ($(UNAME),Linux)
	TEST_ENV_NAME = LD_LIBRARY_PATH
endif

ifeq ($(GPU_ON), Y)
	CUDA_SOURCES  = $(wildcard *.cu)
	OBJECTS  += $(CUDA_SOURCES:.cu=.o)
endif

DEPENDS  = $(OBJECTS:.o=.d)

GOOGLE-PROFILE-LOG = $(TARGET).prof

$(TARGET): $(OBJECTS) $(LIBS)
	$(LINKER) -o $@ $(OBJECTS) $(ALL_LDFLAGS)

%.o: %.cpp
	$(CPP) $(ALL_CFLAGS) $(INCLUDE) -o $@ -c $<

%.o: %.cu
	$(CPP) $(ALL_CFLAGS) $(INCLUDE) -o $@ -c $<

all: $(TARGET)

test: $(TARGET)
	env $(TEST_ENV_NAME)=$(LLVM_LIBRARY_PATH):$(OPENCV_LIB_DIR):$(MODULE_DIR):$($(TEST_ENV_NAME)) ./$(TARGET) $(FILE_DUMP) $(CHUNK_SIZE) $(USE_CUDA) $(NUM_OF_THREADS) $(NUM_OF_GPUS) $(USE_LAST_DERIV_MIN_MAX) $(USE_USER_DEFINED_GPU_DYNSYS_FUNC)

google-prof: $(GOOGLE-PROFILE-LOG)

$(GOOGLE-PROFILE-LOG): $(TARGET)
	env LD_PRELOAD=$(GOOGLE_PERF_LIB):$(LD_PRELOAD) CPUPROFILE=./$(GOOGLE-PROFILE-LOG) CPUPROFILE_FREQUENCY=$(CPUPROFILE_FREQUENCY) $(TEST_ENV_NAME)=$(LLVM_LIBRARY_PATH):$(OPENCV_LIB_DIR):$(MODULE_DIR):$($(TEST_ENV_NAME)) ./$(TARGET) $(FILE_DUMP) $(CHUNK_SIZE) $(USE_CUDA) $(NUM_OF_THREADS) $(NUM_OF_GPUS) $(USE_LAST_DERIV_MIN_MAX) $(USE_USER_DEFINED_GPU_DYNSYS_FUNC)

show-google-prof: $(TARGET) $(GOOGLE-PROFILE-LOG)
	google-pprof $(TARGET) $(GOOGLE-PROFILE-LOG) --text

kcachegrind-google-prof: $(GOOGLE-PROFILE-LOG)	$(TARGET)
	google-pprof $(TARGET) --callgrind $(GOOGLE-PROFILE-LOG) > $(GOOGLE-PROFILE-LOG).cg
	kcachegrind $(GOOGLE-PROFILE-LOG).cg &

time: $(TARGET)
	$(TIME) env $(TEST_ENV_NAME)=$(LLVM_LIBRARY_PATH):$(OPENCV_LIB_DIR):$(MODULE_DIR):$($(TEST_ENV_NAME)) ./$(TARGET) $(FILE_DUMP) $(CHUNK_SIZE) $(USE_CUDA) $(NUM_OF_THREADS) $(NUM_OF_GPUS) $(USE_LAST_DERIV_MIN_MAX) $(USE_USER_DEFINED_GPU_DYNSYS_FUNC)

time_chunk_sizes: $(TARGET)
	$(TIME) env $(TEST_ENV_NAME)=$(LLVM_LIBRARY_PATH):$(OPENCV_LIB_DIR):$(MODULE_DIR):$($(TEST_ENV_NAME)) ./$(TARGET) $(FILE_DUMP) 1 $(USE_CUDA) $(NUM_OF_THREADS) $(NUM_OF_GPUS) $(USE_LAST_DERIV_MIN_MAX) $(USE_USER_DEFINED_GPU_DYNSYS_FUNC)
	$(TIME) env $(TEST_ENV_NAME)=$(LLVM_LIBRARY_PATH):$(OPENCV_LIB_DIR):$(MODULE_DIR):$($(TEST_ENV_NAME)) ./$(TARGET) $(FILE_DUMP) 2 $(USE_CUDA) $(NUM_OF_THREADS) $(NUM_OF_GPUS) $(USE_LAST_DERIV_MIN_MAX) $(USE_USER_DEFINED_GPU_DYNSYS_FUNC)
	$(TIME) env $(TEST_ENV_NAME)=$(LLVM_LIBRARY_PATH):$(OPENCV_LIB_DIR):$(MODULE_DIR):$($(TEST_ENV_NAME)) ./$(TARGET) $(FILE_DUMP) 4 $(USE_CUDA) $(NUM_OF_THREADS) $(NUM_OF_GPUS) $(USE_LAST_DERIV_MIN_MAX) $(USE_USER_DEFINED_GPU_DYNSYS_FUNC)
	$(TIME) env $(TEST_ENV_NAME)=$(LLVM_LIBRARY_PATH):$(OPENCV_LIB_DIR):$(MODULE_DIR):$($(TEST_ENV_NAME)) ./$(TARGET) $(FILE_DUMP) 8 $(USE_CUDA) $(NUM_OF_THREADS) $(NUM_OF_GPUS) $(USE_LAST_DERIV_MIN_MAX) $(USE_USER_DEFINED_GPU_DYNSYS_FUNC)
	$(TIME) env $(TEST_ENV_NAME)=$(LLVM_LIBRARY_PATH):$(OPENCV_LIB_DIR):$(MODULE_DIR):$($(TEST_ENV_NAME)) ./$(TARGET) $(FILE_DUMP) 16 $(USE_CUDA) $(NUM_OF_THREADS) $(NUM_OF_GPUS) $(USE_LAST_DERIV_MIN_MAX) $(USE_USER_DEFINED_GPU_DYNSYS_FUNC)
	$(TIME) env $(TEST_ENV_NAME)=$(LLVM_LIBRARY_PATH):$(OPENCV_LIB_DIR):$(MODULE_DIR):$($(TEST_ENV_NAME)) ./$(TARGET) $(FILE_DUMP) 32 $(USE_CUDA) $(NUM_OF_THREADS) $(NUM_OF_GPUS) $(USE_LAST_DERIV_MIN_MAX) $(USE_USER_DEFINED_GPU_DYNSYS_FUNC)
	$(TIME) env $(TEST_ENV_NAME)=$(LLVM_LIBRARY_PATH):$(OPENCV_LIB_DIR):$(MODULE_DIR):$($(TEST_ENV_NAME)) ./$(TARGET) $(FILE_DUMP) 64 $(USE_CUDA) $(NUM_OF_THREADS) $(NUM_OF_GPUS) $(USE_LAST_DERIV_MIN_MAX) $(USE_USER_DEFINED_GPU_DYNSYS_FUNC)

iprofiler: $(TARGET)
	cp -rp $(OPENCV_LIB_DIR):$(MODULE_DIR)/liblevelset.so .
	iprofiler -T 0 -$(TIME)profiler ./$(TARGET) $(FILE_DUMP) $(CHUNK_SIZE) $(USE_CUDA) $(NUM_OF_THREADS) $(NUM_OF_GPUS) $(USE_LAST_DERIV_MIN_MAX) $(USE_USER_DEFINED_GPU_DYNSYS_FUNC)
#	rm -rf liblevelset.so

valgrind: $(TARGET)
	env $(TEST_ENV_NAME)=$(LLVM_LIBRARY_PATH):$(OPENCV_LIB_DIR):$(MODULE_DIR):$($(TEST_ENV_NAME)) valgrind --leak-check=full --show-reachable=yes --track-origins=yes --error-limit=no ./$(TARGET) $(FILE_DUMP) $(CHUNK_SIZE) $(USE_CUDA) $(NUM_OF_THREADS) $(NUM_OF_GPUS) $(USE_LAST_DERIV_MIN_MAX) $(USE_USER_DEFINED_GPU_DYNSYS_FUNC)


massif: $(TARGET)
	env $(TEST_ENV_NAME)=$(LLVM_LIBRARY_PATH):$(OPENCV_LIB_DIR):$(MODULE_DIR):$($(TEST_ENV_NAME)) valgrind --tool=massif ./$(TARGET) $(FILE_DUMP) $(CHUNK_SIZE) $(USE_CUDA) $(NUM_OF_THREADS) $(NUM_OF_GPUS) $(USE_LAST_DERIV_MIN_MAX) $(USE_USER_DEFINED_GPU_DYNSYS_FUNC)

massif_chunk_sizes: $(TARGET)
	env $(TEST_ENV_NAME)=$(LLVM_LIBRARY_PATH):$(OPENCV_LIB_DIR):$(MODULE_DIR):$($(TEST_ENV_NAME)) valgrind --tool=massif ./$(TARGET) $(FILE_DUMP) 1 $(USE_CUDA) $(NUM_OF_THREADS) $(NUM_OF_GPUS) $(USE_LAST_DERIV_MIN_MAX) $(USE_USER_DEFINED_GPU_DYNSYS_FUNC)
	env $(TEST_ENV_NAME)=$(LLVM_LIBRARY_PATH):$(OPENCV_LIB_DIR):$(MODULE_DIR):$($(TEST_ENV_NAME)) valgrind --tool=massif ./$(TARGET) $(FILE_DUMP) 2 $(USE_CUDA) $(NUM_OF_THREADS) $(NUM_OF_GPUS) $(USE_LAST_DERIV_MIN_MAX) $(USE_USER_DEFINED_GPU_DYNSYS_FUNC)
	env $(TEST_ENV_NAME)=$(LLVM_LIBRARY_PATH):$(OPENCV_LIB_DIR):$(MODULE_DIR):$($(TEST_ENV_NAME)) valgrind --tool=massif ./$(TARGET) $(FILE_DUMP) 4 $(USE_CUDA) $(NUM_OF_THREADS) $(NUM_OF_GPUS) $(USE_LAST_DERIV_MIN_MAX) $(USE_USER_DEFINED_GPU_DYNSYS_FUNC)
	env $(TEST_ENV_NAME)=$(LLVM_LIBRARY_PATH):$(OPENCV_LIB_DIR):$(MODULE_DIR):$($(TEST_ENV_NAME)) valgrind --tool=massif ./$(TARGET) $(FILE_DUMP) 8 $(USE_CUDA) $(NUM_OF_THREADS) $(NUM_OF_GPUS) $(USE_LAST_DERIV_MIN_MAX) $(USE_USER_DEFINED_GPU_DYNSYS_FUNC)
	env $(TEST_ENV_NAME)=$(LLVM_LIBRARY_PATH):$(OPENCV_LIB_DIR):$(MODULE_DIR):$($(TEST_ENV_NAME)) valgrind --tool=massif ./$(TARGET) $(FILE_DUMP) 16 $(USE_CUDA) $(NUM_OF_THREADS) $(NUM_OF_GPUS) $(USE_LAST_DERIV_MIN_MAX) $(USE_USER_DEFINED_GPU_DYNSYS_FUNC)
	env $(TEST_ENV_NAME)=$(LLVM_LIBRARY_PATH):$(OPENCV_LIB_DIR):$(MODULE_DIR):$($(TEST_ENV_NAME)) valgrind --tool=massif ./$(TARGET) $(FILE_DUMP) 32 $(USE_CUDA) $(NUM_OF_THREADS) $(NUM_OF_GPUS) $(USE_LAST_DERIV_MIN_MAX) $(USE_USER_DEFINED_GPU_DYNSYS_FUNC)
	env $(TEST_ENV_NAME)=$(LLVM_LIBRARY_PATH):$(OPENCV_LIB_DIR):$(MODULE_DIR):$($(TEST_ENV_NAME)) valgrind --tool=massif ./$(TARGET) $(FILE_DUMP) 64 $(USE_CUDA) $(NUM_OF_THREADS) $(NUM_OF_GPUS) $(USE_LAST_DERIV_MIN_MAX) $(USE_USER_DEFINED_GPU_DYNSYS_FUNC)

helgrind: $(TARGET)
	env $(TEST_ENV_NAME)=$(LLVM_LIBRARY_PATH):$(OPENCV_LIB_DIR):$(MODULE_DIR):$($(TEST_ENV_NAME)) valgrind --tool=helgrind --histroy-level=approx ./$(TARGET) $(FILE_DUMP) $(CHUNK_SIZE) $(USE_CUDA) $(NUM_OF_THREADS) $(NUM_OF_GPUS) $(USE_LAST_DERIV_MIN_MAX) $(USE_USER_DEFINED_GPU_DYNSYS_FUNC)


check:
	python checker.py ref_all_loop.csv   all_loop.csv
	python checker.py ref_short_loop.csv short_loop.csv
	python checker.py ref_mid_loop.csv   mid_loop.csv

clean:
	rm -rf $(OBJDIR)
	rm -f $(OBJECTS)
	rm -f gmon.out
	rm -f $(TARGET)
	rm -f *.mat
	rm -rf $(GOOGLE-PROFILE-LOG) $(GOOGLE-PROFILE-LOG).cg

cleanlog:
	rm -f gmon.out
	rm -f *.mat
	rm -rf $(GOOGLE-PROFILE-LOG) $(GOOGLE-PROFILE-LOG).cg

-include $(DEPENDS)
