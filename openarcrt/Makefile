include ../make.header

# C compiler
COMMONCFLAGS = $(GMACROS) $(GBACKEND_FLAGS)
COMMONCLIBS = $(GBACKEND_LIBS)
COMMONCINCLUDE = $(GBACKEND_INCLUDES)
ifeq ($(OPENARC_ARCH),0)
# CUDA target
OPENARCLIB_SUFFIX = cuda
else
# OpenCL target
OPENARCLIB_SUFFIX = opencl
endif

# Archive option
AR = ar
ARFLAGS = rs

TARGET = .

ifeq ($(OMP),1)
CLIBS = $(COMMONCLIBS) $(GOMP_LIBS)
ifneq ($(GOMP_FLAGS),"")
OMP_FLAGS = $(CC_FLAGS_PREFIX) $(GOMP_FLAGS)
endif
OUTPUT_BASE = libopenaccrtomp_$(OPENARCLIB_SUFFIX)
else
CLIBS = $(COMMONCLIBS)
OMP_FLAGS =
OUTPUT_BASE = libopenaccrt_$(OPENARCLIB_SUFFIX)
endif

MODE ?= normal
ifeq ($(MODE),profile)
CC_FLAGS = $(GBACKEND_DEBUG) $(GPROFILEMACROS) $(OMP_FLAGS)
BENCHMARK = $(TARGET)/$(OUTPUT_BASE)pf.a
else
CC_FLAGS = $(OMP_FLAGS)
BENCHMARK = $(TARGET)/$(OUTPUT_BASE).a
endif

RESLIB = $(TARGET)/libresilience.a

OMPHELPLIB = $(TARGET)/libomphelper.a

MCLEXTLIB = $(TARGET)/libmclext.a

PRINT_LOG ?= 0
OMP ?= 0
DEFSET ?= -DPRINT_LOG=$(PRINT_LOG) -DOMP=$(OMP)

ifeq ($(OPENARC_ARCH),0)
OBJS = openacc.o openaccrt.o cudadriver.o
else
OBJS = openacc.o openaccrt.o opencldriver.o
endif

all: $(OBJS)
	$(AR) $(ARFLAGS) $(BENCHMARK) $(OBJS)

res: resilience.o
	$(AR) $(ARFLAGS) $(RESLIB) resilience.o

omphelp: omp_helper.o
	$(AR) $(ARFLAGS) $(OMPHELPLIB) omp_helper.o

mcl: mcl_accext.o
	$(AR) $(ARFLAGS) $(MCLEXTLIB) mcl_accext.o

openacc.o: openacc.cpp openacc.h openaccrt.h
	$(BACKENDCXX) $(DEFSET) $(COMMONCINCLUDE) $(COMMONCFLAGS) $(CC_FLAGS) -o openacc.o openacc.cpp -c

openaccrt.o: openaccrt.cpp openacc.h openaccrt.h resilience.cpp omp_helper.cpp
	$(BACKENDCXX) $(DEFSET) $(COMMONCINCLUDE) $(COMMONCFLAGS) $(CC_FLAGS) -o openaccrt.o openaccrt.cpp -c

opencldriver.o: opencldriver.cpp openacc.h openaccrt.h resilience.cpp omp_helper.cpp
	$(BACKENDCXX) $(DEFSET) $(COMMONCINCLUDE) $(COMMONCFLAGS) $(CC_FLAGS) -o opencldriver.o opencldriver.cpp -c

cudadriver.o: cudadriver.cpp openacc.h openaccrt.h resilience.cpp omp_helper.cpp
	$(BACKENDCXX) $(DEFSET) $(COMMONCINCLUDE) $(COMMONCFLAGS) $(CC_FLAGS) -o cudadriver.o cudadriver.cpp -c

resilience.o: resilience.cpp
	$(CXX) $(DEFSET) -o resilience.o resilience.cpp -c

omp_helper.o: omp_helper.cpp
	$(CXX) $(DEFSET) -o omp_helper.o omp_helper.cpp -c

mcl_accext.o: mcl_accext.cpp
	$(CXX) $(DEFSET) $(COMMONCINCLUDE) -I${mclinstallroot}/include -o mcl_accext.o mcl_accext.cpp -c
	
binUtil: binBuilder.cpp
	$(BACKENDCXX) $(DEFSET) $(COMMONCINCLUDE) $(COMMONCFLAGS) $(CC_FLAGS) -o ./binBuilder_$(OPENARCLIB_SUFFIX) binBuilder.cpp $(CLIBS)

clean:
	rm -f *.o *~

purge:
	rm -f *.o *~ libopenaccrt_$(OPENARCLIB_SUFFIX).a libopenaccrtomp_$(OPENARCLIB_SUFFIX).a libopenaccrt_$(OPENARCLIB_SUFFIX)pf.a libopenaccrtomp_$(OPENARCLIB_SUFFIX)pf.a binBuilder_$(OPENARCLIB_SUFFIX) libresilience.a libomphelper.a Timer
