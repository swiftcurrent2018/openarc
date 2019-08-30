include ../../../../make.header

########################
# Set the program name #
########################
BENCHMARK = matmul

OMP ?= 1

OBJS_CU = matmul.o kernel.o
OBJS_ACC = acc_helper.o

COMMONCLIBS = $(GFRONTEND_LIBS)
COMMONCINCLUDE = -I$(OPENARCLIB) $(GBACKEND_INCLUDES) $(GFRONTEND_INCLUDES)

ifeq ($(OMP),1)
CLIBS = $(COMMONCLIBS) $(GOMP_LIBS)
ifneq ($(GOMP_FLAGS),"")
OMP_FLAGS = $(CC_FLAGS_PREFIX) $(GOMP_FLAGS)
endif
ACCRTLIB = openaccrtomp_cuda
else
CLIBS = $(COMMONCLIBS)
OMP_FLAGS =
ACCRTLIB = openaccrt_cuda
endif

MODE ?= normal
ifeq ($(MODE),profile)
CC_FLAGS = $(GBACKEND_DEBUG) $(GPROFILEMACROS) 
else
CC_FLAGS = 
endif


$(BENCHMARK)_CUDA: ${OBJS_CU} ${OBJS_ACC}
	mkdir -p bin
	$(BACKENDCXX) $(DEFSET) $(COMMONCINCLUDE) $(GMACROS) $(GBACKEND_FLAGS) $(CC_FLAGS) $(OMP_FLAGS) -o ./bin/$(BENCHMARK)_CUDA $(OBJS_CU) $(OBJS_ACC) $(CLIBS) -L$(OPENARCLIB) -l$(ACCRTLIB) -lomphelper
	cp -f openarc_kernel.* ./bin/

$(OBJS_ACC): acc_helper.cpp
	$(CXX) $(DEFSET) $(COMMONCINCLUDE) $(GMACROS) $(GFRONTEND_FLAGS) $(CC_FLAGS) -c -o $(OBJS_ACC) acc_helper.cpp $(CLIBS)

matmul.o: matmul.cu
	$(BACKENDCXX) $(DEFSET) $(COMMONCINCLUDE) $(GMACROS) $(GBACKEND_FLAGS) $(CC_FLAGS) $(OMP_FLAGS) -c -o matmul.o matmul.cu $(CLIBS)

kernel.o: kernel.cu
	$(BACKENDCXX) $(DEFSET) $(COMMONCINCLUDE) $(GMACROS) $(GBACKEND_FLAGS) $(CC_FLAGS) $(OMP_FLAGS) -c -o kernel.o kernel.cu $(CLIBS)

clean:
	rm -f *.o *.bc *~ 

purge: clean
	rm -rf $(BENCHMARK)_CUDA openarcConf.txt *.log cetus_output openarc_kernel* acc_helper.cpp bin
