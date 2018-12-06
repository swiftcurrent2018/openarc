include ../../../../../make.header

########################
# Set the program name #
########################
BENCHMARK = bfs

########################################
# Set the input C source files (CSRCS) #
########################################
CSRCS = bfs.c

#########################################
# Set macros used for the input program #
#########################################
VERIFICATION ?= 1
R_MODE ?= 0
# Input Data: graph4096.txt
NUM_OF_NODES ?= 4096
EDGELIST_SIZE ?= 24576
# Input Data: graph1M.txt
#NUM_OF_NODES ?= 1000000
#EDGELIST_SIZE ?= 5999970
# Input Data: graph4M.txt
#NUM_OF_NODES ?= 4194304
#EDGELIST_SIZE ?= 25159848
# Input Data: graph16M.txt
#NUM_OF_NODES ?= 16777216
#EDGELIST_SIZE ?= 100666228

DEFSET_CPU = -DVERIFICATION=$(VERIFICATION) -DNUM_OF_NODES=$(NUM_OF_NODES) -DEDGELIST_SIZE=$(EDGELIST_SIZE)
DEFSET_LLVM = -DVERIFICATION=$(VERIFICATION) -DNUM_OF_NODES=$(NUM_OF_NODES) -DEDGELIST_SIZE=$(EDGELIST_SIZE) -DENABLE_OPENACC=0 -Warc,-macro=NUM_OF_NODES=$(NUM_OF_NODES),EDGELIST_SIZE=$(EDGELIST_SIZE) $(ARCMACRO)
DEFSET_ACC = -DVERIFICATION=$(VERIFICATION) -DNUM_OF_NODES=$(NUM_OF_NODES) -DEDGELIST_SIZE=$(EDGELIST_SIZE)

##########################################################
# Makefile options that the user can overwrite           #
# OMP: set to 1 to use OpenMP (default: 0)               # 
# MODE: set to profile to use a built-in profiling tool  #
#       (default: normal)                                #
#       If this is set to profile, the runtime system    #
#       will print profiling results according to the    #
#       verbosity level set by OPENARCRT_VERBOSITY       #
#       environment variable.                            # 
# AOCL_FLAGS: set Altera OpenCL Compiler (AOC) flags     #
#    - commonly used options                             #
#      -march=emulator //compile a kernel for emulation  #
#      -v //show progress of the compilation on-screen   # 
#      -c //compile the kernel and generate a Quartus II #
#         //hardware design project without creating a   #
#         //hardware configuration file.                 #
#      -profile //instrument the OpenCL kernel pipeline  #
#                //with performance counters.            #
#      -report  //display estimated resource usage on    #
#                //the screen.                           #
#    (default: -march=emulator)                          #
# AOCL_BOARD: set a target Altera FPGA board             #
#    - "-board=$(AOCL_BOARD)" will be added to the AOC   #
#     in addition to the above flags                     # 
#    - Examples                                          #
#    p385_hpc_d5 //for Stratix V                         #
#    p510t_sch_ax115 //for Arria 10 (Nallatech 510T)     #
##########################################################
OMP ?= 0
MODE ?= normal
#AOCL_BOARD ?= p385_hpc_d5
AOCL_BOARD ?= p510t_sch_ax115
#AOCL_FLAGS ?= -march=emulator
#AOCL_FLAGS ?= -march=emulator -c
#AOCL_FLAGS ?= -v -g -c -report
AOCL_FLAGS ?= -v -c -report
#AOCL_FLAGS ?= -v -report

#########################################################
# Use the following macros to give program-specific     #
# compiler flags and libraries                          #
# - CFLAGS1 and CLIBS1 to compile the input C program   #
# - CFLAGS2 and CLIBS2 to compile the OpenARC-generated #
#   output C++ program                                  # 
#########################################################
#CFLAGS1 =  
#CFLAGS2 =  
CLIBS1 = -lm
CLIBS1_LLVM = -lm
#CLIBS2 = 

################################################
# TARGET is where the output binary is stored. #
################################################
#TARGET ?= ./bin

include ../../../../../make.template
