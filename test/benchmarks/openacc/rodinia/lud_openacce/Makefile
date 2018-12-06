include ../../../../../make.header

########################
# Set the program name #
########################
BENCHMARK = lud

########################################
# Set the input C source files (CSRCS) #
########################################
CSRCS = lud.c lud_omp.c common.c

#########################################
# Set macros used for the input program #
#########################################
VERIFICATION ?= 1
R_MODE ?= 0
# matrix_dim = 64 #
_M_SIZE ?= 4096
# matrix_dim = 256 #
#_M_SIZE ?= 65536
# matrix_dim = 512 #
#_M_SIZE ?= 262144
# matrix_dim = 2048 #
#_M_SIZE ?= 4194304


DEFSET_CPU = -DVERIFICATION=$(VERIFICATION) -D_M_SIZE=$(_M_SIZE) -DENABLE_OPENACC=0
DEFSET_LLVM = -DVERIFICATION=$(VERIFICATION) -D_M_SIZE=$(_M_SIZE) -DENABLE_OPENACC=0 -Warc,-macro=_M_SIZE=$(_M_SIZE) $(ARCMACRO)
DEFSET_ACC = -DVERIFICATION=$(VERIFICATION) -D_M_SIZE=$(_M_SIZE) -DENABLE_OPENACC=0

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
#CLIBS1 = 
#CLIBS2 = 

################################################
# TARGET is where the output binary is stored. #
################################################
#TARGET ?= ./bin

include ../../../../../make.template
