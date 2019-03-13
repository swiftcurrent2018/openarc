include $(openarc)/make.header

########################
# Set the program name #
########################
BENCHMARK = srad

########################################
# Set the input C source files (CSRCS) #
########################################
CSRCS = *.c

#########################################
# Set macros used for the input program #
#########################################
VERIFICATION ?= 0
R_MODE ?= 0
#_ROWS_ ?= 256
#_COLS_ ?= 256
#_ROWS_ ?= 1024
#_COLS_ ?= 1024
#_ROWS_ ?= 2048
#_COLS_ ?= 2048
#_ROWS_ ?= 4096
#_COLS_ ?= 4096 


DEFSET_CPU = -DVERIFICATION=$(VERIFICATION) -D_ROWS_=$(_ROWS_) -D_COLS_=$(_COLS_) -DENABLE_OPENACC=0
DEFSET_LLVM = -DVERIFICATION=$(VERIFICATION) -D_ROWS_=$(_ROWS_) -D_COLS_=$(_COLS_) -DENABLE_OPENACC=0 -Warc,-macro=_ROWS_=$(_ROWS_),_COLS_=$(_COLS_) $(ARCMACRO)
DEFSET_ACC = -DVERIFICATION=$(VERIFICATION) -D_ROWS_=$(_ROWS_) -D_COLS_=$(_COLS_) -DENABLE_OPENACC=1

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
#      --profile //instrument the OpenCL kernel pipeline #
#                //with performance counters.            #
#      --report  //display estimated resource usage on   #
#                //the screen.                           #
#    (default: -march=emulator)                          #
# AOCL_BOARD: set a target Altera FPGA board             #
#    - "--board $(AOCL_BOARD)" will be added to the AOC  #
#     in addition to the above flags                     # 
#    - Examples                                          #
#    p385_hpc_d5 //for Stratix V (default)               #
#    p510t_sch_ax115 //for Arria 10 (Nallatech 510T)     #
##########################################################
OMP ?= 0
MODE ?= normal
#AOCL_BOARD ?= p385_hpc_d5
AOCL_BOARD ?= p510t_sch_ax115

AOCL_FLAGS ?= -v -report -march=emulator
#AOCL_FLAGS ?= -v -c -g -report
#AOCL_FLAGS ?= -v -report

#########################################################
# Use the following macros to give program-specific     #
# compiler flags and libraries                          #
# - CFLAGS1 and CLIBS1 to compile the input C program   #
# - CFLAGS2 and CLIBS2 to compile the OpenARC-generated #
#   output C++ program                                  # 
#########################################################
CFLAGS1 = -I ./inc 
CFLAGS2 = -I ../inc
CLIBS1 =  -lm
CLIBS1_LLVM =  -lm
#CLIBS2 = 

################################################
# TARGET is where the output binary is stored. #
################################################
#TARGET ?= ./bin

include $(openarc)/make.template
