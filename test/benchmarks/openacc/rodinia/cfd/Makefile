include ../../../../../make.header

########################
# Set the program name #
########################
BENCHMARK = cfd

########################################
# Set the input C source files (CSRCS) #
########################################
CSRCS = euler3d_cpu.c

#########################################
# Set macros used for the input program #
#########################################
VERIFICATION ?= 0
R_MODE ?= 0
###########################
# Input: fvcorr.domn.097K #
###########################
NEL ?= 97046
BLOCK_LENGTH ?= 256
NELR ?= 97280
#BLOCK_LENGTH ?= 128
#NELR ?= 97152
#BLOCK_LENGTH ?= 64
#NELR ?= 97088
#BLOCK_LENGTH ?= 32
#NELR ?= 97056
###########################
# Input: fvcorr.domn.193K #
###########################
#NEL ?= 193474
#BLOCK_LENGTH ?= 256
#NELR ?= 193536
#BLOCK_LENGTH ?= 128 
#NELR ?= 193536
#BLOCK_LENGTH ?= 64
#NELR ?= 193536
#BLOCK_LENGTH ?= 32
#NELR ?= 193504
###########################
# Input: missile.domn.0.2M #
###########################
#NEL ?= 232536
#BLOCK_LENGTH ?= 128 
#NELR ?= 232544
#BLOCK_LENGTH ?= 64
#NELR ?= 232576
#BLOCK_LENGTH ?= 32
#NELR ?= 232576


DEFSET_CPU = -DVERIFICATION=$(VERIFICATION) -DNEL=$(NEL) -Dblock_length=$(BLOCK_LENGTH) -DNELR=$(NELR) -DENABLE_OPENACC=0
DEFSET_LLVM = -DVERIFICATION=$(VERIFICATION) -DNEL=$(NEL) -Dblock_length=$(BLOCK_LENGTH) -DNELR=$(NELR) -DENABLE_OPENACC=0 -Warc,-macro=NEL=$(NEL),block_length=$(BLOCK_LENGTH),NELR=$(NELR) $(ARCMACRO)
DEFSET_ACC = -DVERIFICATION=$(VERIFICATION) -DNEL=$(NEL) -Dblock_length=$(BLOCK_LENGTH) -DNELR=$(NELR) -DENABLE_OPENACC=1

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
