include ../../../../../make.header

########################
# Set the program name #
########################
BENCHMARK = hotspot

########################################
# Set the input C source files (CSRCS) #
########################################
CSRCS = hotspot.c

#########################################
# Set macros used for the input program #
#########################################
VERIFICATION ?= 1
R_MODE ?= 0
# For temp_64 and  power_64 #
ROW_SIZE ?= 64
COL_SIZE ?= 64
# For temp_512 and  power_512 #
#ROW_SIZE ?= 512
#COL_SIZE ?= 512
# For temp_1024 and  power_1024 #
#ROW_SIZE ?= 1024
#COL_SIZE ?= 1024
# For temp_4096 and  power_4096 #
#ROW_SIZE ?= 4096
#COL_SIZE ?= 4096


DEFSET_CPU = -DVERIFICATION=$(VERIFICATION) -DROW_SIZE=$(ROW_SIZE) -DCOL_SIZE=$(COL_SIZE) -DENABLE_OPENACC=0
DEFSET_LLVM = -DVERIFICATION=$(VERIFICATION) -DROW_SIZE=$(ROW_SIZE) -DCOL_SIZE=$(COL_SIZE) -DENABLE_OPENACC=0 -Warc,-macro=ROW_SIZE=$(ROW_SIZE),COL_SIZE=$(COL_SIZE) $(ARCMACRO)
DEFSET_ACC = -DVERIFICATION=$(VERIFICATION) -DROW_SIZE=$(ROW_SIZE) -DCOL_SIZE=$(COL_SIZE) -DENABLE_OPENACC=1

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
