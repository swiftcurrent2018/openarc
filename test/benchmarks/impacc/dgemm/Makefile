include ../../../../make.header

########################
# Set the program name #
########################
BENCHMARK = dgemm

########################################
# Set the input C source files (CSRCS) #
########################################
CSRCS = dgemm.c

#########################################
# Set macros used for the input program #
#########################################

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
#    (default: p385_hpc_a7)                              #
##########################################################
OMP ?= 1
MODE ?= normal

#########################################################
# Use the following macros to give program-specific     #
# compiler flags and libraries                          #
# - CFLAGS1 and CLIBS1 to compile the input C program   #
# - CFLAGS2 and CLIBS2 to compile the OpenARC-generated #
#   output C++ program                                  # 
#########################################################
#CFLAGS1 = 
#CFLAGS1_LLVM =  
CFLAGS2 = -I ../../../../../impacc/src $(MPI_FLAGS)
# For Translated CPU
CLIBS1 = -lm
# For LLVM
CLIBS1_LLVM = -lm
CLIBS2 = -limpacc -lcuda

################################################
# TARGET is where the output binary is stored. #
################################################
#TARGET ?= ./bin

include ../../../../make.template
