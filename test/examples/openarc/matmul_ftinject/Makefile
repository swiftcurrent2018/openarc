include ../../../../make.header

########################
# Set the program name #
########################
BENCHMARK = matmul

########################################
# Set the input C source files (CSRCS) #
########################################
CSRCS = matmul.c

#########################################
# Set macros used for the input program #
#########################################
_N_ ?= 512
DEFSET_CPU = -D_N_=$(_N_) -DENABLE_OPENACC=0
DEFSET_LLVM = -D_N_=$(_N_) -DENABLE_OPENACC=0 -ffitl -Warc,-macro=_N_=$(_N_)
DEFSET_ACC = -D_N_=$(_N_) -DENABLE_OPENACC=1

#########################################################
# Makefile options that the user can overwrite          #
# OMP: set to 1 to use OpenMP (default: 0)              # 
# MODE: set to profile to use a built-in profiling tool #
#       (default: normal)                               #
#       If this is set to profile, the runtime system   #
#       will print profiling results according to the   #
#       verbosity level set by OPENARCRT_VERBOSITY      #
#       environment variable.                           # 
#########################################################
OMP ?= 0
MODE ?= normal

#########################################################
# Use the following macros to give program-specific     #
# compiler flags and libraries                          #
# - CFLAGS1 and CLIBS1 to compile the input C program   #
# - CFLAGS2 and CLIBS2 to compile the OpenARC-generated #
#   output C++ program                                  # 
#########################################################
#CFLAGS1 =  
#CFLAGS2 =  
CLIBS1 =  -lresilience
CLIBS1_LLVM = -lresilience -lstdc++
#CLIBS2 = 

################################################
# TARGET is where the output binary is stored. #
################################################
#TARGET ?= ./bin

include ../../../../make.template
