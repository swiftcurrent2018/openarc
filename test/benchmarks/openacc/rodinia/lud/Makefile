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
# matrix_dim = 64 #
_M_SIZE = 4096
# matrix_dim = 256 #
#_M_SIZE = 65536
# matrix_dim = 512 #
#_M_SIZE = 262144
# matrix_dim = 2048 #
#_M_SIZE = 4194304


DEFSET_CPU = -DVERIFICATION=$(VERIFICATION) -D_M_SIZE=$(_M_SIZE)
DEFSET_ACC = -DVERIFICATION=$(VERIFICATION) -D_M_SIZE=$(_M_SIZE)

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
#CLIBS1 = 
#CLIBS2 = 

################################################
# TARGET is where the output binary is stored. #
################################################
#TARGET ?= ./bin

include ../../../../../make.template
