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
###########################
# Input: fvcorr.domn.097K #
###########################
NEL = 97046
BLOCK_LENGTH = 256
NELR = 97280
#BLOCK_LENGTH = 128
#NELR = 97152
#BLOCK_LENGTH = 64
#NELR = 97088
#BLOCK_LENGTH = 32
#NELR = 97056
###########################
# Input: fvcorr.domn.193K #
###########################
#NEL = 193474
#BLOCK_LENGTH = 256
#NELR = 193536
#BLOCK_LENGTH = 128 
#NELR = 193536
#BLOCK_LENGTH = 64
#NELR = 193536
#BLOCK_LENGTH = 32
#NELR = 193504
###########################
# Input: missile.domn.0.2M #
###########################
#NEL = 232536
#BLOCK_LENGTH = 128 
#NELR = 232544
#BLOCK_LENGTH = 64
#NELR = 232576
#BLOCK_LENGTH = 32
#NELR = 232576


DEFSET_CPU = -DVERIFICATION=$(VERIFICATION) -DNEL=$(NEL) -Dblock_length=${BLOCK_LENGTH} -DNELR=$(NELR)
DEFSET_ACC = -DVERIFICATION=$(VERIFICATION) -DNEL=$(NEL) -Dblock_length=${BLOCK_LENGTH} -DNELR=$(NELR)

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
