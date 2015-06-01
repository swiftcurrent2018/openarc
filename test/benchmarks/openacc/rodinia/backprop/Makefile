include ../../../../../make.header

########################
# Set the program name #
########################
BENCHMARK = backprop

########################################
# Set the input C source files (CSRCS) #
########################################
CSRCS = backprop.c backprop_kernel.c facetrain.c imagenet.c

#########################################
# Set macros used for the input program #
#########################################
VERIFICATION ?= 1
SAVE_OUTPUT ?= 1
# Input Size: 65536
I_SIZE ?= 65537
# Input Size: 655360
#I_SIZE ?= 655361
# Input Size: 6553600
#I_SIZE ?= 6553601
DEFSET_CPU = -DSAVE_OUTPUT=$(SAVE_OUTPUT) -DVERIFICATION=$(VERIFICATION) -DI_SIZE=$(I_SIZE)
DEFSET_ACC = -DSAVE_OUTPUT=$(SAVE_OUTPUT) -DVERIFICATION=$(VERIFICATION) -DI_SIZE=$(I_SIZE)

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
