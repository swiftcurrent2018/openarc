include ../../../../make.header

################################################
# Makefile options that the user can overwrite #
################################################
OMP = 1

########################
# Set the program name #
########################
BENCHMARK = openmp

########################################
# Set the input C source files (CSRCS) #
########################################
CSRCS = openmp.c

#########################################
# Set macros used for the input program #
#########################################
_N_ ?= 512
DEFSET_CPU = -D_N_=$(_N_)
DEFSET_ACC = -D_N_=$(_N_)

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

include ../../../../make.template
