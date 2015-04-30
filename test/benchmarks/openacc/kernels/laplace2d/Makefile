include ../../../../../make.header

########################
# Set the program name #
########################
BENCHMARK = laplace2d

########################################
# Set the input C source files (CSRCS) #
########################################
CSRCS = laplace2d.c

#########################################
# Set macros used for the input program #
#########################################
VERIFICATION ?= 0
_NN_ ?= 2048
_NM_ ?= 2048
DEFSET_CPU = -DVERIFICATION=$(VERIFICATION) -DNN=$(_NN_) -DNM=$(_NM_)
DEFSET_ACC = -DVERIFICATION=$(VERIFICATION) -DNN=$(_NN_) -DNM=$(_NM_)

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
