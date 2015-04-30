include ../../../../../make.header

########################
# Set the program name #
########################
BENCHMARK = bfs

########################################
# Set the input C source files (CSRCS) #
########################################
CSRCS = bfs.c

#########################################
# Set macros used for the input program #
#########################################
VERIFICATION ?= 1
# Input Data: graph4096.txt
NUM_OF_NODES ?= 4096
EDGELIST_SIZE ?= 24576
# Input Data: graph1M.txt
#NUM_OF_NODES ?= 1000000
#EDGELIST_SIZE ?= 5999970
# Input Data: graph4M.txt
#NUM_OF_NODES ?= 4194304
#EDGELIST_SIZE ?= 25159848
# Input Data: graph16M.txt
#NUM_OF_NODES ?= 16777216
#EDGELIST_SIZE ?= 100666228

DEFSET_CPU = -DVERIFICATION=$(VERIFICATION) -DNUM_OF_NODES=$(NUM_OF_NODES) -DEDGELIST_SIZE=$(EDGELIST_SIZE)
DEFSET_ACC = -DVERIFICATION=$(VERIFICATION) -DNUM_OF_NODES=$(NUM_OF_NODES) -DEDGELIST_SIZE=$(EDGELIST_SIZE)

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
