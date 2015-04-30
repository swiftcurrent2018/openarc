include ../../../../../make.header

########################
# Set the program name #
########################
BENCHMARK = kmeans

########################################
# Set the input C source files (CSRCS) #
########################################
CSRCS = kmeans.c kmeans_clustering.c cluster.c getopt.c

#########################################
# Set macros used for the input program #
#########################################
VERIFICATION ?= 1
# For input 204800.txt #
_NCLUSTERS=5
_NATTRIBUTES=34
_NPOINTS=204800
_UNROLLFAC_=1
# For input 494020.txt #
#_NCLUSTERS=5
#_NATTRIBUTES=34
#_NPOINTS=494020
#_UNROLLFAC_=10
# For input 819200.txt #
#_NCLUSTERS=5
#_NATTRIBUTES=34
#_NPOINTS=819200
#_UNROLLFAC_=100


DEFSET_CPU = -DVERIFICATION=$(VERIFICATION) -D_NCLUSTERS=$(_NCLUSTERS) -D_NATTRIBUTES=${_NATTRIBUTES} -D_NPOINTS=${_NPOINTS} -D_UNROLLFAC=${_UNROLLFAC}
DEFSET_ACC = -DVERIFICATION=$(VERIFICATION) -D_NCLUSTERS=$(_NCLUSTERS) -D_NATTRIBUTES=${_NATTRIBUTES} -D_NPOINTS=${_NPOINTS} -D_UNROLLFAC=${_UNROLLFAC}

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
