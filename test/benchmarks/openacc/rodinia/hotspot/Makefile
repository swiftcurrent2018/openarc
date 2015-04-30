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
# For temp_64 and  power_64 #
ROW_SIZE=64
COL_SIZE=64
# For temp_512 and  power_512 #
#ROW_SIZE=512
#COL_SIZE=512
# For temp_1024 and  power_1024 #
#ROW_SIZE=1024
#COL_SIZE=1024
# For temp_4096 and  power_4096 #
#ROW_SIZE=4096
#COL_SIZE=4096


DEFSET_CPU = -DVERIFICATION=$(VERIFICATION) -DROW_SIZE=$(ROW_SIZE) -DCOL_SIZE=${COL_SIZE}
DEFSET_ACC = -DVERIFICATION=$(VERIFICATION) -DROW_SIZE=$(ROW_SIZE) -DCOL_SIZE=${COL_SIZE}

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
