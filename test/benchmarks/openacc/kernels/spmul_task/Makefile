include ../../../../../make.header

########################
# Set the program name #
########################
BENCHMARK = spmul

########################################
# Set the input C source files (CSRCS) #
########################################
CSRCS = spmul.c timer.c

#########################################
# Set macros used for the input program #
#########################################
DEFSET_CPU = -DENABLE_OPENACC=0 -DSPMUL_INPUTDIR='"$(SPMUL_INPUTDIR)"'
DEFSET_LLVM = -DENABLE_OPENACC=0 -Warc,-macro=SPMUL_INPUTDIR='"$(SPMUL_INPUTDIR)"' $(ARCMACRO)

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
#      -profile //instrument the OpenCL kernel pipeline  #
#                //with performance counters.            #
#               //Use --profile for AOC V16.x or older.  #
#      -report  //display estimated resource usage on    #
#                //the screen.                           #
#               //Use --report for AOC V16.x or older.   #
#    (default: -march=emulator)                          #
# AOCL_BOARD: set a target Altera FPGA board             #
#    - "-board=$(AOCL_BOARD)" will be added to the AOC   #
#     in addition to the above flags                     # 
#    - Examples                                          #
#    p385_hpc_d5 //for Stratix V                         #
#    p510t_sch_ax115 //for Arria 10 (Nallatech 510T)     #
#    - For AOC V16.X or older, the following should be   #   
#    enabled too.                                        #   
#    BOARD_OPTION = "--board $(AOCL_BOARD)"              #   
# PRECMD: optional commands to be executed before the    #   
#    main compilation                                    #   
# POSTCMD: optional commands to be executed after the    #   
#    main compilation                                    #
##########################################################
OMP ?= 0
MODE ?= normal
#AOCL_BOARD ?= p385_hpc_d5
AOCL_BOARD ?= p510t_sch_ax115
#BOARD_OPTION = "--board $(AOCL_BOARD)"
#AOCL_FLAGS ?= -march=emulator
#AOCL_FLAGS ?= -march=emulator -c
#AOCL_FLAGS ?= -v -g -c -report
AOCL_FLAGS ?= -v -c -report
#AOCL_FLAGS ?= -v -report

#########################################################
# Use the following macros to give program-specific     #
# compiler flags and libraries                          #
# - CFLAGS1 and CLIBS1 to compile the input C program   #
# - CFLAGS2 and CLIBS2 to compile the OpenARC-generated #
#   output C++ program                                  # 
#########################################################
#CFLAGS1 =  
#CFLAGS2 =  
CLIBS1 = -lm
#CLIBS2 = 

################################################
# TARGET is where the output binary is stored. #
################################################
#TARGET ?= ./bin

include ../../../../../make.template
