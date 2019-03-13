include $(openarc)/make.header

########################
# Set the program name #
########################
BENCHMARK = sobel

########################################
# Set the input C source files (CSRCS) #
########################################
CSRCS = *.c

#########################################
# Set macros used for the input program #
#########################################
_N_ ?= 512
DEFSET_CPU = -D_N_=$(_N_)
DEFSET_ACC = -D_N_=$(_N_)

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
#      --profile //instrument the OpenCL kernel pipeline #
#                //with performance counters.            #
#      --report  //display estimated resource usage on   #
#                //the screen.                           #
#    (default: -march=emulator)                          #
# AOCL_BOARD: set a target Altera FPGA board             #
#    - "--board $(AOCL_BOARD)" will be added to the AOC  #
#     in addition to the above flags                     # 
#    - Examples                                          #
#    p385_hpc_d5 //for Stratix V (default)               #
#    p510t_sch_ax115 //for Arria 10 (Nallatech 510T)     #
##########################################################
OMP ?= 0
MODE ?= normal

#AOCL_BOARD ?= p385_hpc_d5
AOCL_BOARD ?= p510t_sch_ax115

# More AOCL Flags
# -c: Stop after generating a <file>.aoco
# -v: verbose
# -g: add debug data
# --profile: enable profile support (small performance penalty)
# -shared: compile into object file, implies -c
# --high-effort: Increases aocx compile effort to impove ability to fit kernel on device
# --fpc: rounding stuff
# --fp-relaxed: relax order for arithmetic operations
# --report: print area astimates to screen after inital compilation

AOCL_FLAGS ?= -v -g -march=emulator
#AOCL_FLAGS ?= -v -g -c --report
#AOCL_FLAGS ?= -v --report

#########################################################
# Use the following macros to give program-specific     #
# compiler flags and libraries                          #
# - CFLAGS1 and CLIBS1 to compile the input C program   #
# - CFLAGS2 and CLIBS2 to compile the OpenARC-generated #
#   output C++ program                                  # 
#########################################################
CFLAGS1 = -I ./inc 
CFLAGS2 = -I ../inc 
CLIBS1 = -lm -lrt
CLIBS2 = -lm -lrt

################################################
# TARGET is where the output binary is stored. #
################################################
#TARGET ?= ./bin

include $(openarc)/make.template
