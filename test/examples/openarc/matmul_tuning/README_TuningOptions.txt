###########################################################################
# Avaliable ACC-to-GPU tuning configuration parameters                    #
###########################################################################
# Safe, always-beneficial Options #
###################################
#############################################################################
# Safe, always-beneficial Options, but resources may limit its application. #
#############################################################################
# useMatrixTranspose
#    - Apply MatrixTranspose optimization in allocating private data.
#    - Applicable when a private variable is an array of 1 dimension
#    - Not implemented yet
# useMallocPitch
#    - Use cudaMallocPitch() in ACC2CUDA translation
#    - Applicable when a shared variable is an array of 2 dimensions
#############################################################
# May-beneficial Options, which interact with other options #
#############################################################
# defaultNumWorkers=N
#    - Set default number of workers
# maxNumGangs=N
#    - Set maximum number of gangs
# defaultNumComputeUnits=N
#    - Default number of physical compute units (default value = 1); 
#      applicable only to Altera-OpenCL devices
# defaultNumSIMDWorkItems=N
#    - Default number of work-items within a work-group executing in an SIMD manner 
#      (default value = 1); applicable only to Altera-OpenCL devices
# useLoopCollapse
#    - Apply LoopCollapse optimization in ACC2GPU translation
# useParallelLoopSwap
#    - Apply ParallelLoopSwap optimization in ACC2GPU translation
# useUnrollingOnReduction
# 	 - Apply loop unrolling optimization for in-block reduction
#		   in ACC2GPU translation; to apply this opt, number of workers, 
#		   WORKER_SIZE = 2^m and m > 0. 
#    - Each kernel region can be tuned using noreductionunroll
#      clause.
###################################################################
# Always-beneficial Options, but inaccuracy of the analysis may   #
# break correctness.                                              #
###################################################################
# gpuMemTrOptLevel=N
#    - CPU-GPU memory transfer optimization level (0-4) (default is 2)
#    - Not implemented yet
# gpuMallocOptLevel=N
#    - GPU Malloc optimization level (0-1) (default is 0)
#    - Not implemented yet
####################################################################
# Safe, always-beneficial Options, but user's approval is required #
####################################################################
# assumeNonZeroTripLoops
#    - Assume that all loops hava non-zero iterations.
#    - Not implemented yet
# assumeNoAliasingAmongKernelArgs
#    - Assume that there is no aliasing among kernel arguments
# skipKernelLoopBoundChecking
#    - Skip kernel-loop-boundary-checking code when generating a device kernel; 
#      it is safe only if total number of workers equals to that of the kernel 
#      loop iterations
##############################################################
# May-beneficial options, which interact with other options. #
# These options are not needed if kernel-specific options    #
# are used.                                                  #       
##############################################################
#	shrdSclrCachingOnReg
#    - Cache shared scalar variables onto GPU registers
#    - Each kernel region can be tuned using registerRO, registerRW,
#      and noregister clauses.
#	shrdArryElmtCachingOnReg
#    - Cache shared array elements onto GPU registers
#    - Each kernel region can be tuned using registerRO, registerRW,
#      and noregister clauses.
# shrdSclrCachingOnSM
#    - Cache shared scalar variables onto GPU shared memory
#    - Each kernel region can be tuned using sharedRO, sharedRW,
#      and noshared clauses.
# prvtArryCachingOnSM
#    - Cache private array variables onto GPU shared memory
#    - Each kernel region can be tuned using sharedRO, sharedRW,
#      and noshared clauses.
# shrdArryCachingOnTM
#    - Cache 1-dimensional, R/O shared array variables onto GPU Texture memory
#    - Each kernel region can be tuned using texture and notexture clauses.
####################################################
# Non-tunable options, but user may need to apply  #
# some of these to generate correct output code.   #
####################################################
# doNotRemoveUnusedSymbols
#   - Do not remove unused local symbols in procedures.
# UEPRemovalOptLevel=N,
#   - Optimization level (0-3) to remove upwardly exposed private (UEP)
#     variables (default is 0, which does not apply this optimization).
#     This optimization may be unsafe; this should 
#     be enabled only if UEP problems occur, and programmer should verify 
#     the correctness manually. 
#    - Not implemented yet
# forceSyncKernelCall  
#   - If enabled, cudaThreadSynchronize() call is inserted right after
#     each kernel call to force explicit synchronization;
#     useful for debugging.
# AccPrivatization=N
#   - Set the level of automatic privatization optimization
#     N = 0 disable automatic privatization
#     N = 1 enable only scalar privatization (default)
#     N = 2 enable both scalar and array variable privatization
# AccReduction=N
#   - Set the level of automatic reduction-recognition optimization
#     N = 0 disable automatic reduction recognition
#     N = 1 enable only scalar reduction analysis (default)
#     N = 2 enable both scalar and array reduction variable analysis
############################################################################
# The following user directives are supported:                             #
############################################################################
# kernelid(id) procname(name) [clause[[,] clause]...]
# where clause is one of the following
#       registerRO(list)          # for R/O shared scalar or array data caching 
#       registerRW(list)          # for R/W shared scalar or array data caching 
#       noregister(list)          
#       sharedRO(list)            # for R/O private array caching, 
#                                 # R/O shared scalar caching, and  
#                                 # array expansion of private scalar variable.
#       sharedRW(list)            # for R/W private array caching and 
#                                 # array expansion of private scalar variable.
#       noshared(list)
#       texture(list)           # for R/O 1-dim Shared array caching 
#       notexture(list)
#       constant(list)          
#       noconstant(list)        
#       global(list)
#       noreductionunroll(list) # loop unrolling optimization is not applied
#                               # to variables in the list, even if 
#                               # useUnrollingOnReduction optin is used.
#       noploopswap
#       noloopcollapse
#       permute(list)
#       multisrccg(list)
#       multisrcgc(list)
#       conditionalsrc(list)
#       enclosingloops(list)
#       if( condition)
#       async( [( scalar-integer-expression )]
#       num_gangs( scalar-integer-expression )
#       num_workers( scalar-integer-expression )
#       vector_length( scalar-integer-expression )
#       reduction( operator:list )
#       copy(list)
#       copyin(list)
#       copyout(list)
#       create(list)
#       present(list)
#       pcopy(list)
#       pcopyin(list)
#       pcopyout(list)
#       pcreate(list)
#       deviceptr(list)
#       private(list)
#       firstprivate(list)
#       collapse(n)
#       gang [(scalar-integer-expression )]
#       worker [(scalar-integer-expression )]
#       vector [(scalar-integer-expression )]
#       seq
#       independent
############################################################################
# registerRO may contain
#   R/O shared scalar variables
#   R/O shared array element (ex: a[i])
# registerRW may contain
#   R/W shared scalar variables
#   R/W shared array element (ex: a[i])
# noregister may contain
#   R/O or R/W shared scalar variables
#   R/O or R/W shared array element (ex: a[i])
# sharedRO may contain
#   R/O shared scalar variables
#   R/O private array variables
# sharedRW may contain
#   R/W shared scalar variables (not yet implemented)
#   R/W private array variables
# noshared may contain
#   R/O or R/W shared scalar variables
#   R/O or R/W private array variables
# texture may contain
#   R/O 1-dimensional shared array
# notexture may contain
#   R/O 1-dimensional shared array
##########################
# Possible cache mapping #
##########################
# R/O shared scalar variables => registerRO, noregister, sharedRO, noshared
# R/W shared scalar variables => registerRW, noregister, sharedRW, noshared
# R/O shared array elements => registerRO, noregister
# R/W shared array elements => registerRW, noregister
# R/O 1-dimensional shared array variables => texture, notexture
# R/O private array variables => sharedRO, noshared
# R/W private array variables => sharedRW, noshared
