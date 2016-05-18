package openacc.exec;

import java.io.*;
import java.util.*;

import cetus.analysis.*;
import cetus.hir.*;
import cetus.transforms.*;
import cetus.codegen.*;
import cetus.exec.*;
import openacc.analysis.ACCAnalysis;
import openacc.analysis.ACCParser;
import openacc.codegen.*;
import openacc.transforms.ACCAnnotationParser;

/**
 * <b> ACC2GPUDriver </b> implements the command line parser and controls pass ordering.
 * Users may extend this class by overriding runPasses
 * (which provides a default sequence of passes).  The derived
 * class should pass an instance of itself to the run method.
 * Derived classes have access to a protected {@link Program Program} object.
 * 
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public class ACC2GPUDriver extends Driver
{
	private final String preprocessorDefault;
	private Set<String> optionsWithIntArgument;
	static private String openacc_version = "201306";
	private static final Set<String> cachingOpts =
		new HashSet<String>(Arrays.asList("shrdArryCachingOnConst",
				"shrdArryCachingOnTM", 
				"shrdArryElmtCachingOnReg", 
				"shrdSclrCachingOnConst", 
				"shrdSclrCachingOnReg", 
				"shrdSclrCachingOnSM" 
				));

	private BuildLLVMDelegate buildLLVM = null;
	/**
	 * This method's purpose is to give test cases access to the LLVM IR
	 * constructed by the -emitLLVM pass.
	 * 
	 * @return the {@link BuildLLVMDelegate} constructed during the -emitLLVM
	 *         pass, or null if that pass was not run
	 */
	public BuildLLVMDelegate getBuildLLVM() {
		return buildLLVM;
	}

	protected ACC2GPUDriver() {
		super();
		
		optionsWithIntArgument = new HashSet<String>();
		
        options.add(options.UTILITY,
            "debug_parser_output",
            "Print a parser output file before running any analysis/transformation passes and exit");
        
        options.add(options.UTILITY,
            "addIncludePath", "DIR",
            "Add the directory DIR to the list of directories to be searched for header files; to add multiple directories, " +
            "use this option multiple times. " +
            "(Current directory is included by default.)");

      options.add(options.UTILITY, "emitLLVM", null, ""/*default != 1*/, null,
        "Emit LLVM IR instead of source code. Optionally, target strings for"
        + " LLVM can be specified as an argument in the form:\n"
        + "\n"
        + "  [target-triple][;target-data-layout]\n"
        + "\n"
        + "For example:\n"
        + "\n"
        + "  -emitLLVM='x86_64-apple-macosx10.9.0;e-m:o-i64:64-f80:128-n8:16:32:64-S128'\n"
        + "\n"
        + "For details see:\n"
        + "\n"
        + "  http://llvm.org/docs/LangRef.html#target-triple\n"
        + "  http://llvm.org/docs/LangRef.html#data-layout\n"
        + "\n"
        + "To select the target strings configured when OpenARC was built,"
        + " specify either of the following:\n"
        + "\n"
        + "  -emitLLVM\n"
        + "  -emitLLVM=");

      options.add(options.UTILITY, "debugLLVM",
        "Generate debug output for BuildLLVM pass. Has no effect if BuildLLVM"
        + " pass is not enabled (see -emitLLVM)");

      options.add(options.UTILITY, "WerrorLLVM",
        "Report all BuildLLVM warnings as errors. Has no effect if BuildLLVM"
        + " pass is not enabled (see -emitLLVM)");

      options.add(options.UTILITY, "noPrintCode",
        "do not print final code, whether C or LLVM IR");

        options.add(options.UTILITY,
            "debug_preprocessor_command",
            "Print the command and options to be used for preprocessing and exit");
	    
	    options.add(options.UTILITY, "acc2gpu", "N",
        "Generate a Host+Accelerator program from OpenACC program: \n" +
        "        =0 disable this option\n" +
        "        =1 enable this option (default)\n" +
        "        =2 enable this option for distribued OpenACC program");
		optionsWithIntArgument.add("acc2gpu");
		
	    options.add(options.UTILITY, "targetArch", "N",
        "Set a target architecture: \n" +
        "        =0 for CUDA\n" +
        "        =1 for general OpenCL \n" +
        "        =2 for Xeon Phi with OpenCL \n" +
        "        =3 for Altera with OpenCL \n" +
        "If not set, the target is decided by OPENARC_ARCH env variable." );
		optionsWithIntArgument.add("targetArch");

        options.add(options.UTILITY, "omp2acc", "N",
                "Generate OpenACC program from OpenMP 4.0 program: \n" + 
        "        =0 disable this option (default)\n" +
        "        =1 enable this option");
		optionsWithIntArgument.add("omp2acc");
	    
		options.add(options.ANALYSIS, "AccAnalysisOnly", "N",
		"Conduct OpenACC analysis only and exit if option value > 0\n" +
		"        =0 disable this option (default)\n" +
		"        =1 OpenACC Annotation parsing\n" +
		"        =2 OpenACC Annotation parsing + initial code restructuring\n" + 
		"        =3 OpenACC parsing + code restructuring + OpenACC loop directive preprocessing\n" +
		"        =4 option3 + OpenACC annotation analysis\n" + 
		"        =5 option4 + privatization/reduction analyses");
		optionsWithIntArgument.add("AccAnalysisOnly");
		
		options.add(options.ANALYSIS, "SkipGPUTranslation", "N",
		"Skip the final GPU translation\n" +
		"        =1 exit after all analyses are done (default)\n" + 
		"        =2 exit before the final GPU translation\n" + 
		"        =3 exit after private variable transformaion\n" +
		"        =4 exit after reduction variable transformation");
		optionsWithIntArgument.add("SkipGPUTranslation");

		options.add(options.ANALYSIS, "AccPrivatization", "N",
		"Privatize scalar/array variables accessed in compute regions (parallel loops and kernels loops)\n"
        + "      =0 disable automatic privatization\n"
        + "      =1 enable only scalar privatization (default)\n"
        + "      =2 enable both scalar and array variable privatization\n"
        + "(this option is always applied unless explicitly disabled by setting the value to 0)");
		optionsWithIntArgument.add("AccPrivatization");
		
		options.add(options.ANALYSIS, "AccReduction","N",
        "Perform reduction variable analysis\n"
        + "      =0 disable reduction analysis \n"
        + "      =1 enable only scalar reduction analysis (default)\n"
        + "      =2 enable array reduction analysis and transformation\n"
        + "(this option is always applied unless explicitly disabled by setting the value to 0)");
		optionsWithIntArgument.add("AccReduction");
		
		options.add(options.ANALYSIS, "AccParallelization","N",
        "Find parallelizable loops\n"
        + "      =0 disable automatic parallelization analysis (default) \n"
        + "      =1 add independent clauses to OpenACC loops if they are parallelizable " +
        		"but don't have any work-sharing clauses");
		optionsWithIntArgument.add("AccParallelization");
		
		options.add(options.UTILITY, "defaultNumWorkers", "N",
		"Default number of workers per gang for compute regions (default value = 64)");
		optionsWithIntArgument.add("defaultNumWorkers");

		options.add(options.UTILITY, "defaultNumComputeUnits", "N",
		"Default number of physical compute units (default value = 1); "
		+ "applicable only to Altera-OpenCL devices");
		optionsWithIntArgument.add("defaultNumComputeUnits");

		options.add(options.UTILITY, "defaultNumSIMDWorkItems", "N",
		"Default number of work-items within a work-group executing in an SIMD manner (default value = 1); "
		+ "applicable only to Altera-OpenCL devices");
		optionsWithIntArgument.add("defaultNumSIMDWorkItems");
		
		options.add(options.TRANSFORM, "maxNumGangs", "N",
		"Maximum number of gangs per a compute region; this option will be applied to all gang loops in the program.");
		optionsWithIntArgument.add("maxNumGangs");
		
		options.add(options.TRANSFORM, "maxNumWorkers", "N",
		"Maximum number of workers per a compute region; this option will be applied to all gang loops in the program.");
		optionsWithIntArgument.add("maxNumWorkers");
		
		options.add(options.UTILITY, "showInternalAnnotations", "N",
		"Show internal annotations added by translator\n" +
		"        =0 does not show any OpenACC/internal annotations\n" +
		"        =1 show only OpenACC annotations (default)\n" +
		"        =2 show both OpenACC and acc internal annotations\n" +
		"        =3 show all annotations(OpenACC, acc internal, and cetus annotations)\n" +
		"(this option can be used for debugging purpose.)");
		optionsWithIntArgument.add("showInternalAnnotations");
		
		options.add(options.TRANSFORM, "disableWorkShareLoopCollapsing",
		"disable automatic collapsing of work-share loops in compute regions.");
		
		options.add(options.TRANSFORM, "disableStatic2GlobalConversion",
		"disable automatic converstion of static variables in procedures except for main into global variables.");
		
		options.add(options.UTILITY, "useMallocPitch", 
		"Use cudaMallocPitch() in ACC2GPU translation");
		
		options.add(options.TRANSFORM, "useMatrixTranspose",
		"Apply MatrixTranspose optimization in ACC2GPU translation");
		
		options.add(options.TRANSFORM, "useParallelLoopSwap",
		"Apply ParallelLoopSwap optimization in OpenACC2GPU translation");
		
		options.add(options.TRANSFORM, "useLoopCollapse",
		"Apply LoopCollapse optimization in ACC2GPU translation");
		
		options.add(options.TRANSFORM, "useUnrollingOnReduction",
		"Apply loop unrolling optimization for in-block reduction in ACC2GPU translation;" +
		"to apply this opt, number of workers in a gang should be 2^m.");
		
		options.add(options.ANALYSIS, "gpuMallocOptLevel", "N",
        "GPU Malloc optimization level (0-1) (default is 0)");
		optionsWithIntArgument.add("gpuMallocOptLevel");
		
		options.add(options.ANALYSIS, "gpuMemTrOptLevel", "N",
        "CPU-GPU memory transfer optimization level (0-4) (default is 3);" +
        "if N > 3, aggressive optimizations such as array-name-only analysis will be applied.");
		optionsWithIntArgument.add("gpuMemTrOptLevel");
		
		options.add(options.TRANSFORM, "UEPRemovalOptLevel", "N",
        "Optimization level (0-2) to remove upwardly exposed private (UEP) variables (default is 0). " +
        "This optimization may be unsafe; this should be enabled only if UEP problems occur, and" +
        "programmer should verify the correctness manually.");
		optionsWithIntArgument.add("UEPRemovalOptLevel");
		
		options.add(options.TRANSFORM, "MemTrOptOnLoops",
		"Memory transfer optimization on loops whose bodies contain only parallel regions.");
		
		options.add(options.TRANSFORM, "localRedVarConf", "N",
		"Configure how local reduction variables are generated; \n" + 
		"N = 2 (local scalar reduction variables are allocated in the GPU shared memory and local array reduction variables are cached on the shared memory) \n" +
		"N = 1 (local scalar reduction variables are allocated in the GPU shared memory and local array reduction variables are cached on the shared memory if included in CUDA sharedRO/sharedRW clause) (default) \n" +
		"N = 0 (All local reduction variables are allocated in the GPU global memory and not cached in the GPU shared memory.)");
		optionsWithIntArgument.add("localRedVarConf");

		options.add(options.TRANSFORM, "CloneKernelCallingProcedures", "N",
		"Clone procedures calling compute regions; \n" + 
		"N = 1 (Enable this kernel-calling-procedure cloning) (default) \n" +
		"N = 0 (Disable this kernel-calling-procedure cloning)");
		optionsWithIntArgument.add("CloneKernelCallingProcedures");
		
		options.add(options.UTILITY, "assumeNonZeroTripLoops",
		"Assume that all loops have non-zero iterations");

		options.add(options.UTILITY, "assumeNoAliasingAmongKernelArgs",
		"Assume that there is no aliasing among kernel arguments");

		options.add(options.UTILITY, "skipKernelLoopBoundChecking",
		"Skip kernel-loop-boundary-checking code when generating a device kernel; it is safe only if total number of workers equals to that of the kernel loop iterations");
		
		options.add(options.UTILITY, "programVerification", "N",
        "Perform program verfication for debugging; \n" +
        "N = 1 (verify the correctness of CPU-GPU memory transfers) (default)\n" +
		"N = 2 (verify the correctness of GPU kernel translation)");
		optionsWithIntArgument.add("programVerification");
		
		options.add(options.UTILITY, "verificationOptions", "complement=0|1:kernels=kernel1,kernel2,...",
		"Set options used for GPU kernel verification (programVerification == 1); \n" +
		"complement = 0 (consider kernels provided in the commandline with \"kernels\" sub-option)\n" +
		"           = 1 (consider all kernels except for those provided in the commandline with \"kernels\" sub-option (default))\n" +
		"kernels = [comma-separated list] consider the provided kernels.\n" + 
		"      (Note: It is used with \"complement\" sub-option to determine which kernels should be considered.)" );
		
		options.add(options.UTILITY, "defaultMarginOfError", "E",
        "Set the default value of the allowable margin of error for program verification (default E = 1.0e-6)");
		
		options.add(options.UTILITY, "minValueToCheck", "M",
        "Set the minimum value for error-checking; data with values lower than this will not be checked.\n" +
        "If this option is not provided, all GPU-written data will be checked for kernel verification.");
		
		options.add(options.UTILITY, "enableFaultInjection",
		"Enable directive-based fault injection; otherwise, fault-injection-related direcitves are ignored.\n" +
		"(If this option is set to 0 (enableFaultInjection=0), faults will be injected to each GPU thread; otherwise, faults " +
		"will be injected to only one GPU thread in each kernel. If -emitLLVM is also specified, fault injection is enabled, " +
		"but the -enableFaultInjection argument is ignored.)");
		
		options.add(options.UTILITY, "enableCustomProfiling",
		"Enable directive-based custom profiling; otherwise, profile-related directives are ignored.");
		
		options.add(options.UTILITY, "ASPENModelGen",
            "modelname=name:mode=number:entryfunction=entryfunc:complement=0|1:functions=foo,bar:postprocessing=number",
        "Generate ASPEN model for the input program; \n"
        + "modelname = [name of generated Aspen model]\n"
        + "mode = 0 (skip the whole Aspen model gereation passes)\n"
        + "       1 (analyze an input program and generated output C program annotated with Aspen directives)\n"
        + "       2 (skip analysis pass and generate output Aspen model only with Aspen directives annotated in the input program)\n"
        + "       3 (mode 1 + 2; analyze an input program, annotate it with Aspen directives, and generate output Aspen model (default))\n"
        + "       4 (mode 3 + modify the input OpenACC program such that each compute region is selectively offloaded using HI_aspenpredic() function)\n"
        + "entryfunction = [entry function to generate Aspen model]\n"
        + "functions = [comma-separated list of functions]\n"
        + "complement = 0 (ignore functions if specified in functions sub-option (default))\n"
        + "             1 (ignore functions if not specified in functions sub-option)\n"
        + "postprocessing = 0 (does not perform any Aspen IR flattening transformation)\n"
        + "                 1 (inline Aspen kernels called within Aspen maps)\n"
        + "                 2 (inline Aspen kernels + merge Aspen maps if directly nested (default))"
        + "");
		
		options.add(options.UTILITY, "addSafetyCheckingCode",
		"Add GPU-memory-usage-checking code just before each kernel call; used for debugging.");
		
		options.add(options.UTILITY, "doNotRemoveUnusedSymbols", "N",
		"Do not remove unused local symbols in procedures.\n" +
		"N = 0 (ignore this option; remove both unused symbols and procedures)\n" +
		"  = 1 (do not remove unused symbols or procedures; default)\n" + 
		"  = 2 (do not remove unused procedures)\n" +
		"  = 3 (do not remove unused symbols)");
		optionsWithIntArgument.add("doNotRemoveUnusedSymbols");
		
		options.add(options.UTILITY, "gpuConfFile", "filename",
				"Name of the file that contains OpenACC configuration parameters. " + 
				"(Any valid OpenACC-to-GPU compiler flags can be put in the file.) " + 
				"The file should exist in the current directory.");
		
		options.add(options.UTILITY, "extractTuningParameters", "filename",
				"Extract tuning parameters; output will be stored in the specified file. " +
				"(Default is TuningOptions.txt)" +
				"The generated file contains information on tuning parameters applicable " +
				"to current input program.");
		
		options.add(options.UTILITY, "genTuningConfFiles", "tuningdir",
				"Generate tuning configuration files and/or userdirective files; " +
				"output will be stored in the specified directory. " +
				"(Default is tuning_conf)");
		
		options.add(options.UTILITY, "tuningLevel", "N",
				"Set tuning level when genTuningConfFiles is on; \n" +
				"N = 1 (exhaustive search on program-level tuning options, default), \n" +
				"N = 2 (exhaustive search on kernel-level tuning options)");
		optionsWithIntArgument.add("tuningLevel");
		
		options.add(options.UTILITY, "defaultTuningConfFile", "filename",
				"Name of the file that contains default GPU tuning configurations. " +
				"(Default is gpuTuning.config) If the file does not exist, system-default setting will be used. ");
		
		options.add(options.UTILITY, "UserDirectiveFile", "filename",
				"Name of the file that contains user directives. " + 
				"The file should exist in the current directory.");
		
		options.add(options.UTILITY, "SetAccEntryFunction", "filename",
				"Name of the entry function, from which all device-related codes will be executed. " +
				"(Default is main.)");
		
		options.add(options.UTILITY, "printConfigurations",
				"Generate output codes to print applied configurations/optimizations at the program exit");
		
		
		////////////////////////////////////////////
		//Add Cuda-Specific command-line options. //
		////////////////////////////////////////////

		options.add(options.UTILITY, "cudaGlobalMemSize", "size in bytes",
		"Size of CUDA global memory in bytes (default value = 1600000000); used for debugging");
		optionsWithIntArgument.add("cudaGlobalMemSize");
		
		options.add(options.UTILITY, "cudaSharedMemSize", "size in bytes",
		"Size of CUDA shared memory in bytes (default value = 16384); used for debugging");
		optionsWithIntArgument.add("cudaSharedMemSize");

		options.add(options.UTILITY, "CUDACompCapability", "1.1",
		"CUDA compute capability of a target GPU");
		
		options.add(options.UTILITY, "addErrorCheckingCode",
		"Add CUDA-error-checking code right after each kernel call (If this option is on, forceSyncKernelCall" +
		"option is suppressed, since the error-checking code contains a built-in synchronization call.); used for debugging.");
		
		options.add(options.UTILITY, "cudaMaxGridDimSize", "number",
		"Maximum size of each dimension of a grid of thread blocks ( System max = 65535)");
		optionsWithIntArgument.add("cudaMaxGridDimSize");
		
		options.add(options.TRANSFORM, "forceSyncKernelCall", 
		"If enabled, HI_synchronize(1) call is inserted right after each kernel call in the default queue" +
		" to force explicit synchronization; useful for debugging or timing the kernel execution.");
		
		options.add(options.TRANSFORM, "shrdSclrCachingOnReg",
		"Cache shared scalar variables onto GPU registers");
		
		options.add(options.TRANSFORM, "shrdArryElmtCachingOnReg",
		"Cache shared array elements onto GPU registers; this option may not be used " +
		"if aliasing between array accesses exists.");
		
		options.add(options.TRANSFORM, "shrdSclrCachingOnSM",
		"Cache shared scalar variables onto GPU shared memory");
		
		options.add(options.TRANSFORM, "prvtArryCachingOnSM",
		"Cache private array variables onto GPU shared memory");
		
		options.add(options.TRANSFORM, "shrdArryCachingOnTM",
		"Cache 1-dimensional, R/O shared array variables onto GPU texture memory");
		
		options.add(options.TRANSFORM, "shrdSclrCachingOnConst",
		"Cache R/O shared scalar variables onto GPU constant memory");
		
		options.add(options.TRANSFORM, "shrdArryCachingOnConst",
		"Cache R/O shared array variables onto GPU constant memory");

		options.add(options.TRANSFORM, "disableDefaultCachingOpts",
		"Disable default caching optimizations so that they are applied only if explicitly requested");

    options.add(options.TRANSFORM, "loopUnrollFactor", "N",
            "Unroll loops inside OpenACC compute regions\n" +
                    "        N Specifies the unroll factor");
    optionsWithIntArgument.add("AccAnalysisOnly");

    // Specify "cc -E" not "cpp" as the preprocessor because, if it's clang
    // (3.5.1), "cpp" implies "-traditional-cpp", which is not compatible with
    // some system header files (MAC OS X 10.9.5).

    //setOptionValue("acc2gpu", "1");
    //Replace default preprocessor options.
    //setOptionValue("preprocessor", "cc -E -CC -I. -D _OPENARC_ -D _OPENACC="+openacc_version);
    //DEBUG: for Mac, the following option should be used, instead.

    // Do not let java String interning happen here so that we can compare by
    // reference later to see if the default value is still set.
    String cpp = BuildConfig.getBuildConfig().getProperty("cpp");
    preprocessorDefault = cpp + " -CC";
    setOptionValue("preprocessor", preprocessorDefault);
	}
	
	/**
	 * Set additional passes for ACC2GPU translation.
	 */
	public void setPasses() 
	{
		int MemTrOptLevel = 2;
		int MallocOptLevel = 0;
		boolean	useGlobalGMalloc = false;
		boolean	globalGMallocOpt = false;
		boolean implicitMemTrOptSet = false;
		boolean implicitMallocOptSet = false;
		boolean disableDefaultCachingOpts = false;
		
		String value = getOptionValue("omp2acc");
		if( value != null ) {
			if( Integer.valueOf(value).intValue() == 1 ) {
				value = getOptionValue("acc2gpu");
				if( value == null ) {
					setOptionValue("acc2gpu", "1");
				}
			}
		}
		
		value = getOptionValue("defaultNumWorkers");
		if( value == null ) {
			setOptionValue("defaultNumWorkers", "64");
		}
		
		value = getOptionValue("AccAnalysisOnly");
		if( value == null ) {
			setOptionValue("AccAnalysisOnly", "0");
		}
		
		value = getOptionValue("AccPrivatization");
		if( value == null ) {
			setOptionValue("AccPrivatization", "1");
		}
		
		value = getOptionValue("AccReduction");
		if( value == null ) {
			setOptionValue("AccReduction", "1");
		}
		
		value = getOptionValue("AccParallelization");
		if( value == null ) {
			setOptionValue("AccParallelization", "0");
		}
		
		value = getOptionValue("doNotRemoveUnusedSymbols");
		if( value == null ) {
			setOptionValue("doNotRemoveUnusedSymbols", "0");
		}
		
		value = getOptionValue("programVerification");
		if( value == null ) {
			setOptionValue("programVerification", "0");
		}
		
		value = getOptionValue("defaultMarginOfError");
		if( value == null ) {
			setOptionValue("defaultMarginOfError", "1.0e-6");
		}
		
		if(getOptionValue("genTuningConfFiles") != null)
		{
			if(getOptionValue("extractTuningParameters") == null) {
				setOptionValue("extractTuningParameters", "TuningOptions.txt");
			}
		}

		if(getOptionValue("extractTuningParameters") != null)
		{
			setOptionValue("useParallelLoopSwap", "1");
		}

		if(getOptionValue("useParallelLoopSwap") != null)
		{
			setOptionValue("ddt", "1");
		}

		value = getOptionValue("localRedVarConf");
		if( value == null ) {
			setOptionValue("localRedVarConf", "1");
		}
		value = getOptionValue("showInternalAnnotations");
		if( value == null ) {
			setOptionValue("showInternalAnnotations", "1");
		}

		if(getOptionValue("disableDefaultCachingOpts") != null)
		{
			disableDefaultCachingOpts = true;
		}
		///////////////////////////////////////////////////////////////
		//If no caching optimization is specified, default options   //
		//are enabled unless disableDefaultCachingOpts is specified. //
		///////////////////////////////////////////////////////////////
		boolean cachingOptFound = false;
		for( String tOpt : cachingOpts ) {
			value = getOptionValue(tOpt);
			if( value != null ) {
				cachingOptFound = true;
				break;
			}
		}
		if( !cachingOptFound && !disableDefaultCachingOpts ) {
			setOptionValue("shrdArryElmtCachingOnReg", "1");
			setOptionValue("shrdSclrCachingOnReg", "1");
			setOptionValue("shrdSclrCachingOnSM", "1");
		}

        Map<String, String> env = System.getenv();
        //If use OpenCL, the use of texture memory is not allowed
		value = getOptionValue("targetArch");
		int targetArch = 0;
		if( value != null ) {
			targetArch = Integer.valueOf(value).intValue();
		} else {
			if( env.containsKey("OPENARC_ARCH") ) {
				value = env.get("OPENARC_ARCH");
				targetArch = Integer.valueOf(value).intValue();
			}
		}
        if(targetArch != 0)
        {
            setOptionValue("shrdArryCachingOnTM", null);
        }
		
		//////////////////////////
		//Verify option values. //
		//////////////////////////
		for( String option : optionsWithIntArgument ) {
			value = getOptionValue(option);
			if( value != null ) {
				Expression expr = ACCParser.ExpressionParser.parse(value);
				if( (expr == null) || !(expr instanceof IntegerLiteral) ) {
					Tools.exit("[ERROR in commandline input parsing] wrong argument (" + value + 
					") for " + option + " option; argument should be an integer constant.");
				}
			}
		}
		
		//////////////////////////////////
		//Below options are deprecated. //
		//////////////////////////////////
		value = getOptionValue("useGlobalGMalloc");
		if( value != null ) {
			setOptionValue("ddt", "1");
			useGlobalGMalloc = true;
		}
		value = getOptionValue("globalGMallocOpt");
		if( value != null ) {
			globalGMallocOpt = true;
		}
		value = getOptionValue("gpuMemTrOptLevel");
		if( value != null ) {
			MemTrOptLevel = Integer.valueOf(value).intValue();
		} else {
			setOptionValue("gpuMemTrOptLevel", "2");
			implicitMemTrOptSet = true;
		}
		value = getOptionValue("gpuMallocOptLevel");
		if( value != null ) {
			MallocOptLevel = Integer.valueOf(value).intValue();
		} else {
			setOptionValue("gpuMallocOptLevel", "0");
			implicitMallocOptSet = true;
		}
		if( useGlobalGMalloc && globalGMallocOpt ) {
			if( implicitMemTrOptSet ) {
				setOptionValue("cudaMemTrOptLevel", "3");
			}
			if( implicitMallocOptSet ) {
				setOptionValue("cudaMallocOptLevel", "1");
			}
		}
		if( (MemTrOptLevel > 2) && (MallocOptLevel == 0) ) {
			PrintTools.println("\n[WARNING] if cudaMemTrOptLevel > 2, cudaMallocOptLevel should be bigger than 0;" +
					" system sets cudaMallocOptLevel to 1.\n", 0);
			setOptionValue("cudaMallocOptLevel", "1");
		}
		
	}
	
    /**
    * Prints the list of options that OpenARC accepts.
    */
    public void printUsage() {
        String usage = "\nopenacc.exec.ACC2GPUDriver [option]... [file]...\n";
        usage += options.getUsage();
        System.err.println(usage);
    }
	
    /**
    * Runs analysis and optimization passes on the program.
    * This override Driver.runPasses() to change execution orders of the built-in compiler
    * passes like InlineExpansionPass. 
    */
    public void runPasses() {
    	//[DEBUG: modified by Seyong Lee] Remove ompGen from prerequisite list of
    	//parallelize-loops pass.
        /* check for option dependences */
        /* in each set of option strings, the first option requires the
           rest of the options to be set for it to run effectively */
        String[][] pass_prerequisites = {
            {"inline",
                "tsingle-call","tsingle-return"},
            {"parallelize-loops",
                "alias","ddt","privatize","reduction","induction"},
            {"loop-interchange",
                "ddt"}
        };
        for (String[] pass_prerequisite : pass_prerequisites) {
            if (getOptionValue(pass_prerequisite[0]) != null) {
                for (int j = 1; j < pass_prerequisite.length; ++j) {
                    if (getOptionValue(pass_prerequisite[j]) == null) {
                        System.out.println("[Driver] turning on required pass "
                            + pass_prerequisite[j] + " for "
                            + pass_prerequisite[0]);
                        options.setValue(pass_prerequisite[j]);
                    }
                }
            }
        }
        if (getOptionValue("teliminate-branch") != null) {
            TransformPass.run(new BranchEliminator(program));
        }
        if (getOptionValue("callgraph") != null) {
            CallGraph cg = new CallGraph(program);
            cg.print(System.out);
        }
        if (getOptionValue("tsingle-declarator") != null) {
            TransformPass.run(new SingleDeclarator(program));
        }
        if (getOptionValue("tsingle-call") != null) {
            TransformPass.run(new SingleCall(program));
        }
        if (getOptionValue("tsingle-return") != null) {
            TransformPass.run(new SingleReturn(program));
        }
        //[FIXME] this pass should have been called after ACCAnnotation parsing.
        // However, it could not be deferred since variable-renaming pass in 
        //InlineExpansionPass works only on Annotations with String values.
        if (getOptionValue("tinline") != null) {
            TransformPass.run(new InlineExpansionPass(program));
        }
        if (getOptionValue("normalize-loops") != null) {
            TransformPass.run(new LoopNormalization(program));
        }
        if (getOptionValue("normalize-return-stmt") != null) {
            TransformPass.run(new NormalizeReturn(program));
        }
        if (getOptionValue("induction") != null) {
            TransformPass.run(new IVSubstitution(program));
        }
        if (getOptionValue("privatize") != null) {
            AnalysisPass.run(new ArrayPrivatization(program));
        }
        if (getOptionValue("ddt") != null) {
            AnalysisPass.run(new DDTDriver(program));
        }
        if (getOptionValue("reduction") != null) {
            AnalysisPass.run(new Reduction(program));
        }
/*
        if (getOptionValue("openmp") != null) {
            AnalysisPass.run(new OmpAnalysis(program));
        }
*/
        if (getOptionValue("parallelize-loops") != null) {
            AnalysisPass.run(new LoopParallelizationPass(program));
        }
        if (getOptionValue("ompGen") != null) {
            CodeGenPass.run(new ompGen(program));
        }
/*
        if (getOptionValue("loop-interchange") != null) {
            TransformPass.run(new LoopInterchange(program));
        }
        if (getOptionValue("loop-tiling") != null) {
            AnalysisPass.run(new LoopTiling(program));
        }
*/
        if (getOptionValue("profile-loops") != null) {
            TransformPass.run(new LoopProfiler(program));
        }
    }

	/**
	 * Runs this driver with args as the command line.
	 *
	 * @param args The command line from main.
	 */
	public void run(String[] args)
	{
		parseCommandLine(args);
		if( getOptionValue("emitLLVM") == null )
		{
			String newCPPCMD = getOptionValue("preprocessor") + " -I.";
			setOptionValue("preprocessor", newCPPCMD);
		}
		//Add include-path to the preprocessor command if existing.
		//[DEBUG] Current implementation allows only one "addIncludePath" commandline option.
		//To specify multiple include-paths, they should be specified in the configuration file.
		String value = getOptionValue("addIncludePath");
		if( value != null )
		{
			String newCPPCMD = getOptionValue("preprocessor") + " -I" + value;
			setOptionValue("preprocessor", newCPPCMD);
		}
		parseGpuConfFile();
		//If help, version, dump-options, or dump-system-options is included
		//in the user-configuration file, exit after running specified task.
		if (getOptionValue("help") != null ||
				getOptionValue("usage") != null) {
			printUsage();
			Tools.exit(0);
		}
		if (getOptionValue("version") != null) {
			printVersion();
			Tools.exit(0);
		}
		if (getOptionValue("dump-options") != null) {
			setOptionValue("dump-options", null);
			dumpOptionsFile();
			Tools.exit(0);
		}
		if (getOptionValue("dump-system-options") != null) {
			setOptionValue("dump-system-options", null);
			dumpSystemOptionsFile();
			Tools.exit(0);
		}
		if (getOptionValue("acc2gpu") == null && getOptionValue("emitLLVM") == null)
		{
			//The main acc2gpu pass is always executed unless a user 
			//explicitly disables it.
			setOptionValue("acc2gpu", "1");
		}

		// Currently, the LLVM backend does not support OpenACC, so do not define
		// _OPENACC if -emitLLVM.
		String preprocessorString = getOptionValue("preprocessor");
		if( getOptionValue("emitLLVM") == null )
		{
			preprocessorString += " -D_OPENACC=" + openacc_version;
		}
		//_OPENARC_ internal macro is always added.
		preprocessorString += " -D_OPENARC_";
		setOptionValue("preprocessor",preprocessorString);
        if(getOptionValue("debug_preprocessor_command") != null) {
        	Tools.exit("\npreprocessor command to be used = " + getOptionValue("preprocessor") + "\n");
        }
		
		HashMap<String, HashMap<String,Object>> userDirectives = parseUserDirectiveFile();
		HashMap<String, Object> tuningConfigs = parseTuningConfig();

		parseFiles();
		
		if (getOptionValue("debug_parser_output") != null)
		{
			System.err.println("print parsed output and exit without doing any analysis/transformation!");
			try {
				program.print();
			} catch (IOException e) {
				System.err.println("could not write output files: " + e);
				System.exit(1);
			}
			System.exit(0);
		}

		setPasses();
		
		runPasses();

		value = getOptionValue("acc2gpu");
		final int acc2gpu = value == null ? 0 : Integer.valueOf(value);
		
		value = getOptionValue("debugLLVM");
		final boolean debugLLVM = value != null && Integer.valueOf(value) != 0;
		
		value = getOptionValue("WerrorLLVM");
		final boolean WerrorLLVM = value != null && Integer.valueOf(value) != 0;
		
		final String llvmTargetInfo = getOptionValue("emitLLVM");
		if (llvmTargetInfo != null)
		{
			if (acc2gpu != 0) {
				System.err.println("-acc2gpu="+acc2gpu+" and -emitLLVM cannot be combined");
				System.exit(1);
			}
			final String targetTriple;
			final String targetDataLayout;
			if (llvmTargetInfo.isEmpty()) {
				final BuildConfig buildConfig = BuildConfig.getBuildConfig();
				targetTriple = buildConfig.getProperty("llvmTargetTriple");
				targetDataLayout = buildConfig.getProperty("llvmTargetDataLayout");
			}
			else {
				final String[] arr = llvmTargetInfo.split(";", 2);
				int i = 0;
				targetTriple     = i < arr.length ? arr[i++] : "";
				targetDataLayout = i < arr.length ? arr[i++] : "";
			}
			TransformPass.run(new ACCAnnotationParser(program));
			ACCAnalysis.updateSymbolsInACCAnnotations(program, null);
			try {
				buildLLVM = BuildLLVMDelegate.make(
				  targetTriple, targetDataLayout, program, debugLLVM, WerrorLLVM,
				  getOptionValue("enableFaultInjection") != null);
			} catch (BuildLLVMDelegate.BuildLLVMDisabledException e) {
				System.err.println("-emitLLVM is disabled because OpenARC was built"
				                   + " without LLVM");
				System.exit(1);
				return; // suppress uninit warnings for buildLLVM
			}
			CodeGenPass.run(buildLLVM);
			if (getOptionValue("noPrintCode") != null) {
				return;
			}
			buildLLVM.printLLVM(Driver.getOptionValue("outdir"));
			return;
		}
		
		if (acc2gpu != 0)
			CodeGenPass.run(new acc2gpu(program, userDirectives, tuningConfigs));
		
		if (getOptionValue("noPrintCode") != null) {
			return;
		}

		PrintTools.printlnStatus("Printing...", 1);

		try {
			program.print();
		} catch (IOException e) {
			System.err.println("could not write output files: " + e);
			System.exit(1);
		}
	}
	/**
	 * Entry point for OpenARC; creates a new Driver object,
	 * and calls run on it with args.
	 *
	 * @param args Command line options.
	 */
	public static void main(String[] args)
	{
		/* Set default options for acc2gpu translator. */
		ACC2GPUDriver O2GDriver = new ACC2GPUDriver();
		O2GDriver.run(args);
	}

	protected void parseGpuConfFile() {
		String value = getOptionValue("gpuConfFile");
		if( value != null ) {
			if( value.equals("1") ) {
				PrintTools.println("\n[WARNING] no GPU-configuration file is specified; " +
						"gpuConfFile option will be ignored.\n", 0);
				return;
			}
			//Read contents of the file and parse configuration parameters.
			try {
				FileReader fr = new FileReader(value);
				BufferedReader br = new BufferedReader(fr);
				String inputLine = null;
				while( (inputLine = br.readLine()) != null ) {
					String opt = inputLine.trim();
					if( opt.length() == 0 ) {
						continue;
					}
					if( opt.charAt(0) == '#' ) {
						//Ignore comment line.
						continue;
					}
					int eq = opt.indexOf('=');

					if (eq == -1)
					{
						/* no value on the command line, so just set it to "1" */
						String option_name = opt.substring(0);

						if (options.contains(option_name)) {
							// Commandline input has higher priority than this configuration input.
							if( getOptionValue(option_name) == null ) {
								setOptionValue(option_name, "1");
							} else {
								continue;
							}
						} else {
							System.err.println("ignoring unrecognized option " + option_name);
						}
					}
					else
					{
						/* use the value from the command line */
						String option_name = opt.substring(0, eq);

						if (options.contains(option_name)) {
							// Commandline input has higher priority than this configuration input.
							if( option_name.equals("preprocessor") ) {
								Tools.exit("[ERROR in configuration file parsing] preprocessor option should not be specified in the configuration file; exit!"); 
							} else if( getOptionValue(option_name) == null ) {
								setOptionValue(option_name, opt.substring(eq + 1));
								if( option_name.equals("addIncludePath") ) {
									String tInc = opt.substring(eq + 1);
									String newCPPCMD = getOptionValue("preprocessor") + " -I" + tInc;
									setOptionValue("preprocessor", newCPPCMD);
								}
							} else {
								if( option_name.equals("verbosity") || option_name.equals("outdir") ) {
									setOptionValue(option_name, opt.substring(eq + 1));
								} else if( option_name.equals("addIncludePath") ) {
									String tInc = opt.substring(eq + 1);
									String newCPPCMD = getOptionValue("preprocessor") + " -I" + tInc;
									setOptionValue("preprocessor", newCPPCMD);
								} else if( option_name.equals("macro") ) {
									String tMacro = opt.substring(eq + 1);
									String newMacro = getOptionValue("macro") + "," + tMacro;
									setOptionValue("macro", newMacro);
								} else {
									continue;
								}
							}
						} else {
							System.err.println("ignoring unrecognized option " + option_name);
						}
					}
				}
				br.close();
				fr.close();
			} catch (Exception e) {
				PrintTools.println("Error in readling gpuConfFile!!", 0);
				PrintTools.println("Caught error message: " + e + "\n", 1);
			}
		}
	}

	protected HashMap<String, HashMap<String, Object>> parseUserDirectiveFile() {
		HashMap<String, HashMap<String, Object>> userDirectiveMap = 
				new HashMap<String, HashMap<String, Object>>();
		String value = getOptionValue("UserDirectiveFile");
		if( value != null ) {
			if( value.equals("1") ) {
				PrintTools.println("\n[WARNING] no User-Directive file is specified; " +
						"UserDirectiveFile option will be ignored.\n", 0);
				return userDirectiveMap;
			}
			//Read contents of the file and parse configuration parameters.
			try {
				FileReader fr = new FileReader(value);
				BufferedReader br = new BufferedReader(fr);
				String inputLine = null;
				HashMap<String, HashMap<String, Object>> uDirectives = 
						new HashMap<String, HashMap<String, Object>>();
				while( (inputLine = br.readLine()) != null ) {
					String opt = inputLine.trim();
					if( opt.length() == 0 ) {
						continue;
					}
					if( opt.charAt(0) == '#' ) {
						//Ignore comment line.
						continue;
					}
					/////////////////////////////////////////////////////////
					//Input string, opt, should have spaces before and     //
					//after the following tokens: '(', ')', ','            //
					/////////////////////////////////////////////////////////
					//opt = opt.replaceAll("\\(", " ( ");
					//opt = opt.replaceAll("\\)", " ) ");
					//opt = opt.replaceAll(",", " , ");
					opt = opt.replace("(", " ( ");
					opt = opt.replace(")", " ) ");
					opt = opt.replace("[", " [ ");
					opt = opt.replace("]", " ] ");
					opt = opt.replace(",", " , ");
					opt = opt.replace(":", " : ");
					opt = opt.replace("#", " # ");
					opt = opt.replace("+", " + ");
					opt = opt.replace("-", " - ");
					opt = opt.replace("*", " * ");
					opt = opt.replace("/", " / ");
					opt = opt.replace("%", " % ");
					String[] token_array = opt.split("\\s+");
					uDirectives = ACCParser.parse_userdirective(token_array);
					if( uDirectives == null ) {
						continue;
					} else {
						Set<String> fKeySet = userDirectiveMap.keySet();
						Set<String> tKeySet = uDirectives.keySet();
						for( String tKey : tKeySet ) {
							HashMap<String, Object> tMap = uDirectives.get(tKey);
							if( fKeySet.contains(tKey) ) {
								HashMap<String, Object> fMap = userDirectiveMap.get(tKey);
								Set<String> ffKeySet = fMap.keySet();
								Set<String> ttKeySet = tMap.keySet();
								for( String ttKey : ttKeySet ) {
									Object ttObj = tMap.get(ttKey);
									if( ffKeySet.contains(ttKey) ) {
										Object ffObj = fMap.get(ttKey);
										if( ffObj instanceof Set ) {
											Set ffSet = (Set)ffObj;
											ffSet.addAll((Set)ttObj);
										} else if( ffObj instanceof List ) {
											List ffList = (List)ffObj;
											ffList.addAll((List)ttObj);
										} else if( ffObj instanceof Map ) {
											Map ffMap = (Map)ffObj;
											Map ttMap = (Map)ttObj;
											for( Object tttKey : ttMap.keySet() ) {
												if( ffMap.containsKey(tttKey) ) {
													Set fffSet = (Set)ffMap.get(tttKey);
													fffSet.addAll((Set)ttMap.get(tttKey));
												} else {
													Set fffSet =new HashSet((Set)ttMap.get(tttKey));
													ffMap.put(tttKey, fffSet);
												}
											}
										} else {
											fMap.put(ttKey, ttObj);
										}
									} else {
										fMap.put(ttKey, ttObj);
									}

								}

							} else {
								userDirectiveMap.put(tKey, tMap);
							}
						}
					}
				}
				br.close();
				fr.close();
			} catch (Exception e) {
				PrintTools.println("Error in readling cuda user-directive file!!", 0);
			}
		} 
		return userDirectiveMap;
	}

	protected HashMap<String, Object> parseTuningConfig() {
		HashMap<String, Object> tuningConfigMap = 
				new HashMap<String, Object>();
		String value = getOptionValue("defaultTuningConfFile");
		if( value != null ) {
			if( value.equals("1") ) {
				PrintTools.println("[INFO] no GPU Tuning Config file is specified; " +
						"default configuration file (gpuTuning.config) will be used.", 0);
				value = "gpuTuning.config";
			}
			//Read contents of the file and parse configuration parameters.
			FileReader fr = null;
			try {
				fr = new FileReader(value);
			} catch (Exception e) {
				PrintTools.println("[INFO] no GPU tuning configuration file is found;" +
						" default configuration will be used.", 0);
				return tuningConfigMap;
			}
			try {
				BufferedReader br = new BufferedReader(fr);
				String inputLine = null;
				HashMap<String, Object> tConfigMap = null;
				while( (inputLine = br.readLine()) != null ) {
					String opt = inputLine.trim();
					if( opt.length() == 0 ) {
						continue;
					}
					if( opt.charAt(0) == '#' ) {
						//Ignore comment line.
						continue;
					}
					/////////////////////////////////////////////////////////
					//Input string, opt, should have spaces before and     //
					//after the following tokens: '(', ')', ',', '='       //
					/////////////////////////////////////////////////////////
					opt = opt.replaceAll("\\(", " ( ");
					opt = opt.replaceAll("\\)", " ) ");
					opt = opt.replaceAll(",", " , ");
					opt = opt.replaceAll("=", " = ");
					String[] token_array = opt.split("\\s+");
					tConfigMap = ACCParser.parse_tuningconfig(token_array);
					if( tConfigMap == null ) {
						continue;
					} else {
						Set<String> fKeySet = tuningConfigMap.keySet();
						Set<String> tKeySet = tConfigMap.keySet();
						for( String tKey : tKeySet ) {
							Object tObj = tConfigMap.get(tKey);
							if( fKeySet.contains(tKey) ) {
								Object fObj = tuningConfigMap.get(tKey);
								if( fObj instanceof Set && tObj instanceof Set ) {
									((Set)fObj).addAll((Set)tObj);
								} else if( fObj instanceof List && tObj instanceof List ) {
									((List)fObj).addAll((List)tObj);
								} else if( (tObj instanceof String) || (tObj instanceof Expression) || (tObj instanceof Symbol) ) {
									//Overwrite value.
									tuningConfigMap.put(tKey, tObj);
								} else {
									PrintTools.println("[ERROR in parseTuningConfig()] unsuppored input found; " +
											"remaining input configuraiton will be ignored.", 0);
									return tuningConfigMap;
								}
							} else {
								tuningConfigMap.put(tKey, tObj);
							}
						}
					}
				}
				br.close();
				fr.close();
			} catch (Exception e) {
				PrintTools.println("Error in readling GPU tuning configuration file!!\n" +
						"Error Message: " + e + "\n", 0);
			}
		} 
		return tuningConfigMap;
	}

}

