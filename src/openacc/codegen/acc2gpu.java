package openacc.codegen;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

import cetus.codegen.*;
import cetus.hir.*;
import cetus.exec.*;
import cetus.analysis.AnalysisPass;
import cetus.analysis.ArrayPrivatization;
import cetus.transforms.*;
import openacc.hir.*;
import openacc.analysis.*;
import openacc.transforms.*;

/**
 * <b>acc2gpu</b> converts an OpenACC program into a CUDA GPU program
 * 
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public class acc2gpu extends CodeGenPass
{
	private int AccAnalysisOnly = 0;
	private int AccPrivatization = 1;
	private int AccReduction = 1;
	private int AccParallelization = 0;
	private int programVerification = 0;
	private int FaultInjectionOption = 1;
    private boolean convertOpenMPtoOpenACC = false;
	private boolean enableFaultInjection = false;
	private boolean enableCustomProfiling = false;
	private boolean ParallelLoopSwap = false;
	private boolean useGlobalGMalloc = false;
	private boolean globalGMallocOpt = false;
	private boolean addSafetyCheckingCode = false;
	private boolean disableStatic2GlobalConversion = false;
	private boolean assumeNoAliasingAmongKernelArgs = false;
	private boolean opt_GenDistOpenACC = false;
	private int cloneKernelCallingProcedures = 1;
	private int doNotRemoveUnusedSymbols = 0;
	//private boolean kernelCallingProcCloning = false;
	private boolean MemTrOptOnLoops = false;
	//private boolean applyInlining = false;
	private int SkipGPUTranslation = 0;
	private int MemTrOptLevel = 2;
	private int MallocOptLevel = 0;
	private int tuningLevel = 1;
	private int UEPRemovalOptLevel = 0;
	private int showInternalAnnotations = 0;
	private int ASPENModelGenMode = 0;
	private HashMap<String, HashMap<String, Object>> userDirectives;
	private HashMap<String, Object> tuningConfigs;
	private String tuningParamFile= null;
	private String tuningConfDir = null;
	private boolean IRSymbolOnly = true;
	private Map<String, String> env = null;
	private int unrollFactor = 1;
	private int tNumComputeUnits = 0;
	private int tNumSIMDWorkItems = 0;
	private int OPENARC_ARCH = 0;
	
	public acc2gpu(Program program, HashMap<String, HashMap<String, Object>> uDirectives,
			HashMap<String, Object> tConfigs)
	{
		super(program);
		userDirectives = uDirectives;
		tuningConfigs = tConfigs;
        env = System.getenv();
	}

	public String getPassName()
	{
		return new String("[acc2gpu]");
	}
	
	protected void cleanAnnotations() {
		/////////////////////////////////////////////////////////////////////////
		// Clean internal Cetus annotations if showInternalAnnotations is off. //
		/////////////////////////////////////////////////////////////////////////
		if( showInternalAnnotations == 0) {
			AnalysisTools.removePragmas(program, ACCAnnotation.class);
			AnalysisTools.removePragmas(program, ARCAnnotation.class);
		} 
		if( showInternalAnnotations < 3) {
			AnalysisTools.removePragmas(program, CetusAnnotation.class);
		} 
		if( showInternalAnnotations > 1 ){
			List<ACCAnnotation> internalAnnots = IRTools.collectPragmas(program, ACCAnnotation.class, "internal");
			if( internalAnnots != null ) {
				for(ACCAnnotation iAnnot : internalAnnots) {
					iAnnot.setSkipPrint(false);
				}
			}
		}
		//////////////////////////////////
		// Clean empty OpenACC clauses. //
		//////////////////////////////////
		ACCAnalysis.removeEmptyAccClauses(program);
	}

	public void start()
	{
		/////////////////////////////////////////////////////////////////
		// Read command-line options and set corresponding parameters. //
		/////////////////////////////////////////////////////////////////
		String value = null;
		value = Driver.getOptionValue("targetArch");
		if( value != null ) {
			OPENARC_ARCH = Integer.valueOf(value).intValue();
		} else {
			if( env.containsKey("OPENARC_ARCH") ) {
				value = env.get("OPENARC_ARCH");
				OPENARC_ARCH = Integer.valueOf(value).intValue();
			}
		}
		value = Driver.getOptionValue("acc2gpu");
		if( value != null ) {
			if( Integer.valueOf(value).intValue() == 2) {
				opt_GenDistOpenACC = true;
			}
		}
        value = Driver.getOptionValue("omp2acc");
        if( value != null ) {
        	if( Integer.valueOf(value).intValue() == 1 )
        		convertOpenMPtoOpenACC = true;
        }
		value = Driver.getOptionValue("AccAnalysisOnly");
		if( value != null ) {
			AccAnalysisOnly = Integer.valueOf(value).intValue();
		}
		value = Driver.getOptionValue("SkipGPUTranslation");
		if( value != null ) {
			SkipGPUTranslation = Integer.valueOf(value).intValue();
		}
		value = Driver.getOptionValue("AccPrivatization");
		if( value != null ) {
			AccPrivatization = Integer.valueOf(value).intValue();
		} else {
			AccPrivatization = 1;
		}
						
		value = Driver.getOptionValue("AccReduction");
		if( value != null ) {
			AccReduction = Integer.valueOf(value).intValue();
		} else {
			AccReduction = 1;
		}
		
		value = Driver.getOptionValue("AccParallelization");
		if( value != null ) {
			AccParallelization = Integer.valueOf(value).intValue();
		} else {
			AccParallelization = 1;
		}
		
		value = Driver.getOptionValue("programVerification");
		if( value != null ) {
			programVerification = Integer.valueOf(value).intValue();
		}
		
		value = Driver.getOptionValue("enableFaultInjection");
		if( value != null ) {
			enableFaultInjection = true;
			FaultInjectionOption = Integer.valueOf(value).intValue();
		}
		
		value = Driver.getOptionValue("enableCustomProfiling");
		if( value != null ) {
			enableCustomProfiling = true;
		}
		
		value = Driver.getOptionValue("ASPENModelGen");
		if( value != null ) {
			ASPENModelAnalysis.ASPENConfiguration.setConfiguration(value);
			ASPENModelGenMode = ASPENModelAnalysis.ASPENConfiguration.mode;
		}
		
		value = Driver.getOptionValue("showInternalAnnotations");
		if( value != null ) {
			showInternalAnnotations = Integer.valueOf(value).intValue();
		}
		
		value = Driver.getOptionValue("disableStatic2GlobalConversion");
		if( value != null ) {
				disableStatic2GlobalConversion = true;
		}
		
		value = Driver.getOptionValue("useParallelLoopSwap");
		if( value != null ) {
			ParallelLoopSwap = true;
		}
		value = Driver.getOptionValue("gpuMemTrOptLevel");
		if( value != null ) {
			MemTrOptLevel = Integer.valueOf(value).intValue();
		}
		value = Driver.getOptionValue("gpuMallocOptLevel");
		if( value != null ) {
			MallocOptLevel = Integer.valueOf(value).intValue();
		}
		value = Driver.getOptionValue("useGlobalGMalloc");
		if( value != null ) {
			useGlobalGMalloc = true;
		}
		value = Driver.getOptionValue("globalGMallocOpt");
		if( value != null ) {
			globalGMallocOpt = true;
		}
		value = Driver.getOptionValue("genTuningConfFiles");
		if( value != null ) {
			if( value.equals("1") ) {
				PrintTools.println("[INFO] directory to store the generated Tuning configuration files is not specified;" +
						" default directory (tuning_conf) will be used.", 0);
				tuningConfDir="tuning_conf";
			} else {
				tuningConfDir=value;
			}
		}
		value = Driver.getOptionValue("extractTuningParameters");
		if( value != null ) {
			if( value.equals("1") ) {
				PrintTools.println("[INFO] file to store the extracted Tuning parameters is not specified;" +
						" default file name (TuningOptions.txt) will be used.", 0);
				tuningParamFile="TuningOptions.txt";
			} else {
				tuningParamFile=value;
			}
			//Enable passes needed to extract tuning parameters.
			ParallelLoopSwap = true;
		}
		value = Driver.getOptionValue("tuningLevel");
		if( value != null ) {
			tuningLevel = Integer.valueOf(value).intValue();
		}
		value = Driver.getOptionValue("addSafetyCheckingCode");
		if( value != null ) {
			addSafetyCheckingCode = true;
		}
		
		value = Driver.getOptionValue("doNotRemoveUnusedSymbols");
		if( value != null ) {
			doNotRemoveUnusedSymbols = Integer.valueOf(value).intValue();
		}
		value = Driver.getOptionValue("UEPRemovalOptLevel");
		if( value != null ) {
			UEPRemovalOptLevel = Integer.valueOf(value).intValue();
		}
		value = Driver.getOptionValue("MemTrOptOnLoops");
		if( value != null ) {
			MemTrOptOnLoops = true;
		}

		value = Driver.getOptionValue("loopUnrollFactor");
		if( value != null ) {
			unrollFactor = Integer.valueOf(value).intValue();
		}

		value = Driver.getOptionValue("assumeNoAliasingAmongKernelArgs");
		if( value != null ) {
			assumeNoAliasingAmongKernelArgs = true;
		}

		value = Driver.getOptionValue("CloneKernelCallingProcedures");
		if( value != null ) {
			cloneKernelCallingProcedures = Integer.valueOf(value).intValue();
		}

		if( OPENARC_ARCH == 3 ) {
			value = Driver.getOptionValue("defaultNumComputeUnits");
			if( value != null ) {
				tNumComputeUnits = Integer.valueOf(value).intValue();
			}

			value = Driver.getOptionValue("defaultNumSIMDWorkItems");
			if( value != null ) {
				tNumSIMDWorkItems = Integer.valueOf(value).intValue();
			}
		}
		
/*		value = Driver.getOptionValue("tinline");
		if( value != null ) {
			applyInlining = true;
		}*/

        if(convertOpenMPtoOpenACC)
        {
            OMP2ACCTranslator omp2ACCTranslator = new OMP2ACCTranslator(program);
            TransformPass.run(omp2ACCTranslator);
        }

		/*****************************************************************/
		/* cetus.transforms.AnnotationParser stores a OpenACC annotation */
		/* as stand-alone PragmaAnnotation in an AnnotationStatement or  */
		/* AnnotationDeclaration. ACCAnnotationParser() converts this to */
		/* ACCAnnotation and attach it to the next annotatable object.   */
		/*****************************************************************/
		TransformPass.run(new ACCAnnotationParser(program));

		if( AccAnalysisOnly == 1 ) {
			cleanAnnotations();
			return;
		}
		
		TransformPass.run(new PostParserProcessor(program));
		
		/* "int x, y,z;" becomes "int x; int y; int z;" */
		TransformPass.run(new SingleDeclarator(program));
		
		/* "int *pt = (void *)0;" is changed to "int *pt = 0;" */
		/* if "(void *)0" is the value of NULL macro.          */
		/* This is required because C++ only allow 0 for NULL  */
		/* pointer value while C allows both (void *)0 and 0.  */
		TransformTools.NULLPointerCorrection(program);

		/* "int x = b;" becomes "int x; x = b;" */
		TransformPass.run(new DeclarationInitSeparator(program));
		/* 
		 * Update Symbol links of each IDExpression. 
		 * SingleDeclarator modifies some Symbols, and thus affected
		 * IDExpression should be updated.
		 */
		SymbolTools.linkSymbol(program);
		
		TransformPass.run(new NormalizeReturn(program));
		TransformTools.initializeMainReturnVariable(program);
		
/*		if(applyInlining) {
            TransformPass.run(new InlineExpansionPass(program));
		}*/
		
		//Normalize OpenACC gang/worker/vector loops since the following passes such as 
		//ComputeRegionConfAnalysis works only on loops with stride 1.
		TransformPass.run(new ACCWorkshareLoopNormalization(program));
		
		
		//////////////////////////////////////////////////////////////////
		// Convert static variables in a procedure into global one.     //
		// This conversion is necessary for the next procedure cloning. //
		//////////////////////////////////////////////////////////////////
		if( opt_GenDistOpenACC || !disableStatic2GlobalConversion ) {
			TransformPass.run(new ConvertStatic2Global(program));
		}

		////////////////////////////////////////////////////////////////////////////////
		// Clone procedures containing compute regions and are called more than once. //
		////////////////////////////////////////////////////////////////////////////////
		if( cloneKernelCallingProcedures != 0 ) {
			TransformPass.run(new KernelCallingProcCloning(program));
		}
		
		if( AccAnalysisOnly == 2 ) {
			cleanAnnotations();
			return;
		}
		
		if( AccParallelization > 0 ) {
			AnalysisPass.run(new AccParallelization(program, AccParallelization, IRSymbolOnly));
		}

        /////////////////////////////////////////////
        //Loop tiling with tile clauses            //
        /////////////////////////////////////////////
        TransformPass.run(new LoopTilingTransform(program));

		////////////////////////////////////////////
		// Apply parallel loop swap optimization. //
		////////////////////////////////////////////
		if( ParallelLoopSwap ) {
			TransformPass.run(new ParallelLoopSwap(program, 1));
		}

		//////////////////////////////////////////////////////////////////////////////////
		// Preprocess OpenACC loop directives, which includes collapse clause handling, //
		// work-share loop reordering, valid ordering checking, and independent clause  //
		// handling.                                                                    //
		// For nested gang loops, each gang loop is annotated with gangdim(n) internal  //
		// annotation, where n is the number of nested gang loops including the current //
		// gang loop itself.                                                            // 
		// For nested worker loops, workerdim(n) is added to each worker loop too.      //
		//////////////////////////////////////////////////////////////////////////////////
		TransformPass.run(new ACCLoopDirectivePreprocessor(program));

		TransformPass.run(new PipeTransformation(program, OPENARC_ARCH==3));
		
		if( AccAnalysisOnly == 3 ) {
			cleanAnnotations();
			return;
		}
		
		//Create an internal ACCAnnotation, which will contain internal data used
		//for various analysis/transformation passes.
		ACCAnnotation programAnnot = new ACCAnnotation("internal", "_directive");
		programAnnot.setSkipPrint(true);
		ARCAnnotation programCudaAnnot = new ARCAnnotation("cuda", "_directive");
		programCudaAnnot.setSkipPrint(true);

		/* First, call OpenACC analysis pass to update OpenACC annotations
		 * with correct Symbols and add missing, implicit clauses */
		AnalysisPass.run(new ACCAnalysis(program, IRSymbolOnly, programAnnot, programCudaAnnot));
		
		if( ASPENModelGenMode > 0 ) {
			ASPENModelAnalysis.updateSymbolsInASPENAnnotation(program, null);
		}
		
		if( (ASPENModelGenMode == 1) || (ASPENModelGenMode == 3) ) {
			AnalysisPass.run(new ASPENModelAnalysis(program, IRSymbolOnly));
		}
		if( (ASPENModelGenMode >= 1) && (ASPENModelGenMode <= 3) ) {
			if( ASPENModelGenMode == 1 ) {
				TransformPass.run(new ASPENModelGen(program, IRSymbolOnly, false));
			} else {
				TransformPass.run(new ASPENModelGen(program, IRSymbolOnly, true));
			}
			ASPENModelGen.cleanASPENAnnotations(program);
			cleanAnnotations();
			return;
		}
		if( ASPENModelGenMode == 4 ) {
			ASPENModelGen.ASPENPostProcessing(program);
			ASPENModelGen.cleanASPENAnnotations(program);
		}

		if( AccAnalysisOnly == 4 ) {
			cleanAnnotations();
			return;
		}
		
		//[FIXME] This may no be the right position to call this pass; we have to
		//check the interaction of this pass with other passes.
		TransformPass.run(new DataLayoutTransform(program, IRSymbolOnly));
		
		if( AccPrivatization > 0 ) {
			AnalysisPass.run(new AccPrivatization(program, AccPrivatization, IRSymbolOnly));
		}
		
		if( AccReduction > 0 ) {
			AnalysisPass.run(new AccReduction(program, AccReduction, IRSymbolOnly));
		}
		
		
		if( AccAnalysisOnly == 5 ) {
			cleanAnnotations();
			return;
		}
		
		
		////////////////////////////////////////////////////////////////////////
		// Apply transformation to remove upwardly-exposed private variables. //
		////////////////////////////////////////////////////////////////////////
		if( UEPRemovalOptLevel > 0 ) {
			//TransformPass.run(new UEPRemoval(program));
		}
		////////////////////////////////////////////////////////////////
		// If GPU variables are globally allocated, the following     //
		// analysis computes resident GPU variables interprocedurally //
		// to reduce redundant cudaMalloc() and CPU-to-GPU memory     //
		// transfers.                                                 //
		////////////////////////////////////////////////////////////////
		if( globalGMallocOpt ) {
			//AnalysisPass.run(new IpResidentGVariableAnalysis(program));
			//AnalysisPass.run(new IpG2CMemTrAnalysis(program));
		}
		
		if( (doNotRemoveUnusedSymbols != 1) && (doNotRemoveUnusedSymbols != 2) ) {
			PrintTools.println("[removeUnusedProcedures] begin", 0);
			TransformTools.removeUnusedProcedures(program);
			PrintTools.println("[removeUnusedProcedures] end", 0);
		}
		
		//////////////////////////////////////////////
		// Intraprocedural Cuda Malloc optimization //
		//////////////////////////////////////////////
		if( MallocOptLevel > 0 ) {
			//AnalysisPass.run(new CudaMallocAnalysis(program));
		}
		////////////////////////////////////////////////////////
		// Intraprocedural CPU-to-GPU memory transfer         //
		// analysis to identify unnecessary memory transfers. //
		///////////////////////////////////////////////////////////////
		// FIXME: if MemTrOPtLevel <= 1, MemTrAnalysis will not      //
		// be executed. In this case, if the same variable is        //
		// passed for two separate parameters and one is R/O and     //
		// the other is R/W, R/W data should be transfered back      //
		// to CPU later than R/O one. However, current O2GTranslator //
		// does not consider the transfer order; possible wrong      //
		// output.                                                   //
		///////////////////////////////////////////////////////////////
		if( MemTrOptLevel > 1 ) {
			//AnalysisPass.run(new MemTrAnalysis(program));
		}
		
		//TODO: mallochost optimization for pinned host memory allocation.
		
		//After this pass; all kernels regions are loops, but parallel regions 
		//can be either CompoundStatement or loops.
		//If the original kernels region was a compoundstatement, it is changed
		//to a data region, and enclosed gang loops become kernels loops with
		//present data clauses.
		//All internal annotations should be moved to each kernel loops if necessary.
		KernelsSplitting KSPass = new KernelsSplitting(program, IRSymbolOnly);
		TransformPass.run(KSPass);
		
		//////////////////////////////////////////////////////////////
		// Annotate each kernel region with parent procedure name   //
		// and kernel id, and apply user directives if existing.    //
		//////////////////////////////////////////////////////////////
		AnalysisTools.annotateUserDirectives(program, userDirectives, tNumComputeUnits, tNumSIMDWorkItems);
		
		/////////////////////////////////////////////////////
		// Analyze locality of shared variables to exploit //
		// GPU registers and shared memory as caches.      //
		/////////////////////////////////////////////////////
		//[CAUTION] LocalityAnalysis() removes ACCAnnotations without a clause,
		//as a side effect; if any valid directive without a clause exists,
		//such as fault injection directives, the analysis should be modified
		//not to delete that.
		AnalysisPass.run(new LocalityAnalysis(program, true, programAnnot, programCudaAnnot));
		KSPass.updateDataClauses();
		
		//DEBUG: don't put any pass between LoacalityAnalysis pass and the following
		//tuning-parameter-extracting pass, since internal tuningparameters annotation
		//contains string sets for the same cuda clauses (ex: constant, registerRO, etc.). 
		
		/////////////////////////////////
		// Extract tunable parameters. //
		/////////////////////////////////
		if( tuningParamFile != null ) {
			List<HashMap> TuningOptions = storeTuningParameters();
			if( tuningConfDir != null ) {
				double timer = Tools.getTime();
				PrintTools.println("[genTuningConf] begin", 0);
				if( tuningLevel == 1 ) {
					genTuningConfs1(TuningOptions);
				} else {
					genTuningConfs2(TuningOptions);
				}
				PrintTools.println("[genTuningConf] end in " +
						String.format("%.2f seconds", Tools.getTime(timer)), 0);
			}
		}
		
		//GlobalVariableParameterization for global variables accessed 
		//in a function called in a compute region.
		//ACCAnalysis.shared_analysis() put all global symbols accessed in called functions,
		//but these should be explicitly passed as parameters to the functions.
		TransformPass.run(new GlobalVariableParameterization(program, true, assumeNoAliasingAmongKernelArgs));
		
		//Normalize OpenACC gang/worker/vector loops since the following passes such as 
		//ComputeRegionConfAnalysis works only on loops with stride 1.
		//[DEBUG] This pass should be called before privatization pass; moved.
		//TransformPass.run(new ACCWorkshareLoopNormalization(program));

        //Kernel configuration calculation, stored in gang/num_gangs clauses.
		//If maxNumGangs is specified, min(maxNumGangs, calculatedNumGangs) is applied.
		AnalysisPass.run(new CompRegionConfAnalysis(program, OPENARC_ARCH));
		
		//Reduction malloc-point analysis
		//internal rcreate(list) will specify the reduction malloc point for the given reduction symbols.
		//Separate host variable => GPU reduction variable mapping is needed 
		//(<= No, since currently this is intraprocedural opt.)
		//AnalysisPass.run(new ReductionMallocAnalysis(program, true));
		
		//If resilience directive contains repeat clause, programVerification option should be set to 2.
		if( enableFaultInjection ) {
			List<ACCAnnotation> resAnnots = IRTools.collectPragmas(program, ACCAnnotation.class, "resilience");
			if( resAnnots != null ) {
				boolean containsRepeatClause = false;
				for( ACCAnnotation rAnnot : resAnnots ) {
					if( rAnnot.containsKey("repeat") ) {
						Expression ftcond = rAnnot.get("ftcond");
						if( (ftcond ==null) || !(ftcond instanceof IntegerLiteral) 
								|| (((IntegerLiteral)ftcond).getValue() != 0) ) {
							containsRepeatClause = true;
							break;
						}
					}
				}
				if( containsRepeatClause ) {
					Driver.setOptionValue("programVerification", "2");
					programVerification = 2;
				}
			}
		}
		
		if( programVerification == 1 ) {
			//Verify CPU-GPU data communications.
			TransformPass.run(new MemTrVerifyTransformation(program, IRSymbolOnly));
		} else if( programVerification == 2 ) {
			//Verify GPU kernels by comparing GPU results against CPU results.
			//For this, demote data clauses from data regions to compute regions, and disable
			//all existing declare/update directives.
			TransformPass.run(new kernelVerifyTransformation(program, IRSymbolOnly));
		}
		
		if( enableFaultInjection ) {
			TransformPass.run(new FaultInjectionTransformation(program, IRSymbolOnly, FaultInjectionOption, OPENARC_ARCH));
		}
		
		if( enableCustomProfiling ) {
			TransformPass.run(new CustomProfilingTransformation(program, IRSymbolOnly));
		}
		
		
		
		if( SkipGPUTranslation == 1 ) {
			cleanAnnotations();
			return;
		} else {
			//////////////////////////////////
			// Clean empty OpenACC clauses. //
			//////////////////////////////////
			ACCAnalysis.removeEmptyAccClauses(program);
		}
		
		//Wrap code sections in worker-single mode, which is necessary for correct GPU kernel
		//transformation.
		//Insert #pragma acc barrier at the end of pure-worker loop if it does not
		//have reduction clause and it is not the last worker loop.
		//If nowait clause exists, do not insert the barrier.
		TransformPass.run(new WorkerSingleModeTransformation(program, OPENARC_ARCH));
		
		/////////////////////////////////////////////
		//Collapse loops with collapse(n) clauses. //
		/////////////////////////////////////////////
		TransformPass.run(new CollapseTransformation(program, false));


		/////////////////////////////////////////////////////////
		//Unroll loops if unrollFactor is positive integer.    //
		//(default value = 1, which will unroll only if unroll //
		//OpenARC clause exists.)                              //
		/////////////////////////////////////////////////////////
		if(unrollFactor > 0) {
			TransformPass.run(new LoopUnrollTransformation(program, unrollFactor));
		}

		if( SkipGPUTranslation == 2 ) {
			cleanAnnotations();
			return;
		}
		///////////////////////////////////////////////////////
		// Do the actual OpenACC-to-Accelerator translation. //
		///////////////////////////////////////////////////////
        if( OPENARC_ARCH == 0 )
        {
            PrintTools.println("[ACC2CUDATranslator] begin", 0);
            ACC2CUDATranslator a2cpass = new ACC2CUDATranslator(program);
            a2cpass.start();
            PrintTools.println("[ACC2CUDATranslator] end", 0);
        }
        else
        {
            PrintTools.println("[ACC2OPENCLTranslator] begin", 0);
            ACC2OPENCLTranslator a2cpass = new ACC2OPENCLTranslator(program);
            a2cpass.start();
            PrintTools.println("[ACC2OPENCLTranslator] end", 0);
            PrintTools.println("[OpenCLArrayFlattener] begin", 0);
            OpenCLArrayFlattener arrayFlattener = new OpenCLArrayFlattener(program, assumeNoAliasingAmongKernelArgs);
            arrayFlattener.start();
            PrintTools.println("[OpenCLArrayFlattener] end", 0);
            PrintTools.println("[OpenCLTranslationTools] begin", 0);
            OpenCLKernelTranslationTools openclTool = new OpenCLKernelTranslationTools(program);
            openclTool.start();
            PrintTools.println("[OpenCLTranslationTools] end", 0);
        }

		///////////////////////////////////////////////////////////////////////////////
		// After kernel transformation, original functions called in a kernel region //
		// may not be used; delete these unused functions.                           //
		///////////////////////////////////////////////////////////////////////////////
		if( (doNotRemoveUnusedSymbols != 1) && (doNotRemoveUnusedSymbols != 2) ) {
			PrintTools.println("[removeUnusedProcedures] begin", 0);
			TransformTools.removeUnusedProcedures(program);
			PrintTools.println("[removeUnusedProcedures] end", 0);
		}

		// Update symbol pointers again. 
		SymbolTools.linkSymbol(program);

		///////////////////////////////////////////////////////////////////////////////
		// After kernel transformation, kernel-containing procedures may have unused //
		// symbols; delete these unused symbols.                                     //
		///////////////////////////////////////////////////////////////////////////////
		if( (doNotRemoveUnusedSymbols != 1) && (doNotRemoveUnusedSymbols != 3) ) {
			PrintTools.println("[removeUnusedSymbols] begin", 0);
			TransformTools.removeUnusedSymbols(program);
			PrintTools.println("[removeUnusedSymbols] end", 0);
		}
		/////////////////////////////////////////////////////////////////////////
		// Remove device function declarations in host-side translation units. //
		/////////////////////////////////////////////////////////////////////////
		if( doNotRemoveUnusedSymbols != 1 ) {
			PrintTools.println("[removeUnusedDeviceFunctionDeclarations] begin", 0);
			TransformTools.removeUnusedDeviceFunctionDeclarations(program);
			PrintTools.println("[removeUnusedDeviceFunctionDeclarations] end", 0);
		}

		/////////////////////////////////////////////////////////////////////////
		// Check whether a kernel function or a split parallel region contains //
		// upward-exposed private variables.                                   //
		/////////////////////////////////////////////////////////////////////////
		//AnalysisPass.run( new UEPrivateAnalysis(program) );

		//////////////////////////////////////////////////////////////////////////////
		// Check whether a CUDA kernel function calls C standard library functions  //
		// that are not supported by CUDA runtime systems.                          //
		// If so, CUDA compiler will fail if they are not inlinable.                //
		//////////////////////////////////////////////////////////////////////////////
		PrintTools.println("[CheckKernelFunctions] begin", 0);
        if( OPENARC_ARCH == 0 ) {
        	AnalysisTools.checkKernelFunctions(program, "CUDA");
        } else {
        	AnalysisTools.checkKernelFunctions(program, "OPENCL");
        }
		PrintTools.println("[CheckKernelFunctions] end", 0);

		cleanAnnotations();

		//////////////////////////////////////////////
		// Rename output filenames from *.c to *.cu //
		//////////////////////////////////////////////
		renameOutputFiles(program);
	}

	private void renameOutputFiles(Program program ) {
		for ( Traversable tt : program.getChildren() )
		{
			TranslationUnit tu = (TranslationUnit)tt;
			//DEBUG: tu.getInputFilename() may include path name, but default TranslationUnit.output_filename
			//does not have path name.
			//String iFileName = tu.getInputFilename();
			String iFileName = tu.getOutputFilename();
			int dot = iFileName.lastIndexOf(".c");
			if( dot == -1 ) {
				dot = iFileName.lastIndexOf(".h");
				if( dot != -1 ) {
					PrintTools.println("\n[WARNING] Translating a header file , " + iFileName + 
							", makes all macros in the header file inlined with values defined in this O2G tranlsation, " + 
							"which will cause problems if NVCC compiles the output codes with different macro definitions or " +
							"may cause redundant including of this header file.\n", 0);
				} else {
					PrintTools.println("\n[WARNING] Input file name, " + iFileName + 
							", does not end with C suffix (.c); " +
							"translated output file may behave incorrectly if macros used in " +
							"this O2G translation is different from the ones that will be used in NVCC.\n", 0);
				}
				continue;
			} else {
				String suffix = iFileName.substring(dot);
				if( !suffix.equalsIgnoreCase(".cu") && !suffix.equalsIgnoreCase(".cl")) {
					String fNameStem = iFileName.substring(0, dot);
					//DEBUG: In the new TranslationUnit, output_filename is a file name without path name.
					//String oFileName = Driver.getOptionValue("outdir") + File.separatorChar + 
					//String oFileName = fNameStem.concat(".cu");
                    String oFileName = fNameStem.concat(".cpp");
					tu.setOutputFilename(oFileName);
				}
			}
		}
	}
	
	////////////////////////////////////////////////////////////////////////////
	//Array of local caching-related parameters inserted by LocalityAnalysis. //
	////////////////////////////////////////////////////////////////////////////
	String[] cachingParams = {"tmp_registerRO", "tmp_registerRW", "tmp_sharedRO", "tmp_sharedRW", "tmp_texture",
			"ROShSclrNL", "ROShSclr", "RWShSclr", "ROShArEl", "RWShArEl", "RO1DShAr",
			"PrvAr", "SclrConst", "ArryConst", "tmp_constant"};
	String[] uDirectives = {"tmp_registerRO", "tmp_registerRW", "tmp_sharedRO", "tmp_sharedRW", "tmp_texture", "tmp_constant"};
	
	protected List<HashMap> storeTuningParameters() {
		/////////////////////////////////////////////////////////////////////////////////////////
		//Assume that AnalysisTools.annotateUserDirectives() annotates each kernel region with //
		//a ACCAnnotaion containing "ainfo" clause.                                           // 
		/////////////////////////////////////////////////////////////////////////////////////////
		List<ARCAnnotation> cAnnotList = IRTools.collectPragmas(program, ARCAnnotation.class, "ainfo");
		if( cAnnotList == null ) {
			PrintTools.println("\n[WARNING in storeTuningParameters()] couldn't find any kernel region " +
					"containing acc ainfo clause.\n",0);
			return null;
		}
		
		HashSet<String> gOptionSet1 = new HashSet<String>();
		HashSet<String> gOptionSet2 = new HashSet<String>();
		HashSet<String> gOptionSet3 = new HashSet<String>();
		HashSet<String> gOptionSet4 = new HashSet<String>();
		HashSet<String> gOptionSet5 = new HashSet<String>();
		HashSet<String> gOptionSet6 = new HashSet<String>();
		HashMap<String, HashSet<String>> gOptionMap = new HashMap<String, HashSet<String>>();
		/////////////////////////////////////////////////////////////////////////////////////////////////
		// DEBUG: equal() and hashCode() methods of all PragmaAnnotations use Java.utilMap.entrySet(), //
		// and thus if two PragmaAnnotations contains same string entrySets, both hashCode() will      //
		// return the same output. However, below kOptionMap is OK, since ACCAnnotations used for     //
		// keys are unique; each kernel region has a unique ainfo directive.                           //
		/////////////////////////////////////////////////////////////////////////////////////////////////
		HashMap<ARCAnnotation, HashMap<String, Object>> kOptionMap = 
			new HashMap<ARCAnnotation, HashMap<String, Object>>();
		List<HashMap> TuningOptions = new LinkedList<HashMap>();
		TuningOptions.add(gOptionMap);
		TuningOptions.add(kOptionMap);
		LoopCollapse lcHandler = new LoopCollapse(program);
		BufferedWriter out = null;
		try {
			out = new BufferedWriter(new FileWriter(tuningParamFile));
			///////////////////////////
			// Check global options. //
			///////////////////////////
			boolean MatrixTransposeApplicable = false;
			boolean MallocPitchApplicable = false;
			boolean LoopCollapseApplicable = false;
			boolean PLoopSwapApplicable = false;
			boolean UnrollingReductionApplicable = false;
			boolean ReductionPatternExists = false;
			boolean shrdSclrCachingOnReg = false;
			boolean shrdArryElmtCachingOnReg = false;
			boolean shrdSclrCachingOnSM = false;
			boolean prvtArryCachingOnSM = false;
			boolean shrdArryCachingOnTM = false;
			boolean shrdSclrCachingOnConst = false;
			boolean shrdArryCachingOnConst = false;
			for( ARCAnnotation cAnnot : cAnnotList ) {
				Annotatable at = cAnnot.getAnnotatable();
				ACCAnnotation oAnnot = at.getAnnotation(ACCAnnotation.class, "accshared");
				if( oAnnot != null ) {
					// Check whether MallocPitch is applicable.
					// FIXME: this assumes no access symbols in the accshared set.
					Set<Symbol> sharedSyms = oAnnot.get("accshared");
					if( (sharedSyms != null) && !MallocPitchApplicable ) {
						for( Symbol sym : sharedSyms ) {
							if(SymbolTools.isArray(sym)) {
								List aspecs = sym.getArraySpecifiers();
								ArraySpecifier aspec = (ArraySpecifier)aspecs.get(0);
								if( sym instanceof NestedDeclarator ) {
									if( aspec.getNumDimensions() == 1 ) {
										MallocPitchApplicable = true;
										break;
									}
								} else {
									if( aspec.getNumDimensions() == 2 ) {
										MallocPitchApplicable = true;
										break;
									}
								}
							}
						}
					}
				}
				// Check whether MatrixTranspose is applicable.
				// DEBUG: temporarily disabled since no transformation pass is available.
/*				Set<Symbol> thPrivSyms = oAnnot.get("accprivate");
				if( (thPrivSyms != null) && !MatrixTransposeApplicable ) {
					for( Symbol sym : thPrivSyms ) {
						if(SymbolTools.isArray(sym)) {
							List aspecs = sym.getArraySpecifiers();
							ArraySpecifier aspec = (ArraySpecifier)aspecs.get(0);
							if( aspec.getNumDimensions() == 1 ) {
								MatrixTransposeApplicable = true;
								break;
							}
						} else if( SymbolTools.isPointer(sym) ) {
							MatrixTransposeApplicable = true;
							break;
						}
					}
				}*/
			}
			////////////////////////////////////////////////////////////////////////
			// Check whether input program contains acc reduction clauses or not. //
			////////////////////////////////////////////////////////////////////////
			List<ACCAnnotation> oAnnotList = IRTools.collectPragmas(program, ACCAnnotation.class, "reduction");
			if( (oAnnotList != null) && !oAnnotList.isEmpty() ) {
				ReductionPatternExists = true;
			}
			StringBuilder str = new StringBuilder(1024);
			str.append("#############################\n");
			str.append("# Applicable global options #\n");
			str.append("##########################################\n");
			str.append("# Safe, always-beneficial options, but   #\n");
			str.append("# resource may limit their applications. #\n");
			str.append("##########################################\n");
			str.append("#pragma optionType1\n");
			//str.append("localRedVarConf = 1 (use 0 if shared memory overflows)\n");
			//gOptionSet1.add("localRedVarConf");
			if( MallocPitchApplicable ) {
				str.append("useMallocPitch\n");
				gOptionSet1.add("useMallocPitch");
			}
			if( MatrixTransposeApplicable ) {
				str.append("useMatrixTranspose\n");
				gOptionSet1.add("useMatrixTranspose");
			}
			gOptionMap.put("optionType1", gOptionSet1);
			////////////////////////////////////
			// Check kernel specific options. //
			////////////////////////////////////
			StringBuilder str2 = new StringBuilder(1024);
			str2.append("\n");
			str2.append("###########################\n");
			str2.append("# Kernel-specific options #\n");
			str2.append("###########################\n");
			for( ARCAnnotation cAnnot : cAnnotList ) {
				Annotatable at = cAnnot.getAnnotatable();
				ARCAnnotation aAnnot = at.getAnnotation(ARCAnnotation.class, "ainfo");
				str2.append(aAnnot.toString()+"\n");
				HashMap<String, Object> kMap = new HashMap<String, Object>();
				//Add maxNumGangs(numgangs) parameters
				//[DEBUG] for now, maxNumGangs is applied globally, but not in each kernel.
				//str2.append("maxNumGangs(numgangs)\n");
				//kMap.put("maxNumGangs", "_clause");
				//Check LoopCollapse parameters.
				if( lcHandler.handleSMVP((Statement)at, true) ) {
					str2.append("loopcollapse\n");
					LoopCollapseApplicable = true;
					ReductionPatternExists = true;
					kMap.put("loopcollapse", "_clause");
				}
				//Check reduction-unrolling parameters
				ACCAnnotation iAnnot = at.getAnnotation(ACCAnnotation.class, "accreduction");
				Set<Symbol> redSyms = new HashSet<Symbol>();
				if( iAnnot != null ) {
					redSyms.addAll((Set<Symbol>)iAnnot.get("accreduction"));
				}
				if( redSyms.size() > 0 ) {
					str2.append("noreductionunroll(" + AnalysisTools.symbolsToString(redSyms, ",") + ")\n");
					UnrollingReductionApplicable = true;
					kMap.put("noreductionunroll", AnalysisTools.symbolsToStringSet(redSyms));
				}
				//"tuningparameter" clause is used only internally .
				ACCAnnotation tAnnot = at.getAnnotation(ACCAnnotation.class, "tuningparameters");
				if( tAnnot != null ) {
					//Check ParallelLoopSwap parameters.
					Object obj = tAnnot.get("ploopswap");
					if( obj != null ) {
						str2.append("ploopswap\n");
						PLoopSwapApplicable = true;
						kMap.put("ploopswap", "_clause");
					}
					//Check local caching-related parameters.
					Set<String> clauses = new HashSet<String>(Arrays.asList(cachingParams));
					Set<String> uDirSet = new HashSet<String>(Arrays.asList(uDirectives));
					for( String clause : clauses ) {
						Set<String> symbols = tAnnot.get(clause); 
						if( symbols != null ) {
							if( uDirSet.contains(clause) ) {
								str2.append(clause.substring(4) + "(" + PrintTools.collectionToString(symbols, ",") + ")\n");
							} else {
								kMap.put(clause, new HashSet<String>(symbols));
								if( clause.equals("ROShSclrNL") ) {
									shrdSclrCachingOnSM = true;
								}
								if( clause.equals("ROShSclr") || clause.equals("RWShSclr") ) {
									shrdSclrCachingOnReg = true;
									shrdSclrCachingOnSM = true;
								}
								if( clause.equals("ROShArEl") || clause.equals("RWShArEl") ) {
									shrdArryElmtCachingOnReg = true;
								}
								if( clause.equals("RO1DShAr") ) {
									shrdArryCachingOnTM = true;
								}
								if( clause.equals("PrvAr") ) {
									prvtArryCachingOnSM = true;
								}
								if( clause.equals("SclrConst") ) {
									shrdSclrCachingOnConst = true;
								}
								if( clause.equals("ArryConst") ) {
									shrdArryCachingOnConst = true;
								}
							}
						}
					}
					//Remove ACCAnnotation containing tuning parameters.
					List<ACCAnnotation> cudaAnnots = at.getAnnotations(ACCAnnotation.class);
					at.removeAnnotations(ACCAnnotation.class);
					for( ACCAnnotation annot : cudaAnnots ) {
						if( !annot.containsKey("tuningparameters") ) {
							at.annotate(annot);
						}
					}
				}
				str2.append("\n");
				kOptionMap.put((ARCAnnotation)aAnnot.clone(), kMap);
			}
			str.append("\n");
			str.append("#################################\n");
			str.append("# May-beneficial options, which #\n");
			str.append("# interact with other options.  #\n");
			str.append("#################################\n");
			str.append("#pragma optionType2\n");
			str.append("defaultNumWorkers=N\n");
			gOptionSet2.add("defaultNumWorkers");
			str.append("maxNumGangs=N\n");
			gOptionSet2.add("maxNumGangs");
			str.append("defaultNumComputeUnits=N\n");
			gOptionSet2.add("defaultNumComputeUnits");
			str.append("defaultNumSIMDWorkItems=N\n");
			gOptionSet2.add("defaultNumSIMDWorkItems");
			if( ReductionPatternExists ) {
				str.append("localRedVarConf=1 (use 0 if shared memory overflows)\n");
				gOptionSet2.add("localRedVarConf");
			}
			if( LoopCollapseApplicable ) {
				str.append("useLoopCollapse\n");
				gOptionSet2.add("useLoopCollapse");
			}
			if( PLoopSwapApplicable ) {
				str.append("useParallelLoopSwap\n");
				gOptionSet2.add("useParallelLoopSwap");
			}
			if( UnrollingReductionApplicable ) {
				str.append("useUnrollingOnReduction\n");
				gOptionSet2.add("useUnrollingOnReduction");
			}
			gOptionMap.put("optionType2", gOptionSet2);
			str.append("\n");
			str.append("#############################################\n");
			str.append("# Always-beneficial options, but inaccuracy #\n");
			str.append("# of the analysis may break correctness.    #\n");
			str.append("#############################################\n");
			str.append("#pragma optionType3\n");
			//DEBUG: Temporarily disabled.
/*			str.append("gpuMemTrOptLevel=N\n");
			str.append("gpuMallocOptLevel=N\n");
			gOptionSet3.add("gpuMemTrOptLevel");
			gOptionSet3.add("gpuMallocOptLevel");*/
			gOptionMap.put("optionType3", gOptionSet3);
			str.append("\n");
			str.append("#########################################\n");
			str.append("# Always-beneficial options, but user's #\n");
			str.append("# approval is required.                 #\n");
			str.append("#########################################\n");
			str.append("#pragma optionType4\n");
			str.append("assumeNonZeroTripLoops\n");
			str.append("assumeNoAliasingAmongKernelArgs\n");
			str.append("skipKernelLoopBoundChecking\n");
			gOptionSet4.add("assumeNonZeroTripLoops");
			gOptionSet4.add("assumeNoAliasingAmongKernelArgs");
			gOptionSet4.add("skipKernelLoopBoundChecking");
			gOptionMap.put("optionType4", gOptionSet4);
			str.append("\n");
			str.append("##############################################################\n");
			str.append("# May-beneficial options, which interact with other options. #\n");
			str.append("# These options are not needed if kernel-specific options    #\n");
			str.append("# are used.                                                  #\n");
			str.append("##############################################################\n");
			str.append("#pragma optionType5\n");
			if( shrdSclrCachingOnReg ) {
				str.append("shrdSclrCachingOnReg\n");
				gOptionSet5.add("shrdSclrCachingOnReg");
			}
			if( shrdSclrCachingOnSM ) {
				str.append("shrdSclrCachingOnSM\n");
				gOptionSet5.add("shrdSclrCachingOnSM");
			}
			if( shrdArryElmtCachingOnReg ) {
				str.append("shrdArryElmtCachingOnReg\n");
				gOptionSet5.add("shrdArryElmtCachingOnReg");
			}
			if( shrdArryCachingOnTM ) {
				str.append("shrdArryCachingOnTM\n");
				gOptionSet5.add("shrdArryCachingOnTM");
			}
			if( prvtArryCachingOnSM ) {
				str.append("prvtArryCachingOnSM\n");
				gOptionSet5.add("prvtArryCachingOnSM");
			}
			if( shrdSclrCachingOnConst ) {
				str.append("shrdSclrCachingOnConst\n");
				gOptionSet5.add("shrdSclrCachingOnConst");
			}
			if( shrdArryCachingOnConst ) {
				str.append("shrdArryCachingOnConst\n");
				gOptionSet5.add("shrdArryCachingOnConst");
			}
			gOptionMap.put("optionType5", gOptionSet5);
			str.append("\n");
			str.append("######################################################\n");
			str.append("# Non-tunable options, but user may need to apply    #\n");
			str.append("# some of these either to generate correct output    #\n");
			str.append("# code or to apply some user-assisted optimizations. #\n");
			str.append("######################################################\n");
			str.append("#pragma optionType6\n");
			str.append("UEPRemovalOptLevel=N, where N=1,2,or 3\n");
			//gOptionSet6.add("UEPRemovalOptLevel");
			str.append("forceSyncKernelCall\n");
			//gOptionSet6.add("forceSyncKernelCall");
			str.append("doNotRemoveUnusedSymbols\n");
			//gOptionSet6.add("doNotRemoveUnusedSymbols");
			//str.append("MemTrOptOnLoops\n");
			//gOptionSet6.add("MemTrOptOnLoops");
			str.append("AccPrivatization=N\n");
			//gOptionSet6.add("AccPrivatization");
			str.append("AccReduction=N\n");
			//gOptionSet6.add("AccReduction");
			gOptionMap.put("optionType6", gOptionSet6);
			out.write(str.toString());
			out.write(str2.toString());
			out.close();
		} catch (Exception e) {
			PrintTools.println("Creaing a file, "+ tuningParamFile + ", failed; " +
					"tuning parameters can not be saved.", 0);
		}
		return TuningOptions;
	}
	
	///////////////////////////////////////////////////////////////
	// Safe, always-beneficial global options, but resources may // 
	// limit their applications.                                 //
	///////////////////////////////////////////////////////////////
	static private String[] gOptions1 = {"useMatrixTranspose", "useMallocPitch"};
	
	///////////////////////////////////////////////////////////////
	// May-beneficial global options, which interact with other  //
	// options.                                                  // 
	///////////////////////////////////////////////////////////////
	static private String[] gOptions2 = {"defaultNumWorkers", "useLoopCollapse", 
		"useParallelLoopSwap", "useUnrollingOnReduction", "maxNumGangs",
		"localRedVarConf"};
	
	//////////////////////////////////////////////////////////////////////
	// Always-beneficial global options, but inaccuracy of the analysis //
	// may break correctness.                                           //
	//////////////////////////////////////////////////////////////////////
	static private String[] gOptions3 = {"gpuMallocOptLevel", "gpuMemTrOptLevel"};
	
	//////////////////////////////////////////////////////////////
	// Always-beneficial global options, but user's approval is // 
	// required.                                                //
	//////////////////////////////////////////////////////////////
	static private String[] gOptions4 = {"assumeNonZeroTripLoops", "assumeNoAliasingAmongKernelArgs",
		"skipKernelLoopBoundChecking"};
	
	///////////////////////////////////////////////////////////////
	// May-beneficial global options, which interact with other  //
	// options. These options are not needed if kernel-specific  // 
	// options are used.                                         //
	///////////////////////////////////////////////////////////////
	static private String[] gOptions5 = {"shrdSclrCachingOnReg", "shrdSclrCachingOnSM", 
		"shrdArryElmtCachingOnReg", "shrdArryCachingOnTM", "prvtArryCachingOnSM",
		"shrdSclrCachingOnConst", "shrdArryCachingOnConst"};
	
	/////////////////////////////////////////////////////////////////////////
	// Non-tunable options, but user may add these for correctness purpose //
	// or to apply user-assisted, unsafe optimizations                     //
	/////////////////////////////////////////////////////////////////////////
	static private String[] gOptions6 = {"UEPRemovalOptLevel", "forceSyncKernelCall", "doNotRemoveUnusedSymbols",
		"MemTrOptOnLoops", "AccPrivatization", "AccReduction"};
	
	///////////////////////////////////////////////////////////////////////
	// Global options that will be always applied unless explicitly      //
	// excluded by a user using excludedGOptionSet option.               //
	// These options are applied for both Program-level and Kernel-level //
	// tunings even if the user does not specify in defaultGOptionSet.   //
	///////////////////////////////////////////////////////////////////////
	//DEBUG: Temporarily disabled.
	//static private String[] defaultGOptions0 = { 
	//	"gpuMallocOptLevel",  "gpuMemTrOptLevel"}; 
	static private String[] defaultGOptions0 = {};
	/////////////////////////////////////////////////////////////////
	// Global options that will be applied by default if existing. //
	// (Below options are used for Program-level tuning.)          //
	/////////////////////////////////////////////////////////////////
	//DEBUG: Temporarily disabled.
	//static private String[] defaultGOptions1 = {
		//"gpuMallocOptLevel", "gpuMemTrOptLevel", "defaultNumWorkers"};
	static private String[] defaultGOptions1 = {"defaultNumWorkers"};
	//////////////////////////////////////////////////////////////////////////////
	// Global options that will be applied by default if existing.              //
	// (Below options are used for GPU-kernel-level tuning.)                    //
	// DEBUG: useLoopCollapse, useParallelLoopSwap, and useUnrollingOnReduction //
	// optimizations may not be always beneficial, but added by default, since  //
	// they can be controlled by user directives.                               //
	//////////////////////////////////////////////////////////////////////////////
	//DEBUG: Temporarily disabled.
	//static private String[] defaultGOptions2 = {
		//"gpuMallocOptLevel", "gpuMemTrOptLevel", "useLoopCollapse", "useParallelLoopSwap", 
		//"useUnrollingOnReduction", "defaultNumWorkers"};
//	static private String[] defaultGOptions2 = {
//		"useLoopCollapse", "useParallelLoopSwap", 
//		"useUnrollingOnReduction", "defaultNumWorkers"};
	static private String[] defaultGOptions2 = {
		"useUnrollingOnReduction", "defaultNumWorkers"};
	////////////////////////////////////////////////////////////////////////
	// Global options that will be always applied if existing and unless  //
	// explicitly excluded by a user using excludedGOptionSet option.     //
	// These options are applied for Kernel-level tuning even if the user //
	// does not specify in defaultGOptionSet. These options are always    //
	// applied since these can be overwritten by user directives.         //
	////////////////////////////////////////////////////////////////////////
//	static private String[] defaultGOptions3 = { "useLoopCollapse", 
	//	"useParallelLoopSwap", "useUnrollingOnReduction"};
	static private String[] defaultGOptions3 = { "useUnrollingOnReduction"};
	
	static private String[] defaultNumWorkers = {"32", "64", "128", "256", "384"};
	static private String[] defaultNumGangs = {"NONE"};
	static private String[] defaultNumComputeUnits = {"NONE"};
	static private String[] defaultNumSIMDWorkItems = {"NONE"};
	
	/**
	 * Program-level tuning configuration output generator.
	 * 
	 * @param TuningOptions applicable tuning options suggested by O2G translator
	 */
	protected void genTuningConfs1( List<HashMap> TuningOptions ) {
		if( TuningOptions.size() != 2 ) {
			PrintTools.println("[ERROR in genTuningConfs()] input TuningOptions list does not contain enough data.", 0);
			return;
		}
		
	    /* make sure the tuning-configuration directory exists */
	    File dir = null;
	    File fname = null;
	    try {
	      dir = new File(".");
	      fname = new File(dir.getCanonicalPath(), tuningConfDir);
	      if (!fname.exists())
	      {
	        if (!fname.mkdir())
	          throw new IOException("mkdir failed");
	      }
	    } catch (Exception e) {
	      System.err.println("cetus: could not create tuning-configuration directory, " + e);
	      System.exit(1);
	    }
	    
	    PrintTools.println("Generate configuration files for program-level tuning", 0);
	    String dirPrefix = fname.getAbsolutePath() + File.separatorChar;
	    
		HashMap<String, HashSet<String>> gOptionMap = 
			(HashMap<String, HashSet<String>>)TuningOptions.get(0);
		HashMap<ARCAnnotation, HashMap<String, Object>> kOptionMap = 
			(HashMap<ARCAnnotation, HashMap<String, Object>>)TuningOptions.get(1);
		HashMap<String, List<Boolean>> gOptMap = new HashMap<String, List<Boolean>>();
		HashSet<String> confSet = new HashSet<String>();
		HashSet<String> gOptions = new HashSet<String>();
		gOptions.addAll(Arrays.asList(gOptions1));
		gOptions.addAll(Arrays.asList(gOptions2));
		gOptions.addAll(Arrays.asList(gOptions3));
		gOptions.addAll(Arrays.asList(gOptions4));
		gOptions.addAll(Arrays.asList(gOptions5));
		gOptions.addAll(Arrays.asList(gOptions6));
		
		//////////////////////////////////
		// Check default configurations //
		//////////////////////////////////
		HashSet<String> defaultGOption = null;
		HashSet<String> excludedGOption = null;
		String memTrOptValue = "2";
		String mallocOptValue = "0";
		String UEPRemovalOptValue = "3";
		String AccPrivatizationValue = "1";
		String AccReductionValue = "1";
		Set<String> numWorkersSet = null;
		Set<String> maxNumGangsSet = null;
		Set<String> numComputeUnitsSet = null;
		Set<String> numSIMDWorkItemsSet = null;
		if( tuningConfigs == null || tuningConfigs.isEmpty() ) {
			defaultGOption = new HashSet<String>(Arrays.asList(defaultGOptions1));
			excludedGOption = new HashSet<String>();
			numWorkersSet = new HashSet<String>(Arrays.asList(defaultNumWorkers));
			maxNumGangsSet = new HashSet<String>(Arrays.asList(defaultNumGangs));
			numComputeUnitsSet = new HashSet<String>(Arrays.asList(defaultNumComputeUnits));
			numSIMDWorkItemsSet = new HashSet<String>(Arrays.asList(defaultNumSIMDWorkItems));
		} else {
			defaultGOption = (HashSet<String>)tuningConfigs.get("defaultGOptionSet");
			if( defaultGOption == null ) {
				defaultGOption = new HashSet<String>(Arrays.asList(defaultGOptions1));
			} else {
				//Check whether defaultGOption contains illegal options or not.
				for( String gOpt : defaultGOption ) {
					if( !gOptions.contains(gOpt) ) {
						PrintTools.println("\n[WARNING in genTuningConfs()] defaultGOptions set contains " +
								"unsupported option: " + gOpt + ".\n", 0);
					}
				}
			}
			excludedGOption = (HashSet<String>)tuningConfigs.get("excludedGOptionSet");
			if( excludedGOption == null ) {
				excludedGOption = new HashSet<String>();
			} else {
				//Check whether excludedGOption contains illegal options or not.
				for( String gOpt : excludedGOption ) {
					if( !gOptions.contains(gOpt) ) {
						PrintTools.println("\n[WARNING in genTuningConfs()] excludedGOption set contains " +
								"unsupported option: " + gOpt + ".\n", 0);
					}
				}
			}
			//////////////////////////////////////////////////////////////////
			// Options in defaultGOptions0 are always applied unless a user //
			// explicitly excludes using excludedGOptionSet option.         //
			//////////////////////////////////////////////////////////////////
			for( String gOpt : defaultGOptions0 ) {
				if( !excludedGOption.contains(gOpt) ) {
					defaultGOption.add(gOpt);
				}
			}
			numWorkersSet = AnalysisTools.expressionsToStringSet((Set<Expression>)tuningConfigs.get("defaultNumWorkersSet"));
			if( (numWorkersSet == null) || numWorkersSet.isEmpty() ) {
				numWorkersSet = new HashSet<String>(Arrays.asList(defaultNumWorkers));
			} else if( !excludedGOption.contains("defaultNumWorkers") ) {
				////////////////////////////////////////////////////////////
				// If defaultNumWorkers option is used, defaultNumWorkers //
				// option is always included.                             //
				////////////////////////////////////////////////////////////
				defaultGOption.add("defaultNumWorkers");
			}
			maxNumGangsSet = AnalysisTools.expressionsToStringSet((Set<Expression>)tuningConfigs.get("maxNumGangsSet"));
			if( (maxNumGangsSet == null) || maxNumGangsSet.isEmpty() ) {
				maxNumGangsSet = new HashSet<String>(Arrays.asList(defaultNumGangs));
			}
			numComputeUnitsSet = AnalysisTools.expressionsToStringSet((Set<Expression>)tuningConfigs.get("defaultNumComputeUnits"));
			if( (numComputeUnitsSet == null) || numComputeUnitsSet.isEmpty() ) {
				numComputeUnitsSet = new HashSet<String>(Arrays.asList(defaultNumComputeUnits));
			} else if( !excludedGOption.contains("defaultNumComputeUnits") ) {
				//////////////////////////////////////////////////////////////////////
				// If defaultNumComputeUnits option is used, defaultNumComputeUnits //
				// option is always included.                                       //
				//////////////////////////////////////////////////////////////////////
				defaultGOption.add("defaultNumComputeUnits");
			}
			numSIMDWorkItemsSet = AnalysisTools.expressionsToStringSet((Set<Expression>)tuningConfigs.get("defaultNumSIMDWorkItems"));
			if( (numSIMDWorkItemsSet == null) || numSIMDWorkItemsSet.isEmpty() ) {
				numSIMDWorkItemsSet = new HashSet<String>(Arrays.asList(defaultNumSIMDWorkItems));
			} else if( !excludedGOption.contains("defaultNumSIMDWorkItems") ) {
				////////////////////////////////////////////////////////////////////////
				// If defaultNumSIMDWorkItems option is used, defaultNumSIMDWorkItems //
				// option is always included.                                         //
				////////////////////////////////////////////////////////////////////////
				defaultGOption.add("defaultNumSIMDWorkItems");
			}
			Expression  tExp = ((Expression)tuningConfigs.get("gpuMemTrOptLevel"));
			if( tExp != null ) {
				memTrOptValue = tExp.toString();
			} else {
				memTrOptValue = "2";
			}
			tExp = ((Expression)tuningConfigs.get("gpuMallocOptLevel"));
			if( tExp != null ) {
				mallocOptValue = tExp.toString();
			} else {
				mallocOptValue = "0";
			}
			tExp = ((Expression)tuningConfigs.get("UEPRemovalOptLevel"));
			if( tExp != null ) {
				UEPRemovalOptValue = tExp.toString();
			} else {
				UEPRemovalOptValue = "3";
			}
			tExp = ((Expression)tuningConfigs.get("AccPrivatization"));
			if( tExp != null ) {
				AccPrivatizationValue = tExp.toString();
			} else {
				AccPrivatizationValue = "1";
			}
			tExp = ((Expression)tuningConfigs.get("AccReduction"));
			if( tExp != null ) {
				AccReductionValue = tExp.toString();
			} else {
				AccReductionValue = "1";
			}
		}
		
		Set<String> gKeySet = gOptionMap.keySet();
		for( String gKey : gKeySet ) {
			HashSet<String> gOptionSet = gOptionMap.get(gKey);
			for( String option : gOptionSet ) {
				if( excludedGOption.contains(option) ) {
					gOptMap.put(option, new ArrayList<Boolean>(Arrays.asList(new Boolean(false))));
				} else if( defaultGOption.contains(option) ) {
					gOptMap.put(option, new ArrayList<Boolean>(Arrays.asList(new Boolean(true))));
				} else {
					gOptMap.put(option, new ArrayList<Boolean>(Arrays.asList(
							new Boolean(false), new Boolean(true))));
				}
			}
		}
		gKeySet = gOptMap.keySet();
		for( String option : gOptions ) {
			if( !gKeySet.contains(option) ) {
				if( defaultGOption.contains(option) ) {
					//An option in gOptions6 set is included in the defaultGOption set.
					gOptMap.put(option, new ArrayList<Boolean>(Arrays.asList(new Boolean(true))));
				} else {
					gOptMap.put(option, new ArrayList<Boolean>(Arrays.asList(new Boolean(false))));
				}
			}
		}
		int confID = 0;
		for( boolean gOpt1 : gOptMap.get("assumeNonZeroTripLoops") ) {
			for( boolean gOpt2 : gOptMap.get("AccPrivatization") ) {
				for( boolean gOpt3 : gOptMap.get("AccReduction") ) {
					for( boolean gOpt4 : gOptMap.get("gpuMallocOptLevel") ) {
						for( boolean gOpt5 : gOptMap.get("gpuMemTrOptLevel") ) {
							for( boolean gOpt6 : gOptMap.get("useMatrixTranspose") ) {
								for( boolean gOpt7 : gOptMap.get("useMallocPitch") ) {
									for( boolean gOpt8 : gOptMap.get("useLoopCollapse") ) {
										for( boolean gOpt9 : gOptMap.get("useParallelLoopSwap") ) {
											for( boolean gOpt10 : gOptMap.get("useUnrollingOnReduction") ) {
												for( boolean gOpt11 : gOptMap.get("shrdSclrCachingOnReg") ) {
													for( boolean gOpt12 : gOptMap.get("shrdArryElmtCachingOnReg") ) {
														for( boolean gOpt13 : gOptMap.get("shrdSclrCachingOnSM") ) {
															for( boolean gOpt14 : gOptMap.get("prvtArryCachingOnSM") ) {
																for( boolean gOpt15 : gOptMap.get("shrdArryCachingOnTM") ) {
																	for( boolean gOpt16 : gOptMap.get("defaultNumWorkers") ) {
																		for( boolean gOpt17 : gOptMap.get("maxNumGangs") ) {
																			//for( boolean gOpt18 : gOptMap.get("disableCritical2ReductionConv") ) {
																				for( boolean gOpt19 : gOptMap.get("UEPRemovalOptLevel") ) {
																					for( boolean gOpt20 : gOptMap.get("forceSyncKernelCall") ) {
																						for( boolean gOpt21 : gOptMap.get("doNotRemoveUnusedSymbols") ) {
																							for( boolean gOpt22 : gOptMap.get("shrdSclrCachingOnConst") ) {
																								for( boolean gOpt23 : gOptMap.get("shrdArryCachingOnConst") ) {
																									for( boolean gOpt24 : gOptMap.get("localRedVarConf") ) {
																										for( boolean gOpt25 : gOptMap.get("MemTrOptOnLoops") ) {
																											for( boolean gOpt26 : gOptMap.get("assumeNoAliasingAmongKernelArgs") ) {
																												for( boolean gOpt27 : gOptMap.get("skipKernelLoopBoundChecking") ) {
																													for( boolean gOpt28 : gOptMap.get("defaultNumComputeUnits") ) {
																														for( boolean gOpt29 : gOptMap.get("defaultNumSIMDWorkItems") ) {
																															StringBuilder str1 = new StringBuilder(256);
																															if( addSafetyCheckingCode ) {
																																str1.append("addSafetyCheckingCode\n");
																															}
																															if( gOpt1 ) {
																																str1.append("assumeNonZeroTripLoops\n");
																															}
																															if( gOpt26 ) {
																																str1.append("assumeNoAliasingAmongKernelArgs\n");
																															}
																															if( gOpt27 ) {
																																str1.append("skipKernelLoopBoundChecking\n");
																															}
																															if( gOpt2 ) {
																																str1.append("AccPrivatization="+AccPrivatizationValue+"\n");
																															}
																															if( gOpt3 ) {
																																str1.append("AccReduction="+AccReductionValue+"\n");
																															}
																															if( gOpt4 ) {
																																str1.append("gpuMallocOptLevel="+mallocOptValue+"\n");
																															}
																															if( gOpt5 ) {
																																str1.append("gpuMemTrOptLevel="+memTrOptValue+"\n");
																															}
																															if( gOpt6 ) {
																																str1.append("useMatrixTranspose\n");
																															}
																															if( gOpt7 ) {
																																str1.append("useMallocPitch\n");
																															}
																															if( gOpt8 ) {
																																str1.append("useLoopCollapse\n");
																															}
																															if( gOpt9 ) {
																																str1.append("useParallelLoopSwap\n");
																															}
																															if( gOpt10 ) {
																																str1.append("useUnrollingOnReduction\n");
																															}
																															if( gOpt11 ) {
																																str1.append("shrdSclrCachingOnReg\n");
																															}
																															if( gOpt12 ) {
																																str1.append("shrdArryElmtCachingOnReg\n");
																															}
																															if( gOpt13 ) {
																																str1.append("shrdSclrCachingOnSM\n");
																															}
																															if( gOpt14 ) {
																																str1.append("prvtArryCachingOnSM\n");
																															}
																															if( gOpt15 ) {
																																str1.append("shrdArryCachingOnTM\n");
																															}
																															//if( gOpt18 ) {
																															//	str1.append("disableCritical2ReductionConv\n");
																															//}
																															if( gOpt19 ) {
																																str1.append("UEPRemovalOptLevel="+UEPRemovalOptValue+"\n");
																															}
																															if( gOpt20 ) {
																																str1.append("forceSyncKernelCall\n");
																															}
																															if( gOpt21 ) {
																																str1.append("doNotRemoveUnusedSymbols\n");
																															}
																															if( gOpt22 ) {
																																str1.append("shrdSclrCachingOnConst\n");
																															}
																															if( gOpt23 ) {
																																str1.append("shrdArryCachingOnConst\n");
																															}
																															if( gOpt24 ) {
																																str1.append("localRedVarConf=1\n");
																															} else {
																																str1.append("localRedVarConf=0\n");
																															}
																															if( gOpt25 ) {
																																str1.append("MemTrOptOnLoops\n");
																															}
																															String confString = str1.toString();
																															String confString2 = "";
																															String confString3 = "";
																															String confString4 = "";
																															String confString5 = "";
																															for( String tsS1 : numWorkersSet ) {
																																confString2 = "";
																																if( gOpt16 && !tsS1.equals("NONE") ) {
																																	str1 = new StringBuilder(256);
																																	str1.append("defaultNumWorkers="+tsS1+"\n");
																																	confString2 = str1.toString();
																																}
																																for( String tsS2 : maxNumGangsSet ) {
																																	confString3 = "";
																																	if( gOpt17 && !tsS2.equals("NONE") ) {
																																		str1 = new StringBuilder(256);
																																		str1.append("maxNumGangs="+tsS2+"\n");
																																		confString3 = str1.toString();
																																	}
																																	for( String tsS3 : numComputeUnitsSet ) {
																																		confString4 = "";
																																		if( gOpt28 && !tsS3.equals("NONE") ) {
																																			str1 = new StringBuilder(256);
																																			str1.append("defaultNumComputeUnits="+tsS3+"\n");
																																			confString4 = str1.toString();
																																		}
																																		for( String tsS4 : numSIMDWorkItemsSet ) {
																																			confString5 = "";
																																			if( gOpt29 && !tsS4.equals("NONE") ) {
																																				str1 = new StringBuilder(256);
																																				str1.append("defaultNumSIMDWorkItems="+tsS4+"\n");
																																				confString5 = str1.toString();
																																			}
																																			str1 = new StringBuilder(256);
																																			str1.append(confString);
																																			str1.append(confString2);
																																			str1.append(confString3);
																																			str1.append(confString4);
																																			str1.append(confString5);
																																			confString5 = str1.toString();
																																			if( !confSet.contains(confString5) ) {
																																				confSet.add(confString5);
																																				String confFile = "confFile"+confID+".txt";
																																				try {
																																					BufferedWriter out1 = 
																																							new BufferedWriter(new FileWriter(dirPrefix+confFile));
																																					out1.write(confString5);
																																					out1.close();
																																				} catch( Exception e ) {
																																					PrintTools.println("Creaing a file, "+ confFile + ", failed; " +
																																							"tuning parameters can not be saved.", 0);
																																				}
																																				confID++;
																																			}
																																		}
																																	}
																																}
																															}
																														}
																													}
																												}
																											}
																										}
																									}
																								}
																							}
																						}
																					}
																				}
																			//}
																		}
																	}
																}
															}
														}
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
		PrintTools.println("\n\n    Number of created tuning-configuration files: "+confID+"\n\n", 0);
	}
	
	/**
	 * GPU-kernel-level tuning configuration output generator.
	 * 
	 * @param TuningOptions applicable tuning options suggested by O2G translator
	 */
	protected void genTuningConfs2( List<HashMap> TuningOptions ) {
		if( TuningOptions.size() != 2 ) {
			PrintTools.println("[ERROR in genTuningConfs()] input TuningOptions list does not contain enough data.", 0);
			return;
		}
		
	    /* make sure the tuning-configuration directory exists */
	    File dir = null;
	    File fname = null;
	    try {
	      dir = new File(".");
	      fname = new File(dir.getCanonicalPath(), tuningConfDir);
	      if (!fname.exists())
	      {
	        if (!fname.mkdir())
	          throw new IOException("mkdir failed");
	      }
	    } catch (Exception e) {
	      System.err.println("cetus: could not create tuning-configuration directory, " + e);
	      System.exit(1);
	    }
	    PrintTools.println("Generate configuration files for GPU-kernel-level tuning", 0);
	    String dirPrefix = fname.getAbsolutePath() + File.separatorChar;
	    
		HashMap<String, HashSet<String>> gOptionMap = 
			(HashMap<String, HashSet<String>>)TuningOptions.get(0);
		HashMap<ARCAnnotation, HashMap<String, Object>> kOptionMap = 
			(HashMap<ARCAnnotation, HashMap<String, Object>>)TuningOptions.get(1);
		HashMap<String, List<Boolean>> gOptMap = new HashMap<String, List<Boolean>>();
		HashSet<String> confSet = new HashSet<String>();
		HashSet<String> gOptions = new HashSet<String>();
		gOptions.addAll(Arrays.asList(gOptions1));
		gOptions.addAll(Arrays.asList(gOptions2));
		gOptions.addAll(Arrays.asList(gOptions3));
		gOptions.addAll(Arrays.asList(gOptions4));
		gOptions.addAll(Arrays.asList(gOptions6));
		
		//////////////////////////////////
		// Check default configurations //
		//////////////////////////////////
		HashSet<String> defaultGOption = null;
		HashSet<String> excludedGOption = null;
		String memTrOptValue = "2";
		String mallocOptValue = "0";
		String UEPRemovalOptValue = "3";
		String AccPrivatizationValue = "1";
		String AccReductionValue = "1";
		Set<String> numWorkersSet = null;
		Set<String> maxNumGangsSet = null;
		Set<String> numComputeUnitsSet = null;
		Set<String> numSIMDWorkItemsSet = null;
		if( tuningConfigs == null || tuningConfigs.isEmpty() ) {
			defaultGOption = new HashSet<String>(Arrays.asList(defaultGOptions1));
			excludedGOption = new HashSet<String>();
			numWorkersSet = new HashSet<String>(Arrays.asList(defaultNumWorkers));
			maxNumGangsSet = new HashSet<String>(Arrays.asList(defaultNumGangs));
			numComputeUnitsSet = new HashSet<String>(Arrays.asList(defaultNumComputeUnits));
			numSIMDWorkItemsSet = new HashSet<String>(Arrays.asList(defaultNumSIMDWorkItems));
		} else {
			defaultGOption = (HashSet<String>)tuningConfigs.get("defaultGOptionSet");
			if( defaultGOption == null ) {
				defaultGOption = new HashSet<String>(Arrays.asList(defaultGOptions2));
			} else {
				//Check whether defaultGOption contains illegal options or not.
				HashSet<String> gOption5 = new HashSet<String>();
				gOption5.addAll(Arrays.asList(gOptions5));
				for( String gOpt : defaultGOption ) {
					if( !gOptions.contains(gOpt) && !gOption5.contains(gOpt) ) {
						PrintTools.println("\n[WARNING in genTuningConfs()] defaultGOptions set contains " +
								"unsupported option: " + gOpt + "\n", 0);
					}
				}
			}
			excludedGOption = (HashSet<String>)tuningConfigs.get("excludedGOptionSet");
			if( excludedGOption == null ) {
				excludedGOption = new HashSet<String>();
			} else {
				//Check whether defaultGOption contains illegal options or not.
				HashSet<String> gOption5 = new HashSet<String>();
				gOption5.addAll(Arrays.asList(gOptions5));
				for( String gOpt : excludedGOption ) {
					if( !gOptions.contains(gOpt) && !gOption5.contains(gOpt) ) {
						PrintTools.println("\n[WARNING in genTuningConfs()] excludedGOptions set contains " +
								"unsupported option: " + gOpt + ".\n", 0);
					}
				}
			}
			//////////////////////////////////////////////////////////////////
			// Options in defaultGOptions0 are always applied unless a user //
			// explicitly excludes using excludedGOptionSet option.         //
			//////////////////////////////////////////////////////////////////
			for( String gOpt : defaultGOptions0 ) {
				if( !excludedGOption.contains(gOpt) ) {
					defaultGOption.add(gOpt);
				}
			}
			numWorkersSet = AnalysisTools.expressionsToStringSet((Set<Expression>)tuningConfigs.get("defaultNumWorkersSet"));
			if( (numWorkersSet == null) || numWorkersSet.isEmpty() ) {
				numWorkersSet = new HashSet<String>(Arrays.asList(defaultNumWorkers));
			} else if( !excludedGOption.contains("defaultNumWorkers") ) {
				///////////////////////////////////////////////////////////////
				// If cudaThreadBlockSet option is used, cudaThreadBlockSize //
				// option is always included.                                //
				///////////////////////////////////////////////////////////////
				defaultGOption.add("defaultNumWorkers");
			}
			maxNumGangsSet = AnalysisTools.expressionsToStringSet((Set<Expression>)tuningConfigs.get("maxNumGangsSet"));
			if( (maxNumGangsSet == null) || maxNumGangsSet.isEmpty() ) {
				maxNumGangsSet = new HashSet<String>(Arrays.asList(defaultNumGangs));
			}
			numComputeUnitsSet = AnalysisTools.expressionsToStringSet((Set<Expression>)tuningConfigs.get("defaultNumComputeUnits"));
			if( (numComputeUnitsSet == null) || numComputeUnitsSet.isEmpty() ) {
				numComputeUnitsSet = new HashSet<String>(Arrays.asList(defaultNumComputeUnits));
			} else if( !excludedGOption.contains("defaultNumComputeUnits") ) {
				//////////////////////////////////////////////////////////////////////
				// If defaultNumComputeUnits option is used, defaultNumComputeUnits //
				// option is always included.                                       //
				//////////////////////////////////////////////////////////////////////
				defaultGOption.add("defaultNumComputeUnits");
			}
			numSIMDWorkItemsSet = AnalysisTools.expressionsToStringSet((Set<Expression>)tuningConfigs.get("defaultNumSIMDWorkItems"));
			if( (numSIMDWorkItemsSet == null) || numSIMDWorkItemsSet.isEmpty() ) {
				numSIMDWorkItemsSet = new HashSet<String>(Arrays.asList(defaultNumSIMDWorkItems));
			} else if( !excludedGOption.contains("defaultNumSIMDWorkItems") ) {
				////////////////////////////////////////////////////////////////////////
				// If defaultNumSIMDWorkItems option is used, defaultNumSIMDWorkItems //
				// option is always included.                                         //
				////////////////////////////////////////////////////////////////////////
				defaultGOption.add("defaultNumSIMDWorkItems");
			}
			Expression  tExp = ((Expression)tuningConfigs.get("gpuMemTrOptLevel"));
			if( tExp != null ) {
				memTrOptValue = tExp.toString();
			} else {
				memTrOptValue = "2";
			}
			tExp = ((Expression)tuningConfigs.get("gpuMallocOptLevel"));
			if( tExp != null ) {
				mallocOptValue = tExp.toString();
			} else {
				mallocOptValue = "0";
			}
			tExp = ((Expression)tuningConfigs.get("UEPRemovalOptLevel"));
			if( tExp != null ) {
				UEPRemovalOptValue = tExp.toString();
			} else {
				UEPRemovalOptValue = "3";
			}
			tExp = ((Expression)tuningConfigs.get("AccPrivatization"));
			if( tExp != null ) {
				AccPrivatizationValue = tExp.toString();
			} else {
				AccPrivatizationValue = "1";
			}
			tExp = ((Expression)tuningConfigs.get("AccReduction"));
			if( tExp != null ) {
				AccReductionValue = tExp.toString();
			} else {
				AccReductionValue = "1";
			}
		}
		
		HashSet<String> defaultGOptSet3 = new HashSet<String>(Arrays.asList(defaultGOptions3));
		Set<String> gKeySet = gOptionMap.keySet();
		for( String gKey : gKeySet ) {
			HashSet<String> gOptionSet = gOptionMap.get(gKey);
			for( String option : gOptionSet ) {
				if( excludedGOption.contains(option) ) {
					gOptMap.put(option, new ArrayList<Boolean>(Arrays.asList(new Boolean(false))));
				} else if( defaultGOption.contains(option) ) {
					gOptMap.put(option, new ArrayList<Boolean>(Arrays.asList(new Boolean(true))));
				} else if( defaultGOptSet3.contains(option) ) {
					////////////////////////////////////////////////////////////////////////
					// Options in defaultGOptions3 are always applied if existing and     //
					// unless a user explicitly excludes using excludedGOptionSet option. //
					////////////////////////////////////////////////////////////////////////
					gOptMap.put(option, new ArrayList<Boolean>(Arrays.asList(new Boolean(true))));
				} else {
					gOptMap.put(option, new ArrayList<Boolean>(Arrays.asList(
							new Boolean(false), new Boolean(true))));
				}
			}
		}
		gKeySet = gOptMap.keySet();
		for( String option : gOptions ) {
			if( !gKeySet.contains(option) ) {
				if( defaultGOption.contains(option) ) {
					//An option in gOptions6 set is included in the defaultGOption set.
					gOptMap.put(option, new ArrayList<Boolean>(Arrays.asList(new Boolean(true))));
				} else {
					gOptMap.put(option, new ArrayList<Boolean>(Arrays.asList(new Boolean(false))));
				}
			}
		}
		int confID = 0;
		for( boolean gOpt1 : gOptMap.get("assumeNonZeroTripLoops") ) {
			for( boolean gOpt2 : gOptMap.get("AccPrivatization") ) {
				for( boolean gOpt3 : gOptMap.get("AccReduction") ) {
					for( boolean gOpt4 : gOptMap.get("gpuMallocOptLevel") ) {
						for( boolean gOpt5 : gOptMap.get("gpuMemTrOptLevel") ) {
							for( boolean gOpt6 : gOptMap.get("useMatrixTranspose") ) {
								for( boolean gOpt7 : gOptMap.get("useMallocPitch") ) {
									for( boolean gOpt8 : gOptMap.get("useLoopCollapse") ) {
										for( boolean gOpt9 : gOptMap.get("useParallelLoopSwap") ) {
											for( boolean gOpt10 : gOptMap.get("useUnrollingOnReduction") ) {
												for( boolean gOpt11 : gOptMap.get("defaultNumWorkers") ) {
													for( boolean gOpt12 : gOptMap.get("maxNumGangs") ) {
														//for( boolean gOpt18 : gOptMap.get("disableCritical2ReductionConv") ) {
														for( boolean gOpt19 : gOptMap.get("UEPRemovalOptLevel") ) {
															for( boolean gOpt20 : gOptMap.get("forceSyncKernelCall") ) {
																for( boolean gOpt21 : gOptMap.get("doNotRemoveUnusedSymbols") ) {
																	for( boolean gOpt22 : gOptMap.get("localRedVarConf") ) {
																		for( boolean gOpt23 : gOptMap.get("MemTrOptOnLoops") ) {
																			for( boolean gOpt24 : gOptMap.get("assumeNoAliasingAmongKernelArgs") ) {
																				for( boolean gOpt25 : gOptMap.get("skipKernelLoopBoundChecking") ) {
																					for( boolean gOpt26 : gOptMap.get("defaultNumComputeUnits") ) {
																						for( boolean gOpt27 : gOptMap.get("defaultNumSIMDWorkItems") ) {
																							StringBuilder str1 = new StringBuilder(256);
																							if( addSafetyCheckingCode ) {
																								str1.append("addSafetyCheckingCode\n");
																							}
																							if( gOpt1 ) {
																								str1.append("assumeNonZeroTripLoops\n");
																							}
																							if( gOpt24 ) {
																								str1.append("assumeNoAliasingAmongKernelArgs\n");
																							}
																							if( gOpt25 ) {
																								str1.append("skipKernelLoopBoundChecking\n");
																							}
																							if( gOpt2 ) {
																								str1.append("AccPrivatization="+AccPrivatizationValue+"\n");
																							}
																							if( gOpt3 ) {
																								str1.append("AccReduction="+AccReductionValue+"\n");
																							}
																							if( gOpt4 ) {
																								str1.append("gpuMallocOptLevel="+mallocOptValue+"\n");
																							}
																							if( gOpt5 ) {
																								str1.append("gpuMemTrOptLevel="+memTrOptValue+"\n");
																							}
																							if( gOpt6 ) {
																								str1.append("useMatrixTranspose\n");
																							}
																							if( gOpt7 ) {
																								str1.append("useMallocPitch\n");
																							}
																							if( gOpt8 ) {
																								str1.append("useLoopCollapse\n");
																							}
																							if( gOpt9 ) {
																								str1.append("useParallelLoopSwap\n");
																							}
																							if( gOpt10 ) {
																								str1.append("useUnrollingOnReduction\n");
																							}
																							//if( gOpt18 ) {
																							//	str1.append("disableCritical2ReductionConv\n");
																							//}
																							if( gOpt19 ) {
																								str1.append("UEPRemovalOptLevel="+UEPRemovalOptValue+"\n");
																							}
																							if( gOpt20 ) {
																								str1.append("forceSyncKernelCall\n");
																							}
																							if( gOpt21 ) {
																								str1.append("doNotRemoveUnusedSymbols\n");
																							}
																							if( gOpt22 ) {
																								str1.append("localRedVarConf=1\n");
																							} else {
																								str1.append("localRedVarConf=0\n");
																							}
																							if( gOpt23 ) {
																								str1.append("MemTrOptOnLoops\n");
																							}
																							String confString = str1.toString();
																							String confString2 = "";
																							String confString3 = "";
																							String confString4 = "";
																							String confString5 = "";
																							for( String tsS1 : numWorkersSet ) {
																								confString2 = "";
																								if( gOpt11 && !tsS1.equals("NONE") ) {
																									str1 = new StringBuilder(256);
																									str1.append("defaultNumWorkers="+tsS1+"\n");
																									confString2 = str1.toString();
																								}
																								for( String tsS2 : maxNumGangsSet ) {
																									confString3 = "";
																									if( gOpt12 && !tsS2.equals("NONE") ) {
																										str1 = new StringBuilder(256);
																										str1.append("maxNumGangs="+tsS2+"\n");
																										confString3 = str1.toString();
																									}
																									for( String tsS3 : numComputeUnitsSet ) {
																										confString4 = "";
																										if( gOpt26 && !tsS3.equals("NONE") ) {
																											str1 = new StringBuilder(256);
																											str1.append("defaultNumComputeUnits="+tsS3+"\n");
																											confString4 = str1.toString();
																										}
																										for( String tsS4 : numSIMDWorkItemsSet ) {
																											confString5 = "";
																											if( gOpt27 && !tsS4.equals("NONE") ) {
																												str1 = new StringBuilder(256);
																												str1.append("defaultNumSIMDWorkItems="+tsS4+"\n");
																												confString5 = str1.toString();
																											}
																											str1 = new StringBuilder(256);
																											str1.append(confString);
																											str1.append(confString2);
																											str1.append(confString3);
																											str1.append(confString4);
																											str1.append(confString5);
																											confString5 = str1.toString();
																											if( !confSet.contains(confString5) ) {
																												confSet.add(confString5);
																												Set<String> userDirectives = 
																														genKTuningConf(gOpt8, gOpt9, gOpt10, kOptionMap, maxNumGangsSet);
																												if( userDirectives == null ) {
																													return;
																												}
																												for( String uDir : userDirectives ) {
																													String confFile = "confFile"+confID+".txt";
																													String uDirFile = "userDirective"+confID+".txt";
																													str1 = new StringBuilder(256);
																													str1.append(confString5);
																													str1.append("UserDirectiveFile="+uDirFile+"\n");
																													try {
																														BufferedWriter out1 = 
																																new BufferedWriter(new FileWriter(dirPrefix+confFile));
																														out1.write(str1.toString());
																														out1.close();
																														BufferedWriter out2 = 
																																new BufferedWriter(new FileWriter(dirPrefix+uDirFile));
																														out2.write(uDir);
																														out2.close();
																													} catch( Exception e ) {
																														PrintTools.println("Creaing a file, "+ confFile + ", failed; " +
																																"tuning parameters can not be saved.", 0);
																													}
																													confID++;
																												}
																											}
																										}
																									}
																								}
																							}
																						}
																					}																														
																				}
																			}
																		}
																	}
																}
															}
														}
														//}
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
		}
		PrintTools.println("\n\n    Number of created tuning-configuration files: "+confID+"\n\n", 0);
	}

	/**
	 * Kernel-level user directive output generator
	 * 
	 * @param useLoopCollapse true if loop collapse optimization is applied
	 * @param useParallelLoopSwap true if parallel loopswap optimization is applied
	 * @param useUnrolling true if unrolling-on-reduction optimization is applied
	 * @param kOptionMap hashMap of kernel-level options
	 * @return set of user directive outputs
	 */
	protected Set<String> genKTuningConf(boolean useLoopCollapse, boolean useParallelLoopSwap, boolean useUnrolling,
			HashMap<ARCAnnotation, HashMap<String, Object>> kOptionMap, Set<String> maxNumGangsSet) {
		Set<String> oldSet = new HashSet<String>();
		Set<String> newSet = new HashSet<String>();
		Set<ARCAnnotation> keySet = kOptionMap.keySet();
		for( ARCAnnotation cAnnot : keySet ) {
			StringBuilder id_str = new StringBuilder(32);
			id_str.append("kernelid("+cAnnot.get("kernelid")+") ");
			id_str.append("procname("+cAnnot.get("procname")+") ");
			String idString = id_str.toString();
			HashMap<String, Object> kMap = kOptionMap.get(cAnnot);
			Set<String> kSet = kMap.keySet();
			HashMap<String, List<String>> kOptMap = new HashMap<String, List<String>>();
			if( useLoopCollapse && kSet.contains("loopcollapse") ) {
				kOptMap.put("loopcollapse", new ArrayList<String>(Arrays.asList(
						"false", "true")));
			} else {
				kOptMap.put("loopcollapse", new ArrayList<String>(Arrays.asList(
						"none")));
			}
			if( useParallelLoopSwap && kSet.contains("ploopswap") ) {
				kOptMap.put("ploopswap", new ArrayList<String>(Arrays.asList(
						"false", "true")));
			} else {
				kOptMap.put("ploopswap", new ArrayList<String>(Arrays.asList(
						"none")));
			}
			if( useUnrolling && kSet.contains("noreductionunroll") ) {
				kOptMap.put("noreductionunroll", new ArrayList<String>(Arrays.asList(
						"false", "true")));
			} else {
				kOptMap.put("noreductionunroll", new ArrayList<String>(Arrays.asList(
						"none")));
			}
			if( kSet.contains("ROShSclrNL") ) {
				kOptMap.put("ROShSclrNL", new ArrayList<String>(Arrays.asList(
						"none", "sharedRO")));
			} else {
				kOptMap.put("ROShSclrNL", new ArrayList<String>(Arrays.asList(
						"none")));
			}
			if( kSet.contains("ROShSclr") ) {
				kOptMap.put("ROShSclr", new ArrayList<String>(Arrays.asList(
						"none", "registerRO", "sharedRO")));
			} else {
				kOptMap.put("ROShSclr", new ArrayList<String>(Arrays.asList(
						"none")));
			}
			if( kSet.contains("RWShSclr") ) {
				kOptMap.put("RWShSclr", new ArrayList<String>(Arrays.asList(
						"none", "registerRW", "sharedRW")));
			} else {
				kOptMap.put("RWShSclr", new ArrayList<String>(Arrays.asList(
						"none")));
			}
			if( kSet.contains("ROShArEl") ) {
				kOptMap.put("ROShArEl", new ArrayList<String>(Arrays.asList(
						"none", "registerRO")));
			} else {
				kOptMap.put("ROShArEl", new ArrayList<String>(Arrays.asList(
						"none")));
			}
			if( kSet.contains("RWShArEl") ) {
				kOptMap.put("RWShArEl", new ArrayList<String>(Arrays.asList(
						"none", "registerRW")));
			} else {
				kOptMap.put("RWShArEl", new ArrayList<String>(Arrays.asList(
						"none")));
			}
			if( kSet.contains("RO1DShAr") ) {
				kOptMap.put("RO1DShAr", new ArrayList<String>(Arrays.asList(
						"none", "texture")));
			} else {
				kOptMap.put("RO1DShAr", new ArrayList<String>(Arrays.asList(
						"none")));
			}
			if( kSet.contains("RO1DShAr") ) {
				kOptMap.put("RO1DShAr", new ArrayList<String>(Arrays.asList(
						"none", "texture")));
			} else {
				kOptMap.put("RO1DShAr", new ArrayList<String>(Arrays.asList(
						"none")));
			}
			if( kSet.contains("PrvAr") ) {
				kOptMap.put("PrvAr", new ArrayList<String>(Arrays.asList(
						"none", "sharedRW")));
			} else {
				kOptMap.put("PrvAr", new ArrayList<String>(Arrays.asList(
						"none")));
			}
			if( kSet.contains("SclrConst") ) {
				kOptMap.put("SclrConst", new ArrayList<String>(Arrays.asList(
						"none", "constant")));
			} else {
				kOptMap.put("SclrConst", new ArrayList<String>(Arrays.asList(
						"none")));
			}
			if( kSet.contains("ArryConst") ) {
				kOptMap.put("ArryConst", new ArrayList<String>(Arrays.asList(
						"none", "constant")));
			} else {
				kOptMap.put("ArryConst", new ArrayList<String>(Arrays.asList(
						"none")));
			}
			if( maxNumGangsSet == null || maxNumGangsSet.isEmpty() ) {
				kOptMap.put("maxNumGangs", new ArrayList<String>(Arrays.asList(
				"none")));
			} else {
/*				kOptMap.put("maxNumGangs", new ArrayList<String>(Arrays.asList(
						"none", "maxNumGangs")));*/
				//////////////////////////////////////////////////////////////////
				//DEBUG: for now, this option is applied to a program globally, //
				// and thus not applied for each kernel here.                   //
				//////////////////////////////////////////////////////////////////
				kOptMap.put("maxNumGangs", new ArrayList<String>(Arrays.asList(
				"none")));
			}
			for( String kOpt1 : kOptMap.get("loopcollapse") ) {
				for( String kOpt2 : kOptMap.get("ploopswap") ) {
					for( String kOpt3 : kOptMap.get("noreductionunroll") ) {
						for( String kOpt4 : kOptMap.get("ROShSclrNL") ) {
							for( String kOpt5 : kOptMap.get("ROShSclr") ) {
								for( String kOpt6 : kOptMap.get("RWShSclr") ) {
									for( String kOpt7 : kOptMap.get("ROShArEl") ) {
										for( String kOpt8 : kOptMap.get("RWShArEl") ) {
											for( String kOpt9 : kOptMap.get("RO1DShAr") ) {
												for( String kOpt10 :kOptMap.get("PrvAr") ) {
													for( String kOpt11 :kOptMap.get("maxNumGangs") ) {
														for( String kOpt12 :kOptMap.get("SclrConst") ) {
															for( String kOpt13 :kOptMap.get("ArryConst") ) {
																StringBuilder str1 = new StringBuilder(256);
																str1.append(idString);
																if( kOpt1.equals("false") ) {
																	str1.append("noloopcollapse ");
																}
																if( kOpt2.equals("false") ) {
																	str1.append("noploopswap ");
																}
																if( kOpt3.equals("false") ) {
																	Set<String> rSet = (Set<String>)kMap.get("noreductionunroll");
																	str1.append("noreductionunroll("+
																			PrintTools.collectionToString(rSet, ",")+") ");
																}
																HashSet<String> registerRO = new HashSet<String>();
																HashSet<String> registerRW = new HashSet<String>();
																HashSet<String> sharedRO = new HashSet<String>();
																HashSet<String> sharedRW = new HashSet<String>();
																HashSet<String> texture = new HashSet<String>();
																HashSet<String> constant = new HashSet<String>();
																if( kOpt4.equals("sharedRO") ) {
																	sharedRO.addAll((Set<String>)kMap.get("ROShSclrNL"));
																}
																if( kOpt5.equals("sharedRO") ) {
																	sharedRO.addAll((Set<String>)kMap.get("ROShSclr"));
																} else if( kOpt5.equals("registerRO") ) {
																	registerRO.addAll((Set<String>)kMap.get("ROShSclr"));
																}
																if( kOpt6.equals("sharedRW") ) {
																	sharedRW.addAll((Set<String>)kMap.get("RWShSclr"));
																} else if( kOpt5.equals("registerRW") ) {
																	registerRW.addAll((Set<String>)kMap.get("RWShSclr"));
																}
																if( kOpt9.equals("texture") ) {
																	texture.addAll((Set<String>)kMap.get("RO1DShAr"));
																}
																if( kOpt12.equals("constant") ) {
																	constant.addAll((Set<String>)kMap.get("SclrConst"));
																}
																if( kOpt13.equals("constant") ) {
																	constant.addAll((Set<String>)kMap.get("ArryConst"));
																}
																if( kOpt7.equals("registerRO") ) {
																	Set<String> sSet = new HashSet<String>((Set<String>)kMap.get("ROShArEl"));
																	Set<String> removeSet = new HashSet<String>();
																	///////////////////////////////////////////////////
																	// If an array element in ROShArEl set refers to //
																	// an array in texture set, the element should   //
																	// not be included in the registerRO set.        //
																	///////////////////////////////////////////////////
																	if( !texture.isEmpty() ) {
																		for( String element : sSet ) {
																			int bracket = element.indexOf('[');
																			if( bracket != -1 ) {
																				String sym = element.substring(0, bracket);
																				if( texture.contains(sym) ) {
																					removeSet.add(element);
																				}
																			}
																		}
																		sSet.removeAll(removeSet);
																	}
																	registerRO.addAll(sSet);
																}
																if( kOpt8.equals("registerRW") ) {
																	registerRW.addAll((Set<String>)kMap.get("RWShArEl"));
																}
																if( kOpt10.equals("sharedRW") ) {
																	sharedRW.addAll((Set<String>)kMap.get("PrvAr"));
																}
																if( !sharedRO.isEmpty() ) {
																	str1.append("sharedRO("+PrintTools.collectionToString(sharedRO, ",")+") ");
																}
																if( !sharedRW.isEmpty() ) {
																	str1.append("sharedRW("+PrintTools.collectionToString(sharedRW, ",")+") ");
																}
																if( !texture.isEmpty() ) {
																	str1.append("texture("+PrintTools.collectionToString(texture, ",")+") ");
																}
																if( !registerRO.isEmpty() ) {
																	str1.append("registerRO("+PrintTools.collectionToString(registerRO, ",")+") ");
																}
																if( !registerRW.isEmpty() ) {
																	str1.append("registerRW("+PrintTools.collectionToString(registerRW, ",")+") ");
																}
																if( !constant.isEmpty() ) {
																	str1.append("constant("+PrintTools.collectionToString(constant, ",")+") ");
																}
																str1.append("\n\n");
																String tuningConf = str1.toString();
																if( kOpt11.equals("maxNumGangs") ) {
																	//////////////////////////////////////////////////////////////////
																	//DEBUG: for now, this option is applied to a program globally, //
																	// and thus not applied for each kernel here.                   //
																	//////////////////////////////////////////////////////////////////
																} else {
																	if( oldSet.isEmpty() ) {
																		newSet.add(tuningConf);
																	} else {
																		//DEBUG
																		//System.out.println("newSet size: " + newSet.size());
																		try {
																			for( String confS : oldSet ) {
																				String newStr = confS.concat(tuningConf);
																				newSet.add(newStr);
																			}
																		} catch( Exception e) {
																			Tools.exit("[ERROR in genKTuningConf()] there exist too many tuning " +
																			"configurations; file generation will be skipped.");
																		}
																	}
																}
															}
														}
													}
												}
											}
										}
									}
								}
							}
						}
					}
				}
			}
			oldSet.clear();
			oldSet.addAll(newSet);
			// DEBUG
			//System.out.println("newSet size: " + newSet.size());
			newSet.clear();
		}
		return oldSet;
	}
}
