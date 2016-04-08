/**
 * 
 */
package openacc.transforms;

import cetus.hir.*;
import cetus.exec.*;
import cetus.analysis.LoopTools;

import java.util.HashMap;
import java.util.List;
import java.util.LinkedList;
import java.util.Collection;
import java.util.Map;
import java.util.Set;
import java.util.HashSet;
import java.util.NoSuchElementException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Stack;

import openacc.analysis.AnalysisTools;
import openacc.analysis.SubArray;
import openacc.hir.*;

/**
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group, Oak Ridge National Laboratory
 *
 */
public abstract class ACC2GPUTranslator {
	protected String pass_name = "[ACC2GPUTranslator]";
	protected Program program;
	//Main refers either a procedure containing acc_init() call or main() procedure if no explicit acc_init() call exists.
	protected TranslationUnit main_TrUnt;
	protected Procedure main;
	protected List<TranslationUnit> main_TrUnt_List = new LinkedList<TranslationUnit>();
	protected List<Procedure> main_List = new LinkedList<Procedure>();
	protected Statement accInitStmt = null;
	protected List<Statement> accInitStmt_List = new LinkedList<Statement>();
	//protected SymbolTable main_global_table;
	protected Statement firstMainStmt;  // The first statement in main procedure;
	protected List<FunctionCall> acc_init_list = new LinkedList<FunctionCall>();
	protected List<FunctionCall> acc_shutdown_list = new LinkedList<FunctionCall>();
	protected List<Statement> optPrintStmts = new LinkedList<Statement>();
	protected List<Statement> confPrintStmts = new LinkedList<Statement>();
	
	protected boolean opt_MallocPitch = false;
	protected boolean opt_MatrixTranspose = false;
	protected boolean opt_LoopCollapse = false;
	protected boolean opt_UnrollingOnReduction = false;
	protected boolean opt_addSafetyCheckingCode = false;
	protected boolean opt_forceSyncKernelCall = false;
	protected boolean opt_MemTrOptOnLoops = false;
	protected boolean opt_GenDistOpenACC = false;
	protected boolean opt_PrintConfigurations = false;
	protected boolean opt_AssumeNoAliasing = false;
	protected boolean opt_skipKernelLoopBoundChecking = false;
	
	protected int defaultNumWorkers = 128;
	protected int maxNumGangs = 0; //0 if undefined.
	///////////////////////////////////////////////////////////////
	// DEBUG: below two variables should have the same values as //
	// the ones in acc2gpu.java.                                 //
	///////////////////////////////////////////////////////////////
	protected int gpuMemTrOptLevel = 2;
	protected int gpuMallocOptLevel = 0;
	protected int localRedVarConf = 1;
	protected int SkipGPUTranslation = 0;
	protected String tuningParamFile = null;

    protected int targetArch = 0;
    protected int targetModel = 0; //0 for CUDA, 1 for OpenCL

	protected enum MemTrType {NoCopy, CopyIn, CopyOut, CopyInOut}
	protected enum DataClauseType {CheckOnly, CheckNMalloc, Malloc, UpdateOnly, Pipe, PipeIn, PipeOut}
	protected enum MallocType {ConstantMalloc, TextureMalloc, PitchedMalloc, NormalMalloc, PipeMalloc}
	
	////////////////////////////////////////////////////////////////////
	// HashMap containing TranlationUnit to hidden comment indicating //
	// the last of OpenACC-related headers.                           //
	////////////////////////////////////////////////////////////////////
	protected static HashMap<TranslationUnit, Declaration> OpenACCHeaderEndMap = null;
	
	protected enum DataRegionType {
		ImplicitProgramRegion,
		ImplicitProcedureRegion,
		ExplicitDataRegion,
		ComputeRegion
	}
	
	protected int numComputeRegions = 0;
	
	protected static LoopCollapse loopCollapseHandler = null;
	
	protected boolean IRSymbolOnly = true;
	
	protected boolean kernelVerification = false;
	protected boolean memtrVerification = false;
	protected boolean enableFaultInjection = false;
	protected boolean enableCustomProfiling = false;
	protected boolean enablePipeTransformation = false;
	
	protected FloatLiteral marginOfError = new FloatLiteral(1.0e-6);
	protected FloatLiteral minCheckValue = null;
	
    protected List<String> accKernelsList = new ArrayList<String>();
    
    //Keep the last user-declarations added to the translation unit containing kernel files.
    //This information is needed to insert user declarations in a correct order.
    protected Map<TranslationUnit, Declaration> lastUserDeclMap = new HashMap<TranslationUnit, Declaration>();

    protected Expression getAsyncExpression(ACCAnnotation tAnnot)
    {
                if( tAnnot != null ) 
                {
                    Object obj = tAnnot.get("async");
                    if( obj instanceof String ) 
                    {
                        //asyncID = new NameID("INT_MAX");
                        return new NameID("DEFAULT_ASYNC_QUEUE");
                    } 
                    else if( obj instanceof Expression ) 
                    {
                        return (Expression)obj;
                    }
                    else
                    {
                        PrintTools.println("[Warning] Unsupported Async Clause " + obj,0);
                    }
                }

                return new NameID("DEFAULT_QUEUE");
    }

	protected ACC2GPUTranslator(Program prog) {
		program = prog;
		GPUInitializer();
	}
	
	public void start() {
		Set<String> searchKeys = new HashSet<String>();
		searchKeys.add("declare");
		searchKeys.add("update");
		searchKeys.add("host_data");
		searchKeys.add("wait");
		List<ACCAnnotation> declareAnnots = new LinkedList<ACCAnnotation>();
		List<ACCAnnotation> updateAnnots = new LinkedList<ACCAnnotation>();
		List<ACCAnnotation> hostDataAnnots = new LinkedList<ACCAnnotation>();
		List<ACCAnnotation> waitAnnots = new LinkedList<ACCAnnotation>();
		List<ACCAnnotation> miscAnnots = 
					AnalysisTools.collectPragmas(program, ACCAnnotation.class, searchKeys, false);
		if( miscAnnots != null ) {
			for( ACCAnnotation mAnnot : miscAnnots ) {
				if( mAnnot.containsKey("declare") ) {
					declareAnnots.add(mAnnot);
				} else if( mAnnot.containsKey("update") ) {
					updateAnnots.add(mAnnot);
				} else if( mAnnot.containsKey("host_data") ) {
					hostDataAnnots.add(mAnnot);
				} else if( mAnnot.containsKey("wait") ) {
					boolean isWaitClause = false;
					for( String tDir : ACCAnnotation.OpenACCDirectivesWithWait ) {
						if( mAnnot.containsKey(tDir) ) {
							isWaitClause = true;
							break;
						}
					}
					if( !isWaitClause ) {
						waitAnnots.add(mAnnot);
					}
				}
			}
		}
		handleDeclareDirectives(declareAnnots);
		
		List<Procedure> procedureList = IRTools.getProcedureList(program);
		for( Procedure cProc : procedureList ) {
			List<ACCAnnotation> dataAnnots = 
					AnalysisTools.collectPragmas(cProc, ACCAnnotation.class, ACCAnnotation.dataRegions, false);

			List<ACCAnnotation> atomicAnnots =
                                        AnalysisTools.collectPragmas(cProc, ACCAnnotation.class, new HashSet(Arrays.asList("atomic")), false);
			
			List<FunctionCall> fCallList = IRTools.getFunctionCalls(cProc);
			List<FunctionCall> memoryAPIList = new LinkedList<FunctionCall>();
			for( FunctionCall fCall : fCallList ) {
				if(OpenACCRuntimeLibrary.isMemoryAPI(fCall)) {
					memoryAPIList.add(fCall);
				}
			}
			if( !memoryAPIList.isEmpty() ) {
				handleMemoryRuntimeAPIs(cProc, memoryAPIList);
			}

			List<ACCAnnotation> parallelRegionAnnots = new LinkedList<ACCAnnotation>();
			List<ACCAnnotation> kernelsRegionAnnots = new LinkedList<ACCAnnotation>();
			List<ACCAnnotation> dataRegionAnnots = new LinkedList<ACCAnnotation>();

            handleAtomicAnnots(atomicAnnots);

			if( (dataAnnots != null) && !dataAnnots.isEmpty() ) {
				TranslationUnit trUnt = (TranslationUnit)cProc.getParent();
				if( AnalysisTools.isInHeaderFile(cProc, trUnt) ) {
					Tools.exit("[ERROR in ACC2GPUTranslator] For correct translation, input files containing any " +
							"OpenACC data regions or compute regions should be fed to the compiler as separate input files, " +
							"but the following input file includes another input file that contains data/compute regions; " +
							"to fix this, do not include input C files if they contains data/compute regions.\n" +
							"Input file: " + trUnt.getInputFilename() + "\n");
				}
				for( ACCAnnotation annot : dataAnnots ) {
					if( annot.containsKey("data") ) {
						dataRegionAnnots.add(annot);
					} else if( annot.containsKey("parallel") ) {
						parallelRegionAnnots.add(annot);
					} else if( annot.containsKey("kernels") ) {
						kernelsRegionAnnots.add(annot);
					}
				}
				handleDataRegions(cProc, dataRegionAnnots);
				convComputeRegionsToGPUKernels(cProc, parallelRegionAnnots, kernelsRegionAnnots);
				//Postprocessing for pipe clauses.
				List<ACCAnnotation>  pipeAnnots = IRTools.collectPragmas(program, ACCAnnotation.class, "pipe");
				List<FunctionCall> funcCallList = IRTools.getFunctionCalls(program);
				for( ACCAnnotation tAnnot : pipeAnnots ) {
					if( tAnnot.containsKey("declare") ) {
						Traversable ptt = tAnnot.getAnnotatable().getParent();
						List<ACCAnnotation>  computeAnnots = AnalysisTools.ipCollectPragmas(ptt, ACCAnnotation.class, 
								ACCAnnotation.pipeIOClauses, false, null);
						Set<Annotatable> visitedRegions = new HashSet<Annotatable>();
						for( ACCAnnotation cAnnot : computeAnnots ) {
							Annotatable pAt = cAnnot.getAnnotatable();
							if( !visitedRegions.contains(pAt) ) {
								visitedRegions.add(pAt);
								ACCAnnotation dAnnot = AnalysisTools.ipFindFirstPragmaInParent(pAt, ACCAnnotation.class, "data", funcCallList, null);
								if( dAnnot != null ) {
									Annotatable dAt = dAnnot.getAnnotatable();
									if( !visitedRegions.contains(dAt) ) {
										visitedRegions.add(dAt);
										FunctionCall waitCall = new FunctionCall(new NameID("acc_wait_all"));
										Statement waitCallStmt = new ExpressionStatement(waitCall);
										Traversable parent = dAt.getParent();
										if( (parent instanceof CompoundStatement) && (dAt instanceof Statement) ) {
											((CompoundStatement)parent).addStatementAfter((Statement)dAt, waitCallStmt);
										} else {
											Tools.exit("[ERROR in the ACC2GPUTranslator.start()] pipe clauses are attached to wrong IR object:\n" + 
													"ACC Annotation: " + dAnnot +
													AnalysisTools.getEnclosingContext(dAt));
										}
									}
								}
							}
						}
					} else {
						Annotatable pAt = tAnnot.getAnnotatable();
						FunctionCall waitCall = new FunctionCall(new NameID("acc_wait_all"));
						Statement waitCallStmt = new ExpressionStatement(waitCall);
						Traversable parent = pAt.getParent();
						if( (parent instanceof CompoundStatement) && (pAt instanceof Statement) ) {
							((CompoundStatement)parent).addStatementAfter((Statement)pAt, waitCallStmt);
						} else {
							Tools.exit("[ERROR in the ACC2GPUTranslator.start()] pipe clauses are attached to wrong IR object:\n" + 
									"ACC Annotation: " + tAnnot +
									AnalysisTools.getEnclosingContext(pAt));
						}
					}
				}
			}
		}
		VariableDeclarator kernel_declarator = new VariableDeclarator(new NameID("kernel_str"), new ArraySpecifier(new IntegerLiteral(accKernelsList.size())));
		Identifier kernel_str = new Identifier(kernel_declarator);
		Declaration kernel_decl = new VariableDeclaration(OpenACCSpecifier.STRING, kernel_declarator);
		DeclarationStatement kernel_stmt = new DeclarationStatement(kernel_decl);
		for( int i=0; i<main_List.size(); i++ ) {
			Procedure tmain = main_List.get(i);
			Statement taccInitStmt = accInitStmt_List.get(i);
			TransformTools.addStatementBefore(tmain.getBody(), taccInitStmt, kernel_stmt.clone());
			for(int j = accKernelsList.size()-1; j >= 0; j--)
			{
				AssignmentExpression assignmentExpression = new AssignmentExpression(
						new ArrayAccess(new NameID("kernel_str"), new IntegerLiteral(j)),
						AssignmentOperator.NORMAL,
						new StringLiteral(accKernelsList.get(j))
						);
				TransformTools.addStatementBefore(tmain.getBody(), taccInitStmt, new ExpressionStatement(assignmentExpression));
			}

			FunctionCall initCall = (FunctionCall)((ExpressionStatement)taccInitStmt).getExpression();
			initCall.addArgument(new IntegerLiteral(accKernelsList.size()));
			initCall.addArgument(kernel_str.clone());
		}

		handleUpdateDirectives(updateAnnots);
		handleHostDataDirectives(hostDataAnnots);
		handleWaitDirectives(waitAnnots);
	}
	protected void GPUInitializer() {
		/////////////////////////////////////////////////////////////////
		// Read command-line options and set corresponding parameters. //
		/////////////////////////////////////////////////////////////////
		
		String value = Driver.getOptionValue("useMallocPitch");
		if( value != null ) {
			opt_MallocPitch = true;
			FunctionCall optMPPrintCall = new FunctionCall(new NameID("printf"));
			optMPPrintCall.addArgument(new StringLiteral("====> MallocPitch Opt is used.\\n"));
			optPrintStmts.add( new ExpressionStatement(optMPPrintCall) );
		}

		value = Driver.getOptionValue("useMatrixTranspose");
		if( value != null ) {
			opt_MatrixTranspose = true;
			FunctionCall optMTPrintCall = new FunctionCall(new NameID("printf"));
			optMTPrintCall.addArgument(new StringLiteral("====> MatrixTranspose Opt is used.\\n"));
			optPrintStmts.add( new ExpressionStatement(optMTPrintCall) );
		}

		value = Driver.getOptionValue("useParallelLoopSwap");
		if( value != null ) {
			FunctionCall optPLPrintCall = new FunctionCall(new NameID("printf"));
			optPLPrintCall.addArgument(new StringLiteral("====> ParallelLoopSwap Opt is used.\\n"));
			optPrintStmts.add( new ExpressionStatement(optPLPrintCall) );
		}

		value = Driver.getOptionValue("useLoopCollapse");
		if( value != null ) {
			opt_LoopCollapse = true;
			FunctionCall optLCPrintCall = new FunctionCall(new NameID("printf"));
			optLCPrintCall.addArgument(new StringLiteral("====> LoopCollapse Opt is used.\\n"));
			optPrintStmts.add( new ExpressionStatement(optLCPrintCall) );
		}

		value = Driver.getOptionValue("addSafetyCheckingCode");
		if( value != null ) {
			opt_addSafetyCheckingCode = true;
			FunctionCall optSCPrintCall = new FunctionCall(new NameID("printf"));
			optSCPrintCall.addArgument(new StringLiteral("====> Safety-checking code is added.\\n"));
			optPrintStmts.add( new ExpressionStatement(optSCPrintCall) );
		}

		value = Driver.getOptionValue("forceSyncKernelCall");
		if( value != null ) {
			opt_forceSyncKernelCall = true;
			FunctionCall optSKCPrintCall = new FunctionCall(new NameID("printf"));
			optSKCPrintCall.addArgument(new StringLiteral("====> Explicit synchronization is forced.\\n"));
			optPrintStmts.add( new ExpressionStatement(optSKCPrintCall) );
		}

		value = Driver.getOptionValue("maxNumGangs");
		if( value != null ) {
			FunctionCall maxNumBLKPrintCall = new FunctionCall(new NameID("printf"));
			maxNumBLKPrintCall.addArgument(new StringLiteral("====> Maximum number of Gangs: "+value+" \\n"));
			confPrintStmts.add( new ExpressionStatement(maxNumBLKPrintCall) );
			maxNumGangs = Integer.valueOf(value).intValue();
		}
		value = Driver.getOptionValue("defaultNumWorkers");
		if( value != null ) {
			defaultNumWorkers = Integer.valueOf(value).intValue();
		} else {
			if( opt_LoopCollapse ) {
				defaultNumWorkers = 512;
			}
		}
		FunctionCall BLKSizePrintCall = new FunctionCall(new NameID("printf"));
		BLKSizePrintCall.addArgument(new StringLiteral("====> Default Number of Workers per Gang: "
				+defaultNumWorkers+" \\n"));
		confPrintStmts.add( new ExpressionStatement(BLKSizePrintCall) );
		
		value = Driver.getOptionValue("programVerification");
		if( value != null ) {
			FunctionCall VerifyCall = new FunctionCall(new NameID("printf"));
			if( value.equals("1") ) {
				VerifyCall.addArgument(new StringLiteral("====> Verify the correctness of CPU-GPU memory transfers\\n"));
				confPrintStmts.add( new ExpressionStatement(VerifyCall) );
				memtrVerification = true;
			} else if( value.equals("2") ) {
				VerifyCall.addArgument(new StringLiteral("====> Verify the correctness of GPU kernel translations\\n"));
				confPrintStmts.add( new ExpressionStatement(VerifyCall) );
				kernelVerification = true;
			}
		}
		
		value = Driver.getOptionValue("defaultMarginOfError");
		if( value != null ) {
			double epsilon = Double.valueOf(value).doubleValue();
			marginOfError = new FloatLiteral(epsilon);
			if( kernelVerification ) {
				FunctionCall marginCall = new FunctionCall(new NameID("printf"));
				marginCall.addArgument(new StringLiteral("      (Acceptable margin of errors = " + value + ")\\n"));
				confPrintStmts.add( new ExpressionStatement(marginCall) );
			}
		}
		
		value = Driver.getOptionValue("minValueToCheck");
		if( value != null ) {
			double epsilon = Double.valueOf(value).doubleValue();
			minCheckValue = new FloatLiteral(epsilon);
			if( kernelVerification ) {
				FunctionCall minCheckCall = new FunctionCall(new NameID("printf"));
				minCheckCall.addArgument(new StringLiteral("      (Minimum value to check errors = " + value + ")\\n"));
				confPrintStmts.add( new ExpressionStatement(minCheckCall) );
			}
		}
		
		value = Driver.getOptionValue("enableFaultInjection");
		if( value != null ) {
			enableFaultInjection = true;
		}
		
		value = Driver.getOptionValue("enableCustomProfiling");
		if( value != null ) {
			enableCustomProfiling = true;
		}

		value = Driver.getOptionValue("useUnrollingOnReduction");
		if( value != null ) {
			opt_UnrollingOnReduction = true;
			FunctionCall optRUPrintCall = new FunctionCall(new NameID("printf"));
			optRUPrintCall.addArgument(new StringLiteral("====> Unrolling-on-reduction Opt is used.\\n"));
			optPrintStmts.add( new ExpressionStatement(optRUPrintCall) );
		}

		value = Driver.getOptionValue("gpuMemTrOptLevel");
		if( value != null ) {
			gpuMemTrOptLevel = Integer.valueOf(value).intValue();
		}
		FunctionCall optMemTrPrintCall = new FunctionCall(new NameID("printf"));
		optMemTrPrintCall.addArgument(new StringLiteral("====> CPU-GPU Mem Transfer Opt Level: " 
				+ gpuMemTrOptLevel + "\\n"));
		optPrintStmts.add( new ExpressionStatement(optMemTrPrintCall) );

		value = Driver.getOptionValue("gpuMallocOptLevel");
		if( value != null ) {
			gpuMallocOptLevel = Integer.valueOf(value).intValue();
		}
		FunctionCall optCudaMallocPrintCall = new FunctionCall(new NameID("printf"));
		optCudaMallocPrintCall.addArgument(new StringLiteral("====> GPU Malloc Opt Level: " 
				+ gpuMallocOptLevel + "\\n"));
		optPrintStmts.add( new ExpressionStatement(optCudaMallocPrintCall) );

		value = Driver.getOptionValue("assumeNonZeroTripLoops");
		if( value != null ) {
			FunctionCall assmNnZrTrLpsPrintCall = new FunctionCall(new NameID("printf"));
			assmNnZrTrLpsPrintCall.addArgument(new StringLiteral("====> Assume that all loops have non-zero iterations.\\n"));
			optPrintStmts.add( new ExpressionStatement(assmNnZrTrLpsPrintCall) );
		}

		value = Driver.getOptionValue("assumeNoAliasingAmongKernelArgs");
		if( value != null ) {
			FunctionCall assmNoAliasingPrintCall = new FunctionCall(new NameID("printf"));
			assmNoAliasingPrintCall.addArgument(new StringLiteral("====> Assume that there is no aliasing among kernel arguments.\\n"));
			optPrintStmts.add( new ExpressionStatement(assmNoAliasingPrintCall) );
		}

		value = Driver.getOptionValue("UEPRemovalOptLevel");
		if( value != null ) {
			FunctionCall UEPRemovalCall = new FunctionCall(new NameID("printf"));
			UEPRemovalCall.addArgument(new StringLiteral("====> UEPRemoval Opt. Level: "+value+"\\n"));
			optPrintStmts.add( new ExpressionStatement(UEPRemovalCall) );
		}

		value = Driver.getOptionValue("doNotRemoveUnusedSymbols");
		if( value != null ) {
			int optLevel = Integer.valueOf(value).intValue();
			FunctionCall NoUnusedSymDelCall = new FunctionCall(new NameID("printf"));
			StringLiteral str1;
			if( optLevel == 1 ) {
				str1 = new StringLiteral("====> Do not remove unused symbols or procedures.\\n");
			} else if( optLevel == 2 ) {
				str1 = new StringLiteral("====> Do not remove unused procedures.\\n");
			} else if( optLevel == 3 ) {
				str1 = new StringLiteral("====> Do not remove unused symbols.\\n");
			} else {
				str1 = new StringLiteral("====> Remove both unused symbols and procedures.\\n");
			}
			NoUnusedSymDelCall.addArgument(str1);
			optPrintStmts.add( new ExpressionStatement(NoUnusedSymDelCall) );
		}

		value = Driver.getOptionValue("extractTuningParameters");
		if( value != null ) {
			if( value.equals("1") ) {
				tuningParamFile="TuningOptions.txt";
			} else {
				tuningParamFile=value;
			}
		}

		value = Driver.getOptionValue("MemTrOptOnLoops");
		if( value != null ) {
			opt_MemTrOptOnLoops = true;
			FunctionCall MemTrOptOnLoopsCall = new FunctionCall(new NameID("printf"));
			MemTrOptOnLoopsCall.addArgument(new StringLiteral("====> Apply Memory Transfer Optimization on Loops.\\n"));
			optPrintStmts.add( new ExpressionStatement(MemTrOptOnLoopsCall) );
		}

		value = Driver.getOptionValue("localRedVarConf");
		if( value != null ) {
			localRedVarConf = Integer.valueOf(value).intValue();
			FunctionCall localRedVarConfCall = new FunctionCall(new NameID("printf"));
			localRedVarConfCall.addArgument(new StringLiteral("====> local array reduction variable configuration = " +
					localRedVarConf + "\\n"));
			optPrintStmts.add( new ExpressionStatement(localRedVarConfCall) );
		}
		
		value = Driver.getOptionValue("AccPrivatization");
		if( value != null ) {
			FunctionCall AccPrivatizationCall = new FunctionCall(new NameID("printf"));
			AccPrivatizationCall.addArgument(new StringLiteral("====> AccPrivatization Opt. Level: "+value+"\\n"));
			optPrintStmts.add( new ExpressionStatement(AccPrivatizationCall) );
		}
		
		value = Driver.getOptionValue("AccReduction");
		if( value != null ) {
			FunctionCall AccReductionCall = new FunctionCall(new NameID("printf"));
			AccReductionCall.addArgument(new StringLiteral("====> AccReduction Opt. Level: "+value+"\\n"));
			optPrintStmts.add( new ExpressionStatement(AccReductionCall) );
		}
		
		value = Driver.getOptionValue("AccParallelization");
		if( value != null ) {
			FunctionCall AccParallelCall = new FunctionCall(new NameID("printf"));
			AccParallelCall.addArgument(new StringLiteral("====> AccParallelization Opt. Level: "+value+"\\n"));
			optPrintStmts.add( new ExpressionStatement(AccParallelCall) );
		}
		
		value = Driver.getOptionValue("SkipGPUTranslation");
		if( value != null ) {
			SkipGPUTranslation = Integer.valueOf(value).intValue();
		}
		
		value = Driver.getOptionValue("printConfigurations");
		if( value != null ) {
			opt_PrintConfigurations = true;
		}

		value = Driver.getOptionValue("acc2gpu");
		if( value != null ) {
			if( Integer.valueOf(value).intValue() == 2) {
				opt_GenDistOpenACC = true;
			}
		}

		value = Driver.getOptionValue("targetArch");
		if( value != null ) {
			targetArch = Integer.valueOf(value).intValue();
		} else {
			value = System.getenv("OPENARC_ARCH");
			if( value != null)
			{
				targetArch = Integer.valueOf(value).intValue();
			}
		}
		
		if( targetArch == 3 ) {
			enablePipeTransformation = true;
		}

		value = Driver.getOptionValue("assumeNoAliasingAmongKernelArgs");
		if( value != null ) {
			opt_AssumeNoAliasing = true;
		}

		value = Driver.getOptionValue("skipKernelLoopBoundChecking");
		if( value != null ) {
			opt_skipKernelLoopBoundChecking = true;
		}
		
		OpenACCHeaderEndMap = new HashMap<TranslationUnit, Declaration>();
	}
	
	protected void handleDeclareDirectives(List<ACCAnnotation> declareAnnots) {
		Set<String> mallocCallSet =
		new HashSet<String>(Arrays.asList("malloc", "calloc", "_mm_malloc", "posix_memalign", "aligned_alloc", "valloc"));
		for( ACCAnnotation dAnnot : declareAnnots ) {
			DataRegionType regionT;
			List<Statement> firstStmts = new LinkedList<Statement>();
			List<Statement> lastStmts = new LinkedList<Statement>();
			Annotatable at = dAnnot.getAnnotatable();
			Procedure cProc = IRTools.getParentProcedure(at);
			//[FIXME] firstStmts should be the malloc statements for the data in the declare directive.
			//For now, we simply checks malloc/calloc/_mm_malloc/posix_memalign/alligned_alloc/valloc functions in the direct children.
			if( cProc == null ) {
				//Decalre directive is for implicit region for the whole program.
				regionT = DataRegionType.ImplicitProgramRegion;
				for( FunctionCall acc_init : acc_init_list ) {
					Statement refStmt = acc_init.getStatement();
					Statement firstMallocStmt = null;
					Statement lastMallocStmt = null;
					if( refStmt.getParent() instanceof CompoundStatement ) {
						List<Traversable> childlist = refStmt.getParent().getChildren();
						for( int i=0; i<childlist.size(); i++ ) {
							List<FunctionCall> fCallList = IRTools.getFunctionCalls(childlist.get(i));
							boolean foundMallocStmt = false;
							if( fCallList != null ) {
								for( FunctionCall fCall : fCallList ) {
									if( mallocCallSet.contains(fCall.getName().toString()) ) {
										foundMallocStmt = true;
										break;
									}
								}
							}
							if( foundMallocStmt ) {
								if( firstMallocStmt == null ) {
									firstMallocStmt = (Statement)childlist.get(i);
								}
								lastMallocStmt = (Statement)childlist.get(i);
							} else {
								if( firstMallocStmt != null ) {
									break;
								}
							}
						}
					}
					if( lastMallocStmt != null ) {
						firstStmts.add(lastMallocStmt);
					} else {
						firstStmts.add(refStmt);
					}
				}
				for( FunctionCall acc_shutdown : acc_shutdown_list ) {
					lastStmts.add(acc_shutdown.getStatement());
				}
				if( firstStmts.isEmpty() ) {
					TranslationUnit tu = null;
					Traversable t = at;
					while( t != null ) {
						if( t instanceof TranslationUnit ) {
							tu = (TranslationUnit)t;
							break;
						}
						t = t.getParent();
					}
					Tools.exit("[ERROR in ACC2GPUTranslator.handleDeclareDirectives()] the following declare directive is" +
							"for implicit program-level data region, but the compiler can not determine the entry to the" +
							"implicit data region. At least one acc_init() function or main function should exist; exit!\n" +
							"OpenACC declare directive: " + dAnnot + "\n" +
							"Enclosing file: " + tu.getInputFilename() + "\n");
				}
			} else {
				//Declare directive is for implicit region within a function.
				regionT = DataRegionType.ImplicitProcedureRegion;
				CompoundStatement funcBody = cProc.getBody();
				Statement refStmt = IRTools.getFirstNonDeclarationStatement(funcBody);
				Statement firstMallocStmt = null;
				Statement lastMallocStmt = null;
				List<Traversable> childlist = funcBody.getChildren();
				for( int i=0; i<childlist.size(); i++ ) {
					List<FunctionCall> fCallList = IRTools.getFunctionCalls(childlist.get(i));
					boolean foundMallocStmt = false;
					if( fCallList != null ) {
						for( FunctionCall fCall : fCallList ) {
							if( mallocCallSet.contains(fCall.getName().toString()) ) {
								foundMallocStmt = true;
								break;
							}
						}
					}
					if( foundMallocStmt ) {
						if( firstMallocStmt == null ) {
							firstMallocStmt = (Statement)childlist.get(i);
						}
						lastMallocStmt = (Statement)childlist.get(i);
					} else {
						if( firstMallocStmt != null ) {
							break;
						}
					}
				}
				if( lastMallocStmt != null ) {
					firstStmts.add(lastMallocStmt);
				} else {
					firstStmts.add(refStmt);
				}
				/*
				 * Find return statements in the implicit function.
				 */
				BreadthFirstIterator riter = new BreadthFirstIterator(funcBody);
				riter.pruneOn(Expression.class); /* optimization */
				for (;;)
				{
					ReturnStatement stmt = null;

					try {
						stmt = (ReturnStatement)riter.next(ReturnStatement.class);
					} catch (NoSuchElementException e) {
						break;
					}

					lastStmts.add(stmt);
				}
			}
			//Current implementation handles IR-symbols only (GPU memory allocation unit is a class object, but not each class member.) 
			handleDataClauses(dAnnot, firstStmts, lastStmts, regionT, IRSymbolOnly);
		}
	}
	
	protected void handleUpdateDirectives(List<ACCAnnotation> updateAnnots) {
		for( ACCAnnotation uAnnot : updateAnnots ) {
			handleUpdateClauses(uAnnot, IRSymbolOnly);
		}
	}
	
	protected void handleHostDataDirectives(List<ACCAnnotation> hostDataAnnots) {
		if( !hostDataAnnots.isEmpty() ) {
			for( ACCAnnotation uAnnot : hostDataAnnots ) {
				handleUseDevicesClauses(uAnnot, IRSymbolOnly);
			}
		}
	}
	
	protected void handleWaitDirectives(List<ACCAnnotation> waitAnnots) {
		for(ACCAnnotation wAnnot : waitAnnots ) {
			Annotatable at = wAnnot.getAnnotatable();
			Object waitArg = wAnnot.get("wait");
			ExpressionStatement waitCallStmt = null;
			if( waitArg instanceof Expression ) {
				FunctionCall waitCall = new FunctionCall(new NameID("acc_wait"), (Expression)waitArg);
				waitCallStmt = new ExpressionStatement(waitCall);
			} else {
				FunctionCall waitCall = new FunctionCall(new NameID("acc_wait_all"));
				waitCallStmt = new ExpressionStatement(waitCall);
			}
			if( at instanceof Statement ) {
				((Statement)at).swapWith(waitCallStmt);
				waitCallStmt.annotate(wAnnot);
			} else {
				Tools.exit("[ERROR in ACC2GPUTranslator.handleWaitDirectives()] unexpected type of wait annotation; exit\n" +
						"ACCAnnotation: " + wAnnot + "\n");
			}
		}
		
	}
	
	protected void handleDataRegions(Procedure cProc, List<ACCAnnotation> dataRegionAnnots) {
		DataRegionType regionT = DataRegionType.ExplicitDataRegion;
		for(ACCAnnotation dAnnot : dataRegionAnnots ) {
			Annotatable at = dAnnot.getAnnotatable();
			if( !(at instanceof Procedure) ) { //Data directive for a procedure is an internal one; skip it.
				List<Statement> inStmts = new LinkedList<Statement>();
				List<Statement> outStmts = new LinkedList<Statement>();
				inStmts.add((Statement)at);
				outStmts.add((Statement)at);
				//Current implementation handles IR-symbols only (GPU memory allocation unit is a class object, but not each class member.) 
				handleDataClauses(dAnnot, inStmts, outStmts, regionT, IRSymbolOnly);
			}
		}
	}
	
	protected void handleMemoryRuntimeAPIs(Procedure cProc, List<FunctionCall> fCallList) {
		//Transform OpenACC runtime library APIs to allocate data on constant memory.
		runtimeTransformationForConstMemory(cProc, fCallList);
		//Transform OpenACC runtime library APIs to allocate data on texture memory.
	}
	
	protected void convComputeRegionsToGPUKernels(Procedure cProc, List<ACCAnnotation> parallelRegionAnnots,
			List<ACCAnnotation> kernelsRegionAnnots) {
		DataRegionType regionT = DataRegionType.ComputeRegion;
		List<ACCAnnotation> cRegionAnnots = new LinkedList<ACCAnnotation>();
		cRegionAnnots.addAll(parallelRegionAnnots);
		cRegionAnnots.addAll(kernelsRegionAnnots);
		int kernelCnt = 0;
		//Find an optimal point to insert kernel-configuration-related statements.
		for( ACCAnnotation cAnnot : cRegionAnnots ) {
			String GPUKernelName = cProc.getName().toString() + "_kernel" + kernelCnt++;
			Annotatable at = cAnnot.getAnnotatable();
			String kernelType = "kernels";
			if( at.containsAnnotation(ACCAnnotation.class, "parallel") ) {
				kernelType = "parallel";
			}
			//DEBUG: if ACCAnnotation key, parallel or kernels, is set to "false", GPU translation
			//will be skipped.
			ACCAnnotation tAnnot = at.getAnnotation(ACCAnnotation.class, kernelType);
			if( tAnnot.get(kernelType).equals("false") ) {
				continue;
			}
			Annotatable confRefStmt = (Annotatable)findKernelConfInsertPoint(at, IRSymbolOnly);
			Annotation iAnnot = confRefStmt.getAnnotation(ACCAnnotation.class, "internal");
			if( iAnnot == null ) {
				iAnnot = new ACCAnnotation("internal", "_directive");
				iAnnot.setSkipPrint(true);
				confRefStmt.annotate(iAnnot);
			}
			iAnnot.put("kernelConfPt_"+GPUKernelName, "_clause");
		}
		
		//If nested gang/worker/vector loops contain private/reduction clauses,
		//these clauses are merged into the outermost loop, which is necessary for 
		//correct privateTransformation/reductionTransformation.
		List<ACCAnnotation> loopAnnots = null;
		for( ACCAnnotation cAnnot : cRegionAnnots ) {
			Annotatable at = cAnnot.getAnnotatable();
			for( String tClause : ACCAnnotation.worksharingClauses ) {
				loopAnnots = AnalysisTools.ipCollectPragmas(at, ACCAnnotation.class, tClause, null);
				if( (loopAnnots != null) && !loopAnnots.isEmpty() ) {
					for( ACCAnnotation tAnnot : loopAnnots ) {
						ForLoop tLoop = (ForLoop)tAnnot.getAnnotatable();
						ForLoop oLoop = null;
						Traversable tt = tLoop.getParent();
						boolean outermostloop = true;
						while( tt != null ) {
							if( (tt instanceof Annotatable) && ((Annotatable)tt).containsAnnotation(ACCAnnotation.class, tClause) ) {
								outermostloop = false;
								oLoop = (ForLoop)tt;
							}
							tt = tt.getParent();
						}
						if( !outermostloop && (oLoop != null) ) {
							//Current loop is not the outermost loop; move private/reduction clauses to the outermost loop.
							ACCAnnotation otAnnot = oLoop.getAnnotation(ACCAnnotation.class, "private");
							ACCAnnotation ttAnnot = tLoop.getAnnotation(ACCAnnotation.class, "private");
							ACCAnnotation iotAnnot = oLoop.getAnnotation(ACCAnnotation.class, "internal");
							ACCAnnotation ittAnnot = tLoop.getAnnotation(ACCAnnotation.class, "internal");
							if( ttAnnot != null ) {
								Set<SubArray> ttPSet = ttAnnot.get("private");
								Set<Symbol> ittPSet = ittAnnot.get("accprivate");
								Set<SubArray> otPSet = null;
								Set<Symbol> iotPSet = null;
								if( otAnnot == null ) {
									tAnnot.put("private", ttPSet);
									if( ittPSet != null ) {
										ittAnnot.put("accprivate", ittPSet);
									}
								} else {
									otPSet = otAnnot.get("private");
									iotPSet = iotAnnot.get("accprivate");
									if( iotPSet == null ) {
										iotPSet = new HashSet<Symbol>();
										iotAnnot.put("accprivate", iotPSet);
									}
									Set<Symbol> otPSSet = AnalysisTools.subarraysToSymbols(otPSet, IRSymbolOnly);
									for(SubArray tSub : ttPSet) {
										Symbol tSym = AnalysisTools.subarrayToSymbol(tSub, IRSymbolOnly);
										if( (tSym != null) && !otPSSet.contains(tSym) ) {
											otPSet.add(tSub);
											otPSSet.add(tSym);
											iotPSet.add(tSym);
										}
									}
								}
								ttAnnot.remove("private");
								ittAnnot.remove("accprivate");
							}
							otAnnot = oLoop.getAnnotation(ACCAnnotation.class, "reduction");
							ttAnnot = tLoop.getAnnotation(ACCAnnotation.class, "reduction");
							if( ttAnnot != null ) {
								Map<ReductionOperator, Set<SubArray>> ttRMap = 
									(Map<ReductionOperator, Set<SubArray>>)ttAnnot.get("reduction");
								Set<Symbol> ittRSet = ittAnnot.get("accreduction");
								Map<ReductionOperator, Set<SubArray>> otRMap = null;
								Set<Symbol> iotRSet = null;
								if( otAnnot == null ) {
									tAnnot.put("reduction", ttRMap);
									if( ittRSet != null ) {
										ittAnnot.put("accreduction", ittRSet);
									}
								} else {
									otRMap = 
										(Map<ReductionOperator, Set<SubArray>>)otAnnot.get("reduction");
									iotRSet = iotAnnot.get("accreduction");
									if( iotRSet == null ) {
										iotRSet = new HashSet<Symbol>();
										iotAnnot.put("accreduction", iotRSet);
									}
									for( ReductionOperator rOp : ttRMap.keySet() ) {
										if( otRMap.containsKey(rOp) ) {
											Set<SubArray> ttRSet = ttRMap.get(rOp);
											Set<SubArray> otRSet = otRMap.get(rOp);
											Set<Symbol> otRSSet = AnalysisTools.subarraysToSymbols(otRSet, IRSymbolOnly);
											for(SubArray tSub : ttRSet ) {
												Symbol tSym = AnalysisTools.subarrayToSymbol(tSub, IRSymbolOnly);
												if( (tSym != null) && !otRSSet.contains(tSym) ) {
													otRSet.add(tSub);
													otRSSet.add(tSym);
													iotRSet.add(tSym);
												}
											}
										} else {
											Set<SubArray> ttRSet = ttRMap.get(rOp);
											otRMap.put(rOp, ttRSet);
											iotRSet.addAll(AnalysisTools.subarraysToSymbols(ttRSet, IRSymbolOnly));
										}
									}
								}
								ttAnnot.remove("reduction");
								ittAnnot.remove("accreduction");
							}
						}
					}
				}
			}
		}
		
		//Perform actual compute-region to GPU kernel conversion.
		kernelCnt = 0;
		for( ACCAnnotation pAnnot : parallelRegionAnnots ) {
			String GPUKernelName = cProc.getName().toString() + "_kernel" + kernelCnt++;
			Annotatable at = pAnnot.getAnnotatable();
			//DEBUG: if ACCAnnotation key, parallel or kernels, is set to "false", GPU translation
			//will be skipped.
			if( pAnnot.get("parallel").equals("false") ) {
				at.removeAnnotations(ACCAnnotation.class);
				CommentAnnotation cmt = new CommentAnnotation(GPUKernelName);
				at.annotate(cmt);
				continue;
			}
			//HI_set_async will be generated before the data clause instead
			/*if(pAnnot.containsKey("async"))
            {
                FunctionCall setAsyncCall = new FunctionCall(new NameID("HI_set_async"));
                if(pAnnot.get("async") != null)
                {
                    setAsyncCall.addArgument(new NameID(pAnnot.get("async").toString()));
                }
                else
                {
                    setAsyncCall.addArgument(new NameID("DEFAULT_ASYNC_QUEUE"));
                }
                TransformTools.addStatementBefore((CompoundStatement)((Statement)at).getParent(), (Statement)at, new ExpressionStatement(setAsyncCall));
            }*/
			List<Statement> inStmts = new LinkedList<Statement>();
			List<Statement> outStmts = new LinkedList<Statement>();
			inStmts.add((Statement)at);
			outStmts.add((Statement)at);
			handleDataClauses(pAnnot, inStmts, outStmts, regionT, IRSymbolOnly);
			extractComputeRegion(cProc, pAnnot, "parallel", GPUKernelName, IRSymbolOnly);
		}
		for( ACCAnnotation kAnnot : kernelsRegionAnnots ) {
			String GPUKernelName = cProc.getName().toString() + "_kernel" + kernelCnt++;
			Annotatable at = kAnnot.getAnnotatable();
			//DEBUG: if ACCAnnotation key, parallel or kernels, is set to "false", GPU translation
			//will be skipped.
			if( kAnnot.get("kernels").equals("false") ) {
				at.removeAnnotations(ACCAnnotation.class);
				CommentAnnotation cmt = new CommentAnnotation(GPUKernelName);
				at.annotate(cmt);
				continue;
			}

			/*if(kAnnot.containsKey("async"))
            {
                FunctionCall setAsyncCall = new FunctionCall(new NameID("HI_set_async"));
                if(kAnnot.get("async") != null)
                {
                    setAsyncCall.addArgument(new NameID(kAnnot.get("async").toString()));
                }
                else
                {
                    setAsyncCall.addArgument(new NameID("DEFAULT_ASYNC_QUEUE"));
                }
                TransformTools.addStatementBefore((CompoundStatement)((Statement)at).getParent(), (Statement)at, new ExpressionStatement(setAsyncCall));
            }*/
			List<Statement> inStmts = new LinkedList<Statement>();
			List<Statement> outStmts = new LinkedList<Statement>();
			inStmts.add((Statement)at);
			outStmts.add((Statement)at);
			handleDataClauses(kAnnot, inStmts, outStmts, regionT, IRSymbolOnly);
			extractComputeRegion(cProc, kAnnot, "kernels", GPUKernelName, IRSymbolOnly);
		}
		numComputeRegions += kernelCnt;
	}

    protected abstract void handleAtomicAnnots(List<ACCAnnotation> atomicAnnots);

	protected abstract void handleDataClauses(ACCAnnotation dAnnot, List<Statement> inStmts, List<Statement> outStmts, DataRegionType dReginType,
			boolean IRSymbolOnly);
	
	protected abstract void handleUpdateClauses(ACCAnnotation uAnnot, boolean IRSymbolOnly);
	
	protected abstract void handleUseDevicesClauses(ACCAnnotation uAnnot, boolean IRSymbolOnly);
	
	protected abstract void extractComputeRegion(Procedure cProc, ACCAnnotation cAnnot, String cRegionKind, String new_func_name,
			boolean IRSymbolOnly);
	
	protected abstract void runtimeTransformationForConstMemory(Procedure cProc, List<FunctionCall> fCallList );

	protected Traversable findKernelConfInsertPoint(Annotatable region, boolean IRSymbolOnly) {
		Set<Symbol> reductionSymbols = null;
		Traversable refstmt = region; //Refer to the optimal kernel configuration insertion point.
		if( region.containsAnnotation(ACCAnnotation.class, "reduction") ) {
			//Find reduction symbols.
			reductionSymbols = AnalysisTools.getReductionSymbols(region, IRSymbolOnly);
		}
		//Find symbols used in iterspace expressions.
		Set<Symbol> itrSymbols = new HashSet<Symbol>();
		List<ACCAnnotation> itsAnnots = IRTools.collectPragmas(region, ACCAnnotation.class, "iterspace");
		for( ACCAnnotation itsAnnot : itsAnnots ) {
			Expression exp = (Expression)itsAnnot.get("iterspace");
			itrSymbols.addAll(SymbolTools.getAccessedSymbols(exp));
		}
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//Find optimal kernel configuration statement insertion point.                                                      //
		//If iteration space size is not changed, the configuraiton-insertion point can be moved up to enclosing statement. //
		//FIXME: If reduction symbol is local, the insertion point can not go beyond the symbol scope, but not checked      //
		//in the current implementation.                                                                                    //
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		Traversable tChild = region;
		Traversable tt = tChild.getParent();
		while( (tt != null) && !(tt instanceof Procedure) ) {
			if( !(tt instanceof CompoundStatement) ) {
				tChild = tt;
				tt = tt.getParent();
				continue;
			}
			//TODO: for now, optimal point is searched within a procedure boundary, but it can be further optimized
			//across procedure boundary.
			if( tt.getParent() instanceof Procedure ) {
				refstmt = tChild;
				break;
			}
			Set<Symbol> initSet = AnalysisTools.getInitializedSymbols(tt);
			initSet.retainAll(itrSymbols);
			Set<Symbol> defSet = DataFlowTools.getDefSymbol(tt);
			boolean gridSizeNotChanged = false;
			defSet.retainAll(itrSymbols);
			if( defSet.isEmpty() && initSet.isEmpty() ) {
				gridSizeNotChanged = true;
				if( reductionSymbols != null ) {
					for( Traversable t : tt.getChildren() ) {
						if( (t != tChild) && (t instanceof Annotatable) ) {
							if( AnalysisTools.containsReductionSymbols(t, reductionSymbols, IRSymbolOnly) ) {
								//TODO: too conservative.
								gridSizeNotChanged = false;
								break;
							}
						}
					}
				}
				if( gridSizeNotChanged ) {
					//////////////////////////////////////////////////////////////////////////
					//If a function called in the tt has compute regions, conservatively    //
					//assume that the grid-size may be changed in the called function.      //
					//FIXME: To be precise, we should check whether itrSymbols are modified //
					//       or not interprocedurally, but not checked here.                //
					//////////////////////////////////////////////////////////////////////////
					List<FunctionCall> callList = IRTools.getFunctionCalls(tt);
					for( FunctionCall fcall : callList ) {
						if( (fcall instanceof KernelFunctionCall) || AnalysisTools.isCudaCall(fcall) 
								|| StandardLibrary.contains(fcall) ) {
							continue;
						}
						Procedure cProc = fcall.getProcedure();
						if( cProc == null ) {
							continue;
						} else {
							List<ACCAnnotation> cList = 
								AnalysisTools.collectPragmas(cProc, ACCAnnotation.class, ACCAnnotation.computeRegions, false);
							if( cList.size() > 0 ) {
								gridSizeNotChanged = false;
								break;
							}
						}
					}
				}
			}
			if( gridSizeNotChanged ) {
				Traversable tGrandChild = tChild;
				tChild = tt;
				tt = tt.getParent();
				if( tt instanceof IfStatement ) {
					refstmt = tt;
				} else if( (tt instanceof DoLoop) || (tt instanceof WhileLoop) ) {
					refstmt = tt;
				} else if( tt instanceof ForLoop ) {
					Expression iVar = LoopTools.getIndexVariable((ForLoop)tt);
					Set<Symbol> iSyms = SymbolTools.getAccessedSymbols(iVar);
					iSyms.retainAll(itrSymbols);
					if( iSyms.isEmpty() ) {
						refstmt = tt;
					} else {
						refstmt = tGrandChild;
						break;
					}
				} else {
					refstmt = tChild;
				}
			} else {
				break;
			}
		}
		return refstmt;
	}
	
	/**
	 * Apply stripmining to fit the iteration size of a worksharing loop into the specified gang/worker configuration.
	 *
	 * @param cAnnot
	 * @param cRegionKind
	 */
	protected abstract ForLoop worksharingLoopStripmining(Procedure cProc, ACCAnnotation cAnnot, String cRegionKind);
	
	protected void copyUserSpecifierDeclarations(TranslationUnit tUnit, Traversable tr, Set<Symbol> accessedSymbols,
			Declaration refDecl) {
		Declaration prevDecl = null;
		Declaration newDecl = null;
		SymbolTable symTable = null;
		while( tr != null ) {
			if( tr instanceof SymbolTable ) {
				symTable = (SymbolTable)tr;
				break;
			} else {
				tr = tr.getParent();
			}
		}
		if( symTable == null ) {
			PrintTools.println("[ERROR in ACC2GPUTranslator.copyUserSpecifierDeclarations()] " +
					"user has to manually include necessary header file or declaration to " +
					"the generated openarc_kernel.cu/cl file.\n", 0);
			return;
		}
		if( lastUserDeclMap.containsKey(tUnit) ) {
			prevDecl = lastUserDeclMap.get(tUnit);
		}
		//Procedure tProc = AnalysisTools.findFirstProcedure(tUnit);
		PrintTools.println("[INFO from copyUserSpecifierDeclarations()] User symbols accessed in the kernel: " + accessedSymbols, 2);
		for( Symbol tSym : accessedSymbols ) {
			Stack<Declaration> declStack = new Stack<Declaration>();
			List typeList = tSym.getTypeSpecifiers();
			Stack<List> typeStack = new Stack<List>();
			typeStack.push(typeList);
			while( !typeStack.isEmpty() ) {
				typeList = typeStack.pop();
				for( Object tObj : typeList ) {
					if( tObj instanceof UserSpecifier ) {
						IDExpression tExp = ((UserSpecifier)tObj).getIDExpression();
						if( !tExp.toString().matches("type\\d+b") ) { //[DEBUG] this reg-exp should be changed if AnalysisTools.getBitVecType()
							//is changed.
							if( SymbolTools.findSymbol(tUnit, tExp) == null ) {
								Declaration tDecl = SymbolTools.findSymbol(symTable, tExp);
								if( tDecl != null ) {
									if( tDecl instanceof Enumeration ) {
										declStack.push(tDecl.clone());
									} else if( tDecl instanceof ClassDeclaration ) {
										Set<Declaration> memberDecls = ((ClassDeclaration)tDecl).getDeclarations();
										for( Declaration mDecl : memberDecls ) {
											if( mDecl instanceof VariableDeclaration ) {
												List ttypeList = ((VariableDeclaration)mDecl).getSpecifiers();
												if( ttypeList != null ) {
													typeStack.push(ttypeList);
												}
											}
										}
										declStack.push(tDecl.clone());
									} else if( tDecl instanceof VariableDeclaration ) {
										List ttypeList = ((VariableDeclaration)tDecl).getSpecifiers();
										if( ttypeList != null ) {
											typeStack.push(ttypeList);
										}
										declStack.push(tDecl.clone());
									}
								} else {
									//[DEBUG] While ClassDeclaration.getDeclaredIDs() returns class itself ("struct foo"), 
									//Enumeration.getDeclaredIDs() returns its enumerated types, but not the enumeration itself.
									//Therefore, the enumeration definition can not be found by SymbolTools.findSymbol().
									tDecl = tSym.getDeclaration();
									if( tDecl instanceof Enumeration ) {
										if( SymbolTools.findSymbol(tUnit, tSym.getSymbolName()) == null ) {
											declStack.push(tDecl.clone());
										}
									} else {
										PrintTools.println("[ERROR in ACC2GPUTranslator.copyUserSpecifierDeclarations()] " +
												"can not find definition of derived type(" + tExp + ") used in an OpenACC kernel; user has to manually " +
												"include necessary header file or declaration to the generated openarc_kernel.cu/cl file.\n", 0);
									}
								}
							}
						}
					}
				}
			}
			if( !declStack.isEmpty() ) {
				while( !declStack.isEmpty() ) {
					newDecl = declStack.pop();
					if( prevDecl == null ) {
						tUnit.addDeclarationAfter(refDecl, newDecl);
					} else {
						tUnit.addDeclarationAfter(prevDecl, newDecl);
					}
					prevDecl = newDecl;
/*                    if( tProc == null ) {
                        tUnit.addDeclaration(declStack.pop());
                    } else {
                        tUnit.addDeclarationBefore(tProc, declStack.pop());
                    }   */ 
				}
			}
		}
		
		if( newDecl != null ) {
			lastUserDeclMap.put(tUnit, newDecl);
		}
	}
	
	
	/**
	 * Convert a variable-length array (VLA) in a function parameter to a pointer type, and add typecasting statement at the 
	 * beginning of the function body.
	 * 
	 * For example:
	 *     //from
	 *     zero3(double (*z)[n2][n1], int n2, int n1) { ... }
	 *     //to 
	 *     zero3(void *oz) {
	 *         double (*z)[n2][n1] = (double (*)[n2][n1])oz;
	 *         ... 
	 *     }
	 *     
	 * [FIXME] VLA is a C99 feature but not allowed in C++. (C++14 may include this feature.)
	 * 
	 * @param tProc
	 * @param fCallList
	 */
	public void varLengthArrayTransforamtion(Procedure tProc, List<FunctionCall> fCallList) {
		List paramList = tProc.getParameters();
		int list_size = paramList.size();
		if( list_size == 1 ) {
			Object obj = paramList.get(0);
			String paramS = obj.toString();
			// Remove any leading or trailing whitespace.
			paramS = paramS.trim();
			if( paramS.equals(Specifier.VOID.toString()) ) {
				list_size = 0;
			}
		}
		if( list_size == 0 ) {return;}
		else {
			for( int i=0; i<list_size; i++ ) {}
		}
	}
}
