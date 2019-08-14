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
	protected TranslationUnit kernelsTranslationUnit = null;
	protected AnnotationDeclaration accHeaderDecl = null;
	protected String mainEntryFunc = null;
	protected boolean kernelContainsStdioCalls = false;
	protected boolean kernelContainsStdlibCalls = false;
	
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
	protected int defaultNumAsyncQueues = 4;
	///////////////////////////////////////////////////////////////
	// DEBUG: below two variables should have the same values as //
	// the ones in acc2gpu.java.                                 //
	///////////////////////////////////////////////////////////////
	protected int gpuMemTrOptLevel = 2;
	protected int gpuMallocOptLevel = 0;
	protected int localRedVarConf = 1;
	protected int SkipGPUTranslation = 0;
	protected String tuningParamFile = null;
	protected String kernelFileNameBase = "openarc_kernel";

    protected int targetArch = 0;
    protected String acc_device_type = "acc_device_default"; //contains ACC_DEVICE_TYPE environment value (default: acc_device_default);
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
    			return new NameID("acc_async_noval");
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

    protected List<Expression> getWaitList(ACCAnnotation tAnnot)
    {
    	List<Expression> waitslist = null;
    	if( tAnnot == null ) 
    	{
    		return null;
    	} else {
    		Object obj = tAnnot.get("wait");
    		if( obj instanceof String ) 
    		{
    			waitslist = new LinkedList<Expression>();
    			return waitslist;
    		} 
    		else if( obj instanceof List ) 
    		{
    			waitslist = new LinkedList<Expression>();
    			List tlist = (List)obj;
    			if( tlist.isEmpty() ) {
    				return waitslist;
    			} else if( tlist.get(0) instanceof Expression ) {
    				for( Object tobj : tlist ) {
    					waitslist.add((Expression)tobj);
    				}
    				return waitslist;

    			} else {
    				Tools.exit("[ERROR] Unsupported Wait Clause Argument Type 1: " 
    						+ obj + "\n" + AnalysisTools.getEnclosingAnnotationContext(tAnnot));
    				return null;
    			}
    		}
    		else
    		{
    			Tools.exit("[ERROR] Unsupported Wait Clause Argument Type 2: " 
    					+ obj + "\n" + AnalysisTools.getEnclosingAnnotationContext(tAnnot));
    			return null;
    		}
    	}
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
		searchKeys.add("set");
		List<ACCAnnotation> declareAnnots = new LinkedList<ACCAnnotation>();
		List<ACCAnnotation> updateAnnots = new LinkedList<ACCAnnotation>();
		List<ACCAnnotation> hostDataAnnots = new LinkedList<ACCAnnotation>();
		List<ACCAnnotation> waitAnnots = new LinkedList<ACCAnnotation>();
		List<ACCAnnotation> setAnnots = new LinkedList<ACCAnnotation>();
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
				} else if( mAnnot.containsKey("set") ) {
					setAnnots.add(mAnnot);
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
		if( targetArch != 4 ) {
			handleDeclareDirectives(declareAnnots);
		}

		if( opt_GenDistOpenACC ) {
			for(Traversable tChild : program.getChildren() ) {
				TranslationUnit trUnt = (TranslationUnit)tChild;
				Set<Symbol> accessedSymbols = AnalysisTools.getAccessedVariables(trUnt, true);
				Set<String> accessedSymStrings = new HashSet<String>();
				for( Symbol tSym : accessedSymbols ) {
					accessedSymStrings.add(tSym.getSymbolName());
				}
				List<Traversable> childList = trUnt.getChildren();
				Declaration firstDecl = trUnt.getFirstDeclaration();
				boolean isInHeader = true;
				//Find user-declared threadprivate symbols.
				Set<String> threadPrivateSet = new HashSet<String>();
				//Find ignoreglobal set 
				Set<String> ignoreGlobalSet = new HashSet<String>();
				for( Traversable child : childList ) {
					if( child instanceof AnnotationDeclaration ) {
						OmpAnnotation ompAnnot = ((AnnotationDeclaration)child).getAnnotation(OmpAnnotation.class, "threadprivate");
						if(ompAnnot != null) {
							threadPrivateSet.addAll((Set<String>)ompAnnot.get("threadprivate"));
						}
						ARCAnnotation arcAnnot = ((AnnotationDeclaration)child).getAnnotation(ARCAnnotation.class, "ignoreglobal");
						if( arcAnnot != null) {
							ignoreGlobalSet.addAll((Set<String>)arcAnnot.get("ignoreglobal"));
							//PrintTools.println("Found ignoreglobal clause: " + ignoreGlobalSet, 0);
						}
					}
				}
				Set<String> threadPrivateInHeaderSet = new HashSet<String>();
				Map<Declaration, Declaration> insertMap = new HashMap<Declaration, Declaration>();
				for( Traversable child : childList ) {
					if( isInHeader ) {
						if( child == firstDecl ) {
							isInHeader = false;
						}
					}
					if( child instanceof VariableDeclaration ) {
						VariableDeclaration vDecl = (VariableDeclaration)child;
						List vDeclSpecs = vDecl.getSpecifiers();
						if( (vDeclSpecs != null) && vDeclSpecs.contains(Specifier.TYPEDEF) ) {
							continue;
						}
						if( vDecl.getDeclarator(0) instanceof ProcedureDeclarator ) {
							continue;
						}
						List<IDExpression> declIDs = vDecl.getDeclaredIDs();
						if( (declIDs == null) || declIDs.isEmpty() ) {
							continue;
						}
						Set<String> tSet = new HashSet<String>();
						for( IDExpression dID : declIDs ) {
							String dIDStr = dID.toString();
							if( !threadPrivateSet.contains(dIDStr) && !ignoreGlobalSet.contains(dIDStr) ) {
								tSet.add(dIDStr);
							}
						}
						if( !tSet.isEmpty() ) {
							if( isInHeader ) {
								threadPrivateInHeaderSet.addAll(tSet);
							} else {
								threadPrivateInHeaderSet.retainAll(accessedSymStrings);
								if( !threadPrivateInHeaderSet.isEmpty() ) {
									tSet.addAll(threadPrivateInHeaderSet);
									threadPrivateInHeaderSet.clear();
								}
								OmpAnnotation ompAnnot = new OmpAnnotation();
								ompAnnot.put("threadprivate", tSet);
								threadPrivateSet.addAll(tSet);
								AnnotationDeclaration annotDecl = new AnnotationDeclaration(ompAnnot);
								insertMap.put((Declaration)child, annotDecl);
							}
						}
					}
				}
				if( !insertMap.isEmpty() ) {
					for( Declaration dKey : insertMap.keySet() ) {
						Declaration dValue = insertMap.get(dKey);
						trUnt.addDeclarationAfter(dKey, dValue);
					}
				} else {
					threadPrivateInHeaderSet.retainAll(accessedSymStrings);
					if( !threadPrivateInHeaderSet.isEmpty() ) {
						OmpAnnotation ompAnnot = new OmpAnnotation();
						ompAnnot.put("threadprivate", threadPrivateInHeaderSet);
						AnnotationDeclaration annotDecl = new AnnotationDeclaration(ompAnnot);
						if( firstDecl == null ) {
							trUnt.addDeclaration(annotDecl);
						} else {
							trUnt.addDeclarationBefore(firstDecl, annotDecl);
						}
					}

				}
			}
		}
		
		List<FunctionCall> allFuncCalls = IRTools.getFunctionCalls(program);
		List<Procedure> procedureList = IRTools.getProcedureList(program);
		for( Procedure cProc : procedureList ) {
			List<ACCAnnotation> dataAnnots = 
					AnalysisTools.collectPragmas(cProc, ACCAnnotation.class, ACCAnnotation.dataRegions, false);

			List<ACCAnnotation> atomicAnnots =
                                        AnalysisTools.collectPragmas(cProc, ACCAnnotation.class, new HashSet(Arrays.asList("atomic")), false);
			
			List<FunctionCall> fCallList = IRTools.getFunctionCalls(cProc);
			if( opt_GenDistOpenACC ) {
				ImpaccRuntimeTransformation(cProc, fCallList, allFuncCalls);
			}
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
			List<ACCAnnotation> enterExitDataAnnots = new LinkedList<ACCAnnotation>();

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
						if( annot.containsKey("enter") || annot.containsKey("exit") ) {
							enterExitDataAnnots.add(annot);
						} else {
							dataRegionAnnots.add(annot);
						}
					} else if( annot.containsKey("parallel") ) {
						parallelRegionAnnots.add(annot);
					} else if( annot.containsKey("kernels") ) {
						kernelsRegionAnnots.add(annot);
					}
				}
				if( targetArch != 4 ) {
					handleEnterExitData(cProc, enterExitDataAnnots, IRSymbolOnly);
					handleDataRegions(cProc, dataRegionAnnots);
				}
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
		addStandardHeadersToKernelsTranlationUnit();

		if( targetArch == 4 ) {
			for( int i=0; i<main_List.size(); i++ ) {
				Procedure tmain = main_List.get(i);
				Statement taccInitStmt = accInitStmt_List.get(i);
				Procedure pProc = taccInitStmt.getProcedure();
				TranslationUnit pTrUnit = null;
				IDExpression srcStringPtrID = null;
				if( pProc != null ) {
					pTrUnit = (TranslationUnit) pProc.getParent();
					if( pTrUnit != null ) {
						String iFileNameBase = "";
						String iFileName = pTrUnit.getOutputFilename();
						int dot = iFileName.lastIndexOf(".");
						if( dot >= 0 ) {
							iFileNameBase = iFileName.substring(0, dot);
						}
						Declaration srcPtrDecl = SymbolTools.findSymbol(pTrUnit, "src_code_"+iFileNameBase);
						if( (srcPtrDecl != null) && (srcPtrDecl instanceof VariableDeclaration) ) {
							Declarator vDeclr = ((VariableDeclaration)srcPtrDecl).getDeclarator(0);
							srcStringPtrID = vDeclr.getID();
						}
					}
				}
				if( srcStringPtrID != null ) {
					FunctionCall srcLoadCall = new FunctionCall(new NameID("mcl_load"));
					srcLoadCall.addArgument(new StringLiteral(kernelFileNameBase + ".cl"));
					srcLoadCall.addArgument(new UnaryExpression(UnaryOperator.ADDRESS_OF, srcStringPtrID.clone()));
					TransformTools.addStatementAfter(tmain.getBody(), taccInitStmt, new ExpressionStatement(srcLoadCall));
				} else {
					Tools.exit("[ERROR in ACC2GPUTranslator.start()] cannot find the symbol of the src_code variable; exit!\n" +
							"Enclosing file: " + pTrUnit.getInputFilename() + "\n");
				}
			}
		} else {
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
				initCall.addArgument(new StringLiteral(kernelFileNameBase));
			}
		}

		if( targetArch != 4 ) {
			handleUpdateDirectives(updateAnnots);
			handleHostDataDirectives(hostDataAnnots);
		}
		handleWaitDirectives(waitAnnots);
		if( targetArch == 4 ) {
			MCLRuntimeTransformation();
		}
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

		value = Driver.getOptionValue("defaultNumAsyncQueues");
		if( value != null ) {
			defaultNumAsyncQueues = Integer.valueOf(value).intValue();
		}
		FunctionCall AsyncQueuePrintCall = new FunctionCall(new NameID("printf"));
		AsyncQueuePrintCall.addArgument(new StringLiteral("====> Default Number of Async Queues per Device: "
				+defaultNumAsyncQueues+" \\n"));
		confPrintStmts.add( new ExpressionStatement(AsyncQueuePrintCall) );
		
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

		value = System.getenv("ACC_DEVICE_TYPE");
		if( value != null)
		{
			acc_device_type = value;
		}

		value = Driver.getOptionValue("assumeNoAliasingAmongKernelArgs");
		if( value != null ) {
			opt_AssumeNoAliasing = true;
		}

		value = Driver.getOptionValue("skipKernelLoopBoundChecking");
		if( value != null ) {
			opt_skipKernelLoopBoundChecking = true;
		}

		value = Driver.getOptionValue("SetOutputKernelFileNameBase");
		if( value != null ) {
			kernelFileNameBase = value;
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
				for( FunctionCall acc_shutdown : acc_shutdown_list ) {
					Procedure tProc = IRTools.getParentProcedure(acc_shutdown.getStatement());
					if( (tProc != null) && tProc.getName().equals(cProc.getName()) ) {
						lastStmts.add(acc_shutdown.getStatement());
					}
				}
				if( lastStmts.isEmpty() ) {
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

	protected void handleSetDirectives(List<ACCAnnotation> setAnnots) {
		for(ACCAnnotation sAnnot : setAnnots ) {
			Annotatable at = sAnnot.getAnnotatable();
			Expression default_async = sAnnot.get("default_async");
			Expression device_num = sAnnot.get("device_num");
			Expression device_type = sAnnot.get("device_type");
		}
	}

	protected void handleEnterExitData(Procedure cProc, List<ACCAnnotation> dataRegionAnnots, boolean IRSymOnly) {
		for(ACCAnnotation dAnnot : dataRegionAnnots ) {
			Annotatable at = dAnnot.getAnnotatable();
			Statement atStmt = null;
			CompoundStatement cStmt = null;
			Expression asyncID = new NameID("acc_async_sync");
			Expression waitID = new NameID("acc_async_sync");
			Expression ifCond = null;
			Set<SubArray> dataSet = null;
			if( at instanceof Statement ) {
				atStmt = (Statement)at;
			} else {
				Traversable  t = at.getParent();
				while( (t != null) && !(t instanceof Statement) ) {
					t = t.getParent();
				}
				if( t instanceof Statement ) {
					atStmt = (Statement)t;
				}
			}
			if( (atStmt != null) && (atStmt.getParent() !=null) && (atStmt.getParent() instanceof CompoundStatement) ) {
				cStmt = (CompoundStatement)atStmt.getParent();
			}
			if( dAnnot.containsKey("async") ) {
				Object obj = dAnnot.get("async");
				if( obj instanceof String ) {
					asyncID = new NameID("acc_async_noval");
				} else if( obj instanceof Expression ) {
					asyncID = (Expression)obj;
				}
			}
			if( dAnnot.containsKey("wait") ) {
				Object obj = dAnnot.get("wait");
				if( obj instanceof String ) {
					waitID = new NameID("acc_async_noval");
				} else if( obj instanceof Expression ) {
					waitID = (Expression)obj;
				}
			}
			if( dAnnot.containsKey("if") ) {
				ifCond = (Expression)dAnnot.get("if");
			}
			for( String dClause : ACCAnnotation.unstructuredDataClauses ) {
				if( dAnnot.containsKey(dClause) ) {
					dataSet = (Set<SubArray>)dAnnot.get(dClause);
					if( (dataSet != null) && (!dataSet.isEmpty()) ) {
						FunctionCall fCallOrg = null;
						switch(dClause) {
						case "copyin": fCallOrg = new FunctionCall(new NameID("acc_copyin_async_wait"));
						break;
						case "pcopyin": fCallOrg = new FunctionCall(new NameID("acc_pcopyin_async_wait"));
						break;
						case "create": fCallOrg = new FunctionCall(new NameID("acc_create_async_wait"));
						break;
						case "pcreate": fCallOrg = new FunctionCall(new NameID("acc_pcreate_async_wait"));
						break;
						case "copyout": fCallOrg = new FunctionCall(new NameID("acc_copyout_async_wait"));
						break;
						//case "pcopyout": fCallOrg = new FunctionCall(new NameID("acc_pcopyout_async_wait"));
						//break;
						case "delete": fCallOrg = new FunctionCall(new NameID("acc_delete_async_wait"));
						break;
						default: Tools.exit("[ERROR in ACC2GPUTranslator.handleEnterExitData()] unexpected data clause (" 
										+ dClause + ") is found for the following annotation; exit!\nACCAnnotation: " 
										+ dAnnot + "\n" + AnalysisTools.getEnclosingAnnotationContext(dAnnot));
						break;
						}
						for( SubArray sArray : dataSet ) {
							FunctionCall fCall = fCallOrg.clone();
							Expression varName = sArray.getArrayName();
							Symbol sym = SymbolTools.getSymbolOf(varName);
							List<Expression> startList = new LinkedList<Expression>();
							List<Expression> lengthList = new LinkedList<Expression>();
							boolean foundDimensions = AnalysisTools.extractDimensionInfo(sArray, startList, lengthList, IRSymbolOnly, at);
							if( !foundDimensions ) {
								Tools.exit("[ERROR in ACC2GPUTranslator.handleEnterExitData()] Dimension information " +
										"of the following variable is unknown; exit.\n" + 
										"Variable: " + sArray.getArrayName() + "\n" +
										"ACCAnnotation: " + dAnnot + "\n" +
										"Enclosing Procedure: " + cProc.getSymbolName() + "\n");
							}
							List<Specifier> typeSpecs = new ArrayList<Specifier>();
							Symbol IRSym = sym;
							if( sym instanceof PseudoSymbol ) {
								IRSym = ((PseudoSymbol)sym).getIRSymbol();
							}
							if( IRSymbolOnly ) {
								sym = IRSym;
								varName = new Identifier(sym);
								typeSpecs.addAll(((VariableDeclaration)sym.getDeclaration()).getSpecifiers());
							} else {
								Symbol tSym = sym;
								while( tSym instanceof AccessSymbol ) {
									tSym = ((AccessSymbol)tSym).getMemberSymbol();
								}
								typeSpecs.addAll(((VariableDeclaration)tSym.getDeclaration()).getSpecifiers());
							}
							typeSpecs.remove(Specifier.STATIC);
							typeSpecs.remove(Specifier.EXTERN);
							typeSpecs.remove(Specifier.CONST);
							SizeofExpression sizeof_expr = new SizeofExpression(typeSpecs);
							Expression sizeExp = sizeof_expr.clone();
							for( int i=0; i<lengthList.size(); i++ )
							{
								sizeExp = new BinaryExpression(sizeExp, BinaryOperator.MULTIPLY, lengthList.get(i).clone());
							}
							List<Expression> arg_list = new ArrayList<Expression>();
							if( lengthList.size() == 0 ) { //hostVar is scalar.
								arg_list.add( new UnaryExpression(UnaryOperator.ADDRESS_OF, 
										varName.clone()));
							} else {
								arg_list.add(varName.clone());
							}
							arg_list.add(sizeExp);
							arg_list.add(asyncID.clone());
							arg_list.add(waitID.clone());
							fCall.setArguments(arg_list);
							Statement fCallStmt = new ExpressionStatement(fCall);
							if(ifCond != null) {
								fCallStmt = new IfStatement(ifCond.clone(), fCallStmt);
							}
							if( (cStmt != null) && (atStmt != null) ) {
								cStmt.addStatementAfter(atStmt, fCallStmt);
							} else {
								Tools.exit("[ERROR in ACC2GPUTranslator.handleEnterExitData()] Cannot find " +
										"the parent compoundstatement of the following annotation; exit.\n" + 
										"ACCAnnotation: " + dAnnot + "\n" +
										"Enclosing Procedure: " + cProc.getSymbolName() + "\n");
							}
						}
					}
				}
			}
			cStmt.removeChild(atStmt);
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

	/**
	 * Modify OpenACC runtime API calls to work with MCL.
	 */
	protected void MCLRuntimeTransformation() {
		List<FunctionCall> fCallList = IRTools.getFunctionCalls(program);
		for( FunctionCall fCall : fCallList ) {
			if( OpenACCLibrary.contains(fCall) ) {
				String fCallName = fCall.getName().toString();
				if( fCallName.equals("acc_init") || fCallName.equals("acc_shutdown") ) {
					Tools.exit("[ERROR in ACC2GPUTranslator.MCLRuntimeTransformation()] found acc_init() or acc_shutdown() API calls, "
							+ "which should have been replaced by mcl_init() or mcl_finit() calls; exit!" 
							+ AnalysisTools.getEnclosingContext(fCall));
				} else if( fCallName.equals("acc_wait_all") || fCallName.equals("acc_wait_all_async") 
						|| fCallName.equals("acc_async_wait_all")) {
					FunctionCall mclCall = new FunctionCall(new NameID("mcl_wait_all"));
					mclCall.swapWith(fCall);
				} else if( fCallName.equals("acc_wait") || fCallName.equals("acc_wait_async") 
						|| fCallName.equals("acc_async_wait")) {
					FunctionCall mclCall = new FunctionCall(new NameID("mcl_acc_wait"));
					Expression arg = fCall.getArgument(0).clone();
					mclCall.addArgument(arg);
					mclCall.swapWith(fCall);
				} else if( fCallName.equals("acc_async_test") ) {
					FunctionCall mclCall = new FunctionCall(new NameID("mcl_acc_test"));
					Expression arg = fCall.getArgument(0).clone();
					mclCall.addArgument(arg);
					mclCall.swapWith(fCall);
				} else if( fCallName.equals("acc_async_test_all") ) {
					FunctionCall mclCall = new FunctionCall(new NameID("mcl_acc_test_all"));
					mclCall.swapWith(fCall);
				} else if( fCallName.equals("acc_on_device") ) {
					FunctionCall mclCall = new FunctionCall(new NameID("mcl_on_device"));
					Expression arg = fCall.getArgument(0).clone();
					mclCall.addArgument(arg);
					mclCall.swapWith(fCall);
				} else if( fCallName.equals("acc_malloc") ) {
					FunctionCall mclCall = new FunctionCall(new NameID("malloc"));
					Expression arg = fCall.getArgument(0).clone();
					mclCall.addArgument(arg);
					mclCall.swapWith(fCall);
				} else if( fCallName.equals("acc_free") ) {
					FunctionCall mclCall = new FunctionCall(new NameID("free"));
					mclCall.swapWith(fCall);
				} else if( fCallName.equals("acc_deviceptr") || fCallName.equals("acc_hostptr") ) {
					Expression arg = fCall.getArgument(0).clone();
					arg.swapWith(fCall);
/*				} else if( fCallName.equals("acc_get_num_devices") || fCallName.equals("acc_set_device_num")
						|| fCallName.equals("acc_get_device_type") || fCallName.equals("acc_set_device_type") 
						|| fCallName.equals("acc_copyin") || fCallName.equals("acc_copyin_async")
						|| fCallName.equals("acc_pcopyin") || fCallName.equals("acc_present_or_copyin")
						|| fCallName.equals("acc_create") || fCallName.equals("acc_create_async")
						|| fCallName.equals("acc_pcreate") || fCallName.equals("acc_present_or_create")
						|| fCallName.equals("acc_copyout") || fCallName.equals("acc_copyout_async")
						|| fCallName.equals("acc_delete") || fCallName.equals("acc_delete_async")
						|| fCallName.equals("acc_update_device") || fCallName.equals("acc_update_device_async")
						|| fCallName.equals("acc_update_self") || fCallName.equals("acc_update_self_async")
						|| fCallName.equals("acc_memcpy_to_device") || fCallName.equals("acc_memcpy_from_device")
						|| fCallName.equals("acc_memcpy_device") || fCallName.equals("acc_memcpy_device_async")
						|| fCallName.equals("acc_map_data") || fCallName.equals("acc_unmap_data")) {*/
				} else {
					Traversable child = fCall.getParent();
					Traversable parent = null;
					while( !(child instanceof Statement) && (child != null) ) {
						child = child.getParent();
					}
					if( child != null ) {
						parent = child.getParent();
					}
					if( (parent != null) && (child != null) ) {
						TransformTools.removeChild(parent, child);
					}
				}
			} else {
				String fCallName = fCall.getName().toString();
				if( fCallName.equals("acc_copyin_unified") || fCallName.equals("acc_pcopyin_unified") 
						|| fCallName.equals("acc_present_or_copyin_unified") || fCallName.equals("acc_create_unified")
						|| fCallName.equals("acc_pcreate_unified") || fCallName.equals("acc_present_or_create_unified")
						|| fCallName.equals("acc_copyout_unified")) {
					FunctionCall mclCall = new FunctionCall(new NameID("malloc"));
					Expression arg = fCall.getArgument(1).clone();
					mclCall.addArgument(arg);
					mclCall.swapWith(fCall);
				} else if( fCallName.equals("acc_delete_unified") ) {
					FunctionCall mclCall = new FunctionCall(new NameID("free"));
					Expression arg = fCall.getArgument(0).clone();
					mclCall.addArgument(arg);
					mclCall.swapWith(fCall);
				}
				
			}
		}
	}
	
	protected void ImpaccRuntimeTransformation(Procedure cProc, List<FunctionCall> fCallList, List<FunctionCall> allFuncCalls) {
		for(FunctionCall fCall : fCallList) {
			Statement fStmt = fCall.getStatement();
			if( fStmt != null ) {
				Expression optionFlag = null;
				Expression asyncID = null;
				Expression tPointer = null;
				ACCAnnotation mAnnot = fStmt.getAnnotation(ACCAnnotation.class, "mpi");
				if( mAnnot != null ) {
					Expression SROFlag = null;
					Expression SDFlag = null;
					Expression RROFlag = null;
					Expression RDFlag = null;
					Set<Expression> optionSet = mAnnot.get("sendbuf");
					if( optionSet != null ) {
						for( Expression tExp : optionSet ) {
							if( tExp.toString().equals("readonly") ) {
								SROFlag = new NameID("IMPACC_MEM_S_RO");
							} else if( tExp.toString().equals("device") ) {
								SDFlag = new NameID("IMPACC_MEM_S_DEV");
							} else {
								Tools.exit("[ERROR in ACC2GPUTranslator.ImpactRuntimeTransformation()] unexpected argument (" 
										+ tExp.toString() + ") is found for the sendbuf clause in the following annotation; exit!\nACCAnnotation: " 
										+ mAnnot + "\n" + AnalysisTools.getEnclosingAnnotationContext(mAnnot));
							}
						}
					}
					optionSet = mAnnot.get("recvbuf");
					if( optionSet != null ) {
						for( Expression tExp : optionSet ) {
							if( tExp.toString().equals("readonly") ) {
								RROFlag = new NameID("IMPACC_MEM_R_RO");
							} else if( tExp.toString().equals("device") ) {
								RDFlag = new NameID("IMPACC_MEM_R_DEV");
							} else {
								Tools.exit("[ERROR in ACC2GPUTranslator.ImpactRuntimeTransformation()] unexpected argument (" 
										+ tExp.toString() + ") is found for the sendbuf clause in the following annotation; exit!\nACCAnnotation: " 
										+ mAnnot + "\n" + AnalysisTools.getEnclosingAnnotationContext(mAnnot));
							}
						}
					}
					if( SROFlag != null ) {
						if( optionFlag == null ) {
							optionFlag = SROFlag;
						} else {
							optionFlag = new BinaryExpression(optionFlag, BinaryOperator.LOGICAL_OR, SROFlag);
						}
					}
					if( SDFlag != null ) {
						if( optionFlag == null ) {
							optionFlag = SDFlag;
						} else {
							optionFlag = new BinaryExpression(optionFlag, BinaryOperator.LOGICAL_OR, SDFlag);
						}
					}
					if( RROFlag != null ) {
						if( optionFlag == null ) {
							optionFlag = RROFlag;
						} else {
							optionFlag = new BinaryExpression(optionFlag, BinaryOperator.LOGICAL_OR, RROFlag);
						}
					}
					if( RDFlag != null ) {
						if( optionFlag == null ) {
							optionFlag = RDFlag;
						} else {
							optionFlag = new BinaryExpression(optionFlag, BinaryOperator.LOGICAL_OR, RDFlag);
						}
					}
					Object obj = mAnnot.get("async");
					if( obj instanceof String ) { //async ID is not specified by a user; use default async queue.
						asyncID = new NameID("acc_async_noval");
					} else if( obj instanceof Expression ) {
						asyncID = (Expression)obj;
					} else { //async clause is not specified; use default sync queue.
						asyncID = new NameID("DEFAULT_QUEUE");
					}
				}
				if( optionFlag == null ) {
					optionFlag = new IntegerLiteral(0);
				}
				if( asyncID == null ) {
					asyncID = new NameID("DEFAULT_QUEUE");
				}
				String fName = fCall.getName().toString();
				switch(fName) {
				case "MPI_Init": fCall.setFunction(new NameID("IMPACC_MPI_Init"));
				break;
				case "MPI_Finalize": fCall.setFunction(new NameID("IMPACC_MPI_Finalize"));
				break;
				case "MPI_Comm_size": fCall.setFunction(new NameID("IMPACC_MPI_Comm_size"));
				break;
				case "MPI_Comm_rank": fCall.setFunction(new NameID("IMPACC_MPI_Comm_rank"));
				break;
				case "MPI_Send": fCall.setFunction(new NameID("IMPACC_MPI_Send"));
				fCall.addArgument(optionFlag);
				break;
				case "MPI_Recv": fCall.setFunction(new NameID("IMPACC_MPI_Recv"));
				fCall.addArgument(optionFlag);
				break;
				case "MPI_Isend": fCall.setFunction(new NameID("IMPACC_MPI_Isend"));
				fCall.addArgument(optionFlag);
				fCall.addArgument(asyncID);
				break;
				case "MPI_Irecv": fCall.setFunction(new NameID("IMPACC_MPI_Irecv"));
				fCall.addArgument(optionFlag);
				fCall.addArgument(asyncID);
				break;
				case "MPI_Wait": fCall.setFunction(new NameID("IMPACC_MPI_Wait"));
				break;
				case "MPI_Waitall": fCall.setFunction(new NameID("IMPACC_MPI_Waitall"));
				break;
				case "MPI_Barrier": fCall.setFunction(new NameID("IMPACC_MPI_Barrier"));
				break;
				case "MPI_Bcast": fCall.setFunction(new NameID("IMPACC_MPI_Bcast"));
				fCall.addArgument(optionFlag);
				break;
				case "MPI_Reduce": fCall.setFunction(new NameID("IMPACC_MPI_Reduce"));
				fCall.addArgument(optionFlag);
				break;
				case "MPI_Allreduce": fCall.setFunction(new NameID("IMPACC_MPI_Allreduce"));
				fCall.addArgument(optionFlag);
				break;
				case "free": fCall.setFunction(new NameID("IMPACC_free"));
				break;
				case "malloc": fCall.setFunction(new NameID("IMPACC_malloc"));
				tPointer = AnalysisTools.findAllocatedPointer(fCall, allFuncCalls, true);
				if( tPointer != null ) {
					Expression addExp = new UnaryExpression(UnaryOperator.ADDRESS_OF, tPointer.clone());
					List<Specifier> specs = new ArrayList<Specifier>(4);
					specs.add(Specifier.VOID);
					specs.add(PointerSpecifier.UNQUALIFIED);
					specs.add(PointerSpecifier.UNQUALIFIED);
					addExp = new Typecast(specs, addExp);
					fCall.addArgument(addExp);
				}
				break;
				case "calloc": fCall.setFunction(new NameID("IMPACC_calloc"));
				tPointer = AnalysisTools.findAllocatedPointer(fCall, allFuncCalls, true);
				if( tPointer != null ) {
					Expression addExp = new UnaryExpression(UnaryOperator.ADDRESS_OF, tPointer.clone());
					List<Specifier> specs = new ArrayList<Specifier>(4);
					specs.add(Specifier.VOID);
					specs.add(PointerSpecifier.UNQUALIFIED);
					specs.add(PointerSpecifier.UNQUALIFIED);
					addExp = new Typecast(specs, addExp);
					fCall.addArgument(addExp);
				}
				break;
				}
			}
		}
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
						//ForLoop tLoop = (ForLoop)tAnnot.getAnnotatable();
						Annotatable tAnnotObj = tAnnot.getAnnotatable();
						if( !(tAnnotObj instanceof ForLoop) ) {
							continue;
						}
						ForLoop tLoop = (ForLoop)tAnnotObj;
						ForLoop oLoop = null;
						Traversable tt = tLoop.getParent();
						boolean outermostloop = true;
						while( tt != null ) {
							if( (tt instanceof ForLoop) && ((Annotatable)tt).containsAnnotation(ACCAnnotation.class, tClause) ) {
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
			//[DEBUG] If the compute region does not have any data clause, HI_set_async should
			//be inserted before the kernel call; re-enable below as an easy but inefficient fix
			if(pAnnot.containsKey("async"))
            {
				if( targetArch != 4 ) {
					FunctionCall setAsyncCall = new FunctionCall(new NameID("HI_set_async"));
					if(pAnnot.get("async") != null)
					{
						setAsyncCall.addArgument(new NameID(pAnnot.get("async").toString()));
					}
					else
					{
						setAsyncCall.addArgument(new NameID("acc_async_noval"));
					}
					TransformTools.addStatementBefore((CompoundStatement)((Statement)at).getParent(), (Statement)at, new ExpressionStatement(setAsyncCall));
				}
            }
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

			//HI_set_async will be generated before the data clause instead
			//[DEBUG] If the compute region does not have any data clause, HI_set_async should
			//be inserted before the kernel call; re-enable below as an easy but inefficient fix
			if(kAnnot.containsKey("async"))
            {
				if( targetArch != 4 ) {
					FunctionCall setAsyncCall = new FunctionCall(new NameID("HI_set_async"));
					if(kAnnot.get("async") != null)
					{
						setAsyncCall.addArgument(new NameID(kAnnot.get("async").toString()));
					}
					else
					{
						setAsyncCall.addArgument(new NameID("acc_async_noval"));
					}
					TransformTools.addStatementBefore((CompoundStatement)((Statement)at).getParent(), (Statement)at, new ExpressionStatement(setAsyncCall));
				}
            }
			List<Statement> inStmts = new LinkedList<Statement>();
			List<Statement> outStmts = new LinkedList<Statement>();
			inStmts.add((Statement)at);
			outStmts.add((Statement)at);
			if( targetArch != 4 ) {
				handleDataClauses(kAnnot, inStmts, outStmts, regionT, IRSymbolOnly);
			}
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
		String newDeclStr = null;
		SymbolTable symTable = null;
		Set<String> addedDeclSet = new HashSet<String>();
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
					"the generated" + kernelFileNameBase + ".cu/cl file.\n", 0);
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
												"include necessary header file or declaration to the generated " + kernelFileNameBase + ".cu/cl file.\n", 0);
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
					newDeclStr = newDecl.toString();
					if( !addedDeclSet.contains(newDeclStr) ) {
						addedDeclSet.add(newDeclStr);
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
					} else {
						newDecl = prevDecl;
					}
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
	
	protected void addStandardHeadersToKernelsTranlationUnit () {
		StringBuilder kernelStr = null;
		if( targetArch == 0 ) {
			if( kernelContainsStdioCalls ) {
				kernelStr = new StringBuilder(64);
				kernelStr.append("#include <stdio.h>\n");
			} 
			if( kernelContainsStdlibCalls ) {
				if( kernelStr == null ) {
					kernelStr = new StringBuilder(64);
				}
				kernelStr.append("#include <stdlib.h>\n");
			} 
		}
		if( kernelStr != null ) {
			CodeAnnotation accHeaderAnnot2 = new CodeAnnotation(kernelStr.toString());
	        AnnotationDeclaration accHeaderDecl2 = new AnnotationDeclaration(accHeaderAnnot2);
			if( accHeaderDecl != null ) {
				kernelsTranslationUnit.addDeclarationAfter(accHeaderDecl, accHeaderDecl2);	
			}
		}
	}

	protected void removeBackendSpecificSpecifiers(Traversable region, Set<Specifier> removeSpecs) {
		Set<Symbol> localSymbols = SymbolTools.getLocalSymbols(region);
		if( localSymbols != null ) {
			for(Symbol lSym : localSymbols ) {
				if( lSym instanceof Declarator ) {
					List<Specifier> declspecs = null;
					List<Specifier> declrspecs = ((Declarator)lSym).getSpecifiers();
					Declaration decl = lSym.getDeclaration();
					if( decl instanceof VariableDeclaration ) {
						declspecs = ((VariableDeclaration)decl).getSpecifiers();
					}
					if( declrspecs != null ) {
						if( removeSpecs != null ) {
							declrspecs.removeAll(removeSpecs);
						} else {
							removeSpecs = new HashSet<Specifier>();
							for(Specifier tspec : declrspecs) {
								if( (tspec instanceof OpenCLSpecifier) || (tspec instanceof CUDASpecifier) ) {
									removeSpecs.add(tspec);
								}
							}
							declrspecs.removeAll(removeSpecs);
							removeSpecs = null;
						}
					}
					if( declspecs != null ) {
						if( removeSpecs != null ) {
							declspecs.removeAll(removeSpecs);
						} else {
							removeSpecs = new HashSet<Specifier>();
							for(Specifier tspec : declspecs) {
								if( (tspec instanceof OpenCLSpecifier) || (tspec instanceof CUDASpecifier) ) {
									removeSpecs.add(tspec);
								}
							}
							declspecs.removeAll(removeSpecs);
							removeSpecs = null;
						}
					}
				}
			}
		}
	}
}
