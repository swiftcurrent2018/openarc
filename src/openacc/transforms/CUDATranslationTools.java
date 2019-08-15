package openacc.transforms;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.TreeMap;
import java.util.Map;
import java.io.*;

import cetus.hir.*;
import cetus.analysis.LoopTools;
import openacc.hir.*;
import openacc.analysis.*;

/**
 * <b>CUDATranslationTools</b> provides tools for various transformation tools for CUDA-specific translation.
 * 
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group, Computer Science and Mathematics Division,
 *         Oak Ridge National Laboratory
 */
public abstract class CUDATranslationTools {
	private static int tempIndexBase = 1000;

	/**
	 * Java doesn't allow a class to be both abstract and final,
	 * so this private constructor prevents any derivations.
	 */
	private CUDATranslationTools()
	{
	}
	
	/**
	 * This method performs reduction transformation.
	 * This transformation is intraprocedural; functions called in a compute region should be handled
	 * separately.
	 * 
	 * CAUTION: 1) This translation assumes that there is only one reduction clause per nested gang loops or 
	 * nested worker loops. 2) This pass does not transform seq kernels/parallel loops, which should be handled 
	 * separately.
	 * 
	 * @param region
	 */
	protected static void reductionTransformation(Procedure cProc, Statement region, String cRegionKind, Expression ifCond,
			Expression asyncID, Statement confRefStmt,
			CompoundStatement prefixStmts, CompoundStatement postscriptStmts,
			List<Statement> preList, List<Statement> postList,
			FunctionCall call_to_new_proc, Procedure new_proc, TranslationUnit main_TrUnt, 
			Map<TranslationUnit, Declaration> OpenACCHeaderEndMap, boolean IRSymbolOnly,
			boolean opt_addSafetyCheckingCode, boolean opt_UnrollOnReduction, int maxBlockSize,
			Expression totalnumgangs, boolean kernelVerification, boolean memtrVerification, FloatLiteral EPSILON,
			int warpSize, FloatLiteral minCheckValue, int localRedVarConf, boolean isSingleTask) {
		PrintTools.println("[reductionTransformation() begins] current procedure: " + cProc.getSymbolName() +
				"\ncompute region type: " + cRegionKind + "\n", 2);
		
		List<ACCAnnotation> reduction_annots = IRTools.collectPragmas(region, ACCAnnotation.class, "reduction");
		if( (reduction_annots == null) || reduction_annots.isEmpty() ) {
			PrintTools.println("[reductionTransformation() ends] current procedure: " + cProc.getSymbolName() +
					"\ncompute region type: " + cRegionKind + "\n", 2);
			return;
		}
		
		CompoundStatement scope = null;
		CompoundStatement regionParent = (CompoundStatement)region.getParent();
		SymbolTable global_table = (SymbolTable) cProc.getParent();
		//CompoundStatement mallocScope = (CompoundStatement)confRefStmt.getParent();
		
		
		// Auxiliary variables used for GPU kernel conversion 
		VariableDeclaration bytes_decl = (VariableDeclaration)SymbolTools.findSymbol(global_table, "gpuBytes");
		Identifier cloned_bytes = new Identifier((VariableDeclarator)bytes_decl.getDeclarator(0));					
		VariableDeclaration gmem_decl = null;
		Identifier gmemsize = null;
		VariableDeclaration smem_decl = null;
		Identifier smemsize = null;
		ExpressionStatement gMemAdd_stmt = null;
		ExpressionStatement gMemSub_stmt = null;
		if( opt_addSafetyCheckingCode ) {
			gmem_decl = (VariableDeclaration)SymbolTools.findSymbol(global_table, "gpuGmemSize");
			gmemsize = new Identifier((VariableDeclarator)gmem_decl.getDeclarator(0));					
			smem_decl = (VariableDeclaration)SymbolTools.findSymbol(global_table, "gpuSmemSize");
			smemsize = new Identifier((VariableDeclarator)smem_decl.getDeclarator(0));					
			gMemAdd_stmt = new ExpressionStatement( new AssignmentExpression((Identifier)gmemsize.clone(),
					AssignmentOperator.ADD, (Identifier)cloned_bytes.clone()) );
			gMemSub_stmt = new ExpressionStatement( new AssignmentExpression((Identifier)gmemsize.clone(),
					AssignmentOperator.SUBTRACT, (Identifier)cloned_bytes.clone()) );
		}
		VariableDeclaration numBlocks_decl = (VariableDeclaration)SymbolTools.findSymbol(global_table, "gpuNumBlocks");
		Identifier numBlocks = new Identifier((VariableDeclarator)numBlocks_decl.getDeclarator(0));					
		VariableDeclaration numThreads_decl = (VariableDeclaration)SymbolTools.findSymbol(global_table, "gpuNumThreads");
		Identifier numThreads = new Identifier((VariableDeclarator)numThreads_decl.getDeclarator(0));					
		VariableDeclaration totalNumThreads_decl = (VariableDeclaration)SymbolTools.findSymbol(global_table, "totalGpuNumThreads");
		Identifier totalNumThreads = new Identifier((VariableDeclarator)totalNumThreads_decl.getDeclarator(0));					
		ExpressionStatement gpuBytes_stmt = null;
		ExpressionStatement orgGpuBytes_stmt = null;
		
		//Auxiliary structures to assist translation
		//Host reduction symbol to subarray mapping
		HashMap<Symbol, SubArray> redSubMap = new HashMap<Symbol, SubArray>();
		//Host reduction symbol to reduction operator mapping
		HashMap<Symbol, ReductionOperator> redOpMap = new HashMap<Symbol, ReductionOperator>();
		//Set of allocated gang-reduction symbols
		Set<Symbol> allocatedGangReductionSet = new HashSet<Symbol>();
		//Host gang-reduction symbol to GPU reduction parameter variable mapping
		Map<Symbol, Identifier> tGangParamRedVarMap = new HashMap<Symbol, Identifier>();
		
		//Set of allocated worker-reduction symbols
		Set<Symbol> allocatedWorkerReductionSet = new HashSet<Symbol>();
		Map<CompoundStatement, Map<Symbol, Identifier>> RegionToWorkerRedParamMap = 
				new HashMap<CompoundStatement, Map<Symbol, Identifier>>();
		Map<CompoundStatement, Statement> RegionToLastInBlockWorkerRedStmtMap = 
				new HashMap<CompoundStatement, Statement>();
		Map<CompoundStatement, Statement> RegionToEnclosingWorkerLoopMap = 
				new HashMap<CompoundStatement, Statement>();
		
		//If this kernel is asynchronous, the final gang-level reduction should be performed
		//after matching synchronization statement.
		//FIXME: below code can not check whether "acc_async_test" or "acc_async_test_all" is used 
		//for synchronization.
		Statement waitStmt = null;
		Statement waitStmtCandidate = null;
		Statement asyncConfRefStmt = null;
		CompoundStatement asyncConfRefPStmt = null;
		boolean asyncConfRefChanged = false;
		if( asyncID != null ) {
			CompoundStatement tCStmt = regionParent;
			Traversable tKernel = region;
			while ( (waitStmt == null ) && (tCStmt != null) ) {
				boolean foundKernel = false;
				for( Traversable t : tCStmt.getChildren() ) {
					if( !foundKernel ) {
						if( tKernel == t ) {
							foundKernel = true;
						}
						continue;
					} else {
						Annotatable tAt = (Annotatable)t;
						ACCAnnotation wAnnot = tAt.getAnnotation(ACCAnnotation.class, "wait");
						if( wAnnot != null ) {
							Object arg = wAnnot.get("wait");
							if( arg instanceof Expression ) {
								if( asyncID.equals(arg) ) {
									waitStmt = (Statement)t;
									break;
								} else if( waitStmtCandidate == null ) {
									waitStmtCandidate = (Statement)t;
								}
							} else {
								waitStmt = (Statement)t;
								break;
							}
						} else if( (tAt instanceof ExpressionStatement) ) {
							Expression exp = ((ExpressionStatement)tAt).getExpression();
							if( exp instanceof FunctionCall ) {
								String fName = ((FunctionCall)exp).getName().toString();
								if( fName.equals("acc_wait_all") ) {
									waitStmt = (Statement)tAt;
									break;
								} else if( fName.equals("acc_wait") ) {
									Expression arg = ((FunctionCall)exp).getArgument(0);
									if( arg.equals(asyncID) ) {
										waitStmt = (Statement)tAt;
										break;
									} else if( waitStmtCandidate == null ) {
										waitStmtCandidate = (Statement)tAt;
									}
								}
							}
						}
					}
				}
				if( waitStmt == null ) {
					tKernel = tCStmt;
					Traversable tParent = tKernel.getParent();
					while( (tParent != null) && !(tParent instanceof CompoundStatement) ) {
						tKernel = tParent;
						tParent = tKernel.getParent();
					}
					if( tParent instanceof CompoundStatement ) {
						tCStmt = (CompoundStatement)tParent;
					} else {
						tCStmt = null;
					}
				}
			}
			if( (waitStmt == null) && (waitStmtCandidate != null) ) {
				//If we cannot find wait statement with matching asyncID, but if there is a wait statement
				//with some asyncID expression, that statement can be the matching wait statement.
				waitStmt = waitStmtCandidate;
			}
			if( waitStmt == null ) {
				ACCAnnotation cAnnot = region.getAnnotation(ACCAnnotation.class, cRegionKind);
				Tools.exit("[ERROR in CUDATranslationTools.reductionTransformation()] the final " +
						"reduction codes for the following asynchronous " +
						"kernel should be inserted after the matching synchronization statement, " +
						"but the compiler can not find the statement; exit.\n" +
						"Current implementation can not handle asynchronous reduction if acc_async_test() " +
						"or acc_async_test_all() function is used for synchronization; please change these " +
						"to acc_wait() or acc_wait_all() function.\n" +
						"OpenACC Annotation: " + cAnnot + "\nEnclosing Procedure: " + 
						cProc.getSymbolName() + "\n" );
			} else {
				if( confRefStmt == region ) {
					//asyncConfRefStmt = waitStmt;
					asyncConfRefChanged =  true;
				} else { //If confRefStmt does not include waitStmt, asynchronous final reduction should be inserted 
					     //after waitStmt.
					Traversable tt = waitStmt;
					while ( (tt != null) && (tt != confRefStmt ) ) {
						tt = tt.getParent();
					}
					if( tt == null ) { //confRefStmt does not include waitStmt.
						//FIXME: if confRefStmt does not include waitStmt, reduction-related CPU/GPU memory
						//will not be freely in timely manner, incurring memory leakage problems.
						//==> Fixed at runtime using HI_tempMalloc().
						//asyncConfRefStmt = waitStmt;
						asyncConfRefChanged =  true;
					} else { //confRefStmt contains waitStmt.
						//asyncConfRefStmt = confRefStmt;
						asyncConfRefChanged =  false;
					}
				}
				asyncConfRefStmt = waitStmt;
				asyncConfRefPStmt = (CompoundStatement)asyncConfRefStmt.getParent();
			}
		}
		
		
		
		//Check the number of workers for this compute region, which may be needed for worker-private-caching on shared memory.
		long num_workers = 0;
		Expression num_workersExp = null;
		if( cRegionKind.equals("parallel") ) {
			ACCAnnotation tAnnot = region.getAnnotation(ACCAnnotation.class, "num_workers");
			if( tAnnot != null ) {
				num_workersExp = tAnnot.get("num_workers");
				if( num_workersExp instanceof IntegerLiteral ) {
					num_workers = ((IntegerLiteral)num_workersExp).getValue();
				}
			}
		} else {
			ACCAnnotation tAnnot = region.getAnnotation(ACCAnnotation.class, "totalnumworkers");
			if( tAnnot != null ) {
				num_workersExp = tAnnot.get("totalnumworkers");
				if( num_workersExp instanceof IntegerLiteral ) {
					num_workers = ((IntegerLiteral)num_workersExp).getValue();
				}
			}
		}
		
		//For correct translation, worker-reduction loops should be handled before gang-reduction regions.
		//CAUTION: This translation assumes that there is only one reduction clause per nested gang loops or 
		//nested worker loops.
		List<ACCAnnotation> workerred_regions = new LinkedList<ACCAnnotation>();
		List<ACCAnnotation> gangred_regions = new LinkedList<ACCAnnotation>();
		if( reduction_annots != null ) {
			for ( ACCAnnotation pannot : reduction_annots ) {
				if( pannot.containsKey("worker") ) {
					workerred_regions.add(pannot);
				} else {
					gangred_regions.add(pannot);
				}
			}
		}
		List<ACCAnnotation> reduction_regions = new LinkedList<ACCAnnotation>();
		reduction_regions.addAll(workerred_regions);
		reduction_regions.addAll(gangred_regions);
		
		for ( ACCAnnotation pannot : reduction_regions ) {
			Annotatable at = pannot.getAnnotatable();
			//Host reduction symbol to subarray mapping
			HashMap<Symbol, SubArray> redSubMapLocal = new HashMap<Symbol, SubArray>();
			//Host reduction symbol to reduction operator mapping
			HashMap<Symbol, ReductionOperator> redOpMapLocal = new HashMap<Symbol, ReductionOperator>();
			Map<Symbol, SubArray> sharedCachingMap = new HashMap<Symbol, SubArray>();
			Map<Symbol, SubArray> regROCachingMap = new HashMap<Symbol, SubArray>();
			Set<Symbol> transNoRedUnrollSet = new HashSet<Symbol>();
			Set<String> searchKeys = new HashSet<String>();
			searchKeys.add("sharedRO");
			searchKeys.add("sharedRW");
			for( String key : searchKeys ) {
				ARCAnnotation ttAnt = at.getAnnotation(ARCAnnotation.class, key);
				if( ttAnt != null ) {
					Set<SubArray> DataSet = (Set<SubArray>)ttAnt.get(key);
					for( SubArray sAr : DataSet ) {
						Symbol tSym = AnalysisTools.subarrayToSymbol(sAr, IRSymbolOnly);
						sharedCachingMap.put(tSym, sAr);
					}
				}
			}
			ARCAnnotation ttAnt = at.getAnnotation(ARCAnnotation.class, "registerRO");
			if( ttAnt != null ) {
				Set<SubArray> DataSet = (Set<SubArray>)ttAnt.get("registerRO");
				for( SubArray sAr : DataSet ) {
					Symbol tSym = AnalysisTools.subarrayToSymbol(sAr, IRSymbolOnly);
					regROCachingMap.put(tSym, sAr);
				}
			}
			//CAUTION: "noreductionunroll" clause should be attached to a worker loop to be applicable.
			ttAnt = at.getAnnotation(ARCAnnotation.class, "noreductionunroll");
			if( ttAnt != null ) {
				Set<SubArray> DataSet = (Set<SubArray>)ttAnt.get("noreductionunroll");
				for( SubArray sAr : DataSet ) {
					Symbol tSym = AnalysisTools.subarrayToSymbol(sAr, IRSymbolOnly);
					transNoRedUnrollSet.add(tSym);
				}
			}
			boolean isWorkerReduction = false;
			if( at.containsAnnotation(ACCAnnotation.class, "worker") ) {
				isWorkerReduction = true;
			}
			boolean isGangReduction = false;
			if( at.containsAnnotation(ACCAnnotation.class, "gang") ) {
				isGangReduction = true;
			}
			if( at.containsAnnotation(ACCAnnotation.class, "parallel") ) {
				isGangReduction = true;
			}
            {
            	ACCAnnotation tAnnot = at.getAnnotation(ACCAnnotation.class, "totalnumworkers");
            	if( tAnnot != null ) {
            		num_workersExp = tAnnot.get("totalnumworkers");
            		if( num_workersExp instanceof IntegerLiteral ) {
            			num_workers = ((IntegerLiteral)num_workersExp).getValue();
            		}
            	} else {
            		tAnnot = at.getAnnotation(ACCAnnotation.class, "num_workers");
            		if( tAnnot != null ) {
            			num_workersExp = tAnnot.get("num_workers");
            			if( num_workersExp instanceof IntegerLiteral ) {
            				num_workers = ((IntegerLiteral)num_workersExp).getValue();
            			}
            		}
            	}
            }
			scope = null;
			if( isWorkerReduction ) {//A worker reduction variable is declared in the innermost worker loop body.
				ACCAnnotation tAnnot = AnalysisTools.findInnermostPragma(at, ACCAnnotation.class, "worker");
				ForLoop wLoop = (ForLoop)tAnnot.getAnnotatable();
				scope = (CompoundStatement)(wLoop).getBody();
			} else if( isGangReduction ) {
				/////////////////////////////////////////////////////////////////////////////////////
				//A gang reduction variable is declared either in the enclosing compound statement //
				//or in the innermost gang loop body if region is a loop.                          //
				/////////////////////////////////////////////////////////////////////////////////////
				if( region instanceof ForLoop ) {
					ACCAnnotation tAnnot = AnalysisTools.findInnermostPragma(region, ACCAnnotation.class, "gang");
					ForLoop gLoop = (ForLoop)tAnnot.getAnnotatable();
					scope = (CompoundStatement)(gLoop).getBody();
				} else if( region instanceof CompoundStatement ) {
					scope = (CompoundStatement)region;
				}
			} else {
				continue; //vector reduction is ignored.
			}
			if( scope == null ) {
				Tools.exit("[ERROR in CUDATranslationTools.reductionTransformation() cannot find the scope where reduction" +
						"symbols in the following compute region are declared: \n" + region + "\n");
			}

			Map<ReductionOperator, Set<SubArray>> redMap = pannot.get("reduction");
			for(ReductionOperator op : redMap.keySet() ) {
				Set<SubArray> redSet = redMap.get(op);
				for( SubArray sArray : redSet ) {
					Symbol rSym = AnalysisTools.subarrayToSymbol(sArray, IRSymbolOnly); 
					redSubMapLocal.put(rSym, sArray);
					redOpMapLocal.put(rSym, op);
					redSubMap.put(rSym, sArray);
					redOpMap.put(rSym, op);
				}
			}
			
			Map<Symbol, Identifier> tWorkerParamRedVarMap = new HashMap<Symbol, Identifier>();

			//List to keep post-reduction statements if asyncID != null.
			List<Statement> gPostRedStmts = new LinkedList<Statement>();
			Collection<Symbol> sortedSet = AnalysisTools.getSortedCollection(redSubMapLocal.keySet());
			for( Symbol redSym : sortedSet ) {
				//PrintTools.println("[reductionTransformation] Transform reduction variable " + redSym, 2);
				
				List<Statement> postRedStmts = new LinkedList<Statement>();
				SubArray sArray = redSubMapLocal.get(redSym);
				ReductionOperator redOp = redOpMapLocal.get(redSym);
				Boolean isArray = SymbolTools.isArray(redSym);
				Boolean isPointer = SymbolTools.isPointer(redSym);
				if( redSym instanceof NestedDeclarator ) {
					isPointer = true;
				}
				//////////////////////////////////////////////////////////////////////////////////
				//FIXME: if redSym is a parameter of a function called in the parallel region, //
				//below checking may be incorrect.                                              //
				//////////////////////////////////////////////////////////////////////////////////
				SymbolTable targetSymbolTable = AnalysisTools.getIRSymbolScope(redSym, region);
				if( isGangReduction && ((targetSymbolTable == null) || (targetSymbolTable == region)) ) {
					ACCAnnotation cAnnot = region.getAnnotation(ACCAnnotation.class, cRegionKind);
					Tools.exit("[ERROR in CUDATTranslationTools.reductionTransformation()] gang-reduction variable should" +
							" be visible outside of the enclosing compute region, but the following reduction variable is not; exit!\n" +
							"Gang-reduction variable: " + redSym.getSymbolName() + "\n" +
							"Current reduction loop: " + pannot + "\n" +
							"Enclosing compute region: " + cAnnot + "\n" +
							"Enclosing procedure: " + cProc.getSymbolName() );
				}
				//DEBUG: extractComputeRegion() in ACC2CUDATranslator/ACC2OPENCLTranslator may promote 
				//reduction-related statements above the scope where the reduction variable is declared.
				//In this case, the below targetSymbolTable will not work.
				//To handle this promotion, we simply used the enclosing function body as targetSymbolTable.
/*				if( targetSymbolTable instanceof Procedure ) {
					targetSymbolTable = ((Procedure)targetSymbolTable).getBody();
				}
				if( targetSymbolTable == null ) {
					targetSymbolTable = (SymbolTable) cProc.getBody();
				}*/
				targetSymbolTable = cProc.getBody();
				
				List<Expression> startList = new LinkedList<Expression>();
				List<Expression> lengthList = new LinkedList<Expression>();
				boolean foundDimensions = AnalysisTools.extractDimensionInfo(sArray, startList, lengthList, IRSymbolOnly, at);
				if( !foundDimensions ) {
					Tools.exit("[ERROR in CUDATranslationTools.reductionTransformation()] Dimension information of the following " +
							"reduction variable is unknown: " + sArray.getArrayName() + ", Enclosing procedure: " + 
							cProc.getSymbolName() + "; the ACC2GPU translation failed!");
				}
				int dimsize = lengthList.size();

				/* 
				 * Create a new temporary variable for the reduction variable.
				 */
				VariableDeclaration gpu_gred_decl = null;
				VariableDeclaration gpu_wred_decl = null;
				Identifier ggred_var = null;
				Identifier lgred_var = null;
				Identifier lgcred_var = null;
				Identifier gwred_var = null;
				Identifier lwred_var = null;
				String symNameBase = null;
				if( redSym instanceof AccessSymbol) {
					symNameBase = TransformTools.buildAccessSymbolName((AccessSymbol)redSym);
				} else {
					symNameBase = redSym.getSymbolName();
				}
				String gpuGRedSymName = "ggred__" + symNameBase;
				String localGRedSymName = "lgred__" + symNameBase;
				String localGCRedSymName = "lgcred__" + symNameBase;
				String gpuWRedSymName = "gwred__" + symNameBase;
				String localWRedSymNameOnShared = "lwreds__" + symNameBase;
				String localWRedSymNameOnGlobal = "lwredg__" + symNameBase;
				
				/////////////////////////////////////////////////////////////////////////////////
				// __device__ and __global__ functions can not declare static/extern variables //
				// inside their body.                                                          //
				/////////////////////////////////////////////////////////////////////////////////
				List<Specifier> typeSpecs = new ArrayList<Specifier>();
				Symbol IRSym = redSym;
				if( IRSym instanceof PseudoSymbol ) {
					IRSym = ((PseudoSymbol)IRSym).getIRSymbol();
				}
				if( IRSymbolOnly ) {
					typeSpecs.addAll(((VariableDeclaration)IRSym.getDeclaration()).getSpecifiers());
				} else {
					Symbol tSym = redSym;
					while( tSym instanceof AccessSymbol ) {
						tSym = ((AccessSymbol)tSym).getMemberSymbol();
					}
					typeSpecs.addAll(((VariableDeclaration)tSym.getDeclaration()).getSpecifiers());
				}
				typeSpecs.remove(Specifier.STATIC);
				typeSpecs.remove(Specifier.CONST);
				typeSpecs.remove(Specifier.EXTERN);
				SizeofExpression sizeof_expr = new SizeofExpression(typeSpecs);
				
				List<Specifier> removeSpecs = new ArrayList<Specifier>(1);
				removeSpecs.add(Specifier.STATIC);
				removeSpecs.add(Specifier.CONST);
				removeSpecs.add(Specifier.EXTERN);
				List<Specifier> addSpecs = null;
				
				boolean workerRedCachingOnShared = false;
				boolean gangRedCachingOnShared = false;
				if( localRedVarConf == 1 ) {
					//local scalar reduction variables are allocated in the GPU 
					//shared memory and local array reduction variables are cached 
					//on the shared memory if included in CUDA sharedRO/sharedRW clause.
					if( sharedCachingMap.keySet().contains(redSym) ) {
						if( isWorkerReduction ) {
							if( num_workers > 0 ) {
								workerRedCachingOnShared = true;
							} else {
								PrintTools.println("\n[WARNING] caching of worker-reduction variable, " + redSym.getSymbolName() +
										", on the shared memory is not appplicable, since the number of workers are not " +
										"compile-time constant.\nEnclosing procedure: " + cProc.getSymbolName() + "\n"  ,0);
							}
						} else if( isGangReduction ) { //pure gang-reduction
							addSpecs = new ArrayList<Specifier>(1);
							addSpecs.add(CUDASpecifier.CUDA_SHARED);
							gangRedCachingOnShared = true;
						}
					} 

					if( isWorkerReduction && !isArray && !isPointer ) {
						//If worker-reduction variable is scalar, always cache it on the shared memory.
						workerRedCachingOnShared = true;
					}
				} else if( localRedVarConf == 2 ) {
					//local scalar reduction variables are allocated in the GPU 
					//shared memory and local array reduction variables are cached 
					//on the shared memory.
					if( isWorkerReduction ) {
						workerRedCachingOnShared = true;
					} else if( isGangReduction ) {
						addSpecs = new ArrayList<Specifier>(1);
						addSpecs.add(CUDASpecifier.CUDA_SHARED);
						gangRedCachingOnShared = true;
					}
				}
				
				boolean insertGMalloc = false;
				boolean insertWMalloc = false;
				
				if( isWorkerReduction ) {
					//PrintTools.println("[reductionTransformation] start worker-reduction handling", 2);
					//[DEBUG] in-block reduction variable should be volatile to exploit implicit
					//synchronization within a warp.
					List<Specifier> wTypeSpecs = new LinkedList<Specifier>(typeSpecs);
					if( !wTypeSpecs.contains(Specifier.VOLATILE) ) {
						wTypeSpecs.add(0, Specifier.VOLATILE);
					}
					if( workerRedCachingOnShared ) {
						//////////////////////////////////////////////////////
						//Create a worker-private variable on shared memory //
						//using array expansion.                            //
						/////////////////////////////////////////////////////////////
						// __shared__ volatile float lwreds__x[SIZE][num_workers]; //
						// (To use implicit synchronization within a warp, global  //
						// or shared variable should be volatile.)                 //
						/////////////////////////////////////////////////////////////
						VariableDeclarator arrayV_declarator =	
								arrayWorkerPrivCachingOnSM(redSym, localWRedSymNameOnShared, wTypeSpecs, startList,
								lengthList, scope, num_workersExp.clone());
						lwred_var = new Identifier(arrayV_declarator);
                        if( at != region ) {
                        	VariableDeclaration arrayV_decl = (VariableDeclaration)arrayV_declarator.getDeclaration();
                        	Statement arrayV_Stmt = (Statement)arrayV_decl.getParent();
                        	scope.removeStatement(arrayV_Stmt);
                        	arrayV_decl.setParent(null);
                        	CompoundStatement atP = (CompoundStatement)at.getParent();
                        	atP.addDeclaration(arrayV_decl);
                        }
					} else {
						//////////////////////////////////////////////////////
						//Create a worker-private variable on a global      //
						//memory using array expansion.                     //
						////////////////////////////////////////////////////////////
						//     float volatile gwred__x[totalNumThreads][SIZE1];   //
						// (To use implicit synchronization within a warp, global //
						// or shared variable should be volatile.)                //
						////////////////////////////////////////////////////////////
						///////////////////////////////////////////////////////////////
						// Create a GPU device variable corresponding to privSym     //
						// Ex: float * gwred__x; //GPU variable for worker-reduction //
						///////////////////////////////////////////////////////////////
						// Give a new name for the device variable 
						wTypeSpecs.remove(CUDASpecifier.CUDA_SHARED);
						gpu_wred_decl =  TransformTools.getGPUVariable(gpuWRedSymName, targetSymbolTable, 
								wTypeSpecs, main_TrUnt, OpenACCHeaderEndMap, new IntegerLiteral(0));
						gwred_var = new Identifier((VariableDeclarator)gpu_wred_decl.getDeclarator(0));
						/////////////////////////////////////////////////////
						// Memory allocation for the device variable       //
						// Insert cudaMalloc() function before the region  //
						///////////////////////////////////////////	//////////
						// Ex: cudaMalloc(((void *  * )( & gwred__x)), gpuBytes);

                        FunctionCall malloc_call = new FunctionCall(new NameID("HI_tempMalloc1D"));
						List<Specifier> specs = new ArrayList<Specifier>(4);
						specs.add(Specifier.VOID);
						specs.add(PointerSpecifier.UNQUALIFIED);
						specs.add(PointerSpecifier.UNQUALIFIED);
						List<Expression> arg_list = new ArrayList<Expression>();
						arg_list.add(new Typecast(specs, new UnaryExpression(UnaryOperator.ADDRESS_OF, 
								(Identifier)gwred_var.clone())));

						if( !allocatedWorkerReductionSet.contains(redSym) ) {
							insertWMalloc = true;
							allocatedWorkerReductionSet.add(redSym);
						}

						// Insert "gpuBytes = totalGpuNumThreads * (dimension1 * dimension2 * ..) 
						// * sizeof(varType);" statement
						sizeof_expr.getTypes().remove(CUDASpecifier.CUDA_SHARED);
						Expression biexp = null;
						Expression biexp2 = null;
						if( dimsize > 0 ) {
							biexp = lengthList.get(0).clone();
							for( int i=1; i<dimsize; i++ )
							{
								biexp = new BinaryExpression(biexp, BinaryOperator.MULTIPLY, lengthList.get(i).clone());
							}
							biexp2 = new BinaryExpression(biexp, BinaryOperator.MULTIPLY, sizeof_expr);
						} else {
							biexp2 = sizeof_expr;
						}
						AssignmentExpression assignex = new AssignmentExpression((Expression)cloned_bytes.clone(),
								AssignmentOperator.NORMAL, biexp2);
						orgGpuBytes_stmt = new ExpressionStatement(assignex);
						biexp = new BinaryExpression((Expression)totalNumThreads.clone(), 
								BinaryOperator.MULTIPLY, (Expression)biexp2.clone());
						assignex = new AssignmentExpression((Expression)cloned_bytes.clone(),AssignmentOperator.NORMAL, 
								biexp);
						gpuBytes_stmt = new ExpressionStatement(assignex);

						// Create a parameter Declaration for the kernel function
						// Create an extended array type
						// Ex1: "float lwredg__x[][SIZE1]"
						// Ex2: "float lwredg__x[][SIZE1][SIZE2]"
						VariableDeclarator arrayV_declarator =	arrayPrivConv(redSym, localWRedSymNameOnGlobal, wTypeSpecs, startList,
								lengthList, new_proc, region, scope, 1, call_to_new_proc, gwred_var.clone());
						lwred_var = new Identifier(arrayV_declarator);

						// Add gpuBytes argument to cudaMalloc() call
						arg_list.add((Identifier)cloned_bytes.clone());
                        arg_list.add(new NameID("acc_device_current"));
						malloc_call.setArguments(arg_list);
						ExpressionStatement malloc_stmt = new ExpressionStatement(malloc_call);
						// Insert malloc statement.
						if( insertWMalloc ) {
							//mallocScope.addStatementBefore(confRefStmt, gpuBytes_stmt);
							//mallocScope.addStatementBefore(confRefStmt, malloc_stmt);
							Symbol gpu_wred_sym = AnalysisTools.findsSymbol( ((CompoundStatement)confRefStmt.getParent()).getSymbols(), gpuWRedSymName);
							if(gpu_wred_sym == null)
							{
								((CompoundStatement)confRefStmt.getParent()).addDeclaration(gpu_wred_decl.clone());
							}
							prefixStmts.addStatement(gpuBytes_stmt);
							prefixStmts.addStatement(malloc_stmt);
							if( opt_addSafetyCheckingCode ) {
								// Insert "gpuGmemSize += gpuBytes;" statement 
								//mallocScope.addStatementBefore(gpuBytes_stmt.clone(), 
								//		(Statement)gMemAdd_stmt.clone());
								prefixStmts.addStatement(gMemAdd_stmt.clone());
							}
							/*
							 * Insert cudaFree() to deallocate device memory for worker-private variable. 
							 * Because cuda-related statements are added in reverse order, 
							 * this function call is added first.
							 */
							// Insert "cudaFree(gwred__x);"
							FunctionCall cudaFree_call = new FunctionCall(new NameID("HI_tempFree"));
							specs = new ArrayList<Specifier>(4);
							specs.add(Specifier.VOID);
							specs.add(PointerSpecifier.UNQUALIFIED);
							specs.add(PointerSpecifier.UNQUALIFIED);
							cudaFree_call.addArgument(new Typecast(specs, new UnaryExpression(UnaryOperator.ADDRESS_OF,
									(Identifier)gwred_var.clone())));
							cudaFree_call.addArgument(new NameID("acc_device_current"));
							ExpressionStatement cudaFree_stmt = new ExpressionStatement(cudaFree_call);
							if( confRefStmt != region ) {
								postscriptStmts.addStatement(gpuBytes_stmt.clone());
								if( opt_addSafetyCheckingCode  ) {
									postscriptStmts.addStatement(cudaFree_stmt);
									postscriptStmts.addStatement(gMemSub_stmt.clone());
								}
							} else {
								if( opt_addSafetyCheckingCode  ) {
									regionParent.addStatementAfter(region, gMemSub_stmt.clone());
									regionParent.addStatementAfter(region, gpuBytes_stmt.clone());
								}
								regionParent.addStatementAfter(region, cudaFree_stmt);
							}
						}
					}
					tWorkerParamRedVarMap.put(redSym, lwred_var.clone());
					//Reset below to be checked later.
					gpuBytes_stmt = null;
					orgGpuBytes_stmt = null;
				} 
				//PrintTools.println("[reductionTransformation] finish worker-reduction handling", 2);
				Statement refSt = null;
				if( isGangReduction ) {
					if( gangRedCachingOnShared && !isWorkerReduction 
							&& !AnalysisTools.ipContainPragmas(scope, ACCAnnotation.class, "worker", null) ) {
						//pure gang-reduction variable is cached on the GPU shared memory.
						/////////////////////////////////////////////////////
						//Create a gang-private variable on shared memory. //
						//(No array-expansion is needed.)                  //
						/////////////////////////////////////////////////////
						//     __shared__ float lgcred__x;                 //
						//     __shared__ float lgcred__x[SIZE];           //
						/////////////////////////////////////////////////////
						if( redSym instanceof AccessSymbol ) {
							Symbol tSym = redSym;
							while( tSym instanceof AccessSymbol ) {
								tSym = ((AccessSymbol)tSym).getMemberSymbol();
							}
							lgcred_var = TransformTools.declareClonedArrayVariable(scope, sArray, localGCRedSymName, 
									removeSpecs, addSpecs, false);
						} else {
							lgcred_var = TransformTools.declareClonedArrayVariable(scope, sArray, localGCRedSymName, 
									removeSpecs, addSpecs, false);
						}
						/////////////////////////////////////////////////////////////////////////////
						// Replace the gang-private variable with this new local private variable. //
						/////////////////////////////////////////////////////////////////////////////
						if( redSym instanceof AccessSymbol ) {
							TransformTools.replaceAccessExpressions(at, (AccessSymbol)redSym, lgcred_var);
						} else {
							TransformTools.replaceAll(at, new Identifier(redSym), lgcred_var);
						}
						//Reset below to be checked later.
						gpuBytes_stmt = null;
						orgGpuBytes_stmt = null;
						//PrintTools.println("[reductionTransformation] gang reduction point 1", 2);
					} 
					/////////////////////////////////////////////////////////////
					// Create a GPU device variable corresponding to redSym    //
					// Ex: float * ggred__x; //GPU variable for gang-reduction //
					/////////////////////////////////////////////////////////////
					// Give a new name for the device variable 
					gpu_gred_decl =  TransformTools.getGPUVariable(gpuGRedSymName, targetSymbolTable, 
							typeSpecs, main_TrUnt, OpenACCHeaderEndMap, new IntegerLiteral(0));
					ggred_var = new Identifier((VariableDeclarator)gpu_gred_decl.getDeclarator(0));
					/////////////////////////////////////////////////////
					// Memory allocation for the device variable       //
					// Insert cudaMalloc() function before the region  //
					/////////////////////////////////////////////////////
					// Ex: cudaMalloc(((void *  * )( & gpriv__x)), gpuBytes);
					// ==> Changed to the following call.
					// Ex: HI_tempMalloc1D(((void *  * )( & gpriv__x)), gpuBytes, acc_device_nvidia);
					//FunctionCall malloc_call = new FunctionCall(new NameID("cudaMalloc"));
					FunctionCall malloc_call = new FunctionCall(new NameID("HI_tempMalloc1D"));
					List<Specifier> specs = new ArrayList<Specifier>(4);
					specs.add(Specifier.VOID);
					specs.add(PointerSpecifier.UNQUALIFIED);
					specs.add(PointerSpecifier.UNQUALIFIED);
					List<Expression> arg_list = new ArrayList<Expression>();
					arg_list.add(new Typecast(specs, new UnaryExpression(UnaryOperator.ADDRESS_OF, 
							(Identifier)ggred_var.clone())));
					
					if( !allocatedGangReductionSet.contains(redSym) ) {
						insertGMalloc = true;
						allocatedGangReductionSet.add(redSym);
					}
					
					//PrintTools.println("[reductionTransformation] gang reduction point 2", 2);
					if( !isArray && !isPointer ) { //scalar variable
						// Insert "gpuBytes = gpuNumBlocks * sizeof(varType);" statement 
						AssignmentExpression assignex = new AssignmentExpression((Identifier)cloned_bytes.clone(),
								AssignmentOperator.NORMAL, new BinaryExpression((Expression)numBlocks.clone(), 
										BinaryOperator.MULTIPLY, sizeof_expr.clone()));
						gpuBytes_stmt = new ExpressionStatement(assignex);
						AssignmentExpression assignex2 = new AssignmentExpression((Identifier)cloned_bytes.clone(),
								AssignmentOperator.NORMAL, sizeof_expr.clone());
						orgGpuBytes_stmt = new ExpressionStatement(assignex2);

						// Create a parameter Declaration for the kernel function
						// Change the scalar variable to a pointer type 
						// ex: float *lgred__x,
						boolean registerRO = false;
						if( !gangRedCachingOnShared && regROCachingMap.keySet().contains(redSym) ) {
							registerRO = true;
						}
						boolean isPureGangReduction = false;
						if( !isWorkerReduction 
								&& !AnalysisTools.ipContainPragmas(scope, ACCAnnotation.class, "worker", null) ) {
							isPureGangReduction = true;
						}
						VariableDeclarator pointerV_declarator = 
							scalarGangPrivConv(redSym, localGRedSymName, typeSpecs, new_proc, region, scope, 
									registerRO, isPureGangReduction, call_to_new_proc, ggred_var.clone());
						lgred_var = new Identifier(pointerV_declarator);
						//PrintTools.println("[reductionTransformation] gang reduction point 3", 2);
					} else { //non-scalar variables
						// Insert "gpuBytes = gpuNumBlocks * (dimension1 * dimension2 * ..) 
						// * sizeof(varType);" statement
						Expression biexp = lengthList.get(0).clone();
						for( int i=1; i<dimsize; i++ )
						{
							biexp = new BinaryExpression(biexp, BinaryOperator.MULTIPLY, lengthList.get(i).clone());
						}
						BinaryExpression biexp2 = new BinaryExpression(biexp, BinaryOperator.MULTIPLY, sizeof_expr.clone());
						AssignmentExpression assignex = new AssignmentExpression((Expression)cloned_bytes.clone(),
								AssignmentOperator.NORMAL, biexp2);
						orgGpuBytes_stmt = new ExpressionStatement(assignex);
						biexp = new BinaryExpression((Expression)numBlocks.clone(), 
								BinaryOperator.MULTIPLY, (Expression)biexp2.clone());
						assignex = new AssignmentExpression((Expression)cloned_bytes.clone(),AssignmentOperator.NORMAL, 
								biexp);
						gpuBytes_stmt = new ExpressionStatement(assignex);

						// Create a parameter Declaration for the kernel function
						// Create an extended array type
						// Ex1: "float b[][SIZE1]"
						// Ex2: "float b[][SIZE1][SIZE2]"
						VariableDeclarator arrayV_declarator =	arrayPrivConv(redSym, localGRedSymName, typeSpecs, startList,
								lengthList, new_proc, region, scope, 0, call_to_new_proc, ggred_var.clone() );
						lgred_var = new Identifier(arrayV_declarator);
						//PrintTools.println("[reductionTransformation] gang reduction point 4", 2);
					}
					// Add gpuBytes argument to cudaMalloc() call
					arg_list.add((Identifier)cloned_bytes.clone());
					// Add acc_device_nvidia argument to HI_tempMalloc1D() call
					arg_list.add(new NameID("acc_device_current"));
					malloc_call.setArguments(arg_list);
					ExpressionStatement malloc_stmt = new ExpressionStatement(malloc_call);
					if( insertGMalloc ) {
						// Insert malloc statement.
						//mallocScope.addStatementBefore(confRefStmt, gpuBytes_stmt);
						//mallocScope.addStatementBefore(confRefStmt, malloc_stmt);
						prefixStmts.addStatement(gpuBytes_stmt);
						prefixStmts.addStatement(malloc_stmt);
						if( opt_addSafetyCheckingCode ) {
							// Insert "gpuGmemSize += gpuBytes;" statement 
							//mallocScope.addStatementBefore(gpuBytes_stmt.clone(), 
								//	(Statement)gMemAdd_stmt.clone());
							prefixStmts.addStatement(gMemAdd_stmt.clone());
						}
						/*
						 * Insert cudaFree() to deallocate device memory for gang-reduction variable. 
						 * Because cuda-related statements are added in reverse order, 
						 * this function call is added first.
						 */
/*						if( opt_addSafetyCheckingCode  ) {
							// Insert "gpuGmemSize -= gpuBytes;" statement 
							mallocScope.addStatementAfter(confRefStmt, gMemSub_stmt.clone());
						}
*/						// Insert "cudaFree(ggred__x);"
						// Changed to "HI_tempFree((void **)(& ggred__x), acc_device_nvidia)";
						FunctionCall cudaFree_call = new FunctionCall(new NameID("HI_tempFree"));
						specs = new ArrayList<Specifier>(4);
						specs.add(Specifier.VOID);
						specs.add(PointerSpecifier.UNQUALIFIED);
						specs.add(PointerSpecifier.UNQUALIFIED);
						cudaFree_call.addArgument(new Typecast(specs, new UnaryExpression(UnaryOperator.ADDRESS_OF, 
								(Identifier)ggred_var.clone())));
						cudaFree_call.addArgument(new NameID("acc_device_current"));
						ExpressionStatement cudaFree_stmt = new ExpressionStatement(cudaFree_call);
						//postscriptStmts.addStatement(gpuBytes_stmt.clone());
						refSt = cudaFree_stmt;
						if( asyncID == null ) {
							if( confRefStmt != region ) {
								postscriptStmts.addStatement(cudaFree_stmt);
								if( opt_addSafetyCheckingCode  ) {
									postscriptStmts.addStatement(gMemSub_stmt.clone());
								}
							} else {
								if( opt_addSafetyCheckingCode  ) {
									regionParent.addStatementAfter(region, gMemSub_stmt.clone());
								}
								regionParent.addStatementAfter(region, cudaFree_stmt);
							}
						} else {
							if( asyncConfRefChanged ) {
								postRedStmts.add(cudaFree_stmt);
								if( opt_addSafetyCheckingCode  ) {
									postRedStmts.add(gMemSub_stmt.clone());
								}
							} else {
								postscriptStmts.addStatement(cudaFree_stmt);
								if( opt_addSafetyCheckingCode  ) {
									postscriptStmts.addStatement(gMemSub_stmt.clone());
								}
							}
						}
					}
					tGangParamRedVarMap.put(redSym, lgred_var.clone());
					//PrintTools.println("[reductionTransformation] gang reduction point 5", 2);
					
					/////////////////////////////////////////////////////////////////////////////////////////////////////
					// Pure gang-reduction variable cached on the shared memory should be flushed back to the original //
					// gang-reduction variable on the global memory.                                                   //
					/////////////////////////////////////////////////////////////////////////////////////////////////////
					if( gangRedCachingOnShared && !isWorkerReduction ) { //pure gang-reduction variable is cached on the GPU shared memory.
						Statement estmt = null;
						if( !isArray && !isPointer ) { //scalar variable
							Expression LHS = new ArrayAccess(lgred_var.clone(), SymbolTools.getOrphanID("_bid"));
							Expression RHS = lgcred_var.clone();
							estmt = new ExpressionStatement(new AssignmentExpression(LHS, AssignmentOperator.NORMAL,
									RHS));
						} else {
							List<Identifier> index_vars = new LinkedList<Identifier>();
							for( int i=0; i<=dimsize; i++ ) {
								index_vars.add(TransformTools.getTempIndex(scope, tempIndexBase+i));
							}
							List<Expression> indices = new ArrayList<Expression>(dimsize+1);
							indices.add(SymbolTools.getOrphanID("_bid"));
							for( int k=0; k<dimsize; k++ ) {
								indices.add((Expression)index_vars.get(k).clone());
							}
							Expression LHS = new ArrayAccess(lgred_var.clone(), indices);
							indices = new ArrayList<Expression>(dimsize);
							for( int k=0; k<dimsize; k++ ) {
								indices.add((Expression)index_vars.get(k).clone());
							}
							Expression RHS = new ArrayAccess(lgcred_var.clone(), indices);
							estmt = TransformTools.genArrayCopyLoop(index_vars, lengthList, LHS, RHS);
						}
						scope.addStatement(estmt);
					}
				}

				//PrintTools.println("[reductionTransformation] insert reduction init statement", 2);
				////////////////////////////////////////////////
				// Insert reduction-initialization statement. //
				////////////////////////////////////////////////
				Expression LHS = null;
				Expression RHS = TransformTools.getRInitValue(redOp, typeSpecs);
				Statement estmt = null;
				if( !isArray && !isPointer ) { //scalar variable
					if( isWorkerReduction ) {
						//Initialize worker-reduction variable
						//Ex2: lwred__x[_gtid] = initValue;
						if( workerRedCachingOnShared ) { //allocated on the shared memory
							//Ex: lwred__x[_tid] = initValue;
							LHS = new ArrayAccess(lwred_var.clone(), SymbolTools.getOrphanID("_tid"));
						} else { //allocated on the global memory
							//Ex: lwred__x[_gtid] = initValue;
							LHS = new ArrayAccess(lwred_var.clone(), SymbolTools.getOrphanID("_gtid"));
						}
					} else if( isGangReduction ){
						//Initialize pure-gang-reduction variable.
						if( gangRedCachingOnShared ) {
							//Ex2: lgcred__x = initValue;
							LHS = lgcred_var.clone();
						} else {
							//Ex1: lgred__x[_bid] = initValue;
							LHS = new ArrayAccess(lgred_var.clone(), SymbolTools.getOrphanID("_bid"));
						}
					}
					estmt = new ExpressionStatement(
							new AssignmentExpression((Expression)LHS.clone(), 
									AssignmentOperator.NORMAL, RHS));
				} else { //non-scalar variable
					///////////////////////////////////////////////////////
					// Ex1: worker-reduction allocated on shared memory  //
					//      for(i=0; i<SIZE1; i++) {                     //
					//         for(k=0; k<SIZE2; k++) {                  //
					//             lwred__x[i][k][_tid] = initValue;     //
					//         }                                         //
					//      }                                            //
					///////////////////////////////////////////////////////
					// Ex2: worker-reduction allocated on global memory  //
					//      for(i=0; i<SIZE1; i++) {                     //
					//         for(k=0; k<SIZE2; k++) {                  //
					//             lwred__x[_gtid][i][k] = initValue;    //
					//         }                                         //
					//      }                                            //
					///////////////////////////////////////////////////////
					// Ex3: gang-reduction allocated on global memory    //
					//      for(i=0; i<SIZE1; i++) {                     //
					//          lgred__x[_bid][i] = initValue;           //
					//      }                                            //
					///////////////////////////////////////////////////////
					// Ex4: gang-reduction cached on global memory       //
					//      for(i=0; i<SIZE1; i++) {                     //
					//          lgcred__x[i] = initValue;                //
					//      }                                            //
					///////////////////////////////////////////////////////
					//////////////////////////////////////// //////
					// Create or find temporary index variables. // 
					//////////////////////////////////////// //////
					List<Identifier> index_vars = new LinkedList<Identifier>();
					CompoundStatement tScope = scope;
					if( isWorkerReduction ) {
						if( at != region ) {
							tScope = (CompoundStatement)at.getParent();
						}
					}
					for( int i=0; i<=dimsize; i++ ) {
						index_vars.add(TransformTools.getTempIndex(tScope, tempIndexBase+i));
					}
					List<Expression> indices = new LinkedList<Expression>();
					if( isWorkerReduction ) {
						if( !workerRedCachingOnShared ) {
							indices.add(SymbolTools.getOrphanID("_gtid"));
						}
						for( int k=0; k<dimsize; k++ ) {
							indices.add((Expression)index_vars.get(k).clone());
						}
						if( workerRedCachingOnShared ) {
							indices.add(SymbolTools.getOrphanID("_tid"));
						}
						LHS = new ArrayAccess(lwred_var.clone(), indices);
					} else if( isGangReduction ) {
						if( !gangRedCachingOnShared ) {
							indices.add(SymbolTools.getOrphanID("_bid"));
						}
						for( int k=0; k<dimsize; k++ ) {
							indices.add((Expression)index_vars.get(k).clone());
						}
						if( gangRedCachingOnShared ) {
							LHS = new ArrayAccess(lgcred_var.clone(), indices);
						} else {
							LHS = new ArrayAccess(lgred_var.clone(), indices);
						}
					}
					estmt = TransformTools.genArrayCopyLoop(index_vars, lengthList, LHS, RHS);
				}

				if( isWorkerReduction ) {
					if( at == region ) {
						if( at instanceof ForLoop ) {
							preList.add(estmt);
						} else {
							Tools.exit("[ERROR in CUDATranslationTools.reductionTransformation()] " +
									"unexpected worker reduction for a compound statement; exit!\n" +
									"OpenACC Annotation: " + pannot + "\n" +
											"Enclosing procedure: " + cProc.getSymbolName() + "\n");
						}
					} else {
						CompoundStatement atP = (CompoundStatement)at.getParent();
						atP.addStatementBefore((Statement)at, estmt);
					}
				} else if( isGangReduction ) {
					if( region instanceof ForLoop ) {
						preList.add(estmt);
					} else {
						Statement last_decl_stmt;
						last_decl_stmt = IRTools.getLastDeclarationStatement(scope);
						if( last_decl_stmt != null ) {
							scope.addStatementAfter(last_decl_stmt, estmt);
						} else {
							last_decl_stmt = (Statement)scope.getChildren().get(0);
							scope.addStatementBefore(last_decl_stmt, estmt);
						}
					}
				}

				//PrintTools.println("[reductionTransformation] insert final global reduction statement", 2);
				////////////////////////////////////////////////////////////////
				// Insert final global reduction for gang-reduction variable. //
				// The reduction statement is executed separately for each    //
				// gang-reduction for CPU locality.                           //
				////////////////////////////////////////////////////////////////
				String extGRedName = "extred__" + symNameBase;
				String orgGRedName = "orgred__" + symNameBase;
				VariableDeclaration cpu_extred_decl = null;
				VariableDeclaration cpu_orgred_decl = null;
				Identifier extended_var = null;
				Identifier orgred_var = null;
				///////////////////////////////////////////////////////////////////////////////////////
				// Create a temporary array that is an extended version of the reduction variable.   //
				// - The extended array is used for final reduction across thread blocks on the CPU. //
				// ex: float * extred__x;                                                            //
				///////////////////////////////////////////////////////////////////////////////////////
				// If kernelVerification == true, create another temporary variable, which contains  //
				// the value of the original reduction variable and is used for final reduction for  //
				// GPU kernel.                                                                       //
				// ex: float * orgred__x;                                                            //
				///////////////////////////////////////////////////////////////////////////////////////
				if( insertGMalloc ) {
					cpu_extred_decl =  TransformTools.getGPUVariable(extGRedName, targetSymbolTable, 
							typeSpecs, main_TrUnt, OpenACCHeaderEndMap, new IntegerLiteral(0));
					extended_var = new Identifier((VariableDeclarator)cpu_extred_decl.getDeclarator(0));
					if( kernelVerification ) {
						cpu_orgred_decl =  TransformTools.getGPUVariable(orgGRedName, targetSymbolTable, 
								typeSpecs, main_TrUnt, OpenACCHeaderEndMap, new IntegerLiteral(0));
						orgred_var = new Identifier((VariableDeclarator)cpu_orgred_decl.getDeclarator(0));
					}
					//////////////////////////////////////////////////////////////////////////////////////
					// Create a malloc statement:                                                       //
					//     HI_tempMalloc1D(((void *  * )( & extred__x)), gpuBytes, acc_device_host); //
					//////////////////////////////////////////////////////////////////////////////////////
					FunctionCall tempMalloc_call = new FunctionCall(new NameID("HI_tempMalloc1D"));
					List<Specifier> castspecs = new ArrayList<Specifier>(4);
					castspecs.add(Specifier.VOID);
					castspecs.add(PointerSpecifier.UNQUALIFIED);
					castspecs.add(PointerSpecifier.UNQUALIFIED);
					List<Expression> arg_list = new ArrayList<Expression>();
					arg_list.add(new Typecast(castspecs, new UnaryExpression(UnaryOperator.ADDRESS_OF, 
							(Identifier)extended_var.clone())));
					arg_list.add(cloned_bytes.clone());
					arg_list.add(new NameID("acc_device_host"));
					tempMalloc_call.setArguments(arg_list);
					ExpressionStatement eMallocStmt = new ExpressionStatement(tempMalloc_call);
					prefixStmts.addStatement(eMallocStmt);
/*					/////////////////////////////////////////////////////////////////////////
					// Create malloc() statement, "extred__x = (float *)malloc(gpuBytes);" //
					/////////////////////////////////////////////////////////////////////////
					FunctionCall tempMalloc_call = new FunctionCall(new NameID("malloc"));
					tempMalloc_call.addArgument((Expression)cloned_bytes.clone());
					List<Specifier> castspecs = new LinkedList<Specifier>();
					castspecs.addAll(typeSpecs);
					castspecs.add(PointerSpecifier.UNQUALIFIED);
					Expression assignex = new AssignmentExpression((Identifier)extended_var.clone(),
							AssignmentOperator.NORMAL, new Typecast(castspecs, tempMalloc_call));
					ExpressionStatement eMallocStmt = new ExpressionStatement(assignex);
					////mallocScope.addStatementBefore(
					////	confRefStmt, eMallocStmt);
					prefixStmts.addStatement(eMallocStmt);*/
					if( kernelVerification ) {
						//////////////////////////////////////////////////////////////////////////////////////	
						//     gpuBytes = sizeof(float);                                                    //
						//     HI_tempMalloc1D(((void *  * )( & orgred__x)), gpuBytes, acc_device_host); //
						//////////////////////////////////////////////////////////////////////////////////////
						prefixStmts.addStatement(orgGpuBytes_stmt.clone());
						tempMalloc_call = new FunctionCall(new NameID("HI_tempMalloc1D"));
						castspecs = new ArrayList<Specifier>(4);
						castspecs.add(Specifier.VOID);
						castspecs.add(PointerSpecifier.UNQUALIFIED);
						castspecs.add(PointerSpecifier.UNQUALIFIED);
						arg_list = new ArrayList<Expression>();
						arg_list.add(new Typecast(castspecs, new UnaryExpression(UnaryOperator.ADDRESS_OF, 
								(Identifier)orgred_var.clone())));
						arg_list.add(cloned_bytes.clone());
						arg_list.add(new NameID("acc_device_host"));
						tempMalloc_call.setArguments(arg_list);
						eMallocStmt = new ExpressionStatement(tempMalloc_call);
						prefixStmts.addStatement(eMallocStmt);
						//////////////////////////////////////////////////////////
						//Copy initial value of the original reduction variable.//
						//////////////////////////////////////////////////////////
						// for( k=0; k<SIZE1; k++ ) {                           //
						//     for( m=0; m<SIZE2; m++ ) {                       //
						//         orgred__x[k*SIZE2+m] = x[k][m];              //
						//     }                                                //
						// }                                                    //
						//////////////////////////////////////////////////////////
						// Create or find temporary index variables. 
						List<Identifier> index_vars = new LinkedList<Identifier>();
						for( int i=0; i<dimsize; i++ ) {
							index_vars.add(TransformTools.getTempIndex(cProc.getBody(), tempIndexBase+i));
						}
						LHS = null;
						if( dimsize == 0 ) {
							//ex: (*orgred__x)
							LHS = new UnaryExpression(UnaryOperator.DEREFERENCE, orgred_var.clone());
						} else {
							//ex: orgred__x[k*SIZE2 + m]
							Expression indexEx = null;
							for( int k=0; k<dimsize; k++ ) {
								Expression tExp = null;
								if( k+1 < dimsize ) {
									tExp = lengthList.get(k+1).clone();
									for( int m=k+2; m<dimsize; m++ ) {
										tExp = new BinaryExpression(tExp, BinaryOperator.MULTIPLY, lengthList.get(m).clone());
									} 
									tExp = new BinaryExpression(index_vars.get(k).clone(), BinaryOperator.MULTIPLY, tExp); 
								} else {
									tExp = index_vars.get(k).clone();
								}
								if( indexEx == null ) {
									indexEx = tExp;
								} else {
									indexEx = new BinaryExpression(indexEx, BinaryOperator.ADD, tExp);
								}
							}
							LHS = new ArrayAccess(orgred_var.clone(), indexEx);
						}
						RHS = null;
						if( dimsize == 0 ) {
							//ex: x;
							if( redSym instanceof AccessSymbol ) {
								RHS = AnalysisTools.accessSymbolToExpression((AccessSymbol)redSym, null);
							} else {
								RHS = new Identifier(redSym);
							}
						} else {
							//ex: x[k][m];
							List<Expression> indices = new LinkedList<Expression>();
							for( int k=0; k<dimsize; k++ ) {
								indices.add((Expression)index_vars.get(k).clone());
							}
							if( redSym instanceof AccessSymbol ) {
								RHS = AnalysisTools.accessSymbolToExpression((AccessSymbol)redSym, indices);
							} else {
								RHS = new ArrayAccess(new Identifier(redSym), indices);
							}
						}
						Statement initCopyStmt = TransformTools.genArrayCopyLoop(index_vars, lengthList, LHS, RHS);
						//Insert the init statement right before this kernel region.
						if( confRefStmt == region ) {
							prefixStmts.addStatement(initCopyStmt);
						} else {
							regionParent.addStatementBefore(region, initCopyStmt);
						}
					}
					
					/////////////////////////////////////////////////////////////////////////////////////////
					// Insert free(extred__x); ==> HI_tempFree((void **)(& extred__x), acc_device_host) //
					/////////////////////////////////////////////////////////////////////////////////////////
					FunctionCall free_call = new FunctionCall(new NameID("HI_tempFree"));
					castspecs = new ArrayList<Specifier>(4);
					castspecs.add(Specifier.VOID);
					castspecs.add(PointerSpecifier.UNQUALIFIED);
					castspecs.add(PointerSpecifier.UNQUALIFIED);
					free_call.addArgument(new Typecast(castspecs, new UnaryExpression(UnaryOperator.ADDRESS_OF, 
							(Identifier)extended_var.clone())));
					free_call.addArgument(new NameID("acc_device_host"));
					Statement free_stmt = new ExpressionStatement(free_call);
					////mallocScope.addStatementAfter(
					////	confRefStmt, free_stmt);
					if( asyncID == null ) {
						if( confRefStmt != region ) {
							postscriptStmts.addStatement(free_stmt);
						} else {
							regionParent.addStatementAfter(region, free_stmt);
						}
					} else {
						if( asyncConfRefChanged ) {
							postRedStmts.add(0, free_stmt);
						} else {
							postscriptStmts.addStatement(free_stmt);
						}
					}
					if( kernelVerification ) {
						/////////////////////////////////////////////////////////////
						// HI_tempFree((void **)(& orgred__x), acc_device_host) //
						/////////////////////////////////////////////////////////////
						free_call = new FunctionCall(new NameID("HI_tempFree"));
						castspecs = new ArrayList<Specifier>(4);
						castspecs.add(Specifier.VOID);
						castspecs.add(PointerSpecifier.UNQUALIFIED);
						castspecs.add(PointerSpecifier.UNQUALIFIED);
						free_call.addArgument(new Typecast(castspecs, new UnaryExpression(UnaryOperator.ADDRESS_OF, 
								(Identifier)orgred_var.clone())));
						free_call.addArgument(new NameID("acc_device_host"));
						free_stmt = new ExpressionStatement(free_call);
						if( asyncID == null ) {
							if( confRefStmt != region ) {
								postscriptStmts.addStatement(free_stmt);
							} else {
								regionParent.addStatementAfter(region, free_stmt);
							}
						} else {
							if( asyncConfRefChanged ) {
								postRedStmts.add(0, free_stmt);
							} else {
								postscriptStmts.addStatement(free_stmt);
							}
						}
					}
					
					List<Statement> resultCompareStmts = null;
					if( kernelVerification ) {
						// If kernelVerification is true, insert CPU-GPU result compare statements.
						Expression hostVar = null;
						if( redSym instanceof AccessSymbol ) {
							hostVar = AnalysisTools.accessSymbolToExpression((AccessSymbol)redSym, null);
						} else {
							hostVar = new Identifier(redSym);
						}
						ACCAnnotation cAnnot = region.getAnnotation(ACCAnnotation.class, cRegionKind);
						resultCompareStmts = TransformTools.genResultCompareCodes(cProc, null, 
								hostVar, orgred_var.clone(), null, lengthList, typeSpecs, cAnnot, 
								EPSILON, false, minCheckValue);
					}

					////////////////////////////////////////////////////
					// Insert codes for final reduction on the CPU.   //
					////////////////////////////////////////////////////
					////////////////////////////////////////////////////////////////////////////////
					// Ex1: for(i=0; i<gpuNumBlocks; i++) {                                       //
					//         for(k=0; k<SIZE1; k++) {                                           //
					//             for(m=0; m<SIZE2; m++) {                                       //
					//                x[k][m] += extred__x[i*SIZE1*SIZE2+k*SIZE2+m];              //
					//             }                                                              //
					//         }                                                                  //
					//      }                                                                     //
					// Ex2: for(i=0; i<gpuNumBlocks; i++) {                                       //
					//         for(k=0; k<SIZE1; k++) {                                           //
					//             for(m=0; m<SIZE2; m++) {                                       //
					//                orgred__x[k*SIZE2+m] += extred__x[i*SIZE1*SIZE2+k*SIZE2+m]; //
					//             }                                                              //
					//         }                                                                  //
					//      }                                                                     //
					////////////////////////////////////////////////////////////////////////////////
					// Create or find temporary index variables. 
					List<Identifier> index_vars = new LinkedList<Identifier>();
					for( int i=0; i<=dimsize; i++ ) {
						index_vars.add(TransformTools.getTempIndex(cProc.getBody(), tempIndexBase+i));
					}
					List<Expression> edimensions = new LinkedList<Expression>();
					edimensions.add((Expression)numBlocks.clone());
					for( int i=0; i<dimsize; i++ )
					{
						edimensions.add((Expression)lengthList.get(i).clone());
					}
					// Create LHS expression (ex: x or x[k][m]) 
					// and RHS expression (ex: extred__x[i] or extred__x[i*SIZE1*SIZE2 + k*SIZE2 + m])
					LHS = null;
					RHS = null;
					if( kernelVerification ) {
						if( dimsize == 0 ) {
							//ex: (*orgred__x)
							LHS = new UnaryExpression(UnaryOperator.DEREFERENCE, orgred_var.clone());
						} else {
							//ex: orgred__x[k*SIZE2 + m]
							Expression indexEx = null;
							for( int k=1; k<=dimsize; k++ ) {
								Expression tExp = null;
								if( k < dimsize ) {
									tExp = lengthList.get(k).clone();
									for( int m=k+1; m<dimsize; m++ ) {
										tExp = new BinaryExpression(tExp, BinaryOperator.MULTIPLY, lengthList.get(m).clone());
									} 
									tExp = new BinaryExpression(index_vars.get(k).clone(), BinaryOperator.MULTIPLY, tExp); 
								} else {
									tExp = index_vars.get(k).clone();
								}
								if( indexEx == null ) {
									indexEx = tExp;
								} else {
									indexEx = new BinaryExpression(indexEx, BinaryOperator.ADD, tExp);
								}
							}
							LHS = new ArrayAccess(orgred_var.clone(), indexEx);
						}
					} else {
						if( dimsize == 0 ) {
							//ex: x;
							if( redSym instanceof AccessSymbol ) {
								LHS = AnalysisTools.accessSymbolToExpression((AccessSymbol)redSym, null);
							} else {
								LHS = new Identifier(redSym);
							}
						} else {
							//ex: x[k][m];
							List<Expression> indices = new LinkedList<Expression>();
							for( int k=1; k<=dimsize; k++ ) {
								indices.add((Expression)index_vars.get(k).clone());
							}
							if( redSym instanceof AccessSymbol ) {
								LHS = AnalysisTools.accessSymbolToExpression((AccessSymbol)redSym, indices);
							} else {
								LHS = new ArrayAccess(new Identifier(redSym), indices);
							}
						}
					}
					Expression indexEx = null;
					for( int k=0; k<=dimsize; k++ ) {
						Expression tExp = null;
						if( k < dimsize ) {
							tExp = lengthList.get(k).clone();
							for( int m=k+1; m<dimsize; m++ ) {
								tExp = new BinaryExpression(tExp, BinaryOperator.MULTIPLY, lengthList.get(m).clone());
							} 
							tExp = new BinaryExpression(index_vars.get(k).clone(), BinaryOperator.MULTIPLY, tExp); 
						} else {
							tExp = index_vars.get(k).clone();
						}
						if( indexEx == null ) {
							indexEx = tExp;
						} else {
							indexEx = new BinaryExpression(indexEx, BinaryOperator.ADD, tExp);
						}
					}
					RHS = new ArrayAccess(extended_var.clone(), indexEx);
					ForLoop innerLoop = (ForLoop)TransformTools.genReductionLoop(index_vars, edimensions, LHS, RHS, redOp);
					
					//DEBUG: below code is more optimized, but not used for code re-usablity; 
					//this should be outlined as a separate function.
					/*					Identifier index_var = null;
					Statement loop_init = null;
					Expression condition = null;
					Expression step = null;
					CompoundStatement loop_body = null;
					ForLoop innerLoop = null;
					/////////////////////////////////////////////////////////////////////////////
					// Create or find temporary pointers that are used to pointer calculation. //
					/////////////////////////////////////////////////////////////////////////////
					List<Identifier> row_temps = null;
					row_temps = new ArrayList<Identifier>(dimsize+1);
					for( int i=0; i<dimsize; i++ ) {
						row_temps.add(TransformTools.getPointerTempIndex(cProc.getBody(), 
								typeSpecs, i));
					}
					row_temps.add((Identifier)extended_var.clone());
					////////////////////////////////////////////////////////////////////////////////
					// Insert codes for final reduction on the CPU.                               //
					////////////////////////////////////////////////////////////////////////////////
					////////////////////////////////////////////////////////////////////////////////
					// Ex: for(i=0; i<gpuNumBlocks; i++) {                                        //
					// 		row_temp1 = (float*)((char*)extred__x + i*SIZE1*SIZE2*sizeof(float)); //
					//         for(k=0; k<SIZE1; k++) {                                           //
					// 			row_temp0 = (float*)((char*)row_temp1 + k*SIZE2*sizeof(float));   //
					//             for(m=0; m<SIZE2; m++) {                                       //
					//                x[k][m] += row_temp0[m];                                    //
					//             }                                                              //
					//         }                                                                  //
					//      }                                                                     //
					////////////////////////////////////////////////////////////////////////////////
					// Create the nested loops.
					for( int i=0; i<=dimsize; i++ ) {
						index_var = index_vars.get(i);
						Expression assignex = new AssignmentExpression((Identifier)index_var.clone(),
								AssignmentOperator.NORMAL, new IntegerLiteral(0));
						loop_init = new ExpressionStatement(assignex);
						if( i<dimsize ) {
							condition = new BinaryExpression((Identifier)index_var.clone(),
									BinaryOperator.COMPARE_LT, 
									(Expression)lengthList.get(dimsize-1-i).clone());
						} else {
							condition = new BinaryExpression((Identifier)index_var.clone(),
									BinaryOperator.COMPARE_LT, (Identifier)numBlocks.clone());
						}
						step = new UnaryExpression(UnaryOperator.POST_INCREMENT, 
								(Identifier)index_var.clone());
						loop_body = new CompoundStatement();
						if( i==0  ) {
							if( dimsize == 0 ) {
								Expression hostSymExp = null;
								if( redSym instanceof AccessSymbol ) {
									hostSymExp = AnalysisTools.accessSymbolToExpression((AccessSymbol)redSym, null);
								} else {
									hostSymExp = new Identifier(redSym);
								}
								assignex = TransformTools.RedExpression(hostSymExp, 
										redOp, new ArrayAccess(
												(Identifier)row_temps.get(0).clone(), 
												(Identifier)index_var.clone())); 
							} else {
								List<Expression> indices = new LinkedList<Expression>();
								for( int k=dimsize-1; k>=0; k-- ) {
									indices.add((Expression)index_vars.get(k).clone());
								}
								Expression hostSymExp = null;
								if( redSym instanceof AccessSymbol ) {
									hostSymExp = AnalysisTools.accessSymbolToExpression((AccessSymbol)redSym, indices);
								} else {
									hostSymExp = new Identifier(redSym);
								}
								assignex = TransformTools.RedExpression(hostSymExp,
										redOp, new ArrayAccess(
												(Identifier)row_temps.get(0).clone(), 
												(Identifier)index_var.clone())); 
							}
						} else {
							castspecs = new ArrayList<Specifier>(2);
							castspecs.add(Specifier.CHAR);
							castspecs.add(PointerSpecifier.UNQUALIFIED);
							Typecast tcast1 = new Typecast(castspecs, (Identifier)row_temps.get(i).clone()); 
							BinaryExpression biexp1 = new BinaryExpression((Expression)sizeof_expr.clone(), 
									BinaryOperator.MULTIPLY, (Expression)lengthList.get(dimsize-1).clone());
							BinaryExpression biexp2 = null;
							for( int k=1; k<i; k++ ) {
								biexp2 = new BinaryExpression(biexp1, BinaryOperator.MULTIPLY,
										(Expression)lengthList.get(dimsize-1-k).clone());
								biexp1 = biexp2;
							}
							biexp2 = new BinaryExpression((Expression)index_var.clone(), 
									BinaryOperator.MULTIPLY, biexp1);
							biexp1 = new BinaryExpression(tcast1, BinaryOperator.ADD, biexp2);
							castspecs = new ArrayList<Specifier>();
							castspecs.addAll(typeSpecs);
							castspecs.add(PointerSpecifier.UNQUALIFIED);
							tcast1 = new Typecast(castspecs, biexp1);
							assignex = new AssignmentExpression((Identifier)row_temps.get(i-1).clone(),
									AssignmentOperator.NORMAL, tcast1);
						}
						loop_body.addStatement(new ExpressionStatement(assignex));
						if( innerLoop != null ) {
							loop_body.addStatement(innerLoop);
						}
						innerLoop = new ForLoop(loop_init, condition, step, loop_body);
					}*/

					
					Statement resetStatusCallStmt = null;
					if( memtrVerification ) {
						//Add "HI_reset_status(hostPtr, acc_device_nvidia, HI_stale, INT_MIN)" call.
						Expression hostVar = null;
						if( redSym instanceof AccessSymbol ) {
							hostVar = AnalysisTools.accessSymbolToExpression((AccessSymbol)redSym, null);
						} else {
							hostVar = new Identifier(redSym);
						}
						FunctionCall setStatusCall = new FunctionCall(new NameID("HI_reset_status"));
						if( lengthList.size() == 0 ) { //hostVar is scalar.
							setStatusCall.addArgument( new UnaryExpression(UnaryOperator.ADDRESS_OF, 
									hostVar.clone()));
						} else {
							setStatusCall.addArgument(hostVar.clone());
						}
						setStatusCall.addArgument(new NameID("acc_device_current"));
						setStatusCall.addArgument(new NameID("HI_stale"));
						//setStatusCall.addArgument(new NameID("INT_MIN"));
						setStatusCall.addArgument(new NameID("DEFAULT_QUEUE"));
						resetStatusCallStmt = new ExpressionStatement(setStatusCall);
					}
					
					//mallocScope.addStatementAfter(confRefStmt, innerLoop);
					if( asyncID == null ) {
						if( ifCond == null ) {
							if( resetStatusCallStmt != null ) {
								regionParent.addStatementAfter(region, resetStatusCallStmt);
							}
							regionParent.addStatementAfter(region, innerLoop);
						} else {
							//Put the resetStatusCallStmt later.
							//Put the innerLoop later.
						}
					} else {
						//Put the resetStatusCallStmt later.
						//Put the innerLoop later.
						//postRedStmts.add(0, innerLoop);
					}

					//PrintTools.println("[reductionTransformation] insert memory copy statement", 2);
					////////////////////////////////////////////////////////////////////
					// Insert memory copy function from GPU to CPU.                   //
					////////////////////////////////////////////////////////////////////
					// Ex: gpuBytes= gpuNumBlocks * (SIZE1 * SIZE2 * sizeof (float)); //
					//     cudaMemcpy(extred__x, ggred__x, gpuBytes,                  //
					//     cudaMemcpyDeviceToHost);                                   //
					////////////////////////////////////////////////////////////////////
					/////////////////////////////////////////////////////////////////////////////////////////////
					// HI_memcpy(extred__x, ggred__x, gpuBytes, HI_MemcpyDeviceToHost, 0);               //
					// HI_memcpy_async(extred__x, ggred__x, gpuBytes, HI_MemcpyDeviceToHost, 0, asyncID);//
					// DEBUG: async transfer will not be used, since this will be called after wait statement. //
					/////////////////////////////////////////////////////////////////////////////////////////////
					FunctionCall memCopy_call2 = null;
					memCopy_call2 = new FunctionCall(new NameID("HI_memcpy"));
/*					if( asyncID == null ) {
						memCopy_call2 = new FunctionCall(new NameID("HI_memcpy"));
					} else {
						memCopy_call2 = new FunctionCall(new NameID("HI_memcpy_async"));
					}*/
					List<Expression> arg_list3 = new ArrayList<Expression>();
					arg_list3.add((Identifier)extended_var.clone());
					arg_list3.add((Identifier)ggred_var.clone());
					arg_list3.add((Identifier)cloned_bytes.clone());
					arg_list3.add(new NameID("HI_MemcpyDeviceToHost"));
					arg_list3.add(new IntegerLiteral(0));
/*					if( asyncID != null ) {
						arg_list3.add(asyncID.clone());
					}*/
					memCopy_call2.setArguments(arg_list3);
					ExpressionStatement memCopy_stmt;
					memCopy_stmt = new ExpressionStatement(memCopy_call2);
					//mallocScope.addStatementAfter(confRefStmt, memCopy_stmt);
					//mallocScope.addStatementAfter(confRefStmt, gpuBytes_stmt.clone());
					/*				if( refSt == null ) {
					postscriptStmts.addStatement(gpuBytes_stmt.clone());
					postscriptStmts.addStatement(memCopy_stmt);
				} else {
					postscriptStmts.addStatementBefore(refSt, gpuBytes_stmt.clone());
					postscriptStmts.addStatementBefore(refSt, memCopy_stmt);
				}
					 */				
					if( asyncID == null ) {
						if( ifCond == null ) {
							regionParent.addStatementAfter(region, memCopy_stmt);
							regionParent.addStatementAfter(region, gpuBytes_stmt.clone());
						} else {
							CompoundStatement ifBody2 = new CompoundStatement();
							ifBody2.addStatement(gpuBytes_stmt.clone());
							ifBody2.addStatement(memCopy_stmt);
							ifBody2.addStatement(innerLoop);
							if( resetStatusCallStmt != null ) {
								ifBody2.addStatement(resetStatusCallStmt);
							}
							IfStatement ifStmt2 = new IfStatement(ifCond.clone(), ifBody2);
							regionParent.addStatementAfter(region, ifStmt2);
						}
					} else {
						if( kernelVerification && (resultCompareStmts != null) ) {
							for( int k=resultCompareStmts.size()-1; k>=0; k-- ) {
								postRedStmts.add(0, resultCompareStmts.get(k));
							}
						}
						if( resetStatusCallStmt != null ) {
							postRedStmts.add(0, resetStatusCallStmt);
						}
						postRedStmts.add(0, innerLoop);
						postRedStmts.add(0, memCopy_stmt);
						postRedStmts.add(0, gpuBytes_stmt.clone());
					}
				}
				gPostRedStmts.addAll(postRedStmts);

			} //end of sortedSet loop
			if( !gPostRedStmts.isEmpty() ) {
				AssignmentExpression assignExp = new AssignmentExpression(numBlocks.clone(), AssignmentOperator.NORMAL, totalnumgangs);
				gPostRedStmts.add(0, new ExpressionStatement(assignExp));
				if( ifCond == null ) {
					for( int k=gPostRedStmts.size()-1; k>=0; k-- ) {
						asyncConfRefPStmt.addStatementAfter(asyncConfRefStmt, gPostRedStmts.get(k));
					}
				} else {
					CompoundStatement asyncIfBody = new CompoundStatement();
					for( int k=0; k<gPostRedStmts.size(); k++ ) {
						asyncIfBody.addStatement(gPostRedStmts.get(k));
					}
					asyncConfRefPStmt.addStatementAfter(asyncConfRefStmt, new IfStatement(ifCond.clone(), asyncIfBody));
				}
			}
			///////////////////////////////////////////////////////////////////
			//Add in-block reduction code for worker-reductions.             //
			//This also assign the in-block reduction result back to gang    //
			//reduction variable if current region is a both gang and worker //
			//reduction loop. Otherwise, the assignment is done later.       //
			///////////////////////////////////////////////////////////////////
			//PrintTools.println("[reductionTransformation] insert in-block reduction codes", 2);
			if( isWorkerReduction ) {
				Statement lastS = InBlockReductionConv(tWorkerParamRedVarMap, redSubMapLocal, redOpMapLocal,tGangParamRedVarMap,
						region, (Statement)at, scope, postList, transNoRedUnrollSet, opt_UnrollOnReduction, num_workersExp, maxBlockSize,
						isGangReduction, IRSymbolOnly, warpSize);
				if( !isGangReduction ) {
					RegionToWorkerRedParamMap.put(scope, tWorkerParamRedVarMap);
					RegionToLastInBlockWorkerRedStmtMap.put(scope, lastS);
					RegionToEnclosingWorkerLoopMap.put(scope, (Statement)at);
				}
			}
			
		}
		
		//PrintTools.println("[reductionTransformation] store in-block reduction results", 2);
		//Assign the in-block reduction result back to gang reduction variable 
		//for worker-reduction loops not handled yet.
		Identifier tid = SymbolTools.getOrphanID("_tid");
		Identifier gtid = SymbolTools.getOrphanID("_gtid");
		Identifier bid = SymbolTools.getOrphanID("_bid");
		for( CompoundStatement wscope : RegionToWorkerRedParamMap.keySet() ) {
			Map<Symbol, Identifier> tWorkerParamRedVarMap = RegionToWorkerRedParamMap.get(wscope);
			Statement lastS = RegionToLastInBlockWorkerRedStmtMap.get(wscope);
			Statement ttWLoop = RegionToEnclosingWorkerLoopMap.get(wscope);
			CompoundStatement ifBody = new CompoundStatement();
			CompoundStatement workerIfBody = new CompoundStatement();
			Collection<Symbol> sortedSet = AnalysisTools.getSortedCollection(tWorkerParamRedVarMap.keySet());
			for( Symbol redSym : sortedSet ) {
				ReductionOperator redOp = redOpMap.get(redSym);
				SubArray sArray = redSubMap.get(redSym);
				Identifier lwred_var = tWorkerParamRedVarMap.get(redSym);
				Identifier lgred_var = tGangParamRedVarMap.get(redSym);
				int gpoffset = 1;
				boolean isPureWorkerReduction = false;
				if( lgred_var == null ) { //this wscope has pure worker reduction.
					isPureWorkerReduction = true;
					String symNameBase = null;
					if( redSym instanceof AccessSymbol) {
						symNameBase = TransformTools.buildAccessSymbolName((AccessSymbol)redSym);
					} else {
						symNameBase = redSym.getSymbolName();
					}
					String localGPSymName = "lgpriv__" + symNameBase;
					VariableDeclaration gangPriv_decl = 
							(VariableDeclaration)SymbolTools.findSymbol(wscope, localGPSymName);
					if( gangPriv_decl == null ) {
						//Check again to see if it is a gang-private variable.
						gangPriv_decl = 
								(VariableDeclaration)SymbolTools.findSymbol(new_proc, localGPSymName);
					}
					if( gangPriv_decl == null ) {
						//System.out.println("Pure worker reduction symbol is locally declared: " + redSym);
						//local variable declared within a compute region but outside of worker loop is 
						//gang-private, but in this case, variable name not changed.
						//[FIXME] This will not work for access symbol.
						localGPSymName = symNameBase;
						gangPriv_decl = 
							(VariableDeclaration)SymbolTools.findSymbol(wscope, localGPSymName);
					}
					VariableDeclarator gangPrivSym = null;
					if( gangPriv_decl != null ) {
						gangPrivSym = (VariableDeclarator)gangPriv_decl.getDeclarator(0);
						lgred_var = new Identifier(gangPrivSym);
						//System.out.println("Declaration of the pure worker reduction symbol: " + gangPriv_decl);
						//[DEBUG] Local variable declared within a compute region but outside of worker loop is gang-private and
						//will be cached on the shared memory later in the private transformation pass.
						//if( SymbolTools.containsSpecifier(gangPrivSym, CUDASpecifier.CUDA_SHARED) ) {
							gpoffset = 0; //gang-private is cached on the shared memory.
						//}
					} else {
						Tools.exit("[ERROR in CUDATranslationTools.reductionTransformation()] Gang-private variable for the " +
								"following worker-reduction variable is not visible: " + sArray.getArrayName() 
								+ "; the ACC2GPU translation failed!\n" + "Enclosing procedure: " + cProc.getSymbolName() + "\n");
					}
				}
				int offset = 0;
				if( lwred_var.getName().startsWith("lwredg_") ) {
					offset = 1;
				}
				List<Expression> startList = new LinkedList<Expression>();
				List<Expression> lengthList = new LinkedList<Expression>();
				boolean foundDimensions = AnalysisTools.extractDimensionInfo(sArray, startList, lengthList, IRSymbolOnly, wscope);
				if( !foundDimensions ) {
					Tools.exit("[ERROR in CUDATranslationTools.reductionTransformation()] Dimension information of the following " +
							"reduction variable is unknown: " + sArray.getArrayName() 
							+ "; the ACC2GPU translation failed!\n" + "Enclosing procedure: " + cProc.getSymbolName() + "\n");
				}
				int dimsize = lengthList.size();
				if( dimsize == 0 ) {
					Expression LHS = null;
					if( gpoffset == 1 ) {
						LHS = new ArrayAccess(lgred_var.clone(), bid.clone());
					} else {
						LHS = lgred_var.clone();
					}
					Identifier ttid = tid.clone();
					if( offset == 1 ) {
						ttid = gtid.clone();
					}
					Expression assignex = new AssignmentExpression( LHS, AssignmentOperator.NORMAL, 
							new ArrayAccess(lwred_var.clone(), ttid.clone()));
					if( isPureWorkerReduction ) {
						workerIfBody.addStatement(new ExpressionStatement(assignex));
					} else {
						ifBody.addStatement(new ExpressionStatement(assignex));
					}
				} else {
					//////////////////////////////////////// //////
					// Create or find temporary index variables. // 
					//////////////////////////////////////// //////
					List<Identifier> index_vars = new LinkedList<Identifier>();
					for( int i=0; i<dimsize; i++ ) {
						index_vars.add(TransformTools.getTempIndex(wscope, tempIndexBase+i));
					}
					List<Expression> indices1 = new LinkedList<Expression>();
					List<Expression> indices2 = new LinkedList<Expression>();
					if( gpoffset == 1 ) {
						indices1.add((Identifier)bid.clone());
					}
					if( offset == 1 ) {
						indices2.add((Identifier)gtid.clone());
					}
					for( int k=0; k<dimsize; k++ ) {
						indices1.add((Expression)index_vars.get(k).clone());
						indices2.add((Expression)index_vars.get(k).clone());
					}
					if( offset == 0 ) {
						indices2.add(tid.clone());
					}
					Expression LHS = new ArrayAccess(lgred_var.clone(), indices1);
					Expression RHS = new ArrayAccess(lwred_var.clone(), indices2);
					Statement redLoop = TransformTools.genReductionLoop(index_vars, lengthList, LHS, RHS, redOp);
					if( isPureWorkerReduction ) {
						workerIfBody.addStatement(redLoop);
					} else {
						ifBody.addStatement(redLoop);
					}
				}
			}
			Expression condition = new BinaryExpression(tid.clone(), BinaryOperator.COMPARE_EQ,
					new IntegerLiteral(0));
			condition.setParens(false);
			//[DEBUG] If this is pure worker loop, the assignment statement should be put at the end of
			//the worker loop.
			if( !workerIfBody.getChildren().isEmpty() ) {
				IfStatement nIfStmt = new IfStatement(condition.clone(), workerIfBody);
				//wscope.addStatement( nIfStmt );
				CompoundStatement ttWLoopP = (CompoundStatement)ttWLoop.getParent();
				ttWLoopP.addStatementAfter(lastS, nIfStmt);
			}
			if( !ifBody.getChildren().isEmpty() ) {
				IfStatement nIfStmt = new IfStatement(condition.clone(), ifBody);
				//wscope.addStatement( nIfStmt );
				if( region instanceof ForLoop ) {
					postList.add(nIfStmt);
				} else if (region instanceof CompoundStatement) {
					((CompoundStatement)region).addStatement(nIfStmt);
				}
			}
		}
		PrintTools.println("[reductionTransformation() ends] current procedure: " + cProc.getSymbolName() +
				"\ncompute region type: " + cRegionKind + "\n", 2);
	}
	
	
	private static Statement InBlockReductionConv(Map<Symbol, Identifier> tWorkerParamRedVarMap, 
			Map<Symbol, SubArray> redSubMap, Map<Symbol, ReductionOperator> redOpMap,
			Map<Symbol, Identifier> tGangParamRedVarMap,
			Statement region, Statement at,
			CompoundStatement scope, List<Statement> postList, Set<Symbol> transNoRedUnrollSet,
			boolean opt_UnrollingOnReduction, Expression num_workersExp, int maxBlockSize,
			boolean isGangReduction, boolean IRSymbolOnly, int warpSize)  {
		ArrayList<Symbol> redArgList1 = new ArrayList<Symbol>();
		ArrayList<Symbol> redArgList2 = new ArrayList<Symbol>();
		ReductionOperator redOp = null;
		SubArray sArray = null;
		int num_workers = 0;
		Expression assignex = null;
		Statement loop_init = null;
		Expression condition = null;
		Expression step = null;
		CompoundStatement loop_body = null;
		Identifier tid = SymbolTools.getOrphanID("_tid");
		Identifier gtid = SymbolTools.getOrphanID("_gtid");
		Identifier bid = SymbolTools.getOrphanID("_bid");
		Identifier bsize = SymbolTools.getOrphanID("_bsize");
		
		if( num_workersExp instanceof IntegerLiteral ) {
			num_workers = (int)((IntegerLiteral)num_workersExp).getValue();
		}
		
		boolean usePostList = false;
		CompoundStatement atP = null;
		Statement lastS = null;
		if( at == region ) {
			usePostList = true;
		} else {
			atP = (CompoundStatement)at.getParent();
			lastS = at;
		}
		
		/////////////////////////////////////////////////////////////////////
		// Add in-block reduction codes at the end of the parallel region. //
		/////////////////////////////////////////////////////////////////////
		Statement ifstmt = null;
		CompoundStatement ifBody = null;
		FunctionCall syncCall = new FunctionCall(new NameID("__syncthreads"));
		Statement syncCallStmt = new ExpressionStatement(syncCall);
		//scope.addStatement(syncCallStmt);
		if( usePostList ) {
			postList.add(syncCallStmt);
		} else {
			atP.addStatementAfter(lastS, syncCallStmt);
			lastS = syncCallStmt;
		}
		
		Collection<Symbol> sortedSet = AnalysisTools.getSortedCollection(tWorkerParamRedVarMap.keySet());
		if( opt_UnrollingOnReduction && (num_workers > 0) ) { //unrolling is applicable only if num_workers is compile-time constant.
			for( Symbol redSym : sortedSet ) {
				if( transNoRedUnrollSet.contains(redSym) ) {
					redArgList2.add(redSym);
				} else {
					redArgList1.add(redSym);
				}
			}
		} else {
			redArgList2.addAll(sortedSet);
			
		}
		if( redArgList1.size() > 0 ) {
			///////////////////////////////////////////////////
			// Version1:  reduction with loop unrolling code //
			// this version works only if BLOCK_SIZE = 2^m   //
			///////////////////////////////////////////////////
			// Case1:  reduction variable is scalar.         //
			// (assume warpSize == 32.)                      //
			///////////////////////////////////////////////////
			// Assume that BLOCK_SIZE = 512.
		    //if (_tid < 256) {
		    //     lwred__x[_tid] += lwred__x[_tid + 256];
		    // }    
		    // __syncthreads();
		    //if (_tid < 128) {
		    //     lwred__x[_tid] += lwred__x[_tid + 128];
		    // }    
		    // __syncthreads();
		    // if (_tid < 64) {
		    //     lwred__x[_tid] += lwred__x[_tid + 64]; 
		    // }    
		    // __syncthreads();
		    // if (_tid < 32)
		    // {    
		    //     lwred__x[_tid] += lwred__x[_tid + 32]; 
		    // }    
		    // if (_tid < 16)
		    // {    
		    //     lwred__x[_tid] += lwred__x[_tid + 16]; 
		    // }    
		    // if (_tid < 8)
		    // {    
		    //     lwred__x[_tid] += lwred__x[_tid + 8];
		    // }    
		    // if (_tid < 4)
		    // {    
		    //     lwred__x[_tid] += lwred__x[_tid + 4];
		    // }    
		    // if (_tid < 2)
		    // {    
		    //     lwred__x[_tid] += lwred__x[_tid + 2];
		    // }    
		    // if (_tid < 1)
		    // {    
		    //     lwred__x[_tid] += lwred__x[_tid + 1];
		    // }    
			///////////////////////////////////////////////////
			// Case2:  reduction variable is array.          //
			// (assume warpSize == 32.)                      //
			///////////////////////////////////////////////////
			// Assume that BLOCK_SIZE = 512.
		    //if (_tid < 256) {
			//    for (i=0; i<SIZE1; i++) {
			//        for (k=0; k<SIZE2; k++) {
			//            lwred__x[i][k][_tid] += lwred__x[i][k][_tid + 256];
			//        }
			//    }
		    // }    
		    // __syncthreads();
		    //if (_tid < 128) {
			//    for (i=0; i<SIZE1; i++) {
			//        for (k=0; k<SIZE2; k++) {
			//            lwred__x[i][k][_tid] += lwred__x[i][k][_tid + 128];
			//        }
			//    }
		    // }    
		    // __syncthreads();
		    // if (_tid < 64) {
			//    for (i=0; i<SIZE1; i++) {
			//        for (k=0; k<SIZE2; k++) {
			//            lwred__x[i][k][_tid] += lwred__x[i][k][_tid + 64];
			//        }
			//    }
		    // }    
		    // __syncthreads();
		    // if (_tid < 32)
		    // {    
			//    for (i=0; i<SIZE1; i++) {
			//        for (k=0; k<SIZE2; k++) {
			//            lwred__x[i][k][_tid] += lwred__x[i][k][_tid + 32];
			//        }
			//    }
		    // }    
		    // if (_tid < 16)
		    // {    
			//    for (i=0; i<SIZE1; i++) {
			//        for (k=0; k<SIZE2; k++) {
			//            lwred__x[i][k][_tid] += lwred__x[i][k][_tid + 16];
			//        }
			//    }
		    // }    
		    // if (_tid < 8)
		    // {    
			//    for (i=0; i<SIZE1; i++) {
			//        for (k=0; k<SIZE2; k++) {
			//            lwred__x[i][k][_tid] += lwred__x[i][k][_tid + 8];
			//        }
			//    }
		    // }    
		    // if (_tid < 4)
		    // {    
			//    for (i=0; i<SIZE1; i++) {
			//        for (k=0; k<SIZE2; k++) {
			//            lwred__x[i][k][_tid] += lwred__x[i][k][_tid + 4];
			//        }
			//    }
		    // }    
		    // if (_tid < 2)
		    // {    
			//    for (i=0; i<SIZE1; i++) {
			//        for (k=0; k<SIZE2; k++) {
			//            lwred__x[i][k][_tid] += lwred__x[i][k][_tid + 2];
			//        }
			//    }
		    // }    
		    // if (_tid < 1)
		    // {    
			//    for (i=0; i<SIZE1; i++) {
			//        for (k=0; k<SIZE2; k++) {
			//            lwred__x[i][k][_tid] += lwred__x[i][k][_tid + 1];
			//        }
			//    }
		    // }    
			///////////////////////////////////////////////////
			int halfBlockSize = maxBlockSize/2;
			for( int _bsize_ = halfBlockSize; _bsize_ > 0; _bsize_>>=1 ) {
				if( num_workers >= 2*_bsize_ ) {
					ifBody = new CompoundStatement();
					for( Symbol redSym : redArgList1 ) {
						redOp = redOpMap.get(redSym);
						sArray = redSubMap.get(redSym);
						Identifier lwred_var = tWorkerParamRedVarMap.get(redSym);
						int offset = 0;
						if( lwred_var.getName().startsWith("lwredg_") ) {
							offset = 1;
						}
						List<Expression> startList = new LinkedList<Expression>();
						List<Expression> lengthList = new LinkedList<Expression>();
						boolean foundDimensions = AnalysisTools.extractDimensionInfo(sArray, startList, lengthList, IRSymbolOnly, at);
						if( !foundDimensions ) {
							Tools.exit("[ERROR in CUDATranslationTools.reductionTransformation()] Dimension information of the following " +
									"reduction variable is unknown: " + sArray.getArrayName() + "; the ACC2GPU translation failed!");
						}
						int dimsize = lengthList.size();
						if( dimsize == 0 ) {
							Identifier ttid = tid.clone();
							if( offset == 1 ) {
								ttid = gtid.clone();
							}
							assignex = TransformTools.RedExpression(new ArrayAccess((Identifier)lwred_var.clone(), 
									ttid.clone()) , redOp, 
									new ArrayAccess((Identifier)lwred_var.clone(), 
											new BinaryExpression(tid.clone(), BinaryOperator.ADD,
													new IntegerLiteral(_bsize_))));
							ifBody.addStatement(new ExpressionStatement(assignex));
						} else {
							//////////////////////////////////////// //////
							// Create or find temporary index variables. // 
							//////////////////////////////////////// //////
							List<Identifier> index_vars = new LinkedList<Identifier>();
							for( int i=0; i<dimsize; i++ ) {
								if( usePostList ) {
									index_vars.add(TransformTools.getTempIndex(scope, tempIndexBase+i));
								} else {
									index_vars.add(TransformTools.getTempIndex(atP, tempIndexBase+i));
								}
							}
							List<Expression> indices1 = new LinkedList<Expression>();
							List<Expression> indices2 = new LinkedList<Expression>();
							if( offset == 1 ) {
								indices1.add((Identifier)gtid.clone());
								indices2.add(new BinaryExpression((Identifier)gtid.clone(),
										BinaryOperator.ADD, new IntegerLiteral(_bsize_)));
							}
							for( int k=0; k<dimsize; k++ ) {
								indices1.add((Expression)index_vars.get(k).clone());
								indices2.add((Expression)index_vars.get(k).clone());
							}
							if( offset == 0 ) {
								indices1.add(tid.clone());
								indices2.add(new BinaryExpression(tid.clone(),
										BinaryOperator.ADD, new IntegerLiteral(_bsize_)));
							}
							Expression LHS = new ArrayAccess(lwred_var.clone(), indices1);
							Expression RHS = new ArrayAccess(lwred_var.clone(), indices2);
							Statement redLoop = TransformTools.genReductionLoop(index_vars, lengthList, LHS, RHS, redOp);
							ifBody.addStatement(redLoop);
						}
					}
					condition = new BinaryExpression(tid.clone(), 
							BinaryOperator.COMPARE_LT, new IntegerLiteral(_bsize_) );
					condition.setParens(false);
					ifstmt = new IfStatement(condition, ifBody);
					//scope.addStatement(ifstmt);
					if( usePostList ) {
						postList.add(ifstmt);
					} else {
						atP.addStatementAfter(lastS, ifstmt);
						lastS = ifstmt;
					}
					if( _bsize_ > warpSize ) {
						//scope.addStatement((Statement)syncCallStmt.clone());
						if( usePostList ) {
							postList.add(syncCallStmt.clone());
						} else {
							Statement tStmt = syncCallStmt.clone();
							atP.addStatementAfter(lastS, tStmt);
							lastS = tStmt;
						}
					}
				}
			}
		} 
		if( redArgList2.size() > 0 ) {
			/////////////////////////////////////////////////////////////////////
			// Version2: Unoptimized reduction code                            //
			/////////////////////////////////////////////////////////////////////
			// Case1: reduction variable is scalar.                            //
			/////////////////////////////////////////////////////////////////////
			//     _bsize = blockDim.x * blockDim.y * blockDim.z;
			//     _tid = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDimx*blockDim.y;
			//     oldSize = _bsize;
			//     for (s=(_bsize>>1); s>0; s>>=1) {
			//         if(_tid < s) { 
			//             lwreds__x[_tid] += lwreds__x[_tid + s];
			//             lwreds__y[_gtid] += lwredg__y[_gtid + s];
			//         }    
			//         oddNum = oldSize & (0x01);
			//         if ( oddNum == 1 ) {
			//             if (_tid == 0) { 
			//                 lwreds__x[_tid] += lwreds__x[_tid + oldSize-1];
			//                 lwredg__y[_gtid] += lwredg__y[_gtid + oldSize-1];
			//             }    
			//         }    
			//         oldSize = s; 
			//         if( s > WARPSIZE ) {
			//         		__syncthreads();
			//         }
			//     }    
			/////////////////////////////////////////////////////////////////////
			// Case2: reduction variable is an array.                          //
			/////////////////////////////////////////////////////////////////////
			//     _bsize = blockDim.x * blockDim.y * blockDim.z;
			//     _tid = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDimx*blockDim.y;
			//     oldSize = _bsize;
			//     for (s=(_bsize>>1); s>0; s>>=1) {
			//         if(_tid < s) { 
			//             for (i=0; i<SIZE1; i++) {
			//                 for (k=0; k<SIZE2; k++) {
			//                     lwreds__x[i][k][_tid] += lwreds__x[i][k][_tid + s];
			//                     lwredg__y[_gtid][i][k] += lwreds__y[_gtid + s][i][k];
			//                 }
			//             }
			//         }    
			//         oddNum = oldSize & (0x01);
			//         if ( oddNum == 1 ) {
			//             if (_tid == 0) { 
			//                 for (i=0; i<SIZE1; i++) {
			//                     for (k=0; k<SIZE2; k++) {
			//                         lwreds__x[i][k][_tid] += lwreds__x[i][k][_tid + oldSize-1];
			//                         lwredg__y[_gtid][i][k] += lwredg__y[_gtid + oldSize-1][i][k];
			//                     }
			//                 }
			//             }    
			//         }    
			//         oldSize = s; 
			//         if( s > WARPSIZE ) {
			//         		__syncthreads();
			//         }
			//     }    
			/////////////////////////////////////////////////////////////////////
			// Find the max value of dimensions of reduction variables.
			int maxdimsize = 0;
			for( Symbol redSym : redArgList2 ) {
				Identifier lwred_var = tWorkerParamRedVarMap.get(redSym);
				List aspecs = lwred_var.getSymbol().getArraySpecifiers();
				ArraySpecifier aspec = (ArraySpecifier)aspecs.get(0);
				int dimsize = aspec.getNumDimensions();
				if( dimsize > maxdimsize ) {
					maxdimsize = dimsize;
				}
			}
			maxdimsize += 1;
			Identifier index_var2;
			Identifier oldSize;
			Identifier oddNum;
			if( usePostList ) {
				index_var2 = TransformTools.getTempIndex(scope, tempIndexBase+maxdimsize-1);
				oldSize = TransformTools.getTempIndex(scope, tempIndexBase+maxdimsize);
				oddNum = TransformTools.getTempIndex(scope, tempIndexBase+maxdimsize+1);
			} else {
				index_var2 = TransformTools.getTempIndex(atP, tempIndexBase+maxdimsize-1);
				oldSize = TransformTools.getTempIndex(atP, tempIndexBase+maxdimsize);
				oddNum = TransformTools.getTempIndex(atP, tempIndexBase+maxdimsize+1);
			}
			Statement estmt = new ExpressionStatement(
					new AssignmentExpression((Expression)oldSize.clone(), 
							AssignmentOperator.NORMAL, bsize.clone()));
			//scope.addStatement(estmt);
			if( usePostList ) {
				postList.add(estmt);
			} else {
				atP.addStatementAfter(lastS, estmt);
				lastS = estmt;
			}
			assignex = new AssignmentExpression((Identifier)index_var2.clone(),
					AssignmentOperator.NORMAL, new BinaryExpression(bsize.clone(),
							BinaryOperator.SHIFT_RIGHT, new IntegerLiteral(1)));
			loop_init = new ExpressionStatement(assignex);
			condition = new BinaryExpression((Identifier)index_var2.clone(),
					BinaryOperator.COMPARE_GT, new IntegerLiteral(0));
			step = new AssignmentExpression( (Identifier)index_var2.clone(),
					AssignmentOperator.SHIFT_RIGHT, new IntegerLiteral(1));
			CompoundStatement loopbody = new CompoundStatement();
			ForLoop reductionLoop = new ForLoop(loop_init, condition, step, loopbody);
			ifBody = new CompoundStatement();
			for( Symbol redSym : redArgList2 ) {
				redOp = redOpMap.get(redSym);
				sArray = redSubMap.get(redSym);
				Identifier lwred_var = tWorkerParamRedVarMap.get(redSym);
				int offset = 0;
				if( lwred_var.getName().startsWith("lwredg_") ) {
					offset = 1;
				}
				List<Expression> startList = new LinkedList<Expression>();
				List<Expression> lengthList = new LinkedList<Expression>();
				boolean foundDimensions = AnalysisTools.extractDimensionInfo(sArray, startList, lengthList, IRSymbolOnly, at);
				if( !foundDimensions ) {
					Tools.exit("[ERROR in CUDATranslationTools.reductionTransformation()] Dimension information of the following " +
							"reduction variable is unknown: " + sArray.getArrayName() + "; the ACC2GPU translation failed!");
				}
				int dimsize = lengthList.size();
				if( dimsize == 0 ) {
					Identifier ttid = tid.clone();
					if( offset == 1 ) {
						ttid = gtid.clone();
					}
					assignex = TransformTools.RedExpression(new ArrayAccess(lwred_var.clone(), 
							ttid.clone()) , redOp, 
							new ArrayAccess(lwred_var.clone(), 
									new BinaryExpression(ttid.clone(), BinaryOperator.ADD,
											(Identifier)index_var2.clone())));
					ifBody.addStatement(new ExpressionStatement(assignex));
				} else {
					//////////////////////////////////////// //////
					// Create or find temporary index variables. // 
					//////////////////////////////////////// //////
					List<Identifier> index_vars = new LinkedList<Identifier>();
					for( int i=0; i<dimsize; i++ ) {
						if( usePostList ) {
							index_vars.add(TransformTools.getTempIndex(scope, tempIndexBase+i));
						} else {
							index_vars.add(TransformTools.getTempIndex(atP, tempIndexBase+i));
						}
					}
					List<Expression> indices1 = new LinkedList<Expression>();
					List<Expression> indices2 = new LinkedList<Expression>();
					if( offset == 1 ) {
						indices1.add((Identifier)gtid.clone());
						indices2.add(new BinaryExpression((Identifier)gtid.clone(),
								BinaryOperator.ADD, index_var2.clone()));
					}
					for( int k=0; k<dimsize; k++ ) {
						indices1.add((Expression)index_vars.get(k).clone());
						indices2.add((Expression)index_vars.get(k).clone());
					}
					if( offset == 0 ) {
						indices1.add(tid.clone());
						indices2.add(new BinaryExpression(tid.clone(),
								BinaryOperator.ADD, index_var2.clone()));
					}
					Expression LHS = new ArrayAccess(lwred_var.clone(), indices1);
					Expression RHS = new ArrayAccess(lwred_var.clone(), indices2);
					Statement redLoop = TransformTools.genReductionLoop(index_vars, lengthList, LHS, RHS, redOp);
					ifBody.addStatement(redLoop);
				}
			}
			condition = new BinaryExpression(tid.clone(), BinaryOperator.COMPARE_LT,
					(Identifier)index_var2.clone());
			condition.setParens(false);
			loopbody.addStatement( new IfStatement(condition, ifBody) );
			assignex = new AssignmentExpression( (Identifier)oddNum.clone(), AssignmentOperator.NORMAL,
					new BinaryExpression((Identifier)oldSize.clone(), BinaryOperator.BITWISE_AND,
							new IntegerLiteral(0x01)));
			loopbody.addStatement(new ExpressionStatement(assignex));
			ifBody = new CompoundStatement();
			for( Symbol redSym : redArgList2 ) {
				redOp = redOpMap.get(redSym);
				sArray = redSubMap.get(redSym);
				Identifier lwred_var = tWorkerParamRedVarMap.get(redSym);
				int offset = 0;
				if( lwred_var.getName().startsWith("lwredg_") ) {
					offset = 1;
				}
				List<Expression> startList = new LinkedList<Expression>();
				List<Expression> lengthList = new LinkedList<Expression>();
				boolean foundDimensions = AnalysisTools.extractDimensionInfo(sArray, startList, lengthList, IRSymbolOnly, at);
				if( !foundDimensions ) {
					Tools.exit("[ERROR in CUDATranslationTools.reductionTransformation()] Dimension information of the following " +
							"reduction variable is unknown: " + sArray.getArrayName() + "; the ACC2GPU translation failed!");
				}
				int dimsize = lengthList.size();
				if( dimsize == 0 ) {
					Identifier ttid = tid.clone();
					if( offset == 1 ) {
						ttid = gtid.clone();
					}
					assignex = TransformTools.RedExpression( new ArrayAccess(lwred_var.clone(),
							ttid.clone()), redOp, 
							new ArrayAccess(lwred_var.clone(), 
									new BinaryExpression( ttid.clone(), BinaryOperator.ADD, new BinaryExpression( (Identifier)oldSize.clone(), BinaryOperator.SUBTRACT,
											new IntegerLiteral(1)))));
					ifBody.addStatement(new ExpressionStatement(assignex));
				} else {
					//////////////////////////////////////// //////
					// Create or find temporary index variables. // 
					//////////////////////////////////////// //////
					List<Identifier> index_vars = new LinkedList<Identifier>();
					for( int i=0; i<dimsize; i++ ) {
						if( usePostList ) {
							index_vars.add(TransformTools.getTempIndex(scope, tempIndexBase+i));
						} else {
							index_vars.add(TransformTools.getTempIndex(atP, tempIndexBase+i));
						}
					}
					List<Expression> indices1 = new LinkedList<Expression>();
					List<Expression> indices2 = new LinkedList<Expression>();
					if( offset == 1 ) {
						indices1.add((Identifier)gtid.clone());
						indices2.add(new BinaryExpression( gtid.clone(), BinaryOperator.ADD, 
								new BinaryExpression(oldSize.clone(), BinaryOperator.SUBTRACT, new IntegerLiteral(1))));
					}
					for( int k=0; k<dimsize; k++ ) {
						indices1.add((Expression)index_vars.get(k).clone());
						indices2.add((Expression)index_vars.get(k).clone());
					}
					if( offset == 0 ) {
						indices1.add(tid.clone());
						indices2.add(new BinaryExpression(tid.clone(), BinaryOperator.ADD, 
								new BinaryExpression(oldSize.clone(), BinaryOperator.SUBTRACT, new IntegerLiteral(1))));
					}
					Expression LHS = new ArrayAccess(lwred_var.clone(), indices1);
					Expression RHS = new ArrayAccess(lwred_var.clone(), indices2);
					Statement redLoop = TransformTools.genReductionLoop(index_vars, lengthList, LHS, RHS, redOp);
					ifBody.addStatement(redLoop);
				}
			}
			condition = new BinaryExpression(tid.clone(), BinaryOperator.COMPARE_EQ,
					new IntegerLiteral(0));
			condition.setParens(false);
			ifstmt = new IfStatement(condition, ifBody);
			condition = new BinaryExpression( (Identifier)oddNum.clone(), BinaryOperator.COMPARE_EQ,
					new IntegerLiteral(1));
			condition.setParens(false);
			loopbody.addStatement( new IfStatement(condition, ifstmt) );
			estmt = new ExpressionStatement(
					new AssignmentExpression((Expression)oldSize.clone(), 
							AssignmentOperator.NORMAL, (Identifier)index_var2.clone()));
			loopbody.addStatement( estmt );
			//loopbody.addStatement((Statement)syncCallStmt.clone());
			condition =	new BinaryExpression(index_var2.clone(), BinaryOperator.COMPARE_GT, new IntegerLiteral(warpSize));
			condition.setParens(false);
			Statement condStmt = new IfStatement(condition, syncCallStmt.clone());
			loopbody.addStatement(condStmt);
			//scope.addStatement(reductionLoop);
			if( usePostList ) {
				postList.add(reductionLoop);
			} else {
				atP.addStatementAfter(lastS, reductionLoop);
				lastS = reductionLoop;
			}
		}
		//////////////////////////////////////////////////////////////////////////
		// Write the in-block reduction result back to the gang reduction array //
		// if current loop is both gang and worker reduction loop.              //
		//////////////////////////////////////////////////////////////////////////
		// Case1: Reduction variable is scalar.                                 //
		//////////////////////////////////////////////////////////////////////////
		//     if( _tid == 0 ) {
		//         lgred__x[_bid] = lwreds__x[_tid];
		//         lgred__y[_bid] = lwredg__y[_gtid];
		//     }
		//////////////////////////////////////////////////////////////////////////
		// Case1: Reduction variable is an array.                               //
		//////////////////////////////////////////////////////////////////////////
		//     if( _tid == 0 ) {
		//         for( i=0; i<SIZE1; i++ ) {
		//             for( k=0; k<SIZE2; k++ ) {
		//                 lgred__x[_bid][i][k] = lwreds__x[i][k][_tid];
		//                 lgred__y[_bid][i][k] = lwredg__y[_gtid][i][k];
		//             }
		//         }
		//     }
		//////////////////////////////////////////////////////////////////////////
		if( isGangReduction && sortedSet.size() > 0 ) {
			ifBody = new CompoundStatement();
			for( Symbol redSym : sortedSet ) {
				redOp = redOpMap.get(redSym);
				sArray = redSubMap.get(redSym);
				Identifier lwred_var = tWorkerParamRedVarMap.get(redSym);
				Identifier lgred_var = tGangParamRedVarMap.get(redSym);
				int offset = 0;
				if( lwred_var.getName().startsWith("lwredg_") ) {
					offset = 1;
				}
				List<Expression> startList = new LinkedList<Expression>();
				List<Expression> lengthList = new LinkedList<Expression>();
				boolean foundDimensions = AnalysisTools.extractDimensionInfo(sArray, startList, lengthList, IRSymbolOnly, at);
				if( !foundDimensions ) {
					Tools.exit("[ERROR in CUDATranslationTools.reductionTransformation()] Dimension information of the following " +
							"reduction variable is unknown: " + sArray.getArrayName() + "; the ACC2GPU translation failed!");
				}
				int dimsize = lengthList.size();
				if( dimsize == 0 ) {
					Identifier ttid = tid.clone();
					if( offset == 1 ) {
						ttid = gtid.clone();
					}
					assignex = new AssignmentExpression( new ArrayAccess(lgred_var.clone(),
							(Identifier)bid.clone()), AssignmentOperator.NORMAL, 
							new ArrayAccess(lwred_var.clone(), ttid.clone()));
					ifBody.addStatement(new ExpressionStatement(assignex));
				} else {
					//////////////////////////////////////// //////
					// Create or find temporary index variables. // 
					//////////////////////////////////////// //////
					List<Identifier> index_vars = new LinkedList<Identifier>();
					for( int i=0; i<dimsize; i++ ) {
						if( usePostList ) {
							index_vars.add(TransformTools.getTempIndex(scope, tempIndexBase+i));
						} else {
							index_vars.add(TransformTools.getTempIndex(atP, tempIndexBase+i));
						}
					}
					List<Expression> indices1 = new LinkedList<Expression>();
					List<Expression> indices2 = new LinkedList<Expression>();
					indices1.add((Identifier)bid.clone());
					if( offset == 1 ) {
						indices2.add((Identifier)gtid.clone());
					}
					for( int k=0; k<dimsize; k++ ) {
						indices1.add((Expression)index_vars.get(k).clone());
						indices2.add((Expression)index_vars.get(k).clone());
					}
					if( offset == 0 ) {
						indices2.add(tid.clone());
					}
					Expression LHS = new ArrayAccess(lgred_var.clone(), indices1);
					Expression RHS = new ArrayAccess(lwred_var.clone(), indices2);
					//Statement redLoop = TransformTools.genReductionLoop(index_vars, lengthList, LHS, RHS, redOp);
					Statement redLoop = TransformTools.genArrayCopyLoop(index_vars, lengthList, LHS, RHS);
					ifBody.addStatement(redLoop);
				}
			}
			condition = new BinaryExpression(tid.clone(), BinaryOperator.COMPARE_EQ,
					new IntegerLiteral(0));
			condition.setParens(false);
			IfStatement nIfStmt = new IfStatement(condition, ifBody);
			//scope.addStatement( nIfStmt );
			if( usePostList ) {
				postList.add(nIfStmt);
			} else {
				atP.addStatementAfter(lastS, nIfStmt);
				lastS = nIfStmt;
			}
		}
		return lastS;
	}
	
	/**
	 * Convert the access of a scalar shared variable into a pointer access expression.
	 * @param targetSym symbol of a target OpenMP shared variable
	 * @param new_proc a new function where the target symbol will be accessed 
	 * @param region original code region, which will be transformed into the new function, new_proc
	 * @return
	 */
	protected static VariableDeclarator scalarSharedConv(Symbol sharedSym, String symNameBase, List<Specifier> typeSpecs,
			Symbol gpuSym, Statement region, Procedure new_proc, FunctionCall call_to_new_proc, boolean useRegister, 
			boolean useSharedMemory, boolean ROData, boolean isSingleTask, List<Statement> preList, List<Statement> postList) {
		// Create a parameter Declaration for the kernel function
		// Change the scalar variable to a pointer type 
		VariableDeclarator kParam_declarator = new VariableDeclarator(PointerSpecifier.RESTRICT, 
				new NameID(symNameBase));
		VariableDeclaration kParam_decl = new VariableDeclaration(typeSpecs,
				kParam_declarator);
		Identifier kParamVar = new Identifier(kParam_declarator);
		new_proc.addDeclaration(kParam_decl);

		// Insert argument to the kernel function call
		call_to_new_proc.addArgument(new Identifier(gpuSym));

		CompoundStatement targetStmt = null;
		if( region instanceof CompoundStatement ) {
			targetStmt = (CompoundStatement)region;
		} else if( region instanceof ForLoop ) {
			targetStmt = (CompoundStatement)((ForLoop)region).getBody();
		} else {
			Tools.exit("[ERROR] Unknown region in extractKernelRegion(): "
					+ region.toString());
		}
		if( useSharedMemory || useRegister ) {
			Identifier local_var = null;
			if( useSharedMemory ) {
				// Create a temp variable on shared memory.
				// Ex: "__shared__ float b"
				StringBuilder str = new StringBuilder(80);
				str.append("sh__");
				str.append(symNameBase);
				VariableDeclarator sharedV_declarator = new VariableDeclarator(new NameID(str.toString()));
				List<Specifier> clonedspecs2 = new ChainedList<Specifier>();
				/////////////////////////////////////////////////////////////////////////////////////
				// CAUTION: VariableDeclarator.getTypeSpecifiers() returns both specifiers of      //
				// its parent VariableDeclaration and the VariableDeclarator's leading specifiers. //
				// Therefore, if VariableDeclarator is a pointer symbol, this method will return   //
				// pointer specifiers too.                                                         //
				/////////////////////////////////////////////////////////////////////////////////////
				clonedspecs2.add(CUDASpecifier.CUDA_SHARED);
				clonedspecs2.addAll(typeSpecs);
				VariableDeclaration sharedV_decl = 
					new VariableDeclaration(clonedspecs2, sharedV_declarator); 
				local_var = new Identifier(sharedV_declarator);
				targetStmt.addDeclaration(sharedV_decl);

			} else if( useRegister ) {
				// SymbolTools.getTemp() inserts the new temp symbol to the symbol table of the closest parent
				// if region is a loop.
				local_var = SymbolTools.getTemp(targetStmt, typeSpecs, symNameBase);
			}
			/////////////////////////////////////////////////////////////////////////////////////////
			// Insert a statement to load the global variable to a local variable at the beginning //
			//and a statement to dump the local variable to the global variable at the end.        //
			/////////////////////////////////////////////////////////////////////////////////////////
			Statement estmt = new ExpressionStatement(new AssignmentExpression(local_var, 
					AssignmentOperator.NORMAL, 
					new UnaryExpression(UnaryOperator.DEREFERENCE, (Identifier)kParamVar.clone())));
			Statement astmt = new ExpressionStatement(new AssignmentExpression( 
					new UnaryExpression(UnaryOperator.DEREFERENCE, (Identifier)kParamVar.clone()),
					AssignmentOperator.NORMAL,(Identifier)local_var.clone()));
			// Replace all instances of the shared variable to the local variable
			//TransformTools.replaceAll((Traversable) targetStmt, cloned_ID, local_var);
			if( sharedSym instanceof AccessSymbol ) {
				TransformTools.replaceAccessExpressions(region, (AccessSymbol)sharedSym, local_var);
			} else {
				TransformTools.replaceAll(region, new Identifier(sharedSym), local_var);
			}
			/////////////////////////////////////////////////////////////////////////////////////////
			// If the address of the local variable is passed as an argument of a function called  //
			// in the parallel region, revert the instance of the local variable back to the       //
			// pointer variable; in CUDA, dereferencing of device variables is not allowed.        //
			/////////////////////////////////////////////////////////////////////////////////////////
			List<FunctionCall> funcCalls = IRTools.getFunctionCalls(region); 
			for( FunctionCall calledProc : funcCalls ) {
				List<Expression> argList = (List<Expression>)calledProc.getArguments();
				List<Expression> newList = new LinkedList<Expression>();
				boolean foundArg = false;
				for( Expression arg : argList ) {
					arg.setParent(null);
					if(arg instanceof UnaryExpression) {
						UnaryExpression uarg = (UnaryExpression)arg;
						if( uarg.getOperator().equals(UnaryOperator.ADDRESS_OF) 
								&& uarg.getExpression().equals(local_var) ) {
							newList.add((Expression)kParamVar.clone());
							foundArg = true;
						} else {
							newList.add(arg);
						}
					} else {
						newList.add(arg);
					}
				}
				calledProc.setArguments(newList);
				if( !ROData ) {
					if( foundArg ) {
						CompoundStatement ttCompStmt = (CompoundStatement)calledProc.getStatement().getParent();
						ttCompStmt.addStatementBefore(calledProc.getStatement(),
								(Statement)astmt.clone());
						ttCompStmt.addStatementAfter(calledProc.getStatement(),
								(Statement)estmt.clone());
					} else {
						///////////////////////////////////////////////////////////////////////////
						// If the address of the shared variable is not passed as an argument    //
						// of a function called in the kernel region, but accessed in the called //
						// function, load&store statements should be inserted before&after the   //
						// function call site.                                                   //
						///////////////////////////////////////////////////////////////////////////
						Procedure proc = calledProc.getProcedure();
						if( proc != null ) {
							Statement body = proc.getBody();
							if( IRTools.containsSymbol(body, sharedSym) ) {
								CompoundStatement ttCompStmt = (CompoundStatement)calledProc.getStatement().getParent();
								ttCompStmt.addStatementBefore(calledProc.getStatement(),
										(Statement)astmt.clone());
								ttCompStmt.addStatementAfter(calledProc.getStatement(),
										(Statement)estmt.clone());
							}
						}
					}
				}
			}

			if( region instanceof ForLoop ) {
				///////////////////////////////////////////////////////////////////////////////////
				// If caching optimizations are applied, shared variables with locality will be  //
				// loaded into caches (registers or shared memory) at the beginning of kernel    //
				// region. However, if the region is a for-loop and the loaded variable is used  //
				// in the condition expression of the loop, the initial loading statement should //
				// be inserted before the converted kernel region. For this, the statement has   //
				// to be inserted after the for-loop is converted into a kernel function.        //
				///////////////////////////////////////////////////////////////////////////////////
				if( preList == null ) {
					new_proc.getBody().addStatement(estmt);
				} else {
					preList.add(estmt);
				}
			} else {
				Statement last_decl_stmt;
				last_decl_stmt = IRTools.getLastDeclarationStatement(targetStmt);
				if( last_decl_stmt != null ) {
					targetStmt.addStatementAfter(last_decl_stmt,(Statement)estmt);
				} else {
					last_decl_stmt = (Statement)targetStmt.getChildren().get(0);
					targetStmt.addStatementBefore(last_decl_stmt,(Statement)estmt);
				}
			}
			if( !ROData ) {
				if( region instanceof CompoundStatement ) {
					if( isSingleTask ) {
						targetStmt.addStatement(astmt.clone());
					} else {
						IfStatement ifStmt = new IfStatement(new BinaryExpression(new NameID("_tid"), BinaryOperator.COMPARE_EQ,
								new IntegerLiteral(0)), astmt.clone());
						targetStmt.addStatement(ifStmt);
					}
				} else {
					if( region.containsAnnotation(ACCAnnotation.class, "worker") ) {
						if( postList == null ) {
							targetStmt.addStatement(astmt.clone());
						} else {
							postList.add(astmt.clone());
						}
					} else {
						if( isSingleTask ) {
							if( postList == null ) {
								targetStmt.addStatement(astmt.clone());
							} else {
								postList.add(astmt.clone());
							}
						} else {
							IfStatement ifStmt = new IfStatement(new BinaryExpression(new NameID("_tid"), BinaryOperator.COMPARE_EQ,
									new IntegerLiteral(0)), astmt.clone());
							if( postList == null ) {
								targetStmt.addStatement(ifStmt);
							} else {
								postList.add(ifStmt);
							}
						}
					}
				}
			}
		} else {
			Expression deref_expr = new UnaryExpression(UnaryOperator.DEREFERENCE, 
					(Identifier)kParamVar.clone());
			// Replace all instances of the shared variable to a pointer-dereferencing expression (ex: *x).
			//TransformTools.replaceAll((Traversable) targetStmt, cloned_ID, deref_expr);
			if( sharedSym instanceof AccessSymbol ) {
				TransformTools.replaceAccessExpressions(region, (AccessSymbol)sharedSym, deref_expr);
			} else {
				TransformTools.replaceAll(region, new Identifier(sharedSym), deref_expr);
				TransformTools.replaceAll(region, new UnaryExpression(UnaryOperator.ADDRESS_OF, deref_expr),new Identifier(sharedSym));
			}
		}
		
		return kParam_declarator;
	}
	
	/**
	 * Convert the access of a scalar gang-private variable into an array access using array extension.
	 * @param privSym symbol of a target gang-private variable
	 * @param symName name of the new symbol to be created
	 * @param typeSpecs a list of specifiers for the new symbol
	 * @param new_proc a new function where the target symbol will be accessed 
	 * @param region original code region, which will be transformed into the new function, {@ code new_proc}
	 * @param scope the code region where the new symbol will be declared.
	 * @param RegisterRO true if the target symbol is read-only and cached on the register
	 * @return a new symbol that is expanded on the global memory for the input symbol
	 */
	protected static VariableDeclarator scalarGangPrivConv(Symbol privSym, String symName, List<Specifier> typeSpecs,
			Procedure new_proc, Statement region, CompoundStatement scope, boolean RegisterRO, boolean isPureGangRed, 
			FunctionCall call_to_new_proc, Expression ggpriv_var) {
		Set<Symbol> symSet = new_proc.getSymbols();
		Symbol param_sym = AnalysisTools.findsSymbol(symSet, symName);
		VariableDeclaration pointerV_decl = null;
		VariableDeclarator pointerV_declarator = null;
		Identifier pointer_var = null;
		if( param_sym != null ) {
			pointerV_decl = (VariableDeclaration)param_sym.getDeclaration();
			pointerV_declarator = (VariableDeclarator)pointerV_decl.getDeclarator(0);
			pointer_var = new Identifier(pointerV_declarator);
		} else {
			// Create a parameter Declaration for the kernel function
			// Change the scalar variable to a pointer type 
			// ex: float * restrict lprev_x;
			pointerV_declarator = new VariableDeclarator(PointerSpecifier.RESTRICT, 
					new NameID(symName));
			pointerV_decl = new VariableDeclaration(typeSpecs,
					pointerV_declarator);
			pointer_var = new Identifier(pointerV_declarator);
			new_proc.addDeclaration(pointerV_decl);
			// Insert argument to the kernel function call
			call_to_new_proc.addArgument((Identifier)ggpriv_var.clone());
		}

		if( RegisterRO ) {
			/////////////////////////////////////////////////////////////////////////////////
			// Insert a statement to load the global variable to register at the beginning.//
			// SymbolTools.getTemp() inserts the new temp symbol to the symbol table of    //
			// the closest parent if region is a loop.                                     //
			/////////////////////////////////////////////////////////////////////////////////
			Identifier local_var = SymbolTools.getTemp(scope, typeSpecs, symName);
			// lprev_x_0 = lprev_x + _bid;
			// Identifier "_bid" should be updated later so that it can point to a corresponding symbol.
			Statement estmt = new ExpressionStatement(new AssignmentExpression((Identifier)local_var.clone(), 
					AssignmentOperator.NORMAL, new ArrayAccess((Identifier)pointer_var.clone(), 
							SymbolTools.getOrphanID("_bid"))));
			//Statement astmt = new ExpressionStatement(new AssignmentExpression( 
			//		new ArrayAccess((Identifier)pointer_var.clone(), SymbolTools.getOrphanID("_bid")),
			//		AssignmentOperator.NORMAL,(Identifier)local_var.clone()));
			// Replace all instances of the shared variable to the local variable
			if( privSym instanceof AccessSymbol ) {
				TransformTools.replaceAccessExpressions(region, (AccessSymbol)privSym, local_var);
			} else {
				TransformTools.replaceAll(region, new Identifier(privSym), local_var);
			}
			/////////////////////////////////////////////////////////////////////////////////
			// If the address of the local variable is passed as an argument of a function //
			// called in the parallel region, revert the instance of the local variable    //
			// back to a pointer expression.                                               //
			// DEBUG: do we still need this conversion?                                    //
			/////////////////////////////////////////////////////////////////////////////////
			List<FunctionCall> funcCalls = IRTools.getFunctionCalls(region); 
			for( FunctionCall calledProc : funcCalls ) {
				List<Expression> argList = (List<Expression>)calledProc.getArguments();
				List<Expression> newList = new LinkedList<Expression>();
				boolean foundArg = false;
				for( Expression arg : argList ) {
					arg.setParent(null);
					if(arg instanceof UnaryExpression) {
						UnaryExpression uarg = (UnaryExpression)arg;
						if( uarg.getOperator().equals(UnaryOperator.ADDRESS_OF) 
								&& uarg.getExpression().equals(local_var) ) {
							newList.add(new BinaryExpression((Identifier)pointer_var.clone(), 
									BinaryOperator.ADD, SymbolTools.getOrphanID("_bid")));
							foundArg = true;
						} else {
							newList.add(arg);
						}
					} else {
						newList.add(arg);
					}
				}
				calledProc.setArguments(newList);
			}

			Statement last_decl_stmt;
			last_decl_stmt = IRTools.getLastDeclarationStatement(scope);
			if( last_decl_stmt != null ) {
				scope.addStatementAfter(last_decl_stmt,(Statement)estmt);
			} else {
				last_decl_stmt = (Statement)scope.getChildren().get(0);
				scope.addStatementBefore(last_decl_stmt,(Statement)estmt);
			}
			if( isPureGangRed ) {
				//Flush gang-private output back to the original gang-private variable.
				Statement astmt = new ExpressionStatement(new AssignmentExpression(
						new ArrayAccess((Identifier)pointer_var.clone(),SymbolTools.getOrphanID("_bid")), 
						AssignmentOperator.NORMAL, local_var.clone()));
				scope.addStatement(astmt);
			}
		} else {
			Expression array_expr =  new ArrayAccess((Identifier)pointer_var.clone(), SymbolTools.getOrphanID("_bid"));
			// Replace all instances of the gang-private variable to an array expression (ex: lpriv__x[_bid]).
			if( privSym instanceof AccessSymbol ) {
				TransformTools.replaceAccessExpressions(region, (AccessSymbol)privSym, array_expr);
			} else {
				TransformTools.replaceAll(region, new Identifier(privSym), array_expr);
			}
		}

		return pointerV_declarator;
	}
	
	/**
	 * Convert the access of array private variable into an array access using array extension.
	 * @param privSym symbol of a target array-type private variable
	 * @param symName name of the new symbol to be created
	 * @param typeSpecs a list of specifiers for the new symbol
	 * @param startList a list of start indices 
	 * @param lengthList a list of lengths of each array dimension
	 * @param new_proc a new function where the target symbol will be accessed 
	 * @param region original code region, which will be transformed into the new function, {@ code new_proc}
	 * @param scope the code region where the new symbol will be declared.
	 * @param privType type of the input private symbol (0 if gang-private, 1 if worker-private)
	 * @return a new symbol that is expanded on the global memory for the input symbol
	 */
	protected static VariableDeclarator arrayPrivConv(Symbol privSym, String symName, List<Specifier> typeSpecs,
			List<Expression> startList, List<Expression> lengthList, Procedure new_proc, Statement region, 
			CompoundStatement scope, int privType, FunctionCall call_to_new_proc, Expression ggpriv_var) {
		// Create an extended array type
		// Ex: "float b[][SIZE1][SIZE2]"
		int dimsize = lengthList.size();
		Set<Symbol> symSet = new_proc.getSymbols();
		Symbol param_sym = AnalysisTools.findsSymbol(symSet, symName);
		VariableDeclaration arrayV_decl = null;
		VariableDeclarator arrayV_declarator = null;
		Identifier array_var = null;
		if( param_sym != null ) {
			arrayV_decl = (VariableDeclaration)param_sym.getDeclaration();
			arrayV_declarator = (VariableDeclarator)arrayV_decl.getDeclarator(0);
			array_var = new Identifier(arrayV_declarator);
		} else {
			List edimensions = new LinkedList();
			edimensions.add(null);
			for( int i=0; i<dimsize; i++ )
			{
				edimensions.add(lengthList.get(i).clone());
			}
			ArraySpecifier easpec = new ArraySpecifier(edimensions);
			arrayV_declarator = new VariableDeclarator(new NameID(symName), easpec);
			arrayV_decl = 
					new VariableDeclaration(typeSpecs, arrayV_declarator); 
			array_var = new Identifier(arrayV_declarator);
			new_proc.addDeclaration(arrayV_decl);
			// Insert argument to the kernel function call
			//Cast the gpu variable to pointer-to-array type 
			// Ex: (float (*)[dimesion2]) glpriv__x
			List castspecs = new LinkedList();
			castspecs.addAll(typeSpecs);
			/*
			 * FIXME: NestedDeclarator was used for (*)[SIZE1][SIZE2], but this may not be 
			 * semantically correct way to represent (*)[SIZE1][SIZE2] in IR.
			 */
			List tindices = new LinkedList();
			for( int i=0; i<dimsize; i++) {
				tindices.add(lengthList.get(i).clone());
			}
			ArraySpecifier aspec = new ArraySpecifier(tindices);
			List tailSpecs = new ArrayList(1);
			tailSpecs.add(aspec);
			VariableDeclarator childDeclr = new VariableDeclarator(PointerSpecifier.UNQUALIFIED, new NameID(""));
			NestedDeclarator nestedDeclr = new NestedDeclarator(new ArrayList(), childDeclr, null, tailSpecs);
			castspecs.add(nestedDeclr);
			call_to_new_proc.addArgument(new Typecast(castspecs, (Identifier)ggpriv_var.clone()));
		}
		/* 
		 * Replace array access expression with extended access expression.
		 */
		Identifier leftmostIndex  = null;
		if( privType == 0 ) { //gang-private variable
			leftmostIndex = SymbolTools.getOrphanID("_bid");
		} else { //worker-private variable
			leftmostIndex = SymbolTools.getOrphanID("_gtid");
		}
		if( dimsize == 1 ) {
			// Insert "float* lpriv__x_0 = (float *)((float *)lpriv__x + _bid * SIZE1);" and 
			// replace x with lpriv__x_0
			Statement estmt = null;
			Identifier pointer_var = SymbolTools.getPointerTemp(scope, typeSpecs, symName);
			// Identifier "_bid" should be updated later so that it can point to a corresponding symbol.
			List<Specifier> clonedPspecs = new ChainedList<Specifier>();
			clonedPspecs.addAll(typeSpecs);
			clonedPspecs.add(PointerSpecifier.UNQUALIFIED);
			BinaryExpression biexp = new BinaryExpression(new Typecast(clonedPspecs, (Identifier)array_var.clone()),
					BinaryOperator.ADD, new BinaryExpression(leftmostIndex.clone(), 
							BinaryOperator.MULTIPLY, lengthList.get(0).clone()));
			estmt = new ExpressionStatement(new AssignmentExpression((Identifier)pointer_var.clone(), 
					AssignmentOperator.NORMAL, biexp));
			// Replace all instances of the gang-private variable to the local variable
			if( privSym instanceof AccessSymbol ) {
				TransformTools.replaceAccessExpressions(region, (AccessSymbol)privSym, pointer_var);
			} else {
				TransformTools.replaceAll(region, new Identifier(privSym), pointer_var);
			}
/*			// Revert instances of the local variable used in function calls to the original threadprivae
			// variable; this revert is needed for interProcLoopTransform().
			List<FunctionCall> funcCalls = IRTools.getFunctionCalls(region); 
			for( FunctionCall calledProc : funcCalls ) {
				List<Expression> argList = (List<Expression>)calledProc.getArguments();
				for( Expression arg : argList ) {
					TransformTools.replaceAll(arg, pointer_var, array_var);
				}
			} 
*/			
			Statement last_decl_stmt;
			last_decl_stmt = IRTools.getLastDeclarationStatement(scope);
			if( last_decl_stmt != null ) {
				scope.addStatementAfter(last_decl_stmt, estmt);
			} else {
				last_decl_stmt = (Statement)scope.getChildren().get(0);
				scope.addStatementBefore(last_decl_stmt, estmt);
			}
		} else {
			//replace x[k][m] with x[_bid][k][m]
			if( privSym instanceof AccessSymbol ) {
				List<Expression> matches = new ArrayList<Expression>(4);
				DFIterator<Expression> iter =
					new DFIterator<Expression>(region, Expression.class);
				while (iter.hasNext()) {
					Expression child = iter.next();
					if (SymbolTools.getSymbolOf(child).equals(privSym)) {
						matches.add(child);
					}
				}
				for (int i = 0; i < matches.size(); i++) {
					Expression match = matches.get(i);
					Traversable parent = match.getParent();
					if (parent instanceof AccessExpression &&
							match == ((AccessExpression)parent).getRHS()) {
						/* don't replace these */
					} else {
						Expression tExp = match;
						while( tExp instanceof AccessExpression ) {
							tExp = ((AccessExpression)tExp).getRHS();
						}
						Expression newExp = array_var.clone();
						if( tExp instanceof ArrayAccess ) {
							List<Expression> indices = ((ArrayAccess)tExp).getIndices();
							List<Expression> newIndices = new ArrayList<Expression>(indices.size()+1);
							newIndices.add(leftmostIndex.clone());
							for( Expression index : indices ) {
								newIndices.add(index.clone());
							}
							newExp = new ArrayAccess(array_var.clone(), newIndices);
						}
						match.swapWith(newExp);
					}
				}
			} else {
				Identifier cloned_ID = new Identifier(privSym);
				DFIterator<ArrayAccess> iter =
					new DFIterator<ArrayAccess>(region, ArrayAccess.class);
				while (iter.hasNext()) {
					ArrayAccess aAccess = iter.next();
					IDExpression arrayID = (IDExpression)aAccess.getArrayName();
					if( arrayID.equals(cloned_ID) ) {
						List<Expression> indices = aAccess.getIndices();
						List<Expression> newIndices = new ArrayList<Expression>(indices.size()+1);
						newIndices.add(leftmostIndex.clone());
						for( Expression index : indices ) {
							newIndices.add(index.clone());
						}
						ArrayAccess extendedAccess = new ArrayAccess((IDExpression)array_var.clone(), newIndices);
						aAccess.swapWith(extendedAccess);
					}
				}
				// Replace all instances of the shared variable to the parameter variable
				Expression array_expr =  new ArrayAccess((Identifier)array_var.clone(), leftmostIndex.clone());
				TransformTools.replaceAll((Traversable) region, cloned_ID, array_expr);	
			}
		}
		return arrayV_declarator;
	}
	
	protected static VariableDeclarator arrayWorkerPrivCachingOnSM(Symbol privSym, String symName, List<Specifier> typeSpecs,
			List<Expression> startList, List<Expression> lengthList,
			CompoundStatement scope, Expression num_workers) {
		// Create an extended array type
		// Ex: "__shared__ float b[SIZE1][SIZE2][num_workers]"
		int dimsize = lengthList.size();
		List edimensions = new LinkedList();
		for( int i=0; i<dimsize; i++ )
		{
			edimensions.add(lengthList.get(i).clone());
		}
		edimensions.add(num_workers);
		ArraySpecifier easpec = new ArraySpecifier(edimensions);
		VariableDeclarator arrayV_declarator = new VariableDeclarator(new NameID(symName), easpec);
		List<Specifier> eTypeSpecs = new ArrayList<Specifier>(typeSpecs.size()+1);
		eTypeSpecs.addAll(typeSpecs);
		if( !eTypeSpecs.contains(CUDASpecifier.CUDA_SHARED) ) {
			eTypeSpecs.add(CUDASpecifier.CUDA_SHARED);
		}
		VariableDeclaration arrayV_decl = 
			new VariableDeclaration(eTypeSpecs, arrayV_declarator); 
		Identifier array_var = new Identifier(arrayV_declarator);
		scope.addDeclaration(arrayV_decl);
		/* 
		 * Replace array access expression with extended access expression.
		 */
		//replace x[k][m] with x[k][m][_tid]
		if( privSym instanceof AccessSymbol ) {
			List<Expression> matches = new ArrayList<Expression>(4);
			DFIterator<Expression> iter =
				new DFIterator<Expression>(scope, Expression.class);
			while (iter.hasNext()) {
				Expression child = iter.next();
				if (SymbolTools.getSymbolOf(child).equals(privSym)) {
					matches.add(child);
				}
			}
			for (int i = 0; i < matches.size(); i++) {
				Expression match = matches.get(i);
				Traversable parent = match.getParent();
				if (parent instanceof AccessExpression &&
						match == ((AccessExpression)parent).getRHS()) {
					/* don't replace these */
				} else {
					Expression tExp = match;
					while( tExp instanceof AccessExpression ) {
						tExp = ((AccessExpression)tExp).getRHS();
					}
					Expression newExp = array_var.clone();
					if( tExp instanceof ArrayAccess ) {
						List<Expression> indices = ((ArrayAccess)tExp).getIndices();
						List<Expression> newIndices = new ArrayList<Expression>(indices.size()+1);
						for( Expression index : indices ) {
							newIndices.add(index.clone());
						}
						newIndices.add(SymbolTools.getOrphanID("_tid"));
						newExp = new ArrayAccess(array_var.clone(), newIndices);
					}
					match.swapWith(newExp);
				}
			}
		} else {
			Identifier cloned_ID = new Identifier(privSym);
			DFIterator<ArrayAccess> iter =
				new DFIterator<ArrayAccess>(scope, ArrayAccess.class);
			while (iter.hasNext()) {
				ArrayAccess aAccess = iter.next();
				IDExpression arrayID = (IDExpression)aAccess.getArrayName();
				if( arrayID.equals(cloned_ID) ) {
					List<Expression> indices = aAccess.getIndices();
					List<Expression> newIndices = new ArrayList<Expression>(indices.size()+1);
					for( Expression index : indices ) {
						newIndices.add(index.clone());
					}
					newIndices.add(SymbolTools.getOrphanID("_tid"));
					ArrayAccess extendedAccess = new ArrayAccess((IDExpression)array_var.clone(), newIndices);
					aAccess.swapWith(extendedAccess);
				}
			}
			// Replace all instances of the shared variable to the parameter variable
			Expression array_expr =  new ArrayAccess((Identifier)array_var.clone(), SymbolTools.getOrphanID("_tid"));
			TransformTools.replaceAll((Traversable) scope, cloned_ID, array_expr);	
		}
		return arrayV_declarator;
	}
	
	/**
	 * Convert 2D array access expression (aAccess) to a pointer access expression using pitch
	 * Example: x[i][j] is converted into "*(((float *)((char *)x + i*pitch_x)) + j)".
	 * @param aAccess : 2D array
	 * @param pitch : pitch used in cudaMallocPitch() for the array aAccess
	 * @return : pointer access expression using pitch
	 */
	protected static Expression convArray2Pointer( ArrayAccess aAccess, Identifier kParamVar, Identifier pitch,
			List<Specifier> typeSpecs) {
		List<Specifier> specs = new ArrayList<Specifier>(2);
		specs.add(Specifier.CHAR);
		specs.add(PointerSpecifier.UNQUALIFIED);
		Typecast tcast1 = new Typecast(specs, kParamVar.clone()); 
		BinaryExpression biexp1 = new BinaryExpression((Expression)aAccess.getIndex(0).clone(), 
				BinaryOperator.MULTIPLY, (Identifier)pitch.clone());
		BinaryExpression biexp2 = new BinaryExpression(tcast1, BinaryOperator.ADD, biexp1);
		List<Specifier> specs2 = new ArrayList<Specifier>();
		specs2.addAll(typeSpecs);
		specs2.add(PointerSpecifier.UNQUALIFIED);
		Typecast tcast2 = new Typecast(specs2, biexp2);
		BinaryExpression biexp3 = new BinaryExpression(tcast2, BinaryOperator.ADD,
				(Expression)aAccess.getIndex(1).clone());
		UnaryExpression uexp = new UnaryExpression(UnaryOperator.DEREFERENCE, biexp3);
		return uexp;
	}
	
	/**
	 * Convert 1D array access expression (aAccess) to a pointer access expression using pitch.
	 * This conversion is used for MatrixTranspose optimization on Threadprivate data.
	 * Example: x[i] is converted into "*((float *)((char *)x + i*pitch_x))".
	 * @param aAccess : 1D array
	 * @param pitch : pitch used in cudaMallocPitch() for the array aAccess
	 * @return : pointer access expression using pitch
	 */
	protected static Expression convArray2Pointer2( ArrayAccess aAccess, Identifier pitch ) {
		List<Specifier> specs = new ArrayList<Specifier>(2);
		specs.add(Specifier.CHAR);
		specs.add(PointerSpecifier.UNQUALIFIED);
		Typecast tcast1 = new Typecast(specs, (Expression)aAccess.getArrayName().clone()); 
		BinaryExpression biexp1 = new BinaryExpression((Expression)aAccess.getIndex(0).clone(), 
				BinaryOperator.MULTIPLY, (Identifier)pitch.clone());
		BinaryExpression biexp2 = new BinaryExpression(tcast1, BinaryOperator.ADD, biexp1);
		List<Specifier> specs2 = new ArrayList<Specifier>();
		/////////////////////////////////////////////////////////////////////////////////////
		// CAUTION: VariableDeclarator.getTypeSpecifiers() returns both specifiers of      //
		// its parent VariableDeclaration and the VariableDeclarator's leading specifiers. //
		// Therefore, if VariableDeclarator is a pointer symbol, this method will return   //
		// pointer specifiers too.                                                         //
		/////////////////////////////////////////////////////////////////////////////////////
		specs2.addAll(((Identifier)aAccess.getArrayName()).getSymbol().getTypeSpecifiers());
		specs2.remove(Specifier.STATIC);
		specs2.add(PointerSpecifier.UNQUALIFIED);
		Typecast tcast2 = new Typecast(specs2, biexp2);
		UnaryExpression uexp = new UnaryExpression(UnaryOperator.DEREFERENCE, tcast2);
		return uexp;
	}
	
	/**
	 * This method performs code transformation to handle private/firstprivate clauses.
	 * This transformation is intraprocedural; functions called in a compute region should be handled
	 * separately.
	 * CAUTION: This translation assumes that there is only one private clause per nested gang loops or 
	 * nested worker loops.
	 * 
	 * @param region
	 */
	protected static void privateTransformation(Procedure cProc, Statement region, String cRegionKind, Expression ifCond,
			Expression asyncID, Statement confRefStmt,
			CompoundStatement prefixStmts, CompoundStatement postscriptStmts,
			List<Statement> preList, List<Statement> postList,
			FunctionCall call_to_new_proc, Procedure new_proc, TranslationUnit main_TrUnt, 
			Map<TranslationUnit, Declaration> OpenACCHeaderEndMap, boolean IRSymbolOnly,
			boolean opt_addSafetyCheckingCode, Set<Symbol> arrayElmtCacheSymbols, boolean isSingleTask ) {
		PrintTools.println("[privateTransformation() begins] current procedure: " + cProc.getSymbolName() +
				"\ncompute region type: " + cRegionKind + "\n", 2);
		
		CompoundStatement scope = null;
		SymbolTable global_table = (SymbolTable) cProc.getParent();
		//CompoundStatement mallocScope = (CompoundStatement)confRefStmt.getParent();
		
		
		// Auxiliary variables used for GPU kernel conversion 
		VariableDeclaration bytes_decl = (VariableDeclaration)SymbolTools.findSymbol(global_table, "gpuBytes");
		Identifier cloned_bytes = new Identifier((VariableDeclarator)bytes_decl.getDeclarator(0));					
		VariableDeclaration gmem_decl = null;
		Identifier gmemsize = null;
		VariableDeclaration smem_decl = null;
		Identifier smemsize = null;
		ExpressionStatement gMemAdd_stmt = null;
		ExpressionStatement gMemSub_stmt = null;
		if( opt_addSafetyCheckingCode ) {
			gmem_decl = (VariableDeclaration)SymbolTools.findSymbol(global_table, "gpuGmemSize");
			gmemsize = new Identifier((VariableDeclarator)gmem_decl.getDeclarator(0));					
			smem_decl = (VariableDeclaration)SymbolTools.findSymbol(global_table, "gpuSmemSize");
			smemsize = new Identifier((VariableDeclarator)smem_decl.getDeclarator(0));					
			gMemAdd_stmt = new ExpressionStatement( new AssignmentExpression((Identifier)gmemsize.clone(),
					AssignmentOperator.ADD, (Identifier)cloned_bytes.clone()) );
			gMemSub_stmt = new ExpressionStatement( new AssignmentExpression((Identifier)gmemsize.clone(),
					AssignmentOperator.SUBTRACT, (Identifier)cloned_bytes.clone()) );
		}
		VariableDeclaration numBlocks_decl = (VariableDeclaration)SymbolTools.findSymbol(global_table, "gpuNumBlocks");
		Identifier numBlocks = new Identifier((VariableDeclarator)numBlocks_decl.getDeclarator(0));					
		VariableDeclaration numThreads_decl = (VariableDeclaration)SymbolTools.findSymbol(global_table, "gpuNumThreads");
		Identifier numThreads = new Identifier((VariableDeclarator)numThreads_decl.getDeclarator(0));					
		VariableDeclaration totalNumThreads_decl = (VariableDeclaration)SymbolTools.findSymbol(global_table, "totalGpuNumThreads");
		Identifier totalNumThreads = new Identifier((VariableDeclarator)totalNumThreads_decl.getDeclarator(0));					
		ExpressionStatement gpuBytes_stmt = null;
		ExpressionStatement orgGpuBytes_stmt = null;
		
		
		//Auxiliary structures to assist translation
		Set<Symbol> allocatedGangPrivSet = new HashSet<Symbol>();
		Set<Symbol> allocatedWorkerPrivSet = new HashSet<Symbol>();
		
		
		//Check the number of workers for this compute region, which may be needed for worker-private-caching on shared memory.
		long num_workers = 0;
		if( cRegionKind.equals("parallel") ) {
			ACCAnnotation tAnnot = region.getAnnotation(ACCAnnotation.class, "num_workers");
			if( tAnnot != null ) {
				Expression numExp = tAnnot.get("num_workers");
				if( numExp instanceof IntegerLiteral ) {
					num_workers = ((IntegerLiteral)numExp).getValue();
				}
			}
		} else {
			ACCAnnotation tAnnot = region.getAnnotation(ACCAnnotation.class, "totalnumworkers");
			if( tAnnot != null ) {
				Expression numExp = tAnnot.get("totalnumworkers");
				if( numExp instanceof IntegerLiteral ) {
					num_workers = ((IntegerLiteral)numExp).getValue();
				}
			}
		}
		
		ACCAnnotation readonlyprivateAnnot = region.getAnnotation(ACCAnnotation.class, "accreadonlyprivate");
		Set<Symbol> accreadonlyprivateSet = null;
		if( readonlyprivateAnnot != null ) {
			accreadonlyprivateSet = readonlyprivateAnnot.get("accreadonlyprivate");
		} else {
			accreadonlyprivateSet = new HashSet<Symbol>();
		}
		
		//Find index symbols for work-sharing loops, which are private to each thread by default.
		Set<Symbol> loopIndexSymbols = AnalysisTools.getWorkSharingLoopIndexVarSet(region);
		
		//The local variables defined in a pure gang loop are gang-private, 
		//which will be allocated on the GPU shared memory by default.
		//The local variable defined outside of gang loops but within a compute region
		//will be alloced on the GPU shared memory if they are not included in any gang private clauses.
		Set<Symbol> localGangPrivateSymbols = new HashSet<Symbol>();
        Set<Symbol> ipLocalGangPrivateSymbols = new HashSet<Symbol>();
        Set<Symbol> localGangPrivateSymbolsAll = new HashSet<Symbol>();
        CompoundStatement ttCStmt = null;
        if( !isSingleTask ) {
        	List<FunctionCall> fCallList = null;
        	if( region instanceof CompoundStatement ) {
        		ttCStmt = (CompoundStatement)region;
        	} else if( region instanceof Loop) {
        		ttCStmt = (CompoundStatement)((Loop)region).getBody();
        		if( region.containsAnnotation(ACCAnnotation.class, "worker") ) {
        			ttCStmt = null;
        		}
        	} else {
        		fCallList = IRTools.getFunctionCalls(region);
        	}
        	Set<Procedure> visitedProcedures = new HashSet<Procedure>();
        	if( ttCStmt != null ) {
        		localGangPrivateSymbols.addAll(ttCStmt.getSymbols());
        		localGangPrivateSymbols.removeAll(AnalysisTools.getWorkSharingLoopIndexVarSet(ttCStmt));
        		localGangPrivateSymbols.addAll(AnalysisTools.getLocalGangPrivateSymbols(ttCStmt, false, visitedProcedures));
        		fCallList = IRTools.getFunctionCalls(ttCStmt);
        	}
        	if( fCallList != null ) {
        		for(FunctionCall tfCall : fCallList) {
        			Procedure tProc = tfCall.getProcedure();
        			if( tProc != null ) {
        				boolean foundWorkerLoop = false;
        				Traversable tt = tfCall.getParent();
        				while (tt != null) {
        					if( tt instanceof Annotatable ) {
        						Annotatable at = (Annotatable)tt;
        						ACCAnnotation lAnnot = at.getAnnotation(ACCAnnotation.class, "loop");
        						if( (lAnnot != null) && (lAnnot.containsKey("worker")) ) {
        							foundWorkerLoop = true;
        							break;
        						}
        					}
        					tt = tt.getParent();
        				}
        				if( !foundWorkerLoop ) {
        					CompoundStatement tBody = tProc.getBody();
        					ipLocalGangPrivateSymbols.addAll(tBody.getSymbols());
        					ipLocalGangPrivateSymbols.removeAll(AnalysisTools.getWorkSharingLoopIndexVarSet(tBody));
        					ipLocalGangPrivateSymbols.addAll(AnalysisTools.getLocalGangPrivateSymbols(tBody, true, visitedProcedures));
        				}

        			}
        		}
        	}
        	localGangPrivateSymbols.removeAll(arrayElmtCacheSymbols);
/*			if( region instanceof CompoundStatement ) {
				//The local variables defined outside of gang loops but within a compute region are gang-private.
				localGangPrivateSymbols.addAll(((CompoundStatement) region).getSymbols());
				//symbols used to cache array elements on register should be worker-private.
				localGangPrivateSymbols.removeAll(arrayElmtCacheSymbols);
			}
			List<ACCAnnotation> gangLoopAnnots = AnalysisTools.ipCollectPragmas(region, ACCAnnotation.class, "gang", null);
			if( gangLoopAnnots != null ) {
				for( ACCAnnotation gAnnot : gangLoopAnnots ) {
					if( gAnnot.containsKey("worker") ) {
						continue;
					} else {
						//Local variables defined in a pure gang loop are gang-private.
						Annotatable gAt = gAnnot.getAnnotatable();
						if( gAt instanceof ForLoop ) {
							CompoundStatement gBody = (CompoundStatement)((ForLoop)gAt).getBody();
							localGangPrivateSymbols.addAll(gBody.getSymbols());
							//If the inner loop is part of the stripmined gang loop, the local variables in the inner
							//loop should be also included.
							Statement child = IRTools.getFirstNonDeclarationStatement(gBody);
							if ( (child != null) && child.containsAnnotation(ACCAnnotation.class, "innergang") ) {
								if( child instanceof ForLoop ) {
									gBody = (CompoundStatement)((ForLoop)child).getBody();
									localGangPrivateSymbols.addAll(gBody.getSymbols());
								}
							}
						}
					}
				}
			}*/
        	localGangPrivateSymbolsAll.addAll(localGangPrivateSymbols);
        	localGangPrivateSymbolsAll.addAll(ipLocalGangPrivateSymbols);
		}
		
		//For correct translation, worker-private loops should be handled before gang-private regions.
		//CAUTION: This translation assumes that there is only one private clause per nested gang loops or 
		//nested worker loops.
		List<ACCAnnotation> private_annots = AnalysisTools.collectPragmas(region, ACCAnnotation.class, 
				ACCAnnotation.privateClauses, false);
		List<ACCAnnotation> workerpriv_regions = new LinkedList<ACCAnnotation>();
		List<ACCAnnotation> gangpriv_regions = new LinkedList<ACCAnnotation>();
		if( private_annots != null ) {
			for ( ACCAnnotation pannot : private_annots ) {
				if( pannot.containsKey("worker") ) {
					workerpriv_regions.add(pannot);
				} else if( pannot.containsKey("seq") ) {
					Annotatable at = pannot.getAnnotatable();
					if( AnalysisTools.ipContainPragmas(at, ACCAnnotation.class, ACCAnnotation.parallelWorksharingClauses, false, null) ) {
						// private variables in seq kernel loops are treated as worker-private variables.
						workerpriv_regions.add(pannot);
					}
                } else if( isSingleTask ) {
                	// private variables in single-task region are treated as worker-private variables.
                	workerpriv_regions.add(pannot);
				} else {
					gangpriv_regions.add(pannot);
				}
			}
		}
		List<ACCAnnotation> priv_regions = new LinkedList<ACCAnnotation>();
		priv_regions.addAll(workerpriv_regions);
		priv_regions.addAll(gangpriv_regions);
		
		for ( ACCAnnotation pannot : priv_regions ) {
			Annotatable at = pannot.getAnnotatable();
			Map<Symbol, SubArray> sharedCachingMap = new HashMap<Symbol, SubArray>();
			Map<Symbol, SubArray> regROCachingMap = new HashMap<Symbol, SubArray>();
			Map<Symbol, SubArray> globalMap = new HashMap<Symbol, SubArray>();
			Set<String> searchKeys = new HashSet<String>();
			searchKeys.add("sharedRO");
			searchKeys.add("sharedRW");
			for( String key : searchKeys ) {
				ARCAnnotation ttAnt = at.getAnnotation(ARCAnnotation.class, key);
				if( ttAnt != null ) {
					Set<SubArray> DataSet = (Set<SubArray>)ttAnt.get(key);
					for( SubArray sAr : DataSet ) {
						Symbol tSym = AnalysisTools.subarrayToSymbol(sAr, IRSymbolOnly);
						sharedCachingMap.put(tSym, sAr);
					}
				}
			}
			ARCAnnotation ttAnt = at.getAnnotation(ARCAnnotation.class, "registerRO");
			if( ttAnt != null ) {
				Set<SubArray> DataSet = (Set<SubArray>)ttAnt.get("registerRO");
				for( SubArray sAr : DataSet ) {
					Symbol tSym = AnalysisTools.subarrayToSymbol(sAr, IRSymbolOnly);
					regROCachingMap.put(tSym, sAr);
				}
			}
			ttAnt = at.getAnnotation(ARCAnnotation.class, "global");
			if( ttAnt != null ) {
				Set<SubArray> DataSet = (Set<SubArray>)ttAnt.get("global");
				for( SubArray sAr : DataSet ) {
					Symbol tSym = AnalysisTools.subarrayToSymbol(sAr, IRSymbolOnly);
					globalMap.put(tSym, sAr);
				}
			}
			
			boolean isWorkerPrivate = false;
			if( at.containsAnnotation(ACCAnnotation.class, "worker") ) {
				isWorkerPrivate = true;
			} else if( at.containsAnnotation(ACCAnnotation.class, "seq") ) {
				if( AnalysisTools.ipContainPragmas(at, ACCAnnotation.class, ACCAnnotation.parallelWorksharingClauses, false, null) ) {
					isWorkerPrivate = true;
				}
            } else if (isSingleTask) {
                isWorkerPrivate = true;
			}
			boolean isGangPrivate = false;
			if( !isSingleTask ) {
				if( at.containsAnnotation(ACCAnnotation.class, "gang") ) {
					isGangPrivate = true;
				}
				if( at.containsAnnotation(ACCAnnotation.class, "parallel") ) {
					isGangPrivate = true;
				}
			}
			scope = null;
			if( isWorkerPrivate ) {//A worker private variable is declared in the innermost worker loop body.
				if( isSingleTask ) {
					if( region instanceof ForLoop ) {
						ACCAnnotation tAnnot = AnalysisTools.findInnermostPragma(region, ACCAnnotation.class, "gang");
						if( tAnnot != null ) {
							ForLoop gLoop = (ForLoop)tAnnot.getAnnotatable();
							scope = (CompoundStatement)(gLoop).getBody();
						} else {
							scope = (CompoundStatement)((ForLoop)region).getBody();
						}
					} else if( region instanceof CompoundStatement ) {
						scope = (CompoundStatement)region;
					}
				} else {
					ACCAnnotation tAnnot = AnalysisTools.findInnermostPragma(at, ACCAnnotation.class, "worker");
					ForLoop wLoop = (ForLoop)tAnnot.getAnnotatable();
					scope = (CompoundStatement)(wLoop).getBody();
				}
			} else if( isGangPrivate ) {
				///////////////////////////////////////////////////////////////////////////////////
				//A gang private variable is declared either in the enclosing compound statement //
				//or in the innermost gang loop body if region is a loop.                        //
				///////////////////////////////////////////////////////////////////////////////////
				if( region instanceof ForLoop ) {
					ACCAnnotation tAnnot = AnalysisTools.findInnermostPragma(region, ACCAnnotation.class, "gang");
					ForLoop gLoop = (ForLoop)tAnnot.getAnnotatable();
					scope = (CompoundStatement)(gLoop).getBody();
				} else if( region instanceof CompoundStatement ) {
					scope = (CompoundStatement)region;
				}
			} else {
				continue; //vector private is ignored.
			}
			if( scope == null ) {
				Tools.exit("[ERROR in CUDATranslationTools.privateTransformation() cannot find the scope where private" +
						"symbols in the following compute region are declared: \n" + region + "\n");
			}
			
			HashSet<SubArray> PrivSet = null; 
			HashSet<SubArray> FirstPrivSet = null; 
			HashMap<Symbol, SubArray> PrivSymMap = new HashMap<Symbol, SubArray>();
			HashSet<Symbol> FirstPrivSymSet = new HashSet<Symbol>();
			PrivSet = (HashSet<SubArray>) pannot.get("private");
			if( PrivSet != null ) {
				for( SubArray sAr : PrivSet ) {
					Symbol tSym = AnalysisTools.subarrayToSymbol(sAr, IRSymbolOnly);
					if( tSym != null ) {
						PrivSymMap.put(tSym, sAr);
					} else {
						Tools.exit("[ERROR in CUDATranslationTools.privateTransformation()] found null in the private set; exit!\n" +
								"Enclosing procedure: " + cProc.getSymbolName() + "\n" +
										"OpenACC Annotation: " + pannot +"\n");
					}
				}
			}
			FirstPrivSet = (HashSet<SubArray>) pannot.get("firstprivate");
			if( FirstPrivSet != null ) {
				for( SubArray sAr : FirstPrivSet ) {
					Symbol tSym = AnalysisTools.subarrayToSymbol(sAr, IRSymbolOnly);
					PrivSymMap.put(tSym, sAr);
					FirstPrivSymSet.add(tSym);
				}
			}

			Collection<Symbol> sortedSet = AnalysisTools.getSortedCollection(PrivSymMap.keySet());
			for( Symbol privSym : sortedSet ) {
				SubArray sArray = PrivSymMap.get(privSym);
				Boolean isArray = SymbolTools.isArray(privSym);
				Boolean isPointer = SymbolTools.isPointer(privSym);
				if( privSym instanceof NestedDeclarator ) {
					isPointer =  true;
				}
				
				//If local gang-private symbols are included in any OpenACC private clause, 
				//remove them from localGangPrivateSymbols set.
				localGangPrivateSymbols.remove(privSym);
				
				//////////////////////////////////////////////////////////////////////////////////
				//FIXME: if privSym is a parameter of a function called in the parallel region, //
				//below checking may be incorrect.                                              //
				//////////////////////////////////////////////////////////////////////////////////
				//DEBUG: extractComputeRegion() in ACC2CUDATranslator/ACC2OPENCLTranslator may promote 
				//privatization-related statements above the scope where the private variable is declared.
				//In this case, the below targetSymbolTable will not work.
				//To handle this promotion, we simply used the enclosing function body as targetSymbolTable.
/*				SymbolTable targetSymbolTable = AnalysisTools.getIRSymbolScope(privSym, region);
				if( targetSymbolTable instanceof Procedure ) {
					targetSymbolTable = ((Procedure)targetSymbolTable).getBody();
				}
				if( targetSymbolTable == null ) {
					targetSymbolTable = (SymbolTable) cProc.getBody();
				}*/
				SymbolTable targetSymbolTable = cProc.getBody();
				
				
				List<Expression> startList = new LinkedList<Expression>();
				List<Expression> lengthList = new LinkedList<Expression>();
				boolean foundDimensions = AnalysisTools.extractDimensionInfo(sArray, startList, lengthList, IRSymbolOnly, at);
				if( !foundDimensions ) {
					Tools.exit("[ERROR in CUDATranslationTools.privateTransformation()] Dimension information of the following " +
							"private variable is unknown: " + sArray.getArrayName() + ", Enclosing procedure: " + 
							cProc.getSymbolName() + "; the ACC2GPU translation failed!");
				}
				int dimsize = lengthList.size();

				/* 
				 * Create a new temporary variable for the private variable.
				 */
				VariableDeclaration gpu_priv_decl = null;
				Identifier ggpriv_var = null;
				Identifier gwpriv_var = null;
				Identifier lpriv_var = null;
				String symNameBase = null;
				if( privSym instanceof AccessSymbol) {
					symNameBase = TransformTools.buildAccessSymbolName((AccessSymbol)privSym);
				} else {
					symNameBase = privSym.getSymbolName();
				}
				String gpuGPSymName = "ggpriv__" + symNameBase;
				String gpuWPSymName = "gwpriv__" + symNameBase;
				String localWPSymName = "lwpriv__" + symNameBase;
				String localGPSymName = "lgpriv__" + symNameBase;
				
				/////////////////////////////////////////////////////////////////////////////////
				// __device__ and __global__ functions can not declare static/extern variables //
				// inside their body.                                                          //
				/////////////////////////////////////////////////////////////////////////////////
				List<Specifier> typeSpecs = new ArrayList<Specifier>();
				Symbol IRSym = privSym;
				if( IRSym instanceof PseudoSymbol ) {
					IRSym = ((PseudoSymbol)IRSym).getIRSymbol();
				}
				if( IRSymbolOnly ) {
					typeSpecs.addAll(((VariableDeclaration)IRSym.getDeclaration()).getSpecifiers());
				} else {
					Symbol tSym = privSym;
					while( tSym instanceof AccessSymbol ) {
						tSym = ((AccessSymbol)tSym).getMemberSymbol();
					}
					typeSpecs.addAll(((VariableDeclaration)tSym.getDeclaration()).getSpecifiers());
				}
				typeSpecs.remove(Specifier.STATIC);
				typeSpecs.remove(Specifier.CONST);
				typeSpecs.remove(Specifier.EXTERN);
				
				List<Specifier> removeSpecs = new ArrayList<Specifier>(1);
				removeSpecs.add(Specifier.STATIC);
				removeSpecs.add(Specifier.CONST);
				removeSpecs.add(Specifier.EXTERN);
				boolean gangPrivCachingOnShared = false;
				boolean workerPrivCachingOnShared = false;
				boolean workerPrivOnGlobal = false;
				List<Specifier> addSpecs = null;
				if( sharedCachingMap.keySet().contains(privSym) ) {
					if( isWorkerPrivate ) {
						if( num_workers > 0 ) {
							workerPrivCachingOnShared = true;
						} else {
							PrintTools.println("\n[WARNING] caching of worker-private variable, " + privSym.getSymbolName() +
									", on the shared memory is not appplicable, since the number of workers are not " +
									"compile-time constant.\nEnclosing procedure: " + cProc.getSymbolName() + "\n"  ,0);
						}
					} else {
						addSpecs = new ArrayList<Specifier>(1);
						addSpecs.add(CUDASpecifier.CUDA_SHARED);
						gangPrivCachingOnShared = true;
					}
				} 
				if( globalMap.keySet().contains(privSym) ) {
					if( isWorkerPrivate && ( isArray || isPointer ) ) {
						workerPrivOnGlobal = true;
					}
				}
				
				//work-sharing loop index variable is treated as if thread-private variable.
				if( loopIndexSymbols.contains(privSym) ) {
					if( isWorkerPrivate ) {
						workerPrivCachingOnShared = false;
						workerPrivOnGlobal = false;
					} else {
						addSpecs = new ArrayList<Specifier>(1);
						//addSpecs.add(CUDASpecifier.CUDA_SHARED);
						gangPrivCachingOnShared = true;
					}
				}
				
				boolean insertMalloc = false;
				boolean insertWMalloc = false;
				
				if( isGangPrivate && (FirstPrivSymSet != null) && FirstPrivSymSet.contains(privSym) &&
						accreadonlyprivateSet.contains(privSym) && !isArray && !isPointer ) {
					Identifier lfpriv_var = null;
					String localFPSymName = "lfpriv__" + symNameBase;
					//////////////////////////////////////////////////////////////////////////////
					// If firstprivate variable is scalar, the corresponding shared variable is //
					// passed as a kernel parameter instead of using GPU global memory, which   //
					// has the effect of caching it on the GPU Shared Memory.                   //
					//////////////////////////////////////////////////////////////////////////////
					// Create a GPU kernel parameter corresponding to privSym
					// ex: float lfpriv__x;
					VariableDeclarator lfpriv_declarator = new VariableDeclarator(new NameID(localFPSymName));
					VariableDeclaration lfpriv_decl = new VariableDeclaration(typeSpecs, 
							lfpriv_declarator);
					lfpriv_var = new Identifier(lfpriv_declarator);
					new_proc.addDeclaration(lfpriv_decl);

					// Insert argument to the kernel function call
					if( privSym instanceof AccessSymbol ) {
						AccessExpression accExp = AnalysisTools.accessSymbolToExpression((AccessSymbol)privSym, null);
						call_to_new_proc.addArgument(accExp);
					} else {
						call_to_new_proc.addArgument(new Identifier(privSym));
					}
					TransformTools.replaceAll(region, new Identifier(privSym), lfpriv_var);
					continue;
				}else if( isWorkerPrivate ) {
					if( workerPrivOnGlobal ) {
						//Option to allocate worker-private variable on global memory is checked first, since it may be
						//mandatory due to too large private array size.
						//////////////////////////////////////////////////////
						//Create a worker-private variable on global memory //
						//using array expansion.                            //
						//    - applied only non-scalar type                //
						//////////////////////////////////////////////////////
						// float lprev__x[totalnumworkers][SIZE];           //
						//////////////////////////////////////////////////////
						/////////////////////////////////////////////////////////////
						// Create a GPU device variable corresponding to privSym   //
						// Ex: float * gwpriv__x; //GPU variable for gang-private  //
						/////////////////////////////////////////////////////////////
						// Give a new name for the device variable 
						gpu_priv_decl =  TransformTools.getGPUVariable(gpuWPSymName, targetSymbolTable, 
								typeSpecs, main_TrUnt, OpenACCHeaderEndMap, null);
						gwpriv_var = new Identifier((VariableDeclarator)gpu_priv_decl.getDeclarator(0));
						/////////////////////////////////////////////////////
						// Memory allocation for the device variable       //
						// Insert cudaMalloc() function before the region  //
						/////////////////////////////////////////////////////
						// Ex: cudaMalloc(((void *  * )( & gwpriv__x)), gpuBytes);
						FunctionCall malloc_call = new FunctionCall(new NameID("HI_tempMalloc1D"));
						List<Specifier> specs = new ArrayList<Specifier>(4);
						specs.add(Specifier.VOID);
						specs.add(PointerSpecifier.UNQUALIFIED);
						specs.add(PointerSpecifier.UNQUALIFIED);
						List<Expression> arg_list = new ArrayList<Expression>();
						arg_list.add(new Typecast(specs, new UnaryExpression(UnaryOperator.ADDRESS_OF, 
								(Identifier)gwpriv_var.clone())));
						SizeofExpression sizeof_expr = new SizeofExpression(typeSpecs);

						if( !allocatedWorkerPrivSet.contains(privSym) ) {
							insertWMalloc = true;
							allocatedWorkerPrivSet.add(privSym);
						}

						// Insert "gpuBytes = totalGpuNumThreads * (dimension1 * dimension2 * ..) 
						// * sizeof(varType);" statement
						Expression biexp = null;
						Expression biexp2 = null;
						if( dimsize > 0 ) {
							biexp = lengthList.get(0).clone();
							for( int i=1; i<dimsize; i++ )
							{
								biexp = new BinaryExpression(biexp, BinaryOperator.MULTIPLY, lengthList.get(i).clone());
							}
							biexp2 = new BinaryExpression(biexp, BinaryOperator.MULTIPLY, sizeof_expr);
						} else {
							biexp2 = sizeof_expr;
						}
						AssignmentExpression assignex = new AssignmentExpression((Expression)cloned_bytes.clone(),
								AssignmentOperator.NORMAL, biexp2);
						orgGpuBytes_stmt = new ExpressionStatement(assignex);
						biexp = new BinaryExpression((Expression)totalNumThreads.clone(), 
								BinaryOperator.MULTIPLY, (Expression)biexp2.clone());
						assignex = new AssignmentExpression((Expression)cloned_bytes.clone(),AssignmentOperator.NORMAL, 
								biexp);
						gpuBytes_stmt = new ExpressionStatement(assignex);

						// Create a parameter Declaration for the kernel function
						// Create an extended array type
						// Ex1: "float lwpriv__x[][SIZE1]"
						// Ex2: "float lwpriv__x[][SIZE1][SIZE2]"
						VariableDeclarator arrayV_declarator =	arrayPrivConv(privSym, localWPSymName, typeSpecs, startList,
								lengthList, new_proc, region, scope, 1, call_to_new_proc, gwpriv_var.clone());
						lpriv_var = new Identifier(arrayV_declarator);

						// Add gpuBytes argument to cudaMalloc() call
						arg_list.add((Identifier)cloned_bytes.clone());
                        arg_list.add(new NameID("acc_device_current"));
						malloc_call.setArguments(arg_list);
						ExpressionStatement malloc_stmt = new ExpressionStatement(malloc_call);
						// Insert malloc statement.
						if( insertWMalloc ) {
							//mallocScope.addStatementBefore(confRefStmt, gpuBytes_stmt);
							//mallocScope.addStatementBefore(confRefStmt, malloc_stmt);
							prefixStmts.addStatement(gpuBytes_stmt);
							prefixStmts.addStatement(malloc_stmt);
							if( opt_addSafetyCheckingCode ) {
								// Insert "gpuGmemSize += gpuBytes;" statement 
								//mallocScope.addStatementBefore(gpuBytes_stmt.clone(), 
								//		(Statement)gMemAdd_stmt.clone());
								prefixStmts.addStatement(gMemAdd_stmt.clone());
							}
							/*
							 * Insert cudaFree() to deallocate device memory for worker-private variable. 
							 * Because cuda-related statements are added in reverse order, 
							 * this function call is added first.
							 */
							//if( opt_addSafetyCheckingCode  ) {
							//	// Insert "gpuGmemSize -= gpuBytes;" statement 
							//	mallocScope.addStatementAfter(confRefStmt, gMemSub_stmt.clone());
							//}
							// Insert "cudaFree(gwpriv__x);"
							FunctionCall cudaFree_call = new FunctionCall(new NameID("HI_tempFree"));
							specs = new ArrayList<Specifier>(4);
							specs.add(Specifier.VOID);
							specs.add(PointerSpecifier.UNQUALIFIED);
							specs.add(PointerSpecifier.UNQUALIFIED);
							cudaFree_call.addArgument(new Typecast(specs, new UnaryExpression(UnaryOperator.ADDRESS_OF,
									(Identifier)gwpriv_var.clone())));
							cudaFree_call.addArgument(new NameID("acc_device_current"));
							ExpressionStatement cudaFree_stmt = new ExpressionStatement(cudaFree_call);
							//mallocScope.addStatementAfter(confRefStmt, cudaFree_stmt);
							//mallocScope.addStatementAfter(confRefStmt, gpuBytes_stmt.clone());
							postscriptStmts.addStatement(gpuBytes_stmt.clone());
							postscriptStmts.addStatement(cudaFree_stmt);
							if( opt_addSafetyCheckingCode  ) {
								postscriptStmts.addStatement(gMemSub_stmt.clone());
							}
						}
					} else if( workerPrivCachingOnShared ) {
						//////////////////////////////////////////////////////
						//Create a worker-private variable on shared memory //
						//using array expansion.                            //
						//////////////////////////////////////////////////////
						// __shared__ float lwprev__x[SIZE][num_workers];   //
						//////////////////////////////////////////////////////
						VariableDeclarator arrayV_declarator =	arrayWorkerPrivCachingOnSM(privSym, localWPSymName, typeSpecs, startList,
								lengthList, scope, new IntegerLiteral(num_workers));
						lpriv_var = new Identifier(arrayV_declarator);
					} else {
						//////////////////////////////////////////////////////
						//Create a worker-private variable on local memory. //
						//(No array-expansion is needed.)                   //
						//////////////////////////////////////////////////////
						//     float lwprev__x;                             //
						//     float lwprev__x[SIZE];                       //
						//////////////////////////////////////////////////////
						if( privSym instanceof AccessSymbol ) {
							Symbol tSym = privSym;
							while( tSym instanceof AccessSymbol ) {
								tSym = ((AccessSymbol)tSym).getMemberSymbol();
							}
							//lpriv_var = TransformTools.declareClonedVariable(scope, tSym, localWPSymName, 
							//		removeSpecs, addSpecs);
							lpriv_var = TransformTools.declareClonedArrayVariable(scope, sArray, localWPSymName, 
									removeSpecs, addSpecs, true);
						} else {
							//lpriv_var = TransformTools.declareClonedVariable(scope, privSym, localWPSymName, 
							//		removeSpecs, addSpecs);
							lpriv_var = TransformTools.declareClonedArrayVariable(scope, sArray, localWPSymName, 
									removeSpecs, addSpecs, true);
						}
						////////////////////////////////////////////////////////////////////////
						// Replace the private variable with this new local private variable. //
						////////////////////////////////////////////////////////////////////////
						if( privSym instanceof AccessSymbol ) {
							TransformTools.replaceAccessExpressions(at, (AccessSymbol)privSym, lpriv_var);
						} else {
							TransformTools.replaceAll(at, new Identifier(privSym), lpriv_var);
						}
					}
					//Reset below to be checked later.
					gpuBytes_stmt = null;
					orgGpuBytes_stmt = null;
				} else if( gangPrivCachingOnShared ) { //gang-private variable cached on the GPU shared memory.
					/////////////////////////////////////////////////////
					//Create a gang-private variable on shared memory. //
					//(No array-expansion is needed.)                  //
					/////////////////////////////////////////////////////
					//     __shared__ float lgprev__x;                 //
					//     __shared__ float lgprev__x[SIZE];           //
					/////////////////////////////////////////////////////
					if( privSym instanceof AccessSymbol ) {
						Symbol tSym = privSym;
						while( tSym instanceof AccessSymbol ) {
							tSym = ((AccessSymbol)tSym).getMemberSymbol();
						}
						lpriv_var = TransformTools.declareClonedArrayVariable(scope, sArray, localGPSymName, 
								removeSpecs, addSpecs, false);
					} else {
						lpriv_var = TransformTools.declareClonedArrayVariable(scope, sArray, localGPSymName, 
								removeSpecs, addSpecs, false);
					}
					/////////////////////////////////////////////////////////////////////////////
					// Replace the gang-private variable with this new local private variable. //
					/////////////////////////////////////////////////////////////////////////////
					if( privSym instanceof AccessSymbol ) {
						TransformTools.replaceAccessExpressions(at, (AccessSymbol)privSym, lpriv_var);
					} else {
						TransformTools.replaceAll(at, new Identifier(privSym), lpriv_var);
					}
					//Reset below to be checked later.
					gpuBytes_stmt = null;
					orgGpuBytes_stmt = null;
				} else { //gang-private variable to be allocated on the GPU global memory through array expansion.
					/////////////////////////////////////////////////////////////
					// Create a GPU device variable corresponding to privSym   //
					// Ex: float * ggpriv__x; //GPU variable for gang-private  //
					/////////////////////////////////////////////////////////////
					// Give a new name for the device variable 
					gpu_priv_decl =  TransformTools.getGPUVariable(gpuGPSymName, targetSymbolTable, 
							typeSpecs, main_TrUnt, OpenACCHeaderEndMap, null);
					ggpriv_var = new Identifier((VariableDeclarator)gpu_priv_decl.getDeclarator(0));
					/////////////////////////////////////////////////////
					// Memory allocation for the device variable       //
					// Insert cudaMalloc() function before the region  //
					/////////////////////////////////////////////////////
					// Ex: cudaMalloc(((void *  * )( & ggpriv__x)), gpuBytes);
					FunctionCall malloc_call = new FunctionCall(new NameID("HI_tempMalloc1D"));
					List<Specifier> specs = new ArrayList<Specifier>(4);
					specs.add(Specifier.VOID);
					specs.add(PointerSpecifier.UNQUALIFIED);
					specs.add(PointerSpecifier.UNQUALIFIED);
					List<Expression> arg_list = new ArrayList<Expression>();
					arg_list.add(new Typecast(specs, new UnaryExpression(UnaryOperator.ADDRESS_OF, 
							(Identifier)ggpriv_var.clone())));
					SizeofExpression sizeof_expr = new SizeofExpression(typeSpecs);
					
					if( !allocatedGangPrivSet.contains(privSym) ) {
						insertMalloc = true;
						allocatedGangPrivSet.add(privSym);
					}
					
					if( !isArray && !isPointer ) { //scalar variable
						// Insert "gpuBytes = gpuNumBlocks * sizeof(varType);" statement 
						AssignmentExpression assignex = new AssignmentExpression((Identifier)cloned_bytes.clone(),
								AssignmentOperator.NORMAL, new BinaryExpression((Expression)numBlocks.clone(), 
										BinaryOperator.MULTIPLY, sizeof_expr.clone()));
						gpuBytes_stmt = new ExpressionStatement(assignex);
						AssignmentExpression assignex2 = new AssignmentExpression((Identifier)cloned_bytes.clone(),
								AssignmentOperator.NORMAL, sizeof_expr.clone());
						orgGpuBytes_stmt = new ExpressionStatement(assignex2);

						// Create a parameter Declaration for the kernel function
						// Change the scalar variable to a pointer type 
						// ex: float *lgpriv__x,
						boolean registerRO = false;
						if( regROCachingMap.keySet().contains(privSym) ) {
							registerRO = true;
						}
						VariableDeclarator pointerV_declarator = 
							scalarGangPrivConv(privSym, localGPSymName, typeSpecs, new_proc, region, scope, 
									registerRO, false, call_to_new_proc, ggpriv_var.clone());
						lpriv_var = new Identifier(pointerV_declarator);

					} else { //non-scalar variables
						// Insert "gpuBytes = gpuNumBlocks * (dimension1 * dimension2 * ..) 
						// * sizeof(varType);" statement
						Expression biexp = lengthList.get(0).clone();
						for( int i=1; i<dimsize; i++ )
						{
							biexp = new BinaryExpression(biexp, BinaryOperator.MULTIPLY, lengthList.get(i).clone());
						}
						BinaryExpression biexp2 = new BinaryExpression(biexp, BinaryOperator.MULTIPLY, sizeof_expr);
						AssignmentExpression assignex = new AssignmentExpression((Expression)cloned_bytes.clone(),
								AssignmentOperator.NORMAL, biexp2);
						orgGpuBytes_stmt = new ExpressionStatement(assignex);
						biexp = new BinaryExpression((Expression)numBlocks.clone(), 
								BinaryOperator.MULTIPLY, (Expression)biexp2.clone());
						assignex = new AssignmentExpression((Expression)cloned_bytes.clone(),AssignmentOperator.NORMAL, 
								biexp);
						gpuBytes_stmt = new ExpressionStatement(assignex);

						// Create a parameter Declaration for the kernel function
						// Create an extended array type
						// Ex1: "float lgpriv__x[][SIZE1]"
						// Ex2: "float lgpriv__x[][SIZE1][SIZE2]"
						VariableDeclarator arrayV_declarator =	arrayPrivConv(privSym, localGPSymName, typeSpecs, startList,
								lengthList, new_proc, region, scope, 0, call_to_new_proc, ggpriv_var.clone());
						lpriv_var = new Identifier(arrayV_declarator);

					}
					// Add gpuBytes argument to cudaMalloc() call
					arg_list.add((Identifier)cloned_bytes.clone());
                    arg_list.add(new NameID("acc_device_current"));
					malloc_call.setArguments(arg_list);
					ExpressionStatement malloc_stmt = new ExpressionStatement(malloc_call);
					// Insert malloc statement.
					if( insertMalloc ) {
						//mallocScope.addStatementBefore(confRefStmt, gpuBytes_stmt);
						//mallocScope.addStatementBefore(confRefStmt, malloc_stmt);
						prefixStmts.addStatement(gpuBytes_stmt);
						prefixStmts.addStatement(malloc_stmt);
						if( opt_addSafetyCheckingCode ) {
							// Insert "gpuGmemSize += gpuBytes;" statement 
							//mallocScope.addStatementBefore(gpuBytes_stmt.clone(), 
							//		(Statement)gMemAdd_stmt.clone());
							prefixStmts.addStatement(gMemAdd_stmt.clone());
						}
					}
				}

				if( insertMalloc ) {
					/*
					 * Insert cudaFree() to deallocate device memory for gang-private variable. 
					 * Because cuda-related statements are added in reverse order, 
					 * this function call is added first.
					 */
					//if( opt_addSafetyCheckingCode  ) {
					//	// Insert "gpuGmemSize -= gpuBytes;" statement 
					//	mallocScope.addStatementAfter(confRefStmt, gMemSub_stmt.clone());
					//}
					// Insert "cudaFree(ggpriv__x);"
					FunctionCall cudaFree_call = new FunctionCall(new NameID("HI_tempFree"));
					ArrayList<Specifier> specs = new ArrayList<Specifier>(4);
					specs.add(Specifier.VOID);
					specs.add(PointerSpecifier.UNQUALIFIED);
					specs.add(PointerSpecifier.UNQUALIFIED);
					cudaFree_call.addArgument(new Typecast(specs, new UnaryExpression(UnaryOperator.ADDRESS_OF,
							(Identifier)ggpriv_var.clone())));
					cudaFree_call.addArgument(new NameID("acc_device_current"));
					ExpressionStatement cudaFree_stmt = new ExpressionStatement(cudaFree_call);
					//mallocScope.addStatementAfter(confRefStmt, cudaFree_stmt);
					//mallocScope.addStatementAfter(confRefStmt, gpuBytes_stmt.clone());
					postscriptStmts.addStatement(gpuBytes_stmt.clone());
					postscriptStmts.addStatement(cudaFree_stmt);
					if( opt_addSafetyCheckingCode  ) {
						postscriptStmts.addStatement(gMemSub_stmt.clone());
					}
				}

				///////////////////////////////////////////////////////////////////
				// Load the value of host variable to the firstprivate variable. //
				///////////////////////////////////////////////////////////////////
				if( isGangPrivate && (FirstPrivSymSet != null) && FirstPrivSymSet.contains(privSym) ) {
					VariableDeclaration gfpriv_decl = null;
					Identifier gfpriv_var = null;
					Identifier lfpriv_var = null;
					String gpuFPSymName = "gfpriv__" + symNameBase;
					String localFPSymName = "lfpriv__" + symNameBase;
					//////////////////////////////////////////////////////////////////////////////
					// If firstprivate variable is scalar, the corresponding shared variable is //
					// passed as a kernel parameter instead of using GPU global memory, which   //
					// has the effect of caching it on the GPU Shared Memory.                   //
					//////////////////////////////////////////////////////////////////////////////
					if( !isArray && !isPointer ) { //scalar variable
						// Create a GPU kernel parameter corresponding to privSym
						// ex: float lfpriv__x;
						VariableDeclarator lfpriv_declarator = new VariableDeclarator(new NameID(localFPSymName));
						VariableDeclaration lfpriv_decl = new VariableDeclaration(typeSpecs, 
								lfpriv_declarator);
						lfpriv_var = new Identifier(lfpriv_declarator);
						new_proc.addDeclaration(lfpriv_decl);

						// Insert argument to the kernel function call
						if( privSym instanceof AccessSymbol ) {
							AccessExpression accExp = AnalysisTools.accessSymbolToExpression((AccessSymbol)privSym, null);
							call_to_new_proc.addArgument(accExp);
						} else {
							call_to_new_proc.addArgument(new Identifier(privSym));
						}
						
						///////////////////////////////////////////////////////////////////////////////
						// Load the value of the passed shared variable to the firstprivate variable //
						///////////////////////////////////////////////////////////////////////////////
						Statement estmt = null;
						if( gangPrivCachingOnShared || (isWorkerPrivate && !workerPrivCachingOnShared) ) { 
							//worker-private on a local memory or gang-private cached on Shared memory.
							//No array expansion is used.
							//ex: lgpriv__x = lfpriv__x;
							estmt = new ExpressionStatement(new AssignmentExpression(lpriv_var.clone(), 
									AssignmentOperator.NORMAL, lfpriv_var.clone()));
						} else if( isWorkerPrivate && workerPrivCachingOnShared ) {
							//worker-private is cached on a shared memory using array expansion.
							//ex: lpriv__x[_tid] = lfpriv__x;
							estmt = new ExpressionStatement(new AssignmentExpression(
									new ArrayAccess(lpriv_var.clone(), SymbolTools.getOrphanID("_tid")), 
									AssignmentOperator.NORMAL, lfpriv_var.clone()));
							
						} else { //gang-private allocated on the global memory using array expansion.
							//ex: lpriv__x[_bid] = lfpriv__x;
							estmt = new ExpressionStatement(new AssignmentExpression(
									new ArrayAccess(lpriv_var.clone(), SymbolTools.getOrphanID("_bid")), 
									AssignmentOperator.NORMAL, lfpriv_var.clone()));
						}
						if( isWorkerPrivate ) {
							if( at == region ) {
								if( at instanceof ForLoop ) {
									preList.add(estmt);
								} else {
									Tools.exit("[ERROR in CUDATranslationTools.privateTransformation()] " +
											"unexpected worker-level firstprivate variable for a compound statement; exit!\n" +
											"OpenACC Annotation: " + pannot + "\n" +
											"Enclosing procedure: " + cProc.getSymbolName() + "\n");
								}
							} else {
								CompoundStatement atP = (CompoundStatement)at.getParent();
								atP.addStatementBefore((Statement)at, estmt);
							}
						} else if( isGangPrivate ) {
							if( region instanceof ForLoop ) {
								preList.add(estmt);
							} else {
								Statement last_decl_stmt;
								last_decl_stmt = IRTools.getLastDeclarationStatement(scope);
								if( last_decl_stmt != null ) {
									scope.addStatementAfter(last_decl_stmt, estmt);
								} else {
									last_decl_stmt = (Statement)scope.getChildren().get(0);
									scope.addStatementBefore(last_decl_stmt, estmt);
								}
							}
						}
					} else { //non-scalar variable
						////////////////////////////////////////////////////////////////////////////////////////
						// Create a GPU device variable to carry initial values of the firstprivate variable. //
						// Ex: float * gfpriv__x; //GPU variable for gang-firstprivate                        //
						////////////////////////////////////////////////////////////////////////////////////////
						// Give a new name for the device variable 
						gfpriv_decl =  TransformTools.getGPUVariable(gpuFPSymName, targetSymbolTable, 
								typeSpecs, main_TrUnt, OpenACCHeaderEndMap, null);
						gfpriv_var = new Identifier((VariableDeclarator)gfpriv_decl.getDeclarator(0));
						
						if( !allocatedGangPrivSet.contains(privSym) ) {
							insertMalloc = true;
							allocatedGangPrivSet.add(privSym);
						}

						/////////////////////////////////////////////////////////////////////////
						// Memory allocation for the device variable                           //
						/////////////////////////////////////////////////////////////////////////
						// - Insert cudaMalloc() function before the region.                   //
						// Ex: cudaMalloc(((void *  * )( & gfpriv__x)), gpuBytes);             //
						/////////////////////////////////////////////////////////////////////////
						FunctionCall malloc_call = new FunctionCall(new NameID("HI_tempMalloc1D"));
						List<Specifier> specs = new ArrayList<Specifier>(4);
						specs.add(Specifier.VOID);
						specs.add(PointerSpecifier.UNQUALIFIED);
						specs.add(PointerSpecifier.UNQUALIFIED);
						List<Expression> arg_list = new ArrayList<Expression>();
						arg_list.add(new Typecast(specs, new UnaryExpression(UnaryOperator.ADDRESS_OF, 
								(Identifier)gfpriv_var.clone())));
						SizeofExpression sizeof_expr = new SizeofExpression(typeSpecs);
						// Insert "gpuBytes = (dimension1 * dimension2 * ..) * sizeof(varType);" statement

						// Add malloc size (gpuBytes) statement
						// Ex: gpuBytes=(((2048+2)*(2048+2))*sizeof (float));
						if( orgGpuBytes_stmt == null ) {
							Expression biexp = lengthList.get(0).clone();
							for( int i=1; i<dimsize; i++ )
							{
								biexp = new BinaryExpression(biexp, BinaryOperator.MULTIPLY, lengthList.get(i).clone());
							}
							BinaryExpression biexp2 = new BinaryExpression(biexp, BinaryOperator.MULTIPLY, sizeof_expr);
							AssignmentExpression assignex = new AssignmentExpression((Expression)cloned_bytes.clone(),
									AssignmentOperator.NORMAL, biexp2);
							orgGpuBytes_stmt = new ExpressionStatement(assignex);
						}
						// Add gpuBytes argument to cudaMalloc() call
						arg_list.add((Identifier)cloned_bytes.clone());
                        arg_list.add(new NameID("acc_device_current"));
						malloc_call.setArguments(arg_list);
						ExpressionStatement malloc_stmt = new ExpressionStatement(malloc_call);
						if( insertMalloc ) {
							//mallocScope.addStatementBefore(confRefStmt, orgGpuBytes_stmt.clone());
							//mallocScope.addStatementBefore(confRefStmt, malloc_stmt);
							prefixStmts.addStatement(orgGpuBytes_stmt.clone());
							prefixStmts.addStatement(malloc_stmt);
							if( opt_addSafetyCheckingCode ) {
								// Insert "gpuGmemSize += gpuBytes;" statement 
								//mallocScope.addStatementBefore(orgGpuBytes_stmt.clone(), 
								//		(Statement)gMemAdd_stmt.clone());
								prefixStmts.addStatement(gMemAdd_stmt.clone());
							}
						}

						/////////////////////////////////////////////////////////////
						// Create a parameter Declaration for the kernel function. //
						// Keep the original array type, but change name           //
						// Ex: "float lfpriv_b[(2048+2)][(2048+2)]"                //
						/////////////////////////////////////////////////////////////
						List edimensions = new LinkedList();
						edimensions.add(null);
						for( int i=1; i<dimsize; i++ )
						{
							edimensions.add(lengthList.get(i).clone());
						}
						ArraySpecifier easpec = new ArraySpecifier(edimensions);
						VariableDeclarator lfpriv_declarator = new VariableDeclarator(new NameID(localFPSymName), easpec);
						VariableDeclaration lfpriv_decl = 
							new VariableDeclaration(typeSpecs, lfpriv_declarator); 
						lfpriv_var = new Identifier(lfpriv_declarator);
						new_proc.addDeclaration(lfpriv_decl);
						
						//////////////////////////////////////////////////
						// Insert argument to the kernel function call. //
						//////////////////////////////////////////////////
						if( dimsize == 1 ) {
							// Simply pass address of the pointer
							// Ex:  "gfpriv_x"
							call_to_new_proc.addArgument((Identifier)gfpriv_var.clone());
						} else {
							//Cast the gpu variable to pointer-to-array type 
							// Ex: (float (*)[dimesion2]) gfpriv_x
							List castspecs = new LinkedList();
							castspecs.addAll(typeSpecs);
							/*
							 * FIXME: NestedDeclarator was used for (*)[SIZE2], but this may not be 
							 * semantically correct way to represent (*)[SIZE2] in IR.
							 */
							List tindices = new LinkedList();
							for( int i=1; i<dimsize; i++) {
								tindices.add(lengthList.get(i).clone());
							}
							ArraySpecifier aspec = new ArraySpecifier(tindices);
							List tailSpecs = new ArrayList(1);
							tailSpecs.add(aspec);
							VariableDeclarator childDeclr = new VariableDeclarator(PointerSpecifier.UNQUALIFIED, new NameID(""));
							NestedDeclarator nestedDeclr = new NestedDeclarator(new ArrayList(), childDeclr, null, tailSpecs);
							castspecs.add(nestedDeclr);
							call_to_new_proc.addArgument(new Typecast(castspecs, (Identifier)gfpriv_var.clone()));
						}

						///////////////////////////////////////////////////////////////////////////////
						// Load the value of the passed shared variable to the firstprivate variable //
						// TODO: loading gang-private variable can be parallelized.                  //
						// FIXME: loading gang-private variable should be synchornized after.        //
						///////////////////////////////////////////////////////////////////////////////
						///////////////////////////////////////////////////////////////////////////////////////////
						// Ex1: No array expansion used (worker-private on a local memory or gang-private cached //
						//      on shared memory)                                                                //
						//      for(i=0; i<SIZE1; i++) {                                                         //
						//         for(k=0; k<SIZE2; k++) {                                                      //
						//             lpriv_x[i][k] = lfpriv_x[i][k];                                           //
						//         }                                                                             //
						//      }                                                                                //
						// Ex2: array expansion used (worker-private cached on the shared memory)                //
						//      for(i=0; i<SIZE1; i++) {                                                         //
						//         for(k=0; k<SIZE2; k++) {                                                      //
						//             lpriv_x[i][k][_tid] = lfpriv_x[i][k];                                     //
						//         }                                                                             //
						//      }                                                                                //
						// Ex3: array expansion used (worker-private allocated on the global memory)             //
						//      for(i=0; i<SIZE1; i++) {                                                         //
						//         for(k=0; k<SIZE2; k++) {                                                      //
						//             lpriv_x[_gtid][i][k] = lfpriv_x[i][k];                                    //
						//         }                                                                             //
						//      }                                                                                //
						// Ex4: array expansion used (gang-private allocated on the global memory)               //
						//      for(i=0; i<SIZE1; i++) {                                                         //
						//         for(k=0; k<SIZE2; k++) {                                                      //
						//             lpriv_x[_bid][i][k] = lfpriv_x[i][k];                                     //
						//         }                                                                             //
						//      }                                                                                //
						///////////////////////////////////////////////////////////////////////////////////////////
						//////////////////////////////////////// //////
						// Create or find temporary index variables. // 
						//////////////////////////////////////// //////
						List<Identifier> index_vars = new LinkedList<Identifier>();
						CompoundStatement tScope = scope;
						if( isWorkerPrivate ) {
							if( at != region ) {
								tScope = (CompoundStatement)at.getParent();
							}
						}
						for( int i=0; i<dimsize; i++ ) {
							index_vars.add(TransformTools.getTempIndex(tScope, tempIndexBase+i));
						}
						List<Expression> indices1 = new LinkedList<Expression>();
						List<Expression> indices2 = new LinkedList<Expression>();
						if( !isWorkerPrivate && !gangPrivCachingOnShared ) { 
							indices1.add(SymbolTools.getOrphanID("_bid"));
						} else if( isWorkerPrivate && workerPrivOnGlobal ) { 
							indices1.add(SymbolTools.getOrphanID("_gtid"));
						} 
						for( int k=0; k<dimsize; k++ ) {
							indices1.add((Expression)index_vars.get(k).clone());
							indices2.add((Expression)index_vars.get(k).clone());
						}
						if( isWorkerPrivate && workerPrivCachingOnShared && !workerPrivOnGlobal ) { 
							indices1.add(SymbolTools.getOrphanID("_tid"));
						} 
						Expression LHS = new ArrayAccess(lpriv_var.clone(), indices1);
						Expression RHS = new ArrayAccess(lfpriv_var.clone(), indices2);
						Statement estmt = TransformTools.genArrayCopyLoop(index_vars, lengthList, LHS, RHS);
						if( isWorkerPrivate ) {
							if( at == region ) {
								if( at instanceof ForLoop ) {
									preList.add(estmt);
								} else {
									Tools.exit("[ERROR in CUDATranslationTools.privateTransformation()] " +
											"unexpected worker-level firstprivate variable for a compound statement; exit!\n" +
											"OpenACC Annotation: " + pannot + "\n" +
											"Enclosing procedure: " + cProc.getSymbolName() + "\n");
								}
							} else {
								CompoundStatement atP = (CompoundStatement)at.getParent();
								atP.addStatementBefore((Statement)at, estmt);
							}
						} else if( isGangPrivate ) {
							if( region instanceof ForLoop ) {
								preList.add(estmt);
							} else {
								Statement last_decl_stmt;
								last_decl_stmt = IRTools.getLastDeclarationStatement(scope);
								if( last_decl_stmt != null ) {
									scope.addStatementAfter(last_decl_stmt, estmt);
								} else {
									last_decl_stmt = (Statement)scope.getChildren().get(0);
									scope.addStatementBefore(last_decl_stmt, estmt);
								}
							}
						}

						if( insertMalloc ) {
							/*
							 * Insert cudaFree() to deallocate device memory. 
							 * Because cuda-related statements are added in reverse order, 
							 * this function call is added first.
							 */
							//if( opt_addSafetyCheckingCode  ) {
							//	// Insert "gpuGmemSize -= gpuBytes;" statement 
							//	mallocScope.addStatementAfter(confRefStmt, gMemSub_stmt.clone());
							//}
							// Insert "cudaFree(gfpriv__x);"
							FunctionCall cudaFree_call = new FunctionCall(new NameID("HI_tempFree"));
							specs = new ArrayList<Specifier>(4);
							specs.add(Specifier.VOID);
							specs.add(PointerSpecifier.UNQUALIFIED);
							specs.add(PointerSpecifier.UNQUALIFIED);
							cudaFree_call.addArgument(new Typecast(specs, new UnaryExpression(UnaryOperator.ADDRESS_OF,
									(Identifier)gfpriv_var.clone())));
                            cudaFree_call.addArgument(new NameID("acc_device_current"));
							ExpressionStatement cudaFree_stmt = new ExpressionStatement(cudaFree_call);
							//mallocScope.addStatementAfter(confRefStmt, cudaFree_stmt);
							//mallocScope.addStatementAfter(confRefStmt, orgGpuBytes_stmt.clone());
							postscriptStmts.addStatement(orgGpuBytes_stmt.clone());
							postscriptStmts.addStatement(cudaFree_stmt);
							if( opt_addSafetyCheckingCode  ) {
								postscriptStmts.addStatement(gMemSub_stmt.clone());
							}
							
							/* Insert memory copy function from CPU to GPU */
							CompoundStatement ifBody = new CompoundStatement();
							IfStatement ifStmt = null;
							if( ifCond != null ) {
								ifStmt = new IfStatement(ifCond.clone(), ifBody);
							}
							if( confRefStmt != region ) {
								if( ifCond == null ) {
									((CompoundStatement)region.getParent()).addStatementBefore(region, 
											orgGpuBytes_stmt.clone());
								} else {
									ifBody.addStatement(orgGpuBytes_stmt.clone());
								}
							} else { //duplicated insert; ignore this.
								//prefixStmts.addStatement(orgGpuBytes_stmt.clone());
							}

							/////////////////////////////////////////////////////////////////////////////////////////
							// HI_memcpy(gpuPtr, hostPtr, gpuBytes, HI_MemcpyHostToDevice, 0);               //
							// HI_memcpy_async(gpuPtr, hostPtr, gpuBytes, HI_MemcpyHostToDevice, 0, asyncID);//
							/////////////////////////////////////////////////////////////////////////////////////////
							FunctionCall copyinCall = null;
							if( asyncID == null ) {
								copyinCall = new FunctionCall(new NameID("HI_memcpy"));
							} else {
								copyinCall = new FunctionCall(new NameID("HI_memcpy_async"));
							}
							List<Expression> arg_list2 = new ArrayList<Expression>();
							arg_list2.add((Identifier)gfpriv_var.clone());
							if( privSym instanceof AccessSymbol ) {
								AccessExpression accExp = AnalysisTools.accessSymbolToExpression((AccessSymbol)privSym,null);
								arg_list2.add(accExp);
							} else {
								arg_list2.add(new Identifier(privSym));
							}
							arg_list2.add((Identifier)cloned_bytes.clone());
							arg_list2.add(new NameID("HI_MemcpyHostToDevice"));
							arg_list2.add(new IntegerLiteral(0));
							if( asyncID != null ) {
								arg_list2.add(asyncID.clone());
							}
							copyinCall.setArguments(arg_list2);
							Statement copyin_stmt = new ExpressionStatement(copyinCall);
							if( confRefStmt != region ) {
								if( ifCond != null ) {
									ifBody.addStatement(copyin_stmt);
									((CompoundStatement)region.getParent()).addStatementBefore(region, ifStmt);
								} else {
									((CompoundStatement)region.getParent()).addStatementBefore(region, copyin_stmt);
								}
							} else {
								prefixStmts.addStatement(copyin_stmt);
							}
						}
					}
				} //end of firstprivate translation loop.
			}
		}
		
        if( !localGangPrivateSymbolsAll.isEmpty() ) {
			//Put any implicit local gang-private variables not included in any OpenACC private clause
			//in CUDA shared memory; the only exception is when the local symbol is an index variable of gang loop.
			//[DEBUG] this may not work if the local variable is too big.
			///////////////////////////////////////////////////////////////////////////////////
			//A gang private variable is declared either in the enclosing compound statement //
			//or in the innermost gang loop body if region is a loop.                        //
			///////////////////////////////////////////////////////////////////////////////////
			if( region instanceof ForLoop ) {
				ACCAnnotation tAnnot = AnalysisTools.findInnermostPragma(region, ACCAnnotation.class, "gang");
				ForLoop gLoop = (ForLoop)tAnnot.getAnnotatable();
				scope = (CompoundStatement)(gLoop).getBody();
			} else if( region instanceof CompoundStatement ) {
				scope = (CompoundStatement)region;
			}
			for( Symbol lgSym : localGangPrivateSymbolsAll ) {
				if( loopIndexSymbols.contains(lgSym) ) {
					continue;
				}
				Declaration decl = lgSym.getDeclaration();
				if( decl != null ) {
					if( decl instanceof VariableDeclaration ) {
						VariableDeclaration vaDecl = (VariableDeclaration)decl;
						boolean ispointer = false;
						List<Specifier> symspecs = null;
						if( lgSym instanceof VariableDeclarator ) {
							symspecs = ((VariableDeclarator)lgSym).getSpecifiers();
						} else if( lgSym instanceof NestedDeclarator ) {
							Declarator nestedSym = ((NestedDeclarator)lgSym).getDeclarator();
							if( nestedSym instanceof VariableDeclarator) {
								symspecs = ((VariableDeclarator)nestedSym).getSpecifiers();
							}
						}
						if( symspecs == null ) {
                        	Tools.exit("[ERROR in OpenCLTranslation.privateTransformation()] error in handling local," +
                        			" implicit gang-private variable: " + lgSym);
						}
						for(Specifier tspec : symspecs) {
							if( tspec instanceof PointerSpecifier ) {
								ispointer = true;
								break;
							}
						}
						if( ispointer ) {
							if( !symspecs.contains(CUDASpecifier.CUDA_SHARED) ) {
								symspecs.add(CUDASpecifier.CUDA_SHARED);
							}
						} else {
							List<Specifier> specs = vaDecl.getSpecifiers();
							if( !specs.contains(CUDASpecifier.CUDA_SHARED) ) {
								specs.add(0, CUDASpecifier.CUDA_SHARED);
							}
						}
						Declarator declr = vaDecl.getDeclarator(0);
						Traversable parent = decl.getParent(); //parent should be DeclarationStatement.
						CompoundStatement cStmt = null;
						if( parent instanceof DeclarationStatement ) {
							cStmt = (CompoundStatement)parent.getParent();
						} else {
							Tools.exit("[ERROR in CUDATranslation.privateTransformation()] error in handling local," +
									" implicit gang-private variable: " + lgSym);
						}
						Initializer lsm_init = declr.getInitializer();
						if( lsm_init != null ) {
							//CUDA shared variable cannot have initialization at its declaration statement.
							Object initObj = lsm_init.getChildren().get(0);
							if( initObj instanceof Expression ) {
								Expression initValue = (Expression)initObj;
								declr.setInitializer(null);
								initValue.setParent(null);
								AssignmentExpression lAssignExp = new AssignmentExpression(new Identifier(lgSym),AssignmentOperator.NORMAL,
										initValue);
								Statement lAssignStmt = new ExpressionStatement(lAssignExp);
								Statement fStmt = IRTools.getFirstNonDeclarationStatement(cStmt);
								if( fStmt == null ) {
									cStmt.addStatement(lAssignStmt);
								} else {
									cStmt.addStatementBefore(fStmt, lAssignStmt);
								}
							} else {
								Tools.exit("The following gang-private variable can not be alloced on the CUDA shared memory, since CUDA" +
										"does not allow initialization in the CUDA shared variable declaration, and the following" +
										"variable has inseparable initialization: "+ lgSym);
							}
						}
						if( localGangPrivateSymbols.contains(lgSym) ) {
							//Move the declaration statement into the enclosing compute region if symbols with the sam name
							//does not exist.
							if( !AnalysisTools.containsSymbol(scope.getSymbols(), lgSym.getSymbolName())) {
								cStmt.removeChild(parent);
								decl.setParent(null);
								//parent.removeChild(decl); //disallowed.
								scope.addDeclaration(decl);
							}
						}
					}
				}
			}
		}
		PrintTools.println("[privateTransformation() ends] current procedure: " + cProc.getSymbolName() +
				"\ncompute region type: " + cRegionKind + "\n", 2);
	}
	
	protected static void textureConv(Symbol sharedSym, Identifier textureRefID, Statement region, Expression offset ) {
		if( sharedSym instanceof AccessSymbol ) {
			List<Expression> matches = new ArrayList<Expression>(4);
			DFIterator<Expression> iter =
					new DFIterator<Expression>(region, Expression.class);
			while (iter.hasNext()) {
				Expression child = iter.next();
				if (SymbolTools.getSymbolOf(child).equals(sharedSym)) {
					matches.add(child);
				}
			}
			for (int i = 0; i < matches.size(); i++) {
				Expression match = matches.get(i);
				Traversable parent = match.getParent();
				if (parent instanceof AccessExpression &&
						match == ((AccessExpression)parent).getRHS()) {
					/* don't replace these */
				} else {
					Expression tExp = match;
					while( tExp instanceof AccessExpression ) {
						tExp = ((AccessExpression)tExp).getRHS();
					}
					if( tExp instanceof ArrayAccess ) {
						ArrayAccess aAccess = (ArrayAccess)tExp;
						if( aAccess.getNumIndices() == 1 ) {
							Expression indexExp = null;
							if( offset == null ) {
								indexExp = aAccess.getIndex(0).clone();
							} else {
								indexExp = new BinaryExpression(aAccess.getIndex(0).clone(), 
										BinaryOperator.ADD, offset.clone());
							}
							FunctionCall texAccessCall = new FunctionCall(new NameID("tex1Dfetch"));
							texAccessCall.addArgument((Identifier)textureRefID.clone());
							texAccessCall.addArgument(indexExp);
							match.swapWith(texAccessCall);
						}
					}
				}
			}
		} else {
			Identifier hostVar = new Identifier(sharedSym);
			DepthFirstIterator iter = new DepthFirstIterator(region);
			for (;;)
			{
				ArrayAccess aAccess = null;

				try {
					aAccess = (ArrayAccess)iter.next(ArrayAccess.class);
				} catch (NoSuchElementException e) {
					break;
				}
				Expression arrayID = aAccess.getArrayName();
				if( arrayID.equals(hostVar) ) {
					Expression indexExp = null;
					if( offset == null ) {
						indexExp = aAccess.getIndex(0).clone();
					} else {
						indexExp = new BinaryExpression(aAccess.getIndex(0).clone(), 
								BinaryOperator.ADD, offset.clone());
					}
					FunctionCall texAccessCall = new FunctionCall(new NameID("tex1Dfetch"));
					texAccessCall.addArgument((Identifier)textureRefID.clone());
					texAccessCall.addArgument(indexExp);
					aAccess.swapWith(texAccessCall);
				}
			}
		}
	}
	
	protected static void pitchedAccessConv(Symbol sharedSym, Identifier kParamVar, List<Specifier> typeSpecs,
			Identifier pitchVar, Statement region) {
		/* 
		 * If MallocPitch is used to allocate 2 dimensional array, gpu_a,
		 * replace array access expression with pointer access expression with pitch
		 * Ex: gpu__a[i][k] => *((float *)((char *)gpu__a + i * pitch__a) + k)
		 */
		if( sharedSym instanceof AccessSymbol ) {
			List<Expression> matches = new ArrayList<Expression>(4);
			DFIterator<Expression> iter =
					new DFIterator<Expression>(region, Expression.class);
			while (iter.hasNext()) {
				Expression child = iter.next();
				if (SymbolTools.getSymbolOf(child).equals(sharedSym)) {
					matches.add(child);
				}
			}
			for (int i = 0; i < matches.size(); i++) {
				Expression match = matches.get(i);
				Traversable parent = match.getParent();
				if (parent instanceof AccessExpression &&
						match == ((AccessExpression)parent).getRHS()) {
					/* don't replace these */
				} else {
					Expression tExp = match;
					while( tExp instanceof AccessExpression ) {
						tExp = ((AccessExpression)tExp).getRHS();
					}
					if( tExp instanceof ArrayAccess ) {
						ArrayAccess aAccess = (ArrayAccess)tExp;
						if( aAccess.getNumIndices() < 2 ) {
							Tools.exit("[ERROR in pitchedAccessConv()] an array access (" + aAccess + 
									") uses only the first dimension, and current translator can not convert this " +
									"to a pitched access correctly; please turn off \"useMallocPitch\" option.");
						}
						Expression pAccess = convArray2Pointer(aAccess, kParamVar, 
								pitchVar, typeSpecs); 
						aAccess.swapWith(pAccess);
						match.swapWith(pAccess);
					}
				}
			}
		} else {
			Identifier hostVar = new Identifier(sharedSym);
			DepthFirstIterator iter = new DepthFirstIterator(region);
			for (;;)
			{
				ArrayAccess aAccess = null;

				try {
					aAccess = (ArrayAccess)iter.next(ArrayAccess.class);
				} catch (NoSuchElementException e) {
					break;
				}
				IDExpression arrayID = (IDExpression)aAccess.getArrayName();
				if( arrayID.equals(hostVar) ) {
					if( aAccess.getNumIndices() < 2 ) {
						Tools.exit("[ERROR in pitchedAccessConv()] an array access (" + aAccess + 
								") uses only the first dimension, and current translator can not convert this " +
								"to a pitched access correctly; please turn off \"useMallocPitch\" option.");
					}
					Expression pAccess = convArray2Pointer(aAccess, kParamVar,
							pitchVar, typeSpecs); 
					aAccess.swapWith(pAccess);
				}
			}
		}

	}
	
	protected static void worksharingLoopTransformation(Procedure cProc, CompoundStatement kernelRegion, 
			Statement region, String cRegionKind,
			int defaultNumWorkers, boolean opt_skipKernelLoopBoundChecking, boolean isSingleTask) {
		PrintTools.println("[worksharingLoopTransformation() begins]", 2);
		List<ACCAnnotation> lAnnots = AnalysisTools.ipCollectPragmas(kernelRegion, ACCAnnotation.class, "loop", null);
		if( lAnnots == null ) {
			return;
		}
		for( ACCAnnotation lAnnot : lAnnots ) {
			Annotatable at = lAnnot.getAnnotatable();
			if( at instanceof ForLoop ) {
				Expression num_gangs = null;
				Expression num_workers = null;
				ForLoop ploop = (ForLoop)at;
				ACCAnnotation tAnnot = at.getAnnotation(ACCAnnotation.class, "gang");
				boolean isGangLoop = false;
				boolean outermostloop = false;
				if( tAnnot != null ) {
					isGangLoop = true;
					outermostloop = true;
					Traversable tt = at.getParent();
					while( tt != null ) {
						if( (tt instanceof Annotatable) && ((Annotatable)tt).containsAnnotation(ACCAnnotation.class, "gang") ) {
							outermostloop = false;
							break;
						}
						tt = tt.getParent();
					}
				}
				long gangdim = 1;
				if( isGangLoop && cRegionKind.equals("kernels")) {
					tAnnot = at.getAnnotation(ACCAnnotation.class, "gangdim");
					if( tAnnot != null ) {
						gangdim = ((IntegerLiteral)tAnnot.get("gangdim")).getValue();
					} else {
						Tools.exit("[ERROR in CUDATranslationTools.worksharingLoopTransformation()] internal gangdim clause is missing.\n" +
								"Enclosing Procedure: " + cProc.getSymbolName() + "\nACCAnnotation: " + lAnnot + "\n");
					}
				}
				tAnnot = at.getAnnotation(ACCAnnotation.class, "worker");
				boolean isWorkerLoop = false;
				if( tAnnot != null ) {
					isWorkerLoop = true;
				}
				long workerdim = 1;
				if( isWorkerLoop && cRegionKind.equals("kernels")) {
					tAnnot = at.getAnnotation(ACCAnnotation.class, "workerdim");
					if( tAnnot != null ) {
						workerdim = ((IntegerLiteral)tAnnot.get("workerdim")).getValue();
					} else {
						Tools.exit("[ERROR in CUDATranslationTools.worksharingLoopTransformation()] internal workerdim clause is missing.\n" +
								"Enclosing Procedure: " + cProc.getSymbolName() + "\nACCAnnotation: " + lAnnot + "\n");
					}
				}
				boolean isSeqKernelLoop = false;
				if( !isGangLoop && !isWorkerLoop ) {
					tAnnot = at.getAnnotation(ACCAnnotation.class, "seq");
					if( (tAnnot != null) && (at == region) ) {
						isSeqKernelLoop = true;
					} else {
						continue; //non-worker/gang loop is skipped.
					}
				}
				
				
				CompoundStatement loopbody = (CompoundStatement)ploop.getBody();
				boolean lexicallyIncluded = false;
				Traversable tt = ploop;
				while( tt != null ) {
					if( tt instanceof Procedure ) {
						break;
					} else if( tt.equals(region) ) {
						lexicallyIncluded = true;
						break;
					} else {
						tt = tt.getParent();
					}
				}

				Set<Symbol> localSymbols = SymbolTools.getVariableSymbols(loopbody);
				//[FIXME] Below will not work since if multiple local variables with the same name exist; 
				//temporarily disabled.
				//Set<Symbol> localSymbols = SymbolTools.getLocalSymbols(loopbody);
				for( Symbol sm : localSymbols ) {
/*					Declaration lsm_decl = SymbolTools.findSymbol((SymbolTable)loopbody, 
							new Identifier(sm));*/
					String smName = sm.getSymbolName();
					Declaration lsm_decl = sm.getDeclaration();
					DeclarationStatement lsm_stmt = (DeclarationStatement)lsm_decl.getParent();
					CompoundStatement cParent = (CompoundStatement)lsm_stmt.getParent();
					//
					// If local variable declared in a parallel for-loop has initial value depending on the enclosing loop,
					// the initilization should be separated from the declaration statement for correct O2G translation. 
					// For simplicity, if the initialization statement has initialization, the initialization expression 
					// is always separated unless it is const variable.
					//
					if( lsm_decl instanceof VariableDeclaration ) {
						Declarator lsm_declarator = ((VariableDeclaration)lsm_decl).getDeclarator(0);
						Initializer lsm_init = lsm_declarator.getInitializer();
						// Const variable should be initialized at declaration; don't separate initialization from separation.
						// For now, declaration of const variable is not moved to the beginning of kernel region.
						List lspecs = new ChainedList();
						lspecs.addAll(((VariableDeclaration)lsm_decl).getSpecifiers());
						List declrSpecs = new ChainedList();
						declrSpecs.addAll(lsm_declarator.getSpecifiers());
						if( declrSpecs.contains(PointerSpecifier.CONST) || declrSpecs.contains(PointerSpecifier.CONST_RESTRICT) || 
								declrSpecs.contains(PointerSpecifier.CONST_RESTRICT_VOLATILE) || declrSpecs.contains(PointerSpecifier.CONST_VOLATILE) ||
								(lspecs.contains(Specifier.CONST)&&!SymbolTools.isPointer(sm)) || (lspecs.contains(Specifier.STATIC)) ) {
							//Initialization of constant/static variable/pointer should not be separated.
							//System.err.println("Found constant/static variable/pointer: " + decl);
							continue;
						} else {
							if( !lexicallyIncluded ) {
								continue;
							}
							if( lsm_init == null ) {
								cParent.removeChild(lsm_stmt);
							} else {
								int listSize = lsm_init.getChildren().size();
								if( listSize == 1 ) {
									Object initObj = lsm_init.getChildren().get(0);
									if( initObj instanceof Expression ) {
										lsm_declarator.setInitializer(null);
										Expression initValue = (Expression)initObj;
										initValue.setParent(null);
										AssignmentExpression lAssignExp = new AssignmentExpression(new Identifier(sm),AssignmentOperator.NORMAL,
												initValue);
										ExpressionStatement lAssignStmt = new ExpressionStatement(lAssignExp);
										lsm_stmt.swapWith(lAssignStmt);
									} else {
										cParent.removeChild(lsm_stmt);
									}
								} else {
									cParent.removeChild(lsm_stmt);
								}
							}
							Set<Symbol> kRegionSymbols = SymbolTools.getVariableSymbols(kernelRegion);
							Symbol tSym = AnalysisTools.findsSymbol(kRegionSymbols, smName);
							if( tSym == null ) {
								lsm_decl.setParent(null);
								kernelRegion.addDeclaration(lsm_decl);
							} else {
								//Multiple symbols with the same name exist in the kernelRegion.
								Declaration tSymDecl = tSym.getDeclaration();
								if( !tSymDecl.toString().equals(lsm_decl.toString()) ) {
									//If two symbols are different types, rename the new symbol.
									int tcnt = 0;
									String newSmName = smName+"_"+tcnt;
									while(AnalysisTools.findsSymbol(kRegionSymbols, newSmName) != null) {
										tcnt++;
										newSmName = smName+"_"+tcnt;
									}
									cParent.addStatement(lsm_stmt);
									SymbolTools.setSymbolName(sm, newSmName, cParent);
									cParent.removeChild(lsm_stmt);
									lsm_decl.setParent(null);
									kernelRegion.addDeclaration(lsm_decl);
								}
							}
						}
					} else {
						//DEBUG: unexpected symbol type
						Set<Symbol> kRegionSymbols = SymbolTools.getVariableSymbols(kernelRegion);
						Symbol tSym = AnalysisTools.findsSymbol(kRegionSymbols, smName);
						if( tSym == null ) {
							cParent.removeChild(lsm_stmt);
							lsm_decl.setParent(null);
							kernelRegion.addDeclaration(lsm_decl);
						} else {
							//Multiple symbols with the same name exist in the kernelRegion.
							Declaration tSymDecl = tSym.getDeclaration();
							if( !tSymDecl.toString().equals(lsm_decl.toString()) ) {
								//If two symbols are different types, rename the new symbol.
								int tcnt = 0;
								String newSmName = smName+"_"+tcnt;
								while(AnalysisTools.findsSymbol(kRegionSymbols, newSmName) != null) {
									tcnt++;
									newSmName = smName+"_"+tcnt;
								}
								SymbolTools.setSymbolName(sm, newSmName, cParent);
								cParent.removeChild(lsm_stmt);
								lsm_decl.setParent(null);
								kernelRegion.addDeclaration(lsm_decl);
							}
						}
					}
				}
				//[DEBUG] this will cause too many temporary warnings; disable this.
/*				if( !localSymbols.isEmpty() ) {
					SymbolTools.linkSymbol(ploop);
				}*/
				
				if( isSeqKernelLoop || isSingleTask ) {
					//return;
					continue;
				}
				
				if( cRegionKind.equals("parallel") ) {
					tAnnot = kernelRegion.getAnnotation(ACCAnnotation.class, "num_gangs");
					if( tAnnot != null ) {
						num_gangs = tAnnot.get("num_gangs");
					}
					tAnnot = kernelRegion.getAnnotation(ACCAnnotation.class, "num_workers");
					if( tAnnot != null ) {
						num_workers = tAnnot.get("num_workers");
					} else {
						num_workers = new IntegerLiteral(defaultNumWorkers);
					}
				} else {
					tAnnot = at.getAnnotation(ACCAnnotation.class, "gang");
					if( tAnnot != null ) {
						num_gangs = tAnnot.get("gang");
					}
					tAnnot = at.getAnnotation(ACCAnnotation.class, "worker");
					if( tAnnot != null ) {
						num_workers = tAnnot.get("worker");
					} else {
						num_workers = new IntegerLiteral(defaultNumWorkers);
					}
				}
				if( num_workers != null ) {
					num_workers = Symbolic.simplify(num_workers);
				}
				if( num_gangs != null ) {
					num_gangs = Symbolic.simplify(num_gangs);
				}
				
				// identify the loop index variable 
				Expression ivar = LoopTools.getIndexVariable(ploop);
				Expression lb = LoopTools.getLowerBoundExpressionNS(ploop);
				Expression ub = LoopTools.getUpperBoundExpressionNS(ploop);
				Expression incr = LoopTools.getIncrementExpression(ploop);
				boolean increasingOrder = true;
				if( incr instanceof IntegerLiteral ) {
					long IntIncr = ((IntegerLiteral)incr).getValue();
					if( IntIncr < 0 ) {
						increasingOrder = false;
					}
					if( Math.abs(IntIncr) != 1 ) {
						Tools.exit("[ERROR in CUDATranslationTools.worksharingLoopTransformation()] A worksharing loop with a stride > 1 is found;" +
								"current implmentation can not handle the following loop: \nEnclosing procedure: " + cProc.getSymbolName() + 
								"\nOpenACC Annotation: " + lAnnot);
					}
				} else {
					Tools.exit("[ERROR in CUDATranslationTools.worksharingLoopTransformation()] The stride of a worksharing loop is not constant;" +
							"current implmentation can not handle the following loop: \nEnclosing procedure: " + cProc.getSymbolName() + 
							"\nOpenACC Annotation: " + lAnnot);

				}
				if( !increasingOrder ) {
					//Swap lower-bound and upper-bound if iteration decreases.
					Expression tmp = lb;
					lb = ub;
					ub = tmp;
				}
				BinaryExpression biexp3 = null;
				AssignmentExpression assgn = null;
				Expression bid = null;
				if( gangdim == 1 ) {
					bid = new NameID("blockIdx.x");
				} else if( gangdim == 2 ) {
					bid = new NameID("blockIdx.y");
				} else {
					bid = new NameID("blockIdx.z");
				}
				Expression tid = null;
				if( workerdim == 1 ) {
					tid = new NameID("threadIdx.x");
				} else if( workerdim == 2 ) {
					tid = new NameID("threadIdx.y");
				} else {
					tid = new NameID("threadIdx.z");
				}
				Expression base = null;
				if( isGangLoop ) {
					if( isWorkerLoop ) {
						//base = tid + bid * num_workers;
						base = new BinaryExpression(tid, BinaryOperator.ADD, 
								new BinaryExpression(bid, BinaryOperator.MULTIPLY, num_workers.clone()));
					} else {
						base = bid;
					}
				} else if( isWorkerLoop ) {
					base = tid;
				}
				if( !(lb instanceof Literal) || !lb.toString().equals("0") ) {
					biexp3 = new BinaryExpression(base, BinaryOperator.ADD, lb);
					assgn = new AssignmentExpression(ivar, 
							AssignmentOperator.NORMAL, biexp3);
				} else {
					assgn = new AssignmentExpression(ivar, 
							AssignmentOperator.NORMAL, base);
				}
				Statement thrmapstmt = new ExpressionStatement(assgn);
				CompoundStatement pParent = (CompoundStatement)ploop.getParent();
				pParent.addStatementBefore(ploop, thrmapstmt);
				Symbol ivarSym = SymbolTools.getSymbolOf(ivar);
				if( ivarSym != null ) {
					Declaration ivarSymDecl = ivarSym.getDeclaration(); 
					Traversable tivar = ivarSymDecl.getParent();
					boolean declaredInLoopBody = false;
					while( tivar != null ) {
						if( tivar.equals(ploop) ) {
							declaredInLoopBody = true;
							break;
						} else {
							tivar = tivar.getParent();
						}
					}
					if( declaredInLoopBody ) {
						Statement ivarSymStmt = (Statement)ivarSymDecl.getParent();
						CompoundStatement ivarPStmt = (CompoundStatement)ivarSymStmt.getParent();
						ivarPStmt.removeStatement(ivarSymStmt);
						ivarSymDecl.setParent(null);
						pParent.addDeclaration(ivarSymDecl);
					}
				}
			
				boolean addBoundaryCheck = true;
				if( opt_skipKernelLoopBoundChecking ) {
					addBoundaryCheck = false;
				} else {
					//[DEBUG] To disable the below optimization, comment out the following code:
					if( (lb != null) && (ub != null) ) {
						//If the iteration size matches number of gangs of the pure gang loop or number of workers of the
						//pure worker loop, we don't need to add boundary-checkeing code.
						Expression itrSize = Symbolic.simplify(Symbolic.add(Symbolic.subtract(ub,lb),new IntegerLiteral(1)));
						if( isGangLoop && (!isWorkerLoop) && (num_gangs != null) && (itrSize.equals(num_gangs)) ) {
							addBoundaryCheck = false;
						} else if( (!isGangLoop) && isWorkerLoop && (num_workers != null) && (itrSize.equals(num_workers)) ) {
							addBoundaryCheck = false;
						} else if( isGangLoop && isWorkerLoop && (num_gangs != null) && (num_workers != null) && (itrSize.equals(Symbolic.simplify(Symbolic.multiply(num_gangs, num_workers)))) ) {
							addBoundaryCheck = false;
						}
					}
				}
				if( addBoundaryCheck ) {
					/*
					 * Replace the worksharing loop with if-statement containing the loop body 
					 */
					loopbody.setParent(null);
					IfStatement ifstmt = new IfStatement((Expression)ploop.getCondition().clone(),
							loopbody);
					ploop.swapWith(ifstmt);
					//DEBUG: Below code is disabled since OpenACC has 'nowait'
					//for gang loops in a parallel region by definition.
					/*				if( isGangLoop && outermostloop && cRegionKind.equals("parallel") &&
						(ploop != region) ) {
					FunctionCall syncCall = new FunctionCall(new NameID("__syncthreads"));
					ExpressionStatement syncCallStmt = new ExpressionStatement(syncCall);
					((CompoundStatement)ifstmt.getParent()).addStatementBefore(ifstmt, syncCallStmt);
				}*/
					// Move all annotations from the loop to the if-statement, which will break
					// the OpenACC semantics.
					List<Annotation> aAnnots = ploop.getAnnotations();
					if( aAnnots != null ) {
						for(Annotation aAn : aAnnots ) {
							ifstmt.annotate(aAn);
						}
					}
					ploop.removeAnnotations();
				} else {
					//Replace the loop with its body.
					loopbody.setParent(null);
					ploop.swapWith(loopbody);
					List<Annotation> aAnnots = ploop.getAnnotations();
					if( aAnnots != null ) {
						for(Annotation aAn : aAnnots ) {
							loopbody.annotate(aAn);
						}
					}
					ploop.removeAnnotations();
				}
			}
		}
		PrintTools.println("[worksharingLoopTransformation() ends]", 2);
	}

}
