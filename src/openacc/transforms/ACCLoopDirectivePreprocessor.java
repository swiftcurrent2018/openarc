/**
 * 
 */
package openacc.transforms;

import cetus.analysis.LoopTools;
import cetus.exec.Driver;
import cetus.hir.*;
import openacc.analysis.AnalysisTools;
import openacc.analysis.SubArray;
import openacc.hir.*;
import cetus.transforms.TransformPass;
import java.util.*;

/**
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group, Oak Ridge National Laboratory
 *
 */
public class ACCLoopDirectivePreprocessor extends TransformPass {
	private static String pass_name = "[ACCLoopDirectivePreprocessor]";
	private boolean disableWorkShareLoopCollapsing = false;

	/**
	 * @param program
	 */
	public ACCLoopDirectivePreprocessor(Program program) {
		super(program);
		// TODO Auto-generated constructor stub
	}

	/* (non-Javadoc)
	 * @see cetus.transforms.TransformPass#getPassName()
	 */
	@Override
	public String getPassName() {
		return pass_name;
	}
	
	/**
	 * Perform basic preprocessing for compute-region loops.
	 * 
	 */
	private void baseComputeRegionPreprocessor() {
		
		List<ACCAnnotation>  cRegionAnnots = AnalysisTools.collectPragmas(program, ACCAnnotation.class, ACCAnnotation.computeRegions, false);
		if( cRegionAnnots != null ) {
			for( ACCAnnotation cAnnot : cRegionAnnots ) {
				Annotatable at = cAnnot.getAnnotatable();
				boolean isParallelRegion = false; //is kernels region.
				if( cAnnot.containsKey("parallel") ) {
					isParallelRegion = true;
				}
				if( at instanceof Loop ) {
					if( !cAnnot.containsKey("loop") ) {
						//If kernels loop has separate kernels annotation and loop annotaion, merge them together.
						ACCAnnotation lAnnot = at.getAnnotation(ACCAnnotation.class, "loop");
						if( (lAnnot != null) && (!lAnnot.equals(cAnnot)) ) {
							for( String key : lAnnot.keySet() ) {
								if( !key.equals("pragma") ) {
									Object val = lAnnot.get(key);
									cAnnot.put(key, val);
								}
							}
							List<ACCAnnotation> aList = at.getAnnotations(ACCAnnotation.class);
							at.removeAnnotations(ACCAnnotation.class);
							for( ACCAnnotation tAnnot : aList ) {
								if( !tAnnot.equals(lAnnot) ) {
									at.annotate(tAnnot);
								}
							}
						}
					}
					if( !at.containsAnnotation(ACCAnnotation.class, "loop") ) {
						//If a compute region is a loop, but not having loop directive, add it.
						cAnnot.put("loop", "_directive");
					}
					if( !cAnnot.containsKey("gang") && !cAnnot.containsKey("worker") && !cAnnot.containsKey("seq")) {
						if( isParallelRegion ) {
							cAnnot.put("gang", "_clause");
							if( !AnalysisTools.ipContainPragmas(at, ACCAnnotation.class, "worker", null) ) {
								cAnnot.put("worker", "_clause");
							}
						} else {
							boolean isIndependent = false;
							if( cAnnot.containsKey("independent") ) {
								isIndependent = true;
							} else {
								CetusAnnotation tcAnnot = at.getAnnotation(CetusAnnotation.class, "parallel");
								if( tcAnnot != null ) {
									isIndependent = true;
								}
							}
							if( isIndependent ) {
								cAnnot.put("gang", "_clause");
								if( !AnalysisTools.ipContainPragmas(at, ACCAnnotation.class, "worker", null) ) {
									cAnnot.put("worker", "_clause");
								}
							} else {
							Procedure cProc = IRTools.getParentProcedure(at);
							PrintTools.println("\n[WARNING] the following kernels region does not have any loop directive with " +
									"worksharing clause (gang or worker) or independent clause, which will be executed sequentially by default. " +
									"To enable automatic parallelization, set AccParallelization to 1.\n" +
									"Enclosing procedure: " + cProc.getSymbolName() + "\n" +
									"Compute region: " + cAnnot + "\n", 0);
							//For OpenACC V2.0 or higher, "auto" clause can be added too.
							//cAnnot.put("auto", "_clause");
							//[DEBUG] To enable auto-parallelization, below should be disabled.
							cAnnot.put("seq", "_clause");
							}
						}
					}
				} else {
					//Compute region is a compoundStatement.
					List<ACCAnnotation> loopAnnots = AnalysisTools.ipCollectPragmas(at, ACCAnnotation.class, "loop", null);
					if( loopAnnots != null ) {
						for( ACCAnnotation lAnnot : loopAnnots ) {
							Annotatable gLoop = lAnnot.getAnnotatable();
							Traversable tt = gLoop.getParent();
							boolean isOutermostACCLoop = true;
							boolean isLexicallyIncluded = false;
							while (tt != null ) {
								if( tt instanceof Annotatable ) {
									if( ((Annotatable)tt).containsAnnotation(ACCAnnotation.class, "loop") ) {
										isOutermostACCLoop = false;
										break;
									}
								}
								if( tt == at ) {
									isLexicallyIncluded = true;
								}
								tt = tt.getParent();
							}
							if( !isLexicallyIncluded && (tt != null) ) {
								while( tt != null ) {
									if( tt == at ) {
										isLexicallyIncluded = true;
										break;
									}
									tt = tt.getParent();
								}
							}
							if( isOutermostACCLoop ) {
								if( gLoop.containsAnnotation(ACCAnnotation.class, "gang") || 
										gLoop.containsAnnotation(ACCAnnotation.class, "worker") ||
										gLoop.containsAnnotation(ACCAnnotation.class, "seq") ) {
									continue;
								} else {
									Traversable lBody = ((Loop)gLoop).getBody();
									if( !AnalysisTools.ipContainPragmas(lBody, ACCAnnotation.class, "gang", null) ) {
										if( isParallelRegion ) {
											//OpenACC loop without worksharing clauses in a parallel region is implicitly parallel. 
											lAnnot.put("gang", "_clause");
											if( !AnalysisTools.ipContainPragmas(lBody, ACCAnnotation.class, "worker", null) ) {
												lAnnot.put("worker", "_clause");
											}
										} else {
											//OpenACC loop without worksharing clauses in a kernels region is implementation-dependent.
											if( lAnnot.containsKey("independent") || gLoop.containsAnnotation(CetusAnnotation.class, "parallel") ) {
												lAnnot.put("gang", "_clause");
											} else {
												lAnnot.put("seq", "_clause");
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

	/**
	 * - Find loops with collapse clauses in the input codes, input
	 * - For each loop with collapse clause
	 *     - Merge loop directives in the loops to be collapsed, and put them
	 *     into the outermost loop where collapse clause exists.
	 *         - If any conflict occurs during directive merging, error (incorrect collapse)
	 * 
	 */
	private void CollapseClausePreprocessor(Traversable input) {
		List<ForLoop> outer_loops = new ArrayList<ForLoop>();
		// Find loops containing OpenACC collapse clauses with parameter value > 1.
		List<ACCAnnotation> collapseAnnotList = IRTools.collectPragmas(input, ACCAnnotation.class, "collapse");
		if( collapseAnnotList == null ) return;
		for( ACCAnnotation cAnnot : collapseAnnotList ) {
			Annotatable at = cAnnot.getAnnotatable();
			if( at instanceof ForLoop ) {
				Object cObj = cAnnot.get("collapse");
				if( !(cObj instanceof IntegerLiteral) ) {
					Tools.exit("[ERROR] the argument of OpenACC collapse clause must be a constant positive integer expression"
							+ ", but the following loop construct has an incompatible argument:\n" + at.toString());
				}
				int collapseLevel = (int)((IntegerLiteral)cObj).getValue();
				if( collapseLevel > 1 ) {
					outer_loops.add((ForLoop)at);
				} else {
					PrintTools.println("[INFO] OpenACC collapse clause whose argument value is less than 2 will be ignored.", 1);
				}
			}
		}
		if( outer_loops.isEmpty() ) {
			return;
		} else {
			PrintTools.println("[INFO] Found OpenACC-collapse clauses that associate more than one loop.",1);
			for( ForLoop accLoop : outer_loops ) {
				Traversable t = (Traversable)accLoop;
				while(true) {
					if (t instanceof Procedure) break;
					t = t.getParent(); 
				}
				Procedure proc = (Procedure)t;
				ACCAnnotation collapseAnnot = accLoop.getAnnotation(ACCAnnotation.class, "collapse");
				ARCAnnotation openarcAnnot = null;
				if( collapseAnnot == null ) {
					Tools.exit("[ERROR] a loop associated with a collapse clause can not have another collapse clause:\n" +
							"Enclosing procedure: " + proc.getSymbolName() + "\n");
				}
				int collapseLevel = (int)((IntegerLiteral)collapseAnnot.get("collapse")).getValue();
				if( collapseLevel < 2 ) {
					continue;
				}
				boolean pnest = true;
				List<AnnotationStatement> commentList = new LinkedList<AnnotationStatement>();
				List<ForLoop> nestedLoops = new LinkedList<ForLoop>();
				pnest = AnalysisTools.extendedPerfectlyNestedLoopChecking(accLoop, collapseLevel, nestedLoops, commentList);
				if( !pnest ) {
					Tools.exit("[ERROR] OpenACC collapse clause is applicable only to perfectly nested loops;\n"
							+ "Procedure name: " + proc.getSymbolName() + "\nTarget loop: \n" +
							accLoop.toString() + "\n");
				} else {
					//Step1: move OpenACC/OpenARC annotations in the nested loops into the loop with collapse clause (accLoop).
					for( ForLoop currLoop : nestedLoops ) {
						List<PragmaAnnotation> tAList = new LinkedList<PragmaAnnotation>();
						List<PragmaAnnotation> pragmaAList = currLoop.getAnnotations(PragmaAnnotation.class);
						if( pragmaAList != null ) {
							for( PragmaAnnotation ptAnnot : pragmaAList ) {
								if( (ptAnnot instanceof ACCAnnotation) || (ptAnnot instanceof ARCAnnotation) ) {
									tAList.add(ptAnnot);
								}
							}
						}
						if( !tAList.isEmpty() ) {
							for( PragmaAnnotation tAnnot : tAList ) {
								Set<String> keySet = tAnnot.keySet();
								for( String tKey : keySet ) {
									if( tKey.equals("pragma") ) continue;
									else if( tKey.equals("collapse") ) {
										Tools.exit("[ERROR] a loop associated with a collapse clause can not have another collapse clause:\n" +
												"Enclosing procedure: " + proc.getSymbolName() + "\n" +
												"Target loop:\n" + accLoop.toString() + "\n");
									} else {
										Object tVal = tAnnot.get(tKey);
										PragmaAnnotation cAnnot = accLoop.getAnnotation(PragmaAnnotation.class, tKey);
										if( cAnnot == null ) {
											if( ACCAnnotation.OpenACCDirectiveSet.contains(tKey) || ACCAnnotation.OpenACCClauseSet.contains(tKey) ) {
												collapseAnnot.put(tKey, tVal);
											} else if( ARCAnnotation.OpenARCDirectiveSet.contains(tKey) ) {
												openarcAnnot = new ARCAnnotation(tKey, tVal);
												accLoop.annotate(openarcAnnot);
											} else {
												for( String tDirective : ARCAnnotation.OpenARCDirectiveSet ) {
													if( keySet.contains(tDirective) ) {
														openarcAnnot = new ARCAnnotation(tDirective, "_directive");
														openarcAnnot.put(tKey, tVal);
														accLoop.annotate(openarcAnnot);
														break;
													}
												}
											}
										} else {
											Object cVal = cAnnot.get(tKey);
											boolean conflict = false;
											if( tVal instanceof String ) {
												if( !tVal.equals(cVal) ) {
													conflict = true;
												}
											} else if( tVal instanceof Expression ) {
												if( cVal instanceof Expression ) {
													if( ACCAnnotation.worksharingClauses.contains(tKey) ||
															ACCAnnotation.parallelDimensionClauses.contains(tKey) ) {
														//TODO
														Expression biExp = new BinaryExpression((Expression)cVal, BinaryOperator.MULTIPLY,
																(Expression)tVal);
														collapseAnnot.put(tKey, Symbolic.simplify(biExp));
													}
												} else {
													conflict = true;
												}
											} else if( tVal instanceof Set ) {
												if( !(cVal instanceof Set) ) {
													conflict = true;
												} else {
													((Set)cVal).addAll((Set)tVal);
												}
											} else if( tVal instanceof Map ) {
												if( !(cVal instanceof Map) ) {
													conflict = true;
												} else {
													try { 
														Map tValMap = (Map)tVal;
														Map cValMap = (Map)cVal;
														for( String op : (Set<String>)tValMap.keySet() ) {
															Set tValSet = (Set)tValMap.get(op); 
															Set cValSet = (Set)cValMap.get(op); 
															if( cValSet == null ) {
																cValMap.put(op, tValSet);
															} else {
																cValSet.addAll(tValSet);
															}
														}
													} catch( Exception e ) {
														Tools.exit("[ERROR in ACCLoopDirectivePreprocessor.CollapseClausePreprocessor()]: "+
																"<String, Set> type is expected for the value" +
																" of key," + tKey + " in ACCAnnotation, " + tAnnot);
													}	
												}
											} else {
												Tools.exit("[ERROR] unexpected argument (" + tVal.toString() + 
														") is found whilie merging OpenACC annotations in loops associated with a collapse clause:\n" +
														"Enclosing procedure: " + proc.getSymbolName() + "\n" +
														"For loop:\n" + accLoop.toString() + "\n");

											}
											if( conflict ) {
												Tools.exit("[ERROR] error occurred in merging OpenACC annotations in loops associated with a collapse clause:\n" +
														"Enclosing procedure: " + proc.getSymbolName() + "\n" +
														"For loop:\n" + accLoop.toString() + "\n");
											}
										}
									}
								}
							}
						}
						currLoop.removeAnnotations(ACCAnnotation.class);
						currLoop.removeAnnotations(ARCAnnotation.class);
					}

					//Step2: If comment statements exist around the nested loops, move them before the collapse loop (accLoop)
					if( commentList.size() > 0 ) {
						CompoundStatement pStmt = (CompoundStatement)accLoop.getParent();
						for( Statement stmt : commentList ) {
							CompoundStatement tParent = (CompoundStatement)stmt.getParent();
							tParent.removeChild(stmt);
							pStmt.addStatementBefore(accLoop, stmt);
						}
					}
				}	
			}
		}
	}

	/**
	 * - Find gang loops in the program, and store them to a list in descending order.
	 * - For each gang loop in the descending list
	 *     - If there exists a pure worker loop enclosing the current gang loop
	 *         - swap these loops.                    
	 *             - If collapse clause exists, associated loops are moved together.
	 *         - Repeat this until it finds another gang loop enclosing the current gang 
	 *         loop, or there is no enclosing worker loop. 
	 *     - If there exists another gang loop enclosing the current gang loop
	 *         - If these two gang loops are not directly nested
	 *             - swap the current gang loop with the direct child loop of the outer gang
	 *             loop.
	 *                 - If collapse clause exists, associated loops are moved together.
	 * - Relative nesting order in the gang/worker loops should be preserved respectively.
	 *
	 * - Find pure worker loops in the program, and store them to a list in descending order.
	 * - For each pure worker loop in the descending list
	 *     - If there exists a pure vector loop enclosing the current worker loop                
	 *         - swap these loops.
	 *             - If collapse clause exists, associated loops are moved together.
	 *             - If not swappable, error.
	 *         - Repeat this until it finds another pure worker loop enclosing the current worker loop, 
	 *         or there is no enclosing pure vector loop. 
	 *     - If there exists another pure worker loop enclosing the current worker loop
	 *         - If these two worker loops are not directly nested
	 *             - swap the current worker loop with the direct child loop of the outer 
	 *             worker loop.
	 *                 - If collapse clause exists, associated loops are moved together.
	 *                 - If not swappable, error.
	 *   - Relative nesting order in the worker/vector loops should be preserved respectively.
	 *  
	 * @param outerWorkshareClause coarse-grained worksharing clause (gang or worker)
	 * @param innerWorkshareClause fine-grained worksharing clause (worker or vector)
	 */
	private void WorksharingLoopsPreprocessor(String outerWorkshareClause, String innerWorkshareClause) {
		// Find loops containing outerWorkshareClause clauses.
		List<ACCAnnotation> outerLoopAnnotList = IRTools.collectPragmas(program, ACCAnnotation.class, outerWorkshareClause);
		if( outerLoopAnnotList == null ) return;
		//CAVEAT: Below algorithm assumes that outerLoopAnnotList returns outer worksharing loop first if worksharing loops are nested.
		//(This is true in the current IRTools.collectPragmas() implementation.)
		for( ACCAnnotation gAnnot : outerLoopAnnotList ) {
			Annotatable at = gAnnot.getAnnotatable();
			//If outerWorkshareClause is "worker", handle only pure worker loops w/o gang clause.
			if( outerWorkshareClause.equals("worker") ) {
				if( at.containsAnnotation(ACCAnnotation.class, "gang") ) {
					continue;
				}
			}
			if( at instanceof ForLoop ) {
				ForLoop gloop = (ForLoop)at;
				Traversable t = gloop.getParent();
				int nestLevel = 1;
				//If there exists a pure inner-worksharing loop that enclose the current outer-worksharing loop, 
				//    - Swap these loops.
				//    - Repeat this until it finds another outer-worksharing loop or there is no enclosing inner-worksharing loop.
				//If there exists another outer-worksharing loop that enclose the current outer-worksharing loop, 
				//    - Swap the current outer-worksharing loop with the direct child of the outer outer-worksharing loop
				//      if it is not directly nested by the outer outer-worksharing loop.
				while( t != null ) {
					if( t instanceof ForLoop ) {
						ForLoop tloop = (ForLoop)t;
						boolean isOuterWorkshareLoop = false;
						boolean isInnerWorkshareLoop = false;
						nestLevel++;
						if( tloop.containsAnnotation(ACCAnnotation.class, outerWorkshareClause) ) {
							isOuterWorkshareLoop = true;
						}
						if( tloop.containsAnnotation(ACCAnnotation.class, innerWorkshareClause) ) {
							isInnerWorkshareLoop = true;
						}
						if( isOuterWorkshareLoop ) { //found an outer outerWorkshareClause loop
							boolean notSwappable = false;
							//swap the inner outerWorkshareClause loop if the outer outerWorkshareClause loop does not nest the inner outerWorkshareClause loop directly.
							if( nestLevel > 2 ) {
								int targetLoopLevel = 2;
								ACCAnnotation tAnnot = tloop.getAnnotation(ACCAnnotation.class, "collapse");
								if( tAnnot != null ) {
									int collapseLevel = (int)((IntegerLiteral)tAnnot.get("collapse")).getValue();
									if( collapseLevel > 1 ) {
										targetLoopLevel = collapseLevel+1;
									}
								}
								List<ForLoop> nestedLoops = new LinkedList<ForLoop>();
								if( AnalysisTools.extendedPerfectlyNestedLoopChecking(tloop, targetLoopLevel, nestedLoops, null) ) {
									if( targetLoopLevel < nestLevel ) {
										ForLoop targetLoop = nestedLoops.get(targetLoopLevel-2);		
										//swap the direct child loop of the outer outerWorkshareClause loop with inner outerWorkshareClause loop.
										if( AnalysisTools.extendedPerfectlyNestedLoopChecking(targetLoop, (nestLevel-targetLoopLevel+1), null, null) ) {
											TransformTools.extendLoopSwap(targetLoop, gloop);
										} else {
											notSwappable = true;
										}
									}
								} else {
									notSwappable = true;
								}
							}
							if( notSwappable ) {
								Procedure pProc = IRTools.getParentProcedure(tloop);
								Tools.exit("[ERROR] In current implementation, if multiple " + outerWorkshareClause + " loops exist in nested loops, " +
										"the compiler swaps these so that all these loops are directly nested together," +
										" but swapping failed since these loops are not interchangable; exit!\n" +
										"Parent procedure: " + pProc.getSymbolName() +"\nTarget loop: \n" + tloop + "\n");
							}
							break;
						} else if( isInnerWorkshareLoop ) {
							//found pure inner-worksharing loop. 
							if( AnalysisTools.extendedPerfectlyNestedLoopChecking(tloop, nestLevel, null, null) ) {
								//swap outer inner-workshareing loop with inner outer-worksharing loop.
								TransformTools.extendLoopSwap(tloop, gloop);
								nestLevel=1; //reset nestLevel.
							} else {
								Procedure pProc = IRTools.getParentProcedure(tloop);
								Tools.exit("[ERROR] In current implementation, if a pure " + innerWorkshareClause + " loop contain an inner "+ outerWorkshareClause +
										" loop, the compiler swaps these," +
										" but swapping failed since these loops are not interchangable; exit!\n" +
										"Parent procedure: " + pProc.getSymbolName() +"\nTarget loop: \n" + tloop + "\n");
							}
						}
					}
					t = t.getParent();
				}
			}
		}
	}

	/**
	 * This method check the following:
	 * - If multiple gang loops exist in nested loops, all gang loops should be directly nested together.
	 * - If a gang loop does not contain any worker loop, worker clause is automatically added if in kernels region.
	 * - For nested gang loops, 
	 *   If they are in Kernel regions, each gang loop is annotated with gangdim(n) clause, where n is the number of nested gang
	 *   loops including the current gang loop itself.
	 *   If they are in Parallel region, the nested gang loops are collapsed to make 1D mapping, and gangdim(1) clause is added.
	 *   If they are in seq kernels region, the nested gang loops are collapsed to make 1D mapping, and gangdim(1) clause is added.
	 * 
	 * - If multiple worker loops exist in nested loops, all worker loops should be directly nested together.
	 * - If a pure worker loop contains a gang loop, error.
	 * - If a pure worker loop is not enclosed by a gang loop, error if in kernels region
	 * - For nested worker loops, 
	 *   If they are in Kernel region, and if they are not pure worker loops,
	 *   each worker loop is annotated with workerdim(n) clause, where n is the number of nested worker
	 *   loops including the current worker loop itself.
	 *   If they are in Kernel region, and if they are pure worker loops,
	 *   the nested worker loops are collapsed to make 1D mapping, and workerdim(1) clause is added.
	 *   If they are in Parallel region, the nested worker loops are collapsed to make 1D mapping, and workerdim(1) clause is added.
	 * 
	 * - If a pure vector loop contains a gang/worker loop, error.
	 * - If a pure vector loop is not enclosed by a worker or a gang loop, error if in kernels region
	 * 
	 * - If a parallel/kernels loop does not have any gang/independent/seq/worker/vector clause, add seq clause to it.
	 * - If a seq loop contains any gang/worker/vector clause, error.
	 *   [TODO] if a seq loop contains gang/worker/vector/independent inner loops and if they are all 
	 *   perfectly nested, perform loop interchange so that the seq loop becomes the innermost loop,
	 *   which is a valid transformation.
	 * 
	 * - If a parallel region has a loop containing inner gang/worker loops, and if the loop does not have any OpenACC loop directive,
	 *   add "loop seq" clauses, which will be used later for privatization.
	 * 
	 * TODO: Each nested gang/worker loops should have at most one private/reduction clause in the outermost loop.
	 * 
	 * 
	 * 
	 */
	private void CheckWorkSharingLoopNestingOrder() {
		List<ACCAnnotation> computeRegions = AnalysisTools.collectPragmas(program, ACCAnnotation.class, ACCAnnotation.computeRegions,
				false);
		if( computeRegions == null ) return;
		for( ACCAnnotation rAnnot : computeRegions ) {
			Annotatable rAt = rAnnot.getAnnotatable();
			String computeRegion;
			if( rAt.containsAnnotation(ACCAnnotation.class, "parallel") ) {
				computeRegion = "parallel";
			} else {
				computeRegion = "kernels";
			}
			boolean isSeqKernelLoop = false;
			if( (rAt instanceof ForLoop) && rAt.containsAnnotation(ACCAnnotation.class, "loop") ) {
				if( !rAt.containsAnnotation(ACCAnnotation.class, "gang") 
						&& !rAt.containsAnnotation(ACCAnnotation.class, "worker") 
						&& !rAt.containsAnnotation(ACCAnnotation.class, "vector") 
						&& !rAt.containsAnnotation(ACCAnnotation.class, "independent") 
						&& !rAt.containsAnnotation(ACCAnnotation.class, "seq") ) {
					rAnnot.put("seq", "_clause");
				}
			}
			if( computeRegion.equals("kernels") && rAnnot.containsKey("seq") ) {
				isSeqKernelLoop = true;
			}
			List<ACCAnnotation> gangAnnots = AnalysisTools.ipCollectPragmas(rAt, ACCAnnotation.class, "gang", null);
			if( gangAnnots != null ) {
				for( ACCAnnotation gAnnot : gangAnnots ) {
					ForLoop gloop = (ForLoop)gAnnot.getAnnotatable();
					//Check1: if multiple gang loops exist in nested loops, all gang loops should be directly nested together.
					List<ForLoop> nestedGLoops = AnalysisTools.findDirectlyNestedLoopsWithClause(gloop, "gang");
					if( nestedGLoops.isEmpty() ) {
						continue; //nestedWLoops can be empty due to CollapseClausePreprocessor() called below.
					}
					ForLoop innermostgloop = nestedGLoops.get(nestedGLoops.size()-1);
					ForLoop outermostgloop = nestedGLoops.get(0);
					if( AnalysisTools.ipContainPragmas(innermostgloop.getBody(), ACCAnnotation.class, "gang", null) ) {
						Procedure pProc = IRTools.getParentProcedure(rAt);
						Tools.exit("[ERROR] Gang loop can not contain other gang loops unless they are directly nested together possibly through" +
								"loop-interchage transformation; this loop can not be handled by the current compiler version.\n" +
								"Enclosing procedure: " + pProc.getSymbolName() +"\nEnclosing " + computeRegion + " region: " +
								rAnnot);
					}
					//Check2: If a gang loop does not contain any worker loop, worker clause is automatically added if in kernels region.
					if( computeRegion.equals("kernels") ) {
						boolean noWorkerLoops = true;
						for( ForLoop gl : nestedGLoops ) {
							if( gl.containsAnnotation(ACCAnnotation.class, "worker") ) {
								noWorkerLoops = false;
								break;
							}
						}
						if( noWorkerLoops ) {
							if( !AnalysisTools.ipContainPragmas(innermostgloop.getBody(), ACCAnnotation.class, "worker", null) ) {
								for( ForLoop gl : nestedGLoops ) {
									ACCAnnotation tGA = gl.getAnnotation(ACCAnnotation.class, "gang");
									tGA.put("worker", "_clause");
								}	
							} else {
								//gang loop does not contain worker clause in the same loop, but include inner worker loops.
								//index variable of the gang-only loop is cached on the shared memory.
								Set<SubArray> sharedRWSet = null;
								ARCAnnotation cudaAnnot = outermostgloop.getAnnotation(ARCAnnotation.class, "sharedRW");
								if( cudaAnnot == null ) {
									cudaAnnot = outermostgloop.getAnnotation(ARCAnnotation.class, "cuda");
									if( cudaAnnot == null ) {
										cudaAnnot = new ARCAnnotation("cuda", "_clause");
										outermostgloop.annotate(cudaAnnot);
									}
									sharedRWSet = new HashSet<SubArray>();
									cudaAnnot.put("sharedRW", sharedRWSet);
								} else {
									sharedRWSet = cudaAnnot.get("sharedRW");
								}
								for( ForLoop gl : nestedGLoops ) {
									ACCAnnotation tGA = gl.getAnnotation(ACCAnnotation.class, "gang");
									Identifier indexV = (Identifier)LoopTools.getIndexVariable(gl);
									if( indexV != null ) {
										sharedRWSet.add(AnalysisTools.createSubArray(indexV.getSymbol(), true, null));
									}
								}	
							}
						}
					}
					int nestLevel = nestedGLoops.size();
					if( (computeRegion.equals("kernels") && disableWorkShareLoopCollapsing) || ((computeRegion.equals("kernels") && !isSeqKernelLoop)) ) {
						//Add gangdim(n) internal annotations to each gang loop.
						for( ForLoop tloop : nestedGLoops ) {
							if( tloop.containsAnnotation(ACCAnnotation.class, "gangdim") ) {
								break; //gangdim annotation is already added.
							} else {
								ACCAnnotation iAnnot = tloop.getAnnotation(ACCAnnotation.class, "internal");
								if( iAnnot == null ) {
									iAnnot = new ACCAnnotation("internal", "_directive");
									iAnnot.setSkipPrint(true);
									tloop.annotate(iAnnot);
								}
								iAnnot.put("gangdim", new IntegerLiteral(nestLevel));
								nestLevel--;
							}
						}
					} else { //In a parallel region/seq Kernel loop, collapse nested gang loops to make 1D mapping.
						//If disableWorkShareLoopCollapsing is true, below is skipped for kernels loop.
						ForLoop outermostloop = nestedGLoops.get(0);
						if( nestLevel > 1 ) {
							Traversable tt = outermostloop.getParent();
							boolean isOutermostGangLoop = true;
							while (tt != null ) {
								if( tt instanceof Annotatable ) {
									if( ((Annotatable)tt).containsAnnotation(ACCAnnotation.class, "gang") ) {
										isOutermostGangLoop = false;
										break;
									}
								}
								tt = tt.getParent();
							}
							if( isOutermostGangLoop ) {
								Expression colLevel = null;
								for( ForLoop tloop : nestedGLoops ) {
									Expression tLevel = new IntegerLiteral(1);
									if( tloop.containsAnnotation(ACCAnnotation.class, "collapse") ) {
										ACCAnnotation aAnnot = tloop.getAnnotation(ACCAnnotation.class, "collapse");
										tLevel = (Expression)aAnnot.remove("collapse");
									}
									if( colLevel == null ) {
										colLevel = tLevel;
									} else {
										colLevel = Symbolic.simplify(Symbolic.add(colLevel, tLevel));
									}
								}
								ACCAnnotation aAnnot = outermostloop.getAnnotation(ACCAnnotation.class, "loop");
								aAnnot.put("collapse", colLevel);
								CollapseClausePreprocessor(outermostloop);
							}
						}
						if( outermostloop.containsAnnotation(ACCAnnotation.class, "gangdim") ) {
							break; //gangdim annotation is already added.
						} else {
							ACCAnnotation iAnnot = outermostloop.getAnnotation(ACCAnnotation.class, "internal");
							if( iAnnot == null ) {
								iAnnot = new ACCAnnotation("internal", "_directive");
								iAnnot.setSkipPrint(true);
								outermostloop.annotate(iAnnot);
							}
							iAnnot.put("gangdim", new IntegerLiteral(1));
						}
					}
				}
			}
			List<ACCAnnotation> workerAnnots = AnalysisTools.ipCollectPragmas(rAt, ACCAnnotation.class, "worker", null);
			if( workerAnnots != null ) {
				for( ACCAnnotation wAnnot : workerAnnots ) {
					ForLoop wloop = (ForLoop)wAnnot.getAnnotatable();
					//Check3: If multiple worker loops exist in nested loops, all worker loops should be directly nested together.
					List<ForLoop> nestedWLoops = AnalysisTools.findDirectlyNestedLoopsWithClause(wloop, "worker");
					if( nestedWLoops.isEmpty() ) {
						continue; //nestedWLoops can be empty due to CollapseClausePreprocessor() called below.
					}
					ForLoop innermostwloop = nestedWLoops.get(nestedWLoops.size()-1);
					if( AnalysisTools.ipContainPragmas(innermostwloop.getBody(), ACCAnnotation.class, "worker", null) ) {
						Procedure pProc = IRTools.getParentProcedure(rAt);
						Tools.exit("[ERROR] Worker loop can not contain other worker loops unless they are directly nested together possibly through" +
								"loop-interchage transformation; can not be handled by the current compiler version.\n" +
								"Enclosing procedure: " + pProc.getSymbolName() +"\nEnclosing " + computeRegion + " region: " +
								rAnnot);
					}
					//Check4: If pure worker loop contains a gang loop, error.
					if( AnalysisTools.ipContainPragmas(innermostwloop.getBody(), ACCAnnotation.class, "gang", null) ) {
						Procedure pProc = IRTools.getParentProcedure(rAt);
						Tools.exit("[ERROR] Worker loop can not contain inner gang loop unless they are swappable through" +
								"loop-interchage transformation; can not be handled by the current compiler version.\n" +
								"Enclosing procedure: " + pProc.getSymbolName() +"\nEnclosing " + computeRegion + " region: " +
								rAnnot);
					}
					//Check5: If pure worker loop is not enclosed by a gang loop, error if in kernels region
					boolean noGangLoops = true;
					for( ForLoop wl : nestedWLoops ) {
						if( wl.containsAnnotation(ACCAnnotation.class, "gang") ) {
							noGangLoops = false;
							break;
						}
					}
					if( computeRegion.equals("kernels") ) {
						if( noGangLoops ) { //This is a pure worker loop.
							//CAVEAT: Below checking assumes that each compute region has unique context, 
							//which can be ensured by procedure cloning transformation.
							ACCAnnotation tGA = AnalysisTools.ipFindFirstPragmaInParent(wloop, ACCAnnotation.class, "gang", null, null);
							if( tGA == null ) {
								Procedure pProc = IRTools.getParentProcedure(rAt);
								Tools.exit("[ERROR] In kernels region, a pure worker loop not enclosed by a gang loop is not allowed; " +
										"can not be handled by the current compiler version.\n" +
										"Enclosing procedure: " + pProc.getSymbolName() +"\nEnclosing " + computeRegion + " region: " +
										rAnnot);
							}
						}
					}
					//Add workerdim(n) internal annotations to each worker loop.
					int nestLevel = nestedWLoops.size();
					if( (computeRegion.equals("kernels") && disableWorkShareLoopCollapsing) || ((computeRegion.equals("kernels") && !noGangLoops && !isSeqKernelLoop)) ) {
						for( ForLoop tloop : nestedWLoops ) {
							if( tloop.containsAnnotation(ACCAnnotation.class, "workerdim") ) {
								break; //workerdim annotation is already added.
							} else {
								ACCAnnotation iAnnot = tloop.getAnnotation(ACCAnnotation.class, "internal");
								if( iAnnot == null ) {
									iAnnot = new ACCAnnotation("internal", "_directive");
									iAnnot.setSkipPrint(true);
									tloop.annotate(iAnnot);
								}
								iAnnot.put("workerdim", new IntegerLiteral(nestLevel));
								nestLevel--;
							}
						}
					} else { 
						//In a parallel region/seq kernel loop, collapse nested worker loops to make 1D mapping.
						//In a kernel loop, pure worker loops are also collapsed into 1D mapping.
						//If disableWorkShareLoopCollapsing is true, below is skipped for kernels loop.
						ForLoop outermostloop = nestedWLoops.get(0);
						if( nestLevel > 1 ) {
							Traversable tt = outermostloop.getParent();
							boolean isOutermostWorkerLoop = true;
							while (tt != null ) {
								if( tt instanceof Annotatable ) {
									if( ((Annotatable)tt).containsAnnotation(ACCAnnotation.class, "worker") ) {
										isOutermostWorkerLoop = false;
										break;
									}
								}
								tt = tt.getParent();
							}
							if( isOutermostWorkerLoop ) {
								Expression colLevel = null;
								for( ForLoop tloop : nestedWLoops ) {
									Expression tLevel = new IntegerLiteral(1);
									if( tloop.containsAnnotation(ACCAnnotation.class, "collapse") ) {
										ACCAnnotation aAnnot = tloop.getAnnotation(ACCAnnotation.class, "collapse");
										tLevel = (Expression)aAnnot.remove("collapse");
									}
									if( colLevel == null ) {
										colLevel = tLevel;
									} else {
										colLevel = Symbolic.simplify(Symbolic.add(colLevel, tLevel));
									}
								}
								ACCAnnotation aAnnot = outermostloop.getAnnotation(ACCAnnotation.class, "loop");
								aAnnot.put("collapse", colLevel);
								CollapseClausePreprocessor(outermostloop);
							}
						}
						if( outermostloop.containsAnnotation(ACCAnnotation.class, "workerdim") ) {
							break; //workerdim annotation is already added.
						} else {
							ACCAnnotation iAnnot = outermostloop.getAnnotation(ACCAnnotation.class, "internal");
							if( iAnnot == null ) {
								iAnnot = new ACCAnnotation("internal", "_directive");
								iAnnot.setSkipPrint(true);
								outermostloop.annotate(iAnnot);
							}
							iAnnot.put("workerdim", new IntegerLiteral(1));
						}
					}
				}
			}
			List<ACCAnnotation> vectorAnnots = AnalysisTools.ipCollectPragmas(rAt, ACCAnnotation.class, "vector", null);
			if( vectorAnnots != null ) {
				for( ACCAnnotation vAnnot : vectorAnnots ) {
					ForLoop vloop = (ForLoop)vAnnot.getAnnotatable();
					if( vloop.containsAnnotation(ACCAnnotation.class, "gang") || vloop.containsAnnotation(ACCAnnotation.class, "worker") ) {
						continue;
					}
					//Check6: If a pure vector loop contains a gang/worker loop, error.
					if( AnalysisTools.ipContainPragmas(vloop.getBody(), ACCAnnotation.class, ACCAnnotation.parallelWorksharingClauses, false, null) ) {
						Procedure pProc = IRTools.getParentProcedure(rAt);
						Tools.exit("[ERROR] Pure vector loop can not contain inner gang/worker loop unless they are swappable through" +
								"loop-interchage transformation; can not be handled by the current compiler version.\n" +
								"Enclosing procedure: " + pProc.getSymbolName() +"\nEnclosing " + computeRegion + " region: " +
								rAnnot);
					}
					//Check7: If a pure vector loop is not enclosed by a worker or a gang loop, error if in kernels region
					if( computeRegion.equals("kernels") ) {
						//CAVEAT: Below checking assumes that each compute region has unique context, 
						//which can be ensured by procedure cloning transformation.
						ACCAnnotation tGWA = AnalysisTools.ipFindFirstPragmaInParent(vloop, ACCAnnotation.class, 
								ACCAnnotation.parallelWorksharingClauses, false, null, null);
						if( tGWA == null ) {
							Procedure pProc = IRTools.getParentProcedure(rAt);
							Tools.exit("[ERROR] In kernels region, a pure vector loop not enclosed by a gang/worker loop is not allowed; " +
									"can not be handled by the current compiler version.\n" +
									"Enclosing procedure: " + pProc.getSymbolName() +"\nEnclosing " + computeRegion + " region: " +
									rAnnot + "\n");
						}

					}
				}
			}
			List<ACCAnnotation> seqAnnots = AnalysisTools.ipCollectPragmas(rAt, ACCAnnotation.class, "seq", null);
			if( seqAnnots != null ) {
				for( ACCAnnotation sAnnot : seqAnnots ) {
					ForLoop sloop = (ForLoop)sAnnot.getAnnotatable();
					if( sloop.containsAnnotation(ACCAnnotation.class, "gang") || sloop.containsAnnotation(ACCAnnotation.class, "worker")
							|| sloop.containsAnnotation(ACCAnnotation.class, "vector") ) {
						Procedure pProc = IRTools.getParentProcedure(rAt);
						Tools.exit("[ERROR] a seq loop can not have other worksharing clauses (gang/worker/vector); exit!\n" +
								"Enclosing procedure: " + pProc.getSymbolName() +"\nEnclosing " + computeRegion + " region: " +
								rAnnot + "\n");
					}
				}
			}
			if( computeRegion.equals("parallel") && (rAt instanceof CompoundStatement) ) {
				List<ACCAnnotation> workshareAnnots = AnalysisTools.ipCollectPragmas(rAt, ACCAnnotation.class, 
						ACCAnnotation.parallelWorksharingClauses, false, null);
				if( workshareAnnots != null ) {
					for( ACCAnnotation wsAn : workshareAnnots ) {
						Annotatable wsAt = wsAn.getAnnotatable();
						List<ForLoop> forLoops = new LinkedList<ForLoop>();
						Traversable t = wsAt.getParent();
						boolean isOuterMostLoop = true;
						while( (t != null) && (t != rAt) ) {
							if( (t instanceof Annotatable) && (((Annotatable)t).containsAnnotation(ACCAnnotation.class, "gang")
									|| ((Annotatable)t).containsAnnotation(ACCAnnotation.class, "worker")) ) {
								isOuterMostLoop = false;
								break;
							}
							if( (t instanceof ForLoop) && !((ForLoop)t).containsAnnotation(ACCAnnotation.class, "loop") ) {
								forLoops.add((ForLoop)t);
							}
							t = t.getParent();
						}
						if( isOuterMostLoop && !forLoops.isEmpty() ) {
							for( ForLoop floop : forLoops ) {
								ACCAnnotation newAnnot = new ACCAnnotation("loop", "_directive");
								newAnnot.put("seq", "_clause");
								floop.annotate(newAnnot);
							}
						}
					}
				}
			}
		}
	}
	
	/**
	 * In OpenACC V1.0, "independent" clause is allowed on loop directives in kernels region.
	 * For now, if a pure independent loop is not enclosed by any gang/worker loop, the compiler adds
	 * gang and worker clauses to it.
	 * If it is used in other region, it will be ignored.
	 * 
	 */
	private void IndependentLoopsPreprocessor() {
		List<ACCAnnotation> computeRegions = IRTools.collectPragmas(program, ACCAnnotation.class, "kernels");
		for( ACCAnnotation rAnnot : computeRegions ) {
			Annotatable rAt = rAnnot.getAnnotatable();
			List<ACCAnnotation> indAnnots = AnalysisTools.ipCollectPragmas(rAt, ACCAnnotation.class, "independent", null);
			if( indAnnots != null ) { 
				for( ACCAnnotation iAt : indAnnots ) {
					Annotatable at = iAt.getAnnotatable();
					if( (at instanceof ForLoop) && (!at.containsAnnotation(ACCAnnotation.class, "gang")) &&
							(!at.containsAnnotation(ACCAnnotation.class, "worker")) &&
							(!at.containsAnnotation(ACCAnnotation.class, "vector")) &&
							(!at.containsAnnotation(ACCAnnotation.class, "seq"))  ) {
						ACCAnnotation tGWA = AnalysisTools.ipFindFirstPragmaInParent(at, ACCAnnotation.class, 
								ACCAnnotation.parallelWorksharingClauses, false, null, null);
						if( tGWA == null ) {
							List<ForLoop> indLoops = AnalysisTools.findDirectlyNestedLoopsWithClause((ForLoop)at, "independent");
							for( ForLoop indL : indLoops ) {
								ACCAnnotation tAt = indL.getAnnotation(ACCAnnotation.class, "independent");
								if( tAt != null ) {
									if( !tAt.containsKey("seq") ) {
										tAt.put("gang", "_clause");
										tAt.put("worker", "_clause");
									}
								}
							}
						}
					}
				}
			}
		}
	}

	/* (non-Javadoc)
	 * @see cetus.transforms.TransformPass#start()
	 */
	@Override
	public void start() {
		
		String value = Driver.getOptionValue("disableWorkShareLoopCollapsing");
		if( value != null ) {
			disableWorkShareLoopCollapsing = true;
		}
		baseComputeRegionPreprocessor();
		CollapseClausePreprocessor(program);
		WorksharingLoopsPreprocessor("gang", "worker");
		WorksharingLoopsPreprocessor("worker", "vector");
		IndependentLoopsPreprocessor();
		CheckWorkSharingLoopNestingOrder();
	}

}
