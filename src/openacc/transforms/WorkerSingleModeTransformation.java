/**
 * 
 */
package openacc.transforms;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import cetus.hir.Annotatable;
import cetus.hir.Annotation;
import cetus.hir.AnnotationStatement;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.BreakStatement;
import cetus.hir.Case;
import cetus.hir.CommentAnnotation;
import cetus.hir.CompoundStatement;
import cetus.hir.ContinueStatement;
import cetus.hir.DFIterator;
import cetus.hir.DataFlowTools;
import cetus.hir.DeclarationStatement;
import cetus.hir.Default;
import cetus.hir.DoLoop;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.ForLoop;
import cetus.hir.FunctionCall;
import cetus.hir.GotoStatement;
import cetus.hir.IDExpression;
import cetus.hir.IRTools;
import cetus.hir.IfStatement;
import cetus.hir.IntegerLiteral;
import cetus.hir.Label;
import cetus.hir.Loop;
import cetus.hir.NameID;
import cetus.hir.PrintTools;
import cetus.hir.Procedure;
import cetus.hir.Program;
import cetus.hir.ReturnStatement;
import cetus.hir.StandardLibrary;
import cetus.hir.Statement;
import cetus.hir.SwitchStatement;
import cetus.hir.Symbol;
import cetus.hir.SymbolTools;
import cetus.hir.Tools;
import cetus.hir.Traversable;
import cetus.transforms.TransformPass;
import openacc.analysis.AnalysisTools;
import openacc.hir.ACCAnnotation;
import openacc.transforms.FaultInjectionTransformation;

/**
 * Wrap code sections in worker-single mode so that they can be executed only by one worker, 
 * which is necessary for correct GPU kernel transformation.
 * This pass also adds "#pragma acc barrier" to the end of each worker loop unless it is the last worker loop
 * in the enclosing compute region, it has reduction clause, or it has nowait clause, which is not a standard clause in OpenACC V2.0.
 * 
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group
 *         Oak Ridge National Laboratory
 *
 */
public class WorkerSingleModeTransformation extends TransformPass {
	private static String pass_name = "[WorkerSingleModeTransformation]";
	private static IDExpression threadID = null;
	private int target_arch = 0;

	/**
	 * @param program
	 */
	public WorkerSingleModeTransformation(Program program, int tarch) {
		super(program);
		//threadID = SymbolTools.getOrphanID("_tid");
		threadID = new NameID("_tid");
		target_arch = tarch;
		verbosity = 1;
	}
	
	public WorkerSingleModeTransformation(Program program, IDExpression tid, int tarch) {
		super(program);
		threadID = tid;
		target_arch = tarch;
	}

	/* (non-Javadoc)
	 * @see cetus.transforms.TransformPass#getPassName()
	 */
	@Override
	public String getPassName() {
		return pass_name;
	}

	/* (non-Javadoc)
	 * @see cetus.transforms.TransformPass#start()
	 */
	@Override
	public void start() {
		List<ACCAnnotation> compRegAnnots = 
			AnalysisTools.collectPragmas(program, ACCAnnotation.class, ACCAnnotation.computeRegions, false);
		if( compRegAnnots != null ) {
			for( ACCAnnotation cAnnot : compRegAnnots ) {
				//DEBUG: if ACCAnnotation key, parallel or kernels, is set to "false", GPU translation
				//will be skipped.
				String kernelType = "kernels";
				if( cAnnot.containsKey("parallel") ) {
					kernelType = "parallel";
				}
				if( cAnnot.get(kernelType).equals("false") ) {
					//System.err.println("Following compute region will be skipped " + cAnnot);
					continue;
				}
				//if( target_arch == 3 ) {
					if( kernelType.equals("parallel") ) {
						Expression num_gangs = cAnnot.get("num_gangs");
						Expression num_workers = cAnnot.get("num_workers");
						if( (num_gangs != null) && (num_gangs instanceof IntegerLiteral) 
								&& (num_workers != null) && (num_workers instanceof IntegerLiteral) ) {
							if( (((IntegerLiteral)num_gangs).getValue() == 1) && (((IntegerLiteral)num_workers).getValue() == 1) ) {
								//Skip a single worker-item kernel.
								continue;
							}
						}
					}
					if( cAnnot.containsKey("seq") ) {
						if( !AnalysisTools.ipContainPragmas(cAnnot.getAnnotatable(), ACCAnnotation.class, ACCAnnotation.worksharingClauses, false, null) ) {
							//Skip a single worker-item kernel.
							continue;
						}
					}
				//}
				Annotatable at = cAnnot.getAnnotatable();
				if ( at instanceof ForLoop ) {
					if( cAnnot.containsKey("worker") ) {
						//The attached compute region is in worker-partitioned mode.
						continue;
					} else {
						handleWorkerSingleMode((Statement)at, true);
					}
				} else if (at instanceof CompoundStatement ){
					handleWorkerSingleMode((Statement)at, true);
				} else {
					Procedure proc = IRTools.getParentProcedure(at);
					Tools.exit("[ERROR in WorkerSingleModeTransformation] unexpected type of compute region is found; eixt\n" +
							"Enclosing procedure : " + proc.getSymbolName() + "\nCompute region : " + cAnnot + "\n");
				}
			}
		}

	}
	
	protected boolean handleWorkerSingleMode(Statement region, boolean newRegion) {
		boolean isWorkerSingleMode = true;
		CompoundStatement targetRegion = null;
		Procedure proc = IRTools.getParentProcedure(region);
		List<ForLoop> LoopsToAddWorkerBarrier = new LinkedList<ForLoop>();
		if( region instanceof ForLoop ) {
			ForLoop accLoop = (ForLoop)region;
			ArrayList<ForLoop> indexedLoops = new ArrayList<ForLoop>();
			ACCAnnotation collapseAnnot = accLoop.getAnnotation(ACCAnnotation.class, "collapse");
			if( collapseAnnot != null ) {
				int collapseLevel = (int)((IntegerLiteral)collapseAnnot.get("collapse")).getValue();
				if( collapseLevel > 1 ) {
					boolean pnest = true;
					pnest = AnalysisTools.extendedPerfectlyNestedLoopChecking(accLoop, collapseLevel, indexedLoops, null);
					if( pnest ) {
						ForLoop iLoop = indexedLoops.get(collapseLevel-2);
						targetRegion = (CompoundStatement)iLoop.getBody();
					} else {
						Tools.exit("[ERROR] OpenACC collapse clause is applicable only to perfectly nested loops;\n"
								+ "Procedure name: " + proc.getSymbolName() + "\nTarget loop: \n" +
								accLoop.toString() + "\n");
					}
				}
			}
			if( targetRegion == null ) {
				targetRegion = (CompoundStatement)accLoop.getBody();
			}
		} else if( region instanceof CompoundStatement ) {
			targetRegion = (CompoundStatement)region;
		}
		if( targetRegion != null ) {
			List<Traversable> children = targetRegion.getChildren();
			List<List<Traversable>> subRegList = new LinkedList<List<Traversable>>();
			List<Traversable> temp_list = new LinkedList<Traversable>();
			List<Traversable> ftinjectCallStmts = new LinkedList<Traversable>();
			int cIndex = 0;
			int kRSize = children.size();
			int tlistSize = 0;
			boolean createSubRegion = false;
			Statement nCompRegion = null;
			while( cIndex < kRSize ) {
				boolean noSplit = true;
				ftinjectCallStmts.clear();
				Traversable child = children.get(cIndex++);
				if( (child instanceof DeclarationStatement) || (child instanceof Case) || 
						(child instanceof BreakStatement) || (child instanceof ContinueStatement) || 
						(child instanceof Default) || (child instanceof GotoStatement) || (child instanceof Label) 
						|| (child instanceof ReturnStatement)) {
					noSplit = false;
				} else if( child instanceof AnnotationStatement ) {
					if( ((AnnotationStatement)child).containsAnnotation(ACCAnnotation.class, "barrier") ) {
						isWorkerSingleMode = false;
						noSplit = false;
					}
				} else {
					if( child instanceof ExpressionStatement )  {
						List<FunctionCall> fCallList = IRTools.getFunctionCalls(child);
						if( fCallList != null ) {
							for( FunctionCall fCall : fCallList ) {
								Procedure tProc = fCall.getProcedure();
								String fCallName = fCall.getName().toString();
								boolean ttWSMode = true;
								if( (tProc != null) && (!StandardLibrary.contains(fCall)) ) {
									ttWSMode = handleWorkerSingleMode(tProc.getBody(), false);
									if( !ttWSMode ) {
										isWorkerSingleMode = false;
										noSplit = false;
									}
								}
								if( noSplit ) {
									//ftinjection calls do not have their procedure definition in the current IR, 
									//and thus it should be checked only with their names.
									if( fCallName.contains(FaultInjectionTransformation.ftinjectionCallBaseName) ) {
										ftinjectCallStmts.add(child);
									}
								}
							}
						}
					} else if( child instanceof IfStatement ) {
						IfStatement tIfStmt = (IfStatement)child;
						Statement tStmt = tIfStmt.getThenStatement();
						if( tStmt != null ) {
							noSplit = handleWorkerSingleMode(tStmt, false);
							if( !noSplit ) {
								isWorkerSingleMode = false;
							}
						}
						tStmt = tIfStmt.getElseStatement();
						if( tStmt != null ) {
							noSplit = handleWorkerSingleMode(tStmt, false);
							if( !noSplit ) {
								isWorkerSingleMode = false;
							}
						}
						Expression condExp = tIfStmt.getControlExpression();
						List<FunctionCall> fCallList = IRTools.getFunctionCalls(condExp);
						if( fCallList != null ) {
							for( FunctionCall fCall : fCallList ) {
								Procedure tProc = fCall.getProcedure();
								String fCallName = fCall.getName().toString();
								boolean ttWSMode = true;
								if( (tProc != null) && (!StandardLibrary.contains(fCall)) ) {
									ttWSMode = handleWorkerSingleMode(tProc.getBody(), false);
									if( !ttWSMode ) {
										isWorkerSingleMode = false;
										noSplit = false;
									}
								}
								if( noSplit ) {
									//ftinjection calls do not have their procedure definition in the current IR, 
									//and thus it should be checked only with their names.
									if( fCallName.contains(FaultInjectionTransformation.ftinjectionCallBaseName) ) {
										ftinjectCallStmts.add(child);
									}
								}
							}
						}
					} else if( child instanceof Loop ) {
						Statement loopBody = null;
						if( child instanceof ForLoop ) {
							ForLoop fLoop = (ForLoop)child;
							if( fLoop.containsAnnotation(ACCAnnotation.class, "worker") ) {
								noSplit = false;
								isWorkerSingleMode = false;
								if( !fLoop.containsAnnotation(ACCAnnotation.class, "nowait") &&
										!fLoop.containsAnnotation(ACCAnnotation.class, "reduction") ) {
									LoopsToAddWorkerBarrier.add(fLoop);
								}
							} else {
								//DEBUG: it seems that below is not necessary.
								/*									if( fLoop.containsAnnotation(ACCAnnotation.class, "gang") ) {
										isWorkerSingleMode = false; //This gang loop is still in worker-single mode, but
										//this work-sharing loop should be executed by all workers anyway for possible
										//worker-partitioned mode inside.
									}*/
								ArrayList<ForLoop> indexedLoops = new ArrayList<ForLoop>();
								ACCAnnotation collapseAnnot = fLoop.getAnnotation(ACCAnnotation.class, "collapse");
								if( collapseAnnot != null ) {
									int collapseLevel = (int)((IntegerLiteral)collapseAnnot.get("collapse")).getValue();
									if( collapseLevel > 1 ) {
										boolean pnest = true;
										pnest = AnalysisTools.extendedPerfectlyNestedLoopChecking(fLoop, collapseLevel, indexedLoops, null);
										if( pnest ) {
											ForLoop iLoop = indexedLoops.get(collapseLevel-2);
											loopBody = (CompoundStatement)iLoop.getBody();
										} else {
											Tools.exit("[ERROR] OpenACC collapse clause is applicable only to perfectly nested loops;\n"
													+ "Procedure name: " + proc.getSymbolName() + "\nTarget loop: \n" +
													fLoop.toString() + "\n");
										}
									}
								}
								if( loopBody == null ) {
									loopBody = fLoop.getBody();
								}
								if( !handleWorkerSingleMode(loopBody, false) ) {
									noSplit = false;
									isWorkerSingleMode = false;
								}
							}
						} else {
							loopBody = ((Loop)child).getBody();
							if( !handleWorkerSingleMode(loopBody, false) ) {
								noSplit = false;
								isWorkerSingleMode = false;
							}
						}
						Expression condExp = ((Loop)child).getCondition();
						List<FunctionCall> fCallList = IRTools.getFunctionCalls(condExp);
						if( fCallList != null ) {
							for( FunctionCall fCall : fCallList ) {
								Procedure tProc = fCall.getProcedure();
								String fCallName = fCall.getName().toString();
								boolean ttWSMode = true;
								if( (tProc != null) && (!StandardLibrary.contains(fCall)) ) {
									ttWSMode = handleWorkerSingleMode(tProc.getBody(), false);
									if( !ttWSMode ) {
										isWorkerSingleMode = false;
										noSplit = false;
									}
								}
								if( noSplit ) {
									//ftinjection calls do not have their procedure definition in the current IR, 
									//and thus it should be checked only with their names.
									if( fCallName.contains(FaultInjectionTransformation.ftinjectionCallBaseName) ) {
										ftinjectCallStmts.add(child);
									}
								}
							}
						}
					} else if( child instanceof CompoundStatement ) {
						if( !handleWorkerSingleMode((CompoundStatement)child, false) ) {
							noSplit = false;
							isWorkerSingleMode = false;
						}
					} else if( child instanceof SwitchStatement ) {
						if( !handleWorkerSingleMode(((SwitchStatement)child).getBody(), false) ) {
							noSplit = false;
							isWorkerSingleMode = false;
						}
					}
				}
				if( noSplit ) {
					temp_list.add(child);
					if( !ftinjectCallStmts.isEmpty() ) {
						//If ftinject calls exist in a worker-single-mode section, remove thread ID checking condition from 
						//its enclosing if-statement.
						FaultInjectionTransformation.removeThreadIDCheckingCondition(ftinjectCallStmts);
					}
				} else {
					if( !temp_list.isEmpty() ) {
						subRegList.add(temp_list);
						temp_list = new LinkedList<Traversable>();
					}
				}
			} //end of while loop
			if( !temp_list.isEmpty() ) {
				subRegList.add(temp_list);
			}
			
			
			boolean genWrapperCode = false;
			if( isWorkerSingleMode ) {
				if( newRegion ) {
					//Create wrapper code to enforce worker-single mode.
					genWrapperCode = true;
				} else {
					//The whole input region is in worker-single mode, and thus the caller will create
					//the wrapper code that include this input region.
					genWrapperCode = false;
				}
			} else {
				//Some children of the input region are not in worker-single mode, and thus
				//wrapper code should be added for each sub-section in worker-single mode.
				genWrapperCode = true;
			}
			if( genWrapperCode && (!subRegList.isEmpty()) ) {
				Traversable tChild = null;
				int subListSize = 0;
				CompoundStatement tParent = null;
				Statement prevStmt = null;
				IfStatement ifStmt = null;
				CompoundStatement ifBody = null;
				for( List<Traversable> subList : subRegList ) {
					subListSize = subList.size();
					//If all statements are comments, we don't have to wrap these.
					boolean allComments = true;
					for( int i=0; i<subListSize; i++ ) {
						tChild = subList.get(i);
						if( tChild instanceof AnnotationStatement ) {
							List<Annotation> aList = ((AnnotationStatement)tChild).getAnnotations();
							if( aList != null ) {
								for( Annotation taat : aList ) {
									if( !(taat instanceof CommentAnnotation) ) {
										allComments = false;
										break;
									}
								}
							} else {
								allComments = false;
								break;
							}
						} else {
							allComments = false;
							break;
						}
					}
					if( allComments ) {
						continue;
					}
					tChild = subList.get(0);
					tParent = (CompoundStatement)tChild.getParent();
					prevStmt = AnalysisTools.getStatementBefore(tParent, (Statement)tChild);
					ifBody = new CompoundStatement();
					for( int i=0; i<subListSize; i++ ) {
						tChild = subList.get(i);
						tParent.removeChild(tChild);
						ifBody.addStatement((Statement)tChild);
					}
					Map<Expression, Set<Integer>> defExpMap = DataFlowTools.getDefMap(ifBody);
					Map<Symbol, Set<Integer>> defSymMap = DataFlowTools.convertExprMap2SymbolMap(defExpMap);
					Set<Symbol> defSymSet = defSymMap.keySet();
					Statement bStmt = null;
					if( !defSymSet.isEmpty() ) {
						ACCAnnotation bAnnot = new ACCAnnotation("barrier", "_directive");
						bStmt = new AnnotationStatement(bAnnot);
					}
					Expression condition = new BinaryExpression(threadID.clone(), BinaryOperator.COMPARE_EQ, new IntegerLiteral(0));
					condition.setParens(false);
					ifStmt = new IfStatement(condition, ifBody);
					if( prevStmt == null ) {
						if( tParent.countStatements() == 0 ) {
							tParent.addStatement(ifStmt);
							if( bStmt != null ) {
								tParent.addStatement(bStmt);
							}
						} else {
							prevStmt = (Statement)tParent.getChildren().get(0);
							tParent.addStatementBefore(prevStmt, ifStmt);
							if( bStmt != null ) {
								tParent.addStatementBefore(prevStmt, bStmt);
							}
						}
					} else {
						if( bStmt != null ) {
							tParent.addStatementAfter(prevStmt, bStmt);
						}
						tParent.addStatementAfter(prevStmt, ifStmt);
					}
				}
			}
			
			//Release temporary lists.
			if( !subRegList.isEmpty() ) {
				for( List subList : subRegList ) {
					subList = null;
				}
			}
			//Add implicit barrier for each worker loop unless it has nowait clause, reduction clause, or the 
			//last code in the enclosing compute region.
			subRegList = null;
			int tSize = LoopsToAddWorkerBarrier.size();
			if( tSize > 0 ) {
				ForLoop wLoop = null;
				CompoundStatement wPStmt = null;
				for( int i=0; i<(tSize-1); i++ ) {
					wLoop = LoopsToAddWorkerBarrier.get(i);
					wPStmt = (CompoundStatement)wLoop.getParent();
					ACCAnnotation bAnnot = new ACCAnnotation("barrier", "_directive");
					Statement bStmt = new AnnotationStatement(bAnnot);
					wPStmt.addStatementAfter(wLoop, bStmt);
				}
				wLoop = LoopsToAddWorkerBarrier.get(tSize-1);
				wPStmt = (CompoundStatement)wLoop.getParent();
				CompoundStatement cRegion = null;
				Statement wChildStmt = null;
				if( wPStmt.containsAnnotation(ACCAnnotation.class, "kernels") || 
						wPStmt.containsAnnotation(ACCAnnotation.class, "parallel") ) {
					cRegion = wPStmt;
					wChildStmt = wLoop;
				} else {
					Traversable tC = wLoop;
					Traversable tP = wPStmt;
					Annotatable tGP = (Annotatable)wPStmt.getParent();
					while ( (tGP != null) && !(tGP instanceof Procedure) ) {
						if( tGP.containsAnnotation(ACCAnnotation.class, "kernels") ||
								tGP.containsAnnotation(ACCAnnotation.class, "parallel") ) {
							if( tGP instanceof CompoundStatement ) {
								cRegion = (CompoundStatement)tGP;
								wChildStmt = (Statement)tP;
							} else if( tGP instanceof ForLoop ) {
								cRegion = (CompoundStatement)tP;
								wChildStmt = (Statement)tC;
							}
							break;
						} else {
							if( tP instanceof CompoundStatement ) {
								Statement nStmt = AnalysisTools.getStatementAfter((CompoundStatement)tP, (Statement)tC);
								if( (nStmt !=null) && !(nStmt instanceof AnnotationStatement) ) {
									//This worker loop is not the last statement in the enclosing compute region.
									break;
								}
							}
							tC = tP;
							tP = tGP;
							tGP = (Annotatable)tGP.getParent();
						}
					}
				}
				boolean insertBarrier = true;
				if( (cRegion != null) && (wChildStmt != null) ) {
					Statement nStmt = AnalysisTools.getStatementAfter(cRegion, wChildStmt);
					if( (nStmt == null) || (nStmt instanceof AnnotationStatement) ) {
						//This worker loop is  the last statement in the enclosing compute region; no barrier is
						//needed.
						insertBarrier = false;
					}
				}
				if( insertBarrier ) {
					ACCAnnotation bAnnot = new ACCAnnotation("barrier", "_directive");
					Statement bStmt = new AnnotationStatement(bAnnot);
					wPStmt.addStatementAfter(wLoop, bStmt);
				}
			}
		}
		
		return isWorkerSingleMode;
	}
	
	//WorkerSingleModeTransformation should be applied only to compute regions that will
	//be converted into GPU kernels. If the compiler generate a CPU version of a compute
	//region, the wrapper should be removed.
	//This also remove acc barrier directives too.
	public static void removeWorkerSingleModeWrapper(Statement region) {
		DFIterator<IfStatement> iter =
			new DFIterator<IfStatement>(region, IfStatement.class);
		IfStatement ifStmt = null;
		Expression ifCond = null;
		CompoundStatement ifBody = null;
		Statement nextStmt = null;
		CompoundStatement cStmt = null;
		while (iter.hasNext()) {
			ifStmt = (IfStatement)iter.next();
			ifCond = ifStmt.getControlExpression();
			if( ifCond instanceof BinaryExpression ) {
				if( ((BinaryExpression)ifCond).getLHS().equals(threadID) ) {
					ifBody = (CompoundStatement)ifStmt.getThenStatement();
					cStmt = (CompoundStatement)ifStmt.getParent();
					nextStmt = AnalysisTools.getStatementAfter(cStmt, ifStmt);
					cStmt.removeStatement(ifStmt);
					if( nextStmt == null ) {
						while( ifBody.countStatements() > 0 ) {
							Statement child = (Statement)ifBody.getChildren().get(0);
							ifBody.removeChild(child);
							cStmt.addStatement(child);
						}
					} else {
						while( ifBody.countStatements() > 0 ) {
							Statement child = (Statement)ifBody.getChildren().get(0);
							ifBody.removeChild(child);
							cStmt.addStatementBefore(nextStmt, child);
						}
					}
				}
			}
		}
		DFIterator<AnnotationStatement> iter2 =
			new DFIterator<AnnotationStatement>(region, AnnotationStatement.class);
		while (iter2.hasNext()) {
			AnnotationStatement annotStmt = (AnnotationStatement)iter2.next();
			if( annotStmt.containsAnnotation(ACCAnnotation.class, "barrier") ) {
				cStmt = (CompoundStatement)annotStmt.getParent();
				cStmt.removeStatement(annotStmt);
			}
		}
		DFIterator<FunctionCall> iter3 =
			new DFIterator<FunctionCall>(region, FunctionCall.class);
		while (iter3.hasNext()) {
			FunctionCall fCall = (FunctionCall)iter3.next();
			Statement fCallStmt = fCall.getStatement();
			if( fCallStmt.containsAnnotation(ACCAnnotation.class, "barrier") ) {
				cStmt = (CompoundStatement)fCallStmt.getParent();
				cStmt.removeStatement(fCallStmt);
			}
		}
	}

}
