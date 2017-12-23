/**
 * 
 */
package openacc.transforms;

import cetus.hir.*;
import cetus.transforms.TransformPass;

import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.Map;
import java.util.HashSet;
import java.util.HashMap;
import java.util.ArrayList;

import openacc.analysis.AnalysisTools;
import openacc.analysis.SubArray;
import openacc.hir.*;

/**
 * If a kernel region is a compound statement, each nested gang loop in the region is changed into a kernels loop,
 * and the enclosing kernel region is changed to a data region.
 * 
 * @author f6l
 *
 */
public class KernelsSplitting extends TransformPass {
	private static String pass_name = "[KernelsSplitting]";
	private boolean IRSymbolOnly = true;
	private List<ACCAnnotation> DataRegions = new LinkedList<ACCAnnotation>();

	/**
	 * @param program
	 */
	public KernelsSplitting(Program program, boolean IRSymOnly) {
		super(program);
		IRSymbolOnly = IRSymOnly;
		// TODO Auto-generated constructor stub
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
		List<ACCAnnotation> kernelsAnnots = IRTools.collectPragmas(program, ACCAnnotation.class, "kernels");
		if( kernelsAnnots != null ) {
			for( ACCAnnotation kAnnot : kernelsAnnots ) {
				Annotatable at = kAnnot.getAnnotatable();
				Procedure cProc = IRTools.getParentProcedure(at);
				if( at instanceof CompoundStatement ) { //Found a kernels region that is not a loop.
					Expression asyncExp = kAnnot.get("async");
					Expression ifExp = kAnnot.get("if");
					CompoundStatement kRegion = (CompoundStatement)at;
					List<Traversable> children = kRegion.getChildren();
					List<Traversable> temp_list = new LinkedList<Traversable>();
					int cIndex = 0;
					int lastInsertedKernel = 0;
					int kRSize = children.size();
					int tlistSize = 0;
					int createKernel = 0;
					Statement nCompRegion = null;
					while( cIndex < kRSize ) {
						Traversable child = children.get(cIndex++);
						temp_list.add(child);
						if( (child instanceof Loop) && ((Annotatable)child).containsAnnotation(ACCAnnotation.class, "loop") ) {
							if( temp_list.size() == 1 ) {
								createKernel = 1; //create a kernel loop
							} else {
								createKernel = 2;
							}
						} else if( cIndex == kRSize ) {
							createKernel = 2; //create a parallel region
						}
						if( createKernel > 0 ) {
							////////////////////////////////////////////////////////////
							// Child is really removed from the parent at this point. //
							////////////////////////////////////////////////////////////
							int CommentAnnotations = 0;
							boolean CommentAnnotationOnly = false;
							tlistSize = temp_list.size();
							if( createKernel == 1 ) {
								nCompRegion = (ForLoop)temp_list.remove(0);
								kRegion.removeChild(nCompRegion);
							} else {
								nCompRegion = new CompoundStatement();
								for( int i=0; i<tlistSize; i++ ) {
									Traversable tchild = temp_list.remove(0);
									//tchild.setParent(null);
									kRegion.removeChild(tchild);
									if( tchild instanceof DeclarationStatement ) {
										Declaration decl = ((DeclarationStatement)tchild).getDeclaration();
										decl.setParent(null);
										((CompoundStatement)nCompRegion).addDeclaration(decl);
									} else if( tchild instanceof AnnotationStatement ) {
										AnnotationStatement tAStmt = (AnnotationStatement)tchild;
										List tAnnotList = tAStmt.getAnnotations(CommentAnnotation.class);
										if( (tAnnotList != null) && !tAnnotList.isEmpty() ) {
											CommentAnnotations++;
										}
										((CompoundStatement)nCompRegion).addStatement((Statement)tchild);
									} else {
										((CompoundStatement)nCompRegion).addStatement((Statement)tchild);
									}
								}
							}
							if( CommentAnnotations == tlistSize ) {
								CommentAnnotationOnly = true;
							}
							cIndex = cIndex - tlistSize;
							kRSize = children.size();
							if( cIndex >= kRSize ) {
								kRegion.addStatement(nCompRegion);
							} else {
								child = children.get(cIndex);
								kRegion.addStatementBefore((Statement)child, nCompRegion);
							}
							cIndex = cIndex + 1;
							kRSize = children.size();
							
							if( CommentAnnotationOnly ) {
								continue; //Skip the remaining kernel generation pass if it contains only comments.
							}
							//Step1: add kernels/parallel directive to the current compute region.
							ACCAnnotation cAnnot = null;
							if( nCompRegion instanceof ForLoop ) {
								cAnnot = nCompRegion.getAnnotation(ACCAnnotation.class, "loop");
								if( cAnnot == null ) {
									cAnnot = new ACCAnnotation("kernels", "_directive");
									nCompRegion.annotate(cAnnot);
								} else {
									cAnnot.put("kernels", "_directive");
								}
							} else {
								cAnnot = new ACCAnnotation("parallel", "_directive");
								nCompRegion.annotate(cAnnot);
								List<ACCAnnotation> tAnnots = IRTools.collectPragmas(nCompRegion, ACCAnnotation.class, "gang");
								if( tAnnots != null ) {
									Expression tExp = null;
									Expression tTemp = null;
									for( ACCAnnotation tGAnnot : tAnnots ) {
										if( tExp == null ) {
											tExp = tGAnnot.get("gang");
										} else {
											tTemp = tGAnnot.get("gang");
											if( tTemp != null ) {
												tExp = Symbolic.multiply(tExp, tTemp);
											}
										}
									}
									if( tExp != null ) {
										cAnnot.put("num_gangs", tExp);
									}
								}
								tAnnots = IRTools.collectPragmas(nCompRegion, ACCAnnotation.class, "worker");
								if( tAnnots != null ) {
									Expression tExp = null;
									Expression tTemp = null;
									for( ACCAnnotation tGAnnot : tAnnots ) {
										if( tExp == null ) {
											tExp = tGAnnot.get("worker");
										} else {
											tTemp = tGAnnot.get("worker");
											if( tTemp != null ) {
												tExp = Symbolic.multiply(tExp, tTemp);
											}
										}
									}
									if( tExp != null ) {
										cAnnot.put("num_workers", tExp);
									}
								}
							}
							//Step2-1: find symbols accessed in the region, and add them to accshared set.
							Set<Symbol> accSharedSymbols = null;
							Set<Symbol> accPrivateSymbols = null;
							Set<Symbol> accReductionSymbols = null;
							Annotation iAnnot = nCompRegion.getAnnotation(ACCAnnotation.class, "internal");
							if( iAnnot == null ) {
								iAnnot = new ACCAnnotation("internal", "_directive");
								accSharedSymbols = new HashSet<Symbol>();
								iAnnot.put("accshared", accSharedSymbols);
								accPrivateSymbols = new HashSet<Symbol>();
								accReductionSymbols = new HashSet<Symbol>();
								iAnnot.put("accprivate", accPrivateSymbols); //compute region may contain private clauses.
								iAnnot.put("accreduction", accReductionSymbols); //compute region may contain reduction clauses.
								iAnnot.setSkipPrint(true);
								nCompRegion.annotate(iAnnot);
							} else {
								accSharedSymbols = (Set<Symbol>)iAnnot.get("accshared");
								if( accSharedSymbols == null ) {
									accSharedSymbols = new HashSet<Symbol>();
									iAnnot.put("accshared", accSharedSymbols);
								}

								accPrivateSymbols = (Set<Symbol>)iAnnot.get("accprivate");
								if( accPrivateSymbols == null ) {
									accPrivateSymbols = new HashSet<Symbol>();
									iAnnot.put("accprivate", accPrivateSymbols);
								}
								accReductionSymbols = (Set<Symbol>)iAnnot.get("accreduction");
								if( accReductionSymbols == null ) {
									accReductionSymbols = new HashSet<Symbol>();
									iAnnot.put("accreduction", accReductionSymbols);
								}
							}
							Set<Symbol> tempSet = AnalysisTools.getAccessedVariables(nCompRegion, IRSymbolOnly);
							if( tempSet != null ) {
								accSharedSymbols.addAll(tempSet);
							}
							//Step2-2: find local symbols defined in the region, and remove them from the accshared set.
							tempSet = SymbolTools.getLocalSymbols(nCompRegion);
							if( tempSet != null ) {
								accSharedSymbols.removeAll(tempSet);
							}
							//Step2-3: find global symbols accessed in the functions called in the region, and add them 
							//to the accshared set.
							Map<String, Symbol> gSymMap = null;
							List<FunctionCall> calledFuncs = IRTools.getFunctionCalls(nCompRegion);
							for( FunctionCall call : calledFuncs ) {
								Procedure called_procedure = call.getProcedure();
								if( called_procedure != null ) {
									if( gSymMap == null ) {
										Set<Symbol> tSet = SymbolTools.getGlobalSymbols(nCompRegion);
										gSymMap = new HashMap<String, Symbol>();
										for( Symbol gS : tSet ) {
											gSymMap.put(gS.getSymbolName(), gS);
										}
									} 
									CompoundStatement body = called_procedure.getBody();
									Set<Symbol> procAccessedSymbols = AnalysisTools.getIpAccessedGlobalSymbols(body, gSymMap, null);
									accSharedSymbols.addAll(procAccessedSymbols);
								}
							}
							//Step2-4: remove index variables used for worksharing loops (gang/worker/vector loops) from 
							//the accshared set.
							Set<Symbol> loopIndexSymbols = AnalysisTools.getWorkSharingLoopIndexVarSet(nCompRegion);
							accSharedSymbols.removeAll(loopIndexSymbols);


							//Step2-5: if loop index variable is not local and not included in the private clause, add it the set.
							Set<SubArray> pSet = cAnnot.get("private");
							if( pSet != null ) {
								accPrivateSymbols.addAll(AnalysisTools.subarraysToSymbols(pSet, IRSymbolOnly));
								//accSharedSymbols.removeAll(accPrivateSymbols);
							}
							for(Symbol IndexSym : loopIndexSymbols ) {
								if( tempSet.contains(IndexSym) ) {
									continue; //loop index variable is local to the compute region.
								} else if(!accPrivateSymbols.contains(IndexSym)) {
									accPrivateSymbols.add(IndexSym);
									SubArray pSubArr = AnalysisTools.createSubArray(IndexSym, false, null);
									if( pSet == null ) {
										pSet = new HashSet<SubArray>();
										cAnnot.put("private", pSet);
									}
									pSet.add(pSubArr);
								}
							}
							//Step2-6: Create accreduction clause if missing.
							Map<ReductionOperator, Set<SubArray>> rMap = cAnnot.get("reduction");
							if( rMap != null ) {
								for( ReductionOperator op : (Set<ReductionOperator>)rMap.keySet() ) {
									Set<SubArray> valSet = (Set<SubArray>)rMap.get(op); 
									Set<Symbol> symDSet = null;
									symDSet = AnalysisTools.subarraysToSymbols(valSet, IRSymbolOnly);
									if( valSet.size() != symDSet.size() ) {
										Tools.exit("[ERROR in ACCAnalysis.declareDirectiveAnalysis()]: cannot find symbols for " +
												"subarrays of reduction clause in ACCAnnotation, " + cAnnot + "\n");
									} else {
										accReductionSymbols.addAll(symDSet);
									}
								}
								accSharedSymbols.removeAll(accReductionSymbols);
							}

							//Step2-7: copy kernels clauses to the new kernels loop.
							if( asyncExp != null ) {
								cAnnot.put("async", asyncExp.clone());
							}
							if( ifExp != null ) {
								cAnnot.put("if", ifExp.clone());
							}
							for( String dataClause : ACCAnnotation.dataClauses ) {
								Set<SubArray> dSet = (Set<SubArray>)kAnnot.get(dataClause);
								if( dSet != null ) {
									Set<SubArray> newSet = new HashSet<SubArray>();
									for( SubArray subArr : dSet ) {
										Symbol sym = AnalysisTools.subarrayToSymbol(subArr, IRSymbolOnly);
										/*											if( !isLexicallyIncluded ) {
												List osymList = new ArrayList(2);
												if( AnalysisTools.SymbolStatus.OrgSymbolFound(
														AnalysisTools.findOrgSymbol(sym, nCompRegion, false, cProc, osymList, null)) ) {
													sym = (Symbol)osymList.get(0);
												}
											}*/
										if( accSharedSymbols.contains(sym) ) {
											if( accPrivateSymbols.contains(sym) ) {
												if( !dataClause.equals("pcopy") && !dataClause.equals("copy") ) {
													PrintTools.println("\n[WARNING] privatization of the following variable conflicts with " +
															"data clause (" + dataClause + ") in the enclosing compute region; " +
															"privatization of this symbol is skipped!\n" +
															"To enforce privatization, put the variable in the pcopy clause.\n" +
															"Symbol: " + sym.getSymbolName() + "\nACCAnnotation: " + cAnnot + "\n" +
															"Enclosing compute region: " + kAnnot + "\n", 0);
													newSet.add(subArr);
													if( pSet != null ) {
														SubArray rmArr = null;
														Expression subName = subArr.getArrayName();
														for( SubArray cArr : pSet ) {
															if( cArr.getArrayName().equals(subName) ) {
																rmArr = cArr;
															}
														}
														if( rmArr != null ) {
															pSet.remove(rmArr);
														}
													}
													accPrivateSymbols.remove(sym);
												} else {
													accSharedSymbols.remove(sym);
												}
											} else {
												newSet.add(subArr);
											}
										}
									}
									if( !newSet.isEmpty() ) {
										if( dataClause.equals("deviceptr") ) {
											cAnnot.put(dataClause, newSet);
										} else {
											Set<SubArray> curSet = cAnnot.get("present");
											if( curSet == null ) {
												cAnnot.put("present", newSet);
											} else {
												curSet.addAll(newSet);
											}
										}
									}
								}
							}
							accSharedSymbols.removeAll(accPrivateSymbols);
						}
					}
					// The correct way to do this is changing to data enter and data exit
					//Convert the kernels region into a data region.
					//if( asyncExp != null ) {
					//	kAnnot.remove("async"); //data region does not have async clause.
					//}
					kAnnot.remove("kernels"); //Remove kernels directive.
					kAnnot.put("data", "_directive"); //Add data directive.
					DataRegions.add(kAnnot);
				}
			}
		}
	}
	
	/**
	 * Update data clauses in the newly created data region based on the results of the locality analysis.
	 * (For now, this handles the case where there is only one compute region in the data region.)
	 * 
	 */
	public void updateDataClauses() {
		for( ACCAnnotation aAnnot : DataRegions ) {
			Annotatable at = aAnnot.getAnnotatable();
			if( !(at instanceof CompoundStatement) ) {
				continue;
			}
			if( at.getChildren().size() > 1 ) {
				continue;
			}
			Annotatable cRegion = (Annotatable)at.getChildren().get(0);
			ACCAnnotation cAnnot = cRegion.getAnnotation(ACCAnnotation.class, "accreadonly");
			if( cAnnot == null ) {
				continue;
			}
			Set<Symbol> accreadonlySet = cAnnot.get("accreadonly");
			Set<Symbol> accpreadonlySet = cAnnot.get("accpreadonly");
			if( accpreadonlySet == null ) {
				accpreadonlySet = new HashSet<Symbol>();
				cAnnot.put("accpreadonly", accpreadonlySet);
			}
			accpreadonlySet.addAll(accreadonlySet);

			ARCAnnotation arcAnnot = cRegion.getAnnotation(ARCAnnotation.class, "sharedRO");
			if( arcAnnot == null ) {
				continue;
			}
			Set<SubArray> sharedROSubArraySet = arcAnnot.get("sharedRO");
			Set<Symbol> sharedROSet = AnalysisTools.subarraysToSymbols(sharedROSubArraySet, true);
			Set<SubArray> psharedROSubArraySet = arcAnnot.get("psharedRO");
			if( psharedROSubArraySet == null ) {
				psharedROSubArraySet = new HashSet<SubArray>();
				arcAnnot.put("psharedRO", psharedROSubArraySet);
			}
			psharedROSubArraySet.addAll(sharedROSubArraySet);

 			cAnnot = at.getAnnotation(ACCAnnotation.class, "accexplicitshared");
			Set<Symbol> accExShared = null;
			if( cAnnot == null ) {
				accExShared = new HashSet<Symbol>();
			} else {
				accExShared = cAnnot.get("accexplicitshared");
			}
			//For R/O shared variables whose data transfers are not explicitly specified by users,
			//move them from pcopy/copy to pcopyin/copyin.
			ACCAnnotation compAnnot = null;
			Map<Symbol, SubArray> pcopyMap = new HashMap<Symbol, SubArray>();
			Map<Symbol, SubArray> copyMap = new HashMap<Symbol, SubArray>();
			Map<Symbol, SubArray> presentMap = new HashMap<Symbol, SubArray>();
			Set<SubArray> pcopySet = null;
			Set<SubArray> copySet = null;
			Set<SubArray> pcopyinSet = null;
			Set<SubArray> copyinSet = null;
			Set<SubArray> copyinSet2 = null;
			Set<SubArray> presentSet = null;
			ACCAnnotation tempAnnot = at.getAnnotation(ACCAnnotation.class, "pcopy");
			if( tempAnnot != null ) {
				compAnnot = tempAnnot;
				pcopySet = tempAnnot.get("pcopy");
				for( SubArray tempSAr : pcopySet ) {
					Symbol temSym = AnalysisTools.subarrayToSymbol(tempSAr, IRSymbolOnly);
					pcopyMap.put(temSym, tempSAr);
				}
			}
			tempAnnot = at.getAnnotation(ACCAnnotation.class, "copy");
			if( tempAnnot != null ) {
				compAnnot = tempAnnot;
				copySet = tempAnnot.get("copy");
				for( SubArray tempSAr : copySet ) {
					Symbol temSym = AnalysisTools.subarrayToSymbol(tempSAr, IRSymbolOnly);
					copyMap.put(temSym, tempSAr);
				}
			}
			tempAnnot = at.getAnnotation(ACCAnnotation.class, "pcopyin");
			if( tempAnnot != null ) {
				compAnnot = tempAnnot;
				pcopyinSet = tempAnnot.get("pcopyin");
			}
			tempAnnot = at.getAnnotation(ACCAnnotation.class, "copyin");
			if( tempAnnot != null ) {
				compAnnot = tempAnnot;
				copyinSet = tempAnnot.get("copyin");
			}
			tempAnnot = cRegion.getAnnotation(ACCAnnotation.class, "present");
			if( tempAnnot != null ) {
				presentSet = tempAnnot.get("present");
				for( SubArray tempSAr : presentSet ) {
					Symbol temSym = AnalysisTools.subarrayToSymbol(tempSAr, IRSymbolOnly);
					presentMap.put(temSym, tempSAr);
				}
			}
			tempAnnot = cRegion.getAnnotation(ACCAnnotation.class, "copyin");
			if( tempAnnot != null ) {
				copyinSet2 = tempAnnot.get("copyin");
			}
			for( Symbol roSym : sharedROSet ) {
				if( !accExShared.contains(roSym) ) {
					if( pcopyMap.keySet().contains(roSym) ) {
						SubArray temSAr = pcopyMap.get(roSym);
						if( pcopyinSet == null ) {
							pcopyinSet = new HashSet<SubArray>();
							compAnnot.put("pcopyin", pcopyinSet);
						}
						pcopySet.remove(temSAr);
						pcopyinSet.add(temSAr);
					}
					if( copyMap.keySet().contains(roSym) ) {
						SubArray temSAr = copyMap.get(roSym);
						if( copyinSet == null ) {
							copyinSet = new HashSet<SubArray>();
							compAnnot.put("copyin", copyinSet);
						}
						copySet.remove(temSAr);
						copyinSet.add(temSAr);
					}
					if( presentMap.keySet().contains(roSym) ) {
						SubArray temSAr = presentMap.get(roSym);
						presentSet.remove(temSAr);
						if( copyinSet2 == null ) {
							copyinSet2 = new HashSet<SubArray>();
							tempAnnot = cRegion.getAnnotation(ACCAnnotation.class, "parallel");
							if( tempAnnot == null ) {
								tempAnnot = cRegion.getAnnotation(ACCAnnotation.class, "kernels");
							}
							if( tempAnnot != null ) {
								tempAnnot.put("copyin", copyinSet2);
							}
						}
						copyinSet2.add(temSAr);
					}
				}
			}
			if( !accreadonlySet.isEmpty() ) {
				tempAnnot = at.getAnnotation(ACCAnnotation.class, "internal");
				if( tempAnnot == null ) {
					tempAnnot = new ACCAnnotation("internal", "_directive");
					at.annotate(tempAnnot);
				}
				Set<Symbol> ROSet = new HashSet<Symbol>();
				ROSet.addAll(accreadonlySet);
				tempAnnot.put("accreadonly", ROSet);
			}
			if( !sharedROSubArraySet.isEmpty() ) {
				arcAnnot = at.getAnnotation(ARCAnnotation.class, "cuda");
				if( arcAnnot == null ) {
					arcAnnot = new ARCAnnotation("cuda", "_directive");
					at.annotate(arcAnnot);
				}
				Set<SubArray> ROSubArraySet = arcAnnot.get("sharedRO");
				if( ROSubArraySet == null ) {
					ROSubArraySet = new HashSet<SubArray>();
					arcAnnot.put("sharedRO", ROSubArraySet);
				}
				ROSubArraySet.addAll(sharedROSubArraySet);
			}
		}
	}

}
