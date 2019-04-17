/**
 * 
 */
package openacc.analysis;

import cetus.analysis.AnalysisPass;
import cetus.analysis.Reduction;
import cetus.hir.*;
import cetus.exec.Driver;
import java.util.*;
import openacc.hir.ReductionOperator;
import openacc.hir.*;

/**
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group, Oak Ridge National Laboratory
 *
 */
public class AccReduction extends AnalysisPass {
	
	private Reduction rudPass;
	private int optionLevel;
	private boolean IRSymbolOnly;

	/**
	 * @param program
	 */
	public AccReduction(Program program, int option, boolean IRSymOnly) {
		super(program);
		optionLevel = option;
		IRSymbolOnly = IRSymOnly;
		Driver.setOptionValue("reduction", new Integer(option).toString());
		rudPass = new Reduction(program);
	}

	/* (non-Javadoc)
	 * @see cetus.analysis.AnalysisPass#getPassName()
	 */
	@Override
	public String getPassName() {
		return "[AccReduction]";
	}

	/* (non-Javadoc)
	 * @see cetus.analysis.AnalysisPass#start()
	 */
	@Override
	public void start() {
		//Step1: Run Cetus reduction pass to identify all reduction variables.
		String value = Driver.getOptionValue("AccParallelization");
		if( (value == null) || (Integer.valueOf(value).intValue() == 0)) {
			rudPass.start();
		}
		//Step2: Update ACCAnnotations in each OpenACC loop directives, which include both parallel loops
		//       and kernels loops.
		List<ACCAnnotation>  accAnnots = IRTools.collectPragmas(program, ACCAnnotation.class, "loop");
		if( accAnnots != null ) {
			for( ACCAnnotation cAnnot : accAnnots ) {
				Annotatable at = cAnnot.getAnnotatable();
				if( at.containsAnnotation(ACCAnnotation.class, "seq") ) {
					//if( !AnalysisTools.ipContainPragmas(at, ACCAnnotation.class, ACCAnnotation.parallelWorksharingClauses, false) ) {
						continue;
					//}
				}
				boolean kernelLoop = false;
				if( at.containsAnnotation(ACCAnnotation.class, "parallel") || 
						at.containsAnnotation(ACCAnnotation.class, "kernels")) {
					kernelLoop = true;
				}
				Set<Symbol> loopPrivSyms = new HashSet<Symbol>();
				ACCAnnotation privAnnot = at.getAnnotation(ACCAnnotation.class, "private");
				if( privAnnot != null ) {
					Set<SubArray> privSet = privAnnot.get("private");
					loopPrivSyms.addAll(AnalysisTools.subarraysToSymbols(privSet, true));
				}
				ACCAnnotation enCompAnnot = null;
				Annotatable enParallelRegion = null;
				Set<Symbol> parallelRegionRedSymbols = new HashSet<Symbol>();
				if( !kernelLoop ) {
					enCompAnnot = AnalysisTools.ipFindFirstPragmaInParent(at, ACCAnnotation.class, ACCAnnotation.computeRegions, false, null, null);
					if( enCompAnnot == null ) {
								PrintTools.println("\n[WARNING in AccReduction.start()] can not find the enclosing compute region of the " +
										"following reduction loop;  " +
										cAnnot + "; reduction transformation for this loop will be skipped!\n" + AnalysisTools.getEnclosingContext(at), 0);
								continue;
						
					}
					if( enCompAnnot.containsKey("parallel") ) {
						enParallelRegion = enCompAnnot.getAnnotatable();
						ACCAnnotation pRedAnnot = enParallelRegion.getAnnotation(ACCAnnotation.class, "accreduction");
						if( pRedAnnot != null ) {
							parallelRegionRedSymbols.addAll((Set<Symbol>)pRedAnnot.get("accreduction"));
						}
					}
				}
				Annotation cetusAnnot = at.getAnnotation(CetusAnnotation.class, "reduction");
				if( cetusAnnot != null ) {
					Map<String, Set<Expression>> reduce_map = (Map<String, Set<Expression>>)cetusAnnot.get("reduction");
					if( (reduce_map == null) || reduce_map.isEmpty() ) {
						continue;
					}
					Map<ReductionOperator, Set<SubArray>> reduce_map2 = null;
					Annotation rAnnot = at.getAnnotation(ACCAnnotation.class, "reduction");
					if( rAnnot == null ) {
						reduce_map2 = new HashMap<ReductionOperator, Set<SubArray>>();
					} else {
						reduce_map2 = (Map<ReductionOperator, Set<SubArray>>)rAnnot.get("reduction");
					}
					Set<Symbol> reductionSyms = null;
					if( kernelLoop ) {
						Annotation iAnnot = at.getAnnotation(ACCAnnotation.class, "accreduction");
						if( iAnnot != null ) {
							reductionSyms = (Set<Symbol>)iAnnot.get("accreduction");
						}
					}
					if( reductionSyms == null ) {
						reductionSyms = new HashSet<Symbol>();
						for(ReductionOperator op : reduce_map2.keySet()) {
							Set<SubArray> redSyms1 = reduce_map2.get(op);
							Set<Symbol> redSyms2 = AnalysisTools.subarraysToSymbols(redSyms1, IRSymbolOnly);
							if( redSyms1.size() != redSyms2.size() ) {
								Tools.exit("[ERROR in AccReduction.start()]: cannot find symbols for " +
										"subarrays in a reduction clause in ACCAnnotation, " + cAnnot + 
										AnalysisTools.getEnclosingAnnotationContext(cAnnot));
							} else {
								reductionSyms.addAll(redSyms2);
							}
						}
					}
					Set<Symbol> newRedSyms = new HashSet<Symbol>();
					for( String op : reduce_map.keySet() ) {
						ReductionOperator redOp = ReductionOperator.fromString(op);
						Set<Expression> redExps = reduce_map.get(op);
						Set<SubArray> redSyms = null;
						if( reduce_map2.containsKey(redOp) ) {
							redSyms = reduce_map2.get(redOp);
						}
						for(Expression rExp : redExps ) {
							Symbol redSym = SymbolTools.getSymbolOf(rExp);
							if( redSym == null ) {
								PrintTools.println("\n[WARNING in AccReduction.start()] can not find the symbol of the " +
										"expression, " + rExp + ", used in a reduction clause in ACCAnnotation, " +
										cAnnot + "; reduction transformation for this reduction variable will be skipped!\n", 0);
							} else {
								//DEBUG: in current implementation, all internal symbol sets contains IR symbols.
								//       (ex: in accshared, accprivate, accfirstprivate, and accreduction)
								Symbol IRSym = redSym;
								if( redSym instanceof PseudoSymbol ) {
									IRSym = ((PseudoSymbol)redSym).getIRSymbol();
								}
								if( !reductionSyms.contains(IRSym) ) {
									//Cetus-found reduction variable should not be included in the OpenACC reduction clause in the
									//following cases:
									//1) current reduction loop is a gang loop, and the new variable is gang-private.
									//2) current reduction loop is a worker loop, and the new variable is worker-private.
									//3) current reduction loop is a vector loop, and the new variable is vector-private.
									//4) current reduction loop is a gang loop in a parallel region, and the parallel region has
									//   a reduction clause for the same variable.
									//Add this new reduction variable into OpenACC reduction clause.
									boolean addRedSym = true;
									if( loopPrivSyms.contains(IRSym) ) {
										addRedSym = false;
									}
									if( addRedSym && !kernelLoop ) {
										if( (enParallelRegion != null) && (at.containsAnnotation(ACCAnnotation.class, "gang")) ) {
											if( parallelRegionRedSymbols.contains(IRSym) ) {
												//Enclosing parallel region has a reduction clause containing the same variable.
												addRedSym = false;
											} else {
												Traversable tt = at.getParent();
												while ((tt != null) && !(tt instanceof Procedure)) {
													if( tt instanceof SymbolTable ) {
														Set<Symbol> localSyms = ((SymbolTable)tt).getSymbols();
														if( localSyms != null ) {
															if( localSyms.contains(IRSym) ) {
																//IRSym is local variable defined in the enclosing parallel region,
																//which is gang private by default.
																addRedSym = false;
																break;
															}
														}
													}
													if( tt == enParallelRegion ) {
														break;
													} else {
														tt = tt.getParent();
													}
												}
											}
										}
									}
									if( addRedSym ) {
										SubArray sArray = AnalysisTools.createSubArray(redSym, true, rExp);
										if( sArray != null ) {
											boolean foundDimInfo = false;
											if( sArray.getArrayDimension() < 0 ) {
												SubArray refArry = AnalysisTools.findSubArrayInDataClauses(enCompAnnot, redSym, IRSymbolOnly);
												if( (refArry != null) && (refArry.getArrayDimension() >= 0 ) ) {
													List<Expression> tStartL = refArry.getStartIndices();
													List<Expression> tLengthL = refArry.getLengths();
													sArray.setRange(tStartL, tLengthL);
													foundDimInfo = true;
												}
											} else {
												foundDimInfo = true;
											}
											if( foundDimInfo ) {
												if( redSyms == null ) {
													redSyms = new HashSet<SubArray>();
													reduce_map2.put(redOp, redSyms);
												}
												redSyms.add(sArray);
												newRedSyms.add(IRSym);
												reductionSyms.add(IRSym);
											} else {
												PrintTools.println("[INFO] AccReduction found the following symbol is a reduction variable, " +
														"but necessary dimension information is missing; reduction substitution of this symbol is skipped!\n" +
														"Symbol: " + redSym.getSymbolName() + "\nACCAnnotation: " + cAnnot + "\n", 0);
											}
										}
									}
								}
							}
						}
					}
					if( (rAnnot == null) && !newRedSyms.isEmpty() ) {
						cAnnot.put("reduction", reduce_map2);
					}
					//FIXME: the following comparison is based on IR symbols, and thus if struct members are allowed in OpenACC 
					// reduction clause, below update may incur incorrect output.
					if( !newRedSyms.isEmpty() && kernelLoop ) {
						Annotation iAnnot = at.getAnnotation(ACCAnnotation.class, "accshared");
						if( iAnnot != null ) {
							Set<Symbol> accSharedSymbols = (Set<Symbol>)iAnnot.get("accshared");
							Set<SubArray> pcopySet =  null;
							Annotation tAnnot = at.getAnnotation(ACCAnnotation.class, "pcopy");
							if( tAnnot != null ) {
								pcopySet =  (Set<SubArray>)tAnnot.get("pcopy");
							} else {
								pcopySet =  new HashSet<SubArray>();
							}
							Set<SubArray> copySet =  null;
							tAnnot = at.getAnnotation(ACCAnnotation.class, "copy");
							if( tAnnot != null ) {
								copySet =  (Set<SubArray>)tAnnot.get("copy");
							} else {
								copySet =  new HashSet<SubArray>();
							}
							for( Symbol rSym : newRedSyms ) {
								if( accSharedSymbols.contains(rSym) ) {
									SubArray sArry = AnalysisTools.subarrayOfSymbol(copySet, rSym);
									Set<SubArray> targetSet = copySet;
									String targetClause = "copy";
									if( sArry == null ) {
										sArry = AnalysisTools.subarrayOfSymbol(pcopySet, rSym);
										targetSet = pcopySet;
										targetClause = "pcopy";
									}
									if( sArry == null ) {
										PrintTools.println("[INFO] AccReduction found the following symbol is a reduction variable, " +
												"but it conflicts with user's annotation; reduction of this symbol may cause unexpected side effect!\n" +
												"Symbol: " + rSym + "\nACCAnnotation: " + cAnnot + "\n", 1);
										continue;
									} else {
										targetSet.remove(sArry);
										if( targetSet.isEmpty() ) {
											tAnnot = at.getAnnotation(ACCAnnotation.class, targetClause);
											tAnnot.remove(targetClause);
										}
										accSharedSymbols.remove(rSym);
									}
								}
							}
						}
					}
				}
			}
		}
		//Step3: if a compute region is not a loop (kernel loops are handled in the previous step), 
		//       of if a seq kernel loop contains inner gang loops,
		//       check whether reduction variables in gang loops of the region are used only as reduction variables.
		//       If so, we can update ACCAnnotations for the region too.
		Map<Symbol, List> redSymMap = new HashMap<Symbol, List>();
		Map<Symbol, List> redSymMapT;
		accAnnots = AnalysisTools.collectPragmas(program, ACCAnnotation.class, ACCAnnotation.computeRegions, false);
		if( accAnnots != null ) {
			for( ACCAnnotation cAnnot : accAnnots ) {
				Annotatable at = cAnnot.getAnnotatable();
				if( at.containsAnnotation(ACCAnnotation.class, "loop") ) {
					if( !at.containsAnnotation(ACCAnnotation.class, "seq") || 
							!AnalysisTools.ipContainPragmas(at, ACCAnnotation.class, "gang", null) ) {
						continue; //skip kernel loops.
					}
				}
				String kernelType = "parallel";
				if( at.containsAnnotation(ACCAnnotation.class, "kernels") ) {
					kernelType = "kernels";
				}
				//Step3-1: check all outermost gang loops lexically contained in this region have common reduction variables.
				redSymMapT = ipCheckRedVariablesInGangLoops(at, kernelType);
				//Step3-2: remove reduction variables from accShared set.
				//FIXME: If current region is parallel region, we need additional check to make sure that a variable
				//is used only as a reduction variable.
				Annotation iAnnot = at.getAnnotation(ACCAnnotation.class, "accshared");
				if( iAnnot != null ) {
					Set<Symbol> accSharedSymbols = (Set<Symbol>)iAnnot.get("accshared");
					Set<Symbol> accReductionSymbols = (Set<Symbol>)iAnnot.get("accreduction");
					Set<SubArray> pcopySet =  null;
					Annotation tAnnot = at.getAnnotation(ACCAnnotation.class, "pcopy");
					if( tAnnot != null ) {
						pcopySet =  (Set<SubArray>)tAnnot.get("pcopy");
					} else {
						pcopySet =  new HashSet<SubArray>();
					}
					Set<SubArray> copySet =  null;
					tAnnot = at.getAnnotation(ACCAnnotation.class, "copy");
					if( tAnnot != null ) {
						copySet =  (Set<SubArray>)tAnnot.get("copy");
					} else {
						copySet =  new HashSet<SubArray>();
					}
					Map<ReductionOperator, Set<SubArray>> reduce_map2 = null;
					Annotation rAnnot = at.getAnnotation(ACCAnnotation.class, "reduction");
					if( rAnnot == null ) {
						reduce_map2 = new HashMap<ReductionOperator, Set<SubArray>>();
					} else {
						reduce_map2 = (Map<ReductionOperator, Set<SubArray>>)rAnnot.get("reduction");
					}
/*					List<ACCAnnotation> gangAnnots = IRTools.collectPragmas(at, ACCAnnotation.class, "gang");
					List<ForLoop> outerGangLoops = new LinkedList<ForLoop>();
					if( gangAnnots != null ) {
						for( ACCAnnotation gAnnot : gangAnnots ) {
							Annotatable gAt = gAnnot.getAnnotatable();
							boolean isInnerGangLoop = false;
							Traversable tt = gAt.getParent();
							while (tt != null ) {
								if( (tt instanceof Annotatable) && ((Annotatable)tt).containsAnnotation(ACCAnnotation.class, "gang") ) {
									isInnerGangLoop = true;
									break;
								} else {
									tt = tt.getParent();
								}
							}
							if( !isInnerGangLoop ) {
								if( gAt.containsAnnotation(ACCAnnotation.class, "reduction") ) {
									outerGangLoops.add((ForLoop)gAt);
								}
							}
						}
					}
*/					for( Symbol rSym : redSymMapT.keySet() ) {
						if( accSharedSymbols.contains(rSym) ) {
							SubArray sArry = AnalysisTools.subarrayOfSymbol(copySet, rSym);
							Set<SubArray> targetSet = copySet;
							String targetClause = "copy";
							if( sArry == null ) {
								sArry = AnalysisTools.subarrayOfSymbol(pcopySet, rSym);
								targetSet = pcopySet;
								targetClause = "pcopy";
							}
							if( sArry == null ) {
								PrintTools.println("[INFO] AccReduction found the following symbol is a reduction variable, " +
										"but it conflicts with user's annotation; reduction of this symbol may cause unexpected side effect!\n" +
										"Symbol: " + rSym + "\nACCAnnotation: " + cAnnot + "\n", 1);
								continue;
							} else {
/*								if( kernelType.equals("parallel") && (rAnnot == null) ) {
									cAnnot.put("reduction", reduce_map2);
								}
*/								targetSet.remove(sArry);
								if( targetSet.isEmpty() ) {
									tAnnot = at.getAnnotation(ACCAnnotation.class, targetClause);
									tAnnot.remove(targetClause);
								}
/*								if( kernelType.equals("parallel") ) {
									List tArray = redSymMapT.get(rSym);
									ReductionOperator op = (ReductionOperator)tArray.get(0);
									SubArray sAr = (SubArray)tArray.get(1);
									Set<SubArray> redSet = null;
									if( reduce_map2.containsKey(op) ) {
										redSet = reduce_map2.get(op);
									} else {
										redSet = new HashSet<SubArray>();
										reduce_map2.put(op, redSet);
									}
									redSet.add(sAr);
									accReductionSymbols.add(rSym);
									//Remove the reduction symbol from the previous clause.
									for( ForLoop gLoop : outerGangLoops ) {
										ACCAnnotation tRAnnot = gLoop.getAnnotation(ACCAnnotation.class, "reduction");
										Map<ReductionOperator, Set<SubArray>> tRedMap = tRAnnot.get("reduction");
										Set<SubArray> tRSet = tRedMap.get(op);
										SubArray remSA = null;
										for( SubArray tSubA : tRSet ) {
											Symbol tRsym = AnalysisTools.subarrayToSymbol(tSubA, IRSymbolOnly);
											if( tRsym.equals(rSym) ) {
												remSA = tSubA;
											}
										}
										if( remSA != null ) {
											tRSet.remove(remSA);
											if( tRSet.isEmpty() ) {
												tRedMap.remove(op);
												if( tRedMap.isEmpty() ) {
													tRAnnot.remove("reduction");
												}
											}
											break;
										}
									}
								}
*/								accSharedSymbols.remove(rSym);
							}
						}
					}
				}
			}
		}
	}
	
	/**
	 * Return reduction variables applicable to all outermost gang loops lexically included in the
	 * input code {@code t}.
	 * @param t input traversable where gang loops will be searched.
	 * @param kernelType kernel type
	 * @return map of reduction symbols.
	 */
	private Map<Symbol, List> ipCheckRedVariablesInGangLoops(Traversable t, String kernelType) {
		Set<Symbol> accessedSymbols = new HashSet<Symbol>();
		Map<Symbol, List> redSymMap = new HashMap<Symbol, List>();
		Traversable tt = t;
		while( tt != null ) {
			if( tt instanceof Procedure ) break;
			tt = tt.getParent();
		}
		Procedure pProc = (Procedure)tt;
		List<ACCAnnotation> gangAnnots = IRTools.collectPragmas(t, ACCAnnotation.class, "gang");
		if( gangAnnots != null ) {
			for( ACCAnnotation gAnnot : gangAnnots ) {
				Annotatable gAt = gAnnot.getAnnotatable();
				boolean isInnerGangLoop = false;
				tt = gAt.getParent();
				while (tt != null ) {
					if( (tt instanceof Annotatable) && ((Annotatable)tt).containsAnnotation(ACCAnnotation.class, "gang") ) {
						isInnerGangLoop = true;
						break;
					} else {
						tt = tt.getParent();
					}
				}
				if( isInnerGangLoop ) {
					continue; //The method checks only the outermost gang loops.
				}
				Map<Symbol, List> redSymMapT = new HashMap<Symbol, List>();
				Set<Symbol> tSet = AnalysisTools.getAccessedVariables(gAt, IRSymbolOnly);
				//Find new reduction variables for the current gang loop.
				Map<ReductionOperator, Set<SubArray>> reduce_map2 = null;
				Annotation rAnnot = gAt.getAnnotation(ACCAnnotation.class, "reduction");
				if( rAnnot != null ) {
					reduce_map2 = (Map<ReductionOperator, Set<SubArray>>)rAnnot.get("reduction");
				}
				if( reduce_map2 != null ) {
					for(ReductionOperator op : reduce_map2.keySet()) {
						Set<SubArray> redSyms1 = reduce_map2.get(op);
						for( SubArray subArr : redSyms1 ) {
							Symbol sym = SymbolTools.getSymbolOf(subArr.getArrayName());
							if( sym == null ) {
								PrintTools.println("\n[WARNING in AccReduction.ipCheckRedVariablesInGangLoops()]: " +
										"cannot find the symbol for the subarray, " + subArr  + "\n", 0);
							} else {
								if( sym instanceof PseudoSymbol ) {
									sym = ((PseudoSymbol)sym).getIRSymbol();
								}
								List tArray = null;
								tArray = new ArrayList(2);
								tArray.add(op);
								tArray.add(subArr);
								if( redSymMapT.containsKey(sym) ) {
									Tools.exit("[ERROR in AccReduction] a reduction variable should" +
											" be specified only once in the same OpenACC reduction clause; \n" + 
											"Reduction variable: " + sym + "\n" + "OpenACC annotation: " + rAnnot + 
											AnalysisTools.getEnclosingAnnotationContext(rAnnot));

								} else {
									redSymMapT.put(sym, tArray);
								}
							}
						}
					}
				}
				//Check existing reduction variables are accessed in the current gang loop.
				//In a parallel region, a gang reduction variable should appear in only one gang loop.
				//In a kernels region, if a gang reduction variable appear in all enclosed gang loops,
				//it can be removed from accshared set.
				Set<Symbol> removeSet = new HashSet<Symbol>();
				for( Symbol tSym : redSymMap.keySet() ) {
					if( kernelType.equals("parallel") ) {
						if( tSet.contains(tSym) ) {
							Tools.exit("[ERROR in AccReduction] in a parallel region, a gang loop reduction variable should" +
									" be used only in one gang loop; \n" + "Reduction variable: " + tSym + "\n" +
									"Parallel region " + t + AnalysisTools.getEnclosingContext(t));
						}
					} else { //kernels region
						if( tSet.contains(tSym) && !redSymMapT.containsKey(tSym) ) {
							removeSet.add(tSym);
						}
					}
					removeSet.add(tSym);
				}
				for( Symbol tSym : removeSet ) {
					redSymMap.remove(tSym);
				}
				//Check whether the current gang loop has new reduction variables.
				for( Symbol tSym : redSymMapT.keySet() ) {
					if( !accessedSymbols.contains(tSym) ) {
						redSymMap.put(tSym, redSymMapT.get(tSym));
					}
				}
				accessedSymbols.addAll(tSet);
			}
		}

		//Check whether reduction variables are accessed in the function called in this region.
		//If so, remove  them from the reduction set conservatively.
		//LIMIT: there will be new reduction variables in the called functions, but they are not 
		//checked in the current version.
		Set<Symbol> accSet = new HashSet<Symbol>();
		List<FunctionCall> gFuncCallList = IRTools.getFunctionCalls(program);
		List<FunctionCall> funcCallList = IRTools.getFunctionCalls(t);
		if( funcCallList != null ) {
			Map<String, Symbol> gSymMap = null;
			if( gSymMap == null ) {
				Set<Symbol> tSet = SymbolTools.getGlobalSymbols(t);
				gSymMap = new HashMap<String, Symbol>();
				for( Symbol gS : tSet ) {
					gSymMap.put(gS.getSymbolName(), gS);
				}
			} 
			for( FunctionCall funCall : funcCallList ) {
				//We don't have to check functions called inside gang loops in the current scope.
				tt = funCall.getParent();
				boolean calledInGangLoop = false;
				while (tt != null ) {
					if( (tt instanceof Annotatable) && ((Annotatable)tt).containsAnnotation(ACCAnnotation.class, "gang") ) {
						calledInGangLoop = true;
						break;
					} else {
						tt = tt.getParent();
					}
				}
				if( calledInGangLoop ) {
					continue;
				}
				accSet = AnalysisTools.getAccessedVariables(funCall, IRSymbolOnly);
				//If reduction variables are used as a function call argument, conservatively remove them
				//from privatisable set.
				for( Symbol tSym : accSet ) {
					redSymMap.remove(tSym);
				}
				//If a reduction variable is global, and it is accessed in a function called in this region,
				//conservatively remove it from reduction set.
				Procedure calledProc = funCall.getProcedure();
				if( calledProc != null ) { 
					accSet = AnalysisTools.getIpAccessedGlobalSymbols(calledProc, gSymMap, null);
					//FIXME: this does not check the case where reduction variable is a global variable passed 
					//as a function parameter to the current scope.
					for( Symbol tSym : accSet ) {
						redSymMap.remove(tSym);
					}
				}
			}
		}
		return redSymMap;
	}

}
