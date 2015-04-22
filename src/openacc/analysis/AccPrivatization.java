/**
 * 
 */
package openacc.analysis;

import cetus.exec.Driver;
import cetus.hir.*;
import cetus.analysis.AnalysisPass;
import cetus.analysis.ArrayPrivatization;
import openacc.hir.ACCAnnotation;
import java.util.List;
import java.util.Set;
import java.util.HashSet;
import java.util.Map;
import java.util.HashMap;

/**
 * Privatize both scalar and array variables accessed in compute regions (parallel loops and kernels loops).
 * This pass should be called after ACCAnalysis pass.
 * 
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group, Oak Ridge National Laboratory
 *
 */
public class AccPrivatization extends AnalysisPass {

	private ArrayPrivatization privatizePass;
	private int optionLevel;
	private boolean IRSymbolOnly = true;
	/**
	 * @param program
	 */
	public AccPrivatization(Program program, int option, boolean IRSymOnly) {
		super(program);
		Driver.setOptionValue("privatize", new Integer(option).toString());
		privatizePass = new ArrayPrivatization(program);
		optionLevel = option;
		IRSymbolOnly = IRSymOnly;
	}

	/* (non-Javadoc)
	 * @see cetus.analysis.AnalysisPass#getPassName()
	 */
	@Override
	public String getPassName() {
		return "[AccPrivatization]";
	}

	/* (non-Javadoc)
	 * @see cetus.analysis.AnalysisPass#start()
	 */
	@Override
	public void start() {
		//TODO: For now, below is always executed, but if optionLevel == 1, simplified, scalar privatization code 
		//should be executed.
		if( optionLevel > 0 ) {
			String value = Driver.getOptionValue("AccParallelization");
			if( (value == null) || (Integer.valueOf(value).intValue() == 0)) {
				privatizePass.start();
			}
			//Step1: add privatisable variables in each OpenACC loop to its private clause.
			List<ACCAnnotation>  accAnnots = IRTools.collectPragmas(program, ACCAnnotation.class, "loop");
			if( accAnnots != null ) {
				for( ACCAnnotation cAnnot : accAnnots ) {
					Annotatable at = cAnnot.getAnnotatable();
					boolean kernelLoop = false;
					if( at.containsAnnotation(ACCAnnotation.class, "parallel") || 
							at.containsAnnotation(ACCAnnotation.class, "kernels")) {
						kernelLoop = true;
					}
					if( at.containsAnnotation(ACCAnnotation.class, "seq") ) {
						if( !kernelLoop && !AnalysisTools.ipContainPragmas(at, ACCAnnotation.class, 
								ACCAnnotation.parallelWorksharingClauses, false, null) ) {
							continue;
						}
					}
					ACCAnnotation enCompAnnot = null;
					Annotatable enCompRegion = null;
					if( !kernelLoop ) {
						enCompAnnot = AnalysisTools.ipFindFirstPragmaInParent(at, ACCAnnotation.class, ACCAnnotation.computeRegions, false, null, null);
/*						if( enCompAnnot.containsKey("parallel") ) {
							enCompRegion = enCompAnnot.getAnnotatable();
						}*/
						if( enCompAnnot != null ) {
							enCompRegion = enCompAnnot.getAnnotatable();
						}
					}
					Annotation cetusAnnot = at.getAnnotation(CetusAnnotation.class, "private");
					if( cetusAnnot != null ) {
						Set<Symbol> privatisables = (Set<Symbol>)cetusAnnot.get("private");
						Set<SubArray> privateSet =  null;
						Annotation tAnnot = at.getAnnotation(ACCAnnotation.class, "private");
						if( tAnnot == null ) {
							privateSet =  new HashSet<SubArray>();
						} else {
							privateSet =  (Set<SubArray>)tAnnot.get("private");
						}
						if( !kernelLoop ) {
							Set<Symbol> privateSymSet = AnalysisTools.subarraysToSymbols(privateSet, IRSymbolOnly);
							for( Symbol pSym : privatisables ) {
								if( !privateSymSet.contains(pSym) ) {
									boolean addPrivSym = true;
									if( (enCompRegion != null) && (at.containsAnnotation(ACCAnnotation.class, "gang")) ) {
										//Traversable tt = at.getParent();
										//[DEBUG] local variable defined in a gang loop is also gang private.
										Traversable tt = ((ForLoop)at).getBody();
										while ((tt != null) && !(tt instanceof Procedure)) {
											if( tt instanceof SymbolTable ) {
												Set<Symbol> localSyms = ((SymbolTable)tt).getSymbols();
												if( localSyms != null ) {
													if( localSyms.contains(pSym) ) {
														//IRSym is local variable defined either in the enclosing compute region
														//or in the current gang loop, which is gang private by default.
														addPrivSym = false;
														break;
													}
												}
											}
											if( tt == enCompRegion ) {
												break;
											} else {
												tt = tt.getParent();
											}
										}
									}
									if( addPrivSym ) {
										SubArray tSubArr = AnalysisTools.createSubArray(pSym, true, null);
										if( tSubArr != null ) {
											boolean foundDimInfo = false;
											if( tSubArr.getArrayDimension() < 0 ) {
												SubArray refArry = AnalysisTools.findSubArrayInDataClauses(enCompAnnot, pSym, IRSymbolOnly);
												if( (refArry != null) && (refArry.getArrayDimension() >= 0 ) ) {
													List<Expression> tStartL = refArry.getStartIndices();
													List<Expression> tLengthL = refArry.getLengths();
													tSubArr.setRange(tStartL, tLengthL);
													foundDimInfo = true;
												}
											} else {
												foundDimInfo = true;
											}
											if( foundDimInfo ) {
												privateSet.add(tSubArr);
											} else {
												PrintTools.println("[INFO] AccPrivatization found the following symbol can be privatized, " +
														"but necessary dimension information is missing; privatization of this symbol is skipped!\n" +
														"Symbol: " + pSym.getSymbolName() + "\nACCAnnotation: " + cAnnot + "\n", 0);
											}
										}
									}
								}
							}
							if( (tAnnot == null) && (!privateSet.isEmpty()) ) {
								cAnnot.put("private", privateSet);
							}
						} else {
							Annotation iAnnot = at.getAnnotation(ACCAnnotation.class, "accshared");
							if( iAnnot != null ) {
								Set<Symbol> accSharedSymbols = (Set<Symbol>)iAnnot.get("accshared");
								Set<Symbol>	accPrivateSymbols = (Set<Symbol>)iAnnot.get("accprivate");
								Set<SubArray> pcopySet =  null;
								tAnnot = at.getAnnotation(ACCAnnotation.class, "pcopy");
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
								for( Symbol pSym : privatisables ) {
									if( accSharedSymbols.contains(pSym) ) {
										SubArray sArry = AnalysisTools.subarrayOfSymbol(copySet, pSym);
										Set<SubArray> targetSet = copySet;
										String targetClause = "copy";
										if( sArry == null ) {
											sArry = AnalysisTools.subarrayOfSymbol(pcopySet, pSym);
											targetSet = pcopySet;
											targetClause = "pcopy";
										}
										if( sArry == null ) {
											PrintTools.println("[INFO] AccPrivatization found the following symbol can be privatized, " +
													"but it conflicts with user's annotation; privatization of this symbol is skipped!\n" +
													"Symbol: " + pSym.getSymbolName() + "\nACCAnnotation: " + cAnnot + "\nTo enforce privatization, put the " +
															"symbol into the pcopy clause.", 0);
											continue;
										} else if( sArry.getArrayDimension() < 0 ) {
											PrintTools.println("[INFO] AccPrivatization found the following symbol can be privatized, " +
													"but necessary dimension information is missing; privatization of this symbol is skipped!\n" +
													"Symbol: " + pSym.getSymbolName() + "\nACCAnnotation: " + cAnnot + "\n", 0);
										} else {
											targetSet.remove(sArry);
											if( targetSet.isEmpty() ) {
												tAnnot = at.getAnnotation(ACCAnnotation.class, targetClause);
												tAnnot.remove(targetClause);
											}
											privateSet.add(sArry);
											accSharedSymbols.remove(pSym);
											accPrivateSymbols.add(pSym);
										}
									}
								}
								if( (!privateSet.isEmpty()) && (!at.containsAnnotation(ACCAnnotation.class, "private")) ) {
									cAnnot.put("private", privateSet);
								}
							}
						}
					}
				}
			}
			//Step2: if a compute region is not a loop (kernel loops are handled in the previous step), 
			//       check privatizable condition according to the compute region types.
			accAnnots = AnalysisTools.collectPragmas(program, ACCAnnotation.class, ACCAnnotation.computeRegions, false);
			if( accAnnots != null ) {
				for( ACCAnnotation cAnnot : accAnnots ) {
					Annotatable at = cAnnot.getAnnotatable();
					if( at.containsAnnotation(ACCAnnotation.class, "loop") ) {
						continue; //skip kernel loops.
					}
					String kernelType = "parallel";
					if( at.containsAnnotation(ACCAnnotation.class, "kernels") ) {
						kernelType = "kernels";
					}
					//Step2-1: check all outermost gang loops lexically contained in this region have common privatisable variables.
					Set<Symbol> privatisables = ipCheckPrivatisablesInGangLoops(at);
					//Step2-2: remove privatisable variables from accShared set.
					//FIXME: If current region is parallel region, we need additional check to make sure that a variable
					//is not upward exposed at the entry of the region.
					Annotation iAnnot = at.getAnnotation(ACCAnnotation.class, "accshared");
					if( iAnnot != null ) {
						Set<Symbol> accSharedSymbols = (Set<Symbol>)iAnnot.get("accshared");
						Set<Symbol>	accPrivateSymbols = (Set<Symbol>)iAnnot.get("accprivate");
						//Set<Symbol> accReductionSymbols = (Set<Symbol>)iAnnot.get("accreduction");
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
						Set<SubArray> privateSet =  null;
						tAnnot = at.getAnnotation(ACCAnnotation.class, "private");
						if( tAnnot != null ) {
							privateSet =  (Set<SubArray>)tAnnot.get("private");
						}
						for( Symbol pSym : privatisables ) {
							if( accSharedSymbols.contains(pSym) ) {
								SubArray sArry = AnalysisTools.subarrayOfSymbol(copySet, pSym);
								Set<SubArray> targetSet = copySet;
								String targetClause = "copy";
								if( sArry == null ) {
									sArry = AnalysisTools.subarrayOfSymbol(pcopySet, pSym);
									targetSet = pcopySet;
									targetClause = "pcopy";
								}
								if( sArry == null ) {
									PrintTools.println("[INFO] AccPrivatization found the following symbol can be privatized, " +
											"but it conflicts with user's annotation; privatization of this symbol is skipped!\n" +
											"To enforce privatization, put the symbol in the pcopy clause.\n" +
											"Symbol: " + pSym.getSymbolName() + "\nACCAnnotation: " + cAnnot + "\n", 0);
									continue;
								} else {
									if( kernelType.equals("parallel") && (privateSet == null) ) {
										privateSet = new HashSet<SubArray>();
										cAnnot.put("private", privateSet);
									}
									targetSet.remove(sArry);
									if( targetSet.isEmpty() ) {
										tAnnot = at.getAnnotation(ACCAnnotation.class, targetClause);
										tAnnot.remove(targetClause);
									}
									if( kernelType.equals("parallel") ) {
										privateSet.add(sArry);
										accPrivateSymbols.add(pSym);
									}
									accSharedSymbols.remove(pSym);
								}
							}
						}
					}
				}
			}
		}
	}
	
	/**
	 * Return privatisable variables applicable to all outermost gang loops lexically included in the
	 * input code {@code t}.
	 * @param t input traversable where gang loops will be searched.
	 * @return set of privatisable symbols.
	 */
	private Set<Symbol> ipCheckPrivatisablesInGangLoops(Traversable t) {
		Set<Symbol> accessedSymbols = new HashSet<Symbol>();
		Set<Symbol> privatisableSymSet = new HashSet<Symbol>();
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
				Set<Symbol> tSet = AnalysisTools.getAccessedVariables(gAt, IRSymbolOnly);
				Set<Symbol> privatisables = new HashSet<Symbol>();
				Annotation privAnnot = gAt.getAnnotation(CetusAnnotation.class, "private");
				if( privAnnot != null ) {
					privatisables.addAll((Set<Symbol>)privAnnot.get("private"));
				}
				privAnnot = gAt.getAnnotation(ACCAnnotation.class, "private");
				if( privAnnot != null ) {
					privatisables.addAll(
							AnalysisTools.subarraysToSymbols((Set<SubArray>)privAnnot.get("private"), IRSymbolOnly));
				}
				Set<Symbol> removeSet = new HashSet<Symbol>();
				//Check existing privatisable variables are still privatisable in the current gang loop.
				for( Symbol tSym : privatisableSymSet ) {
					if( tSet.contains(tSym) && !privatisables.contains(tSym) ) {
						removeSet.add(tSym);
					}
				}
				privatisableSymSet.removeAll(removeSet);
				//Check whether the current gang loop has new privatisable variables.
				for( Symbol tPS : privatisables ) {
					if( !accessedSymbols.contains(tPS) ) {
						privatisableSymSet.add(tPS);
					}
				}
				accessedSymbols.addAll(tSet);
			}
		}

		//Check whether privatisable variables are accessed in the function called in this region.
		//If so, remove  them from the privatisable set.
		//LIMIT: there will be new privatisable variables in the called functions, but they are not 
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
				//If privatisable variables are used as a function call argument, conservatively remove them
				//from privatisable set.
				privatisableSymSet.removeAll(accSet);
				//If a privatisable variable is global, and it is accessed in a function called in this region,
				//conservatively remove it from privatisable set.
				Procedure calledProc = funCall.getProcedure();
				if( calledProc != null ) { 
					accSet = AnalysisTools.getIpAccessedGlobalSymbols(calledProc, gSymMap, null);
					//FIXME: this does not check the case where privatisable variable is a global variable passed 
					//as a function parameter to the current scope.
					privatisableSymSet.removeAll(accSet);
				}
			}

		}
		return privatisableSymSet;
	}
}

