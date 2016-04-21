package openacc.transforms;

import java.util.*;

import cetus.hir.*;
import cetus.analysis.LoopTools;
import openacc.analysis.AnalysisTools;
import openacc.analysis.SubArray;
import openacc.hir.ACCAnnotation;
import openacc.hir.ARCAnnotation;
import openacc.hir.CUDASpecifier;
import openacc.hir.OpenCLSpecifier;
import openacc.hir.ReductionOperator;

public abstract class ACC2GPUTranslationTools {
	private static int tempIndexBase = 1000;
	
	/**
	 * Java doesn't allow a class to be both abstract and final,
	 * so this private constructor prevents any derivations.
	 */
	private ACC2GPUTranslationTools()
	{
	}  
	
	/**
	 * If an shared array element (ex: A[i]) is included in the cuda registerRO or 
	 * cuda registerRW set, the element is cached in the GPU register.
	 * [FIXME] If aliasing between any two array access expressions exists and at least
	 * one array access is DEF expression, below transformation is incorrect.
	 * 
	 * @param region
	 * @param cudaRegisterSet
	 * @param cudaRegisterROSet
	 * @param arraySymbol
	 * return set of strings of cached array elements
	 */
	protected static Set<Symbol> arrayCachingOnRegister(Statement region, Map<Symbol, Set<SubArray>> shrdArryOnRegMap,
			Set<SubArray> ROShrdArryOnRegSet) {
		Set<Symbol> arrayElmtCacheSymbols = new HashSet<Symbol>();
		HashSet<String> cachedArrayElmts = new HashSet<String>();
		HashSet<String> checkedArrayAccessSet = new HashSet<String>();
		CompoundStatement targetStmt = null;
		boolean isCompoundStmt = false;
		//PrintTools.println("arrayCachingOnRegister() begin", 0);
		//PrintTools.println("current symbol: " + arraySymbol, 0);
		if( shrdArryOnRegMap.isEmpty() ) {
			//PrintTools.println("arrayCachingOnRegister() end", 0);
			return arrayElmtCacheSymbols;
		}
		Set<String> cudaRegisterSet = new HashSet<String>();
		Set<String> cudaRegisterROSet = new HashSet<String>();
		for( Symbol tSym : shrdArryOnRegMap.keySet() ) {
			Set<SubArray> sSet = shrdArryOnRegMap.get(tSym);
			for( SubArray tSubA : sSet ) {
				List<Expression> indices = new LinkedList<Expression>();
				for( int i=0; i<tSubA.getArrayDimension(); i++ ) {
					List<Expression> range = tSubA.getRange(i);
					indices.add(range.get(0).clone());
				}
				Expression accessEx = null;
				Expression IDEx = tSubA.getArrayName();
				if( IDEx instanceof IDExpression ) {
					accessEx = new ArrayAccess(IDEx.clone(), indices);
				} else if( IDEx instanceof AccessExpression ) {
					AccessExpression accEx = (AccessExpression)IDEx.clone();
					Expression RHS = accEx.getRHS();
					while ( RHS instanceof AccessExpression ) {
						RHS = ((AccessExpression)RHS).getRHS();
					}
					ArrayAccess aAccess = new ArrayAccess(RHS.clone(), indices);
					RHS.swapWith(aAccess);
					accessEx = accEx;
				}
				cudaRegisterSet.add(accessEx.toString());
				if( ROShrdArryOnRegSet.contains(tSubA) ) {
					cudaRegisterROSet.add(accessEx.toString());
				}
			}
		}
		//PrintTools.println("current registerset: " + cudaRegisterSet, 0);
		if( region instanceof CompoundStatement ) {
			targetStmt = (CompoundStatement)region;
			isCompoundStmt = true;
		} else if( region instanceof ForLoop ) {
			targetStmt = (CompoundStatement)((ForLoop)region).getBody();
		} else {
			Tools.exit("[ERROR] Unknwon region in arrayCachingOnRegister(): "
					+ region.toString());
		}
		for( Symbol tSym : shrdArryOnRegMap.keySet() ) {
			String symName = null;
			if( tSym instanceof AccessSymbol ) {
				symName = TransformTools.buildAccessSymbolName((AccessSymbol)tSym);
			} else {
				symName = tSym.getSymbolName();
			}
			List<Specifier> specs = new LinkedList<Specifier>();
			
			List<Specifier> removeSpecs = new ArrayList<Specifier>();
			removeSpecs.add(Specifier.STATIC);
			removeSpecs.add(Specifier.CONST);
			removeSpecs.add(Specifier.EXTERN);
			for( Specifier obj : (List<Specifier>)tSym.getTypeSpecifiers() ) {
				if( !removeSpecs.contains(obj) && !(obj instanceof PointerSpecifier) ) {
					specs.add(obj);
				}
			}
			
			/////////////////////////////////////////////////////////////////////////
			// Find array access expressions that will be cached on the registers. //
			/////////////////////////////////////////////////////////////////////////
			HashMap<String, Expression> aAccessMap = new HashMap<String, Expression>();
			List<Expression> matches = new ArrayList<Expression>(4);
			DFIterator<Expression> DFiter =
					new DFIterator<Expression>(targetStmt, Expression.class);
			while (DFiter.hasNext()) {
				Expression child = DFiter.next();
				Symbol nSym = SymbolTools.getSymbolOf(child);
				if( (nSym != null) && (nSym.equals(tSym)) ) {
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
					String aAccessString = match.toString();
					////////////////////////////////////
					// Remove any '(', ')', or space. //
					////////////////////////////////////
					StringBuilder strB = new StringBuilder(aAccessString);
					int index = strB.toString().indexOf('(');
					while ( index != -1 ) {
						strB = strB.deleteCharAt(index);
						index = strB.toString().indexOf('(');
					}
					index = strB.toString().indexOf(')');
					while ( index != -1 ) {
						strB = strB.deleteCharAt(index);
						index = strB.toString().indexOf(')');
					}
					index = strB.toString().indexOf(' ');
					while ( index != -1 ) {
						strB = strB.deleteCharAt(index);
						index = strB.toString().indexOf(' ');
					}
					String aAccessString2 = strB.toString();

					if( (cudaRegisterSet.contains(aAccessString) 
							|| cudaRegisterSet.contains(aAccessString2)) ) {
						aAccessMap.put(aAccessString, match);
					}
				}
			}

			for( Expression aAccess : aAccessMap.values() ) {
				String aAccessString = aAccess.toString();
				//PrintTools.println("aAccessString to check " + aAccessString, 0);
				////////////////////////////////////
				// Remove any '(', ')', or space. //
				////////////////////////////////////
				StringBuilder strB = new StringBuilder(aAccessString);
				int index = strB.toString().indexOf('(');
				while ( index != -1 ) {
					strB = strB.deleteCharAt(index);
					index = strB.toString().indexOf('(');
				}
				index = strB.toString().indexOf(')');
				while ( index != -1 ) {
					strB = strB.deleteCharAt(index);
					index = strB.toString().indexOf(')');
				}
				index = strB.toString().indexOf(' ');
				while ( index != -1 ) {
					strB = strB.deleteCharAt(index);
					index = strB.toString().indexOf(' ');
				}
				String aAccessString2 = strB.toString();

				// Insert a statement to load the global variable to register at the beginning
				//and a statement to dump register value to the global variable at the end
				// SymbolTools.getTemp() inserts the new temp symbol to the symbol table of the closest parent
				// if region is a loop.
				Identifier local_var = SymbolTools.getTemp(targetStmt, specs, symName);
				arrayElmtCacheSymbols.add(local_var.getSymbol());
				Statement estmt = new ExpressionStatement(new AssignmentExpression(local_var, 
						AssignmentOperator.NORMAL, aAccess.clone() ));
				Statement astmt = new ExpressionStatement(new AssignmentExpression( 
						aAccess.clone(), AssignmentOperator.NORMAL, local_var.clone()));
				List<Expression> indexList = null;
				if( aAccess instanceof ArrayAccess ) {
					indexList = ((ArrayAccess)aAccess).getIndices();
				} else if( aAccess instanceof AccessExpression ) {
					Expression tExp = ((AccessExpression)aAccess).getRHS();
					while( tExp instanceof AccessExpression ) {
						tExp = ((AccessExpression)tExp).getRHS();
					}
					if( tExp instanceof ArrayAccess ) {
						indexList = ((ArrayAccess)tExp).getIndices();
					} else {
						Tools.exit("[ERROR in ACC2GPUTranslationTools.arrayCachingOnRegister()] unexpected expression ("+ aAccess +  
								") for registerRO or registerRW clause; only scalar or array access expression is allowed; exit");
					}
				} else {
					Tools.exit("[ERROR in ACC2GPUTranslationTools.arrayCachingOnRegister()] unexpected expression ("+ aAccess +  
							") for registerRO or registerRW clause; only scalar or array access expression is allowed; exit");
				}
				boolean constantIndices = true;
				boolean containsIndirectAccess = false;
				List<ArrayAccess> indexArrayList = null;
				for( Expression indExp : indexList ) {
					if( !(indExp instanceof Literal) ) {
						constantIndices = false;
						break;
					}
					indexArrayList = IRTools.getExpressionsOfType(indExp, ArrayAccess.class);
					if( (indexArrayList != null) && (!indexArrayList.isEmpty()) ) {
						containsIndirectAccess = true;
						break;
					}
				}
				//[FIXME] Constant index array transformation is incorrect; disable it for now.
/*				if( constantIndices ) {
					///////////////////////////////////////////////////////////////////////////
					// If array access has constant index expressions, we can insert loading //
					// statement at the beginning and dump statement at the end of the target//
					// region.                                                               //
					///////////////////////////////////////////////////////////////////////////
					if( !isCompoundStmt ) {
						////////////////////////////////////////////////////////////////////////
						// check whether kernels/parallel loop body contains the array access //
						// more than once.                                                    //
						////////////////////////////////////////////////////////////////////////
						boolean foundFirstArrayAccess = false;
						Expression firstAccess = null;
						Expression lastAccess = null;
						Statement firstAccessStmt = null;
						Statement lastAccessStmt = null;
						DepthFirstIterator iter = new DepthFirstIterator(targetStmt);
						for (;;)
						{
							Expression tAccess = null;

							try {
								tAccess = (Expression)iter.next(Expression.class);
							} catch (NoSuchElementException e) {
								break;
							}
							if( aAccess.equals(tAccess) ) {
								if( !foundFirstArrayAccess ) {
									firstAccess = tAccess;
									foundFirstArrayAccess = true;
								}
								lastAccess = tAccess;
							}
						}
						if( (!foundFirstArrayAccess) || (firstAccess == lastAccess) ) {
							continue;
						}
						Traversable t = (Traversable)firstAccess;
						while( !(t instanceof Statement) ) {
							t = t.getParent();
						}
						if( t instanceof Statement ) {
							firstAccessStmt = (Statement)t;
						}
						t = (Traversable)lastAccess;
						while( !(t instanceof Statement) ) {
							t = t.getParent();
						}
						if( t instanceof Statement ) {
							lastAccessStmt = (Statement)t;
						}

						//DEBUG: if below check is for detecting reduction array, it
						//can be removed, since reduction variable has been already excluded.
						//if( firstAccessStmt == lastAccessStmt ) {
						//	continue;
						//}
					}
					if( cudaRegisterSet.contains(aAccessString) ) {
						cachedArrayElmts.add(aAccessString);
					} else if( cudaRegisterSet.contains(aAccessString2) ) {
						cachedArrayElmts.add(aAccessString2);
					}
					// Replace all instances of the shared variable to the local variable
					IRTools.replaceAll((Traversable) targetStmt, aAccess, local_var);
					Statement fStmt = IRTools.getFirstNonDeclarationStatement(targetStmt);
					if( fStmt == null ) {
						targetStmt.addStatement(estmt.clone());
					} else {
						targetStmt.addStatementBefore(fStmt, estmt.clone());
					}
					if( !cudaRegisterROSet.contains(aAccessString) && 
							!cudaRegisterROSet.contains(aAccessString2) ) {
						targetStmt.addStatement(astmt.clone());
					}
				} else */
				if( containsIndirectAccess ) {
					if( !checkedArrayAccessSet.contains(aAccessString) ) {
						PrintTools.println("[INFO in arrayCachingOnRegister()] " +
								"the array access, " + aAccess + 
								", has indirect array access pattern, which is not well supported in the " +
								"current implementation, and thus it will not be cached conservatively," +
								"even though it may have locality.", 0);
						checkedArrayAccessSet.add(aAccessString);
					}
				} else {
					HashSet<ForLoop> targetLoops = new HashSet<ForLoop>();
					//Find symobls used in the index expression
					Set<Symbol> indexSyms = new HashSet<Symbol>();
					if( !constantIndices ) {
						for( Expression tExp : indexList) {
							indexSyms.addAll(DataFlowTools.getUseSymbol(tExp));
						}
					}
					List<Statement> defStmts;
					if( indexSyms.isEmpty() ) {
						defStmts = new LinkedList<Statement>();
					} else {
						defStmts = AnalysisTools.getDefStmts(indexSyms, targetStmt);
					}
					boolean indexLoopExist = false;
					if( !defStmts.isEmpty() ) {
						///////////////////////////////////////////////////////////////////////
						// Find inner-most loops containing the array access and their index //
						// variables are used in the array access.                           //
						///////////////////////////////////////////////////////////////////////
						//DEBUG: Is it OK to use region for targetStmt2, instead of using targetStmt?
						Statement targetStmt2 = region;
						DepthFirstIterator iter = new DepthFirstIterator(targetStmt2);
						for (;;)
						{
							Expression tAccess = null;

							try {
								tAccess = (Expression)iter.next(Expression.class);
							} catch (NoSuchElementException e) {
								break;
							}
							if( aAccess.equals(tAccess) ) {
								Traversable t = tAccess.getParent();
								Symbol indexSym = null;
								Identifier tIndexVar = null;
								boolean foundLoop = false;
								while( t != targetStmt2 ) {
									while( !(t instanceof ForLoop) && !(t == targetStmt2) ) {
										t = t.getParent();
									}
									if( t instanceof ForLoop ) {
										tIndexVar = (Identifier)LoopTools.getIndexVariable((ForLoop)t);
										if( tIndexVar ==  null ) {
											indexSym = null;
										} else {
											indexSym = tIndexVar.getSymbol();
										}
										if( IRTools.containsSymbol(tAccess, indexSym) ) {
											if( !foundLoop ) {
												foundLoop = true;
												indexLoopExist = true;
												ForLoop cLoop = (ForLoop)t;
												Statement cLoopBody = cLoop.getBody();
												//Check whether cLoop is a parent of any defStmts.
												boolean innerDEFExist = false;
												for(Statement ttStmt : defStmts) {
													Traversable pp = ttStmt;
													while( (pp != null) && (pp.getParent() != targetStmt) && (pp.getParent() != cLoopBody) ) {
														pp = pp.getParent();
													}
													if( (pp != null) && (pp.getParent() == cLoopBody) ) {
														innerDEFExist = true;
														break;
													}
												}
												if( (!innerDEFExist) && (!targetLoops.contains(cLoop)) ) {
													boolean addLoop = true;
													//Check whether cLoop is a parent of one of targetLoops.
													for(ForLoop ttL : targetLoops ) {
														Traversable ttt = ttL.getParent();
														while( ttt != null ) {
															if( ttt == t ) {
																//cLoop is parent of ttL.
																addLoop = false;
																break;
															}
															ttt = ttt.getParent();
														}
														if( !addLoop ) {
															break;
														}
													}
													if(addLoop) {
														//Check whether cLoop is a child of one of targetLoops.
														Set<ForLoop> removeSet = new HashSet<ForLoop>();
														for(ForLoop ttL : targetLoops ) {
															Traversable ttt = cLoop.getParent();
															while( ttt != null ) {
																if( ttt == ttL ) {
																	//cLoop is a child of ttL.
																	removeSet.add(ttL);
																	break;
																}
																ttt = ttt.getParent();
															}
														}
														if( !removeSet.isEmpty() ) {
															targetLoops.removeAll(removeSet);
														}
														targetLoops.add(cLoop);
													}
												}
											}
											break;
										} else {
											if( t == targetStmt2 ) {
												break;
											} else {
												t = t.getParent();
											}
										}
									}
								}
								/*							if( !foundLoop ) {
								/////////////////////////////////////////////////////////////
								// FIXME: The array access is independent of any enclosing //
								// loops. If the index expressions of the array access are //
								// not constant, the dependency of the index expressions   //
								// should be checked; for now, array access with variable  //
								// index expressions are not cached.                       //
								/////////////////////////////////////////////////////////////
								if( !checkedArrayAccessSet.contains(aAccessString) ) {
									PrintTools.println("[INFO in arrayCachingOnRegister()] " +
											"index expression of the array access, " + aAccess + 
											", is independent of any enclosing loops, but not constant. " +
											"Current system does not detect the possible dependency problem of this " +
											"index expression, and thus it will not be cached conservatively, " +
											"even though it may have locality.", 0);
									checkedArrayAccessSet.add(aAccessString);
								}
							}*/
							}
						}
						/*					if( targetLoops.size() > 1 ) {
						PrintTools.println("[WARNING in ACC2GPUTranslationToos.arrayCachingOnRegister()] Multiple target loops are found for array " + aAccess + AnalysisTools.getEnclosingContext(aAccess), 0);
					}*/
						for( ForLoop fLoop : targetLoops ) {
							//PrintTools.println("fLoop check point1 ", 0);
							///////////////////////////////////////////////////
							// Find the first instance of this array access. //
							///////////////////////////////////////////////////
							Expression firstAccess = null;
							Expression lastAccess = null;
							Statement firstAccessStmt = null;
							Statement lastAccessStmt = null;
							boolean foundFirstArrayAccess = false;
							iter = new DepthFirstIterator(fLoop);
							for (;;)
							{
								Expression tAccess = null;

								try {
									tAccess = (Expression)iter.next(Expression.class);
								} catch (NoSuchElementException e) {
									break;
								}
								if( aAccess.equals(tAccess) ) {
									if( !foundFirstArrayAccess ) {
										firstAccess = tAccess;
										foundFirstArrayAccess = true;
									}
									lastAccess = tAccess;
								}
							}
							//PrintTools.println("fLoop check point2 ", 0);
							if( (!foundFirstArrayAccess) || (firstAccess == lastAccess) ) {
								continue;
							}
							Traversable t = (Traversable)firstAccess;
							while( !(t instanceof Statement) ) {
								t = t.getParent();
							}
							if( t instanceof Statement ) {
								firstAccessStmt = (Statement)t;
							}
							if( (fLoop == region) && ((firstAccessStmt == fLoop) 
									|| (firstAccessStmt == fLoop.getInitialStatement())) ) {
								continue;
							}
							t = (Traversable)lastAccess;
							while( !(t instanceof Statement) ) {
								t = t.getParent();
							}
							if( t instanceof Statement ) {
								lastAccessStmt = (Statement)t;
							}

							//DEBUG: if below check is for detecting reduction array, it
							//can be removed, since reduction variable has been already excluded.
							//if( firstAccessStmt == lastAccessStmt ) {
							//	continue;
							//}

							if( cudaRegisterSet.contains(aAccessString) ) {
								cachedArrayElmts.add(aAccessString);
							} else if( cudaRegisterSet.contains(aAccessString2) ) {
								cachedArrayElmts.add(aAccessString2);
							}
							// Replace all instances of the shared variable to the local variable
							IRTools.replaceAll((Traversable) fLoop, aAccess, local_var);
							//PrintTools.println("fLoop check point3 ", 0);
							/////////////////////////////////////////////////////////////////////////////////////////
							// If the address of the shared variable is passed as an argument of a function called //
							// in the for-loop, load&store statement should be inserted before&after the function  //
							// call site.                                                                          //
							/////////////////////////////////////////////////////////////////////////////////////////
							List<FunctionCall> funcCalls = IRTools.getFunctionCalls(fLoop); 
							for( FunctionCall calledProc : funcCalls ) {
								List<Expression> argList = (List<Expression>)calledProc.getArguments();
								boolean foundArg = false;
								for( Expression arg : argList ) {
									if(IRTools.containsSymbol(arg, tSym) ) {
										foundArg = true;
										break;
									}    
								}    

								if( !cudaRegisterROSet.contains(aAccessString) && 
										!cudaRegisterROSet.contains(aAccessString2) ) {
									Statement fStmt = calledProc.getStatement();
									if( foundArg ) {
										((CompoundStatement)fStmt.getParent()).addStatementBefore(fStmt,
												(Statement)astmt.clone());
										((CompoundStatement)fStmt.getParent()).addStatementAfter(fStmt,
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
											if( IRTools.containsSymbol(body, tSym) ) {
												((CompoundStatement)fStmt.getParent()).addStatementBefore(fStmt,
														(Statement)astmt.clone());
												((CompoundStatement)fStmt.getParent()).addStatementAfter(fStmt,
														(Statement)estmt.clone());
											}
										}
									}
								}
							}
							//PrintTools.println("fLoop check point4 ", 0);
							if( firstAccessStmt != null ) {
								/*						Traversable p = firstAccessStmt.getParent();
						while( !(p instanceof CompoundStatement) ) {
							p = p.getParent();
						}
						Traversable parent_p = p.getParent();
						if( parent_p instanceof IfStatement ) {
							p = parent_p;
							parent_p = p.getParent();
							((CompoundStatement)parent_p).addStatementBefore(
									(Statement)p, estmt.clone());
						} else {
							((CompoundStatement)p).addStatementBefore(
									firstAccessStmt, estmt.clone());
						}*/
								if( (firstAccessStmt == fLoop) || 
										(firstAccessStmt == fLoop.getInitialStatement()) ) {
									((CompoundStatement)fLoop.getParent()).addStatementBefore(fLoop, estmt.clone());
								} else {
									CompoundStatement fBody = (CompoundStatement)fLoop.getBody();
									Statement fStmt = IRTools.getFirstNonDeclarationStatement(fBody);
									if( fStmt == null ) {
										fBody.addStatement(estmt.clone());
									} else {
										fBody.addStatementBefore(fStmt, estmt.clone());
									}
								}
							} else {
								Tools.exit("[ERROR in arrayCachingOnRegister()] can't find a statement " +
										"containing array access, " + aAccessString + 
										"; remove the array access string from registerRO or registerRW clause.");
							}
							//PrintTools.println("fLoop check point5 ", 0);
							if( !cudaRegisterROSet.contains(aAccessString) && 
									!cudaRegisterROSet.contains(aAccessString2) ) {
								ForLoop commonPLoop = null;
								/*						Traversable parent_t1 = null;
						Traversable parent_t2 = null;
						t = firstAccessStmt.getParent();
						while( !(t instanceof ForLoop) ) {
							t = t.getParent();
						}
						parent_t1 = t;
						t = lastAccessStmt.getParent();
						while( !(t instanceof ForLoop) ) {
							t = t.getParent();
						}
						parent_t2 = t;
						if( !(parent_t1 instanceof ForLoop) || !(parent_t2 instanceof ForLoop) ) {
							commonPLoop = null;
						} else if( parent_t1 == parent_t2 ) {
							// Both statements have the common parent for-loop.
							commonPLoop = (ForLoop)parent_t2;
						} else {
							t = parent_t1;
							while( t != fLoop ) {
								if( t == parent_t2 ) {
									break;
								} else {
									t = t.getParent();
								}
							}
							if( t == parent_t2 ) {
								commonPLoop = (ForLoop)parent_t2;
							} else {
								t = parent_t2;
								while( t != fLoop ) {
									if( t == parent_t1 ) {
										break;
									} else {
										t = t.getParent();
									}
								}
								if( t == parent_t1 ) {
									commonPLoop = (ForLoop)parent_t1;
								} 
							}
						}*/
								commonPLoop = fLoop;
								if( commonPLoop != null ) {
									if( (commonPLoop == fLoop) && ((firstAccessStmt == fLoop) || 
											(firstAccessStmt == fLoop.getInitialStatement())) ) {
										((CompoundStatement)fLoop.getParent()).addStatementAfter(fLoop, astmt.clone());
									} else {
										((CompoundStatement)commonPLoop.getBody()).addStatement(astmt.clone());
									}
								} else {
									PrintTools.println("[Current for-loop] \n" + fLoop + "\n", 0);
									Tools.exit("[ERROR in arrayCachingOnRegister()] can't find a common, enclosing for-loop " +
											"containing all instances of array access, " + aAccessString + 
											"; remove the array access string from registerRO or registerRW clause.");
								}
							}
						}
					}
					if( (!indexLoopExist) && targetLoops.isEmpty() ) {
						//////////////////////////////////////////////////////////////
						// The array access is independent of any enclosing loops.  //
						//////////////////////////////////////////////////////////////
						boolean isCached = true;
						int errorCode = 0;
						Statement defStmt = null;
						Statement parentDefStmt = null;
						boolean sameParentDefStmt = true;
						if( (!constantIndices) && indexSyms.isEmpty() ) {
							//index expressions are not constant, but we could not find symbols for variables used 
							//in the index expression; conservatively skip caching.
							isCached = false;
							errorCode = 1;
						} else {
							if( defStmts.isEmpty() ) {
								//Index expressions are not defined within the target region.
								isCached = true;
							} else {
								//Find a common statement including all statements modifying index expressions.
								for( Statement ttStmt : defStmts ) {
									Traversable pp = ttStmt;
									if( sameParentDefStmt && (parentDefStmt == null) ) {
										parentDefStmt = (Statement)pp.getParent();
									} else if( sameParentDefStmt ) {
										if( ((Statement)pp.getParent()) != parentDefStmt ) {
											sameParentDefStmt = false;
											parentDefStmt = null;
										}
									}
									if( defStmt == null ) {
										if( pp.getParent() == targetStmt ) {
											defStmt = (Statement)pp;
										} else {
											defStmt = (Statement)pp.getParent();
										}
									} else {
										while( (pp != null) && (pp.getParent() != targetStmt) && (pp != defStmt) ) {
											//Find a child statement of targetStmt containing the current DEF statement.
											pp = pp.getParent();
										}
										if( pp == null ) {
											//Could not find child statement of targetStmt including the DEF statement (ttStmt).
											isCached = false;
											errorCode = 2;
											break;
										} else {
											if( defStmt == null ) {
												defStmt = (Statement)pp;
											} else if( pp != defStmt ) {
												Traversable pp2 = defStmt;
												while( (pp2 != null) && (pp2.getParent() != targetStmt) ) {
													pp2 = pp2.getParent();
												}
												if( pp == pp2 ) {
													defStmt = (Statement)pp;
												} else {
													//Multiple child statements of targetStmt that modify index expressions.
													//Conservatively skip caching.
													isCached = false;
													errorCode = 3;
													break;
												}
											}
										}
									}
								}
							}
						}
						if( isCached || sameParentDefStmt ) {
							///////////////////////////////////////////////////
							// Find the first instance of this array access. //
							///////////////////////////////////////////////////
							Expression firstAccess = null;
							Expression lastAccess = null;
							Statement firstAccessStmt = null;
							Statement lastAccessStmt = null;
							boolean foundFirstArrayAccess = false;
							boolean sameParentArrayAccessStmt = true;
							Statement parentArrayAccessStmt = null;
							DepthFirstIterator iter = new DepthFirstIterator(region);
							for (;;)
							{
								Expression tAccess = null;

								try {
									tAccess = (Expression)iter.next(Expression.class);
								} catch (NoSuchElementException e) {
									break;
								}
								if( aAccess.equals(tAccess) ) {
									if( !foundFirstArrayAccess ) {
										firstAccess = tAccess;
										foundFirstArrayAccess = true;
									}
									if( sameParentDefStmt && sameParentArrayAccessStmt ) {
										Traversable t = (Traversable)tAccess;
										while( (t != null) && !(t instanceof Statement) ) {
											t = t.getParent();
										}
										if( t instanceof Statement ) {
											Statement ttStmt = (Statement)t.getParent();
											if( parentArrayAccessStmt == null ) {
												parentArrayAccessStmt = ttStmt;
											} else if ( parentArrayAccessStmt != ttStmt ) {
												parentArrayAccessStmt = null;
												sameParentArrayAccessStmt = false;
											}
										} else {
											parentArrayAccessStmt = null;
											sameParentArrayAccessStmt = false;
										}
									}
									lastAccess = tAccess;
								}
							}
							if( !isCached && (!sameParentArrayAccessStmt || (parentArrayAccessStmt != parentDefStmt)) ) {
								if( !checkedArrayAccessSet.contains(aAccessString) ) {
									errorCode = 4;
									PrintTools.println("[INFO in arrayCachingOnRegister()] " +
											"index expression of the array access, " + aAccess + 
											", is independent of any enclosing loops, but not constant. " +
											"Current implementation fails to detect the possible dependency problem of this " +
											"index expression, and thus it will not be cached conservatively, " +
											"even though it may have locality. (Internal code = " + errorCode + ")" +
											AnalysisTools.getEnclosingContext(region), 0);
									checkedArrayAccessSet.add(aAccessString);
								}
								continue;
							}
							//PrintTools.println("fLoop check point2 ", 0);
							if( (!foundFirstArrayAccess) || (firstAccess == lastAccess) ) {
								continue;
							}
							Traversable t = (Traversable)firstAccess;
							while( !(t instanceof Statement) ) {
								t = t.getParent();
							}
							if( t instanceof Statement ) {
								firstAccessStmt = (Statement)t;
							}
							if( (region instanceof ForLoop) && ((firstAccessStmt == region) 
									|| (firstAccessStmt == ((ForLoop)region).getInitialStatement())) ) {
								continue;
							}
							t = (Traversable)lastAccess;
							while( !(t instanceof Statement) ) {
								t = t.getParent();
							}
							if( t instanceof Statement ) {
								lastAccessStmt = (Statement)t;
							}
							
							if( firstAccessStmt == null ) {
								Tools.exit("[ERROR in arrayCachingOnRegister()] can't find a statement " +
										"containing array access, " + aAccessString + 
								"; remove the array access string from registerRO or registerRW clause.");
							}
							Statement lastDefStmt = null;
							if( !isCached ) {
								//Check whether all index-def-statements are before the array-access-statements.
								//If not, skip caching conservatively.
								if( parentDefStmt instanceof CompoundStatement ) {
									CompoundStatement cStmt = (CompoundStatement)parentDefStmt;
									int lastDefIndex = 0;
									int tmp = 0;
									for( Statement dStmt : defStmts ) {
										tmp = cStmt.getChildren().indexOf(dStmt);
										if( tmp > lastDefIndex ) {
											lastDefIndex = tmp;
											lastDefStmt = dStmt;
										}
									}
									tmp = cStmt.getChildren().indexOf(firstAccessStmt);
									if( (tmp == -1) || (tmp < lastDefIndex) ) {
										if( !checkedArrayAccessSet.contains(aAccessString) ) {
											errorCode = 5;
											PrintTools.println("[INFO in arrayCachingOnRegister()] " +
													"index expression of the array access, " + aAccess + 
													", is independent of any enclosing loops, but not constant. " +
													"Current implementation fails to detect the possible dependency problem of this " +
													"index expression, and thus it will not be cached conservatively, " +
													"even though it may have locality. (Internal code = " + errorCode + ")" +
													AnalysisTools.getEnclosingContext(region), 0);
											checkedArrayAccessSet.add(aAccessString);
										}
										continue;
									}
								} else {
									continue;
								}
							} else {
								Traversable pp = lastAccessStmt;
								while( (pp != null) && (pp.getParent() != defStmt) && (pp.getParent() != targetStmt) ) {
									pp = pp.getParent();
								}
								if( (pp != null) && (defStmt != null) && (pp.getParent() == defStmt) ) {
									//lastAccessStmt is in the same block where index expressions are modified, which
									//implies that caching should be done only in that enclosing block; for now, conservatively
									//skip the caching.
									continue;
								}
							}
							
							if( cudaRegisterSet.contains(aAccessString) ) {
								cachedArrayElmts.add(aAccessString);
							} else if( cudaRegisterSet.contains(aAccessString2) ) {
								cachedArrayElmts.add(aAccessString2);
							}
							
							boolean foundDEFStmt = false;
							if( isCached && (defStmt == null) ) {
								foundDEFStmt = true;
							}
							List<Statement> stmtToInsertLSStmts = new LinkedList<Statement>();
							CompoundStatement cTargetStmt = null;
							if( isCached ) {
								cTargetStmt = targetStmt;
							} else {
								cTargetStmt = (CompoundStatement)parentDefStmt;
							}
							for( Traversable tChildStmt : cTargetStmt.getChildren() ) {
								if( !foundDEFStmt ) {
									if( isCached ) {
										if( tChildStmt == defStmt ) {
											foundDEFStmt = true;
										}
										continue;
									} else {
										if( defStmts.contains(tChildStmt) ) {
											continue;
										}
									}
								} 
								if( !isCached || foundDEFStmt ) {
									// Replace all instances of the shared variable to the local variable
									IRTools.replaceAll(tChildStmt, aAccess, local_var);
									//PrintTools.println("fLoop check point3 ", 0);
									/////////////////////////////////////////////////////////////////////////////////////////
									// If the address of the shared variable is passed as an argument of a function called //
									// in the target region, load&store statement should be inserted before&after the      //
									// function call site.                                                                 //
									// CF: Actual insertion will be done later not to corrupt targetStmt.getChildren()     //
									/////////////////////////////////////////////////////////////////////////////////////////
									List<FunctionCall> funcCalls = IRTools.getFunctionCalls(tChildStmt); 
									for( FunctionCall calledProc : funcCalls ) {
										List<Expression> argList = (List<Expression>)calledProc.getArguments();
										boolean foundArg = false;
										for( Expression arg : argList ) {
											if(IRTools.containsSymbol(arg, tSym) ) {
												foundArg = true;
												break;
											}    
										}    

										if( !cudaRegisterROSet.contains(aAccessString) && 
												!cudaRegisterROSet.contains(aAccessString2) ) {
											Statement fStmt = calledProc.getStatement();
											if( foundArg ) {
												stmtToInsertLSStmts.add((Statement)tChildStmt);
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
													if( IRTools.containsSymbol(body, tSym) ) {
														stmtToInsertLSStmts.add((Statement)tChildStmt);
													}
												}
											}
										}
									}
								}
							}
							for( Statement fStmt : stmtToInsertLSStmts ) {
								//Insert load&store statements after this function call statement.
								((CompoundStatement)fStmt.getParent()).addStatementBefore(fStmt,
										(Statement)astmt.clone());
								((CompoundStatement)fStmt.getParent()).addStatementAfter(fStmt,
										(Statement)estmt.clone());
							}
							
							
							//PrintTools.println("fLoop check point4 ", 0);
							if( isCached && (defStmt != null) ) {
								//Add load statement after the index expressions are modified.
								((CompoundStatement)defStmt.getParent()).addStatementAfter(defStmt, estmt.clone());
							} else if( !isCached && (lastDefStmt != null) ) {
								//Add load statement after the index expressions are modified.
								((CompoundStatement)lastDefStmt.getParent()).addStatementAfter(lastDefStmt, estmt.clone());
							} else {
								//The index expressions remains unchanged within the target region.
								//Add load statement before the first non-declaration statement.
								Statement fStmt = IRTools.getFirstNonDeclarationStatement(targetStmt);
								if( fStmt == null ) {
									targetStmt.addStatement(estmt.clone());
								} else {
									targetStmt.addStatementBefore(fStmt, estmt.clone());
								}
							}
							
							if( !cudaRegisterROSet.contains(aAccessString) && 
									!cudaRegisterROSet.contains(aAccessString2) ) {
								//Add store statement at the end of the target region
								targetStmt.addStatement(astmt.clone());
							}
						} else {
							if( !checkedArrayAccessSet.contains(aAccessString) ) {
								PrintTools.println("[INFO in arrayCachingOnRegister()] " +
										"index expression of the array access, " + aAccess + 
										", is independent of any enclosing loops, but not constant. " +
										"Current implementation fails to detect the possible dependency problem of this " +
										"index expression, and thus it will not be cached conservatively, " +
										"even though it may have locality. (Internal code = " + errorCode + ")" +
										AnalysisTools.getEnclosingContext(region), 0);
								checkedArrayAccessSet.add(aAccessString);
							}
						}
					}
				}
			}
		}
		//PrintTools.println("arrayCachingOnRegister() end", 0);
		return arrayElmtCacheSymbols;
	}
	
	/**
	 * Handle privatization and reduction transformation for a seq kernels/parallel loop.
	 * [FIXME] this should be updated to handle kernel verificaiton and other special modes.
	 * 
	 * @param cProc
	 * @param region
	 * @param cRegionKind
	 * @param ifCond
	 * @param asyncID
	 * @param confRefStmt
	 * @param prefixStmts
	 * @param postscriptStmts
	 * @param call_to_new_proc
	 * @param new_proc
	 * @param main_TrUnt
	 * @param OpenACCHeaderEndMap
	 * @param IRSymbolOnly
	 * @param opt_addSafetyCheckingCode
	 * @param opt_UnrollOnReduction
	 * @param maxBlockSize
	 */
	protected static void seqKernelLoopTransformation(Procedure cProc, ForLoop seqLoop, 
			String cRegionKind, Expression ifCond, Expression asyncID, Statement confRefStmt,
			List<Statement> preList, List<Statement> postList,
			CompoundStatement prefixStmts, CompoundStatement postscriptStmts,
			FunctionCall call_to_new_proc, Procedure new_proc, TranslationUnit main_TrUnt, 
			Map<TranslationUnit, Declaration> OpenACCHeaderEndMap, boolean IRSymbolOnly,
			boolean opt_addSafetyCheckingCode, int targetModel, boolean assumeNoAliasing ) {
		CompoundStatement scope = null;
		CompoundStatement regionParent = (CompoundStatement)seqLoop.getParent();

		SymbolTable global_table = (SymbolTable) cProc.getParent();
		PrintTools.println("[seqKernelLoopTransformation() begins] current procedure: " + cProc.getSymbolName() +
				"\ncompute region type: " + cRegionKind + "\n", 2);


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
		ExpressionStatement gpuBytes_stmt = null;
		ExpressionStatement orgGpuBytes_stmt = null;

		long num_workers = 1;

		if( (seqLoop.containsAnnotation(ACCAnnotation.class, "private") 
				|| seqLoop.containsAnnotation(ACCAnnotation.class, "firstprivate"))
				|| seqLoop.containsAnnotation(ACCAnnotation.class, "reduction")) {
			Map<Symbol, SubArray> sharedCachingMap = new HashMap<Symbol, SubArray>();
			Map<Symbol, SubArray> regROCachingMap = new HashMap<Symbol, SubArray>();
			Map<Symbol, SubArray> globalMap = new HashMap<Symbol, SubArray>();
			Set<String> searchKeys = new HashSet<String>();
			searchKeys.add("sharedRO");
			searchKeys.add("sharedRW");
			for( String key : searchKeys ) {
				ARCAnnotation ttAnt = seqLoop.getAnnotation(ARCAnnotation.class, key);
				if( ttAnt != null ) {
					Set<SubArray> DataSet = (Set<SubArray>)ttAnt.get(key);
					for( SubArray sAr : DataSet ) {
						Symbol tSym = AnalysisTools.subarrayToSymbol(sAr, IRSymbolOnly);
						sharedCachingMap.put(tSym, sAr);
					}
				}
			}
			ARCAnnotation ttAnt = seqLoop.getAnnotation(ARCAnnotation.class, "registerRO");
			if( ttAnt != null ) {
				Set<SubArray> DataSet = (Set<SubArray>)ttAnt.get("registerRO");
				for( SubArray sAr : DataSet ) {
					Symbol tSym = AnalysisTools.subarrayToSymbol(sAr, IRSymbolOnly);
					regROCachingMap.put(tSym, sAr);
				}
			}
			ttAnt = seqLoop.getAnnotation(ARCAnnotation.class, "global");
			if( ttAnt != null ) {
				Set<SubArray> DataSet = (Set<SubArray>)ttAnt.get("global");
				for( SubArray sAr : DataSet ) {
					Symbol tSym = AnalysisTools.subarrayToSymbol(sAr, IRSymbolOnly);
					globalMap.put(tSym, sAr);
				}
			}

			scope = (CompoundStatement)seqLoop.getBody();

			HashSet<SubArray> PrivSet = null; 
			HashSet<SubArray> FirstPrivSet = null; 
			HashMap<Symbol, SubArray> PrivSymMap = new HashMap<Symbol, SubArray>();
			HashSet<Symbol> FirstPrivSymSet = new HashSet<Symbol>();
			ACCAnnotation pannot = seqLoop.getAnnotation(ACCAnnotation.class, "private");
			if( pannot != null ) {
				PrivSet = (HashSet<SubArray>) pannot.get("private");
				if( PrivSet != null ) {
					for( SubArray sAr : PrivSet ) {
						Symbol tSym = AnalysisTools.subarrayToSymbol(sAr, IRSymbolOnly);
						PrivSymMap.put(tSym, sAr);
					}
				}
			}
			pannot = seqLoop.getAnnotation(ACCAnnotation.class, "firstprivate");
			if( pannot != null ) {
				FirstPrivSet = (HashSet<SubArray>) pannot.get("firstprivate");
				if( FirstPrivSet != null ) {
					for( SubArray sAr : FirstPrivSet ) {
						Symbol tSym = AnalysisTools.subarrayToSymbol(sAr, IRSymbolOnly);
						PrivSymMap.put(tSym, sAr);
						FirstPrivSymSet.add(tSym);
					}
				}
			}

			//Host reduction symbol to reduction operator mapping
			HashMap<Symbol, ReductionOperator> redOpMap = new HashMap<Symbol, ReductionOperator>();
			//Set of allocated gang-reduction symbols
			pannot = seqLoop.getAnnotation(ACCAnnotation.class, "reduction");
			if( pannot != null ) {
				Map<ReductionOperator, Set<SubArray>> redMap = pannot.get("reduction");
				for(ReductionOperator op : redMap.keySet() ) {
					Set<SubArray> redSet = redMap.get(op);
					for( SubArray sArray : redSet ) {
						Symbol rSym = AnalysisTools.subarrayToSymbol(sArray, IRSymbolOnly); 
						PrivSymMap.put(rSym, sArray);
						redOpMap.put(rSym, op);
					}
				}
			}
			
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
				Traversable tKernel = seqLoop;
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
					ACCAnnotation cAnnot = seqLoop.getAnnotation(ACCAnnotation.class, cRegionKind);
					Tools.exit("[ERROR in CUDATranslationTools.reductionTransformation()] the final " +
							"reduction codes for the following asynchronous " +
							"kernel should be inserted after the matching synchronization statement, " +
							"but the compiler can not find the statement; exit.\n" +
							"Current implementation can not handle asynchronous reduction if acc_async_test() " +
							"or acc_async_test_all() function is used for synchronization; please change these " +
							"to acc_wait() or acc_waitall() function.\n" +
							"OpenACC Annotation: " + cAnnot + "\nEnclosing Procedure: " + 
							cProc.getSymbolName() + "\n" );
				} else {
					if( confRefStmt == seqLoop ) {
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
			
			//List to keep post-reduction statements if asyncID != null.
			List<Statement> gPostRedStmts = new LinkedList<Statement>();

			Collection<Symbol> sortedSet = AnalysisTools.getSortedCollection(PrivSymMap.keySet());
			for( Symbol privSym : sortedSet ) {
				List<Statement> postRedStmts = new LinkedList<Statement>();
				
				SubArray sArray = PrivSymMap.get(privSym);
				ReductionOperator redOp = null;
				if( redOpMap.containsKey(privSym) ) {
					redOp = redOpMap.get(privSym);
				}
				Boolean isArray = SymbolTools.isArray(privSym);
				Boolean isPointer = SymbolTools.isPointer(privSym);
				if( privSym instanceof NestedDeclarator ) {
					isPointer = true;
				}
				//////////////////////////////////////////////////////////////////////////////////
				//FIXME: if privSym is a parameter of a function called in the parallel region, //
				//below checking may be incorrect.                                              //
				//////////////////////////////////////////////////////////////////////////////////
				//DEBUG: extractComputeRegion() in ACC2CUDATranslator/ACC2OPENCLTranslator may promote 
				//privatization-related statements above the scope where the private variable is declared.
				//In this case, the below targetSymbolTable will not work.
				//To handle this promotion, we simply used the enclosing function body as targetSymbolTable.
/*				SymbolTable targetSymbolTable = AnalysisTools.getIRSymbolScope(privSym, seqLoop);
				if( targetSymbolTable instanceof Procedure ) {
					targetSymbolTable = ((Procedure)targetSymbolTable).getBody();
				}
				if( targetSymbolTable == null ) {
					targetSymbolTable = (SymbolTable) cProc.getBody();
				}*/
				SymbolTable targetSymbolTable = cProc.getBody();

				List<Expression> startList = new LinkedList<Expression>();
				List<Expression> lengthList = new LinkedList<Expression>();
				boolean foundDimensions = AnalysisTools.extractDimensionInfo(sArray, startList, lengthList, IRSymbolOnly, seqLoop);
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
				Identifier gwpriv_var = null;
				Identifier lpriv_var = null;
				Identifier lred_var = null;
				String symNameBase = null;
				if( privSym instanceof AccessSymbol) {
					symNameBase = TransformTools.buildAccessSymbolName((AccessSymbol)privSym);
				} else {
					symNameBase = privSym.getSymbolName();
				}
				String gpuWPSymName = "gwpriv__" + symNameBase;
				String localWPSymName = "lwpriv__" + symNameBase;
				String localRedSymName = "lred__" + symNameBase;

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
				boolean workerPrivCachingOnShared = false;
				boolean workerPrivOnGlobal = false;
				List<Specifier> addSpecs = null;
				if( sharedCachingMap.keySet().contains(privSym) ) {
					workerPrivCachingOnShared = true;
				} 
				if( globalMap.keySet().contains(privSym) ) {
					if( ( isArray || isPointer ) ) {
						workerPrivOnGlobal = true;
					}
				}


				if( workerPrivOnGlobal || (redOp != null) ) {
					//Option to allocate worker-private variable on global memory is checked first, since it may be
					//mandatory due to too large private array size.
					//////////////////////////////////////////////////////
					//Create a worker-private variable on global memory //
					//////////////////////////////////////////////////////
					// float lprev__x[SIZE];                            //
					//////////////////////////////////////////////////////
					/////////////////////////////////////////////////////////////
					// Create a GPU device variable corresponding to privSym   //
					// Ex: float * gwpriv__x; //GPU variable for worker-private//
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


					// Insert "gpuBytes = (dimension1 * dimension2 * ..) * sizeof(varType);" statement
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
					gpuBytes_stmt = new ExpressionStatement(assignex);

					////////////////////////////////////////////////////////////
					// Create a parameter Declaration for the kernel function //
					////////////////////////////////////////////////////////////
					// Ex1: "float lwpriv__x[SIZE1]"
					// Ex2: "float lwpriv__x[SIZE1][SIZE2]"
					// Ex2: "float lred__x[SIZE1]"
					if( dimsize == 0 ) {
						// Create a parameter Declaration for the kernel function
						// Change the scalar variable to a pointer type 
						if( (redOp == null) || ((redOp != null) && workerPrivOnGlobal) ) {
							VariableDeclarator kParam_declarator = new VariableDeclarator(PointerSpecifier.UNQUALIFIED, 
									new NameID(localWPSymName));
							VariableDeclaration kParam_decl = new VariableDeclaration(typeSpecs,
									kParam_declarator);
							lpriv_var = new Identifier(kParam_declarator);
							new_proc.addDeclaration(kParam_decl);
						} else {
							VariableDeclarator kParam_declarator = new VariableDeclarator(PointerSpecifier.UNQUALIFIED, 
									new NameID(localRedSymName));
							VariableDeclaration kParam_decl = new VariableDeclaration(typeSpecs,
									kParam_declarator);
							lred_var = new Identifier(kParam_declarator);
							new_proc.addDeclaration(kParam_decl);
						}

						// Insert argument to the kernel function call
						call_to_new_proc.addArgument(gwpriv_var.clone());

					} else {
						if( (redOp == null) || ((redOp != null) && workerPrivOnGlobal) ) {
							lpriv_var = TransformTools.declareClonedVariable(new_proc, privSym, localWPSymName, removeSpecs, null, true, assumeNoAliasing);
						} else {
							lred_var = TransformTools.declareClonedVariable(new_proc, privSym, localRedSymName, removeSpecs, null, true, assumeNoAliasing);
						}

						// Insert argument to the kernel function call
						if( dimsize == 1 ) {
							call_to_new_proc.addArgument(gwpriv_var.clone());
						} else {
							//Cast the gpu variable to pointer-to-array type 
							// Ex: (float (*)[SIZE2]) gpu__x
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
							call_to_new_proc.addArgument(new Typecast(castspecs, (Identifier)gwpriv_var.clone()));
						}
					}

					// Add gpuBytes argument to cudaMalloc() call
					arg_list.add((Identifier)cloned_bytes.clone());
                    arg_list.add(new NameID("acc_device_current"));
					malloc_call.setArguments(arg_list);
					ExpressionStatement malloc_stmt = new ExpressionStatement(malloc_call);
					// Insert malloc statement.
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
					if( redOp == null ) { //If this is for reduction, it should be inserted at the end.
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
				} 
				if( !workerPrivOnGlobal ) {
					if( workerPrivCachingOnShared ) {
						//////////////////////////////////////////////////////
						//Create a worker-private variable on shared memory //
						//using array expansion.                            //
						//////////////////////////////////////////////////////
						addSpecs = new ArrayList<Specifier>(1);
						if( targetModel == 0 ) { //for CUDA
							addSpecs.add(CUDASpecifier.CUDA_SHARED);
						} else { //for OpenCL
							addSpecs.add(OpenCLSpecifier.OPENCL_LOCAL);
						}
						////////////////////////////////////////////////
						//Create a private variable on shared memory. //
						//(No array-expansion is needed.)             //
						////////////////////////////////////////////////
						//     __shared__ float lwprev__x;            //
						//     __shared__ float lwprev__x[SIZE];      //
						////////////////////////////////////////////////
						if( privSym instanceof AccessSymbol ) {
							Symbol tSym = privSym;
							while( tSym instanceof AccessSymbol ) {
								tSym = ((AccessSymbol)tSym).getMemberSymbol();
							}
							lpriv_var = TransformTools.declareClonedArrayVariable(scope, sArray, localWPSymName, 
									removeSpecs, addSpecs);
						} else {
							lpriv_var = TransformTools.declareClonedArrayVariable(scope, sArray, localWPSymName, 
									removeSpecs, addSpecs);
						}
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
							lpriv_var = TransformTools.declareClonedArrayVariable(scope, sArray, localWPSymName, 
									removeSpecs, addSpecs);
						} else {
							lpriv_var = TransformTools.declareClonedArrayVariable(scope, sArray, localWPSymName, 
									removeSpecs, addSpecs);
						}
					}
				}
				////////////////////////////////////////////////////////////////////////
				// Replace the private variable with this new local private variable. //
				////////////////////////////////////////////////////////////////////////
				if( privSym instanceof AccessSymbol ) {
					TransformTools.replaceAccessExpressions(seqLoop, (AccessSymbol)privSym, lpriv_var);
				} else {
					TransformTools.replaceAll(seqLoop, new Identifier(privSym), lpriv_var);
				}
				//Reset below to be checked later.
				//[DEBUG] temporarily disabled.
				//gpuBytes_stmt = null;
				//orgGpuBytes_stmt = null;
				
				if( redOp != null ) {
					////////////////////////////////////////////////
					// Insert reduction-initialization statement. //
					////////////////////////////////////////////////
					Expression LHS = null;
					Expression RHS = TransformTools.getRInitValue(redOp, typeSpecs);
					Statement estmt = null;
					if( !isArray && !isPointer ) { //scalar variable
						LHS = lpriv_var.clone();
						estmt = new ExpressionStatement(
								new AssignmentExpression((Expression)LHS.clone(), 
										AssignmentOperator.NORMAL, RHS));
					} else { //non-scalar variable
						///////////////////////////////////////////////////////
						// Ex1: worker-reduction allocated on shared memory  //
						//      for(i=0; i<SIZE1; i++) {                     //
						//         for(k=0; k<SIZE2; k++) {                  //
						//             lwpriv__x[i][k] = initValue;          //
						//         }                                         //
						//      }                                            //
						///////////////////////////////////////////////////////
						// Ex2: worker-reduction allocated on global memory  //
						//      for(i=0; i<SIZE1; i++) {                     //
						//         for(k=0; k<SIZE2; k++) {                  //
						//             lwpriv__x[i][k] = initValue;          //
						//         }                                         //
						//      }                                            //
						///////////////////////////////////////////////////////
						//////////////////////////////////////// //////
						// Create or find temporary index variables. // 
						//////////////////////////////////////// //////
						List<Identifier> index_vars = new LinkedList<Identifier>();
						for( int i=0; i<=dimsize; i++ ) {
							index_vars.add(TransformTools.getTempIndex(scope, tempIndexBase+i));
						}
						List<Expression> indices = new LinkedList<Expression>();
						for( int k=0; k<dimsize; k++ ) {
							indices.add((Expression)index_vars.get(k).clone());
						}
						LHS = new ArrayAccess(lpriv_var.clone(), indices);
						estmt = TransformTools.genArrayCopyLoop(index_vars, lengthList, LHS, RHS);
					}
					preList.add(estmt);
				}

				///////////////////////////////////////////////////////////////////
				// Load the value of host variable to the firstprivate variable. //
				///////////////////////////////////////////////////////////////////
				if( (FirstPrivSymSet != null) && FirstPrivSymSet.contains(privSym) ) {
					VariableDeclaration gfpriv_decl = null;
					Identifier gfpriv_var = null;
					Identifier lfpriv_var = null;
					String gpuFPSymName = "gfpriv__" + symNameBase;
					String localFPSymName = "lfpriv__" + symNameBase;
					if( workerPrivOnGlobal ) {
						gpuFPSymName = "gwpriv__" + symNameBase;
						localFPSymName = "lwpriv__" + symNameBase;
					}
					//////////////////////////////////////////////////////////////////////////////
					// If firstprivate variable is scalar, the corresponding shared variable is //
					// passed as a kernel parameter instead of using GPU global memory, which   //
					// has the effect of caching it on the GPU Shared Memory.                   //
					//////////////////////////////////////////////////////////////////////////////
					if( !isArray && !isPointer ) { //scalar variable
						// Create a GPU kernel parameter corresponding to privSym
						// ex: flaot lfpriv__x;
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
						//No array expansion is used.
						//ex: lgpriv__x = lfpriv__x;
						estmt = new ExpressionStatement(new AssignmentExpression(lpriv_var.clone(), 
								AssignmentOperator.NORMAL, lfpriv_var.clone()));
						preList.add(estmt);
					} else { //non-scalar variable
						if( !workerPrivOnGlobal ) {
							////////////////////////////////////////////////////////////////////////////////////////
							// Create a GPU device variable to carry initial values of the firstprivate variable. //
							// Ex: float * gfpriv__x; //GPU variable for gang-firstprivate                        //
							////////////////////////////////////////////////////////////////////////////////////////
							// Give a new name for the device variable 
							gfpriv_decl =  TransformTools.getGPUVariable(gpuFPSymName, targetSymbolTable, 
									typeSpecs, main_TrUnt, OpenACCHeaderEndMap, null);
							gfpriv_var = new Identifier((VariableDeclarator)gfpriv_decl.getDeclarator(0));


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
							Expression biexp = lengthList.get(0).clone();
							for( int i=1; i<dimsize; i++ )
							{
								biexp = new BinaryExpression(biexp, BinaryOperator.MULTIPLY, lengthList.get(i).clone());
							}
							BinaryExpression biexp2 = new BinaryExpression(biexp, BinaryOperator.MULTIPLY, sizeof_expr);
							AssignmentExpression assignex = new AssignmentExpression((Expression)cloned_bytes.clone(),
									AssignmentOperator.NORMAL, biexp2);
							orgGpuBytes_stmt = new ExpressionStatement(assignex);
							// Add gpuBytes argument to cudaMalloc() call
							arg_list.add((Identifier)cloned_bytes.clone());
                            arg_list.add(new NameID("acc_device_current"));
							malloc_call.setArguments(arg_list);
							ExpressionStatement malloc_stmt = new ExpressionStatement(malloc_call);

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

							/////////////////////////////////////////////////////////////
							// Create a parameter Declaration for the kernel function. //
							// Keep the original array type, but change name           //
							// Ex: "float lfpriv_b[(2048+2)][(2048+2)]"                //
							/////////////////////////////////////////////////////////////
							List edimensions = new LinkedList();
							for( int i=0; i<dimsize; i++ )
							{
								edimensions.add(lengthList.get(i).clone());
							}
							ArraySpecifier easpec = new ArraySpecifier(edimensions);
							VariableDeclarator lfpriv_declarator = new VariableDeclarator(new NameID(localFPSymName), easpec);
							VariableDeclaration lfpriv_decl = 
									new VariableDeclaration(typeSpecs, lfpriv_declarator); 
							Identifier array_var = new Identifier(lfpriv_declarator);
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
							///////////////////////////////////////////////////////////////////////////////
							///////////////////////////////////////////////////////////////////////////////
							//      for(i=0; i<SIZE1; i++) {                                             //
							//         for(k=0; k<SIZE2; k++) {                                          //
							//             lpriv_x[i][k] = lfpriv_x[i][k];                               //
							//         }                                                                 //
							//      }                                                                    //
							///////////////////////////////////////////////////////////////////////////////
							//////////////////////////////////////// //////
							// Create or find temporary index variables. // 
							//////////////////////////////////////// //////
							List<Identifier> index_vars = new LinkedList<Identifier>();
							for( int i=0; i<dimsize; i++ ) {
								index_vars.add(TransformTools.getTempIndex(scope, tempIndexBase+i));
							}
							List<Expression> indices1 = new LinkedList<Expression>();
							List<Expression> indices2 = new LinkedList<Expression>();
							for( int k=0; k<dimsize; k++ ) {
								indices1.add((Expression)index_vars.get(k).clone());
								indices2.add((Expression)index_vars.get(k).clone());
							}
							Expression LHS = new ArrayAccess(lpriv_var.clone(), indices1);
							Expression RHS = new ArrayAccess(lfpriv_var.clone(), indices2);
							Statement estmt = TransformTools.genArrayCopyLoop(index_vars, lengthList, LHS, RHS);
							preList.add(estmt);

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
						}

						/* Insert memory copy function from CPU to GPU */
						// Ex: cudaMemcpy(gfpriv__x, x, gpuBytes, cudaMemcpyHostToDevice); 
						CompoundStatement ifBody = new CompoundStatement();
						IfStatement ifStmt = null;
						if( ifCond != null ) {
							ifStmt = new IfStatement(ifCond.clone(), ifBody);
						}
						if( confRefStmt != seqLoop ) {
							if( ifCond == null ) {
								((CompoundStatement)seqLoop.getParent()).addStatementBefore(seqLoop, 
										orgGpuBytes_stmt.clone());
							} else {
								ifBody.addStatement(orgGpuBytes_stmt.clone());
							}
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
						arg_list2.add(new NameID("cudaMemcpyHostToDevice"));
						arg_list2.add(new IntegerLiteral(0));
						if( asyncID != null ) {
							arg_list2.add(asyncID.clone());
						}
						copyinCall.setArguments(arg_list2);
						Statement copyin_stmt = new ExpressionStatement(copyinCall);
						if( ifCond != null ) {
							ifBody.addStatement(copyin_stmt);
							((CompoundStatement)seqLoop.getParent()).addStatementBefore(seqLoop, ifStmt);
						} else {
							((CompoundStatement)seqLoop.getParent()).addStatementBefore(seqLoop, copyin_stmt);
						}
					}
				} //end of firstprivate translation loop.

				//////////////////////////////////////////////////
				// Copy back the reduction results back to CPU. //
				//////////////////////////////////////////////////
				if( redOp != null ) {
					if( !workerPrivOnGlobal ) {
						if( !isArray && !isPointer ) { //scalar variable
							//////////////////////////////////////////////////
							// Store reduction result back to GPU variable. //
							//////////////////////////////////////////////////
							Statement estmt = null;
							//ex: lred__x = lpriv__x;
							estmt = new ExpressionStatement(new AssignmentExpression(new UnaryExpression(UnaryOperator.DEREFERENCE, lred_var.clone()), 
									AssignmentOperator.NORMAL, lpriv_var.clone()));
							postList.add(estmt);
						} else { //non-scalar variable
							//////////////////////////////////////////////////
							// Store reduction result back to GPU variable. //
							//////////////////////////////////////////////////
							//////////////////////////////////////////////////
							//      for(i=0; i<SIZE1; i++) {                //
							//         for(k=0; k<SIZE2; k++) {             //
							//             lred_x[i][k] = lpriv_x[i][k];    //
							//         }                                    //
							//      }                                       //
							//////////////////////////////////////////////////
							//////////////////////////////////////// //////
							// Create or find temporary index variables. // 
							//////////////////////////////////////// //////
							List<Identifier> index_vars = new LinkedList<Identifier>();
							for( int i=0; i<dimsize; i++ ) {
								index_vars.add(TransformTools.getTempIndex(scope, tempIndexBase+i));
							}
							List<Expression> indices1 = new LinkedList<Expression>();
							List<Expression> indices2 = new LinkedList<Expression>();
							for( int k=0; k<dimsize; k++ ) {
								indices1.add((Expression)index_vars.get(k).clone());
								indices2.add((Expression)index_vars.get(k).clone());
							}
							Expression LHS = new ArrayAccess(lred_var.clone(), indices1);
							Expression RHS = new ArrayAccess(lpriv_var.clone(), indices2);
							Statement estmt = TransformTools.genArrayCopyLoop(index_vars, lengthList, LHS, RHS);
							postList.add(estmt);
						}
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
						ArrayList<Specifier> specs = new ArrayList<Specifier>(4);
                        specs.add(Specifier.VOID);
                        specs.add(PointerSpecifier.UNQUALIFIED);
                        specs.add(PointerSpecifier.UNQUALIFIED);
                        cudaFree_call.addArgument(new Typecast(specs, new UnaryExpression(UnaryOperator.ADDRESS_OF,
(Identifier)gwpriv_var.clone())));
                    cudaFree_call.addArgument(new NameID("acc_device_current"));
					ExpressionStatement cudaFree_stmt = new ExpressionStatement(cudaFree_call);
/*					//mallocScope.addStatementAfter(confRefStmt, cudaFree_stmt);
					//mallocScope.addStatementAfter(confRefStmt, gpuBytes_stmt.clone());
					postscriptStmts.addStatement(gpuBytes_stmt.clone());
					postscriptStmts.addStatement(cudaFree_stmt);
					if( opt_addSafetyCheckingCode  ) {
						postscriptStmts.addStatement(gMemSub_stmt.clone());
					}*/
					if( asyncID == null ) {
						if( confRefStmt != seqLoop ) {
							postscriptStmts.addStatement(cudaFree_stmt);
							if( opt_addSafetyCheckingCode  ) {
								postscriptStmts.addStatement(gMemSub_stmt.clone());
							}
						} else {
							if( opt_addSafetyCheckingCode  ) {
								regionParent.addStatementAfter(seqLoop, gMemSub_stmt.clone());
							}
							regionParent.addStatementAfter(seqLoop, cudaFree_stmt);
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

					/* Insert memory copy function from GPU to CPU */
					CompoundStatement ifBody = new CompoundStatement();
					IfStatement ifStmt = null;
					if( ifCond != null ) {
						ifStmt = new IfStatement(ifCond.clone(), ifBody);
					}
					/////////////////////////////////////////////////////////////////////////////////////////
					// HI_memcpy(hostPtr, gpuPtr, gpuBytes, HI_MemcpyDeviceToHost, 0);               //
					// HI_memcpy_async(hostPtr, gpuPtr, gpuBytes, HI_MemcpyDeviceToHost, 0, asyncID);//
					/////////////////////////////////////////////////////////////////////////////////////////
					FunctionCall copyinCall = null;
					if( asyncID == null ) {
						copyinCall = new FunctionCall(new NameID("HI_memcpy"));
					} else {
						copyinCall = new FunctionCall(new NameID("HI_memcpy_async"));
					}
					List<Expression> arg_list2 = new ArrayList<Expression>();
					if( privSym instanceof AccessSymbol ) {
						AccessExpression accExp = AnalysisTools.accessSymbolToExpression((AccessSymbol)privSym,null);
						if( !isArray && !isPointer ) { //scalar variable
							arg_list2.add(new UnaryExpression(UnaryOperator.ADDRESS_OF, accExp));
						} else {
							arg_list2.add(accExp);
						}
					} else {
						if( !isArray && !isPointer ) { //scalar variable
							arg_list2.add(new UnaryExpression(UnaryOperator.ADDRESS_OF, new Identifier(privSym)));
						} else {
							arg_list2.add(new Identifier(privSym));
						}
					}
					arg_list2.add((Identifier)gwpriv_var.clone());
					arg_list2.add((Identifier)cloned_bytes.clone());
					arg_list2.add(new NameID("HI_MemcpyDeviceToHost"));
					arg_list2.add(new IntegerLiteral(0));
					if( asyncID != null ) {
						arg_list2.add(asyncID.clone());
					}
					copyinCall.setArguments(arg_list2);
					Statement copyin_stmt = new ExpressionStatement(copyinCall);
/*					if( ifCond != null ) {
						if( confRefStmt != seqLoop ) {
							ifBody.addStatement(orgGpuBytes_stmt.clone());
						}
						ifBody.addStatement(copyin_stmt);
						((CompoundStatement)seqLoop.getParent()).addStatementAfter(seqLoop, ifStmt);
					} else {
						((CompoundStatement)seqLoop.getParent()).addStatementAfter(seqLoop, copyin_stmt);
						if( confRefStmt != seqLoop ) {
							((CompoundStatement)seqLoop.getParent()).addStatementAfter(seqLoop, 
									orgGpuBytes_stmt.clone());
						}
					}*/
					if( asyncID == null ) {
						if( ifCond == null ) {
							regionParent.addStatementAfter(seqLoop, copyin_stmt);
							regionParent.addStatementAfter(seqLoop, gpuBytes_stmt.clone());
						} else {
							CompoundStatement ifBody2 = new CompoundStatement();
							ifBody2.addStatement(gpuBytes_stmt.clone());
							ifBody2.addStatement(copyin_stmt);
/*							if( resetStatusCallStmt != null ) {
								ifBody2.addStatement(resetStatusCallStmt);
							}*/
							IfStatement ifStmt2 = new IfStatement(ifCond.clone(), ifBody2);
							regionParent.addStatementAfter(seqLoop, ifStmt2);
						}
					} else {
/*						if( kernelVerification && (resultCompareStmts != null) ) {
							for( int k=resultCompareStmts.size()-1; k>=0; k-- ) {
								postRedStmts.add(0, resultCompareStmts.get(k));
							}
						}
						if( resetStatusCallStmt != null ) {
							postRedStmts.add(0, resetStatusCallStmt);
						}*/
						postRedStmts.add(0, copyin_stmt);
						postRedStmts.add(0, gpuBytes_stmt.clone());
					}
					gPostRedStmts.addAll(postRedStmts);
					
				} //end of reduction copyout loop
			}
			if( !gPostRedStmts.isEmpty() ) {
				//AssignmentExpression assignExp = new AssignmentExpression(numBlocks.clone(), AssignmentOperator.NORMAL, totalnumgangs);
				//gPostRedStmts.add(0, new ExpressionStatement(assignExp));
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
		}
		PrintTools.println("[seqKernelLoopTransformation() ends] current procedure: " + cProc.getSymbolName() +
				"\ncompute region type: " + cRegionKind + "\n", 2);
	}

}
