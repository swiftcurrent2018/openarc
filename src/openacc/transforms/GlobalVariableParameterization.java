/**
 * 
 */
package openacc.transforms;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Set;

import openacc.analysis.ACCAnalysis;
import openacc.analysis.AnalysisTools;
import openacc.hir.ACCAnnotation;
import cetus.analysis.CallGraph;
import cetus.analysis.CallGraph.Node;
import cetus.exec.Driver;
import cetus.hir.AccessExpression;
import cetus.hir.AccessSymbol;
import cetus.hir.Annotatable;
import cetus.hir.Annotation;
import cetus.hir.BreadthFirstIterator;
import cetus.hir.CompoundStatement;
import cetus.hir.DFIterator;
import cetus.hir.Declaration;
import cetus.hir.Expression;
import cetus.hir.FlatIterator;
import cetus.hir.FunctionCall;
import cetus.hir.IRTools;
import cetus.hir.Identifier;
import cetus.hir.NameID;
import cetus.hir.NestedDeclarator;
import cetus.hir.PointerSpecifier;
import cetus.hir.PrintTools;
import cetus.hir.Procedure;
import cetus.hir.ProcedureDeclarator;
import cetus.hir.Program;
import cetus.hir.PseudoSymbol;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.Symbol;
import cetus.hir.SymbolTools;
import cetus.hir.Tools;
import cetus.hir.TranslationUnit;
import cetus.hir.Traversable;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import cetus.transforms.TransformPass;

/**
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group
 *         Oak Ridge National Laboratory
 *
 */
public class GlobalVariableParameterization extends TransformPass {
	private Map<Procedure, Set<Symbol>> proc2gsymMap;
	private boolean IRSymbolOnly;
	private boolean AssumeNoAliasingAmongKernelArgs = false;
	private boolean shrdArryCachingOnConst = false;

	/**
	 * @param program
	 */
	public GlobalVariableParameterization(Program program, boolean IRSymOnly, boolean NoAliasing) {
		super(program);
		IRSymbolOnly = IRSymOnly;
		AssumeNoAliasingAmongKernelArgs = NoAliasing;
		disable_protection = true;
		verbosity = 1;
	}

	/* (non-Javadoc)
	 * @see cetus.transforms.TransformPass#getPassName()
	 */
	@Override
	public String getPassName() {
		// TODO Auto-generated method stub
		return "[GlobalVariableParameterization]";
	}

	/* (non-Javadoc)
	 * @see cetus.transforms.TransformPass#start()
	 */
	@Override
	public void start() {
		String mainEntryFunc = null;
		String value = Driver.getOptionValue("SetAccEntryFunction");
		if( (value != null) && !value.equals("1") ) {
			mainEntryFunc = value;
		}
		value = Driver.getOptionValue("shrdArryCachingOnConst");
		if( value != null ) {
			shrdArryCachingOnConst = true;
		}
		//Step1: Generate a list of device functions called in GPU kernels
		Map<TranslationUnit, Integer> trCntMap = new HashMap<TranslationUnit, Integer>(); 
		Map<TranslationUnit, Set<Annotatable>> tr2kernelsMap = new HashMap<TranslationUnit, Set<Annotatable>>();
		Set<Procedure> devProcSet = new HashSet<Procedure>();
		List<Traversable> tList = program.getChildren();
		if( tList != null ) {
			int i=0;
			for( Traversable t : tList ) {
				trCntMap.put((TranslationUnit)t, new Integer(i++));
				List<ACCAnnotation> kAnnots = AnalysisTools.collectPragmas(t, ACCAnnotation.class, 
						ACCAnnotation.computeRegions, false);
				if( kAnnots != null ) {
					Set<Annotatable> kernels = new HashSet<Annotatable>();
					for( ACCAnnotation kAnnot : kAnnots ) {
						Annotatable at = kAnnot.getAnnotatable();
						Set<Procedure> procList = AnalysisTools.ipGetCalledProcedures(at, null);
						if( procList != null ) {
							devProcSet.addAll(procList);
							kernels.add(at); //add only if the kernel contains function calls.
						}
					}
					if( !kernels.isEmpty() ) {
						tr2kernelsMap.put((TranslationUnit)t, kernels);
					}
				}
			}
		}
		if( devProcSet.isEmpty() ) {
			return;
		} else {
			Procedure main = AnalysisTools.findMainEntryFunction(program, mainEntryFunc);
			if( main == null ) {
				PrintTools.println("\n[WARNING in GlobalVariableParameterization] This transform pass is skipped " +
						"since the compiler can not find accelerator entry function or main function; " +
						"if any function called in a compute region accesses global variables, user SHOULD manually change " +
						"the function such that the accessed global variables are " +
						"passed explicitly as function parameters.\n", 0);
			PrintTools.println("To enable this pass, the main entry function should be explicitly specified using " +
					"\"SetAccEntryFunction\" option.\n", 0);
				return;
			}
		}
		//Step2: Generate (procedure, global variables) mapping for each device function
		//       using bottom-up callgraph traversing.
		// generate a list of procedures in post-order traversal
		//CallGraph callgraph = new CallGraph(program);
		CallGraph callgraph = new CallGraph(program, mainEntryFunc);
		HashMap cgMap = callgraph.getCallGraph();
		// procedureList contains Procedure in ascending order; the last one is main
		List<Procedure> procedureList = callgraph.getTopologicalCallList();
		Map<TranslationUnit, Map<String, Symbol>> tr2gsymmap = new HashMap<TranslationUnit, Map<String, Symbol>>();
		proc2gsymMap = new HashMap<Procedure, Set<Symbol>>();
		boolean foundDevFuncWGSyms = false;
		for( Procedure tProc : procedureList ) {
			if( devProcSet.contains(tProc) ) {
				Set<Symbol> gSymSet = proc2gsymMap.get(tProc);
				if( gSymSet == null ) {
					gSymSet = new HashSet<Symbol>();
					proc2gsymMap.put(tProc, gSymSet);
				}
				TranslationUnit tr = (TranslationUnit)tProc.getParent();
				Map<String, Symbol> visibleGSymMap = tr2gsymmap.get(tr);
				if( visibleGSymMap == null ) {
					Set<Symbol> tSet = SymbolTools.getGlobalSymbols(tr);
					visibleGSymMap = new HashMap<String, Symbol>();
					for( Symbol gS : tSet ) {
						if( !shrdArryCachingOnConst || !SymbolTools.isArray(gS) || SymbolTools.isPointer(gS) 
								|| !gS.getTypeSpecifiers().contains(Specifier.CONST) ) {
							visibleGSymMap.put(gS.getSymbolName(), gS);
						}
					}
					tr2gsymmap.put(tr, visibleGSymMap);
				}
				CompoundStatement cBody = tProc.getBody();
				if( cBody != null ) {
					//DEBUG: below call will add IR symbols only.
					//gSymSet.addAll(AnalysisTools.getAccessedGlobalSymbols(cBody, visibleGSymMap));
					Set<Symbol> ttSymSet = AnalysisTools.getAccessedGlobalSymbols(cBody, visibleGSymMap, false);
					for(Symbol ttSym : ttSymSet ) {
						if( !shrdArryCachingOnConst || !SymbolTools.isArray(ttSym) || SymbolTools.isPointer(ttSym) 
								|| !ttSym.getTypeSpecifiers().contains(Specifier.CONST) ) {
							gSymSet.add(ttSym);
						}
					}
					Node node = (Node)cgMap.get(tProc);
					List<Procedure> callees = node.getCallees();
					if( callees != null ) {
						for( Procedure tP : callees ) {
							Set<Symbol> tSet = proc2gsymMap.get(tP);
							if( tSet != null ) {
								gSymSet.addAll(tSet);
							}
						}
					}
					if( !gSymSet.isEmpty() ) {
						foundDevFuncWGSyms = true;
					}
				}
			}
		}
		if( foundDevFuncWGSyms ) {
			Map<Procedure, Procedure> devProcMap = new HashMap<Procedure, Procedure>();
			Map<Procedure, Map<Symbol, Symbol>> proc2gsymParamMap = new HashMap<Procedure, Map<Symbol, Symbol>>();
			for( TranslationUnit trUnt : tr2kernelsMap.keySet() ) {
				devProcMap.clear();
				proc2gsymParamMap.clear();
				//Step3: Generate (procedure, new device procedure) and 
				//       (new procedure, (global variable, function parameter)) mapping for each
				//       device function.
				//       - Add routine directive if not existing
				//       - New procedure name will be "dev__" + procname + "TU" + TrUntCnt
				//Step4: Update each device function callsite in a Depth-First-Search manner.
				Set<Annotatable> kernels = tr2kernelsMap.get(trUnt);
				for( Annotatable at : kernels ) {
					ACCAnnotation iAnnot = at.getAnnotation(ACCAnnotation.class, "accreadonly");
					Set<Symbol> accROSet = null;
					if( iAnnot != null ) {
						accROSet = iAnnot.get("accreadonly");
					} else {
						accROSet = new HashSet<Symbol>();
					}
					devProcCloning(at, trUnt, trCntMap.get(trUnt).toString(), null, devProcMap, proc2gsymParamMap, accROSet);
				}
			}
		}
		//Remove unused procedures created in this pass.
		TransformTools.removeUnusedProcedures(program);
	}
	
	private void devProcCloning(Traversable at, TranslationUnit trUnt, String TrCnt, Procedure callerProc,
			Map<Procedure, Procedure> devProcMap, Map<Procedure, Map<Symbol, Symbol>> proc2gsymParamMap,
			Set<Symbol> accROSet) {
		List<FunctionCall> funcList = IRTools.getFunctionCalls(at);
		if( funcList != null ) {
			for( FunctionCall fCall : funcList ) {
				Procedure c_proc = AnalysisTools.findProcedure(fCall);
				if( (c_proc != null) && proc2gsymMap.containsKey(c_proc) ) {
					Set<Symbol> gSymSet = proc2gsymMap.get(c_proc);
					if( !gSymSet.isEmpty() ) {
						String new_proc_name = c_proc.getSymbolName() + "_GP" + TrCnt;
						NameID new_procID = new NameID(new_proc_name);
						FunctionCall new_fCall = null;
						Map<Symbol, Symbol> parentGsymParamMap = null;
						if( callerProc != null ) {
							parentGsymParamMap = proc2gsymParamMap.get(callerProc);
						}
						//Declaration decl = SymbolTools.findSymbol(trUnt, new_procID);
						Collection<Symbol> sortedGSyms = AnalysisTools.getSortedCollection(gSymSet);
						if( !devProcMap.containsKey(c_proc) ) {
							/////////////////////////////
							// Clone current procedure //
							/////////////////////////////
							//FIXME: if the current procedure has routine bind clause, and if the argument is different
							//from the procedure name, below transformation should be skipped, but the current kernel
							//translation pass may not recognize the other implementation correctly. To fix this, the
							//other implementation should not have any global variable accessed without parameter passing.
							//If a procedure has a static variable, it should not be cloned.
							//Set<Symbol> symSet = SymbolTools.getVariableSymbols(c_proc.getBody());
							Set<Symbol> symSet = SymbolTools.getLocalSymbols(c_proc.getBody());
							Set<Symbol> staticSyms = AnalysisTools.getStaticVariables(symSet);
							if( !staticSyms.isEmpty() ) {
								Tools.exit("[ERROR in GlobalVariableParameterization] if a procedure has static variables," +
										"it can not be cloned; for correct transformation, either \"disableStatic2GlobalConversion\" " +
										"option should be disabled or static variables should be manually promoted to global ones.\n" +
										"Procedure name: " + c_proc.getSymbolName() + "\n");
							}
							List<Specifier> return_types = c_proc.getReturnType();
							List<VariableDeclaration> oldParamList = 
								(List<VariableDeclaration>)c_proc.getParameters();
							CompoundStatement body = (CompoundStatement)c_proc.getBody().clone();
							Procedure new_proc = new Procedure(return_types,
									new ProcedureDeclarator(new_procID,
											new LinkedList()), body);	
							devProcMap.put(c_proc, new_proc);
							/////////////////////////////////
							// Update function parameters. //
							/////////////////////////////////
							int oldParamListSize = oldParamList.size();
							if( oldParamListSize == 1 ) {
								Object obj = oldParamList.get(0);
								String paramS = obj.toString();
								// Remove any leading or trailing whitespace.
								paramS = paramS.trim();
								if( paramS.equals(Specifier.VOID.toString()) ) {
									oldParamListSize = 0;
								}
							}
							if( oldParamListSize > 0 ) {
								for( VariableDeclaration param : oldParamList ) {
									Symbol param_declarator = (Symbol)param.getDeclarator(0);
									VariableDeclaration cloned_decl = (VariableDeclaration)param.clone();
									Identifier paramID = new Identifier(param_declarator);
									Identifier cloned_ID = new Identifier((Symbol)cloned_decl.getDeclarator(0));
									new_proc.addDeclaration(cloned_decl);
									TransformTools.replaceAll((Traversable) body, paramID, cloned_ID);
								}
							}
							//////////////////////////////////////////////////////////
							// Create a new function call for the cloned procedure. //
							//////////////////////////////////////////////////////////
							if( fCall != null ) {
								new_fCall = new FunctionCall(new NameID(new_proc_name));
								List<Expression> argList = (List<Expression>)fCall.getArguments();
								if( argList != null ) {
									for( Expression exp : argList ) {
										new_fCall.addArgument(exp.clone());
									}
								}
								fCall.swapWith(new_fCall);
							}
							/////////////////////////////////////////////////////////////////////////////////////////
							//Add new parameters/arguments for each global symbol accessed in the device function. //
							/////////////////////////////////////////////////////////////////////////////////////////
							Map<Symbol, Symbol> gsymParamMap = new HashMap<Symbol, Symbol>();
							proc2gsymParamMap.put(new_proc, gsymParamMap);
							List<Specifier> removeSpecs = new ArrayList<Specifier>();
							removeSpecs.add(Specifier.STATIC);
							removeSpecs.add(Specifier.CONST);
							removeSpecs.add(Specifier.EXTERN);
							for( Symbol gSym : sortedGSyms ) {
								Boolean isArray = SymbolTools.isArray(gSym);
								Boolean isPointer = SymbolTools.isPointer(gSym);
								if( gSym instanceof NestedDeclarator ) {
									isPointer = true;
								}
								Boolean isScalar = !isArray && !isPointer;
								List<Specifier> typeSpecs = new ArrayList<Specifier>();
								Boolean isStruct = false;
								Symbol IRSym = gSym;
								if( gSym instanceof PseudoSymbol ) {
									IRSym = ((PseudoSymbol)gSym).getIRSymbol();
								}
								if( IRSymbolOnly ) {
									typeSpecs.addAll(((VariableDeclaration)IRSym.getDeclaration()).getSpecifiers());
									isStruct = SymbolTools.isStruct(IRSym, at);
								} else {
									Symbol tSym = gSym;
									while( tSym instanceof AccessSymbol ) {
										tSym = ((AccessSymbol)tSym).getMemberSymbol();
									}
									typeSpecs.addAll(((VariableDeclaration)tSym.getDeclaration()).getSpecifiers());
									isStruct = SymbolTools.isStruct(tSym, at);
								}
								typeSpecs.removeAll(removeSpecs);
								Symbol argSym = gSym;
								if( parentGsymParamMap != null ) {
									argSym = parentGsymParamMap.get(gSym);
								}

								Identifier kParamVar = null;
								String symNameBase = null;
								if( gSym instanceof AccessSymbol) {
									symNameBase = TransformTools.buildAccessSymbolName((AccessSymbol)gSym);
								} else {
									symNameBase = gSym.getSymbolName();
								}
								//Create a kernel parameter for the global variable.
								Expression replaceExp;
								if( isScalar ) {
									//DEBUG: currently, gSym is IR symbol.
									if( !isStruct && accROSet.contains(gSym) ) {
										// Parameter is passed by value.
										// Create a GPU kernel parameter corresponding to shared_var
										VariableDeclarator kParam_declarator = new VariableDeclarator(new NameID(symNameBase));
										VariableDeclaration kParam_decl = new VariableDeclaration(typeSpecs,
												kParam_declarator);
										kParamVar = new Identifier(kParam_declarator);
										new_proc.addDeclaration(kParam_decl);
										// Replace all instances of the global variable to the parameter variable
										replaceExp = kParamVar.clone();
									} else {
										// Parameter is passed by reference.
										// Create a parameter Declaration for the kernel function
										// Change the scalar variable to a pointer type 
										VariableDeclarator kParam_declarator = new VariableDeclarator(PointerSpecifier.UNQUALIFIED, 
												new NameID(symNameBase));
										VariableDeclaration kParam_decl = new VariableDeclaration(typeSpecs,
												kParam_declarator);
										kParamVar = new Identifier(kParam_declarator);
										new_proc.addDeclaration(kParam_decl);
										// Replace all instances of the global variable to a pointer-dereferencing expression (ex: *x).
										replaceExp = new UnaryExpression(UnaryOperator.DEREFERENCE, 
												(Identifier)kParamVar.clone());
									}
								} else {
									kParamVar = TransformTools.declareClonedVariable(new_proc, gSym, symNameBase, removeSpecs, null, true, AssumeNoAliasingAmongKernelArgs);
									replaceExp = kParamVar.clone();
								}
								gsymParamMap.put(gSym, kParamVar.getSymbol());
								/////////////////////////////////////////////////
								// Insert argument to the kernel function call //
								/////////////////////////////////////////////////
								Expression argExp;
								if( argSym instanceof AccessSymbol ) {
									argExp = AnalysisTools.accessSymbolToExpression((AccessSymbol)argSym, null);
								} else {
									argExp = new Identifier(argSym);
								}
								if( isScalar && (isStruct || !accROSet.contains(gSym)) && (gSym == argSym) ) {
									new_fCall.addArgument(new UnaryExpression( UnaryOperator.ADDRESS_OF, argExp));
								} else {
									new_fCall.addArgument(argExp);
								}
								////////////////////////////////////////////////////////////////////////////
								// Replace all instances of the global variable to the parameter variable //
								////////////////////////////////////////////////////////////////////////////
								if( gSym instanceof AccessSymbol ) {
									TransformTools.replaceAccessExpressions(body, (AccessSymbol)gSym, replaceExp);
								} else {
									TransformTools.replaceAll(body, new Identifier(gSym), replaceExp);
								}
							}
							
							TranslationUnit tu = (TranslationUnit)c_proc.getParent();
							////////////////////////////
							// Add the new procedure. //
							////////////////////////////
							if( (trUnt == tu) && (!AnalysisTools.isInHeaderFile(c_proc, trUnt)) ) {
								trUnt.addDeclarationAfter(c_proc, new_proc);
							} else {
								Procedure firstProc = AnalysisTools.findFirstProcedure(trUnt);
								trUnt.addDeclarationBefore(firstProc, new_proc);
							}
							////////////////////////////////////////////////////////////////////////
							//If the current procedure has annotations, copy them to the new one. //
							////////////////////////////////////////////////////////////////////////
							List<Annotation> cAnnotList = c_proc.getAnnotations();
							if( (cAnnotList != null) && (!cAnnotList.isEmpty()) ) {
								for( Annotation cAn : cAnnotList ) {
									new_proc.annotate(cAn.clone());
								}
							}
							///////////////////////////////////////////////////
							// Add routine directive for this new procedure. //
							///////////////////////////////////////////////////
							ACCAnnotation rAnnot = new_proc.getAnnotation(ACCAnnotation.class, "routine");
							if( rAnnot == null ) {
								rAnnot = new ACCAnnotation("routine", "_directive");
								new_proc.annotate(rAnnot);
							}
							rAnnot.put("nohost", "_clause");
							
							//////////////////////////////////////////////////////////////////
							//If declaration statement exists for the original procedure,   //
							//create a new declaration statement for the new procedure too. //
							//////////////////////////////////////////////////////////////////
							FlatIterator Fiter = new FlatIterator(program);
							while (Fiter.hasNext())
							{
								TranslationUnit cTu = (TranslationUnit)Fiter.next();
								DFIterator<ProcedureDeclarator> iter = new DFIterator<ProcedureDeclarator>(cTu, ProcedureDeclarator.class);
								iter.pruneOn(ProcedureDeclarator.class);
								iter.pruneOn(Procedure.class);
								iter.pruneOn(Statement.class);
								for (;;)
								{
									ProcedureDeclarator procDeclr = null;

									try {
										procDeclr = (ProcedureDeclarator)iter.next();
									} catch (NoSuchElementException e) {
										break;
									}
									if( procDeclr.getID().equals(c_proc.getName()) ) {
										Traversable parent = procDeclr.getParent();
										if( parent instanceof VariableDeclaration ) {
											//Found function declaration.
											VariableDeclaration procDecl = (VariableDeclaration)parent;
											//Create a new function declaration.
											VariableDeclaration newProcDecl = 
												new VariableDeclaration(procDecl.getSpecifiers(), new_proc.getDeclarator().clone());
											//Insert the new function declaration.
											if( !AnalysisTools.isInHeaderFile(procDecl, cTu) ) {
												cTu.addDeclarationAfter(procDecl, newProcDecl);
											} else {
												Procedure firstProc = AnalysisTools.findFirstProcedure(cTu);
												if( firstProc == null ) {
													cTu.addDeclaration(newProcDecl);
												} else {
													cTu.addDeclarationBefore(firstProc, newProcDecl);
												}
											}
											////////////////////////////////////////////////////////////////////////////////////
											//If the current procedure declaration has annotations, copy them to the new one. //
											////////////////////////////////////////////////////////////////////////////////////
											cAnnotList = procDecl.getAnnotations();
											if( (cAnnotList != null) && (!cAnnotList.isEmpty()) ) {
												for( Annotation cAn : cAnnotList ) {
													newProcDecl.annotate(cAn.clone());
												}
											}
											rAnnot = newProcDecl.getAnnotation(ACCAnnotation.class, "routine");
											if( rAnnot == null ) {
												rAnnot = new ACCAnnotation("routine", "_directive");
												newProcDecl.annotate(rAnnot);
											}
											rAnnot.put("nohost", "_clause");

											ACCAnalysis.updateSymbolsInACCAnnotations(newProcDecl, null);
											break;
										}
									}
								}
							}
							/////////////////////////////////////////////////////////////////////////
							// Update the newly cloned procedure:                                  //
							//     1) Update symbols in the new procedure, including symbols       //
							//        in ACCAnnoations.                                            //
							/////////////////////////////////////////////////////////////////////////
							SymbolTools.linkSymbol(new_proc, 1);
							ACCAnalysis.updateSymbolsInACCAnnotations(new_proc, null);
							
							////////////////////////////////////////////////////////////////////////
							// Check functions called in the current device function recursively. //
							////////////////////////////////////////////////////////////////////////
							devProcCloning(body, trUnt, TrCnt, new_proc, devProcMap, proc2gsymParamMap, accROSet);
						} else {
							//cloned device procedure already exist; just change function calls.
							Procedure new_proc = devProcMap.get(c_proc);
							//////////////////////////////////////////////////////////
							// Create a new function call for the cloned procedure. //
							//////////////////////////////////////////////////////////
							if( fCall != null ) {
								new_fCall = new FunctionCall(new NameID(new_proc_name));
								List<Expression> argList = (List<Expression>)fCall.getArguments();
								if( argList != null ) {
									for( Expression exp : argList ) {
										new_fCall.addArgument(exp.clone());
									}
								}
								fCall.swapWith(new_fCall);
							}
							for( Symbol gSym : sortedGSyms ) {
								Boolean isArray = SymbolTools.isArray(gSym);
								Boolean isPointer = SymbolTools.isPointer(gSym);
								if( gSym instanceof NestedDeclarator ) {
									isPointer = true;
								}
								Boolean isScalar = !isArray && !isPointer;
								Boolean isStruct = false;
								Symbol IRSym = gSym;
								if( gSym instanceof PseudoSymbol ) {
									IRSym = ((PseudoSymbol)gSym).getIRSymbol();
								}
								if( IRSymbolOnly ) {
									isStruct = SymbolTools.isStruct(IRSym, at);
								} else {
									Symbol tSym = gSym;
									while( tSym instanceof AccessSymbol ) {
										tSym = ((AccessSymbol)tSym).getMemberSymbol();
									}
									isStruct = SymbolTools.isStruct(tSym, at);
								}
								Symbol argSym = gSym;
								if( parentGsymParamMap != null ) {
									argSym = parentGsymParamMap.get(gSym);
								}
								/////////////////////////////////////////////////
								// Insert argument to the kernel function call //
								/////////////////////////////////////////////////
								Expression argExp;
								if( argSym instanceof AccessSymbol ) {
									argExp = AnalysisTools.accessSymbolToExpression((AccessSymbol)argSym, null);
								} else {
									argExp = new Identifier(argSym);
								}
								if( isScalar && (isStruct || !accROSet.contains(gSym)) && (gSym == argSym) ) {
									new_fCall.addArgument(new UnaryExpression( UnaryOperator.ADDRESS_OF, argExp));
								} else {
									new_fCall.addArgument(argExp);
								}
							}
						}
					}
				}
			}
		}
	}

}
