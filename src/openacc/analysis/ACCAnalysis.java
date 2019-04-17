package openacc.analysis;

import java.io.*;
import java.lang.reflect.Method;
import java.lang.Math;
import java.util.*;

import cetus.hir.*;
import cetus.hir.Enumeration;
import cetus.exec.*;
import cetus.analysis.*;
import openacc.hir.*;
import openacc.transforms.TransformTools;

/**
 * This pass analyzes OpenACC pragmas, update symbols, check correctness, add implicit clauses,
 * and add implicit variables, which are referenced but not explicitly included in the data clauses,
 * into proper data clauses.
 * 
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public class ACCAnalysis extends AnalysisPass
{
	private int debug_level;
	private int debug_tab;
	private ACCAnnotation programAnnot;
	private ARCAnnotation programCudaAnnot;
	private HashSet<String> visitedProcs;
	private	HashSet<String> threadprivSet;
	private boolean IRSymbolOnly;

	public ACCAnalysis(Program program, boolean IRSymOnly, ACCAnnotation inAnnot, ARCAnnotation inCAnnot)
	{
		super(program);
		debug_level = Integer.valueOf(Driver.getOptionValue("verbosity")).intValue();
		debug_tab = 0;
		IRSymbolOnly = IRSymOnly;
		programAnnot = inAnnot;
		programCudaAnnot = inCAnnot;
	}

	public String getPassName()
	{
		return new String("[ACCAnalysis]");
	}

	public void start()
	{
		//////////////////////////////////////////////////////////////////////////////////////////////
		// Analysis passes in openacc assume that structured block is either CompoundStatement or   //
		// ForLoop. However, a single executable statement with a single entry at the top and a     //
		// single exit at the bottom can constitute the structured block.                           //
		// For analysis passes to handle the single statement case, a "single-statement" structured //
		// block is replaced with a CompoundStatement where the single statement is contained.      //
		//////////////////////////////////////////////////////////////////////////////////////////////
		for( String pragma : ACCAnnotation.directivesForStructuredBlock ) {
			List<ACCAnnotation> ACCAnnots = 
				IRTools.collectPragmas(program, ACCAnnotation.class, pragma);
			for( ACCAnnotation pAnnot : ACCAnnots ) {
				Annotatable at = pAnnot.getAnnotatable();
				if( at instanceof ExpressionStatement ) {
					Statement stmt = (ExpressionStatement)at;
					CompoundStatement cStmt = new CompoundStatement();
					List<Annotation> annotList = stmt.getAnnotations();
					stmt.removeAnnotations();
					for( Annotation annot : annotList ) {
						cStmt.annotate(annot);
					}
					stmt.swapWith(cStmt);
					cStmt.addStatement(stmt);
				}
			}
		}
		
		computeRegionAnalysis();
		
		// Replace NameIDs in the ACCAnnotation clauses with Identifiers.
		updateSymbolsInACCAnnotations(program, null);
		
		declareDirectiveAnalysis(IRSymbolOnly);
		
		shared_analysis(IRSymbolOnly);
		
		updateAnalysis();
		
		asyncAnalysis();
		
		hostDataAnalysis();
		
		routineDirectiveAnalysis();
	}

	/**
	 * Check nested compute-regions, which are not allowed in OpenACC V1.0.
	 * 
	 */
	private void computeRegionAnalysis() {
		List<ACCAnnotation>  cRegionAnnots = AnalysisTools.collectPragmas(program, ACCAnnotation.class, ACCAnnotation.computeRegions, false);
		if( cRegionAnnots != null ) {
			for( ACCAnnotation cAnnot : cRegionAnnots ) {
				Annotatable at = cAnnot.getAnnotatable();
				for( Traversable child : at.getChildren() ) {
					if( AnalysisTools.ipContainPragmas(child, ACCAnnotation.class, ACCAnnotation.computeRegions, false, null) ) {
						Tools.exit("[ERROR] Nested compute-region is not allowed in OpenACC V1.0; exit!\n" +
								"OpenACC compute region having nested compute-region: " +
								cAnnot + AnalysisTools.getEnclosingAnnotationContext(cAnnot));	
					}
				}
				if( at instanceof Loop ) {
					if( !at.containsAnnotation(ACCAnnotation.class, "loop") ) {
						Tools.exit("[ERROR] the following compute region is a loop, but not a combined loop (paralllel loop or kernels loop); " +
								"for correct translation, this region should be changed to either \"a data region with inner compute regions\" or " +
								"\"a combined loop\", depending on the program semantics.\n" +
								"OpenACC compute region: " +
								cAnnot + AnalysisTools.getEnclosingAnnotationContext(cAnnot));	
					}
				}
				if( at.containsAnnotation(ACCAnnotation.class, "parallel") ) { //parallel region
					List<ACCAnnotation> workshareclauses = AnalysisTools.ipCollectPragmas(at, ACCAnnotation.class, ACCAnnotation.parallelWorksharingClauses, 
							false, null);
					if( workshareclauses != null ) {
						for( ACCAnnotation wAnnot : workshareclauses ) {
							for(String wclause : ACCAnnotation.parallelWorksharingClauses ) {
								Object obj = wAnnot.get(wclause);
								if( (obj != null) && (obj instanceof Expression) ) {
									Tools.exit("[ERROR] In a parallel region, gang/worker clauses cannot have arguments; exit\n" +
											"OpenACC compute region: " +
											cAnnot + AnalysisTools.getEnclosingAnnotationContext(cAnnot));	
								}
							}
						}
					}
				}
			}
		}
	}
	
	/**
	 * OpenACC asysnc clause analysis
	 * 
	 * For each async clause, 
	 *     //- If its argument is empty, add unique integer value as its argument
	 *     - If its argument is empty, use acc_async_noval as its argument
	 *     - If its argument is not an integer constant, error. (this constrain is removed.)
	 */
	private void asyncAnalysis() {
		List<ACCAnnotation>  asyncAnnots = IRTools.collectPragmas(program, ACCAnnotation.class, "async");
		List<ACCAnnotation> emptyAsyncs = new LinkedList<ACCAnnotation>();
		long maxVal = 0;
		if( asyncAnnots != null ) {
			boolean firstAnnot = true;
			boolean allIntArgs = true;
			for( ACCAnnotation cAnnot : asyncAnnots ) {
				Object val = cAnnot.get("async");
				if( val instanceof Expression ) {
					Expression argument = Symbolic.simplify((Expression)val);
					if( argument instanceof IntegerLiteral ) {
						if( firstAnnot ) {
							maxVal = ((IntegerLiteral)argument).getValue();
							firstAnnot = false;
						} else {
							maxVal = java.lang.Math.max(maxVal, ((IntegerLiteral)argument).getValue());
						}
					} else {
						allIntArgs = false;
/*						Annotatable at = cAnnot.getAnnotatable();
						Procedure pProc = IRTools.getParentProcedure(at);
						Tools.exit("[ERROR] in the implementation, the argument of async clause should an integer constant, " +
								"but the following async clause has non-constant expression as its argument.\n" +
								"OpenACC annotation: " +
								cAnnot + AnalysisTools.getEnclosingAnnotationContext(cAnnot);	*/
					}
				} else { //argument is unspecified.
					emptyAsyncs.add(cAnnot);
				}
			}
			if( !emptyAsyncs.isEmpty() ) {
				//DEBUG: if async argument is not specified, use a value distinct from all existing async arguments in the program.
				/*Expression uniqueID;
				if( allIntArgs ) {
					uniqueID = new IntegerLiteral(++maxVal);
				} else {
					uniqueID = new NameID("INT_MAX");
				}
				for( ACCAnnotation cAnnot : emptyAsyncs ) {
					cAnnot.put("async", uniqueID.clone());
				}*/
				for( ACCAnnotation cAnnot : emptyAsyncs ) {
					cAnnot.put("async", new NameID("acc_async_noval"));
				}
			}
		}
	}
	
	/**
	 *  UpdateAnalysis do the following things:
	 *  1) If a variable in an update directive is not included in any of enclosing explicit/implicit data region, error.
	 *     Else if the subarray in the update directive does not have array boundary information, 
	 *         copy the boundary information from the one in the enclosing data region.     
	 * 
	 */
	private void updateAnalysis() {
		List<ACCAnnotation>  updateAnnots = IRTools.collectPragmas(program, ACCAnnotation.class, "update");
		if( updateAnnots != null ) {
			List<FunctionCall> gFuncCallList = IRTools.getFunctionCalls(program);
			for( ACCAnnotation uAnnot : updateAnnots ) {
				//Step1: find symbols included in the update clauses.
				Set<Symbol> uSymSet = new HashSet<Symbol>();
				Set<SubArray> uSArraySet = new HashSet<SubArray>();
				Set<String> updateClauses = new HashSet(Arrays.asList("host", "device"));
				for(String uClause : updateClauses ) {
					Set<SubArray> uSArrays = uAnnot.get(uClause);
					if( uSArrays != null ) {
						uSArraySet.addAll(uSArrays);
						uSymSet.addAll(AnalysisTools.subarraysToSymbols(uSArrays, IRSymbolOnly));
					}
				}
				//Step2: find enclosing explicit/implicit data regions
				Annotatable at = uAnnot.getAnnotatable();
				Procedure parentProc = IRTools.getParentProcedure(at);
				LinkedList<ACCAnnotation> enclosingDataRegions = new LinkedList<ACCAnnotation>();
				ACCAnnotation drAnnot = AnalysisTools.ipFindFirstPragmaInParent(at, ACCAnnotation.class, "data", gFuncCallList, null);
				while( drAnnot != null ) {
					enclosingDataRegions.add(drAnnot);
					drAnnot = AnalysisTools.ipFindFirstPragmaInParent(drAnnot.getAnnotatable(), ACCAnnotation.class, "data", gFuncCallList, null);
				}
				//Step3: for each symbol in the update clauses,
				//       - If it is not included in any enclosing explicit/implicit data region, error.
				Set<Symbol> DSharedSymbols = new HashSet<Symbol>();
				Map<Procedure, List<Set<Symbol>>> dRegionMap = new HashMap<Procedure, List<Set<Symbol>>>();
				Map<Procedure, List<Set<String>>> dRegionGMap = new HashMap<Procedure, List<Set<String>>>();
				Map<Procedure, List<ACCAnnotation>> dRegionAnnotMap = new HashMap<Procedure, List<ACCAnnotation>>();
				while( !enclosingDataRegions.isEmpty() ) {
					drAnnot = enclosingDataRegions.removeFirst();
					Annotatable dAt = drAnnot.getAnnotatable();
					Procedure dPproc = IRTools.getParentProcedure(dAt);
					ACCAnnotation idAnnot = dAt.getAnnotation(ACCAnnotation.class, "accshared");
					if( (dPproc !=null) && (idAnnot != null) ) {
						Set<Symbol> idSymSet = (Set<Symbol>)idAnnot.get("accshared");
						Set<String> idGSymSSet = null;
						if( idAnnot.containsKey("accglobal") ) {
							idGSymSSet = (Set<String>)idAnnot.get("accglobal");
						} else {
							idGSymSSet = new HashSet<String>();
							idAnnot.put("accglobal", idGSymSSet);
						}
						List<Set<Symbol>> accSharedSetList = null;
						if( dRegionMap.containsKey(dPproc) ) {
							accSharedSetList = (List<Set<Symbol>>)dRegionMap.get(dPproc);
						} else {
							accSharedSetList = new LinkedList<Set<Symbol>>();
							dRegionMap.put(dPproc, accSharedSetList);
						}
						accSharedSetList.add(idSymSet);
						List<Set<String>> accGlobalSetList = null;
						if( dRegionGMap.containsKey(dPproc) ) {
							accGlobalSetList = (List<Set<String>>)dRegionGMap.get(dPproc);
						} else {
							accGlobalSetList = new LinkedList<Set<String>>();
							dRegionGMap.put(dPproc, accGlobalSetList);
						}
						accGlobalSetList.add(idGSymSSet);
						List<ACCAnnotation> accDAnnotList = null;
						if( dRegionAnnotMap.containsKey(dPproc) ) {
							accDAnnotList = (List<ACCAnnotation>)dRegionAnnotMap.get(dPproc);
						} else {
							accDAnnotList = new LinkedList<ACCAnnotation>();
							dRegionAnnotMap.put(dPproc, accDAnnotList);
						}
							accDAnnotList.add(drAnnot);
								
					} else {
						Tools.exit("[ERROR] Internal error in ACCAnalysis.shared_analysis(): internal data, accshared set, is missing!\n" +
							"Procedure containing a data region: " + dPproc.getSymbolName() + "\nEnclosing Translation Unit: " + 
							((TranslationUnit)dPproc.getParent()).getOutputFilename() + "\n");
					}
				}
				for( Symbol dSym : uSymSet ) {
					boolean isIncluded = false;
					Symbol tdGSym = null;
					List osymList = null;
					//Step3-1: check procedure-level implicit data regions and explicit data regions.
					SubArray enSubArray = null;
					for( Procedure ttproc : dRegionMap.keySet() ) {
						List<Set<Symbol>> accSharedSetList = dRegionMap.get(ttproc);
						List<Set<String>> accGlobalSetList = dRegionGMap.get(ttproc);
						List<ACCAnnotation> accDAnnotList = dRegionAnnotMap.get(ttproc);
						int i = 0;
						for( Set<Symbol> dSymSet : accSharedSetList) {
							Set<String> dGSymSSet = accGlobalSetList.get(i);
							ACCAnnotation tDAnnot = accDAnnotList.get(i);
							if( dSymSet == null ) {
								continue;
							}
							if( dSymSet.contains(dSym) ) {
								//dSym is visible in a procedure, ttproc, and included in the data clauses of the data region in the procedure.
								isIncluded = true;
								enSubArray = AnalysisTools.findSubArrayInDataClauses(tDAnnot, dSym, IRSymbolOnly);
								break;
							} else {
								//dSym is invisible in a procedure, ttproc.
								//find a symbol visible in the ttproc, if dSym is a parameter symbol.
								osymList = new ArrayList(2);
								if( AnalysisTools.SymbolStatus.OrgSymbolFound(
										AnalysisTools.findOrgSymbol(dSym, at, false, ttproc, osymList, gFuncCallList)) ) {
									Symbol odSym = (Symbol)osymList.get(0);
									if( dSymSet.contains(odSym) ) {
										isIncluded = true;
										enSubArray = AnalysisTools.findSubArrayInDataClauses(tDAnnot, odSym, IRSymbolOnly);
										break;
									} else if( SymbolTools.isGlobal(odSym) ){ 
										//Remaining cases: 1) odSym is global and a symbol in dSymSet is also global, but they refer to
										//                    different extern symbols for the same symbol.
										//                 2) odSym is global and a symbol in dSymSet is a parameter symbol, but they
										//                    refer to the same global symbol.
										if( dGSymSSet.contains(odSym.getSymbolName()) ) {
											isIncluded = true;
											enSubArray = 
													AnalysisTools.findSubArrayInDataClauses(tDAnnot, odSym.getSymbolName(), IRSymbolOnly);
											break;
										}
									}
								}
							}
							i++;
						}
						if( isIncluded ) {
							break;
						}
					}
					//Step3-2: check program-level implicit data region.
					if( !isIncluded ) {
						Set<String> dGSymSSet = programAnnot.get("accglobal");
						osymList = new ArrayList(2);
						if( AnalysisTools.SymbolStatus.OrgSymbolFound(
								AnalysisTools.findOrgSymbol(dSym, at, true, null, osymList, gFuncCallList)) ) {
							Symbol odSym = (Symbol)osymList.get(0);
							if( SymbolTools.isGlobal(odSym) ) {
								if( dGSymSSet.contains(odSym.getSymbolName()) ) {
									isIncluded = true;
									enSubArray = 
											AnalysisTools.findSubArrayInDataClauses(programAnnot, odSym.getSymbolName(), IRSymbolOnly);
								}
							}
						}
					}
					//Step3-3: If symbol is not included in any explicit/implicit data region, error.
					//         Otherwise, update subArray information, if necessary.
					//[DEBUG] Below error message is changed to warning to enable separate compilation.
					if( !isIncluded ) {
						PrintTools.println("[WARNING] variable, " + dSym.getSymbolName() + ", in the following update directive should be included in" +
								" an explicit/implicit data region enclosing the update directive, but the compiler can not detect any enclosing data region.\n" +
								"Update directive: " + uAnnot + "\n" + AnalysisTools.getEnclosingAnnotationContext(uAnnot), 0);
					} 
					//Find corresponding subarray for the current symbol.
					SubArray sArr = AnalysisTools.subarrayOfSymbol(uSArraySet, dSym);
					if( sArr.getArrayDimension() < 0 ) {
						if( enSubArray != null ) {
							if( enSubArray.getArrayDimension() > 0 ) {
								List<Expression> tStartL = enSubArray.getStartIndices();
								List<Expression> tLengthL = enSubArray.getLengths();
								sArr.setRange(tStartL, tLengthL);
							}

						}
					}
				}
			}
		}
	}
	
	/**
	 * hostData analysis checks the following:
	 * 
	 *     - Each host_data region should be enclosed by an explicit/implicit data region.
	 *     - Each variable in a use_device clause must be present in the accelerator memory.
	 */
	private void hostDataAnalysis() {
		List<ACCAnnotation>  hostDataAnnots = IRTools.collectPragmas(program, ACCAnnotation.class, "host_data");
		if( hostDataAnnots != null ) {
			List<FunctionCall> funcCallList = IRTools.getFunctionCalls(program);
			for( ACCAnnotation cAnnot : hostDataAnnots ) {
				Annotatable at = cAnnot.getAnnotatable();
				ACCAnnotation dAnnot = AnalysisTools.ipFindFirstPragmaInParent(at, ACCAnnotation.class, "data", funcCallList, null);
				if( dAnnot == null ) {
					Tools.exit("[ERROR in ACCAnalysis.hostDataAnalysis()] host_data constuct should be enclosed by a data region," +
							" but no enclosing data region is found for the following construct:\n " +
							"host_data construct: " + cAnnot + AnalysisTools.getEnclosingAnnotationContext(cAnnot));
				} else {
					//TODO: check each variable in use_device clause is included in the enclosing data region.
				}
			}
		}
	}
	
	/**
	 * routine directive analysis checks the following:
	 * 
	 *     - Each routine directive should be attached to either function declaration or definition.
	 */
	private void routineDirectiveAnalysis() {
		List<ACCAnnotation>  routineAnnots = IRTools.collectPragmas(program, ACCAnnotation.class, "routine");
		if( routineAnnots != null ) {
			for( ACCAnnotation cAnnot : routineAnnots ) {
				Annotatable at = cAnnot.getAnnotatable();
				if( at instanceof Procedure ) {
					continue;
				} else if( at instanceof VariableDeclaration ) {
					Declarator fDeclr = ((VariableDeclaration)at).getDeclarator(0);
					if( fDeclr instanceof ProcedureDeclarator ) {
						continue;
					}
				}
				Tools.exit("[ERROR in ACCAnalysis.routineDirectiveAnalysis()] routine directive should be attached" +
						" to either function declaration or function definition," +
						" but the following directive is wrongly attached:\n " +
						"routine construct:\n" + at + AnalysisTools.getEnclosingContext(at));
			}
		}
	}
	
	/**
	 * This method checks the following restrictions on the declare directives (OpenACC V1.0 ch2.11):
	 * 
	 * - A variable or array may appear at most once in all the clauses of declare directives for a function,
	 * subroutine, program, or module. (check1)
	 * - Subarrays are not allowed in declare directives. (check2)
	 * - If a variable or array appears in a declare directive, the same variable or array may not appear
	 * in a data clause for any construct where the declaration of the variable is visible. (check3)
	 * 
	 * At the end of this analysis, the following internal annotations are added:
	 * 
	 * - If there exists declare directives for implicit program-level data region
	 *     - Add an empty data directive to the programAnnot
	 *     - Add an accshared clause whose set contains symbols bound to the program-level implicit data region.
	 *         - If a symbol is extern, corresponding original symbol is stored if existing.
	 *     - Each declare directive is annotated with accshared clause.
	 *     - CUDA data clauses in declare directives are copied to the program-level data directive.
	 * - If there exists declare directives for implicit procedure-level data region
	 *     - Add an OpenACC annotation with a data directive to the procedure
	 *     - Add an OpenACC annotation with an internal directive to the procedure
	 *         - The internal annotation contains an accshared clause whose set contains symbols bound to 
	 *         the procedure-level implicit data region.
	 *     - Each declare directive is annotated with accshared clause.
	 *     - CUDA data clauses in declare directives are copied to the procedure-level data directive.
	 * - For each explicit data region, parallel region, or kernels region
	 *     - Add an OpenACC annotation with an internal directive to the region.
	 *         - The internal annotation contains an accshared clause whose set contains symbols existing
	 *         in data clauses of the region.
	 *         - If the region is compute region (parallel region or kernels region)
	 *             - Add accprivate, accfirstprivate, and accreduction clauses to the internal annotation.
	 * - The symbols in an accshared/accprivate/accfirstprivate/accreduction set can be one of the following types:
	 *     VariableDeclarator or NestedDeclarator (If PseudoSymbol exists, its IRSymbol will be inserted instead.)
	 */
	private void declareDirectiveAnalysis(boolean IRSymbolOnly) {
		PrintTools.println("[ACCAnalysis.declareDirectiveAnalysis()] begin", 2);
		List<FunctionCall> gFuncCallList = IRTools.getFunctionCalls(program);
		//Step1: Create internal data structure to keep information on program-level internal data region.
		Set<Symbol> progAccSharedSymbols = new HashSet<Symbol>();
		//Set<Symbol> progAccPipeSymbols = new HashSet<Symbol>();
		Set<Symbol> tProgAccSharedSymbols = new HashSet<Symbol>();
		Set<String> progAccSharedSymbolStrings = new HashSet<String>();
		programAnnot.put("accshared", progAccSharedSymbols);
		//programAnnot.put("accpipe", progAccPipeSymbols);
		programAnnot.put("accglobal", progAccSharedSymbolStrings);
		//Step2: For each explicit data region or compute region, create accshared set which contains symbols used 
		//       in the data clauses of the construct. The accshared set will be stored in "acc internal" internal ACCAnnotation.
		//       The accshared set exclude symbols in private/firstprivate clauses
		List<ACCAnnotation> dataAnnots = null;
		dataAnnots = AnalysisTools.collectPragmas(program, ACCAnnotation.class, ACCAnnotation.dataRegions, false);
		if( dataAnnots != null ) {
			for( ACCAnnotation dAnnot : dataAnnots ) {
				Set<Symbol> accSharedSymbols = null;
				Set<Symbol> accPrivateSymbols = null;
				Set<Symbol> accFirstPrivateSymbols = null;
				Set<Symbol> accReductionSymbols = null;
				Annotatable at = dAnnot.getAnnotatable();
				String directiveType;
				if( at.containsAnnotation(ACCAnnotation.class, "data") ) {
					directiveType = "data";
				} else if( at.containsAnnotation(ACCAnnotation.class, "parallel") ) {
					directiveType = "parallel";
				} else {
					directiveType = "kernels";
				}
				//PrintTools.println("[ACCAnalysis.declareDirectiveAnalysis()] data region annotation: " + dAnnot, 3);
				Annotation iAnnot = at.getAnnotation(ACCAnnotation.class, "internal");
				if( iAnnot == null ) {
					iAnnot = new ACCAnnotation("internal", "_directive");
					accSharedSymbols = new HashSet<Symbol>();
					iAnnot.put("accshared", accSharedSymbols);
					if( directiveType.equals("data") ) {
						iAnnot.put("accglobal", new HashSet<String>()); //Create an empty set, which will be filled in next step.
					} else {
						accPrivateSymbols = new HashSet<Symbol>();
						accFirstPrivateSymbols = new HashSet<Symbol>();
						accReductionSymbols = new HashSet<Symbol>();
						iAnnot.put("accprivate", accPrivateSymbols); //compute region may contain private clauses.
						iAnnot.put("accfirstprivate", accFirstPrivateSymbols); //compute region may contain private clauses.
						iAnnot.put("accreduction", accReductionSymbols); //compute region may contain reduction clauses.
					}
					iAnnot.setSkipPrint(true);
					at.annotate(iAnnot);
				} else {
					accSharedSymbols = (Set<Symbol>)iAnnot.get("accshared");
					if( accSharedSymbols == null ) {
						accSharedSymbols = new HashSet<Symbol>();
						iAnnot.put("accshared", accSharedSymbols);
					}

					if( !directiveType.equals("data") ) {
						accPrivateSymbols = (Set<Symbol>)iAnnot.get("accprivate");
						if( accPrivateSymbols == null ) {
							accPrivateSymbols = new HashSet<Symbol>();
							iAnnot.put("accprivate", accPrivateSymbols);
						}
						accFirstPrivateSymbols = (Set<Symbol>)iAnnot.get("accfirstprivate");
						if( accFirstPrivateSymbols == null ) {
							accFirstPrivateSymbols = new HashSet<Symbol>();
							iAnnot.put("accfirstprivate", accFirstPrivateSymbols);
						}
						accReductionSymbols = (Set<Symbol>)iAnnot.get("accreduction");
						if( accReductionSymbols == null ) {
							accReductionSymbols = new HashSet<Symbol>();
							iAnnot.put("accreduction", accReductionSymbols);
						}
					}
				}
				//Handle reduction clause first to check duplicate variables in both reduction clause and other dataclauses.
				if( dAnnot.containsKey("reduction") ) {
					Object val = dAnnot.get("reduction");
					if( (!directiveType.equals("data")) && (val instanceof Map) ) {
						try { 
							Map valMap = (Map)val;
							for( ReductionOperator op : (Set<ReductionOperator>)valMap.keySet() ) {
								Set<SubArray> valSet = (Set<SubArray>)valMap.get(op); 
								Set<Symbol> symDSet = null;
								symDSet = AnalysisTools.subarraysToSymbols(valSet, IRSymbolOnly);
								if( valSet.size() != symDSet.size() ) {
									Tools.exit("[ERROR in ACCAnalysis.declareDirectiveAnalysis()]: cannot find symbols for " +
											"subarrays of key," + "reduction" + ", in ACCAnnotation, " + dAnnot + AnalysisTools.getEnclosingAnnotationContext(dAnnot));
								} else {
									accReductionSymbols.addAll(symDSet);
								}
							}
						} catch( Exception e ) {
							Tools.exit("[ERROR in ACCAnalysis.declareDirectiveAnalysis()]: <ReductionOperator, Set<SubArray>> type " +
									"is expected for the value of key," + "reduction" + " in ACCAnnotation, " + dAnnot + AnalysisTools.getEnclosingAnnotationContext(dAnnot));
						}
					}
				}
				//If the compute region contains any gang reductions in it's body, include those in the accReductionSymbols set.
				if( !directiveType.equals("data") ) {
					List<ACCAnnotation> loopAnnotList = IRTools.collectPragmas(at, ACCAnnotation.class, "loop");
					if( (loopAnnotList != null) && (!loopAnnotList.isEmpty()) ) {
						for( ACCAnnotation lAnnot : loopAnnotList ) {
							if( (lAnnot.containsKey("reduction")) && (lAnnot.containsKey("gang")) ) {
								Object val = lAnnot.get("reduction");
								try { 
									Map valMap = (Map)val;
									for( ReductionOperator op : (Set<ReductionOperator>)valMap.keySet() ) {
										Set<SubArray> valSet = (Set<SubArray>)valMap.get(op); 
										Set<Symbol> symDSet = null;
										symDSet = AnalysisTools.subarraysToSymbols(valSet, IRSymbolOnly);
										if( valSet.size() != symDSet.size() ) {
											Tools.exit("[ERROR in ACCAnalysis.declareDirectiveAnalysis()]: cannot find symbols for " +
													"subarrays of key," + "reduction" + ", in ACCAnnotation, " + lAnnot + AnalysisTools.getEnclosingAnnotationContext(lAnnot));
										} else {
											accReductionSymbols.addAll(symDSet);
										}
									}
								} catch( Exception e ) {
									Tools.exit("[ERROR 2 in ACCAnalysis.declareDirectiveAnalysis()]: <ReductionOperator, Set<SubArray>> type " +
											"is expected for the value of key," + "reduction" + " in ACCAnnotation, " + lAnnot + AnalysisTools.getEnclosingAnnotationContext(lAnnot));
								}

							}
						}
					}
				}
				
				//Put symbols in each dataclause into the internal symbol set.
				for( String aKey : dAnnot.keySet() ) {
					Object val = dAnnot.get(aKey);
					if( val instanceof Set ) {
                        if(aKey.compareTo("tile") == 0)
                            continue;
						try {
							Set<SubArray> dataSet = (Set<SubArray>)val;
							Set<Symbol> symDSet = null;
							symDSet = AnalysisTools.subarraysToSymbols(dataSet, IRSymbolOnly);
							if( dataSet.size() != symDSet.size() ) {
								Tools.exit("[ERROR in ACCAnalysis.declareDirectiveAnalysis()]: cannot find symbols for " +
										"subarrays of key," + aKey + ", in ACCAnnotation, " + dAnnot + AnalysisTools.getEnclosingAnnotationContext(dAnnot));
							} else {
								if( !directiveType.equals("data") && ACCAnnotation.privateClauses.contains(aKey) ) {
									if( aKey.equals("firstprivate") ) {
										accFirstPrivateSymbols.addAll(symDSet);
										//accPrivateSymbols.addAll(symDSet);
									} else {
										accPrivateSymbols.addAll(symDSet);
									}
								//} else if( !ACCAnnotation.pipeClauses.contains(aKey) ){
								} else {
									Set<Symbol> removeSet = new HashSet<Symbol>();
									if( (accReductionSymbols != null) && (!accReductionSymbols.isEmpty()) ) {
										for( Symbol ttSym : symDSet ) {
											if( accReductionSymbols.contains(ttSym) ) {
												SubArray ttSubA = AnalysisTools.subarrayOfSymbol(dataSet, ttSym);
												if( ttSubA != null ) {
													removeSet.add(ttSym);
													dataSet.remove(ttSubA);
												}
											}
										}
									}
									symDSet.removeAll(removeSet);
									accSharedSymbols.addAll(symDSet);
								}
							}
						} catch (Exception e) {
							Tools.exit("[ERROR in ACCAnalysis.declareDirectiveAnalysis()]: Set<SubArray> type is expected " +
									"for the value of key," + aKey + " in ACCAnnotation, " + dAnnot + AnalysisTools.getEnclosingAnnotationContext(dAnnot));
						}
					}
				}
			}
		}
		//Step3: For each declare directive, if it belongs to a procedure, put it into implicit data region for the procedure.
		//       Else, put it into implicit data region for program.
		//Visit each data directive and fill associated implicit data region.
		dataAnnots = IRTools.collectPragmas(program, ACCAnnotation.class, "declare");
		if( dataAnnots != null ) {
			for( ACCAnnotation dAnnot : dataAnnots ) {
				Annotatable at = dAnnot.getAnnotatable();
				//Create an internal annotation for this declare directive, and add addshared clause.
				Set<Symbol> decSharedSymbols = null;
				ACCAnnotation idAnnot;
				idAnnot = at.getAnnotation(ACCAnnotation.class, "internal");
					if( idAnnot == null ) {
						idAnnot = new ACCAnnotation("internal", "_directive");
						idAnnot.setSkipPrint(true);
						decSharedSymbols = new HashSet<Symbol>();
						idAnnot.put("accshared", decSharedSymbols);
						//idAnnot.put("accglobal", new HashSet<String>());
						//idAnnot.setSkipPrint(true); //This is an internal annotation.
						at.annotate(idAnnot);
					} else {
						decSharedSymbols = (Set<Symbol>)idAnnot.get("accshared");
					}
				Traversable t = (Traversable)at;
				Procedure tProc = null;
				while ( t != null ) {
					if( t instanceof Procedure ) {
						tProc = (Procedure)t;
						break;
					}
					t = t.getParent();
				}
				ACCAnnotation pAnnot;
				ACCAnnotation iAnnot;
				ARCAnnotation cAnnot;
				Set<Symbol> accSharedSymbols = null;
				String implicitDataRegion = null;
				if( tProc != null ) { //current declare directive is associated to a function.
					pAnnot = tProc.getAnnotation(ACCAnnotation.class, "data");
					if( pAnnot == null ) {
						pAnnot = new ACCAnnotation("data", "_directive");
						pAnnot.setSkipPrint(true); //This is an internal annotation.
						tProc.annotate(pAnnot);
					}
					iAnnot = tProc.getAnnotation(ACCAnnotation.class, "internal");
					if( iAnnot == null ) {
						iAnnot = new ACCAnnotation("internal", "_directive");
						accSharedSymbols = new HashSet<Symbol>();
						iAnnot.put("accshared", accSharedSymbols);
						iAnnot.put("accglobal", new HashSet<String>()); //Procedure-level implicit data region does not have any global variable.
						iAnnot.setSkipPrint(true); //This is an internal annotation.
						tProc.annotate(iAnnot);
					} else {
						accSharedSymbols = (Set<Symbol>)iAnnot.get("accshared");
					}
					cAnnot = tProc.getAnnotation(ARCAnnotation.class, "cuda");
					if( cAnnot == null ) {
						cAnnot = new ARCAnnotation("cuda", "_directive");
						cAnnot.setSkipPrint(true); //This is an internal annotation.
						tProc.annotate(cAnnot);
					}
					implicitDataRegion = tProc.getSymbolName();
				} else { //current declare directive is associated to program
					pAnnot = programAnnot;
					iAnnot = programAnnot;
					cAnnot = programCudaAnnot;
					accSharedSymbols = tProgAccSharedSymbols;
					implicitDataRegion = "program";
				}
				//Put symbols in each declare dataclauses into the internal symbol set.
				//Perform check1: a variable may appear at most once in all the clauses of declare directives
				//                for a function or program.
				for( String aKey : dAnnot.keySet() ) {
					Object val = dAnnot.get(aKey);
					if( val instanceof Set ) {
						try {
							Set<SubArray> dataSet = (Set<SubArray>)val;
							Set<SubArray> tDataSet = null;
/*							if( pAnnot == programAnnot ) {
								programAnnot.put(aKey, new HashSet<SubArray>(dataSet));
							}*/
							if( pAnnot.containsKey(aKey) ) {
								tDataSet = (Set<SubArray>)pAnnot.get(aKey);
							} else {
								tDataSet = new HashSet<SubArray>();
								pAnnot.put(aKey, tDataSet);
							}
							for( SubArray subArr : dataSet ) {
								Symbol sym = AnalysisTools.subarrayToSymbol(subArr, IRSymbolOnly);
								if( sym == null ) {
									Tools.exit("[ERROR in ACCAnalysis.declareDirectiveAnalysis()]: cannot find the symbol for the subarray, " +
											subArr + ", of key," + aKey + " in ACCAnnotation, " + dAnnot + AnalysisTools.getEnclosingAnnotationContext(dAnnot));
								}
								decSharedSymbols.add(sym);
								if( accSharedSymbols.contains(sym) ) {
									Tools.exit("[ERROR in ACCAnalysis.declareDirectiveAnalysis()]: a variable, " + sym.getSymbolName() +
											", appears more than once in the data clauses for  the same implicit data region:\n" + 
											"Implicit data region: " + implicitDataRegion + "\nDeclare directive: " + dAnnot + AnalysisTools.getEnclosingAnnotationContext(dAnnot));
								//} else if( !ACCAnnotation.pipeClauses.contains(aKey) ){
								} else {
									accSharedSymbols.add(sym);
									tDataSet.add(subArr);
								}
							}
						} catch (Exception e) {
							Tools.exit("[ERROR in ACCAnalysis.declareDirectiveAnalysis()]: Set<SubArray> type is expected for the value" +
									" of key," + aKey + " in ACCAnnotation, " + dAnnot + AnalysisTools.getEnclosingAnnotationContext(dAnnot));
						}
					}
				}
				for( String cudaDclause : ARCAnnotation.cudaDataClauses ) {
					ARCAnnotation cudaAnnot = at.getAnnotation(ARCAnnotation.class, cudaDclause);
					if( cudaAnnot != null ) {
						Set<SubArray> cudaDSet = cudaAnnot.get(cudaDclause);
						Set<SubArray> cudaSet = cAnnot.get(cudaDclause);
						if( cudaSet == null ) {
							cudaSet = new HashSet<SubArray>(cudaDSet);
							cAnnot.put(cudaDclause, cudaSet);
						} else {
							cudaSet.addAll(cudaDSet);
						}
					}
				}
			}
		}
		//Replace extern symbol to the original one, if existing.
		if( !tProgAccSharedSymbols.isEmpty() ) {
			for( Symbol gSym : tProgAccSharedSymbols ) {
				Symbol oSym = AnalysisTools.getOrgSymbolOfExternOne(gSym, program);
				if( oSym == null ) {
					progAccSharedSymbols.add(gSym);
				} else {
					progAccSharedSymbols.add(oSym);
				}
				progAccSharedSymbolStrings.add(gSym.getSymbolName());
			}
		}
		//Check2 is done during ACCAnnotation parsing.
		//Step4: Perform check3: If a variable or array appears in a declare directive, the same variable or 
		//       array may not appear in a data clause for any construct where the declaration of the variable 
		//       is visible.
		List<Procedure> procList = IRTools.getProcedureList(program);
		if( procList != null ) {
			for( Procedure cProc : procList ) {
				Set<Symbol> dataSymbols = new HashSet<Symbol>();
				//Check whether variable in dataclauses included in the current procedure appears in the program-level
				//implicit data region.
				dataAnnots = AnalysisTools.ipCollectPragmas(cProc.getBody(), ACCAnnotation.class, ACCAnnotation.dataRegions, false, null);
				if( dataAnnots != null ) {
					for( ACCAnnotation dAnnot : dataAnnots ) {
						Annotatable at = dAnnot.getAnnotatable();
						ACCAnnotation iAnnot = at.getAnnotation(ACCAnnotation.class, "internal");
						if( iAnnot != null ) {
							Object val = iAnnot.get("accshared");
							if( (val != null) && (val instanceof Set) ) {
								Set<String> accGlobalSet = null;
								Object gVal = iAnnot.get("accglobal");
								if( gVal != null ) {
									accGlobalSet = (Set<String>)gVal;
								}
								for( Symbol tSym : (Set<Symbol>)val ) {
									List symbolInfo = new ArrayList(2);
									if( AnalysisTools.SymbolStatus.OrgSymbolFound(
											AnalysisTools.findOrgSymbol(tSym, at, true, null, symbolInfo, gFuncCallList)) ) {
										Symbol tOSym = (Symbol)symbolInfo.get(0);
										dataSymbols.add(tOSym);
										if( SymbolTools.isGlobal(tOSym) && (accGlobalSet != null) ) {
											accGlobalSet.add(tOSym.getSymbolName());
										}
									} else {
										dataSymbols.add(tSym);
									}
								}
							}
						}
					}
				}
				if( !progAccSharedSymbols.isEmpty() ) {
					for( Symbol sym : dataSymbols ) {
						if( SymbolTools.isGlobal(sym) ) {
							if( progAccSharedSymbolStrings.contains(sym.getSymbolName()) ) {
								PrintTools.println("[WARNING in ACCAnalysis.declareDirectiveAnalysis()]: a variable, " + sym.getSymbolName() +
										", appears in data clauses for both program-level implicit data region" + 
										" and data region or compute region in a procedure, " + cProc.getSymbolName() + 
										"\nEnclosing Translatin Unit: " + ((TranslationUnit)cProc.getParent()).getOutputFilename() + "\n", 1);
							}
						}
					}
				}
				//Check if this procedure has an internal data region; if not, skip this procedure.
				Annotation dAnnot = cProc.getAnnotation(ACCAnnotation.class, "data"); 
				Set<Symbol> procDataSymbols = null;
				if( dAnnot == null ) {
					continue;
				} else {
					//Check whether variable in dataclauses included in the current procedure appears in the procedure-level
					//implicit data region.
					Annotation iAnnot = cProc.getAnnotation(ACCAnnotation.class, "internal");
					if( iAnnot != null ) {
						Object val = iAnnot.get("accshared");
						if( (val !=null) && (val instanceof Set) ) {
							procDataSymbols = (Set<Symbol>)val;
							for( Symbol sym : dataSymbols ) {
								if( procDataSymbols.contains(sym) ) {
									Tools.exit("[ERROR in ACCAnalysis.declareDirectiveAnalysis()]: a variable, " + sym.getSymbolName() +
											", appears in data clauses for both procedure-level implicit data region" + 
											" and data region or compute region in the same procedure, " + cProc.getSymbolName() + 
										"\nEnclosing Translatin Unit: " + ((TranslationUnit)cProc.getParent()).getOutputFilename() + "\n");
								}
							}
						}
					}
				}
			}
		}
		PrintTools.println("[ACCAnalysis.declareDirectiveAnalysis()] end", 2);
	}
	
	/**
	 * Update IDExressions in ACCAnnotation clauses with Identifiers that have links to 
	 * corresponding Symbols and update old symbols in a set with new ones.
	 * 
	 */
	public static void updateSymbolsInACCAnnotations(Traversable inputIR, Map<String, String> nameChangeMap)
	{
		DFIterator<Annotatable> iter = new DFIterator<Annotatable>(inputIR, Annotatable.class);
		while(iter.hasNext())
		{
			Annotatable at = iter.next();
			List<PragmaAnnotation> ACCAnnotList = new LinkedList<PragmaAnnotation>();
			List<PragmaAnnotation> PragmaAnnotList = at.getAnnotations(PragmaAnnotation.class);
			if( (PragmaAnnotList == null) || (PragmaAnnotList.isEmpty()) ) {
				continue;
			} else {
				for( PragmaAnnotation pragAnnot : PragmaAnnotList ) {
					if( pragAnnot instanceof ACCAnnotation ) {
						ACCAnnotList.add(pragAnnot);
					} else if( pragAnnot instanceof ARCAnnotation ) {
						ACCAnnotList.add(pragAnnot);
					} else if( pragAnnot instanceof NVLAnnotation ) {
						ACCAnnotList.add(pragAnnot);
					}
				}
			}
			if ( ACCAnnotList.size() > 0 ) 
			{
				Traversable inputTR = at;
				if( at instanceof Procedure ) {
					inputTR = ((Procedure)at).getBody();
				}
				for( PragmaAnnotation annot : ACCAnnotList ) {
					for( String key: annot.keySet() ) {
						Object value = annot.get(key);
						if( value instanceof Expression ) {
							Expression newExp = updateSymbolsInExpression((Expression)value, inputTR, nameChangeMap, annot);
							if( newExp != null ) {
								annot.put(key, newExp);
							}
						} else if( value instanceof List ) {
							//transform permute and window clauses have argument lists.
							//[CAUTION] this assumes that arguments are either subarray or expression.
							List vList = (List)value;
							List newList = new LinkedList<Object>();
							for( Object elm : vList ) {
								if( elm instanceof Expression ) {
									Expression newExp = updateSymbolsInExpression((Expression)elm, inputTR, nameChangeMap, annot);
									if( newExp == null ) {
										newList.add(elm);
									} else {
										newList.add(newExp);
									}
								} else if( elm instanceof SubArray ) {
									updateSymbolsInSubArray((SubArray)elm, inputTR, nameChangeMap, annot);
									newList.add(elm);
								} else {
									Tools.exit("[ERROR in ACCAnalysis.updateSymbolsInACCAnnotations()]: The value of key, " + key +
											", in the  following ACCAnnotation should be either expression or SubArray.\n" + 
											"ACCAnnotation: " + annot + 
											AnalysisTools.getEnclosingAnnotationContext(annot));
								}
							}
							annot.put(key, newList);
						} else if( value instanceof Set ) {
							Set vSet = (Set)value;
							String elmType = "subarray";
							for( Object elm : vSet ) {
								if( elm instanceof SubArray ) {
									updateSymbolsInSubArray((SubArray)elm, inputTR, nameChangeMap, annot);
								} else if( elm instanceof SubArrayWithConf ) {
									elmType = "subarrayconf";
									SubArrayWithConf sArrayConf = (SubArrayWithConf)elm;
									SubArray sArray = sArrayConf.getSubArray();
									updateSymbolsInSubArray(sArray, inputTR, nameChangeMap, annot);
									List<ConfList> listOfConfList = sArrayConf.getListOfConfList();
									if( listOfConfList != null ) {
										for( ConfList tConfList : listOfConfList ) {
											List<Expression> tList = tConfList.getConfigList();
											List<Expression> newList = new LinkedList<Expression>();
											for( Expression tExp : tList ) {
												Expression newExp = updateSymbolsInExpression(tExp, inputTR, nameChangeMap, annot);
												if( newExp == null ) {
													newList.add(tExp);
												} else {
													newList.add(newExp);
												}
											}
											tConfList.setConfigList(newList);
										}
									}
								} else if( elm instanceof Expression ) {
									elmType = "exprssion";
									break;
								} else if( elm instanceof Symbol ) {
									elmType = "symbol";
									break;
								} else if( elm instanceof String ) {
									elmType = "string";
									break;
								} else {
									Tools.exit("[ERROR in ACCAnalysis.updateSymbolsInACCAnnotations()]: Set<SubArray>, Set<Expression>, " +
											"Set<String>, Set<Symbol>, or Set<SubArrayWithConf> type is expected " +
											"for the value of key, " + key + ", in ACCAnnotation, " + annot + 
											AnalysisTools.getEnclosingAnnotationContext(annot));
								}
							}
							if( elmType.equals("symbol") ) {
								Set<Symbol> new_set = new HashSet<Symbol>();
								updateSymbols(inputTR, (Set<Symbol>)vSet, new_set, nameChangeMap);
								annot.put(key, new_set); //update symbol set in the annotation
							} else if( elmType.equals("string") ) {
								Set<String> new_set = new HashSet<String>();
								new_set.addAll(vSet);
								annot.put(key, new_set); //update symbol set in the annotation
							} else if( elmType.equals("expression") ) {
								Set newSet = new HashSet<Expression>();
								for( Object elm : vSet ) {
									Expression newExp = updateSymbolsInExpression((Expression)elm, inputTR, nameChangeMap, annot);
									if( newExp == null ) {
										newSet.add(elm);
									} else {
										newSet.add(newExp);
									}
								}
								annot.put(key, newSet);
							}
						} else if( value instanceof Map ) {
							try { 
								Map valMap = (Map)value;
								for( ReductionOperator op : (Set<ReductionOperator>)valMap.keySet() ) {
									Set<SubArray> valSet = (Set<SubArray>)valMap.get(op); 
									for( SubArray sArray : valSet ) {
										updateSymbolsInSubArray(sArray, inputTR, nameChangeMap, annot);
									}
								}
							} catch( Exception e ) {
								Tools.exit("[ERROR in ACCAnalysis.updateSymbolsInACCAnnotations()]: <String, Set<SubArray>> type is expected for the value" +
										" of key," + key + " in ACCAnnotation, " + annot + AnalysisTools.getEnclosingAnnotationContext(annot));
							}
						} else if( !(value instanceof String) ) {
							Tools.exit("[ERROR in ACCAnalysis] Unexpected value type for a key, "+ key + 
									" in  ACCAnnotation, " + annot + AnalysisTools.getEnclosingAnnotationContext(annot));
						}
					}
				}
			}
		}
	}

	/**
		* shared_analysis 
		* For each compute region
		*    - Move accshared set into dataSet, which contains symbols existing in data clauses of the region.
		*    - Find shared symbols interprocedurally, and put them in the accshared set.
		*    - Find enclosing explicit/implicit data regions
		*    - For each symbol that is in accshared set, but not in dataSet
		*        - If it is not included in accshared set in any enclosing explicit/implicit data region 
		*        	- If it is scalar variable in a parallel region, put it into firstprivate.
		*           - Else, put it into pcopy set of the region.
		*        - Else, put it into present set of the region.
		*            
		* CAVEAT: current analysis compares IR symbols if PseudoSymbols exist. To allow comparison of PseudoSymbols,
		* {@code declareDirectiveAnalysis()} and other analysis passes should be modified too.
		*/
	private void shared_analysis(boolean IRSymbolOnly)
	{
		PrintTools.println("[ACCAnalysis.shared_analysis()] begin", 2);

		List<FunctionCall> gFuncCallList = IRTools.getFunctionCalls(program);
		List<ACCAnnotation>  cRegionAnnots = AnalysisTools.collectPragmas(program, ACCAnnotation.class, ACCAnnotation.computeRegions, false);
		if( cRegionAnnots != null ) {
			for( ACCAnnotation cAnnot : cRegionAnnots ) {
				Annotatable at = cAnnot.getAnnotatable();
				Annotation iAnnot = at.getAnnotation(ACCAnnotation.class, "accshared");
				Procedure parentProc = IRTools.getParentProcedure(at);
				TranslationUnit parentTu = IRTools.getParentTranslationUnit(at);
				Set<Symbol> accSharedSymbols = null;
				Set<Symbol> accPrivateSymbols = null;
				Set<Symbol> accFirstPrivateSymbols = null;
				Set<Symbol> accReductionSymbols = null;
				Set<Symbol> dataSet = new HashSet<Symbol>();
				Set<Symbol> newPrivSymbols = new HashSet<Symbol>();
				boolean privateClauseAllowed = false;
				boolean isParallelConstruct = false;
				if( at.containsAnnotation(ACCAnnotation.class, "parallel") ) {
					privateClauseAllowed = true;
					isParallelConstruct = true;
				} else if( at.containsAnnotation(ACCAnnotation.class, "loop") ) {
					privateClauseAllowed = true;
				}
				if( iAnnot != null ) {
					accSharedSymbols = (Set<Symbol>)iAnnot.get("accshared");
					if( accSharedSymbols != null ) {
						//Step1: move accshared set into dataSet.
						dataSet.addAll(accSharedSymbols);
						//Step1-2: store shared variables explicitly specified by users into accexplicitshared set
						Set<Symbol> exSharedSet = iAnnot.get("accexplicitshared");
						if( exSharedSet == null ) {
							exSharedSet = new HashSet<Symbol>();
							iAnnot.put("accexplicitshared", exSharedSet);
						}
						exSharedSet.addAll(accSharedSymbols);
						accSharedSymbols.clear();
						accPrivateSymbols = (Set<Symbol>)iAnnot.get("accprivate");
						if(accPrivateSymbols == null)	
							accPrivateSymbols = new HashSet<Symbol>();
						accFirstPrivateSymbols = (Set<Symbol>)iAnnot.get("accfirstprivate");
						if(accFirstPrivateSymbols == null)	
							accFirstPrivateSymbols = new HashSet<Symbol>();
						accReductionSymbols = (Set<Symbol>)iAnnot.get("accreduction");
						if(accReductionSymbols == null)	
							accReductionSymbols = new HashSet<Symbol>();
						//Step2: find shared symbols interprocedurally, and put them in the accshared set.
						// accshared set = symbols accessed in the region - local symbols 
						//                 + global symbols accessed in functions called in the region 
						//                 - OpenACC private set
						//                 - Loop index variables for Gang/Worker/Vector loops.
						//Step2-1: find symbols accessed in the region, and add them to accshared set.
						Set<Symbol> tempSet = AnalysisTools.getAccessedVariables(at, IRSymbolOnly);
						if( tempSet != null ) {
							accSharedSymbols.addAll(tempSet);
						}
						//Step2-2: find local symbols defined in the region, and remove them from the accshared set.
						tempSet = SymbolTools.getLocalSymbols(at);
						if( tempSet != null ) {
							accSharedSymbols.removeAll(tempSet);
						}
						//Step2-3: find global symbols accessed in the functions called in the region, and add them 
						//to the accshared set.
						Map<String, Symbol> gSymMap = null;
						List<FunctionCall> calledFuncs = IRTools.getFunctionCalls(at);
						for( FunctionCall call : calledFuncs ) {
							Procedure called_procedure = call.getProcedure();
							if( called_procedure != null ) {
								if( gSymMap == null ) {
									Set<Symbol> tSet = SymbolTools.getGlobalSymbols(at);
									gSymMap = new HashMap<String, Symbol>();
									for( Symbol gS : tSet ) {
										if( !(gS.getDeclaration() instanceof Enumeration) ) {
											//Skip member symbols of an enumeration.
											gSymMap.put(gS.getSymbolName(), gS);
										}
									}
								} 
								CompoundStatement body = called_procedure.getBody();
								Set<Symbol> procAccessedSymbols = AnalysisTools.getIpAccessedGlobalSymbols(body, gSymMap, null);
								for( Symbol tgSym : procAccessedSymbols ) {
									if( !gSymMap.containsValue(tgSym) ) {
										//tgSym is not visible in the current scope.
										Declaration tgSymDecl = tgSym.getDeclaration();
										if( (tgSymDecl instanceof VariableDeclaration) && !(tgSym instanceof ProcedureDeclarator) ) {
											VariableDeclaration newDecl = TransformTools.addExternVariableDeclaration((VariableDeclaration)tgSymDecl, parentTu);
											Declarator newExtSym = newDecl.getDeclarator(0);
											if( newExtSym instanceof Symbol ) {
												gSymMap.put(((Symbol) newExtSym).getSymbolName(), (Symbol)newExtSym);
											}
										} else {
											PrintTools.println("\n[WARNING] an unexpected type of global symbol (" + tgSym.getSymbolName() + ") is" +
													" accessed by a function called in the following compute region:\n" +
													cAnnot + AnalysisTools.getEnclosingContext(at), 0);
										}
									}
								}
								accSharedSymbols.addAll(procAccessedSymbols);
							}
						}
						//Step2-4: remove index variables used for worksharing loops (gang/worker/vector loops) from 
						//the accshared set.
						Set<Symbol> loopIndexSymbols = AnalysisTools.getWorkSharingLoopIndexVarSet(at);
						accSharedSymbols.removeAll(loopIndexSymbols);
						//Step2-5: if loop index variable is not local and not included in the private clause, add it the set.
						//         (private clause exists only in parallel regions or in parallel/kernels loops.)
						if( privateClauseAllowed ) {
							for(Symbol IndexSym : loopIndexSymbols ) {
								if( tempSet.contains(IndexSym) ) {
									continue; //loop index variable is local to the compute region.
								} else if(!accPrivateSymbols.contains(IndexSym)) {
									newPrivSymbols.add(IndexSym);
								}
							}
						}
						//Step3: Check whether variables in data clauses are not accessed in the current region.
						for( Symbol tSym : dataSet ) {
							if( !accSharedSymbols.contains(tSym) ) {
								PrintTools.println("\n[WARNING in ACCAnalysis.shared_analysis()] a variable, " + tSym.getSymbolName() + 
										", appears in a data clause of the following compute region, but it is not accessed in the" +
										" region. To reduce redandunt transfers, remove the variable from its data clause.\n" +
										"Compute region: " + cAnnot + "\nEnclosing Procedure: " + parentProc.getSymbolName() + "\n", 0);
							}
						}

						//Step4: find enclosing explicit/implicit data regions
						LinkedList<ACCAnnotation> enclosingDataRegions = new LinkedList<ACCAnnotation>();
						ACCAnnotation drAnnot = AnalysisTools.ipFindFirstPragmaInParent(at, ACCAnnotation.class, "data", gFuncCallList, null);
						while( drAnnot != null ) {
							enclosingDataRegions.add(drAnnot);
							drAnnot = AnalysisTools.ipFindFirstPragmaInParent(drAnnot.getAnnotatable(), ACCAnnotation.class, "data", gFuncCallList, null);
						}
						//Step5: for each symbol that is in accshared set, but not in dataSet
						//       - If it is not included in accshared set in any enclosing explicit/implicit data region 
						//           - put it into pcopy/copy set of the region.
						//       - Otherwise, put it into present/deviceptr set of the region.
						Set<Symbol> DSharedSymbols = new HashSet<Symbol>();
						Map<Procedure, List<Set<Symbol>>> dRegionMap = new HashMap<Procedure, List<Set<Symbol>>>();
						Map<Procedure, List<Set<String>>> dRegionGMap = new HashMap<Procedure, List<Set<String>>>();
						Map<Procedure, List<ACCAnnotation>> dRegionAnnotMap = new HashMap<Procedure, List<ACCAnnotation>>();
						while( !enclosingDataRegions.isEmpty() ) {
							drAnnot = enclosingDataRegions.removeFirst();
							Annotatable dAt = drAnnot.getAnnotatable();
							Procedure dPproc = IRTools.getParentProcedure(dAt);
							ACCAnnotation idAnnot = dAt.getAnnotation(ACCAnnotation.class, "accshared");
							if( (dPproc !=null) && (idAnnot != null) ) {
								Set<Symbol> idSymSet = (Set<Symbol>)idAnnot.get("accshared");
								Set<String> idGSymSSet = null;
								if( idAnnot.containsKey("accglobal") ) {
									idGSymSSet = (Set<String>)idAnnot.get("accglobal");
								} else {
									idGSymSSet = new HashSet<String>();
									idAnnot.put("accglobal", idGSymSSet);
								}
								List<Set<Symbol>> accSharedSetList = null;
								if( dRegionMap.containsKey(dPproc) ) {
									accSharedSetList = (List<Set<Symbol>>)dRegionMap.get(dPproc);
								} else {
									accSharedSetList = new LinkedList<Set<Symbol>>();
									dRegionMap.put(dPproc, accSharedSetList);
								}
								accSharedSetList.add(idSymSet);
								List<Set<String>> accGlobalSetList = null;
								if( dRegionGMap.containsKey(dPproc) ) {
									accGlobalSetList = (List<Set<String>>)dRegionGMap.get(dPproc);
								} else {
									accGlobalSetList = new LinkedList<Set<String>>();
									dRegionGMap.put(dPproc, accGlobalSetList);
								}
								accGlobalSetList.add(idGSymSSet);
								List<ACCAnnotation> accDAnnotList = null;
								if( dRegionAnnotMap.containsKey(dPproc) ) {
									accDAnnotList = (List<ACCAnnotation>)dRegionAnnotMap.get(dPproc);
								} else {
									accDAnnotList = new LinkedList<ACCAnnotation>();
									dRegionAnnotMap.put(dPproc, accDAnnotList);
								}
								accDAnnotList.add(drAnnot);
								
							} else {
								Tools.exit("[ERROR] Internal error in ACCAnalysis.shared_analysis(): internal data, accshared set, is missing!\n" +
										"Procedure containing a data region: " + dPproc.getSymbolName() + "\nEnclosing Translation Unit: " + 
										((TranslationUnit)dPproc.getParent()).getOutputFilename() + "\n");
							}
						}
						accSharedSymbols.removeAll(accPrivateSymbols);
						accSharedSymbols.removeAll(accFirstPrivateSymbols);
						accSharedSymbols.removeAll(accReductionSymbols);
						for( Symbol dSym : accSharedSymbols ) {
							boolean isIncluded = false;
							Symbol tdGSym = null;
							List osymList = null;
							if( dataSet.contains(dSym) ) { 
								//dSym is included in the data clauses of the current compute region.
								continue;
							} else if( accPrivateSymbols.contains(dSym) || accFirstPrivateSymbols.contains(dSym) ||
									accReductionSymbols.contains(dSym) ) {
								//dSym is included in the private/firstprivate/reduction clauses of the 
								//current compute region.
								continue;
							}
							//Step5-1: check procedure-level implicit data regions and explicit data regions.
							SubArray enSubArray = null;
							String tDataClause = null;
							List retList = null;
							for( Procedure ttproc : dRegionMap.keySet() ) {
								List<Set<Symbol>> accSharedSetList = dRegionMap.get(ttproc);
								List<Set<String>> accGlobalSetList = dRegionGMap.get(ttproc);
								List<ACCAnnotation> accDAnnotList = dRegionAnnotMap.get(ttproc);
								int i = 0;
								for( Set<Symbol> dSymSet : accSharedSetList) {
									Set<String> dGSymSSet = accGlobalSetList.get(i);
									ACCAnnotation tDAnnot = accDAnnotList.get(i);
									if( dSymSet == null ) {
										continue;
									}
									if( dSymSet.contains(dSym) ) {
										//dSym is visible in a procedure, ttproc, and included in the data clauses of the data region in the procedure.
										isIncluded = true;
										//enSubArray = AnalysisTools.findSubArrayInDataClauses(tDAnnot, dSym, IRSymbolOnly);
										retList = AnalysisTools.findClauseNSubArrayInDataClauses(tDAnnot, dSym, IRSymbolOnly);
										if( (retList != null) && (retList.size()==2) ) {
											tDataClause = (String)retList.get(0);
											enSubArray = (SubArray)retList.get(1);
										}
										break;
									} else {
										//dSym is invisible in a procedure, ttproc.
										//find a symbol visible in the ttproc, if dSym is a parameter symbol.
										osymList = new ArrayList(2);
										if( AnalysisTools.SymbolStatus.OrgSymbolFound(
												AnalysisTools.findOrgSymbol(dSym, at, false, ttproc, osymList, gFuncCallList)) ) {
											Symbol odSym = (Symbol)osymList.get(0);
											if( dSymSet.contains(odSym) ) {
												isIncluded = true;
												//enSubArray = AnalysisTools.findSubArrayInDataClauses(tDAnnot, odSym, IRSymbolOnly);
												retList = AnalysisTools.findClauseNSubArrayInDataClauses(tDAnnot, odSym, IRSymbolOnly);
												if( (retList != null) && (retList.size()==2) ) {
													tDataClause = (String)retList.get(0);
													enSubArray = (SubArray)retList.get(1);
												}
												break;
											} else if( SymbolTools.isGlobal(odSym) ){ 
												//Remaining cases: 1) odSym is global and a symbol in dSymSet is also global, but they refer to
												//                    different extern symbols for the same symbol.
												//                 2) odSym is global and a symbol in dSymSet is a parameter symbol, but they
												//                    refer to the same global symbol.
												if( dGSymSSet.contains(odSym.getSymbolName()) ) {
													isIncluded = true;
													//enSubArray = 
													//		AnalysisTools.findSubArrayInDataClauses(tDAnnot, odSym.getSymbolName(), IRSymbolOnly);
													retList = AnalysisTools.findClauseNSubArrayInDataClauses(tDAnnot, odSym.getSymbolName(), IRSymbolOnly);
													if( (retList != null) && (retList.size()==2) ) {
														tDataClause = (String)retList.get(0);
														enSubArray = (SubArray)retList.get(1);
													}
													break;
												}
											}
										}
									}
									i++;
								}
								if( isIncluded ) {
									break;
								}
							}
							//Step5-2: check program-level implicit data region.
							if( !isIncluded ) {
								Set<String> dGSymSSet = programAnnot.get("accglobal");
								osymList = new ArrayList(2);
								if( AnalysisTools.SymbolStatus.OrgSymbolFound(
										AnalysisTools.findOrgSymbol(dSym, at, true, null, osymList, gFuncCallList)) ) {
									Symbol odSym = (Symbol)osymList.get(0);
									if( SymbolTools.isGlobal(odSym) ) {
										if( dGSymSSet.contains(odSym.getSymbolName()) ) {
											isIncluded = true;
											//enSubArray = 
											//		AnalysisTools.findSubArrayInDataClauses(programAnnot, odSym.getSymbolName(), IRSymbolOnly);
											retList = AnalysisTools.findClauseNSubArrayInDataClauses(programAnnot, odSym.getSymbolName(), IRSymbolOnly);
											if( (retList != null) && (retList.size()==2) ) {
												tDataClause = (String)retList.get(0);
												enSubArray = (SubArray)retList.get(1);
											}
										}
									}
								}
							}
							//Step5-3: add necessary data clause (present/copy/pcopy)
							Set<SubArray> targetSet = null;
							String dataClause = "present"; //symbol, dSym, is included in an enclosing data region.
							if( !isIncluded ) {
								if( SymbolTools.isScalar(dSym) && !SymbolTools.isPointer(dSym) && !(dSym instanceof NestedDeclarator)) {
									//[OpenACC V2.7 Section 2.5.1 Line 675 - 677] "A scalar variable referenced in the parallel construct 
									//that does not appear in a data clause for the construct or any enclosing data construct will be 
									//treated as if it appeared in a firstprivate clause."
									if( isParallelConstruct ) {
										dataClause = "firstprivate";
										accFirstPrivateSymbols.add(dSym);
									} else {
										if( SymbolTools.containsSpecifier(dSym, Specifier.CONST) ) {
											dataClause = "copyin";
										} else {
											dataClause = "copy";
										}
									}
								} else {
									dataClause = "pcopy";
								}
							} else if( (tDataClause != null) && tDataClause.equals("deviceptr") ) {
								dataClause = "deviceptr";
							}
							Annotation pAnnot = at.getAnnotation(ACCAnnotation.class, dataClause);
							if( pAnnot == null ) {
								targetSet = new HashSet<SubArray>();
								cAnnot.put(dataClause, targetSet);
							} else {
								targetSet = pAnnot.get(dataClause);
							}
							osymList = new ArrayList(2);
							if( AnalysisTools.SymbolStatus.OrgSymbolFound(
									AnalysisTools.findOrgSymbol(dSym, at, false, parentProc, osymList, gFuncCallList)) ) {
								Symbol odSym = (Symbol)osymList.get(0);
								SubArray newSubArray = AnalysisTools.createSubArray(odSym, true, null);
								if( enSubArray != null ) {
									if( enSubArray.getArrayDimension() > 0 ) {
										List<Expression> tStartL = enSubArray.getStartIndices();
										List<Expression> tLengthL = enSubArray.getLengths();
										newSubArray.setRange(tStartL, tLengthL);
									}
								}
								targetSet.add(newSubArray);
							} else {
								Tools.exit("[ERROR in ACCAnalysis.shared_analysis()] symbol, " + dSym.getSymbolName() +
										", is accessed in the following compute region, but not visible." + AnalysisTools.getEnclosingContext(at));
							}
						}
						accSharedSymbols.removeAll(accFirstPrivateSymbols);
						if( privateClauseAllowed && !newPrivSymbols.isEmpty() ) {
							Set<SubArray> privSet = null; 
							ACCAnnotation privAnnot = at.getAnnotation(ACCAnnotation.class, "private");
							if( privAnnot == null ) {
								privSet = new HashSet<SubArray>();
								cAnnot.put("private", privSet);
							} else {
								privSet = (Set<SubArray>)privAnnot.get("private");
							}
							for( Symbol pSym : newPrivSymbols ) {
								privSet.add(AnalysisTools.createSubArray(pSym, true, null));
								accPrivateSymbols.add(pSym);
							}
						}
					} else {
						Tools.exit("[ERROR] Internal error in ACCAnalysis.shared_analysis(): internal data, accshared set, is missing!\n" +
								"OpenACC annotation: " + cAnnot + AnalysisTools.getEnclosingAnnotationContext(cAnnot));
					}
				} else {
					Tools.exit("[ERROR] Internal error in ACCAnalysis.shared_analysis(): internal data, accshared set, is missing!\n" +
								"OpenACC annotation: " + cAnnot + AnalysisTools.getEnclosingAnnotationContext(cAnnot));
				}
			}
		}

		PrintTools.println("[ACCAnalysis.shared_analysis()] end", 2);
	}


	public static void updateSymbolsInSubArray(SubArray sArray, Traversable at, Map<String, String> nameChangeMap,
			PragmaAnnotation annot)
	{
		if( (sArray == null) || (at == null) ) return;
		Expression aName = sArray.getArrayName();
		//TODO: if aName is a class member, it should be handled too.
		if( aName instanceof IDExpression ) {
			String oldName = aName.toString();
			if( oldName.equals("*") ) {
				//We don't need to update symbols.
				return;
			}
			String newName = oldName;
			if( (nameChangeMap != null) && nameChangeMap.containsKey(oldName) ) {
				newName = nameChangeMap.get(oldName);
			}
			Symbol sym = SymbolTools.getSymbolOfName(newName, at);
			if( sym == null ) {
				Tools.exit("[ERROR in ACCAnalysis.updateSymbolsInSubArray()] variable, " + newName + 
						", in the following OpenACC annotatin is not visible in the current scope; " +
						"please check whether the declaration of the variable is visible.\n" +
						"OpenACC annotation: " + annot + "\n" + AnalysisTools.getEnclosingAnnotationContext(annot));
			}
			Identifier id = new Identifier(sym);
			sArray.setArrayName(id);
			int dimension = sArray.getArrayDimension();
			if( dimension == -1 ) { //dimension is not known.
				//DEBUG: if symbol is NestedDeclarator, it does not have all dimension information.
				if( !(sym instanceof NestedDeclarator) ) {
					boolean isArray = SymbolTools.isArray(sym);
					boolean isPointer = SymbolTools.isPointer(sym);
					//FIXME: Updating array dimension in compute-region may conflict with 
					//user-provided information in enclosing data region.
					//Disabled temporarily
					if( isArray && !isPointer ) { //array variable
						/*					List aspecs = sym.getArraySpecifiers();
					ArraySpecifier aspec = (ArraySpecifier)aspecs.get(0);
					int dimsize = aspec.getNumDimensions();
					ArrayList<Expression> startIndices = new ArrayList<Expression>(dimsize);
					ArrayList<Expression> lengths = new ArrayList<Expression>(dimsize);
					for( int i=0; i<dimsize; i++ ) {
						startIndices.add(i, new IntegerLiteral(0));
						Expression length = aspec.getDimension(i);
						if( length == null ) {
							lengths.add(i, null);
						} else {
							lengths.add(i, length.clone());
						}
					}
					sArray.setRange(startIndices, lengths);*/
					} else if( !isArray && !isPointer ) { //scalar, non-pointer variable
						List<Expression> startIndices = new LinkedList<Expression>();
						List<Expression> lengths = new LinkedList<Expression>();
						sArray.setRange(startIndices, lengths); //this will set dimension to 0.
					}
				}
			} else if( dimension > 0 ) {//update symbols in subarray bound expressions.
				for( int i=0; i<dimension; i++ ) {
					List<Expression> range = sArray.getRange(i);
					Expression newExp = updateSymbolsInExpression(range.get(0), at, nameChangeMap, annot);
					if( newExp != null ) {
						range.set(0, newExp);
					}
					newExp = updateSymbolsInExpression(range.get(1), at, nameChangeMap, annot);
					if( newExp != null ) {
						range.set(1, newExp);
					}
					sArray.setRange(i, range);
				}
			}
		} else if ( aName instanceof AccessExpression ) {
			AccessExpression accExp = (AccessExpression)aName;
			//TODO: we need methods similar to SymbolTools.searchDeclaration() to implement this section.
			Tools.exit("[ERROR in ACCAnalysis.updateSymbolsInSubArray()] the name of a subarray should be a simple variable name;" +
					"the current implementation does not support class member to be used as a subarray name.\n" +
					"Current subarray: " + sArray + AnalysisTools.getEnclosingContext(at));
		} else {
			Tools.exit("[ERROR in ACCAnalysis.updateSymbolsInSubArray()] the name of a subarray should be a simple variable name;" +
					"the current implementation does not support the following subarray name.\n" +
					"Current subarray: " + sArray + AnalysisTools.getEnclosingContext(at));
		}
	}

	/**
	 * Return the new expression if input expression {@code exp} is IDExpression; otherwise, return
	 * null
	 * 
	 * @param exp
	 * @param at
	 * @param nameChangeMap
	 * @param annot
	 * @return
	 */
	protected static Expression updateSymbolsInExpression(Expression exp, Traversable at, Map<String, String> nameChangeMap,
			PragmaAnnotation annot)
	{
		Expression newExp = null;
		if( (exp == null) || (at == null) ) return newExp;
		DFIterator<IDExpression> eiter = new DFIterator<IDExpression>(exp, IDExpression.class);
		while(eiter.hasNext())
		{
			IDExpression nid = eiter.next();
			String oldName = nid.getName();
			String newName = oldName;
			if( (nameChangeMap != null) && nameChangeMap.containsKey(oldName) ) {
				newName = nameChangeMap.get(oldName);
			}
			Symbol sym = null;
			sym = SymbolTools.getSymbolOfName(newName, at);
			if( sym == null ) {
				//OpenARC internal variables (HI_*) don't need to be updated.
				if( !newName.startsWith("HI_") && !newName.startsWith("openarc_") ) {
					Tools.exit("[ERROR in ACCAnalysis.updateSymbolsInExpression()] variable, " + newName + 
							", in the following OpenACC annotatin is not visible in the current scope; " +
							"please check whether the declaration of the variable is visible.\n" +
							"OpenACC annotation: " + annot + AnalysisTools.getEnclosingAnnotationContext(annot));
				}
			} else {
				Identifier id = new Identifier(sym);
				id.swapWith(nid);
				if( exp.equals(nid) ) {
					newExp = id;
				}
			}
		}
		return newExp;
	}
	
	/**
	 * For each symbol in the old_set, 
	 *     - find a symbol with the same name in the SymbolTable, 
	 *       and put the new symbol into the new_set.
	 *     - If no symbol is found in the table, put the old symbol into the new_set
	 *     
	 * @param t region, from which symbol search starts.
	 * @param old_set Old Symbol data set
	 * @param new_set New Symbol data set to be replaced for the old_set.
	 */
	static public void updateSymbols(Traversable t, Set<Symbol> old_set, Set<Symbol> new_set, Map<String, String> nameChangeMap)
	{
		VariableDeclaration sm_decl = null;
		Traversable tt = t;
		while( !(tt instanceof SymbolTable) ) {
			tt = tt.getParent();
		}
		///////////////////////////////////////////////////////////////////////////////////
		// CAVEAT: Below SymbolTools.findSymbol() checks symbols upwardly, and thus if   //
		// the old_set contains local symbol not visible in the current scope, below     //
		// checking will return wrong symbol; if a function has the same name as the     //
		// local static variable, the function symbol (ProcedureDeclarator) will be      //
		// returned.                                                                     //
		///////////////////////////////////////////////////////////////////////////////////
		for( Symbol sm : old_set) {
			Symbol newSym = null;
			Symbol IRSym = sm;
			if( (sm instanceof Procedure) || (sm instanceof ProcedureDeclarator) ) {
				continue;
			} else if( sm instanceof PseudoSymbol ) {
				IRSym = ((PseudoSymbol)sm).getIRSymbol();
			}
			String oldName = IRSym.getSymbolName();
			String newName = oldName;
			if( (nameChangeMap != null) && nameChangeMap.containsKey(oldName) ) {
				newName = nameChangeMap.get(oldName);
			}
			sm_decl = (VariableDeclaration)SymbolTools.findSymbol((SymbolTable)tt, 
					newName);
			if( sm_decl == null ) {
				IRSym.setName(newName);
				newSym = IRSym;
			} else {
				boolean found_sm = false;
				for( int i=0; i<sm_decl.getNumDeclarators(); i++ ) {
					Declarator tDeclr = sm_decl.getDeclarator(i);
					//Both NestedDeclarator and VariableDeclarator are symbol type.
					if( (tDeclr instanceof NestedDeclarator) || (tDeclr instanceof VariableDeclarator) ) {
						if( ((Symbol)tDeclr).getSymbolName().compareTo(IRSym.getSymbolName()) == 0 ) {
							newSym = (Symbol)tDeclr;
							found_sm = true;
							break;
						}
					} 
				}
				if( !found_sm ) {
					IRSym.setName(newName);
					newSym = IRSym;
				}
			}
			if( sm instanceof PseudoSymbol ) {
				newSym = AnalysisTools.replaceIRSymbol((PseudoSymbol)sm, IRSym, newSym);
			}
			new_set.add(newSym);
		}
	}
	
	/**
	 * Remove OpenACC clauses whose argument is an empty list.
	 * 
	 * @param tr
	 */
	static public void removeEmptyAccClauses(Traversable tr) {
		List<PragmaAnnotation> pragmaAnnots = AnalysisTools.collectPragmas(tr, PragmaAnnotation.class);
		if( pragmaAnnots != null ) {
			Set<String> removeSet;
			for( PragmaAnnotation aAnnot : pragmaAnnots ) {
				if( (aAnnot instanceof ACCAnnotation) || (aAnnot instanceof ARCAnnotation) ) {
					removeSet = new HashSet<String>();
					for( String key : aAnnot.keySet() ) {
						Object val = aAnnot.get(key);
						if( val instanceof Collection ) {
							if( ((Collection)val).isEmpty() ) {
								removeSet.add(key);
							}
						}
					}
					if( !removeSet.isEmpty() ) {
						for( String key : removeSet ) {
							aAnnot.remove(key);
						}
					}
					//If Cuda annoatation does not have any clause, remove it.
					if( aAnnot.keySet().contains("cuda") && (aAnnot.keySet().size() < 3) ) {
						Annotatable at = aAnnot.getAnnotatable();
						List<ARCAnnotation> aList = at.getAnnotations(ARCAnnotation.class);
						at.removeAnnotations(ARCAnnotation.class);
						for( ARCAnnotation tAnnot : aList ) {
							if( tAnnot != aAnnot ) {
								at.annotate(tAnnot);
							}
						}
					}
				}
			}
		}
	}

}


