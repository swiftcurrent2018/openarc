package openacc.analysis;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.HashMap;
import java.util.List;
import java.util.LinkedList;
import java.util.Map;
import java.util.Set;

import openacc.hir.ACCAnnotation;
import openacc.hir.ARCAnnotation;
import cetus.hir.Annotatable;
import cetus.analysis.AnalysisPass;
import cetus.analysis.CallGraph;
import cetus.exec.Driver;
import cetus.hir.AccessSymbol;
import cetus.hir.AnnotationDeclaration;
import cetus.hir.ArraySpecifier;
import cetus.hir.CompoundStatement;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.Identifier;
import cetus.hir.FunctionCall;
import cetus.hir.PragmaAnnotation;
import cetus.hir.Procedure;
import cetus.hir.ProcedureDeclarator;
import cetus.hir.Program;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.Symbol;
import cetus.hir.SymbolTools;
import cetus.hir.DataFlowTools;
import cetus.hir.IRTools;
import cetus.hir.PrintTools;
import cetus.hir.ArrayAccess;
import cetus.hir.Traversable;
import cetus.hir.NestedDeclarator;
import cetus.hir.VariableDeclarator;

/**
 * Locality Analysis (outdated).
 * If shrdSclrCachingOnReg == true,
 *     - if R/O shared scalar variables have locality, 
 *         add them into cuda registerRO clause. 
 *     - if R/W shared scalar variables have locality,
 *         add them into cuda registerRW clause. 
 * If shrdSclrCachingOnSM == true,
 *     - if R/O shared scalar variables exist, 
 *         add them into cuda sharedRO clause. 
 *     - if R/W shared scalar variables have locality,
 *         add them into cuda sharedRW clause. 
 *         (not yet implemented). 
 * If both shrdSclrCachingOnReg and shrdSclrCachingOnSM are on,        
 *     - R/O shared scalar variables with locality will be put
 *       into sharedRO clause.
 *     - R/W shared scalar variables with locality will be put
 *       into registerRW clause.
 * If shrdArryElmtCachingOnReg == true,
 *     - if R/O shared array elements have locality, 
 *         add them into cuda registerRO clause. 
 *     - if R/W shared array elements have locality,
 *         add them into cuda registerRW clause. 
 * If prvtArryCachingOnSM == true,
 *     - add private array variables into sharedRW clause.
 * If shrdArryCachingOnTM == true,
 *     - If one-dimensional, R/O shared arrays have locality,
 *         add them into cuda texture clause.
 *     - Current CUDA implementations do not support double texture.
 *     If you don't need texture interpolation you can utilize the texture cache 
 *     by accessing double data via tex1Dfetch(). To do so, you would bind an int2 
 *     texture to the array of doubles, and re-interprete the int2 returned by 
 *     tex1Dfetch() using __hiloint2double():
 *     int2 val = tex1Dfetch (texture, index);
 *     double r = __hiloint2double (val.y, val.x);
 *     
 *     However, the above scheme is not supported in the current implementation.
 *
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group
 *         Oak Ridge National Laboratory
 */
public class LocalityAnalysis extends AnalysisPass {
	private boolean shrdSclrCachingOnReg;
	private boolean shrdSclrCachingOnSM;
	private boolean shrdArryElmtCachingOnReg;
	private boolean prvtArryCachingOnSM;
	private boolean shrdArryCachingOnTM;
	private boolean shrdSclrCachingOnConst;
	private boolean shrdArryCachingOnConst;
	private boolean extractTuningParameters;
	private boolean localityAnalysisOn;
	private int programVerification;

	private boolean IRSymbolOnly;
	private ACCAnnotation programAnnot;
	private ARCAnnotation programCudaAnnot;

	/**
	 * @param program
	 */
	public LocalityAnalysis(Program program, boolean IRSymOnly, ACCAnnotation inAnnot, ARCAnnotation inCAnnot) {
		super(program);
		IRSymbolOnly = IRSymOnly;
		programAnnot = inAnnot;
		programCudaAnnot = inCAnnot;
	}

	/* (non-Javadoc)
	 * @see cetus.analysis.AnalysisPass#getPassName()
	 */
	@Override
	public String getPassName() {
		return new String("[LocalityAnalysis]");
	}

	/* (non-Javadoc)
	 * @see cetus.analysis.AnalysisPass#start()
	 */
	@Override
	public void start() {
		localityAnalysisOn = false;
		shrdSclrCachingOnReg = false;
		String value = Driver.getOptionValue("shrdSclrCachingOnReg");
		if( value != null ) {
			shrdSclrCachingOnReg = true;
			localityAnalysisOn = true;
		}
		shrdArryElmtCachingOnReg = false;
		value = Driver.getOptionValue("shrdArryElmtCachingOnReg");
		if( value != null ) {
			shrdArryElmtCachingOnReg = true;
			localityAnalysisOn = true;
		}
		shrdSclrCachingOnSM = false;
		value = Driver.getOptionValue("shrdSclrCachingOnSM");
		if( value != null ) {
			shrdSclrCachingOnSM = true;
			localityAnalysisOn = true;
		}
		prvtArryCachingOnSM = false;
		value = Driver.getOptionValue("prvtArryCachingOnSM");
		if( value != null ) {
			prvtArryCachingOnSM = true;
			localityAnalysisOn = true;
		}
		shrdArryCachingOnTM = false;
		value = Driver.getOptionValue("shrdArryCachingOnTM");
		if( value != null ) {
			shrdArryCachingOnTM = true;
			localityAnalysisOn = true;
		}
		shrdSclrCachingOnConst = false;
		value = Driver.getOptionValue("shrdSclrCachingOnConst");
		if( value != null ) {
			shrdSclrCachingOnConst = true;
			localityAnalysisOn = true;
		}
		shrdArryCachingOnConst = false;
		value = Driver.getOptionValue("shrdArryCachingOnConst");
		if( value != null ) {
			shrdArryCachingOnConst = true;
			localityAnalysisOn = true;
		}
		extractTuningParameters = false;
		value = Driver.getOptionValue("extractTuningParameters");
		if( value != null ) {
			extractTuningParameters = true;
			localityAnalysisOn = true;
		}
		programVerification = 0;
		value = Driver.getOptionValue("programVerification");
		if( value != null ) {
			programVerification = Integer.valueOf(value).intValue();
			if( programVerification > 0 ) {
				localityAnalysisOn = true;
			}
		}

		/////////////////////////////////////////////////////////////////////////////////////
		//DEBUG: CallGraph.getTopologicalCallList() returns only procedures reachable from //
		// the main procedure. To access all procedures, use an iterator.                  //
		/////////////////////////////////////////////////////////////////////////////////////
		/*		// generate a list of procedures in post-order traversal
		CallGraph callgraph = new CallGraph(program);
		// procedureList contains Procedure in ascending order; the last one is main
		List<Procedure> procedureList = callgraph.getTopologicalCallList();*/
		/* iterate to search for all Procedures */
		DepthFirstIterator proc_iter = new DepthFirstIterator(program);
		Set<Procedure> procedureList = (Set<Procedure>)(proc_iter.getSet(Procedure.class));

		/* drive the engine; visit every procedure */
		for (Procedure proc : procedureList)
		{
			List<ACCAnnotation> computeRegions = AnalysisTools.collectPragmas(proc.getBody(), ACCAnnotation.class, 
					ACCAnnotation.computeRegions, false);
			if( localityAnalysisOn ) {
				//DEBUG: if a share symbol is written in a previous kernel in the same procedure,
				//the symbol will not be cached in CUDA constant memory conservatively.
				//prevDefSymSet is used for this checking.
				//[FIXME] If a shared symbol is written in a kernel in other procedure, prevDefSymSet can not
				//check the writes; conservatively the symbol should not be cached on the constant memory.
				Set<Symbol> prevDefSymSet = new HashSet<Symbol>();
				boolean constSetAdded = false;
				List<Statement> modifiedConstRegionList = new LinkedList<Statement>();
				//FIXME: Filling prevDefSymSet will work correctly only if the below 
				//for-loop iterates cAnnots in a lexical order. 
				for( ACCAnnotation cAnnot : computeRegions ) {
					Statement pstmt = (Statement)cAnnot.getAnnotatable();
					ACCAnnotation annot = pstmt.getAnnotation(ACCAnnotation.class, "accshared");
					if( annot != null ) {
						Set<Symbol> sharedVars = annot.get("accshared");
						Set<Symbol> privVars = annot.get("accprivate");
						if(privVars == null)
							privVars = new HashSet<Symbol>();
						Set<Symbol> firstprivVars = annot.get("accfirstprivate");
						if(firstprivVars == null)
							firstprivVars = new HashSet<Symbol>();
						Set<Symbol> accreadonlySet = annot.get("accreadonly");
						if( accreadonlySet == null ) {
							accreadonlySet = new HashSet<Symbol>();
							annot.put("accreadonly", accreadonlySet);
						}
						Set<Symbol> accreadonlyprivateSet = annot.get("accreadonlyprivate");
						if( accreadonlyprivateSet == null ) {
							accreadonlyprivateSet = new HashSet<Symbol>();
							annot.put("accreadonlyprivate", accreadonlyprivateSet);
						}
						Set<Symbol> accExShared = annot.get("accexplicitshared");
						if( accExShared == null ) {
							accExShared = new HashSet<Symbol>();
							annot.put("accexplicitshared", accExShared);
						}
						Map<Expression, Set<Integer>> useExpMap = DataFlowTools.getUseMap(pstmt);
						Map<Expression, Set<Integer>> defExpMap = DataFlowTools.getDefMap(pstmt);
						Map<Symbol, Set<Integer>> useSymMap = DataFlowTools.convertExprMap2SymbolMap(useExpMap);
						Map<Symbol, Set<Integer>> defSymMap = DataFlowTools.convertExprMap2SymbolMap(defExpMap);
						Set<Symbol> useSymSet = useSymMap.keySet();
						Set<Symbol> defSymSet = defSymMap.keySet();
						Set<Expression> useExpSet = useExpMap.keySet();
						Set<Expression> defExpSet = defExpMap.keySet();
						Set<Symbol> defStructSymSet = new HashSet<Symbol>();
						for( Symbol dSym : defSymSet ) {
							if( dSym instanceof AccessSymbol ) {
								defStructSymSet.add(((AccessSymbol)dSym).getIRSymbol());
							}
						}

						HashMap<Symbol, SubArray> regROMap = new HashMap<Symbol, SubArray>();
						Set<SubArray> regROSet = null;
						HashSet<Symbol> cudaRegROSet = new HashSet<Symbol>();
						HashSet<SubArray> tCudaRegROSet = new HashSet<SubArray>();

						HashMap<Symbol, SubArray> regRWMap = new HashMap<Symbol, SubArray>();
						Set<SubArray> regRWSet = null;
						HashSet<Symbol> cudaRegRWSet = new HashSet<Symbol>();
						HashSet<SubArray> tCudaRegRWSet = new HashSet<SubArray>();

						HashMap<Symbol, SubArray> noRegMap = new HashMap<Symbol, SubArray>();
						Set<SubArray> noRegSet = new HashSet<SubArray>();

						HashMap<Symbol, SubArray> sharedROMap = new HashMap<Symbol, SubArray>();
						Set<SubArray> sharedROSet = null;
						HashSet<Symbol> cudaSharedROSet = new HashSet<Symbol>();

						HashMap<Symbol, SubArray> sharedRWMap = new HashMap<Symbol, SubArray>();
						Set<SubArray> sharedRWSet = null;
						HashSet<Symbol> cudaSharedRWSet = new HashSet<Symbol>();

						HashMap<Symbol, SubArray> noSharedMap = new HashMap<Symbol, SubArray>();
						Set<SubArray> noSharedSet = new HashSet<SubArray>();

						HashMap<Symbol, SubArray> textureMap = new HashMap<Symbol, SubArray>();
						Set<SubArray> textureSet = null;
						HashSet<Symbol> cudaTextureSet = new HashSet<Symbol>();

						HashMap<Symbol, SubArray> noTextureMap = new HashMap<Symbol, SubArray>();
						Set<SubArray> noTextureSet = new HashSet<SubArray>();

						HashMap<Symbol, SubArray> constMap = new HashMap<Symbol, SubArray>();
						Set<SubArray> constSet = null;
						HashSet<Symbol> cudaConstSet = new HashSet<Symbol>();

						HashMap<Symbol, SubArray> noConstMap = new HashMap<Symbol, SubArray>();
						Set<SubArray> noConstSet = new HashSet<SubArray>();

						////////////////////////////
						// Tunable parameter sets //
						////////////////////////////
						HashSet<String> tRegisterROSet = new HashSet<String>();
						HashSet<String> tRegisterRWSet = new HashSet<String>();
						HashSet<String> tSharedROSet = new HashSet<String>();
						HashSet<String> tSharedRWSet = new HashSet<String>();
						HashSet<String> tTextureSet = new HashSet<String>();
						HashSet<String> tConstantSet = new HashSet<String>();
						HashSet<String> tSclrConstSet = new HashSet<String>();
						HashSet<String> tArryConstSet = new HashSet<String>();
						HashSet<String> tROShSclrNL = new HashSet<String>();
						HashSet<String> tROShSclr = new HashSet<String>();
						HashSet<String> tRWShSclr = new HashSet<String>();
						HashSet<String> tROShArEl = new HashSet<String>();
						HashSet<String> tRWShArEl = new HashSet<String>();
						HashSet<String> tRO1DShAr = new HashSet<String>();
						HashSet<String> tPrvAr = new HashSet<String>();
						ARCAnnotation aInfoAnnot = pstmt.getAnnotation(ARCAnnotation.class, "ainfo");
						List<ARCAnnotation> arcAnnots = pstmt.getAnnotations(ARCAnnotation.class);
						ARCAnnotation cudaAnnot = null;
						if( arcAnnots != null ) {
							for( ARCAnnotation cannot : arcAnnots ) {
								if( cannot.containsKey("cuda") ) {
									cudaAnnot = cannot;
								} else {
									continue;
								}
								Set<SubArray> dataSet = (Set<SubArray>)cannot.get("registerRO");
								if( dataSet != null ) {
									for( SubArray sAr : dataSet ) {
										Symbol tSym = AnalysisTools.subarrayToSymbol(sAr, IRSymbolOnly);
										regROMap.put(tSym, sAr);
									}
									regROSet = dataSet;
								}
								dataSet = (Set<SubArray>)cannot.get("registerRW");
								if( dataSet != null ) {
									for( SubArray sAr : dataSet ) {
										Symbol tSym = AnalysisTools.subarrayToSymbol(sAr, IRSymbolOnly);
										regRWMap.put(tSym, sAr);
									}
									regRWSet = dataSet;
								}
								dataSet = (Set<SubArray>)cannot.get("noregister");
								if( dataSet != null ) {
									for( SubArray sAr : dataSet ) {
										Symbol tSym = AnalysisTools.subarrayToSymbol(sAr, IRSymbolOnly);
										noRegMap.put(tSym, sAr);
									}
									noRegSet.addAll(dataSet);
								}
								dataSet = (Set<SubArray>)cannot.get("sharedRO");
								if( dataSet != null ) {
									for( SubArray sAr : dataSet ) {
										Symbol tSym = AnalysisTools.subarrayToSymbol(sAr, IRSymbolOnly);
										sharedROMap.put(tSym, sAr);
									}
									sharedROSet = dataSet;
								}
								dataSet = (Set<SubArray>)cannot.get("sharedRW");
								if( dataSet != null ) {
									for( SubArray sAr : dataSet ) {
										Symbol tSym = AnalysisTools.subarrayToSymbol(sAr, IRSymbolOnly);
										sharedRWMap.put(tSym, sAr);
									}
									sharedRWSet = dataSet;
								}
								dataSet = (Set<SubArray>)cannot.get("noshared");
								if( dataSet != null ) {
									for( SubArray sAr : dataSet ) {
										Symbol tSym = AnalysisTools.subarrayToSymbol(sAr, IRSymbolOnly);
										noSharedMap.put(tSym, sAr);
									}
									noSharedSet.addAll(dataSet);
								}
								dataSet = (Set<SubArray>)cannot.get("texture");
								if( dataSet != null ) {
									for( SubArray sAr : dataSet ) {
										Symbol tSym = AnalysisTools.subarrayToSymbol(sAr, IRSymbolOnly);
										textureMap.put(tSym, sAr);
									}
									textureSet = dataSet;
								}
								dataSet = (Set<SubArray>)cannot.get("notexture");
								if( dataSet != null ) {
									for( SubArray sAr : dataSet ) {
										Symbol tSym = AnalysisTools.subarrayToSymbol(sAr, IRSymbolOnly);
										noTextureMap.put(tSym, sAr);
									}
									noTextureSet.addAll(dataSet);
								}
								dataSet = (Set<SubArray>)cannot.get("constant");
								if( dataSet != null ) {
									for( SubArray sAr : dataSet ) {
										Symbol tSym = AnalysisTools.subarrayToSymbol(sAr, IRSymbolOnly);
										constMap.put(tSym, sAr);
									}
									constSet = dataSet;
								}
								dataSet = (Set<SubArray>)cannot.get("noconstant");
								if( dataSet != null ) {
									for( SubArray sAr : dataSet ) {
										Symbol tSym = AnalysisTools.subarrayToSymbol(sAr, IRSymbolOnly);
										noConstMap.put(tSym, sAr);
									}
									noConstSet.addAll(dataSet);
								}
							}
						}
						int useCnt = 0;
						int defCnt = 0;
						for( Symbol sym: sharedVars ) {
							boolean isStruct = false;
							///////////////////////////////////////////////////////////////////////////////////////////
							// DEBUG: Parameter symbol is a child of ProcedureDeclarator, which is a member field    //
							// of the enclosing Procedure, but not a child. Therefore, traversing from the parameter //
							// symbol can not reach the Procedure.                                                   //
							///////////////////////////////////////////////////////////////////////////////////////////
							if( proc.containsSymbol(sym) ) {
								isStruct = SymbolTools.isStruct(sym, proc);
							} else {
								isStruct = SymbolTools.isStruct(sym, (Traversable)sym);
							}
							useCnt = 0;
							defCnt = 0;
							if( useSymSet.contains(sym) ) {
								useCnt = useSymMap.get(sym).size();
							}
							if( defSymSet.contains(sym) ) {
								defCnt = defSymMap.get(sym).size();
								prevDefSymSet.add(sym);
							}
							if( defStructSymSet.contains(sym) ) {
								defCnt = 1;
								prevDefSymSet.add(sym);
							}
							if( (defCnt == 0) && AnalysisTools.ipaIsDefined(sym, pstmt, null) ) {
								defCnt = 2; //Assume that there is locality.
								prevDefSymSet.add(sym);
							}
							if( (useCnt <= 1) && (defCnt <= 1) ) {
								////////////////////////
								//No locality exists. //
								///////////////////////////////////////////////////////
								//Even if there is no locality, passing R/O shared   //
								// scalar variable as kernel parameter can save GPU  //
								// global memory access.                             //
								///////////////////////////////////////////////////////
								if( defCnt == 0 ) { 
									//R/O shared variable
									accreadonlySet.add(sym);
									if( SymbolTools.isScalar(sym) && !SymbolTools.isPointer(sym) ) {
										tSharedROSet.add(sym.getSymbolName());
										tROShSclrNL.add(sym.getSymbolName());
										tSclrConstSet.add(sym.getSymbolName());
										tConstantSet.add(sym.getSymbolName());
										constSetAdded = true;
										if( isStruct ) {
											if( shrdSclrCachingOnConst ) {
												if( !prevDefSymSet.contains(sym) ) {
													cudaConstSet.add(sym);
													constSetAdded = true;
												}
											} else if( shrdSclrCachingOnSM ) {
												cudaSharedROSet.add(sym);
											} else {
												continue;
											}
										} else {
											if( shrdSclrCachingOnSM ) {
												cudaSharedROSet.add(sym);
											} else if( shrdSclrCachingOnConst ) {
												if( !prevDefSymSet.contains(sym) ) {
													cudaConstSet.add(sym);
													constSetAdded = true;
												}
											} else {
												continue;
											}
										}
									} else {
										tArryConstSet.add(sym.getSymbolName());
										tConstantSet.add(sym.getSymbolName());
										constSetAdded = true;
										int dimsize;
										if( SymbolTools.isArray(sym) ) {
											List aspecs = sym.getArraySpecifiers();
											ArraySpecifier aspec = (ArraySpecifier)aspecs.get(0);
											dimsize = aspec.getNumDimensions();
											if( sym instanceof NestedDeclarator ) {
												dimsize += 1;
											}
										} else {
											dimsize = 1;
										}
										if( (dimsize == 1) && !isStruct ) {
											if( !sym.getTypeSpecifiers().contains(Specifier.DOUBLE) ) {
												tTextureSet.add(sym.getSymbolName());
											}
											tRO1DShAr.add(sym.getSymbolName());
										}
										if (shrdArryCachingOnTM) {
											if( (dimsize == 1) && !isStruct ) {
												if( !sym.getTypeSpecifiers().contains(Specifier.DOUBLE) ) {
													cudaTextureSet.add(sym);
												}
											} else if( shrdArryCachingOnConst ) {
												if( !prevDefSymSet.contains(sym) ) {
													cudaConstSet.add(sym);
													constSetAdded = true;
												}
											} else {
												continue;
											}
										} else if( shrdArryCachingOnConst ) {
											if( !prevDefSymSet.contains(sym) ) {
												cudaConstSet.add(sym);
												constSetAdded = true;
											}
										} else {
											continue;
										}
									}
								} else {
									continue;
								}
							} else if ( defCnt == 0 ) {
								//R/O shared variable
								accreadonlySet.add(sym);
								if( SymbolTools.isScalar(sym) && !SymbolTools.isPointer(sym) ) {
									if( isStruct ) {
										/////////////////////////////////////////////
										// For R/O shared scalar struct variables, //
										// caching on Const is preferred method.   //
										/////////////////////////////////////////////
										if( shrdSclrCachingOnConst ) {
											if( !prevDefSymSet.contains(sym) ) {
												cudaConstSet.add(sym);
												constSetAdded = true;
											}
										} else if( shrdSclrCachingOnSM ) {
											cudaSharedROSet.add(sym);
										}
										tSharedROSet.add(sym.getSymbolName());
										tROShSclr.add(sym.getSymbolName());
										tSclrConstSet.add(sym.getSymbolName());
										tConstantSet.add(sym.getSymbolName());
										constSetAdded = true;
									} else {
										////////////////////////////////////////
										// For R/O shared scalar variables,   //
										// caching on SM is preferred method. //
										////////////////////////////////////////
										if( shrdSclrCachingOnSM ) {
											cudaSharedROSet.add(sym);
										} else if( shrdSclrCachingOnConst ) {
											if( !prevDefSymSet.contains(sym) ) {
												cudaConstSet.add(sym);
												constSetAdded = true;
											}
										} else if( shrdSclrCachingOnReg ) {
											cudaRegROSet.add(sym);
										}
										tSharedROSet.add(sym.getSymbolName());
										tRegisterROSet.add(sym.getSymbolName());
										tROShSclr.add(sym.getSymbolName());
										tSclrConstSet.add(sym.getSymbolName());
										tConstantSet.add(sym.getSymbolName());
										constSetAdded = true;
									}
								} else {
									int dimsize;
									if( SymbolTools.isArray(sym) ) {
										List aspecs = sym.getArraySpecifiers();
										ArraySpecifier aspec = (ArraySpecifier)aspecs.get(0);
										dimsize = aspec.getNumDimensions();
										if( sym instanceof NestedDeclarator ) {
											dimsize += 1;
										}
									} else {
										dimsize = 1;
									}
									if( (dimsize == 1) && !isStruct ) {
										if( shrdArryCachingOnTM ) {
											if( !sym.getTypeSpecifiers().contains(Specifier.DOUBLE) ) {
												cudaTextureSet.add(sym);
											}
										}
										if( !sym.getTypeSpecifiers().contains(Specifier.DOUBLE) ) {
											tTextureSet.add(sym.getSymbolName());
										}
										tRO1DShAr.add(sym.getSymbolName());
									}
									tArryConstSet.add(sym.getSymbolName());
									tConstantSet.add(sym.getSymbolName());
									constSetAdded = true;
									for( Expression exp : useExpSet ) {
										if( exp instanceof ArrayAccess ) {
											ArrayAccess aa = (ArrayAccess)exp;
											if( IRTools.containsSymbol(aa.getArrayName(), sym) ) {
												if( useExpMap.get(exp).size() > 1 ) {
													if( shrdArryElmtCachingOnReg && !cudaTextureSet.contains(sym) ) {
														tCudaRegROSet.add(AnalysisTools.arrayAccessToSubArray(aa));
													}
													tRegisterROSet.add(aa.toString());
													tROShArEl.add(aa.toString());
												}
											}
										}
									}
									if( !cudaTextureSet.contains(sym) && shrdArryCachingOnConst ) {
										if( !prevDefSymSet.contains(sym) ) {
											cudaConstSet.add(sym);
											constSetAdded = true;
										}
									}
								}
							} else {
								//R/W shared variable
								if( SymbolTools.isScalar(sym) && !SymbolTools.isPointer(sym) ) {
									///////////////////////////////////////
									// For R/W shared scalar variables,  //
									// caching on Register is preferred. //
									///////////////////////////////////////
									if( shrdSclrCachingOnReg && !isStruct ) {
										cudaRegRWSet.add(sym);
									} else if( shrdSclrCachingOnSM ) {
										cudaSharedRWSet.add(sym);
									}
									if(!isStruct ) {
										tRegisterRWSet.add(sym.getSymbolName());
									}
									tRWShSclr.add(sym.getSymbolName());
									tSharedRWSet.add(sym.getSymbolName());
								} else if( !isStruct ) {
									for( Expression exp : defExpSet ) {
										if( exp instanceof ArrayAccess ) {
											ArrayAccess aa = (ArrayAccess)exp;
											if( IRTools.containsSymbol(aa.getArrayName(), sym) ) {
												if( (defExpMap.get(exp).size() > 1) || 
														(useExpSet.contains(exp) && (useExpMap.get(exp).size() > 1)) ) {
													if( shrdArryElmtCachingOnReg ) {
														SubArray tSubArr = AnalysisTools.arrayAccessToSubArray(aa);
														if( !noRegSet.contains(tSubArr) ) {
															tCudaRegRWSet.add(tSubArr);
														}
													}
													tRegisterRWSet.add(aa.toString());
													tRWShArEl.add(aa.toString());
												}
											}
										}
									}
								}
							}
						}
						for( Symbol sym: privVars ) {
							if( SymbolTools.isArray(sym) || SymbolTools.isPointer(sym) ) {
								if( prvtArryCachingOnSM ) {
									cudaSharedRWSet.add(sym);
								}
								tSharedRWSet.add(sym.getSymbolName());
								tPrvAr.add(sym.getSymbolName());
							}
							if( !defSymSet.contains(sym) ) {
								accreadonlyprivateSet.add(sym);
							}
						}
						for( Symbol sym: firstprivVars ) {
							if( SymbolTools.isArray(sym) || SymbolTools.isPointer(sym) ) {
								if( prvtArryCachingOnSM ) {
									cudaSharedRWSet.add(sym);
								}
								tSharedRWSet.add(sym.getSymbolName());
								tPrvAr.add(sym.getSymbolName());
							}
							if( !defSymSet.contains(sym) ) {
								accreadonlyprivateSet.add(sym);
							}
						}
						cudaRegROSet.removeAll(noRegMap.keySet());
						if( cudaRegROSet.size() > 0 ) {
							if( regROSet == null ) {
								if( cudaAnnot == null ) {
									cudaAnnot = new ARCAnnotation("cuda", "_directive");
									pstmt.annotate(cudaAnnot);
								}
								regROSet = new HashSet<SubArray>();
								cudaAnnot.put("registerRO", regROSet);
							}
							for( Symbol tSym : cudaRegROSet ) {
								if( !regROMap.containsKey(tSym) ) {
									regROSet.add(AnalysisTools.createSubArray(tSym, false, null));
								}
							}
						}
						if( tCudaRegROSet.size() > 0 ) {
							if( regROSet == null ) {
								if( cudaAnnot == null ) {
									cudaAnnot = new ARCAnnotation("cuda", "_directive");
									pstmt.annotate(cudaAnnot);
								}
								regROSet = new HashSet<SubArray>();
								cudaAnnot.put("registerRO", regROSet);
							}
							regROSet.addAll(tCudaRegROSet);
						}
						cudaRegRWSet.removeAll(noRegMap.keySet());
						if( cudaRegRWSet.size() > 0 ) {
							if( regRWSet == null ) {
								if( cudaAnnot == null ) {
									cudaAnnot = new ARCAnnotation("cuda", "_directive");
									pstmt.annotate(cudaAnnot);
								}
								regRWSet = new HashSet<SubArray>();
								cudaAnnot.put("registerRW", regRWSet);
							}
							for( Symbol tSym : cudaRegRWSet ) {
								if( !regRWMap.containsKey(tSym) ) {
									regRWSet.add(AnalysisTools.createSubArray(tSym, false, null));
								}
							}
						}
						if( tCudaRegRWSet.size() > 0 ) {
							if( regRWSet == null ) {
								if( cudaAnnot == null ) {
									cudaAnnot = new ARCAnnotation("cuda", "_directive");
									pstmt.annotate(cudaAnnot);
								}
								regRWSet = new HashSet<SubArray>();
								cudaAnnot.put("registerRW", regRWSet);
							}
							regRWSet.addAll(tCudaRegRWSet);
						}
						cudaSharedROSet.removeAll(noSharedMap.keySet());
						if( cudaSharedROSet.size() > 0 ) {
							if( sharedROSet == null ) {
								if( cudaAnnot == null ) {
									cudaAnnot = new ARCAnnotation("cuda", "_directive");
									pstmt.annotate(cudaAnnot);
								}
								sharedROSet = new HashSet<SubArray>();
								cudaAnnot.put("sharedRO", sharedROSet);
							}
							for( Symbol tSym : cudaSharedROSet ) {
								if( !sharedROMap.containsKey(tSym) ) {
									sharedROSet.add(AnalysisTools.createSubArray(tSym, false, null));
								}
							}
						}
						cudaSharedRWSet.removeAll(noSharedMap.keySet());
						if( cudaSharedRWSet.size() > 0 ) {
							if( sharedRWSet == null ) {
								if( cudaAnnot == null ) {
									cudaAnnot = new ARCAnnotation("cuda", "_directive");
									pstmt.annotate(cudaAnnot);
								}
								sharedRWSet = new HashSet<SubArray>();
								cudaAnnot.put("sharedRW", sharedRWSet);
							}
							for( Symbol tSym : cudaSharedRWSet ) {
								if( !sharedRWMap.containsKey(tSym) ) {
									sharedRWSet.add(AnalysisTools.createSubArray(tSym, false, null));
								}
							}
						}
						cudaTextureSet.removeAll(noTextureMap.keySet());
						if( cudaTextureSet.size() > 0 ) {
							if( textureSet == null ) {
								if( cudaAnnot == null ) {
									cudaAnnot = new ARCAnnotation("cuda", "_directive");
									pstmt.annotate(cudaAnnot);
								}
								textureSet = new HashSet<SubArray>();
								cudaAnnot.put("texture", textureSet);
							}
							for( Symbol tSym : cudaTextureSet ) {
								if( !textureMap.containsKey(tSym) ) {
									textureSet.add(AnalysisTools.createSubArray(tSym, false, null));
								}
							}
						}
						cudaConstSet.removeAll(noConstMap.keySet());
						if( cudaConstSet.size() > 0 ) {
							if( constSet == null ) {
								if( cudaAnnot == null ) {
									cudaAnnot = new ARCAnnotation("cuda", "_directive");
									pstmt.annotate(cudaAnnot);
								}
								constSet = new HashSet<SubArray>();
								cudaAnnot.put("constant", constSet);
							}
							for( Symbol tSym : cudaConstSet ) {
								if( !constMap.containsKey(tSym) ) {
									constSet.add(AnalysisTools.createSubArray(tSym, false, null));
								}
							}
						}
						if( constSetAdded ) {
							modifiedConstRegionList.add(pstmt);
							constSetAdded = false;
						}
						if( extractTuningParameters && (aInfoAnnot != null) ) {
							//"tuningparameter" directive is used only internally .
							ACCAnnotation tAnnot = pstmt.getAnnotation(ACCAnnotation.class, "tuningparameters");
							if( tAnnot == null ) {
								tAnnot = new ACCAnnotation("tuningparameters", "_directive");
								pstmt.annotate(tAnnot);
							}
							if( tRegisterROSet.size() > 0 ) {
								tAnnot.put("tmp_registerRO", tRegisterROSet);
							}
							if( tRegisterRWSet.size() > 0 ) {
								tAnnot.put("tmp_registerRW", tRegisterRWSet);
							}
							if( tSharedROSet.size() > 0 ) {
								tAnnot.put("tmp_sharedRO", tSharedROSet);
							}
							if( tSharedRWSet.size() > 0 ) {
								tAnnot.put("tmp_sharedRW", tSharedRWSet);
							}
							if( tTextureSet.size() > 0 ) {
								tAnnot.put("tmp_texture", tTextureSet);
							}
							if( tConstantSet.size() > 0 ) {
								tAnnot.put("tmp_constant", tConstantSet);
							}
							if( tSclrConstSet.size() > 0 ) {
								tAnnot.put("SclrConst", tSclrConstSet);
							}
							if( tArryConstSet.size() > 0 ) {
								tAnnot.put("ArryConst", tArryConstSet);
							}
							if( tROShSclrNL.size() > 0 ) {
								tAnnot.put("ROShSclrNL", tROShSclrNL);
							}
							if( tROShSclr.size() > 0 ) {
								tAnnot.put("ROShSclr", tROShSclr);
							}
							if( tRWShSclr.size() > 0 ) {
								tAnnot.put("RWShSclr", tRWShSclr);
							}
							if( tROShArEl.size() > 0 ) {
								tAnnot.put("ROShArEl", tROShArEl);
							}
							if( tRWShArEl.size() > 0 ) {
								tAnnot.put("RWShArEl", tRWShArEl);
							}
							if( tRO1DShAr.size() > 0 ) {
								tAnnot.put("RO1DShAr", tRO1DShAr);
							}
							if( tPrvAr.size() > 0 ) {
								tAnnot.put("PrvAr", tPrvAr);
							}
						}
						//For R/O shared variables whose data transfers are not explicitly specified by users,
						//move them from pcopy/copy to pcopyin/copyin.
						ACCAnnotation compAnnot = null;
						Map<Symbol, SubArray> pcopyMap = new HashMap<Symbol, SubArray>();
						Map<Symbol, SubArray> copyMap = new HashMap<Symbol, SubArray>();
						Set<SubArray> pcopySet = null;
						Set<SubArray> copySet = null;
						Set<SubArray> pcopyinSet = null;
						Set<SubArray> copyinSet = null;
						ACCAnnotation tempAnnot = pstmt.getAnnotation(ACCAnnotation.class, "pcopy");
						if( tempAnnot != null ) {
							compAnnot = tempAnnot;
							pcopySet = tempAnnot.get("pcopy");
							for( SubArray tempSAr : pcopySet ) {
								Symbol temSym = AnalysisTools.subarrayToSymbol(tempSAr, IRSymbolOnly);
								pcopyMap.put(temSym, tempSAr);
							}
						}
						tempAnnot = pstmt.getAnnotation(ACCAnnotation.class, "copy");
						if( tempAnnot != null ) {
							compAnnot = tempAnnot;
							copySet = tempAnnot.get("copy");
							for( SubArray tempSAr : copySet ) {
								Symbol temSym = AnalysisTools.subarrayToSymbol(tempSAr, IRSymbolOnly);
								copyMap.put(temSym, tempSAr);
							}
						}
						tempAnnot = pstmt.getAnnotation(ACCAnnotation.class, "pcopyin");
						if( tempAnnot != null ) {
							compAnnot = tempAnnot;
							pcopyinSet = tempAnnot.get("pcopyin");
						}
						tempAnnot = pstmt.getAnnotation(ACCAnnotation.class, "copyin");
						if( tempAnnot != null ) {
							compAnnot = tempAnnot;
							copyinSet = tempAnnot.get("copyin");
						}
						for( Symbol roSym : accreadonlySet ) {
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
							}
						}
					}
				}
				//DEBUG: if a shared variable is R/W in a kernel region and R/O in other kernel region
				//in the same procedure, caching the shared variable on CUDA constant memory for the
				//R/O kernel may be incorrect depending on other memory-transfer optimizations.
				//Therefore, if the shared variable is R/W in any of a kernel region in a procedure,
				//conservatively remove it from the CUDA constant clause.
				if( (!prevDefSymSet.isEmpty()) && (!modifiedConstRegionList.isEmpty()) ) {
					Set<String> prevDefStringSet = AnalysisTools.symbolsToStringSet(prevDefSymSet);
					for( Statement pstmt : modifiedConstRegionList ) {
						List<PragmaAnnotation> accAnnots = new LinkedList<PragmaAnnotation>();
						List<PragmaAnnotation> pragmaAnnots = pstmt.getAnnotations(PragmaAnnotation.class);
						if( (pragmaAnnots == null) || pragmaAnnots.isEmpty() ) {
							continue;
						} else {
							for( PragmaAnnotation pAnnot : pragmaAnnots ) {
								if( (pAnnot instanceof ACCAnnotation) || (pAnnot instanceof ARCAnnotation) ) {
									accAnnots.add(pAnnot);
								}
							}
						}
						if( !accAnnots.isEmpty() ) {
							boolean emptyAnnot =false;
							for( PragmaAnnotation cannot : accAnnots ) {
								if( cannot.containsKey("tuningparameters") ) {
									HashSet<String> dataSet = (HashSet<String>)cannot.get("tmp_constant");
									if( dataSet != null ) {
										Set<String> removeSet = new HashSet<String>();
										for( String sArr : dataSet ) {
											if( prevDefStringSet.contains(sArr) ) {
												//dataSet.remove(sArr);
												removeSet.add(sArr);
											}
										}
										dataSet.removeAll(removeSet);
										if( dataSet.isEmpty() ) {
											cannot.remove("tmp_constant");
											emptyAnnot = true;
										}
										dataSet = (HashSet<String>)cannot.get("SclrConst");
										if( dataSet != null ) {
											dataSet.removeAll(removeSet);
											if( dataSet.isEmpty() ) {
												cannot.remove("SclrConst");
											}
										}
										dataSet = (HashSet<String>)cannot.get("ArryConst");
										if( dataSet != null ) {
											dataSet.removeAll(removeSet);
											if( dataSet.isEmpty() ) {
												cannot.remove("ArryConst");
											}
										}
									}
								} else {
									HashSet<SubArray> dataSet = (HashSet<SubArray>)cannot.get("constant");
									if( dataSet != null ) {
										Set<SubArray> removeSet = new HashSet<SubArray>();
										for( SubArray sArr : dataSet ) {
											Symbol cSym = AnalysisTools.subarrayToSymbol(sArr, IRSymbolOnly);
											if( prevDefSymSet.contains(cSym) ) {
												//dataSet.remove(sArr);
												removeSet.add(sArr);
											}
										}
										dataSet.removeAll(removeSet);
										if( dataSet.isEmpty() ) {
											cannot.remove("constant");
											emptyAnnot = true;
										}
									}
								}
							}
							if( emptyAnnot ) {
								//[CAUTION] Below code will remove ACCAnnotations if they don't have any clause, 
								//but some ACCAnnotations (e.g., resilience, ftregion, ftinject) may not have 
								//any clause legally; these should not be deleted.
								List<PragmaAnnotation> newList = new LinkedList<PragmaAnnotation>();
								for( PragmaAnnotation cAnnot : accAnnots ) {
									if( cAnnot.size() > 2 ) {
										newList.add(cAnnot);
									} else if( cAnnot.size() == 2 ) {
										boolean isResDirective = false;
										for (String resDirective : ARCAnnotation.resilienceDirectives ) {
											if( cAnnot.containsKey(resDirective) ) {
												isResDirective = true;
												break;
											}
										}
										if( isResDirective ) {
											newList.add(cAnnot);
										}
									}
								}
								pstmt.removeAnnotations(PragmaAnnotation.class);
								if( newList.size() > 0 ) {
									for( PragmaAnnotation newAnnot : newList ) {
										pstmt.annotate(newAnnot);
									}
								} 
							}
						}	
					}
				}
			} //End of localityAnalysis
			
			
			//Check whether CUDA data clauses exist in both compute regions and enclosing data regions.
			//If not, add necessary data clauses so that both have the same CUDA data clauses.
			List<FunctionCall> gFuncCallList = IRTools.getFunctionCalls(program);
			for( ACCAnnotation cAnnot : computeRegions ) {
				Statement pstmt = (Statement)cAnnot.getAnnotatable();
				Set<Symbol> readonlySet = new HashSet<Symbol>();
				ACCAnnotation roAnnot = pstmt.getAnnotation(ACCAnnotation.class, "accreadonly");
				if( roAnnot != null ) {
					readonlySet = roAnnot.get("accreadonly");
				}
				Set<Symbol> kSharedSet = new HashSet<Symbol>();
				ACCAnnotation sAnnot = pstmt.getAnnotation(ACCAnnotation.class, "accshared");
				if( sAnnot != null ) {
					kSharedSet.addAll((Set<Symbol>)sAnnot.get("accshared"));
				}
				//Step1: find enclosing explicit/implicit data regions
				LinkedList<ACCAnnotation> enclosingDataRegions = new LinkedList<ACCAnnotation>();
				ACCAnnotation drAnnot = AnalysisTools.ipFindFirstPragmaInParent(pstmt, ACCAnnotation.class, "data", gFuncCallList, null);
				while( drAnnot != null ) {
					enclosingDataRegions.add(drAnnot);
					drAnnot = AnalysisTools.ipFindFirstPragmaInParent(drAnnot.getAnnotatable(), ACCAnnotation.class, "data", gFuncCallList, null);
				}
				enclosingDataRegions.add(programAnnot);
				//Step2: if variables exist only in one clause, add it to the other one.
				for( String dClause : ARCAnnotation.cudaMDataClauses ) {
					ARCAnnotation kAnnot = pstmt.getAnnotation(ARCAnnotation.class, dClause);
					Set<SubArray> kDSet = null;
					Map<Symbol, SubArray> kDMap = new HashMap<Symbol, SubArray>();
					if( kAnnot == null ) {
						kDSet = new HashSet<SubArray>();
					} else {
						kDSet = (Set<SubArray>)kAnnot.get(dClause);
						for( SubArray tSArr : kDSet ) {
							Symbol tSym = AnalysisTools.subarrayToSymbol(tSArr, IRSymbolOnly);
							if( tSym != null ) {
								kDMap.put(tSym, tSArr);
							}
						}
					}
					for( ACCAnnotation dAnnot : enclosingDataRegions ) {
						Annotatable dAt = dAnnot.getAnnotatable();
						Procedure dPproc = null;
						if( dAt != null ) {
							dPproc = IRTools.getParentProcedure(dAt);
						}
						Set<Symbol> enSharedSet = new HashSet<Symbol>();
						ACCAnnotation ensAnnot = null;
						ARCAnnotation enAnnot = null;
						if( dAt != null ) {
							ensAnnot = dAt.getAnnotation(ACCAnnotation.class, "accshared");
							enAnnot = dAt.getAnnotation(ARCAnnotation.class, dClause);
						} else { //dAnnot is for the entire program.
							if( dAnnot.containsKey("accshared") ) {
								ensAnnot = dAnnot;
							}
							if( dAnnot.equals(programAnnot) ) {
								if( programCudaAnnot.containsKey(dClause) ) {
									enAnnot = programCudaAnnot;
								}
							}
						}
						if( ensAnnot != null ) {
							enSharedSet.addAll((Set<Symbol>)ensAnnot.get("accshared"));
						}
						if( (kAnnot == null) && (enAnnot == null) ) {
							continue;
						}
						Set<SubArray> enDSet = null;
						Map<Symbol, SubArray> enDMap = new HashMap<Symbol, SubArray>();
						if( enAnnot == null ) {
							enDSet = new HashSet<SubArray>();
						} else {
							enDSet = (Set<SubArray>)enAnnot.get(dClause);
							for( SubArray tSArr : enDSet ) {
								Symbol tSym = AnalysisTools.subarrayToSymbol(tSArr, IRSymbolOnly);
								if( tSym != null ) {
									enDMap.put(tSym, tSArr);
								}
							}
						}
						// Check symbol scope if enclosing procedures are different.
						Map<Symbol, SubArray> tkDMap = null;
						if( dPproc != proc ) {
							tkDMap = new HashMap<Symbol, SubArray>();
							Traversable refTr = dAt;
							if( dAt == null ) {
								refTr = proc;
							}
							for( Symbol tSym : tkDMap.keySet() ) {
								List osymList = new ArrayList(2);
								if( AnalysisTools.SymbolStatus.OrgSymbolFound(
									AnalysisTools.findOrgSymbol(tSym, refTr, false, dPproc, osymList, gFuncCallList)) ) {
									Symbol odSym = (Symbol)osymList.get(0);
									//FIXME: below assumes that odSym is IR symbol.
									SubArray odSArray = new SubArray(new Identifier(odSym));
									tkDMap.put(odSym, odSArray);
								}
							}
						} else {
							tkDMap = kDMap;
						}
						Set<Symbol> uniqKDSet = new HashSet<Symbol>();
						uniqKDSet.addAll(tkDMap.keySet());
						uniqKDSet.removeAll(enDMap.keySet());
						if( !uniqKDSet.isEmpty() ) {
							boolean isAdded = false;
							for( Symbol uSym : uniqKDSet ) {
								if( enSharedSet.contains(uSym) ) {
									enDSet.add(tkDMap.get(uSym).clone());
									isAdded = true;
								}
							}
							if( isAdded && (enAnnot == null) ) {
								if( dAt != null ) {
									enAnnot = dAt.getAnnotation(ARCAnnotation.class, "cuda");
									if( enAnnot == null ) {
										enAnnot = new ARCAnnotation("cuda", "_directive");
										dAt.annotate(enAnnot);
									}
								} else {
									enAnnot = programCudaAnnot;
								}
								enAnnot.put(dClause, enDSet);
							}
						}
						Set<Symbol> uniqEnDSet = new HashSet<Symbol>();
						uniqEnDSet.addAll(enDMap.keySet());
						uniqEnDSet.removeAll(tkDMap.keySet());
						if( !uniqEnDSet.isEmpty() ) {
							boolean isAdded = false;
							for( Symbol uSym : uniqEnDSet ) {
								//FIXME: below will not work if uSym is not visible in proc scope.
								if( kSharedSet.contains(uSym) ) {
									if( ARCAnnotation.cudaRODataClauses.contains(dClause) ) {
										if( readonlySet.contains(uSym) ) {
											kDSet.add(enDMap.get(uSym).clone());
											isAdded = true;
										}
									} else {
										kDSet.add(enDMap.get(uSym).clone());
										isAdded = true;
									}
								}
							}
							if( isAdded && (kAnnot == null) ) {
								kAnnot = pstmt.getAnnotation(ARCAnnotation.class, "cuda");
								if( kAnnot == null ) {
									kAnnot = new ARCAnnotation("cuda", "_directive");
									pstmt.annotate(kAnnot);
								}
								kAnnot.put(dClause, kDSet);
							}
						}
					}
				}
				//If enclosing data region is implicit, corresponding declare directives should 
				// be updated.
				for( ACCAnnotation dAnnot : enclosingDataRegions ) {
					Annotatable dAt = dAnnot.getAnnotatable();
					if( (dAt == null) || (dAt instanceof Procedure) ) {
						//enclosing region is an implicit region.
						Traversable impT = null;
						if( dAt == null ) {
							impT = proc.getParent();
						} else {
							impT = ((Procedure)dAt).getBody();
						}
						List<ACCAnnotation> declareAnnots = 
								IRTools.collectPragmas(impT, ACCAnnotation.class, "update");
						if( declareAnnots != null ) {
							for( ACCAnnotation decAnnot : declareAnnots ) {
								Annotatable decAt = decAnnot.getAnnotatable();
								if( dAt == null ) {
									if( !(decAt instanceof AnnotationDeclaration) ) {
										continue;
									}
								}
								ACCAnnotation iAnnot = decAt.getAnnotation(ACCAnnotation.class, "accshared");
								Set<Symbol> accSharedSet = new HashSet<Symbol>();
								if( iAnnot != null ) {
									accSharedSet.addAll((Set<Symbol>)iAnnot.get("accshared"));
								}
								for( String dClause : ARCAnnotation.cudaMDataClauses ) {
									ARCAnnotation enAnnot = null;
									if( dAt != null ) {
										enAnnot = dAt.getAnnotation(ARCAnnotation.class, dClause);
									} else if( programCudaAnnot.containsKey(dClause)) {
										enAnnot = programCudaAnnot;
									}
									if( enAnnot != null ) {
										Set<SubArray> subArraySet = enAnnot.get(dClause);
										Set<Symbol> symSet = AnalysisTools.subarraysToSymbols(subArraySet, IRSymbolOnly);
										symSet.retainAll(accSharedSet);
										if( !symSet.isEmpty() ) {
											ARCAnnotation tAnnot = decAt.getAnnotation(ARCAnnotation.class, "cuda");
											if( tAnnot == null ) {
												tAnnot = new ARCAnnotation("cuda", "_directive");
												decAt.annotate(tAnnot);
											}
											Set<SubArray> tSArrays = null;
											if( !tAnnot.containsKey(dClause) ) {
												tSArrays = new HashSet<SubArray>();
												tAnnot.put(dClause, tSArrays);
											} else {
												tSArrays = tAnnot.get(dClause);
											}
											Set<Symbol> tSyms = AnalysisTools.subarraysToSymbols(tSArrays, IRSymbolOnly);
											for( Symbol nSym : symSet ) {
												if( !tSyms.contains(nSym) ) {
													//FIXME: below assumes IR symbol.
													tSArrays.add(new SubArray(new Identifier(nSym)));
												}
											}
										}
									}
								}
							}
						}
					}
				}
				//If an enclosing data region contains update directive, cuda directives may need to be updated
				//to each update directive.
				for( ACCAnnotation dAnnot : enclosingDataRegions ) {
					Annotatable dAt = dAnnot.getAnnotatable();
					if( (dAt != null) ) {
						//Update directive can exist only within procedure body.
						Traversable impT = dAt;
						if( dAt instanceof Procedure ) {
							impT = ((Procedure)dAt).getBody();
						}
						List<ACCAnnotation> updateAnnots = 
								IRTools.collectPragmas(impT, ACCAnnotation.class, "update");
						if( updateAnnots != null ) {
							for( ACCAnnotation updAnnot : updateAnnots ) {
								Annotatable updAt = updAnnot.getAnnotatable();
								//For now, we only care about update device clause.
								ACCAnnotation iAnnot = updAt.getAnnotation(ACCAnnotation.class, "device");
								Set<Symbol> accSharedSet = new HashSet<Symbol>();
								if( iAnnot != null ) {
									accSharedSet.addAll(AnalysisTools.subarraysToSymbols((Set<SubArray>)iAnnot.get("device"), IRSymbolOnly));
								}
								for( String dClause : ARCAnnotation.cudaMDataClauses ) {
									ARCAnnotation enAnnot = null;
									if( dAt != null ) {
										enAnnot = dAt.getAnnotation(ARCAnnotation.class, dClause);
									} else if( programCudaAnnot.containsKey(dClause)) {
										enAnnot = programCudaAnnot;
									}
									if( enAnnot != null ) {
										Set<SubArray> subArraySet = enAnnot.get(dClause);
										Set<Symbol> symSet = AnalysisTools.subarraysToSymbols(subArraySet, IRSymbolOnly);
										symSet.retainAll(accSharedSet);
										if( !symSet.isEmpty() ) {
											ARCAnnotation tAnnot = updAt.getAnnotation(ARCAnnotation.class, "cuda");
											if( tAnnot == null ) {
												tAnnot = new ARCAnnotation("cuda", "_directive");
												updAt.annotate(tAnnot);
											}
											Set<SubArray> tSArrays = null;
											if( !tAnnot.containsKey(dClause) ) {
												tSArrays = new HashSet<SubArray>();
												tAnnot.put(dClause, tSArrays);
											} else {
												tSArrays = tAnnot.get(dClause);
											}
											Set<Symbol> tSyms = AnalysisTools.subarraysToSymbols(tSArrays, IRSymbolOnly);
											for( Symbol nSym : symSet ) {
												if( !tSyms.contains(nSym) ) {
													//FIXME: below assumes IR symbol.
													tSArrays.add(new SubArray(new Identifier(nSym)));
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

	}

}
