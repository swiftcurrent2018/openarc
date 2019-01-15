/**
 * 
 */
package openacc.transforms;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import openacc.analysis.ASPENModelAnalysis;
import openacc.analysis.AnalysisTools;
import openacc.analysis.SubArray;
import openacc.hir.ACCAnnotation;
import openacc.hir.ASPENAnnotation;
import openacc.hir.ASPENCompoundStatement;
import openacc.hir.ASPENControlExecuteStatement;
import openacc.hir.ASPENControlIfStatement;
import openacc.hir.ASPENControlIterateStatement;
import openacc.hir.ASPENControlKernelCallStatement;
import openacc.hir.ASPENControlMapStatement;
import openacc.hir.ASPENControlParallelStatement;
import openacc.hir.ASPENControlProbabilityStatement;
import openacc.hir.ASPENControlSeqStatement;
import openacc.hir.ASPENControlStatement;
import openacc.hir.ASPENData;
import openacc.hir.ASPENDataDeclaration;
import openacc.hir.ASPENDeclaration;
import openacc.hir.ASPENExpressionStatement;
import openacc.hir.ASPENTrait;
//import openacc.hir.ASPENExposesExpressionStatement;
import openacc.hir.ASPENKernel;
import openacc.hir.ASPENMemoryExpressionStatement;
import openacc.hir.ASPENModel;
import openacc.hir.ASPENParam;
import openacc.hir.ASPENParamDeclaration;
import openacc.hir.ASPENRequiresExpressionStatement;
import openacc.hir.ASPENResource;
import openacc.hir.ASPENStatement;
import cetus.analysis.CallGraph;
import cetus.hir.AccessExpression;
import cetus.hir.AccessSymbol;
import cetus.hir.Annotatable;
import cetus.hir.AnnotationDeclaration;
import cetus.hir.AnnotationStatement;
import cetus.hir.AssignmentExpression;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.CommentAnnotation;
import cetus.hir.CompoundStatement;
import cetus.hir.DFIterator;
import cetus.hir.DataFlowTools;
import cetus.hir.Declaration;
import cetus.hir.Declarator;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.FlatIterator;
import cetus.hir.FunctionCall;
import cetus.hir.IDExpression;
import cetus.hir.IRTools;
import cetus.hir.Identifier;
import cetus.hir.IfStatement;
import cetus.hir.IntegerLiteral;
import cetus.hir.Literal;
import cetus.hir.Loop;
import cetus.hir.NameID;
import cetus.hir.PrintTools;
import cetus.hir.Procedure;
import cetus.hir.ProcedureDeclarator;
import cetus.hir.Program;
import cetus.hir.SomeExpression;
import cetus.hir.Specifier;
import cetus.hir.StandardLibrary;
import cetus.hir.Statement;
import cetus.hir.SwitchStatement;
import cetus.hir.Symbol;
import cetus.hir.SymbolTable;
import cetus.hir.SymbolTools;
import cetus.hir.Symbolic;
import cetus.hir.Tools;
import cetus.hir.TranslationUnit;
import cetus.hir.Traversable;
import cetus.hir.Typecast;
import cetus.hir.VariableDeclaration;
import cetus.transforms.TransformPass;

/**
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group
 *         Oak Ridge National Laboratory
 *
 */
public class ASPENModelGen extends TransformPass {
	private boolean IRSymbolOnly = true;
	private Procedure main = null;
	private TranslationUnit mainTrUnt = null;
	private ASPENModel aspenModel = null;
	private Set<IDExpression> ignoredKernels = new HashSet<IDExpression>();
	private	int exeBlockCnt = 0;
	private int maxLoopCnt = 20;
	private int postprocessing = 2;
	private boolean inASPENModelRegion = false;
	private boolean IRMerged = false;
	
	private static String ASPENPredictFuncName = "HI_aspenpredict";
	public static Procedure ASPENPredictFunc = null;
	private static List<String> tuningParameters = new ArrayList<String>();
	private Map<ASPENDeclaration, Traversable> aspenDeclToIRMap = new HashMap<ASPENDeclaration, Traversable>();
	private int genASPENModel = 2;
    static public Set<String> ASPENKeywords = 
    	new HashSet<String>(Arrays.asList("param", "in", "with", "as", "of", "size", //common keywords
    			"model", "kernel", "data", "import", "to", "from", "iterate", "map", 
    			"par", "seq", "execute", "if", "else", "probability", 				//application keywords
    			"node", "nodes", "machine", "socket", "sockets", "cache", "link",
    			"linked", "resource", "core", "cores", "memory", "interconnet", 
    			"conflict", "power", "static", "dynamic",                           //machine keywords
    			"nano", "micro", "milli", "mega", "giga", "tera", "peta", "exa",	//built-in constants
    			"and", "or"));                                                      //symbols
    			
	
	/**
	 * @param program
	 */
	public ASPENModelGen(Program program, boolean IRSymOnly, int genModel) {
		super(program);
		IRSymbolOnly = IRSymOnly;
		genASPENModel = genModel;
		postprocessing = ASPENModelAnalysis.ASPENConfiguration.postprocessing;
	}

	/* (non-Javadoc)
	 * @see cetus.transforms.TransformPass#getPassName()
	 */
	@Override
	public String getPassName() {
		return "[ASPENModelGen]";
	}
	
	public static void ASPENPostProcessing(Program prog) {
		//Step0: Find ASPEN prediction function.
		List<Procedure> procList = IRTools.getProcedureList(prog);
		for( Procedure tProc : procList ) {
			if( tProc.getSymbolName().equals(ASPENPredictFuncName) ) {
				ASPENPredictFunc = tProc;
				break;
			}
		}
		if( ASPENPredictFunc == null ) {
			Tools.exit("\n[ERROR in ASPENModelGen.ASPENPostProcessing()] This transform pass is skipped " +
					"since the compiler can not find the ASPEN prediction function (" + ASPENPredictFuncName +
					"); exit.\n");
		}
		for( VariableDeclaration tDecl : (List<VariableDeclaration>)ASPENPredictFunc.getParameters() ) {
			tuningParameters.add(tDecl.getDeclarator(0).getID().toString());
		}
		Set<String> searchKeys = new HashSet<String>();
		searchKeys.addAll(ACCAnnotation.OpenACCDirectivesWithConditional);
		searchKeys.add("wait");
		FlatIterator Fiter = new FlatIterator(prog);
		while (Fiter.hasNext())
		{
			TranslationUnit cTu = (TranslationUnit)Fiter.next();
			List<ACCAnnotation> accAnnots = 
				AnalysisTools.collectPragmas(cTu, ACCAnnotation.class, searchKeys, false);
			if( (accAnnots != null) && (!accAnnots.isEmpty()) ) {
				TransformTools.addExternProcedureDeclaration(ASPENPredictFunc, cTu);
				for( ACCAnnotation aAnnot : accAnnots ) {
					Annotatable at = aAnnot.getAnnotatable();
					Procedure cProc = IRTools.getParentProcedure(at);
					SymbolTable symTable = null;
					Traversable tt = at.getParent();
					while ( (tt != null) && !(tt instanceof SymbolTable) ) {
						tt = tt.getParent();
					}
					if( tt != null ) {
						symTable = (SymbolTable)tt;
					}
					FunctionCall ASPENPredictFuncCall = new FunctionCall(new NameID(ASPENPredictFuncName));
					for( String tParam : tuningParameters ) {
						VariableDeclaration symDecl = (VariableDeclaration)SymbolTools.findSymbol(symTable, tParam);
						IDExpression argID = null;
						if( symDecl != null ) {
							for( int i=0; i<symDecl.getNumDeclarators(); i++ ) {
								Declarator tDeclr = symDecl.getDeclarator(i);
								if( tDeclr.getID().toString().equals(tParam) ) {
									argID = new Identifier((Symbol)tDeclr);
									break;
								}
							}
						}
						if( argID == null ) {
							Tools.exit("[ERROR in ASPENModelGen.ASPENPostProcessing()] an argument variable (" + tParam +
									") needed for " + ASPENPredictFuncName + "() function is not visible in the current scope; exit.\n" +
											"Enclosing Procedure: " + cProc.getSymbolName() + "\nOpenACC Annotation: " +
											aAnnot + "\nEnclosing Translation Unit: " + cTu.getOutputFilename() + "\n");
						} else {
							List<Specifier> typeList = new ArrayList<Specifier>(1);
							typeList.add(Specifier.DOUBLE);
							ASPENPredictFuncCall.addArgument(new Typecast(typeList, argID));
						}
					}
					boolean isWaitDirective = true;
					if( aAnnot.containsKey("wait") ) {
						for( String tDir : ACCAnnotation.OpenACCDirectivesWithWait ) {
							if( aAnnot.containsKey(tDir) ) {
								isWaitDirective = false;
								break;
							}
						}
					} else {
						isWaitDirective = false;
					}
					if( isWaitDirective ) {
						CompoundStatement ifBody = new CompoundStatement();
						IfStatement ifStmt = new IfStatement(ASPENPredictFuncCall.clone(), ifBody);
						ifStmt.swapWith((Statement)at);
						ifBody.addStatement((Statement)at);
					} else {
						if( aAnnot.containsKey("if") ) {
							Expression ifCond = aAnnot.get("if");
							ifCond = new BinaryExpression(ASPENPredictFuncCall.clone(), BinaryOperator.LOGICAL_AND,
									ifCond);
							aAnnot.put("if", ifCond);
						} else {
							aAnnot.put("if", ASPENPredictFuncCall.clone());
						}
					}
				}
			}
		}

	}
	
	public static void cleanASPENAnnotations(Traversable tt) {
		List<ASPENAnnotation> aspenAnnots = IRTools.collectPragmas(tt, ASPENAnnotation.class, "control");
		if( aspenAnnots != null ) {
			for( ASPENAnnotation aAnnot : aspenAnnots ) {
				Annotatable at = aAnnot.getAnnotatable();
				if( (aAnnot.size() == 2) || ((aAnnot.size() == 3) && (aAnnot.containsKey("label"))) ) {
					//Remove the empty ASPEN annotation.
					List<ASPENAnnotation> cAnnots = at.getAnnotations(ASPENAnnotation.class);
					at.removeAnnotations(ASPENAnnotation.class);
					for( ASPENAnnotation tAnnot : cAnnots ) {
						if( !tAnnot.equals(aAnnot) ) {
							at.annotate(tAnnot);
						}
					}
				} else if( ((aAnnot.size() == 3) && aAnnot.containsKey("execute")) || 
						((aAnnot.size() == 4) && aAnnot.containsKey("execute") && aAnnot.containsKey("label")) )  {
					//Replace the "execute" clause with "ignore" clause.
					aAnnot.remove("execute");
					aAnnot.put("ignore", "_clause");
					aAnnot.remove("label");
				} else if( (aAnnot.size() == 4) && aAnnot.containsKey("label") && aAnnot.containsKey("ignore") ) {
					//#pragma aspen control label(string) ignore.
					aAnnot.remove("label");
				}
			}
		}
	}

	/* (non-Javadoc)
	 * @see cetus.transforms.TransformPass#start()
	 */
	@Override
	public void start() {
		/////////////////////////////////////////////////////////////////////////////////
		//Step0: Parse ASPENModelGen options, and initialize internal data structures. //
		/////////////////////////////////////////////////////////////////////////////////
		String mainEntryFunc = ASPENModelAnalysis.ASPENConfiguration.mainEntryFunc;
		String modelName = ASPENModelAnalysis.ASPENConfiguration.modelName;
		
		main = AnalysisTools.findMainEntryFunction(program, mainEntryFunc);
		if( main == null ) {
			Tools.exit("\n[ERROR in ASPENModelGen] This transform pass is skipped " +
					"since the compiler can not find the main entry function; " +
					"if the input program does not have a main function, user should specify a main entry function " +
					"using \"SetAccEntryFunction\" option.\n");
			return;
		}
		if( modelName == null ) {
			//If no model name is given, use the name of the file containing main function.
			//DEBUG: TranslationUnit.getInputFilename() may include path name, 
			//but default TranslationUnit.output_filename does not have path name.
			modelName = ((TranslationUnit)main.getParent()).getOutputFilename();
			int dotIndex = modelName.indexOf('.'); 
			if( dotIndex != -1 ) {
				modelName = modelName.substring(0, dotIndex).trim();
			}
		}
		mainTrUnt = (TranslationUnit)main.getParent(); 
		/////////////////////////////////////////////
		//Step1: Add global ASPEN parameters/data. //
		/////////////////////////////////////////////
		aspenModel = new ASPENModel(new NameID(modelName));
		aspenModel.setEntryFuncID(main.getName());
		List<Traversable> trList = program.getChildren();
		for( Traversable trUnt : trList ) {
			List<Traversable> declList = trUnt.getChildren();
			for( Traversable tr : declList ) {
				if( tr instanceof AnnotationDeclaration ) {
					AnnotationDeclaration annot = (AnnotationDeclaration)tr;
					ASPENAnnotation aspenAnnot = annot.getAnnotation(ASPENAnnotation.class, "param");
					if( aspenAnnot != null ) {
						Set<ASPENParam> paramSet = aspenAnnot.get("param");
						for( ASPENParam aParam : paramSet ) {
							ASPENParamDeclaration paramDecl = new ASPENParamDeclaration(aParam);
							IDExpression ID = paramDecl.getDeclaredID();
							if( !aspenModel.containsParam(ID) ) {
								aspenModel.addASPENDeclaration(paramDecl);
								aspenDeclToIRMap.put(paramDecl, tr);
							} else {
								if( !ASPENModelAnalysis.isInternalParam(ID.getName()) ) {
									PrintTools.println("\n[WARNING in ASPENModelGen] duplicate ASPEN parameter is found:\n" +
											"ASPEN Parameter: " + aParam.toString() + 
											" Enclosing Translation Unit: " + ((TranslationUnit)trUnt).getOutputFilename() +"\n", 2);
								}
								ASPENParamDeclaration cParamDecl = aspenModel.getParamDeclaration(ID);
								ASPENParam tParam = cParamDecl.getASPENParam();
								if( tParam.getInitVal() == null ) {
									ASPENDeclaration aspenDecl = aspenModel.removeParam(ID);
									if( aspenDecl != null ) {
										Traversable IRtr = aspenDeclToIRMap.get(aspenDecl);
										if( IRtr != null ) {
											Traversable IRtrP = IRtr.getParent();
											if( IRtrP != null ) {
												IRtrP.removeChild(IRtr);
											}
										}
										aspenModel.addASPENDeclaration(paramDecl);
										aspenDeclToIRMap.put(paramDecl, tr);
									}
								}
							}
						}
					}
					aspenAnnot = annot.getAnnotation(ASPENAnnotation.class, "data");
					if( aspenAnnot != null ) {
						Set<ASPENData> dataSet = aspenAnnot.get("data");
						for( ASPENData aData : dataSet ) {
							ASPENDataDeclaration dataDecl = new ASPENDataDeclaration(aData);
							IDExpression ID = dataDecl.getDeclaredID();
							if( !aspenModel.containsData(ID) ) {
								aspenModel.addASPENDeclaration(dataDecl);
								aspenDeclToIRMap.put(dataDecl, tr);
							} else {
								PrintTools.println("\n[WARNING in ASPENModelGen] duplicate ASPEN data is found:\n" +
										"ASPEN Data: " + aData.toString() + 
										"\tEnclosing Translation Unit: " + ((TranslationUnit)trUnt).getOutputFilename() +"\n", 2);
								ASPENDataDeclaration cDataDecl = aspenModel.getDataDeclaration(ID);
								ASPENData tData = cDataDecl.getASPENData();
								if( (tData.getCapacity() == null) && (tData.getTraitSize() == 0) ) {
									ASPENData nData = dataDecl.getASPENData();
									if( nData.getCapacity() != null ) {
										tData.setCapacity(nData.getCapacity().clone());
									} 
									if( nData.getTraitSize() > 0 ) {
										for( ASPENTrait tTrait : nData.getTraits() ) {
											tData.addTrait(tTrait.clone());
										}
									} 
									
									//[DEBUG] Don't replace the current data directive to the new one, since
									//current Cetus implementation keeps only the first symbol in the symbol table
									//if both global and extern symbols exist.
/*									ASPENDeclaration aspenDecl = aspenModel.removeData(ID);
									if( aspenDecl != null ) {
										Traversable IRtr = aspenDeclToIRMap.get(aspenDecl);
										if( IRtr != null ) {
											Traversable IRtrP = IRtr.getParent();
											if( IRtrP != null ) {
												IRtrP.removeChild(IRtr);
											}
										}
										aspenModel.addASPENDeclaration(dataDecl);
										aspenDeclToIRMap.put(dataDecl, tr);
									}*/
								}
							}
						}
					}
				}
			}
		}
		
		///////////////////////////////////
		//Step2: Generate ASPEN kernels. //
		///////////////////////////////////
		// generate a list of procedures in post-order traversal
		//CallGraph callgraph = new CallGraph(program);
		CallGraph callgraph = new CallGraph(program, mainEntryFunc);
		// procedureList contains Procedure in ascending order; the last one is main
		List<Procedure> procedureList = callgraph.getTopologicalCallList();
		
		/////////////////////////////////////////////////////////////////////////////////////////////////
		//Step2-1: If a function has an ignore clause, all functions called from the function are also //
		//ignored, unless they are called by another function without an ignore clause.                //
		/////////////////////////////////////////////////////////////////////////////////////////////////
		Map<Procedure, CallGraph.Node> callgraphMap = callgraph.getCallGraph();
		List<Procedure> ignoredProcs = new LinkedList<Procedure>();
		Set<Procedure> visitedProcs = new HashSet<Procedure>();
		CallGraph.Node pNode = null;
		Procedure tpProc = null;
		int itrCnt = 0;
		boolean isChanged = false;
		do {
			ignoredProcs.clear();
			visitedProcs.clear();
			isChanged = false;
			ignoredProcs.add(main);
			while( !ignoredProcs.isEmpty() ) {
				tpProc = ignoredProcs.remove(0);
				visitedProcs.add(tpProc);
				pNode = callgraphMap.get(tpProc); 
				List<Procedure> callees = pNode.getCallees();
				for( Procedure tcProc : callees ) {
					boolean ignoreThis = false;
					if( tcProc.containsAnnotation(ASPENAnnotation.class, "ignore") ) {
						if( !visitedProcs.contains(tcProc) ) {
							ignoredProcs.add(tcProc);
							visitedProcs.add(tcProc);
						}
						ignoreThis = true;
					} else {
						List<ProcedureDeclarator> procDeclrList = AnalysisTools.getProcedureDeclarators(tcProc);
						for( ProcedureDeclarator tProcDeclr : procDeclrList ) {
							if( tProcDeclr.getID().equals(tcProc.getName()) ) {
								//Found procedure declarator for tcProc.
								Declaration tProcDecl = tProcDeclr.getDeclaration();
								if( (tProcDecl != null) && (tProcDecl.containsAnnotation(ASPENAnnotation.class, "ignore")) ) {
									if( !visitedProcs.contains(tcProc) ) {
										ignoredProcs.add(tcProc);
										visitedProcs.add(tcProc);
									}
									ignoreThis = true;
								}
								break;
							}
						}
					}
					if( !ignoreThis ) {
						//If all caller procedures are ignored, this procedure can be ignored too.
						CallGraph.Node cNode = callgraphMap.get(tcProc);
						List<CallGraph.Caller> callers = cNode.getCallers();
						boolean allParentIgnored = true;
						for( CallGraph.Caller tCaller : callers ) {
							Procedure pProc = tCaller.getCallingProc();
							if( !pProc.containsAnnotation(ASPENAnnotation.class, "ignore") ) {
								Statement callerStmt = tCaller.getCallSite();
								if( !callerStmt.containsAnnotation(ASPENAnnotation.class, "ignore") ) {
									List<ProcedureDeclarator> procDeclrList = AnalysisTools.getProcedureDeclarators(pProc);
									boolean procDeclExist = false;
									for( ProcedureDeclarator tProcDeclr : procDeclrList ) {
										if( tProcDeclr.getID().equals(pProc.getName()) ) {
											//Found procedure declarator for pProc.
											procDeclExist = true;
											Declaration tProcDecl = tProcDeclr.getDeclaration();
											if( (tProcDecl != null) && (!tProcDecl.containsAnnotation(ASPENAnnotation.class, "ignore")) ) {
												allParentIgnored = false;
											}
											break;
										}
									}
									if( !procDeclExist )  {
										allParentIgnored = false;
									}

								}
							}
							if( !allParentIgnored ) {
								break;
							}
						}
						if( allParentIgnored ) {
							//Ignore this procedure, tcProc.
							ASPENAnnotation tAnnot = tcProc.getAnnotation(ASPENAnnotation.class, "control");
							if( tAnnot == null ) {
								tAnnot = new ASPENAnnotation();
								tAnnot.put("control", "_directive");
								tcProc.annotate(tAnnot);
							}
							if( !tAnnot.containsKey("ignore") ) {
								tAnnot.put("ignore", "_clause");
								isChanged = true;
								if( !visitedProcs.contains(tcProc) ) {
									ignoredProcs.add(tcProc);
									visitedProcs.add(tcProc);
								}
							}
						}
					}
				}
			}
			itrCnt++;
		} while( isChanged && (itrCnt++ < maxLoopCnt) );
		if( itrCnt >= maxLoopCnt ) {
			PrintTools.println("[WARNING in ASPENModelGen.start()] " +
					"Loop to annotate functions to ignore seems to iterate infinitely.", 0);
		}
		
		/* drive the engine; visit every procedure */
		IDExpression mainID = new NameID("main");
		for (Procedure proc : procedureList)
		{
			exeBlockCnt = 0;
			ASPENAnnotation aspenAnnot = proc.getAnnotation(ASPENAnnotation.class, "ignore");
			if( aspenAnnot == null ) {
				//PrintTools.println("[INFO in ASPENModelGen] Current procedure: \n" + proc + "\n", 0);
				List<ProcedureDeclarator> procDeclrList = AnalysisTools.getProcedureDeclarators(proc);
				CompoundStatement pBody = proc.getBody();
				IDExpression kernelID = new NameID(proc.getSymbolName());
				if( !aspenModel.containsKernel(kernelID) && !ignoredKernels.contains(kernelID) && 
						ASPENModelAnalysis.ASPENConfiguration.accessedFunctions.contains(kernelID.getName()) ) {
					ASPENCompoundStatement aspenCStmt = new ASPENCompoundStatement();
					if( !kernelID.toString().equals(main.getSymbolName()) ) {
						inASPENModelRegion = true;
					} else if( ASPENModelAnalysis.ASPENConfiguration.modelRegionIsFunction ) {
						inASPENModelRegion = true;
					} else {
						inASPENModelRegion = false;
					}
					fillASPENStatements(aspenCStmt, pBody, proc, procDeclrList);
					if( aspenCStmt.isEmpty() ) {
						ignoredKernels.add(kernelID.clone());
					} else {
						//[CAUTION] The current ASPEN model assumes that the name of an entry kernel is "main".
						//Therefore, if the entry kernel should be renamed to "main".
						if( kernelID.equals(main.getName()) && !kernelID.equals(mainID) ) {
							kernelID = mainID.clone();
						}
						ASPENKernel aProc = new ASPENKernel(kernelID, null, aspenCStmt);
						aspenModel.addASPENDeclaration(aProc);
					}
				}
			}
		}
		
		/////////////////////////////////////////////////////////////////
		//Step3: Remove parameters/data that are not used in the model //
		/////////////////////////////////////////////////////////////////
		List<FunctionCall> gFuncCallList = IRTools.getFunctionCalls(program);
		Set<SomeExpression> accessedSomeExps = new HashSet<SomeExpression>();
		Map<Symbol, Symbol> dataGSymMap = new HashMap<Symbol, Symbol>();
		Map<Symbol, Symbol> paramSymMap = null;
		Map<Symbol, Symbol> accParamSymMap = new HashMap<Symbol, Symbol>();
		Map<Symbol, Symbol> accessedGSymMap = null;
		Map<Symbol, Procedure> formalSym2ProcMap = null;
		Set<Symbol> indirectParamUseSet = new HashSet<Symbol>();
		isChanged = false;
		itrCnt = 0;
		do {
			isChanged = false;
			accessedGSymMap = new HashMap<Symbol, Symbol>();
			formalSym2ProcMap = new HashMap<Symbol, Procedure>();
			paramSymMap = new HashMap<Symbol, Symbol>(); //contains mapping of the current iteration.
			Set<Symbol> accessedSymbols = new HashSet<Symbol>();
			Set<NameID> accessedNameIDs = new HashSet<NameID>();
			ASPENParam tParam = null;
			ASPENData tData = null;
			Expression ttExp = null;
			DFIterator<Traversable> model_iter =
					new DFIterator<Traversable>(aspenModel);
			model_iter.pruneOn(ASPENParam.class);
			model_iter.pruneOn(ASPENData.class);
			while (model_iter.hasNext()) {
				Traversable t = model_iter.next();
				if( t instanceof ASPENParam ) {
					tParam = (ASPENParam)t;
					Symbol pSym = SymbolTools.getSymbolOf(tParam.getID());
					Symbol gPSym = null;
					if( pSym != null ) {
						ASPENDeclaration pDecl = aspenModel.getASPENDeclaration(pSym);
						Traversable pt = aspenDeclToIRMap.get(pDecl);
						Procedure ttProc = IRTools.getParentProcedure(pt);
						if( SymbolTools.isFormal(pSym) ) {
							//System.err.println("Formal parameter found: " + pSym.getSymbolName() + " in " + ttProc.getSymbolName());
							List symbolInfo = new ArrayList(2);
							if( AnalysisTools.SymbolStatus.SrcScalarSymbolFound(
									AnalysisTools.findSrcScalarSymbol(pSym, pt, true, null, symbolInfo, gFuncCallList))) {
								gPSym = (Symbol)symbolInfo.get(0);
								//System.err.println("Source parameter found: " + pSym.getSymbolName() + " : " + gPSym.getSymbolName());
								if( (gPSym != null) && !pSym.equals(gPSym) ) {
									paramSymMap.put(pSym, gPSym);
									//System.err.println("Add paramSymMap for parameter found: (" + pSym.getSymbolName() + ", " + gPSym.getSymbolName() + ")");
								}
							}
						}
					}
					ttExp = tParam.getInitVal();
					if( ttExp != null ) {
						accessedSymbols.addAll(SymbolTools.getAccessedSymbols(ttExp));
						accessedNameIDs.addAll(IRTools.getExpressionsOfType(ttExp, NameID.class));
						accessedSomeExps.addAll(IRTools.getExpressionsOfType(ttExp, SomeExpression.class));
						/*					List<NameID> tNameList = IRTools.getExpressionsOfType(ttExp, NameID.class);
					if( !tNameList.isEmpty() ) {
						PrintTools.println("Case1: current t = "+ t, 0);
						PrintTools.println("Name IDs: " + PrintTools.listToString(tNameList, ","), 0);
					}*/
					}
				} else if( t instanceof ASPENData ) {
					tData = (ASPENData)t;
					ttExp = tData.getCapacity();
					if( ttExp != null ) {
						accessedSymbols.addAll(SymbolTools.getAccessedSymbols(ttExp));
						accessedNameIDs.addAll(IRTools.getExpressionsOfType(ttExp, NameID.class));
						accessedSomeExps.addAll(IRTools.getExpressionsOfType(ttExp, SomeExpression.class));
					}
					int nTraits = tData.getTraitSize();
					if( nTraits > 0 ) {
						for( int i=0; i<nTraits; i++ ) {
							accessedSymbols.addAll(SymbolTools.getAccessedSymbols(tData.getTrait(i)));
							accessedNameIDs.addAll(IRTools.getExpressionsOfType(tData.getTrait(i), NameID.class));
							accessedSomeExps.addAll(IRTools.getExpressionsOfType(tData.getTrait(i), SomeExpression.class));
						}
					}
				} else if( t instanceof Identifier ) {
					accessedSymbols.add(((Identifier)t).getSymbol());
				} else if( t instanceof NameID ) {
					accessedNameIDs.add((NameID)t);
				} else if( t instanceof SomeExpression ) {
					accessedSomeExps.add((SomeExpression)t);
				}
			}
			Set<Symbol> accessedGSymbols = new HashSet<Symbol>(accessedSymbols.size());
			ASPENDeclaration aDecl = null;
			Traversable t = null;
			Symbol gSym = null;
			List osymList = new ArrayList(2);
			for( Symbol aSym : accessedSymbols ) {
				Procedure tProc = null;
				if( SymbolTools.isFormal(aSym) ) {
					//Procedure parameter is not traversable from the main IR tree.
					//We have to find a procedure manually.
					tProc = AnalysisTools.findProcedureForFormalSymol(program, aSym);
					if( tProc != null ) {
						formalSym2ProcMap.put(aSym, tProc);
					}
				}
				gSym = null;
				aDecl = aspenModel.getASPENDeclaration(aSym);
				if( aDecl != null ) {
					t = aspenDeclToIRMap.get(aDecl);
					if( t != null ) {
						osymList.clear();
						if( AnalysisTools.SymbolStatus.OrgSymbolFound(
								AnalysisTools.findOrgSymbol(aSym, t, true, null, osymList, gFuncCallList)) ) {
							gSym = (Symbol)osymList.get(0);
							accessedGSymbols.add(gSym);
							accessedGSymMap.put(aSym, gSym);
						}
					}
				} else {
					if( SymbolTools.isFormal(aSym) ) {
						if( tProc != null ) {
							osymList.clear();
							if( AnalysisTools.SymbolStatus.OrgSymbolFound(
									AnalysisTools.findOrgSymbol(aSym, tProc, true, null, osymList, gFuncCallList)) ) {
								gSym = (Symbol)osymList.get(0);
								accessedGSymbols.add(gSym);
								accessedGSymMap.put(aSym, gSym);
							}
						}
					} else {
						Declaration tDecl = aSym.getDeclaration();
						if( tDecl != null ) {
							osymList.clear();
							if( AnalysisTools.SymbolStatus.OrgSymbolFound(
									AnalysisTools.findOrgSymbol(aSym, tDecl, true, null, osymList, gFuncCallList)) ) {
								gSym = (Symbol)osymList.get(0);
								accessedGSymbols.add(gSym);
								accessedGSymMap.put(aSym, gSym);
							}
						}
					}
				}
				if( gSym == null ) {
					accessedGSymbols.add(aSym);
					accessedGSymMap.put(aSym, aSym);
				}
			}
			//PrintTools.println("Accessed Org Symbols: " + PrintTools.collectionToString(accessedGSymbols, ","), 0);
			//PrintTools.println("Accessed NameIDs: " + PrintTools.collectionToString(accessedNameIDs, ","), 0);
			//Handle ASPEN Data.
			Set<Symbol> tSymbols = new HashSet<Symbol>();
			Set<Symbol> tGSymbols = new HashSet<Symbol>();
			tSymbols.addAll(aspenModel.getDataSymbols());
			//PrintTools.println("Accessed Data Symbols: " + PrintTools.collectionToString(tSymbols, ","), 0);
			for( Symbol aSym : tSymbols ) {
				gSym = null;
				aDecl = aspenModel.getASPENDeclaration(aSym);
				if( aDecl != null ) {
					t = aspenDeclToIRMap.get(aDecl);
					if( t != null ) {
						gSym = dataGSymMap.get(aSym);
						if( gSym == null ) {
							osymList.clear();
							if( AnalysisTools.SymbolStatus.OrgSymbolFound(
									AnalysisTools.findOrgSymbol(aSym, t, true, null, osymList, gFuncCallList)) ) {
								gSym = (Symbol)osymList.get(0);
								tGSymbols.add(gSym);
								dataGSymMap.put(aSym, gSym);
							}
						} else {
							tGSymbols.add(gSym);
						}
					}
				}
				if( gSym == null ) {
					tGSymbols.add(aSym);
					dataGSymMap.put(aSym, aSym);
				}
			}
			ASPENDeclaration aspenDecl = null;
			Traversable IRtr = null;
			Traversable IRtrP = null;
			for( Symbol pSym : tGSymbols ) {
				if( !accessedGSymbols.contains(pSym) ) {
					aspenDecl = aspenModel.removeDataSymbol(pSym);
					if( aspenDecl != null ) {
						IRtr = aspenDeclToIRMap.get(aspenDecl);
						if( IRtr != null ) {
							IRtrP = IRtr.getParent();
							if( IRtrP != null ) {
								IRtrP.removeChild(IRtr);
								isChanged = true;
							}
						}
					}
				}
			}
			//Handle ASPEN Parameters.
			tSymbols.clear();
			tGSymbols.clear();
			tSymbols.addAll(aspenModel.getParamSymbols());
			//[DEBUG] below set should be global since the indirect parameter access can be checked twice if the original argument is a literal.
			//Set<Symbol> indirectParamUseSet = new HashSet<Symbol>();
			Set<Symbol> removeSet = new HashSet<Symbol>();
			for( Symbol aSym : tSymbols ) {
				gSym = null;
				aDecl = aspenModel.getASPENDeclaration(aSym);
				if( aDecl != null ) {
					t = aspenDeclToIRMap.get(aDecl);
					if( t != null ) {
						osymList.clear();
						if( AnalysisTools.SymbolStatus.OrgSymbolFound(
								AnalysisTools.findOrgSymbol(aSym, t, true, null, osymList, gFuncCallList)) ) {
							gSym = (Symbol)osymList.get(0);
							tGSymbols.add(gSym);
						}
					}
				}
				if( gSym == null ) {
					tGSymbols.add(aSym);
				}
			}
			for( Symbol pSym : tGSymbols ) {
				if( accessedGSymbols.contains(pSym) ) {
					if( paramSymMap.containsKey(pSym) ) {
						Symbol oSym = paramSymMap.get(pSym);
						PrintTools.println("[INFO in ASPENModelGen] Rename a formal parameter, " + pSym.getSymbolName() + ", to " + oSym.getSymbolName(), 1);
						indirectParamUseSet.add(oSym);
						removeSet.add(pSym);
						if( SymbolTools.isFormal(pSym) ) {
							Procedure tProc = formalSym2ProcMap.get(pSym);
							if( tProc != null ) {
								SymbolTools.setSymbolName(pSym, oSym.getSymbolName(), tProc);
								//System.err.println("current procedure: " + tProc.getSymbolName());
							}
						} else {
							Traversable tt = pSym.getDeclaration();
							SymbolTable symTable = null;
							while ( (tt != null) && !(tt instanceof SymbolTable) ) {
								if( tt instanceof SymbolTable ) {
									if( ((SymbolTable)tt).containsSymbol(pSym) ) {
										symTable = (SymbolTable)tt;
										break;
									} else {
										tt = tt.getParent();
									}
								}
							}
							if( symTable != null ) {
								SymbolTools.setSymbolName(pSym, oSym.getSymbolName(), symTable);
							}
						}
						aspenDecl = aspenModel.removeParamSymbol(pSym);
						if( aspenDecl != null ) {
							IRtr = aspenDeclToIRMap.get(aspenDecl);
							if( IRtr != null ) {
								IRtrP = IRtr.getParent();
								if( IRtrP != null ) {
									IRtrP.removeChild(IRtr);
									isChanged = true;
								}
							}
						}
					}
				} 
			}
			tGSymbols.removeAll(removeSet);
			for( Symbol pSym : tGSymbols ) {
				if( !accessedGSymbols.contains(pSym) ) {
					boolean mayBeUsed = false;
					if( !indirectParamUseSet.contains(pSym) ) {
						for( SomeExpression tSome : accessedSomeExps ) {
							if( tSome.toString().contains(pSym.getSymbolName()) ) {
								mayBeUsed = true;
								break;
							}
						}
						if( !mayBeUsed ) {
							aspenDecl = aspenModel.removeParamSymbol(pSym);
							if( aspenDecl != null ) {
								IRtr = aspenDeclToIRMap.get(aspenDecl);
								if( IRtr != null ) {
									IRtrP = IRtr.getParent();
									if( IRtrP != null ) {
										IRtrP.removeChild(IRtr);
										isChanged = true;
									}
								}
							}
						}
					}
				}
			}
			Set<IDExpression> tNameIDs = new HashSet<IDExpression>();
			tNameIDs.addAll(aspenModel.getInternalParamIDs());
			for( IDExpression tNameID : tNameIDs ) {
				if( !accessedNameIDs.contains(tNameID) ) {
					boolean mayBeUsed = false;
					String tNameStr = tNameID.toString();
					if( tNameStr.contains("aspen_param_type_") ) {
						for( SomeExpression tSome : accessedSomeExps ) {
							if( tSome.toString().contains(tNameID.toString()) ) {
								mayBeUsed = true;
								break;
							}
						}
					}
					//Remove unused internal Aspen parameters only if they represent type sizes.
					if( !mayBeUsed && tNameStr.contains("aspen_param_sizeof_") ) {
						aspenDecl = aspenModel.removeParam(tNameID);
						if( aspenDecl != null ) {
							IRtr = aspenDeclToIRMap.get(aspenDecl);
							if( IRtr != null ) {
								IRtrP = IRtr.getParent();
								if( IRtrP != null ) {
									IRtrP.removeChild(IRtr);
									isChanged = true;
								}
							}
						}
					}
				}
			}
			accParamSymMap.putAll(paramSymMap);
			//PrintTools.println("Accessed Some-Expressions: " + accessedSomeExps, 0);
		} while( isChanged && (itrCnt++ < maxLoopCnt) );
		if( itrCnt >= maxLoopCnt ) {
			PrintTools.println("[WARNING in ASPENModelGen.start()] " +
					"Loop to remove unused ASPEN variables seems to iterate infinitely.", 0);
		}
		
		//System.err.println("accParamSymMap: " + accParamSymMap);
		
		///////////////////////////////////////////////////////
		//Step4: Check parameters/data with duplicate names. //
		///////////////////////////////////////////////////////
		boolean duplicatedNameFound = false;
		Set<String> duNames = new HashSet<String>();
		Set<Symbol> paramSet = aspenModel.getParamSymbols();
		Set<Symbol> dSymSet = null;
		Map<String, Set<Symbol>> duplicatedParamMap = new HashMap<String, Set<Symbol>>();
		for( Symbol paramSym : paramSet ) {
			String pName = paramSym.getSymbolName();
			if( duplicatedParamMap.containsKey(pName) ) {
				dSymSet = duplicatedParamMap.get(pName);
				dSymSet.add(paramSym);
				duplicatedNameFound = true;
				duNames.add(pName);
			} else {
				dSymSet = new HashSet<Symbol>();
				dSymSet.add(paramSym);
				duplicatedParamMap.put(pName, dSymSet);
			}
		}
		if( duplicatedNameFound ) {
			PrintTools.println("[INFO in ASPENModelGen] parameters with duplicated names are found: " +
					duNames.toString()+".\nThese names will be renamed for correct ASPEN-model generation.", 0);
			for( Set<Symbol> tSymSet : duplicatedParamMap.values() ) {
				if( tSymSet.size() > 1 ) {
					Set<String> symNameSet = new HashSet<String>();
					for( Symbol tSym : tSymSet ) {
						String symNameBase = tSym.getSymbolName();
						String symName = symNameBase;
						ASPENDeclaration aDecl = aspenModel.getASPENDeclaration(tSym);
						Traversable tIR = aspenDeclToIRMap.get(aDecl);
						SymbolTable symTable = IRTools.getAncestorOfType(tIR, SymbolTable.class);
						Procedure pProc = null;
						while( tIR != null ) {
							if( tIR instanceof Procedure ) {
								pProc = (Procedure)tIR;
								break;
							}
							tIR = tIR.getParent();
						}
						if( (symTable != null) ) {
							if( pProc != null ) {
								symNameBase = symNameBase + "_" + pProc.getSymbolName();
							}
							symName = symNameBase;
							int i = 0;
							while(true) {
								if( !symNameSet.contains(symName) ) {
									SymbolTools.setSymbolName(tSym, symName, symTable);
									symNameSet.add(symName);
									break;
								}
								symName = symNameBase + i++;
							}
						} else {
							symNameSet.add(symName);
						}
						if( accParamSymMap.containsValue(tSym) ) {
							//If original argument parameter is renamed, corresponding formal parameter 
							//should be renamed too.
							for( Symbol iSym : accParamSymMap.keySet() ) {
								if( accParamSymMap.get(iSym).equals(tSym) ) {
									Procedure ttProc = formalSym2ProcMap.get(iSym);
									SymbolTools.setSymbolName(iSym, symName, ttProc);
								}
							}
						}
					}
				}
			}
		}
		duplicatedNameFound = false;
		duNames = new HashSet<String>();
		Set<Symbol> dataSet = aspenModel.getDataSymbols();
		Map<String, Set<Symbol>> duplicatedDataMap = new HashMap<String, Set<Symbol>>();
		for( Symbol dataSym : dataSet ) {
			String pName = dataSym.getSymbolName();
			if( duplicatedDataMap.containsKey(pName) ) {
				dSymSet = duplicatedDataMap.get(pName);
				dSymSet.add(dataSym);
				duplicatedNameFound = true;
				duNames.add(pName);
			} else {
				dSymSet = new HashSet<Symbol>();
				dSymSet.add(dataSym);
				duplicatedDataMap.put(pName, dSymSet);
			}
		}
		if( duplicatedNameFound ) {
			PrintTools.println("[INFO in ASPENModelGen] data with duplicated names are found: " +
					duNames.toString()+".\nThese names will be renamed for correct ASPEN-model generation.", 0);
			for( Set<Symbol> tSymSet : duplicatedDataMap.values() ) {
				if( tSymSet.size() > 1 ) {
					Set<String> symNameSet = new HashSet<String>();
					for( Symbol tSym : tSymSet ) {
						String symNameBase = tSym.getSymbolName();
						String symName = symNameBase;
						ASPENDeclaration aDecl = aspenModel.getASPENDeclaration(tSym);
						Traversable tIR = aspenDeclToIRMap.get(aDecl);
						SymbolTable symTable = IRTools.getAncestorOfType(tIR, SymbolTable.class);
						Procedure pProc = null;
						while( tIR != null ) {
							if( tIR instanceof Procedure ) {
								pProc = (Procedure)tIR;
								break;
							}
							tIR = tIR.getParent();
						}
						if( (symTable != null) ) {
							if( pProc != null ) {
								symNameBase = symNameBase + "_" + pProc.getSymbolName();
							}
							if( (pProc != null) || (SymbolTools.containsSpecifier(tSym, Specifier.STATIC)) ) {
								//Local variables or static variables with the same name should be renamed.
								symName = symNameBase;
								int i = 0;
								while(true) {
									if( !symNameSet.contains(symName) ) {
										SymbolTools.setSymbolName(tSym, symName, symTable);
										symNameSet.add(symName);
										break;
									}
									symName = symNameBase + i++;
								}
							}
						} else {
							symNameSet.add(symName);
						}
					}
				}
			}
		}
		
		///////////////////////////////////////////////////////////////////////////////////////////////
		//Step5: Rename function parameter-type data symbols to the names of their original symbols. //
		///////////////////////////////////////////////////////////////////////////////////////////////
		Set<Symbol>	dSymbols = new HashSet<Symbol>();
		dSymbols.addAll(aspenModel.getDataSymbols());
		Traversable IRtr = null;
		Traversable IRtrP = null;
		for(Symbol dSym : dSymbols ) {
			Symbol gSym = dataGSymMap.get(dSym);
			if( (gSym != null) && !gSym.equals(dSym) && dSymbols.contains(gSym) ) {
				ASPENDeclaration aDecl = aspenModel.getASPENDeclaration(dSym);
				Traversable tIR = aspenDeclToIRMap.get(aDecl);
				SymbolTable symTable = IRTools.getAncestorOfType(tIR, SymbolTable.class);
				if( symTable != null ) {
					SymbolTools.setSymbolName(dSym, gSym.getSymbolName(), symTable);
					ASPENDeclaration aspenDecl = aspenModel.removeDataSymbol(dSym);
					if( aspenDecl != null ) {
						IRtr = aspenDeclToIRMap.get(aspenDecl);
						if( IRtr != null ) {
							IRtrP = IRtr.getParent();
							if( IRtrP != null ) {
								IRtrP.removeChild(IRtr);
							}
						}
					}
				}
			}
		}
		for( Symbol aSym : accessedGSymMap.keySet() ) {
			Symbol gSym = accessedGSymMap.get(aSym);
			if( (gSym != null) && !gSym.equals(aSym) && dSymbols.contains(gSym) ) {
				if( SymbolTools.isFormal(aSym) ) {
					Procedure tProc = formalSym2ProcMap.get(aSym);
					if( tProc != null ) {
						SymbolTools.setSymbolName(aSym, gSym.getSymbolName(), tProc);
					}
				} else {
					Traversable tt = aSym.getDeclaration();
					SymbolTable symTable = null;
					while ( (tt != null) && !(tt instanceof SymbolTable) ) {
						if( tt instanceof SymbolTable ) {
							if( ((SymbolTable)tt).containsSymbol(aSym) ) {
								symTable = (SymbolTable)tt;
								break;
							} else {
								tt = tt.getParent();
							}
						}
					}
					if( symTable != null ) {
						SymbolTools.setSymbolName(aSym, gSym.getSymbolName(), symTable);
					}
				}
			}
		}
		
		////////////////////////////////////////////////////////////////////////////////////
		//Step6: Rename parameter/data variables whose names are ASPEN-reserved keywords. //
		////////////////////////////////////////////////////////////////////////////////////
		Set<Symbol> formalSet = new HashSet<Symbol>();
		Set<Symbol> tSymSet = new HashSet<Symbol>();
		paramSet = aspenModel.getParamSymbols();
		for( Symbol pSym : paramSet ) {
			if( ASPENKeywords.contains(pSym.getSymbolName()) ) {
				tSymSet.add(pSym);
				//formal parameter symbol that was masked by pSym should be renamed too.
				for( Symbol fSym: accessedGSymMap.keySet() ) {
					if( fSym.getSymbolName().equals(pSym.getSymbolName()) && !fSym.equals(pSym) ) {
						formalSet.add(fSym);
					}
				}
			}
		}
		dataSet = aspenModel.getDataSymbols();
		for( Symbol pSym : dataSet ) {
			if( ASPENKeywords.contains(pSym.getSymbolName()) ) {
				tSymSet.add(pSym);
				//formal data symbol that was masked by pSym should be renamed too.
				for( Symbol fSym: accessedGSymMap.keySet() ) {
					if( fSym.getSymbolName().equals(pSym.getSymbolName()) && !fSym.equals(pSym) ) {
						formalSet.add(fSym);
					}
				}
			}
		}
		if( !tSymSet.isEmpty() ) {
			PrintTools.println("[INFO in ASPENModelGen] The following parameters/data variables will be renamed," +
					" since they are reserved keywords in ASPEN: " + AnalysisTools.symbolsToString(tSymSet, ", "), 0);
			for( Symbol tSym : tSymSet ) {
				String symName = tSym.getSymbolName() + "_renamed";
				ASPENDeclaration aDecl = aspenModel.getASPENDeclaration(tSym);
				Traversable tIR = aspenDeclToIRMap.get(aDecl);
				SymbolTable symTable = IRTools.getAncestorOfType(tIR, SymbolTable.class);
				SymbolTools.setSymbolName(tSym, symName, symTable);
/*				for( SomeExpression tSome : accessedSomeExps ) {
					if( tSome.toString().contains(tSym.getSymbolName()) ) {
					}
				}*/
			}
			for( Symbol tSym : formalSet ) {
				String symName = tSym.getSymbolName() + "_renamed";
				SymbolTable symTable = formalSym2ProcMap.get(tSym);
				if( symTable == null ) {
					symTable = AnalysisTools.findProcedureForFormalSymol(program, tSym);
				}
				SymbolTools.setSymbolName(tSym, symName, symTable);
			}
		}
		
		//////////////////////////////////////////////////////////////////////
		//Step7: convert scalar struct members to internal Aspen parameter. //
		//////////////////////////////////////////////////////////////////////
		DFIterator<AccessExpression> eiter = new DFIterator<AccessExpression>(aspenModel, AccessExpression.class);
		while(eiter.hasNext())
		{
			AccessExpression tAccExp = eiter.next();
			AccessSymbol tAccSym = new AccessSymbol(tAccExp);
			Symbol tMemSym = tAccSym.getMemberSymbol();
			if( SymbolTools.isArray(tMemSym) || SymbolTools.isPointer(tMemSym) ) {
				continue;
			}
			StringBuilder str = new StringBuilder(32);
			str.append("aspen_param_");
			str.append(TransformTools.buildAccessSymbolName(tAccSym));
			NameID nParamID = new NameID(str.toString());
			if( !aspenModel.containsParam(nParamID) ) {
				Set<ASPENParam> nParamSet = new HashSet<ASPENParam>();
				ASPENParam aParam = new ASPENParam(nParamID);
				nParamSet.add(aParam);
				ASPENAnnotation nParamAnnot = new ASPENAnnotation("declare", "_directive");
				nParamAnnot.put("param", nParamSet);
				AnnotationDeclaration nParamDecl = new AnnotationDeclaration(nParamAnnot);
				if( aspenDeclToIRMap.isEmpty() ) {
					mainTrUnt.addDeclarationBefore(mainTrUnt.getFirstDeclaration(), nParamDecl);
				} else {
					mainTrUnt.addDeclarationAfter(mainTrUnt.getFirstDeclaration(), nParamDecl);
				}
				ASPENParamDeclaration aspenParamDecl = new ASPENParamDeclaration(aParam.clone());
				aspenModel.addASPENDeclarationFirst(aspenParamDecl);
				aspenDeclToIRMap.put(aspenParamDecl, nParamDecl);
			}
			tAccExp.swapWith(nParamID.clone());
		}
		
		/////////////////////////////////////////////////////////////////////////////
		//Step8: Check uninitialized parameters and data without size information. //
		/////////////////////////////////////////////////////////////////////////////
		tSymSet = new HashSet<Symbol>();
		paramSet = aspenModel.getParamSymbols();
		for( Symbol pSym : paramSet ) {
			ASPENParamDeclaration pDecl = (ASPENParamDeclaration)aspenModel.getASPENDeclaration(pSym);
			if( pDecl.getASPENParam().getInitVal() == null ) {
				tSymSet.add(pSym);
			}
		}
		if( !tSymSet.isEmpty() ) {
			//Try a simple optimization to find an initial value.
			Set<Symbol> removeSet = new HashSet<Symbol>();
			for( Symbol pSym : tSymSet ) {
				ASPENDeclaration aDecl = aspenModel.getASPENDeclaration(pSym);
				Traversable tIR = aspenDeclToIRMap.get(aDecl);
				SymbolTable symTable = IRTools.getAncestorOfType(tIR, SymbolTable.class);
				CompoundStatement checkRegion = null;
				if( symTable instanceof TranslationUnit ) {
					checkRegion = main.getBody();
				} else if( symTable instanceof CompoundStatement ) {
					checkRegion = (CompoundStatement)symTable;
				} else {
					Traversable tt = symTable.getParent();
					while( (tt != null) && !(tt instanceof CompoundStatement) ) {
						tt = tt.getParent();
					}
					if( tt instanceof CompoundStatement ) {
						checkRegion = (CompoundStatement)tt;
					}
				}
				if( checkRegion != null ) {
					for( Traversable t : checkRegion.getChildren() ) {
						if( t instanceof ExpressionStatement ) {
							Set<Symbol> defSyms = DataFlowTools.getDefSymbol(t);
							if( defSyms.contains(pSym) ) {
								Expression exp = ((ExpressionStatement)t).getExpression();
								if( exp instanceof AssignmentExpression ) {
									Expression initExp = ((AssignmentExpression)exp).getRHS();
									while ( initExp instanceof AssignmentExpression ) {
										initExp = ((AssignmentExpression)initExp).getRHS();
									}
									Set<Symbol> useSyms = DataFlowTools.getUseSymbol(initExp);
									boolean nonParamExists = false;
									for( Symbol uSym : useSyms ) {
										if( !paramSet.contains(uSym) ) {
											nonParamExists = true;
											break;
										}
									}
									if( !nonParamExists ) {
										if( !IRTools.containsFunctionCall(initExp) ) {
											ASPENParamDeclaration pDecl = (ASPENParamDeclaration)aspenModel.getASPENDeclaration(pSym);
											pDecl.getASPENParam().setInitVal(initExp.clone());
											removeSet.add(pSym);
										}
									}
								}
								break;
							}
						}
					}
				}
			}
			tSymSet.removeAll(removeSet);
		}
		
		if( !tSymSet.isEmpty() ) {
			PrintTools.println("[WARNING in ASPENModelGen] The following parameters are not initialized. " +
							"Initial values should be provided either by annotating the input program or by " +
							"modifying the output ASPEN model.", 0);
			for( Symbol pSym : tSymSet ) {
				ASPENDeclaration aDecl = aspenModel.getASPENDeclaration(pSym);
				Traversable tIR = aspenDeclToIRMap.get(aDecl);
				Procedure pProc = IRTools.getParentProcedure(tIR);
				TranslationUnit pTrUnt = IRTools.getParentTranslationUnit(tIR);
				if( pProc == null ) {
					PrintTools.println("Symbol: " + pSym.getSymbolName() +"\tTranslationUnit: " + pTrUnt.getOutputFilename(), 0);
				} else {
					PrintTools.println("Symbol: " + pSym.getSymbolName() +"\tTranslationUnit: " + pTrUnt.getOutputFilename() + 
							"\tProcedure: " + pProc.getSymbolName(), 0);
				}
			}
		}
		
		tSymSet.clear();
		dataSet = aspenModel.getDataSymbols();
		for( Symbol pSym : dataSet ) {
			ASPENDataDeclaration pDecl = (ASPENDataDeclaration)aspenModel.getASPENDeclaration(pSym);
			ASPENData tData = pDecl.getASPENData();
			if( (tData.getCapacity() == null) && (tData.getTraitSize() == 0) ) {
				tSymSet.add(pSym);
			}
		}
		if( !tSymSet.isEmpty() ) {
			//Try to find data size information from OpenACC data clauses.
			//System.err.println("ASPEN Data without size information found: " + AnalysisTools.symbolsToString(tSymSet, ", "));
			List<ACCAnnotation> sharedAnnots = AnalysisTools.ipCollectPragmas(program, ACCAnnotation.class, "accshared", null);
			if( sharedAnnots != null ) {
				for( ACCAnnotation aAnnot: sharedAnnots ) {
					Set<Symbol> symSet = aAnnot.get("accshared"); 
					if( symSet != null ) {
						Set<Symbol> removeSet = new HashSet<Symbol>();
						for( Symbol dSym : tSymSet ) {
							boolean found = false;
							Symbol gSym = null;
							if( symSet.contains(dSym) ) {
								found = true;
							} else {
								gSym = dataGSymMap.get(dSym);
								if( (gSym != null) && symSet.contains(gSym) ) {
									found = true;
								}
							}
							if( found ) {
								if (gSym == null) { gSym = dSym; }
								Annotatable at = aAnnot.getAnnotatable();
								List<ACCAnnotation> ACCAnnots = at.getAnnotations(ACCAnnotation.class);
								boolean dClauseFound = false;
								for( ACCAnnotation tAnnot : ACCAnnots ) {
									if( tAnnot.equals(aAnnot) ) { continue; }
									else if( dClauseFound ) {
										break;
									} else {
										SubArray subArr = AnalysisTools.findSubArrayInDataClauses(tAnnot, gSym, IRSymbolOnly);
										if( subArr != null ) {
											int dimSize = subArr.getArrayDimension();
											if( dimSize > 0 ) {
												List<Expression> lengthList = subArr.getLengths();
												List<Expression> lengthList2 = new ArrayList<Expression>(dimSize);
												for( Expression length : lengthList ) {
													lengthList2.add(Symbolic.simplify(length.clone()));
												}
												ASPENDataDeclaration pDecl = (ASPENDataDeclaration)aspenModel.getASPENDeclaration(dSym);
												ASPENData tData = pDecl.getASPENData();
												ASPENTrait nTrait = null;
												if( dimSize == 1 ) {
													nTrait = new ASPENTrait("Array", lengthList2);
												} else if( dimSize == 2 ) {
													nTrait = new ASPENTrait("Matrix", lengthList2);
												} else if( dimSize == 3 ) {
													nTrait = new ASPENTrait("3DVolume", lengthList2);
												}
												if( nTrait != null ) {
													tData.addTrait(nTrait);
												}
												removeSet.add(dSym);
												dClauseFound = true;
											}
										}
									}
								}
							}
						}
						if( !removeSet.isEmpty() ) {
							tSymSet.removeAll(removeSet);
						}
					}
				}
			}
			if( !tSymSet.isEmpty() ) {
				PrintTools.println("[WARNING in ASPENModelGen] The following data do not have size information. " +
						"The size information should be provided either by annotating the input program or by " +
						"modifying the output ASPEN model.", 0);
				for( Symbol pSym : tSymSet ) {
					ASPENDeclaration aDecl = aspenModel.getASPENDeclaration(pSym);
					Traversable tIR = aspenDeclToIRMap.get(aDecl);
					Procedure pProc = IRTools.getParentProcedure(tIR);
					TranslationUnit pTrUnt = IRTools.getParentTranslationUnit(tIR);
					if( pProc == null ) {
						PrintTools.println("Symbol: " + pSym.getSymbolName() +"\tTranslationUnit: " + pTrUnt.getOutputFilename(), 0);
					} else {
						PrintTools.println("Symbol: " + pSym.getSymbolName() +"\tTranslationUnit: " + pTrUnt.getOutputFilename() + 
								"\tProcedure: " + pProc.getSymbolName(), 0);
					}
				}
			}
		}
		
		/////////////////////////////////////////////////
		//Step9: Flatten ASPEN IRs to meet the current //
		//ASPEN parallelism mapping strategy.          //                                                          //
		/////////////////////////////////////////////////
		///////////////////////////////////////////////////////////
		//Step9-1: Inline ASPEN kernels called within ASPEN maps //
		///////////////////////////////////////////////////////////
		if( postprocessing > 0 ) {
			do {
				IRMerged = false;
				for( Traversable t : aspenModel.getChildren() ) {
					if( t instanceof ASPENKernel ) {
						inlineKernels(((ASPENKernel)t).getKernelBody(), false);
					}
				}
			} while( IRMerged );
		}
		/////////////////////////////////////
		//Step9-2: Merge nested ASPEN maps //
		/////////////////////////////////////
		if( postprocessing > 1 ) {
			for( Traversable t : aspenModel.getChildren() ) {
				if( t instanceof ASPENKernel ) {
					mergeMaps(((ASPENKernel)t).getKernelBody());
				}
			}
		}
		
		///////////////////////////////////////////////
		//Step10: Write the model to the output file. //
		///////////////////////////////////////////////
		if( genASPENModel > 0 ) {
			PrintTools.printlnStatus("Printing ASPEN Model...", 1);
			try {
				aspenModel.print();
			} catch (IOException e) {
				System.err.println("could not write ASPEN output files: " + e);
				System.exit(1);
			}
		}
	}
	
	
	public static class ASPENAnnotData {
		private Annotatable targetAnnotatable = null;
		public Set<ASPENParam> paramSet = new HashSet<ASPENParam>();
		public Set<ASPENData> dataSet = new HashSet<ASPENData>();
		public Set<ASPENResource> loadsSet = new HashSet<ASPENResource>();
		public Set<ASPENResource> storesSet = new HashSet<ASPENResource>();
		public Set<ASPENResource> flopsSet = new HashSet<ASPENResource>();
		public Set<ASPENResource> messagesSet = new HashSet<ASPENResource>();
		public Set<ASPENResource> intracommSet = new HashSet<ASPENResource>();
		public Set<ASPENData> allocatesSet = new HashSet<ASPENData>();
		public Set<ASPENData> resizesSet = new HashSet<ASPENData>();
		public Set<ASPENData> freesSet = new HashSet<ASPENData>();
		public boolean ASPENAnnotExists = false;
		public boolean skipThis = false;
		public boolean isAspenBlock = false;
		public boolean aspenDeclare = false;
		public boolean aspenControl = false;
		public boolean aspenModelRegion = false;
		public String modelName = null;
		public int modelRegionType = 0;
		public ASPENResource parallelArg = null;
		public List<Expression> ifCond = new ArrayList<Expression>();
		public Expression loopCnt = null;
		public List<Expression> probability = new ArrayList<Expression>();
		public IDExpression label = null;
		public IDExpression preLabel = null;
		public IDExpression postLabel = null;
		protected ASPENControlStatement preStmt = null;
		protected ASPENControlStatement postStmt = null;
		
		public ASPENAnnotData() {
		}
		
		public void resetASPENAnnotData() {
			targetAnnotatable = null;
			paramSet.clear();
			dataSet.clear();
			loadsSet.clear();
			storesSet.clear();
			flopsSet.clear();
			messagesSet.clear();
			intracommSet.clear();
			allocatesSet.clear();
			resizesSet.clear();
			freesSet.clear();
			ASPENAnnotExists = false;
			skipThis = false;
			isAspenBlock = false;
			aspenDeclare = false;
			aspenControl = false;
			aspenModelRegion = false;
			modelName = null;
			modelRegionType = 0;
			parallelArg = null;
			ifCond.clear();
			loopCnt = null;
			probability.clear();
			label = null;
			preLabel = null;
			postLabel = null;
			preStmt = null;
			postStmt = null;
		}
		
		public ASPENControlStatement getPreStmt() {
			return preStmt;
		}
		public ASPENControlStatement getPostStmt() {
			return postStmt;
		}
		
		public void analyzeASPENAnnots(Annotatable at, String newLabel) {
			targetAnnotatable = at;
			List<ASPENAnnotation> aspenAnnots = at.getAnnotations(ASPENAnnotation.class);
			if( (aspenAnnots != null) && (!aspenAnnots.isEmpty()) ) {
				ASPENAnnotExists = true;
				for( ASPENAnnotation asAnnot : aspenAnnots ) {
					if( asAnnot.containsKey("ignore") ) {
						skipThis = true;
						break;
					}
					if( asAnnot.containsKey("execute") ) {
						isAspenBlock = true;
					}
					if( asAnnot.containsKey("control") ) {
						aspenControl = true;
					} else if( asAnnot.containsKey("declare") ) {
						aspenDeclare = true;
					} else if( asAnnot.containsKey("modelregion") ) {
						aspenModelRegion = true;
					}
				}
				if( !skipThis ) {
					if( aspenDeclare ) {
						for( ASPENAnnotation asAnnot : aspenAnnots ) {
							if( asAnnot.containsKey("param") ) {
								paramSet.addAll((Set<ASPENParam>)asAnnot.get("param"));
							} else if( asAnnot.containsKey("data") ) {
								dataSet.addAll((Set<ASPENData>)asAnnot.get("data"));
							}
						}
					} else if( aspenControl ) {
						for( ASPENAnnotation asAnnot : aspenAnnots ) {
							Object val = null;
							val = asAnnot.get("loop");
							if( (val != null) && (val instanceof Expression) ) {
								loopCnt = Symbolic.simplify((Expression)val);
							}
							val = asAnnot.get("if");
							if( (val != null) && (val instanceof List) ) {
								ifCond.addAll((List<Expression>)val);
								if( ifCond.size() > 1 ) {
									//If multiple if-conditions exist, distribute them into the else-if paths.
									if( at instanceof IfStatement ) {
										IfStatement ifStmt = (IfStatement)at;
										while( ifCond.size() > 1 ) {
											Expression tCond = ifCond.remove(1);
											CompoundStatement elseStmt = (CompoundStatement)ifStmt.getElseStatement();
											if( (elseStmt == null) || (elseStmt.getChildren().isEmpty()) ) {
												Tools.exit("[ERROR in ASPENAnnotData.analyzeASPENAnnots()] the number of if-conditions in an ASPEN annotation does not " +
														"match the if-paths in the attached if-statement; exit\n" + "ASPEN annotation: " + asAnnot + 
														AnalysisTools.getEnclosingAnnotationContext(asAnnot));
											} else {
												ifStmt = null;
												for( Traversable tChild : elseStmt.getChildren() ) {
													if( tChild instanceof IfStatement ) {
														ifStmt = (IfStatement)tChild;
														break;
													}
												}
												if( ifStmt == null ) {
													Tools.exit("[ERROR in ASPENAnnotData.analyzeASPENAnnots()] the number of if-conditions in an ASPEN annotation does not " +
															"match the if-paths in the attached if-statement; exit\n" + "ASPEN annotation: " + asAnnot + 
															AnalysisTools.getEnclosingAnnotationContext(asAnnot));
												} else {
													ASPENAnnotation tASAnnot = ifStmt.getAnnotation(ASPENAnnotation.class, "if");
													if( tASAnnot == null ) {
														tASAnnot = ifStmt.getAnnotation(ASPENAnnotation.class, "control");
														if( tASAnnot == null ) {
															tASAnnot = new ASPENAnnotation();
															tASAnnot.put("control", "_directive");
															ifStmt.annotate(tASAnnot);
														}
														List<Expression> newIfCond = new ArrayList<Expression>(1);
														newIfCond.add(tCond.clone());
														tASAnnot.put("if", newIfCond);
													}
												}
											}
										}
									} else {
										Tools.exit("[ERROR in ASPENAnnotData.analyzeASPENAnnots()] multiple if-conditions are allowed " +
												"only to if-statements; exit\n" + "ASPEN annotation: " + asAnnot + 
												AnalysisTools.getEnclosingAnnotationContext(asAnnot));
									}
								}
							}
							val = asAnnot.get("probability");
							if( (val != null) && (val instanceof List) ) {
								probability.addAll((List<Expression>)val);
								if( probability.size() > 1 ) {
									//If multiple if-probabilities exist, distribute them into the else-if paths.
									if( at instanceof IfStatement ) {
										IfStatement ifStmt = (IfStatement)at;
										while( probability.size() > 1 ) {
											Expression tCond = probability.remove(1);
											CompoundStatement elseStmt = (CompoundStatement)ifStmt.getElseStatement();
											if( (elseStmt == null) || (elseStmt.getChildren().isEmpty()) ) {
												Tools.exit("[ERROR in ASPENAnnotData.analyzeASPENAnnots()] the number of if-probabilities in an ASPEN annotation does not " +
														"match the if-paths in the attached if-statement; exit\n" + "ASPEN annotation: " + asAnnot + 
														AnalysisTools.getEnclosingAnnotationContext(asAnnot));
											} else {
												ifStmt = null;
												for( Traversable tChild : elseStmt.getChildren() ) {
													if( tChild instanceof IfStatement ) {
														ifStmt = (IfStatement)tChild;
														break;
													}
												}
												if( ifStmt == null ) {
													Tools.exit("[ERROR in ASPENAnnotData.analyzeASPENAnnots()] the number of if-probabilities in an ASPEN annotation does not " +
															"match the if-paths in the attached if-statement; exit\n" + "ASPEN annotation: " + asAnnot + 
															AnalysisTools.getEnclosingAnnotationContext(asAnnot));
												} else {
													ASPENAnnotation tASAnnot = ifStmt.getAnnotation(ASPENAnnotation.class, "probability");
													if( tASAnnot == null ) {
														tASAnnot = ifStmt.getAnnotation(ASPENAnnotation.class, "control");
														if( tASAnnot == null ) {
															tASAnnot = new ASPENAnnotation();
															tASAnnot.put("control", "_directive");
															ifStmt.annotate(tASAnnot);
														}
														List<Expression> newIfProb = new ArrayList<Expression>(1);
														newIfProb.add(tCond.clone());
														tASAnnot.put("probability", newIfProb);
													}
												}
											}
										}
									} else {
										Tools.exit("[ERROR in ASPENAnnotData.analyzeASPENAnnots()] multiple if-probabilities are allowed " +
												"only to if-statements; exit\n" + "ASPEN annotation: " + asAnnot + 
												AnalysisTools.getEnclosingAnnotationContext(asAnnot));
									}
									
								}
							}
							val = asAnnot.get("label");
							if( (val != null) && (val instanceof String) ) {
								label = new NameID((String)val);
								preLabel = new NameID(((String)val).concat("__intracommIN"));
								postLabel = new NameID(((String)val).concat("__intracommOUT"));
							} else {
								label = new NameID(newLabel);
								asAnnot.put("label", newLabel);
								preLabel = new NameID(newLabel.concat("__intracommIN"));
								postLabel = new NameID(newLabel.concat("__intracommOUT"));
								
							}
							if( asAnnot.containsKey("parallelism") ) {
								parallelArg = (ASPENResource)asAnnot.get("parallelism");
							}
							if( asAnnot.containsKey("flops") ) {
								flopsSet.addAll((Set<ASPENResource>)asAnnot.get("flops"));
							}
							if( asAnnot.containsKey("loads") ) {
								loadsSet.addAll((Set<ASPENResource>)asAnnot.get("loads"));
							}
							if( asAnnot.containsKey("stores") ) {
								storesSet.addAll((Set<ASPENResource>)asAnnot.get("stores"));
							}
							if( asAnnot.containsKey("messages") ) {
								messagesSet.addAll((Set<ASPENResource>)asAnnot.get("messages"));
							}
							if( asAnnot.containsKey("intracomm") ) {
								intracommSet.addAll((Set<ASPENResource>)asAnnot.get("intracomm"));
							}
							if( asAnnot.containsKey("allocates") ) {
								allocatesSet.addAll((Set<ASPENData>)asAnnot.get("allocates"));
							}
							if( asAnnot.containsKey("resizes") ) {
								resizesSet.addAll((Set<ASPENData>)asAnnot.get("resizes"));
							}
							if( asAnnot.containsKey("frees") ) {
								freesSet.addAll((Set<ASPENData>)asAnnot.get("frees"));
							}
						}
					} else if (aspenModelRegion) {
						for( ASPENAnnotation asAnnot : aspenAnnots ) {
							if( asAnnot.containsKey("label") ) {
								modelName = asAnnot.get("label");
							}
							if( asAnnot.containsKey("enter") ) {
								modelRegionType = 1;
							} else if( asAnnot.containsKey("exit") ) {
								modelRegionType = 2;
							} else {
								modelRegionType = 0;
							}
						}	
					}
				}
			}
		}
		
		public ASPENControlStatement genASPENStatement() {
			ASPENControlStatement retStmt = null;
			if( ASPENAnnotExists ) {
				ASPENCompoundStatement tCStmt = new ASPENCompoundStatement();
				ASPENCompoundStatement tPreCStmt = new ASPENCompoundStatement();
				ASPENCompoundStatement tPostCStmt = new ASPENCompoundStatement();
				//[DEBUG] @deprecated
/*				if( parallelArg != null ) {
					ASPENExposesExpressionStatement tStmt = new ASPENExposesExpressionStatement(parallelArg);
					tCStmt.addASPENStatement(tStmt);
				}*/
				for( ASPENResource tData : loadsSet ) {
					ASPENRequiresExpressionStatement tStmt = new ASPENRequiresExpressionStatement("loads", tData);
					tCStmt.addASPENStatement(tStmt);
				}
				for( ASPENResource tData : storesSet ) {
					ASPENRequiresExpressionStatement tStmt = new ASPENRequiresExpressionStatement("stores", tData);
					tCStmt.addASPENStatement(tStmt);
				}
				for( ASPENResource tData : flopsSet ) {
					ASPENRequiresExpressionStatement tStmt = new ASPENRequiresExpressionStatement("flops", tData);
					tCStmt.addASPENStatement(tStmt);
				}
				for( ASPENResource tData : messagesSet ) {
					ASPENRequiresExpressionStatement tStmt = new ASPENRequiresExpressionStatement("messages", tData);
					tCStmt.addASPENStatement(tStmt);
				}
				for( ASPENResource tData : intracommSet ) {
					int trSize = tData.getTraitSize();
					for( int i=0; i<trSize; i++ ) {
						ASPENTrait tr = tData.getTrait(i);
						String dtrait = tr.getTrait();
						//[DEBUG] Changed not to generate intracomm statement for present_or_* clauses.
						//if( dtrait.equals("copy") || dtrait.equals("pcopy") ) {
						if( dtrait.equals("copy") ) {
							ASPENResource clonedRSC = tData.clone();
							String property = null;
							if( dtrait.equals("copy") ) {
								property = "copyin";
							} else {
								property = "pcopyin";
							}
							clonedRSC.setTrait(i, new ASPENTrait(property));
							ASPENRequiresExpressionStatement tStmt = new ASPENRequiresExpressionStatement("intracomm", clonedRSC);
							tPreCStmt.addASPENStatement(tStmt);
							clonedRSC = tData.clone();
							if( dtrait.equals("copy") ) {
								property = "copyout";
							} else {
								property = "pcopyout";
							}
							clonedRSC.setTrait(i, new ASPENTrait(property));
							tStmt = new ASPENRequiresExpressionStatement("intracomm", clonedRSC);
							tPostCStmt.addASPENStatement(tStmt);
						//} else if( dtrait.equals("copyin") || dtrait.equals("pcopyin")) {
						} else if( dtrait.equals("copyin") ) {
							ASPENRequiresExpressionStatement tStmt = new ASPENRequiresExpressionStatement("intracomm", tData);
							tPreCStmt.addASPENStatement(tStmt);
						//} else if( dtrait.equals("copyout") || dtrait.equals("pcopyout")) {
						} else if( dtrait.equals("copyout") ) {
							ASPENRequiresExpressionStatement tStmt = new ASPENRequiresExpressionStatement("intracomm", tData);
							tPostCStmt.addASPENStatement(tStmt.clone());
						}
					}
					//tCStmt.addASPENStatement(tStmt);
				}
				//[FIXME] Memory clauses (allocates/resizes/frees) are temporarily disabled, 
				//since current ASPEN implementation does not accept these.
/*				for( ASPENData tData : allocatesSet ) {
					ASPENMemoryExpressionStatement tStmt = new ASPENMemoryExpressionStatement("allocates", tData);
					tCStmt.addASPENStatement(tStmt);
				}
				for( ASPENData tData : resizesSet ) {
					ASPENMemoryExpressionStatement tStmt = new ASPENMemoryExpressionStatement("resizes", tData);
					tCStmt.addASPENStatement(tStmt);
				}
				for( ASPENData tData : freesSet ) {
					ASPENMemoryExpressionStatement tStmt = new ASPENMemoryExpressionStatement("frees", tData);
					tCStmt.addASPENStatement(tStmt);
				}*/
				if( !tCStmt.isEmpty() ) {
					ASPENControlExecuteStatement aBlockStmt = null;
					if (parallelArg == null) {
						aBlockStmt = new ASPENControlExecuteStatement(label, tCStmt); 
					} else {
						aBlockStmt = new ASPENControlExecuteStatement(label, tCStmt, parallelArg); 
					}
					if( !isAspenBlock && 
							((targetAnnotatable instanceof IfStatement) || (targetAnnotatable instanceof Loop)) ) {
						retStmt = aBlockStmt;
					} else {
						ASPENControlStatement targetStmt = null;
						if( (parallelArg == null) && (loopCnt != null) ) {
							ASPENCompoundStatement ttCStmt = new ASPENCompoundStatement();
							ttCStmt.addASPENStatement(aBlockStmt);
							targetStmt = new ASPENControlIterateStatement(loopCnt.clone(), ttCStmt);
						} else {
							targetStmt = aBlockStmt;
						}
						if( (probability != null) && !probability.isEmpty() ) {
							ASPENCompoundStatement ttCStmt = new ASPENCompoundStatement();
							ttCStmt.addASPENStatement(targetStmt);
							if( probability.size() > 1 ) {
								Tools.exit("[ERROR in ASPENAnnotData.genASPENStatement()] multiple if-probabilities exist where " +
										"at most one probability is expected; exit\n" + "Attached region: " + targetAnnotatable + 
										AnalysisTools.getEnclosingContext(targetAnnotatable));
							}
							ASPENControlProbabilityStatement probStmt = new ASPENControlProbabilityStatement(probability.get(0).clone(), ttCStmt);
							retStmt = probStmt;
						} else if( (ifCond != null) && !ifCond.isEmpty() ) {
							ASPENCompoundStatement ttCStmt = new ASPENCompoundStatement();
							ttCStmt.addASPENStatement(targetStmt);
							if( ifCond.size() > 1 ) {
								Tools.exit("[ERROR in ASPENAnnotData.genASPENStatement()] multiple if-conditions exist where " +
										"at most one if-condition is expected; exit\n" + "Attached region: " + targetAnnotatable + 
										AnalysisTools.getEnclosingContext(targetAnnotatable));
							}
							ASPENControlIfStatement ifStmt = new ASPENControlIfStatement(ifCond.get(0).clone(), ttCStmt);
							retStmt = ifStmt;
						} else {
							retStmt = targetStmt;
						}
					}
				}
				if( !tPreCStmt.isEmpty() ) {
					ASPENControlExecuteStatement aBlockStmt = new ASPENControlExecuteStatement(preLabel, tPreCStmt); 
					if( (probability != null) && !probability.isEmpty() ) {
						ASPENCompoundStatement ttCStmt = new ASPENCompoundStatement();
						ttCStmt.addASPENStatement(aBlockStmt);
						if( probability.size() > 1 ) {
							Tools.exit("[ERROR in ASPENAnnotData.genASPENStatement()] multiple if-probabilities exist where " +
									"at most one probability is expected; exit\n" + "Attached region: " + targetAnnotatable + 
									AnalysisTools.getEnclosingContext(targetAnnotatable));
						}
						ASPENControlProbabilityStatement probStmt = new ASPENControlProbabilityStatement(probability.get(0).clone(), ttCStmt);
						preStmt = probStmt;
					} else if( (ifCond != null) && !ifCond.isEmpty() ) {
						ASPENCompoundStatement ttCStmt = new ASPENCompoundStatement();
						ttCStmt.addASPENStatement(aBlockStmt);
						if( ifCond.size() > 1 ) {
							Tools.exit("[ERROR in ASPENAnnotData.genASPENStatement()] multiple if-conditions exist where " +
									"at most one if-condition is expected; exit\n" + "Attached region: " + targetAnnotatable + 
									AnalysisTools.getEnclosingContext(targetAnnotatable));
						}
						ASPENControlIfStatement ifStmt = new ASPENControlIfStatement(ifCond.get(0).clone(), ttCStmt);
						preStmt = ifStmt;
					} else {
						preStmt = aBlockStmt;
					}
				}
				if( !tPostCStmt.isEmpty() ) {
					ASPENControlExecuteStatement aBlockStmt = new ASPENControlExecuteStatement(postLabel, tPostCStmt); 
					if( (probability != null) && !probability.isEmpty() ) {
						ASPENCompoundStatement ttCStmt = new ASPENCompoundStatement();
						ttCStmt.addASPENStatement(aBlockStmt);
						if( probability.size() > 1 ) {
							Tools.exit("[ERROR in ASPENAnnotData.genASPENStatement()] multiple if-probabilities exist where " +
									"at most one probability is expected; exit\n" + "Attached region: " + targetAnnotatable + 
									AnalysisTools.getEnclosingContext(targetAnnotatable));
						}
						ASPENControlProbabilityStatement probStmt = new ASPENControlProbabilityStatement(probability.get(0).clone(), ttCStmt);
						postStmt = probStmt;
					} else if( (ifCond != null) && !ifCond.isEmpty() ) {
						ASPENCompoundStatement ttCStmt = new ASPENCompoundStatement();
						ttCStmt.addASPENStatement(aBlockStmt);
						if( ifCond.size() > 1 ) {
							Tools.exit("[ERROR in ASPENAnnotData.genASPENStatement()] multiple if-conditions exist where " +
									"at most one if-condition is expected; exit\n" + "Attached region: " + targetAnnotatable + 
									AnalysisTools.getEnclosingContext(targetAnnotatable));
						}
						ASPENControlIfStatement ifStmt = new ASPENControlIfStatement(ifCond.get(0).clone(), ttCStmt);
						postStmt = ifStmt;
					} else {
						postStmt = aBlockStmt;
					}
				}
			}
			return retStmt;
		}
	}
	
	private void fillASPENStatements(ASPENCompoundStatement aspenCStmtOpt, CompoundStatement inputBody,
			Procedure proc, List<ProcedureDeclarator> procDeclrList) {
		//PrintTools.println("Current procedure = " + proc.getSymbolName(), 0);
		String blockName = "block_" + proc.getSymbolName();
		ASPENCompoundStatement aspenCStmt = new ASPENCompoundStatement();
		ASPENAnnotData aspenAnnotData1 = new ASPENAnnotData();
		List<Traversable> IRsToRemove = new LinkedList<Traversable>();
		for( Traversable child : inputBody.getChildren() ) {
			aspenAnnotData1.resetASPENAnnotData();
			Annotatable at = (Annotatable)child;
			aspenAnnotData1.analyzeASPENAnnots(at, blockName + exeBlockCnt++);
			if( aspenAnnotData1.skipThis ) { continue; }
			else {
				if( aspenAnnotData1.aspenModelRegion ) {
					if( aspenAnnotData1.modelRegionType == 0 ) { //aspen modelregion directive found.
						inASPENModelRegion = true;
					} else if( aspenAnnotData1.modelRegionType == 1 ) {//aspen enter modelregion directive found
						inASPENModelRegion = true;
					} else if( aspenAnnotData1.modelRegionType == 2 ) { //aspen exit modelregion directive found.
						inASPENModelRegion = false;
					}
				} else if( aspenAnnotData1.aspenDeclare ) {
					for( ASPENParam aParam : aspenAnnotData1.paramSet ) {
						if( aParam.getParent() != null ) {
							if( aParam.toString().startsWith("_ret_val_") ) {
								continue;
							} else {
								Tools.exit("\n[ERROR in ASPENModelGen] Duplicated declaration of the Aspen parameter, " +
										aParam + " in the following statement; exit!\n" +
										"Statement: " + at + 
										"\nEnclosing Procedure: " + proc.getSymbolName() +"\n" +
										"Enclosing Translation Unit: " + ((TranslationUnit)proc.getParent()).getOutputFilename() + "\n");
							}
						}
						ASPENParamDeclaration paramDecl = new ASPENParamDeclaration(aParam);
						IDExpression ID = paramDecl.getDeclaredID();
						if( aspenModel.containsParam(ID) ) {
							if( !ASPENModelAnalysis.isInternalParam(ID.getName()) ) {
								if( aParam.toString().startsWith("_ret_val_") ) {
									continue;
								} else {
									PrintTools.println("\n[WARNING in ASPENModelGen] duplicate ASPEN parameter is found:\n" +
											"ASPEN Parameter: " + aParam.toString() + 
											" Enclosing Procedure: " + proc.getSymbolName() +"\n", 1);
								}
							}
							ASPENParamDeclaration cParamDecl = aspenModel.getParamDeclaration(ID);
							ASPENParam tParam = cParamDecl.getASPENParam();
							if( tParam.getInitVal() == null ) {
								ASPENDeclaration aspenDecl = aspenModel.removeParam(ID);
								if( aspenDecl != null ) {
									Traversable IRtr = aspenDeclToIRMap.get(aspenDecl);
									if( IRtr != null ) {
										Traversable IRtrP = IRtr.getParent();
										if( IRtrP != null ) {
											//IRtrP.removeChild(IRtr);
											IRsToRemove.add(IRtr);
										}
									}
									aspenModel.addASPENDeclaration(paramDecl);
									aspenDeclToIRMap.put(paramDecl, child);
								}
							}
						} else {
							aspenModel.addASPENDeclaration(paramDecl);
							aspenDeclToIRMap.put(paramDecl, at);
						}
					}
					for( ASPENData aData : aspenAnnotData1.dataSet ) {
						if( aData.getParent() != null ) {
							Tools.exit("\n[ERROR in ASPENModelGen] Duplicated declaration of the Aspen data, " +
									aData + " in the following statement; exit!\n" +
									"Statement: " + at + 
									"\nEnclosing Procedure: " + proc.getSymbolName() +"\n" +
									"Enclosing Translation Unit: " + ((TranslationUnit)proc.getParent()).getOutputFilename() + "\n");
						}
						ASPENDataDeclaration dataDecl = new ASPENDataDeclaration(aData);
						IDExpression ID = dataDecl.getDeclaredID();
						if( aspenModel.containsData(ID) ) {
							PrintTools.println("\n[WARNING in ASPENModelGen] duplicate ASPEN data is found:\n" +
									"ASPEN Data: " + aData.toString() + 
									" Enclosing Procedure: " + proc.getSymbolName() +"\n", 0);
							ASPENDataDeclaration cDataDecl = aspenModel.getDataDeclaration(ID);
							ASPENData tData = cDataDecl.getASPENData();
							if( (tData.getCapacity() == null) && (tData.getTraitSize() == 0) ) {
								ASPENDeclaration aspenDecl = aspenModel.removeData(ID);
								if( aspenDecl != null ) {
									Traversable IRtr = aspenDeclToIRMap.get(aspenDecl);
									if( IRtr != null ) {
										Traversable IRtrP = IRtr.getParent();
										if( IRtrP != null ) {
											//IRtrP.removeChild(IRtr);
											IRsToRemove.add(IRtr);
										}
									}
									aspenModel.addASPENDeclaration(dataDecl);
									aspenDeclToIRMap.put(dataDecl, child);
								}
							}
						} else {
							aspenModel.addASPENDeclaration(dataDecl);
							aspenDeclToIRMap.put(dataDecl, at);
						}
					}
				}
			}
			
			if( !inASPENModelRegion ) {
				continue;
			}
			
			boolean isConditional = false;
			if( (!aspenAnnotData1.probability.isEmpty()) || (!aspenAnnotData1.ifCond.isEmpty()) ) {
				isConditional = true;
			}
			
			ASPENControlStatement aspenStmt = aspenAnnotData1.genASPENStatement();
			if( aspenAnnotData1.isAspenBlock ) {
				if( aspenAnnotData1.getPreStmt() != null ) {
					aspenCStmt.addASPENStatement(aspenAnnotData1.getPreStmt());
				}
				if( aspenStmt != null ) {
					aspenCStmt.addASPENStatement(aspenStmt);
				}
				if( aspenAnnotData1.getPostStmt() != null ) {
					aspenCStmt.addASPENStatement(aspenAnnotData1.getPostStmt());
				}
			} else {
				if( at instanceof ExpressionStatement ) {
					if( aspenAnnotData1.getPreStmt() != null ) {
						aspenCStmt.addASPENStatement(aspenAnnotData1.getPreStmt());
					}
					if( aspenStmt != null ) {
						aspenCStmt.addASPENStatement(aspenStmt);
					}
					if( aspenAnnotData1.getPostStmt() != null ) {
						aspenCStmt.addASPENStatement(aspenAnnotData1.getPostStmt());
					}
					List<FunctionCall> fCallList = IRTools.getFunctionCalls(at);
					if( fCallList != null ) {
						ASPENCompoundStatement ttCStmt = new ASPENCompoundStatement();
						for( FunctionCall fCall : fCallList ) {
							IDExpression kernelID = (IDExpression)fCall.getName();
							Procedure tProc = fCall.getProcedure();
							if( StandardLibrary.contains(fCall) ) {
								//[TODO] C standard library should be treated differently.
								continue;
							}
							if( ignoredKernels.contains(kernelID) ) {
								continue;
							} else if( aspenModel.containsKernel(kernelID) ) {
								ASPENControlKernelCallStatement kCallStmt = new ASPENControlKernelCallStatement(kernelID.clone(), null);
								if( isConditional ) {
									ttCStmt.addASPENStatement(kCallStmt);
								} else {
									aspenCStmt.addASPENStatement(kCallStmt);
								}
							} else {
								ASPENAnnotData aspenAnnotData2 = new ASPENAnnotData();
								ASPENAnnotData aspenAnnotData3 = new ASPENAnnotData();
								ASPENCompoundStatement kernelBody = new ASPENCompoundStatement();
								Declaration procDecl = null;
								for( ProcedureDeclarator procDeclr : procDeclrList ) {
									if( procDeclr.getID().equals(kernelID) ) {
										procDecl = procDeclr.getDeclaration();
										break;
									}
								}
								if( procDecl != null ) {
									aspenAnnotData2.analyzeASPENAnnots(procDecl, "block_" + kernelID.toString());
									if( aspenAnnotData2.skipThis ) {
										ignoredKernels.add(kernelID.clone());
										continue;
									} else if( aspenAnnotData2.ASPENAnnotExists ) {
										ASPENStatement procAnnotStmt = aspenAnnotData2.genASPENStatement();
										if( procAnnotStmt != null ) {
											if( aspenAnnotData2.getPreStmt() != null ) {
												aspenCStmt.addASPENStatement(aspenAnnotData2.getPreStmt());
											}
											kernelBody.addASPENStatement(procAnnotStmt);
											if( aspenAnnotData2.getPostStmt() != null ) {
												aspenCStmt.addASPENStatement(aspenAnnotData2.getPostStmt());
											}
										}
									} 
								}
								if( tProc == null ) {
									if( kernelBody.isEmpty() ) {
										ignoredKernels.add(kernelID.clone());
										continue;
									}
								} else {
									aspenAnnotData3.analyzeASPENAnnots(tProc, "block_" + kernelID.toString());
									if( aspenAnnotData3.skipThis ) {
										ignoredKernels.add(kernelID.clone());
										continue;
									} else if( aspenAnnotData3.ASPENAnnotExists ) {
										//[CAUTION] This assumes that annotations in the procedure declaration have higher
										//priority than those in the procedure; to minimize confusion, both procedure declaration
										//and procedure should have the same annotations.
										if( kernelBody.isEmpty() ) {
											ASPENStatement procAnnotStmt = aspenAnnotData3.genASPENStatement();
											if( aspenAnnotData3.getPreStmt() != null ) {
												aspenCStmt.addASPENStatement(aspenAnnotData3.getPreStmt());
											}
											if( procAnnotStmt != null ) {
												kernelBody.addASPENStatement(procAnnotStmt);
											}
											if( aspenAnnotData3.getPostStmt() != null ) {
												aspenCStmt.addASPENStatement(aspenAnnotData3.getPostStmt());
											}
										}
									}
								}
								List<ProcedureDeclarator> tProcDeclrList = AnalysisTools.getProcedureDeclarators(tProc);
								fillASPENStatements(kernelBody, tProc.getBody(), tProc, tProcDeclrList);
								if( kernelBody.isEmpty() ) {
									ignoredKernels.add(kernelID.clone());
									continue;
								} else {
									ASPENKernel newKernel = new ASPENKernel(kernelID.clone(), null, kernelBody);
									aspenModel.addASPENDeclaration(newKernel);
									ASPENControlStatement kCallStmt = new ASPENControlKernelCallStatement(kernelID.clone(), null);
									Expression tProbability = null;
									if( aspenAnnotData2.probability.isEmpty() ) {
										if( !aspenAnnotData3.probability.isEmpty() ) {
											tProbability = aspenAnnotData3.probability.get(0).clone();
										}	
									} else {
										tProbability = aspenAnnotData2.probability.get(0).clone();
									}
									if( tProbability != null ) {
										ASPENCompoundStatement tttCStmt = new ASPENCompoundStatement();
										tttCStmt.addASPENStatement(kCallStmt);
										kCallStmt = new ASPENControlProbabilityStatement(tProbability.clone(), tttCStmt);
									} else {
										Expression tIfCond = null;
										if( aspenAnnotData2.ifCond.isEmpty() ) {
											if( !aspenAnnotData3.ifCond.isEmpty() ) {
												tIfCond = aspenAnnotData3.ifCond.get(0).clone();
											}	
										} else {
											tIfCond = aspenAnnotData2.ifCond.get(0).clone();
										}
										if( tIfCond != null ) {
											ASPENCompoundStatement tttCStmt = new ASPENCompoundStatement();
											tttCStmt.addASPENStatement(kCallStmt);
											kCallStmt = new ASPENControlIfStatement(tIfCond.clone(), tttCStmt);
										}
									}
									if( isConditional ) {
										ttCStmt.addASPENStatement(kCallStmt);
									} else {
										aspenCStmt.addASPENStatement(kCallStmt);
									}
								}
							}
						}
						if( isConditional && !ttCStmt.isEmpty() ) {
							if( !aspenAnnotData1.probability.isEmpty() ) {
								ASPENControlProbabilityStatement probStmt = new ASPENControlProbabilityStatement(aspenAnnotData1.probability.get(0).clone(), ttCStmt);
								aspenCStmt.addASPENStatement(probStmt);
							} else if( !aspenAnnotData1.ifCond.isEmpty() ) {
								ASPENControlIfStatement ifStmt = new ASPENControlIfStatement(aspenAnnotData1.ifCond.get(0).clone(), ttCStmt);
								aspenCStmt.addASPENStatement(ifStmt);
							}
						}
					}
				} else if( at instanceof Loop ) {
					if( aspenStmt != null ) {
						//Add ASPEN statement for loop condition expression.
						//Assume that this expression is evaluated only once due to caching.
						aspenCStmt.addASPENStatement(aspenStmt);
					}
					CompoundStatement ttBody = (CompoundStatement)((Loop)at).getBody();
					ASPENCompoundStatement ttCStmt = new ASPENCompoundStatement();
					fillASPENStatements(ttCStmt,ttBody, proc, procDeclrList);
					if( !ttCStmt.isEmpty() ) {
						if( (aspenAnnotData1.loopCnt == null) && (aspenAnnotData1.parallelArg == null) ) {
							Tools.exit("\n[ERROR in ASPENModelGen] Loop statement without ignore or execute ASPEN clause should have " +
									"loop count information in either loop or parallelism ASPEN clause.\n" +
									"Loop Statement:\n" + at + 
									"\nEnclosing Procedure: " + proc.getSymbolName() +"\n" +
									"Enclosing Translation Unit: " + ((TranslationUnit)proc.getParent()).getOutputFilename() + "\n");
						} else {
							ASPENControlStatement targetStmt = null;
							if( aspenAnnotData1.parallelArg != null ) {
								targetStmt = 
									new ASPENControlMapStatement(aspenAnnotData1.parallelArg.getValue().clone(), ttCStmt, new NameID("map"+aspenAnnotData1.label.toString()));
							} else if( aspenAnnotData1.loopCnt != null ) {
								targetStmt = 
									new ASPENControlIterateStatement(aspenAnnotData1.loopCnt.clone(), ttCStmt);
							}
							if( targetStmt != null ) {
								if( isConditional ) {
									ASPENCompoundStatement tttCStmt = new ASPENCompoundStatement();
									if( aspenAnnotData1.getPreStmt() != null ) {
										tttCStmt.addASPENStatement(aspenAnnotData1.getPreStmt());
									}
									tttCStmt.addASPENStatement(targetStmt);
									if( aspenAnnotData1.getPostStmt() != null ) {
										tttCStmt.addASPENStatement(aspenAnnotData1.getPostStmt());
									}
									if( !aspenAnnotData1.probability.isEmpty() ) {
										ASPENControlProbabilityStatement probStmt = 
											new ASPENControlProbabilityStatement(aspenAnnotData1.probability.get(0).clone(), tttCStmt);
										aspenCStmt.addASPENStatement(probStmt);
									} else if( !aspenAnnotData1.ifCond.isEmpty() ) {
										ASPENControlIfStatement ifStmt = new ASPENControlIfStatement(aspenAnnotData1.ifCond.get(0).clone(), tttCStmt);
										aspenCStmt.addASPENStatement(ifStmt);
									}
								} else {
									if( aspenAnnotData1.getPreStmt() != null ) {
										aspenCStmt.addASPENStatement(aspenAnnotData1.getPreStmt());
									}
									aspenCStmt.addASPENStatement(targetStmt);
									if( aspenAnnotData1.getPostStmt() != null ) {
										aspenCStmt.addASPENStatement(aspenAnnotData1.getPostStmt());
									}
								}
							}
						}
					}
				} else if( at instanceof IfStatement ) {
					if( aspenStmt != null ) {
						//Add ASPEN statement for if condition expression.
						//Assume that this expression is evaluated only once due to caching.
						aspenCStmt.addASPENStatement(aspenStmt);
					}
					long condType = -1; //unknown
					if( (!aspenAnnotData1.ifCond.isEmpty()) && (aspenAnnotData1.ifCond.get(0) instanceof IntegerLiteral) ) {
						IntegerLiteral intL = (IntegerLiteral)aspenAnnotData1.ifCond.get(0);
						if( intL.getValue() == 0 ) {
							condType = 0; //false
						} else {
							condType = 1; //true
						}
					} else if( (!aspenAnnotData1.probability.isEmpty()) && (aspenAnnotData1.probability.get(0) instanceof IntegerLiteral) ) {
						IntegerLiteral intL = (IntegerLiteral)aspenAnnotData1.probability.get(0);
						if( intL.getValue() == 0 ) {
							condType = 0; //false
						} else if( intL.getValue() == 1 ) {
							condType = 1; //true
						}
					}
					ASPENCompoundStatement ttCStmt1 = new ASPENCompoundStatement();
					if( condType != 0 ) {
						CompoundStatement thenBody = (CompoundStatement)((IfStatement)at).getThenStatement();
						fillASPENStatements(ttCStmt1,thenBody, proc, procDeclrList);
					}
					ASPENCompoundStatement ttCStmt2 = new ASPENCompoundStatement();
					if( condType != 1 ) {
						CompoundStatement elseBody = (CompoundStatement)((IfStatement)at).getElseStatement();
						if( elseBody != null ) {
							fillASPENStatements(ttCStmt2,elseBody, proc, procDeclrList);
						}
					}
					if( (!ttCStmt1.isEmpty()) || (!ttCStmt2.isEmpty()) ) {
						if( (aspenAnnotData1.ifCond.isEmpty()) && (aspenAnnotData1.probability.isEmpty()) ) {
							Tools.exit("\n[ERROR in ASPENModelGen] If statement without ignore or execute ASPEN clause should have " +
									"either if or probability ASPEN clause.\n" +
									"If Statement:\n" + at + 
									"\nEnclosing Procedure: " + proc.getSymbolName() +"\n" + 
									"Enclosing Translation Unit: " + ((TranslationUnit)proc.getParent()).getOutputFilename() + "\n");
						} else {
							ASPENControlStatement targetStmt = null;
							if( aspenAnnotData1.getPreStmt() != null ) {
								aspenCStmt.addASPENStatement(aspenAnnotData1.getPreStmt());
							}
							if( !aspenAnnotData1.probability.isEmpty() ) {
								if( condType == -1 ) {
									targetStmt = 
										new ASPENControlProbabilityStatement(aspenAnnotData1.probability.get(0).clone(), ttCStmt1, ttCStmt2);
									((ASPENControlProbabilityStatement)targetStmt).normalize();
									aspenCStmt.addASPENStatement(targetStmt);
								} else if( condType == 1 ) { //add then-part only
									for( Traversable tCh : ttCStmt1.getChildren() ) {
										if( tCh instanceof ASPENStatement ) {
											tCh.setParent(null);
											aspenCStmt.addASPENStatement((ASPENStatement)tCh);
										}
									}
								} else { //add else-part only
									for( Traversable tCh : ttCStmt2.getChildren() ) {
										if( tCh instanceof ASPENStatement ) {
											tCh.setParent(null);
											aspenCStmt.addASPENStatement((ASPENStatement)tCh);
										}
									}
								}
							} else if( !aspenAnnotData1.ifCond.isEmpty() ) {
								if( condType == -1 ) {
									targetStmt = 
										new ASPENControlIfStatement(aspenAnnotData1.ifCond.get(0).clone(), ttCStmt1, ttCStmt2);
									((ASPENControlIfStatement)targetStmt).normalize();
									aspenCStmt.addASPENStatement(targetStmt);
								} else if( condType == 1 ) { //add then-part only
									for( Traversable tCh : ttCStmt1.getChildren() ) {
										if( tCh instanceof ASPENStatement ) {
											tCh.setParent(null);
											aspenCStmt.addASPENStatement((ASPENStatement)tCh);
										}
									}
								} else { //add else-part only
									for( Traversable tCh : ttCStmt2.getChildren() ) {
										if( tCh instanceof ASPENStatement ) {
											tCh.setParent(null);
											aspenCStmt.addASPENStatement((ASPENStatement)tCh);
										}
									}
								}
							}
							if( aspenAnnotData1.getPostStmt() != null ) {
								aspenCStmt.addASPENStatement(aspenAnnotData1.getPostStmt());
							}
						}
					}
				} else if( at instanceof CompoundStatement ) {
					ASPENCompoundStatement ttCStmt = new ASPENCompoundStatement();
					if( aspenAnnotData1.getPreStmt() != null ) {
						ttCStmt.addASPENStatement(aspenAnnotData1.getPreStmt());
					}
					fillASPENStatements(ttCStmt,(CompoundStatement)at, proc, procDeclrList);
					if( aspenAnnotData1.getPostStmt() != null ) {
						ttCStmt.addASPENStatement(aspenAnnotData1.getPostStmt());
					}
					//[DEBUG] How does ASPEN handle nested compound statements? 
					//e.g., par { A, {B, C}, D}
					//In current implementation, the nested compound statement will be flattened
					//if no conditional clause exists.
					if( !ttCStmt.isEmpty() ) { 
						ASPENControlStatement targetStmt = null;
						if( isConditional ) {
							if( !aspenAnnotData1.probability.isEmpty() ) {
								targetStmt = 
									new ASPENControlProbabilityStatement(aspenAnnotData1.probability.get(0).clone(), ttCStmt);
							} else if( !aspenAnnotData1.ifCond.isEmpty() ) {
								targetStmt =  new ASPENControlIfStatement(aspenAnnotData1.ifCond.get(0).clone(), ttCStmt);
							}
							if( targetStmt != null ) {
								aspenCStmt.addASPENStatement(targetStmt);
							}
						} else {
							//aspenCStmt.addASPENStatement(ttCStmt); 
							for( Traversable tCh : ttCStmt.getChildren() ) {
								if( tCh instanceof ASPENStatement ) {
									tCh.setParent(null);
									aspenCStmt.addASPENStatement((ASPENStatement)tCh);
								}
							}
						}
					}
				} else if( at instanceof SwitchStatement ) {
					//[FIXME] IN current implementation, each statement in switch body will be treated
					//equally, ignoring original control flows.
					CompoundStatement swBody = ((SwitchStatement)at).getBody();
					ASPENCompoundStatement ttCStmt = new ASPENCompoundStatement();
					if( aspenAnnotData1.getPreStmt() != null ) {
						ttCStmt.addASPENStatement(aspenAnnotData1.getPreStmt());
					}
					fillASPENStatements(ttCStmt,swBody, proc, procDeclrList);
					if( aspenAnnotData1.getPostStmt() != null ) {
						ttCStmt.addASPENStatement(aspenAnnotData1.getPostStmt());
					}
					//[DEBUG] How does ASPEN handle nested compound statements? 
					//e.g., par { A, {B, C}, D}
					//In current implementation, the nested compound statement will be flattened.
					//if no conditional clause exists.
					if( !ttCStmt.isEmpty() ) { 
						if( isConditional ) {
							if( !aspenAnnotData1.probability.isEmpty() ) {
								ASPENControlProbabilityStatement probStmt = 
									new ASPENControlProbabilityStatement(aspenAnnotData1.probability.get(0).clone(), ttCStmt);
								aspenCStmt.addASPENStatement(probStmt);
							} else if( !aspenAnnotData1.ifCond.isEmpty() ) {
								ASPENControlIfStatement ifStmt = new ASPENControlIfStatement(aspenAnnotData1.ifCond.get(0).clone(), ttCStmt);
								aspenCStmt.addASPENStatement(ifStmt);
							}
						} else {
							//aspenCStmt.addASPENStatement(ttCStmt); 
							for( Traversable tCh : ttCStmt.getChildren() ) {
								if( tCh instanceof ASPENStatement ) {
									tCh.setParent(null);
									aspenCStmt.addASPENStatement((ASPENStatement)tCh);
								}
							}
						}
					}
				} else { //There may exist ASPEN directives even in other statement.
					//e.g., acc update directive exists as standalone annotation statement.
					if( aspenAnnotData1.getPreStmt() != null ) {
						aspenCStmt.addASPENStatement(aspenAnnotData1.getPreStmt());
					}
					if( aspenStmt != null ) {
						aspenCStmt.addASPENStatement(aspenStmt);
					}
					if( aspenAnnotData1.getPostStmt() != null ) {
						aspenCStmt.addASPENStatement(aspenAnnotData1.getPostStmt());
					}
				}
			}
			//Reset inASPENModelRegion if aspen modelregion directive is found.
			if( aspenAnnotData1.aspenModelRegion ) {
				if( aspenAnnotData1.modelRegionType == 0 ) { //aspen modelregion directive found.
					inASPENModelRegion = false;
				}
			}
		}
		if( !IRsToRemove.isEmpty() ) {
			for( Traversable removeTR : IRsToRemove ) {
				Traversable removeTRP = removeTR.getParent();
				if( removeTRP != null ) {
					//System.out.println("IR to be remove:");
					//System.out.println(removeTR + "\nEnclosing context:\n" + AnalysisTools.getEnclosingContext(removeTR));
					removeTRP.removeChild(removeTR);
				}
			}
		}
		if( !aspenCStmt.isEmpty() ) {
			ASPENControlExecuteStatement blockStmt = null;
			for( Traversable ttt : aspenCStmt.getChildren() ) {
				if( ttt instanceof ASPENStatement ) {
					ttt.setParent(null);
					boolean isMerged = false;
					if( ttt instanceof ASPENControlExecuteStatement ) {
						ASPENControlExecuteStatement curBlockStmt = (ASPENControlExecuteStatement)ttt;
/*						List<ASPENExposesExpressionStatement> expList = 
							curBlockStmt.getExecuteBody().getASPENStatements(ASPENExposesExpressionStatement.class);*/
						ASPENResource parallelism = curBlockStmt.getExecuteParallelism();
						if( blockStmt == null ) {
							if( parallelism == null ) {
								//This is a simple block statement.
								blockStmt = curBlockStmt;
							}
						} else {
							if( parallelism == null ) {
								//This is a simple block statement.
								//Merge this to the existing block statement.
								isMerged = true;
								ASPENCompoundStatement bCStmt = blockStmt.getExecuteBody();
								ASPENCompoundStatement curBCStmt = curBlockStmt.getExecuteBody();
								for( Traversable tc : curBCStmt.getChildren() ) {
									tc.setParent(null);
									bCStmt.addASPENStatement((ASPENStatement)tc);
								}
							} else {
								blockStmt = null;
							}
						}
					} else if( blockStmt != null ) {
						blockStmt = null;
					}
					if( !isMerged ) {
						aspenCStmtOpt.addASPENStatement((ASPENStatement)ttt);
					}
				}
			}
		}
	}
	
	private void inlineKernels(ASPENCompoundStatement cStmt, boolean inlining) {
		if( cStmt == null ) {
			return;
		}
		ASPENControlExecuteStatement pBlockStmt = null;
		List<Traversable> childList = new ArrayList<Traversable>(cStmt.getChildren().size());
		childList.addAll(cStmt.getChildren());
		for( Traversable t : childList ) {
			if( t instanceof ASPENStatement ) {
				ASPENStatement aStmt = (ASPENStatement)t;
				if( aStmt instanceof ASPENControlExecuteStatement ) {
					ASPENControlExecuteStatement curBlockStmt = (ASPENControlExecuteStatement)aStmt;
					ASPENResource parallelism = curBlockStmt.getExecuteParallelism();
					if( pBlockStmt == null ) {
						if( parallelism == null ) {
							//This is a simple block statement.
							pBlockStmt = curBlockStmt;
						}
					} else {
						if( parallelism == null ) {
							//This is a simple block statement.
							//Merge this to the existing block statement.
							IRMerged = true;
							ASPENCompoundStatement bCStmt = pBlockStmt.getExecuteBody();
							ASPENCompoundStatement curBCStmt = curBlockStmt.getExecuteBody();
							for( Traversable tc : curBCStmt.getChildren() ) {
								tc.setParent(null);
								bCStmt.addASPENStatement((ASPENStatement)tc);
							}
							curBlockStmt.detach();
						} else {
							pBlockStmt = null;
						}
					}
					continue;
				} else if( aStmt instanceof ASPENControlIfStatement ) {
					pBlockStmt = null;
					inlineKernels(((ASPENControlIfStatement)aStmt).getIfBody(), inlining);
					inlineKernels(((ASPENControlIfStatement)aStmt).getElseBody(), inlining);
				} else if( aStmt instanceof ASPENControlIterateStatement ) {
					boolean isMerged = false;
					ASPENCompoundStatement ttCStmt = ((ASPENControlIterateStatement)aStmt).getIterateBody();
					if( ttCStmt.getChildren().size() == 1 ) {
						Traversable tt = ttCStmt.getChildren().get(0);
						if( tt instanceof ASPENControlExecuteStatement ) {
							ASPENControlExecuteStatement blockStmt = (ASPENControlExecuteStatement)tt;
							if( blockStmt.getExecuteParallelism() == null ) {
								isMerged = true;
								IRMerged = true;
								Expression itrCnt = ((ASPENControlIterateStatement)aStmt).getItrCnt();
								for( Traversable ttt : blockStmt.getExecuteBody().getChildren() ) {
									if( ttt instanceof ASPENMemoryExpressionStatement ) {
										ASPENData tData = ((ASPENMemoryExpressionStatement)ttt).getASPENData();
										Expression newCapacity = new BinaryExpression(tData.getCapacity().clone(),
												BinaryOperator.MULTIPLY, itrCnt.clone());
										tData.setCapacity(newCapacity);
									} else if( ttt instanceof ASPENRequiresExpressionStatement ) {
										ASPENResource tRSC = ((ASPENRequiresExpressionStatement)ttt).getASPENResource();
										Expression newValue = new BinaryExpression(tRSC.getValue().clone(),
												BinaryOperator.MULTIPLY, itrCnt.clone());
										tRSC.setValue(newValue);
									} else {
										Tools.exit("[ERROR in ASPENModelGen.inlineKernels()] unexpected child is found " +
												"in the following ASPENControlExecuteStatement:\n" + blockStmt + "\n");
									}
								}
								blockStmt.detach();
								aStmt.swapWith(blockStmt);
								if( pBlockStmt != null ) {
									//Add to existing execute block.
									blockStmt.detach();
									ASPENCompoundStatement bCStmt = pBlockStmt.getExecuteBody();
									ASPENCompoundStatement curBCStmt = blockStmt.getExecuteBody();
									for( Traversable tc : curBCStmt.getChildren() ) {
										tc.setParent(null);
										bCStmt.addASPENStatement((ASPENStatement)tc);
									}
								}
							}
						}
					}
					if( !isMerged ) {
						pBlockStmt = null;
						inlineKernels(((ASPENControlIterateStatement)aStmt).getIterateBody(), inlining);
					}
				} else if( aStmt instanceof ASPENControlKernelCallStatement ) {
					pBlockStmt = null;
					if( inlining ) {
						IRMerged = true;
						IDExpression kernelID = ((ASPENControlKernelCallStatement)aStmt).getKernelID();
						ASPENKernel kernel = aspenModel.getKernel(kernelID);
						if( kernel == null ) {
							Tools.exit("[ERROR in ASPENModelGen.inlineKernels()] cannot find " +
									"an ASPENKernel named " + kernelID + "; exit!\n");
						} else {
							ASPENCompoundStatement kernelBody = kernel.getKernelBody();
							List<Traversable> inlineList = getInlineList(kernelBody);
							ASPENControlExecuteStatement blockStmt = null;
							for( Traversable ttt : inlineList ) {
								if( ttt instanceof ASPENStatement ) {
									boolean isMerged = false;
									if( ttt instanceof ASPENControlExecuteStatement ) {
										ASPENControlExecuteStatement curBlockStmt = (ASPENControlExecuteStatement)ttt;
										ASPENResource parallelism = curBlockStmt.getExecuteParallelism();
										if( blockStmt == null ) {
											if( parallelism == null ) {
												//This is a simple block statement.
												blockStmt = curBlockStmt;
											}
										} else {
											if( parallelism == null ) {
												//This is a simple block statement.
												//Merge this to the existing block statement.
												isMerged = true;
												ASPENCompoundStatement bCStmt = blockStmt.getExecuteBody();
												ASPENCompoundStatement curBCStmt = curBlockStmt.getExecuteBody();
												for( Traversable tc : curBCStmt.getChildren() ) {
													tc.setParent(null);
													bCStmt.addASPENStatement((ASPENStatement)tc);
												}
											} else {
												blockStmt = null;
											}
										}
									} else if( blockStmt != null ) {
										blockStmt = null;
									}
									if( !isMerged ) {
										cStmt.addASPENStatementBefore(aStmt, (ASPENStatement)ttt);
									}
								}
							}
							cStmt.removeASPENStatement(aStmt);
						}
					}
				} else if( aStmt instanceof ASPENControlMapStatement ) {
					pBlockStmt = null;
					inlineKernels(((ASPENControlMapStatement)aStmt).getMapBody(), true);
				} else if( aStmt instanceof ASPENControlParallelStatement ) {
					pBlockStmt = null;
					inlineKernels(((ASPENControlParallelStatement)aStmt).getParallelBody(), inlining);
				} else if( aStmt instanceof ASPENControlProbabilityStatement ) {
					pBlockStmt = null;
					inlineKernels(((ASPENControlProbabilityStatement)aStmt).getIfBody(), inlining);
					inlineKernels(((ASPENControlProbabilityStatement)aStmt).getElseBody(), inlining);
				} else if( aStmt instanceof ASPENControlSeqStatement ) {
					pBlockStmt = null;
					inlineKernels(((ASPENControlSeqStatement)aStmt).getSeqBody(), inlining);
				} else {
					Tools.exit("[ERROR in ASPENModelGen.inlineKernels()] Only ASPENControlStatement is allowed, but " +
							"the following ASPENStatement is found:\n" + aStmt + "\n" +
									"Enclosing statement:\n" + cStmt + "\n");
				}
			} else {
				Tools.exit("[ERROR in ASPENModelGen.inlineKernels()] unexpected child is found " +
						"in the following ASPENCompoundStatement:\n" + cStmt + "\n");
			}
		}
	}
	
	private List<Traversable> getInlineList(Traversable aaStmt) {
		List<Traversable> inlineList = new LinkedList<Traversable>();
		if( aaStmt instanceof ASPENCompoundStatement ) {
			for( Traversable t : ((ASPENCompoundStatement)aaStmt).getChildren() ) {
				inlineList.addAll(getInlineList(t));
			}
		} else if( aaStmt instanceof ASPENExpressionStatement ) {
			inlineList.add(((ASPENExpressionStatement) aaStmt).clone());
		} else if( aaStmt instanceof ASPENControlExecuteStatement ) {
			inlineList.add(((ASPENControlExecuteStatement) aaStmt).clone());
		} else if( aaStmt instanceof ASPENControlIfStatement ) {
			ASPENControlIfStatement clonedStmt = (ASPENControlIfStatement)((ASPENControlIfStatement)aaStmt).clone();
			inlineKernels(clonedStmt.getIfBody(), true);
			inlineKernels(clonedStmt.getElseBody(), true);
			inlineList.add(clonedStmt);
		} else if( aaStmt instanceof ASPENControlIterateStatement ) {
			ASPENControlIterateStatement clonedStmt = (ASPENControlIterateStatement)((ASPENControlIterateStatement)aaStmt).clone();
			inlineKernels(clonedStmt.getIterateBody(), true);
			inlineList.add(clonedStmt);
		} else if( aaStmt instanceof ASPENControlKernelCallStatement ) {
				IDExpression kernelID = ((ASPENControlKernelCallStatement)aaStmt).getKernelID();
				ASPENKernel kernel = aspenModel.getKernel(kernelID);
				if( kernel == null ) {
					Tools.exit("[ERROR in ASPENModelGen.getInlineList()] cannot find " +
							"an ASPENKernel named " + kernelID + "; exit!\n");
				} else {
					ASPENCompoundStatement kernelBody = kernel.getKernelBody();
					inlineList.addAll(getInlineList(kernelBody));
				}
		} else if( aaStmt instanceof ASPENControlMapStatement ) {
			ASPENControlMapStatement clonedStmt = (ASPENControlMapStatement)((ASPENControlMapStatement)aaStmt).clone();
			inlineKernels(clonedStmt.getMapBody(), true);
			inlineList.add(clonedStmt);
		} else if( aaStmt instanceof ASPENControlParallelStatement ) {
			ASPENControlParallelStatement clonedStmt = (ASPENControlParallelStatement)((ASPENControlParallelStatement)aaStmt).clone();
			inlineKernels(clonedStmt.getParallelBody(), true);
			inlineList.add(clonedStmt);
		} else if( aaStmt instanceof ASPENControlProbabilityStatement ) {
			ASPENControlProbabilityStatement clonedStmt = (ASPENControlProbabilityStatement)((ASPENControlProbabilityStatement)aaStmt).clone();
			inlineKernels(clonedStmt.getIfBody(), true);
			inlineKernels(clonedStmt.getElseBody(), true);
			inlineList.add(clonedStmt);
		} else if( aaStmt instanceof ASPENControlSeqStatement ) {
			ASPENControlSeqStatement clonedStmt = (ASPENControlSeqStatement)((ASPENControlSeqStatement)aaStmt).clone();
			inlineKernels(clonedStmt.getSeqBody(), true);
			inlineList.add(clonedStmt);
		} else {
				Tools.exit("[ERROR in ASPENModelGen.getInlineList()] Only ASPENControlStatement is allowed, but " +
						"the following ASPENStatement is found:\n" + aaStmt + "\n");
		}

		return inlineList;
	}
	private void mergeMaps(ASPENCompoundStatement cStmt) {
		if( cStmt == null ) {
			return;
		}
		List<Traversable> childList = new ArrayList<Traversable>(cStmt.getChildren().size());
		childList.addAll(cStmt.getChildren());
		for( Traversable t : childList ) {
			if( t instanceof ASPENStatement ) {
				if( t instanceof ASPENControlMapStatement ) {
					ASPENControlMapStatement mapStmt = (ASPENControlMapStatement)t;
					while( mapStmt != null ) {
						Expression mapCnt = mapStmt.getMapCnt();
						ASPENCompoundStatement mBody = mapStmt.getMapBody();
						if( mBody.getChildren().size() == 1 ) {
							Traversable tt = mBody.getChildren().get(0);
							if( tt instanceof ASPENControlExecuteStatement ) {
								ASPENControlExecuteStatement exeStmt = (ASPENControlExecuteStatement)tt;
								ASPENResource pRSC = exeStmt.getExecuteParallelism();
								if( pRSC != null ) {
									pRSC.setValue(new BinaryExpression(pRSC.getValue().clone(),
											BinaryOperator.MULTIPLY, mapCnt.clone()));
								} else {
									pRSC = new ASPENResource(mapCnt.clone());
									exeStmt.setExecuteParallelism(pRSC);
								}
								exeStmt.detach();
								exeStmt.swapWith(mapStmt);
								mapStmt = null;
							} else if( tt instanceof ASPENControlMapStatement ) {
								ASPENControlMapStatement mapStmt2 = (ASPENControlMapStatement)tt;
								Expression mapCnt2 = mapStmt2.getMapCnt();
								mapStmt2.setMapCnt(new BinaryExpression(mapCnt.clone(), BinaryOperator.MULTIPLY,
										mapCnt2.clone()));
								mapStmt2.detach();
								mapStmt2.swapWith(mapStmt);
								mapStmt = mapStmt2;
							} else {
								mergeMaps(mBody);
								mapStmt = null;
							}
						} else {
							mergeMaps(mBody);
							mapStmt = null;
						}
					}
				} else if( t instanceof ASPENControlIfStatement ) {
					ASPENControlIfStatement ifStmt = (ASPENControlIfStatement)t;
					mergeMaps(ifStmt.getIfBody());
					mergeMaps(ifStmt.getElseBody());
				} else if( t instanceof ASPENControlProbabilityStatement ) {
					ASPENControlProbabilityStatement probStmt = (ASPENControlProbabilityStatement)t;
					mergeMaps(probStmt.getIfBody());
					mergeMaps(probStmt.getElseBody());
				} else if( t instanceof ASPENControlStatement ) {
					if( !(t instanceof ASPENControlExecuteStatement) ) {
						ASPENCompoundStatement bodyStmt = ((ASPENControlStatement)t).getBody();
						mergeMaps(bodyStmt);
					}
				} else {
					Tools.exit("[ERROR in ASPENModelGen.mergeMaps()] Only ASPENControlStatement is allowed, but " +
							"the following ASPENStatement is found:\n" + t + "\n" +
									"Enclosing statement:\n" + cStmt + "\n");
				}
			} else {
				Tools.exit("[ERROR in ASPENModelGen.inlineKernels()] unexpected child is found " +
						"in the following ASPENCompoundStatement:\n" + cStmt + "\n");
			}
		}
	}

}
