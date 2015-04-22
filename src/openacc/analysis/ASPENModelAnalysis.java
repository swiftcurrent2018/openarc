/**
 * 
 */
package openacc.analysis;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.StringTokenizer;
import java.util.TreeMap;

import openacc.hir.ACCAnnotation;
import openacc.hir.ASPENAnnotation;
import openacc.hir.ASPENCompoundStatement;
import openacc.hir.ASPENControlExecuteStatement;
import openacc.hir.ASPENControlIfStatement;
import openacc.hir.ASPENControlIterateStatement;
import openacc.hir.ASPENControlKernelCallStatement;
import openacc.hir.ASPENControlMapStatement;
import openacc.hir.ASPENControlProbabilityStatement;
import openacc.hir.ASPENControlStatement;
import openacc.hir.ASPENData;
import openacc.hir.ASPENDataDeclaration;
import openacc.hir.ASPENExpression;
import openacc.hir.ASPENKernel;
import openacc.hir.ASPENModel;
import openacc.hir.ASPENParam;
import openacc.hir.ASPENParamDeclaration;
import openacc.hir.ASPENResource;
import openacc.hir.ASPENStatement;
import openacc.hir.ASPENTrait;
import openacc.transforms.ASPENModelGen;

import cetus.analysis.AnalysisPass;
import cetus.analysis.CFGraph;
import cetus.analysis.CallGraph;
import cetus.analysis.LoopTools;
import cetus.exec.Driver;
import cetus.hir.AccessExpression;
import cetus.hir.Annotatable;
import cetus.hir.AnnotationDeclaration;
import cetus.hir.AnnotationStatement;
import cetus.hir.ArrayAccess;
import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.CommaExpression;
import cetus.hir.CompoundStatement;
import cetus.hir.ConditionalExpression;
import cetus.hir.DFIterator;
import cetus.hir.Declaration;
import cetus.hir.DeclarationStatement;
import cetus.hir.Declarator;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.FloatLiteral;
import cetus.hir.ForLoop;
import cetus.hir.FunctionCall;
import cetus.hir.IDExpression;
import cetus.hir.IRTools;
import cetus.hir.Identifier;
import cetus.hir.IfStatement;
import cetus.hir.Initializer;
import cetus.hir.IntegerLiteral;
import cetus.hir.Literal;
import cetus.hir.Loop;
import cetus.hir.MinMaxExpression;
import cetus.hir.NameID;
import cetus.hir.NestedDeclarator;
import cetus.hir.OmpAnnotation;
import cetus.hir.PointerSpecifier;
import cetus.hir.PrintTools;
import cetus.hir.Procedure;
import cetus.hir.ProcedureDeclarator;
import cetus.hir.Program;
import cetus.hir.PseudoSymbol;
import cetus.hir.ReturnStatement;
import cetus.hir.SizeofExpression;
import cetus.hir.Specifier;
import cetus.hir.StandardLibrary;
import cetus.hir.Statement;
import cetus.hir.StringLiteral;
import cetus.hir.SwitchStatement;
import cetus.hir.Symbol;
import cetus.hir.SymbolTable;
import cetus.hir.SymbolTools;
import cetus.hir.Symbolic;
import cetus.hir.Tools;
import cetus.hir.TranslationUnit;
import cetus.hir.Traversable;
import cetus.hir.Typecast;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;

/**
 * @author lee222<lees2@ornl.gov>
 *         Future Technologies Group
 *         Oak Ridge National Laboratory
 */
public class ASPENModelAnalysis extends AnalysisPass {
	private boolean IRSymbolOnly = true;
	private boolean assumeNonZeroTripLoops = false;
	private boolean inASPENModelRegion = false;
	private boolean inComputeRegion = false;
	private Procedure main = null;
	private TranslationUnit mainTrUnt = null;
	private ASPENModel aspenModel = null;
	private	int exeBlockCnt = 0;
    private Map<Symbol, Annotatable> paramMap;
    private Map<Symbol, Annotatable> dataMap;
    private Set<IDExpression> kernelSet;
    private Map<IDExpression, Annotatable> internalParamMap;
    private Map<Symbol, Symbol> orgDataSymMap;
    private List osymList;
    private List<FunctionCall> gFuncCallList;
    static public List<String> internalParamPrefixList = 
    	new ArrayList<String>(Arrays.asList("HI_", "aspen_"));
    
	private Set<String> memTrClauseSet;
    
	public static final Set<String> pseudoStandardLibrary = new HashSet(Arrays.asList("_mm_malloc", "_free"));
	
	/**
	 * @param program
	 */
	public ASPENModelAnalysis(Program program, boolean IRSymOnly) {
		super(program);
		IRSymbolOnly = IRSymOnly;
		paramMap = new HashMap<Symbol, Annotatable>();
		dataMap = new HashMap<Symbol, Annotatable>();
		kernelSet = new HashSet<IDExpression>();
		internalParamMap = new HashMap<IDExpression, Annotatable>();
		orgDataSymMap = new HashMap<Symbol, Symbol>();
		osymList = new ArrayList(2);
		memTrClauseSet = new HashSet<String>();
		memTrClauseSet.addAll(ACCAnnotation.memTrDataClauses);
		memTrClauseSet.addAll(ACCAnnotation.memTrUpdateClauses);
	}

	/* (non-Javadoc)
	 * @see cetus.analysis.AnalysisPass#getPassName()
	 */
	@Override
	public String getPassName() {
		return new String("[ASPENModelAnalysis]");
	}
	
	public static boolean isInternalParam(String IDName) {
		boolean isInternalParameter = false;
		for( String prefix : internalParamPrefixList ) {
			if( IDName.startsWith(prefix) ) {
				isInternalParameter = true;
				break;
			}
		}
		return isInternalParameter;
	}
	
	public static class ASPENConfiguration {
		private static final String MODELNAME = "modelname";
		private static final String FUNCTIONS = "functions";
		private static final String ENTRYFUNCTION = "entryfunction";
		private static final String COMPLEMENT = "complement";
		private static final String MODE = "mode";
		private static final String POSTPROCESSING = "postprocessing";
		public static String mainEntryFunc = null;
		public static String modelName = null;
		public static int mode = -1;
		public static int postprocessing = 2;
		public static ArrayList<String> functionNames = new ArrayList<String>();
		public static boolean complementFunctions = false;
		public static Set<String> accessedFunctions = new HashSet<String>();
		public static boolean modelRegionIsFunction = true;
		public static void setConfiguration(String options) {
			if( !options.equals("1") ) { //otions.equals("1") means no suboption is specified.
				StringTokenizer tokenizer = new StringTokenizer(options, ":");
				String option;
				while(tokenizer.hasMoreTokens()) {
					option = tokenizer.nextToken();
					int eqIndex = option.indexOf('='); 
					if( eqIndex != -1) {
						String opt = option.substring(0, eqIndex).trim();
						if(opt.equals(FUNCTIONS)) {
							StringTokenizer funcs = new StringTokenizer(option.substring(eqIndex+1), " ,");
							while(funcs.hasMoreTokens()) {
								functionNames.add(funcs.nextToken().trim());
							}
						} else if(opt.equals(MODELNAME)) {
							modelName = option.substring(eqIndex+1).trim();
						} else if(opt.equals(ENTRYFUNCTION)) {
							mainEntryFunc = option.substring(eqIndex+1).trim();
						} else {
							try {
								int value = new Integer(option.substring(eqIndex+1).trim()).intValue();
								if(opt.equals(COMPLEMENT)) {
									complementFunctions = (value == 1? true : false);
								} else if( opt.equals(MODE)) {
									mode = value;
								} else if( opt.equals(POSTPROCESSING)) {
									postprocessing = value;
								}
							}
							catch(NumberFormatException ex){
								PrintTools.println("[ERROR in ASPENModelAnalysis.ASPENConfiguration.setConfiguration()] " +
										"error in reading ASPENModelGen option; this option will be ignored!", 0);
								break;
							}
						}
					} else {
						PrintTools.println("[WARNING in ASPENModelAnalysis.ASPENConfiguration.setConfiguration()] " +
								"invalid ASPENModelGen option (" + option + ") will be ignored!", 0);
					}
				}
			}
			if( mode == -1 ) {
				mode = 3; //default option.
			}
			if( mainEntryFunc == null ) {
				String value = Driver.getOptionValue("SetAccEntryFunction");
				if( (value != null) && !value.equals("1") ) {
					mainEntryFunc = value;
				}
			}
		}
	}


	/* (non-Javadoc)
	 * @see cetus.analysis.AnalysisPass#start()
	 */
	@Override
	public void start() {
		String mainEntryFunc = ASPENModelAnalysis.ASPENConfiguration.mainEntryFunc;
		ArrayList<String> functionNames = ASPENModelAnalysis.ASPENConfiguration.functionNames;
		boolean complementFunctions = ASPENModelAnalysis.ASPENConfiguration.complementFunctions;
		
		assumeNonZeroTripLoops = false;
		String value = Driver.getOptionValue("assumeNonZeroTripLoops");
		if( value != null ) {
			assumeNonZeroTripLoops = true;
		}
		
		//////////////////////////////////////////////////////////////////////////////////////
		// [Step 0] Find an entry function, which contains either "aspen enter modelregion" //
		// or "aspen modelregion" directive.                                                //
		// If no such function is found, the main function becomes the entry function.      //
		//////////////////////////////////////////////////////////////////////////////////////
		List<ASPENAnnotation> modelRegionAnnots = IRTools.collectPragmas(program, ASPENAnnotation.class, "modelregion");
		for(ASPENAnnotation aAnnot : modelRegionAnnots ) {
			if( aAnnot.containsKey("exit") ) {
				continue;
			} else {
				//Found the entry model region.
				Annotatable at = aAnnot.getAnnotatable();
				main = IRTools.getParentProcedure(at);
				ASPENModelAnalysis.ASPENConfiguration.mainEntryFunc = main.getSymbolName();
				if( at == main ) {
					inASPENModelRegion = true;
				} else {
					ASPENModelAnalysis.ASPENConfiguration.modelRegionIsFunction = false;
				}
				//[FIXME] for now, we assume that there is only one modelregion directive.
				break;
			}
		}
		
		if( main == null ) {
			main = AnalysisTools.findMainEntryFunction(program, mainEntryFunc);
			inASPENModelRegion = true;
		}
		if( main == null ) {
			Tools.exit("\n[ERROR in ASPENModelAnalysis] This analysis pass is skipped " +
					"since the compiler can not find the main entry function; " +
					"if the input program does not have a main function, user should specify a main entry function " +
					"using SetAccEntryFunction option.\n");
			return;
		} else {
			mainTrUnt = (TranslationUnit)main.getParent(); 
		}
		
		
		//////////////////////////////////////////////////////////////////////////////////
		// [Step 1] Annotate functions to ignore with "aspen control ignore" directive. //
		//////////////////////////////////////////////////////////////////////////////////
		List<Procedure> procList = IRTools.getProcedureList(program);
		for( Procedure tProc : procList ) {
			String procName = tProc.getSymbolName();
			if( complementFunctions ) {
				if( !functionNames.contains(procName) ) {
					ASPENAnnotation ignoreAnnot = tProc.getAnnotation(ASPENAnnotation.class, "ignore");
					if( ignoreAnnot == null ) {
						ignoreAnnot = tProc.getAnnotation(ASPENAnnotation.class, "control");
						if( ignoreAnnot == null ) {
							ignoreAnnot = new ASPENAnnotation();
							ignoreAnnot.put("control", "_directive");
							tProc.annotate(ignoreAnnot);
						}
						ignoreAnnot.put("ignore", "_clause");
					}
				}
			} else {
				if( functionNames.contains(procName) ) {
					ASPENAnnotation ignoreAnnot = tProc.getAnnotation(ASPENAnnotation.class, "ignore");
					if( ignoreAnnot == null ) {
						ignoreAnnot = tProc.getAnnotation(ASPENAnnotation.class, "control");
						if( ignoreAnnot == null ) {
							ignoreAnnot = new ASPENAnnotation();
							ignoreAnnot.put("control", "_directive");
							tProc.annotate(ignoreAnnot);
						}
						ignoreAnnot.put("ignore", "_clause");
					}
				}
			}
		}
		
		gFuncCallList = IRTools.getFunctionCalls(program);
		
		///////////////////////////////////////////////
		//[Step 2] Add global ASPEN parameters/data. //
		///////////////////////////////////////////////
		List<Traversable> trList = program.getChildren();
		for( Traversable trUnt : trList ) {
			List<Traversable> declList = new ArrayList<Traversable>();
			declList.addAll(trUnt.getChildren());
			for( Traversable tr : declList ) {
				ASPENAnnotation aspenAnnot = ((Annotatable)tr).getAnnotation(ASPENAnnotation.class, "ignore");
				if( aspenAnnot != null ) {
					continue;
				}
				if( tr instanceof AnnotationDeclaration ) {
					//[Step 2-1] Handle existing ASPEN param annotations.
					AnnotationDeclaration annot = (AnnotationDeclaration)tr;
					aspenAnnot = annot.getAnnotation(ASPENAnnotation.class, "param");
					if( aspenAnnot != null ) {
						Set<ASPENParam> paramSet = aspenAnnot.get("param");
						for( ASPENParam aParam : paramSet ) {
							Symbol tSym = null;
							IDExpression ID = aParam.getID();
							if( ID instanceof Identifier ) {
								tSym = ((Identifier)ID).getSymbol();
							}
							if( tSym == null ) {
								String tSymName = ID.toString();
								if( isInternalParam(tSymName) ) {
									if( !internalParamMap.containsKey(ID) ) {
										internalParamMap.put(ID.clone(), annot);
									}
								} else {
									Tools.exit("[ERROR in ASPENModelAnalysis] can not find symbol for the global variable " +
											"(" + ID + ") " + 
											"used in the following ASPEN Param annotation:\n" +
											"ASPEN annotation: " + aspenAnnot + AnalysisTools.getEnclosingAnnotationContext(aspenAnnot));
								}
							} else {
								if( !paramMap.containsKey(tSym) ) {
									paramMap.put(tSym, annot);
								}
							}
						}
					}
					//[Step 2-2] Handle existing ASPEN data annotations.
					aspenAnnot = annot.getAnnotation(ASPENAnnotation.class, "data");
					if( aspenAnnot != null ) {
						Set<ASPENData> dataSet = aspenAnnot.get("data");
						for( ASPENData aData : dataSet ) {
							Symbol tSym = null;
							IDExpression ID = aData.getID();
							if( ID instanceof Identifier ) {
								tSym = ((Identifier)ID).getSymbol();
							}
							if( tSym == null ) {
								Tools.exit("[ERROR in ASPENModelAnalysis] can not find symbol for the global variable " +
										"(" + ID + ") " + 
										"used in the following ASPEN Data annotation:\n" +
										"ASPEN annotation: " + aspenAnnot + AnalysisTools.getEnclosingAnnotationContext(aspenAnnot));
							} else {
								if( !dataMap.containsKey(tSym) ) {
									dataMap.put(tSym, annot);
									osymList.clear();
									if( AnalysisTools.SymbolStatus.OrgSymbolFound(
											AnalysisTools.findOrgSymbol(tSym, annot, true, null, osymList, gFuncCallList)) ) {
										//If the data symbol is aliased, keep the mapping of the symbol to the original symbol.
										orgDataSymMap.put(tSym, (Symbol)osymList.get(0));
									} else {
										orgDataSymMap.put(tSym, tSym);
									}
								}
							}
						}
					}
				} else if( tr instanceof VariableDeclaration ) {
					//[Step 2-3] Analyze variable declarations to extract ASPEN param/data 
					//annotations.
					VariableDeclaration vDecl = (VariableDeclaration)tr;
					if( !(vDecl.getDeclarator(0) instanceof ProcedureDeclarator) ) {
						handleVariableDeclaration(vDecl, (TranslationUnit)trUnt);
					}
				}
			}
		}
		
		/////////////////////////////////////////////////////////////////////
		//[Step 3] Analyze each procedure starting from the entry function //
		/////////////////////////////////////////////////////////////////////
		exeBlockCnt = 0;
		ASPENConfiguration.accessedFunctions.add(main.getSymbolName());
		CompoundStatement pBody = main.getBody();
		//Handle formal parameters of the entry function.
		List<Declaration> paramList = main.getParameters();
		if( paramList != null ) {
			int list_size = paramList.size();
			if( list_size == 1 ) {
				Object obj = paramList.get(0);
				String paramS = obj.toString();
				// Remove any leading or trailing whitespace.
				paramS = paramS.trim();
				if( paramS.equals(Specifier.VOID.toString()) ) {
					list_size = 0;
				}
			}
			if( list_size > 0 ) {
				for( Declaration pDecl : paramList ) {
					if( pDecl instanceof VariableDeclaration ) {
						handleVariableDeclaration((VariableDeclaration)pDecl, main);
					}
				}
			}
		}
		analyzeASPENModel(pBody, main);
		kernelSet.add(main.getName().clone());
	}
	
	private void analyzeASPENModel(CompoundStatement inputBody,
			Procedure proc) {
		String blockName = "block_" + proc.getSymbolName();
		ASPENModelGen.ASPENAnnotData aspenAnnotData1 = new ASPENModelGen.ASPENAnnotData();
		List<Traversable> inChildren = new ArrayList<Traversable>(inputBody.getChildren().size());
		inChildren.addAll(inputBody.getChildren());
		for( Traversable child : inChildren ) {
			aspenAnnotData1.resetASPENAnnotData();
			Annotatable at = (Annotatable)child;
			boolean foundComputeRegion = false;
			if( at.containsAnnotation(ACCAnnotation.class, "kernels") 
					|| at.containsAnnotation(ACCAnnotation.class, "parallel") ) {
				foundComputeRegion = true;
				inComputeRegion = true;
			}
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
					if( aspenAnnotData1.modelName != null ) {
						ASPENModelAnalysis.ASPENConfiguration.modelName = aspenAnnotData1.modelName;
					}
				} else if( aspenAnnotData1.aspenDeclare ) {
					Set<ASPENParam> removeSet1 = new HashSet<ASPENParam>();
					for( ASPENParam aParam : aspenAnnotData1.paramSet ) {
						Symbol tSym = null;
						IDExpression ID = aParam.getID();
						if( ID instanceof Identifier ) {
							tSym = ((Identifier)ID).getSymbol();
						}
						if( tSym == null ) {
							String tSymName = ID.toString();
							if(isInternalParam(tSymName) ) {
								if( !internalParamMap.containsKey(ID) ) {
									internalParamMap.put(ID.clone(), at);
								}
							} else {
								ASPENAnnotation tAnnot = at.getAnnotation(ASPENAnnotation.class, "param");
								Tools.exit("[ERROR in ASPENModelAnalysis] can not find symbol for the variable " +
										"(" + ID + ") " + 
										"used in the following ASPEN Param annotation:\n" +
										"ASPEN annotation: " + tAnnot + AnalysisTools.getEnclosingAnnotationContext(tAnnot));
							}
						} else {
							if( paramMap.containsKey(tSym) ) {
								//Merge two ASPEN Param declarations.
								ASPENParam prevParam = null;
								Expression curInit = aParam.getInitVal();
								Annotatable prevAt = paramMap.get(tSym);
								ASPENAnnotation ttAnnot = prevAt.getAnnotation(ASPENAnnotation.class, "param");
								Set<ASPENParam> prevParamSet = ttAnnot.get("param");
								if( prevParamSet != null ) {
									for( ASPENParam pParam : prevParamSet ) {
										if( pParam.getID().equals(ID) ) {
											Expression prevInit = pParam.getInitVal();
											if( (prevInit != null) && (curInit == null) ) {
												removeSet1.add(aParam);
											} else if( (prevInit == null) && (curInit != null) ) {
												prevParam = pParam;
											}
											break;
										}
									}
									if( prevParam != null ) {
										prevParamSet.remove(prevParam);
										if( prevParamSet.isEmpty() ) {
											//Remove the previous ASPEN Param declaration.
											Traversable prevAtP = prevAt.getParent();
											if( prevAtP != null ) {
												prevAtP.removeChild(prevAt);
											}
										}
										//Update paramMap mapping.
										paramMap.put(tSym, at);
									}
								}
							} else {
								paramMap.put(tSym, at);
							}
						}
					}
					if( !removeSet1.isEmpty() ) {
						aspenAnnotData1.paramSet.removeAll(removeSet1);
						ASPENAnnotation ttAA = at.getAnnotation(ASPENAnnotation.class, "param");
						Set<ASPENParam> currParamSet = ttAA.get("param");
						currParamSet.removeAll(removeSet1);
						if( currParamSet.isEmpty() ) {
							Traversable atP = at.getParent();
							if( atP != null ) {
								atP.removeChild(at);
							}
						}
					}
					Set<ASPENData> removeSet2 = new HashSet<ASPENData>();
					for( ASPENData aData : aspenAnnotData1.dataSet ) {
						Symbol tSym = null;
						IDExpression ID = aData.getID();
						if( ID instanceof Identifier ) {
							tSym = ((Identifier)ID).getSymbol();
						}
						if( tSym == null ) {
							ASPENAnnotation tAnnot = at.getAnnotation(ASPENAnnotation.class, "data");
							Tools.exit("[ERROR in ASPENModelAnalysis] can not find symbol for the variable " +
									"(" + ID + ") " + 
									"used in the following ASPEN Data annotation:\n" +
									"ASPEN annotation: " + tAnnot + AnalysisTools.getEnclosingAnnotationContext(tAnnot));
						} else {
							if( dataMap.containsKey(tSym) ) {
								//Merge two ASPEN Data declarations.
								ASPENData prevData = null;
								Expression curCap = aData.getCapacity();
								int curTraits = aData.getTraitSize();
								Annotatable prevAt = dataMap.get(tSym);
								ASPENAnnotation ttAnnot = prevAt.getAnnotation(ASPENAnnotation.class, "data");
								Set<ASPENData> prevDataSet = ttAnnot.get("data");
								if( prevDataSet != null ) {
									for( ASPENData pData : prevDataSet ) {
										if( pData.getID().equals(ID) ) {
											Expression prevCap = pData.getCapacity();
											int prevTraits = pData.getTraitSize();
											if( (curTraits == 0) && (curCap == null) ) {
												removeSet2.add(aData);
											} else if( (prevCap == null) && (prevTraits == 0) ) {
												prevData = pData;
											}
											break;
										}
									}
									if( prevData != null ) {
										prevDataSet.remove(prevData);
										if( prevDataSet.isEmpty() ) {
											//Remove the previous ASPEN Data declaration.
											Traversable prevAtP = prevAt.getParent();
											if( prevAtP != null ) {
												prevAtP.removeChild(prevAt);
											}
										}
										//Update dataMap mapping.
										dataMap.put(tSym, at);
									}
								}
							} else {
								dataMap.put(tSym, at);
								osymList.clear();
								if( AnalysisTools.SymbolStatus.OrgSymbolFound(
										AnalysisTools.findOrgSymbol(tSym, at, true, null, osymList, gFuncCallList)) ) {
									orgDataSymMap.put(tSym, (Symbol)osymList.get(0));
								} else {
									orgDataSymMap.put(tSym, tSym);
								}
							}
						}
					}
					if( !removeSet2.isEmpty() ) {
						aspenAnnotData1.dataSet.removeAll(removeSet2);
						ASPENAnnotation ttAA = at.getAnnotation(ASPENAnnotation.class, "data");
						Set<ASPENData> currDataSet = ttAA.get("data");
						currDataSet.removeAll(removeSet2);
						if( currDataSet.isEmpty() ) {
							Traversable atP = at.getParent();
							if( atP != null ) {
								atP.removeChild(at);
							}
						}
					}
				}
			}
			
			boolean isConditional = false;
			if( (!aspenAnnotData1.probability.isEmpty()) || (!aspenAnnotData1.ifCond.isEmpty()) ) {
				isConditional = true;
			}
			
			if( !aspenAnnotData1.isAspenBlock ) {
				if( at instanceof ExpressionStatement ) {
					Expression cExpr = ((ExpressionStatement)at).getExpression(); 
					analyzeExpression(cExpr, at, aspenAnnotData1);
				} else if( at instanceof Loop ) {
					Loop ploop = (Loop)at;
					analyzeExpression(ploop.getCondition(), at, aspenAnnotData1);
					CompoundStatement ttBody = (CompoundStatement)ploop.getBody();
					analyzeASPENModel(ttBody, proc);
					if( inASPENModelRegion && (aspenAnnotData1.loopCnt == null) && (aspenAnnotData1.parallelArg == null) ) {
						// check for a canonical loop
						if ( !LoopTools.isCanonical(ploop) ) {
							Tools.exit("\n[ERROR in ASPENModelGen] case1: loop statement without ignore or execute ASPEN clause should have " +
									"loop count information in either loop or parallelism ASPEN clause, unless the compiler can figure out the loop count.\n" +
									"Loop Statement:\n" + at + 
									"\nEnclosing Procedure: " + proc.getSymbolName() +"\nEnclosing Translation Unit: " + 
									((TranslationUnit)proc.getParent()).getOutputFilename() + "\n");
						}
						// check whether loop stride is 1.
						Expression incr = LoopTools.getIncrementExpression(ploop);
						Boolean increasingOrder = true;
						if( incr instanceof IntegerLiteral ) {
							long IntIncr = ((IntegerLiteral)incr).getValue();
							if( IntIncr < 0 ) {
								increasingOrder = false;
							}
							if( Math.abs(IntIncr) != 1 ) {
								Tools.exit("\n[ERROR in ASPENModelGen] case2: loop statement without ignore or execute ASPEN clause should have " +
										"loop count information in either loop or parallelism ASPEN clause, unless the compiler can figure out the loop count.\n" +
										"Loop Statement:\n" + at + 
									"\nEnclosing Procedure: " + proc.getSymbolName() +"\nEnclosing Translation Unit: " + 
									((TranslationUnit)proc.getParent()).getOutputFilename() + "\n");
							}
						} else {
							Tools.exit("\n[ERROR in ASPENModelGen] case3: loop statement without ignore or execute ASPEN clause should have " +
									"loop count information in either loop or parallelism ASPEN clause, unless the compiler can figure out the loop count.\n" +
									"Loop Statement:\n" + at + 
									"\nEnclosing Procedure: " + proc.getSymbolName() +"\nEnclosing Translation Unit: " + 
									((TranslationUnit)proc.getParent()).getOutputFilename() + "\n");

						}
						// identify the loop index variable 
						//Expression ivar = LoopTools.getIndexVariable(ploop);
						Expression lb = Symbolic.simplify(LoopTools.getLowerBoundExpression(ploop));
						Expression ub = Symbolic.simplify(LoopTools.getUpperBoundExpression(ploop));
						Expression tSize = null;
						if( increasingOrder ) {
							tSize = Symbolic.add(Symbolic.subtract(ub,lb),new IntegerLiteral(1));
						} else {
							tSize = Symbolic.add(Symbolic.subtract(lb,ub),new IntegerLiteral(1));
						}
						if( (tSize == null) || !containsParamSymbolsOnly(tSize) ) {
							Tools.exit("\n[ERROR in ASPENModelGen] case4: loop statement without ignore or execute ASPEN clause should have " +
									"loop count information in either loop or parallelism ASPEN clause, unless the compiler can figure out the loop count.\n" +
									"Loop Statement:\n" + at + 
									"\nEnclosing Procedure: " + proc.getSymbolName() +"\nEnclosing Translation Unit: " + 
									((TranslationUnit)proc.getParent()).getOutputFilename() + "\n");
						} else {
							boolean isParallel = false;
							List<ACCAnnotation> accAnnots = at.getAnnotations(ACCAnnotation.class);
							if( accAnnots != null ) {
								for( ACCAnnotation ttAn : accAnnots ) {
									if( ttAn.containsKey("gang") || ttAn.containsKey("worker") || ttAn.containsKey("vector") ||
											ttAn.containsKey("independent") ) {
										isParallel = true;
										break;
									}
								}
							}
							if( !isParallel ) {
								List<OmpAnnotation> ompAnnots = at.getAnnotations(OmpAnnotation.class);
								if( ompAnnots != null ) {
									for( OmpAnnotation ttOAn : ompAnnots ) {
										if( ttOAn.containsKey("for") ) {
											isParallel = true;
											break;
										}
									}
								}
							}
							ASPENAnnotation aspenAnnot = at.getAnnotation(ASPENAnnotation.class, "loop");
							if( aspenAnnot == null ) {
								aspenAnnot = at.getAnnotation(ASPENAnnotation.class, "control");
							}
							if( aspenAnnot == null ) {
								aspenAnnot = new ASPENAnnotation("control", "_directive");
								at.annotate(aspenAnnot);
							}
							if( aspenAnnot != null ) {
								aspenAnnot.put("loop", tSize.clone());
								if( isParallel ) {
									ASPENResource parallelArg = new ASPENResource(tSize.clone());
									aspenAnnot.put("parallelism", parallelArg);
								}
							}
						}
					}
				} else if( at instanceof IfStatement ) {
					Expression ifCond = ((IfStatement)at).getControlExpression();
					Expression simpleCond = null;
					if( ifCond != null ) {
						simpleCond = Symbolic.simplify(ifCond); 
						analyzeExpression(ifCond, at, aspenAnnotData1);
					}
					if( (simpleCond == null) || !simpleCond.equals(new IntegerLiteral(0)) ) {
						CompoundStatement thenBody = (CompoundStatement)((IfStatement)at).getThenStatement();
						analyzeASPENModel(thenBody, proc);
					}
					if( (simpleCond == null) || !simpleCond.equals(new IntegerLiteral(1)) ) {
						CompoundStatement elseBody = (CompoundStatement)((IfStatement)at).getElseStatement();
						if( elseBody != null ) {
							analyzeASPENModel(elseBody, proc);
						}
					}
					if( inASPENModelRegion && (aspenAnnotData1.ifCond.isEmpty()) && (aspenAnnotData1.probability.isEmpty()) ) {
						if( (ifCond == null) || !containsParamSymbolsOnly(ifCond) ) {
							Tools.exit("\n[ERROR in ASPENModelGen] If statement without ignore or execute ASPEN clause should have " +
									"either if or probability ASPEN clause, unless the compiler can figure it out.\n" +
									"If Statement:\n" + at + 
									"\nEnclosing Procedure: " + proc.getSymbolName() +"\nEnclosing Translation Unit: " + 
									((TranslationUnit)proc.getParent()).getOutputFilename() + "\n");
						} else {
							if( (simpleCond != null) && (simpleCond instanceof IntegerLiteral) ) {
								ifCond = simpleCond.clone();
							} else {
								ifCond = ifCond.clone();
								//If ifCond is not binary-comparison, make it an explicit binary-comparison expression,
								//which is required by the current ASPEN model.
								boolean change2BinaryComparison = false;
								if( ifCond instanceof BinaryExpression ) {
									BinaryOperator bOp = ((BinaryExpression)ifCond).getOperator();
									if( bOp.equals(BinaryOperator.COMPARE_EQ) || bOp.equals(BinaryOperator.COMPARE_GE) || 
											bOp.equals(BinaryOperator.COMPARE_GT) || bOp.equals(BinaryOperator.COMPARE_LE) || 
											bOp.equals(BinaryOperator.COMPARE_LT) || bOp.equals(BinaryOperator.COMPARE_NE) ) {
										change2BinaryComparison = false;
									} else {
										change2BinaryComparison = true;
									}
								} else {
									change2BinaryComparison = true;
								}
								if( change2BinaryComparison ) {
									ifCond = new BinaryExpression(ifCond, BinaryOperator.COMPARE_NE, new IntegerLiteral(0));
								}
							}
							ifCond.setParens(false);
							ASPENAnnotation aspenAnnot = at.getAnnotation(ASPENAnnotation.class, "if");
							if( aspenAnnot == null ) {
								aspenAnnot = at.getAnnotation(ASPENAnnotation.class, "control");
							}
							if( aspenAnnot == null ) {
								aspenAnnot = new ASPENAnnotation("control", "_directive");
								at.annotate(aspenAnnot);
							}
							if( aspenAnnot != null ) {
								List<Expression> ifConds = new ArrayList<Expression>(1);
								ifConds.add(ifCond);
								aspenAnnot.put("if", ifConds);
							}
						}
					}
				} else if( at instanceof CompoundStatement ) {
					analyzeASPENModel((CompoundStatement)at, proc);
				} else if( at instanceof SwitchStatement ) {
					//[FIXME] IN current implementation, each statement in switch body will be treated
					//equally, ignoring original control flows.
					CompoundStatement swBody = ((SwitchStatement)at).getBody();
					analyzeASPENModel(swBody, proc);
				} else if( at instanceof DeclarationStatement ) {
					VariableDeclaration vDecl = (VariableDeclaration)((DeclarationStatement)at).getDeclaration();
					handleVariableDeclaration(vDecl, inputBody);
				} else { //what else?
				}
				
				//Handle OpenACC data clauses.
				List<ACCAnnotation> dataAnnots = AnalysisTools.collectPragmas(at, ACCAnnotation.class, ACCAnnotation.memTrDataClauses, false);
				dataAnnots.addAll(IRTools.collectPragmas(at, ACCAnnotation.class, "update"));
				if( inASPENModelRegion && (dataAnnots != null) && !dataAnnots.isEmpty() ) {
					boolean newControlAnnot = false;
					ASPENAnnotation aspenAnnot = at.getAnnotation(ASPENAnnotation.class, "intracomm");
					if( aspenAnnot == null ) {
						aspenAnnot = at.getAnnotation(ASPENAnnotation.class, "control");
					}
					if( aspenAnnot == null ) {
						aspenAnnot = new ASPENAnnotation("control", "_directive");
						newControlAnnot = true;
					}
					Set<ASPENResource> intracommSet = aspenAnnot.get("intracomm");
					if( intracommSet == null ) {
						intracommSet = new HashSet<ASPENResource>();
					}
					Set<SubArray> tSet = null;
					List<Expression> startList = new LinkedList<Expression>();
					List<Expression> lengthList = new LinkedList<Expression>();
					for( ACCAnnotation tAnnot : dataAnnots ) {
						for( String memtrType : memTrClauseSet ) {
							tSet = tAnnot.get(memtrType);
							if( tSet != null ) {
								for(SubArray tSubA : tSet) {
									startList.clear();
									lengthList.clear();
									Symbol lSym = AnalysisTools.subarrayToSymbol(tSubA, IRSymbolOnly);
									if( lSym != null ) {
										if( !SymbolTools.isArray(lSym) && !SymbolTools.isPointer(lSym) ) {
											//Skip scalar variable.
											continue;
										}
										if(AnalysisTools.extractDimensionInfo(tSubA, startList, lengthList, IRSymbolOnly, tAnnot.getAnnotatable())) {
											List specs = lSym.getTypeSpecifiers();
											NameID nParamID = getBuiltInParamForTypes(specs, internalParamMap, mainTrUnt);
											if( nParamID != null ) {
												Expression mSize = nParamID.clone();
												//[FIXME] For now, we assumes that start address is always 0 in startList.
												for( Expression tExp : lengthList ) {
													mSize = new BinaryExpression(mSize, BinaryOperator.MULTIPLY, tExp.clone());
												}
												Expression dID = tSubA.getArrayName().clone();
												Symbol tSym = SymbolTools.getSymbolOf(dID);
												boolean isDataSymbol = false;
												if( tSym != null ) {
													if( orgDataSymMap.containsKey(tSym) ) {
														isDataSymbol = true;
													} else {
														osymList.clear();
														if( AnalysisTools.SymbolStatus.OrgSymbolFound(
																AnalysisTools.findOrgSymbol(tSym, at, true, null, osymList, gFuncCallList)) ) {
															Symbol otSym = (Symbol)osymList.get(0);
															if( orgDataSymMap.containsValue(otSym) ) {
																isDataSymbol = true;
															}
														}
													}
												}
												ASPENResource intraCommData;
												if( isDataSymbol ) {
													intraCommData = new ASPENResource(mSize, null, "to", dID);
												} else {
													intraCommData = new ASPENResource(mSize, null);
												}
												if( memtrType.equals("host") ) {
													intraCommData.addTrait(new ASPENTrait("copyout"));
												} else if( memtrType.equals("device") ) {
													intraCommData.addTrait(new ASPENTrait("copyin"));
												} else {
													intraCommData.addTrait(new ASPENTrait(memtrType));
												}
												boolean rscExist = false;
												for( ASPENResource tRSC : intracommSet ) {
													if( tRSC.getID().equals(dID) ) {
														rscExist = true;
														break;
													}
												}
												if( !rscExist ) {
													intracommSet.add(intraCommData);
												}
											}
										}
									}
								}
							}
						}
					}
					if( !intracommSet.isEmpty() ) {
						aspenAnnot.put("intracomm", intracommSet);
						if( newControlAnnot ) {
							at.annotate(aspenAnnot);
						}
					}
				}
			}
			if( foundComputeRegion ) {
				//Reset inComputeRegion at the end of the region.
				inComputeRegion = false;
			}
			//Reset inASPENModelRegion if aspen modelregion directive is found.
			if( aspenAnnotData1.aspenModelRegion ) {
				if( aspenAnnotData1.modelRegionType == 0 ) { //aspen modelregion directive found.
					inASPENModelRegion = false;
				}
			}
		}
	}
	
	protected void analyzeExpression(Expression cExpr, Annotatable at, ASPENModelGen.ASPENAnnotData aspenAnnotData1) {
		/*					
 					- Analyze the current expression to find flops/loads/stores, and 
		            add these information to the "#pragma aspen control" directive.
		            - If the current expression does not have a user function call,
		                - Add "execute" clause to the ASPEN directive if the enclosing annotatable is ExpressionStatement.
		            - Else
		                - For each function call
		                    - If the function is C library calls, 
		                        - Memory-related calls (malloc, free, memset, etc.) will 
		                        be recognized and necessary ASPEN directive will be added. 
		                        - Other calls will be considered for flops/loads/stores.
		                    - Else if the function is user-function,
		                        - Set the function as the current function.
		 */
		if( (cExpr == null) || (at == null) || (aspenAnnotData1 == null) ) {
			return;
		}
		ExpressionAnalyzer lExpressionAnalyzer = new ExpressionAnalyzer();
		lExpressionAnalyzer.analyzeExpression(cExpr, internalParamMap, mainTrUnt);
		//System.err.println("Analyze Expression in " + at);
		//System.err.println("Loads set: " + lExpressionAnalyzer.LOADS);
		//System.err.println("Stores set: " + lExpressionAnalyzer.STORES);
		ASPENAnnotation controlAnnot = at.getAnnotation(ASPENAnnotation.class, "control");
		if( (controlAnnot == null) ) {
			controlAnnot = new ASPENAnnotation("control", "_directive");
			at.annotate(controlAnnot);
		}
		if( lExpressionAnalyzer.containsUserFunction ) {
			List<FunctionCall> fCallList = IRTools.getFunctionCalls(cExpr);
			List<ProcedureDeclarator> procDeclrList = AnalysisTools.getProcedureDeclarators(at);
			if( fCallList != null ) {
				for( FunctionCall tFCall : fCallList ) {
					if( !StandardLibrary.contains(tFCall) ) {
						Procedure tProc = tFCall.getProcedure();
						if( (tProc != null) && !kernelSet.contains(tFCall.getName()) ) {
							kernelSet.add((IDExpression)tFCall.getName());
							boolean skipThis = false;
							List<ASPENAnnotation> aspenAnnotList = 
								tProc.getAnnotations(ASPENAnnotation.class);
							if( aspenAnnotList != null ) {
								for( ASPENAnnotation tAnnot : aspenAnnotList ) {
									if( tAnnot.containsKey("ignore") || tAnnot.containsKey("execute") ) {
										skipThis = true;
										break;
									}
								}
							}
							if( !skipThis ) {
								Declaration procDecl = null;
								for( ProcedureDeclarator procDeclr : procDeclrList ) {
									if( procDeclr.getID().equals(tFCall.getName()) ) {
										procDecl = procDeclr.getDeclaration();
										break;
									}
								}
								if( procDecl != null ) {
									aspenAnnotList = 
										procDecl.getAnnotations(ASPENAnnotation.class);
									if( aspenAnnotList != null ) {
										for( ASPENAnnotation tAnnot : aspenAnnotList ) {
											if( tAnnot.containsKey("ignore") || tAnnot.containsKey("execute") ) {
												skipThis = true;
												break;
											}
										}
									}
								}
							}
							if( !skipThis ) {
								ASPENConfiguration.accessedFunctions.add(tProc.getSymbolName());
								CompoundStatement ttBody = (CompoundStatement)tProc.getBody();
								List<Declaration> paramList = tProc.getParameters();
								if( paramList != null ) {
									int list_size = paramList.size();
									if( list_size == 1 ) {
										Object obj = paramList.get(0);
										String paramS = obj.toString();
										// Remove any leading or trailing whitespace.
										paramS = paramS.trim();
										if( paramS.equals(Specifier.VOID.toString()) ) {
											list_size = 0;
										}
									}
									if( list_size > 0 ) {
										for( Declaration pDecl : paramList ) {
											if( pDecl instanceof VariableDeclaration ) {
												handleVariableDeclaration((VariableDeclaration)pDecl, tProc);
											}
										}
									}
								}
								analyzeASPENModel(ttBody, tProc);
							}
						}
					}
				}
			}
		} else if( at instanceof ExpressionStatement ) {
			if( inASPENModelRegion ) {
				controlAnnot.put("execute", "_clause");
			}
		}
		if( inASPENModelRegion ) {
			if( lExpressionAnalyzer.getFlops() > 0 ) {
				ASPENAnnotation aspenAnnot = at.getAnnotation(ASPENAnnotation.class, "flops");
				String floatType = null;
				if( lExpressionAnalyzer.dataTypes != null ) {
					if( lExpressionAnalyzer.containsDoubles ) {
						floatType = "dp";
					} else if( lExpressionAnalyzer.containsFloats ) {
						floatType = "sp";
					}
				}
				if( aspenAnnot == null ) {
					aspenAnnot = controlAnnot;
					ASPENResource flops = new ASPENResource(
							new IntegerLiteral(lExpressionAnalyzer.getFlops()));
					if( floatType != null ) {
						ASPENTrait tTrait = new ASPENTrait(floatType);
						flops.addTrait(tTrait);
					}
					if( lExpressionAnalyzer.isSIMDizable ) {
						ASPENTrait tTrait = new ASPENTrait("simd");
						flops.addTrait(tTrait);
					}
					Set<ASPENResource> flopsSet = new HashSet<ASPENResource>();
					flopsSet.add(flops);
					aspenAnnot.put("flops", flopsSet);
				} else {
					Set<ASPENResource> flopsSet = aspenAnnot.get("flops");
					boolean foundSIMD = false;
					boolean foundFlops = false;
					for( ASPENResource flops : flopsSet ) {
						//[FIXME] Current implementation consider traits only partially (float type).
						if( !foundFlops ) {
							Set<ASPENTrait> tTraitSet = flops.getTraitsOfType(ASPENTrait.TraitType.FlopsTrait);
							if( floatType != null ) {
								for( ASPENTrait tTrait : tTraitSet ) {
									if( tTrait.getTrait().equals("simd")) {
										foundSIMD = true;
									}
									if( tTrait.getTrait().equals(floatType)) {
										foundFlops = true;
										break;
									}
								}
							} else {
								if( tTraitSet.isEmpty() ) {
									foundFlops = true;
								}
							}
							if( foundFlops ) {
								Expression value = flops.getValue();
								if( value instanceof IntegerLiteral ) {
									IntegerLiteral nValue = 
										new IntegerLiteral(((IntegerLiteral)value).getValue() + lExpressionAnalyzer.getFlops());
									nValue.swapWith(value);
								} else {
									Expression nValue = new BinaryExpression(value.clone(), BinaryOperator.ADD,
											new IntegerLiteral(lExpressionAnalyzer.getFlops()));
									nValue.swapWith(value);
								}
								if( lExpressionAnalyzer.isSIMDizable && !foundSIMD ) {
									ASPENTrait tTrait = new ASPENTrait("simd");
									flops.addTrait(tTrait);
								}
								break;
							}
						}
					}
				}
			} 
			if( !lExpressionAnalyzer.LOADS.isEmpty() ) {
				Map<Symbol, Map<Expression, Expression>> symStatusMap = lExpressionAnalyzer.LOADS;
				Set<Symbol> symSet = symStatusMap.keySet();
				Set<Symbol> removeSyms = new HashSet<Symbol>();
				for( Symbol tSym : symSet ) {
					osymList.clear();
					Symbol tGSym = orgDataSymMap.get(tSym);
					if( tGSym == null ) {
						if( AnalysisTools.SymbolStatus.OrgSymbolFound(
								AnalysisTools.findOrgSymbol(tSym, at, true, null, osymList, gFuncCallList)) ) {
							tGSym = (Symbol)osymList.get(0);
						} else {
							tGSym = tSym;
						}
					}
					if( !orgDataSymMap.values().contains(tGSym) ) {
						removeSyms.add(tSym);
					}
				}
				for( Symbol tSym : removeSyms ) {
					if( !SymbolTools.isFormal(tSym) || 
							(!SymbolTools.isArray(tSym) && !SymbolTools.isPointer(tSym) && !(tSym instanceof NestedDeclarator)) ) {
						symStatusMap.remove(tSym);
					}
				}
				if( !symStatusMap.isEmpty() ) {
					Set<ASPENResource> loadsSet = aspenAnnotData1.loadsSet;
					Set<ASPENResource> aspenLoadsSet = null;
					ASPENAnnotation aspenAnnot = at.getAnnotation(ASPENAnnotation.class, "loads");
					if( aspenAnnot == null ) {
						aspenAnnot = controlAnnot;
						aspenLoadsSet = new HashSet<ASPENResource>();
						aspenAnnot.put("loads", aspenLoadsSet);
					} else {
						aspenLoadsSet =  aspenAnnot.get("loads");
					}
					for( Symbol lSym : symStatusMap.keySet() ) {
						boolean loadSymExists = false;
						for( ASPENResource lRSC : loadsSet ) {
							if((lRSC.getID() != null) && (lRSC.getID().toString().equals(lSym.getSymbolName())) ) {
								loadSymExists = true;
								break;
							}
						}
						if( !loadSymExists ) {
							List specs = lSym.getTypeSpecifiers();
							NameID nParamID = getBuiltInParamForTypes(specs, internalParamMap, mainTrUnt);
							Map<Expression, Expression> statusMap = symStatusMap.get(lSym);
							for(Expression tstatus : statusMap.keySet()) {
								Expression loadSize = new BinaryExpression(statusMap.get(tstatus).clone(), 
										BinaryOperator.MULTIPLY, nParamID.clone());
								if( tstatus instanceof StringLiteral ) {
									ASPENResource lRSC = null;
									if( removeSyms.contains(lSym) ) {
										lRSC = new ASPENResource(loadSize, null); 
									} else {
										lRSC = new ASPENResource(loadSize, null, "from", new Identifier(lSym)); 
									}
									//If duplicate ASPENResources exist, we have to manually multiply  them;
									//otherwise, they will be overwritten.
									while ( aspenLoadsSet.contains(lRSC) ) {
										aspenLoadsSet.remove(lRSC);
										loadSize = Symbolic.simplify(new BinaryExpression(new IntegerLiteral(2),
												BinaryOperator.MULTIPLY, loadSize.clone()));
										if( removeSyms.contains(lSym) ) {
											lRSC = new ASPENResource(loadSize, null); 
										} else {
											lRSC = new ASPENResource(loadSize, null, "from", new Identifier(lSym)); 
										}
									}
									aspenLoadsSet.add(lRSC);
								} else {
									List<ASPENTrait> traitList = new ArrayList<ASPENTrait>(1);
									List<Expression> argList = new ArrayList<Expression>(1);
									argList.add(tstatus.clone());
									traitList.add(new ASPENTrait("stride", argList));
									ASPENResource lRSC = null;
									if( removeSyms.contains(lSym) ) {
										lRSC = new ASPENResource(loadSize, traitList); 
									} else {
										lRSC = new ASPENResource(loadSize, traitList, "from", new Identifier(lSym)); 
									}
									//System.err.println("Current lRSC: " + lRSC);
									while ( aspenLoadsSet.contains(lRSC) ) {
										aspenLoadsSet.remove(lRSC);
										loadSize = Symbolic.simplify(new BinaryExpression(new IntegerLiteral(2),
												BinaryOperator.MULTIPLY, loadSize.clone()));
										traitList = new ArrayList<ASPENTrait>(1);
										argList = new ArrayList<Expression>(1);
										argList.add(tstatus.clone());
										traitList.add(new ASPENTrait("stride", argList));
										if( removeSyms.contains(lSym) ) {
											lRSC = new ASPENResource(loadSize, traitList); 
										} else {
											lRSC = new ASPENResource(loadSize, traitList, "from", new Identifier(lSym)); 
										}
									}
									//System.err.println("lRSC to be added: " + lRSC);
									aspenLoadsSet.add(lRSC);
								}
							}
						}
					}
				}
			}
		}
		if( !lExpressionAnalyzer.STORES.isEmpty() ) {
			Map<Symbol, Map<Expression, Expression>> symStatusMap = lExpressionAnalyzer.STORES;
			Set<Symbol> symSet = symStatusMap.keySet();
			if( inASPENModelRegion && (lExpressionAnalyzer.allocatedElems == null) ) {
				Set<Symbol> removeSyms = new HashSet<Symbol>();
				for( Symbol tSym : symSet ) {
					osymList.clear();
					Symbol tGSym = orgDataSymMap.get(tSym);
					if( tGSym == null ) {
						if( AnalysisTools.SymbolStatus.OrgSymbolFound(
								AnalysisTools.findOrgSymbol(tSym, at, true, null, osymList, gFuncCallList)) ) {
							tGSym = (Symbol)osymList.get(0);
						} else {
							tGSym = tSym;
						}
					}
					if( !orgDataSymMap.values().contains(tGSym) ) {
						removeSyms.add(tSym);
					}
				}
				for( Symbol tSym : removeSyms ) {
					if( !SymbolTools.isFormal(tSym) || 
							(!SymbolTools.isArray(tSym) && !SymbolTools.isPointer(tSym) && !(tSym instanceof NestedDeclarator)) ) {
						symStatusMap.remove(tSym);
					}
				}
				if( !symStatusMap.isEmpty() ) {
					Set<ASPENResource> storesSet = aspenAnnotData1.storesSet;
					Set<ASPENResource> aspenStoresSet = null;
					ASPENAnnotation aspenAnnot = at.getAnnotation(ASPENAnnotation.class, "stores");
					if( aspenAnnot == null ) {
						aspenAnnot = controlAnnot;
						aspenStoresSet = new HashSet<ASPENResource>();
						aspenAnnot.put("stores", aspenStoresSet);
					} else {
						aspenStoresSet =  aspenAnnot.get("stores");
					}
					for( Symbol lSym : symStatusMap.keySet() ) {
						boolean storeSymExists = false;
						for( ASPENResource lRSC : storesSet ) {
							if((lRSC.getID() != null) && (lRSC.getID().toString().equals(lSym.getSymbolName())) ) {
								storeSymExists = true;
								break;
							}
						}
						if( !storeSymExists ) {
							List specs = lSym.getTypeSpecifiers();
							NameID nParamID = getBuiltInParamForTypes(specs, internalParamMap, mainTrUnt);
							Map<Expression, Expression> statusMap = symStatusMap.get(lSym);
							for(Expression tstatus : statusMap.keySet()) {
								Expression storeSize = new BinaryExpression(statusMap.get(tstatus).clone(), 
										BinaryOperator.MULTIPLY, nParamID.clone());
								if( tstatus instanceof StringLiteral ) {
									if( !((StringLiteral)tstatus).getValue().equals("PointerAssignment") ) {
										//Simple pointer assignment is excluded.
										ASPENResource lRSC = null;
										if( removeSyms.contains(lSym) ) {
											lRSC = new ASPENResource(storeSize, null); 
										} else {
											lRSC = new ASPENResource(storeSize, null, "to", new Identifier(lSym)); 
										}
										while ( aspenStoresSet.contains(lRSC) ) {
											aspenStoresSet.remove(lRSC);
											storeSize = Symbolic.simplify(new BinaryExpression(new IntegerLiteral(2),
													BinaryOperator.MULTIPLY, storeSize.clone()));
											if( removeSyms.contains(lSym) ) {
												lRSC = new ASPENResource(storeSize, null); 
											} else {
												lRSC = new ASPENResource(storeSize, null, "to", new Identifier(lSym)); 
											}
										}
										aspenStoresSet.add(lRSC);
									}
								} else {
									List<ASPENTrait> traitList = new ArrayList<ASPENTrait>(1);
									List<Expression> argList = new ArrayList<Expression>(1);
									argList.add(tstatus.clone());
									traitList.add(new ASPENTrait("stride", argList));
									ASPENResource lRSC = null;
									if( removeSyms.contains(lSym) ) {
										lRSC = new ASPENResource(storeSize, traitList); 
									} else {
										lRSC = new ASPENResource(storeSize, traitList, "to", new Identifier(lSym)); 
									}
									while ( aspenStoresSet.contains(lRSC) ) {
										aspenStoresSet.remove(lRSC);
										storeSize = Symbolic.simplify(new BinaryExpression(new IntegerLiteral(2),
												BinaryOperator.MULTIPLY, storeSize.clone()));
										traitList = new ArrayList<ASPENTrait>(1);
										argList = new ArrayList<Expression>(1);
										argList.add(tstatus.clone());
										traitList.add(new ASPENTrait("stride", argList));
										if( removeSyms.contains(lSym) ) {
											lRSC = new ASPENResource(storeSize, traitList); 
										} else {
											lRSC = new ASPENResource(storeSize, traitList, "to", new Identifier(lSym)); 
										}
									}
									aspenStoresSet.add(lRSC);
								}
							}
						}
					}
					if( aspenStoresSet.isEmpty() ) {
						aspenAnnot = at.getAnnotation(ASPENAnnotation.class, "stores");
						if( aspenAnnot != null ) {
							aspenStoresSet = aspenAnnot.get("stores");
							if( (aspenStoresSet == null) || (aspenStoresSet.isEmpty()) ) {
								aspenAnnot.remove("stores");
							}
						}

					}
				}
			} else if( lExpressionAnalyzer.allocatedElems != null ) {
				//ASPEN Data size information is updated even if not in the model region.
				Symbol allocSym = null;
				for( Symbol tSym : symSet ) {
					osymList.clear();
					Symbol tGSym = orgDataSymMap.get(tSym);
					if( tGSym == null ) {
						if( AnalysisTools.SymbolStatus.OrgSymbolFound(
								AnalysisTools.findOrgSymbol(tSym, at, true, null, osymList, gFuncCallList)) ) {
							tGSym = (Symbol)osymList.get(0);
						} else {
							tGSym = tSym;
						}
					}
					allocSym = tSym;
				}
				Annotatable tat = dataMap.get(allocSym);
				if( tat != null ) {
					ASPENAnnotation tannot = tat.getAnnotation(ASPENAnnotation.class, "data");
					Set<ASPENData> dataSet = tannot.get("data");
					ASPENTrait nTrait = null;
					List<Expression> lengthList = new ArrayList<Expression>(1);
					lengthList.add(lExpressionAnalyzer.allocatedElems.clone());
					if( lExpressionAnalyzer.allocatedUnit != null ) {
						lengthList.add(lExpressionAnalyzer.allocatedUnit.clone());
						nTrait = new ASPENTrait("Array", lengthList);
					}
					if( dataSet != null ) {
						for( ASPENData tData : dataSet ) {
							IDExpression ID = tData.getID();
							if( ID instanceof Identifier ) {
								if( ((Identifier)ID).getSymbol().equals(allocSym) ) {
									if( (tData.getCapacity() == null) && (tData.getTraitSize() == 0) ) {
										//We update data size info only if it does not exist.
										if( nTrait == null ) {
											tData.setCapacity(lExpressionAnalyzer.allocatedElems.clone());
										} else {
											tData.addTrait(nTrait);
										}
									}
									break;
								}
							}
						}
					}
				}
			}
		}

	}
	
	/**
	 * Update IDExressions in ASPENAnnotation clauses with Identifiers that have links to 
	 * corresponding Symbols and update old symbols in a set with new ones.
	 * 
	 */
	public static void updateSymbolsInASPENAnnotation(Traversable inputIR, Map<String, String> nameChangeMap)
	{
		DFIterator<Annotatable> iter = new DFIterator<Annotatable>(inputIR, Annotatable.class);
		while(iter.hasNext())
		{
			Annotatable at = iter.next();
			List<ASPENAnnotation> ASPENAnnotList = at.getAnnotations(ASPENAnnotation.class);
			if ( (ASPENAnnotList != null) && (ASPENAnnotList.size() > 0) ) 
			{
				Traversable inputTR = at;
				if( at instanceof Procedure ) {
					inputTR = ((Procedure)at).getBody();
				}
				for( ASPENAnnotation annot : ASPENAnnotList ) {
					for( String key: annot.keySet() ) {
						Object value = annot.get(key);
						if( value instanceof Expression ) {
							Expression newExp = updateSymbolsInExpression((Expression)value, inputTR, nameChangeMap, annot);
							if( newExp != null ) {
								annot.put(key, newExp);
							}
						} else if( value instanceof ASPENExpression ) {
							updateSymbolsInASPENExpression((ASPENExpression)value, inputTR, nameChangeMap, annot);
						} else if( value instanceof List ) {
							List vSet = (List)value;
							String elmType = "null";
							for( Object elm : vSet ) {
								if( elm instanceof Expression ) {
									elmType = "exprssion";
									break;
								} else {
									Tools.exit("[ERROR in ASPENModelAnalysis.updateSymbolsInASPENAnnotation()]: List<Expression> is expected " +
											"for the value of key, " + key + ", in ASPENAnnotation, " + annot + AnalysisTools.getEnclosingAnnotationContext(annot));
								}
							}
							if( elmType.equals("expression") ) {
								List newList = new ArrayList<Expression>(vSet.size());
								for( Object elm : vSet ) {
									Expression newExp = updateSymbolsInExpression((Expression)elm, inputTR, nameChangeMap, annot);
									if( newExp == null ) {
										newList.add(elm);
									} else {
										newList.add(newExp);
									}
								}
								annot.put(key, newList);
							}
						} else if( value instanceof Set ) {
							Set vSet = (Set)value;
							String elmType = "null";
							for( Object elm : vSet ) {
								if( elm instanceof Expression ) {
									elmType = "exprssion";
									break;
								} else if( elm instanceof ASPENExpression ) {
									elmType = "aspenexp";
									break;
								} else if( elm instanceof Symbol ) {
									elmType = "symbol";
									break;
								} else if( elm instanceof String ) {
									elmType = "string";
									break;
								} else {
									Tools.exit("[ERROR in ASPENModelAnalysis.updateSymbolsInASPENAnnotation()]: Set<ASPENExpression>, Set<Expression>, " +
											"Set<String>, or Set<Symbol> type is expected " +
											"for the value of key, " + key + ", in ASPENAnnotation, " + annot + AnalysisTools.getEnclosingAnnotationContext(annot));
								}
							}
							if( elmType.equals("symbol") ) {
								Set<Symbol> new_set = new HashSet<Symbol>();
								ACCAnalysis.updateSymbols(inputTR, (Set<Symbol>)vSet, new_set, nameChangeMap);
								annot.put(key, new_set); //update symbol set in the annotation
							} else if( elmType.equals("string") ) {
								Set<String> new_set = new HashSet<String>();
								new_set.addAll(vSet);
								annot.put(key, new_set); //Change set type to HashSet.
							} else if( elmType.equals("aspenexp") ) {
								Set<ASPENExpression> newSet = new HashSet<ASPENExpression>();
								for( ASPENExpression elm : (Set<ASPENExpression>)vSet ) {
									updateSymbolsInASPENExpression(elm, inputTR, nameChangeMap, annot);
									newSet.add(elm);
								}
								annot.put(key, newSet);
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
						} else if( !(value instanceof String) ) {
							Tools.exit("[ERROR in ASPENModelAnalysis.updateSymbolsInASPENAnnotation()] Unexpected value type for a key, "+ key + 
									" in  ASPENAnnotation, " + annot + AnalysisTools.getEnclosingAnnotationContext(annot));
						}
					}
				}
			}
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
			ASPENAnnotation annot)
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
				if( nid.getParent() instanceof FunctionCall ) {
					//nid is a function name, and its procedure declaration is not found, but we can 
					//ignore this for ASPEN model generation.
					Procedure cProc = IRTools.getParentProcedure(at);
					PrintTools.println("[WARNING in ASPENModelAnalysis.updateSymbolsInExpression()] a function, " +
							newName + ", in the following ASPEN annotation is not visible in the current scope.\n" +
							"ASPEN annotation: " + annot + "\n" +
							"Enclosing procedure: " + cProc.getSymbolName() + "\n", 0);
				} else if( !isInternalParam(newName) ) {
					//OpenARC internal variables (HI_* or aspen_*) don't need to be updated.
					Procedure cProc = IRTools.getParentProcedure(at);
					if( cProc != null ) {
						Tools.exit("[ERROR in ASPENModelAnalysis.updateSymbolsInExpression()] a variable, " + newName + 
								", in the following ASPEN annotatin is not visible in the current scope; " +
								"please check whether the declaration of the variable is visible.\n" +
								"ASPEN annotation: " + annot + "\n" +
								"Enclosing Procedure: " + cProc.getSymbolName() +"\nEnclosing Translation Unit: " + 
								((TranslationUnit)cProc.getParent()).getOutputFilename() + "\n");
					} else {
						Tools.exit("[ERROR in ASPENModelAnalysis.updateSymbolsInExpression()] a variable, " + newName + 
								", in the following ASPEN annotatin is not visible in the current scope; " +
								"please check whether the declaration of the variable is visible.\n" +
								"ASPEN annotation: " + annot + "\n");
					}
				}
			} else {
/*				List symbolInfo = new ArrayList(2);
				if( AnalysisTools.SymbolStatus.OrgSymbolFound(
						AnalysisTools.findOrgSymbol(sym, at, true, null, symbolInfo, null)) ) {
					sym = (Symbol)symbolInfo.get(0);
				}*/
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
	 * Return the new expression if input expression {@code exp} is IDExpression; otherwise, return
	 * null
	 * 
	 * @param exp
	 * @param at
	 * @param nameChangeMap
	 * @param annot
	 * @return
	 */
	protected static void updateSymbolsInASPENExpression(ASPENExpression aExp, Traversable at, Map<String, String> nameChangeMap,
			ASPENAnnotation annot)
	{
		if( (aExp == null) || (at == null) ) return;
		for( Traversable tt : aExp.getChildren() ) {
			if( tt instanceof Expression ) {
				updateSymbolsInExpression((Expression)tt, at, nameChangeMap, annot);
			} else if( tt instanceof ASPENExpression ) {
				updateSymbolsInASPENExpression((ASPENExpression)tt, at, nameChangeMap, annot);
			}
		}
		if( (aExp instanceof ASPENData) && ((ASPENData)aExp).getCapacity() != null ) {
			((ASPENData)aExp).updateCapacity();
		}
		if( (aExp instanceof ASPENResource) && ((ASPENResource)aExp).getID() != null ) {
			((ASPENResource)aExp).updateID();
		}
	}
	
	protected boolean containsParamSymbolsOnly(Expression exp) {
		boolean ret = true;
		if( exp != null ) {
			DFIterator<Identifier> eiter = new DFIterator<Identifier>(exp, Identifier.class);
			while(eiter.hasNext())
			{
				Identifier nid = eiter.next();
				if( !paramMap.keySet().contains(nid.getSymbol()) ) {
					ret = false;
					break;
				}
			}
		}
		return ret;
	}
	
	protected static NameID getBuiltInParamForTypes(List tspecs, Map<IDExpression, Annotatable> internalParamMap,
			TranslationUnit mainTrUnt) {
		List specs = new ArrayList(tspecs.size());
		//[FIXME] below use the string comparison because the pointer in a sizeof(float *) expression 
		//is not recognized as PointerSpecifier (recognized as Declarator).
		//====> Fixed.
		for( Object tObj : tspecs ) {
			//[DEBUG] below is for debugging purpose only.
/*			if( tObj.toString().contains("*") && !(tObj instanceof PointerSpecifier) ) {
				System.err.println("* is found in the typelist");
				if( tObj instanceof Declarator ) {
					System.err.println("Type of this object: Declarator");
				} else if( tObj instanceof Expression ) {
					System.err.println("Type of this object: Expression");
				} else if( tObj instanceof Symbol ) {
					System.err.println("Type of this object: Symbol");
				} else if( tObj instanceof Specifier ) {
					System.err.println("Type of this object: Specifier");
				}
				System.err.println("The whole typelist: " + tspecs);
			}*/
			if( !(tObj instanceof PointerSpecifier) && !tObj.equals(Specifier.STATIC) 
					&& !tObj.equals(Specifier.EXTERN) && !tObj.equals(Specifier.CONST) 
					&& !(tObj instanceof Declarator)
					&& !tObj.equals(Specifier.RESTRICT) ) {
				specs.add(tObj);
			}
		}
		NameID nParamID = null;
		if( !specs.isEmpty() ) {
			StringBuilder str = new StringBuilder(32);
			str.append("aspen_param_");
			for( Object obj : specs ) {
				str.append(obj.toString());
			}
			nParamID = new NameID(str.toString());
			int bitLength = AnalysisTools.getBitLengthOfType(specs);
			int byteLength = bitLength/8;
			Expression byteExp = null;
			if( byteLength > 0 ) {
				byteExp = new IntegerLiteral(byteLength);
			}
			if( !internalParamMap.containsKey(nParamID) ) {
				Set<ASPENParam> paramSet = new HashSet<ASPENParam>();
				ASPENParam aParam = new ASPENParam(nParamID, byteExp);
				paramSet.add(aParam);
				ASPENAnnotation nParamAnnot = new ASPENAnnotation("declare", "_directive");
				nParamAnnot.put("param", paramSet);
				AnnotationDeclaration nParamDecl = new AnnotationDeclaration(nParamAnnot);
				if( internalParamMap.isEmpty() ) {
					mainTrUnt.addDeclarationBefore(mainTrUnt.getFirstDeclaration(), nParamDecl);
				} else {
					mainTrUnt.addDeclarationAfter(mainTrUnt.getFirstDeclaration(), nParamDecl);
				}
				internalParamMap.put(nParamID, nParamDecl);
			}
		}
		return nParamID;
	}
	
	protected void handleVariableDeclaration(VariableDeclaration vDecl, SymbolTable symTable) {
		TranslationUnit trUnt = null;
		CompoundStatement cStmt = null;
		Statement declStmt = null;
		boolean isFormalParameter = false;
		if( vDecl.getParent() instanceof Statement ) {
			declStmt = (Statement)vDecl.getParent();
		}
		if( symTable instanceof TranslationUnit ) {
			trUnt = (TranslationUnit)symTable;
		} else if( symTable instanceof CompoundStatement ) {
			cStmt = (CompoundStatement)symTable;
		} else if( symTable instanceof Procedure ) {
			isFormalParameter = true;
			cStmt = ((Procedure)symTable).getBody();
		} else {
			return;
		}
		List specs = vDecl.getSpecifiers();
		if( specs.contains(Specifier.TYPEDEF) ) {
			//Typedef declaration is skipped.
			return;
		}
		ASPENAnnotation aspenAnnot = null;
		DFIterator<Traversable> symbol_iter =
				new DFIterator<Traversable>(vDecl);
		symbol_iter.pruneOn(NestedDeclarator.class);
		while (symbol_iter.hasNext()) {
			Traversable t = symbol_iter.next();
			if (!(t instanceof Symbol)) {
				continue;
			}
			Symbol symbol = (Symbol)t;
			boolean isArray = SymbolTools.isArray(symbol);
			boolean isPointer = SymbolTools.isPointer(symbol);
			boolean isStruct = SymbolTools.isStruct(symbol, vDecl);
			if( symbol instanceof NestedDeclarator ) {
				isPointer = true;
			}
			if( !isArray && !isPointer && !isStruct ) {
				//Primitive scalar variables are translated to ASPEN param variables.
				//(Formal parameters are also translated to ASPEN param variables.)
				if( !paramMap.containsKey(symbol) ) {
					//Add a new ASPEN parameter
					Set<ASPENParam> paramSet = new HashSet<ASPENParam>();
					aspenAnnot = new ASPENAnnotation("declare", "_directive");
					aspenAnnot.put("param", paramSet);
					if( trUnt != null ) {
						AnnotationDeclaration nDecl = new AnnotationDeclaration(aspenAnnot);
						trUnt.addDeclarationBefore(vDecl, nDecl);
						paramMap.put(symbol, nDecl);
					} else if( cStmt != null ) {
						AnnotationStatement anStmt = new AnnotationStatement(aspenAnnot);
						if( !isFormalParameter && (declStmt != null) ) {
							cStmt.addStatementBefore(declStmt, anStmt);
						} else {
							Statement nonDeclStmt = IRTools.getFirstNonDeclarationStatement(cStmt);
							if( nonDeclStmt != null ) {
								cStmt.addStatementBefore(nonDeclStmt, anStmt);
							} else {
								cStmt.addStatement(anStmt);
							}
						}
						paramMap.put(symbol, anStmt);
					}
					ASPENParam aParam = new ASPENParam(new Identifier(symbol));
					if( symbol instanceof VariableDeclarator ) {
						Initializer init = ((VariableDeclarator)symbol).getInitializer();
						if( init != null ) {
							//[CAUTION] Does this have only one child?
							//==> Generally not true, but it's OK for scalar variable initialization.
							Expression initExp = (Expression)init.getChildren().get(0);
							while ( initExp instanceof AssignmentExpression ) {
								initExp = ((AssignmentExpression)initExp).getRHS();
							}
							aParam.setInitVal(Symbolic.simplify(initExp.clone()));
						}
					}
					paramSet.add(aParam);
				}
			} else if( !isFormalParameter && ((trUnt != null) || (isPointer)) ) {
				//Pointer variables and global array variables are translated to ASPEN data variables
				//if they are not formal parameters.
				if( !dataMap.containsKey(symbol) ) {
					//Add a new ASPEN data
					Set<ASPENData> dataSet = new HashSet<ASPENData>();
					aspenAnnot = new ASPENAnnotation("declare", "_directive");
					aspenAnnot.put("data", dataSet);
					if( trUnt != null ) {
						AnnotationDeclaration nDecl = new AnnotationDeclaration(aspenAnnot);
						((TranslationUnit)trUnt).addDeclarationBefore(vDecl, nDecl);
						dataMap.put(symbol, nDecl);
					} else if( cStmt != null ) {
						AnnotationStatement anStmt = new AnnotationStatement(aspenAnnot);
						cStmt.addStatementBefore(declStmt, anStmt);
						dataMap.put(symbol, anStmt);
					}
					osymList.clear();
					if( AnalysisTools.SymbolStatus.OrgSymbolFound(
							AnalysisTools.findOrgSymbol(symbol, vDecl, true, null, osymList, gFuncCallList)) ) {
						//If the data variable is aliased, keep the mapping of the symbol and the original symbol.
						orgDataSymMap.put(symbol, (Symbol)osymList.get(0));
					} else {
						orgDataSymMap.put(symbol, symbol);
					}
					ASPENData aData = new ASPENData(new Identifier(symbol));
					List<Expression> lengthList = new LinkedList<Expression>();
					if( AnalysisTools.extractDimensionInfo(symbol, lengthList, IRSymbolOnly)) {
						NameID nParamID = getBuiltInParamForTypes(specs, internalParamMap, mainTrUnt);
						int dimSize = lengthList.size();
						if( nParamID != null ) {
							lengthList.add(nParamID.clone());
						} else {
							Tools.exit("[ERROR in ASPENModelAnalysis.handleVariableDeclaration()] unexpected declaration: " + vDecl +
									AnalysisTools.getEnclosingContext(vDecl));
						}
						ASPENTrait nTrait = null;
						if( dimSize == 1 ) {
							nTrait = new ASPENTrait("Array", lengthList);
						} else if( dimSize == 2 ) {
							nTrait = new ASPENTrait("Matrix", lengthList);
						} else if( dimSize == 3 ) {
							nTrait = new ASPENTrait("3DVolume", lengthList);
						}
						if( nTrait != null ) {
							aData.addTrait(nTrait);
						}
					} else if( symbol instanceof VariableDeclarator ) {
						Initializer init = ((VariableDeclarator)symbol).getInitializer();
						if( init != null ) {
							//[CAUTION] Does this have only one child?
							//No, but we are interested in the case where only one child of Expression type 
							//exists, which is true if the initialization has malloc fuction.
							Object initObj = init.getChildren().get(0);
							Expression cExpr = null;
							if( initObj instanceof Expression ) {
								cExpr = ((Expression)initObj).clone();
								ExpressionAnalyzer lExpressionAnalyzer = new ExpressionAnalyzer();
								lExpressionAnalyzer.analyzeExpression(cExpr, internalParamMap, mainTrUnt);
								if( lExpressionAnalyzer.allocatedElems != null ) {
									if( lExpressionAnalyzer.allocatedUnit != null ) {
										List<Expression> tlengthList = new ArrayList<Expression>(1);
										tlengthList.add(lExpressionAnalyzer.allocatedElems.clone());
										tlengthList.add(lExpressionAnalyzer.allocatedUnit.clone());
										ASPENTrait nTrait = null;
										nTrait = new ASPENTrait("Array", tlengthList);
										aData.addTrait(nTrait);
									} else {
										aData.setCapacity(lExpressionAnalyzer.allocatedElems.clone());
									}
								}
							} else {
								//Nested Initializer case: the current ASPEN analysis pass
								//can not handle this case; ignore it.
							}
						}
					}
					dataSet.add(aData);
				}
			}
		}

	}
	
	protected  class ExpressionAnalyzer {
		protected int FADDS = 0;
		protected int FSUBS = 0;
		protected int FMULS = 0;
		protected int FDIVS = 0;
		protected int FOTHERS = 0;
		protected int INTADDS = 0;
		protected int INTSUBS = 0;
		protected int INTMULS = 0;
		protected int INTDIVS = 0;
		protected int INTOTHERS = 0;
		protected Map<Symbol, Map<Expression, Expression>> LOADS = new HashMap<Symbol, Map<Expression, Expression>>();
		protected Map<Symbol, Map<Expression, Expression>> STORES = new HashMap<Symbol, Map<Expression, Expression>>();
		protected Expression allocatedElems = null;
		protected Expression allocatedUnit = null;
		boolean containsUserFunction = false;
		Specifier dataTypes = null;
		boolean containsFloats = false;
		boolean containsDoubles = false;
		boolean isExpressionRoot = true;
		boolean isSIMDizable = false; //Set to true for scalar variable or array variables with affine expressions 
									//with respect to the index variable of the innermost enclosing loop.
		
		protected ExpressionAnalyzer() {
			//reset();
		}
		
		protected int getFlops() {
			return FADDS + FSUBS + FMULS + FDIVS + FOTHERS;
		}
		
		protected int getIntops() {
			return INTADDS + INTSUBS + INTMULS + INTDIVS + INTOTHERS;
		}
		
		void reset() {
			FADDS = 0;
			FSUBS = 0;
			FMULS = 0;
			FDIVS = 0;
			FOTHERS = 0;
			INTADDS = 0;
			INTSUBS = 0;
			INTMULS = 0;
			INTDIVS = 0;
			INTOTHERS = 0;
			for( Symbol tSym: LOADS.keySet() ) {
				Map tMap = LOADS.get(tSym);
				tMap.clear();
			}
			LOADS.clear();
			for( Symbol tSym: STORES.keySet() ) {
				Map tMap = STORES.get(tSym);
				tMap.clear();
			}
			STORES.clear();
			allocatedElems = null;
			allocatedUnit = null;
			containsUserFunction = false;
			dataTypes = null;
			containsFloats = false;
			containsDoubles = false;
			isExpressionRoot = true;
			isSIMDizable = false;
		}
		
		Set<Symbol> getFloatVariables(Expression tExp) {
			Set<Symbol> fSymSet = new HashSet<Symbol>();
			Set<Symbol> accessedSymbols = SymbolTools.getAccessedSymbols(tExp);
			if( accessedSymbols != null ) {
				for(Symbol aSym : accessedSymbols) {
					List specs = aSym.getTypeSpecifiers();
					if( specs != null ) {
						if( specs.contains(Specifier.DOUBLE) ) {
							fSymSet.add(aSym);
							dataTypes = Specifier.DOUBLE;
						} else if( specs.contains(Specifier.FLOAT) ) {
							fSymSet.add(aSym);
							dataTypes = Specifier.FLOAT;
						}
					}
				}
			}
			return fSymSet;
		}
		
		Map<Symbol, Expression> analyzeFloatVariableAccesses(Expression tExp, boolean isLValue) {
			Map<Symbol, Expression> retMap = new HashMap<Symbol, Expression>();
			if( tExp != null ) {
				Traversable t = tExp.getParent();
				ForLoop pLoop = null;
				while( (t != null) && !(t instanceof ForLoop) ) {
					t = t.getParent();
				}
				Expression indexVar = null;
				if( t instanceof ForLoop ) {
					pLoop = (ForLoop)t;
					indexVar = LoopTools.getIndexVariable(pLoop);
				}
				Identifier tID = null;
				DFIterator<Identifier> iter =
						new DFIterator<Identifier>(tExp, Identifier.class);
				while (iter.hasNext()) {
					tID = iter.next();
					Symbol symbol = tID.getSymbol();
					if (symbol != null) {
						List specs = symbol.getTypeSpecifiers();
						if( specs != null ) {
							boolean isFloat = false;
							if( specs.contains(Specifier.DOUBLE) ) {
								isFloat = true;
								dataTypes = Specifier.DOUBLE;
								containsDoubles = true;
							} else if( specs.contains(Specifier.FLOAT) ) {
								isFloat = true;
								dataTypes = Specifier.FLOAT;
								containsFloats = true;
							}
							if( isFloat ) {
								boolean isArray = SymbolTools.isArray(symbol);
								boolean isPointer = SymbolTools.isPointer(symbol);
								if( symbol instanceof NestedDeclarator ) {
									isPointer = true;
								}
								if( !isArray && !isPointer ) {
									//Scalar variable.
									retMap.put(symbol, new IntegerLiteral(0));
									if( inComputeRegion ) {
										isSIMDizable = true;
									}
								} else {
									if( tID.getParent() instanceof ArrayAccess ) {
										if( indexVar != null ) {
											ArrayAccess tAccess = (ArrayAccess)tID.getParent();
											Expression tIndex = tAccess.getIndex(tAccess.getNumIndices()-1);
											if( tIndex instanceof IntegerLiteral ) {
												List<Expression> indices = tAccess.getIndices();
												int i=0;
												boolean foundIndexDim = false;
												for( i=0; i<indices.size()-1; i++ ) {
													tIndex = indices.get(i);
													if( IRTools.containsExpression(tIndex, indexVar) ) {
														foundIndexDim = true;
													}
												}
												if( !foundIndexDim ) {
													//A[1];
													retMap.put(symbol, new IntegerLiteral(0));
													isSIMDizable = true;
												}
											} else if( tIndex.equals(indexVar) ) {
												//A[i];
												retMap.put(symbol, new IntegerLiteral(1));
													isSIMDizable = true;
											} else if( tIndex instanceof BinaryExpression ) {
												BinaryExpression bExp = (BinaryExpression)tIndex;
												if( bExp.getLHS().equals(indexVar) || bExp.getRHS().equals(indexVar) ) {
													BinaryOperator bOp = bExp.getOperator();
													if( bOp.equals(BinaryOperator.ADD) || bOp.equals(BinaryOperator.SUBTRACT) ) {
														retMap.put(symbol, new IntegerLiteral(1));
														isSIMDizable = true;
													}
												}
											}
										} else {
											ArrayAccess tAccess = (ArrayAccess)tID.getParent();
											List<Expression> indices = tAccess.getIndices();
											Expression tIndex = null;
											int i=0;
											boolean foundNonConstantIndex = false;
											for( i=0; i<indices.size()-1; i++ ) {
												tIndex = indices.get(i);
												List<IDExpression> tExpList = IRTools.getExpressionsOfType(tIndex, IDExpression.class);
												if( (tExpList != null) && !tExpList.isEmpty() ) {
													foundNonConstantIndex = true;
													break;
												}
											}
											if( !foundNonConstantIndex ) {
												//A[1];
												retMap.put(symbol, new IntegerLiteral(0));
												isSIMDizable = true;
											}
											if( !retMap.containsKey(symbol) ) {
												retMap.put(symbol, new StringLiteral("Unknown"));
											}
										}
									} else {
										boolean dereferenced = false;
										Traversable tt = tID;
										UnaryExpression uExp = null;
										while( tt != null ) {
											tt = tt.getParent();
											if( tt instanceof UnaryExpression ) {
												uExp = (UnaryExpression)tt;
												if( uExp.getOperator().equals(UnaryOperator.DEREFERENCE) ) {
													dereferenced = true;
													break;
												}
											}
										}
										if( dereferenced ) {
											retMap.put(symbol, new StringLiteral("PointerDereference"));
										}
									}
									if( isLValue ) {
										if( tExp instanceof Identifier ) {
											retMap.put(symbol, new StringLiteral("PointerAssignment"));
										}
										if( !retMap.containsKey(symbol) ) {
											retMap.put(symbol, new StringLiteral("Unknown"));
										}
									}
								}
							}
						}
					}
				}
			}
			return retMap;
		}
		
		Map<Symbol, Expression> analyzeVariableAccesses(Expression tExp, boolean isLValue) {
			Map<Symbol, Expression> retMap = new HashMap<Symbol, Expression>();
			if( tExp != null ) {
				Traversable t = tExp.getParent();
				ForLoop pLoop = null;
				while( (t != null) && !(t instanceof ForLoop) ) {
					t = t.getParent();
				}
				Expression indexVar = null;
				if( t instanceof ForLoop ) {
					pLoop = (ForLoop)t;
					indexVar = LoopTools.getIndexVariable(pLoop);
				}
				Identifier tID = null;
				DFIterator<Identifier> iter =
						new DFIterator<Identifier>(tExp, Identifier.class);
				while (iter.hasNext()) {
					tID = iter.next();
					Symbol symbol = tID.getSymbol();
					if (symbol != null) {
						List specs = symbol.getTypeSpecifiers();
						if( specs != null ) {
							if( specs.contains(Specifier.DOUBLE) ) {
								dataTypes = Specifier.DOUBLE;
								containsDoubles = true;
							} else if( specs.contains(Specifier.FLOAT) ) {
								dataTypes = Specifier.FLOAT;
								containsFloats = true;
							} else if( specs.contains(Specifier.CHAR) ) {
								dataTypes = Specifier.CHAR;
							} else if( specs.contains(Specifier.SHORT) ) {
								dataTypes = Specifier.SHORT;
							} else if( specs.contains(Specifier.LONG) ) {
								dataTypes = Specifier.LONG;
							} else if( specs.contains(Specifier.INT) ) {
								dataTypes = Specifier.INT;
							} else if( specs.contains(Specifier.UNSIGNED) ) {
								dataTypes = Specifier.INT;
							}
							boolean isArray = SymbolTools.isArray(symbol);
							boolean isPointer = SymbolTools.isPointer(symbol);
							if( symbol instanceof NestedDeclarator ) {
								isPointer = true;
							}
							if( !isArray && !isPointer ) {
								//Scalar variable.
								retMap.put(symbol, new IntegerLiteral(0));
								if( inComputeRegion ) {
									isSIMDizable = true;
								}
							} else {
								if( tID.getParent() instanceof ArrayAccess ) {
									if( indexVar != null ) {
										ArrayAccess tAccess = (ArrayAccess)tID.getParent();
										Expression tIndex = tAccess.getIndex(tAccess.getNumIndices()-1);
										if( tIndex instanceof IntegerLiteral ) {
											List<Expression> indices = tAccess.getIndices();
											int i=0;
											boolean foundIndexDim = false;
											for( i=0; i<indices.size()-1; i++ ) {
												tIndex = indices.get(i);
												if( IRTools.containsExpression(tIndex, indexVar) ) {
													foundIndexDim = true;
												}
											}
											if( !foundIndexDim ) {
												//A[1];
												retMap.put(symbol, new IntegerLiteral(0));
												isSIMDizable = true;
											}
										} else if( tIndex.equals(indexVar) ) {
											//A[i];
											retMap.put(symbol, new IntegerLiteral(1));
											isSIMDizable = true;
										} else if( tIndex instanceof BinaryExpression ) {
											BinaryExpression bExp = (BinaryExpression)tIndex;
											if( bExp.getLHS().equals(indexVar) || bExp.getRHS().equals(indexVar) ) {
												BinaryOperator bOp = bExp.getOperator();
												if( bOp.equals(BinaryOperator.ADD) || bOp.equals(BinaryOperator.SUBTRACT) ) {
													retMap.put(symbol, new IntegerLiteral(1));
													isSIMDizable = true;
												}
											}
										}
									} else {
										ArrayAccess tAccess = (ArrayAccess)tID.getParent();
											List<Expression> indices = tAccess.getIndices();
											Expression tIndex = null;
											int i=0;
											boolean foundNonConstantIndex = false;
											for( i=0; i<indices.size()-1; i++ ) {
												tIndex = indices.get(i);
												List<IDExpression> tExpList = IRTools.getExpressionsOfType(tIndex, IDExpression.class);
												if( (tExpList != null) && !tExpList.isEmpty() ) {
													foundNonConstantIndex = true;
													break;
												}
											}
											if( !foundNonConstantIndex ) {
												//A[1];
												retMap.put(symbol, new IntegerLiteral(0));
												isSIMDizable =  true;
											}
									}
									if( !retMap.containsKey(symbol) ) {
										retMap.put(symbol, new StringLiteral("Unknown"));
									}
								} else {
									boolean dereferenced = false;
									Traversable tt = tID;
									UnaryExpression uExp = null;
									while( tt != null ) {
										tt = tt.getParent();
										if( tt instanceof UnaryExpression ) {
											uExp = (UnaryExpression)tt;
											if( uExp.getOperator().equals(UnaryOperator.DEREFERENCE) ) {
												dereferenced = true;
												break;
											}
										}
									}
									if( dereferenced ) {
										retMap.put(symbol, new StringLiteral("PointerDereference"));
									}
								}
								if( isLValue ) {
									if( tExp instanceof Identifier ) {
										retMap.put(symbol, new StringLiteral("PointerAssignment"));
									}
									if( !retMap.containsKey(symbol) ) {
										retMap.put(symbol, new StringLiteral("Unknown"));
									}
								}
							}
						}
					}
				}
			}
			return retMap;
		}
		
		void analyzeExpression(Expression inExp, Map<IDExpression, Annotatable> internalParamMap, TranslationUnit mainTrUnt) {
			if( inExp == null ) {
				return;
			}
			if( inExp instanceof BinaryExpression ) {
				BinaryExpression bExp = (BinaryExpression)inExp;
				if( inExp instanceof AssignmentExpression ) {
					//Set<Symbol> fSymSet = getFloatVariables(bExp.getLHS());
					//Map<Symbol, Expression> fSymMap = analyzeFloatVariableAccesses(bExp.getLHS(), true);
					Map<Symbol, Expression> fSymMap = analyzeVariableAccesses(bExp.getLHS(), true);
					Set<Symbol> fSymSet = fSymMap.keySet();
					for( Symbol tSym : fSymSet ) {
						Map<Expression, Expression> statusMap = STORES.get(tSym);
						Expression cstatus = fSymMap.get(tSym);
						if( statusMap == null ) {
							statusMap = new HashMap<Expression, Expression>();
							statusMap.put(cstatus.clone(), new IntegerLiteral(1));
							STORES.put(tSym, statusMap);
						} else {
							Expression tInt = null;
							if( statusMap.containsKey(cstatus) ) {
								tInt = statusMap.get(cstatus);
								if( tInt instanceof IntegerLiteral ) {
									tInt = new IntegerLiteral(((IntegerLiteral)tInt).getValue()+1);
								} else {
									tInt = new BinaryExpression(tInt, BinaryOperator.ADD, new IntegerLiteral(1));
								}
							} else {
								tInt = new IntegerLiteral(1);
							}
							statusMap.put(cstatus.clone(), tInt);
						}
					}
					isExpressionRoot = false;
					analyzeExpression(bExp.getRHS(), internalParamMap, mainTrUnt);
					if( containsFloats || containsDoubles ) {
						AssignmentOperator aOp = (AssignmentOperator)bExp.getOperator();
						if( aOp.equals(AssignmentOperator.ADD) ) {
							FADDS++;
						} else if( aOp.equals(AssignmentOperator.SUBTRACT) ) {
							FSUBS++;
						} else if( aOp.equals(AssignmentOperator.MULTIPLY) ) {
							FMULS++;
						} else if( aOp.equals(AssignmentOperator.DIVIDE) ) {
							FDIVS++;
						}
					} else {
						AssignmentOperator aOp = (AssignmentOperator)bExp.getOperator();
						if( aOp.equals(AssignmentOperator.ADD) ) {
							INTADDS++;
						} else if( aOp.equals(AssignmentOperator.SUBTRACT) ) {
							INTSUBS++;
						} else if( aOp.equals(AssignmentOperator.MULTIPLY) ) {
							INTMULS++;
						} else if( aOp.equals(AssignmentOperator.DIVIDE) ) {
							INTDIVS++;
						}
					}
				} else if( inExp instanceof AccessExpression ) {
					//[FIXME] how to handle member variables?
					//Set<Symbol> fSymSet = getFloatVariables(inExp);
					//Map<Symbol, Expression> fSymMap = analyzeFloatVariableAccesses(inExp, false);
					Map<Symbol, Expression> fSymMap = analyzeVariableAccesses(inExp, false);
					Set<Symbol> fSymSet = fSymMap.keySet();
					for( Symbol tSym : fSymSet ) {
						Map<Expression, Expression> statusMap = LOADS.get(tSym);
						Expression cstatus = fSymMap.get(tSym);
						if( statusMap == null ) {
							statusMap = new HashMap<Expression, Expression>();
							statusMap.put(cstatus.clone(), new IntegerLiteral(1));
							LOADS.put(tSym, statusMap);
						} else {
							Expression tInt = null;
							if( statusMap.containsKey(cstatus) ) {
								tInt = statusMap.get(cstatus);
								if( tInt instanceof IntegerLiteral ) {
									tInt = new IntegerLiteral(((IntegerLiteral)tInt).getValue()+1);
								} else {
									tInt = new BinaryExpression(tInt, BinaryOperator.ADD, new IntegerLiteral(1));
								}
							} else {
								tInt = new IntegerLiteral(1);
							}
							statusMap.put(cstatus.clone(), tInt);
						}
					}
				} else {
					BinaryOperator bOp = bExp.getOperator();
					isExpressionRoot = false;
					analyzeExpression(bExp.getLHS(), internalParamMap, mainTrUnt);
					analyzeExpression(bExp.getRHS(), internalParamMap, mainTrUnt);
					if( containsFloats || containsDoubles ) {
						if( bOp.equals(BinaryOperator.ADD) ) {
							FADDS++;
						} else if( bOp.equals(BinaryOperator.SUBTRACT) ) {
							FSUBS++;
						} else if( bOp.equals(BinaryOperator.MULTIPLY) ) {
							FMULS++;
						} else if( bOp.equals(BinaryOperator.DIVIDE) ) {
							FDIVS++;
						} else if( bOp.equals(BinaryOperator.COMPARE_EQ) || 
								bOp.equals(BinaryOperator.COMPARE_GE) ||
								bOp.equals(BinaryOperator.COMPARE_GT) ||
								bOp.equals(BinaryOperator.COMPARE_LE) ||
								bOp.equals(BinaryOperator.COMPARE_LT) ||
								bOp.equals(BinaryOperator.COMPARE_NE) ) {
							FOTHERS++;
						}
					} else {
						if( bOp.equals(BinaryOperator.ADD) ) {
							INTADDS++;
						} else if( bOp.equals(BinaryOperator.SUBTRACT) ) {
							INTSUBS++;
						} else if( bOp.equals(BinaryOperator.MULTIPLY) ) {
							INTMULS++;
						} else if( bOp.equals(BinaryOperator.DIVIDE) ) {
							INTDIVS++;
						} else if( bOp.equals(BinaryOperator.COMPARE_EQ) || 
								bOp.equals(BinaryOperator.COMPARE_GE) ||
								bOp.equals(BinaryOperator.COMPARE_GT) ||
								bOp.equals(BinaryOperator.COMPARE_LE) ||
								bOp.equals(BinaryOperator.COMPARE_LT) ||
								bOp.equals(BinaryOperator.COMPARE_NE) ) {
							INTOTHERS++;
						}
					}
				}
			} else if( inExp instanceof UnaryExpression ) {
				UnaryExpression uExp = (UnaryExpression)inExp;
				UnaryOperator uOp = uExp.getOperator();
				isExpressionRoot = false;
				analyzeExpression(uExp.getExpression(), internalParamMap, mainTrUnt);
				if( containsFloats || containsDoubles ) {
					if( uOp.equals(UnaryOperator.POST_INCREMENT) || 
							uOp.equals(UnaryOperator.PRE_INCREMENT) ) {
						FADDS++;
					} else if( uOp.equals(UnaryOperator.POST_DECREMENT) || 
							uOp.equals(UnaryOperator.PRE_DECREMENT) ) {
						FSUBS++;
					} else if( uOp.equals(UnaryOperator.POST_DECREMENT) || 
							uOp.equals(UnaryOperator.PRE_DECREMENT) ) {
						FSUBS++;
					} else {
						//Remaining unary operations are not float operations.
						//FOTHERS++;
					}
				} else {
					if( uOp.equals(UnaryOperator.POST_INCREMENT) || 
							uOp.equals(UnaryOperator.PRE_INCREMENT) ) {
						INTADDS++;
					} else if( uOp.equals(UnaryOperator.POST_DECREMENT) || 
							uOp.equals(UnaryOperator.PRE_DECREMENT) ) {
						INTSUBS++;
					} else if( uOp.equals(UnaryOperator.POST_DECREMENT) || 
							uOp.equals(UnaryOperator.PRE_DECREMENT) ) {
						INTSUBS++;
					} else {
						//Remaining unary operations are not float operations.
						//INTOTHERS++;
					}
					
				}
			} else if( inExp instanceof Typecast ) {
				Typecast tExp = (Typecast)inExp;
				boolean prev_containsFloats = containsFloats;
				boolean prev_containsDoubles = containsDoubles;
				Expression tCastExp = ((Typecast)inExp).getExpression();
				isExpressionRoot = false;
				analyzeExpression(tCastExp, internalParamMap, mainTrUnt);
				if( (tCastExp instanceof IDExpression) || (tCastExp instanceof Literal) ) {
					containsFloats = prev_containsFloats;
					containsDoubles = prev_containsDoubles;
				}
				List typeList = tExp.getSpecifiers();
				if( typeList.contains(Specifier.FLOAT) ) {
					containsFloats = true;
				} else if( typeList.contains(Specifier.DOUBLE) ) {
					containsDoubles = true;
				}
			} else if( inExp instanceof FunctionCall ) {
				FunctionCall fCall = (FunctionCall)inExp;
				String fCallName = fCall.getName().toString();
				if( !StandardLibrary.contains(fCall) && !pseudoStandardLibrary.contains(fCallName) ) {
					containsUserFunction = true;
				}
				if( fCallName.equals("malloc") || fCallName.equals("_mm_malloc") ) {
					//void * malloc(size_t size);
					Expression sizeExp = fCall.getArgument(0).clone();
					List<SizeofExpression> sizeofExps = 
							IRTools.getExpressionsOfType(sizeExp, SizeofExpression.class);
					if( (sizeofExps != null) && !sizeofExps.isEmpty() ) {
						SizeofExpression sizeofExp = sizeofExps.get(0);
						if( allocatedUnit == null ) {
							NameID nParamID = getBuiltInParamForTypes(sizeofExp.getTypes(), 
									internalParamMap, mainTrUnt);
							if( nParamID != null ) {
								allocatedUnit = nParamID.clone();
								sizeofExp.swapWith(new IntegerLiteral(1));
								allocatedElems = Symbolic.simplify(sizeExp);
							}
						} else {
							Statement expStmt = inExp.getStatement();
							Procedure cProc = expStmt.getProcedure();
							Tools.exit("[ERROR in ASPENModelAnalysis.ExpressionAnalyzer] multiple memory allocation calls " +
									"exist in the current expression statement; not analyzable; exit! \n" +
									"Current Expression Statement: " + inExp.getStatement() + "\n" +
									"Enclosing Procedure: " + cProc.getSymbolName() +"\nEnclosing Translation Unit: " + 
									((TranslationUnit)cProc.getParent()).getOutputFilename() + "\n");
						}
					} else {
						allocatedElems = Symbolic.simplify(sizeExp.clone());
					}
				} else if( fCallName.equals("calloc") ) {
					//void * calloc(size_t num, size_t size);
					Expression numElms = fCall.getArgument(0).clone();
					Expression sizeExp = fCall.getArgument(1).clone();
					if( allocatedUnit == null ) {
						if( sizeExp instanceof SizeofExpression) {
							NameID nParamID = getBuiltInParamForTypes(((SizeofExpression)sizeExp).getTypes(), 
									internalParamMap, mainTrUnt);
							if( nParamID != null ) {
								allocatedUnit = nParamID.clone();
								allocatedElems = Symbolic.simplify(numElms);
							}
						} else {
							allocatedUnit = sizeExp;
							allocatedElems = Symbolic.simplify(numElms);
						}
					} else {
						Statement expStmt = inExp.getStatement();
						Procedure cProc = expStmt.getProcedure();
						Tools.exit("[ERROR in ASPENModelAnalysis.ExpressionAnalyzer] multiple memory allocation calls " +
								"exist in the current expression statement; not analyzable; exit! \n" +
								"Current Expression Statement: " + inExp.getStatement() + "\n" +
								"Enclosing Procedure: " + cProc.getSymbolName() +"\nEnclosing Translation Unit: " + 
								((TranslationUnit)cProc.getParent()).getOutputFilename() + "\n");
					}
				} else if( fCallName.equals("realloc") ) {
					//void *realloc(void *ptr, size_t size);
					Expression sizeExp = fCall.getArgument(1).clone();
					List<SizeofExpression> sizeofExps = 
							IRTools.getExpressionsOfType(sizeExp, SizeofExpression.class);
					if( (sizeofExps != null) && !sizeofExps.isEmpty() ) {
						SizeofExpression sizeofExp = sizeofExps.get(0);
						NameID nParamID = getBuiltInParamForTypes(sizeofExp.getTypes(), 
								internalParamMap, mainTrUnt);
						if( nParamID != null ) {
							allocatedUnit = nParamID.clone();
							sizeofExp.swapWith(new IntegerLiteral(1));
							allocatedElems = Symbolic.simplify(sizeExp);
						}
					} else {
						allocatedElems = Symbolic.simplify(sizeExp);
					}
				} else if( fCallName.equals("memset") ) {
					//void *memset(void *, int c, size_t n);
					Expression arg = fCall.getArgument(0).clone();
					Set<Symbol> symSet = SymbolTools.getAccessedSymbols(arg);
					Expression sizeExp = fCall.getArgument(2).clone();
					List<SizeofExpression> sizeofExps = 
							IRTools.getExpressionsOfType(sizeExp, SizeofExpression.class);
					if( (sizeofExps != null) && !sizeofExps.isEmpty() ) {
						SizeofExpression sizeofExp = sizeofExps.get(0);
						NameID nParamID = getBuiltInParamForTypes(sizeofExp.getTypes(), 
								internalParamMap, mainTrUnt);
						if( nParamID != null ) {
							sizeofExp.swapWith(new IntegerLiteral(1));
							for( Symbol tSym : symSet ) {
								if( SymbolTools.isPointer(tSym) || SymbolTools.isArray(tSym) ) {
									Map<Expression, Expression> statusMap = STORES.get(tSym);
									Expression cstatus = new IntegerLiteral(1);
									if( statusMap == null ) {
										statusMap = new HashMap<Expression, Expression>();
										statusMap.put(cstatus.clone(), sizeExp);
										STORES.put(tSym, statusMap);
									} else {
										Expression tInt = null;
										if( statusMap.containsKey(cstatus) ) {
											tInt = statusMap.get(cstatus);
											tInt = new BinaryExpression(tInt, BinaryOperator.ADD, sizeExp);
										} else {
											tInt = sizeExp;
										}
										statusMap.put(cstatus.clone(), tInt);
									}
								}
							}
						}
					}
				} else if( StandardLibrary.hasSideEffectOnParameter(fCall) ) { 
					//If this function call modify float variables, this should be handled separately.
					for(Expression arg : fCall.getArguments() ) {
						//Set<Symbol> fSymSet = getFloatVariables(arg);
						//Map<Symbol, Expression> fSymMap = analyzeFloatVariableAccesses(arg, false);
						Map<Symbol, Expression> fSymMap = analyzeVariableAccesses(arg, false);
						Set<Symbol> fSymSet = fSymMap.keySet();
						for( Symbol tSym : fSymSet ) {
							Map<Expression, Expression> statusMap = STORES.get(tSym);
							Expression cstatus = fSymMap.get(tSym);
							if( statusMap == null ) {
								statusMap = new HashMap<Expression, Expression>();
								statusMap.put(cstatus.clone(), new IntegerLiteral(1));
								STORES.put(tSym, statusMap);
							} else {
								Expression tInt = null;
								if( statusMap.containsKey(cstatus) ) {
									tInt = statusMap.get(cstatus);
									if( tInt instanceof IntegerLiteral ) {
										tInt = new IntegerLiteral(((IntegerLiteral)tInt).getValue()+1);
									} else {
										tInt = new BinaryExpression(tInt, BinaryOperator.ADD, new IntegerLiteral(1));
									}
								} else {
									tInt = new IntegerLiteral(1);
								}
								statusMap.put(cstatus.clone(), tInt);
							}
						}
					}
				} else {
					isExpressionRoot = false;
					for(Expression arg : fCall.getArguments() ) {
						analyzeExpression(arg, internalParamMap, mainTrUnt);
					}
					if( StandardLibrary.isSideEffectFree(fCall) && 
							!StandardLibrary.isSideEffectFreeExceptIO(fCall)) {
						//Standard library function with no side effect.
						//Conservatively assumes it as float operation.
						String fNameS = fCall.getName().toString();
						if( !fNameS.equals("exit") && !fNameS.equals("fopen") ) {
							if( containsFloats || containsDoubles ) {
								FOTHERS++;
							}
						}
					}
				}
				List returnTypes = fCall.getReturnType();
				if( returnTypes != null ) {
					if( returnTypes.contains(Specifier.DOUBLE) ) {
						containsDoubles = true;
					} else if( returnTypes.contains(Specifier.FLOAT)) {
						containsFloats = true;
					}
					//[DEBUG] Below simple optimization may not work if the indirect malloc call is called in multiple contexts.
					if( containsUserFunction ) {
						for( Object tObj : returnTypes ) {
							if( tObj instanceof PointerSpecifier ) {
								Procedure ttProc = fCall.getProcedure();
								if( ttProc != null ) {
									CompoundStatement ttBody = ttProc.getBody();
									List<ReturnStatement> retStmtList = IRTools.getStatementsOfType(ttBody, ReturnStatement.class);
									Set<Expression> retExpList = new HashSet<Expression>();
									for( ReturnStatement retStmt : retStmtList ) {
										retExpList.add(retStmt.getExpression());
									}
									List<FunctionCall> inFCallList = IRTools.getFunctionCalls(ttBody);
									if( inFCallList != null ) {
										for( FunctionCall rFcall : inFCallList ) {
											String rFcallName = rFcall.getName().toString();
											if( rFcallName.equals("malloc") || rFcallName.equals("calloc") ||
													rFcallName.equals("_mm_malloc") ) {
												Statement inStmt = rFcall.getStatement();
												if( inStmt != null ) {
													if( inStmt instanceof ReturnStatement ) {
														analyzeExpression(((ReturnStatement) inStmt).getExpression(), internalParamMap, mainTrUnt);
													} else if( inStmt instanceof ExpressionStatement) {
														Expression inTemExp = ((ExpressionStatement)inStmt).getExpression();
														if( inTemExp instanceof AssignmentExpression ) {
															Expression inTemLHS = ((AssignmentExpression)inTemExp).getLHS();
															if( inTemLHS.toString().startsWith("_ret_val") || retExpList.contains(inTemLHS) ) {
																analyzeExpression(((AssignmentExpression)inTemExp).getRHS(), internalParamMap, mainTrUnt);
																if( allocatedElems != null ) {
																	List<Expression> argList = fCall.getArguments();
																	List<Declaration> paramList = ttProc.getParameters();
																	int list_size = paramList.size();
																	if( list_size == 1 ) {
																		Object obj = paramList.get(0);
																		String paramS = obj.toString();
																		// Remove any leading or trailing whitespace.
																		paramS = paramS.trim();
																		if( paramS.equals(Specifier.VOID.toString()) ) {
																			list_size = 0;
																		}
																	}	
																	if( list_size > 0 ) {
																		int i = 0;
																		for( Declaration tParam : paramList ) {
																			Expression tParamID = tParam.getDeclaredIDs().get(0);
																			Symbol tParamSym = SymbolTools.getSymbolOf(tParamID);
																			Expression tArg = argList.get(i);
																			if( allocatedElems.equals(tParamID) ) {
																				allocatedElems = tArg.clone();
																				break;
																			} else {
																				IRTools.replaceAll(allocatedElems, tParamID, tArg);
																			}
																			i++;
																		}
																	}
																}
															}
														}
													}
												}
												break;
											}
										}
									}
								}
								break;
							}
						}
					}
				}
			} else if( inExp instanceof ConditionalExpression ) {
				ConditionalExpression cExp = (ConditionalExpression)inExp;
				isExpressionRoot = false;
				analyzeExpression(cExp.getCondition(), internalParamMap, mainTrUnt);
				analyzeExpression(cExp.getTrueExpression(), internalParamMap, mainTrUnt);
				analyzeExpression(cExp.getFalseExpression(), internalParamMap, mainTrUnt);
			} else if( (inExp instanceof CommaExpression) || (inExp instanceof MinMaxExpression) ) {
				isExpressionRoot = false;
				for( Traversable tt : inExp.getChildren() ) {
					analyzeExpression((Expression)tt, internalParamMap, mainTrUnt);
				}
			} else if( inExp instanceof FloatLiteral ) {
				if( inExp.toString().contains("f") || inExp.toString().contains("F") ) {
					containsFloats = true;
				} else {
					containsDoubles = true;
				}
			} else {
				//Set<Symbol> fSymSet = getFloatVariables(inExp);
				//Map<Symbol, Expression> fSymMap = analyzeFloatVariableAccesses(inExp, false);
				Map<Symbol, Expression> fSymMap = analyzeVariableAccesses(inExp, false);
				Set<Symbol> fSymSet = fSymMap.keySet();
				for( Symbol tSym : fSymSet ) {
					Map<Expression, Expression> statusMap = LOADS.get(tSym);
					Expression cstatus = fSymMap.get(tSym);
					if( statusMap == null ) {
						statusMap = new HashMap<Expression, Expression>();
						statusMap.put(cstatus.clone(), new IntegerLiteral(1));
						LOADS.put(tSym, statusMap);
					} else {
						Expression tInt = null;
						if( statusMap.containsKey(cstatus) ) {
							tInt = statusMap.get(cstatus);
							if( tInt instanceof IntegerLiteral ) {
								tInt = new IntegerLiteral(((IntegerLiteral)tInt).getValue()+1);
							} else {
								tInt = new BinaryExpression(tInt, BinaryOperator.ADD, new IntegerLiteral(1));
							}
						} else {
							tInt = new IntegerLiteral(1);
						}
						statusMap.put(cstatus.clone(), tInt);
					}
				}
			}
		}
		
	}
	
}
