/**
 * 
 */
package openacc.transforms;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import openacc.analysis.AnalysisTools;
import openacc.analysis.SubArray;
import openacc.hir.ACCAnnotation;
import openacc.hir.ARCAnnotation;
import openacc.hir.OpenCLSpecifier;
import cetus.exec.Driver;
import cetus.hir.AccessExpression;
import cetus.hir.AccessSymbol;
import cetus.hir.Annotatable;
import cetus.hir.AssignmentExpression;
import cetus.hir.AssignmentOperator;
import cetus.hir.BinaryExpression;
import cetus.hir.BinaryOperator;
import cetus.hir.ChainedList;
import cetus.hir.CompoundStatement;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.ForLoop;
import cetus.hir.FunctionCall;
import cetus.hir.IDExpression;
import cetus.hir.IRTools;
import cetus.hir.Identifier;
import cetus.hir.IfStatement;
import cetus.hir.Initializer;
import cetus.hir.IntegerLiteral;
import cetus.hir.Loop;
import cetus.hir.NameID;
import cetus.hir.PointerSpecifier;
import cetus.hir.PrintTools;
import cetus.hir.Procedure;
import cetus.hir.Program;
import cetus.hir.PseudoSymbol;
import cetus.hir.SizeofExpression;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.StringLiteral;
import cetus.hir.SwitchStatement;
import cetus.hir.Symbol;
import cetus.hir.SymbolTools;
import cetus.hir.Tools;
import cetus.hir.TranslationUnit;
import cetus.hir.Traversable;
import cetus.hir.Typecast;
import cetus.hir.UnaryExpression;
import cetus.hir.UnaryOperator;
import cetus.hir.UserSpecifier;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import cetus.transforms.TransformPass;

/**
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group
 *         Oak Ridge National Laboratory
 *
 */
public class FaultInjectionTransformation extends TransformPass {
	private boolean IRSymbolOnly;
	private int FaultInjectionOption;
	private int OPENARC_ARCH = 0;
	private String passName = "[FaultInjectionTransformation]";
	static public String ftinjectionCallBaseName = "HI_ftinjection";

	/**
	 * 
	 * @param program
	 */
	public FaultInjectionTransformation(Program program, boolean IRSymOnly, int FTOption, int targetARCH) {
		super(program);
		IRSymbolOnly = IRSymOnly;
		FaultInjectionOption = FTOption;
		OPENARC_ARCH = targetARCH;
		//disable_protection = true;
	}

	/* (non-Javadoc)
	 * @see cetus.transforms.TransformPass#getPassName()
	 */
	@Override
	public String getPassName() {
		return passName;
	}

	/* (non-Javadoc)
	 * @see cetus.transforms.TransformPass#start()
	 */
	@Override
	public void start() {
		//for each resilience,
		//Step0: if ifcond exists and its argument is 0, skip the current resilience region.
		//Step1: - If the resilience region is inside of compute regions, error.
		//Step1.1: Create temporary variable (int _ti_targetThread) to choose target thread to inject faults.
		//Step1.5: - If repeat clause exists, and if the resilience region is a compute region,
		//         generate an enclosing loop for the compute region and host side codes to 
		//         store reference values of copyout data and their CPU output values.
		//Step1.9: [TODO] if resilience region contains compute regions as children, but if the compute
		//         regions don't have ftregion clauses, add them to the compute regions and also
		//         add ftdata with subset of ftdata of the resilience region if existing.
		//         To avoid this problem, each compute region should have explicit ftregion clause.
		//Step2: - Count number of ftregions/ftinject including the one in the resilience directive.
		//           - Assign a unique id for each ftregion and ftinject.
		//               - ftregion => ftregionid  and ftinject => ftinjectid mapping
		//           - If no ftregions/ftinject exists, add ftregion to the resilience directive.
		//           - each ftregion/ftinject has internal pointer to the enclosing resilience region
		//               - tresilience => enclosing resilience region mapping
		//           - each ftregion/ftinject is annotated with tftsymbols containing symbols in the ftdata clause.
		//Step3: - If ftdata(*), find shared symbols accessed in the attached region.
		//           - If the attached region is a compute region, accshared internal clause
		//           is used to find accessed shared symbols.
		//           - Else if all ftregions/ftinjects in the current resilience region has explicit ftdata clauses,
		//               - add all symbols in the ftdata clauses.
		//           - Otherwise, analyze accessed shared symbols.
		//           - Keep internal tftsymbols set containing symbols in the ftdata clause.
		//Step4: - For each symbol in ftdata clause, 
		//           - Create temporary variables (ipos) to decide at which statement to inject 
		//           fault for each ftregion.
		//               - Store symbol => temp variable symbol list mapping internally in the resilience directive.
		//           - Create temporary variables (epos) to decide at which array element to
		//           inject fault for each ftregion/ftinject
		//               - Store symbol => temp variable symbol list mapping internally in the resilience directive.
		//           - Create random bitvectors for each ftregion/ftinject.
		//               - Store symbol => bitvector symbol list mapping internally in the resilience directive.
		//Step5: - for each symbol in ftdata clause,
		//           - Set values for ipos/epos/bitvector
		//Step6: - handle ftcond/num_faults clauses.
		//           - Store the condition symbol into tftcondsym clause of the resilience directive.
		//Step7: - for each ftregion/ftinject,
		//           - Generate fault-injection code for each symbol in ftdata
		//Step8: - for each compute region enclosed in the resilience region
		//           - add temp varaibles (ipos, epos, bitvector) for each symbol in tftsymbols set to copyin, 
		//           sharedRO, accshared, and accreadonly clauses.
		//Step9: - Remove temp clauses (tftsymbols, tresilience, tiposmap, teposmap, tbitvecmap, tftcondsym)
		//[DEBUG] In current implementation that assumes all ftregion/ftinject constructs are lexically included,
		//        most of these temp clauses are not needed, but for interprocedural handling, these will be needed.
		int programVerification = 0;
		String value = Driver.getOptionValue("programVerification");
		if( value != null ) {
			programVerification = Integer.valueOf(value).intValue();
		}
		List<FunctionCall> funcCallList = IRTools.getFunctionCalls(program);
		List<ARCAnnotation> resilienceAnnots = IRTools.collectPragmas(program, ARCAnnotation.class, "resilience");
		if( (resilienceAnnots != null) && (!resilienceAnnots.isEmpty()) ) {
			List<ARCAnnotation> ftregionAnnots = new LinkedList<ARCAnnotation>();
			List<ARCAnnotation> ftinjectAnnots = new LinkedList<ARCAnnotation>();
			int resR_cnt = 0;
			for( ARCAnnotation resAnnot : resilienceAnnots ) {
				Statement resRegion = (Statement)resAnnot.getAnnotatable();
				CompoundStatement cStmt = (CompoundStatement)resRegion.getParent();
				Procedure cProc = IRTools.getParentProcedure(resRegion);
				CompoundStatement cBody = cProc.getBody();
				TranslationUnit parentTrUnt = (TranslationUnit)cProc.getParent();
				int num_ftregions = 0;
				int num_ftinjects = 0;
				Expression num_ftbits = resAnnot.get("num_ftbits");
				if( num_ftbits == null ) {
					num_ftbits = new IntegerLiteral(1);
				}
				//Step0: if ifcond exists and its argument is 0, skip the current resilience region.
				Expression ftcond = resAnnot.get("ftcond");
				Expression tftcondExp = null;
				if( ftcond != null ) {
					tftcondExp = ftcond.clone();
					if( (tftcondExp instanceof IntegerLiteral) && ( ((IntegerLiteral)tftcondExp).getValue() == 0 ) ) {
						resR_cnt++;
						continue;
					}
				}
				//Step0.8: Create a counter variable for each resilience region.
				List<Specifier>	ftSpecs = new ArrayList<Specifier>(2);
				ftSpecs.add(Specifier.STATIC);
				ftSpecs.add(Specifier.INT);
				Identifier ftCntID = TransformTools.getTempScalar(cBody, ftSpecs, "_ti_ftcnt" + resR_cnt, 0);
				VariableDeclarator ftCntSym = (VariableDeclarator)ftCntID.getSymbol();
				ftCntSym.setInitializer(new Initializer(new IntegerLiteral(0)));
				//Step0.9: Insert "HI_set_srand() function at the begining of each resilience region.
				//[DEBUG] C srand() function is global, but it does not seem to be global when compiled with
				//NVCC; to avoid this weird scope issue, srand() function is called for each resilience region.
				Expression tIfExp = new BinaryExpression(ftCntID.clone(), BinaryOperator.COMPARE_EQ, new IntegerLiteral(0));
				IfStatement srandStmt = new IfStatement(tIfExp, new ExpressionStatement(
						new FunctionCall(new NameID("HI_set_srand"))));
				cStmt.addStatementBefore(resRegion, srandStmt);
				//Step1
				ACCAnnotation compAnnot = null;
				compAnnot = resRegion.getAnnotation(ACCAnnotation.class, "kernels");
				if( compAnnot == null ) {
					compAnnot = resRegion.getAnnotation(ACCAnnotation.class, "parallel");
				}
				ACCAnnotation pcompAnnot = AnalysisTools.ipFindFirstPragmaInParent(resRegion, ACCAnnotation.class, 
						ACCAnnotation.computeRegions, false, funcCallList, null);
				if( pcompAnnot != null ) {
					Tools.exit("[ERROR in FaultInjectionTransformation()] resilience construct can not exist inside of " +
							"any compute regions (kerenls/parallel regions):\n" +
							"Enclosing procedure: " + cProc.getSymbolName() + "\nOpenACC annotation: " + resAnnot + "\n");
				}
				//Step1.1: Create temporary variable (int _ti_targetThread) and assign random value.
				List<Specifier> ttSpecs = new ArrayList<Specifier>(2);
				//ttSpecs.add(Specifier.LONG);
				ttSpecs.add(Specifier.INT);
				Identifier targetThread = null;
				if( FaultInjectionOption > 0 ) {
					targetThread = TransformTools.getTempScalar(cBody, ttSpecs, "_ti_targetThread"+resR_cnt, 0);
				}
				//Step1.5
				if( resAnnot.containsKey("repeat") ) {
					if( compAnnot == null ) {
						PrintTools.println("\n[WARNING in FaultInjectionTransformation()] in current implementation, " +
								"repeat clause in a resilience directive " +
								"is allowed only if the attached region is a compute region; repeat clause will be ignored!\n" +
								"Enclosing procedure: " + cProc.getSymbolName() + "\nOpenACC annotation: " + resAnnot + "\n", 0);
					} else {
						if( programVerification != 2 ) {
							Tools.exit("[ERROR in FaultInjectionTransformation()] for correct translation of repeat clause " +
									"in the following resilience construct, \"programVerification=2\" option should be set; exit!\n" +
									"Enclosing procedure: " + cProc.getSymbolName() + "\nOpenACC annotation: " + resAnnot + "\n");
						}
						Expression rpt_cnt = resAnnot.get("repeat");
						Identifier rpt_indx = TransformTools.getNewTempIndex(cBody); 
						resAnnot.put("rpt_index", rpt_indx); //store the loop-index variable for later use; this should
															//be removed before the final output printing.
						Expression initExp = new AssignmentExpression(rpt_indx.clone(), AssignmentOperator.NORMAL,
								new IntegerLiteral(0));
						Expression condExp = new BinaryExpression(rpt_indx.clone(), BinaryOperator.COMPARE_LT,
								rpt_cnt.clone());
						Expression stepExp = new UnaryExpression(UnaryOperator.POST_INCREMENT, rpt_indx.clone());
						CompoundStatement loopBody = new CompoundStatement();
						ForLoop rptLoop = new ForLoop(new ExpressionStatement(initExp), condExp, stepExp, loopBody);
						Statement compRegion = (Statement)compAnnot.getAnnotatable();
						CompoundStatement parentStmt = (CompoundStatement)compRegion.getParent();
						compRegion.swapWith(rptLoop);
						loopBody.addStatement(compRegion);
						
						//Create a variable to count the number of detected errors.
						List<Specifier> tSpecs = new ArrayList<Specifier>(2);
						tSpecs.add(Specifier.STATIC);
						tSpecs.add(Specifier.INT);
						//Identifier totalNumErrors = TransformTools.getTempScalar(cBody, tSpecs, "_ti_totalfaults" + resR_cnt, 0);
						//DEBUG: resilience construct with repeat clause can not be nested, and thus having one variable is OK
						//for all resilience constructs.
						Identifier totalNumErrors = TransformTools.getTempScalar(cBody, tSpecs, "_ti_totalfaults", 0);
						VariableDeclarator tNESym = (VariableDeclarator)totalNumErrors.getSymbol();
						tNESym.setInitializer(new Initializer(new IntegerLiteral(0)));
						//resAnnot.put("totalnumerrors", totalNumErrors); //store the variable temporarily.
						ExpressionStatement expStmt = new ExpressionStatement(new AssignmentExpression(
								totalNumErrors.clone(), AssignmentOperator.NORMAL, new IntegerLiteral(0)));
						//parentStmt.addStatementBefore(rptLoop, expStmt);
						loopBody.addStatementBefore(compRegion, expStmt);
						FunctionCall fCall = new FunctionCall(new NameID("printf"));
						fCall.addArgument(new StringLiteral("Total number of deteced faults: %d\\n"));
						fCall.addArgument(totalNumErrors.clone());
						expStmt = new ExpressionStatement(fCall);
						loopBody.addStatement(expStmt);
					
						//Parent region of the current resilience region is changed to the loopbody.
						cStmt = loopBody;
						Set<Symbol> outSymbols = new HashSet<Symbol>();
						ACCAnnotation tiAnnot = compRegion.getAnnotation(ACCAnnotation.class, "accshared");
						if( tiAnnot != null ) {
							Set<Symbol> tSet = (Set<Symbol>)tiAnnot.get("accshared");
							if( tSet != null ) {
								outSymbols.addAll(tSet);
							}
							tSet = null;
							tiAnnot = compRegion.getAnnotation(ACCAnnotation.class, "accreadonly");
							if( tiAnnot != null ) {
								tSet = (Set<Symbol>)tiAnnot.get("accreadonly");
								if( tSet != null ) {
									outSymbols.removeAll(tSet);
								}

							}
						}
						/*							VariableDeclaration bytes_decl = (VariableDeclaration)SymbolTools.findSymbol(parentTrUnt, "gpuBytes");
							Identifier cloned_bytes = new Identifier((VariableDeclarator)bytes_decl.getDeclarator(0));*/			
						//gpuBytes variable has not been created yet, and thus use fake Identifier for now.
						//[FIXME] if SymbolTools.getOrphanID() is used, the identifier will be linked to correct symbol
						//in the later pass, but it will generate several warning messages after this pass.
						//Therefore, NameID() will be used for now, but it should be changed to correct identifier later.
						//Identifier cloned_bytes = SymbolTools.getOrphanID("gpuBytes");
						IDExpression cloned_bytes = new NameID("gpuBytes");
						for( Symbol outSym : outSymbols ) {
							SubArray outSA = AnalysisTools.findSubArrayInDataClauses(compAnnot, outSym, IRSymbolOnly);
							if( outSA == null ) {
							Tools.exit("[ERROR in FaultInjectionTransformation()] can not find subarray for symbol " + 
									outSym.getSymbolName() + " in the following OpenACC annotation:\n" + 
									"Enclosing procedure: " + cProc.getSymbolName() + "\nOpenACC annotation: " + resAnnot + "\n");
							}
							Expression hostVar = null;
							Expression ftrefVar = null;
							Expression ftoutVar = null;
							if( outSym instanceof AccessSymbol ) {
								hostVar = AnalysisTools.accessSymbolToExpression((AccessSymbol)outSym, null);
							} else {
								hostVar = new Identifier(outSym);
							}
							List<Expression> startList = new LinkedList<Expression>();
							List<Expression> lengthList = new LinkedList<Expression>();
							boolean foundDimensions = AnalysisTools.extractDimensionInfo(outSA, startList, lengthList, IRSymbolOnly, compAnnot.getAnnotatable());
							if( !foundDimensions ) {
								Tools.exit("[ERROR in FaultInjectionTransformation()] Dimension information " +
										"of the following variable is" +
										"unknown: " + outSA.getArrayName() + ", OpenACC directive: " + compAnnot +
								"; the ACC2GPU translation failed!");
							}

							List<Specifier> typeSpecs = new ArrayList<Specifier>();
							Symbol IRSym = outSym;
							Symbol sym = outSym;
							if( outSym instanceof PseudoSymbol ) {
								IRSym = ((PseudoSymbol)outSym).getIRSymbol();
							}
							if( IRSymbolOnly ) {
								sym = IRSym;
								typeSpecs.addAll(((VariableDeclaration)outSym.getDeclaration()).getSpecifiers());
							} else {
								Symbol tSym = outSym;
								while( tSym instanceof AccessSymbol ) {
									tSym = ((AccessSymbol)tSym).getMemberSymbol();
								}
								typeSpecs.addAll(((VariableDeclaration)tSym.getDeclaration()).getSpecifiers());
							}
							StringBuilder str = new StringBuilder(80);
							if( hostVar instanceof AccessExpression ) {
								str.append(TransformTools.buildAccessExpressionName((AccessExpression)hostVar));
							} else {
								str.append(hostVar.toString());
							}
							String symNameBase = str.toString();
							String ftrefName = "ftref__" + symNameBase;
							String ftoutName = "ftout__" + symNameBase;
							List<Specifier> clonedspecs = new ChainedList<Specifier>();
							clonedspecs.addAll(typeSpecs);
							clonedspecs.remove(Specifier.STATIC);
							///////////////////////////////////////////
							// GPU variables should not be constant. //
							///////////////////////////////////////////
							clonedspecs.remove(Specifier.CONST);
							//////////////////////////////
							// Remove extern specifier. //
							//////////////////////////////
							clonedspecs.remove(Specifier.EXTERN);
							//Add "gpuBytes = SIZE_a * sizeof(float);" before the repeat loop.
							SizeofExpression sizeof_expr = new SizeofExpression(clonedspecs);
							Expression biexp = sizeof_expr.clone();
							for( int i=0; i<lengthList.size(); i++ )
							{
								biexp = new BinaryExpression(biexp, BinaryOperator.MULTIPLY, lengthList.get(i).clone());
							}
							AssignmentExpression assignex = new AssignmentExpression(cloned_bytes.clone(),AssignmentOperator.NORMAL, 
									biexp);
							Statement gpuBytes_stmt = new ExpressionStatement(assignex);
							parentStmt.addStatementBefore(rptLoop, gpuBytes_stmt);
							/////////////////////////////////////////////////////////////////////////
							// Create a variable to keep the original value (ex: float *ftref__a;) //
							/////////////////////////////////////////////////////////////////////////
							Set<Symbol> symSet = cBody.getSymbols();
							Symbol ftref_sym = AnalysisTools.findsSymbol(symSet, ftrefName);
							if( ftref_sym != null ) {
								ftrefVar = new Identifier(ftref_sym);
							} else {
								// Create a new ftref-host variable.
								// The type of the ftref-host symbol should be a pointer type 
								VariableDeclarator ftref_declarator = new VariableDeclarator(PointerSpecifier.UNQUALIFIED, 
										new NameID(ftrefName));
								VariableDeclaration ftref_decl = new VariableDeclaration(clonedspecs, 
										ftref_declarator);
								ftrefVar = new Identifier(ftref_declarator);
								ftref_sym = ftref_declarator;
								cBody.addDeclaration(ftref_decl);
							}
							//Add malloc() statement for the ft-ref host variable.
							FunctionCall malloc_call = new FunctionCall(new NameID("malloc"));
							malloc_call.addArgument(cloned_bytes.clone());
							List<Specifier> specs = new ArrayList<Specifier>(4);
							specs.addAll(ftref_sym.getTypeSpecifiers());
							Statement malloc_stmt = new ExpressionStatement(new AssignmentExpression(ftrefVar.clone(),
									AssignmentOperator.NORMAL, new Typecast(specs, malloc_call)));
							parentStmt.addStatementBefore(rptLoop, malloc_stmt);
							//Add memcpy() statement from host variable to  the ft-ref host variable
							FunctionCall copy_call = new FunctionCall(new NameID("memcpy"));
							copy_call.addArgument(ftrefVar.clone());
							if( lengthList.size() == 0 ) { //hostVar is scalar.
								copy_call.addArgument( new UnaryExpression(UnaryOperator.ADDRESS_OF, 
										hostVar.clone()));
							} else {
								copy_call.addArgument(hostVar.clone());
							}
							copy_call.addArgument(cloned_bytes.clone());
							Statement memcpy_stmt = new ExpressionStatement(copy_call);
							parentStmt.addStatementBefore(rptLoop, memcpy_stmt);
							//Add memcpy() statement from the ft-ref host variable to the host variable in the loopbody.
							loopBody.addStatementBefore(compRegion, gpuBytes_stmt.clone());
							copy_call = new FunctionCall(new NameID("memcpy"));
							if( lengthList.size() == 0 ) { //hostVar is scalar.
								copy_call.addArgument( new UnaryExpression(UnaryOperator.ADDRESS_OF, 
										hostVar.clone()));
							} else {
								copy_call.addArgument(hostVar.clone());
							}
							copy_call.addArgument(ftrefVar.clone());
							copy_call.addArgument(cloned_bytes.clone());
							memcpy_stmt = new ExpressionStatement(copy_call);
							loopBody.addStatementBefore(compRegion, memcpy_stmt);
							//Add free() statement for the ft-ref host variable.
							FunctionCall free_call = new FunctionCall(new NameID("free"));
							free_call.addArgument(ftrefVar.clone());
							Statement free_stmt = new ExpressionStatement(free_call);
							parentStmt.addStatementAfter(rptLoop, free_stmt);
							///////////////////////////////////////////////////////////////////////////
							// Create a variable to keep the CPU-output value (ex: float *ftout__a;) //
							///////////////////////////////////////////////////////////////////////////
							symSet = cBody.getSymbols();
							Symbol ftout_sym = AnalysisTools.findsSymbol(symSet, ftoutName);
							if( ftout_sym != null ) {
								ftoutVar = new Identifier(ftout_sym);
							} else {
								// Create a new ftout-host variable.
								// The type of the ftout-host symbol should be a pointer type 
								VariableDeclarator ftout_declarator = new VariableDeclarator(PointerSpecifier.UNQUALIFIED, 
										new NameID(ftoutName));
								VariableDeclaration ftout_decl = new VariableDeclaration(clonedspecs, 
										ftout_declarator);
								ftoutVar = new Identifier(ftout_declarator);
								ftout_sym = ftout_declarator;
								cBody.addDeclaration(ftout_decl);
							}
							//Add malloc() statement for the ft-ref host variable.
							malloc_call = new FunctionCall(new NameID("malloc"));
							malloc_call.addArgument(cloned_bytes.clone());
							specs = new ArrayList<Specifier>(4);
							specs.addAll(ftout_sym.getTypeSpecifiers());
							malloc_stmt = new ExpressionStatement(new AssignmentExpression(ftoutVar.clone(),
									AssignmentOperator.NORMAL, new Typecast(specs, malloc_call)));
							parentStmt.addStatementBefore(rptLoop, malloc_stmt);
							//Add memcpy() statement from host variable to  the ft-out host variable
							//[DEBUG] this will be added by ACC2CUDATranslator.
							//Add free() statement for the ft-out host variable.
							free_call = new FunctionCall(new NameID("free"));
							free_call.addArgument(ftoutVar.clone());
							free_stmt = new ExpressionStatement(free_call);
							parentStmt.addStatementAfter(rptLoop, free_stmt);
						}
					}
				}
				//Step1.9: [TODO] [FIXME] if resilience/ftregion region contains compute regions as children, but if the compute
				//         regions don't have ftregion clauses, add them to the compute regions and also
				//         add ftdata with subset of ftdata of the resilience region if existing.
				//         To avoid this problem, each compute region should have explicit ftregion clause.
				//Step2
				List<ARCAnnotation> tAnnots = AnalysisTools.ipCollectPragmas(resRegion, ARCAnnotation.class, 
						ARCAnnotation.ftinjecttargets, false, null);
				ftregionAnnots.clear();
				ftinjectAnnots.clear();
				if( tAnnots != null ) {
					for( ARCAnnotation tAnnot : tAnnots ) {
						if( tAnnot.containsKey("ftregion") ) {
							tAnnot.put("ftregion", new IntegerLiteral(num_ftregions++));
							tAnnot.put("tresilience", resRegion);
							ftregionAnnots.add(tAnnot);
						} else if( tAnnot.containsKey("ftinject") ) {
							tAnnot.put("ftinject", new IntegerLiteral(num_ftinjects++));
							tAnnot.put("tresilience", resRegion);
							ftinjectAnnots.add(tAnnot);
						}
					}
				}
				if( (num_ftregions == 0) && (num_ftinjects == 0) ) {
					resAnnot.put("ftregion", new IntegerLiteral(num_ftregions++));
					resAnnot.put("tresilience", resRegion);
					ftregionAnnots.add(resAnnot);
				}
				boolean allExplicitFTData = true;
				Set<SubArray> tftdata = new HashSet<SubArray>();
				Set<Symbol> ttftsymbols = new HashSet<Symbol>();
				List<ARCAnnotation> ftAnnotsToFill = new LinkedList<ARCAnnotation>();
				List<ARCAnnotation> ftRIAnnots = new LinkedList<ARCAnnotation>(ftregionAnnots);
				ftRIAnnots.addAll(ftinjectAnnots);
				for( ARCAnnotation tAnnot : ftRIAnnots ) {
					Annotatable at = tAnnot.getAnnotatable();
					if( at == resRegion ) {
						continue;
					}
					Set<Symbol> tftsymbols = resAnnot.get("tftsymbols");
					if( tftsymbols == null ) {
						tftsymbols = new HashSet<Symbol>();
						tAnnot.put("tftsymbols", tftsymbols);
					}
					Set<SubArray> tSSet  = tAnnot.get("ftdata");
					if( tSSet == null ) {
						allExplicitFTData = false;
						//[DEBUG] should we fill ftdata clause for this ftregion/ftinject?
					} else {
						for( SubArray tSAr : tSSet ) {
							if( tSAr.getArrayName().toString().equals("*") ) {
								allExplicitFTData = false;
								ftAnnotsToFill.add(tAnnot);
								break;
							} else {
								Symbol tSym = AnalysisTools.subarrayToSymbol(tSAr, IRSymbolOnly);
								tftsymbols.add(tSym);
								if( !ttftsymbols.contains(tSym) ) {
									ttftsymbols.add(tSym);
									tftdata.add(tSAr.clone());
								}
							}
						}
					}
				}
				if( !ftAnnotsToFill.isEmpty() ) {
					//Fill ftdata set for fault-inject regions with ftdata(*).
					for( ARCAnnotation ftA : ftAnnotsToFill ) {
						ftA.remove("ftdata");
						ftA.remove("tftsymbols");
						fillFTData(ftA);
					}
				}
				//Step3
				Set<SubArray> ftdata = resAnnot.get("ftdata");
				boolean findData = false;
				if( ftdata == null ) {
					ftdata = new HashSet<SubArray>();
					resAnnot.put("ftdata", ftdata);
					findData = true;
				} else {
					if( ftdata.isEmpty() ) {
						findData = true;
					} else {
						for( SubArray sArr : ftdata ) {
							if( sArr.getArrayName().toString().equals("*") ) {
								findData = true;
								break;
							}
						}
					}
				}
				if( findData ) {
					ftdata.clear();
					if( compAnnot != null ) {
						//If resilience region is a compute region, use the shared symbols
						//in the compute region.
						ACCAnnotation cAnnot = resRegion.getAnnotation(ACCAnnotation.class, "accshared");
						if( cAnnot != null ) {
							Set<Symbol> accsharedSet = cAnnot.get("accshared");
							//Set<Symbol> tftsymbols = new HashSet<Symbol>(accsharedSet);
							Set<Symbol> tftsymbols = new HashSet<Symbol>();
							for( Symbol ttSym : accsharedSet ) {
								String ttSymName = ttSym.getSymbolName();
								if( !ttSymName.startsWith("ipos__") && !ttSymName.startsWith("epos__") &&
										!ttSymName.startsWith("bitv__") && !ttSymName.startsWith("_ti_ftcond") &&
										!ttSymName.startsWith("_ti_targetThread") ) {
									tftsymbols.add(ttSym);
								}
							}
							resAnnot.put("tftsymbols", tftsymbols);
							for( Symbol inSym : tftsymbols ) {
								SubArray tSArr = AnalysisTools.createSubArray(inSym, false, null);
								if( tSArr == null ) {
									PrintTools.println("\n[WARNING in FaultInjectionTransformation()] can not create " +
											"a subarray for following symbol: \n" +
											"Symbol: " + inSym.getSymbolName() + "\nACCAnnotation: " + resAnnot + 
											"\nEnclosing Procedure: " + cProc.getSymbolName() + "\n", 0);
								} else {
									ftdata.add(tSArr);
								}
							}
						}
					} else {
						//If all ftregion or ftinject construct in the resilience region contain explicit ftdata,
						//the union of the ftdata set becomes the ftdata for this resilience region.
						if( allExplicitFTData ) {
							//FIXME: if symbols is not visible in the resilience region scope, symbols should be
							//updated as necessary, but not handled in the current implementation.
							ftdata.addAll(tftdata);
							resAnnot.put("tftsymbols", ttftsymbols);
						}
						if( !allExplicitFTData || ftdata.isEmpty() ) {
							fillFTData(resAnnot);
						}
					}
				}
				Set<Symbol> tftsymbols = resAnnot.get("tftsymbols");
				if( tftsymbols == null ) {
					//Fill tftsymbols set
					tftsymbols = new HashSet<Symbol>();
					for( SubArray tSArr : ftdata ) {
						Symbol tSym = AnalysisTools.subarrayToSymbol(tSArr, IRSymbolOnly);
						if( tSym != null ) {
							tftsymbols.add(tSym);
						}
					}
					resAnnot.put("tftsymbols", tftsymbols);
				}
				//Step4
				Map<Symbol, Map<String, Symbol>> iposMap = new HashMap<Symbol, Map<String, Symbol>>();
				Map<Symbol, Map<String, Symbol>> eposMap = new HashMap<Symbol, Map<String, Symbol>>();
				Map<Symbol, Map<String, Symbol>> bitVecMap = new HashMap<Symbol, Map<String, Symbol>>();
				resAnnot.put("tiposmap", iposMap);
				resAnnot.put("teposmap", eposMap);
				resAnnot.put("tbitvecmap", bitVecMap);
				for( Symbol ftSym : tftsymbols ) {
					String symNameBase = null;
					if( ftSym instanceof AccessSymbol) {
						symNameBase = TransformTools.buildAccessSymbolName((AccessSymbol)ftSym);
					} else {
						symNameBase = ftSym.getSymbolName();
					}
					//Create a temp variable to decide at which statement to inject fault for each ftregion.
					//Naming rule: ipos__[symname]_FR[ftregionid]
					//             e.g., unsigned long int ipos__a_FR0;
					Map<String, Symbol> tiposMap = new HashMap<String, Symbol>();
					for( ARCAnnotation ftRAnnot : ftregionAnnots ) {
						Set<Symbol> l_tftsymbols = ftRAnnot.get("tftsymbols");
						//FIXME: below assumes that ftregion is lexically included in the resilience region.
						if( (l_tftsymbols != null) && (l_tftsymbols.contains(ftSym))) {
							IntegerLiteral tInt = ftRAnnot.get("ftregion");
							String iposName = "ipos__" + symNameBase + "_FR" + tInt.toString();
							List<Specifier> tSpecs = new ArrayList<Specifier>(3);
							//tSpecs.add(Specifier.UNSIGNED);
							//tSpecs.add(Specifier.LONG);
							tSpecs.add(Specifier.INT);
							Identifier iposID = TransformTools.getTempScalar(cBody, tSpecs, iposName, 0);
							tiposMap.put(iposName, iposID.getSymbol());
						}
					}
					iposMap.put(ftSym, tiposMap);
					//Create a temp variable to decide at which array element to inject fault for each ftregion
					//and ftinject.
					//Naming rule: epos__[symname]_FR[ftregionid] or epos__[symname]_FI[ftinjectid]
					//             e.g., unsigned long int epos__a_FR0, epos__a_FI0;
					Map<String, Symbol> teposMap = new HashMap<String, Symbol>();
					for( ARCAnnotation ftRAnnot : ftregionAnnots ) {
						Set<Symbol> l_tftsymbols = ftRAnnot.get("tftsymbols");
						//FIXME: below assumes that ftregion is lexically included in the resilience region.
						if( (l_tftsymbols != null) && (l_tftsymbols.contains(ftSym))) {
							IntegerLiteral tInt = ftRAnnot.get("ftregion");
							String eposName = "epos__" + symNameBase + "_FR" + tInt.toString();
							List<Specifier> tSpecs = new ArrayList<Specifier>(3);
							tSpecs.add(Specifier.UNSIGNED);
							//tSpecs.add(Specifier.LONG);
							tSpecs.add(Specifier.INT);
							Identifier eposID = TransformTools.getTempScalar(cBody, tSpecs, eposName, 0);
							teposMap.put(eposName, eposID.getSymbol());
						}
					}
					for( ARCAnnotation ftRAnnot : ftinjectAnnots ) {
						Set<Symbol> l_tftsymbols = ftRAnnot.get("tftsymbols");
						//FIXME: below assumes that ftinject is lexically included in the resilience region.
						if( (l_tftsymbols != null) && (l_tftsymbols.contains(ftSym))) {
							IntegerLiteral tInt = ftRAnnot.get("ftinject");
							String eposName = "epos__" + symNameBase + "_FI" + tInt.toString();
							List<Specifier> tSpecs = new ArrayList<Specifier>(3);
							tSpecs.add(Specifier.UNSIGNED);
							//tSpecs.add(Specifier.LONG);
							tSpecs.add(Specifier.INT);
							Identifier eposID = TransformTools.getTempScalar(cBody, tSpecs, eposName, 0);
							teposMap.put(eposName, eposID.getSymbol());
						}
					}
					eposMap.put(ftSym, teposMap);
					
					//Create a bitvector for each ftregion and ftinject.
					//Naming rule: bitv__[symname]_FR[ftregionid] or bitv__[symname]_FI[ftinjectid]
					//             e.g., type32 bitv__a_FR0; type64b bitv__a_FI0;
					Specifier bitvType = AnalysisTools.getBitVecType(ftSym);
					Map<String, Symbol> tbitVecMap = new HashMap<String, Symbol>();
					for( ARCAnnotation ftRAnnot : ftregionAnnots ) {
						Set<Symbol> l_tftsymbols = ftRAnnot.get("tftsymbols");
						//FIXME: below assumes that ftregion is lexically included in the resilience region.
						if( (l_tftsymbols != null) && (l_tftsymbols.contains(ftSym))) {
							IntegerLiteral tInt = ftRAnnot.get("ftregion");
							String bitvName = "bitv__" + symNameBase + "_FR" + tInt.toString();
							List<Specifier> tSpecs = new ArrayList<Specifier>(1);
							tSpecs.add(bitvType);
							Identifier bitvID = TransformTools.getTempScalar(cBody, tSpecs, bitvName, 0);
							tbitVecMap.put(bitvName, bitvID.getSymbol());
						}
					}
					for( ARCAnnotation ftRAnnot : ftinjectAnnots ) {
						Set<Symbol> l_tftsymbols = ftRAnnot.get("tftsymbols");
						//FIXME: below assumes that ftinject is lexically included in the resilience region.
						if( (l_tftsymbols != null) && (l_tftsymbols.contains(ftSym))) {
							IntegerLiteral tInt = ftRAnnot.get("ftinject");
							String bitvName = "bitv__" + symNameBase + "_FI" + tInt.toString();
							List<Specifier> tSpecs = new ArrayList<Specifier>(1);
							tSpecs.add(bitvType);
							Identifier bitvID = TransformTools.getTempScalar(cBody, tSpecs, bitvName, 0);
							tbitVecMap.put(bitvName, bitvID.getSymbol());
						}
					}
					bitVecMap.put(ftSym, tbitVecMap);
				}
				//Step5
				//Create a temp variable to keep array sizes.
				List<Specifier> tSpecs = new ArrayList<Specifier>(2);
				//tSpecs.add(Specifier.LONG);
				tSpecs.add(Specifier.INT);
				Identifier arraySizeID = TransformTools.getTempScalar(cBody, tSpecs, "_ti_numElements", 0);
				for( SubArray sArray : ftdata ) {
					Symbol ftSym = AnalysisTools.subarrayToSymbol(sArray, IRSymbolOnly);
					String symNameBase = null;
					if( ftSym instanceof AccessSymbol) {
						symNameBase = TransformTools.buildAccessSymbolName((AccessSymbol)ftSym);
					} else {
						symNameBase = ftSym.getSymbolName();
					}
					List<Expression> startList = new LinkedList<Expression>();
					List<Expression> lengthList = new LinkedList<Expression>();
					boolean foundDimensions = AnalysisTools.extractDimensionInfo(sArray, startList, lengthList, IRSymbolOnly, resAnnot.getAnnotatable());
					if( !foundDimensions && (compAnnot != null) ) {
						SubArray tSArray = AnalysisTools.findSubArrayInDataClauses(compAnnot, ftSym, IRSymbolOnly);
						if( tSArray != null ) {
							startList.clear();
							lengthList.clear();
							foundDimensions = AnalysisTools.extractDimensionInfo(tSArray, startList, lengthList, IRSymbolOnly, compAnnot.getAnnotatable());
						}
					}
					if( !foundDimensions ) {
						Tools.exit("[ERROR in FaultInjectionTransformaion()] Can not find dimension information of the following symbol:\n" +
								"Symbol: " + ftSym.getSymbolName() + "\nACAnnotation: " + resAnnot + 
								"\nEnclosing Procedure: " + cProc.getSymbolName() + "\n");
					}
					//Set random value for epos for each ftregion and ftinject directive.
					//e.g., epos__a_FR0 = HI_genrandom_int(1024);
					Map<String, Symbol> teposMap = eposMap.get(ftSym);
					Expression biexp = null;
					if( lengthList.size() > 0 ) {//ftSym is array variable.
						biexp = lengthList.get(0).clone();
						for( int i=1; i<lengthList.size(); i++ )
						{
							biexp = new BinaryExpression(biexp, BinaryOperator.MULTIPLY, lengthList.get(i).clone());
						}
					}
					if( biexp == null ) {
						biexp = new IntegerLiteral(1);
					}
					ExpressionStatement eStmt = new ExpressionStatement(
							new AssignmentExpression(arraySizeID.clone(), AssignmentOperator.NORMAL,biexp.clone()));
					cStmt.addStatementBefore(resRegion, eStmt);
					FunctionCall fCall = new FunctionCall(new NameID("HI_genrandom_int"));
					fCall.addArgument(arraySizeID.clone());
					for( Symbol eposSym : teposMap.values() ) {
						AssignmentExpression assignExp = new AssignmentExpression(new Identifier(eposSym), AssignmentOperator.NORMAL,
								fCall.clone());
						eStmt = new ExpressionStatement(assignExp);
						cStmt.addStatementBefore(resRegion, eStmt);
					}
					//Set random value for ipos
					//e.g., ipose__a_FR = HI_genrandom_int(# of statements in a ftregion);
					Map<String, Symbol> tiposMap = iposMap.get(ftSym);
					for( ARCAnnotation ftRAnnot : ftregionAnnots ) {
						Statement ftregion = (Statement)ftRAnnot.getAnnotatable();
						List<Statement> childList = findChildStatements(ftregion, true);
						int num_stmts = childList.size();
						num_stmts += 1; //Fault may be injected right after the last child statement.
						IntegerLiteral tInt = ftRAnnot.get("ftregion");
						String iposName = "ipos__" + symNameBase + "_FR" + tInt.toString();
						Symbol iposSym = tiposMap.get(iposName);
						if( iposSym != null ) {
							fCall = new FunctionCall(new NameID("HI_genrandom_int"));
							fCall.addArgument(new IntegerLiteral(num_stmts));
							AssignmentExpression assignExp = new AssignmentExpression(new Identifier(iposSym), AssignmentOperator.NORMAL,
									fCall.clone());
							eStmt = new ExpressionStatement(assignExp);
							cStmt.addStatementBefore(resRegion, eStmt);
						}
					}
					//Set random bitvector for each ftregion and ftinject directive.
					Map<String, Symbol> tbitVecMap = bitVecMap.get(ftSym);
					Specifier bitvType = AnalysisTools.getBitVecType(ftSym);
					if( bitvType == null ) {
						Tools.exit("[ERROR in FaultInjectionTransformation] cannot find corresponding bit-vector type " +
								"for the following symbol: " + ftSym + "\n");
					}
					if( bitvType.toString().equals("type8b") ) {
						fCall = new FunctionCall(new NameID("HI_genbitvector8b"));
					} else if( bitvType.toString().equals("type16b") ) {
						fCall = new FunctionCall(new NameID("HI_genbitvector16b"));
					} else if( bitvType.toString().equals("type32b") ) {
						fCall = new FunctionCall(new NameID("HI_genbitvector32b"));
					} else if( bitvType.toString().equals("type64b") ) {
						fCall = new FunctionCall(new NameID("HI_genbitvector64b"));
					}
					fCall.addArgument(num_ftbits.clone());
					for( Symbol bitvSym : tbitVecMap.values() ) {
						AssignmentExpression assignExp = new AssignmentExpression(new Identifier(bitvSym), 
								AssignmentOperator.NORMAL, fCall.clone());
						eStmt = new ExpressionStatement(assignExp);
						cStmt.addStatementBefore(resRegion, eStmt);
					}
				}
				//Step6: - handle ftcond/num_faults clauses.
				//Create a temp variable to keep the condition
				Identifier condID = null;
				Expression num_faults = resAnnot.get("num_faults");
				if( (ftcond != null) || (num_faults != null) ) {
					tSpecs = new ArrayList<Specifier>(1);
					tSpecs.add(Specifier.INT);
					condID = TransformTools.getTempScalar(cBody, tSpecs, "_ti_ftcond" + resR_cnt, 0);
					resAnnot.put("tftcondsym", condID.getSymbol());
				}
				if( num_faults != null ) {
	//				Expression biExp = new BinaryExpression(new UnaryExpression(UnaryOperator.POST_INCREMENT, ftCntID.clone()), 
	//						BinaryOperator.COMPARE_LT, num_faults.clone());
					Expression biExp = new BinaryExpression(ftCntID.clone(), 
							BinaryOperator.COMPARE_LT, num_faults.clone());
					if( ftcond == null ) {
						tftcondExp = biExp;
					} else {
						tftcondExp = new BinaryExpression(ftcond.clone(), BinaryOperator.LOGICAL_AND,
								biExp);
					}
				} else {
					Statement eStmt = new ExpressionStatement(new UnaryExpression(UnaryOperator.POST_INCREMENT, 
							ftCntID.clone()));
					cStmt.addStatementBefore(resRegion, eStmt);
				}
				if( (tftcondExp != null) && (condID != null) ) {
					//Expression assignExp = new AssignmentExpression(condID.clone(), AssignmentOperator.NORMAL,
					//		tftcondExp);
					Expression assignExp = new AssignmentExpression(condID.clone(), AssignmentOperator.NORMAL,
							new IntegerLiteral(1));
					Statement eStmt = new ExpressionStatement(assignExp);
					CompoundStatement confIfBody = new CompoundStatement();
					confIfBody.addStatement(eStmt);
					assignExp = new AssignmentExpression(ftCntID.clone(), AssignmentOperator.ADD,
							new IntegerLiteral(1));
					eStmt = new ExpressionStatement(assignExp);
					confIfBody.addStatement(eStmt);
					CompoundStatement confElseBody = new CompoundStatement();
					assignExp = new AssignmentExpression(condID.clone(), AssignmentOperator.NORMAL,
							new IntegerLiteral(0));
					eStmt = new ExpressionStatement(assignExp);
					confElseBody.addStatement(eStmt);
					IfStatement condIfStmt = new IfStatement(tftcondExp, confIfBody, confElseBody);
					cStmt.addStatementBefore(resRegion, condIfStmt);
				}
				resR_cnt++;
				//Step7: - for each ftregion/ftinject,
				//           - Generate fault-injection code for each symbol in ftdata
				//[FIXME] this assumes that each ftregion does not contain any compute regions inside; if so, 
				//below transformation is incorrect.
				for( ARCAnnotation ftRAnnot : ftregionAnnots ) {
					Statement ftregion = (Statement)ftRAnnot.getAnnotatable();
					Expression ftthreadExp = ftRAnnot.get("ftthread");
					boolean inCompRegion = false;
					if( ftregion.containsAnnotation(ACCAnnotation.class, "kernels") ) {
						inCompRegion = true;
					} else if( ftregion.containsAnnotation(ACCAnnotation.class, "parallel") ) {
						inCompRegion = true;
					} else {
						ACCAnnotation tcompAnnot = AnalysisTools.ipFindFirstPragmaInParent(ftregion, ACCAnnotation.class, 
								ACCAnnotation.computeRegions, false, funcCallList, null);
						if( tcompAnnot != null ) {
							inCompRegion = true;
						}
					}
					String fcall_prefix = "";
					if( inCompRegion ) {
						fcall_prefix = "dev__";
					}
					CompoundStatement ftParent = (CompoundStatement)ftregion.getParent();
					List<Statement> childList = findChildStatements(ftregion, true);
					int num_stmts = childList.size();
					num_stmts += 1; //Fault may be injected right after the last child statement.
					Set<Symbol> l_tftsymbols = ftRAnnot.get("tftsymbols");
					if( l_tftsymbols != null ) {
						for( Symbol ftSym : l_tftsymbols ) {
							Map<String, Symbol> tiposMap = iposMap.get(ftSym);
							Map<String, Symbol> teposMap = eposMap.get(ftSym);
							Map<String, Symbol> tbitVecMap = bitVecMap.get(ftSym);
							Specifier bitvType = AnalysisTools.getBitVecType(ftSym);
							Expression pointerExp = null;
							if( SymbolTools.isScalar(ftSym) && !SymbolTools.isPointer(ftSym) ) {
								pointerExp = new UnaryExpression(UnaryOperator.ADDRESS_OF, new Identifier(ftSym));
							} else {
								pointerExp = new Identifier(ftSym);
							}
							IntegerLiteral tInt = ftRAnnot.get("ftregion");
							String symNameBase = null;
							if( ftSym instanceof AccessSymbol) {
								symNameBase = TransformTools.buildAccessSymbolName((AccessSymbol)ftSym);
							} else {
								symNameBase = ftSym.getSymbolName();
							}
							String iposName = "ipos__" + symNameBase + "_FR" + tInt.toString();
							Symbol iposSym = tiposMap.get(iposName);
							String eposName = "epos__" + symNameBase + "_FR" + tInt.toString();
							Symbol eposSym = teposMap.get(eposName);
							String bitvName = "bitv__" + symNameBase + "_FR" + tInt.toString();
							Symbol bitvSym = tbitVecMap.get(bitvName);
							FunctionCall fCall = null;
							int typeKind = 0;
							if( bitvType.toString().equals("type8b") ) {
								fCall = new FunctionCall(new NameID(fcall_prefix + ftinjectionCallBaseName+"_int8b"));
							} else if( bitvType.toString().equals("type16b") ) {
								fCall = new FunctionCall(new NameID(fcall_prefix + ftinjectionCallBaseName+"_int16b"));
							} else if( bitvType.toString().equals("type32b") ) {
								if( ftSym.getTypeSpecifiers().contains(Specifier.FLOAT) ) {
									fCall = new FunctionCall(new NameID(fcall_prefix + ftinjectionCallBaseName+"_float"));
									typeKind = 1;
								} else {
									fCall = new FunctionCall(new NameID(fcall_prefix + ftinjectionCallBaseName+"_int32b"));
								}
							} else if( bitvType.toString().equals("type64b") ) {
								if( ftSym.getTypeSpecifiers().contains(Specifier.DOUBLE) ) {
									fCall = new FunctionCall(new NameID(fcall_prefix + ftinjectionCallBaseName+"_double"));
									typeKind = 2;
								} else {
									fCall = new FunctionCall(new NameID(fcall_prefix + ftinjectionCallBaseName+"_int64b"));
								}
							}
							if( typeKind == 0 ) {
								List<Specifier> types = new ArrayList<Specifier>(2);
								types.add(bitvType);
								//[FIXME] if the target is OpenCL (OPENARC_ARCH > 0), OpenCL address space qualifier (__global,
								//__constant, or __local) should be added too. However, in current implementation, only 
								//__global is used, which may be changed later.
								if( inCompRegion && (OPENARC_ARCH > 0) && (OPENARC_ARCH != 5) ) {
									types.add(OpenCLSpecifier.OPENCL_GLOBAL);
								}
								types.add(PointerSpecifier.UNQUALIFIED);
								fCall.addArgument(new Typecast(types, pointerExp));
							} else if( typeKind == 1 ) {
								List<Specifier> types = new ArrayList<Specifier>(2);
								types.add(Specifier.FLOAT);
								if( inCompRegion && (OPENARC_ARCH > 0) && (OPENARC_ARCH != 5) ) {
									types.add(OpenCLSpecifier.OPENCL_GLOBAL);
								}
								types.add(PointerSpecifier.UNQUALIFIED);
								fCall.addArgument(new Typecast(types, pointerExp));
							} else {
								List<Specifier> types = new ArrayList<Specifier>(2);
								types.add(Specifier.DOUBLE);
								if( inCompRegion && (OPENARC_ARCH > 0) && (OPENARC_ARCH != 5) ) {
									types.add(OpenCLSpecifier.OPENCL_GLOBAL);
								}
								types.add(PointerSpecifier.UNQUALIFIED);
								fCall.addArgument(new Typecast(types, pointerExp));
							}
							for( int i=0; i<num_stmts; i++ ) {
								Statement cChild = null;
								if( i == num_stmts-1 ) {
									cChild = childList.get(i-1);
								} else {
									cChild = childList.get(i);
								}
								CompoundStatement pStmt = (CompoundStatement)cChild.getParent();
								CompoundStatement ifBody = new CompoundStatement();
								Expression tCondExp = new BinaryExpression(new Identifier(iposSym), 
										BinaryOperator.COMPARE_EQ, new IntegerLiteral(i));
								if( condID != null ) {
									tCondExp = new BinaryExpression(condID.clone(), BinaryOperator.LOGICAL_AND,
											tCondExp);
								}
								if( inCompRegion && (targetThread != null) ) {
									if( ftthreadExp == null ) {
										tCondExp = new BinaryExpression(new BinaryExpression(new NameID("_gtid"), 
												BinaryOperator.COMPARE_EQ, targetThread.clone()),
												BinaryOperator.LOGICAL_AND, tCondExp.clone());
									} else {
										tCondExp = new BinaryExpression(new BinaryExpression(new NameID("_gtid"), 
												BinaryOperator.COMPARE_EQ, ftthreadExp.clone()),
												BinaryOperator.LOGICAL_AND, tCondExp.clone());
									}
								}
								IfStatement ifStmt = new IfStatement(tCondExp, ifBody);
								FunctionCall lfCall = fCall.clone();
								lfCall.addArgument(new IntegerLiteral(1));
								lfCall.addArgument(new Identifier(eposSym));
								lfCall.addArgument(new Identifier(bitvSym));
								Statement eStmt = new ExpressionStatement(lfCall);
								ifBody.addStatement(eStmt);
								eStmt = new ExpressionStatement(new AssignmentExpression(new Identifier(iposSym), 
										AssignmentOperator.NORMAL, new IntegerLiteral(-1)));
								ifBody.addStatement(eStmt);
								//[DEBUG] ft-inject statement will be inserted before a target statement.
								//except for the last target statement, where ft-inject statement will be inserted
								//both before and after the target statement.
								if( i == num_stmts-1 ) {
									pStmt.addStatementAfter(cChild, ifStmt);
								} else {
									pStmt.addStatementBefore(cChild, ifStmt);
								}
							}
						}
					}
				}
				for( ARCAnnotation ftRAnnot : ftinjectAnnots ) {
					Annotatable ftinject = ftRAnnot.getAnnotatable();
					Expression ftthreadExp = ftRAnnot.get("ftthread");
					boolean inCompRegion = false;
					if( ftinject.containsAnnotation(ACCAnnotation.class, "kernels") ) {
						inCompRegion = true;
					} else if( ftinject.containsAnnotation(ACCAnnotation.class, "parallel") ) {
						inCompRegion = true;
					} else {
						ACCAnnotation tcompAnnot = AnalysisTools.ipFindFirstPragmaInParent(ftinject, ACCAnnotation.class, 
								ACCAnnotation.computeRegions, false, funcCallList, null);
						if( tcompAnnot != null ) {
							inCompRegion = true;
						}
					}
					String fcall_prefix = "";
					if( inCompRegion ) {
						fcall_prefix = "dev__";
					}
					CompoundStatement ftParent = (CompoundStatement)ftinject.getParent();
					Set<Symbol> l_tftsymbols = ftRAnnot.get("tftsymbols");
					if( l_tftsymbols != null ) {
						for( Symbol ftSym : l_tftsymbols ) {
							Map<String, Symbol> teposMap = eposMap.get(ftSym);
							Map<String, Symbol> tbitVecMap = bitVecMap.get(ftSym);
							Specifier bitvType = AnalysisTools.getBitVecType(ftSym);
							Expression pointerExp = null;
							if( SymbolTools.isScalar(ftSym) && !SymbolTools.isPointer(ftSym) ) {
								pointerExp = new UnaryExpression(UnaryOperator.ADDRESS_OF, new Identifier(ftSym));
							} else {
								pointerExp = new Identifier(ftSym);
							}
							IntegerLiteral tInt = ftRAnnot.get("ftinject");
							String symNameBase = null;
							if( ftSym instanceof AccessSymbol) {
								symNameBase = TransformTools.buildAccessSymbolName((AccessSymbol)ftSym);
							} else {
								symNameBase = ftSym.getSymbolName();
							}
							String eposName = "epos__" + symNameBase + "_FI" + tInt.toString();
							Symbol eposSym = teposMap.get(eposName);
							String bitvName = "bitv__" + symNameBase + "_FI" + tInt.toString();
							Symbol bitvSym = tbitVecMap.get(bitvName);
							FunctionCall fCall = null;
							int typeKind = 0;
							if( bitvType.toString().equals("type8b") ) {
								fCall = new FunctionCall(new NameID(fcall_prefix + ftinjectionCallBaseName+"_int8b"));
							} else if( bitvType.toString().equals("type16b") ) {
								fCall = new FunctionCall(new NameID(fcall_prefix + ftinjectionCallBaseName+"_int16b"));
							} else if( bitvType.toString().equals("type32b") ) {
								if( ftSym.getTypeSpecifiers().contains(Specifier.FLOAT) ) {
									fCall = new FunctionCall(new NameID(fcall_prefix + ftinjectionCallBaseName+"_float"));
									typeKind = 1;
								} else {
									fCall = new FunctionCall(new NameID(fcall_prefix + ftinjectionCallBaseName+"_int32b"));
								}
							} else if( bitvType.toString().equals("type64b") ) {
								if( ftSym.getTypeSpecifiers().contains(Specifier.DOUBLE) ) {
									fCall = new FunctionCall(new NameID(fcall_prefix + ftinjectionCallBaseName+"_double"));
									typeKind = 2;
								} else {
									fCall = new FunctionCall(new NameID(fcall_prefix + ftinjectionCallBaseName+"_int64b"));
								}
							}
							if( typeKind == 0 ) {
								List<Specifier> types = new ArrayList<Specifier>(2);
								types.add(bitvType);
								if( inCompRegion && (OPENARC_ARCH > 0) && (OPENARC_ARCH != 5) ) {
									types.add(OpenCLSpecifier.OPENCL_GLOBAL);
								}
								types.add(PointerSpecifier.UNQUALIFIED);
								fCall.addArgument(new Typecast(types, pointerExp));
							} else if( typeKind == 1 ) {
								List<Specifier> types = new ArrayList<Specifier>(2);
								types.add(Specifier.FLOAT);
								if( inCompRegion && (OPENARC_ARCH > 0) && (OPENARC_ARCH != 5) ) {
									types.add(OpenCLSpecifier.OPENCL_GLOBAL);
								}
								types.add(PointerSpecifier.UNQUALIFIED);
								fCall.addArgument(new Typecast(types, pointerExp));
							} else {
								List<Specifier> types = new ArrayList<Specifier>(2);
								types.add(Specifier.DOUBLE);
								if( inCompRegion && (OPENARC_ARCH > 0) && (OPENARC_ARCH != 5) ) {
									types.add(OpenCLSpecifier.OPENCL_GLOBAL);
								}
								types.add(PointerSpecifier.UNQUALIFIED);
								fCall.addArgument(new Typecast(types, pointerExp));
							}
							Expression tCondExp = null;
							if( inCompRegion && (targetThread != null) ) {
								if( condID != null ) {
									if( ftthreadExp == null ) {
										tCondExp = new BinaryExpression(new BinaryExpression(new NameID("_gtid"), 
												BinaryOperator.COMPARE_EQ, targetThread.clone()),
												BinaryOperator.LOGICAL_AND, condID.clone());
									} else {
										tCondExp = new BinaryExpression(new BinaryExpression(new NameID("_gtid"), 
												BinaryOperator.COMPARE_EQ, ftthreadExp.clone()),
												BinaryOperator.LOGICAL_AND, condID.clone());
									}
								} else {
									//DEBUG: Is this correct?
									tCondExp = new BinaryExpression(new NameID("_gtid"), 
											BinaryOperator.COMPARE_EQ, new NameID("_ti_targetThread"));
									if( ftthreadExp == null ) {
										tCondExp = new BinaryExpression(new NameID("_gtid"), 
												BinaryOperator.COMPARE_EQ,targetThread.clone());
									} else {
										tCondExp = new BinaryExpression(new NameID("_gtid"), 
												BinaryOperator.COMPARE_EQ, ftthreadExp.clone());
									}
								}
								fCall.addArgument(new IntegerLiteral(1));
								fCall.addArgument(new Identifier(eposSym));
								fCall.addArgument(new Identifier(bitvSym));
								Statement eStmt = new ExpressionStatement(fCall);
								IfStatement ifStmt = new IfStatement(tCondExp, eStmt);
								ftParent.addStatementAfter((Statement)ftinject, ifStmt);

							} else {
								if( condID != null ) {
									tCondExp = condID.clone();
								} else {
									tCondExp = new IntegerLiteral(1);
								}
								fCall.addArgument(tCondExp);
								fCall.addArgument(new Identifier(eposSym));
								fCall.addArgument(new Identifier(bitvSym));
								Statement eStmt = new ExpressionStatement(fCall);
								ftParent.addStatementAfter((Statement)ftinject, eStmt);
							}
						}
					}
				}
				//Step8: - for each compute region enclosed in the resilience region
				//           - add temp varaibles (ipos, epos, bitvector) for each symbol in tftsymbols set to copyin, 
				//           sharedRO, accshared, and accreadonly clauses.
				//           - Assign random value to targetThread for each compute region and add to the 
				//           copyin set of the compute region.
				if( compAnnot != null ) {
					//current resilience region is a compute region.
					Set<Symbol> l_tftsymbols = resAnnot.get("tftsymbols");
					if( l_tftsymbols != null ) {
						Set<SubArray> copyinSet = compAnnot.get("copyin");
						if( copyinSet == null ) {
							copyinSet = new HashSet<SubArray>();
							compAnnot.put("copyin", copyinSet);
						}
						Set<SubArray> sharedROSet = null;
						ARCAnnotation cudaAnnot = resRegion.getAnnotation(ARCAnnotation.class, "sharedRO");
						if( cudaAnnot == null ) {
							cudaAnnot = resRegion.getAnnotation(ARCAnnotation.class, "cuda");
							if( cudaAnnot == null ) {
								cudaAnnot = new ARCAnnotation("cuda", "_directive");
								resRegion.annotate(cudaAnnot);
							}
							sharedROSet = new HashSet<SubArray>();
							cudaAnnot.put("sharedRO", sharedROSet);
						} else {
							sharedROSet = cudaAnnot.get("sharedRO");
						}
						Set<Symbol> accsharedSet = null;
						ACCAnnotation iAnnot = resRegion.getAnnotation(ACCAnnotation.class, "accshared");
						if( iAnnot == null ) {
							iAnnot = resRegion.getAnnotation(ACCAnnotation.class, "internal");
							if( iAnnot == null ) {
								iAnnot = new ACCAnnotation("internal", "_directive");
								resRegion.annotate(iAnnot);
							}
							accsharedSet = new HashSet<Symbol>();
							iAnnot.put("accshared", accsharedSet);
						} else {
							accsharedSet = iAnnot.get("accshared");
						}
						Set<Symbol> accreadonlySet = null;
						iAnnot = resRegion.getAnnotation(ACCAnnotation.class, "accreadonly");
						if( iAnnot == null ) {
							iAnnot = resRegion.getAnnotation(ACCAnnotation.class, "internal");
							if( iAnnot == null ) {
								iAnnot = new ACCAnnotation("internal", "_directive");
								resRegion.annotate(iAnnot);
							}
							accreadonlySet = new HashSet<Symbol>();
							iAnnot.put("accreadonly", accreadonlySet);
						} else {
							accreadonlySet = iAnnot.get("accreadonly");
						}
						for( Symbol ftSym : l_tftsymbols ) {
							Map<String, Symbol> tiposMap = iposMap.get(ftSym);
							Map<String, Symbol> teposMap = eposMap.get(ftSym);
							Map<String, Symbol> tbitVecMap = bitVecMap.get(ftSym);
							Set<Symbol> newSymSet = new HashSet<Symbol>();
							newSymSet.addAll(tiposMap.values());
							newSymSet.addAll(teposMap.values());
							newSymSet.addAll(tbitVecMap.values());
							for( Symbol inSym : newSymSet ) {
								SubArray tSArr = AnalysisTools.createSubArray(inSym, true, null);
								if( tSArr != null ) {
									copyinSet.add(tSArr);
									sharedROSet.add(tSArr.clone());
									accsharedSet.add(inSym);
									accreadonlySet.add(inSym);
								} else {
									Tools.exit("[ERROR in FaultInjectionTransformation() can not create a subarray of symbol, "
											+ inSym.getSymbolName() + "\nOpenACC Annotation: " + resAnnot + 
											"\nEnclosing procedure: " + cProc.getSymbolName());
								}
							}
						}
						if( condID != null ) {
							Symbol condSym = condID.getSymbol();
							SubArray tSArr = AnalysisTools.createSubArray(condSym, true, null);
							if( tSArr != null ) {
								copyinSet.add(tSArr);
								sharedROSet.add(tSArr.clone());
								accsharedSet.add(condSym);
								accreadonlySet.add(condSym);
							} else {
								Tools.exit("[ERROR in FaultInjectionTransformation() can not create a subarray of symbol, "
										+ condSym.getSymbolName() + "\nOpenACC Annotation: " + resAnnot + 
										"\nEnclosing procedure: " + cProc.getSymbolName());
							}
						}
						if( targetThread != null ) {
							Symbol tThreadSym = targetThread.getSymbol();
							SubArray tSArr = AnalysisTools.createSubArray(tThreadSym, true, null);
							if( tSArr != null ) {
								copyinSet.add(tSArr);
								sharedROSet.add(tSArr.clone());
								accsharedSet.add(tThreadSym);
								accreadonlySet.add(tThreadSym);
							} else {
								Tools.exit("[ERROR in FaultInjectionTransformation() can not create a subarray of symbol, "
										+ tThreadSym.getSymbolName() + "\nOpenACC Annotation: " + resAnnot + 
										"\nEnclosing procedure: " + cProc.getSymbolName());
							}
							//Store the target thread symbol to be used later by ACC2CUDATranslator
							compAnnot.put("targetThread", tThreadSym);
						}
					}
				} else {
					List<ACCAnnotation> compAnnots = AnalysisTools.ipCollectPragmas(resRegion, ACCAnnotation.class, 
							ACCAnnotation.computeRegions, false, null);
					if( compAnnots != null ) {
						for( ACCAnnotation cAnnot : compAnnots ) {
							Annotatable cRegion = cAnnot.getAnnotatable();
							List<ARCAnnotation> ftAnnots = AnalysisTools.ipCollectPragmas(cRegion, ARCAnnotation.class, 
									ARCAnnotation.ftinjecttargets, false, null);
							if( ftAnnots == null ) {
								continue;
							} else {
								Set<SubArray> copyinSet = cAnnot.get("copyin");
								if( copyinSet == null ) {
									copyinSet = new HashSet<SubArray>();
									cAnnot.put("copyin", copyinSet);
								}
								Set<SubArray> sharedROSet = null;
								ARCAnnotation cudaAnnot = cRegion.getAnnotation(ARCAnnotation.class, "sharedRO");
								if( cudaAnnot == null ) {
									cudaAnnot = cRegion.getAnnotation(ARCAnnotation.class, "cuda");
									if( cudaAnnot == null ) {
										cudaAnnot = new ARCAnnotation("cuda", "_directive");
										cRegion.annotate(cudaAnnot);
									}
									sharedROSet = new HashSet<SubArray>();
									cudaAnnot.put("sharedRO", sharedROSet);
								} else {
									sharedROSet = cudaAnnot.get("sharedRO");
								}
								Set<Symbol> accsharedSet = null;
								ACCAnnotation iAnnot = cRegion.getAnnotation(ACCAnnotation.class, "accshared");
								if( iAnnot == null ) {
									iAnnot = cRegion.getAnnotation(ACCAnnotation.class, "internal");
									if( iAnnot == null ) {
										iAnnot = new ACCAnnotation("internal", "_directive");
										cRegion.annotate(iAnnot);
									}
									accsharedSet = new HashSet<Symbol>();
									iAnnot.put("accshared", accsharedSet);
								} else {
									accsharedSet = iAnnot.get("accshared");
								}
								Set<Symbol> accreadonlySet = null;
								iAnnot = cRegion.getAnnotation(ACCAnnotation.class, "accreadonly");
								if( iAnnot == null ) {
									iAnnot = cRegion.getAnnotation(ACCAnnotation.class, "internal");
									if( iAnnot == null ) {
										iAnnot = new ACCAnnotation("internal", "_directive");
										cRegion.annotate(iAnnot);
									}
									accreadonlySet = new HashSet<Symbol>();
									iAnnot.put("accreadonly", accreadonlySet);
								} else {
									accreadonlySet = iAnnot.get("accreadonly");
								}
								if( condID != null ) {
									Symbol condSym = condID.getSymbol();
									SubArray tSArr = AnalysisTools.createSubArray(condSym, true, null);
									if( tSArr != null ) {
										copyinSet.add(tSArr);
										sharedROSet.add(tSArr.clone());
										accsharedSet.add(condSym);
										accreadonlySet.add(condSym);
									} else {
										Procedure tProc = IRTools.getParentProcedure(cRegion);
										Tools.exit("[ERROR in FaultInjectionTransformation() can not create a subarray of symbol, "
												+ condSym.getSymbolName() + "\nOpenACC Annotation: " + cAnnot + 
												"\nEnclosing procedure: " + tProc.getSymbolName());
									}
								}
								if( targetThread != null ) {
									Symbol tThreadSym = targetThread.getSymbol();
									SubArray tSArr = AnalysisTools.createSubArray(tThreadSym, true, null);
									if( tSArr != null ) {
										copyinSet.add(tSArr);
										sharedROSet.add(tSArr.clone());
										accsharedSet.add(tThreadSym);
										accreadonlySet.add(tThreadSym);
									} else {
										Procedure tProc = IRTools.getParentProcedure(cRegion);
										Tools.exit("[ERROR in FaultInjectionTransformation() can not create a subarray of symbol, "
												+ tThreadSym.getSymbolName() + "\nOpenACC Annotation: " + cAnnot + 
												"\nEnclosing procedure: " + tProc.getSymbolName());
									}
									//Store the target thread symbol to be used later by ACC2CUDATranslator
									cAnnot.put("targetThread", tThreadSym);
								}
								for( ARCAnnotation ftAnn : ftAnnots ) {
									int ftRType = 0;
									if( ftAnn.containsKey("ftinject") ) {
										ftRType = 1;
									}
									Set<Symbol> l_tftsymbols = ftAnn.get("tftsymbols");
									if( l_tftsymbols == null ) {
										continue;
									} else {
										for( Symbol ftSym : l_tftsymbols ) {
											Map<String, Symbol> tiposMap = iposMap.get(ftSym);
											Map<String, Symbol> teposMap = eposMap.get(ftSym);
											Map<String, Symbol> tbitVecMap = bitVecMap.get(ftSym);
											IntegerLiteral tInt = null;
											String suffix = null;
											if( ftRType == 0 ) {
												tInt = ftAnn.get("ftregion");
												suffix = "_FR";
											} else {
												tInt = ftAnn.get("ftinject");
												suffix = "_FI";
											}
											String symNameBase = null;
											if( ftSym instanceof AccessSymbol) {
												symNameBase = TransformTools.buildAccessSymbolName((AccessSymbol)ftSym);
											} else {
												symNameBase = ftSym.getSymbolName();
											}
											Set<Symbol> newSymSet = new HashSet<Symbol>();
											String iposName = "ipos__" + symNameBase + suffix + tInt.toString();
											Symbol iposSym = tiposMap.get(iposName);
											if( iposSym != null ) {
												newSymSet.add(iposSym);
											}
											String eposName = "epos__" + symNameBase + suffix + tInt.toString();
											Symbol eposSym = teposMap.get(eposName);
											if( eposSym != null ) {
												newSymSet.add(eposSym);
											}
											String bitvName = "bitv__" + symNameBase + suffix + tInt.toString();
											Symbol bitvSym = tbitVecMap.get(bitvName);
											if( bitvSym != null ) {
												newSymSet.add(bitvSym);
											}
											for( Symbol inSym : newSymSet ) {
												SubArray tSArr = AnalysisTools.createSubArray(inSym, true, null);
												if( tSArr != null ) {
													if( !accsharedSet.contains(inSym)) {
														copyinSet.add(tSArr);
														sharedROSet.add(tSArr.clone());
														accsharedSet.add(inSym);
														accreadonlySet.add(inSym);
													}
												} else {
													Procedure tProc = IRTools.getParentProcedure(cRegion);
													Tools.exit("[ERROR in FaultInjectionTransformation() can not create a subarray of symbol, "
															+ inSym.getSymbolName() + "\nOpenACC Annotation: " + ftAnn + 
															"\nEnclosing procedure: " + tProc.getSymbolName());
												}
											}
										}
									}
								}
							}
						}
					}
				}
				//Step9: - Remove temp clauses (tftsymbols, tresilience, tiposmap, teposmap, tbitvecmap, tftcondsym)
				List<ARCAnnotation> ftAnnotList = new ArrayList<ARCAnnotation>();
				ftAnnotList.add(resAnnot);
				ftAnnotList.addAll(ftregionAnnots);
				ftAnnotList.addAll(ftinjectAnnots);
				for( ARCAnnotation tAnnot : ftAnnotList ) {
					tAnnot.remove("tftsymbols");
					tAnnot.remove("tresilience");
					tAnnot.remove("tiposmap");
					tAnnot.remove("teposmap");
					tAnnot.remove("tbitvecmap");
					tAnnot.remove("tftcondsym");
				}
			} //end of resilienceAnnots loop
		}
	}
	
	private void fillFTData(ARCAnnotation resAnnot) {
		Annotatable tRegion = resAnnot.getAnnotatable();
		Annotatable targetRegion = null;
		if( tRegion.containsAnnotation(ARCAnnotation.class, "ftregion") ) {
			targetRegion = tRegion;
		} else if( tRegion.containsAnnotation(ARCAnnotation.class, "ftinject") ) {
			targetRegion = (Annotatable)tRegion.getParent();
		}
		Set<SubArray> tftdata = resAnnot.get("ftdata");
		if( tftdata == null ) {
			tftdata = new HashSet<SubArray>();
			resAnnot.put("ftdata", tftdata);
		}
		Set<Symbol> ttftsymbols = resAnnot.get("tftsymbols");
		if( ttftsymbols == null ) {
			ttftsymbols = new HashSet<Symbol>();
			resAnnot.put("tftsymbols", ttftsymbols);
		}

		//Find shared symbols interprocedurally, and put them in the tftsymbols set.
		// tftsymbols set = symbols accessed in the region - local symbols 
		//                 + global symbols accessed in functions called in the region 
		//Find symbols accessed in the region, and add them to accshared set.
		Set<Symbol> tempSet = AnalysisTools.getAccessedVariables(targetRegion, IRSymbolOnly);
		if( tempSet != null ) {
			//ttftsymbols.addAll(tempSet);
			for( Symbol ttSym : tempSet ) {
				String ttSymName = ttSym.getSymbolName();
				if( !ttSymName.startsWith("ipos__") && !ttSymName.startsWith("epos__") &&
						!ttSymName.startsWith("bitv__") && !ttSymName.startsWith("_ti_ftcond") &&
						!ttSymName.startsWith("_ti_targetThread") ) {
					ttftsymbols.add(ttSym);
				}
			}
		}
		//Find local symbols defined in the region, and remove them from the accshared set.
		tempSet = SymbolTools.getLocalSymbols(targetRegion);
		if( tempSet != null ) {
			ttftsymbols.removeAll(tempSet);
		}
		//Find global symbols accessed in the functions called in the region, and add them 
		//to the accshared set.
		Map<String, Symbol> gSymMap = null;
		List<FunctionCall> calledFuncs = IRTools.getFunctionCalls(targetRegion);
		for( FunctionCall call : calledFuncs ) {
			Procedure called_procedure = call.getProcedure();
			if( called_procedure != null ) {
				if( gSymMap == null ) {
					Set<Symbol> tSet = SymbolTools.getGlobalSymbols(targetRegion);
					gSymMap = new HashMap<String, Symbol>();
					for( Symbol gS : tSet ) {
						gSymMap.put(gS.getSymbolName(), gS);
					}
				} 
				CompoundStatement body = called_procedure.getBody();
				Set<Symbol> procAccessedSymbols = AnalysisTools.getIpAccessedGlobalSymbols(body, gSymMap, null);
				if( procAccessedSymbols != null ) {
					//ttftsymbols.addAll(procAccessedSymbols);
					for( Symbol ttSym : procAccessedSymbols ) {
						String ttSymName = ttSym.getSymbolName();
						if( !ttSymName.startsWith("ipos__") && !ttSymName.startsWith("epos__") &&
								!ttSymName.startsWith("bitv__") && !ttSymName.startsWith("_ti_ftcond") &&
								!ttSymName.startsWith("_ti_targetThread") ) {
							ttftsymbols.add(ttSym);
						}
					}
				}
			}
		}
		for( Symbol inSym : ttftsymbols ) {
			SubArray tSArr = AnalysisTools.createSubArray(inSym, false, null);
			if( tSArr == null ) {
				Procedure cProc = IRTools.getParentProcedure(tRegion);
				PrintTools.println("\n[WARNING in FaultInjectionTransformation()] can not create " +
						"a subarray for following symbol: \n" +
						"Symbol: " + inSym.getSymbolName() + "\nACCAnnotation: " + resAnnot + 
						"\nEnclosing Procedure: " + cProc.getSymbolName() + "\n", 0);
			} else {
				tftdata.add(tSArr);
			}
		}
	}
	
	private List<Statement> findChildStatements(Statement inStmt, boolean isRoot) {
		List<Statement> childList = new  ArrayList<Statement>();
		if( inStmt != null ) {
			Traversable parent = inStmt.getParent();
			if( inStmt instanceof CompoundStatement ) {
				if( isRoot || (!inStmt.containsAnnotation(ARCAnnotation.class, "ftregion"))) {
					for( Traversable t : inStmt.getChildren() ) {
						childList.addAll(findChildStatements((Statement)t, false));
					}
				} else if( !(parent instanceof Loop) && !(parent instanceof IfStatement)) {
					childList.add(inStmt);
				}
			} else if( inStmt instanceof Loop ) {
				if( isRoot || (!inStmt.containsAnnotation(ARCAnnotation.class, "ftregion"))) {
					Statement tStmt = ((Loop)inStmt).getBody();
					childList.addAll(findChildStatements(tStmt, false));
				} else {
					childList.add(inStmt);
				} 
			} else if( inStmt instanceof IfStatement ) {
				if( isRoot || (!inStmt.containsAnnotation(ARCAnnotation.class, "ftregion"))) {
					IfStatement ifStmt = (IfStatement)inStmt;
					childList.addAll(findChildStatements(ifStmt.getThenStatement(), false));
					childList.addAll(findChildStatements(ifStmt.getElseStatement(), false));
					childList.add(ifStmt);
				} else {
					childList.add(inStmt);
				} 
			} else if( inStmt instanceof ExpressionStatement ) {
				if( isRoot || (!inStmt.containsAnnotation(ARCAnnotation.class, "ftregion"))) {
					childList.add(inStmt);
				} else {
					childList.add(inStmt);
				} 
			} else if( inStmt instanceof SwitchStatement ) {
				childList.add(inStmt);
			}
		}
		return childList;
	}
	
	//If a ftinjection call is inserted within a worker-single-mode section, thread ID-checking
	//condition should be removed from its enclosing if-statement.
	//This method will be called later by WorkerSingleModeTransformation pass.
	public static void removeThreadIDCheckingCondition(List<Traversable> ftinjectCallStmts) {
		IfStatement ifStmt = null;
		Traversable t1 = null;
		Traversable t2 = null;
		Expression ifCond = null;
		Expression gtidExp = new NameID("_gtid");
		Expression gtID = null;
		BinaryExpression biExp = null;
		Expression exp1 = null;
		Expression exp2 = null;
		for( Traversable tt : ftinjectCallStmts ) {
			t1 = tt.getParent().getParent();
			if( t1 instanceof IfStatement ) {
				ifStmt = (IfStatement)t1;
				ifCond = ifStmt.getControlExpression();
				gtID = IRTools.findExpression(ifCond, gtidExp);
				if( gtID != null ) {
					t1 = gtID.getParent(); //BinaryExpression containing gtid.
					t2 = t1.getParent(); //Parent of the above binaryExpression
					if( t2 instanceof BinaryExpression ) {
						biExp = (BinaryExpression)t2;
						if( biExp.getLHS().equals(t1) ) {
							exp1 = biExp.getRHS();
						} else {
							exp1 = biExp.getLHS();
						}
						exp2 = new NameID("dummyExp");
						exp1.swapWith(exp2);
						biExp.swapWith(exp1);
					} else {
						exp1 = (Expression)t1;
						exp2 = new IntegerLiteral(1);
						exp1.swapWith(exp2);
					}
				}
			}
		}
	}
	
}
