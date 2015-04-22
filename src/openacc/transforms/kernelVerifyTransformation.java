/**
 * 
 */
package openacc.transforms;

import openacc.analysis.AnalysisTools;
import openacc.hir.ACCAnnotation;
import openacc.analysis.SubArray;
import cetus.hir.Program;
import cetus.hir.Procedure;
import cetus.hir.Annotatable;
import cetus.hir.FunctionCall;
import cetus.hir.Specifier;
import cetus.hir.PointerSpecifier;
import cetus.hir.PseudoSymbol;
import cetus.hir.AccessSymbol;
import cetus.hir.Symbol;
import cetus.hir.SymbolTable;
import cetus.hir.SymbolTools;
import cetus.hir.Statement;
import cetus.hir.AccessExpression;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.CompoundStatement;
import cetus.hir.TranslationUnit;
import cetus.hir.Tools;
import cetus.hir.IRTools;
import cetus.hir.IntegerLiteral;
import cetus.hir.NameID;
import cetus.hir.Declaration;
import cetus.hir.AnnotationDeclaration;
import cetus.hir.AnnotationStatement;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;
import cetus.hir.Identifier;
import cetus.transforms.TransformPass;
import cetus.exec.Driver;
import java.util.List;
import java.util.LinkedList;
import java.util.ArrayList;
import java.util.Set;
import java.util.HashSet;
import java.util.StringTokenizer;

/**
 * @author f6l
 *
 */
public class kernelVerifyTransformation extends TransformPass {
	private boolean IRSymbolOnly = true;
	private static final String KERNELS = "kernels";
	private static final String COMPLEMENT = "complement";
	private boolean complement = true;
	private Set<String> kernelsSet;

	/**
	 * @param program
	 */
	public kernelVerifyTransformation(Program program, boolean IRSymOnly) {
		super(program);
		IRSymbolOnly = IRSymOnly;
	}

	/* (non-Javadoc)
	 * @see cetus.transforms.TransformPass#getPassName()
	 */
	@Override
	public String getPassName() {
		return new String("[kernelVerifyTransformation");
	}

	/* (non-Javadoc)
	 * @see cetus.transforms.TransformPass#start()
	 */
	@Override
	public void start() {
		kernelsSet = new HashSet<String>();
		String options = Driver.getOptionValue("verificationOptions");
		if( options != null ) {
			StringTokenizer tokenizer = new StringTokenizer(options, ":");
			String option;
			while(tokenizer.hasMoreTokens()) {
				option = tokenizer.nextToken();
				int eqIndex = option.indexOf('='); 
				if( eqIndex != -1) {
					String opt = option.substring(0, eqIndex).trim();
					if(!opt.equals(KERNELS)) {
						try {
							int value = new Integer(option.substring(eqIndex+1).trim()).intValue();
							if(opt.equals(COMPLEMENT)) {
								if( value == 1 ) {
									complement = true;
								} else {
									complement = false;
								}
							}
						}
						catch(NumberFormatException ex){
						}
					}
					else {
						StringTokenizer kernels = new StringTokenizer(option.substring(eqIndex+1), " ,");
						while(kernels.hasMoreTokens()) {
							kernelsSet.add(kernels.nextToken().trim());
						}
					}
				}
			}
		}
		//Step1: for each compute region, 
		//       - Move all shared variables to copy or copyin clauses.
		//         (if accreadonly internal clause has the variable, put it into copyin.)
		//       - Remove if clauses, and add async(1) clause.
		//       - Add HI_waitS1(1) and HI_waitS2(1) at the end of it.
		List<Procedure> procedureList = IRTools.getProcedureList(program);
		for( Procedure cProc : procedureList ) {
			String procName = cProc.getSymbolName().toString();
			List<ACCAnnotation> compAnnotsOrg = AnalysisTools.collectPragmas(cProc, ACCAnnotation.class, ACCAnnotation.computeRegions, false);
			if( compAnnotsOrg != null ) {
				List<ACCAnnotation> parallelRegionAnnots = new LinkedList<ACCAnnotation>();
				List<ACCAnnotation> kernelsRegionAnnots = new LinkedList<ACCAnnotation>();
				for( ACCAnnotation cAnnot : compAnnotsOrg ) {
					if( cAnnot.containsKey("kernels") ) {
						kernelsRegionAnnots.add(cAnnot);
					} else {
						parallelRegionAnnots.add(cAnnot);
					}
				}
				List<ACCAnnotation> compAnnots = new LinkedList<ACCAnnotation>();
				compAnnots.addAll(parallelRegionAnnots);
				compAnnots.addAll(kernelsRegionAnnots);
				int cnt = 0;
				for( ACCAnnotation cAnnot : compAnnots ) {
					Annotatable at = cAnnot.getAnnotatable();
					String refName = procName + "_kernel" + cnt++;	
					
					//if verificationOption is provided, only selected kernels will be compared, other kernels will
					//be executed on the CPU.
					boolean skipKernel = false;
					if( kernelsSet.contains(refName) ) {
						if( complement ) {
							skipKernel = true;
						}
					} else if( !complement ) {
						skipKernel = true;
					}
					if( skipKernel ) {
						String kernelType = "kernels";
						if( at.containsAnnotation(ACCAnnotation.class, "parallel") ) {
							kernelType = "parallel";
						}
						//Insert fake kernel directives for consistent kernel naming.
						//In the later ACC2GPUTranslator, if the value of OpenACC annotation key, kernelType,
						//is "false", translation will be skipped.
						at.removeAnnotations(ACCAnnotation.class);
						ACCAnnotation annot = new ACCAnnotation(kernelType, "false");
						at.annotate(annot);
						continue;
					}
					
					ACCAnnotation iAnnot = at.getAnnotation(ACCAnnotation.class, "internal");
					if( iAnnot == null ) {
						iAnnot = new ACCAnnotation("internal", "_directive");
						iAnnot.setSkipPrint(true);
						at.annotate(iAnnot);
					}
					if( !iAnnot.containsKey("refname") ) {
						iAnnot.put("refname", refName);
					}
					CompoundStatement cStmt = (CompoundStatement)at.getParent();
					Set<Symbol> ROSyms = new HashSet<Symbol>();
					Set<SubArray> copyASet = new HashSet<SubArray>();
					Set<SubArray> copyinASet = new HashSet<SubArray>();
					/*				ACCAnnotation copyAnnot = at.getAnnotation(ACCAnnotation.class, "copy");
				if( copyAnnot != null ) {
					copyASet.addAll((Set<SubArray>)copyAnnot.get("copy"));
				}*/
					/*				ACCAnnotation copyinAnnot = at.getAnnotation(ACCAnnotation.class, "copyin");
				if( copyinAnnot != null ) {
					copyinASet.addAll((Set<SubArray>)copyinAnnot.get("copyin"));
				}*/
					ACCAnnotation accROAnnot = at.getAnnotation(ACCAnnotation.class, "accreadonly");
					if( accROAnnot != null ) {
						ROSyms.addAll((Set<Symbol>)accROAnnot.get("accreadonly"));
					}
					for( String dclause : ACCAnnotation.dataClauses ) {
						if( dclause.equals("deviceptr") ) {
							if( at.containsAnnotation(ACCAnnotation.class, dclause) ) {
								Tools.exit("[ERROR] current kernel verification pass can not handle deviceptr data clause; " +
										"please disable kernel verification option (programVerification != 2)");
							} else {
								continue;
							}
							//} else if( dclause.equals("copy") || dclause.equals("copyin") ) {
							//	continue;
						} else {
							ACCAnnotation dAnnot = at.getAnnotation(ACCAnnotation.class, dclause);
							if( dAnnot != null ) {
								Set<SubArray> SubArrays = dAnnot.get(dclause);
								for( SubArray sArry : SubArrays ) {
									List<Expression> startList = new LinkedList<Expression>();
									List<Expression> lengthList = new LinkedList<Expression>();
									AnalysisTools.extractDimensionInfo(sArry, startList, lengthList, IRSymbolOnly, at);
									Symbol sym = AnalysisTools.subarrayToSymbol(sArry, IRSymbolOnly);
									if( sym != null ) {
										if( ROSyms.contains(sym) ) {
											copyinASet.add(sArry);
										} else {
											copyASet.add(sArry);
										}
									}
								}
								dAnnot.remove(dclause);
							}
						}
					}
					if( !copyASet.isEmpty() ) {
						cAnnot.put("copy", copyASet);
					}
					if( !copyinASet.isEmpty() ) {
						cAnnot.put("copyin", copyinASet);
					}
					cAnnot.remove("if");
					cAnnot.remove("async");
					IntegerLiteral one = new IntegerLiteral(1);
					cAnnot.put("async", one);
					//Below is moved to ACC2CUDATranslator.
					/*				FunctionCall asyncW2Call = new FunctionCall(new NameID("HI_waitS2"), new IntegerLiteral(1));
				ExpressionStatement asyncW2Stmt = new ExpressionStatement(asyncW2Call);
				cStmt.addStatementAfter((Statement)at, asyncW2Stmt);
				FunctionCall asyncW1Call = new FunctionCall(new NameID("HI_waitS1"), new IntegerLiteral(1));
				ExpressionStatement asyncW1Stmt = new ExpressionStatement(asyncW1Call);
				cStmt.addStatementAfter((Statement)at, asyncW1Stmt);*/
					//Step2: Insert result compare codes for each variable in copy clause.
				}
			}
		}
		//Step3: remove all data directives, declare directives, update directives, 
		//       wait directives, and OpenACC runtimes related to async execution.
		//       (deviceptr or device_resident clauses are not yet supported for kernel verification; exit!)
		Set<String> searchKeys = new HashSet<String>();
		searchKeys.add("data");
		searchKeys.add("declare");
		searchKeys.add("update");
		searchKeys.add("wait");
		List<ACCAnnotation> miscAnnots = AnalysisTools.collectPragmas(program, ACCAnnotation.class, searchKeys, false);
		if( miscAnnots != null ) {
			for( ACCAnnotation annot : miscAnnots ) {
				Annotatable at = annot.getAnnotatable();
				if( annot.containsKey("deviceptr") || annot.containsKey("device_resident") ) {
					Tools.exit("[ERROR] current kernel verification pass can not handle deviceptr/device_resident clauses; " +
							"please disable kernel verification option (programVerification != 2)");
				} 
				if( annot.containsKey("data") ) {
					at.removeAnnotations(ACCAnnotation.class);
				} else {
					if( at instanceof AnnotationStatement ) {
						Statement aStmt = (AnnotationStatement)at;
						CompoundStatement pStmt = (CompoundStatement)aStmt.getParent();
						pStmt.removeStatement(aStmt);
					} else if( at instanceof AnnotationDeclaration ) {
						Declaration aDecl = (AnnotationDeclaration)at;
						TranslationUnit tu = (TranslationUnit)at.getParent();
						tu.removeChild(aDecl);
					} else {
						Tools.exit("[ERROR in kernelVerifyTransformation.start(); unexpected type of Annotatable, " 
					+ at + ", exit!\n");
					}
					
				}
			}
		}
		List<FunctionCall> funcCalls = IRTools.getFunctionCalls(program);
		if( funcCalls != null ) {
			Set<String> searchFuncs = new HashSet<String>();
			searchFuncs.add("acc_async_test");
			searchFuncs.add("acc_async_test_all");
			searchFuncs.add("acc_wait");
			searchFuncs.add("acc_wait_all");
			for(FunctionCall fCall : funcCalls ) {
				String fName = fCall.getName().toString();
				if( searchFuncs.contains(fName) ) {
					Statement pStmt = fCall.getStatement();
					CompoundStatement cStmt = (CompoundStatement)pStmt.getParent();
					cStmt.removeStatement(pStmt);
				}
			}
		}
		//Step4: generate both CPU versions and GPU versions
		//       (This will be done by ACC2CUDATranslator or ACC2OPENCLTranslator.)

	}

}
