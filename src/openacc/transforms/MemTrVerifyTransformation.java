/**
 * 
 */
package openacc.transforms;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import cetus.analysis.AnalysisPass;
import cetus.analysis.LoopTools;
import cetus.exec.Driver;
import cetus.hir.AccessSymbol;
import cetus.hir.Annotatable;
import cetus.hir.CompoundStatement;
import cetus.hir.DFIterator;
import cetus.hir.Declaration;
import cetus.hir.DeclarationStatement;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.ForLoop;
import cetus.hir.FunctionCall;
import cetus.hir.Identifier;
import cetus.hir.NameID;
import cetus.hir.PrintTools;
import cetus.hir.ProcedureDeclarator;
import cetus.hir.Program;
import cetus.hir.Procedure;
import cetus.hir.IRTools;
import cetus.hir.Statement;
import cetus.hir.StringLiteral;
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
import openacc.analysis.AnalysisTools;
import openacc.analysis.IpFirstAccessAnalysis;
import openacc.analysis.IpRedundantMemTrAnalysis;
import openacc.hir.ACCAnnotation;

/**
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group
 *         Oak Ridge National Laboratory
 *
 */
public class MemTrVerifyTransformation extends TransformPass {
	private boolean IRSymbolOnly;

	/**
	 * @param program
	 */
	public MemTrVerifyTransformation(Program program, boolean IRSymOnly) {
		super(program);
		IRSymbolOnly = IRSymOnly;
	}

	/* (non-Javadoc)
	 * @see cetus.transforms.TransformPass#getPassName()
	 */
	@Override
	public String getPassName() {
		return new String("[memoryTrasferVerifyTransformation");
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
		//Step1: add memtrref(refname) internal clause for each data-transfer-involving directive 
		//       (update, declare, data, kernel, and parallel directives) and check_read/check_write call
		//       - For compute regions,
		//           - use "${procedure}_kernel${counter}" as names.
		//       - For data regions,
		//           - use "${procedure}_data${counter}" as names.
		//       - For update directives,
		//           - use "{$procedure}_update${counter}" as names.
		//       - For declare directives in a procedure,
		//           - use "{$procedure}_declare${counter}" as names.
		//       - For declare directives out of a procedure, 
		//           - use "{$filename}_declare${counter}" as names.
		//       - For each check_read() or check_write() call, 
		//           - Use "${procedure}_hostcheck${counter}" for CPU-access checking
		//           - Use "${procedure}_kernel${counter}" for GPU-access checking
		Set<String> searchKeys = new HashSet<String>(ACCAnnotation.dataRegions);
		searchKeys.add("update");
		searchKeys.add("declare");
		List<Procedure> procedureList = IRTools.getProcedureList(program);
		for( Procedure cProc : procedureList ) {
			String procName = cProc.getSymbolName().toString();
			List<ACCAnnotation> dataAnnots = 
					AnalysisTools.collectPragmas(cProc, ACCAnnotation.class, searchKeys, false);
			List<ACCAnnotation> parallelRegionAnnots = new LinkedList<ACCAnnotation>();
			List<ACCAnnotation> kernelsRegionAnnots = new LinkedList<ACCAnnotation>();
			List<ACCAnnotation> dataRegionAnnots = new LinkedList<ACCAnnotation>();
			List<ACCAnnotation> updateAnnots = new LinkedList<ACCAnnotation>();
			List<ACCAnnotation> declareAnnots = new LinkedList<ACCAnnotation>();
			if( dataAnnots != null ) {
				for( ACCAnnotation annot : dataAnnots ) {
					if( annot.containsKey("data") ) {
						dataRegionAnnots.add(annot);
					} else if( annot.containsKey("parallel") ) {
						parallelRegionAnnots.add(annot);
					} else if( annot.containsKey("kernels") ) {
						kernelsRegionAnnots.add(annot);
					} else if( annot.containsKey("update") ) {
						updateAnnots.add(annot);
					} else if( annot.containsKey("declare") ) {
						declareAnnots.add(annot);
					}
				}
				int cnt = 0;
				for( ACCAnnotation tAnnot : dataRegionAnnots ) {
					String refName = procName + "_data" + cnt++;	
					Annotatable at = tAnnot.getAnnotatable();
					ACCAnnotation iAnnot = at.getAnnotation(ACCAnnotation.class, "internal");
					if( iAnnot == null ) {
						iAnnot = new ACCAnnotation("internal", "_directive");
						iAnnot.setSkipPrint(true);
						at.annotate(iAnnot);
					}
					if( !iAnnot.containsKey("refname") ) {
						iAnnot.put("refname", refName);
					}
				}
				cnt = 0;
				for( ACCAnnotation tAnnot : parallelRegionAnnots ) {
					String refName = procName + "_kernel" + cnt++;	
					Annotatable at = tAnnot.getAnnotatable();
					ACCAnnotation iAnnot = at.getAnnotation(ACCAnnotation.class, "internal");
					if( iAnnot == null ) {
						iAnnot = new ACCAnnotation("internal", "_directive");
						iAnnot.setSkipPrint(true);
						at.annotate(iAnnot);
					}
					if( !iAnnot.containsKey("refname") ) {
						iAnnot.put("refname", refName);
					}
				}
				for( ACCAnnotation tAnnot : kernelsRegionAnnots ) {
					String refName = procName + "_kernel" + cnt++;	
					Annotatable at = tAnnot.getAnnotatable();
					ACCAnnotation iAnnot = at.getAnnotation(ACCAnnotation.class, "internal");
					if( iAnnot == null ) {
						iAnnot = new ACCAnnotation("internal", "_directive");
						iAnnot.setSkipPrint(true);
						at.annotate(iAnnot);
					}
					if( !iAnnot.containsKey("refname") ) {
						iAnnot.put("refname", refName);
					}
				}
				cnt = 0;
				for( ACCAnnotation tAnnot : updateAnnots ) {
					String refName = procName + "_update" + cnt++;	
					Annotatable at = tAnnot.getAnnotatable();
					ACCAnnotation iAnnot = at.getAnnotation(ACCAnnotation.class, "internal");
					if( iAnnot == null ) {
						iAnnot = new ACCAnnotation("internal", "_directive");
						iAnnot.setSkipPrint(true);
						at.annotate(iAnnot);
					}
					if( !iAnnot.containsKey("refname") ) {
						iAnnot.put("refname", refName);
					}
				}
				cnt = 0;
				for( ACCAnnotation tAnnot : declareAnnots ) {
					String refName = procName + "_declare" + cnt++;	
					Annotatable at = tAnnot.getAnnotatable();
					ACCAnnotation iAnnot = at.getAnnotation(ACCAnnotation.class, "internal");
					if( iAnnot == null ) {
						iAnnot = new ACCAnnotation("internal", "_directive");
						iAnnot.setSkipPrint(true);
						at.annotate(iAnnot);
					}
					if( !iAnnot.containsKey("refname") ) {
						iAnnot.put("refname", refName);
					}
				}
			}
		}
		//Handle declare directives for implicit program-level data regions.
		for( Traversable t : program.getChildren() ) {
			TranslationUnit trUnt = (TranslationUnit)t;
			String fileName = trUnt.getInputFilename();
			int cnt = 0;
			for( Traversable tt : trUnt.getChildren() ) {
				if( tt instanceof Annotatable ) {
					Annotatable at = (Annotatable)tt;
					if( at.containsAnnotation(ACCAnnotation.class, "declare") ) {
						String refName = fileName + "_declare" + cnt++;
						ACCAnnotation iAnnot = at.getAnnotation(ACCAnnotation.class, "internal");
						if( iAnnot == null ) {
							iAnnot = new ACCAnnotation("internal", "_directive");
							iAnnot.setSkipPrint(true);
							at.annotate(iAnnot);
						}
						if( !iAnnot.containsKey("refname") ) {
							iAnnot.put("refname", refName);
						}
					}
				}
			}
		}
		/*
		 * [Example Code]
		 * check_read(a, CPU);  //error if CPU is stale
		 * check_write(a, CPU); //warn if CPU is stale; set GPU stale
		 * check_read(b, CPU);
		 * reset_status(a, GPU, not-stale); //insert this if GPU does not access a any more.
		 * reset_status(a, GPU, may-stale); //insert this if a is written before read in the following GPU kernel.
		 * a = a + b;
		 * #pragma acc data copy(a)
		 * {
		 *     set_status(a, GPU, not-stale); //due to copyin(a); check redundant copyin().
		 *     check_read(a, GPU);  //due to R/W GPUKernel(a) (if a is W/R in the kernel, this call will not be inserted.)
		 *     check_write(a, GPU); //due to R/W GPUKernel(a) 
		 *     GPUKernel(a); //assume a is R/W
		 *     reset_status(a, CPU, not-stale); //insert this if CPU does not access a after this kernel call.
		 *     reset_status(a, CPU, may-stale); //insert this if a is written before read in the following CPU region.
		 *     set_status(a, CPU, not-stale); //due to copyout(a); check redundant copyout().
		 *     reset_status(a, GPU, stale);     //due to cudaFree(a)
		 * }
		 * check_read(c, CPU);
		 * check_write(a, CPU);
		 * a = c;
		 */
		//Step2: perform data flow analysis to find first read/write accesses, and insert check_read()/check_write() calls.
		//    - insert check_read(var, CPU) before the first CPU read access from the beginning or after a previous kernel call.
		//    - insert check_write(var, CPU) before the first CPU write access from the beginning or after a previous kernel call.
		//    - insert check_read(var, GPU) before a kernel call if the var is read in the kernel.
		//    - insert check_write(var, GPU) before a kernel call if the var is written in the kernel.
		//Step3: perform redundant transfer analysis, and insert reset_status() calls.
		//    - insert reset_status() calls right after a kernel if a variable is not upward-exposed at that point. 
		//        - insert reset_status(var, CPU, may-stale) if the CPU variable is written before read.
		//        - insert reset_status(var, CPU, not-stale) if the CPU variable is not used
		//    - insert reset_status() calls right after CPU-write statement if a variable is not upward-exposed in following compute regions.
		//        - insert reset_status(var, GPU, may-stale) if the GPU variable is written before read.
		//        - insert reset_status(var, GPU, not-stale) if the GPU variable is not used anymore
		//    - insert reset_status(var, GPU, stale) calls right after a kernel if a variable is reduction variable.
		//        - This will be done by CUDATranslationTools.reductionTransformation().
		Set<Symbol>	targetSymbols = AnalysisTools.getAllACCSharedVariables(program);
		AnalysisTools.markIntervalForComputeRegions(program);
		AnalysisPass.run(new IpFirstAccessAnalysis(program, IRSymbolOnly, targetSymbols, true));
		AnalysisPass.run(new IpRedundantMemTrAnalysis(program, IRSymbolOnly, targetSymbols, true));
		AnalysisTools.deleteBarriers(program);
		List<FunctionCall> fCallList = IRTools.getFunctionCalls(program);
		// Insert check_read()/check_write() calls for each statement containing firstreadSet/firstwriteSet clauses.
		// and delete the temporary internal annotation.
		for( Procedure tProc : procedureList ) {
			String procName = tProc.getSymbolName();
			Map<Symbol, Symbol> g2lSymMap = new HashMap<Symbol, Symbol>();
			List<ACCAnnotation> tempInternalAnnots = IRTools.collectPragmas(tProc, ACCAnnotation.class, "tempinternal");
			if( tempInternalAnnots != null ) {
				int cnt = 0;
				for( ACCAnnotation tAnnot : tempInternalAnnots ) {
					Annotatable at = tAnnot.getAnnotatable();
					Expression loopIndex = null;
					//Find enclosing ForLoop if existing.
					Traversable tt = at.getParent();
					while( (tt != null) && !(tt instanceof Procedure) ) {
						if( tt instanceof ForLoop ) {
							loopIndex = LoopTools.getIndexVariable((ForLoop)tt);
							break;
						} else {
							tt = tt.getParent();
						}
					}
					List<ACCAnnotation> pragmas = at.getAnnotations(ACCAnnotation.class);
					boolean isComputeRegion = false;
					for( ACCAnnotation nAnnot : pragmas ) {
						if( nAnnot.containsKey("kernels") || nAnnot.containsKey("parallel") ) {
							isComputeRegion = true;
							break;
						}
					}
					if( isComputeRegion ) {
						//Compute region will be handled later by ACC2CUDATranslator.
/*						PrintTools.println("[INFO in MemTrVerifyTransformation] found tempinternal annotation for " +
								"compute region:\n" +
								at.toString() + "\n", 0);*/
						continue;
					} else {
/*						PrintTools.println("[INFO in MemTrVerifyTransformation] found tempinternal annotation:\n" +
								at.toString() + "\n", 0);*/
						//System.err.println("Region containing tempinternal directive (before):\n" + at + "\n\n");
						at.removeAnnotations(ACCAnnotation.class);
						for( ACCAnnotation nAnnot : pragmas ) {
							if( !nAnnot.containsKey("tempinternal") ) {
								at.annotate(nAnnot);
							}
						}
						CompoundStatement cStmt = (CompoundStatement)at.getParent();
						Statement refStmt = (Statement)at;
						boolean isDataRegion = false;
						if( refStmt.containsAnnotation(ACCAnnotation.class, "data") ) {
							isDataRegion = true;
						}
						String refname = null;
						ACCAnnotation rAnnot = at.getAnnotation(ACCAnnotation.class, "refname");
						if( rAnnot != null ) {
							refname = rAnnot.get("refname");
						}
						if( refname == null ) {
							refname = procName + "_hostcheck" + cnt++;
						}
						StringLiteral refNameConst = new StringLiteral(refname);
						Set<Symbol> accessedSyms = null;
						Set<Symbol> firstWriteSet = tAnnot.get("firstwriteSet");
						Set<Symbol> gfirstWriteSet = tAnnot.get("gfirstwriteSet");
						Set<Symbol> firstReadSet = tAnnot.get("firstreadSet");
						Set<Symbol> mayKilledSet = tAnnot.get("maykilled");
						Set<Symbol> deadSet = tAnnot.get("dead");
						Symbol tSO = tAnnot.get("callocsym");
						Set<Symbol> checkSet = new HashSet<Symbol>();
						Set<Symbol> firstWSet = new HashSet<Symbol>();
						if( firstWriteSet != null ) {
							checkSet.addAll(firstWriteSet);
							firstWSet.addAll(firstWriteSet);
						}
						if( gfirstWriteSet != null ) {
							if( isDataRegion ) {
								ACCAnnotation ttAnnot = refStmt.getAnnotation(ACCAnnotation.class, "accshared");
								if( ttAnnot != null ) {
									Set<Symbol> excludeSet = new HashSet<Symbol>();
									Set<Symbol> lDataSyms = ttAnnot.get("accshared");
									Set<Symbol> gDataSyms = new HashSet<Symbol>();
									for( Symbol lSym : lDataSyms ) {
										List symbolInfo = new ArrayList(2);
										if( AnalysisTools.SymbolStatus.OrgSymbolFound(
												AnalysisTools.findOrgSymbol(lSym, at, true, null, symbolInfo, fCallList)) ) {
											Symbol gSym = (Symbol)symbolInfo.get(0);
											gDataSyms.add(gSym);
										}
									}
									for( Symbol wSym : gfirstWriteSet ) {
										if( gDataSyms.contains(wSym) ) {
											excludeSet.add(wSym);
										} else {
											checkSet.add(wSym);
											firstWSet.add(wSym);
										}
									}
									if( !excludeSet.isEmpty() ) {
										ACCAnnotation tempAnnot = new ACCAnnotation("tempinternal", "_directive");
										tempAnnot.put("gfirstwriteSet", excludeSet);
										at.annotate(tempAnnot);
									}
								} else {
									checkSet.addAll(gfirstWriteSet);
									firstWSet.addAll(gfirstWriteSet);
								}
							} else {
								checkSet.addAll(gfirstWriteSet);
								firstWSet.addAll(gfirstWriteSet);
							}
						} else {
							gfirstWriteSet = new HashSet<Symbol>();
						}
						if( firstReadSet != null ) {
							checkSet.addAll(firstReadSet);
						}
						if( mayKilledSet != null ) {
							checkSet.addAll(mayKilledSet);
						}
						if( deadSet != null ) {
							checkSet.addAll(deadSet);
						}
						if( !checkSet.isEmpty() ) {
							//Find local symbol visible in the current procedure scope.
							accessedSyms = AnalysisTools.getAccessedVariables(at, IRSymbolOnly);
							if( (accessedSyms != null) && !accessedSyms.isEmpty() ) {
								for( Symbol lSym : accessedSyms ) {
									List symbolInfo = new ArrayList(2);
									if( AnalysisTools.SymbolStatus.OrgSymbolFound(
											AnalysisTools.findOrgSymbol(lSym, at, true, null, symbolInfo, fCallList)) ) {
										Symbol gSym = (Symbol)symbolInfo.get(0);
										if( checkSet.contains(gSym) ) {
											g2lSymMap.put(gSym, lSym);
										}
									}
								}
								//System.err.println("Found accessSymbols:\n" + accessedSyms + "\nStatement:\n" + at + "\n\n");
							} else if( at instanceof DeclarationStatement ) {
								//System.err.println("Found declarationStatement:\n" + at + "\n\n");
								DFIterator<VariableDeclarator> decIter =
									new DFIterator<VariableDeclarator>(at,
											VariableDeclarator.class);
								decIter.pruneOn(VariableDeclarator.class);
								while (decIter.hasNext()) {
									VariableDeclarator o = decIter.next();
									if( checkSet.contains(o) ) {
										g2lSymMap.put(o, o);
									}
								}
							}
						}
						if( !firstWSet.isEmpty() ) {
							for( Symbol gsym : firstWSet ) {
								FunctionCall checkCall = new FunctionCall(new NameID("HI_check_write"));
								Expression hostVar = null;
								Symbol lsym = g2lSymMap.get(gsym);
								if( lsym == null ) {
									Tools.exit("[ERROR in MemTrVerifyTransformation] can't find locally visible symbol " +
											"for the first-write symbol: " + gsym + "\nEnclosing procedure: " + 
											tProc.getSymbolName() + "\n");
								}
								if( lsym instanceof AccessSymbol ) {
									hostVar = AnalysisTools.accessSymbolToExpression((AccessSymbol)lsym, null);
								} else {
									hostVar = new Identifier(lsym);
								}
								if( !SymbolTools.isArray(lsym) && !SymbolTools.isPointer(lsym) ) { //scalar
									checkCall.addArgument( new UnaryExpression(UnaryOperator.ADDRESS_OF, 
											hostVar.clone()));
								} else {
									checkCall.addArgument(hostVar.clone());
								}
								if( gfirstWriteSet.contains(gsym) ) {
									checkCall.addArgument(new NameID("acc_device_nvidia"));
								} else {
									checkCall.addArgument(new NameID("acc_device_host"));
								}
								checkCall.addArgument(new StringLiteral(hostVar.toString()));
								checkCall.addArgument(refNameConst.clone());
								if( loopIndex != null ) {
									checkCall.addArgument(loopIndex.clone());
								} else {
									checkCall.addArgument(new NameID("INT_MIN"));
								}
								if( (tSO != null) && (gsym == tSO) ) {
									//check_write() should be inserted after calloc() statement.
									cStmt.addStatementAfter(refStmt, new ExpressionStatement(checkCall));
								} else if( refStmt instanceof DeclarationStatement ) {
									Statement trefStmt = IRTools.getFirstNonDeclarationStatement(cStmt);
									if( trefStmt == null ) {
										cStmt.addStatementAfter(refStmt, new ExpressionStatement(checkCall));
									} else {
										cStmt.addStatementBefore(trefStmt, new ExpressionStatement(checkCall));
									}
								} else {
									cStmt.addStatementBefore(refStmt, new ExpressionStatement(checkCall));
								}
							}
						}
						if( firstReadSet != null ) {
							for( Symbol gsym : firstReadSet ) {
								FunctionCall checkCall = new FunctionCall(new NameID("HI_check_read"));
								Expression hostVar = null;
								Symbol lsym = g2lSymMap.get(gsym);
								if( lsym == null ) {
									Tools.exit("[ERROR in MemTrVerifyTransformation] can't find locally visible symbol " +
											"for the first-read symbol: " + gsym + "\nEnclosing procedure: " + 
											tProc.getSymbolName() + "\n");
								}
								if( lsym instanceof AccessSymbol ) {
									hostVar = AnalysisTools.accessSymbolToExpression((AccessSymbol)lsym, null);
								} else {
									hostVar = new Identifier(lsym);
								}
								if( !SymbolTools.isArray(lsym) && !SymbolTools.isPointer(lsym) ) { //scalar
									checkCall.addArgument( new UnaryExpression(UnaryOperator.ADDRESS_OF, 
											hostVar.clone()));
								} else {
									checkCall.addArgument(hostVar.clone());
								}
								checkCall.addArgument(new NameID("acc_device_host"));
								checkCall.addArgument(new StringLiteral(hostVar.toString()));
								checkCall.addArgument(refNameConst.clone());
								if( loopIndex != null ) {
									checkCall.addArgument(loopIndex.clone());
								} else {
									checkCall.addArgument(new NameID("INT_MIN"));
								}
								cStmt.addStatementBefore(refStmt, new ExpressionStatement(checkCall));
							}
						}
						if( mayKilledSet != null ) {
							for( Symbol gsym : mayKilledSet ) {
								FunctionCall checkCall = new FunctionCall(new NameID("HI_reset_status"));
								Expression hostVar = null;
								Symbol lsym = g2lSymMap.get(gsym);
								if( lsym == null ) {
									Tools.exit("[ERROR in MemTrVerifyTransformation] can't find locally visible symbol " +
											"for the symbol: " + gsym + "\nEnclosing procedure: " + 
											tProc.getSymbolName() + "\n");
								}
								if( lsym instanceof AccessSymbol ) {
									hostVar = AnalysisTools.accessSymbolToExpression((AccessSymbol)lsym, null);
								} else {
									hostVar = new Identifier(lsym);
								}
								if( !SymbolTools.isArray(lsym) && !SymbolTools.isPointer(lsym) ) { //scalar
									checkCall.addArgument( new UnaryExpression(UnaryOperator.ADDRESS_OF, 
											hostVar.clone()));
								} else {
									checkCall.addArgument(hostVar.clone());
								}
								checkCall.addArgument(new NameID("acc_device_nvidia"));
								checkCall.addArgument(new NameID("HI_maystale"));
								//checkCall.addArgument(new NameID("INT_MIN"));
								checkCall.addArgument(new NameID("DEFAULT_QUEUE"));
								if( (tSO != null) && (gsym == tSO) ) {
									cStmt.addStatementAfter(refStmt, new ExpressionStatement(checkCall));
								} else {
									cStmt.addStatementBefore(refStmt, new ExpressionStatement(checkCall));
								}
							}
						}
						if( deadSet != null ) {
							for( Symbol gsym : deadSet ) {
								FunctionCall checkCall = new FunctionCall(new NameID("HI_reset_status"));
								Expression hostVar = null;
								Symbol lsym = g2lSymMap.get(gsym);
								if( lsym == null ) {
									Tools.exit("[ERROR in MemTrVerifyTransformation] can't find locally visible symbol " +
											"for the symbol: " + gsym + "\nEnclosing procedure: " + 
											tProc.getSymbolName() + "\n");
								}
								if( lsym instanceof AccessSymbol ) {
									hostVar = AnalysisTools.accessSymbolToExpression((AccessSymbol)lsym, null);
								} else {
									hostVar = new Identifier(lsym);
								}
								if( !SymbolTools.isArray(lsym) && !SymbolTools.isPointer(lsym) ) { //scalar
									checkCall.addArgument( new UnaryExpression(UnaryOperator.ADDRESS_OF, 
											hostVar.clone()));
								} else {
									checkCall.addArgument(hostVar.clone());
								}
								checkCall.addArgument(new NameID("acc_device_nvidia"));
								checkCall.addArgument(new NameID("HI_notstale"));
								//checkCall.addArgument(new NameID("INT_MIN"));
								checkCall.addArgument(new NameID("DEFAULT_QUEUE"));
								if( (tSO != null) && (gsym == tSO) ) {
									cStmt.addStatementAfter(refStmt, new ExpressionStatement(checkCall));
								} else {
									cStmt.addStatementBefore(refStmt, new ExpressionStatement(checkCall));
								}
							}
						}
						//System.err.println("Region containing tempinternal directive (after):\n" + at + "\n\n");
					}
				}
			}
		}
		
		//Step4: add check_write(var, CPU) at the entrance of a procedure if var is a scalar parameter of the procedure,
		//and if var is accessed by GPU kernels in the procedure. This is because current IpFirstAccessAnalysis does not
		//track the first write across procedure boundary if a shared variable is a scalar function parameter.
		Procedure main = null;
		TranslationUnit mainTr = null;
		Set<Symbol> allSharedSyms = new HashSet<Symbol>();
		Set<Symbol> scalarParams = new HashSet<Symbol>();
		Set<Symbol> fWSSet = new HashSet<Symbol>();
		for( Procedure tProc : procedureList ) {
			String procname = tProc.getSymbolName();
			/* f2c code uses MAIN__ */
			if ( ((mainEntryFunc != null) && procname.equals(mainEntryFunc)) || 
					((mainEntryFunc == null) && (procname.equals("main") || procname.equals("MAIN__"))) ) {
				main = tProc;
				mainTr = (TranslationUnit)main.getParent();
			}
			scalarParams.clear();
			fWSSet.clear();
			ProcedureDeclarator pdecl = (ProcedureDeclarator)tProc.getDeclarator();
			List<Declaration> params = pdecl.getParameters();
			for (int i = 0; i < params.size(); i++) {
				Declaration d = params.get(i);
				Symbol p = (Symbol)d.getChildren().get(0);
				if( SymbolTools.isScalar(p) && !SymbolTools.isPointer(p) ) {
					scalarParams.add(p);
				}
			}
			List<ACCAnnotation> sharedAnnots = IRTools.collectPragmas(tProc, ACCAnnotation.class, "accshared");
			if( sharedAnnots != null ) {
				for( ACCAnnotation sAnnot : sharedAnnots ) {
					Set<Symbol> accshared = sAnnot.get("accshared");
					allSharedSyms.addAll(accshared);
					if( !scalarParams.isEmpty() ) {
						for( Symbol tSym : scalarParams ) {
							if( accshared.contains(tSym) ) {
								fWSSet.add(tSym);
							}
						}
					}
				}
			}
			if( !fWSSet.isEmpty() ) {
				StringLiteral refNameConst = new StringLiteral(procname + "_hostcheck01M");
				CompoundStatement body = tProc.getBody();
				Statement refStmt = IRTools.getFirstNonDeclarationStatement(body);
				for( Symbol lsym : fWSSet ) {
					FunctionCall checkCall = new FunctionCall(new NameID("HI_check_write"));
					Expression hostVar = null;
					if( lsym instanceof AccessSymbol ) {
						hostVar = AnalysisTools.accessSymbolToExpression((AccessSymbol)lsym, null);
					} else {
						hostVar = new Identifier(lsym);
					}
					checkCall.addArgument( new UnaryExpression(UnaryOperator.ADDRESS_OF, 
							hostVar.clone()));
					checkCall.addArgument(new NameID("acc_device_host"));
					checkCall.addArgument(new StringLiteral(hostVar.toString()));
					checkCall.addArgument(refNameConst.clone());
					checkCall.addArgument(new NameID("INT_MIN"));
					body.addStatementBefore(refStmt, new ExpressionStatement(checkCall));
				}
			}
		}
		Set<Symbol> globalCWSet = new HashSet<Symbol>();
		Map<String, Symbol> symMapInMainTr = new HashMap<String, Symbol>();
		List<Traversable> trUnits = program.getChildren();
		for( Traversable t : trUnits ) {
			List<Traversable> tChildren = t.getChildren();
			for( Traversable tt : tChildren ) {
				if( tt instanceof VariableDeclaration ) {
					DFIterator<VariableDeclarator> decIter =
							new DFIterator<VariableDeclarator>(tt,
									VariableDeclarator.class);
					decIter.pruneOn(VariableDeclarator.class);
					while (decIter.hasNext()) {
						VariableDeclarator o = decIter.next();
						if( (o.getInitializer() != null) && allSharedSyms.contains(o) ) {
							globalCWSet.add(o);
						}	
						if( t == mainTr ) {
							symMapInMainTr.put(o.getSymbolName(), o);
						}
					}
				}
			}
		}
		if( !globalCWSet.isEmpty() ) {
			StringLiteral refNameConst = new StringLiteral(main.getSymbolName() + "_hostcheck00M");
			CompoundStatement body = main.getBody();
			Statement refStmt = IRTools.getFirstNonDeclarationStatement(body);
			for( Symbol sym : globalCWSet ) {
				Symbol cSym = symMapInMainTr.get(sym.getSymbolName());
				if( cSym != null ) {
					FunctionCall checkCall = new FunctionCall(new NameID("HI_check_write"));
					Expression hostVar = null;
					if( cSym instanceof AccessSymbol ) {
						hostVar = AnalysisTools.accessSymbolToExpression((AccessSymbol)cSym, null);
					} else {
						hostVar = new Identifier(cSym);
					}
					checkCall.addArgument( new UnaryExpression(UnaryOperator.ADDRESS_OF, 
							hostVar.clone()));
					checkCall.addArgument(new NameID("acc_device_host"));
					checkCall.addArgument(new StringLiteral(hostVar.toString()));
					checkCall.addArgument(refNameConst.clone());
					checkCall.addArgument(new NameID("INT_MIN"));
					body.addStatementBefore(refStmt, new ExpressionStatement(checkCall));
				} else {
					PrintTools.println("\n[WARNING in MemTrVerfyTransformation()] global variable (" + sym.getSymbolName() + 
							") has initial value, but not visiable by main() procedure!\n", 0);
				}
			}
		}
		
		//Step5: add set_status() calls for each memory transfer and reset_status() calls after a cudaFree() call.
		//    - insert set_status(var, GPU, not-stale) for variables in copyin/update device clause
		//        - This will be done by ACC2CUDATranslator.
		//    - insert set_status(var, CPU, not-stale) for variables in copyout/update host clause
		//        - This will be done by ACC2CUDATranslator.
		//    - insert reset_status(var, GPU, stale) calls right after a cudaFree() call.
		//        - This will be done by ACC2CUDATranslator
	}

}

