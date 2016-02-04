/**
 * 
 */
package openacc.analysis;

import openacc.hir.ACCAnnotation;
import openacc.transforms.TransformTools;
import cetus.analysis.AnalysisPass;
import cetus.analysis.CFGraph;
import cetus.analysis.DFANode;
import cetus.analysis.LoopTools;
import cetus.exec.Driver;
import cetus.hir.AccessSymbol;
import cetus.hir.Annotatable;
import cetus.hir.Annotation;
import cetus.hir.AnnotationStatement;
import cetus.hir.BinaryExpression;
import cetus.hir.BreadthFirstIterator;
import cetus.hir.CommentAnnotation;
import cetus.hir.CompoundStatement;
import cetus.hir.Declaration;
import cetus.hir.Declarator;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.FlatIterator;
import cetus.hir.ForLoop;
import cetus.hir.FunctionCall;
import cetus.hir.IDExpression;
import cetus.hir.Identifier;
import cetus.hir.Loop;
import cetus.hir.NameID;
import cetus.hir.OmpAnnotation;
import cetus.hir.Procedure;
import cetus.hir.ProcedureDeclarator;
import cetus.hir.Program;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.SymbolTools;
import cetus.hir.Tools;
import cetus.hir.DataFlowTools;
import cetus.hir.IRTools;
import cetus.hir.PrintTools;
import cetus.hir.Symbol;
import cetus.hir.StandardLibrary;
import cetus.hir.TranslationUnit;
import cetus.hir.Traversable;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.Stack;
import java.util.TreeMap;
import java.util.Iterator;

/**
 * Inter-procedural analysis to identify first statements that 
 * reads/write OpenACC shared variables. 
 *  
 * <p>
 * Input  : input program 
 * Output : set of OpenARC internal clauses(firstwriteSet, firstreadSet) annotated to the first statements
 *          that access OpenACC shared variables.
 *          For compute regions (kernels/parallel regions), these clauses will be attached only to each compute 
 *          region, but not to statements within compute regions.
 *          For regions outside of compute regions, these clauses will be attached to each first-access statement.
 *          (If the first-access statement is in a loop, the enclosing loop may be annotated if it is safe to move 
 *          the annotation upto the loop.)
 * <p>  
 *[Interprocedural first-access analysis]
 * readSet_in(program entry-node) = {}
 * writeSet_in(program entry-node) = {}
 *  for ( node m : predecessor nodes of node n ) {
 * 	    readSet_in(n)  ^= readSet_out(m) // ^ : intersection
 * 	    writeSet_in(n)  ^= writeSet_out(m) // ^ : intersection
 *  }
 *  writeSet_out(n) = writeSet_in(n) + writes(n) - kills(n) // + : union
 *      where,
 *      writes(n) = a set of OpenACC shared variables that are written by CPU if a node n is not in a compute region
 *                  {} if a node is in a compute region
 *      kills(n) = a set of OpenACC shared variables that are written by GPU if a node n is in a compute region
 *                 {} if a node n is not in a compute region
 *  firstwriteSet(n) = writeSet_out(n) - writeSet_in(n) 
 *  readSet_out(n) = readSet_in(n) + reads(n) - kills(n) // + : union
 *      where,
 *      reads(n) = a set of OpenACC shared variables that are read by CPU if a node n is not in a compute region
 *                 {} if a node is in a compute region
 *      kills(n) = a set of OpenACC shared variables that are written by GPU if a node n is in a compute region
 *                 {} if a node n is not in a compute region
 *  firstreadSet(n) = readSet_out(n) - readSet_in(n) - writeSet_in(n)
 * <p>  
 * 
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group
 *         Oak Ridge National Laboratory
 */
public class IpFirstAccessAnalysis extends AnalysisPass {
	private boolean assumeNonZeroTripLoops;
	private HashMap<Symbol, Symbol> l2gGVMap;
	private Stack<HashMap<Symbol, Symbol>> l2gGVMapStack;
	private HashMap<Procedure, Set<Symbol>> procWritesMap;
	private HashMap<Procedure, Set<Symbol>> procReadsMap;
	private String currentRegion;
	private Procedure main;
	private Set<Symbol> targetSymbols;
	private boolean IRSymbolOnly;
	private boolean barrierInserted;

	/**
	 * @param program
	 */
	public IpFirstAccessAnalysis(Program program, boolean IRSymOnly, Set<Symbol> targetSyms, boolean BARInserted) {
		super(program);
		IRSymbolOnly = IRSymOnly;
		if( targetSyms != null ) {
			targetSymbols = new HashSet<Symbol>(targetSyms);
		} else {
			targetSyms = null;
		}
		barrierInserted = BARInserted;
	}

	/* (non-Javadoc)
	 * @see cetus.analysis.AnalysisPass#getPassName()
	 */
	@Override
	public String getPassName() {
		return new String("[IpFirstAccessAnalysis]");
	}

	/* (non-Javadoc)
	 * @see cetus.analysis.AnalysisPass#start()
	 */
	@Override
	public void start() {
		assumeNonZeroTripLoops = false;
		String value = Driver.getOptionValue("assumeNonZeroTripLoops");
		if( value != null ) {
			assumeNonZeroTripLoops = true;
		}
		String mainEntryFunc = null;
		value = Driver.getOptionValue("SetAccEntryFunction");
		if( (value != null) && !value.equals("1") ) {
			mainEntryFunc = value;
		}
		
		l2gGVMapStack = new Stack<HashMap<Symbol, Symbol>>();
		
		main = AnalysisTools.findMainEntryFunction(program, mainEntryFunc);
		if( main == null ) {
			List<ACCAnnotation> dAnnots = AnalysisTools.collectPragmas(program, ACCAnnotation.class, 
					ACCAnnotation.dataRegions, false);
			Procedure cProc = null;
			if( dAnnots != null ) {
				for( ACCAnnotation dAn : dAnnots ) {
					Annotatable at = dAn.getAnnotatable();
					Procedure tProc = IRTools.getParentProcedure(at);
					if( cProc == null ) {
						cProc = tProc;
					} else {
						if( !cProc.getSymbolName().equals(tProc.getSymbolName())) {
							cProc = null;
							break;
						}
					}
				}
				if( cProc != null ) {
					//All data regions are in the same procedure, which will be a main entry.
					main = cProc;
				}
			}
		}
		if( main == null ) {
			//FIXME: for now, this assumes that input program contains a main procedure.
			Tools.exit("[ERROR in IpFirstAccessAnalysis] can't find a main-entry function; diable memory-transfer verification pass!");
		}
		
		if( !barrierInserted ) {
			AnalysisTools.markIntervalForComputeRegions(program);
		}
		// Collect shared variables, which are target of this analysis.
		if( targetSymbols == null ) {
			targetSymbols = AnalysisTools.getAllACCSharedVariables(program);
		}
		PrintTools.println("Symbols of interest: " + AnalysisTools.symbolsToString(targetSymbols, ","), 3);
		
		// Initialize currentRegion.
		currentRegion = new String("CPU");
		procWritesMap = new HashMap<Procedure, Set<Symbol>>();
		procReadsMap = new HashMap<Procedure, Set<Symbol>>();
		Set<Symbol> dummySet1 = new HashSet<Symbol>();
		Set<Symbol> dummySet2 = new HashSet<Symbol>();
		List<FunctionCall> fCallList = IRTools.getFunctionCalls(program);
		// Start interprocedural analysis from main() procedure.
		firstAccessAnalysis(main, dummySet1, dummySet2, currentRegion, fCallList, null);
		
		if( !barrierInserted ) {
			AnalysisTools.deleteBarriers(program);
		}

	}
	
	private boolean firstAccessAnalysis(Procedure proc, Set<Symbol> writeSet, Set<Symbol> readSet,
			String currentRegion, List<FunctionCall> fCallList, Set<Procedure> accessedProcs ) {
		if( accessedProcs == null ) {
			accessedProcs = new HashSet<Procedure>();
		}
		boolean AnnotationAdded = false;
		l2gGVMap = new HashMap<Symbol, Symbol>();
		Statement computeRegion = null;
		Set<Symbol> writeSet_in = new HashSet<Symbol>();
		Set<Symbol> readSet_in = new HashSet<Symbol>();
		///////////////////////////////////////////////////////////////////////////////////
		//If the same procedure is called with different context, current context should //
		//be intersected with the previous context conservatively.                       //
		///////////////////////////////////////////////////////////////////////////////////
		if( procWritesMap.containsKey(proc) ) {
			Set<Symbol> prevWSet = procWritesMap.get(proc);
			prevWSet.retainAll(writeSet);
			writeSet_in.addAll(prevWSet);
		} else {
			writeSet_in.addAll(writeSet);
			procWritesMap.put(proc, new HashSet<Symbol>(writeSet));
		}
		if( procReadsMap.containsKey(proc) ) {
			Set<Symbol> prevRSet = procReadsMap.get(proc);
			prevRSet.retainAll(readSet);
			readSet_in.addAll(prevRSet);
		} else {
			readSet_in.addAll(readSet);
			procReadsMap.put(proc, new HashSet<Symbol>(readSet));
		}
		
		PrintTools.println("[firstAccessAnalysis] analyze " + proc.getSymbolName(), 2);
		
		OCFGraph.setNonZeroTripLoops(assumeNonZeroTripLoops);
		//CFGraph cfg = new OCFGraph(proc, null);
		CFGraph cfg = new OCFGraph(proc, null, true);
		
		// sort the control flow graph
		cfg.topologicalSort(cfg.getNodeWith("stmt", "ENTRY"));
		
		TreeMap work_list = new TreeMap();
		
		// Enter the entry node in the work_list
		DFANode entry = cfg.getNodeWith("stmt", "ENTRY");
		Set<Symbol> writeSet_out = new HashSet<Symbol>(writeSet_in);
		Set<Symbol> readSet_out = new HashSet<Symbol>(readSet_in);
		entry.putData("writeSet_in", writeSet_in);
		entry.putData("writeSet_out", writeSet_out);
		entry.putData("readSet_in", readSet_in);
		entry.putData("readSet_out", readSet_out);
		//work_list.put(entry.getData("top-order"), entry);
		// work_list contains all nodes except for the entry node.
		for ( DFANode succ : entry.getSuccs() ) {
			work_list.put(succ.getData("top-order"), succ);
		}
		
		// Do iterative steps
		while ( !work_list.isEmpty() )
		{
			DFANode node = (DFANode)work_list.remove(work_list.firstKey());

			String tag = (String)node.getData("tag");
			// Check whether the node is in the kernel region or not.
			boolean isBarrierNode = false;
			if( tag != null && tag.equals("barrier") ) {
				isBarrierNode = true;
				String type = (String)node.getData("type");
				Statement bStmt = (Statement)node.getData("ir");
				CompoundStatement pStmt = (CompoundStatement)bStmt.getParent(); 
				if( type != null ) {
					if( type.equals("S2P") ) {
						currentRegion = new String("GPU");
						computeRegion = AnalysisTools.getStatementAfter(pStmt, bStmt);

					} else if( type.equals("P2S") ) {
						currentRegion = new String("CPU");
						computeRegion = null;
					}
				}
			}
			
			//PrintTools.println("[DEBUG in firstAccessAnalysis] curren region " + currentRegion + ", old current node:\n" + node + "\n", 0);
			
			writeSet_in = null;
			readSet_in = null;

			DFANode temp = (DFANode)node.getData("back-edge-from");
			for ( DFANode pred : node.getPreds() )
			{
				Set<Symbol> pred_writeSet_out = pred.getData("writeSet_out");
				if ( writeSet_in == null ) {
					if ( pred_writeSet_out != null ) {
						writeSet_in = new HashSet<Symbol>(pred_writeSet_out);
					}
				} else {
					// Calculate intersection of previous nodes.
					if ( pred_writeSet_out != null ) {
						if( (temp != null) && (temp == pred) ) {
							// this data is from a back-edge, union it with the current data
							// [DEBUG] to union, additional information such as multisrc should be added.
							// For now, conservatively intersected.
							//writeSet_in.addAll(pred_writeSet_out);
							writeSet_in.retainAll(pred_writeSet_out);
						} else {
							// this is an if-else branch, thus intersect it with the current data
							writeSet_in.retainAll(pred_writeSet_out);
						}
					}  else {
						//This is the first visit to this node.
						writeSet_in.clear();
					} 
				}
				Set<Symbol> pred_readSet_out = pred.getData("readSet_out");
				if ( readSet_in == null ) {
					if ( pred_readSet_out != null ) {
						readSet_in = new HashSet<Symbol>(pred_readSet_out);
					}
				} else {
					// Calculate intersection of previous nodes.
					if ( pred_readSet_out != null ) {
						if( (temp != null) && (temp == pred) ) {
							// this data is from a back-edge, union it with the current data
							// [DEBUG] to union, additional information such as multisrc should be added.
							// For now, conservatively intersected.
							//readSet_in.addAll(pred_readSet_out);
							readSet_in.retainAll(pred_readSet_out);
						} else {
							// this is an if-else branch, thus intersect it with the current data
							readSet_in.retainAll(pred_readSet_out);
						}
					}  else {
						//This is the first visit to this node.
						readSet_in.clear();
					} 
				}
			}

			// previous writeSet_in
			Set<Symbol> p_writeSet_in = node.getData("writeSet_in");
			// previous readSet_in
			Set<Symbol> p_readSet_in = node.getData("readSet_in");
			// previous writeSet_out
			Set<Symbol> p_writeSet_out = null;
			// previous readSet_in
			Set<Symbol> p_readSet_out = null;

			if ( (writeSet_in == null) || (p_writeSet_in == null) || !writeSet_in.equals(p_writeSet_in) ||
					(readSet_in == null) || (p_readSet_in == null) || !readSet_in.equals(p_readSet_in) ) {
				node.putData("writeSet_in", writeSet_in);
				node.putData("readSet_in", readSet_in);
				
				if( isBarrierNode && (computeRegion != null) ) {
					p_writeSet_out = node.getData("writeSet_out");
					p_readSet_out = node.getData("readSet_out");
				}

				// compute writeSet_out, a set of modified variables.
				writeSet_out = new HashSet<Symbol>();
				// compute readSet_out, a set of read variables.
				readSet_out = new HashSet<Symbol>();
				
				if( writeSet_in != null ) {
					writeSet_out.addAll(writeSet_in);
				}
				if( readSet_in != null ) {
					readSet_out.addAll(readSet_in);
				}

				////////////////////////////
				// Handle GEN & KILL set. //
				////////////////////////////
				Traversable ir = node.getData("ir");
				if( currentRegion.equals("CPU") ) {
					if( ir != null ) {
						//Handle GEN_WRITE set.
						Set<Symbol> tDefSyms = DataFlowTools.getDefSymbol(ir);
						Set<Symbol> defSyms = AnalysisTools.getIRSymbols(tDefSyms);
						if( ir instanceof ExpressionStatement ) {
							Expression expr = ((ExpressionStatement)ir).getExpression();
							//If LHS symbol is malloced at this node, this should not be in
							//a DefSyms set.
							if( expr instanceof BinaryExpression ) {
								Expression LHS = ((BinaryExpression)expr).getLHS();
								Expression RHS = ((BinaryExpression)expr).getRHS();
								Symbol LSym = SymbolTools.getSymbolOf(LHS);
								if( LSym instanceof AccessSymbol ) {
									LSym = ((AccessSymbol)LSym).getIRSymbol();
								}
								List<FunctionCall> fcalls = IRTools.getFunctionCalls(RHS);
								if( fcalls != null ) {
									for(FunctionCall tC : fcalls ) {
										String fName = tC.getName().toString();
										if( fName.equals("malloc") ) {
											defSyms.remove(LSym);
										} else if( fName.equals("calloc") ) {
											node.putData("callocsym", LSym);
										}
									}
								}
							}
						} else if( ir instanceof VariableDeclarator ) {
							VariableDeclarator vDeclr = (VariableDeclarator)ir;
							if( vDeclr.getInitializer() != null ) {
								defSyms.add(vDeclr);
							}
						}
						for( Symbol sym : defSyms ) {
							Symbol gSym = null;
							if( l2gGVMap.containsKey(sym) ) {
								gSym = l2gGVMap.get(sym);
							} else {
								List symbolInfo = new ArrayList(2);
								if( AnalysisTools.SymbolStatus.OrgSymbolFound(
										AnalysisTools.findOrgSymbol(sym, ir, true, null, symbolInfo, fCallList)) ) {
									gSym = (Symbol)symbolInfo.get(0);
									l2gGVMap.put(sym, gSym);
								}
							}
							if( (gSym != null) && targetSymbols.contains(gSym) ) {
								writeSet_out.add(gSym);
								Symbol LSym = (Symbol)node.getData("callocsym");
								if( LSym == sym ) {
									node.putData("callocsym", gSym);
								}
							}
						}
						//Handle GEN_READ/WRITE set.
						Set<Symbol> gUseSyms = new HashSet<Symbol>();
						Set<Symbol> gDefSyms = new HashSet<Symbol>();
						boolean simpleFuncCall = false;
						boolean isStandardLibraryCall = false;
						FunctionCall fCall = null;
						if( ir instanceof ExpressionStatement ) {
							Expression expr = ((ExpressionStatement)ir).getExpression();
							if( expr instanceof FunctionCall ) {
								fCall = (FunctionCall)expr;
								simpleFuncCall = true;
								//FIXME: If a function call is standard library, use cetus.hir.StandardLibray 
								//class to detect any side effect; if a parameter is written, use that into
								//DefSyms set.
								isStandardLibraryCall = StandardLibrary.contains(fCall);
								boolean isSideEffectFunction = false;
								int writeIndices [] = null;
								if( isStandardLibraryCall ) {
									isSideEffectFunction = StandardLibrary.hasSideEffectOnParameter(fCall);
									writeIndices = StandardLibrary.getSideEffectParamIndices(fCall);
/*									if( isSideEffectFunction ) {
										System.err.println("Found STL with side effect: " + fCall + "\n");
										for( int k=0; k<writeIndices.length; k++ ) {
											System.err.println("Write indices[" + k + "] = " + writeIndices[k] + "\n");
										}
									}*/
								}
								List<Expression> argList = (List<Expression>)fCall.getArguments();
								if( argList != null ) {
									int i = 0;
									for( Expression argExp : argList ) {
										boolean hasSideEffect = false;
										if( isStandardLibraryCall && isSideEffectFunction ) {
											for( int k=0; k<writeIndices.length; k++ ) {
												if( writeIndices[k] == i ) {
													hasSideEffect = true;
													break;
												}
											}
										}
										Set<Symbol> tUseSyms = DataFlowTools.getUseSymbol(argExp);
										Set<Symbol> useSyms = AnalysisTools.getIRSymbols(tUseSyms);
										for( Symbol sym : useSyms ) {
											Symbol gSym = null;
											if( l2gGVMap.containsKey(sym) ) {
												gSym = l2gGVMap.get(sym);
											} else {
												List symbolInfo = new ArrayList(2);
												if( AnalysisTools.SymbolStatus.OrgSymbolFound(
														AnalysisTools.findOrgSymbol(sym, ir, true, null, symbolInfo, fCallList)) ) {
													gSym = (Symbol)symbolInfo.get(0);
													l2gGVMap.put(sym, gSym);
												}
											}
											if( (gSym != null) && targetSymbols.contains(gSym) ) {
												////////////////////////////////////////////////////////////////////
												// DEBUG: If shared array variable is passed as a simple function //
												// argument, it is handled in the called function; skip it.       //
												// Exception #1: if the function is free(), we don't have to add  //
												// the shared variable into gUseSyms set.                         //
												// Exception #2: if the function is other standard library call,  //
												// the shared variable should be handled here.                    //
												////////////////////////////////////////////////////////////////////
												String fName = fCall.getName().toString();
												if( !fName.equals("free") ) {
													if( isStandardLibraryCall ) {
														if( hasSideEffect ) {
															gDefSyms.add(gSym);
														} else {
															gUseSyms.add(gSym);
														}
													} else if((!SymbolTools.isArray(gSym) && !SymbolTools.isPointer(gSym)) ) {
														gUseSyms.add(gSym);
													}
												}
											}
										}
										i++;
									}
								}
							}
						}
						if( !simpleFuncCall ) {
							Set<Symbol> tUseSyms = DataFlowTools.getUseSymbol(ir);
							Set<Symbol> useSyms = AnalysisTools.getIRSymbols(tUseSyms);
							for( Symbol sym : useSyms ) {
								Symbol gSym = null;
								if( l2gGVMap.containsKey(sym) ) {
									gSym = l2gGVMap.get(sym);
								} else {
									List symbolInfo = new ArrayList(2);
									if( AnalysisTools.SymbolStatus.OrgSymbolFound(
											AnalysisTools.findOrgSymbol(sym, ir, true, null, symbolInfo, fCallList)) ) {
										gSym = (Symbol)symbolInfo.get(0);
										l2gGVMap.put(sym, gSym);
									}
								}
								if( (gSym != null) && targetSymbols.contains(gSym) ) {
									gUseSyms.add(gSym);
								}
							}
						}
						readSet_out.addAll(gUseSyms);
						writeSet_out.addAll(gDefSyms);
						
						//////////////////////////////////////////////
						// Handle function calls interprocedurally. //
						//////////////////////////////////////////////
						if( ir instanceof ExpressionStatement ) {
							ExpressionStatement estmt = (ExpressionStatement)ir;
							Expression expr = estmt.getExpression();
							List<FunctionCall> fcalls = IRTools.getFunctionCalls(expr);
							if( fcalls !=null ) {
								for( FunctionCall funCall : fcalls ) {
									if( !isStandardLibraryCall ) {
										Procedure calledProc = funCall.getProcedure();
										if( (calledProc != null) && !accessedProcs.contains(calledProc) ) {
											accessedProcs.add(calledProc);
											l2gGVMapStack.push(l2gGVMap);
											if( firstAccessAnalysis(calledProc, writeSet_out, readSet_out, currentRegion, fCallList, accessedProcs) ) {
												AnnotationAdded = true;
											}
											l2gGVMap = l2gGVMapStack.pop();
										}
									}
								}
							}
						}
						
						//////////////////////////////////////////
						//Calculate First-Read/First-Write set. //
						//////////////////////////////////////////
						//////////////////////////////////////////////////////////////////////////
						// DEBUG: if this node is a function call, first-read/write set are not //
						// needed to be annotated, since they are annotated in its body.        //
						// FIXME: if this node contains assignment statement where a function   //
						// call is also included, first-read/write set related to the function  //
						// should not be included.                                              //
						// ==> simpleFuncCall does not have any LHS, and thus OK.               //
						//////////////////////////////////////////////////////////////////////////
						if( (!simpleFuncCall) || (simpleFuncCall && isStandardLibraryCall) ) {
							Set<Symbol> firstWriteSet = new HashSet<Symbol>();
							Set<Symbol> firstReadSet = new HashSet<Symbol>();
							firstWriteSet.addAll(writeSet_out);
							firstWriteSet.removeAll(writeSet_in);
							if( firstWriteSet.isEmpty() ) {
								node.removeData("firstwriteSet");
							} else {
								node.putData("firstwriteSet", firstWriteSet);
							}
							firstReadSet.addAll(readSet_out);
							firstReadSet.removeAll(readSet_in);
							firstReadSet.removeAll(writeSet_in);
							//DEBUG: firstWriteSet contains both R/W and W/O data.
							firstReadSet.removeAll(firstWriteSet);
							if( firstReadSet.isEmpty() ) {
								node.removeData("firstreadSet");
							} else {
								node.putData("firstreadSet", firstReadSet);
							}
						}
					}
				} else if( isBarrierNode ) {
					//this node is an entry barrier to a compute region (S2P).
					//CAUTION: this analysis assumes that LocalisyAnalysis is called before this.
					Set<Symbol> killSet = node.getData("killSet");
					if( killSet == null ) {
						killSet = new HashSet<Symbol>();
						node.putData("killSet", killSet);
						Set<Symbol> firstWriteSet = new HashSet<Symbol>();
						Set<Symbol> firstReadSet = new HashSet<Symbol>();
						ACCAnnotation iAnnot = computeRegion.getAnnotation(ACCAnnotation.class, "accshared");
						if( iAnnot != null ) {
							Set<Symbol> accShared = iAnnot.get("accshared");
							for( Symbol tSym : accShared ) {
								Symbol gSym = null;
								if( l2gGVMap.containsKey(tSym) ) {
									gSym = l2gGVMap.get(tSym);
								} else {
									List symbolInfo = new ArrayList(2);
									if( AnalysisTools.SymbolStatus.OrgSymbolFound(
											AnalysisTools.findOrgSymbol(tSym, computeRegion, true, null, symbolInfo, fCallList)) ) {
										gSym = (Symbol)symbolInfo.get(0);
										l2gGVMap.put(tSym, gSym);
									}
								}
								if( gSym != null ) {
									killSet.add(gSym);
									firstWriteSet.add(gSym);
									//firstReadSet.add(gSym);
								}
							}
							Set<Symbol> accReadOnly = iAnnot.get("accreadonly");
							for( Symbol tSym : accReadOnly ) {
								Symbol gSym = null;
								if( l2gGVMap.containsKey(tSym) ) {
									gSym = l2gGVMap.get(tSym);
								} else {
									List symbolInfo = new ArrayList(2);
									if( AnalysisTools.SymbolStatus.OrgSymbolFound(
											AnalysisTools.findOrgSymbol(tSym, computeRegion, true, null, symbolInfo, fCallList)) ) {
										gSym = (Symbol)symbolInfo.get(0);
										l2gGVMap.put(tSym, gSym);
									}
								}
								if( gSym != null ) {
									killSet.remove(gSym);
									firstWriteSet.remove(gSym);
									firstReadSet.add(gSym);
								}
							}
							if( !firstWriteSet.isEmpty() ) {
								node.putData("firstwriteSet", firstWriteSet);
							}
							if( !firstReadSet.isEmpty() ) {
								node.putData("firstreadSet", firstReadSet);
							}
						} else {
							ACCAnnotation cAnnot = computeRegion.getAnnotation(ACCAnnotation.class, "kernels");
							if( cAnnot == null ) {
								cAnnot = computeRegion.getAnnotation(ACCAnnotation.class, "parallel");
							}
							Tools.exit("[ERROR in IpFirstAccAnalysis()] Cannot find accshared internal clause for the following compute region:\n" +
									"OpenACCAnnotation: " + cAnnot +"\nEnclosing procedure: " + proc.getSymbolName() + "\n");
						}
					}
					writeSet_out.removeAll(killSet);
					readSet_out.removeAll(killSet);
				}

				node.putData("writeSet_out", writeSet_out);
				node.putData("readSet_out", readSet_out);
				
				//PrintTools.println("[DEBUG in firstAccessAnalysis] curren region " + currentRegion + ", new current node:\n" + node + "\n", 0);

				boolean checkSuccNodes = true;
				if( isBarrierNode && (computeRegion != null) ) {
					if( writeSet_out.equals(p_writeSet_out) && readSet_out.equals(p_readSet_out) ) {
						checkSuccNodes = false;
						//We will not visit the kernel next to the current barrier, and thus currentRegion should be back to "CPU".
						currentRegion = new String("CPU");
/*						PrintTools.println("[DEBUG in firstAccessAnalysis] curren region " + currentRegion + ", new current node:\n" + node + "\n" +
							"prev_writeSet_out: " + p_writeSet_out + "\nprev_readSet_out: " + p_readSet_out +"\n", 0);*/
					}
				}
				if( checkSuccNodes ) {
/*					PrintTools.println("[DEBUG in firstAccessAnalysis] curren region " + currentRegion + ", new current node:\n" + node + "\n" +
							"prev_writeSet_out: " + p_writeSet_out + "\nprev_readSet_out: " + p_readSet_out +"\n", 0);*/
					for ( DFANode succ : node.getSuccs() ) {
						work_list.put(succ.getData("top-order"), succ);
					}
				}
			} else if( isBarrierNode && (computeRegion != null) ) {
				//We will not visit the kernel next to the current barrier, and thus currentRegion should be back to "CPU".
				currentRegion = new String("CPU");
			}
		}
		// Create a new writeSet/readSet at the end of this procedure execution.
		writeSet.clear();
		readSet.clear();
		List<DFANode> exit_nodes = cfg.getExitNodes();
		boolean firstNode = true;
		// If multiple exit nodes exist, intersect writeSet_out sets.
		Set<Symbol> localSyms = SymbolTools.getLocalSymbols(proc.getBody());
		if( localSyms == null ) {
			localSyms = new HashSet<Symbol>();
		}
		for( DFANode exit_node : exit_nodes ) {
			Set<Symbol> lwset;
			lwset =	(Set<Symbol>)exit_node.getData("writeSet_out");
			Set<Symbol> lrset;
			lrset =	(Set<Symbol>)exit_node.getData("readSet_out");
			if( lwset == null ) {
				PrintTools.println("\n[WARNING in IpFirstAccessAnalysis()] the following exit node does not have writeSet_out set:\n" +
						exit_node.toString() + "\n", 1);
			} else if( lrset == null ) {
				PrintTools.println("\n[WARNING in IpFirstAccessAnalysis()] the following exit node does not have readSet_out set:\n" +
						exit_node.toString() + "\n", 1);
			} else {
				Set<Symbol> tlwset = new HashSet<Symbol>();
				for( Symbol tSym : lwset ) {
					if( !localSyms.contains(tSym) ) {
						tlwset.add(tSym);
					}
				}
				Set<Symbol> tlrset = new HashSet<Symbol>();
				for( Symbol tSym : lrset ) {
					if( !localSyms.contains(tSym) ) {
						tlrset.add(tSym);
					}
				}
				if( firstNode ) {
					writeSet.addAll(tlwset);
					readSet.addAll(tlrset);
					firstNode = false;
				} else {
					writeSet.retainAll(tlwset);
					readSet.retainAll(tlrset);
				}
			}
		}
		
		/////////////////////////////////////////////////////////////////////
		// Annotate nodes containing firstwriteSet/firstreadSet with       //
		// #pragma acc tempinternal firstwriteSet(list) firstreadSet(list) //
		// internal annotation, which will be removed at the end of this   //
		// analysis pass.                                                  //
		/////////////////////////////////////////////////////////////////////
		List<String> firstClauses = new ArrayList<String>(2);
		firstClauses.add("firstwriteSet");
		firstClauses.add("firstreadSet");
		Iterator<DFANode> iter = cfg.iterator();
		while ( iter.hasNext() )
		{
			DFANode node = iter.next();
			//PrintTools.println("[INFO in IpFirstAccessAnalysis] post node: " + node + "\n", 0);
			Object obj = node.getData(firstClauses);
			if( obj == null ) {
				continue;
			}
			//PrintTools.println("[INFO in IpFirstAccessAnalysis] first-access node: " + node + "\n", 0);
			Statement IRStmt = null;
			Set<Symbol> firstWriteSet = null;
			Set<Symbol> firstReadSet = null;
			firstWriteSet = node.getData("firstwriteSet");
			firstReadSet = node.getData("firstreadSet");
			obj = node.getData("ir");
			if( obj instanceof Statement ) {
				IRStmt = (Statement)obj;
			} else if( obj instanceof Expression ) {
				Traversable t = ((Expression)obj).getParent();
				while ( (t != null) && !(t instanceof Statement) ) {
					t = t.getParent();
				}
				if( t instanceof Statement ) {
					IRStmt = (Statement)t;
				} else {
					continue;
				}
			} else if( obj instanceof VariableDeclarator ) {
				Traversable t = ((VariableDeclarator)obj).getParent();
				while ( (t != null) && !(t instanceof Statement) ) {
					t = t.getParent();
				}
				if( t instanceof Statement ) {
					IRStmt = (Statement)t;
				} else {
					continue;
				}
			} else {
				continue;
			}
			//If current statement is an AnnotationStatement for a barrier, related compute region
			//should be annotated.
			boolean isComputeRegion = false;
			String tag = (String)node.getData("tag");
			if( (tag != null) && tag.equals("barrier") ) {
				CompoundStatement pStmt = (CompoundStatement)IRStmt.getParent();
				IRStmt = AnalysisTools.getStatementAfter(pStmt, IRStmt);
				isComputeRegion = true;
			} else {
				//If current statement is an init statement for a loop, the parent loop will be used.
				Traversable t = IRStmt.getParent();
				if( t instanceof Loop ) {
					IRStmt = (Statement)t;
				}
			}
			if( IRStmt != null ) {
				//PrintTools.println("[INFO in IpFirstAccessAnalysis] found first-access statement\n"
				//		+ IRStmt + "\n", 0);
				CompoundStatement pStmt = (CompoundStatement)IRStmt.getParent();
				Traversable pp = pStmt.getParent();
				Object ttO = node.getData("callocsym");
				boolean calloced = false;
				if( (ttO != null) && (firstWriteSet != null) && firstWriteSet.contains(ttO) ) {
					calloced = true;
				}
				if( !(pp instanceof Loop) || calloced ) { //IRStmt is not directly enclosed by a loop.
					ACCAnnotation tAnnot = IRStmt.getAnnotation(ACCAnnotation.class, "tempinternal");
					if( tAnnot == null ) {
						tAnnot = new ACCAnnotation("tempinternal", "_directive");
						IRStmt.annotate(tAnnot);
					}
					Set<Symbol> tSet = null;
					if( firstWriteSet != null ) {
						tSet = tAnnot.get("firstwriteSet");
						if( tSet == null ) {
							tSet = new HashSet<Symbol>(firstWriteSet);
							tAnnot.put("firstwriteSet", tSet);
						} else {
							tSet.addAll(firstWriteSet);
						}
						if( calloced ) {
							tAnnot.put("callocsym", ttO);
						}
					}
					tSet = null;
					if( firstReadSet != null ) {
						tSet = tAnnot.get("firstreadSet");
						if( tSet == null ) {
							tSet = new HashSet<Symbol>(firstReadSet);
							tAnnot.put("firstreadSet", tSet);
						} else {
							tSet.addAll(firstReadSet);
						}
					}
				} else { //IRStmt is directly enclosed by a loop.
					//If current statement is in a loop and it is loop invariant, the enclosing loop
					//can be annotated instead of the current statement.
					if( firstWriteSet != null ) {
						for( Symbol tSym : firstWriteSet ) {
							Statement targetStmt = IRStmt;
							pStmt = (CompoundStatement)targetStmt.getParent();
							pp = pStmt.getParent();
							while( (pp instanceof Loop) && !pStmt.containsSymbol(tSym) && 
									(!(pp instanceof ForLoop) || 
									((pp instanceof ForLoop) && !LoopTools.getLoopIndexSymbol((Loop)pp).equals(tSym))) ) {
								if( isComputeRegion ) {
									//System.err.print("==> Enter compute region\n");
									boolean codeMotionIsAllowed = true;
									boolean foundCurrentChild = false;
									List<Traversable> childList = pStmt.getChildren();
									for( Traversable child : childList ) {
										if( child instanceof AnnotationStatement ) {
											List<Annotation> tAList = ((AnnotationStatement)child).getAnnotations();
											if( (tAList.size() == 1) && (tAList.get(0) instanceof CommentAnnotation) ) {
												continue;
											} else {
												if( foundCurrentChild ) {
													//We don't have to check this.
													continue;
												} else {
													ACCAnnotation uAnnot = 
															((AnnotationStatement)child).getAnnotation(ACCAnnotation.class, "update");
													if( uAnnot != null ) {
														Set<SubArray> tSet = uAnnot.get("device");
														if( tSet != null ) {
															Set<Symbol> tSSet = AnalysisTools.subarraysToSymbols(tSet, IRSymbolOnly);
															if( tSSet.contains(tSym) ) {
																codeMotionIsAllowed = false;
																break;
															}
														}
														tSet = uAnnot.get("host");
														if( tSet != null ) {
															Set<Symbol> tSSet = AnalysisTools.subarraysToSymbols(tSet, IRSymbolOnly);
															if( tSSet.contains(tSym) ) {
																codeMotionIsAllowed = false;
																break;
															}
														}
													} else {
														continue;
													}
												}
											}

										} else if( child instanceof Annotatable ) {
											if( !foundCurrentChild && (targetStmt == child) ) {
												foundCurrentChild = true;
												continue;
											} else {
												Annotatable Att = (Annotatable)child;
												if( !Att.containsAnnotation(ACCAnnotation.class, "kernels") 
														&& !Att.containsAnnotation(ACCAnnotation.class, "parallel") ) {
													//[FIXME] below checking does not handle pointer aliasing to global variables.
													Set<Symbol> tSet = AnalysisTools.getAccessedVariables(Att, IRSymbolOnly);
													if( tSet.contains(tSym) ) {
														codeMotionIsAllowed = false;
													} else if( SymbolTools.isGlobal(tSym) ) {
														tSet = AnalysisTools.getAccessedGlobalSymbols(Att, null, true);
														if( tSet.contains(tSym) ) {
															codeMotionIsAllowed = false;
														}
													}
													if( !codeMotionIsAllowed ) {
														break;
													}
												} else {
													for( String dataClause : ACCAnnotation.memTrDataClauses ) {
														ACCAnnotation tDAnnot = Att.getAnnotation(ACCAnnotation.class, dataClause);
														if( tDAnnot != null ) {
															Set<SubArray> tSet = tDAnnot.get(dataClause);
															Set<Symbol> tSSet = AnalysisTools.subarraysToSymbols(tSet, IRSymbolOnly);
															if( tSSet.contains(tSym) ) {
																codeMotionIsAllowed = false;
																break;
															}
														}
													}
													if( !codeMotionIsAllowed ) {
														break;
													}
												}
											}
										} else {
											//Unexpected child is found.
											PrintTools.println("\n[WARNING in IpFirstAccessAnalysis()] unexpected child found:" +
													"child: " + child.toString() + "\n" +
													"Enclosing compoundStatement: " + pStmt + "\n", 0);
										}
									}
									if( codeMotionIsAllowed ) {
										//pStmt contains compute regions only.
										targetStmt = (Statement)pp;
										pStmt = (CompoundStatement)targetStmt.getParent();
										pp = pStmt.getParent();
										//System.err.print("==> enclosing loop contains compute regions only\n");
									} else {
										//System.err.print("==> enclosing loop contains non-compute regions\n");
										break;
									}
								} else if( !AnalysisTools.ipContainPragmas(pStmt, ACCAnnotation.class, ACCAnnotation.computeRegions, false, null) ) {
									//pStmt does not contain any compute region.
									//[DEBUG] to be more efficient, check each compute region whether it accesses the symbol or not.
									targetStmt = (Statement)pp;
									pStmt = (CompoundStatement)targetStmt.getParent();
									pp = pStmt.getParent();
								} else {
									break;
								}
							}
							ACCAnnotation tAnnot = targetStmt.getAnnotation(ACCAnnotation.class, "tempinternal");
							if( tAnnot == null ) {
								tAnnot = new ACCAnnotation("tempinternal", "_directive");
								targetStmt.annotate(tAnnot);
							}
							Set<Symbol> tSet = null;
							String firstWSetName;
							if( isComputeRegion && (IRStmt != targetStmt) ) {
								firstWSetName = "gfirstwriteSet";
							} else {
								firstWSetName = "firstwriteSet";
							}
							tSet = tAnnot.get(firstWSetName);
							if( tSet == null ) {
								tSet = new HashSet<Symbol>();
								tSet.add(tSym);
								tAnnot.put(firstWSetName, tSet);
							} else {
								tSet.add(tSym);
							}
						}
					}
					if( firstReadSet != null ) {
						for( Symbol tSym : firstReadSet ) {
							Statement targetStmt = IRStmt;
							if( !isComputeRegion ) {
								pStmt = (CompoundStatement)targetStmt.getParent();
								pp = pStmt.getParent();
								while( (pp instanceof Loop) && !pStmt.containsSymbol(tSym) && 
										(!(pp instanceof ForLoop) || ((pp instanceof ForLoop) &&
										!LoopTools.getLoopIndexSymbol((Loop)pp).equals(tSym)) ) ) {
									if( !AnalysisTools.ipContainPragmas(pStmt, ACCAnnotation.class, ACCAnnotation.computeRegions, false, null) ) {
										//pStmt does not contain any compute region.
										//[DEBUG] to be more efficient, check each compute region whether it accesses the symbol or not.
										targetStmt = (Statement)pp;
										pStmt = (CompoundStatement)targetStmt.getParent();
										pp = pStmt.getParent();
									} else {
										break;
									}
								}
							}
							ACCAnnotation tAnnot = targetStmt.getAnnotation(ACCAnnotation.class, "tempinternal");
							if( tAnnot == null ) {
								tAnnot = new ACCAnnotation("tempinternal", "_directive");
								targetStmt.annotate(tAnnot);
							}
							Set<Symbol> tSet = null;
							tSet = tAnnot.get("firstreadSet");
							if( tSet == null ) {
								tSet = new HashSet<Symbol>();
								tSet.add(tSym);
								tAnnot.put("firstreadSet", tSet);
							} else {
								tSet.add(tSym);
							}
						}
					}
				}
			}
		}
		
		PrintTools.println("[firstAccessAnalysis] analysis of " + proc.getSymbolName() + " ended", 2);
		return AnnotationAdded;
	}
}
