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
 * Inter-procedural analysis to identify memory transfers 
 *  
 * <p>
 * Input  : input program 
 * Output : set of OpenARC internal clauses(firstwriteSet, firstreadSet) annotated to the first statements
 *          that access OpenACC shared variables.
 *          For compute regions (kernels/parallel regions), these clauses will be attached only to each compute 
 *          region, but not to statements within compute regions.
 *          For regions outside of compute regions, these clauses will be attached to each last-write statement.
 *          (If the last-write statement is in a loop, the enclosing loop may be annotated if it is safe to move 
 *          the annotation upto the loop.)
 * <p>  
 *[Interprocedural first-access analysis]
 * liveCPU_out(program exit-node) = {}
 * mayKilledCPU_out(program exit-node) = {}
 * liveGPU_out(program exit-node) = {}
 * mayKilledGPU_out(program exit-node) = {}
 * cpuWriteSet_out(program exit-node) ={}
 * 
 *  for ( node m : successor nodes of node n ) {
 * 	    liveCPU_out(n)  += liveCPU_in(m) // + : union
 * 	    liveGPU_out(n)  += liveGPU_in(m) // + : union
 * 	    mayKilledCPU_out(n)  ^= mayKilledCPU_in(m) // ^ : intersection
 * 	    mayKilledGPU_out(n)  ^= mayKilledGPU_in(m) // ^ : intersection
 * 	    cpuWriteSet_out(n)  ^= cpuWriteSet_in(m) // ^ : intersection
 *  }
 *  if( n is in a compute region ) {
 *      liveGPU_in(n) = liveGPU_out(n) - GKILL(n) - GDEF(n) + GUSE(n) // + : union
 *      mayKilledGPU_in(n) = mayKilledGPU_out(n) -GKILL(n) + GDEF(n) -GUSE(n) // + : union
 *      liveCPU_in(n) = liveCPU_out(n) - CKILL(n) - CDEF(n) + CUSE(n) // + : union
 *      mayKilledCPU_in(n) = mayKilledCPU_out(n) -CKILL(n) + CDEF(n) - CUSE(n) // + : union
 *  } else {
 *      liveCPU_in(n) = liveCPU_out(n) - CKILL(n) - CDEF(n) + CUSE(n) // + : union
 *      mayKilledCPU_in(n) = mayKilledCPU_out(n) -CKILL(n) + CDEF(n) - CUSE(n) // + : union
 *      liveGPU_in(n) = liveGPU_out(n) - GKILL(n) - GDEF(n) + GUSE(n) // + : union
 *      mayKilledGPU_in(n) = mayKilledGPU_out(n) -GKILL(n) + GDEF(n) -GUSE(n) // + : union
 *  }
 *  cpuWriteSet_in(n) = cpuWriteSet_out(n) + CDEF(n) - GDEF(n) 
 *  lastWriteSet(n) = cpuWriteSet_in() - cpuWriteSet_out(n)
 *      where,
 *      CUSE(n) = a set of OpenACC shared variables that are read by CPU if a node n is not in a compute region
 *               {} if a node is in a compute region
 *      CDEF(n) = a set of OpenACC shared variables that are written by CPU if a node n is not in a compute region
 *               {} if a node is in a compute region
 *      GUSE(n) = a set of OpenACC shared variables that are read by GPU if a node n is in a compute region
 *               {} if a node is not in a compute region
 *      GDEF(n) = a set of OpenACC shared variables that are written by GPU if a node n is in a compute region
 *               {} if a node is not in a compute region
 *      CKILL(n) = a set of OpenACC shared variables that are written by GPU in a node n or or nodes before n
 *                 //CKILL(n)= mayKilledGPU_out(n) if a node n is not in a compute region
 *                 //          mayKilledGPU_in(n) + GDEF(n) if a node n is in a compute region
 *      GKILL(n) = a set of OpenACC shared variables that are written by CPU in a node n or or nodes before n
 *                 //GKILL(n)= mayKilledCPU_out(n) if a node n is in a compute region
 *                 //          mayKilledCPU_in(n) + CDEF(n) if a node n is not in a compute region
 * <p>  
 * 
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group
 *         Oak Ridge National Laboratory
 */
public class IpRedundantMemTrAnalysis extends AnalysisPass {
	private boolean assumeNonZeroTripLoops;
	private HashMap<Symbol, Symbol> l2gGVMap;
	private Stack<HashMap<Symbol, Symbol>> l2gGVMapStack;
	private HashMap<Procedure, Set<Symbol>> procLCPUMap;
	private HashMap<Procedure, Set<Symbol>> procLGPUMap;
	private HashMap<Procedure, Set<Symbol>> procMKCPUMap;
	private HashMap<Procedure, Set<Symbol>> procMKGPUMap;
	private HashMap<Procedure, Set<Symbol>> procCPUWMap;
	private String currentRegion;
	private Procedure main;
	private Set<Symbol> targetSymbols;
	private boolean IRSymbolOnly;
	private boolean barrierInserted;

	/**
	 * @param program
	 */
	public IpRedundantMemTrAnalysis(Program program, boolean IRSymOnly, Set<Symbol> targetSyms, boolean BARInserted) {
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
		return new String("[IpRedundantMemTrAnalysis]");
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
			Tools.exit("[ERROR in IpRedundantMemTrAnalysis] can't find a main-entry function; disable memory-transfer verification pass!");
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
		procLCPUMap = new HashMap<Procedure, Set<Symbol>>();
		procMKCPUMap = new HashMap<Procedure, Set<Symbol>>();
		procLGPUMap = new HashMap<Procedure, Set<Symbol>>();
		procMKGPUMap = new HashMap<Procedure, Set<Symbol>>();
		procCPUWMap = new HashMap<Procedure, Set<Symbol>>();
		Set<Symbol> dummySet1 = new HashSet<Symbol>();
		Set<Symbol> dummySet2 = new HashSet<Symbol>();
		Set<Symbol> dummySet3 = new HashSet<Symbol>();
		Set<Symbol> dummySet4 = new HashSet<Symbol>();
		Set<Symbol> dummySet5 = new HashSet<Symbol>();
		List<FunctionCall> fCallList = IRTools.getFunctionCalls(program);
		// Start interprocedural analysis from main() procedure.
		redundantMemTrAnalysis(main, dummySet1, dummySet2, dummySet3, dummySet4, dummySet5, currentRegion, fCallList, null);
		
		if( !barrierInserted ) {
			AnalysisTools.deleteBarriers(program);
		}

	}
	
	private boolean redundantMemTrAnalysis(Procedure proc, Set<Symbol> LCPUSet, Set<Symbol> MKCPUSet,
			Set<Symbol> LGPUSet, Set<Symbol> MKGPUSet, Set<Symbol> CPUWSet,
			String currentRegion, List<FunctionCall> fCallList, Set<Procedure> accessedProcs ) {
		if( accessedProcs == null ) {
			accessedProcs = new HashSet<Procedure>();
		}
		boolean AnnotationAdded = false;
		l2gGVMap = new HashMap<Symbol, Symbol>();
		Statement computeRegion = null;
		Set<Symbol> LCPUSet_out = new HashSet<Symbol>();
		Set<Symbol> MKCPUSet_out = new HashSet<Symbol>();
		Set<Symbol> LGPUSet_out = new HashSet<Symbol>();
		Set<Symbol> MKGPUSet_out = new HashSet<Symbol>();
		Set<Symbol> CPUWSet_out = new HashSet<Symbol>();
		///////////////////////////////////////////////////////////////////////////////////
		//If the same procedure is called with different context, current context should //
		//be unioned with the previous context conservatively.                           //
		///////////////////////////////////////////////////////////////////////////////////
		if( procLCPUMap.containsKey(proc) ) {
			Set<Symbol> prevLCPUSet = procLCPUMap.get(proc);
			prevLCPUSet.addAll(LCPUSet);
			LCPUSet_out.addAll(prevLCPUSet);
		} else {
			LCPUSet_out.addAll(LCPUSet);
			procLCPUMap.put(proc, new HashSet<Symbol>(LCPUSet));
		}
		if( procLGPUMap.containsKey(proc) ) {
			Set<Symbol> prevLGPUSet = procLGPUMap.get(proc);
			prevLGPUSet.addAll(LGPUSet);
			LGPUSet_out.addAll(prevLGPUSet);
		} else {
			LGPUSet_out.addAll(LGPUSet);
			procLGPUMap.put(proc, new HashSet<Symbol>(LGPUSet));
		}
		if( procMKCPUMap.containsKey(proc) ) {
			Set<Symbol> prevMKCPUSet = procMKCPUMap.get(proc);
			prevMKCPUSet.retainAll(MKCPUSet);
			MKCPUSet_out = new HashSet<Symbol>(prevMKCPUSet);
		} else {
			MKCPUSet_out.addAll(MKCPUSet);
			procMKCPUMap.put(proc, new HashSet<Symbol>(MKCPUSet));
		}
		if( procMKGPUMap.containsKey(proc) ) {
			Set<Symbol> prevMKGPUSet = procMKGPUMap.get(proc);
			prevMKGPUSet.retainAll(MKGPUSet);
			MKGPUSet_out = new HashSet<Symbol>(prevMKGPUSet);
		} else {
			MKGPUSet_out.addAll(MKGPUSet);
			procMKGPUMap.put(proc, new HashSet<Symbol>(MKGPUSet));
		}
		if( procCPUWMap.containsKey(proc) ) {
			Set<Symbol> prevCPUWSet = procCPUWMap.get(proc);
			prevCPUWSet.retainAll(CPUWSet);
			CPUWSet_out = new HashSet<Symbol>(prevCPUWSet);
		} else {
			CPUWSet_out.addAll(CPUWSet);
			procCPUWMap.put(proc, new HashSet<Symbol>(CPUWSet));
		}
		
		PrintTools.println("[redundantMemTrAnalysis] analyze " + proc.getSymbolName(), 2);
		
		OCFGraph.setNonZeroTripLoops(assumeNonZeroTripLoops);
		//CFGraph cfg = new OCFGraph(proc, null);
		CFGraph cfg = new OCFGraph(proc, null, true);
		
		// sort the control flow graph
		cfg.topologicalSort(cfg.getNodeWith("stmt", "ENTRY"));
		
		TreeMap work_list = new TreeMap();
		
		// Enter the exit node in the work_list
		List<DFANode> exit_nodes = cfg.getExitNodes();
		if (exit_nodes.size() > 1)
		{
			PrintTools.println("\n[WARNING in gLiveCVarAnalysis] multiple exits in the program.\n", 1);
		}

		Set<Symbol> LCPUSet_in = null;
		Set<Symbol> LGPUSet_in = null;
		Set<Symbol> MKCPUSet_in = null;
		Set<Symbol> MKGPUSet_in = null;
		Set<Symbol> CPUWSet_in = null;
		for ( DFANode exit_node : exit_nodes ) {
			exit_node.putData("LCPUSet_out", new HashSet<Symbol>(LCPUSet_out));
			exit_node.putData("LGPUSet_out", new HashSet<Symbol>(LGPUSet_out));
			exit_node.putData("MKCPUSet_out", new HashSet<Symbol>(MKCPUSet_out));
			exit_node.putData("MKGPUSet_out", new HashSet<Symbol>(MKGPUSet_out));
			exit_node.putData("CPUWSet_out", new HashSet<Symbol>(CPUWSet_out));
			work_list.put(exit_node.getData("top-order"), exit_node);
		}
		
		Set<Symbol> USE = new HashSet<Symbol>();
		Set<Symbol> DEF = new HashSet<Symbol>();
		// Do iterative steps
		while ( !work_list.isEmpty() )
		{
			DFANode node = (DFANode)work_list.remove(work_list.lastKey());

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
						currentRegion = new String("CPU");
						computeRegion = null;
					} else if( type.equals("P2S") ) {
						currentRegion = new String("GPU");
						computeRegion = AnalysisTools.getStatementBefore(pStmt, bStmt);
					}
				}
			}
			
			//PrintTools.println("[DEBUG in redundantMemTrAnalysis] curren region " + currentRegion + ", old current node:\n" + node + "\n", 0);
			
			LCPUSet_out = null;
			LGPUSet_out = null;
			MKCPUSet_out = null;
			MKGPUSet_out = null;
			CPUWSet_out = null;
			
			// previous LCPUSet_out
			Set<Symbol> p_LCPUSet_out = node.getData("LCPUSet_out");
			Set<Symbol> p_LGPUSet_out = node.getData("LGPUSet_out");
			Set<Symbol> p_MKCPUSet_out = node.getData("MKCPUSet_out");
			Set<Symbol> p_MKGPUSet_out = node.getData("MKGPUSet_out");
			Set<Symbol> p_CPUWSet_out = node.getData("CPUWSet_out");
			
			if( exit_nodes.contains(node) ) {
				LCPUSet_out = p_LCPUSet_out;
				p_LCPUSet_out = null;
				LGPUSet_out = p_LGPUSet_out;
				p_LGPUSet_out = null;
				MKCPUSet_out = p_MKCPUSet_out;
				p_MKCPUSet_out = null;
				MKGPUSet_out = p_MKGPUSet_out;
				p_MKGPUSet_out = null;
				CPUWSet_out = p_CPUWSet_out;
				p_CPUWSet_out = null;
			} else {
				LCPUSet_out = new HashSet<Symbol>();
				LGPUSet_out = new HashSet<Symbol>();
				//MKCPUSet_out = new HashSet<Symbol>();
				//MKGPUSet_out = new HashSet<Symbol>();
				//CPUWSet_out = new HashSet<Symbol>();

				for ( DFANode succ : node.getSuccs() )
				{
					HashSet<Symbol> succ_LCPUSet_in = succ.getData("LCPUSet_in");
					//At the beginning, some nodes may not have LCPUSet_in data.
					if( succ_LCPUSet_in != null ) {
						LCPUSet_out.addAll(succ_LCPUSet_in);
					}
					HashSet<Symbol> succ_LGPUSet_in = succ.getData("LGPUSet_in");
					//At the beginning, some nodes may not have LGPUSet_in data.
					if( succ_LGPUSet_in != null ) {
						LGPUSet_out.addAll(succ_LGPUSet_in);
					}
					HashSet<Symbol> succ_MKCPUSet_in = succ.getData("MKCPUSet_in");
					if( MKCPUSet_out == null ) {
						//At the beginning, some nodes may not have MKCPUSet_in data.
						if( succ_MKCPUSet_in != null ) {
							MKCPUSet_out = new HashSet<Symbol>();
							MKCPUSet_out.addAll(succ_MKCPUSet_in);
						}
					} else {
						if( succ_MKCPUSet_in != null ) {
							MKCPUSet_out.retainAll(succ_MKCPUSet_in);
						} else {
							MKCPUSet_out.clear();
						}
					}
					HashSet<Symbol> succ_MKGPUSet_in = succ.getData("MKGPUSet_in");
					if( MKGPUSet_out == null ) {
						//At the beginning, some nodes may not have MKGPUSet_in data.
						if( succ_MKGPUSet_in != null ) {
							MKGPUSet_out = new HashSet<Symbol>();
							MKGPUSet_out.addAll(succ_MKGPUSet_in);
						}
					} else {
						if( succ_MKGPUSet_in != null ) {
							MKGPUSet_out.retainAll(succ_MKGPUSet_in);
						} else {
							MKGPUSet_out.clear();
						}
					}
					HashSet<Symbol> succ_CPUWSet_in = succ.getData("CPUWSet_in");
					if( CPUWSet_out == null ) {
						//At the beginning, some nodes may not have MKGPUSet_in data.
						if( succ_CPUWSet_in != null ) {
							CPUWSet_out = new HashSet<Symbol>();
							CPUWSet_out.addAll(succ_CPUWSet_in);
						}
					} else {
						if( succ_CPUWSet_in != null ) {
							CPUWSet_out.retainAll(succ_CPUWSet_in);
						} else {
							CPUWSet_out.clear();
						}
					}
				}
			}
			
			// previous LCPUSet_in
			HashSet<Symbol> p_LCPUSet_in = null;
			HashSet<Symbol> p_LGPUSet_in = null;
			HashSet<Symbol> p_MKCPUSet_in = null;
			HashSet<Symbol> p_MKGPUSet_in = null;
			HashSet<Symbol> p_CPUWSet_in = null;

			if ( (p_LCPUSet_out == null) || !LCPUSet_out.equals(p_LCPUSet_out) ||
					(p_LGPUSet_out == null) || !LGPUSet_out.equals(p_LGPUSet_out) ||
					(MKCPUSet_out == null) || (p_MKCPUSet_out == null) || !MKCPUSet_out.equals(p_MKCPUSet_out) ||
					(MKGPUSet_out == null) || (p_MKGPUSet_out == null) || !MKGPUSet_out.equals(p_MKGPUSet_out) ||
					(CPUWSet_out == null) || (p_CPUWSet_out == null) || !CPUWSet_out.equals(p_CPUWSet_out) ) {
				node.putData("LCPUSet_out", LCPUSet_out);
				node.putData("LGPUSet_out", LGPUSet_out);
				node.putData("MKCPUSet_out", MKCPUSet_out);
				node.putData("MKGPUSet_out", MKGPUSet_out);
				node.putData("CPUWSet_out", CPUWSet_out);
				
				if( isBarrierNode && (computeRegion != null) ) {
					p_LCPUSet_in = node.getData("LCPUSet_in");
					p_LGPUSet_in = node.getData("LGPUSet_in");
					p_MKCPUSet_in = node.getData("MKCPUSet_in");
					p_MKGPUSet_in = node.getData("MKGPUSet_in");
					p_CPUWSet_in = node.getData("CPUWSet_in");
				}

				//LCPUSet_in = LCPUSet_out -CDEF + CUSE - MKGPUSet_out
				LCPUSet_in = new HashSet<Symbol>();
				LGPUSet_in = new HashSet<Symbol>();
				MKCPUSet_in = new HashSet<Symbol>();
				MKGPUSet_in = new HashSet<Symbol>();
				CPUWSet_in = new HashSet<Symbol>();
				if( LCPUSet_out != null ) {
					LCPUSet_in.addAll(LCPUSet_out);
				}
				if( LGPUSet_out != null ) {
					LGPUSet_in.addAll(LGPUSet_out);
				}
				if( MKCPUSet_out != null ) {
					MKCPUSet_in.addAll(MKCPUSet_out);
				}
				if( MKGPUSet_out != null ) {
					MKGPUSet_in.addAll(MKGPUSet_out);
				}
				if( CPUWSet_out != null ) {
					CPUWSet_in.addAll(CPUWSet_out);
				}

				////////////////////////////////
				// Calculate USE/DEF/KILL Set //
				////////////////////////////////
				USE.clear();
				DEF.clear();
				Traversable ir = node.getData("ir");
				if( currentRegion.equals("CPU") ) {
					boolean simpleFuncCall = false;
					boolean isStandardLibraryCall = false;
					FunctionCall fCall = null;
					if( ir != null ) {
						//Calculate CDEF set.
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
						} else if ( ir instanceof VariableDeclarator ) {
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
								DEF.add(gSym);
								Symbol LSym = (Symbol)node.getData("callocsym");
								if( LSym == sym ) {
									node.putData("callocsym", gSym);
								}
							}
						}
						//Calculate CUSE/CDEF set
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
								}
								List<Expression> argList = (List<Expression>)fCall.getArguments();
								if( argList != null ) {
									int i=0;
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
															DEF.add(gSym);
														} else {
															USE.add(gSym);
														}
													} else if((!SymbolTools.isArray(gSym) && !SymbolTools.isPointer(gSym)) ) {
														USE.add(gSym);
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
									USE.add(gSym);
								}
							}
						}
						
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
											if( redundantMemTrAnalysis(calledProc, LCPUSet_in, MKCPUSet_in, LGPUSet_in,
													MKGPUSet_in, CPUWSet_in, currentRegion, fCallList, accessedProcs) ) {
												AnnotationAdded = true;
											}
											l2gGVMap = l2gGVMapStack.pop();
										}
									}
								}
							}
						}
					}
					//LCPUSet_in = LCPUSet_out -MKGPUSet_out -GDEF -CDEF + CUSE
					LCPUSet_in.removeAll(MKGPUSet_in); //at this point, MKGPUSet_in == MKGPUSet_out.
					LCPUSet_in.removeAll(DEF);
					LCPUSet_in.addAll(USE);
					//MKCPUSet_in = MKCPUSet_out - MKGPUSet_out -GDEF + CDEF -CUSE
					MKCPUSet_in.removeAll(MKGPUSet_in); //at this point, MKGPUSet_in == MKGPUSet_out.
					MKCPUSet_in.addAll(DEF);
					MKCPUSet_in.removeAll(USE);
					//LGPUSet_in = LGPUSet_out - MKCPUSet_in -CDEF -GDEF + GUSE
					LGPUSet_in.removeAll(MKCPUSet_in);
					LGPUSet_in.removeAll(DEF);
					//MKGPUSet_in = MKGPUSet_out - MKCPUSet_in -CDEF + GDEF -GUSE
					MKGPUSet_in.removeAll(MKCPUSet_in);
					MKGPUSet_in.removeAll(DEF);
					// cpuWriteSet_in(n) = cpuWriteSet_out(n) + CDEF(n) - GDEF(n) 
					CPUWSet_in.addAll(DEF);
					// lastWriteSet(n) = cpuWriteSet_in() - cpuWriteSet_out(n)
					Set<Symbol> lastWriteSet = new HashSet<Symbol>(CPUWSet_in);
					if( CPUWSet_out != null ) {
						lastWriteSet.removeAll(CPUWSet_out);
					}
					//////////////////////////////////////////////////////////////////////////
					// DEBUG: if this node is a function call, last-write set are not       //
					// needed to be annotated, since they are annotated in its body.        //
					// FIXME: if this node contains assignment statement where a function   //
					// call is also included, last-write set related to the function        //
					// should not be included.                                              //
					//////////////////////////////////////////////////////////////////////////
					if( (!simpleFuncCall) || (simpleFuncCall && StandardLibrary.contains(fCall)) ) {
						if( !lastWriteSet.isEmpty() ) {
							node.putData("lastWriteSet", lastWriteSet);
						}
					}
				} else if( isBarrierNode ) {
					//this node is an exit barrier to a compute region (P2S).
					//CAUTION: this analysis assumes that LocalisyAnalysis is called before this.
					Set<Symbol> GDEF = node.getData("GDEF");
					Set<Symbol> GUSE = node.getData("GUSE");
					if( GDEF == null ) {
						GDEF = new HashSet<Symbol>();
						GUSE = new HashSet<Symbol>();
						node.putData("GDEF", GDEF);
						node.putData("GUSE", GUSE);
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
									GUSE.add(gSym);
									GDEF.add(gSym);
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
											AnalysisTools.findOrgSymbol(tSym, computeRegion, true, null, 
													symbolInfo, fCallList)) ) {
										gSym = (Symbol)symbolInfo.get(0);
										l2gGVMap.put(tSym, gSym);
									}
								}
								if( gSym != null ) {
									GUSE.add(gSym);
									GDEF.remove(gSym);
								}
							}
						} else {
							ACCAnnotation cAnnot = computeRegion.getAnnotation(ACCAnnotation.class, "kernels");
							if( cAnnot == null ) {
								cAnnot = computeRegion.getAnnotation(ACCAnnotation.class, "parallel");
							}
							Tools.exit("[ERROR in IpRedundantMemTrAnalysis()] Cannot find accshared internal clause " +
									"for the following compute region:\n" +
									"OpenACCAnnotation: " + cAnnot +"\nEnclosing procedure: " + 
									proc.getSymbolName() + "\n");
						}
					}
					//LGPUSet_in = LGPUSet_out -MKCPUSet_out -CDEF -GDEF + GUSE
					//[DEBUG] calculation of LGPUSet_in is changed as following, since GDEF and GUSE sets are 
					//summary of whole compute region rather than each statement.
					//    - LGPUSet_in = LGPUSet_out -MKCPUSet_out -CDEF +GUSE -GDEF
					//====> Switch back to original version.
					LGPUSet_in.removeAll(MKCPUSet_in); //at this point, MKCPUSet_in == MKCPUSet_out
					LGPUSet_in.removeAll(GDEF);
					LGPUSet_in.addAll(GUSE);
					//MKGPUSet_in = MKGPUSet_out - MKCPUSet_out -CDEF + GDEF -GUSE
					//[DEBUG] calculation of MKGPUSet_in is changed as following, since GDEF and GUSE sets are 
					//summary of whole compute region rather than each statement.
					//    - MKGPUSet_in = MKGPUSet_out - MKCPUSet_out -CDEF -GUSE + GDEF
					MKGPUSet_in.removeAll(MKCPUSet_in); //at this point, MKCPUSet_in == MKCPUSet_out
					MKGPUSet_in.removeAll(GUSE);
					MKGPUSet_in.addAll(GDEF);
					//LCPUSet_in = LCPUSet_out -MKGPUSet_in -GDEF -CDEF + CUSE
					LCPUSet_in.removeAll(MKGPUSet_in);
					LCPUSet_in.removeAll(GDEF);
					//MKCPUSet_in = MKCPUSet_out - MKGPUSet_in -GDEF + CDEF -CUSE
					MKCPUSet_in.removeAll(MKGPUSet_in);
					MKCPUSet_in.removeAll(GDEF);
					// cpuWriteSet_in(n) = cpuWriteSet_out(n) + CDEF(n) - GDEF(n) 
					CPUWSet_in.removeAll(GDEF);
					// lastWriteSet(n) = GDEF(n)
					Set<Symbol> lastWriteSet = new HashSet<Symbol>(GDEF);
					if( !lastWriteSet.isEmpty() ) {
						node.putData("lastWriteSet", lastWriteSet);
					}
				}

				node.putData("LCPUSet_in", LCPUSet_in);
				node.putData("LGPUSet_in", LGPUSet_in);
				node.putData("MKCPUSet_in", MKCPUSet_in);
				node.putData("MKGPUSet_in", MKGPUSet_in);
				node.putData("CPUWSet_in", CPUWSet_in);
				
				//PrintTools.println("[DEBUG in redundantMemTrAnalysis] current region " + currentRegion + 
				//", new current node:\n" + node + "\n", 0);

				boolean checkPrevNodes = true;
				if( isBarrierNode && (computeRegion != null) ) {
					if( LCPUSet_in.equals(p_LCPUSet_in) && LGPUSet_in.equals(p_LGPUSet_in) &&
							MKCPUSet_in.equals(p_MKCPUSet_in) && MKGPUSet_in.equals(p_MKGPUSet_in) &&
							CPUWSet_in.equals(p_CPUWSet_in)) {
						checkPrevNodes = false;
						//We will not visit the kernel next to the current barrier, and thus currentRegion 
						//should be back to "CPU".
						currentRegion = new String("CPU");
/*						PrintTools.println("[DEBUG in redundantMemTrAnalysis] current region " + currentRegion + 
 * 							", new current node:\n" + node + "\n" +
							"prev_writeSet_out: " + p_writeSet_out + "\nprev_readSet_out: " + p_readSet_out +"\n", 0);*/
					}
				}
				if( checkPrevNodes ) {
/*					PrintTools.println("[DEBUG in redundantMemTrAnalysis] curren region " + currentRegion + 
 * 							", new current node:\n" + node + "\n" +
							"prev_writeSet_out: " + p_writeSet_out + "\nprev_readSet_out: " + p_readSet_out +"\n", 0);*/
					for ( DFANode pred : node.getPreds() ) {
						work_list.put(pred.getData("top-order"), pred);
					}
				}
			} else if( isBarrierNode && (computeRegion != null) ) {
				//We will not visit the kernel next to the current barrier, and thus currentRegion should be back to "CPU".
				currentRegion = new String("CPU");
			}
		}
		
		//Update input sets to this analysis.
		LCPUSet.clear();
		LGPUSet.clear();
		MKCPUSet.clear();
		MKGPUSet.clear();
		CPUWSet.clear();
		Set<Symbol> localSyms = SymbolTools.getLocalSymbols(proc.getBody());
		if( localSyms == null ) {
			localSyms = new HashSet<Symbol>();
		}
		List<DFANode> entry_nodes = cfg.getEntryNodes();
		boolean firstNode = true;
		for( DFANode entry : entry_nodes ) {
			//LCPUSet.addAll((Set<Symbol>)entry.getData("LCPUSet_in"));
			Set<Symbol> ttSet = (Set<Symbol>)entry.getData("LCPUSet_in");
			Set<Symbol> tinset = new HashSet<Symbol>();
			for( Symbol tSym : ttSet ) {
				if( !localSyms.contains(tSym) ) {
					tinset.add(tSym);
				}
			}
			LCPUSet.addAll(tinset);
			//LGPUSet.addAll((Set<Symbol>)entry.getData("LGPUSet_in"));
			ttSet = (Set<Symbol>)entry.getData("LGPUSet_in");
			tinset = new HashSet<Symbol>();
			for( Symbol tSym : ttSet ) {
				if( !localSyms.contains(tSym) ) {
					tinset.add(tSym);
				}
			}
			LGPUSet.addAll(tinset);
			if( firstNode ) {
				//MKCPUSet.addAll((Set<Symbol>)entry.getData("MKCPUSet_in"));
				ttSet = (Set<Symbol>)entry.getData("MKCPUSet_in");
				tinset = new HashSet<Symbol>();
				for( Symbol tSym : ttSet ) {
					if( !localSyms.contains(tSym) ) {
						tinset.add(tSym);
					}
				}
				MKCPUSet.addAll(tinset);
				//MKGPUSet.addAll((Set<Symbol>)entry.getData("MKGPUSet_in"));
				ttSet = (Set<Symbol>)entry.getData("MKGPUSet_in");
				tinset = new HashSet<Symbol>();
				for( Symbol tSym : ttSet ) {
					if( !localSyms.contains(tSym) ) {
						tinset.add(tSym);
					}
				}
				MKGPUSet.addAll(tinset);
				//CPUWSet.addAll((Set<Symbol>)entry.getData("CPUWSet_in"));
				ttSet = (Set<Symbol>)entry.getData("CPUWSet_in");
				tinset = new HashSet<Symbol>();
				for( Symbol tSym : ttSet ) {
					if( !localSyms.contains(tSym) ) {
						tinset.add(tSym);
					}
				}
				CPUWSet.addAll(tinset);
				firstNode = false;
			} else {
				//MKCPUSet.retainAll((Set<Symbol>)entry.getData("MKCPUSet_in"));
				ttSet = (Set<Symbol>)entry.getData("MKCPUSet_in");
				tinset = new HashSet<Symbol>();
				for( Symbol tSym : ttSet ) {
					if( !localSyms.contains(tSym) ) {
						tinset.add(tSym);
					}
				}
				MKCPUSet.retainAll(tinset);
				//MKGPUSet.retainAll((Set<Symbol>)entry.getData("MKGPUSet_in"));
				ttSet = (Set<Symbol>)entry.getData("MKGPUSet_in");
				tinset = new HashSet<Symbol>();
				for( Symbol tSym : ttSet ) {
					if( !localSyms.contains(tSym) ) {
						tinset.add(tSym);
					}
				}
				MKGPUSet.retainAll(tinset);
				//CPUWSet.retainAll((Set<Symbol>)entry.getData("CPUWSet_in"));
				ttSet = (Set<Symbol>)entry.getData("CPUWSet_in");
				tinset = new HashSet<Symbol>();
				for( Symbol tSym : ttSet ) {
					if( !localSyms.contains(tSym) ) {
						tinset.add(tSym);
					}
				}
				CPUWSet.retainAll(tinset);
			}
		}
		
		///////////////////////////////////////////////////////////
		// Annotate nodes containing lastWriteSet data with      //
		// #pragma acc tempinternal maykilled(list) ueuse(list)  //
		// internal annotation, which will be removed at the end //
		// of this  analysis pass.                               //
		// [CAUTION] this assumes that IpFirstAccessAnalysis is  //
		// called before this analysis.                          //
		///////////////////////////////////////////////////////////
		boolean isComputeRegion = false;
		Iterator<DFANode> iter = cfg.iterator();
		while ( iter.hasNext() )
		{
			DFANode node = iter.next();
			//PrintTools.println("[INFO in IpRedundantMemTrAnalysis] post node: " + node + "\n", 0);
			Object obj = node.getData("lastWriteSet");
			if( obj == null ) {
				continue;
			}
			//PrintTools.println("[INFO in IpRedundantMemTrAnalysis] first-access node: " + node + "\n", 0);
			Statement IRStmt = null;
			Set<Symbol> lastWriteSet = (Set<Symbol>)obj;
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
			isComputeRegion = false;
			String tag = (String)node.getData("tag");
			if( (tag != null) && tag.equals("barrier") ) {
				isComputeRegion = true;
				CompoundStatement pStmt = (CompoundStatement)IRStmt.getParent();
				IRStmt = AnalysisTools.getStatementBefore(pStmt, IRStmt);
			} else {
				//If current statement is an init statement for a loop, the parent loop will be used.
				Traversable t = IRStmt.getParent();
				if( t instanceof Loop ) {
					IRStmt = (Statement)t;
				}
			}
			if( IRStmt != null ) {
				//PrintTools.println("[INFO in IpRedundantMemTrAnalysis] found last-write statement:\n"
				//		+ IRStmt + "\n", 0);
				Set<Symbol> tLiveSet = null;
				Set<Symbol> tMayKilledSet = null;
				if( isComputeRegion ) {
					tLiveSet = node.getData("LCPUSet_out");
					tMayKilledSet = node.getData("MKCPUSet_out");
				} else {
					tLiveSet = node.getData("LGPUSet_out");
					tMayKilledSet = node.getData("MKGPUSet_out");
				}
				if( tLiveSet == null ) {
					PrintTools.println("\n[WARNING in IpRedundantMemTrAnalysis] no LCPUSet_out/LGPUSet_out data found!\n", 0);
					tLiveSet = new HashSet<Symbol>();
				}
				if( tMayKilledSet == null ) {
					PrintTools.println("\n[WARNING in IpRedundantMemTrAnalysis] no MKCPUSet_out/MKGPUSet_out data found!\n", 0);
					tMayKilledSet = new HashSet<Symbol>();
				}
				CompoundStatement pStmt = (CompoundStatement)IRStmt.getParent();
				Traversable pp = pStmt.getParent();
				Object ttO = node.getData("callocsym");
				boolean calloced = false;
				if( (ttO != null) && (lastWriteSet != null) && lastWriteSet.contains(ttO) ) {
					calloced = true;
				}
				if( ((tag != null) && tag.equals("barrier")) || !(pp instanceof Loop) ||
						calloced ) { //compute region or IRStmt is
					                                                                      //not directly enclosed by a loop.
					ACCAnnotation tAnnot = IRStmt.getAnnotation(ACCAnnotation.class, "tempinternal");
					if( tAnnot == null ) {
						tAnnot = new ACCAnnotation("tempinternal", "_directive");
						IRStmt.annotate(tAnnot);
					}
					Set<Symbol> mayKilledSet = tAnnot.get("maykilled");
					if( mayKilledSet == null ) {
						mayKilledSet = new HashSet<Symbol>();
					}
					Set<Symbol> deadSet = tAnnot.get("dead");
					if( deadSet == null ) {
						deadSet = new HashSet<Symbol>();
					}
					if( lastWriteSet != null ) {
						for( Symbol wSym : lastWriteSet ) {
							if( tMayKilledSet.contains(wSym) ) {
								mayKilledSet.add(wSym);
							} else if( !tLiveSet.contains(wSym) ) {
								deadSet.add(wSym);
							}
						}
						if( !mayKilledSet.isEmpty() ) {
							tAnnot.put("maykilled", mayKilledSet);
						}
						if( !deadSet.isEmpty() ) {
							tAnnot.put("dead", deadSet);
						}
						if( calloced ) {
							tAnnot.put("callocsym", ttO);
						}
					}
				} else { //non-compute region and IRStmt is directly enclosed by a loop.
					//If current statement is in a loop and it is loop invariant, the enclosing loop
					//can be annotated instead of the current statement.
					if( lastWriteSet != null ) {
						for( Symbol tSym : lastWriteSet ) {
							Statement targetStmt = IRStmt;
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
							ACCAnnotation tAnnot = targetStmt.getAnnotation(ACCAnnotation.class, "tempinternal");
							if( tAnnot == null ) {
								tAnnot = new ACCAnnotation("tempinternal", "_directive");
								targetStmt.annotate(tAnnot);
							}
							Set<Symbol> mayKilledSet = tAnnot.get("maykilled");
							if( mayKilledSet == null ) {
								mayKilledSet = new HashSet<Symbol>();
							}
							Set<Symbol> deadSet = tAnnot.get("dead");
							if( deadSet == null ) {
								deadSet = new HashSet<Symbol>();
							}
							if( tMayKilledSet.contains(tSym) ) {
								mayKilledSet.add(tSym);
							} else if( !tLiveSet.contains(tSym) ) {
								deadSet.add(tSym);
							}
							if( !mayKilledSet.isEmpty() ) {
								tAnnot.put("maykilled", mayKilledSet);
							}
							if( !deadSet.isEmpty() ) {
								tAnnot.put("dead", deadSet);
							}
						}
					}
				}
			}
		}
		
		PrintTools.println("[redundantMemTrAnalysis] analysis of " + proc.getSymbolName() + " ended", 2);
		return AnnotationAdded;
	}
}
