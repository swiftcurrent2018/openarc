package cetus.application;

import java.io.IOException;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Stack;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import cetus.analysis.CFGraph;
import cetus.analysis.CallGraph;
import cetus.analysis.CallSite;
import cetus.analysis.DFANode;
import cetus.analysis.IPAGraph;
import cetus.analysis.IPANode;
import cetus.application.ProgramSummaryGraph.Mode;
import cetus.application.ProgramSummaryGraph.PSGPropagator;
import cetus.hir.ArrayAccess;
import cetus.hir.AssignmentExpression;
import cetus.hir.BinaryExpression;
import cetus.hir.BreakStatement;
import cetus.hir.Case;
import cetus.hir.CommaExpression;
import cetus.hir.ContinueStatement;
import cetus.hir.DoLoop;
import cetus.hir.Expression;
import cetus.hir.ExpressionStatement;
import cetus.hir.ForLoop;
import cetus.hir.FunctionCall;
import cetus.hir.IRTools;
import cetus.hir.Identifier;
import cetus.hir.IfStatement;
import cetus.hir.IntegerLiteral;
import cetus.hir.NestedDeclarator;
import cetus.hir.PrintTools;
import cetus.hir.Procedure;
import cetus.hir.ProcedureDeclarator;
import cetus.hir.Program;
import cetus.hir.ReturnStatement;
import cetus.hir.Statement;
import cetus.hir.SwitchStatement;
import cetus.hir.Traversable;
import cetus.hir.UnaryExpression;
import cetus.hir.VariableDeclarator;
import cetus.hir.WhileLoop;

/**
 * This class slices the program using the def-use or use-def chain and
 * the control dependence graph.
 * @author Jae-Woo Lee, <jaewoolee@purdue.edu>
 *         School of ECE, Purdue University
 */
public class ProgramSlicer {
	Program program;
	Set<Procedure> procSet;
	Map<Procedure, CFGraph> cfgMap;
	IPChainAnalysis chainAnalysis;
	IPAGraph ipaGraph;
	
	public enum Criteria {
		PRINT_STMT,
		EXIT_STMT
	};
	
	Map<DFANode, List<Expression>> scPartialMap;
	Set<DFANode> scSet;
	Set<Criteria> criteriaSet;
	boolean customCriteria = false;
	boolean predefinedCriteria = false;
	AnalysisTarget[] globalDefArray;
	
	public ProgramSlicer(Map<Procedure, CFGraph> cfgMap, IPChainAnalysis udChain, Program program) {
		this.cfgMap = cfgMap;
		this.program = program;
		scPartialMap = new HashMap<DFANode, List<Expression>>();
		scSet = new HashSet<DFANode>();
		criteriaSet = new HashSet<Criteria>();
		procSet = cfgMap.keySet();
		this.chainAnalysis = udChain;
		LinkedHashSet<AnalysisTarget> globalDefSet = chainAnalysis.getGlobalDefSet();
		globalDefArray = new AnalysisTarget[globalDefSet.size()];
		globalDefSet.toArray(globalDefArray);
    	ipaGraph = new IPAGraph(program);
	}
	
	/**
	 * Add program slice criteria
	 * Whole DFANode is added to the slice criteria
	 * @param dfaNode
	 * @param proc
	 */
	public void addSlicingCriteria(DFANode dfaNode, Procedure proc) {
		customCriteria = true;
		scSet.add(dfaNode);
	}

	/**
	 * Add program slice criteria
	 * Traversable in the DFANode is only added to the slice criteria
	 * @param dfaNode
	 * @param expr
	 * @param proc
	 */
	public void addSlicingCriteria(DFANode dfaNode, List<Expression> exprList, Procedure proc) {
		customCriteria = true;
		scSet.add(dfaNode);
		scPartialMap.put(dfaNode, exprList);
	}
	
	public void addPreDefinedSlicingCriteria(Criteria cri) {
		predefinedCriteria = true;
		criteriaSet.add(cri);
	}
	
	public void markSlice() {
		if (predefinedCriteria) {
			for (Criteria cri: criteriaSet) {
				addPredefinedDFANodeToSlicingCriteria(cri);
			}
		}
		startSlicing();
		handleSpecialControlStructure();
	}
	
	private void handleSpecialControlStructure() {
		System.out.println("[ProgramSlicer] begin handleSpecialControlStructure, time: " + (new Date()).toString());
		// check all the break statement and see if its control dependent node is in the slice
		// if the controlling node is in the slice, add the break statement in the slice
		for (Procedure proc: procSet) {
			CFGraph cfg = cfgMap.get(proc);
			Iterator<DFANode> cfgIter = cfg.iterator();
			while (cfgIter.hasNext()) {
				DFANode dfaNode = cfgIter.next();
				Traversable ir = (Traversable)CFGraph.getIR(dfaNode);
				if (ir instanceof BreakStatement ||
						ir instanceof ContinueStatement ||
						ir instanceof ReturnStatement) {
					DFANode controlNode = dfaNode.getData("controlNode");
					if (controlNode != null && (CFGraph.getIR(controlNode) instanceof SwitchStatement) == false) {
						Boolean isSlice = controlNode.getData("isSlice");
						if (isSlice == Boolean.TRUE) {
							dfaNode.putData("isSlice", Boolean.TRUE);
						}
					}
				}				
			}
		}
		// handle case statement for switch-case
		// if a stmt is in slice, check controlling node 
		// if control node is switch stmt check, add case which is dominator of current node (handle case)
		for (Procedure proc: procSet) {
			CFGraph cfg = cfgMap.get(proc);
			// get dominator
			ArrayList<DFANode> nodeList = new ArrayList<DFANode>();
			BitSet[] dominators = SlicingTools.getDominators(cfg, nodeList);
			// iterator
			Iterator<DFANode> cfgIter = cfg.iterator();
			while (cfgIter.hasNext()) {
				DFANode dfaNode = cfgIter.next();
				Traversable ir = (Traversable)CFGraph.getIR(dfaNode);
				if (dfaNode.getData("isSlice") == Boolean.TRUE) {
					DFANode controlNode = dfaNode.getData("controlNode");
					if (controlNode != null && CFGraph.getIR(controlNode) instanceof SwitchStatement) {
						// get current node's idx for dominator check
						int currentNodeIdx = nodeList.indexOf(dfaNode);
						Set<DFANode> caseNodes = controlNode.getSuccs();
						for (DFANode caseN: caseNodes) {
							// if case statement is current node's dominator
							if (dominators[currentNodeIdx].get(nodeList.indexOf(caseN))) {
								caseN.putData("isSlice", Boolean.TRUE);
							}
						}
					}
				}
			}
		}
		// handle break statement for switch-case
		// if a statement is break or continue or return, check control node
		// if it is switch-case, check case and is case is in slice, add break/continue/return node to slice
		for (Procedure proc: procSet) {
			CFGraph cfg = cfgMap.get(proc);
			// get dominator
			ArrayList<DFANode> nodeList = new ArrayList<DFANode>();
			BitSet[] dominators = SlicingTools.getDominators(cfg, nodeList);
			// iterator
			Iterator<DFANode> cfgIter = cfg.iterator();
			while (cfgIter.hasNext()) {
				DFANode dfaNode = cfgIter.next();
				Traversable ir = (Traversable)CFGraph.getIR(dfaNode);
				if (ir instanceof BreakStatement ||
						ir instanceof ContinueStatement ||
						ir instanceof ReturnStatement) {
					DFANode controlNode = dfaNode.getData("controlNode");
					if (controlNode != null && CFGraph.getIR(controlNode) instanceof SwitchStatement) {
						// get current node's idx for dominator check
						int currentNodeIdx = nodeList.indexOf(dfaNode);
						Set<DFANode> caseNodes = controlNode.getSuccs();
						for (DFANode caseN: caseNodes) {
							// if case statement is current node's dominator
							if (dominators[currentNodeIdx].get(nodeList.indexOf(caseN))) {
								if (caseN.getData("isSlice") == Boolean.TRUE) {
									dfaNode.putData("isSlice", Boolean.TRUE);
								}
							}
						}
					}
				}
			}
		}
		System.out.println("[ProgramSlicer] end handleSpecialControlStructure, time: " + (new Date()).toString());
	}
	
	private void addPredefinedDFANodeToSlicingCriteria(Criteria cri) {
		// set slice criteria
		for (Procedure proc: procSet) {
			CFGraph cfg = cfgMap.get(proc);
			Iterator<DFANode> cfgIter = cfg.iterator();
			while (cfgIter.hasNext()) {
				DFANode dfaNode = cfgIter.next();
				Traversable ir = (Traversable)CFGraph.getIR(dfaNode);
				if (IRTools.containsFunctionCall(ir)) {
					List<FunctionCall> funcList = IRTools.getFunctionCalls(ir);
					for (FunctionCall fc: funcList) {
						String fname = fc.getName().toString();
						if (cri.equals(Criteria.PRINT_STMT)) {
							if (PrintTools.getVerbosity() > 1) {
								System.out.println("[markSlice]mark slice criteria for print statements");
							}
							if (fname.equals("printf")) {
								if (PrintTools.getVerbosity() > 1) {
									System.out.println("[markSlice]Adding slice criteria: " + ir + ", proc: " + proc.getSymbolName());
								}
								addSlicingCriteria(dfaNode, proc);
							}
						} else if (cri.equals(Criteria.EXIT_STMT)) {
							if (PrintTools.getVerbosity() > 1) {
								System.out.println("[markSlice]mark slice criteria for exit statements");
							}
							if (fname.equals("exit")) {
								if (PrintTools.getVerbosity() > 1) {
									System.out.println("[markSlice]Adding slice criteria: " + ir + ", proc: " + proc.getSymbolName());
								}
								addSlicingCriteria(dfaNode, proc);
							}
						}
					}
				}					
			}
		}
	}
	
	private void addToWorklistReturnNodeOfCallee(FunctionCall fc, LinkedList<DFANode> workList) {
		// add return node of called function
		Procedure callee = fc.getProcedure();
		if (callee != null) {
			CFGraph calleeCFG = cfgMap.get(callee);
			List<DFANode> exitList = calleeCFG.getExitNodes();
			for (DFANode dfaNode: exitList) {
				if (dfaNode.getData("isSlice") == null) {
					workList.add(dfaNode);
				}
			}
		}
	}
	
	private void addToWorklistDefNodeOfSideEffectParamInCallee(DFANode currentNode, LinkedList<DFANode> workList) {
		// add def node of side effect param in callee function
		Set<DFANode> refSet = currentNode.getData("psg_return_ref");
		if (refSet != null) {
			for (DFANode returnRef: refSet) {
				Set<AnalysisTarget> defSet = returnRef.getData("INdef");
				for (AnalysisTarget def: defSet) {
					DFANode refDefNode = def.getDFANode();
					if (refDefNode.getData("isSlice") == null) {
						workList.add(refDefNode);
					}
				}
			}
		}
	}
	
//	private void addToWorklistDefNodeOfGlobalVarInCallee(DFANode currentNode, LinkedList<DFANode> workList) {
//		// add def node of global variable in callee function
//		Set<DFANode> globalSet = currentNode.getData("psg_return_global");
//		if (globalSet != null) {
//			for (DFANode returnGlobal: globalSet) {
//				BitSet globalDefBitSet = returnGlobal.getData("INdef");
//				if (globalDefBitSet != null) {
//					for (int i = globalDefBitSet.nextSetBit(0); i >= 0; i = globalDefBitSet.nextSetBit(i+1)) {
//						if (globalDefArray[i].getDFANode().getData("isSlice") == null) {
//							workList.add(globalDefArray[i].getDFANode());
//						}
//					}
//				}
//			}
//		}
//	}
	
	private void addToWorklistControlDependentNode(DFANode currentNode, LinkedList<DFANode> workList) {
		DFANode controllingNode = currentNode.getData("controlNode");
		if (controllingNode != null && controllingNode.getData("isSlice") == null) {
			workList.add(controllingNode);
		}
	}
	
	private void addToWorklistDefNode(DFANode currentNode, LinkedList<DFANode> workList) {
		Traversable currentIR = (Traversable) CFGraph.getIR(currentNode);
		List<Expression> useList = ChainTools.getUseList(currentIR);
		// if the current node is in partial slicing list
		// only the relevant expression is contained in the slice
		if (scPartialMap.containsKey(currentNode)) {
			useList = scPartialMap.get(currentNode);
		}
		Procedure currentProc = ChainTools.getParentProcedure(currentIR, program);
		for (Expression use: useList) {
			Set<DFANode> defDFANodeSet = chainAnalysis.getDefDFANodeSet(ChainTools.getIDExpression(use));
			for (DFANode dfaNode: defDFANodeSet) {
				if (dfaNode.getData("isSlice") == null &&
						workList.contains(dfaNode) == false) {
					workList.add(dfaNode);
//					Object ir = dfaNode.getData("ir");
//					if (ir instanceof ExpressionStatement) {
//						Expression ex = ((ExpressionStatement)ir).getExpression();
//						if (ex instanceof AssignmentExpression) {
//							Expression lhsEx = ((AssignmentExpression) ex).getLHS();
//							if (ChainTools.hasSameStructureVariableIdentifier(lhsEx, use, currentProc)) {
//								// same struct ID
//							} else if (ChainTools.hasSameToString(lhsEx, use)) {
//								// same string
//							} else {
//								System.out.println("CurrentIR: " + currentIR);
//								System.out.println("  Def Assignment Expression: " + ex);
//								System.out.println("  Has Not Same String: lhsEx: " + lhsEx + ", use: " + use + ", proc: " + currentProc.getSymbolName());
//							}
//						} else {
////							System.out.println("Not Assignment Expression: " + ex);
//						}
//					} else {					
////						System.out.println("Def is Not ExpressionStatement: " + ir + ", class: " + ir.getClass().getCanonicalName());
//						
//					}
//					// if def proc is different from use proc
//					// add call sites of def proc to work list
//					Procedure defProc = ChainTools.getParentProcedure((Traversable)dfaNode.getData("ir"), program);
//					if (currentProc.equals(defProc) == false) {
//						workList.addAll(findCallSites(defProc, ipaGraph));
//					}
				}
			}
		}
		// add all the call site of use proc to worklist
		Set<DFANode> callSiteDFAs = findCallSites(currentProc, ipaGraph);
		for (DFANode n: callSiteDFAs) {
			if (n.getData("isSlice") == null &&
					workList.contains(n) == false) {
				workList.add(n);
			}
		}
	}
	
	class MiniSlicer implements Runnable {

		LinkedList<DFANode> workList;
		
		public MiniSlicer() {
			workList = new LinkedList<DFANode>();
		}
		
		public void addToWorkList(DFANode node) {
			workList.add(node);
		}
		
		public void run() {
			while (workList.isEmpty() == false) {
				DFANode currentNode = workList.remove();
				Traversable currentIR = (Traversable) CFGraph.getIR(currentNode);
				currentNode.putData("isSlice", Boolean.TRUE);
				
				// add control dependence node
				addToWorklistControlDependentNode(currentNode, workList);
				// add def node
				addToWorklistDefNode(currentNode, workList);
				
				if (scPartialMap.containsKey(currentNode) == false) {
					List<FunctionCall> fcList = IRTools.getFunctionCalls(currentIR);
					if (fcList != null) {
						for (FunctionCall fc: fcList) {
							// add return node of called function
							addToWorklistReturnNodeOfCallee(fc, workList);
							// add def node of side effect param in callee function
							addToWorklistDefNodeOfSideEffectParamInCallee(currentNode, workList);
	//						// add def node of global variable in callee function
	//						addToWorklistDefNodeOfGlobalVarInCallee(currentNode, workList);
						}
					}
					// handle procedures which contain slices
					Set<DFANode> callDFANodeSet = findCallSites(ChainTools.getParentProcedure(currentIR, program), ipaGraph);
					for (DFANode n: callDFANodeSet) {
						if (n.getData("isSlice") == null &&
								workList.contains(n) == false) {
							workList.add(n);
						}
					}
				}
			}
		}
	}
	
	private void startSlicing() {
		System.out.println("[ProgramSlicer] begin startSlicing, time: " + (new Date()).toString());
		// mark slice
		LinkedList<DFANode> workList = new LinkedList<DFANode>();
		workList.addAll(scSet);
		while (workList.isEmpty() == false) {
			DFANode currentNode = workList.remove();
			Traversable currentIR = (Traversable) CFGraph.getIR(currentNode);
			currentNode.putData("isSlice", Boolean.TRUE);
			
			// add control dependence node
			addToWorklistControlDependentNode(currentNode, workList);
			// add def node
			addToWorklistDefNode(currentNode, workList);
			
			if (scPartialMap.containsKey(currentNode) == false) {
				List<FunctionCall> fcList = IRTools.getFunctionCalls(currentIR);
				for (FunctionCall fc: fcList) {
					// add return node of called function
					addToWorklistReturnNodeOfCallee(fc, workList);
					// add def node of side effect param in callee function
					addToWorklistDefNodeOfSideEffectParamInCallee(currentNode, workList);
//					// add def node of global variable in callee function
//					addToWorklistDefNodeOfGlobalVarInCallee(currentNode, workList);
				}
				// handle procedures which contain slices
				Set<DFANode> callDFANodeSet = findCallSites(ChainTools.getParentProcedure(currentIR, program), ipaGraph);
				for (DFANode n: callDFANodeSet) {
					if (n.getData("isSlice") == null &&
							workList.contains(n) == false) {
						workList.add(n);
					}
				}
			}
			if (workList.size() > 4) {
				break;
			}
		}
		
		// split worklist
        ExecutorService taskExecutor = Executors.newFixedThreadPool(workList.size());
		for (DFANode node: workList) {
	        MiniSlicer mSlicer = new MiniSlicer();
	        mSlicer.addToWorkList(node);
	       	taskExecutor.execute(mSlicer);
		}
        taskExecutor.shutdown();
        try {
			taskExecutor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		
		System.out.println("[ProgramSlicer] end startSlicing, time: " + (new Date()).toString());
	}
	
	private Set<DFANode> findCallSites(Procedure proc, IPAGraph ipaGraph) {
		IPANode node = ipaGraph.getNode(proc);
		Set<DFANode> returnSet = new HashSet<DFANode>();
		// sites calling this procedure
		List<CallSite> callingSiteList = node.getCallingSites();
		for (CallSite cs: callingSiteList) {
			Procedure callProc = IRTools.getParentProcedure(cs.getFunctionCall());
			CFGraph cfg = cfgMap.get(callProc);
			Iterator<DFANode> cfgIter = cfg.iterator();
			while (cfgIter.hasNext()) {
				DFANode dfaNode = cfgIter.next();
				Traversable ir = (Traversable)CFGraph.getIR(dfaNode);
				if (IRTools.containsFunctionCall(ir)) {
					List<FunctionCall> funcList = IRTools.getFunctionCalls(ir);
					for (FunctionCall fc: funcList) {
						if (cs.getFunctionCall().equals(fc)) {
							returnSet.add(dfaNode);
						}
					}
				}
			}
		}
		return returnSet;
	}
	
	class CodeRemover implements Runnable {
		
		Procedure proc;
		
		public CodeRemover(Procedure proc) {
			this.proc = proc;
		}
		
		public void run() {
			System.out.println("[CodeRemover] begin, proc: " + proc.getSymbolName() + ", time: " + (new Date()).toString());
			CFGraph cfg = cfgMap.get(proc);
			Iterator cfgIter = cfg.iterator();
			while (cfgIter.hasNext()) {
				DFANode dfa = (DFANode) cfgIter.next();
				if (dfa.getData("isSlice") == null) {
					Traversable tr = (Traversable) CFGraph.getIR(dfa);
					if (tr == null) {
						continue;
					}
					if (PrintTools.getVerbosity() > 1) {
						if (tr instanceof SwitchStatement) {
							System.out.println("Removing the traversable[Whole SwitchStatement]: " + ((SwitchStatement)tr).getExpression() + ", proc: " + proc.getSymbolName());
						} else {
							System.out.println("Removing the traversable: " + tr.toString() + ", class: " + tr.getClass().getCanonicalName() + ", proc: " + proc.getSymbolName());
						}
					}
					if (tr instanceof Statement) {
						Statement ex = (Statement) tr;
						Traversable stmtParent = ex.getParent();
						if (stmtParent instanceof ForLoop) {
							// do not remove ForLoop when the init step is "isSlice == null"
							// this will be handled by BinaryExpression
//							Traversable stmtParParent = stmtParent.getParent();
//							stmtParParent.removeChild(stmtParent);
						} else {
							if (stmtParent == null) {
								System.err.println("ProgramSlicer.removeUnmarkedDFANode(not removed, parent is null): " + tr.toString() + ", proc: " + proc.getSymbolName() + ", class: " + tr.getClass().getCanonicalName());
							} else {
								stmtParent.removeChild(ex);
							}
						}
					} else if (tr instanceof BinaryExpression) {
						// This handles IfStatement
						BinaryExpression ex = (BinaryExpression) tr;
						Traversable binaryParent = ex.getParent();
						if (binaryParent instanceof IfStatement) {
							Traversable binaryParParent = binaryParent.getParent();
							binaryParParent.removeChild(binaryParent);
						} else if (binaryParent instanceof ForLoop) {
							Traversable stmtParParent = binaryParent.getParent();
							if (stmtParParent != null)
								stmtParParent.removeChild(binaryParent);
						} else if (binaryParent instanceof WhileLoop) {
							Traversable stmtParParent = binaryParent.getParent();
							stmtParParent.removeChild(binaryParent);
						} else if (binaryParent instanceof DoLoop) {
							Traversable stmtParParent = binaryParent.getParent();
							stmtParParent.removeChild(binaryParent);
						} else {
							throw new RuntimeException("[removeUnmarkedDFANode]Unhandled Type: " + tr.toString() + ", proc: " + proc.getSymbolName() + ", class: " + tr.getClass().getCanonicalName());
						}
					} else if (tr instanceof VariableDeclarator) {
						VariableDeclarator vDeclarator = (VariableDeclarator) tr;
						Traversable parentTrav = vDeclarator.getParent();
						// The following operation is not allowed.
						// parentTrav.removeChild(vDeclarator);
					} else if (tr instanceof UnaryExpression) {
						// the removing is not allowed
					} else if (tr instanceof Identifier) {
						// the removing is not allowed
					} else if (tr instanceof FunctionCall) {
						Traversable fcParent = tr.getParent();
						if (fcParent instanceof IfStatement) {
							Traversable fcParParent = fcParent.getParent();
							fcParParent.removeChild(fcParent);
						} else {
							throw new RuntimeException("[removeUnmarkedDFANode]Unhandled Type: " + tr.toString() + ", proc: " + proc.getSymbolName() + ", class: " + tr.getClass().getCanonicalName());
						}
					} else if (tr instanceof CommaExpression) {
						Traversable ceParent = tr.getParent();
						if (ceParent instanceof ForLoop) {
							// if this is ForLoop, it is already removed
//							Traversable ceParParent = ceParent.getParent();
//							ceParParent.removeChild(ceParent);
						}
					} else if (tr instanceof NestedDeclarator) {
						System.err.println("ProgramSlicer.removeUnmarkedDFANode(not removed): " + tr.toString() + ", proc: " + proc.getSymbolName() + ", class: " + tr.getClass().getCanonicalName());
					} else if (tr instanceof ArrayAccess) {
						System.err.println("ProgramSlicer.removeUnmarkedDFANode(not removed): " + tr.toString() + ", proc: " + proc.getSymbolName() + ", class: " + tr.getClass().getCanonicalName());
					} else if (tr instanceof IntegerLiteral) {
						System.err.println("ProgramSlicer.removeUnmarkedDFANode(not removed): " + tr.toString() + ", proc: " + proc.getSymbolName() + ", class: " + tr.getClass().getCanonicalName());
					} else if (tr instanceof ProcedureDeclarator) {
						System.err.println("ProgramSlicer.removeUnmarkedDFANode(not removed): " + tr.toString() + ", proc: " + proc.getSymbolName() + ", class: " + tr.getClass().getCanonicalName());
					} else {
						throw new RuntimeException("[removeUnmarkedDFANode]Unhandled Type: " + tr.toString() + ", proc: " + proc.getSymbolName() + ", class: " + tr.getClass().getCanonicalName());
					}
				}
			}
			System.out.println("[CodeRemover] begin, proc: " + proc.getSymbolName() + ", time: " + (new Date()).toString());
		}
	}
	
	public void removeUnmarkedDFANodeWithThreads() {
		System.out.println("[ProgramSlicer] begin removeUnmarkedDFANodeWithThreads, time: " + (new Date()).toString());
        ExecutorService taskExecutor = Executors.newFixedThreadPool(procSet.size());
		for (Procedure proc: procSet) {
	        CodeRemover cRemover = new CodeRemover(proc);
	       	taskExecutor.execute(cRemover);
		}
        taskExecutor.shutdown();
        try {
			taskExecutor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
		System.out.println("[ProgramSlicer] end removeUnmarkedDFANodeWithThreads, time: " + (new Date()).toString());
	}
	
	public void removeUnmarkedDFANode() {
		System.out.println("[ProgramSlicer] begin removeUnmarkedDFANode, time: " + (new Date()).toString());
		for (Procedure proc: procSet) {
			CFGraph cfg = cfgMap.get(proc);
			Iterator cfgIter = cfg.iterator();
			while (cfgIter.hasNext()) {
				DFANode dfa = (DFANode) cfgIter.next();
				if (dfa.getData("isSlice") == null) {
					Traversable tr = (Traversable) CFGraph.getIR(dfa);
					if (tr == null) {
						continue;
					}
					if (PrintTools.getVerbosity() > 1) {
						if (tr instanceof SwitchStatement) {
							System.out.println("Removing the traversable[Whole SwitchStatement]: " + ((SwitchStatement)tr).getExpression() + ", proc: " + proc.getSymbolName());
						} else {
							System.out.println("Removing the traversable: " + tr.toString() + ", class: " + tr.getClass().getCanonicalName() + ", proc: " + proc.getSymbolName());
						}
					}
					if (tr instanceof Statement) {
						Statement ex = (Statement) tr;
						Traversable stmtParent = ex.getParent();
						if (stmtParent instanceof ForLoop) {
							// do not remove ForLoop when the init step is "isSlice == null"
							// this will be handled by BinaryExpression
//							Traversable stmtParParent = stmtParent.getParent();
//							stmtParParent.removeChild(stmtParent);
						} else {
							stmtParent.removeChild(ex);
						}
					} else if (tr instanceof BinaryExpression) {
						// This handles IfStatement
						BinaryExpression ex = (BinaryExpression) tr;
						Traversable binaryParent = ex.getParent();
						if (binaryParent instanceof IfStatement) {
							Traversable binaryParParent = binaryParent.getParent();
							binaryParParent.removeChild(binaryParent);
						} else if (binaryParent instanceof ForLoop) {
							Traversable stmtParParent = binaryParent.getParent();
							if (stmtParParent != null)
								stmtParParent.removeChild(binaryParent);
						} else if (binaryParent instanceof WhileLoop) {
							Traversable stmtParParent = binaryParent.getParent();
							stmtParParent.removeChild(binaryParent);
						} else if (binaryParent instanceof DoLoop) {
							Traversable stmtParParent = binaryParent.getParent();
							stmtParParent.removeChild(binaryParent);
						} else {
							throw new RuntimeException("[removeUnmarkedDFANode]Unhandled Type: " + tr.toString() + ", proc: " + proc.getSymbolName() + ", class: " + tr.getClass().getCanonicalName());
						}
					} else if (tr instanceof VariableDeclarator) {
						VariableDeclarator vDeclarator = (VariableDeclarator) tr;
						Traversable parentTrav = vDeclarator.getParent();
						// The following operation is not allowed.
						// parentTrav.removeChild(vDeclarator);
					} else if (tr instanceof UnaryExpression) {
						// the removing is not allowed
					} else if (tr instanceof Identifier) {
						// the removing is not allowed
					} else if (tr instanceof FunctionCall) {
						Traversable fcParent = tr.getParent();
						if (fcParent instanceof IfStatement) {
							Traversable fcParParent = fcParent.getParent();
							fcParParent.removeChild(fcParent);
						} else {
							throw new RuntimeException("[removeUnmarkedDFANode]Unhandled Type: " + tr.toString() + ", proc: " + proc.getSymbolName() + ", class: " + tr.getClass().getCanonicalName());
						}
					} else if (tr instanceof CommaExpression) {
						Traversable ceParent = tr.getParent();
						if (ceParent instanceof ForLoop) {
							// if this is ForLoop, it is already removed
//							Traversable ceParParent = ceParent.getParent();
//							ceParParent.removeChild(ceParent);
						}
					} else if (tr instanceof NestedDeclarator) {
						System.err.println("ProgramSlicer.removeUnmarkedDFANode(not removed): " + tr.toString() + ", proc: " + proc.getSymbolName());
					} else {
						throw new RuntimeException("[removeUnmarkedDFANode]Unhandled Type: " + tr.toString() + ", proc: " + proc.getSymbolName() + ", class: " + tr.getClass().getCanonicalName());
					}
				}
			}
		}
		System.out.println("[ProgramSlicer] end removeUnmarkedDFANode, time: " + (new Date()).toString());
	}
	
	public void writeToDisk() {
		try {
			program.print();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
