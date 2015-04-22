package cetus.application;

import java.util.BitSet;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import cetus.analysis.CFGraph;
import cetus.analysis.DFANode;
import cetus.analysis.Domain;
import cetus.analysis.IPPointsToAnalysis;
import cetus.analysis.PointsToDomain;
import cetus.analysis.PointsToRel;
import cetus.application.ChainTools.MyType;
import cetus.hir.ArrayAccess;
import cetus.hir.BinaryExpression;
import cetus.hir.ConditionalExpression;
import cetus.hir.DepthFirstIterator;
import cetus.hir.Expression;
import cetus.hir.IDExpression;
import cetus.hir.IRTools;
import cetus.hir.Identifier;
import cetus.hir.NestedDeclarator;
import cetus.hir.PrintTools;
import cetus.hir.Procedure;
import cetus.hir.Specifier;
import cetus.hir.Statement;
import cetus.hir.SwitchStatement;
import cetus.hir.Symbol;
import cetus.hir.SymbolTools;
import cetus.hir.Traversable;
import cetus.hir.UnaryExpression;
import cetus.hir.VariableDeclaration;
import cetus.hir.VariableDeclarator;

public class ChainComputer implements Runnable {
	
	private Procedure targetProc;
	private Map<Procedure, CFGraph> cfgMap;
    private LinkedHashSet<AnalysisTarget> globalDefSet;
    private LinkedHashSet<AnalysisTarget> globalUseSet;
    private Map<Procedure, Set<AnalysisTarget>> defTargetSetMap;
    private Map<Procedure, Set<AnalysisTarget>> useTargetSetMap;

	public ChainComputer(
			Procedure proc,
			Map<Procedure, CFGraph> cfgMap,
			LinkedHashSet<AnalysisTarget> globalDefSet,
			LinkedHashSet<AnalysisTarget> globalUseSet,
			Map<Procedure, Set<AnalysisTarget>> defTargetSetMap,
			Map<Procedure, Set<AnalysisTarget>> useTargetSetMap) {
		super();
		this.targetProc = proc;
		this.cfgMap = cfgMap;
		this.globalDefSet = globalDefSet;
		this.globalUseSet = globalUseSet;
		this.defTargetSetMap = defTargetSetMap;
		this.useTargetSetMap = useTargetSetMap;
	}
	
	@Override
	public void run() {
		// TODO Auto-generated method stub
		computeChain();
	}
	
	private void computeChain() {
    	System.out.println("[ChainComputer.computeChain]Begin proc: " + targetProc.getSymbolName() + ", time: " + (new Date()).toString());
        AnalysisTarget[] defTargetArray = getDefTargetArray(targetProc);
        handleLocalStaticVariables(targetProc, defTargetArray);
        CFGraph cfg = cfgMap.get(targetProc);
        Iterator iter = cfg.iterator();
        while (iter.hasNext()) {
            DFANode cfgNode = (DFANode) iter.next();
            Object nodeIR = CFGraph.getIR(cfgNode);
            if (nodeIR instanceof VariableDeclarator) {
                if (ChainTools.isDefinedArrayDeclarator((VariableDeclarator) nodeIR)) {
                    generateChainForDefinedArrayDeclarator(cfgNode, defTargetArray, (VariableDeclarator) nodeIR, targetProc);
                } else {
                    generateChainForPlainDeclarator(cfgNode, defTargetArray, (VariableDeclarator) nodeIR, targetProc);
                }
            } else if (nodeIR instanceof NestedDeclarator) {
                if (ChainTools.isDefinedArrayNestedDeclarator((NestedDeclarator) nodeIR)) {
                    generateChainForDefinedArrayNestedDeclarator(cfgNode, defTargetArray, (NestedDeclarator) nodeIR, targetProc);
                } else {
                    generateChainForPlainNestedDeclarator(cfgNode, defTargetArray, (NestedDeclarator) nodeIR, targetProc);
                }
            } else if (nodeIR instanceof SwitchStatement) {
                SwitchStatement switchStmt = (SwitchStatement) nodeIR;
                Expression expInSwitch = switchStmt.getExpression();
                List<Expression> useList = ChainTools.getUseList(expInSwitch);
                for (Expression currentUse : useList) {
                    if (ChainTools.getIDExpression(currentUse) != null) {
                        if (ChainTools.isArrayAccess(currentUse)) {
                            generateChainForArrayAccess(currentUse, defTargetArray, cfgNode, targetProc);
                        } else if (ChainTools.isPointerAccess(currentUse)) {
                            generateChainForPointerAccess(currentUse, defTargetArray, cfgNode, targetProc);
                        } else if (ChainTools.isStructureAccess(currentUse, targetProc)) {
                            generateChainForStructure(currentUse, defTargetArray, cfgNode, targetProc);
                        } else {
                            generateChainForPlainExpression(currentUse, defTargetArray, cfgNode, targetProc);
                        }
                        generateChainForAlias(nodeIR, currentUse, defTargetArray, cfgNode, targetProc);
                    }
                } // for (Expression currentUse : useTargetSet)
            } else if (nodeIR instanceof Traversable) {
                Traversable traversableNodeIR = (Traversable) nodeIR;
                List<Expression> useList = ChainTools.getUseList(traversableNodeIR);
                for (Expression currentUse : useList) {
                    if (ChainTools.getIDExpression(currentUse) != null) {
                        if (ChainTools.isArrayAccess(currentUse)) {
                            generateChainForArrayAccess(currentUse, defTargetArray, cfgNode, targetProc);
                        } else if (ChainTools.isPointerAccess(currentUse)) {
                            if (ChainTools.isStructureAccess(currentUse, targetProc)) {
                                generateChainForStructure(currentUse, defTargetArray, cfgNode, targetProc);
                            } else {
                                generateChainForPointerAccess(currentUse, defTargetArray, cfgNode, targetProc);
                            }
                        } else if (ChainTools.isStructureAccess(currentUse, targetProc)) {
                            generateChainForStructure(currentUse, defTargetArray, cfgNode, targetProc);
                        } else {
                            generateChainForPlainExpression(currentUse, defTargetArray, cfgNode, targetProc);
                        }
                        generateChainForAlias(nodeIR, currentUse, defTargetArray, cfgNode, targetProc);
                    }
                } // for (Expression currentUse : useTargetSet)
                if (IRTools.containsFunctionCall(traversableNodeIR)) {
                    BitSet inBitSet = cfgNode.getData("InSet"); // reaching definition
                    Set<DFANode> callNodeSetRef = cfgNode.getData("psg_call_ref");
                    if (callNodeSetRef != null) {
                        for (DFANode callNode : callNodeSetRef) {
                            Set<AnalysisTarget> inUSE = callNode.getData("INuse");  // upward uses from callee
                            List<Expression> argsList = callNode.getData("arg");
                            for (Expression callArg: argsList) {
	                            if (callArg instanceof ConditionalExpression) {
	                                Expression trueEx = ((ConditionalExpression) callArg).getTrueExpression();
	                                if (trueEx instanceof BinaryExpression) {
	                                    List<Expression> defList = ChainTools.getUseList(trueEx);
	                                    for (Expression def : defList) {
	                                        Expression idEx = ChainTools.getIDExpression(def);
	                                        if (idEx != null) {
	                                        	//  for (int i = bs.nextSetBit(0); i >= 0; i = bs.nextSetBit(i+1)) 
	                                        	//  { // operate on index i here }
	                                            for (AnalysisTarget target : inUSE) {
	                                                for (int i = inBitSet.nextSetBit(0); i >= 0; i = inBitSet.nextSetBit(i+1)) {
                                                        if (defTargetArray[i].getExpression().equals(idEx)) {
                                                            defTargetArray[i].addUseChain(target);
                                                            addUseDefChain(defTargetArray[i], target.getExpression(), target.getDFANode(), target.getProcedure());
                                                        }
	                                                }
	                                            }
	                                        }
	                                    }
	                                } else {
	                                    Expression idEx = ChainTools.getIDExpression(trueEx);
	                                    if (idEx != null) {
	                                        for (AnalysisTarget target : inUSE) {
	                                            for (int i = inBitSet.nextSetBit(0); i >= 0; i = inBitSet.nextSetBit(i+1)) {
                                                    if (defTargetArray[i].getExpression().equals(idEx)) {
                                                        defTargetArray[i].addUseChain(target);
                                                        addUseDefChain(defTargetArray[i], target.getExpression(), target.getDFANode(), target.getProcedure());
                                                    }
	                                            }
	                                        }
	                                    }
	                                }
	                                Expression falseEx = ((ConditionalExpression) callArg).getTrueExpression();
	                                if (falseEx instanceof BinaryExpression) {
	                                    List<Expression> defList = ChainTools.getUseList(falseEx);
	                                    for (Expression def : defList) {
	                                        Expression idEx = ChainTools.getIDExpression(def);
	                                        if (idEx != null) {
	                                            for (AnalysisTarget target : inUSE) {
	                                                for (int i = inBitSet.nextSetBit(0); i >= 0; i = inBitSet.nextSetBit(i+1)) {
                                                        if (defTargetArray[i].getExpression().equals(idEx)) {
                                                            defTargetArray[i].addUseChain(target);
                                                            addUseDefChain(defTargetArray[i], target.getExpression(), target.getDFANode(), target.getProcedure());
                                                        }
	                                                }
	                                            }
	                                        }
	                                    }
	                                } else {
	                                    Expression idEx = ChainTools.getIDExpression(falseEx);
	                                    if (idEx != null) {
	                                        for (AnalysisTarget target : inUSE) {
	                                            for (int i = inBitSet.nextSetBit(0); i >= 0; i = inBitSet.nextSetBit(i+1)) {
                                                    if (defTargetArray[i].getExpression().equals(idEx)) {
                                                        defTargetArray[i].addUseChain(target);
                                                        addUseDefChain(defTargetArray[i], target.getExpression(), target.getDFANode(), target.getProcedure());
                                                    }
	                                            }
	                                        }
	                                    }
	                                }
	                            } else if (callArg instanceof BinaryExpression) {
	                                List<Expression> defList = ChainTools.getUseList(callArg);
	                                for (Expression def : defList) {
	                                    Expression idEx = ChainTools.getIDExpression(def);
	                                    if (idEx != null) {
	                                        for (AnalysisTarget target : inUSE) {
	                                            for (int i = inBitSet.nextSetBit(0); i >= 0; i = inBitSet.nextSetBit(i+1)) {
                                                    if (defTargetArray[i].getExpression().equals(idEx)) {
                                                        defTargetArray[i].addUseChain(target);
                                                        addUseDefChain(defTargetArray[i], target.getExpression(), target.getDFANode(), target.getProcedure());
                                                    }
	                                            }
	                                        }
	                                    }
	                                }
	                            } else {
	                                Expression idEx = ChainTools.getIDExpression(callArg);
	                                if (idEx != null) {
	                                    for (AnalysisTarget target : inUSE) {
	                                        for (int i = inBitSet.nextSetBit(0); i >= 0; i = inBitSet.nextSetBit(i+1)) {
                                                if (defTargetArray[i].getExpression().equals(idEx)) {
                                                    defTargetArray[i].addUseChain(target);
                                                    addUseDefChain(defTargetArray[i], target.getExpression(), target.getDFANode(), target.getProcedure());
                                                }
	                                        }
	                                    }
	                                }
	                            }
                            }
                        }
                    }
                    Set<DFANode> callNodeSetGlobal = cfgNode.getData("psg_call_global");
                    if (callNodeSetGlobal != null) {
                        AnalysisTarget[] globalUseArray = new AnalysisTarget[globalUseSet.size()];
                        globalUseSet.toArray(globalUseArray);
                        AnalysisTarget[] globalDefArray = new AnalysisTarget[globalDefSet.size()];
                        globalDefSet.toArray(globalDefArray);
                        for (DFANode callNode : callNodeSetGlobal) {
                            BitSet bitInUSE = callNode.getData("INuse");
                            for (int i = inBitSet.nextSetBit(0); i >= 0; i = inBitSet.nextSetBit(i+1)) {
                                for (int idx = bitInUSE.nextSetBit(0); idx >= 0; idx = bitInUSE.nextSetBit(idx+1)) {
                                    if (defTargetArray[i].getExpression().equals(globalUseArray[idx].getExpression())) {
                                        defTargetArray[i].addUseChain(globalUseArray[idx]);
                                        addUseDefChain(defTargetArray[i], globalUseArray[idx].getExpression(), globalUseArray[idx].getDFANode(), globalUseArray[idx].getProcedure());
                                    }
                                }
                            }
                            BitSet globalDefInSet = callNode.getData("DefInSet");
                            for (int i = globalDefInSet.nextSetBit(0); i >= 0; i = globalDefInSet.nextSetBit(i+1)) {
                                for (int idx = bitInUSE.nextSetBit(0); idx >= 0; idx = bitInUSE.nextSetBit(idx+1)) {
                                    if (globalDefArray[i].getExpression().equals(globalUseArray[idx].getExpression())) {
                                    	globalDefArray[i].addUseChain(globalUseArray[idx]);
                                        addUseDefChain(globalDefArray[i], globalUseArray[idx].getExpression(), globalUseArray[idx].getDFANode(), globalUseArray[idx].getProcedure());
                                    }
                                }
                            }
                        }
                    }
                    Set<DFANode> returnNodeSetGlobal = cfgNode.getData("psg_return_global");
                    if (returnNodeSetGlobal != null) {
                        AnalysisTarget[] globalUseArray = new AnalysisTarget[globalUseSet.size()];
                        globalUseSet.toArray(globalUseArray);
                        for (DFANode returnNode : returnNodeSetGlobal) {
                            BitSet bitInUSE = returnNode.getData("INuse");
                            for (int i = inBitSet.nextSetBit(0); i >= 0; i = inBitSet.nextSetBit(i+1)) {
                                for (int idx = bitInUSE.nextSetBit(0); idx >= 0; idx = bitInUSE.nextSetBit(idx+1)) {
                                    if (defTargetArray[i].getExpression().equals(globalUseArray[idx].getExpression())) {
                                        defTargetArray[i].addUseChain(globalUseArray[idx]);
                                        addUseDefChain(defTargetArray[i], globalUseArray[idx].getExpression(), globalUseArray[idx].getDFANode(), globalUseArray[idx].getProcedure());
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // at exit node, should handle INuse
            if (ChainTools.isExitNode(cfgNode, cfg)) {
                BitSet inBitSet = cfgNode.getData("InSet"); // reaching def
                // Ref
                Set<DFANode> exitNodeSetRef = cfgNode.getData("psg_exit_ref");
                if (exitNodeSetRef != null) {
                    for (DFANode exitNode : exitNodeSetRef) {
                        Set<AnalysisTarget> inUSE = exitNode.getData("INuse"); // use from callers
                        IDExpression param = ChainTools.getIDExpression(((AnalysisTarget) exitNode.getData("param")).getExpression());
                        for (AnalysisTarget target : inUSE) {
                            for (int i = inBitSet.nextSetBit(0); i >= 0; i = inBitSet.nextSetBit(i+1)) {
                                if (defTargetArray[i].getExpression().equals(param)) {
                                    defTargetArray[i].addUseChain(target);
                                    addUseDefChain(defTargetArray[i], target.getExpression(), target.getDFANode(), target.getProcedure());
                                } else { // handle special case
                                	if (ChainTools.isStructureAccess(target.getExpression(), target.getProcedure())) {
                	            		if (ChainTools.isStructureAccess(defTargetArray[i].getExpression(), targetProc)) {
                	            			Symbol useSymbol = SymbolTools.getSymbolOf(target.getExpression());
                	            			Symbol defSymbol = SymbolTools.getSymbolOf(defTargetArray[i].getExpression());
                	            			if (useSymbol == null || defSymbol == null) {
                	            				continue;
                	            			}
                	            			if (useSymbol.getTypeSpecifiers().get(0).equals(defSymbol.getTypeSpecifiers().get(0)) == false) {
                	            				continue;
                	            			}
//                	            			System.out.println("Add chain in special alias case: currentUse: " + target.getExpression() + ", reaching def: " + defTargetArray[i].getExpression());
                                            defTargetArray[i].addUseChain(target);
                                            addUseDefChain(defTargetArray[i], target.getExpression(), target.getDFANode(), target.getProcedure());
                	            		}
               	            		}
                                }
                            }
                        }
                    }
                }
                // Global
                DFANode exitNodeGlobal = cfgNode.getData("psg_exit_global");
                if (exitNodeGlobal != null) {
                    AnalysisTarget[] globalUseArray = new AnalysisTarget[globalUseSet.size()];
                    globalUseSet.toArray(globalUseArray);
                    BitSet bitInUSE = exitNodeGlobal.getData("INuse");
                    for (int i = inBitSet.nextSetBit(0); i >= 0; i = inBitSet.nextSetBit(i+1)) {
                        for (int idx = bitInUSE.nextSetBit(0); idx >= 0; idx = bitInUSE.nextSetBit(idx+1)) {
                            if (defTargetArray[i].getExpression().equals(globalUseArray[idx].getExpression())) {
                                defTargetArray[i].addUseChain(globalUseArray[idx]);
                                addUseDefChain(defTargetArray[i], globalUseArray[idx].getExpression(), globalUseArray[idx].getDFANode(), globalUseArray[idx].getProcedure());
                            }
                        }
                    }
                }
            }
        }
    	System.out.println("[ChainComputer.computeChain]End   proc: " + targetProc.getSymbolName() + ", time: " + (new Date()).toString());
	}

    private AnalysisTarget[] getDefTargetArray(Procedure proc) {
        Set<AnalysisTarget> targetSet = defTargetSetMap.get(proc);
        AnalysisTarget defMapEntry[] = new AnalysisTarget[targetSet.size()];
        targetSet.toArray(defMapEntry);
        return defMapEntry;
    }

    private AnalysisTarget[] getUseTargetArray(Procedure proc) {
        Set<AnalysisTarget> targetSet = useTargetSetMap.get(proc);
        if (targetSet == null) {
            return null;
        }
        AnalysisTarget useTargetArray[] = new AnalysisTarget[targetSet.size()];
        targetSet.toArray(useTargetArray);
        return useTargetArray;
    }
	
    private void addUseDefChain(AnalysisTarget def, Expression use, DFANode node, Procedure proc) {
        if (def.getDFANode().getData("param") != null) {
            if (def.getProcedure().equals(proc) == false) {
                return;
            }
        }
        Expression idEx = null;
    	HashMap<MyType, Boolean> typeMap = ChainTools.typeCache.get(System.identityHashCode(use));
    	boolean isArrayAccessWithConstantIndex;
    	if (typeMap != null) {
    		Boolean b = typeMap.get(MyType.ArrayWithConstantIndex);
    		if (b != null) {
    			isArrayAccessWithConstantIndex = b.booleanValue();
    		} else {
    			isArrayAccessWithConstantIndex = ChainTools.isArrayAccessWithConstantIndex(use);
    		}
    	} else {
    		isArrayAccessWithConstantIndex = ChainTools.isArrayAccessWithConstantIndex(use);
    	}
        if (isArrayAccessWithConstantIndex) {
//        if (ChainTools.isArrayAccessWithConstantIndex(use)) {
            idEx = use;
        } else {
            idEx = ChainTools.getIDExpression(use);
        }
        Set<AnalysisTarget> useTargetSet = useTargetSetMap.get(proc);
        if (useTargetSet == null) {
            useTargetSet = new LinkedHashSet<AnalysisTarget>();
            useTargetSetMap.put(proc, useTargetSet);
        }
        boolean useFound = false;
        synchronized (useTargetSetMap) {
            for (AnalysisTarget useTarget : useTargetSet) {
                if (System.identityHashCode(useTarget.getExpression()) == System.identityHashCode(idEx)) {
                    useTarget.addDefChain(def);
                    useFound = true;
                }
            }
	        if (useFound == false) {
	            AnalysisTarget useTarget = new AnalysisTarget(use, node, proc);
	            useTarget.addDefChain(def);
	            useTargetSet.add(useTarget);
	        }
        }
    }
    
    private void generateChainForDefinedArrayDeclarator(
            DFANode cfgNode,
            AnalysisTarget[] defSetArray,
            VariableDeclarator declarator,
            Procedure proc) {
        List<Expression> useList = ChainTools.getUseListInDec(declarator);
        for (Expression currentUse : useList) {
            boolean defFound = false;
            for (int i = 0; i < defSetArray.length; i++) {
                if (ChainTools.hasSameToString(defSetArray[i].getExpression(), currentUse) &&
                        ChainTools.isDefInItself(cfgNode, defSetArray[i].getExpression(), currentUse)) {
                    defSetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
                    addUseDefChain(defSetArray[i], currentUse, cfgNode, proc);
                    defFound = true;
                    break;
                }
            }
            if (defFound == false) {
            	BitSet inBitSet = cfgNode.getData("InSet");
            	for (int i = inBitSet.nextSetBit(0); i >= 0; i = inBitSet.nextSetBit(i+1)) {
                    if (ChainTools.hasSameToString(defSetArray[i].getExpression(), currentUse)) {
                        defSetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
                        addUseDefChain(defSetArray[i], currentUse, cfgNode, proc);
                    }
                }
            }
        }
    }

    private void generateChainForDefinedArrayNestedDeclarator(
            DFANode cfgNode,
            AnalysisTarget[] defSetArray,
            NestedDeclarator declarator,
            Procedure proc) {
        List<Expression> useList = ChainTools.getUseListInNestedDec(declarator);
        for (Expression currentUse : useList) {
            boolean defFound = false;
            for (int i = 0; i < defSetArray.length; i++) {
                if (ChainTools.hasSameToString(defSetArray[i].getExpression(), currentUse) &&
                        ChainTools.isNestedDefInItself(cfgNode, defSetArray[i].getExpression(), currentUse)) {
                    defSetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
                    addUseDefChain(defSetArray[i], currentUse, cfgNode, proc);
                    defFound = true;
                    break;
                }
            }
            if (defFound == false) {
            	BitSet inBitSet = cfgNode.getData("InSet");
            	for (int i = inBitSet.nextSetBit(0); i >= 0; i = inBitSet.nextSetBit(i+1)) {
                    if (ChainTools.hasSameToString(defSetArray[i].getExpression(), currentUse)) {
                        defSetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
                        addUseDefChain(defSetArray[i], currentUse, cfgNode, proc);
                    }
                }
            }
        }
    }

    private void generateChainForPlainDeclarator(
            DFANode cfgNode,
            AnalysisTarget[] defMapEntry,
            VariableDeclarator declarator,
            Procedure proc) {
        List<Expression> useSet = ChainTools.getUseListInDec(declarator);
        for (Expression currentUse : useSet) {
            for (int i = 0; i < defMapEntry.length; i++) {
                if (ChainTools.isArrayAccess(currentUse)) {
                    generateChainForArrayAccess(currentUse, defMapEntry, cfgNode, proc);
                } else if (ChainTools.isPointerAccess(currentUse)) {
                    if (ChainTools.isStructureAccess(currentUse, proc)) {
                        generateChainForStructure(currentUse, defMapEntry, cfgNode, proc);
                    } else {
                        generateChainForPointerAccess(currentUse, defMapEntry, cfgNode, proc);
                    }
                } else if (ChainTools.isStructureAccess(currentUse, proc)) {
                    generateChainForStructure(currentUse, defMapEntry, cfgNode, proc);
                } else {
                    generateChainForPlainExpression(currentUse, defMapEntry, cfgNode, proc);
                }
                Object nodeIR = CFGraph.getIR(cfgNode);
                generateChainForAlias(nodeIR, currentUse, defMapEntry, cfgNode, proc);
            }
        }
    }

    private void generateChainForPlainNestedDeclarator(
            DFANode cfgNode,
            AnalysisTarget[] defMapEntry,
            NestedDeclarator declarator,
            Procedure proc) {
        List<Expression> useSet = ChainTools.getUseListInNestedDec(declarator);
        for (Expression currentUse : useSet) {
            for (int i = 0; i < defMapEntry.length; i++) {
                if (ChainTools.isArrayAccess(currentUse)) {
                    generateChainForArrayAccess(currentUse, defMapEntry, cfgNode, proc);
                } else if (ChainTools.isPointerAccess(currentUse)) {
                    if (ChainTools.isStructureAccess(currentUse, proc)) {
                        generateChainForStructure(currentUse, defMapEntry, cfgNode, proc);
                    } else {
                        generateChainForPointerAccess(currentUse, defMapEntry, cfgNode, proc);
                    }
                } else if (ChainTools.isStructureAccess(currentUse, proc)) {
                    generateChainForStructure(currentUse, defMapEntry, cfgNode, proc);
                } else {
                    generateChainForPlainExpression(currentUse, defMapEntry, cfgNode, proc);
                }
                Object nodeIR = CFGraph.getIR(cfgNode);
                generateChainForAlias(nodeIR, currentUse, defMapEntry, cfgNode, proc);
            }
        }
    }

    private void generateChainForArrayAccess(
            Expression currentUse,
            AnalysisTarget[] defTargetArray,
            DFANode cfgNode,
            Procedure proc) {
    	// check
    	HashMap<MyType, Boolean> typeMap = ChainTools.typeCache.get(System.identityHashCode(currentUse));
    	boolean isArrayAccessWithConstantIndex, isArrayAccessWithNoIndex,
    			isArrayAccessWithPartiallyConstantIndex, isArrayAccessWithVariableIndex;
    	if (typeMap != null) {
    		Boolean b = typeMap.get(MyType.ArrayWithConstantIndex);
    		// constanct idx
    		if (b != null) {
    			isArrayAccessWithConstantIndex = b.booleanValue();
    		} else {
    			isArrayAccessWithConstantIndex = ChainTools.isArrayAccessWithConstantIndex(currentUse);
    		}
    		// with no idx
    		b = typeMap.get(MyType.ArrayWithNoIndex);
    		if (b != null) {
    			isArrayAccessWithNoIndex = b.booleanValue();
    		} else {
    			isArrayAccessWithNoIndex = ChainTools.isArrayAccessWithNoIndex(currentUse);
    		}
    		//
    		b = typeMap.get(MyType.ArrayWithPartiallyConstantIndex);
    		if (b != null) {
    			isArrayAccessWithPartiallyConstantIndex = b.booleanValue();
    		} else {
    			isArrayAccessWithPartiallyConstantIndex = ChainTools.isArrayAccessWithPartiallyConstantIndex(currentUse);
    		}
    		//
    		b = typeMap.get(MyType.ArrayWithVariableIndex);
    		if (b != null) {
    			isArrayAccessWithVariableIndex = b.booleanValue();
    		} else {
    			isArrayAccessWithVariableIndex = ChainTools.isArrayAccessWithVariableIndex(currentUse);
    		}
    	} else {
    		isArrayAccessWithConstantIndex = ChainTools.isArrayAccessWithConstantIndex(currentUse);
    		isArrayAccessWithNoIndex = ChainTools.isArrayAccessWithNoIndex(currentUse);
    		isArrayAccessWithVariableIndex = ChainTools.isArrayAccessWithVariableIndex(currentUse);
    		isArrayAccessWithPartiallyConstantIndex = ChainTools.isArrayAccessWithPartiallyConstantIndex(currentUse);
    	}
    	//
        if (isArrayAccessWithNoIndex ||
        		isArrayAccessWithVariableIndex ||
        		isArrayAccessWithPartiallyConstantIndex) {
//            if (ChainTools.isArrayAccessWithNoIndex(currentUse) ||
//            		ChainTools.isArrayAccessWithVariableIndex(currentUse) ||
//            		ChainTools.isArrayAccessWithPartiallyConstantIndex(currentUse)) {
        	BitSet inBitSet = cfgNode.getData("InSet");
        	for (int i = inBitSet.nextSetBit(0); i >= 0; i = inBitSet.nextSetBit(i+1)) {
                if (ChainTools.hasSameArrayIdentifier(currentUse, defTargetArray[i].getExpression())) {
                    defTargetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
                    addUseDefChain(defTargetArray[i], currentUse, cfgNode, proc);
                }
            }
        } else if (isArrayAccessWithConstantIndex) {
//        } else if (ChainTools.isArrayAccessWithConstantIndex(currentUse)) {
        	BitSet inBitSet = cfgNode.getData("InSet");
        	for (int i = inBitSet.nextSetBit(0); i >= 0; i = inBitSet.nextSetBit(i+1)) {
                if (ChainTools.hasSameArrayIdentifier(currentUse, defTargetArray[i].getExpression())) {
                    if (ChainTools.isArrayAccessWithConstantIndex(defTargetArray[i].getExpression())) {
                        if (ChainTools.hasSameToString(currentUse, defTargetArray[i].getExpression())) {
                            defTargetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
                            addUseDefChain(defTargetArray[i], currentUse, cfgNode, proc);
                        }
                    } else {
                        defTargetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
                        addUseDefChain(defTargetArray[i], currentUse, cfgNode, proc);
                    }
                }
            }
        } else {
            ChainTools.traverseIR(currentUse);
            throw new RuntimeException("Unexpected Expression: " + "currentUse: " + currentUse + " (Unknown ArrayType)");
        }
    }

    private void generateChainForStructure(
            Expression currentUse,
            AnalysisTarget[] defTargetArray,
            DFANode cfgNode,
            Procedure proc) {
    	BitSet inBitSet = cfgNode.getData("InSet");
    	for (int i = inBitSet.nextSetBit(0); i >= 0; i = inBitSet.nextSetBit(i+1)) {
            // handle exact matching
            if (ChainTools.isStructureAccess(defTargetArray[i].getExpression(), proc)) {
                if (ChainTools.getMemberOnlyInStruct(currentUse) != null &&
                        ChainTools.getMemberOnlyInStruct(defTargetArray[i].getExpression()) != null) {
                    // use: var->member, def: var->member case
                    if (ChainTools.isArrayAccessInStruct(currentUse, proc)) {
                        if (ChainTools.hasSameArrayAccessInStruct(defTargetArray[i].getExpression(), currentUse, proc)) {
                            defTargetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
                            addUseDefChain(defTargetArray[i], currentUse, cfgNode, proc);
                        }
                    } else {
                        if (ChainTools.hasSameToString(
                                ChainTools.getIDVariablePlusMemberInStruct(defTargetArray[i].getExpression()),
                                ChainTools.getIDVariablePlusMemberInStruct(currentUse))) {
                            defTargetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
                            addUseDefChain(defTargetArray[i], currentUse, cfgNode, proc);
                        }
                    }
                } else if (ChainTools.getMemberOnlyInStruct(currentUse) == null &&
                        ChainTools.getMemberOnlyInStruct(defTargetArray[i].getExpression()) == null) {
                    // use: var, def: var case
                    if (ChainTools.hasSameToString(defTargetArray[i].getExpression(), currentUse)) {
                        defTargetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
                        addUseDefChain(defTargetArray[i], currentUse, cfgNode, proc);
                    }
                } else if (ChainTools.getMemberOnlyInStruct(currentUse) != null &&
                        ChainTools.getMemberOnlyInStruct(defTargetArray[i].getExpression()) == null) {
                    // use: var->member, def: var case
                    if (ChainTools.isArrayAccessInStruct(currentUse, proc)) {
                        if (ChainTools.hasSameArrayAccessInStruct(defTargetArray[i].getExpression(), currentUse, proc)) {
                            defTargetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
                            addUseDefChain(defTargetArray[i], currentUse, cfgNode, proc);
                        }
                    } else {
                        if (ChainTools.hasSameStructureVariableIdentifier(defTargetArray[i].getExpression(), currentUse, proc)) {
                            defTargetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
                            addUseDefChain(defTargetArray[i], currentUse, cfgNode, proc);
                        }
                    }
                } else if (ChainTools.getMemberOnlyInStruct(currentUse) == null &&
                        ChainTools.getMemberOnlyInStruct(defTargetArray[i].getExpression()) != null) {
                    // use: var, def: var->member case
                    if (ChainTools.hasSameStructureVariableIdentifier(defTargetArray[i].getExpression(), currentUse, proc)) {
                        defTargetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
                        addUseDefChain(defTargetArray[i], currentUse, cfgNode, proc);
                    }
                } else {
                    throw new RuntimeException("currentUse: " + currentUse + ", defMapEntry[i].expression: " + defTargetArray[i].getExpression());
                }
            }
        } //  for (int i = 0; i < useTargetArray.length; i++)
    }

    private void generateChainForPlainExpression(
            Expression currentUse,
            AnalysisTarget[] defTargetArray,
            DFANode cfgNode,
            Procedure proc) {
    	BitSet inBitSet = cfgNode.getData("InSet");
    	for (int i = inBitSet.nextSetBit(0); i >= 0; i = inBitSet.nextSetBit(i+1)) {
            // handle exact matching
            if (ChainTools.hasSameToString(defTargetArray[i].getExpression(), currentUse)) {
                defTargetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
                addUseDefChain(defTargetArray[i], currentUse, cfgNode, proc);
            } else if (defTargetArray[i].getExpression() instanceof ArrayAccess ||
                    defTargetArray[i].getExpression() instanceof UnaryExpression) {
                // Handle the case of accessing array and pointer dereferencing
                Expression id = ChainTools.getIDExpression(defTargetArray[i].getExpression());
                if (ChainTools.hasSameToString(id, currentUse)) {
                    defTargetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
                    addUseDefChain(defTargetArray[i], currentUse, cfgNode, proc);
                }
            }
        } //  for (int i = 0; i < useTargetArray.length; i++)
    }

    private void generateChainForPointerAccess(
            Expression currentUse,
            AnalysisTarget[] defTargetArray,
            DFANode cfgNode,
            Procedure proc) {
    	BitSet inBitSet = cfgNode.getData("InSet");
    	for (int i = inBitSet.nextSetBit(0); i >= 0; i = inBitSet.nextSetBit(i+1)) {
            // handle exact matching
            if (ChainTools.hasSameToString(defTargetArray[i].getExpression(), currentUse)) {
                defTargetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
                addUseDefChain(defTargetArray[i], currentUse, cfgNode, proc);
            } else {
                // Handle the case of accessing array and pointer dereferencing
                Expression defId = ChainTools.getIDExpression(defTargetArray[i].getExpression());
                Expression useId = ChainTools.getIDExpression(currentUse);
                if (defId == null || useId == null) {
                    continue;
                }
                if (currentUse instanceof ArrayAccess) {
                    if (ChainTools.hasSameToString(defId, useId)) {
                        defTargetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
                        addUseDefChain(defTargetArray[i], currentUse, cfgNode, proc);
                    }
                } else {
                    if (ChainTools.hasSameToString(defId, currentUse)) {
                        defTargetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
                        addUseDefChain(defTargetArray[i], currentUse, cfgNode, proc);
                    }
                }
            }
        } //  for (int i = 0; i < useTargetArray.length; i++)
    }

    private void generateChainForAlias(
            Object nodeIR,
            Expression currentUse,
            AnalysisTarget[] defTargetArray,
            DFANode cfgNode,
            Procedure proc) {
        // Apply the alias analysis information to the UD chain
        // If the alias is definite, the def of the alias should be in the def chain
        // If the alias is possible, the def of itself and the alias should be in the def chain
        if (nodeIR instanceof Statement) {
            Set<Symbol> definedSymbolsInProc = ChainTools.getDefSymbol(defTargetSetMap.get(proc), proc);//DataFlowTools.getDefSymbol(proc);
            Expression cuId = ChainTools.getIDExpression(currentUse);
            if (cuId == null || (cuId instanceof Identifier) == false) {
                return;
            }
            Symbol symForCurrentEx = ((Identifier) cuId).getSymbol();
            Domain aliasInfo = IPPointsToAnalysis.getPointsToRelations((Statement) nodeIR);
            PointsToDomain aliasInfoForCurrentStmt = null;
            if (aliasInfo instanceof PointsToDomain) {
                aliasInfoForCurrentStmt = (PointsToDomain) aliasInfo;
            }
            if (aliasInfoForCurrentStmt != null) {
                for (Symbol definedSym : definedSymbolsInProc) {   // def symbols for the proc
                    Set<PointsToRel> aliasSetAffectingCurrentStmt = aliasInfoForCurrentStmt.get(definedSym);
                    if (aliasSetAffectingCurrentStmt != null) {
                        for (PointsToRel aliasInstance : aliasSetAffectingCurrentStmt) {
                            if (symForCurrentEx.getSymbolName().equals(aliasInstance.getPointedToSymbol().getSymbolName())) {
                            	BitSet inBitSet = cfgNode.getData("InSet");
                            	for (int i = inBitSet.nextSetBit(0); i >= 0; i = inBitSet.nextSetBit(i+1)) {
                                    Expression defID = ChainTools.getIDExpression(defTargetArray[i].getExpression());
                                    if (defID.toString().equals(aliasInstance.getPointerSymbol().getSymbolName())) {
                                        defTargetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
                                        addUseDefChain(defTargetArray[i], currentUse, cfgNode, proc);
                                    }
                                }
                            } else if (symForCurrentEx.getSymbolName().equals(aliasInstance.getPointerSymbol().getSymbolName())) {
                            	BitSet inBitSet = cfgNode.getData("InSet");
                            	for (int i = inBitSet.nextSetBit(0); i >= 0; i = inBitSet.nextSetBit(i+1)) {
                                    Expression defID = ChainTools.getIDExpression(defTargetArray[i].getExpression());
                                    if (defID.toString().equals(aliasInstance.getPointedToSymbol().getSymbolName())) {
                                        defTargetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
                                        addUseDefChain(defTargetArray[i], currentUse, cfgNode, proc);
                                    }
                                }
                            } else {
                                // Do nothing
                            }
                        } // for (PointsToRel aliasInstance : aliasSetAffectingCurrentStmt)
                    } // if (aliasSetAffectingCurrentStmt != null)
                } // for (Symbol definedSym : definedSymbolsInProc)
                //
//            } else {// if (aliasInfoForCurrentStmt != null)
//            	// POINTS-TO-UNIVERSE // handle recurring structure case
//            	if (ChainTools.isStructureAccess(currentUse, proc)) {
//	            	BitSet inBitSet = cfgNode.getData("InSet");
//	            	for (int i = inBitSet.nextSetBit(0); i >= 0; i = inBitSet.nextSetBit(i+1)) {
//	            		if (ChainTools.isStructureAccess(defTargetArray[i].getExpression(), proc)) {
//	            			Symbol useSymbol = SymbolTools.getSymbolOf(currentUse);
//	            			Symbol defSymbol = SymbolTools.getSymbolOf(defTargetArray[i].getExpression());
//	            			if (useSymbol == null || defSymbol == null) {
//	            				continue;
//	            			}
//	            			if (useSymbol.getTypeSpecifiers().get(0).equals(defSymbol.getTypeSpecifiers().get(0)) == false) {
//	            				continue;
//	            			}
//	            			if (cfgNode.equals(defTargetArray[i].getDFANode())) {
//	            				continue;
//	            			}
//	            			System.out.println("######### Struct to check #### Stmt: " + nodeIR + ", proc: " + proc.getSymbolName());
//	            			System.out.println("Struct to check: " + useSymbol.getTypeSpecifiers().get(0));
//	            			System.out.println("currentUse: " + currentUse + ", reaching def: " + defTargetArray[i].getExpression());
//		                    Expression defID = ChainTools.getIDExpression(defTargetArray[i].getExpression());
//			                defTargetArray[i].addUseChain(new AnalysisTarget(currentUse, cfgNode, proc));
//			                addUseDefChain(defTargetArray[i], currentUse, cfgNode, proc);
//	            		}
//	            	}
//            	}
            }
        }
    }

    private void handleLocalStaticVariables(Procedure proc, AnalysisTarget[] defTargetArray) {
    	if (PrintTools.getVerbosity() > 1) {
    		System.out.println("[IPChainAnalysis.handleLocalStaticVariables] begin for proc: " + proc.getSymbolName());
    	}
        // keep local static variable. chains for reaching definitions at exit node to upward exposed uses at entry should be added 
        Set<Integer> staticDefIdxs = new HashSet<Integer>();
        DepthFirstIterator<Traversable> dfi = new DepthFirstIterator<Traversable>(proc);
        while (dfi.hasNext()) {
        	Traversable ir = dfi.next();
        	if (ir instanceof VariableDeclaration) {
        		VariableDeclaration vd = (VariableDeclaration) ir;
        		if (vd.getSpecifiers().contains(Specifier.STATIC)) {
        			if (PrintTools.getVerbosity() > 2) {
        				System.out.println("Static Declaration: " + vd);
        			}
        			int numDec = vd.getNumDeclarators();
        			List<IDExpression> idList = vd.getDeclaredIDs();
        			for (IDExpression id: idList) {
        				if (PrintTools.getVerbosity() > 2) {
        					System.out.println("id: " + id);
        				}
        				int idx = 0;
        				for (AnalysisTarget at: defTargetArray) {        					
        					if (at.getExpression() instanceof IDExpression &&
        							((at.getExpression() instanceof Identifier) == false)) {
        						if (at.getExpression().equals(id)) {
        							if (PrintTools.getVerbosity() > 2) {
        								System.out.println("Found in def: " + at.getExpression() +", idx: " + idx);
        							}
        							staticDefIdxs.add(idx);
        						}
        					}
        					idx++;
        				}
        			}
        		}
        	}
        }
        // find all the reaching def node and check the def is used in the node
        Map<Integer, Set<DFANode>> reachingDefNodes = new HashMap<Integer, Set<DFANode>>();
        CFGraph cfg = cfgMap.get(proc);
        Iterator iter = cfg.iterator();
        while (iter.hasNext()) {
            DFANode cfgNode = (DFANode) iter.next();
            Traversable nodeIR = (Traversable) CFGraph.getIR(cfgNode);
            if (nodeIR == null) {
            	continue;
            }
            BitSet inDefSet = cfgNode.getData("InSet");
            for (int idx: staticDefIdxs) {
            	if (inDefSet.get(idx)) {
            		List<Expression> useList = ChainTools.getUseList(nodeIR);
            		for (Expression ex: useList) {
            			if (defTargetArray[idx].getExpression().equals(ex)) {
                    		Set<DFANode> nodeSet = reachingDefNodes.get(idx);
                    		if (nodeSet == null) {
                    			nodeSet = new HashSet<DFANode>();
                    			reachingDefNodes.put(idx, nodeSet);
                    		}
                    		nodeSet.add(cfgNode);
                    		if (PrintTools.getVerbosity() > 2) {
                    			System.out.println("reaching static def found: " + nodeIR);
                    		}
                    		break;
            			}
            		}
            	}
            }
        }
        // find last definition of the static variable
        List<DFANode> exitNodes = cfg.getExitNodes();
        for (DFANode node: exitNodes) {
        	BitSet inDefSet = node.getData("InSet");
        	for (int i = inDefSet.nextSetBit(0); i >= 0; i = inDefSet.nextSetBit(i+1)) {
        	    // operate on index i here
        		for (int idx: staticDefIdxs) {
        			if (defTargetArray[i].getExpression().equals(defTargetArray[idx].getExpression())) {
        		        // chain from last def to all the uses found
        				Set<DFANode> useNodes = reachingDefNodes.get(idx);
        				for (DFANode useNode: useNodes) {
        					List<Expression> useList = ChainTools.getUseList((Traversable)useNode.getData("ir"));
        					for (Expression ex: useList) {
        						if (defTargetArray[i].getExpression().equals(ex)) {
			                        defTargetArray[i].addUseChain(new AnalysisTarget(ex, useNode, proc));
			                        addUseDefChain(defTargetArray[i], ex, useNode, proc);
        						}
        					}
        				}
        			}
        		}
        	}
        }
    	if (PrintTools.getVerbosity() > 1) {
    		System.out.println("[IPChainAnalysis.handleLocalStaticVariables] end for proc: " + proc.getSymbolName());
    	}
    }
    
}
