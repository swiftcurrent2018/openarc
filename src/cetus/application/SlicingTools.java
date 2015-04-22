package cetus.application;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

import cetus.analysis.CFGraph;
import cetus.analysis.DFAGraph;
import cetus.analysis.DFANode;
import cetus.hir.PrintTools;
import cetus.hir.Procedure;

/**
 * This class provides several utility functions for the program slicing.
 * @author Jae-Woo Lee, <jaewoolee@purdue.edu>
 *         School of ECE, Purdue University
 */
public class SlicingTools {
	public static DFAGraph reverseCFGraph(CFGraph cfg) {
		DFAGraph reversedCFG = new DFAGraph();
		Iterator<DFANode> cfgIter = cfg.iterator();
		// reverse CFG for building post dominance tree
		while (cfgIter.hasNext()) {
			DFANode currentNode = cfgIter.next();
			DFANode newRevNode = currentNode.getData("revNode");
			if (newRevNode == null) {
				newRevNode = new DFANode();
			}
			currentNode.putData("revNode", newRevNode);
			newRevNode.putData("cfgNode", currentNode);
			reversedCFG.addNode(newRevNode);
			Set<DFANode> preds = currentNode.getPreds();
			if (preds != null) {
				for (DFANode pred : preds) {
					DFANode revPred = pred.getData("revNode");
					if (revPred == null) {
						revPred = new DFANode();
					}
					revPred.putData("cfgNode", pred);
					pred.putData("revNode", revPred);
					reversedCFG.addNode(revPred);
					revPred.addPred(newRevNode);
					newRevNode.addSucc(revPred);
					reversedCFG.addEdge(newRevNode, revPred);
				}
			}
			Set<DFANode> succs = currentNode.getSuccs();
			if (succs != null) {
				for (DFANode succ : succs) {
					DFANode revSucc = succ.getData("revNode");
					if (revSucc == null) {
						revSucc = new DFANode();
					}
					revSucc.putData("cfgNode", succ);
					succ.putData("revNode", revSucc);
					reversedCFG.addNode(revSucc);
					revSucc.addSucc(newRevNode);
					newRevNode.addPred(revSucc);
					reversedCFG.addEdge(revSucc, newRevNode);
				}
			}
		}
		// verify if the reversed CFG is correct
		if (PrintTools.getVerbosity() > 1)
			System.out.println("###### Reversed CFG #######");
		Iterator<DFANode> revIter = reversedCFG.iterator();
		while (revIter.hasNext()) {
			DFANode revNode = revIter.next();
			DFANode cfgNode = (DFANode) revNode.getData("cfgNode");
			if (PrintTools.getVerbosity() > 1)
				System.out.println("CurrentNode: " + cfgNode.getData("ir"));
			Set<DFANode> preds = revNode.getPreds();
			if (preds != null) {
				for (DFANode pred : preds) {
					DFANode cfgPred = pred.getData("cfgNode");
					if (PrintTools.getVerbosity() > 1)
						System.out.println("    Pred: " + cfgPred.getData("ir"));
				}
			}
			Set<DFANode> succs = revNode.getSuccs();
			if (succs != null) {
				for (DFANode succ : succs) {
					DFANode cfgSucc = succ.getData("cfgNode");
					if (PrintTools.getVerbosity() > 1)
						System.out.println("    Succ: " + cfgSucc.getData("ir"));
				}
			}
		}
		// remove the temporary information from CFGraph
		cfgIter = cfg.iterator();
		while (cfgIter.hasNext()) {
			DFANode currentNode = (DFANode) cfgIter.next();
			currentNode.removeData("revNode");
		}
		return reversedCFG;
	}

	public static BitSet[] getDominators(final DFAGraph graph,
			ArrayList<DFANode> nodeList) {
		Iterator<DFANode> cfgIter = graph.iterator();
		while (cfgIter.hasNext()) {
			DFANode currentNode = cfgIter.next();
			nodeList.add(currentNode);
		}
		// build dominator
		final int nodeSize = graph.size();
		DFANode[] nodeArray = new DFANode[nodeSize];
		nodeList.toArray(nodeArray);
		BitSet[] dominator = new BitSet[nodeSize];
		for (int i = 0; i < nodeSize; i++) {
			dominator[i] = new BitSet(nodeSize);
			dominator[i].clear();
			dominator[i].set(0, nodeSize);
		}
		List<DFANode> entryNodeList = graph.getEntryNodes();
		Set<Integer> entryIdxSet = new HashSet<Integer>();
		for (DFANode entryNode : entryNodeList) {
			int entryIdx = nodeList.indexOf(entryNode);
			entryIdxSet.add(Integer.valueOf(entryIdx));
			dominator[entryIdx].clear();
			dominator[entryIdx].set(entryIdx);
		}

		boolean change = true;
		while (change) {
			change = false;
			for (int i = 0; i < nodeSize; i++) {
				if (entryIdxSet.contains(Integer.valueOf(i)) == false) {
					BitSet tmp = new BitSet(nodeSize);
					tmp.set(0, nodeSize);
					Set<DFANode> preds = nodeArray[i].getPreds();
					if (preds == null) {
						continue;
					}
					for (DFANode pred : preds) {
						tmp.and(dominator[nodeList.indexOf(pred)]);
					}
					BitSet dom = (BitSet) tmp.clone();
					dom.set(i);
					if (dom.equals(dominator[i]) == false) {
						change = true;
						dominator[i] = (BitSet) dom.clone();
					}
				}
			}
		}
		return dominator;
	}

	public static Set<Integer> getEntryIdxSet(DFAGraph graph, List<DFANode> nodeList) {
		List<DFANode> entryNodeList = graph.getEntryNodes();
		Set<Integer> entryIdxSet = new HashSet<Integer>();
		for (DFANode entryNode : entryNodeList) {
			int entryIdx = nodeList.indexOf(entryNode);
			entryIdxSet.add(Integer.valueOf(entryIdx));
		}
		return entryIdxSet;
	}
	
	public static BitSet[] getImmediateDominator(
			BitSet[] dominator, 
			Set<Integer> entryIdxSet,
			List<DFANode> entryNodeList,
			DFANode[] nodeArray) {
		int nodeSize = dominator.length;
        BitSet[] immediateDom = new BitSet[nodeSize];
        // remove myself from dominator
        for (int i = 0; i < nodeSize; i++) {
            immediateDom[i] = (BitSet) dominator[i].clone();
            immediateDom[i].clear(i);
        }
        //
        for (int n = 0; n < nodeSize; n++) {
        	// do not consider the entry node
            if (entryIdxSet.contains(Integer.valueOf(n))) {
                continue;
            }
            // 
            for (int s = 0; s < nodeSize; s++) {
            	// my idom bitset cantains node s --> s is my dom
                if (immediateDom[n].get(s)) {
                	// remove my dom's dom node from my bitset
                    for (int t = 0; t < nodeSize; t++) {
                        if (t == s) {
                            continue;
                        }
                        if ((immediateDom[n].get(t) == true)
                        		&& immediateDom[s].get(t)) {
                            immediateDom[n].clear(t);
                        }
                    }
                }
            }
        }
        // choose only one if there are several idom
        for (int current = 0; current < nodeSize; current++) {
   			if (immediateDom[current].cardinality() > 1) {
       			if (immediateDom[current].cardinality() < 1) {
       				throw new RuntimeException("No idom after remove: " + nodeArray[current].getData("ir") + ", cardinality: " + immediateDom[current].cardinality());
       			} else if (immediateDom[current].cardinality() > 1) {
       				if (PrintTools.getVerbosity() > 1) {
	       				System.out.println("Current: " + nodeArray[current].getData("cfgNode"));
	       				for (int i = immediateDom[current].nextSetBit(0); i >= 0; i = immediateDom[current].nextSetBit(i+1)) {
	       					System.out.println(" immediate dominator: " + nodeArray[i].getData("cfgNode"));
	       				}
       				}
       				int first = immediateDom[current].nextSetBit(0);
       				immediateDom[current].clear(0, nodeSize);
       				immediateDom[current].set(first); // set only first one
       			}
   			}
        }
        return immediateDom;
	}
	
	public static DFAGraph buildImmediateDominanceTree(
			DFAGraph reversedCFG,
			BitSet[] immediateDom, 
			DFANode[] nodeArray,
			List<DFANode> nodeList) {
		int nodeSize = immediateDom.length;
        LinkedList<DFANode> workList = new LinkedList<DFANode>();
        DFAGraph idomTree = new DFAGraph();
        // add exit nodes
        List<DFANode> entryList = reversedCFG.getEntryNodes();
        if (entryList.size() != 1) {
        	throw new RuntimeException("The number of entry node should be one, but it is: " + entryList.size());
        }
        BitSet addedNodeSet = new BitSet(nodeSize);
        // add all the nodes which doesn't have immediate dominator
        for (int i = 0; i < nodeSize; i++) {
        	if (immediateDom[i].cardinality() == 0) {
                DFANode idomTreeEntry = new DFANode();
                idomTreeEntry.putData("revNode", nodeArray[i]);
                if (nodeArray[i].getData("cfgNode") != null) {
                	idomTreeEntry.putData("ir", ((DFANode)nodeArray[i].getData("cfgNode")).getData("ir") + "#" + ((DFANode)nodeArray[i].getData("cfgNode")).getData("tag"));
                }
                nodeArray[i].putData("pdtNode", idomTreeEntry);                
                idomTree.addNode(idomTreeEntry);
                addedNodeSet.set(i);
                workList.add(idomTreeEntry);
        	}
        }
        // process all nodes
        while (workList.isEmpty() == false) {
            DFANode parentIDTNode = (DFANode) workList.remove();
            DFANode parentRevNode = parentIDTNode.getData("revNode");
            int parentIdx = nodeList.indexOf(parentRevNode);
            for (int idx = 0; idx < nodeSize; idx++) {
                if (immediateDom[idx].get(parentIdx)) {
                    DFANode childIDTNode = new DFANode();
                    childIDTNode.putData("revNode", nodeArray[idx]);
                    if (nodeArray[idx].getData("cfgNode") != null) {
                    	childIDTNode.putData("ir", ((DFANode)nodeArray[idx].getData("cfgNode")).getData("ir") + "#" + ((DFANode)nodeArray[idx].getData("cfgNode")).getData("tag"));
                    }
                    parentIDTNode.addSucc(childIDTNode);
                    childIDTNode.addPred(parentIDTNode);
                    idomTree.addNode(childIDTNode);
                    addedNodeSet.set(idx);
                    idomTree.addEdge(parentIDTNode, childIDTNode);
                    nodeArray[idx].putData("pdtNode", childIDTNode);
                    workList.add(childIDTNode);
                }
            }
        }
        if (addedNodeSet.cardinality() != nodeSize) {
        	throw new RuntimeException("There is unchecked node: expected: " + nodeSize + ", actual: " + addedNodeSet.cardinality());
        }
        if (PrintTools.getVerbosity() > 1)
        	System.out.println(idomTree.toDot("ir", 1));
        return idomTree;
	}
	
	public static void printDominator(BitSet[] dominator, DFANode[] nodeArray, Procedure proc) {
		int nodeSize = dominator.length;
        System.out.println("##### Dominator for " + proc.getSymbolName() + " #####");
        for (int i = 0; i < nodeSize; i++) {
            System.out.print("Node: " + CFGraph.getIR((DFANode)nodeArray[i].getData("cfgNode")) + ", Dominator: ");
            for (int idx = 0; idx < nodeSize; idx++) {
                if (dominator[i].get(idx)) {
                    System.out.print(" [" + CFGraph.getIR((DFANode)nodeArray[idx].getData("cfgNode")) + "] ");
                }
            }
            System.out.println("");
        }
	}
	
	public static void printDominanceTree(DFAGraph postDomTree, Procedure proc) {
        System.out.println("###### Post Dominance Tree for : " + proc.getSymbolName());
        Iterator<DFANode> pdtIter = postDomTree.iterator();
        while (pdtIter.hasNext()) {
        	DFANode currentNode = pdtIter.next();
        	DFANode cfgNode = null;
        	if (postDomTree.getEntryNodes().contains(currentNode)) {
        		System.out.println("CurrentNode: ExitROOT");
        	} else {
        		cfgNode = ((DFANode)currentNode.getData("revNode")).getData("cfgNode");
            	System.out.println("CurrentNode: " + cfgNode.getData("ir") + ", tag: " + cfgNode.getData("tag"));
        	}
        	Set<DFANode> preds = currentNode.getPreds();
        	if (preds != null) {
        		for (DFANode pred: preds) {
        			if (postDomTree.getEntryNodes().contains(pred)) {
        				System.out.println("    Pred: ExitROOT");
        			} else {
        				DFANode cfgPred = ((DFANode)pred.getData("revNode")).getData("cfgNode");
        				System.out.println("    Pred: " + cfgPred.getData("ir"));
        			}
        		}
        	}
        	Set<DFANode> succs = currentNode.getSuccs();
        	if (succs != null) {
        		for (DFANode succ: succs) {
        			DFANode cfgSucc = ((DFANode)succ.getData("revNode")).getData("cfgNode");
        			System.out.println("    Succ: " + cfgSucc.getData("ir") + ", " + cfgSucc.getData("tag"));
        		}
        	}
        }
	}
	
    public static int longestSubstr(String first, String second) {
    	int idx;
    	int strLength = first.length() > second.length() ? second.length() : first.length();
    	for (idx = 0; idx < strLength; idx++) {
        	if (first.charAt(idx) != second.charAt(idx)) {
        		break;
        	}
    	}
    	return idx;
    }
}
