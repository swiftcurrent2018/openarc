package cetus.application;

import cetus.analysis.CFGraph;
import cetus.analysis.DFAGraph;
import cetus.analysis.DFANode;
import cetus.hir.Expression;
import cetus.hir.ForLoop;
import cetus.hir.PrintTools;
import cetus.hir.Procedure;
import cetus.hir.SwitchStatement;
import cetus.hir.Traversable;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.Date;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.StringTokenizer;

/**
 * This class builds the control dependence graph using the post dominance tree.
 * @author Jae-Woo Lee, <jaewoolee@purdue.edu>
 *         School of ECE, Purdue University
 */
public class ControlDependenceGraph {

    Map<Procedure, CFGraph> cfgMap;
    Set<Procedure> procSet;

    public void buildGraph(Map<Procedure, CFGraph> cfgMap) {
        this.cfgMap = cfgMap;
        buildCDG();
    }
    
    public void setProcSet(Set<Procedure> procSet) {
    	this.procSet = procSet;
    }

    private void buildCDG() {
    	if (this.procSet == null) {
    		procSet = cfgMap.keySet();
    	} 
        for (Procedure proc : procSet) {
        	System.out.println("Building Control Dependence Graph for Proc: " + proc.getSymbolName() + ", time: " + (new Date()).toString());
            CFGraph cfg = cfgMap.get(proc);
            // reverse the control flow graph
            DFAGraph reversedCFG = SlicingTools.reverseCFGraph(cfg);
            // add additional start node for connecting multiple return node
            // single return node works fine with this.
            DFANode dummyEntryNode = new DFANode();
            List<DFANode> entryList = reversedCFG.getEntryNodes();
            reversedCFG.addNode(dummyEntryNode);
            for (DFANode entry: entryList) {
            	dummyEntryNode.addSucc(entry);
            	reversedCFG.addEdge(dummyEntryNode, entry);
            }
            //
            ArrayList<DFANode> nodeList = new ArrayList<DFANode>();
            
            // extract dominator
            BitSet[] dominator = SlicingTools.getDominators(reversedCFG, nodeList);
            int nodeSize = nodeList.size();
            DFANode[] nodeArray = new DFANode[nodeSize];
            nodeList.toArray(nodeArray);
            Set<Integer> entryIdxSet = SlicingTools.getEntryIdxSet(reversedCFG, nodeList);
            List<DFANode> entryNodeList = reversedCFG.getEntryNodes();
            
            // print dominator
//            SlicingTools.printDominator(dominator, nodeArray, proc);
            
            // extract immediate dominator
            BitSet[] immediateDom = SlicingTools.getImmediateDominator(dominator, entryIdxSet, entryNodeList, nodeArray);
            
            // print immediate dominator
//            SlicingTools.printDominator(immediateDom, nodeArray, proc);
            
            // build Post Dominance Tree
            if (PrintTools.getVerbosity() > 1)
            	System.out.println("############# build Post Dominace Tree (Graph) for " + proc.getSymbolName() + " ###########");
            DFAGraph postDomTree = SlicingTools.buildImmediateDominanceTree(reversedCFG, immediateDom, nodeArray, nodeList);
            
            // print post dominance tree
//            SlicingTools.printDominanceTree(postDomTree, proc);
            
            // extract edges m --> n such that n does not postdominate(dominate in reversed) m
            // CFGEdge's head and tail are on reversed cfg.
            // can get PDTree node by getData("pdtNode")
            Set<CFGEdge> nonPDEdgeSet = extractNonPostdominatingEdges(reversedCFG, postDomTree, proc);

            List<DFANode> pdtNodeList = new ArrayList<DFANode>();
            Iterator<DFANode> pdtIter = postDomTree.iterator();
            int idx = 0;
            while (pdtIter.hasNext()) {
            	DFANode node = pdtIter.next();
            	pdtNodeList.add(node);
//            	System.out.println("[" + idx++ + "]" + ((DFANode)((DFANode)node.getData("revNode")).getData("cfgNode")).getData("ir"));
            }
            for (CFGEdge edge: nonPDEdgeSet) {
                // find lowest common ancestor of m and n (result: l)
            	DFANode commonAncestor = getLowestCommonAncestor(edge, postDomTree, pdtNodeList); 

            	// all nodes in N on the path from l to n in the postdominance tree except l
                // are control dependent on m
            	Set<DFANode> cdSet = getControlDependentSet(edge, commonAncestor);
            	DFANode controllingNode = edge.tail.getData("cfgNode");
            	// TODO: handled some corner cases this way but should be fixed later
            	// Some wrong controlling nodes are inserted, maybe due to the problem of control flow graph 
            	if ((controllingNode.getData("ir") instanceof Expression) == false &&
            			(controllingNode.getData("ir") instanceof SwitchStatement) == false) {
            		// filter if a statement is a controlling node (wrong)
            		continue;
            		//throw new RuntimeException("controllingNode is not expression: " + controllingNode.getData("ir"));
            	} else {
            		if (controllingNode.getData("ir") instanceof Expression) {
	            		// filter a stepping expression in the for loop is a controlling node (wrong) 
	            		Expression exp = controllingNode.getData("ir");
	            		Traversable parent = exp.getParent();
	            		if (parent instanceof ForLoop) {
	            			ForLoop fLoop = (ForLoop) parent;
	            			if (exp.equals(fLoop.getStep())) {
	            				continue;
	            			}
	            		}
            		} else if (controllingNode.getData("ir") instanceof SwitchStatement) {
            			
            		} else {
            			throw new RuntimeException("controllingNode is unexpected type: " + controllingNode.getData("ir"));
            		}
            	}
            	//
            	for (DFANode node: cdSet) {
            		DFANode revNode = node.getData("revNode");
            		if (revNode == null) {
            			continue;
            		}
            		DFANode controlledNode = revNode.getData("cfgNode");
            		controlledNode.putData("controlNode", controllingNode);
            		if (PrintTools.getVerbosity() > 1)
            			System.out.println("[Node]" + controlledNode.getData("ir") + " == control dependent on == [Node]" + controllingNode.getData("ir"));
            	}
            }
            // check inter-dependent node
            // remove the controlling node for coming first node toward coming later node
            Iterator<DFANode> cfgIter = cfg.iterator();
            while (cfgIter.hasNext()) {
            	DFANode n = cfgIter.next();
            	DFANode cdNode = n.getData("controlNode");
            	if (cdNode == null) {
            		continue;
            	}
            	DFANode cdCDNode = cdNode.getData("controlNode");
            	if (n.equals(cdCDNode)) {
            		n.removeData("controlNode");
//            		throw new RuntimeException("My contol node is dependent on me. my: " + n + ",\n cd: " + cdNode);
            	}
            }
            // check a cyclic dependent node and remove it
            cfgIter = cfg.iterator();
            while (cfgIter.hasNext()) {
            	DFANode n = cfgIter.next();
            	DFANode cdNode = n.getData("controlNode");
            	while (cdNode != null) {
            		if (cdNode.equals(n)) {
            			cdNode.removeData("controlNode");
            			break;
//            			throw new RuntimeException("Cyclic dependency was found!!");
            		} else {
            			cdNode = cdNode.getData("controlNode");
            		}
            	}
            }
        }
    }
    
    private Set<CFGEdge> extractNonPostdominatingEdges(
    		DFAGraph reverseCFG, 
    		DFAGraph postDomTree, 
    		Procedure proc) {
    	if (PrintTools.getVerbosity() > 1)
    		System.out.println("[extractNonPostdominatingEdges]Proc: " + proc.getSymbolName());
    	Set<CFGEdge> retSet = new HashSet<CFGEdge>();
    	// extract all the edges from CFG
    	Set<CFGEdge> allEdgeList = new HashSet<CFGEdge>();
    	Iterator<DFANode> iter = reverseCFG.iterator();
    	while(iter.hasNext()) {
    		DFANode currentNode = iter.next();
    		Set<DFANode> predSet = currentNode.getPreds();
    		for (DFANode pred: predSet) {
    			// In the reversed cfg, pred is succ in CFG
    			allEdgeList.add(new CFGEdge(currentNode, pred));
    		}
    	}
    	List<DFANode> entryNodes = postDomTree.getEntryNodes();
    	if (entryNodes.size() != 1) {
    		throw new RuntimeException("Size of Entry node on PD Tree is not 1. Size: " + entryNodes.size());
    	}
    	DFANode entryNode = entryNodes.get(0);
    	// extract edge which head does not postdominate tail
    	// (head is not parent of tail on PostDomTree)
    	for (CFGEdge edge: allEdgeList) {
    		// 
    		DFANode pdtTail = (DFANode)edge.tail.getData("pdtNode");
    		DFANode pdtHead = (DFANode)edge.head.getData("pdtNode");
    		Set<DFANode> predSet = pdtTail.getPreds();
    		boolean postDom = false;
    		while (predSet.size() != 0) {
    			if (predSet.size() > 1) {
    				throw new RuntimeException("The size of predecessor is expected to be less than 2");
    			}
    			DFANode pdtNode = predSet.iterator().next();
    			if (pdtNode.equals(pdtHead)) {
    				postDom = true;
    				break;
    			}
    			predSet = pdtNode.getPreds();
    		}
    		if (postDom == false) {
    			retSet.add(edge);
    			if (PrintTools.getVerbosity() > 1)
    				System.out.println("Found Edge: " + ((DFANode)edge.tail.getData("cfgNode")).getData("ir") + " --> " + ((DFANode)edge.head.getData("cfgNode")).getData("ir") + "(" + ((DFANode)edge.head.getData("cfgNode")).getData("tag") + ")");
    		}
    	}
    	return retSet;
    }
    
    private DFANode getLowestCommonAncestor(CFGEdge edgeOnReversedCFG, DFAGraph postDomTree, List<DFANode> pdtNodeList) {
    	DFANode lcaNode = null;
    	String pathTail = getCommaSeparatedPath((DFANode)edgeOnReversedCFG.tail.getData("pdtNode"), postDomTree, pdtNodeList);
    	String pathHead = getCommaSeparatedPath((DFANode)edgeOnReversedCFG.head.getData("pdtNode"), postDomTree, pdtNodeList);
    	int idxCommon = SlicingTools.longestSubstr(pathHead, pathTail);
    	if (PrintTools.getVerbosity() > 1)
    		System.out.println("CommonString: " + pathHead.substring(0, idxCommon));
    	String commonString = pathHead.substring(0, idxCommon);
    	if (commonString.endsWith(",") == false) {
    		commonString = commonString.substring(0,commonString.lastIndexOf(","));
    	}
    	StringTokenizer tzer = new StringTokenizer(commonString, ",");
    	String lastToken = null;
    	while (tzer.hasMoreTokens()) {
    		lastToken = tzer.nextToken();
    	}
    	if (PrintTools.getVerbosity() > 1)
    		System.out.println("Last Token: " + lastToken);
    	lcaNode = pdtNodeList.get(Integer.parseInt(lastToken));
    	return lcaNode;
    }
    
    private String getCommaSeparatedPath(DFANode node, DFAGraph graph, List<DFANode> pdtNodeList) {
    	LinkedList<DFANode> list = new LinkedList<DFANode>();
    	list.add(node);
    	Set<DFANode> preds = node.getPreds();
    	while (preds != null && preds.size() > 0) {
    		DFANode parent = preds.iterator().next();
    		list.addFirst(parent);
    		preds = parent.getPreds();
    	}
    	StringBuffer retStr = new StringBuffer();
    	for (DFANode dfa: list) {
    		retStr.append(pdtNodeList.indexOf(dfa));
    		retStr.append(",");
    	}
    	if (PrintTools.getVerbosity() > 1) 
    		System.out.println("String: " + retStr.toString() + ", for node: " + ((DFANode)((DFANode)node.getData("revNode")).getData("cfgNode")).getData("ir") + "(" + ((DFANode)((DFANode)node.getData("revNode")).getData("cfgNode")).getData("tag") + ")");
    	return retStr.toString();
    }
    
    private Set<DFANode> getControlDependentSet(CFGEdge edge, DFANode commonAncestor) {
    	Set<DFANode> retSet = new HashSet<DFANode>();
    	DFANode pdtHead = edge.head.getData("pdtNode");
    	Set<DFANode> preds = pdtHead.getPreds();
    	retSet.add(pdtHead);
    	while (preds != null && preds.size() > 0) {
    		DFANode parent = preds.iterator().next();
    		if (parent.equals(commonAncestor)) {
    			break;
    		}
    		retSet.add(parent);
    		preds = parent.getPreds();
    	}
    	return retSet;
    }
}

class CFGEdge {
	DFANode tail;
	DFANode head;
	CFGEdge(DFANode tail, DFANode head) {
		this.tail = tail;
		this.head = head;
	}
}
