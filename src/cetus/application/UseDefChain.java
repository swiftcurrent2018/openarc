package cetus.application;

import cetus.analysis.DFANode;
import cetus.hir.Expression;
import cetus.hir.Traversable;
import java.util.List;
import java.util.Set;

/**
 * A class that implements UseDefChain provides the Use-Def chain information
 * which is computed from the control flow graph.
 * @author Jae-Woo Lee, <jaewoolee@purdue.edu>
 *         School of ECE, Purdue University
 */
public interface UseDefChain {
	
	/**
	 * Returns interprocedural definition list.
	 * 
	 * @param use use expression for use-def chain
	 * @return the resulting definition list
	 */
    public List<Traversable> getDefList(Expression use);
    
    /**
     * Returns interprocedural definition list as list of DFANode in the control flow graph
     * 
     * @param use use expression for use-def chain
     * @return the resulting definition list
     */
    public Set<DFANode> getDefDFANodeSet(Expression use);
    
    /**
     * Returns intraprocedural defintion list
     * 
     * @param use use expression for use-def chain
     * @return the resulting definition list
     */
    public List<Traversable> getLocalDefList(Expression use);
}
