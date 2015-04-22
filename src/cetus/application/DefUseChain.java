package cetus.application;

import cetus.hir.Expression;
import cetus.hir.Traversable;
import java.util.List;

/**
 * A class that implements DefUseChain provides the Def-Use chain information
 * which is computed from the control flow graph.
 * @author Jae-Woo Lee, <jaewoolee@purdue.edu>
 *         School of ECE, Purdue University
 */
public interface DefUseChain {
	
	/**
	 * Returns interprocedural use list.
	 *  
	 * @param def definition expression for def-use chain
	 * @return the resulting interprocedural use list
	 */
    public List<Traversable> getUseList(Expression def);
    
    /**
     * Returns intraprocedural use list.
     *  
     * @param def definition expression for def-use chain
     * @return the resulting interprocedural use list
     */
    public List<Traversable> getLocalUseList(Expression def);
    
    /**
     * Returns if the given definition reaches the given use interprocedurally
     * 
     * @param def definition expression
     * @param use use expression
     * @return the resulting reachability
     */
    public boolean isReachable(Expression def, Expression use);
}
