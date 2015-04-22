/**
 * 
 */
package openacc.hir;

import cetus.hir.IDExpression;

/**
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group
 *         Oak Ridge National Laboratory
 *
 */
public abstract class ASPENControlStatement extends ASPENStatement {
	protected IDExpression label = null;
	/**
	 * 
	 */
	public ASPENControlStatement() {
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param size
	 */
	public ASPENControlStatement(int size) {
		super(size);
		// TODO Auto-generated constructor stub
	}
	
	public abstract ASPENCompoundStatement getBody();
	
}
