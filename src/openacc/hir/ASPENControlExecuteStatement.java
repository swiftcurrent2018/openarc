/**
 * 
 */
package openacc.hir;

import java.io.PrintWriter;

import cetus.hir.IDExpression;
import cetus.hir.TraversableVisitor;

/**
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group
 *         Oak Ridge National Laboratory
 *
 */
public class ASPENControlExecuteStatement extends ASPENControlStatement {
	/**
	 * 
	 */
	public ASPENControlExecuteStatement() {
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param size
	 */
	public ASPENControlExecuteStatement(int size) {
		super(size);
		// TODO Auto-generated constructor stub
	}
	
	public ASPENControlExecuteStatement(IDExpression nlabel, ASPENCompoundStatement cStmt) {
		label = nlabel;
		addChild(cStmt);
	}
	
	public ASPENControlExecuteStatement(IDExpression nlabel, ASPENCompoundStatement cStmt, ASPENResource parallelArg) {
		label = nlabel;
		addChild(cStmt);
		addChild(parallelArg);
	}
	
	public IDExpression getExecuteLabel() {
		return label;
	}
	
	public ASPENCompoundStatement getExecuteBody() {
		return (ASPENCompoundStatement)children.get(0);
	}
	
	public ASPENCompoundStatement getBody() {
		return (ASPENCompoundStatement)children.get(0);
	}
	
	public void setExecuteBody(ASPENCompoundStatement cStmt) {
		setChild(0, cStmt);
	}
	
	public ASPENResource getExecuteParallelism() {
		if( children.size() == 2 ) {
			return (ASPENResource)children.get(1);
		} else {
			return null;
		}
	}
	
	public void setExecuteParallelism( ASPENResource parallelArg ) {
		if( children.size() == 1 ) {
			if( parallelArg != null ) {
				addChild(parallelArg);
			}
		} else {
			if( parallelArg != null ) {
				setChild(1, parallelArg);
			} else {
				children.remove(1);
			}
		}
	}

	/* (non-Javadoc)
	 * @see cetus.hir.Printable#print(java.io.PrintWriter)
	 */
	@Override
	public void print(PrintWriter o) {
		o.print("execute ");
		if( children.size() == 2 ) {
			((ASPENResource)children.get(1)).printASPENModel(o);
		}
		if( label != null ) {
			o.print(" \"" + label.toString() + "\" ");
		}
		((ASPENCompoundStatement)children.get(0)).printASPENModel(o);

	}

	/* (non-Javadoc)
	 * @see openacc.hir.ASPENPrintable#printASPENModel(java.io.PrintWriter)
	 */
	@Override
	public void printASPENModel(PrintWriter o) {
		o.print("execute ");
		if( children.size() == 2 ) {
			((ASPENResource)children.get(1)).printASPENModel(o);
		}
		if( label != null ) {
			o.print(" \"" + label.toString() + "\" ");
		}
		((ASPENCompoundStatement)children.get(0)).printASPENModel(o);
	}

	public void accept(cetus.hir.TraversableVisitor v) {
		((openacc.hir.OpenACCTraversableVisitor)v).visit(this);
	}
}
