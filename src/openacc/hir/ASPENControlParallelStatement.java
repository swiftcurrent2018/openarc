/**
 * 
 */
package openacc.hir;

import java.io.PrintWriter;

import cetus.hir.Expression;
import cetus.hir.TraversableVisitor;

/**
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group
 *         Oak Ridge National Laboratory
 *
 */
public class ASPENControlParallelStatement extends ASPENControlStatement {

	/**
	 * 
	 */
	public ASPENControlParallelStatement() {
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param size
	 */
	public ASPENControlParallelStatement(int size) {
		super(size);
		// TODO Auto-generated constructor stub
	}
	
	public ASPENControlParallelStatement(ASPENCompoundStatement cStmt) {
		addChild(cStmt);
	}
	
	public ASPENCompoundStatement getParallelBody() {
		return (ASPENCompoundStatement)children.get(0);
	}
	
	public ASPENCompoundStatement getBody() {
		return (ASPENCompoundStatement)children.get(0);
	}
	
	public void setParallelBody(ASPENCompoundStatement cStmt) {
		setChild(0, cStmt);
	}

	/* (non-Javadoc)
	 * @see cetus.hir.Printable#print(java.io.PrintWriter)
	 */
	@Override
	public void print(PrintWriter o) {
		o.print("par ");
		((ASPENCompoundStatement)children.get(0)).printASPENModel(o);

	}

	/* (non-Javadoc)
	 * @see openacc.hir.ASPENPrintable#printASPENModel(java.io.PrintWriter)
	 */
	@Override
	public void printASPENModel(PrintWriter o) {
		o.print("par ");
		((ASPENCompoundStatement)children.get(0)).printASPENModel(o);
	}

	@Override
	public void accept(TraversableVisitor v) {
		((OpenACCTraversableVisitor)v).visit(this);
	}
}
