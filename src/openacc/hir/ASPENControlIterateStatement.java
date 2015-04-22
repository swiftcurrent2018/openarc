/**
 * 
 */
package openacc.hir;

import java.io.PrintWriter;

import cetus.hir.Expression;
import cetus.hir.IDExpression;
import cetus.hir.TraversableVisitor;

/**
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group
 *         Oak Ridge National Laboratory
 *
 */
public class ASPENControlIterateStatement extends ASPENControlStatement {

	/**
	 * 
	 */
	public ASPENControlIterateStatement() {
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param size
	 */
	public ASPENControlIterateStatement(int size) {
		super(size);
		// TODO Auto-generated constructor stub
	}
	
	public ASPENControlIterateStatement(Expression itrcnt, ASPENCompoundStatement cStmt) {
		addChild(itrcnt);
		addChild(cStmt);
	}
	
	public ASPENControlIterateStatement(Expression itrcnt, ASPENCompoundStatement cStmt, IDExpression nLabel) {
		addChild(itrcnt);
		addChild(cStmt);
		label = nLabel;
	}
	
	public Expression getItrCnt() {
		return (Expression)children.get(0);
	}
	
	public void setItrCnt(Expression itrcnt) {
		setChild(0, itrcnt);
	}
	
	public ASPENCompoundStatement getIterateBody() {
		return (ASPENCompoundStatement)children.get(1);
	}
	
	public ASPENCompoundStatement getBody() {
		return (ASPENCompoundStatement)children.get(1);
	}
	
	public void setIterateBody(ASPENCompoundStatement cStmt) {
		setChild(1, cStmt);
	}

	/* (non-Javadoc)
	 * @see cetus.hir.Printable#print(java.io.PrintWriter)
	 */
	@Override
	public void print(PrintWriter o) {
		o.print("iterate ");
		o.print("[");
		o.print((Expression)children.get(0));
		o.print("] ");
		if( label != null ) {
			o.print("\""+label.toString() + "\" ");
		}
		((ASPENCompoundStatement)children.get(1)).printASPENModel(o);

	}

	/* (non-Javadoc)
	 * @see openacc.hir.ASPENPrintable#printASPENModel(java.io.PrintWriter)
	 */
	@Override
	public void printASPENModel(PrintWriter o) {
		o.print("iterate ");
		o.print("[");
		o.print((Expression)children.get(0));
		o.print("] ");
		if( label != null ) {
			o.print("\""+label.toString() + "\" ");
		}
		((ASPENCompoundStatement)children.get(1)).printASPENModel(o);
	}

	@Override
	public void accept(TraversableVisitor v) {
		((OpenACCTraversableVisitor)v).visit(this);
	}
}
