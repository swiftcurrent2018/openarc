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
public class ASPENControlMapStatement extends ASPENControlStatement {

	/**
	 * 
	 */
	public ASPENControlMapStatement() {
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param size
	 */
	public ASPENControlMapStatement(int size) {
		super(size);
		// TODO Auto-generated constructor stub
	}
	
	public ASPENControlMapStatement(Expression mapcnt, ASPENCompoundStatement cStmt) {
		addChild(mapcnt);
		addChild(cStmt);
	}
	
	public ASPENControlMapStatement(Expression mapcnt, ASPENCompoundStatement cStmt, IDExpression nLabel) {
		addChild(mapcnt);
		addChild(cStmt);
		label = nLabel;
	}
	
	public Expression getMapCnt() {
		return (Expression)children.get(0);
	}
	
	public void setMapCnt(Expression mapcnt) {
		setChild(0, mapcnt);
	}
	
	public ASPENCompoundStatement getMapBody() {
		return (ASPENCompoundStatement)children.get(1);
	}
	
	public ASPENCompoundStatement getBody() {
		return (ASPENCompoundStatement)children.get(1);
	}
	
	public void setMapBody(ASPENCompoundStatement cStmt) {
		setChild(1, cStmt);
	}

	/* (non-Javadoc)
	 * @see cetus.hir.Printable#print(java.io.PrintWriter)
	 */
	@Override
	public void print(PrintWriter o) {
		o.print("map ");
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
		o.print("map ");
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
