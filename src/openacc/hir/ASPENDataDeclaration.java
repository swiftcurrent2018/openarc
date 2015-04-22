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
public class ASPENDataDeclaration extends ASPENDeclaration {

	/**
	 * 
	 */
	public ASPENDataDeclaration() {
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param size
	 */
	public ASPENDataDeclaration(int size) {
		super(size);
		// TODO Auto-generated constructor stub
	}
	
	public ASPENDataDeclaration(ASPENData data) {
		addChild(data);
	}
	
	public IDExpression getDeclaredID() {
		return ((ASPENData)children.get(0)).getID();
	}
	
	public ASPENData getASPENData() {
		return (ASPENData)children.get(0);
	}
	
	public void setASPENData(ASPENData data) {
		setChild(0, data);
	}

	/* (non-Javadoc)
	 * @see cetus.hir.Printable#print(java.io.PrintWriter)
	 */
	@Override
	public void print(PrintWriter o) {
		o.print("data ");
		((ASPENData)children.get(0)).printASPENModel(o);
	}

	/* (non-Javadoc)
	 * @see openacc.hir.ASPENPrintable#printASPENModel(java.io.PrintWriter)
	 */
	@Override
	public void printASPENModel(PrintWriter o) {
		o.print("data ");
		((ASPENData)children.get(0)).printASPENModel(o);

	}

	@Override
	public void accept(TraversableVisitor v) {
		((OpenACCTraversableVisitor)v).visit(this);
	}
}
