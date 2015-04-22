/**
 * 
 */
package openacc.hir;

import java.io.PrintWriter;

import cetus.hir.TraversableVisitor;

/**
 * @deprecated
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group
 *         Oak Ridge National Laboratory
 *
 */
public class ASPENExposesExpressionStatement extends ASPENExpressionStatement {

	/**
	 * 
	 */
	public ASPENExposesExpressionStatement() {
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param size
	 */
	public ASPENExposesExpressionStatement(int size) {
		super(size);
		// TODO Auto-generated constructor stub
	}
	
	public ASPENExposesExpressionStatement(ASPENResource pRSC) {
		addChild(pRSC);
	}

	/* (non-Javadoc)
	 * @see cetus.hir.Printable#print(java.io.PrintWriter)
	 */
	@Override
	public void print(PrintWriter o) {
        o.print("exposes ");
        ((ASPENResource)children.get(0)).printASPENModel(o);
	}

	/* (non-Javadoc)
	 * @see openacc.hir.ASPENPrintable#printASPENModel(java.io.PrintWriter)
	 */
	@Override
	public void printASPENModel(PrintWriter o) {
        o.print("exposes ");
        ((ASPENResource)children.get(0)).printASPENModel(o);

	}

	@Override
	public void accept(TraversableVisitor v) {
		((OpenACCTraversableVisitor)v).visit(this);
	}
}
