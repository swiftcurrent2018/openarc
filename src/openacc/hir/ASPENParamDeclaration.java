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
public class ASPENParamDeclaration extends ASPENDeclaration {

	/**
	 * 
	 */
	public ASPENParamDeclaration() {
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param size
	 */
	public ASPENParamDeclaration(int size) {
		super(size);
		// TODO Auto-generated constructor stub
	}
	
	public ASPENParamDeclaration(ASPENParam param) {
		addChild(param);
	}
	
	public IDExpression getDeclaredID() {
		return ((ASPENParam)children.get(0)).getID();
	}
	
	public ASPENParam getASPENParam() {
		return (ASPENParam)children.get(0);
	}
	
	public void setASPENParam(ASPENParam param) {
		setChild(0, param);
	}

	/* (non-Javadoc)
	 * @see cetus.hir.Printable#print(java.io.PrintWriter)
	 */
	@Override
	public void print(PrintWriter o) {
		o.print("param ");
		((ASPENParam)children.get(0)).printASPENModel(o);
	}

	/* (non-Javadoc)
	 * @see openacc.hir.ASPENPrintable#printASPENModel(java.io.PrintWriter)
	 */
	@Override
	public void printASPENModel(PrintWriter o) {
		o.print("param ");
		((ASPENParam)children.get(0)).printASPENModel(o);

	}

	@Override
	public void accept(TraversableVisitor v) {
		((OpenACCTraversableVisitor)v).visit(this);
	}
}
