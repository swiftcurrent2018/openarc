/**
 * 
 */
package openacc.hir;

import java.io.PrintWriter;
import java.util.List;

import cetus.hir.Expression;
import cetus.hir.IDExpression;
import cetus.hir.Traversable;
import cetus.hir.TraversableVisitor;

/**
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group
 *         Oak Ridge National Laboratory
 *
 */
public class ASPENKernel extends ASPENDeclaration {

	/**
	 * 
	 */
	public ASPENKernel() {
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param size
	 */
	public ASPENKernel(int size) {
		super(size);
		// TODO Auto-generated constructor stub
	}
	
	public ASPENKernel(IDExpression kernelID, List argList, ASPENCompoundStatement body) {
		addChild(kernelID);
		addChild(body);
		setKernelParams(argList);
	}
	
	public IDExpression getDeclaredID() {
		return (IDExpression)children.get(0);
	}
	
	public IDExpression getKernelID() {
		return (IDExpression)children.get(0);
	}
	
	public void setKernelID(IDExpression kernelID) {
		setChild(0, kernelID);
	}
	
	public ASPENCompoundStatement getKernelBody() {
		return (ASPENCompoundStatement)children.get(1);
	}
	
	public void setKernelBody(ASPENCompoundStatement Body) {
		setChild(1, Body);
	}
	
	public Expression getKernelParam(int i) {
		return (Expression)children.get(i+2);
	}
	
	public int getKernelParamSize() {
		return children.size()-2;
	}
	
	public void setKernelParam(int i, Expression arg) {
		setChild(i+2, arg);
	}
	
	public void addKernelParam(Expression arg) {
		addChild(arg);
	}
	
	/* (non-Javadoc)
	 * @see cetus.hir.Printable#print(java.io.PrintWriter)
	 */
	@Override
	public void print(PrintWriter o) {
		o.print("kernel ");
		o.print((Expression)children.get(0));
		if( getKernelParamSize() > 0 ) {
			o.print("(");
			ASPENPrintTools.printListWithComma(children.subList(2, getKernelParamSize()+2), o);
			o.print(") ");
		} else {
			o.print(" ");
		}
		((ASPENCompoundStatement)children.get(1)).printASPENModel(o);

	}

	/* (non-Javadoc)
	 * @see openacc.hir.ASPENPrintable#printASPENModel(java.io.PrintWriter)
	 */
	@Override
	public void printASPENModel(PrintWriter o) {
		o.print("kernel ");
		o.print((Expression)children.get(0));
		if( getKernelParamSize() > 0 ) {
			o.print("(");
			ASPENPrintTools.printListWithComma(children.subList(2, getKernelParamSize()+2), o);
			o.print(") ");
		} else {
			o.print(" ");
		}
		((ASPENCompoundStatement)children.get(1)).printASPENModel(o);
	}
	
    /**
    * Set the list of arg expressions.
    *
    * @param args A list of Expression.
    * @throws NotAnOrphanException if an element of <b>args</b> has a parent
    * object.
    */
	public void setKernelParams(List args) {
		IDExpression ID = (IDExpression)children.get(0);
		ASPENCompoundStatement body = (ASPENCompoundStatement)children.get(1);
		children.clear();
		children.add(ID);
		children.add(body);
		if( (args != null) && (!args.isEmpty()) ) {
			for (Object o : args) {
				addChild((Traversable)o);
			}
		}
	}

	@Override
	public void accept(TraversableVisitor v) {
		((OpenACCTraversableVisitor)v).visit(this);
	}
}
