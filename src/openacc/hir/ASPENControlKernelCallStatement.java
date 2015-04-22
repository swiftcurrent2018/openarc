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
public class ASPENControlKernelCallStatement extends ASPENControlStatement {

	/**
	 * 
	 */
	public ASPENControlKernelCallStatement() {
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param size
	 */
	public ASPENControlKernelCallStatement(int size) {
		super(size);
		// TODO Auto-generated constructor stub
	}
	
	public ASPENControlKernelCallStatement(IDExpression kernelID, List argList) {
		addChild(kernelID);
		setArgs(argList);
	}
	
	public IDExpression getKernelID() {
		return (IDExpression)children.get(0);
	}
	
	public void setKernelID(IDExpression kernelID) {
		setChild(0, kernelID);
	}
	
	public Expression getArg(int i) {
		return (Expression)children.get(i+1);
	}
	
	public int getArgSize() {
		return children.size()-1;
	}
	
	public void setArg(int i, Expression arg) {
		setChild(i+1, arg);
	}
	
	public void addArg(Expression arg) {
		addChild(arg);
	}
	
	//This class does not have a body.
	public ASPENCompoundStatement getBody() {
		return null;
	}
	
	/* (non-Javadoc)
	 * @see cetus.hir.Printable#print(java.io.PrintWriter)
	 */
	@Override
	public void print(PrintWriter o) {
		o.print("call ");
		o.print((Expression)children.get(0));
		o.print("(");
		ASPENPrintTools.printListWithComma(children.subList(1, getArgSize()+1), o);
		o.print(")");

	}

	/* (non-Javadoc)
	 * @see openacc.hir.ASPENPrintable#printASPENModel(java.io.PrintWriter)
	 */
	@Override
	public void printASPENModel(PrintWriter o) {
		o.print("call ");
		o.print((Expression)children.get(0));
		o.print("(");
		ASPENPrintTools.printListWithComma(children.subList(1, getArgSize()+1), o);
		o.print(")");
	}
	
    /**
    * Set the list of arg expressions.
    *
    * @param args A list of Expression.
    * @throws NotAnOrphanException if an element of <b>args</b> has a parent
    * object.
    */
    public void setArgs(List args) {
		IDExpression ID = (IDExpression)children.get(0);
        children.clear();
        children.add(ID);
		if( args != null ) {
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
