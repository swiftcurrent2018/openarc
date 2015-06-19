/**
 * 
 */
package openacc.hir;

import java.io.PrintWriter;
import java.io.StringWriter;

import cetus.hir.IDExpression;
import cetus.hir.Expression;
import cetus.hir.IntegerLiteral;
import cetus.hir.NameID;
import cetus.hir.TraversableVisitor;

/**
 * @author lee222<lees2@ornl.gov>
 *         Future Technologies Group
 *         Oak Ridge National Laboratory
 */
public class ASPENParam extends ASPENExpression {
    public static IDExpression defaultParamID = new NameID("aspen_param_default");
    public static Expression defaultParamValue = new IntegerLiteral(1);

	/**
	 * 
	 */
	public ASPENParam(IDExpression tID) {
		addChild(tID);
	}
	
	public ASPENParam(IDExpression tID, Expression tInitVal) {
		addChild(tID);
		if( tInitVal != null ) {
			addChild(tInitVal);
		}
	}

	public ASPENParam clone() {
		ASPENParam nParam = (ASPENParam)super.clone();
		return nParam;
	}
	
	public IDExpression getID() {
		return (IDExpression)children.get(0);
	}
	
	public void setID(IDExpression tID) {
		setChild(0, tID);
	}
	
	public Expression getInitVal() {
		if( children.size() == 2 ) {
			return (Expression)children.get(1);
		} else {
			return null;
		}
	}

	public void setInitVal(Expression nInitVal) {
		if( children.size() == 2 ) {
			setChild(1, nInitVal);
		} else if( children.size() == 1 ) {
			addChild(nInitVal);
		}
	}
	
	/* (non-Javadoc)
	 * @see cetus.hir.Printable#print(java.io.PrintWriter)
	 */
	@Override
	public void print(PrintWriter o) {
		o.print(children.get(0));
		if( children.size() == 2 ) {
			o.print(":");
			o.print(children.get(1));
		}
	}

	/* (non-Javadoc)
	 * @see openacc.hir.ASPENPrintable#printASPENModel(java.io.PrintWriter)
	 */
	@Override
	public void printASPENModel(PrintWriter o) {
		o.print(children.get(0));
		if( children.size() == 2 ) {
			o.print(" = ");
			o.print(children.get(1));
		}
	}
	
	@Override
	public void accept(TraversableVisitor v) {
		((OpenACCTraversableVisitor)v).visit(this);
	}
}
