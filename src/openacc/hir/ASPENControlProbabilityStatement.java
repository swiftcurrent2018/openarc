/**
 * 
 */
package openacc.hir;

import java.io.PrintWriter;

import cetus.hir.BinaryOperator;
import cetus.hir.Expression;
import cetus.hir.BinaryExpression;
import cetus.hir.IntegerLiteral;
import cetus.hir.TraversableVisitor;

/**
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group
 *         Oak Ridge National Laboratory
 *
 */
public class ASPENControlProbabilityStatement extends ASPENControlStatement {

	/**
	 * 
	 */
	public ASPENControlProbabilityStatement() {
		// TODO Auto-generated constructor stub
	}

	/**
	 * @param size
	 */
	public ASPENControlProbabilityStatement(int size) {
		super(size);
		// TODO Auto-generated constructor stub
	}
	
	public ASPENControlProbabilityStatement(Expression prob, ASPENCompoundStatement ifBody) {
		super(2);
		addChild(prob);
		addChild(ifBody);
	}
	
	public ASPENControlProbabilityStatement(Expression prob, ASPENCompoundStatement ifBody, ASPENCompoundStatement elseBody) {
		super(3);
		addChild(prob);
		addChild(ifBody);
		addChild(elseBody);
	}
	
	/**
	 * Normalize this conditional statement and return true if the normalized statement is non-empty.
	 * @return
	 */
	public boolean normalize() {
		boolean ret = true;
		Expression prob = ((Expression)children.get(0)).clone();
		ASPENCompoundStatement ifBody = (ASPENCompoundStatement)children.get(1);
		ASPENCompoundStatement elseBody = null;
		if( children.size() == 3 ) {
			elseBody = (ASPENCompoundStatement)children.get(2);
			if( ifBody.isEmpty() ) {
				if( elseBody.isEmpty() ) {
					return false;
				} else {
					prob = new BinaryExpression(new IntegerLiteral(1), BinaryOperator.SUBTRACT, prob);
					children.clear();
					addChild(prob);
					elseBody.setParent(null);
					addChild(elseBody);
				}
				
			} else if( elseBody.isEmpty() ) {
				children.remove(2);
			}
		} else if( ifBody.isEmpty() ) {
			ret = false;
		}
		return ret;
	}
	
	public Expression getProbability() {
		return (Expression)children.get(0);
	}
	
	public void setProbability(Expression ifcond) {
		setChild(0, ifcond);
	}
	
	//This class has two body statements, and thus 
	//this method should not be called.
	public ASPENCompoundStatement getBody() {
		return null;
	}
	
	public ASPENCompoundStatement getIfBody() {
		return (ASPENCompoundStatement)children.get(1);
	}
	
	public void setIfBody(ASPENCompoundStatement ifBody) {
		setChild(1, ifBody);
	}
	
	public ASPENCompoundStatement getElseBody() {
		if( children.size() == 3 ) {
			return (ASPENCompoundStatement)children.get(2);
		} else {
			return null;
		}
	}
	
	public void setElseBody(ASPENCompoundStatement elseBody) {
		setChild(2, elseBody);
	}

	/* (non-Javadoc)
	 * @see cetus.hir.Printable#print(java.io.PrintWriter)
	 */
	@Override
	public void print(PrintWriter o) {
		o.print("probability (");
		o.print((Expression)children.get(0));
		o.print(") ");
		((ASPENCompoundStatement)children.get(1)).printASPENModel(o);
		if( children.size() == 3 ) {
			o.print(" else ");
			((ASPENCompoundStatement)children.get(2)).printASPENModel(o);
		}

	}

	/* (non-Javadoc)
	 * @see openacc.hir.ASPENPrintable#printASPENModel(java.io.PrintWriter)
	 */
	@Override
	public void printASPENModel(PrintWriter o) {
		o.println("probability {");
		o.print("[");
		o.print((Expression)children.get(0));
		o.print("] ");
		((ASPENCompoundStatement)children.get(1)).printASPENModel(o);
		if( children.size() == 3 ) {
			o.print(" else ");
			((ASPENCompoundStatement)children.get(2)).printASPENModel(o);
		}
		o.println("");
		o.print("}");
	}

	@Override
	public void accept(TraversableVisitor v) {
		((OpenACCTraversableVisitor)v).visit(this);
	}
}
