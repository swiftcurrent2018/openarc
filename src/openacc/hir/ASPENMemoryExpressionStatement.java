/**
 * 
 */
package openacc.hir;

import java.io.PrintWriter;
import java.util.Arrays;
import java.util.HashSet;

import cetus.hir.Expression;
import cetus.hir.TraversableVisitor;

/**
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group
 *         Oak Ridge National Laboratory
 *
 */
public class ASPENMemoryExpressionStatement extends ASPENExpressionStatement {
	private String maction;
    private static HashSet<String> memory_actions = 
    		new HashSet<String>(Arrays.asList("allocates", "resizes", "frees"));

	/**
	 * 
	 */
	public ASPENMemoryExpressionStatement() {
		maction = null;
	}

	/**
	 * @param size
	 */
	public ASPENMemoryExpressionStatement(int size) {
		super(size);
		maction = null;
	}
	
	public ASPENMemoryExpressionStatement(String mType, ASPENData Data) {
		super(1);
		checkActionType(mType);
		addChild(Data);
	}
	
	private void checkActionType(String mType) {
		if( memory_actions.contains(mType) ) {
			maction = mType;
		} else {
			System.out.println("Unknown ASPEN Memory Clause: " + mType);
			System.out.println("Exit the ASPEN Model Generator.");
			System.exit(0);
		}
	}
	
	public String getActionType() {
		return maction;
	}
	
	public void setActionType(String mType) {
		checkActionType(mType);
	}
	
	public Expression getData() {
		return (Expression)((ASPENData)children.get(0)).getID();
	}
	
	public ASPENData getASPENData() {
		if( children.size() == 1 ) {
			return (ASPENData)children.get(0);
		} else {
			return null;
		}
	}
	
	public void setASPENData(ASPENData tRSC) {
		setChild(0, tRSC);
	}
	
	public ASPENMemoryExpressionStatement clone() {
		ASPENMemoryExpressionStatement nRES = (ASPENMemoryExpressionStatement)super.clone();
		nRES.maction = this.maction;
		return nRES;
	}

	/* (non-Javadoc)
	 * @see cetus.hir.Printable#print(java.io.PrintWriter)
	 */
	@Override
	public void print(PrintWriter o) {
        o.print(maction);
        o.print(" ");
        if( children.size() == 1 ) {
        	o.print(" ");
        	((ASPENData)children.get(0)).printASPENModel(o);
        }
	}

	/* (non-Javadoc)
	 * @see openacc.hir.ASPENPrintable#printASPENModel(java.io.PrintWriter)
	 */
	@Override
	public void printASPENModel(PrintWriter o) {
        o.print(maction);
        o.print(" ");
        if( children.size() == 1 ) {
        	o.print(" ");
        	((ASPENData)children.get(0)).printASPENModel(o);
        }
	}

	@Override
	public void accept(TraversableVisitor v) {
		((OpenACCTraversableVisitor)v).visit(this);
	}
}
