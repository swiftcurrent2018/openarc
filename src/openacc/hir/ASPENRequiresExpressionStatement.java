/**
 * 
 */
package openacc.hir;

import java.io.PrintWriter;

import cetus.hir.TraversableVisitor;

/**
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group
 *         Oak Ridge National Laboratory
 *
 */
public class ASPENRequiresExpressionStatement extends ASPENExpressionStatement {
	private String resource;

	/**
	 * 
	 */
	public ASPENRequiresExpressionStatement() {
		resource = null;
	}

	/**
	 * @param size
	 */
	public ASPENRequiresExpressionStatement(int size) {
		super(size);
		resource = null;
	}
	
	public ASPENRequiresExpressionStatement(String rType, ASPENResource tRSC) {
		super(1);
		resource = rType;
		addChild(tRSC);
	}
	
	public String getResourceType() {
		return resource;
	}
	
	public void setResourceType(String rType) {
		resource = rType;
	}
	
	public ASPENResource getASPENResource() {
		if( children.size() == 1 ) {
			return (ASPENResource)children.get(0);
		} else {
			return null;
		}
	}
	
	public void setASPENResource(ASPENResource tRSC) {
		setChild(0, tRSC);
	}
	
	public ASPENRequiresExpressionStatement clone() {
		ASPENRequiresExpressionStatement nRES = (ASPENRequiresExpressionStatement)super.clone();
		nRES.resource = this.resource;
		return nRES;
	}

	/* (non-Javadoc)
	 * @see cetus.hir.Printable#print(java.io.PrintWriter)
	 */
	@Override
	public void print(PrintWriter o) {
        //o.print("requires ");
        o.print(resource);
        o.print(" ");
        ((ASPENResource)children.get(0)).printASPENModel(o);
	}

	/* (non-Javadoc)
	 * @see openacc.hir.ASPENPrintable#printASPENModel(java.io.PrintWriter)
	 */
	@Override
	public void printASPENModel(PrintWriter o) {
        //o.print("requires ");
        o.print(resource);
        o.print(" ");
        ((ASPENResource)children.get(0)).printASPENModel(o);
	}

	@Override
	public void accept(TraversableVisitor v) {
		((OpenACCTraversableVisitor)v).visit(this);
	}
}
