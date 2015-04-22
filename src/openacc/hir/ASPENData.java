/**
 * 
 */
package openacc.hir;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import cetus.hir.IDExpression;
import cetus.hir.PrintTools;
import cetus.hir.Expression;
import cetus.hir.Traversable;
import cetus.hir.TraversableVisitor;
import openacc.hir.ASPENTrait;

/**
 * @author lee222<lees2@ornl.gov>
 *         Future Technologies Group
 *         Oak Ridge National Laboratory
 */
public class ASPENData extends ASPENExpression {

	private Expression capacity;
	/**
	 * 
	 */
	public ASPENData(IDExpression tID) {
		addChild(tID);
		capacity = null;
	}
	
	public ASPENData(IDExpression tID, List<ASPENTrait> args) {
		addChild(tID);
		capacity = null;
		if( args != null ) {
			setTraits(args);
		}
	}

	public ASPENData(IDExpression tID, Expression initCap,  List<ASPENTrait> args) {
		addChild(tID);
		if( args != null ) {
			setTraits(args);
		}
		capacity = initCap;
		if( capacity != null ) {
			addChild(capacity);
		}
	}
	
	public boolean equals(Object o) {
		if( super.equals(o) ) {
			ASPENData obj = (ASPENData)o;
			if( capacity == null ) {
				if( obj.capacity != null ) {
					return false;
				}
			} else if( !capacity.equals(obj.capacity) ) {
				return false;
			}
			return true;
		} else {
			return false;
		}
	}
	
	public ASPENData clone() {
		ASPENData nData = (ASPENData)super.clone();
		if( capacity != null ) {
			nData.setCapacity(capacity.clone());
		} else {
			nData.capacity = null;
		}
		return nData;
	}
	
	public IDExpression getID() {
		return (IDExpression)children.get(0);
	}
	
	public void setID(IDExpression tID) {
		setChild(0, tID);
	}
	
	public Expression getCapacity() {
		return capacity;
	}

	public void setCapacity(Expression nCap) {
		if( capacity == null ) {
			capacity = nCap;
			addChild(capacity);
		} else {
			int size = children.size();
			capacity = nCap;
			if( nCap != null ) {
				setChild(size-1, capacity);
			} else {
				children.remove(size-1);
			}
		}
	}
	
	public void updateCapacity() {
		if( capacity != null ) {
			int size = children.size();
			capacity = (Expression)children.get(size-1);
		}
	}
	
	public ASPENTrait getTrait(int i) {
		return (ASPENTrait)children.get(i+1);
	}
	
	public Set<ASPENTrait> getTraits() {
		Set<ASPENTrait> traitSet = new HashSet<ASPENTrait>();
		if( getTraitSize() > 0 ) {
			for(int i=0; i<getTraitSize(); i++) {
				traitSet.add((ASPENTrait)getTrait(i));
			}
		}
		return traitSet;
	}
	
	public int getTraitSize() {
		if( capacity == null ) {
			return children.size()-1;
		} else {
			return children.size()-2;
		}
	}
	
	public void setTrait(int i, ASPENTrait arg) {
		setChild(i+1, arg);
	}
	
	public void addTrait(ASPENTrait arg) {
		if( capacity == null ) {
			addChild(arg);
		} else {
			int size = children.size();
			addChild(size-1, arg);
		}
	}
	
	/* (non-Javadoc)
	 * @see cetus.hir.Printable#print(java.io.PrintWriter)
	 */
	@Override
	public void print(PrintWriter o) {
		int traitSize = children.size()-1;
		o.print(children.get(0));
		if( capacity != null ) {
			o.print(":capacity(");
			o.print(capacity);
			o.print(")");
			traitSize -= 1;
		}
		if( traitSize > 0 ) {
			o.print(":traits(");
			PrintTools.printListWithComma(children.subList(1, traitSize+1), o);
			o.print(")");
		}
	}

	/* (non-Javadoc)
	 * @see openacc.hir.ASPENPrintable#printASPENModel(java.io.PrintWriter)
	 */
	@Override
	public void printASPENModel(PrintWriter o) {
		int traitSize = children.size()-1;
		o.print(children.get(0));
		if( capacity != null ) {
			o.print(" [");
			o.print(capacity);
			o.print("] ");
			traitSize -= 1;
		}
		if( traitSize > 0 ) {
			o.print(" as ");
			ASPENPrintTools.printListWithComma(children.subList(1, traitSize+1), o);
		}
	}
	
    /**
    * Set the list of ASPENTraits
    *
    * @param args A list of Expression.
    * @throws NotAnOrphanException if an element of <b>args</b> has a parent
    * object.
    */
    public void setTraits(List args) {
		IDExpression ID = (IDExpression)children.get(0);
        children.clear();
        children.add(ID);
		if( args != null ) {
        	for (Object o : args) {
            	addChild((Traversable)o);
        	}
		}
		if( capacity != null ) {
			children.add(capacity);
		}
    }

    @Override
    public void accept(TraversableVisitor v) {
      ((OpenACCTraversableVisitor)v).visit(this);
    }
}
