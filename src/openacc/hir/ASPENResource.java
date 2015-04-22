/**
 * 
 */
package openacc.hir;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

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
public class ASPENResource extends ASPENExpression {

    private Expression ID;
    private String memsuffix;
    public static HashSet<String> memSuffixSet = 
    		new HashSet<String>(Arrays.asList("from", "to"));
	/**
	 * 
	 */
	public ASPENResource(Expression tValue) {
		addChild(tValue);
		memsuffix = null;
		ID = null;
	}
	
	public ASPENResource(Expression tValue, List<ASPENTrait> args) {
		addChild(tValue);
		if( args != null ) {
			setTraits(args);
		}
		memsuffix = null;
		ID = null;
	}

	public ASPENResource(Expression tValue, List<ASPENTrait> args, String msuffix, Expression tID) {
		addChild(tValue);
		if( args != null ) {
			setTraits(args);
		}
		memsuffix = msuffix;
		ID = tID;
		if( ID != null ) {
			addChild(ID);
		}
	}
	
	public boolean equals(Object o) {
		if( super.equals(o) ) {
			ASPENResource obj = (ASPENResource)o;
			if( memsuffix == null ) {
				if( obj.memsuffix != null ) {
					return false;
				}
			} else if( !memsuffix.equals(obj.memsuffix) ) {
				return false;
			}
			if( ID == null ) {
				if( obj.ID != null ) {
					return false;
				}
			} else if( !ID.equals(obj.ID) ) {
				return false;
			}
			return true;
		} else {
			return false;
		}
	}
	
	public ASPENResource clone() {
		ASPENResource nRSC = (ASPENResource)super.clone();
		if( memsuffix != null ) {
			nRSC.memsuffix = memsuffix;
		} else {
			nRSC.memsuffix = null;
		}
		if( ID != null ) {
			nRSC.ID = ID.clone();
		} else {
			nRSC.ID = null;
		}
		return nRSC;
	}
	
	public Expression getValue() {
		return (Expression)children.get(0);
	}

	public void setValue(Expression nValue) {
		setChild(0, nValue);
	}
	
	public ASPENTrait getTrait(int i) {
		return (ASPENTrait)children.get(i+1);
	}
	
	public Set<ASPENTrait> getTraitsOfType(ASPENTrait.TraitType tType) {
		Set<ASPENTrait> retSet = new HashSet<ASPENTrait>();
		int traitSize = getTraitSize();
		for( int i=0; i<traitSize; i++ ) {
			ASPENTrait tr = (ASPENTrait)children.get(i+1);
			if( tr.getTraitType() == tType ) {
				retSet.add(tr);
			}
		}
		return retSet;
	}
	
	public int getTraitSize() {
		if( ID == null ) {
			return children.size()-1;
		} else {
			return children.size()-2;
		}
	}
	
	public void setTrait(int i, ASPENTrait arg) {
		setChild(i+1, arg);
	}

	public void addTrait(ASPENTrait arg) {
		if( ID == null ) {
			addChild(arg);
		} else {
			int size = children.size();
			addChild(size-1, arg);
		}
	}
	
	public void setMemSuffix(String msuffix) {
		memsuffix = msuffix;
	}
	
	public Expression getID() {
		return ID;
	}

	public void setID(Expression tID) {
		if( ID == null ) {
			ID = tID;
			addChild(ID);
		} else {
			int size = children.size();
			ID = tID;
			if( tID != null ) {
				setChild(size-1, ID);
			} else {
				children.remove(size-1);
			}
		}
	}
	
	public void updateID() {
		if( ID != null ) {
			int size = children.size();
			ID = (Expression)children.get(size-1);
		}
	}

	/* (non-Javadoc)
	 * @see cetus.hir.Printable#print(java.io.PrintWriter)
	 */
	@Override
	public void print(PrintWriter o) {
		int traitSize = children.size()-1;
		o.print(children.get(0));
		if( (memsuffix != null) && (ID != null) ) {
			o.print(":");
			o.print(memsuffix);
			o.print("(");
			o.print(ID);
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
		o.print("[");
		o.print(children.get(0));
		o.print("]");
		if( (memsuffix != null) && (ID != null) ) {
			o.print(" ");
			o.print(memsuffix);
			o.print(" ");
			o.print(ID);
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
		Expression value = (Expression)children.get(0);
        children.clear();
        children.add(value);
		if( args != null ) {
        	for (Object o : args) {
            	addChild((Traversable)o);
        	}
		}
		if( ID != null ) {
			children.add(ID);
		}
    }

    @Override
    public void accept(TraversableVisitor v) {
      ((OpenACCTraversableVisitor)v).visit(this);
    }
}
