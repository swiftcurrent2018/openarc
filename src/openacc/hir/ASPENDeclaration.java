/**
 * 
 */
package openacc.hir;

import cetus.hir.Expression;
import cetus.hir.IDExpression;
import cetus.hir.NotAnOrphanException;
import cetus.hir.Tools;
import cetus.hir.Traversable;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group
 *         Oak Ridge National Laboratory
 *
 */
public abstract class ASPENDeclaration implements Cloneable, Traversable, ASPENPrintable, ASPENSymbol {

    /** The parent traversable object */
    protected Traversable parent;

    /** The list of children of the statement */
    protected List<Traversable> children;

    /** Empty child list for declarations with no child */
    protected static final List empty_list =
            Collections.unmodifiableList(new ArrayList<Object>(0));

	/**
	 * 
	 */
	protected ASPENDeclaration() {
		parent = null;
		children = new ArrayList<Traversable>(1);
	}

	protected ASPENDeclaration(int size) {
		parent = null;
        if (size < 0) {
            children = empty_list;
        } else {
            children = new ArrayList<Traversable>(size);
        }
	}
	
    @Override
	public ASPENDeclaration clone() {
        ASPENDeclaration o = null;
        try {
            o = (ASPENDeclaration)super.clone();
        } catch(CloneNotSupportedException e) {
            throw new InternalError();
        }
        o.parent = null;
        if (children != null) {
            o.children = new ArrayList<Traversable>(children.size());
            int children_size = children.size();
            for (int i = 0; i < children_size; i++) {
                Traversable child = children.get(i);
                Traversable o_child = null;
                if (child instanceof ASPENParam) {
                    o_child = ((ASPENParam)child).clone();
                } else if (child instanceof ASPENData) {
                    o_child = ((ASPENData)child).clone();
                } else if (child instanceof Expression) {
                    o_child = ((Expression)child).clone();
                } else if (child instanceof ASPENCompoundStatement) {
                    o_child = ((ASPENCompoundStatement)child).clone();
                } else if (child != null) {
                    throw new InternalError(
                            "ASPENDeclaration contains an unknown child type" + this);
                }
                if (o_child != null) {
                    o_child.setParent(o);
                }
                o.children.add(o_child);
            }
        } else {
            o.children = null;
        }
        return o;
	}
    
    @Override
    public boolean equals(Object o) {
		return (o == this);
    }
    
    @Override
    public int hashCode() {
        return System.identityHashCode(this);
    }

   /**
 	* Detaches this statement from it's parent, if it has one.
 	*/
    public void detach() {
        if (parent != null) {
            parent.removeChild(this);
            setParent(null);
        }
    }

    public List<Traversable> getChildren() {
        return children;
    }

    public Traversable getParent() {
        return parent;
    }

    /**
    * Removes a specific child of this statement;
    * some statements do not support this method.
    *
    * @param child The child to remove.
    */
    public void removeChild(Traversable child) {
        throw new UnsupportedOperationException(
            "This statement does not support removal of arbitrary children.");
    }
    
    /**
    * Inserts the specified traversable object at the end of the child list.
    *
    * @param t the traversable object to be inserted.
    * @throws IllegalArgumentException if <b>t</b> is null.
    * @throws NotAnOrphanException if <b>t</b> has a parent.
    */
    protected void addChild(Traversable t) {
        if (t == null) {
            throw new IllegalArgumentException("invalid child inserted.");
        }
        if (t.getParent() != null) {
            throw new NotAnOrphanException(this.getClass().getName());
        }
        children.add(t);
        t.setParent(this);
    }

    public void setChild(int index, Traversable t) {
        if (t == null || index < 0 || index >= children.size()) {
            throw new IllegalArgumentException();
        }
        if (t.getParent() != null) {
            throw new NotAnOrphanException();
        }
        // only certain types of objects can be children of declarations, so
        // check for them
        if (t instanceof ASPENParam || t instanceof ASPENData ||
            t instanceof Expression ||
            t instanceof ASPENCompoundStatement) {
            // detach the old child at position index
            children.get(index).setParent(null);
            // set the new child
            children.set(index, t);
            t.setParent(this);
        } else {
            throw new IllegalArgumentException();
        }
    }

    public void setParent(Traversable t) {
        parent = t;
    }
    
    /** Returns a string representation of the ASPEN Declaration */
    @Override
    public String toString() {
        StringWriter sw = new StringWriter(40);
        printASPENModel(new PrintWriter(sw));
        return sw.toString();
    }
    
    /** Returns a string representation of the ASPEN Declaration */
    public String toASPENString() {
        StringWriter sw = new StringWriter(40);
        printASPENModel(new PrintWriter(sw));
        return sw.toString();
    }

}
