/**
 * 
 */
package openacc.hir;

import cetus.hir.Expression;
import cetus.hir.NotAnOrphanException;
import cetus.hir.Printable;
import cetus.hir.Tools;
import cetus.hir.Traversable;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.ArrayList;
import java.util.List;
import java.util.Collections;

/**
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group
 *         Oak Ridge National Laboratory
 *
 */
public abstract class ASPENExpression implements Cloneable, ASPENPrintable, Traversable {

    /** The parent object of the ASPENExpression */
    protected Traversable parent;

    /** All children must be ASPENExpressions. */
    protected List<Traversable> children;

    /** Empty child list for expressions having no children */
    protected static final List empty_list =
            Collections.unmodifiableList(new ArrayList<Object>(0));

    /** Constructor for derived classes. */
    protected ASPENExpression() {
        parent = null;
        children = new ArrayList<Traversable>(1);
    }

    /**
    * Constructor for derived classes.
    *
    * @param size The initial size for the child list.
    */
    @SuppressWarnings("unchecked")
    protected ASPENExpression(int size) {
        parent = null;
        if (size < 0) {
            children = empty_list;
        } else {
            children = new ArrayList<Traversable>(size);
        }
    }

    /**
    * Creates and returns a deep copy of this expression.
    *
    * @return a deep copy of this expression.
    */
    @Override
    public ASPENExpression clone() {
        ASPENExpression o = null;
        try {
            o = (ASPENExpression)super.clone();
        } catch(CloneNotSupportedException e) {
            throw new InternalError();
        }
        o.parent = null;
        if (children != null) {
            o.children = new ArrayList<Traversable>(children.size());
            for (int i = 0; i < children.size(); i++) {
                Traversable new_child = children.get(i);
                if (new_child instanceof ASPENExpression) {
                    new_child = ((ASPENExpression)new_child).clone();
                } else if (new_child instanceof Expression) {
                    new_child = ((Expression)new_child).clone();
                }
                new_child.setParent(o);
                o.children.add(new_child);
            }
        } else {
            o.children = null;
        }
        return o;
    }

    /**
    * Checks if the given object is has the same type with this expression and
    * its children is same with this expression's. The sub classes of expression
    * should call this method first and proceed with more checking if they
    * have additional fields to be checked.
    * @param o the object to be compared with.
    * @return true if {@code o!=null}, {@code this.getClass()==o.getClass()},
    *       and {@code this.children.equals(o.children) ||
    *       this.children==o.children==null}
    */
    @Override
    public boolean equals(Object o) {
        if (o == null || this.getClass() != o.getClass()) {
            return false;
        }
        if (children == null) {
            return (((ASPENExpression)o).children == null);
        } else {
            return children.equals(((ASPENExpression)o).children);
        }
    }

    /**
    * Returns the hash code of the expression. It returns the hash code of the
    * string representation since expressions are compared lexically.
    *
    * @return the integer hash code of the expression.
    */
    @Override
    public int hashCode() {
        return hashCode(0);
    }

    /**
    * Returns cumulative hash code with the given initial value, {@code h}.
    * This is intended for optimizing memory usage when computing hash code of
    * an expression object. Previous approach constructs a string representation
    * of the expression to get the hash code while this approach uses the same
    * algorithm as {@link String#hashCode()} without creating a string. All sub
    * classes of {@code Expression} have there specific implementations of this
    * method and {@link #hashCode()} simply returns {@code hashCode(0)}.
    */
    // NOTE for developers: Make this method consistent with toString() for all
    // sub classes of expression.
    protected int hashCode(int h) {
        return hashCode(this, h);
    }


    /* Traversable interface */
    public List<Traversable> getChildren() {
        return children;
    }

    /* Traversable interface */
    public Traversable getParent() {
        return parent;
    }

    /**
    * Get the parent Statement containing this Expression.
    *
    * @return the enclosing Statement or null if this Expression
    *   is not inside a Statement.
    */
    public ASPENStatement getASPENStatement() {
        Traversable t = this;
        do {
            t = t.getParent();
        } while (t != null && !(t instanceof ASPENStatement));
        return (ASPENStatement)t;
    }


    /**
    * This operation is not allowed.
    * @throws UnsupportedOperationException always
    */
    public void removeChild(Traversable child) {
        throw new UnsupportedOperationException(
                "ASPENExpressions do not support removal of arbitrary children.");
    }

    /**
    * @throws NotAnOrphanException if <b>t</b> has a parent object.
    * @throws IllegalArgumentException if <b>index</b> is out-of-range or
    * <b>t</b> is not an expression.
    */
    public void setChild(int index, Traversable t) {
        if (t.getParent() != null) {
            throw new NotAnOrphanException();
        }
        if ((!(t instanceof Expression) && !(t instanceof ASPENExpression)) 
			|| index >= children.size()) {
            throw new IllegalArgumentException();
        }
        // Detach the old child
        if (children.get(index) != null) {
            children.get(index).setParent(null);
        }
        children.set(index, t);
        t.setParent(this);
    }

    /* Traversable interface */
    public void setParent(Traversable t) {
        // expressions can appear in many places so it's probably not
        // worth it to try and provide instanceof checks against t here
        parent = t;
    }

    /**
    * Swaps two expression on the IR tree.  If neither this expression nor
    * <var>expr</var> has a parent, then this function has no effect. Otherwise,
    * each expression ends up with the other's parent and exchange positions in
    * the parents' lists of children.
    *
    * @param expr The expression with which to swap this expression.
    * @throws IllegalArgumentException if <var>expr</var> is null.
    * @throws IllegalStateException if the types of the expressions
    *   are such that they would create inconsistent IR when swapped.
    */
    public void swapWith(ASPENExpression expr) {
        if (expr == null) {
            throw new IllegalArgumentException();
        }
        if (this == expr) {
            // swap with self does nothing
            return;
        }
        // The rest of this must be done in a very particular order.
        // Be EXTREMELY careful changing it.
        Traversable this_parent = this.parent;
        Traversable expr_parent = expr.parent;
        int this_index = -1, expr_index = -1;
        if (this_parent != null) {
            this_index = Tools.identityIndexOf(this_parent.getChildren(), this);
            if (this_index == -1) {
                throw new IllegalStateException();
            }
        }
        if (expr_parent != null) {
            expr_index = Tools.identityIndexOf(expr_parent.getChildren(), expr);
            if (expr_index == -1) {
                throw new IllegalStateException();
            }
        }
        // detach both so setChild won't complain
        expr.parent = null;
        this.parent = null;
        if (this_parent != null) {
            this_parent.getChildren().set(this_index, expr);
            expr.setParent(this_parent);
        }
        if (expr_parent != null) {
            expr_parent.getChildren().set(expr_index, this);
            this.setParent(expr_parent);
        }
    }

    /** Returns a string representation of the expression */
    @Override
    public String toString() {
        StringWriter sw = new StringWriter(40);
        print(new PrintWriter(sw));
        return sw.toString();
    }
    
    /** Returns a string representation of the ASPEN expression */
    @Override
    public String toASPENString() {
        StringWriter sw = new StringWriter(40);
        printASPENModel(new PrintWriter(sw));
        return sw.toString();
    }

    /**
    * Verifies three properties of this object:
    * (1) All children are not null, (2) the parent object has this
    * object as a child, (3) all children have this object as the parent.
    *
    * @throws IllegalStateException if any of the properties are not true.
    */
    public void verify() throws IllegalStateException {
        if (parent != null && !parent.getChildren().contains(this)) {
            throw new IllegalStateException(
                    "parent does not think this is a child");
        }
        if (children != null) {
            if (children.contains(null)) {
                throw new IllegalStateException("a child is null");
            }
            for (Traversable t : children) {
                if (t.getParent() != this) {
                    throw new IllegalStateException(
                            "a child does not think this is the parent");
                }
            }
        }
    }

    /**
    * Common operation used in constructors - adds the specified traversable
    * object at the end of the child list.
    *
    * @param t the new child object to be added.
    * @throws NotAnOrphanException
    */
    protected void addChild(Traversable t) {
        if (t.getParent() != null) {
            throw new NotAnOrphanException(this.getClass().getName());
        }
        children.add(t);
        t.setParent(this);
    }
    
    /**
    * Common operation used in constructors - adds the specified traversable
    * object at the specified index.
    *
    * @param index list index to insert the new child
    * @param t the new child object to be added.
    * @throws NotAnOrphanException
    */
    protected void addChild(int index, Traversable t) {
        if (t == null || index < 0 || index > children.size()) {
            throw new IllegalArgumentException("invalid child inserted.");
        }
        if (t.getParent() != null) {
            throw new NotAnOrphanException(this.getClass().getName());
        }
        children.add(index, t);
        t.setParent(this);
    }

    protected static int hashCode(Object o, int h) {
        String s = o.toString();
        for (int i = 0; i < s.length(); i++) {
            h = 31 * h + s.charAt(i);
        }
        return h;
    }

}
