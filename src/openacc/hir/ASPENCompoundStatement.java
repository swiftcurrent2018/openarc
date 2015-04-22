/**
 * 
 */
package openacc.hir;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

import cetus.hir.NotAChildException;
import cetus.hir.PrintTools;
import cetus.hir.Tools;
import cetus.hir.Traversable;
import cetus.hir.TraversableVisitor;

/**
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group
 *         Oak Ridge National Laboratory
 *
 */
public class ASPENCompoundStatement extends ASPENStatement {

	/**
	 * 
	 */
	public ASPENCompoundStatement() {
		super();
	}

	/**
	 * @param size
	 */
	public ASPENCompoundStatement(int size) {
		super(size);
	}
	
    public void removeChild(Traversable child) {
        int index = Tools.identityIndexOf(children, child);
        if (index == -1) {
            throw new NotAChildException();
        }
        child.setParent(null);
        children.remove(index);
    }

	/* (non-Javadoc)
	 * @see cetus.hir.Printable#print(java.io.PrintWriter)
	 */
	@Override
	public void print(PrintWriter o) {
        o.println("{");
        ASPENPrintTools.printlnList(children, o);
        o.print("}");

	}

	/* (non-Javadoc)
	 * @see openacc.hir.ASPENPrintable#printASPENModel(java.io.PrintWriter)
	 */
	@Override
	public void printASPENModel(PrintWriter o) {
        o.println("{");
        ASPENPrintTools.printlnList(children, o);
        o.print("}");
	}
	
    /**
    * Adds a statement to the end of this compound statement. 
    *
    * @param stmt The statement to add.
    * @throws IllegalArgumentException If <b>stmt</b> is null.
    * @throws NotAnOrphanException if <b>stmt</b> has a parent.
    * @throws UnsupportedOperationException if <b>stmt</b> is a declaration
    * statement.
    */
    public void addASPENStatement(ASPENStatement stmt) {
        addChild(stmt);
    }

    /**
    * Add a new statement before the reference statement.
    *
    * @param new_stmt the statement to be added.
    * @param ref_stmt the reference statement.
    * @throws IllegalArgumentException If <b>ref_stmt</b> is not found or
    * <b>new_stmt</b> is null.
    * @throws NotAnOrphanException if <b>new_stmt</b> has a parent.
    * @throws UnsupportedOperationException If <b>new_stmt</b> is a declaration
    * statement.
    */
    public void addASPENStatementBefore(ASPENStatement ref_stmt, ASPENStatement new_stmt) {
        int index = Tools.identityIndexOf(children, ref_stmt);
        if (index == -1) {
            throw new IllegalArgumentException();
        }
        addChild(index, new_stmt);
    }

    /**
    * Add a new statement after the reference statement.
    *
    * @param new_stmt the statement to be added.
    * @param ref_stmt the reference statement.
    * @throws IllegalArgumentException If <b>ref_stmt</b> is not found or
    * <b>new_stmt</b> is null.
    * @throws NotAnOrphanException if <b>new_stmt</b> has a parent.
    * @throws UnsupportedOperationException If <b>stmt</b> is a declaration
    * statement.
    */
    public void addASPENStatementAfter(ASPENStatement ref_stmt, ASPENStatement new_stmt) {
        int index = Tools.identityIndexOf(children, ref_stmt);
        if (index == -1) {
            throw new IllegalArgumentException();
        }
        addChild(index + 1, new_stmt);
    }

    /**
    * Remove the given statement if it exists.
    *
    * @param stmt the statement to be removed.
    */
    public void removeASPENStatement(ASPENStatement stmt) {
        removeChild(stmt);
    }
    
    public int countASPENStatements() {
    	return children.size();
    }
    
    public boolean isEmpty() {
    	return children.isEmpty();
    }
    
    @SuppressWarnings("unchecked")
    public <T extends ASPENStatement> List<T> getASPENStatements(Class<T> type) {
        List<T> ret = new ArrayList<T>(1);
        for (int i = 0; i < children.size(); i++) {
            Traversable tr = children.get(i);
            if (type.isInstance(tr)) {
                ret.add((T)tr);
            }
        }
        return ret;
    }

    @Override
    public void accept(TraversableVisitor v) {
      ((OpenACCTraversableVisitor)v).visit(this);
    }
}
