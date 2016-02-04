package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;

/**
 * Represents {@code __builtin_nvl_get_root} expression in NVL-C programs.
 * 
 * Based on {@link VaArgExpression}, which has a similar syntax.
 *  
 * @author Joel E. Denny <dennyje@ornl.gov> -
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public class NVLGetRootExpression extends Expression {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = NVLGetRootExpression.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }
    
    @SuppressWarnings("rawtypes")
    private List specs;

    /**
    * Constructs a {@code __builtin_nvl_get_root} expression with the
    * specified expression and specs.
    *
    * @param heap the heap operand expression.
    * @param pspecs the list of specifiers for the type operand.
    * @throws NotAnOrphanException if <b>expr</b> has a parent.
    */
    @SuppressWarnings({ "unchecked", "rawtypes" })
    public NVLGetRootExpression(Expression heap, List pspecs) {
        object_print_method = class_print_method;
        addChild(heap);
        specs = new ArrayList(pspecs);
    }

    /**
    * Prints a {@code __builtin_nvl_get_root} expression to a stream.
    *
    * @param e The expression to print.
    * @param o The writer on which to print the expression.
    */
    public static void defaultPrint(NVLGetRootExpression e, PrintWriter o) {
        o.print("__builtin_nvl_get_root(");
        e.getHeapExpression().print(o);
        o.print(",");
        PrintTools.printListWithSpace(e.specs, o);
        o.print(")");
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(32);
        sb.append("__builtin_nvl_get_root(");
        sb.append(getHeapExpression());
        sb.append(",");
        sb.append(PrintTools.listToString(specs, " "));
        sb.append(")");
        return sb.toString();
    }

    @Override
    protected int hashCode(int h) {
        h = hashCode("__builtin_nvl_get_root(", h);
        h = getHeapExpression().hashCode(h);
        h = 31 * h + ',';
        h = hashCode(specs, " ", h);
        h = 31 * h + ')';
        return h;
    }

    /**
    * Returns the heap operand expression.
    *
    * @return the expression.
    */
    public Expression getHeapExpression() {
        return (Expression)children.get(0);
    }

    /**
    * Returns the type argument, from which the return type is computed.
    *
    * @return the list of specifiers.
    */
    @SuppressWarnings("rawtypes")
    public List getSpecifiers() {
        return specs;
    }

    /**
    * Overrides the class print method, so that all subsequently
    * created objects will use the supplied method.
    *
    * @param m The new print method.
    */
    static public void setClassPrintMethod(Method m) {
        class_print_method = m;
    }

    /**
    * Compares the {@code __builtin_nvl_get_root} expression with the
    * specified object for equality.
    */
    @Override
    public boolean equals(Object o) {
        return (super.equals(o) &&
                specs.equals(((NVLGetRootExpression)o).specs));
    }

    @Override
    public void accept(TraversableVisitor v) { v.visit(this); }
}
