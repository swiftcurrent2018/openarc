package cetus.hir;

import java.util.List;
import java.util.ArrayList;
import java.io.PrintWriter;
import java.lang.reflect.Method;

public class CompoundLiteral extends Expression {

    private static Method class_print_method;

    static {
        try {
            class_print_method =
                    CompoundLiteral.class.getMethod("defaultPrint",
                    new Class<?>[] {CompoundLiteral.class, PrintWriter.class});
        } catch (NoSuchMethodException e) {
            throw new RuntimeException("No print method found.");
        }
    }

    /** List of type specifiers */
    private List<Specifier> specifiers;

    /**
    * Constructs a new compound literal with the specified type and the list
    * of literals.
    */
    public CompoundLiteral(List specs, List lits) {
        if (specs==null || specs.isEmpty() || lits==null || lits.isEmpty()) {
            throw new IllegalArgumentException();
        }
        object_print_method = class_print_method;
        specifiers = new ArrayList<Specifier>(specs.size());
        children = new ArrayList<Traversable>(lits.size());
        for (int i = 0; i < specs.size(); i++) {
            Object o = specs.get(i);
            if (o instanceof Specifier) {
                specifiers.add((Specifier)o);
            } else if (o instanceof VariableDeclarator) {
                List trails = ((VariableDeclarator)o).getTrailingSpecifiers();
                for (int j = 0; j < trails.size(); j++) {
                    Object oo = trails.get(j);
                    if (oo instanceof Specifier) {
                        specifiers.add((Specifier)oo);
                    } else {
                        throw new IllegalArgumentException();
                    }
                }
            } else {
                throw new IllegalArgumentException();
            }
        }
        for (int i = 0; i < lits.size(); i++) {
            addChild((Traversable)lits.get(i));
        }
    }

    /**
    * Returns a clone of this compound literal.
    */
    @Override
    public CompoundLiteral clone() {
        CompoundLiteral o = (CompoundLiteral)super.clone();
        o.specifiers = new ArrayList<Specifier>(specifiers.size());
        for (int i = 0; i < specifiers.size(); i++) {
            o.specifiers.add(specifiers.get(i));
        }
        return o;
    }

    /**
    * Prints this compound literal to the specified stream.
    */
    public static void defaultPrint(CompoundLiteral cl, PrintWriter o) {
        o.print("(");
        PrintTools.printListWithSpace(cl.specifiers, o);
        o.print(")");
        o.print("{");
        PrintTools.printListWithComma(cl.children, o);
        o.print("}");
    }

    @Override
    public boolean equals(Object o) {
        if (!super.equals(o)) {
            return false;
        }
        CompoundLiteral other = (CompoundLiteral)o;
        return specifiers.equals(other.specifiers);
    }

    public static void setClassPrintMethod(Method m) {
        class_print_method = m;
    }

    public List<Specifier> getSpecifiers() {
        return new ArrayList<Specifier>(specifiers);
    }

    public List<Traversable> getLiterals() {
        return children;
    }

    @Override
    public void accept(TraversableVisitor v) { v.visit(this); }
}
