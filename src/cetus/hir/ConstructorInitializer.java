package cetus.hir;

import java.io.PrintWriter;
import java.util.List;

/** This class is used only for generating C++ output code.*/
public class ConstructorInitializer extends Initializer {

    public ConstructorInitializer(List values) {
        for (Object o : values) {
            children.add((Traversable)o);
            ((Traversable)o).setParent(this);
        }
    }
    
    public void print(PrintWriter o) {
        o.print("(");
        PrintTools.printListWithComma(children, o);
        o.print(")");
    }

    @Override
    public void accept(TraversableVisitor v) { v.visit(this); }
}
