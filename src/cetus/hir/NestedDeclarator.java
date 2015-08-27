package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.util.Iterator;
import java.util.ArrayList;
import java.util.List;

/**
* Represents a nested declarator that may contain another declarator within
* itself. e.g., function pointers and pointers to a chunck of arrays.
*/
public class NestedDeclarator extends Declarator implements Symbol {

    /** Default class print method. */
    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = NestedDeclarator.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }

    /** Flag for identifying a nested declarator for function pointers. */
    private boolean has_param = false;

    /** Common initialization process. */
    private void initialize(Declarator nested_decl, List params) {
        object_print_method = class_print_method;
        if (nested_decl.getParent() != null) {
            throw new NotAnOrphanException();
        }
        children.add(nested_decl);
        nested_decl.setParent(this);
        if (params != null) {
            has_param = true;
            for (int i = 0; i < params.size(); i++) {
                Traversable decl = (Traversable)params.get(i);
                if (decl.getParent() != null) {
                    throw new NotAnOrphanException();
                }
                children.add(decl);
                decl.setParent(this);
            }
        }
    }

    /**
    * Constructs a new nested declarator with the given child declarator.
    *
    * @param nested_decl the child declarator to be added.
    */
    public NestedDeclarator(Declarator nested_decl) {
        super(1);
        initialize(nested_decl, null);
        leading_specs = new ArrayList<Specifier>(1);
        trailing_specs = new ArrayList<Specifier>(1);
    }

    /**
    * Constructs a new nested declarator with the given child declarator and the
    * list of parameters.
    *
    * @param nested_decl the child declarator to be added.
    * @param params the list of parameters.
    */
    public NestedDeclarator(Declarator nested_decl, List params) {
        super(params == null ? 1 : 1 + params.size());
        initialize(nested_decl, params);
        leading_specs = new ArrayList<Specifier>(1);
        trailing_specs = new ArrayList<Specifier>(1);
    }

    /**
    * Constructs a new nested declarator with the given list of leading
    * specifiers, the child declarator, and the list of parameters.
    *
    * @param leading_specs the list of leading specifiers.
    * @param nested_decl the child declarator to be added.
    * @param params the list of parameters.
    */
    @SuppressWarnings("unchecked")
    public NestedDeclarator(List leading_specs, Declarator nested_decl,
                            List params) {
        super(params == null ? 1 : 1 + params.size());
        initialize(nested_decl, params);
        this.leading_specs = new ArrayList<Specifier>(leading_specs);
        this.trailing_specs = new ArrayList<Specifier>(1);
    }

    /**
    * Constructs a new nested declarator with the given leading specifier, the
    * child declarator, and the list of parameters.
    *
    * @param spec the leading specifier.
    * @param nested_decl the child declarator to be added.
    * @param params the list of parameters.
    */
    public NestedDeclarator(Specifier spec, Declarator nested_decl,
                            List params) {
        super(params == null ? 1 : 1 + params.size());
        initialize(nested_decl, params);
        leading_specs = new ArrayList<Specifier>(1);
        leading_specs.add(spec);
        trailing_specs = new ArrayList<Specifier>(1);
    }

    /**
    * Constructs a new nested declarator with the given list of leading
    * specifers, the child declarator, the list of parameters, and the list of
    * trailing specifiers.
    *
    * @param leading_specs the list of leading specifiers.
    * @param nested_decl the child declarator to be added.
    * @param params the list of parameters.
    * @param trailing_specs the list of trailing specifiers.
    */
    @SuppressWarnings("unchecked")
    public NestedDeclarator(List leading_specs, Declarator nested_decl,
                            List params, List trailing_specs) {
        super(params == null ? 1 : 1 + params.size());
        initialize(nested_decl, params);
        this.leading_specs = new ArrayList<Specifier>(leading_specs);
        this.trailing_specs = new ArrayList<Specifier>(trailing_specs);
    }

    /**
    * Inserts a new parameter declaration to the nested declarator.
    *
    * @param decl the new parameter declaration to be added.
    */
    @Override
    public void addParameter(Declaration decl) {
        has_param = true;
        if (decl.getParent() != null) {
            throw new NotAnOrphanException();
        }
        if (getInitializer() == null) { // initializer is positioned at the end.
            children.add(decl);
        } else {
            children.add(children.size() - 1, decl);
        }
        decl.setParent(this);
    }

    /**
    * Inserts a new parameter declaration to the nested declarator before the
    * reference parameter declaration.
    *
    * @param ref the reference parameter declaration.
    * @param decl the new parameter declaration to be added.
    */
    @Override
    public void addParameterBefore(Declaration ref, Declaration decl) {
        int index = Tools.identityIndexOf(children, ref);
        if (index == -1) {
            throw new IllegalArgumentException();
        }
        if (decl.getParent() != null) {
            throw new NotAnOrphanException();
        }
        children.add(index, decl);
        decl.setParent(this);
    }

    /**
    * Inserts a new parameter declaration to the nested declarator after the
    * reference parameter declaration.
    *
    * @param ref the reference parameter declaration.
    * @param decl the new parameter declaration to be added.
    */
    @Override
    public void addParameterAfter(Declaration ref, Declaration decl) {
        int index = Tools.identityIndexOf(children, ref);
        if (index == -1) {
            throw new IllegalArgumentException();
        }
        if (decl.getParent() != null) {
            throw new NotAnOrphanException();
        }
        children.add(index + 1, decl);
        decl.setParent(this);
    }

    /**
    * 
    * [Added by Seyong Lee]
    * Replace existing parameter with a new one.
    *
    * @param ref the reference parameter declaration to be replaced.
    * @param decl the new parameter declaration to be added.
    */
    public void replaceParameter(Declaration ref, Declaration decl) {
        int index = Tools.identityIndexOf(children, ref);
        if (index == -1) {
            throw new IllegalArgumentException();
        }
        if (decl.getParent() != null) {
            throw new NotAnOrphanException();
        }
        children.set(index, decl);
        decl.setParent(this);
        ref.setParent(null);
    }

    /**
    * Returns a clone of the nested declarator.
    */
    @Override
    public NestedDeclarator clone() {
        NestedDeclarator d = (NestedDeclarator)super.clone();
        Declarator id = getDeclarator().clone();
        d.children.add(id);
        id.setParent(d);
        if (children.size() > 1) {
            for (int i = 1; i < children.size(); i++) {
                Traversable child = children.get(i);
                if (child instanceof Declaration) {
                    Declaration decl = ((Declaration)child).clone();
                    d.children.add(decl);
                    decl.setParent(d);
                // was getting ClassCastException on statements like:
                // int (*abc)[2] = temp;
                // (where: int temp[][2] = {{2,3},{4,5}};)
                } else if (child instanceof Initializer) {
                    Initializer init = ((Initializer)child).clone();
                    d.children.add(init);
                    init.setParent(d);
                }
            }
        }
        d.has_param = has_param;
        return d;
    }

    /**
    * Prints a nested declarator to a stream.
    *
    * @param d The declarator to print.
    * @param o The writer on which to print the declarator.
    */
    public static void defaultPrint(NestedDeclarator d, PrintWriter o) {
        PrintTools.printListWithSpace(d.leading_specs, o);
        o.print("(");
        d.getDeclarator().print(o);
        o.print(")");
        if (d.has_param) {
            o.print("(");
            int num_param = d.children.size() - 1;
            if (d.getInitializer() != null) {
                num_param--;
            }
            if (num_param > 0) {
                PrintTools.printListWithComma(
                        d.children.subList(1, num_param + 1), o);
            }
            o.print(")");
        }
        PrintTools.printListWithSpace(d.trailing_specs, o);
        if (d.getInitializer() != null) {
            d.getInitializer().print(o);
        }
    }

    /**
    * Returns the child declarator of the nested declarator.
    */
    public Declarator getDeclarator() {
        return (Declarator)children.get(0);
    }

    /**
    * Returns the list of parameters if the declarator represents a function.
    * @return the list of parameters if it does, {@code null} otherwise.
    * [Fixed by Joel E. Denny not to try to cast any Initializer child to a
    * Declaration when iterating through children.]
    */
    @Override
    public List<Declaration> getParameters() {
        if (!has_param) {
            return null;
        }
        List<Declaration> ret = new ArrayList<Declaration>(children.size() - 1);
        for (int i = 1; i < children.size(); i++) {
            final Traversable child = children.get(i);
            if (child instanceof Declaration)
                ret.add((Declaration)child);
        }
        return ret;
    }

    /**
    * Returns the name ID declared by the nested declarator.
    */
    public IDExpression getID() {
        return getDeclarator().getDirectDeclarator();
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
    * Returns the initializer of the nested declarator.
    *
    * @return the initializer if exists, null otherwise.
    */
    public Initializer getInitializer() {
        if (!children.isEmpty()) {
            if (children.get(children.size() - 1) instanceof Initializer) {
                return (Initializer)children.get(children.size() - 1);
            }
        }
        return null;
    }

    /**
    * Assigns a new initializer for the nested declarator. The existing
    * initializer is dicarded.
    */
    public void setInitializer(Initializer init) {
        if (getInitializer() != null) {
            getInitializer().setParent(null);
            if (init != null) {
                children.set(children.size() - 1, init);
                init.setParent(this);
            } else { //[DEBUG by Seyong Lee] added to remove existing initializer if init is null.
            	children.remove(children.size()-1);
            }
        } else {
            if (init != null) {
                children.add(init);
                init.setParent(this);
            }
        }
    }

    /**
    * Checks if the nested declarator is used to represent a procedure call.
    */
    public boolean isProcedure() {
        return has_param;
    }

    /* Symbol interface */
    @SuppressWarnings("unchecked")
    public List getTypeSpecifiers() {
        Traversable t = this;
        while (!(t instanceof Declaration)) {
            t = t.getParent();
        }
        List ret = new ArrayList();
        if (t instanceof VariableDeclaration) {
            ret.addAll(((VariableDeclaration)t).getSpecifiers());
        } else if (t instanceof Enumeration) {
            ret.add(((Enumeration)t).getSpecifier());
        } else {
            return null;
        }
        ret.addAll(leading_specs);
        return ret;
    }

    /* Symbol interface */
    public String getSymbolName() {
        return getID().toString();
    }

    /* Symbol interface */
    @SuppressWarnings("unchecked")
    public List getArraySpecifiers() {
        return trailing_specs;
    }

    /* Symbol interface */
    public void setName(String name) {
        SymbolTable symtab = IRTools.getAncestorOfType(this, SymbolTable.class);
        SymbolTools.setSymbolName(this, name, symtab);
    }

    /* Symbol interface */
    public Declaration getDeclaration() {
        return IRTools.getAncestorOfType(this, Declaration.class);
    }

    protected IDExpression getDirectDeclarator() {
        return getDeclarator().getDirectDeclarator();
    }

    protected void setDirectDeclarator(IDExpression direct_decl) {
        getDeclarator().setDirectDeclarator(direct_decl);
    }
    
    //[Added by Seyong Lee] used to treat extern symbols equivalently.
    /**
    * Warning: this is an experimental feature.
    * Checks if the given Object <b>obj</b> refers to the same Symbol as this
    * Symbol.
    * @param obj
    * @return true if <b>obj</b> refers to the same memory location as this
    * Symbol.
    */
    private boolean isSameSymbol(Object obj) {
        // obj should be of type Symbol
        if (!(obj instanceof Symbol)) {
            return false;
        }
        Symbol other = (Symbol)obj;
        // Check text value of both Identifiers
        if (!this.getSymbolName().equals(other.getSymbolName())) {
            return false;
        }
        // If symbols are equal, two Identifiers refer to the same storage
        if (this == other) {
            return true;
        } else {
            // If both symbols
            // 1. Have the same text value and
            // 2. Have external linkage
            return hasExternalLinkage(this) && hasExternalLinkage(other);
        }
    }

    //[Added by Seyong Lee] used to treat extern symbols equivalently.
    /**
    * Check if Symbol <b>sym</b> has external linkage.
    * @param sym
    * @return true if <b>sym</b> has exteranl linkage
    */
    private static boolean hasExternalLinkage(Symbol sym) {
        // A Symbol has external linkage if
        // 1. Symbol has "extern" specifier or
        // 2. parent SymbolTable is of Type TranslationUnit && and not "static"
        try {
            if (SymbolTools.containsSpecifier(sym, Specifier.EXTERN)) {
                return true;
            }
            if (SymbolTools.containsSpecifier(sym, Specifier.STATIC)) {
                return false;
            }
            // getTypeSpecifiers() is not appropriate for this method since this
            // method is called extensibly and getTypeSpecifiers() consumes
            // extra memory for the returned list.
/*
            for(Specifier spec1 : (List<Specifier>)sym.getTypeSpecifiers()) {
                if(spec1 == Specifier.EXTERN)
                    return true;
                if(spec1 == Specifier.STATIC)
                    return false;
            }
*/
        } catch(Exception e) {
            PrintTools.printlnStatus("Fatal Error: Non Specifier List", 0);
            e.printStackTrace();
            System.exit(1);
        }
        Declaration decl = sym.getDeclaration();
        if (decl == null) {
            return false;
        }
        // check if parent SymbolTable is a TranslationUnit
        SymbolTable parent_sym =
                IRTools.getAncestorOfType(decl, SymbolTable.class);
        if (parent_sym instanceof TranslationUnit) {
            return true;
        }
        return false;
    }

    //[Added by Seyong Lee] used to treat extern symbols equivalently.
    /**
    * Checks if the given Object <b>obj</b> refers to the same Symbol as this
    * Symbol.
    * @param obj
    * @return true if <b>obj</b> refers to the same memory location as this
    * Symbol.
    */
    @Override
    public boolean equals(Object obj) {
        return isSameSymbol(obj);
    }

    //[Added by Seyong Lee] used to treat extern symbols equivalently.
    /**
    * If there are multiple Symbols with external linkage and same name,
    * hashCode() returns same value for those symbols
    * @return the computed hash code.
    */
    @Override
    public int hashCode() {
        // If Symbol has external linkage
        // return hashcode for symbol name
        if (hasExternalLinkage(this)) {
            return getSymbolName().hashCode();
        }
        // If Symbol has internal linkage
        // return default hashcode
        return super.hashCode();
    }

    @Override
    public void accept(TraversableVisitor v) { v.visit(this); }
}
