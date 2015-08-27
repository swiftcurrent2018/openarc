package cetus.hir;

import cetus.exec.Driver;

import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.util.*;

/**
* Represents a function, subroutine, or method. A procedure object consists of
* specifiers, a procedure declarator, initializers, and a body, the body being
* the only traversable child.
*/
public final class Procedure extends Declaration
                             implements SymbolTable, Symbol {

    /** Default method for printing */
    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = Procedure.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }

    /**
     * List of specifiers not attached to declarators
     * [Renamed from leading_specs by Joel E. Denny]
     */
    private List<Specifier> specs;

    /** Procedure declarator of the procedure */
    private Declarator declarator;

    /** List of initializers - C++ feature */
    private List initializers;

    /** Look-up table for symbol search */
    private Map<IDExpression, Declaration> symbol_table;

    /** Original source was declared in old sytle */
    private boolean is_old_style_function;

    /** GNU-style attributes or null. [Added by Joel E. Denny] */
    private AttributeSpecifier attrSpec;

    /**
    * Creates a constructor definition (declaration plus body).
    *
    * @param declarator The name and parameter list of the procedure.
    * Must not be null.
    * @param body The body of the procedure.
    * @throws NotAnOrphanException If <b>body</b> has a parent object.
    */
    public Procedure(Declarator declarator, CompoundStatement body) {
        super(1);
        this.is_old_style_function = false;
        object_print_method = class_print_method;
        this.specs = new ArrayList<Specifier>(1);
        this.declarator = declarator;
        symbol_table = new LinkedHashMap<IDExpression, Declaration>();
        if (body.getParent() != null) {
            throw new NotAnOrphanException();
        }
        children.add(body);
        body.setParent(this);
        List<Declaration> params
            = getDeclaratorWithParameters().getParameters();
        for (int i = 0; i < params.size(); i++) {
            SymbolTools.addSymbols(this, params.get(i));
        }
        this.attrSpec = null;
    }

    /**
    * Creates a procedure definition (declaration plus body).
    *
    * @param specs A list of specifiers describing the return type of
    * the procedure.  May be null.
    * @param declarator The name and parameter list of the procedure.
    * Must not be null.
    * @param body The body of the procedure.
    * @throws NotAnOrphanException If <b>body</b> has a parent object.
    */
    public Procedure(List specs,
                     Declarator declarator, CompoundStatement body) {
        super(1);
        this.is_old_style_function = false;
        object_print_method = class_print_method;
        this.specs = new ArrayList<Specifier>(specs.size());
        for (int i = 0; i < specs.size(); i++) {
            this.specs.add((Specifier)specs.get(i));
        }
        this.declarator = declarator;
        symbol_table = new LinkedHashMap<IDExpression, Declaration>();
        if (body.getParent() != null) {
            throw new NotAnOrphanException();
        }
        children.add(body);
        body.setParent(this);
        if (getDeclaratorWithParameters().getParameters() != null) {
            List<Declaration> params
                = getDeclaratorWithParameters().getParameters();
            for (int i = 0; i < params.size(); i++) {
                SymbolTools.addSymbols(this, params.get(i));
            }
        }
        this.attrSpec = null;
    }

    /** Constructor that can preserve the old style of the declaration */
    public Procedure(List specs, Declarator declarator,
                     CompoundStatement body, boolean is_old_style_function) {
        this(specs, declarator, body, is_old_style_function, null);
    }
    public Procedure(List specs, Declarator declarator,
                     CompoundStatement body, boolean is_old_style_function,
                     AttributeSpecifier attrSpec) {
        this(specs, declarator, body);
        this.is_old_style_function = is_old_style_function;
        this.attrSpec = attrSpec;
    }

    /**
    * Creates a procedure definition (declaration plus body).
    *
    * @param spec A specifier describing the return type of the procedure.
    * @param declarator The name and parameter list of the procedure.
    * Must not be null.
    * @param body The body of the procedure.
    * @throws NotAnOrphanException If <b>body</b> has a parent object.
    */
    public Procedure(Specifier spec,
                     Declarator declarator, CompoundStatement body) {
        super(1);
        this.is_old_style_function = false;
        object_print_method = class_print_method;
        this.specs = new ArrayList<Specifier>(1);
        this.specs.add(spec);
        this.declarator = declarator;
        symbol_table = new LinkedHashMap<IDExpression, Declaration>();
        if (body.getParent() != null) {
            throw new NotAnOrphanException();
        }
        children.add(body);
        body.setParent(this);
        if (getDeclaratorWithParameters().getParameters() != null) {
            List<Declaration> params
                = getDeclaratorWithParameters().getParameters();
            for (int i = 0; i < params.size(); i++) {
                SymbolTools.addSymbols(this, params.get(i));
            }
        }
        this.attrSpec = null;
    }

    /**
    * Adds a parameter declaration to the procedure.
    *
    * @param decl the new parameter declaration to be added.
    */
    public void addDeclaration(Declaration decl) {
        getDeclaratorWithParameters().addParameter(decl);
        SymbolTools.addSymbols(this, decl);
    }

    /**
    * Adds a new parameter declaration before the reference declaration.
    *
    * @param ref the reference parameter declaration.
    * @param decl the new parameter declaration to be added.
    */
    public void addDeclarationBefore(Declaration ref, Declaration decl) {
        getDeclaratorWithParameters().addParameterBefore(ref, decl);
        SymbolTools.addSymbols(this, decl);
    }

    /**
    * Adds a new parameter declaration after the reference declaration.
    *
    * @param ref the reference parameter declaration.
    * @param decl the new parameter declaration to be added.
    */
    public void addDeclarationAfter(Declaration ref, Declaration decl) {
        getDeclaratorWithParameters().addParameterAfter(ref, decl);
        SymbolTools.addSymbols(this, decl);
    }

    /**
    * [Added by Seyong Lee]
    * Replace an existing parameter declaration with a new one.
    *
    * @param ref the existing parameter declaration to be replaced.
    * @param decl the new parameter declaration to be added.
    */
    public void replaceDeclaration(Declaration ref, Declaration decl) {
        getDeclaratorWithParameters().replaceParameter(ref, decl);
        SymbolTools.removeSymbols(this, ref);
        SymbolTools.addSymbols(this, decl);
    }

    /**
    * Returns the procedure declarator of the procedure.
    *
    * @return the procedure declarator.
    */
    public Declarator getDeclarator() {
        return declarator;
    }

    /**
    * Prints a procedure to a stream.
    *
    * @param p The procedure to print.
    * @param o The writer on which to print the procedure.
    */
    public static void defaultPrint(Procedure p, PrintWriter o) {
        boolean enable_old_style_function;
        if (Driver.getOptionValue("preserve-KR-function") != null) {
            enable_old_style_function = true;
        } else {
            enable_old_style_function = false;
        }
        PrintTools.printListWithSeparator(p.specs, o, " ");
        o.print(" ");
        if (p.attrSpec != null) {
            p.attrSpec.print(o);
            o.print(" ");
        }
        if (enable_old_style_function && p.is_old_style_function) {
            PrintTools.printList(p.declarator.getSpecifiers(), o);
            p.declarator.getID().print(o);
            o.print("(");
            List<Declaration> params
                = p.getDeclaratorWithParameters().getParameters();
            StringBuilder decl_list = new StringBuilder(80);
            for (int i = 0; i < params.size(); i++) {
                Declaration decl = params.get(i);
                decl_list.append("\n");
                decl_list.append(decl.toString());
                decl_list.append(";");
                if (i > 0) {
                    o.print(", ");
                }
                o.print(decl.getDeclaredIDs().get(0).toString());
            }
            o.print(")");
            o.print(decl_list);
        } else {
            p.declarator.print(o);
        }
        o.println("");
        p.getBody().print(o);
        o.println("");
    }

    /* SymbolTable interface */
    public Declaration findSymbol(IDExpression name) {
        return SymbolTools.findSymbol(this, name);
    }

    /**
    * Returns the body compound statement of this procedure.
    *
    * @return the body of this procedure.
    */
    public CompoundStatement getBody() {
        return (CompoundStatement)children.get(0);
    }

    /**
    * Returns the list of declared name IDs in the procedure.
    *
    * @return the list of declared IDs.
    */
    public List<IDExpression> getDeclaredIDs() {
        List<IDExpression> ret = new ArrayList<IDExpression>(1);
        ret.add(declarator.getID());
        return ret;
    }

    /**
    * Returns the name ID of the procedure.
    *
    * @return the name ID of the procedure.
    */
    public IDExpression getName() {
        return declarator.getID();
    }

    // [Added by Joel E. Denny to handle case where the immediate child
    // Declarator is a NestedDeclarator. We need to look at the
    // ProcedureDeclarator at the bottom of the tree to get the Procedure's
    // parameters.]
    private Declarator getDeclaratorWithParameters() {
      return (Declarator)declarator.getDirectDeclarator().getParent();
    }

    /** Returns the number of parameters of the procedure. */
    public int getNumParameters() {
        return getDeclaratorWithParameters().getParameters().size();
    }

    /** Returs the list of procedure parameters. */
    public List getParameters() {
        return getDeclaratorWithParameters().getParameters();
    }

    /**
    * returns the parameter at specified index
    * @param index - zero-based index of the required parameter
    * @return - parameter at specified index or null 
    */
    public Declaration getParameter(int index) {
        return getDeclaratorWithParameters().getParameter(index);
    }

    /* SymbolTable interface */
    public List<SymbolTable> getParentTables() {
        return SymbolTools.getParentTables(this);
    }

    /**
    * Returns the list of type specifiers of the procedure. WARNING: In some
    * cases, such as returning a pointer to an array, this does not include all
    * specifiers necessary to specify the return type. Instead, call
    * {@link #getSpecifiers}, which unlike this method does not include any
    * specifiers from descendant declarators, and then traverse the hierarchy
    * of descendant declarators to find the remaining specifiers.
    */
    @SuppressWarnings("unchecked")
    public List getReturnType() {
        List list = new ArrayList();
        list.addAll(specs);
        list.addAll(declarator.getSpecifiers());
        return list;
    }

    /**
    * Returns the symbol look-up table of the procedure.
    * This method is protected for consistent management of symbol table.
    *
    * @return the internal look-up table.
    */
    protected Map<IDExpression, Declaration> getTable() {
        return symbol_table;
    }

    /**
    * Assigns a new procedure body for the procedure.
    *
    * @param body the new procedure body to be added.
    * @throws NotAnOrphanException if <b>body</b> has a parent object.
    */
    public void setBody(CompoundStatement body) {
        if (getBody() != null) {
            getBody().setParent(null);
        }
        if (body.getParent() != null) {
            throw new NotAnOrphanException();
        }
        children.set(0, body);
        body.setParent(this);
    }

    /**
    * Overrides the class print method, so that all subsequently
    * created objects will use the supplied method.
    *
    * @param m The new print method.
    */
    public static void setClassPrintMethod(Method m) {
        class_print_method = m;
    }

    /**
    * Sets the list of initializers - not used in a C program.
    */
    public void setConstructorInitializers(List list) {
        initializers = list;
    }

    /**
    * Modifies the name of the procedure.
    */
    public void setName(IDExpression name) {
        declarator.setDirectDeclarator(name);
    }

    /* Symbol interface */
    public void setName(String name) {
        SymbolTable symtab = IRTools.getAncestorOfType(this, SymbolTable.class);
        SymbolTools.setSymbolName(this, name, symtab);
    }

    /* Symbol interface */
    public String getSymbolName() {
        return getName().toString();
    }

    /* Symbol interface */
    public List getTypeSpecifiers() {
        return getReturnType();
    }

    /** Not used in a procedure; it just returns null */
    public List getArraySpecifiers() {
        return null;
    }

    // [Made public by Joel E. Denny]
    public List<Specifier> getSpecifiers() {
        return specs;
    }

    public AttributeSpecifier getAttributeSpecifier() {
        return attrSpec;
    }
    
    public void setAttributeSpecifier(AttributeSpecifier attrib) {
    	this.attrSpec = attrib;
    }

    /** Returns a clone of the procedure. */
    @Override
    @SuppressWarnings("unchecked")
    public Procedure clone() {
        Procedure p = (Procedure)super.clone();
        if (specs != null) {
            p.specs = new ArrayList(specs);
        }
        p.attrSpec = attrSpec == null ? null : attrSpec.clone();
        if (declarator != null) {
            p.declarator = declarator.clone();
        }
        if (initializers != null) {
            p.initializers = new ArrayList(initializers.size());
            for (int i = 0; i < initializers.size(); i++) {
                p.initializers.add(initializers.get(i));
            }
        }
        // Builds an internal look-up table.
        if (symbol_table != null) {
            p.symbol_table = new LinkedHashMap<IDExpression, Declaration>();
            List<Declaration> params
                = p.getDeclaratorWithParameters().getParameters();
            for (int i = 0; i < params.size(); i++) {
                SymbolTools.addSymbols(p, params.get(i));
            }
        }
        // Fixes obsolete symbol references in the IR.
        SymbolTools.relinkSymbols(p);
        return p;
    }

    /* SymbolTable interface */
    public Set<Symbol> getSymbols() {
        return SymbolTools.getSymbols(this);
    }

    /* SymbolTable interface */
    public Set<Declaration> getDeclarations() {
        return new LinkedHashSet<Declaration>(symbol_table.values());
    }

    /* Symbol interface */
    public Declaration getDeclaration() {
        return this;
    }

    /* SymbolTable interface */
    public boolean containsSymbol(Symbol symbol) {
        for (IDExpression id : symbol_table.keySet()) {
            if (id instanceof Identifier &&
                symbol.equals(((Identifier)id).getSymbol())) {
                return true;
            }
        }
        return false;
    }

    /* SymbolTable interface */
    public boolean containsDeclaration(Declaration decl) {
        return symbol_table.containsValue(decl);
    }

    @Override
    public void accept(TraversableVisitor v) { v.visit(this); }
}
