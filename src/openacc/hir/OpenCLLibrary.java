package openacc.hir;

import java.util.*;

import cetus.hir.*;

/**
* Repository for OpenCL library functions. This class provides a basic
* information about the OpenCL library calls. Knowing that if a function call
* may or must not have side effects can greatly improve the precision of a
* program analysis in general. OpenCL library calls may have side effects on
* the parameters, on the automatic storage locations, on the file system, and
* on the execution environment.
*/
public class OpenCLLibrary {

    /** Only a single object is constructed. */
    private static final OpenCLLibrary std = new OpenCLLibrary();

    /** Predefined properties for each library functions */
    private Map<String, Set<Property>> catalog;

    private static Map<String, int[]> seIndices = new HashMap<String, int[]>();

    /** Predefined set of properties */
    private enum Property {
        SIDE_EFFECT_GLOBAL,     // contains side effects on global variables.
        SIDE_EFFECT_PARAMETER,  // contains side effects through parameters.
        SIDE_EFFECT_STDIO,      // the only side effect is on standard io.
        SIDE_EFFECT_FILEIO,     // the only side effect is on file io.
        MAY_NOT_RETURN          // it is possible for the call not to return.
    }

    /**
    * Checks if the given function call is a standard library call.
    * @param fcall the function call to be examined.
    * @return true if the function call exists in the entries.
    */
    public static boolean contains(FunctionCall fcall) {
        return (std.catalog.get(fcall.getName().toString()) != null);
    }

    /**
    * Checks if the given function name is a standard library call.
    * @param fName the function name to be examined.
    * @return true if the function call exists in the entries.
    */
    public static boolean contains(String fName) {
        return (std.catalog.get(fName) != null);
    }

    /**
    * Checks if the given function call may have side effects.
    * @param fcall the function call to be examined.
    * @return true if the function call is standard library call and does not have a side effect.
    */
    public static boolean isSideEffectFree(FunctionCall fcall) {
        if (!contains(fcall)) {
            return false;
        }
        Set<Property> properties = std.catalog.get(fcall.getName().toString());
        return (!properties.contains(Property.SIDE_EFFECT_GLOBAL) &&
                !properties.contains(Property.SIDE_EFFECT_PARAMETER));
    }

    /**
    * Checks if the only side effect in the given function call is IO.
    * @param fcall the function call to be examined.
    * @return true if so.
    */
    public static boolean isSideEffectFreeExceptIO(FunctionCall fcall) {
        if (!contains(fcall)) {
            return false;
        }
        Set<Property> properties = std.catalog.get(fcall.getName().toString());
        return (properties.contains(Property.SIDE_EFFECT_STDIO) ||
                properties.contains(Property.SIDE_EFFECT_FILEIO));
    }

    /** Constructs a new repository */
    private OpenCLLibrary() {
        catalog = new HashMap<String, Set<Property>>();
        addEntries();
    }

    /**
    * Adds each entry to the repository.
    */
    private void addEntries() {
        //add("wprintf", Property.SIDE_EFFECT_GLOBAL,
        //                           Property.SIDE_EFFECT_STDIO);
    	//OpenCL Compute Capability 1.1
        add("mem_fence");
        add("read_mem_fence");
        add("write_mem_fence");
        add("atomic_inc");
        add("atomic_dec");
        add("atomic_cmpxchg");
        add("atomic_min");
        add("atomic_max");
        add("atomic_and");
        add("atomic_or");
        add("atomic_xor");
    	//OpenCL Compute Capability 1.2
        add("barrier");
    }

    /** Adds the specified properties to the call */
    private void add(String name, Property ... properties) {
        catalog.put(name, EnumSet.noneOf(Property.class));
        Set<Property> props = catalog.get(name);
        for (Property property : properties) {
            props.add(property);
        }
    }

    /* Automated categorizers */
    private static boolean takesVoid(ProcedureDeclarator pdecl) {
        return pdecl.getParameters().toString().equals("void ");
    }

    private static boolean returnsVoid(ProcedureDeclarator pdecl) {
        List types = pdecl.getTypeSpecifiers();
        return (types.size() == 1 && types.get(0) == Specifier.VOID);
    }

    private static int getNumMutableArguments(ProcedureDeclarator pdecl) {
        int ret = 0;
        if (takesVoid(pdecl)) {
            return ret;
        }
        for (Declaration decl : pdecl.getParameters()) {
            Symbol param = (Symbol)decl.getChildren().get(0);
            List types = param.getTypeSpecifiers();
            if (SymbolTools.isPointer(param) &&
                !types.contains(Specifier.CONST)) {
                ret++;
            }
        }
        return ret;
    }

    private static boolean containsVarArguments(ProcedureDeclarator pdecl) {
        return (pdecl.toString().contains("..."));
    }

    private static boolean containsStreamArguments(ProcedureDeclarator pdecl) {
        if (takesVoid(pdecl)) {
            return false;
        }
        for (Declaration decl : pdecl.getParameters()) {
            Symbol param = (Symbol)decl.getChildren().get(0);
            for (Object o : param.getTypeSpecifiers()) {
                if (o.toString().equals("FILE") ||
                    o.toString().equals("__FILE")) {
                    return true;
                }
            }
        }
        return false;
    }

    private static boolean returnsPointer(ProcedureDeclarator pdecl) {
        return SymbolTools.isPointer(pdecl);
    }

    public static void addSideEffectParamIndices(String fname, int[]indices) {
        seIndices.put(fname, indices);
    }

    /**
    * Returns the position of the function call arguments that may have a side
    * effect upon a call.
    * @param fcall the function call to be inspected.
    * @return the list of positions having a side effect.
    */
    public static int[] getSideEffectParamIndices(FunctionCall fcall) {
        int num_args = fcall.getNumArguments();
        int[] ret = new int[num_args];
        if (fcall.getName() instanceof Identifier &&
            SymbolTools.getSymbolOf(fcall.getName()) != null) {
            Symbol symbol = ((Identifier)fcall.getName()).getSymbol();
            ProcedureDeclarator pdecl = null;
            if (symbol instanceof ProcedureDeclarator) {
                pdecl = (ProcedureDeclarator)symbol;
            } else if (symbol instanceof Procedure) {
                pdecl =(ProcedureDeclarator)((Procedure)symbol).getDeclarator();
            }
            if (pdecl != null) {
                List<Declaration> params = pdecl.getParameters();
                boolean prev_decision = true;
                int pos = 0;
                for (int i = 0; i < params.size(); i++) {
                    Declaration d = params.get(i);
                    Symbol p = (Symbol)d.getChildren().get(0);
                    // Handling of variable-length arguments
                    if (d.toString().trim().equals("...")) {
                        if (prev_decision) {
                            for (int j = i; j < num_args; j++) {
                                ret[pos++] = j;
                            }
                        }
                    } else {
                        prev_decision = false;
                        Specifier pu = PointerSpecifier.UNQUALIFIED;
                        Specifier cn = Specifier.CONST;
                        List specs = SymbolTools.getNativeSpecifiers(fcall, p);
                        if (specs == null ||
                            specs.contains(PointerSpecifier.CONST) ||
                            specs.contains(PointerSpecifier.CONST_VOLATILE) ||
                            specs.contains(PointerSpecifier.VOLATILE)) {
                            ret[pos++] = i;
                            prev_decision = true;
                        } else if (specs.contains(pu)) {
                            int pu0 = specs.indexOf(pu);
                            int pu1 = specs.lastIndexOf(pu);
                            int cn0 = specs.indexOf(cn);
                            int cn1 = specs.lastIndexOf(cn);
                            if (pu0 >= 0 && cn0 >= 0 &&
                                pu0 == pu1 && cn0 == cn1 &&
                                pu0 > cn0) {
                                // The only case having no side effect.
                            	//[DEBUG by Seyong Lee] ..scanf() has side effects on var_arg list.
                            	if( fcall.getName().toString().contains("scanf") ) {
                            		prev_decision = true;
                            	}
                            } else {
                                ret[pos++] = i;
                                prev_decision = true;
                            }
                        } else {
//                            System.out.println(specs);
                            boolean has_array_spec = false;
                            for (int j = 0; j < specs.size(); j++) {
                                if (specs.get(j) instanceof ArraySpecifier) {
                                    has_array_spec = true;
                                    break;
                                }
                            }
                            if (has_array_spec && !specs.contains(cn)) {
                                ret[pos++] = i;
                                prev_decision = true;
                            }
                        }
                    }
                }
                return Arrays.copyOf(ret, pos);
            }
        }
        // returns conservative result for all other cases
        for (int i = 0; i < num_args; i++) {
            ret[i] = i;
        }
        return ret;
/*
        if (hasSideEffectOnParameter(fcall) == false) {
            return null;
        }
        return seIndices.get(fcall.getName().toString());
*/
    }

    public static boolean hasSideEffectOnParameter(FunctionCall fcall) {
        if (!contains(fcall)) {
            return false;
        }
        Set<Property> properties = std.catalog.get(fcall.getName().toString());
        return properties.contains(Property.SIDE_EFFECT_PARAMETER) &&
                properties.contains(Property.SIDE_EFFECT_FILEIO) == false;
    }

}
