package openacc.hir;

import java.util.*;

import cetus.hir.*;
import openacc.hir.*;
import openacc.analysis.SubArray;

/**
 * ARCAnnotation is used to represent various OpenARC extensions.
 * OpenARC pragmas are raw text right after parsing 
 * but converted to an internal annotation of this type.
 * 
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future TechnologiesGroup, Oak Ridge National Laboratory
 */
public class ARCAnnotation extends PragmaAnnotation
{
/**
 * <p>     
 * #pragma openarc #define macro value
 * <p>     
 * #pragma openarc ainfo procname(proc-name) kernelid(kernel-id)
 * <p>     
 * #pragma openarc cuda [clause[[,] clause]...]
 * <p>     
 * where clause is one of the following
 * 		registerRO(list) 
 * 		registerRW(list) 
 *      noregister(list)
 * 		sharedRO(list) 
 * 		sharedRW(list) 
 *      noshared(list)
 * 		texture(list) 
 *      notexture(list)
 * 		constant(list)
 * 		noconstant(list)
 *      global(list)
 * <p>
 * #pragma openarc opencl [clause[[,] clause]...]
 * <p>     
 * where clause is one of the following
 *      num_simd_work_items(exp)
 *      num_compute_units(exp)
 * <p>     
 * #pragma openarc transform [clause[[,] clause]...]
 * <p>     
 * where clause is one of the following
 *      permute(list)
 *      unroll(exp)
 *      noreductionunroll(list)
 *      transpose(list-of-subarray-with-conflist)
 *      redim(list-of-subarray-with-conflist)
 *      expand(list-of-subarray-with-conflist)
 *      redim_transpose(list-of-subarray-with-conflist)
 *      expand_transpose(list-of-subarray-with-conflist)
 *      transpose_expand(list-of-subarray-with-conflist)
 *      noploopswap
 *      noloopcollapse
 *      window(subarray, exp, exp, exp)
 *      multisrccg(list)
 *      multisrcgc(list)
 *      conditionalsrc(list)
 *      enclosingloops(list)
 *      
 * where subarray-with-conflist has the following format
 *      subarray::[expression list]::[expression list]...
 *     
 * #pragma openarc resilience [clause[[,] clause]...]
 * <p>     
 * where clause is one of the following
 *     ftregion
 *     ftcond(condition)
 *     ftdata(list)
 *     ftkind(list)
 *     num_faults(scalar-integer-expression)
 *     num_ftbits(scalar-integer-expression)
 *     repeat(scalar-integer-expression)
 *     ftthread(scalar-integer-expression)
 *     
 * #pragma openarc ftinject [clause[[,] clause]...]
 * <p>     
 * where clause is one of the following
 *     ftdata(list)
 *     ftthread(scalar-integer-expression)
 *     
 * #pragma openarc ftregion [clause[[,] clause]...]
 * <p>     
 * where clause is one of the following
 *     ftdata(list)
 *     ftkind(list)
 *     ftthread(scalar-integer-expression)
 * 
 * #pragma openarc profile region label(name) [clause[[,] clause]...]
 * <p>
 * where clause is one of the following
 * mode(list)  //a list consists of the following:
 *             //memory, instructions, occupancy, memorytransfer, all 
 * event(list) //a list consists of expressions
 * verbosity(arg) //an arg is a non-negative integer where
 *                //0 is the least verbose mode. 
 * <p>
 * 
 * #pragma openarc enter profile region label(name) [clause[[,] clause]...]
 * <p>
 * where clause is one of the following
 * mode(list)  //a list consists of the following:
 *             //memory, instructions, occupancy, memorytransfer, all 
 * event(list) //a list consists of expressions
 * verbosity(arg) //an arg is a non-negative integer where
 *                //0 is the least verbose mode. 
 * <p>
 * #pragma openarc exit profile region label(name)
 * <p>
 * #pragma openarc profile track [clause [[,] clause] ...]
 * <p>
 * where clause is one of the following
 * event(list) //a list consists of expressions
 * induction(induction-expression) 
 * profcond(expression) 
 * label(name)
 * mode(list)  //a list consists of the following:
 *             //memory, instructions, occupancy, memorytransfer, all 
 * <p>
 * #pragma openarc profile measure [clause [[,] clause] ...]
 * <p>
 * where clause is one of the following
 * event(list) //a list consists of expressions
 * induction(induction-expression) 
 * profcond(expression) 
 * label(name)
 * <p>
 * #pragma openarc impacc [clause [[,] clause] ...]
 * <p>
 * where clause is one of the following
 * ignoreglobal(list) //a list of global variables to be ignored by IMPACC.
 * <p>
 * #pragma openarc devicetask [clause [[,] clause] ...]
 * <p>
 * where clause is one of the following
 * map (task-mapping-scheme) //task-mapping-scheme: included, coarse_grained, or fine_grained
 * schedule (task-scheduling-scheme) //task-scheduling-scheme: LRR, GRR, LF
 * 
 * 
 */

	// Pragmas used without values
	private static final Set<String> no_values =
		new HashSet<String>(Arrays.asList("ainfo", 
				"noploopswap", "noloopcollapse", "transform",
				"resilience", "ftinject", "ftregion",
				"enter", "exit", "profile", "region", "measure", "track",
				"impacc", "devicetask"));
	

	// Pragmas used with collection of values
	private static final Set<String> collection_values =
		new HashSet<String>(Arrays.asList(
		"registerRO", "registerRW", "sharedRO", "psharedRO", "sharedRW", "global", 
		"texture", "constant", "noconstant", "noreductionunroll",
		"procname", "kernelid", "unroll",
		"noregister", "noshared", "notexture", "enclosingloops",
		"multisrccg", "multisrcgc", "conditionalsrc", "permute", 
		"ftdata", "ftkind", "mode", "event", 
		"transpose", "redim", "expand", "redim_transpose", "expand_transpose",
		"transpose_expand", "num_simd_work_items", "num_compute_units",
		"ignoreglobal", "window", "map", "schedule"));
	

	// Pragmas used with optional value
	private static final Set<String> optional_values = 
		new HashSet<String>(Arrays.asList("NULL"));
	
	//List used to set print orders.
	//Clauses not listed here may be printed in a random order.
	private static final List<String> print_order =
			new ArrayList<String>(Arrays.asList( "ainfo", "enter", "exit",  "cuda", "opencl", "impacc",
					"transform", "transpose", "redim", "expand", "redim_transpose", "expand_transpose",
					"transpose_expand", "window",
					"profile", "region", "measure", "track", "if", "resilience", 
					"ftregion", "ftinject", "ftcond", "ftthread", "ftdata", "ftkind", "num_faults", 
					"num_ftbits", "repeat", "label", "mode", "event", "induction", "verbosity",
					"ignoreglobal"));
	
	// List of OpenARC directives.
	public static final Set<String> OpenARCDirectiveSet = new HashSet(Arrays.asList(
			"ainfo", "cuda", "opencl", "enter", "exit", "profile", "datalayout", 
			"ftregion", "ftinject", "resilience", "transform", "impacc", "devicetask"));
	
	// Directives attached to structured blocks
	public static final Set<String> directivesForStructuredBlock = new HashSet(Arrays.asList(
			"ainfo", "cuda", "opencl", "profile", "resilience", "ftregion", "transform", "devicetask" )); 
	
	// Clauses that have a list as arguments.
	public static final Set<String> collectionClauses = collection_values;
	
	public static final Set<String> ftinjecttargets = new HashSet(Arrays.asList("ftregion", "ftinject"));
	
	public static final Set<String> resilienceDirectives = new HashSet(Arrays.asList("resilience", "ftregion", "ftinject"));
	
	// Transform clauses
	public static final Set<String> transformClauses = new HashSet(Arrays.asList(
		"permute", "unroll", "noreductionunroll", "noploopswap", "noloopcollapse", 
		"transpose", "redim", "expand", "redim_transpose", "expand_transpose", "transpose_expand",
		"multisrccg", "multisrcgc", "conditionalsrc", "enclosingloops", "window"
		));
	
	// CUDA clauses
	public static final Set<String> cudaClauses = new HashSet(Arrays.asList(
		"registerRO", "registerRW", "noregister", "sharedRO", "sharedRW", "noshared", "texture",
		"notexture", "constant", "noconstant", "global" 
		));
	
	// CUDA data clauses
	public static final Set<String> cudaDataClauses = new HashSet(Arrays.asList("registerRO", "registerRW", "noregister",
			"sharedRO", "sharedRW", "noshared", "texture", "notexture", "constant", "noconstant", "global" ));
	
	public static final Set<String> cudaRODataClauses = new HashSet(Arrays.asList("registerRO", "sharedRO", "texture", 
			"constant" ));
	
	// CUDA data clauses that affect memory allocation
	public static final Set<String> cudaMDataClauses = new HashSet(Arrays.asList("texture", "notexture", 
			"constant", "noconstant", "global" ));

	// OpenCL clauses
	public static final Set<String> openclClauses = new HashSet(Arrays.asList(
		"num_simd_work_items", "num_compute_units" 
		));
	
	// OpenARC clauses that should be printed in the order as specified in the input program.
	private static final Set<String> inOrderClauses = new HashSet(Arrays.asList("enclosingloops", "permute", "gangconf",
			"workerconf", "window"));
	
	// Resilience clauses 
	private static final Set<String> resilienceClauses = new HashSet(Arrays.asList("ftregion", "ftcond",
			"ftdata", "ftkind", "num_faults", "num_fbits", "repeat", "ftthread" ));
	
	// ftregion clauses 
	private static final Set<String> ftregionClauses = new HashSet(Arrays.asList(
			"ftdata", "ftkind", "ftthread" ));
	
	// profile clauses 
	private static final Set<String> profileClauses = new HashSet(Arrays.asList(
		"region", "track", "measure", "mode", "label", "event", "induction",
		"verbosity", "profcond"
		));

	// devicetask clauses 
	private static final Set<String> deviceTaskClauses = new HashSet(Arrays.asList(
		"map", "schedule"
		));

    /**
    * Returns a clone of this annotation object.

	/**
	 * Constructs an empty OpenARC annotation.
	 */
	public ARCAnnotation()
	{
		super("openarc");
	}

	/**
	 * Constructs an OpenACC annotation with the given key-value pair.
	 */
	public ARCAnnotation(String key, Object value)
	{
		super("openarc");
		put(key, value);
	}
	
	/**
	 * Check whether this annotation is applicable to the input Annotatable, <var>at</var>.
	 * For stand-alone annotations, this will always return false.
	 * 
	 * @param at Annotatable to which this annotation will be attached.
	 * @return true if this annotation is applicable to the input Annotatable <var>at</var>
	 */
	public boolean isValidTo(Annotatable at) {
		boolean isValid = true;
		if( containsKey("ftinject") || containsKey("enter") ||
				containsKey("exit") || containsKey("measure") ) {
			//Stand-alone directives
			//acc profile measure is a stand-alone directive.
			//enter/exit data and enter/exit profile are stand-alone directives.
			isValid = false;
		} else if( containsKey("ainfo") || containsKey("resilience") ||
				containsKey("ftregion") || containsKey("profile") ) {
			if( (at instanceof DeclarationStatement) || (at instanceof AnnotationStatement) ) {
				isValid = false;
			}
		}
		return isValid;
	}
	
    /**
    * Returns a clone of this annotation object.
    * @return a cloned annotation.
    */
/*    @SuppressWarnings("unchecked")
    @Override
    public Annotation clone() {
        Annotation o = (Annotation)super.clone();      // super is cloneable.
        // Overwrite shallow copies.
        o.clear();
        Iterator<String> iter = keySet().iterator();
        while (iter.hasNext()) {
            String key = iter.next();
            Object val = get(key);
            o.put(key, cloneObject(val));
        }
        o.position = this.position;
        o.skip_print = this.skip_print;
        // ir are overwritten only by annotatable.annotate().
        return o;
    }*/
	
    /**
    * returns the deep copy of the given map
    */
/*    @SuppressWarnings("unchecked")
    private HashMap cloneMap(Map map) {
        Iterator<ReductionOperator> iter = map.keySet().iterator();
        HashMap clone = new HashMap();
        while (iter.hasNext()) {
            ReductionOperator redOp = iter.next();
            Object val = map.get(redOp);
            clone.put(redOp, cloneObject(val));
        }
        return clone;
    }*/

    /**
    * returns the deep copy of the given object (which could be String,
    * Collection, Map or null). Symbol is also returned as a shallow copy.
    */
/*    private Object cloneObject(Object obj) {
        if (obj instanceof String || obj instanceof Symbol) {
            return obj;
        } else if (obj instanceof Expression) {
        	return ((Expression)obj).clone();
        } else if (obj instanceof SubArray) {
        	return ((SubArray)obj).clone();
        } else if (obj instanceof Collection) {
            return cloneCollection((Collection)obj);
        } else if (obj instanceof Map) {
            return cloneMap((Map)obj);
        } else if (obj == null) {
            // for some keys in the maps, values are null
            return null;
        } else {
            System.err.println(
                    "Annotation argument, " + obj + ", has unhandled Object type, fix me in AccAnnotation.java");
            return null;
        }
    }*/

    /**
    * returns the deep copy of the given collection 
    */
/*    @SuppressWarnings("unchecked")
    private LinkedList cloneCollection(Collection c) {
        Iterator iter = c.iterator();
        LinkedList list = new LinkedList();
        while (iter.hasNext()) {
            Object val = iter.next();
            list.add(cloneObject(val));
        }
        return list;
    }*/

	/**
	 * Returns the string representation of this cuda annotation.
     *
	 * @return the string representation.
	 */
	public String toString()
	{
		if ( skip_print )
			return "";

		StringBuilder str = new StringBuilder(80);

		str.append(super.toString());
		
		Set<String> directiveSet = new HashSet<String>();
		directiveSet.addAll(keySet());
		directiveSet.remove("pragma");
		
		for( String key : print_order ) {
			if( directiveSet.contains(key) ) {
				printDirective(key, str);
				directiveSet.remove(key);
			}
		}
		if( !directiveSet.isEmpty() ) {
			for( String key: directiveSet ) {
				printDirective(key, str);
			}
		}
		
		return str.toString();
	}
	
	private void printDirective(String key, StringBuilder str) {
		if ( no_values.contains(key) )
			str.append(" "+key);
		else if ( collection_values.contains(key) )
		{
			str.append(" "+key+"(");
			Object value = get(key);
			if ( value instanceof Collection )
				if( value instanceof List ) {
					str.append(PrintTools.listToString((List)value, ", "));
				} else {
					str.append(PrintTools.collectionToString((Collection)value, ", "));
				}
			else // e.g., num_gangs
				str.append(value);
			str.append(")");
		}
		else if ( optional_values.contains(key) )
		{
			str.append(" "+key);
			Object tVal = get(key);
			if ( tVal != null && (!tVal.equals("true") && !tVal.equals("_directive") 
					&& !tVal.equals("_clause")) )
				str.append("("+tVal+")");
		}
		else
		{
			//If this annotation contains annotatable object as value, 
			//printing the value will cause infinite recursion; skip
			//printing annotatable object.
			Object tObj = get(key);
			if( !(tObj instanceof Annotatable) ) {
			str.append(" "+key);
			if ( (tObj != null) && (!"true".equals(tObj) && !"_directive".equals(tObj) 
					&& !"_clause".equals(tObj)) )
				str.append(" ("+tObj+")");
			}
		}
	}
}
