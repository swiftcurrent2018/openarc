package openacc.hir;

import java.util.*;

import cetus.hir.*;
import openacc.hir.*;
import openacc.analysis.SubArray;

/**
 * ACCAnnotation is used for internally representing OpenACC pragmas.
 * OpenACC pragmas are raw text right after parsing 
 * but converted to an internal annotation of this type.
 * 
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future TechnologiesGroup, Oak Ridge National Laboratory
 */
public class ACCAnnotation extends PragmaAnnotation
{
/**
 * OpenACC Annotation List.
 * The following user directives are supported:
 * <p>     
 * #pragma acc parallel [clause[[,] clause]...]
 * <p>     
 * where clause is one of the following:
 *      if( condition )
 *      async [( scalar-integer-expression )]
 *      num_gangs( scalar-integer-expression )
 *      num_workers( scalar-integer-expression )
 *      vector_length( scalar-integer-expression )
 *      reduction( operator:list )
 * 		copy( list ) 
 * 		copyin( list ) 
 * 		copyout( list ) 
 * 		create( list ) 
 * 		present( list ) 
 * 		present_or_copy( list ) 
 * 		pcopy( list ) 
 * 		present_or_copyin( list ) 
 * 		pcopyin( list ) 
 * 		present_or_copyout( list ) 
 * 		pcopyout( list ) 
 * 		present_or_create( list ) 
 * 		pcreate( list ) 
 * 		deviceptr( list ) 
 * 		private( list ) 
 * 		firstprivate( list ) 
 *      pipein( list )
 *      pipeout( list )
 * <p>     
 * #pragma acc kernels [clause[[,] clause]...]
 * <p>     
 * where clause is one of the following:
 *      if( condition )
 *      async [( scalar-integer-expression )]
 * 		copy( list ) 
 * 		copyin( list ) 
 * 		copyout( list ) 
 * 		create( list ) 
 * 		present( list ) 
 * 		present_or_copy( list ) 
 * 		pcopy( list ) 
 * 		present_or_copyin( list ) 
 * 		pcopyin( list ) 
 * 		present_or_copyout( list ) 
 * 		pcopyout( list ) 
 * 		present_or_create( list ) 
 * 		pcreate( list ) 
 * 		deviceptr( list ) 
 *      pipein( list )
 *      pipeout( list )
 * <p>     
 * #pragma acc data [clause[[,] clause]...]
 * <p>     
 * where clause is one of the following:
 *      if( condition )
 * 		copy( list ) 
 * 		copyin( list ) 
 * 		copyout( list ) 
 * 		create( list ) 
 * 		present( list ) 
 * 		present_or_copy( list ) 
 * 		pcopy( list ) 
 * 		present_or_copyin( list ) 
 * 		pcopyin( list ) 
 * 		present_or_copyout( list ) 
 * 		pcopyout( list ) 
 * 		present_or_create( list ) 
 * 		pcreate( list ) 
 * 		deviceptr( list ) 
 *      pipe( list )
 * <p>     
 * #pragma acc host_data [clause[[,] clause]...]
 * <p>     
 * where clause is one of the following:
 *      use_device( condition )
 * <p>     
 * #pragma acc loop [clause[[,] clause]...]
 * <p>     
 * where clause is one of the following:
 *      collapse( n )
 *      gang [( scalar-integer-expression )]
 *      worker [( scalar-integer-expression )]
 *      vector [( scalar-integer-expression )]
 *      seq
 *      independent
 *      private( list )
 *      reduction( operator:list )
 *      tile( list )
 * <p>     
 * #pragma acc parallel loop [clause[[,] clause]...]
 * <p>     
 * where clause is any clause allowed on a parallel or loop directive.
 * <p>     
 * #pragma acc kernels loop [clause[[,] clause]...]
 * <p>     
 * where clause is any clause allowed on a kernels or loop directive.
 * <p>     
 * #pragma acc cache ( list )
 * <p>     
 * #pragma acc declare declclause [[,] declclause]...
 * <p>     
 * where declclause is one of the following:
 * 		copy( list ) 
 * 		copyin( list ) 
 * 		copyout( list ) 
 * 		create( list ) 
 * 		present( list ) 
 * 		present_or_copy( list ) 
 * 		pcopy( list ) 
 * 		present_or_copyin( list ) 
 * 		pcopyin( list ) 
 * 		present_or_copyout( list ) 
 * 		pcopyout( list ) 
 * 		present_or_create( list ) 
 * 		pcreate( list ) 
 * 		deviceptr( list ) 
 * 		device_resident( list ) 
 * 		pipe( list ) 
 * <p>     
 * #pragma acc update clause[[,] clause]...
 * <p>     
 * where clause is one of the following:
 *      host( list )
 *      device( list )
 *      if( condition )
 *      async [( scalar-integer-expression )]
 * <p>     
 * #pragma acc enter data clause[[,] clause]...
 * <p>     
 * where clause is one of the following:
 *      if( condition )
 *      async [( scalar-integer-expression )]
 *      wait [( scalar-integer-expression )]
 *      copyin( list )
 *      create( list )
 * <p>     
 * #pragma acc exit data clause[[,] clause]...
 * <p>     
 * where clause is one of the following:
 *      if( condition )
 *      async [( scalar-integer-expression )]
 *      wait [( scalar-integer-expression )]
 *      copyout( list )
 *      delete( list )
 *      finalize
 * <p>     
 * #pragma acc wait [( scalar-integer-expression )]
 * <p>     
 * #pragma acc routine [clause[[,] clause]...]
 * <p>     
 * where clause is one of the following
 *     bind(name)
 *     nohost
 *     type(workshare type)     
 * <p>     
 * #pragma acc set [clause[[,] clause]...]
 * <p>     
 * where clause is one of the following
 *     default_async ( scalar-integer-expression )
 *     device_num ( scalar-integer-expression )
 *     device_type ( device-type )
 * <p>
 * #pragma acc mpi [clause[[,] clause]...]
 * <p>
 * where clause is one of the following
 *     sendbuf ( [device] [,] [readonly] )
 *     recvbuf ( [device] [,] [readonly] )
 *     async [(int-expr)]    
 * 
 *     
 */

	// Pragmas used without values
	private static final Set<String> no_values =
		new HashSet<String>(Arrays.asList("parallel", "kernels", 
				"data", "loop", "declare", "update", "host_data",
				"seq", "independent", "internal", "routine", 
				"enter", "exit", "set", "finalize", "atomic"));

	// Pragmas used with collection of values
	private static final Set<String> collection_values =
		new HashSet<String>(Arrays.asList("copy", "copyin", "copyout",
		"create", "present", "present_or_copy", "pcopy", "pcopyin",
		"present_or_copyin", "present_or_copyout", "pcopyout",
		"present_or_create", "pcreate", "deviceptr", "private", "firstprivate",
		"if", "num_gangs", "num_workers", "vector_length", "reduction",
		"use_device", "collapse", "cache", "device_resident", "host",
		"device", 
		"accglobal", "accshared", "accprivate", "accreduction", "accdeviceptr",
		"accexplicitshared", "accreadonly", "accpreadonly",
		"iterspace", "rcreate", "gangdim", "workerdim", "gangconf", "workerconf",
		"totalnumgangs", "totalnumworkers", "tile", "pipe", "pipein", "pipeout",
		"default_async", "device_num", "device_type", "delete",
		"sendbuf", "recvbuf"));

	// Pragmas used with optional value
	private static final Set<String> optional_values = 
		new HashSet<String>(Arrays.asList("async", "gang", "worker",
		"vector", "wait"));
	
	//List used to set print orders.
	//Clauses not listed here may be printed in a random order.
	private static final List<String> print_order =
			new ArrayList<String>(Arrays.asList( "parallel", "kernels", "enter", "exit", "mpi",
					"data", "loop", "declare", "update", "atomic", "host_data", "cuda", "internal", "tempinternal",
					"routine", "if", "async", "refname", "num_gang", "num_workers", "vector_length", 
					"collapse", "gang", "worker", "vector", "seq", "independent", "tile",
					"reduction", "copy", "copyin", "copyout", "create", "present", "pcopy", "pcopyin",
					"pcopyout", "pcreate", "pipe", "pipein", "pipeout", "deviceptr", "device_resident", 
					"private", "firstprivate", "finalize", "sendbuf", "recvbuf"
					));
	
	// List of OpenACC directives.
	public static final Set<String> OpenACCDirectiveSet = new HashSet(Arrays.asList("parallel", "kernels", "loop", "data", "host_data",
			"declare", "update", "cache", "wait", "routine", "enter", "exit", "barrier", "set"));
	
	// List of OpenACC directives with optional if clauses
	public static final Set<String> OpenACCDirectivesWithConditional = new HashSet(Arrays.asList("parallel",
			"kernels", "data", "enter", "exit", "update"));
	
	// List of OpenACC directives with optional wait clauses
	public static final Set<String> OpenACCDirectivesWithWait = new HashSet(Arrays.asList("parallel",
			"kernels", "enter", "exit", "update"));
	
	// Directives attached to structured blocks
	public static final Set<String> directivesForStructuredBlock = new HashSet(Arrays.asList("parallel", "kernels", "data", "host_data" )); 
	
	// Directives to specify compute regions
	public static final Set<String> computeRegions = new HashSet(Arrays.asList("parallel", "kernels"));
	
	// Directives that specify explicit/implicit data regions
	public static final Set<String> dataRegions = new HashSet(Arrays.asList("data", "parallel", "kernels"));
	
	// Directives that contains data clauses.
	public static final Set<String> dataDirectives = new HashSet(Arrays.asList("data", "declare", "parallel", "kernels"));
	
	// List of OpenACC clauses
	public static final Set<String> OpenACCClauseSet = new HashSet(Arrays.asList(
	"if", "async", "num_gangs", "num_workers", "vector_length", "reduction", "copy", "copyin", "copyout",
	"create", "present", "present_or_copy", "pcopy", "present_or_copyin", "pcopyin", "present_or_copyout",
	"pcopyout", "present_or_create", "pcreate", "deviceptr", "device_resident", "host", "device", "private",
	"pipe", "pipein", "pipeout",
	"firstprivate", "use_device", "collapse", "gang", "worker", "vector", "seq", "independent", "bind",
	"nohost", "nowait", "type", "tile", "default_async", "device_num", "device_type", "finalize", "delete"
	));
	
	// Clauses that specify worksharing loops
	public static final Set<String> worksharingClauses = new HashSet(Arrays.asList("gang", "worker", "vector"));
	
	// Clauses that specify dimensions of parallel region
	public static final Set<String> parallelDimensionClauses = new HashSet(Arrays.asList("num_gangs", "num_workers", "vector_length"));
	
	// Clauses that specify parallelizable worksharing loops
	public static final Set<String> parallelWorksharingClauses = new HashSet(Arrays.asList("gang", "worker"));
	
	// Clauses that specify private/firstprivate variables
	public static final Set<String> privateClauses = new HashSet(Arrays.asList("private", "firstprivate"));
	
	// Clauses that have a list as arguments.
	public static final Set<String> collectionClauses = collection_values;
	
	// Data clauses
	public static final Set<String> dataClauses = new HashSet(Arrays.asList("copy", "copyin", "copyout",
			"create", "present", "pcopy", "pcopyin", "pcopyout", "pcreate", "deviceptr", "device_resident",
			"pipe", "pipein", "pipeout", "delete"));
	
	public static final Set<String> noMemTrDataClauses = new HashSet(Arrays.asList("create", "present", "pcreate",
			"deviceptr", "device_resident", "pipe", "pipein", "pipeout", "delete"));
	
	public static final Set<String> memTrDataClauses = new HashSet(Arrays.asList("copy", "copyin", "copyout",
			"pcopy", "pcopyin", "pcopyout" ));
	
	public static final Set<String> memTrUpdateClauses = new HashSet(Arrays.asList("host", "device"));

	public static final Set<String> pipeClauses = new HashSet(Arrays.asList("pipe", "pipein", "pipeout"));

	public static final Set<String> pipeIOClauses = new HashSet(Arrays.asList("pipein", "pipeout"));
	
	public static final Set<String> internalDataClauses =
		new HashSet<String>(Arrays.asList("accglobal", "accshared", "accprivate",
		"accreduction", "accdeviceptr", "accexplicitshared", "accreadonly", "rcreate"));
	
	public static final Set<String> internalConfigClauses =
		new HashSet<String>(Arrays.asList("iterspace", "gangdim", "workerdim",
		"gangconf", "workerconf", "totalnumgangs", "totalnumworkers"		));
	
    /**
    * Returns a clone of this annotation object.

	/**
	 * Constructs an empty OpenACC annotation.
	 */
	public ACCAnnotation()
	{
		super("acc");
	}

	/**
	 * Constructs an OpenACC annotation with the given key-value pair.
	 */
	public ACCAnnotation(String key, Object value)
	{
		super("acc");
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
		if( containsKey("loop") ) {
			if( !(at instanceof Loop) ) {
				isValid = false;
			}
		} else if( containsKey("declare") || containsKey("update") || containsKey("cache") ||
				containsKey("barrier") || containsKey("enter") ||
				containsKey("exit") || containsKey("set") ) {
			//Stand-alone directives
			//acc profile measure is a stand-alone directive.
			//enter/exit data and enter/exit profile are stand-alone directives.
			isValid = false;
		} else if( containsKey("wait") && get("wait").equals("_directive") ) {
			//acc wait directive.
			isValid = false;
		} else if( containsKey("parallel") || containsKey("kernels") || 
				containsKey("data") || containsKey("host_data") ) { 
			if( (at instanceof DeclarationStatement) || (at instanceof AnnotationStatement) ) {
				isValid = false;
			}
		} else if( containsKey("routine") ) {
			//acc routine is applicable to either procedure declaration (VariableDeclaration) or definition (Procedure).
			if( !(at instanceof Procedure) && !(at instanceof VariableDeclaration) ) {
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
			if ( key.equals("reduction") ) {
				Map<ReductionOperator, Collection> reduction_map = get(key);
				boolean firstOp = true;
				for (ReductionOperator op : reduction_map.keySet()) {
					if( !firstOp ) {
						str.append(") "+"reduction(");
						firstOp = false;
					}
					str.append(op.toString());
					str.append(": ");
					str.append(PrintTools.collectionToString(
							reduction_map.get(op), ", "));
				}
			} else {
				Object value = get(key);
				if ( value instanceof Collection )
					if( value instanceof List ) {
						str.append(PrintTools.listToString((List)value, ", "));
					} else {
						str.append(PrintTools.collectionToString((Collection)value, ", "));
					}
				else // e.g., num_gangs
					str.append(value);
			}
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
