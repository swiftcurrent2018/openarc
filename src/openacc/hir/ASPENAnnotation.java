package openacc.hir;

import java.util.*;

import cetus.hir.*;
import openacc.hir.*;
import openacc.analysis.SubArray;

/**
 * ASPENAnnotation is used for internally representing ASPEN pragmas.
 * ASPEN pragmas are raw text right after parsing 
 * but converted to an internal annotation of this type.
 * 
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future TechnologiesGroup, Oak Ridge National Laboratory
 */
public class ASPENAnnotation extends PragmaAnnotation
{
/**
 * ASPEN Annotation List.
 *
 * The following user directives are supported:
 * <p>     
 * #pragma aspen modelregion [label(string-name)]
 * <p>     
 * #pragma aspen enter modelregion [label(string-name)]
 * <p>     
 * #pragma aspen exit modelregion [label(string-name)]
 * <p>     
 * #pragma aspen declare [clause[[,] clause]...]
 * <p>     
 * where clause is one of the following
 * param(param-arg-list) //param(n:1000, ntimes:0)
 * data(data-arg-list) //data(matA:traits(Matrix(n, n, wordSize)),
 * 						//matB:traits(Array(n, wordSize))
 * <p>     
 * where param-arg is one of the following:
 * identifier[:init-exp]
 * <p>     
 * where data-arg is one of the following:
 * identifier:traits(trait-list)
 * <p>     
 * #pragma aspen control [clause[[,] clause]...]
 * <p>     
 * where clause is one of the following
 * loop [(itr-size)]
 * if (cond-exp-list)
 * probability (prob-exp-list)
 * ignore
 * parallelism (para-arg])  //parallelClause
 * execute
 * label (string-name)
 * flops (flops-arg-list)
 * loads (loads-arg-list)
 * stores (stores-arg-list)
 * messages (messages-arg-list)
 * intracomm (messages-arg-list)
 * allocates (memory-arg-list)
 * resizes (memory-arg-list)
 * frees (memory-arg-list)
 * <p>     
 * where para-arg is one of the following:
 * exp[:traits(trait-list)]						//ASPENResource
 * <p>     
 * where flops-arg is one of the following:
 * size-exp[:traits(trait-list)]				//ASPENResource
 * <p>     
 * where loads-arg is one of the following:
 * size-exp[:from(ID)][:traits(trait-list)]		//ASPENResource
 * <p>     
 * where stores-arg is one of the following:
 * size-exp[:to(ID)][:traits(trait-list)]		//ASPENResource
 * <p>     
 * where message-arg is one of the following:
 * size-exp[:from(ID)][:traits(trait-list)]		//ASPENResource
 * <p>     
 * where memory-arg is one of the following:
 * identifier:size(exp)[:traits(trait-list)]	//ASPENData
 * 
 */

	// ASPEN directives
	public static final Set<String> aspen_directives =
		new HashSet<String>(Arrays.asList("declare", "control", "enter", "exit", "modelregion"));

	// Pragmas used without values
	private static final Set<String> no_values =
		new HashSet<String>(Arrays.asList("enter", "exit", "modelregion", "declare", "control", "ignore", 
			"execute")); 

	// Pragmas used with collection of values
	private static final Set<String> collection_values =
		new HashSet<String>(Arrays.asList("param", "data", "flops", "if", "probability",
		"loads", "stores", "messages", "intracomm", "allocates", "resizes", "frees"));

	// Pragmas used with optional value
	private static final Set<String> optional_values = 
		new HashSet<String>(Arrays.asList("loop"));
	
	//List used to set print orders.
	//Clauses not listed here may be printed in a random order.
	private static final List<String> print_order =
			new ArrayList<String>(Arrays.asList( "enter", "exit", "modelregion", "declare", "control", "execute",
					"label", "data", "param", "loop", "if", "probability",
					"parallelism", "flops", "loads", "stores", "messages",
					"intracomm", "allocates", 
					"resizes", "frees", "ignore"));
	
	// Directives attached to structured blocks
	public static final Set<String> directivesForStructuredBlock = new HashSet(Arrays.asList("control", "modelregion"));
	
	// Clauses that have a list as arguments.
	public static final Set<String> collectionClauses = collection_values;
	
	// clauses that should be printed in the order as specified in the input program.
	private static final Set<String> inOrderClauses = new HashSet();

    /**
    * Returns a clone of this annotation object.

	/**
	 * Constructs an empty ASPEN annotation.
	 */
	public ASPENAnnotation()
	{
		super("aspen");
	}

	/**
	 * Constructs an ASPEN annotation with the given key-value pair.
	 */
	public ASPENAnnotation(String key, Object value)
	{
		super("aspen");
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
		if( containsKey("enter") || containsKey("exit") || containsKey("declare") ) {
			//Stand-alone directives
			isValid = false;
		}
		return isValid;
	}
	

	/**
	 * Returns the string representation of this annotation.
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
			else // e.g., label
				str.append(value);
			str.append(")");
		}
		else if ( optional_values.contains(key) )
		{
			str.append(" "+key);
			Object tVal = get(key);
			if ( tVal != null && !tVal.equals("true") && !tVal.equals("_directive") 
					&& !tVal.equals("_clause") )
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
				str.append("("+tObj+")");
			}
		}
	}
}
