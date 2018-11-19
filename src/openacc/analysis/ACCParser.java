/**
 * 
 */
package openacc.analysis;

import java.io.*;
import java.util.*;
import java.lang.String;

import cetus.hir.*;
import openacc.hir.ACCAnnotation;
import openacc.hir.ASPENData;
import openacc.hir.ASPENParam;
import openacc.hir.ASPENResource;
import openacc.hir.ASPENTrait;
import openacc.hir.ReductionOperator;
import openacc.transforms.ACCAnnotationParser;

/**
 * OpenACC annotation parser
 * CAUTION: current version stores start/length expressions of subarrays in data clauses as SomeExpression, which
 * is not analyzable by internal analyses passes. For accurate analysis, these should be parsed further by using
 * extra expression parser.
 * 
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group, Oak Ridge National Laboratory
 *
 */
public class ACCParser {
	
	private static String [] token_array;
	private static int token_index;
	private	static HashMap acc_map; // Used (key, value) mapping types:
									// - For ACCAnnotations and user-directives:
									//		(String, String), (String, Expression), (String, Set<SubArray>)
									//		(String, Map<ReductionOperator, Set<SubArray>>)
									// - For Tuning configurations:
									//		(String, Set<String>), (String, Set<Expression>), (Set, Expression)
	private int debug_level;
	private static final int infiniteLoopCheckTh = 1024; //used to detect possible infinite loop due to incorrect parsing.
	
	private static HashMap<String, String> macroMap = null;
	
	public static class ExpressionParser {
		static private String expr;
		static private int expr_pos;
		static private char expr_c;
		static private String token;
		
		public ExpressionParser() 
		{
			expr = "";
			expr_pos = -1;
			expr_c = '\0';
		}
		
	    /** 
	     * checks if the given char c is a letter or undersquare
	     */
	    static boolean isAlpha(final char c)
	    {   
	        char cUpper = Character.toUpperCase(c);
	        return "ABCDEFGHIJKLMNOPQRSTUVWXYZ_".indexOf(cUpper) != -1; 
	    }   
	    /** 
	     * checks if the given char c is a digit or dot
	     */
	    static boolean isNumericDot(final char c)
	    {   
	        return "0123456789.".indexOf(c) != -1; 
	    }   

	    /** 
	     * checks if the given char c is a digit
	     */
	    static boolean isNumeric(final char c)
	    {   
	        return "0123456789".indexOf(c) != -1; 
	    }   
	    
	    /** 
	     * checks if the given char c is a float suffix
	     */
	    static boolean isFloatSuffix(final char c)
	    {   
	        return "fFlL".indexOf(c) != -1; 
	    }   
	    
	    /** 
	     * checks if the given char c is an integer suffix
	     */
	    static boolean isIntegerSuffix(final char c)
	    {   
	        return "uUlL".indexOf(c) != -1; 
	    }   
	    
	    /** 
	     * checks if the given char c is an exponent
	     */
	    static boolean isExponent(final char c)
	    {   
	        return "eE".indexOf(c) != -1; 
	    }   
	    
	    /** 
	     * checks if the given char c is a unary operator 
	     */
	    static boolean isUnaryOp(final char c)
	    {   
	        return "+-".indexOf(c) != -1; 
	    }   
	    
	    /**
	     * Get the next character from the expression.
	     * The character is stored into the char expr_c.
	     * If the end of the expression is reached, the function puts zero ('\0')
	     * in expr_c.
	     */
	    static private void getChar()
	    {
	          expr_pos++;
	          if (expr_pos < expr.length())
	          {
	              expr_c = expr.charAt(expr_pos);
	          }
	          else
	          {
	              expr_c = '\0';
	          }
	      }
	    
	    static private char lookahead(int offset)
	    {
	    	  char retChar;
	          if ((expr_pos+offset) < expr.length())
	          {
	              retChar = expr.charAt(expr_pos+offset);
	          }
	          else
	          {
	              retChar = '\0';
	          }
	          return retChar;
	      }

	    /**
	     * Current parser simply checks whether input string <b>new_expr</b> is integer literal, 
	     * a variable name, or else.
	     * 
	     * @param new_expr input string to be parsed
	     * @return Expression which is either IntegerLiteral, NameID, or SomeExpression type.
	     */
	    public static Expression parse (final String new_expr)
	    {
	    	Expression parsedExpr = null;
	    	boolean containDigits = false;
	    	boolean containDot = false;
	    	boolean containLetters = false;
	    	boolean containFloatSuffix = false;
	    	boolean containIntSuffix = false;
	    	boolean containExponent = false;
	    	boolean containUnaryOp = false;
	    	boolean containNonAlphaNumeric = false;
	    	boolean startWithLetter = false;
	    	if( (new_expr == null) || (new_expr.length() == 0) ) {
	    		return parsedExpr;
	    	}
	    	//System.err.println("parse input expression: " + new_expr);
	    	expr = new_expr;
	    	expr_pos = -1;
	    	expr_c = '\0';
	    	getChar();
	    	boolean isFirstChar = true;
	    	while( expr_c != '\0' ) {
	    		if( isNumeric(expr_c) ) {
	    			containDigits = true;
	    		} else if( expr_c == '.' ) {
	    			containDot = true;
	    		} else if( isFirstChar ) {
	    			if( isUnaryOp(expr_c) ) {
	    				containUnaryOp = true;
	    			} else if( isAlpha(expr_c) ) {
	    				startWithLetter = true;
	    				containLetters = true;
	    			} else {
	    				containNonAlphaNumeric = true;
	    				break;
	    			}
	    		} else {
	    			if( !containLetters && containDigits ) {
	    				if( isFloatSuffix(expr_c) ) {
	    					containFloatSuffix = true;
	    				} else if( isIntegerSuffix(expr_c) ) {
	    					containIntSuffix = true;
	    				} else if( isExponent(expr_c) && !containNonAlphaNumeric ) {
	    					if( isUnaryOp(lookahead(1)) && isNumeric(lookahead(2)) ) {
	    						getChar();
	    						containExponent = true;
	    					} else if( isNumeric(lookahead(1)) ) {
	    						containExponent = true;
	    					} else {
	    						containNonAlphaNumeric = true;
	    						break;
	    					}
	    				} else if( isAlpha(expr_c) ) {
	    					containLetters = true;
	    				} else {
	    					containNonAlphaNumeric = true;
	    					break;
	    				}
	    			} else if( isAlpha(expr_c))  {
	    				containLetters = true;
	    			} else {
	    				containNonAlphaNumeric = true;
	    				break;
	    			}
	    		}
	    		getChar();
	    		isFirstChar = false;
	    	}
	    	if( containDigits && !containLetters && !containNonAlphaNumeric ) {
	    		if( containDot ) {
	    			//this string is FloatLiteral
	    			Double dVal = new Double(expr);
	    			if( expr.contains("f") || expr.contains("F") ) {
	    				parsedExpr = new FloatLiteral(dVal.doubleValue(), "F");
	    			} else {
	    				parsedExpr = new FloatLiteral(dVal.doubleValue());
	    			}
	    		} else {
	    			//this string is IntegerLiteral
	    			parsedExpr = new IntegerLiteral(expr);
	    		}
	    	} else if( !containNonAlphaNumeric && !containDot  && !containUnaryOp ) {
	    		//this string is a simple variable
	    		parsedExpr = new NameID(expr);
	    	} else {
	    		//general expression
	    		List<Traversable> children = new ArrayList<Traversable>();
	    		//DEBUG: below does not print parenthesis for SomeExpression.
	    		//parsedExpr = new SomeExpression(expr, children);
	    		//parsedExpr.setParens(true);
	    		//System.err.println("ParserTools.strToExpression( " + expr + ")");

	    		try {
	    			parsedExpr = ParserTools.strToExpression(expr);
	    			if(parsedExpr == null)
	    			{
	    				parsedExpr = new SomeExpression("("+expr+")", children);
	    				PrintTools.println("[INFO in ACCParser] SomeExpression found (case 1): " + parsedExpr, 2);
	    			}
	    		} catch( Exception e) {
	    			parsedExpr = new SomeExpression("("+expr+")", children);
	    			PrintTools.println("[INFO in ACCParser] SomeExpression found (case 2): " + parsedExpr, 2);
	    		}
	    		
	    	}
	    	return parsedExpr;
	    }
	}

	public ACCParser() {
	}

	private static String get_token()
	{
		if( end_of_token() ) {
			ACCParserError("parsing ended unexpectedly.");
			return null;
		} else {
			String tok = token_array[token_index++];
			if( macroMap.containsKey(tok) ) {
				String macroV = macroMap.get(tok);
				String[] new_token_array = macroV.split("\\s+");
				tok = new_token_array[0];
				if( new_token_array.length > 1 ) {
					String[] result = Arrays.copyOf(token_array, token_array.length + new_token_array.length - 1);
					System.arraycopy(new_token_array, 0, result, token_index-1, new_token_array.length);
					System.arraycopy(token_array, token_index, result, token_index-1+new_token_array.length, token_array.length-token_index);
					token_array = result;
				}
				return tok;
			} else {
				return tok;
			}
		}
	}
	
	// get one token but do not consume the token
	private static String lookahead()
	{
		if( end_of_token() ) {
			return "";
		} else {
			String tok = token_array[token_index];
			if( macroMap.containsKey(tok) ) {
				String macroV = macroMap.get(tok);
				String[] new_token_array = macroV.split("\\s+");
				tok = new_token_array[0];
				if( new_token_array.length > 1 ) {
					String[] result = Arrays.copyOf(token_array, token_array.length + new_token_array.length - 1);
					System.arraycopy(new_token_array, 0, result, token_index, new_token_array.length);
					System.arraycopy(token_array, token_index+1, result, token_index+new_token_array.length, token_array.length-token_index-1);
					token_array = result;
				}
				return tok;
			} else {
				return tok;
			}
		}
	}
	
	// get one token but do not consume the token
	private static String lookahead_debug()
	{
		if( end_of_token() ) {
			return "";
		} else {
			String tok = token_array[token_index];
			System.err.println("current token: " + tok);
			if( macroMap.containsKey(tok) ) {
				String macroV = macroMap.get(tok);
				System.err.println("replaced token: " + macroV);
				String[] new_token_array = macroV.split("\\s+");
				tok = new_token_array[0];
				if( new_token_array.length > 1 ) {
					System.err.print("replaced token array:");
					for( int i=0; i<new_token_array.length; i++ ) {
						System.err.print(" " + new_token_array[i]);
					}
					System.err.println("\ntoken length: " + new_token_array.length);
					String[] result = Arrays.copyOf(token_array, token_array.length + new_token_array.length - 1);
					System.arraycopy(new_token_array, 0, result, token_index, new_token_array.length);
					System.arraycopy(token_array, token_index+1, result, token_index+new_token_array.length, token_array.length-token_index-1);
					token_array = result;
				}
				System.err.println("token to return: " + tok);
				return tok;
			} else {
				System.err.println("token to return: " + tok);
				return tok;
			}
		}
	}

	// consume one token
	private static void eat()
	{
		token_index++;
	}

	// match a token with the given string
	private static boolean match(String istr)
	{
		boolean answer = check(istr);
		if (answer == false) {
			System.out.println("ACCParser Syntax Error: " + istr + " is expected");
			System.out.println("    Current token: " + lookahead());
			System.out.println("    Token index: " + token_index);
			System.out.println(display_tokens());
/*			System.out.println("\nmacroMap\n");
			for( String key : macroMap.keySet() ) {
				System.out.println("key: " + key + ", value: " + macroMap.get(key));
			}*/
			System.exit(0);
		}
		token_index++;
		return answer;
	}

	// match a token with the given string, but do not consume a token
	private static boolean check(String istr)
	{
		if ( end_of_token() ) 
			return false;
		String tok = token_array[token_index];
		if( macroMap.containsKey(tok) ) {
			String macroV = macroMap.get(tok);
			String[] new_token_array = macroV.split("\\s+");
			tok = new_token_array[0];
			if( new_token_array.length > 1 ) {
				String[] result = Arrays.copyOf(token_array, token_array.length + new_token_array.length - 1);
				System.arraycopy(new_token_array, 0, result, token_index, new_token_array.length);
				System.arraycopy(token_array, token_index+1, result, token_index+new_token_array.length, token_array.length-token_index-1);
				token_array = result;
			}
		}
		return ( tok.compareTo(istr) == 0 ) ? true : false;
	}	

	private static String display_tokens()
	{
		StringBuilder str = new StringBuilder(160);

		for (int i=0; i<token_array.length; i++)
		{
			str.append("token_array[" + i + "] = " + token_array[i] + "\n");
		}
		return str.toString();
	}

	private static boolean end_of_token()
	{
		return (token_index >= token_array.length) ? true : false;
	}
	
	public static void preprocess_acc_pragma( String[] str_array, HashMap<String, String> macro_map) {
		token_array = str_array;
		token_index = 2; // "openarc" and "#" have already been matched
		macroMap = macro_map;
		String tok = get_token();
		String construct = "arc_" + tok;
		try {
			switch (arc_preprocessor.valueOf(construct)) {
			case arc_define 		: parse_define_command(); break;
			default : ACCParserError("Not Supported Construct: " + tok);
			}
		} catch( Exception e) {
			ACCParserError("Not Supported Construct: " + tok);
		}
	}
	
	/*
	 * For now, only simple macro variable is allowed, but not function-style one.
	 * 
	 */
	private static void parse_define_command() {
		PrintTools.println("ACCParser is parsing [define] directive", 3);
		//String name = get_token(); //This will cause errors if the same macro is defined more than once.
		String name = token_array[token_index++];
		String value = "";
		if( !end_of_token() ) {
			value = get_token();
			while( !end_of_token() ) {
				value = value + " " + get_token();
			}
		}
		PrintTools.println("Macro name: " + name + ", value: " + value, 3);
		macroMap.put(name, value);
		macroMap.put("\\"+name, value);
		macroMap.put("\\{"+name+"}", value);
		
	}
	
	/**
	 * ACCAnnotation is used for internally representing OpenACC pragmas.
	 * OpenACC pragmas are raw text right after parsing 
	 * but converted to an internal annotation of this type.
	 * 
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
	 *      use_device( list )
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
	 *      nowait
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
	 *      pipe( list )
	 * <p>     
	 * #pragma acc update clause[[,] clause]...
	 * <p>     
	 * where clause is one of the following:
	 *      host( list )
	 *      device( list )
	 *      if( condition )
	 *      async [( scalar-integer-expression )]
	 * <p>     
	 * #pragma acc wait [( scalar-integer-expression )]
	 * <p>     
	 * <p>     
	 * #pragma acc enter data clause[[,] clause]...
	 * <p>     
	 * where clause is one of the following:
	 *      if( condition )
	 *      async [( scalar-integer-expression )]
	 *      wait [( scalar-integer-expression )]
	 *      copyin( list )
	 *      present_or_copyin( list )
	 *      create( list )
	 *      present_or_create( list )
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
	 *     gang
	 *     worker
	 *     vector
	 *     seq     
	 *     device_type(device-type-list)
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
	 * <p>
	 * #pragma openarc ainfo procname(proc-name) kernelid(kernel-id)
	 * <p>     
	 * #pragma openarc cuda [clause[[,] clause]...]
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
	 *      noreductionunroll(list)
	 *      noploopswap
	 *      noloopcollapse
	 *      multisrccg(list)
	 *      multisrcgc(list)
	 *      conditionalsrc(list)
	 *      enclosingloops(list)
	 *      accshared(list)
	 * <p>     
	 *		#pragma openarc resilience [clause [[,] clause] ...] new-line
	 *			structured block
	 * <p>
	 *		where clause is one of the following
	 *		ftregion
	 *     	ftcond(condition)
	 *     	ftdata(list)
	 *     	num_faults(scalar-integer-expression)
	 *     	num_ftbits(scalar-integer-expression)
	 *     	repeat(scalar-integer-expression)
	 *      ftthread(scalar-integer-expression)
	 * <p>
	 *		#pragma openarc ftregion [clause [[,] clause] ...] new-line
	 *			structured block
	 * <p>
	 *		where clause is one of the following
	 *     	ftdata(list)
	 *      ftthread(scalar-integer-expression)
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
	 * <p>
	 * #pragma openarc profile measure [clause [[,] clause] ...]
	 * <p>
	 * where clause is one of the following
	 * event(list) //a list consists of expressions
	 * induction(induction-expression) 
	 * profcond(expression) 
	 * label(name)
	 * 
	 */

    /**
     * Parse OpenACC pragmas, which are stored as raw text after Cetus parsing.
     * 
     * @param input_map HashMap that will contain the parsed output pragmas
     * @param str_array input pragma string that will be parsed.
     * @return true if the pragma will be attached to the following non-pragma
     *         statement. Or, it returns false, if the pragma is stand-alone.
     */
	public static boolean parse_acc_pragma(HashMap input_map, String [] str_array, HashMap<String, String>macro_map)
	{
		acc_map = input_map;
		token_array = str_array;
		token_index = 1; // "acc" has already been matched
		//token_index = 2; // If there is a leading space, use this one.
		macroMap = macro_map;

		PrintTools.println(display_tokens(), 9);

		String token = get_token();
		String construct = "acc_" + token;
		try {
			switch (acc_directives.valueOf(construct)) {
			case acc_parallel 		: parse_acc_parallel(); return true;
			case acc_kernels		: parse_acc_kernels(); return true;
			case acc_loop 		: parse_acc_loop(); return true;
			case acc_data	: parse_acc_data(); return true;
			case acc_host_data	: parse_acc_host_data(); return true;
			case acc_declare	: parse_acc_declare(); return false;
			case acc_update	: parse_acc_update(); return false;
			case acc_cache 		: parse_acc_cache(); return false;
			case acc_wait 		: parse_acc_wait(); return false;
			case acc_routine 		: parse_acc_routine(); return true;
			case acc_barrier 	: parse_acc_barrier(); return false;
			case acc_enter 	: parse_acc_enter(); return false;
			case acc_exit 	: parse_acc_exit(); return false;
			case acc_set 	: parse_acc_set(); return false;
			case acc_mpi 	: parse_acc_mpi(); return true;
			//		default : throw new NonOmpDirectiveException();
			default : ACCParserError("Not Supported Construct");
			}
		} catch( Exception e) {
			ACCParserError("unexpected or wrong token found (" + token + ")");
		}
		return true;		// meaningless return because it is unreachable
	}


	/**
	 * Standalone subarray-list parser, which takes a String containing comma-separated subarray list and 
	 * optional macro_map and returns Set<SubArray> as output.
	 * 
	 * @param inputStr string containing comma-seperated subarray list
	 * @param macro_map optional macro map
	 * @return Set<SubArray>
	 */
	public static Set<SubArray> parse_subarraylist(String inputStr, HashMap<String, String>macro_map)
	{
		String newStr = ACCAnnotationParser.modifyAnnotationString("(" + inputStr +")");
		token_array = newStr.split("\\s+");
		token_index = 1; //Skip leading space before '(' token. 
		if( macro_map == null ) {
			macroMap = new HashMap<String, String>();
		} else {
			macroMap = macro_map;
		}
		PrintTools.println(display_tokens(), 9);
		match("(");
		Set<SubArray> set = new HashSet<SubArray>();
		parse_commaSeparatedSubArrayList(set, 0);
		match(")");

		return set;
	}

	/**
	 * Standalone reduction parser, which takes an input HashMap, a String containing reduction clause and 
	 * optional macro_map and update the input HashMap with reduction mapping.
	 * 
     * @param input_map HashMap that will contain the parsed output pragmas
	 * @param inputStr string containing comma-seperated subarray list
	 * @param macro_map optional macro map
	 * @return Set<SubArray>
	 */
	public static void parse_reductionclause(HashMap input_map, String inputStr, HashMap<String, String>macro_map)
	{
		String newStr = ACCAnnotationParser.modifyAnnotationString("(" + inputStr +")");
		acc_map = input_map;
		token_array = newStr.split("\\s+");
		token_index = 1; //Skip leading space before '(' token. 
		if( macro_map == null ) {
			macroMap = new HashMap<String, String>();
		} else {
			macroMap = macro_map;
		}
		PrintTools.println(display_tokens(), 9);
		parse_acc_reduction("reduction");
	}
	
	private static void parse_acc_enter()
	{
		addToMap("enter", "_directive");
		String token = get_token();
		String construct = "acc_" + token;
		try {
			switch (acc_directives.valueOf(construct)) {
			case acc_data	: parse_acc_enter_data(); break;
			//		default : throw new NonOmpDirectiveException();
			default : ACCParserError("Not Supported Construct");
			}
		} catch( Exception e) {
			ACCParserError("unexpected or wrong token found (" + token + ")");
		}
	}
	
	private static void parse_acc_exit()
	{
		addToMap("exit", "_directive");
		String token = get_token();
		String construct = "acc_" + token;
		try {
			switch (acc_directives.valueOf(construct)) {
			case acc_data	: parse_acc_exit_data(); break;
			//		default : throw new NonOmpDirectiveException();
			default : ACCParserError("Not Supported Construct");
			}
		} catch( Exception e) {
			ACCParserError("unexpected or wrong token found (" + token + ")");
		}
	}
	
	private static void parse_arc_enter()
	{
		addToMap("enter", "_directive");
		String token = get_token();
		String construct = "arc_" + token;
		try {
			switch (arc_directives.valueOf(construct)) {
			case arc_profile 	: parse_arc_profile(); break;
			//		default : throw new NonOmpDirectiveException();
			default : ACCParserError("Not Supported Construct");
			}
		} catch( Exception e) {
			ACCParserError("unexpected or wrong token found (" + token + ")");
		}
	}
	
	private static void parse_arc_exit()
	{
		addToMap("exit", "_directive");
		String token = get_token();
		String construct = "arc_" + token;
		try {
			switch (arc_directives.valueOf(construct)) {
			case arc_profile 	: parse_arc_exit_profile(); break;
			//		default : throw new NonOmpDirectiveException();
			default : ACCParserError("Not Supported Construct");
			}
		} catch( Exception e) {
			ACCParserError("unexpected or wrong token found (" + token + ")");
		}
	}
	
	/** ---------------------------------------------------------------
	 * #pragma openarc profile region label(name) [clause[[,] clause]...]
	 * 		structured-block
	 * <p>
	 * where clause is one of the following
	 * mode(list)  //a list consists of the following:
	 *             //memory, instructions, occupancy, memorytransfer, all 
	 * verbosity(arg) //an arg is a non-negative integer where
	 *                //0 is the least verbose mode. 
	 * <p>
	 * 
	 * #pragma openarc enter profile region label(name) [clause[[,] clause]...]
	 * <p>
	 * where clause is one of the following
	 * mode(list)  //a list consists of the following:
	 *             //memory, instructions, occupancy, memorytransfer, all 
	 * verbosity(arg) //an arg is a non-negative integer where
	 *                //0 is the least verbose mode. 
	 * <p>               
	 * #pragma openarc profile track label(name) [clause [[,] clause] ...]
	 * 		structured-block
	 * <p>
	 * where clause is one of the following
	 * event(list) //a list consists of expressions
	 * induction(induction-expression) 
	 * profcond(expression) 
	 * mode(list)  //a list consists of the following:
	 *             //memory, instructions, occupancy, memorytransfer, all 
	 * <p>
	 * #pragma openarc profile measure label(name) [clause [[,] clause] ...]
	 * <p>
	 * where clause is one of the following
	 * event(list) //a list consists of expressions
	 * induction(induction-expression) 
	 * profcond(expression) 
	 * --------------------------------------------------------------- */
	private static boolean parse_arc_profile()
	{
		PrintTools.println("ACCParser is parsing [profile] directive", 3);
		addToMap("profile", "_directive");
		boolean attachedToNextAnnotatable = true;
		boolean labelExist = false;
		boolean subDirectiveExist = false;
		while (end_of_token() == false) 
		{
			String token = get_token();
			if( token.equals("") ) continue; //Skip empty string, which may occur due to macro.
			String clause = "token_" + token;
			if( token.equals(",") ) continue; //Skip comma between clauses, if existing.
			PrintTools.println("clause=" + clause, 3);
			try {
				switch (profile_clause.valueOf(clause)) {
				case token_region	:	parse_acc_noargclause(token); subDirectiveExist = true; break;
				case token_track	:	parse_arc_track(); subDirectiveExist = true; labelExist = true; break;
				case token_measure	:	parse_arc_measure(); subDirectiveExist = true; labelExist = true; attachedToNextAnnotatable = false; break;
				case token_label		:	parse_acc_stringargclause(token); labelExist = true; break;
				case token_mode		:	parse_conf_stringset(token); break;
				case token_event		:	parse_conf_expressionset(token); break;
				case token_verbosity		:	parse_acc_confclause(token); break;
				case token_profcond		:	parse_acc_confclause(token); break;
				default : ACCParserError("NoSuchProfileConstruct : " + clause);
				}
			} catch( Exception e) {
				ACCParserError("unexpected or wrong token found (" + token + ")");
			}
		}
		if( !subDirectiveExist ) {
			ACCParserError("Profile directive should have one sub-directive (region/track/measure)!");
		}
		if( !labelExist ) {
			ACCParserError("label clause is missing!");
		}
		return attachedToNextAnnotatable;
	}
	
	/** ---------------------------------------------------------------
     * #pragma openarc exit profile label(name) 
	 * --------------------------------------------------------------- */
	private static void parse_arc_exit_profile()
	{
		PrintTools.println("ACCParser is parsing [exit profile] directive", 3);
		addToMap("profile", "_directive");
		boolean labelExist = false;
		while (end_of_token() == false) 
		{
			String token = get_token();
			if( token.equals("") ) continue; //Skip empty string, which may occur due to macro.
			String clause = "token_" + token;
			if( token.equals(",") ) continue; //Skip comma between clauses, if existing.
			PrintTools.println("clause=" + clause, 3);
			try {
				switch (profile_clause.valueOf(clause)) {
				case token_region	:	parse_acc_noargclause(token); break;
				case token_label		:	parse_acc_stringargclause(token); labelExist = true; break;
				default : ACCParserError("NoSuchProfileConstruct : " + clause);
				}
			} catch( Exception e) {
				ACCParserError("unexpected or wrong token found (" + token + ")");
			}
		}
		if( !labelExist ) {
			ACCParserError("label clause is missing!");
		}
	}
	
	/** ---------------------------------------------------------------
     * #pragma openarc measure label(name) event(list) induction(induction-expression) if(expression)
	 * --------------------------------------------------------------- */
	private static void parse_arc_measure()
	{
		PrintTools.println("ACCParser is parsing [measure] directive", 3);
		addToMap("measure", "_directive");
		while (end_of_token() == false) 
		{
			String token = get_token();
			if( token.equals("") ) continue; //Skip empty string, which may occur due to macro.
			String clause = "token_" + token;
			if( token.equals(",") ) continue; //Skip comma between clauses, if existing.
			PrintTools.println("clause=" + clause, 3);
			try {
				switch (profile_clause.valueOf(clause)) {
				case token_event		:	parse_conf_expressionset(token); break;
				case token_induction		:	parse_acc_confclause(token); break;
				case token_profcond		:	parse_acc_confclause(token); break;
				case token_label		:	parse_acc_stringargclause(token); break;
				default : ACCParserError("NoSuchProfileConstruct : " + clause);
				}
			} catch( Exception e) {
				ACCParserError("unexpected or wrong token found (" + token + ")");
			}
		}
		
	}
	
	/** ---------------------------------------------------------------
     * #pragma openarc profile track label(name) event(list) induction(induction-expression) if(expression)
	 * --------------------------------------------------------------- */
	private static void parse_arc_track()
	{
		PrintTools.println("ACCParser is parsing [track] directive", 3);
		addToMap("track", "_directive");
		while (end_of_token() == false) 
		{
			String token = get_token();
			if( token.equals("") ) continue; //Skip empty string, which may occur due to macro.
			String clause = "token_" + token;
			if( token.equals(",") ) continue; //Skip comma between clauses, if existing.
			PrintTools.println("clause=" + clause, 3);
			try {
				switch (profile_clause.valueOf(clause)) {
				case token_event		:	parse_conf_expressionset(token); break;
				case token_induction		:	parse_acc_confclause(token); break;
				case token_profcond		:	parse_acc_confclause(token); break;
				case token_label		:	parse_acc_stringargclause(token); break;
				case token_mode		:	parse_conf_stringset(token); break;
				default : ACCParserError("NoSuchProfileConstruct : " + clause);
				}
			} catch( Exception e) {
				ACCParserError("unexpected or wrong token found (" + token + ")");
			}
		}
		
	}
	
	/** ---------------------------------------------------------------
	 *		OpenARC resilience Construct
	 *
	 *		#pragma openarc resilience [clause [[,] clause] ...] new-line
	 *			structured block
	 *
	 *		where clause is one of the following
	 *		ftregion
	 *     	ftcond(condition)
	 *     	ftdata(list)
	 *     	ftkind(list)
	 *     	num_faults(scalar-integer-expression)
	 *     	num_ftbits(scalar-integer-expression)
	 *     	repeat(scalar-integer-expression)
	 *      ftthread(scalar-integer-expression)
	 *
	 *
	 * --------------------------------------------------------------- */
	private static void parse_arc_resilience()
	{
		PrintTools.println("ACCParser is parsing [resilience] directive", 3);
		addToMap("resilience", "_directive");
		if( check("ftregion") ) {
			eat();
			addToMap("ftregion", "_directive");
		}
		while (end_of_token() == false) 
		{
			String token = get_token();
			if( token.equals("") ) continue; //Skip empty string, which may occur due to macro.
			String clause = "token_" + token;
			if( token.equals(",") ) continue; //Skip comma between clauses, if existing.
			PrintTools.println("clause=" + clause, 3);
			try {
				switch (resilience_clause.valueOf(clause)) {
				case token_ftcond		:	parse_acc_confclause(token); break;
				case token_ftdata		:	parse_acc_dataclause(token); break;
				case token_num_faults		: parse_acc_confclause(token); break;
				case token_num_ftbits		: parse_acc_confclause(token); break;
				case token_repeat		: parse_acc_confclause(token); break;
				case token_ftthread		: parse_acc_confclause(token); break;
				case token_ftkind		:	parse_conf_stringset(token); break;
				case token_ftprofile		:	parse_acc_confclause(token); break;
				case token_ftpredict		:	parse_acc_confclause(token); break;
				default : ACCParserError("NoSuchResilienceConstruct : " + clause);
				}
			} catch( Exception e) {
				ACCParserError("unexpected or wrong token found (" + token + ")");
			}
		}
	}
	
	/** ---------------------------------------------------------------
	 *		OpenARC ftregion Construct
	 *
	 *		#pragma openarc ftregion [clause [[,] clause] ...] new-line
	 *			structured block
	 *
	 *		where clause is one of the following
	 *     	ftdata(list)
	 *     	ftkind(list)
	 *      ftthread(scalar-integer-expression)
	 *
	 *
	 * --------------------------------------------------------------- */
	private static void parse_arc_ftregion()
	{
		PrintTools.println("ACCParser is parsing [ftregion] directive", 3);
		addToMap("ftregion", "_directive");
		while (end_of_token() == false) 
		{
			String token = get_token();
			if( token.equals("") ) continue; //Skip empty string, which may occur due to macro.
			String clause = "token_" + token;
			if( token.equals(",") ) continue; //Skip comma between clauses, if existing.
			PrintTools.println("clause=" + clause, 3);
			try {
				switch (resilience_clause.valueOf(clause)) {
				case token_ftdata		:	parse_acc_dataclause(token); break;
				case token_ftthread		: parse_acc_confclause(token); break;
				case token_ftkind		:	parse_conf_stringset(token); break;
				default : ACCParserError("NoSuchFtregionConstruct : " + clause);
				}
			} catch( Exception e) {
				ACCParserError("unexpected or wrong token found (" + token + ")");
			}
		}
	}
	
	/** ---------------------------------------------------------------
	 *		OpenARC ftinject Construct
	 *
	 *		#pragma openarc ftinject [clause [[,] clause] ...] new-line
	 *
	 *		where clause is one of the following
	 *     	ftdata(list)
	 *      ftthread(scalar-integer-expression)
	 *
	 *
	 * --------------------------------------------------------------- */
	private static void parse_arc_ftinject()
	{
		PrintTools.println("ACCParser is parsing [ftinject] directive", 3);
		addToMap("ftinject", "_directive");
		while (end_of_token() == false) 
		{
			String token = get_token();
			if( token.equals("") ) continue; //Skip empty string, which may occur due to macro.
			String clause = "token_" + token;
			if( token.equals(",") ) continue; //Skip comma between clauses, if existing.
			PrintTools.println("clause=" + clause, 3);
			try {
				switch (resilience_clause.valueOf(clause)) {
				case token_ftdata		:	parse_acc_dataclause(token); break;
				case token_ftthread		: parse_acc_confclause(token); break;
				default : ACCParserError("NoSuchFtinjectConstruct : " + clause);
				}
			} catch( Exception e) {
				ACCParserError("unexpected or wrong token found (" + token + ")");
			}
		}
	}
	
	/** ---------------------------------------------------------------
	 *		OpenACC barrier Construct
	 *
	 *		#pragma acc barrier new-line
	 *
	 * --------------------------------------------------------------- */
	private static void parse_acc_barrier()
	{
		//PrintTools.println("ACCParser is parsing [barrier] directive", 3);
		//addToMap("barrier", "_directive");
		parse_acc_directivewithoptionalstringarg("barrier");
	}
	
	/** ---------------------------------------------------------------
	 *		OpenACC routine Construct
	 *
	 *		#pragma acc routine [clause [[,] clause] ...] new-line
	 *			function declaration or definition
	 *
	 * --------------------------------------------------------------- */
	private static void parse_acc_routine()
	{
		PrintTools.println("ACCParser is parsing [routine] directive", 3);
		addToMap("routine", "_directive");

		while (end_of_token() == false) 
		{
			String token = get_token();
			if( token.equals("") ) continue; //Skip empty string, which may occur due to macro.
			String clause = "acc_" + token;
			if( token.equals(",") ) continue; //Skip comma between clauses, if existing.
			PrintTools.println("clause=" + clause, 3);
			try {
				switch (acc_clause.valueOf(clause)) {
				case acc_gang		: parse_acc_workshareconfclause(token); break;
				case acc_worker		: parse_acc_workshareconfclause(token); break;
				case acc_vector		: parse_acc_workshareconfclause(token); break;
				case acc_seq		: parse_acc_noargclause(token); break;
				case acc_bind		:	parse_acc_bindclause(token); break;
				case acc_nohost		: parse_acc_noargclause(token); break;
				case acc_device_type	:	parse_acc_confclause(token); break;
				default : ACCParserError("NoSuchRoutineConstruct : " + clause);
				}
			} catch( Exception e) {
				ACCParserError("unexpected or wrong token found (" + token + ")");
			}
		}
	}
	
	/** ---------------------------------------------------------------
	 *		OpenARC ainfo Construct
	 *
	 *		#pragma openarc ainfo procname(proc-name) kernelid(id) new-line
	 *			structured-block
	 *
	 * --------------------------------------------------------------- */
	private static void parse_arc_ainfo()
	{
		PrintTools.println("ACCParser is parsing [ainfo] directive", 3);
		addToMap("ainfo", "_directive");

		while (end_of_token() == false) 
		{
			String token = get_token();
			if( token.equals("") ) continue; //Skip empty string, which may occur due to macro.
			String clause = "token_" + token;
			if( token.equals(",") ) continue; //Skip comma between clauses, if existing.
			PrintTools.println("clause=" + clause, 3);
			try {
				switch (ainfo_clause.valueOf(clause)) {
				case token_procname		:	parse_acc_stringargclause(token); break;
				case token_kernelid		:	parse_acc_confclause(token); break;
				default : ACCParserError("NoSuchAinfoConstruct : " + clause);
				}
			} catch( Exception e) {
				ACCParserError("unexpected or wrong token found (" + token + ")");
			}
		}
	}

	/** ---------------------------------------------------------------
	 *		OpenACC parallel Construct
	 *
	 *		#pragma acc parallel [clause[[,] clause]...] new-line
	 *			structured-block
	 *
	 *		where clause is one of the following
	 *      if( condition )
	 *      async [( scalar-integer-expression )]
	 *      wait [( scalar-integer-expression-list )]
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
	 * --------------------------------------------------------------- */
	private static void parse_acc_parallel()
	{
		addToMap("parallel", "_directive");
		if (check("loop")) {
			eat();
			parse_acc_parallel_loop();
		} else {
			PrintTools.println("ACCParser is parsing [parallel] directive", 3);
			while (end_of_token() == false) 
			{
				String tok = get_token();
				if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
				if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
				String clause = "acc_" + tok;
				PrintTools.println("clause=" + clause, 3);
				try {
					switch (acc_clause.valueOf(clause)) {
					case acc_if		:	parse_acc_confclause(tok); break;
					case acc_async	:	parse_acc_optionalconfclause(tok); break;
					case acc_wait	:	parse_acc_optionalconflistclause(tok); break;
					case acc_num_gangs		:	parse_acc_confclause(tok); break;
					case acc_num_workers	:	parse_acc_confclause(tok); break;
					case acc_vector_length		:	parse_acc_confclause(tok); break;
					case acc_reduction		:	parse_acc_reduction(tok); break;
					case acc_copy		:	parse_acc_dataclause(tok); break;
					case acc_copyin 		:	parse_acc_dataclause(tok); break;
					case acc_copyout 		:	parse_acc_dataclause(tok); break;
					case acc_create 		:	parse_acc_dataclause(tok); break;
					case acc_present 		:	parse_acc_dataclause(tok); break;
					case acc_present_or_copy 		:	parse_acc_dataclause("pcopy"); break;
					case acc_pcopy		:	parse_acc_dataclause(tok); break;
					case acc_present_or_copyin		:	parse_acc_dataclause("pcopyin"); break;
					case acc_pcopyin		:	parse_acc_dataclause(tok); break;
					case acc_present_or_copyout	:	parse_acc_dataclause("pcopyout"); break;
					case acc_pcopyout	:	parse_acc_dataclause(tok); break;
					case acc_present_or_create	:	parse_acc_dataclause("pcreate"); break;
					case acc_pcreate	:	parse_acc_dataclause(tok); break;
					case acc_deviceptr	:	parse_acc_dataclause(tok); break;
					case acc_private	:	parse_acc_dataclause(tok); break;
					case acc_firstprivate	:	parse_acc_dataclause(tok); break;
					case acc_pipein 		:	parse_acc_dataclause(tok); break;
					case acc_pipeout 		:	parse_acc_dataclause(tok); break;
					default : ACCParserError("NoSuchOpenACCConstruct : " + clause);
					}
				} catch( Exception e) {
					ACCParserError("unexpected or wrong token found (" + tok + ")");
				}
			}
		}
	}
	
	/** ---------------------------------------------------------------
	 *		OpenACC parallel loop Construct
	 *
	 *		#pragma acc parallel loop [clause[[,] clause]...] new-line
	 *			structured-block
	 *
	 *		where clause is one of the following
	 *      if( condition )
	 *      async [( scalar-integer-expression )]
	 *      wait [( scalar-integer-expression-list )]
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
	 * 		pipein( list ) 
	 * 		pipeout( list ) 
	 * 		collapse( n )
	 * 		gang
	 * 		worker
	 * 		vector
	 * 		seq
	 * 		independent
	 * --------------------------------------------------------------- */
	private static void parse_acc_parallel_loop()
	{
		PrintTools.println("ACCParser is parsing [parallel loop] directive", 3);
		addToMap("loop", "_directive");
		while (end_of_token() == false) 
		{
			String tok = get_token();
			if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
			if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
			String clause = "acc_" + tok;
			PrintTools.println("clause=" + clause, 3);
			try {
				switch (acc_clause.valueOf(clause)) {
				case acc_if		:	parse_acc_confclause(tok); break;
				case acc_async	:	parse_acc_optionalconfclause(tok); break;
				case acc_wait	:	parse_acc_optionalconflistclause(tok); break;
				case acc_num_gangs		:	parse_acc_confclause(tok); break;
				case acc_num_workers	:	parse_acc_confclause(tok); break;
				case acc_vector_length		:	parse_acc_confclause(tok); break;
				case acc_reduction		:	parse_acc_reduction(tok); break;
				case acc_copy		:	parse_acc_dataclause(tok); break;
				case acc_copyin 		:	parse_acc_dataclause(tok); break;
				case acc_copyout 		:	parse_acc_dataclause(tok); break;
				case acc_create 		:	parse_acc_dataclause(tok); break;
				case acc_present 		:	parse_acc_dataclause(tok); break;
				case acc_present_or_copy 		:	parse_acc_dataclause("pcopy"); break;
				case acc_pcopy		:	parse_acc_dataclause(tok); break;
				case acc_present_or_copyin		:	parse_acc_dataclause("pcopyin"); break;
				case acc_pcopyin		:	parse_acc_dataclause(tok); break;
				case acc_present_or_copyout	:	parse_acc_dataclause("pcopyout"); break;
				case acc_pcopyout	:	parse_acc_dataclause(tok); break;
				case acc_present_or_create	:	parse_acc_dataclause("pcreate"); break;
				case acc_pcreate	:	parse_acc_dataclause(tok); break;
				case acc_deviceptr	:	parse_acc_dataclause(tok); break;
				case acc_private	:	parse_acc_dataclause(tok); break;
				case acc_firstprivate	:	parse_acc_dataclause(tok); break;
				case acc_collapse		: parse_acc_confclause(tok); break;
				case acc_gang		: parse_acc_workshareconfclause(tok); break;
				case acc_worker		: parse_acc_workshareconfclause(tok); break;
				case acc_vector		: parse_acc_workshareconfclause(tok); break;
				case acc_seq		: parse_acc_noargclause(tok); break;
				case acc_independent		: parse_acc_noargclause(tok); break;
                case acc_tile		: parse_expressionlist(tok); break;
				case acc_pipein 		:	parse_acc_dataclause(tok); break;
				case acc_pipeout 		:	parse_acc_dataclause(tok); break;
				default : ACCParserError("NoSuchOpenACCConstruct : " + clause);
				}
			} catch( Exception e) {
				ACCParserError("unexpected or wrong token found (" + tok + ")\nError message:" + e);
			}
		}
	}
	
	/** ---------------------------------------------------------------
	 *		OpenACC kernels Construct
	 *
	 *		#pragma acc kernels [clause[[,] clause]...] new-line
	 *			structured-block
	 *
	 *		where clause is one of the following
	 *      if( condition )
	 *      async [( scalar-integer-expression )]
	 *      wait [( scalar-integer-expression-list )]
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
	 * 		pipein( list ) 
	 * 		pipeout( list ) 
	 * --------------------------------------------------------------- */
	private static void parse_acc_kernels()
	{
		addToMap("kernels", "_directive");
		if (check("loop")) {
			eat();
			parse_acc_kernels_loop();
		} else {
			PrintTools.println("ACCParser is parsing [kernels] directive", 3);
			while (end_of_token() == false) 
			{
				String tok = get_token();
				if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
				if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
				String clause = "acc_" + tok;
				PrintTools.println("clause=" + clause, 3);
				try {
					switch (acc_clause.valueOf(clause)) {
					case acc_if		:	parse_acc_confclause(tok); break;
					case acc_async	:	parse_acc_optionalconfclause(tok); break;
					case acc_wait	:	parse_acc_optionalconflistclause(tok); break;
					case acc_copy		:	parse_acc_dataclause(tok); break;
					case acc_copyin 		:	parse_acc_dataclause(tok); break;
					case acc_copyout 		:	parse_acc_dataclause(tok); break;
					case acc_create 		:	parse_acc_dataclause(tok); break;
					case acc_present 		:	parse_acc_dataclause(tok); break;
					case acc_present_or_copy 		:	parse_acc_dataclause("pcopy"); break;
					case acc_pcopy		:	parse_acc_dataclause(tok); break;
					case acc_present_or_copyin		:	parse_acc_dataclause("pcopyin"); break;
					case acc_pcopyin		:	parse_acc_dataclause(tok); break;
					case acc_present_or_copyout	:	parse_acc_dataclause("pcopyout"); break;
					case acc_pcopyout	:	parse_acc_dataclause(tok); break;
					case acc_present_or_create	:	parse_acc_dataclause("pcreate"); break;
					case acc_pcreate	:	parse_acc_dataclause(tok); break;
					case acc_deviceptr	:	parse_acc_dataclause(tok); break;
					case acc_pipein 		:	parse_acc_dataclause(tok); break;
					case acc_pipeout 		:	parse_acc_dataclause(tok); break;
					default : ACCParserError("NoSuchOpenACCConstruct : " + clause);
					}
				} catch( Exception e) {
					ACCParserError("unexpected or wrong token found (" + tok + ")");
				}
			}
		}
	}
	
	/** ---------------------------------------------------------------
	 *		OpenACC kernels loop Construct
	 *
	 *		#pragma acc kernels loop [clause[[,] clause]...] new-line
	 *			structured-block
	 *
	 *		where clause is one of the following
	 *      if( condition )
	 *      async [( scalar-integer-expression )]
	 *      wait [( scalar-integer-expression-list )]
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
	 * 		pipein( list ) 
	 * 		pipeout( list ) 
	 * 		collapse( n )
	 * 		gang [( scalar-integer-expression )]
	 * 		worker [( scalar-integer-expression )]
	 * 		vector [( scalar-integer-expression )]
	 * 		seq
	 * 		independent
	 *      reduction( operator:list )
	 * 		private( list ) 
	 * --------------------------------------------------------------- */
	private static void parse_acc_kernels_loop()
	{
		PrintTools.println("ACCParser is parsing [kernels loop] directive", 3);
		addToMap("loop", "_directive");
		while (end_of_token() == false) 
		{
			String tok = get_token();
			if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
			if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
			String clause = "acc_" + tok;
			PrintTools.println("clause=" + clause, 3);
			try {
				switch (acc_clause.valueOf(clause)) {
				case acc_if		:	parse_acc_confclause(tok); break;
				case acc_async	:	parse_acc_optionalconfclause(tok); break;
				case acc_wait	:	parse_acc_optionalconflistclause(tok); break;
				case acc_copy		:	parse_acc_dataclause(tok); break;
				case acc_copyin 		:	parse_acc_dataclause(tok); break;
				case acc_copyout 		:	parse_acc_dataclause(tok); break;
				case acc_create 		:	parse_acc_dataclause(tok); break;
				case acc_present 		:	parse_acc_dataclause(tok); break;
				case acc_present_or_copy 		:	parse_acc_dataclause("pcopy"); break;
				case acc_pcopy		:	parse_acc_dataclause(tok); break;
				case acc_present_or_copyin		:	parse_acc_dataclause("pcopyin"); break;
				case acc_pcopyin		:	parse_acc_dataclause(tok); break;
				case acc_present_or_copyout	:	parse_acc_dataclause("pcopyout"); break;
				case acc_pcopyout	:	parse_acc_dataclause(tok); break;
				case acc_present_or_create	:	parse_acc_dataclause("pcreate"); break;
				case acc_pcreate	:	parse_acc_dataclause(tok); break;
				case acc_deviceptr	:	parse_acc_dataclause(tok); break;
				case acc_collapse		: parse_acc_confclause(tok); break;
				case acc_gang		: parse_acc_optionalconfclause(tok); break;
				case acc_worker		: parse_acc_optionalconfclause(tok); break;
				case acc_vector		: parse_acc_optionalconfclause(tok); break;
				case acc_seq		: parse_acc_noargclause(tok); break;
				case acc_independent		: parse_acc_noargclause(tok); break;
				case acc_private	:	parse_acc_dataclause(tok); break;
				case acc_reduction		:	parse_acc_reduction(tok); break;
                case acc_tile		: parse_expressionlist(tok); break;
				case acc_pipein 		:	parse_acc_dataclause(tok); break;
				case acc_pipeout 		:	parse_acc_dataclause(tok); break;
				default : ACCParserError("NoSuchOpenACCConstruct : " + tok);
				}
			} catch( Exception e) {
				ACCParserError("unexpected or wrong token found (" + tok + ")");
			}
		}
	}
	
	/** ---------------------------------------------------------------
	 *		OpenACC loop Construct
	 *
	 *		#pragma acc loop [clause[[,] clause]...] new-line
	 *			structured-block
	 *
	 *		where clause is one of the following
	 * 		collapse( n )
	 * 		gang [( scalar-integer-expression )]
	 * 		worker [( scalar-integer-expression )]
	 * 		vector [( scalar-integer-expression )]
	 * 		seq
	 * 		nowait
	 * 		independent
	 *      reduction( operator:list )
	 * 		private( list ) 
	 * --------------------------------------------------------------- */
	private static void parse_acc_loop()
	{
		PrintTools.println("ACCParser is parsing [loop] directive", 3);
		addToMap("loop", "_directive");
		while (end_of_token() == false) 
		{
			String tok = get_token();	
			if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
			if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
			String clause = "acc_" + tok;
			PrintTools.println("clause=" + clause, 3);
			try {
				switch (acc_clause.valueOf(clause)) {
				case acc_collapse		: parse_acc_confclause(tok); break;
				case acc_gang		: parse_acc_optionalconfclause(tok); break;
				case acc_worker		: parse_acc_optionalconfclause(tok); break;
				case acc_vector		: parse_acc_optionalconfclause(tok); break;
				case acc_seq		: parse_acc_noargclause(tok); break;
				case acc_nowait		: parse_acc_noargclause(tok); break;
				case acc_independent		: parse_acc_noargclause(tok); break;
				case acc_private	:	parse_acc_dataclause(tok); break;
				case acc_reduction		:	parse_acc_reduction(tok); break;
                case acc_tile		: parse_expressionlist(tok); break;
				default : ACCParserError("NoSuchOpenACCConstruct : " + clause);
				}
			} catch( Exception e) {
				ACCParserError("unexpected or wrong token found (" + tok + ")");
			}
		}
	}
	
	/** ---------------------------------------------------------------
	 *		OpenACC data Construct
	 *
	 *		#pragma acc data [clause[[,] clause]...] new-line
	 *			structured-block
	 *
	 *		where clause is one of the following
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
	 * 		pipe( list ) 
	 * --------------------------------------------------------------- */
	private static void parse_acc_data()
	{
		addToMap("data", "_directive");
		PrintTools.println("ACCParser is parsing [data] directive", 3);
		while (end_of_token() == false) 
		{
			String tok = get_token();
			if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
			if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
			String clause = "acc_" + tok;
			PrintTools.println("clause=" + clause, 3);
			try {
				switch (acc_clause.valueOf(clause)) {
				case acc_if		:	parse_acc_confclause(tok); break;
				case acc_copy		:	parse_acc_dataclause(tok); break;
				case acc_copyin 		:	parse_acc_dataclause(tok); break;
				case acc_copyout 		:	parse_acc_dataclause(tok); break;
				case acc_create 		:	parse_acc_dataclause(tok); break;
				case acc_present 		:	parse_acc_dataclause(tok); break;
				case acc_present_or_copy 		:	parse_acc_dataclause("pcopy"); break;
				case acc_pcopy		:	parse_acc_dataclause(tok); break;
				case acc_present_or_copyin		:	parse_acc_dataclause("pcopyin"); break;
				case acc_pcopyin		:	parse_acc_dataclause(tok); break;
				case acc_present_or_copyout	:	parse_acc_dataclause("pcopyout"); break;
				case acc_pcopyout	:	parse_acc_dataclause(tok); break;
				case acc_present_or_create	:	parse_acc_dataclause("pcreate"); break;
				case acc_pcreate	:	parse_acc_dataclause(tok); break;
				case acc_deviceptr	:	parse_acc_dataclause(tok); break;
				case acc_pipe 		:	parse_acc_dataclause(tok); break;
				default : ACCParserError("NoSuchOpenACCConstruct : " + clause);
				}
			} catch( Exception e) {
				ACCParserError("unexpected or wrong token found (" + tok + ")");
			}
		}
	}
	
	/** ---------------------------------------------------------------
	 *		OpenACC enter data Construct
	 *
	 *		#pragma acc enter data [clause[[,] clause]...] new-line
	 *
	 *		where clause is one of the following
	 *      if( condition )
	 * 		copyin( list ) 
	 * 		create( list ) 
	 * 		present_or_copyin( list ) 
	 * 		pcopyin( list ) 
	 * 		present_or_create( list ) 
	 * 		pcreate( list ) 
	 *      async [(int-expr)]
	 *      wait [(int-expr-list)]
	 * --------------------------------------------------------------- */
	private static void parse_acc_enter_data()
	{
		addToMap("data", "_directive");
		PrintTools.println("ACCParser is parsing [enter data] directive", 3);
		while (end_of_token() == false) 
		{
			String tok = get_token();
			if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
			if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
			String clause = "acc_" + tok;
			PrintTools.println("clause=" + clause, 3);
			try {
				switch (acc_clause.valueOf(clause)) {
				case acc_if		:	parse_acc_confclause(tok); break;
				case acc_copyin 		:	parse_acc_dataclause(tok); break;
				case acc_create 		:	parse_acc_dataclause(tok); break;
				case acc_present_or_copyin		:	parse_acc_dataclause("pcopyin"); break;
				case acc_pcopyin		:	parse_acc_dataclause(tok); break;
				case acc_present_or_create	:	parse_acc_dataclause("pcreate"); break;
				case acc_pcreate	:	parse_acc_dataclause(tok); break;
				case acc_async	:	parse_acc_optionalconfclause(tok); break;
				case acc_wait	:	parse_acc_optionalconflistclause(tok); break;
				default : ACCParserError("NoSuchOpenACCConstruct : " + clause);
				}
			} catch( Exception e) {
				ACCParserError("unexpected or wrong token found (" + tok + ")");
			}
		}
	}
	
	/** ---------------------------------------------------------------
	 *		OpenACC exit data Construct
	 *
	 *		#pragma acc exit data [clause[[,] clause]...] new-line
	 *
	 *		where clause is one of the following
	 *      if( condition )
	 *      async [(int-expr)]
	 *      wait [(int-expr-list)]
	 * 		copyout( list ) 
	 *      delete( list )
	 *      finalize
	 * --------------------------------------------------------------- */
	private static void parse_acc_exit_data()
	{
		addToMap("data", "_directive");
		PrintTools.println("ACCParser is parsing [exit data] directive", 3);
		while (end_of_token() == false) 
		{
			String tok = get_token();
			if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
			if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
			String clause = "acc_" + tok;
			PrintTools.println("clause=" + clause, 3);
			try {
				switch (acc_clause.valueOf(clause)) {
				case acc_if		:	parse_acc_confclause(tok); break;
				case acc_copyout 		:	parse_acc_dataclause(tok); break;
				case acc_async	:	parse_acc_optionalconfclause(tok); break;
				case acc_wait	:	parse_acc_optionalconflistclause(tok); break;
				case acc_finalize		: parse_acc_noargclause(tok); break;
				case acc_delete 		:	parse_acc_dataclause(tok); break;
				default : ACCParserError("NoSuchOpenACCConstruct : " + clause);
				}
			} catch( Exception e) {
				ACCParserError("unexpected or wrong token found (" + tok + ")");
			}
		}
	}

	/** ---------------------------------------------------------------
	 *		OpenACC set Construct
	 *
	 *		#pragma acc set [clause[[,] clause]...] new-line
	 *
	 *	   where clause is one of the following
	 *     default_async ( scalar-integer-expression )
	 *     device_num ( scalar-integer-expression )
	 *     device_type ( device-type )
	 * --------------------------------------------------------------- */
	private static void parse_acc_set()
	{
		addToMap("set", "_directive");
		PrintTools.println("ACCParser is parsing [set] directive", 3);
		while (end_of_token() == false) 
		{
			String tok = get_token();
			if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
			if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
			String clause = "acc_" + tok;
			PrintTools.println("clause=" + clause, 3);
			try {
				switch (acc_clause.valueOf(clause)) {
				case acc_default_async		:	parse_acc_confclause(tok); break;
				case acc_device_num	:	parse_acc_confclause(tok); break;
				case acc_device_type	:	parse_acc_confclause(tok); break;
				default : ACCParserError("NoSuchOpenACCConstruct : " + clause);
				}
			} catch( Exception e) {
				ACCParserError("unexpected or wrong token found (" + tok + ")");
			}
		}
	}
	
	/** ---------------------------------------------------------------
	 *		OpenACC host_data Construct
	 *
	 *		#pragma acc host_data [clause[[,] clause]...] new-line
	 *			structured-block
	 *
	 *		where clause is one of the following
	 * 		use_device( list ) 
	 * --------------------------------------------------------------- */
	private static void parse_acc_host_data()
	{
		addToMap("host_data", "_directive");
		PrintTools.println("ACCParser is parsing [host_data] directive", 3);
		while (end_of_token() == false) 
		{
			String tok = get_token();
			if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
			if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
			String clause = "acc_" + tok;
			PrintTools.println("clause=" + clause, 3);
			try {
				switch (acc_clause.valueOf(clause)) {
				case acc_use_device		:	parse_acc_dataclause(tok); break;
				default : ACCParserError("NoSuchOpenACCConstruct : " + clause);
				}
			} catch( Exception e) {
				ACCParserError("unexpected or wrong token found (" + tok + ")");
			}
		}
	}
	
	/** ---------------------------------------------------------------
	 *		OpenACC cache Construct
	 *
	 *		#pragma acc cache ( list ) new-line
	 *
	 * --------------------------------------------------------------- */
	private static void parse_acc_cache()
	{
		parse_acc_dataclause("cache");
	}
	
	/** ---------------------------------------------------------------
	 *		OpenACC declare Construct
	 *
	 *		#pragma acc declare declclause [[,] declclause]... new-line
	 *
	 *		where declclause is one of the following
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
	 * --------------------------------------------------------------- */
	private static void parse_acc_declare()
	{
		boolean declclauseexist = false;
		addToMap("declare", "_directive");
		PrintTools.println("ACCParser is parsing [declare] directive", 3);
		while (end_of_token() == false) 
		{
			String tok = get_token();
			if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
			if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
			String clause = "acc_" + tok;
			PrintTools.println("clause=" + clause, 3);
			try {
				switch (acc_clause.valueOf(clause)) {
				case acc_copy		:	parse_acc_declaredataclause(tok); declclauseexist = true; break;
				case acc_copyin 		:	parse_acc_declaredataclause(tok); declclauseexist = true; break;
				case acc_copyout 		:	parse_acc_declaredataclause(tok); declclauseexist = true; break;
				case acc_create 		:	parse_acc_declaredataclause(tok); declclauseexist = true; break;
				case acc_present 		:	parse_acc_declaredataclause(tok); declclauseexist = true; break;
				case acc_present_or_copy 		:	parse_acc_declaredataclause("pcopy"); declclauseexist = true; break;
				case acc_pcopy		:	parse_acc_declaredataclause(tok); declclauseexist = true; break;
				case acc_present_or_copyin		:	parse_acc_declaredataclause("pcopyin"); declclauseexist = true; break;
				case acc_pcopyin		:	parse_acc_declaredataclause(tok); declclauseexist = true; break;
				case acc_present_or_copyout	:	parse_acc_declaredataclause("pcopyout"); declclauseexist = true; break;
				case acc_pcopyout	:	parse_acc_declaredataclause(tok); declclauseexist = true; break;
				case acc_present_or_create	:	parse_acc_declaredataclause("pcreate"); declclauseexist = true; break;
				case acc_pcreate	:	parse_acc_declaredataclause(tok); declclauseexist = true; break;
				case acc_deviceptr	:	parse_acc_declaredataclause(tok); declclauseexist = true; break;
				case acc_device_resident	:	parse_acc_declaredataclause(tok); declclauseexist = true; break;
				case acc_pipe 		:	parse_acc_declaredataclause(tok); declclauseexist = true; break;
				default : ACCParserError("NoSuchOpenACCConstruct : " + clause);
				}
			} catch( Exception e) {
				ACCParserError("unexpected or wrong token found (" + tok + ")");
			}
		}
		if( !declclauseexist ) {
			ACCParserError("No valid declclause is found for the declare directive");
		}
	}
	
	/** ---------------------------------------------------------------
	 *		OpenACC update Construct
	 *
	 *		#pragma acc update clause[[,] clause]... new-line
	 *
	 *		where clause is one of the following
	 * 		use_device( list ) 
	 *      async [( scalar-integer-expression )]
	 *      wait [( scalar-integer-expression-list )]
	 * --------------------------------------------------------------- */
	private static void parse_acc_update()
	{
		boolean clauseexist = false;
		addToMap("update", "_directive");
		PrintTools.println("ACCParser is parsing [update] directive", 3);
		while (end_of_token() == false) 
		{
			String tok = get_token();
			if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
			if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
			String clause = "acc_" + tok;
			PrintTools.println("clause=" + clause, 3);
			try {
				switch (acc_clause.valueOf(clause)) {
				case acc_host		:	parse_acc_dataclause(tok); clauseexist = true; break;
				case acc_self		:	parse_acc_dataclause(tok); clauseexist = true; break;
				case acc_device	:	parse_acc_dataclause(tok); clauseexist = true; break;
				case acc_if		:	parse_acc_confclause(tok); break;
				case acc_async	:	parse_acc_optionalconfclause(tok); break;
				case acc_wait	:	parse_acc_optionalconflistclause(tok); break;
				default : ACCParserError("NoSuchOpenACCConstruct : " + clause);
				}
			} catch( Exception e) {
				ACCParserError("unexpected or wrong token found (" + tok + ")");
			}
		}
		if( !clauseexist ) {
			ACCParserError("No valid dataclause is found for the update directive");
		}
	}
	
	/** ---------------------------------------------------------------
	 *		OpenACC wait Construct
	 *
	 *		#pragma acc wait [( scalar_integer_expression )] new-line
	 *
	 * --------------------------------------------------------------- */
	private static void parse_acc_wait()
	{
		parse_acc_optionaldirective("wait");
	}   

	/** ---------------------------------------------------------------
	 *		OpenACC mpi Construct
	 *
	 *		#pragma acc mpi [clause[[,] clause]...] new-line
	 *
	 *		where clause is one of the following
	 *     sendbuf ( [device] [,] [readonly] )
	 *     recvbuf ( [device] [,] [readonly] )
	 *     async [(int-expr)]   
	 * --------------------------------------------------------------- */
	private static void parse_acc_mpi()
	{
		addToMap("mpi", "_directive");
		PrintTools.println("ACCParser is parsing [mpi] directive", 3);
		while (end_of_token() == false) 
		{
			String tok = get_token();
			if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
			if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
			String clause = "acc_" + tok;
			PrintTools.println("clause=" + clause, 3);
			try {
				switch (acc_clause.valueOf(clause)) {
				case acc_sendbuf		:	parse_conf_expressionset(tok); break;
				case acc_recvbuf		:	parse_conf_expressionset(tok); break;
				case acc_async	:	parse_acc_optionalconfclause(tok); break;
				default : ACCParserError("NoSuchOpenACCConstruct : " + clause);
				}
			} catch( Exception e) {
				ACCParserError("unexpected or wrong token found (" + tok + ")");
			}
		}
	}
	
	/**
	 * OpenARC cuda directive 
	 * 
	 * #pragma openarc cuda [clause[[,] clause]...]
	 * 		structured-block
	 * 
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
	 */
	private static void parse_arc_cuda()
	{
		PrintTools.println("ACCParser is parsing [cuda] directive", 3);
		addToMap("cuda", "_directive");
		while (end_of_token() == false) 
		{
			String tok = get_token();
			if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
			if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
			String clause = "token_" + tok;
			PrintTools.println("clause=" + clause, 3);
			try {
				switch (cuda_clause.valueOf(clause)) {
				case token_registerRO		:	parse_acc_dataclause(tok); break;
				case token_registerRW 		:	parse_acc_dataclause(tok); break;
				case token_noregister 		:	parse_acc_dataclause(tok); break;
				case token_sharedRO 		:	parse_acc_dataclause(tok); break;
				case token_sharedRW 		:	parse_acc_dataclause(tok); break;
				case token_noshared 		:	parse_acc_dataclause(tok); break;
				case token_texture		:	parse_acc_dataclause(tok); break;
				case token_notexture		:	parse_acc_dataclause(tok); break;
				case token_constant		:	parse_acc_dataclause(tok); break;
				case token_noconstant	:	parse_acc_dataclause(tok); break;
				case token_global	:	parse_acc_dataclause(tok); break;
				default : ACCParserError("NoSuchOpenARCConstruct : " + clause);
				}
			} catch( Exception e) {
				ACCParserError("unexpected or wrong token found (" + tok + ")");
			}
		}
	}

	/**
	 * OpenARC opencl directive 
	 * 
	 * #pragma openarc opencl [clause[[,] clause]...]
	 * 		structured-block
	 * 
	 * where clause is one of the following
	 * 		num_simd_work_items(exp) 
	 * 		num_compute_units(exp) 
	 */
	private static void parse_arc_opencl()
	{
		PrintTools.println("ACCParser is parsing [opencl] directive", 3);
		addToMap("opencl", "_directive");
		while (end_of_token() == false) 
		{
			String tok = get_token();
			if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
			if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
			String clause = "token_" + tok;
			PrintTools.println("clause=" + clause, 3);
			try {
				switch (opencl_clause.valueOf(clause)) {
				case token_num_simd_work_items	:	parse_acc_confclause(tok); break;
				case token_num_compute_units	:	parse_acc_confclause(tok); break;
				default : ACCParserError("NoSuchOpenARCConstruct : " + clause);
				}
			} catch( Exception e) {
				ACCParserError("unexpected or wrong token found (" + tok + ")");
			}
		}
	}
	
	/**
	 * OpenARC transform directive 
	 * 
	 * #pragma openarc transform [clause[[,] clause]...]
	 * 		structured-block
	 * 
	 * where clause is one of the following
	 *      permute(list)
	 *      unroll(N)
	 *      noreductionunroll(list)
	 *      noploopswap
	 *      noloopcollapse
	 *      multisrccg(list)
	 *      multisrcgc(list)
	 *      conditionalsrc(list)
	 *      enclosingloops(list)
	 *      window
	 */
	private static void parse_arc_transform()
	{
		PrintTools.println("ACCParser is parsing [transform] directive", 3);
		addToMap("transform", "_directive");
		while (end_of_token() == false) 
		{
			String tok = get_token();
			if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
			if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
			String clause = "token_" + tok;
			PrintTools.println("clause=" + clause, 3);
			try {
				switch (transform_clause.valueOf(clause)) {
				case token_noreductionunroll	:	parse_acc_dataclause(tok); break;
				case token_noploopswap	:	parse_acc_noargclause(tok); break;
				case token_noloopcollapse	:	parse_acc_noargclause(tok); break;
				case token_permute		: parse_expressionlist(tok); break;
				case token_unroll		: parse_acc_confclause(tok); break;
				case token_transpose	: parse_arc_clause_with_subarrayconf(tok); break;
				case token_redim	: parse_arc_clause_with_subarrayconf(tok); break;
				case token_expand	: parse_arc_clause_with_subarrayconf(tok); break;
				case token_expand_transpose	: parse_arc_clause_with_subarrayconf(tok); break;
				case token_redim_transpose	: parse_arc_clause_with_subarrayconf(tok); break;
				case token_transpose_expand	: parse_arc_clause_with_subarrayconf(tok); break;
				case token_window	:	parse_arc_windowclause(tok); break;
				case token_multisrccg	:	parse_acc_dataclause(tok); break;
				case token_multisrcgc	:	parse_acc_dataclause(tok); break;
				case token_conditionalsrc		: parse_acc_dataclause(tok); break;
				case token_enclosingloops		: parse_acc_dataclause(tok); break;
				default : ACCParserError("NoSuchOpenARCConstruct : " + clause);
				}
			} catch( Exception e) {
				ACCParserError("unexpected or wrong token found (" + tok + ")");
			}
		}
	}

	private static void parse_arc_impacc()
	{
		PrintTools.println("ACCParser is parsing [impacc] directive", 3);
		addToMap("impacc", "_directive");
		while (end_of_token() == false) 
		{
			String tok = get_token();
			if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
			if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
			String clause = "token_" + tok;
			PrintTools.println("clause=" + clause, 3);
			try {
				switch (impacc_clause.valueOf(clause)) {
				case token_ignoreglobal		:	parse_conf_stringset(tok); break;
				default : ACCParserError("NoSuchOpenARCConstruct : " + clause);
				}
			} catch( Exception e) {
				ACCParserError("unexpected or wrong token found (" + tok + ")");
			}
		}
	}

	/** ---------------------------------------------------------------
	 *		OpenARC devicetask Construct
	 *
	 *		#pragma openarc devicetask map(task-mapping-scheme) schedule(task-scheduling-scheme) new-line
	 *			structured-block
	 *
	 * --------------------------------------------------------------- */
	private static void parse_arc_devicetask()
	{
		PrintTools.println("ACCParser is parsing [devicetask] directive", 3);
		addToMap("devicetask", "_directive");

		while (end_of_token() == false) 
		{
			String token = get_token();
			if( token.equals("") ) continue; //Skip empty string, which may occur due to macro.
			String clause = "token_" + token;
			if( token.equals(",") ) continue; //Skip comma between clauses, if existing.
			PrintTools.println("clause=" + clause, 3);
			try {
				switch (devicetask_clause.valueOf(clause)) {
				case token_map		:	parse_acc_stringargclause(token); break;
				case token_schedule		:	parse_acc_stringargclause(token); break;
				default : ACCParserError("NoSuchAinfoConstruct : " + clause);
				}
			} catch( Exception e) {
				ACCParserError("unexpected or wrong token found (" + token + ")");
			}
		}
	}
	
    /**
     * Parse OpenARC pragmas, which are stored as raw text after Cetus parsing.
     * 
     * @param input_map HashMap that will contain the parsed output pragmas
     * @param str_array input pragma string that will be parsed.
     * @return true if the pragma will be attached to the following non-pragma
     *         statement. Or, it returns false, if the pragma is stand-alone.
     */
	public static boolean parse_openarc_pragma(HashMap input_map, String [] str_array, HashMap<String, String>macro_map)
	{
		acc_map = input_map;
		token_array = str_array;
		token_index = 1; // "openarc" has already been matched
		//token_index = 2; // If there is a leading space, use this one.
		macroMap = macro_map;

		PrintTools.println(display_tokens(), 9);

		String token = get_token();
		String construct = "arc_" + token;
		try {
			switch (arc_directives.valueOf(construct)) {
			case arc_ainfo 		: parse_arc_ainfo(); return true;
			case arc_cuda 		: parse_arc_cuda(); return true;
			case arc_opencl 		: parse_arc_opencl(); return true;
			case arc_transform 		: parse_arc_transform(); return true;
			case arc_resilience 	: parse_arc_resilience(); return true;
			case arc_ftregion 	: parse_arc_ftregion(); return true;
			case arc_ftinject 	: parse_arc_ftinject(); return false;
			case arc_profile 	: return parse_arc_profile();
			case arc_enter 	: parse_arc_enter(); return false;
			case arc_exit 	: parse_arc_exit(); return false;
			case arc_impacc 		: parse_arc_impacc(); return false;
			case arc_devicetask 		: parse_arc_devicetask(); return true;
			//		default : throw new NonOmpDirectiveException();
			default : ACCParserError("Not Supported Construct");
			}
		} catch( Exception e) {
			ACCParserError("unexpected or wrong token found (" + token + ")");
		}
		return true;		// meaningless return because it is unreachable
	}
	
	/** ---------------------------------------------------------------
	 * 
	 *		User Directive
	 *      (Used in a user-directive file.)
	 *
	 *		kernelid(id) procname(name) [clause[[,] clause]...] new-line
	 *
	 *		where clause is one of the following
	 * 		registerRO(list) 
	 * 		registerRW(list) 
	 * 		noregister(list)
	 * 		sharedRO(list) 
	 * 		sharedRW(list) 
	 * 		noshared(list) 
	 * 		texture(list) 
	 * 		notexture(list) 
	 * 		constant(list)
	 * 		noconstant(list)
	 *      global(list)
	 *      noreductionunroll(list)
	 *      noploopswap
	 *      noloopcollapse
	 *      multisrccg(list)
	 *      multisrcgc(list)
	 *      conditionalsrc(list)
	 *      enclosingloops(list)
	 *      permute(list)
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
	 * 		collapse( n )
	 * 		gang [( scalar-integer-expression )]
	 * 		worker [( scalar-integer-expression )]
	 * 		vector [( scalar-integer-expression )]
	 * 		seq
	 * 		independent
	 * 		permute
	 * 		unroll
	 * 		num_simd_work_items
	 * 		num_compute_units
	 * --------------------------------------------------------------- */

	public static HashMap<String,HashMap<String, Object>> parse_userdirective(String[] str_array)
	{
		acc_map = new HashMap<String, Object>();
		HashMap userDirectives = new HashMap<String, HashMap<String, Object>>();
		token_array = str_array;
		token_index = 0; 
		macroMap = new HashMap<String, String>();
		PrintTools.println(display_tokens(), 9);

		PrintTools.println("ACCParser is parsing user-directves", 2);
		String kernelid = null;
		String procname = null;
		String kernelname = null;
		String clause = "token_" + get_token();
		if( clause.equals("token_kernelid") ) {
			match("(");
			kernelid = get_token();
			match(")");
		} else {
			PrintTools.println("[ERROR in parse_userdirective()] kernelid is missing; " +
					"current user directive line will be ignored.", 0);
			PrintTools.println("Current token is " + clause, 0);
			return null;
		}
		clause = "token_" + get_token();
		if( clause.equals("token_procname") ) {
			match("(");
			procname = get_token();
			match(")");
		} else {
			PrintTools.println("[ERROR in parse_userdirective()] procname is missing; " +
					"current user directive line will be ignored.", 0);
			return null;
		}
		//kernelname = procname.concat(kernelid);
		kernelname = procname + "_kernel" + kernelid;

		while (end_of_token() == false) 
		{
			String tok = get_token();
			if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
			if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
			clause = "token_" + tok;
			PrintTools.println("clause=" + clause, 3);
			try {
				switch (user_clause.valueOf(clause)) {
				case token_registerRO		:	parse_acc_dataclause(tok); break;
				case token_registerRW 		:	parse_acc_dataclause(tok); break;
				case token_noregister 		:	parse_acc_dataclause(tok); break;
				case token_sharedRO 		:	parse_acc_dataclause(tok); break;
				case token_sharedRW 		:	parse_acc_dataclause(tok); break;
				case token_noshared 		:	parse_acc_dataclause(tok); break;
				case token_texture		:	parse_acc_dataclause(tok); break;
				case token_notexture		:	parse_acc_dataclause(tok); break;
				case token_constant		:	parse_acc_dataclause(tok); break;
				case token_noconstant	:	parse_acc_dataclause(tok); break;
				case token_global	:	parse_acc_dataclause(tok); break;
				case token_noreductionunroll	:	parse_acc_dataclause(tok); break;
				case token_noploopswap	:	parse_acc_noargclause(tok); break;
				case token_noloopcollapse	:	parse_acc_noargclause(tok); break;
				case token_multisrccg	:	parse_acc_dataclause(tok); break;
				case token_multisrcgc	:	parse_acc_dataclause(tok); break;
				case token_conditionalsrc		: parse_acc_dataclause(tok); break;
				case token_enclosingloops		: parse_acc_dataclause(tok); break;
				case token_permute		: parse_expressionlist(tok); break;
				case token_unroll		:	parse_acc_confclause(tok); break;
				case token_if		:	parse_acc_confclause(tok); break;
				case token_async	:	parse_acc_optionalconfclause(tok); break;
				case token_wait	:	parse_acc_optionalconflistclause(tok); break;
				case token_num_gangs		:	parse_acc_confclause(tok); break;
				case token_num_workers	:	parse_acc_confclause(tok); break;
				case token_vector_length		:	parse_acc_confclause(tok); break;
				case token_reduction		:	parse_acc_reduction(tok); break;
				case token_copy		:	parse_acc_dataclause(tok); break;
				case token_copyin 		:	parse_acc_dataclause(tok); break;
				case token_copyout 		:	parse_acc_dataclause(tok); break;
				case token_create 		:	parse_acc_dataclause(tok); break;
				case token_present 		:	parse_acc_dataclause(tok); break;
				case token_present_or_copy 		:	parse_acc_dataclause("pcopy"); break;
				case token_pcopy		:	parse_acc_dataclause(tok); break;
				case token_present_or_copyin		:	parse_acc_dataclause("pcopyin"); break;
				case token_pcopyin		:	parse_acc_dataclause(tok); break;
				case token_present_or_copyout	:	parse_acc_dataclause("pcopyout"); break;
				case token_pcopyout	:	parse_acc_dataclause(tok); break;
				case token_present_or_create	:	parse_acc_dataclause("pcreate"); break;
				case token_pcreate	:	parse_acc_dataclause(tok); break;
				case token_deviceptr	:	parse_acc_dataclause(tok); break;
				case token_private	:	parse_acc_dataclause(tok); break;
				case token_firstprivate	:	parse_acc_dataclause(tok); break;
				case token_collapse		: parse_acc_confclause(tok); break;
				case token_gang		: parse_acc_optionalconfclause(tok); break;
				case token_worker		: parse_acc_optionalconfclause(tok); break;
				case token_vector		: parse_acc_optionalconfclause(tok); break;
				case token_seq		: parse_acc_noargclause(tok); break;
				case token_independent		: parse_acc_noargclause(tok); break;
				case token_num_simd_work_items		:	parse_acc_confclause(tok); break;
				case token_num_compute_units		:	parse_acc_confclause(tok); break;
				default : ACCParserError("NoSuchUserConstruct : " + clause);
				}
			} catch( Exception e) {
				ACCParserError("unexpected or wrong token found (" + tok + ")");
			}
		}
		userDirectives.put(kernelname, acc_map);
		return userDirectives;
	}
	
	/*--------------------------------------------------------------------
	 * Available tuning configurations 
	 * -------------------------------------------------------------------
	 * defaultGOptionSet(list) 
	 *     - where list is a comma-seperated list of program-level tuning  
	 *       parameters, which will be always applied.
	 *     - List of program-level tuning parameters
	 *         assumeNonZeroTripLoops
	 *         gpuMallocOptLevel     
	 *         gpuMemTrOptLevel
	 *         useMatrixTranspose
	 *         useMallocPitch
	 *         useLoopCollapse
	 *         useParallelLoopSwap
	 *         useUnrollingOnReduction
	 *         shrdSclrCachingOnReg
	 *         shrdArryElmtCachingOnReg
	 *         shrdSclrCachingOnSM
	 *         prvtArryCachingOnSM
	 *         shrdArryCachingOnTM
	 *         defaultNumWorkers
	 *         maxNumGangs
	 *         AccPrivatization
	 *         AccReduction
	 *         localRedVarConf
	 *         assumeNoAliasingAmongKernelArgs
	 *         skipKernelLoopBoundChecking
	 * excludedGOptionSet(list) 
	 *     - where list is a comma-seperated list of program-level tuning  
	 *       parameters, which will not be applied.
	 *     - List of program-level tuning parameters
	 *         assumeNonZeroTripLoops
	 *         gpuMallocOptLevel     
	 *         gpuMemTrOptLevel
	 *         useMatrixTranspose
	 *         useMallocPitch
	 *         useLoopCollapse
	 *         useParallelLoopSwap
	 *         useUnrollingOnReduction
	 *         shrdSclrCachingOnReg
	 *         shrdArryElmtCachingOnReg
	 *         shrdSclrCachingOnSM
	 *         prvtArryCachingOnSM
	 *         shrdArryCachingOnTM
	 *         defaultNumWorkers
	 *         maxNumGangs
	 *         assumeNoAliasingAmongKernelArgs
	 *         skipKernelLoopBoundChecking
	 * gpuMemTrOptLevel=N
	 * gpuMallocOptLevel=N
	 * UEPRemovalOptLevel=N
	 * AccPrivatization=N
	 * AccReduction=N
	 * localRedVarConf=N
	 * defaultNumWorkersSet(list)
	 *    - where list is the a comma-separated list of numbers.
	 * maxNumGangsSet(list)
	 *    - where list is the a comma-separated list of numbers.
	 * defaultNumComputeUnits(list)
	 *    - where list is the a comma-separated list of numbers.
	 * defaultNumSIMDWorkItems(list)
	 *    - where list is the a comma-separated list of numbers.
	 *----------------------------------------------------------------------*/
	
	public static HashMap<String,Object> parse_tuningconfig(String[] str_array)
	{
		acc_map = new HashMap<String, Object>();
		token_array = str_array;
		token_index = 0; 
		macroMap = new HashMap<String, String>();
		PrintTools.println(display_tokens(), 9);

		PrintTools.println("ACCParser is parsing default tuning configuration", 2);
		String clause = null;

		while (end_of_token() == false) 
		{
			String tok = get_token();
			if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
			if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
			clause = "conf_" + tok;
			PrintTools.println("clause=" + clause, 3);
			try {
				switch (cuda_tuningconf.valueOf(clause)) {
				case conf_defaultGOptionSet		:	parse_conf_stringset(tok); break;
				case conf_excludedGOptionSet		:	parse_conf_stringset(tok); break;
				case conf_gpuMemTrOptLevel	:	parse_conf_expression(tok); break;
				case conf_gpuMallocOptLevel		:	parse_conf_expression(tok); break;
				case conf_AccPrivatization		:	parse_conf_expression(tok); break;
				case conf_AccReduction		:	parse_conf_expression(tok); break;
				case conf_localRedVarConf		:	parse_conf_expression(tok); break;
				case conf_defaultNumWorkersSet		:	parse_conf_expressionset(tok); break;
				case conf_maxNumGangsSet		:	parse_conf_expressionset(tok); break;
				case conf_defaultNumComputeUnits		:	parse_conf_expressionset(tok); break;
				case conf_defaultNumSIMDWorkItems		:	parse_conf_expressionset(tok); break;
				case conf_UEPRemovalOptLevel		:	parse_conf_expression(tok); break;
				default : ACCParserError("NoSuchCudaConstruct : " + clause);
				}
			} catch( Exception e) {
				ACCParserError("unexpected or wrong token found (" + tok + ")");
			}
		}
		return acc_map;
	}
	
    /**
     * Parse ASPEN pragmas, which are stored as raw text after Cetus parsing.
     * 
     * @param input_map HashMap that will contain the parsed output pragmas
     * @param str_array input pragma string that will be parsed.
     * @return true if the pragma will be attached to the following non-pragma
     *         statement. Or, it returns false, if the pragma is stand-alone.
     */
	public static boolean parse_aspen_pragma(HashMap input_map, String [] str_array, HashMap<String, String>macro_map)
	{
		acc_map = input_map;
		token_array = str_array;
		token_index = 1; // "aspen" has already been matched
		//token_index = 2; // If there is a leading space, use this one.
		macroMap = macro_map;

		PrintTools.println(display_tokens(), 9);

		String token = get_token();
		String construct = "aspen_" + token;
		try {
			switch (aspen_directives.valueOf(construct)) {
			case aspen_enter 		: parse_aspen_enter(); return false;
			case aspen_exit 		: parse_aspen_exit(); return false;
			case aspen_modelregion 		: parse_aspen_modelregion(null); return true;
			case aspen_declare 		: parse_aspen_declare(); return false;
			case aspen_control 		: parse_aspen_control(); return true;
			//		default : throw new NonOmpDirectiveException();
			default : ACCParserError("Not Supported Construct");
			}
		} catch( Exception e) {
			ACCParserError("Error occured during parsing " + token + " directive.\n" + "Error message: " + e);
		}
		return true;		// meaningless return because it is unreachable
	}
	
	private static void parse_aspen_enter()
	
	{
		addToMap("enter", "_directive");
		while (end_of_token() == false) 
		{
			String token = get_token();
			if( token.equals("") ) continue; //Skip empty string, which may occur due to macro.
			String directive = "aspen_" + token;
			if( token.equals(",") ) continue; //Skip comma between clauses, if existing.
			PrintTools.println("directive=" + directive, 3);
			try {
				switch (aspen_directives.valueOf(directive)) {
				case aspen_modelregion		:	parse_aspen_modelregion("enter"); break;
				default : ACCParserError("NoSuchASPENConstruct : " + directive);
				}
			} catch( Exception e) {
				ACCParserError("Error occured during parsing " + token + " directive.\n" + "Error message: " + e);
			}
		}
	}
	
	private static void parse_aspen_exit()
	
	{
		addToMap("exit", "_directive");
		while (end_of_token() == false) 
		{
			String token = get_token();
			if( token.equals("") ) continue; //Skip empty string, which may occur due to macro.
			String directive = "aspen_" + token;
			if( token.equals(",") ) continue; //Skip comma between clauses, if existing.
			PrintTools.println("directive=" + directive, 3);
			try {
				switch (aspen_directives.valueOf(directive)) {
				case aspen_modelregion		:	parse_aspen_modelregion("exit"); break;
				default : ACCParserError("NoSuchASPENConstruct : " + directive);
				}
			} catch( Exception e) {
				ACCParserError("Error occured during parsing " + token + " directive.\n" + "Error message: " + e);
			}
		}
	}
	
	private static void parse_aspen_modelregion(String prefix)
	
	{
		String directiveName = null;
		if( prefix == null ) {
			directiveName = "modelregion";
		} else {
			directiveName = prefix + " modelregion";
		}
		PrintTools.println("ACCParser is parsing ["+directiveName+"] directive", 3);
		addToMap("modelregion", "_directive");
		while (end_of_token() == false) 
		{
			String token = get_token();
			if( token.equals("") ) continue; //Skip empty string, which may occur due to macro.
			String clause = "aspen_" + token;
			if( token.equals(",") ) continue; //Skip comma between clauses, if existing.
			PrintTools.println("clause=" + clause, 3);
			try {
				switch (aspen_clauses.valueOf(clause)) {
				case aspen_label		:	parse_acc_stringargclause(token); break;
				default : ACCParserError("NoSuchASPENConstruct : " + clause);
				}
			} catch( Exception e) {
				ACCParserError("Error occured during parsing " + token + " clause.\n" + "Error message: " + e);
			}
		}
	}
	
	private static void parse_aspen_declare()
	{
		PrintTools.println("ACCParser is parsing [declare] directive", 3);
		addToMap("declare", "_directive");

		while (end_of_token() == false) 
		{
			String token = get_token();
			if( token.equals("") ) continue; //Skip empty string, which may occur due to macro.
			String clause = "aspen_" + token;
			if( token.equals(",") ) continue; //Skip comma between clauses, if existing.
			PrintTools.println("clause=" + clause, 3);
			try {
				switch (aspen_clauses.valueOf(clause)) {
				case aspen_param		:	parse_aspen_paramargsetclause(token); break;
				case aspen_data		:	parse_aspen_dataargsetclause(token); break;
				default : ACCParserError("NoSuchASPENConstruct : " + clause);
				}
			} catch( Exception e) {
				ACCParserError("Error occured during parsing " + token + " clause.\n" + "Error message: " + e);
			}
		}
	}
	
	private static void parse_aspen_control()
	{
		PrintTools.println("ACCParser is parsing [control] directive", 3);
		addToMap("control", "_directive");

		while (end_of_token() == false) 
		{
			String token = get_token();
			if( token.equals("") ) continue; //Skip empty string, which may occur due to macro.
			String clause = "aspen_" + token;
			if( token.equals(",") ) continue; //Skip comma between clauses, if existing.
			PrintTools.println("clause=" + clause, 3);
			try {
				switch (aspen_clauses.valueOf(clause)) {
				case aspen_loop		:	parse_acc_optionalconfclause(token); break;
				//case aspen_if		:	parse_acc_confclause(token); break;
				case aspen_if		:	parse_expressionlist(token); break;
				//case aspen_probability		:	parse_acc_confclause(token); break;
				case aspen_probability		:	parse_expressionlist(token); break;
				case aspen_ignore		:	parse_acc_noargclause(token); break;
				case aspen_parallelism		:	parse_aspen_resourceargclause(token); break;
				case aspen_execute		:	parse_acc_noargclause(token); break;
				case aspen_label		:	parse_acc_stringargclause(token); break;
				case aspen_flops		:	parse_aspen_resourceargsetclause(token); break;
				case aspen_loads		:	parse_aspen_resourceargsetclause(token); break;
				case aspen_stores		:	parse_aspen_resourceargsetclause(token); break;
				case aspen_messages		:	parse_aspen_resourceargsetclause(token); break;
				case aspen_intracomm		:	parse_aspen_resourceargsetclause(token); break;
				case aspen_allocates		:	parse_aspen_dataargsetclause(token); break;
				case aspen_resizes		:	parse_aspen_dataargsetclause(token); break;
				case aspen_frees		:	parse_aspen_dataargsetclause(token); break;
				default : ACCParserError("NoSuchASPENConstruct : " + clause);
				}
			} catch( Exception e) {
				ACCParserError("Error occured during parsing " + token + " clause.\n" + "Error message: " + e);
			}
		}
	}
	
	private static void parse_aspen_resourceargclause(String clause)
	{
		PrintTools.println("ACCParser is parsing ["+clause+"] clause", 3);
		match("(");
		String tok = null;
		ASPENResource tRSC = null;
		Expression sizeExp = null;
		List<ASPENTrait> traitList = null;
		String memsuffix = null;
		Expression ID = null;
		//Get size expression.
		sizeExp = parse_expression("(", ")", ":", 1);
		if( check(":") ) {
			eat();
			tok = get_token();
			if( ASPENResource.memSuffixSet.contains(tok) ) {
				//Get memsuffix. 
				memsuffix = tok;
				match("(");
				ID = parse_expression("(", ")", 1);
				match(")");
				if( check(":") ) {
					eat();
				}
				tok = get_token();
			} 
			if( tok.equals("traits") ) {
				//Get traitList. 
				match("(");
				parse_commaSeparatedASPENTraitList(traitList);
				match(")");
				match(")");
			} else if( !tok.equals(")") ) {
				ACCParserError("Unexpected token, " + tok + ", is found: " + clause);
			}
		} else {
			match(")");
		}
		//Generate ASPENResource.
		if( sizeExp != null ) {
			tRSC = new ASPENResource(sizeExp, traitList, memsuffix, ID);
		}
		if( tRSC == null ) {
			ACCParserError("No valid argument is found for the clause, " + clause);
		} else {
			addToMap(clause, tRSC);
		}
	}
	
	private static void parse_aspen_paramargsetclause(String clause)
	{
		PrintTools.println("ACCParser is parsing ["+clause+"] clause", 3);
		match("(");
		Set<ASPENParam> set = new HashSet<ASPENParam>();
		parse_commaSeparatedASPENParamSet(set);
		match(")");
		addToMap(clause, set);
	}
	
	private static void parse_aspen_dataargsetclause(String clause)
	{
		PrintTools.println("ACCParser is parsing ["+clause+"] clause", 3);
		match("(");
		Set<ASPENData> set = new HashSet<ASPENData>();
		parse_commaSeparatedASPENDataSet(set);
		match(")");
		addToMap(clause, set);
	}
	private static void parse_aspen_resourceargsetclause(String clause)
	{
		PrintTools.println("ACCParser is parsing ["+clause+"] clause", 3);
		match("(");
		Set<ASPENResource> set = new HashSet<ASPENResource>();
		parse_commaSeparatedASPENResourceSet(set);
		match(")");
		addToMap(clause, set);
	}

	/**
	 * This method parses input string into an Expression using ExpressionParser.parse() method.
	 * This does not consume closingTok.
	 * 
	 * @param openingTok
	 * @param closingTok
	 * @param initMatchCNT
	 * @return
	 */
	private static Expression parse_expression(String openingTok, String closingTok, int initMatchCNT)
	{
		//System.err.println("Enter parse_expression() type1");
		StringBuilder sb = new StringBuilder(32);
		int matchingCNT = initMatchCNT;
		int counter = 0;
		String tok = lookahead();
		if( tok.equals(closingTok) ) {
			//No valid token to parse
			return null;
		} else if( tok.equals(openingTok) ) {
			matchingCNT++;
		}
//		try {
			for (;;) {
				sb.append(get_token());
				tok = lookahead();
				if ( tok.equals("") ) {
					break; //end of token
				} else if ( tok.equals(closingTok) ) {
					--matchingCNT;
					if( matchingCNT < 0 ) {
						break;
					} else if( (initMatchCNT > 0) && (matchingCNT == 0) ) {
						break;
					}
				} else if ( tok.equals(openingTok) ) {
					matchingCNT++;
				} else if ( counter > infiniteLoopCheckTh ) {
					ACCParserError("Can't find stopping token ("+closingTok+")");
					System.exit(0);
				}
				counter++;
			}
/*		} catch( Exception e) {
			ACCParserError("unexpected error in parsing expression (Error: " + e + ")");
		}*/
		//System.out.println("expression to parse: " + sb.toString());
		return ExpressionParser.parse(sb.toString());
	}
	
	/**
	 * This method parses input string into an Expression using ExpressionParser.parse() method.
	 * This method assumes that stopTok is different from closingTok.
	 * This does not consume stopTok or closingTok.
	 * 
	 * @param openingTok
	 * @param closingTok
	 * @param stopTok
	 * @param initMatchCNT
	 * @return
	 */
	private static Expression parse_expression(String openingTok, String closingTok, String stopTok, int initMatchCNT)
	{
		//System.err.println("Enter parse_expression() type2");
		StringBuilder sb = new StringBuilder(32);
		int matchingCNT = initMatchCNT;
		int counter = 0;
		String tok = lookahead();
//		String tok = lookahead_debug();
		if( tok.equals(closingTok) || tok.equals(stopTok) ) {
			//No valid token to parse
			return null;
		} else if( tok.equals(openingTok) ) {
			matchingCNT++;
		}
//		try {
			for (;;) {
				sb.append(get_token());
				tok = lookahead();
				if ( tok.equals("") ) {
					break; //end of token
				} else if ( tok.equals(closingTok) ) {
					--matchingCNT;
					if( matchingCNT < 0 ) {
						break;
					} else if( (initMatchCNT > 0) && (matchingCNT == 0) ) {
						break;
					}
				} else if ( tok.equals(openingTok) ) {
					matchingCNT++;
				} else if ( tok.equals(stopTok) ) {
					if( tok.equals(":") ) {
						break;
					} else if( matchingCNT <= initMatchCNT ) {
						break;
					}
				} else if ( counter > infiniteLoopCheckTh ) {
					ACCParserError("Can't find stopping token ("+stopTok+")");
					System.exit(0);
				}
				counter++;
			}
/*		} catch( Exception e) {
			ACCParserError("unexpected error in parsing expression (Error: " + e + ")");
		}*/
		return ExpressionParser.parse(sb.toString());
	}
	
	/**
	 * This method parses input string into an Expression using ExpressionParser.parse() method.
	 * This method assumes that stopTok is different from closingTok.
	 * This does not consume stopTok or closingTok.
	 * 
	 * @param openingTok
	 * @param closingTok
	 * @param stopTokSet
	 * @param initMatchCNT
	 * @return
	 */
	private static Expression parse_expression(String openingTok, String closingTok, Set<String> stopTokSet, int initMatchCNT)
	{
		//System.err.println("Enter parse_expression() type3");
		StringBuilder sb = new StringBuilder(32);
		int matchingCNT = initMatchCNT;
		int counter = 0;
		String tok = lookahead();
		if( tok.equals(closingTok) ) {
			//No valid token to parse
			return null;
		} else if( tok.equals(openingTok) ) {
			matchingCNT++;
		} else {
			for( String stopTok : stopTokSet ) {
				if( tok.equals(stopTok) ) {
					return null;
				}
			}
		}
//		try {
			for (;;) {
				sb.append(get_token());
				tok = lookahead();
				if ( tok.equals("") ) {
					break; //end of token
				} else if ( tok.equals(closingTok) ) {
					--matchingCNT;
					if( matchingCNT < 0 ) {
						break;
					} else if( (initMatchCNT > 0) && (matchingCNT == 0) ) {
						break;
					}
				} else if ( tok.equals(openingTok) ) {
					matchingCNT++;
				} else if ( counter > infiniteLoopCheckTh ) {
					ACCParserError("Can't find stopping token: " + stopTokSet);
					System.exit(0);
				} else {
					boolean foundStopTok = false;
					for( String stopTok : stopTokSet ) {
						if( tok.equals(stopTok) ) {
							foundStopTok = true;
							break;
						}
					}
					if( foundStopTok ) {
						if( tok.equals(":") ) {
							break;
						} else if( matchingCNT <= initMatchCNT ) {
							break;
						}
					}

				}
				counter++;
			}
/*		} catch( Exception e) {
			ACCParserError("unexpected error in parsing expression (Error: " + e + ")");
		}*/
		return ExpressionParser.parse(sb.toString());
	}

	private static void parse_acc_noargclause(String clause)
	{
		PrintTools.println("ACCParser is parsing ["+clause+"] clause", 3);
		addToMap(clause, "_clause");
	}
	
	private static void parse_acc_optionaldirective(String directive)
	{
		PrintTools.println("ACCParser is parsing ["+directive+"] directive", 3);
		if( check("(") ) {
			match("(");
			Expression exp = parse_expression("(", ")", 1);
			match(")");
			if( exp == null ) {
				//addToMap(clause, "_clause");
				ACCParserError("No valid argument is found for the clause, " + directive);
			} else {
				addToMap(directive, exp);
			}
		} else {
			addToMap(directive, "_directive");
		}
	}

	private static void parse_acc_directivewithoptionalstringarg(String directive)
	{
		PrintTools.println("ACCParser is parsing ["+directive+"] directive", 3);
		if( check("(") ) {
			match("(");
			String str = get_token();
			match(")");
			if( str == null ) {
				//addToMap(clause, "_clause");
				ACCParserError("No valid argument is found for the clause, " + directive);
			} else {
				addToMap(directive, str);
			}
		} else {
			addToMap(directive, "_directive");
		}
	}

	private static void parse_acc_optionalconfclause(String clause)
	{
		PrintTools.println("ACCParser is parsing ["+clause+"] clause", 3);
		if( check("(") ) {
			match("(");
			Expression exp = parse_expression("(", ")", 1);
			match(")");
			if( exp == null ) {
				//addToMap(clause, "_clause");
				ACCParserError("No valid argument is found for the clause, " + clause);
			} else {
				addToMap(clause, exp);
			}
		} else {
			addToMap(clause, "_clause");
		}
	}

	private static void parse_acc_optionalconflistclause(String clause)
	{
		PrintTools.println("ACCParser is parsing ["+clause+"] clause", 3);
		if( check("(") ) {
			match("(");
			List<Expression> elist = new LinkedList<Expression>(); 
			parse_commaSeparatedExpressionList(elist);
			match(")");
			if( elist.isEmpty() ) {
				//addToMap(clause, "_clause");
				ACCParserError("No valid argument list is found for the clause, " + clause);
			} else {
				addToMap(clause, elist);
			}
		} else {
			addToMap(clause, "_clause");
		}
	}

	private static void parse_acc_confclause(String clause)
	{
		PrintTools.println("ACCParser is parsing ["+clause+"] clause", 3);
		match("(");
		Expression exp = parse_expression("(", ")", 1);
		match(")");
		if( exp == null ) {
			ACCParserError("No valid argument is found for the clause, " + clause);
		} else {
			addToMap(clause, exp);
		}
	}
	
	//Check gang, worker, and vector clauses in a parallel loop construct.
	private static void parse_acc_workshareconfclause(String clause)
	{
		PrintTools.println("ACCParser is parsing ["+clause+"] clause", 3);
		if( check("(") ) {
			ACCParserError("No argument is allowed for the clause, " + clause + ", in the parallel loop or routine construct.");
		} else {
			addToMap(clause, "_clause");
		}
	}
	
	private static void parse_acc_dataclause(String clause)
	{
		PrintTools.println("ACCParser is parsing ["+clause+"] clause", 3);
		match("(");
		Set<SubArray> set = new HashSet<SubArray>();
		parse_commaSeparatedSubArrayList(set, 0);
		match(")");
		addToMap(clause, set);
	}

	private static void parse_arc_windowclause(String clause)
	{
		PrintTools.println("ACCParser is parsing ["+clause+"] clause", 3);
		match("(");
		List<SubArray> slist = new LinkedList<SubArray>();
		parse_commaSeparatedSubArrayList(slist, 2);
		match(")");

                List<Object> wconflist = new ArrayList<Object>(2);
                wconflist.add(slist.get(0));
                wconflist.add(slist.get(1));

                addToMap(clause, wconflist);
	}
	
	private static void parse_arc_clause_with_subarrayconf(String clause)
	{
		PrintTools.println("ACCParser is parsing ["+clause+"] clause", 3);
		match("(");
		Set<SubArrayWithConf> set = new HashSet<SubArrayWithConf>();
		parse_commaSeparatedSubArrayWithConfList(set);
		match(")");
		addToMap(clause, set);
	}

	private static void parse_acc_declaredataclause(String clause)
	{
		PrintTools.println("ACCParser is parsing ["+clause+"] clause", 3);
		match("(");
		Set<SubArray> set = new HashSet<SubArray>();
		//DEBUG: OpenACC spec V1.0 section 2.11 says that subarrays are not allowed in declare
		//directives. However, without subarray syntax, size information of dynamically
		//allocated memory can not be described in the directives. Therefore, subarray syntax is
		//allowed in the declare directives too. (The subarray should be used only to give 
		//dimension information (ex: a[0:SIZE]))
		//parse_commaSeparatedVariableList(set);
		parse_commaSeparatedSubArrayList(set, 0);
		match(")");
		addToMap(clause, set);
	}
	
	private static void parse_acc_reduction(String clause)
	{
		PrintTools.println("ACCParser is parsing ["+clause+"] clause", 3);
		match("(");
        HashMap reduction_map = null;
        Set<SubArray> set = null;
        String op = null;
        ReductionOperator redOp = null;
        // Discover the kind of reduction operator (+, etc)
        if (check("+") || check("*") || check("max") || check("min") ||
            check("&") || check("|") || check("^") || check("&&") || check("||")) {
            op = get_token();
            redOp = ReductionOperator.fromString(op);
            PrintTools.println("reduction op:" + op, 3);
        } else {
            ACCParserError("Undefined reduction operator");
        }

        // check if there is already a reduction annotation with the same
        // operator in the set
        for (String ikey : (Set<String>)(acc_map.keySet())) {
            if (ikey.compareTo("reduction") == 0) {
                reduction_map = (HashMap)(acc_map.get(ikey));
                set = (Set<SubArray>)(reduction_map.get(redOp));
                break;
            }
        }
        if (reduction_map == null) {
            reduction_map = new HashMap(4);
        } 
        if (match(":") == false) {
            ACCParserError(
                    "colon expected before a list of reduction variables");
        }
        // When reduction_map is not null, set can be null for the given
        // reduction operator
        if (set == null) {
            set = new HashSet<SubArray>();
        }
        parse_commaSeparatedSubArrayList(set, 0);
        match(")");
        reduction_map.put(redOp, set);
        //addToMap("reduction", reduction_map);		
		if (acc_map.keySet().contains("reduction"))
		{
			Map orig_map = (Map)acc_map.get("reduction");
			for ( ReductionOperator nop : (Set<ReductionOperator>)reduction_map.keySet() )
			{
				Set new_set = (Set)reduction_map.get(nop);
				if (orig_map.keySet().contains(nop))
				{
					Set orig_set = (Set)orig_map.get(nop);
					orig_set.addAll(new_set);
				}
				else
				{
					orig_map.put(nop, new_set);
				}
			}
		}
		else
		{
			acc_map.put("reduction", reduction_map);
		}
	}
	
	private static void parse_acc_worksharetype(String clause)
	{
		PrintTools.println("ACCParser is parsing ["+clause+"] clause", 3);
		match("(");
		String str = get_token();
		if( str.equals(")") ) {
			addToMap(clause, "seq"); //default type if no argument is given.
		} else {
			if( ACCAnnotation.worksharingClauses.contains(str) || str.equals("seq") ) {
				addToMap(clause, str);
				match(")");
			} else {
				ACCParserError("No valid argument is found for the clause, " + clause);
			}
		}
	}
	
	private static void parse_acc_stringargclause(String clause)
	{
		PrintTools.println("ACCParser is parsing ["+ clause + "] clause", 3);
		match("(");
		String str = get_token();
		match(")");
		if( str.charAt(0) == '"' ) {
			StringLiteral p = new StringLiteral(str);
			p.stripQuotes();
			addToMap(clause, p.getValue());
		} else {
			addToMap(clause, str);
		}
	}

	private static void parse_acc_bindclause(String clause)
	{
		PrintTools.println("ACCParser is parsing ["+ clause + "] clause", 3);
		match("(");
		String str = lookahead();
		if( str.charAt(0) == '"' ) {
			str =  get_token();
			StringLiteral p = new StringLiteral(str);
			p.stripQuotes();
			addToMap(clause, p.getValue());
		} else {
			PrintTools.println("Input string for the bind argument: " + str, 3);
			Expression exp = parse_expression("(", ")", 1);
			if( exp == null ) {
				ACCParserError("No valid argument is found for the clause, " + clause);
			} else {
				addToMap(clause, exp);
			}

		}
		match(")");
	}

	private static void parse_acc_optionalstringargclause(String clause)
	{
		PrintTools.println("ACCParser is parsing ["+clause+"] clause", 3);
		if( check("(") ) {
			match("(");
			String str = get_token();
			match(")");
			if( str == null ) {
				//addToMap(clause, "_clause");
				ACCParserError("No valid argument is found for the clause, " + clause);
			} else {
				addToMap(clause, str);
			}
		} else {
			addToMap(clause, "_clause");
		}
	}

	private static void parse_conf_stringset(String clause)
	{
		PrintTools.println("ACCParser is parsing ["+clause+"] clause", 3);
		match("(");
		Set<String> set = new HashSet<String>();
		parse_commaSeparatedStringList(set);
		match(")");
		addToMap(clause, set);
	}
	
	private static void parse_conf_expressionset(String clause)
	{
		PrintTools.println("ACCParser is parsing ["+clause+"] clause", 3);
		match("(");
		Set<Expression> set = new HashSet<Expression>();
		parse_commaSeparatedExpressionList(set);
		match(")");
		addToMap(clause, set);
	}
	
	
	private static void parse_conf_expression(String clause)
	{
		PrintTools.println("ACCParser is parsing ["+clause+"] clause", 3);
		match("=");
		Expression exp = ExpressionParser.parse(get_token());
		if( exp == null ) {
			ACCParserError("No valid argument is found for the clause, " + clause);
		} else {
			addToMap(clause, exp);
		}
	}
	
	private static void parse_expressionlist(String clause)
	{
		PrintTools.println("ACCParser is parsing ["+clause+"] clause", 3);
		match("(");
		List<Expression> list = new LinkedList<Expression>();
		parse_commaSeparatedExpressionList(list);
		match(")");
		addToMap(clause, list);
	}
	
	/**
		*	This function reads a list of comma-separated string variables
		* It checks the right parenthesis to end the parsing, but does not consume it.
		*/
	private static void parse_commaSeparatedStringList(Collection<String> set)
	{
		for (;;) {
			String str = get_token();
			if( str.charAt(0) == '"' ) {
				StringLiteral p = new StringLiteral(str);
				p.stripQuotes();
				str = p.getValue();
			}
			set.add(str);
			if ( check(")") )
			{
				break;
			}
			else if ( match(",") == false )
			{
				ACCParserError("comma expected in comma separated list");
			}
		}
	}
	
	/**
		*	This function reads a list of comma-separated variables
		* It checks the right parenthesis to end the parsing, but does not consume it.
		*/
	private static void parse_commaSeparatedExpressionList(Set<Expression> set)
	{
		int initMatchCNT = 1;
		boolean firstItr = true;
//		try {
			for (;;) {
				String tok = lookahead();
				if( tok.equals(")") || tok.equals(",") ) {
					ACCParserError("valid list is missing in comma separated list");
				}
				//Expression exp = ExpressionParser.parse(tok);
				Expression exp = parse_expression("(", ")", ",", initMatchCNT);
				set.add(exp);
				if ( check(")") )
				{
					break;
				}
				else if ( match(",") == false )
				{
					ACCParserError("comma expected in comma separated list");
				}
				if( firstItr ) {
					initMatchCNT = 0;
					firstItr = false;
				}
			}
/*		} catch( Exception e) {
			ACCParserError("unexpected error in parsing comma-separated Expression list");
		}*/
	}
	
	/**
		*	This function reads a list of comma-separated variables
		* It checks the right parenthesis to end the parsing, but does not consume it.
		*/
	private static void parse_commaSeparatedExpressionList(List<Expression> list)
	{
		int initMatchCNT = 1;
		boolean firstItr = true;
//		try {
			for (;;) {
				String tok = lookahead();
				if( tok.equals(")") || tok.equals(",") ) {
					ACCParserError("valid list is missing in comma separated list");
				}
				//Expression exp = ExpressionParser.parse(tok);
				Expression exp = parse_expression("(", ")", ",", initMatchCNT);
				list.add(exp);
				if ( check(")") )
				{
					break;
				}
				else if ( match(",") == false )
				{
					ACCParserError("comma expected in comma separated list");
				}
				if( firstItr ) {
					initMatchCNT = 0;
					firstItr = false;
				}
			}
/*		} catch( Exception e) {
			ACCParserError("unexpected error in parsing comma-separated Expression list");
		}*/
	}
	
	/**
		*	This function reads a list of comma-separated variables
		* It checks the right angle bracket to end the parsing, but does not consume it.
		*/
	private static void parse_commaSeparatedConfList(List<Expression> list)
	{
		int initMatchCNT = 1;
		boolean firstItr = true;
//		try {
			for (;;) {
				String tok = lookahead();
				if( tok.equals("]") || tok.equals(",") ) {
					ACCParserError("valid list is missing in comma separated conf list");
				}
				//Expression exp = ExpressionParser.parse(tok);
				Expression exp = parse_expression("[", "]", ",", initMatchCNT);
				list.add(exp);
				if ( check("]") )
				{
					break;
				}
				else if ( match(",") == false )
				{
					ACCParserError("comma expected in comma separated conf list");
				}
				if( firstItr ) {
					initMatchCNT = 0;
					firstItr = false;
				}
			}
/*		} catch( Exception e) {
			ACCParserError("unexpected error in parsing comma-separated Expression list");
		}*/
	}
	
	
	/**
		*	This function reads a list of comma-separated variables and store
		*   them in SubArray.
		* It checks the right parenthesis to end the parsing, but does not consume it.
		*/
	private static void parse_commaSeparatedVariableList(Set<SubArray> set)
	{
		for (;;) {
			String tok = get_token();
			if( tok.equals(")") || tok.equals(",") ) {
				ACCParserError("valid list is missing in comma separated list");
			}
			Expression aName = ExpressionParser.parse(tok);
			if( aName instanceof SomeExpression ) {
				ACCParserError("Current implementation supports only simple scalar or array variables (but not class members) in OpenACC data clauses.");
			}
			SubArray subArr = new SubArray(aName);
			subArr.setPrintRange(false);
			set.add(subArr);
			if ( check(")") )
			{
				break;
			}
			else if ( match(",") == false )
			{
				ACCParserError("comma expected in comma separated list");
			}
		}
	}
	/**
		* This function reads a list of comma-separated subarrays
		* It checks the right parenthesis to end the parsing, but does not consume it.
		* 
		*/
	private static void parse_commaSeparatedSubArrayList(Collection<SubArray> collect, int limit)
	{
		String tok;
		int counter = 0;
//		try {
			for (;;) {
				tok = get_token();
				Expression aName = ExpressionParser.parse(tok);
				if( aName == null ) {
					ACCParserError("null is returned where SubArray name is expected!");
				}
				//"*" is special variable used in resilience ftdata clause.
				if( aName.toString().equals("(*)") ) {
					aName = new NameID("*");
				} else if( (aName instanceof SomeExpression) ) {
					ACCParserError("Current implementation supports only simple scalar or array variables (but not class members) in OpenACC data clauses.");
				}
				SubArray subArr = new SubArray(aName);
				if ( check(")") )
				{
					collect.add(subArr);
					counter++;
					break;
				}
				else if ( check("[") ) 
				{
					tok = lookahead();
					while( tok.equals("[") ) {
						eat();
						List<Expression> range = new ArrayList<Expression>(2);
						//Find start index expression
						tok = get_token();
						if( tok.equals(":") ) {
							range.add(0, new IntegerLiteral(0));
						} else {
							StringBuilder sb = new StringBuilder(32);
							int cnt1 = 0;
							while ( !tok.equals(":") && !tok.equals("]") ) {
								sb.append(tok);
								tok = get_token();
								cnt1++;
								if( cnt1 > infiniteLoopCheckTh ) {
									ACCParserError("Can't find : token in the Subarray expressions");
									System.exit(0);
								}
							}
							if( tok.equals(":") ) {
								range.add(0, ExpressionParser.parse(sb.toString()));
							} else if( tok.equals("]") ) {
								//AA[n] => AA[0:n]
								range.add(0, new IntegerLiteral(0));
								range.add(1, ExpressionParser.parse(sb.toString()));
							}
						}
						if( tok.equals(":") ) {
							//Find length expression
							tok = get_token();
							if( tok.equals("]") ) {
								range.add(1, null);
							} else {
								StringBuilder sb = new StringBuilder(32);
								int cnt1 = 0;
								while ( !tok.equals("]") ) {
									sb.append(tok);
									tok = get_token();
									cnt1++;
									if( cnt1 > infiniteLoopCheckTh ) {
										ACCParserError("Can't find ] token in the Subarray expressions");
										System.exit(0);
									}
								}
								range.add(1, ExpressionParser.parse(sb.toString()));
							}
						}
						subArr.addRange(range);
						tok = lookahead();
					}
					if( tok.equals(")") ) {
						collect.add(subArr);
						counter++;
						break;
					} else if( tok.equals(",") ) {
						collect.add(subArr);
						counter++;
						eat();
						if( (limit > 0) && (counter == limit) ) {
							break;
						}
					} else {
						ACCParserError("comma expected in comma separated list");
					}
				}
				else if ( check(",") )
				{
					collect.add(subArr);
					counter++;
					eat();
					if( (limit > 0) && (counter == limit) ) {
						break;
					}
				}
				else 
				{
					ACCParserError("comma expected in comma separated list");
				}
			}
/*		} catch( Exception e) {
			ACCParserError("unexpected error in parsing comma-separated SubArray list");
		}*/
	}
	
	/**
		* This function reads a list of comma-separated subarrays with confList
		* It checks the right parenthesis to end the parsing, but does not consume it.
		* 
		*/
	private static void parse_commaSeparatedSubArrayWithConfList(Set<SubArrayWithConf> set)
	{
		String tok;
//		try {
			for (;;) {
				tok = get_token();
				Expression aName = ExpressionParser.parse(tok);
				if( aName == null ) {
					ACCParserError("null is returned where SubArray name is expected!");
				}
				//"*" is special variable used in resilience ftdata clause.
				if( aName.toString().equals("(*)") ) {
					aName = new NameID("*");
				} else if( (aName instanceof SomeExpression) ) {
					ACCParserError("Current implementation supports only simple scalar or array variables (but not class members) in OpenACC data clauses.");
				}
				SubArray subArr = new SubArray(aName);
				SubArrayWithConf subArrConf = new SubArrayWithConf(subArr);
				tok = lookahead();
				if ( tok.equals(")") )
				{ //end of the comma-separated list.
					set.add(subArrConf);
					break;
				} else if ( tok.equals(":") ) 
				{
					eat();
					tok = lookahead();
					if( tok.equals(":") ) {
						while( tok.equals(":") ) {
							//found "::" operator.
							eat();
							match("[");
							List<Expression> tExpList = new LinkedList<Expression>();
							parse_commaSeparatedConfList(tExpList);
							match("]");
							subArrConf.addConfList(new ConfList(tExpList));
							tok = lookahead();
							if( tok.equals(":") ) {
								eat();
								tok = lookahead();
							} else {
								break;
							}
						}
					} else {
						ACCParserError(":: operator expected in subarry-with-conflist list");
					}
					if( tok.equals(")") ) {
						set.add(subArrConf);
						break;
					} else if( tok.equals(",") ) {
						set.add(subArrConf);
						eat();
					} else {
						ACCParserError("comma expected in comma separated list");
					}
				} else if ( tok.equals("[") )
				{
					//tok = lookahead();
					while( tok.equals("[") ) {
						eat();
						List<Expression> range = new ArrayList<Expression>(2);
						//Find start index expression
						tok = get_token();
						if( tok.equals(":") ) {
							range.add(0, new IntegerLiteral(0));
						} else {
							StringBuilder sb = new StringBuilder(32);
							int cnt1 = 0;
							while ( !tok.equals(":") && !tok.equals("]") ) {
								sb.append(tok);
								tok = get_token();
								cnt1++;
								if( cnt1 > infiniteLoopCheckTh ) {
									ACCParserError("Can't find : token in the Subarray expressions");
									System.exit(0);
								}
							}
							if( tok.equals(":") ) {
								range.add(0, ExpressionParser.parse(sb.toString()));
							} else if( tok.equals("]") ) {
								//AA[n] => AA[0:n]
								range.add(0, new IntegerLiteral(0));
								range.add(1, ExpressionParser.parse(sb.toString()));
							}
						}
						if( tok.equals(":") ) {
							//Find length expression
							tok = get_token();
							if( tok.equals("]") ) {
								range.add(1, null);
							} else {
								StringBuilder sb = new StringBuilder(32);
								int cnt1 = 0;
								while ( !tok.equals("]") ) {
									sb.append(tok);
									tok = get_token();
									cnt1++;
									if( cnt1 > infiniteLoopCheckTh ) {
										ACCParserError("Can't find ] token in the Subarray expressions");
										System.exit(0);
									}
								}
								range.add(1, ExpressionParser.parse(sb.toString()));
							}
						}
						subArr.addRange(range);
						tok = lookahead();
					}
					if( tok.equals(":") ) {
						eat();
						tok = lookahead();
						if( tok.equals(":") ) {
							while( tok.equals(":") ) {
								//found "::" operator.
								eat();
								match("[");
								List<Expression> tExpList = new LinkedList<Expression>();
								parse_commaSeparatedConfList(tExpList);
								match("]");
								subArrConf.addConfList(new ConfList(tExpList));
								tok = lookahead();
								if( tok.equals(":") ) {
									eat();
									tok = lookahead();
								} else {
									break;
								}
							}
						} else {
							ACCParserError(":: operator expected in subarry-with-conflist list");
						}
					} 
					if( tok.equals(")") ) {
						set.add(subArrConf);
						break;
					} else if( tok.equals(",") ) {
						set.add(subArrConf);
						eat();
					} else {
						ACCParserError("comma expected in comma separated list");
					}
				}
				else if ( tok.equals(",") )
				{
					set.add(subArrConf);
					eat();
				}
				else 
				{
					ACCParserError("comma expected in comma separated list");
				}
			}
/*		} catch( Exception e) {
			ACCParserError("unexpected error in parsing comma-separated SubArray list");
		}*/
	}
	
	private static void parse_commaSeparatedASPENTraitList(List<ASPENTrait> list) {
		String tok = null;
		ASPENTrait tTrait = null;
		String IDExp = null;
		List<Expression> argList = null;
		boolean inProgress = true;
//		try {
			while( inProgress ) {
				tTrait = null;
				IDExp = null;
				argList = new LinkedList<Expression>();
				tok = lookahead();
				if( tok.equals(")") ) {
					inProgress = false;
					break;
				} else if( tok.equals(",") ) {
					ACCParserError("Unexpected token, " + tok + ", is found");
				}
				//Get trait ID string
				IDExp = get_token();
				tok = lookahead();
				if( tok.equals("(") ) {
					eat();
					parse_commaSeparatedExpressionList(argList);
					match(")");
					tok = lookahead();
					if( tok.equals(")") ) {
						inProgress = false;
					} else if( tok.equals(",") ) {
						eat();
					}
				} else if( tok.equals(")") ){
					inProgress = false;
				} else if( tok.equals(",") ) {
					eat();
				}
				//Generate ASPENResource.
				if( IDExp != null ) {
					tTrait = new ASPENTrait(IDExp, argList);
				}
				if( tTrait == null ) {
					ACCParserError("No valid argument is found for the clause:");
				} else {
					list.add(tTrait);
				}
			}
/*		} catch( Exception e) {
			ACCParserError("unexpected error in parsing comma-separated ASPENTrait list");
		}*/
	}
	
	//[FIXME] This pass will fail if a function call with multiple arguments is included in the parameter init expression.
	private static void parse_commaSeparatedASPENParamSet(Set<ASPENParam> set) {
		String tok = null;
		ASPENParam tParam = null;
		Expression IDExp = null;
		Expression InitExp = null;
		Set<String> stopTokSet = new HashSet<String>();
		stopTokSet.add(":");
		stopTokSet.add(",");
		int cMatchCNT = 1;
		boolean isFirstItr = true;
		boolean inProgress = true;
//		try {
			while( inProgress ) {
				tParam = null;
				IDExp = null;
				InitExp = null;
				//Get ID expression.
				IDExp = parse_expression("(", ")", stopTokSet, cMatchCNT);
				if( isFirstItr ) {
					cMatchCNT = 0;
					isFirstItr = false;
				}
				tok = lookahead();
				if( tok.equals(":") ) {
					eat();
					tok = lookahead();
					if( tok.equals("{") ) {
						//Handle initial value list using SomeExpression for now.
						int ParenCnt = 1;
						StringBuilder sb = new StringBuilder(64);
						while( ParenCnt > 0 ) {
							sb.append(get_token());
							tok = lookahead();
							if( tok.equals("{") ) {
								ParenCnt++;
							} else if( tok.equals("}") ) {
								ParenCnt--;
							}
						}
						sb.append(get_token());
						InitExp = new SomeExpression(sb.toString(), new ArrayList<Traversable>(0));
					} else {
						InitExp = parse_expression("(", ")", ",", 0);
					}
					tok = lookahead();
					if( tok.equals(")") ) {
						inProgress = false;
					} else if( tok.equals(",") ) {
						eat();
					}
				} else if( tok.equals(")") ){
					inProgress = false;
				} else if( tok.equals(",") ) {
					eat();
				}
				//Generate ASPENResource.
				if( (IDExp != null) && (IDExp instanceof IDExpression) ) {
					tParam = new ASPENParam((IDExpression)IDExp, InitExp);
				}
				if( tParam == null ) {
					ACCParserError("No valid argument is found for the clause:");
				} else {
					set.add(tParam);
				}
			}
/*		} catch( Exception e) {
			ACCParserError("unexpected error in parsing comma-separated ASPENParam set");
		}*/
	}
	
	//[FIXME] This pass will fail if a function call with multiple arguments is included.
	private static void parse_commaSeparatedASPENDataSet(Set<ASPENData> set) {
		String tok = null;
		ASPENData tData = null;
		Expression IDExp = null;
		Expression CapExp = null;
		List<ASPENTrait> traitList = null;
		Set<String> stopTokSet = new HashSet<String>();
		stopTokSet.add(":");
		stopTokSet.add(",");
		int cMatchCNT = 1;
		boolean isFirstItr = true;
		boolean inProgress = true;
//		try {
			while( inProgress ) {
				tData = null;
				IDExp = null;
				CapExp = null;
				traitList = new LinkedList<ASPENTrait>();
				//Get ID expression.
				IDExp = parse_expression("(", ")", stopTokSet, cMatchCNT);
				if( isFirstItr ) {
					cMatchCNT = 0;
					isFirstItr = false;
				}
				tok = lookahead();
				if( tok.equals(":") ) {
					eat();
					tok = lookahead();
					boolean isConsumed = false;
					if( tok.equals("capacity") ) {
						isConsumed = true;
						eat();
						match("(");
						CapExp = parse_expression("(", ")", 1);
						match(")");
						tok = lookahead();
						if( tok.equals(")") ) {
							inProgress = false;
						} else if( tok.equals(":") || tok.equals(",") ) {
							eat();
							if( tok.equals(":") ) {
								isConsumed = false;
							}
						}
						tok = lookahead();
					} 
					if( tok.equals("traits") ) {
						isConsumed = true;
						eat();
						//Get traitList. 
						match("(");
						parse_commaSeparatedASPENTraitList(traitList);
						match(")");
						tok = lookahead();
						if( tok.equals(")") ) {
							inProgress = false;
						} else if( tok.equals(",") ) {
							eat();
						}
					}
					if( !isConsumed ) {
						ACCParserError("Unexpected token (" + tok + ") is found");
					}
				} else if( tok.equals(")") ){
					inProgress = false;
				} else if( tok.equals(",") ) {
					eat();
				}
				//Generate ASPENResource.
				if( (IDExp != null) && (IDExp instanceof IDExpression) ) {
					tData = new ASPENData((IDExpression)IDExp, CapExp, traitList);
				}
				if( tData == null ) {
					ACCParserError("No valid argument is found for the clause:");
				} else {
					set.add(tData);
				}
			}
/*		} catch( Exception e) {
			ACCParserError("unexpected error in parsing comma-separated ASPENData set");
		}*/
	}
	
	private static void parse_commaSeparatedASPENResourceSet(Set<ASPENResource> set) {
		String tok = null;
		ASPENResource tRSC = null;
		Expression sizeExp = null;
		List<ASPENTrait> traitList = null;
		String memsuffix = null;
		Expression ID = null;
		Set<String> stopTokSet = new HashSet<String>();
		stopTokSet.add(":");
		stopTokSet.add(",");
		int cMatchCNT = 1;
		boolean isFirstItr = true;
		boolean inProgress = true;
//		try {
			while( inProgress ) {
				tRSC = null;
				sizeExp = null;
				traitList = new LinkedList<ASPENTrait>();
				memsuffix = null;
				ID = null;
				//Get size expression.
				sizeExp = parse_expression("(", ")", stopTokSet, cMatchCNT);
				if( isFirstItr ) {
					cMatchCNT = 0;
					isFirstItr = false;
				}
				tok = lookahead();
				if( tok.equals(":") ) {
					eat();
					tok = lookahead();
					boolean isConsumed = false;
					if( ASPENResource.memSuffixSet.contains(tok) ) {
						isConsumed = true;
						//Get memsuffix. 
						memsuffix = get_token();
						match("(");
						ID = parse_expression("(", ")", 1);
						match(")");
						tok = lookahead();
						if( tok.equals(")") ) {
							inProgress = false;
						} else if( tok.equals(":") || tok.equals(",") ) {
							eat();
							if( tok.equals(":") ) {
								isConsumed = false;
							}
						}
						tok = lookahead();
					} 
					if( tok.equals("traits") ) {
						isConsumed = true;
						eat();
						//Get traitList. 
						match("(");
						parse_commaSeparatedASPENTraitList(traitList);
						match(")");
						tok = lookahead();
						if( tok.equals(")") ) {
							inProgress = false;
						} else if( tok.equals(",") ) {
							eat();
						}
					}
					if( !isConsumed ) {
						ACCParserError("Unexpected token (" + tok + ") is found");
					}
				} else if( tok.equals(")") ){
					inProgress = false;
				} else if( tok.equals(",") ) {
					eat();
				}
				//Generate ASPENResource.
				if( sizeExp != null ) {
					tRSC = new ASPENResource(sizeExp, traitList, memsuffix, ID);
				}
				if( tRSC == null ) {
					ACCParserError("No valid argument is found for the clause:");
				} else {
					set.add(tRSC);
				}
			}
/*		} catch( Exception e) {
			ACCParserError("unexpected error in parsing comma-separated ASPENResource set");
		}*/
	}

    /**
     * Parse NVL pragmas, which are stored as raw text after Cetus parsing.
     * 
     * @param input_map HashMap that will contain the parsed output pragmas
     * @param str_array input pragma string that will be parsed.
     * @return true if the pragma will be attached to the following non-pragma
     *         statement. Or, it returns false, if the pragma is stand-alone.
     */
	public static boolean parse_nvl_pragma(HashMap input_map, String [] str_array, HashMap<String, String>macro_map)
	{
		acc_map = input_map;
		token_array = str_array;
		token_index = 1; // "nvl" has already been matched
		//token_index = 2; // If there is a leading space, use this one.
		macroMap = macro_map;

		PrintTools.println(display_tokens(), 9);

		String token = get_token();
		String construct = "nvl_" + token;
		try {
			switch (nvl_directives.valueOf(construct)) {
			case nvl_atomic 		: parse_nvl_atomic(); return true;
			//		default : throw new NonOmpDirectiveException();
			default : ACCParserError("Not Supported Construct");
			}
		} catch( Exception e) {
			ACCParserError("Error occured during parsing " + token + " directive.\n" + "Error message: " + e);
		}
		return true;		// meaningless return because it is unreachable
	}
	
	private static void parse_nvl_atomic() {
		String directiveName = null;
		directiveName = "atomic";
		PrintTools.println("ACCParser is parsing ["+directiveName+"] directive", 3);
		addToMap("atomic", "_directive");
		while (end_of_token() == false) {
			String token = get_token();
			if( token.equals("") ) continue; //Skip empty string, which may occur due to macro.
			String clause = "nvl_" + token;
			if( token.equals(",") ) continue; //Skip comma between clauses, if existing.
			PrintTools.println("clause=" + clause, 3);
			try {
				switch (nvl_clauses.valueOf(clause)) {
				case nvl_heap  : parse_acc_confclause(token); break;
				case nvl_default : parse_acc_stringargclause(token); break;
				case nvl_backup : parse_acc_dataclause(token); break;
				case nvl_backup_writeFirst : parse_acc_dataclause(token); break;
				case nvl_clobber : parse_acc_dataclause(token); break;
				case nvl_readonly : parse_acc_dataclause(token); break;
				case nvl_mpiGroup : parse_acc_confclause(token); break;
				default : ACCParserError("NoSuchNVLConstruct : " + clause);
				}
			} catch( Exception e) {
				ACCParserError("Error occured during parsing " + token + " clause.\n" + "Error message: " + e);
			}
		}
	}
	
	private static void notSupportedWarning(String text)
	{
		System.out.println("Not Supported OpenACC Annotation: " + text); 
	}

	private static void ACCParserError(String text)
	{
		System.out.println("Syntax Error in OpenACC Directive Parsing: " + text);
		System.out.println(display_tokens());
		System.out.println("Exit the OpenACC translator!!");
		System.exit(1);
	}

	private static void addToMap(String key, String new_str)
	{
		if (acc_map.keySet().contains(key))
			Tools.exit("[ERROR] OpenACC Parser detected duplicate pragma: " + key);
		else
			acc_map.put(key, new_str);
	}
	
	private static void addToMap(String key, Expression expr)
	{
		if (acc_map.keySet().contains(key))
			Tools.exit("[ERROR] OpenACC Parser detected duplicate pragma: " + key);
		else
			acc_map.put(key, expr);
	}
	
	private static void addToMap(String key, ASPENResource tRSC)
	{
		if (acc_map.keySet().contains(key))
			Tools.exit("[ERROR] OpenACC Parser detected duplicate pragma: " + key);
		else
			acc_map.put(key, tRSC);
	}

	// When a key already exists in the map
	// clauses may be repeated as needed. 
	private static void addToMap(String key, Set new_set)
	{
		if (acc_map.keySet().contains(key))
		{
			Set set = (Set)acc_map.get(key);
			set.addAll(new_set);
		}
		else
		{
			acc_map.put(key, new_set);
		}
	}
	
	// When a key already exists in the map
	// clauses may be repeated as needed. 
	private static void addToMap(String key, List new_list)
	{
		if (acc_map.keySet().contains(key))
		{
			List list = (List)acc_map.get(key);
			list.addAll(new_list);
		}
		else
		{
			acc_map.put(key, new_list);
		}
	}

	private static void addToMap(String key, Map new_map)
	{
		if (acc_map.keySet().contains(key))
		{
			Map orig_map = (Map)acc_map.get(key);
			for ( String new_str : (Set<String>)new_map.keySet() )
			{
				Set new_set = (Set)new_map.get(new_str);
				if (orig_map.keySet().contains(new_str))
				{
					Set orig_set = (Set)orig_map.get(new_str);
					orig_set.addAll(new_set);
				}
				else
				{
					orig_map.put(new_str, new_set);
				}
			}
		}
		else
		{
			acc_map.put(key, new_map);
		}
	}
	
	public static enum arc_preprocessor
	{
		arc_define
	}

	public static enum acc_directives
	{
		acc_parallel, 
		acc_kernels,
		acc_loop, 
		acc_data,
		acc_host_data,
		acc_declare,
		acc_update,
		acc_cache,
		acc_wait,
		acc_routine,
		acc_barrier,
		acc_enter,
		acc_exit,
		acc_set,
		acc_mpi
	}
	
	public static enum arc_directives
	{
		arc_ainfo,
		arc_cuda, 
		arc_opencl, 
		arc_resilience,
		arc_ftregion,
		arc_ftinject,
		arc_profile,
		arc_enter,
		arc_exit,
		arc_transform,
		arc_impacc,
		arc_devicetask
	}

	public static enum acc_clause
	{
		acc_if,
		acc_async,
		acc_num_gangs,
		acc_num_workers,
		acc_vector_length,
		acc_reduction,
		acc_copy,
		acc_copyin,
		acc_copyout,
		acc_create,
		acc_present,
		acc_present_or_copy,
		acc_pcopy,
		acc_present_or_copyin,
		acc_pcopyin,
		acc_present_or_copyout,
		acc_pcopyout,
		acc_present_or_create,
		acc_pcreate,
		acc_deviceptr,
		acc_device_resident,
		acc_host,
		acc_self,
		acc_device,
		acc_private,
		acc_firstprivate,
		acc_use_device,
		acc_collapse,
		acc_gang,
		acc_worker,
		acc_vector,
		acc_seq,
		acc_independent,
		acc_bind,
		acc_nohost,
		acc_nowait,
		acc_type,
        acc_tile,
        acc_pipe,
        acc_pipein,
        acc_pipeout,
        acc_wait,
        acc_finalize,
        acc_delete,
        acc_default_async,
        acc_device_num,
        acc_device_type,
        acc_sendbuf,
        acc_recvbuf
	}
	
	public static enum ainfo_clause
	{
		token_procname,
		token_kernelid
	}
	
	public static enum cuda_clause
	{
		token_registerRO,
		token_registerRW,
		token_noregister,
		token_sharedRO,
		token_sharedRW,
		token_noshared,
		token_texture,
		token_notexture,
		token_constant,
		token_noconstant,
		token_global
	}

	public static enum opencl_clause
	{
		token_num_simd_work_items,
		token_num_compute_units
	}

	public static enum impacc_clause
	{
		token_ignoreglobal
	}
	
	public static enum transform_clause
	{
		token_noreductionunroll,
		token_noploopswap,
		token_noloopcollapse,
		token_multisrccg,
		token_multisrcgc,
		token_conditionalsrc,
		token_enclosingloops,
		token_permute,
		token_unroll,
		token_transpose,
		token_redim,
		token_expand,
		token_redim_transpose,
		token_expand_transpose,
		token_transpose_expand,
		token_window
	}
	
	public static enum resilience_clause
	{
		token_ftdata,
		token_ftcond,
		token_num_faults,
		token_num_ftbits,
		token_repeat,
		token_ftthread,
		token_ftkind,
		token_ftprofile,
		token_ftpredict
	}
	
	public static enum profile_clause
	{
		token_region,
		token_track,
		token_measure,
		token_mode,
		token_label,
		token_event,
		token_induction,
		token_verbosity,
		token_profcond
	}

	public static enum devicetask_clause
	{
		token_map,
		token_schedule
	}
	
	public static enum user_clause
	{
		token_if,
		token_async,
		token_wait,
		token_num_gangs,
		token_num_workers,
		token_vector_length,
		token_reduction,
		token_copy,
		token_copyin,
		token_copyout,
		token_create,
		token_present,
		token_present_or_copy,
		token_pcopy,
		token_present_or_copyin,
		token_pcopyin,
		token_present_or_copyout,
		token_pcopyout,
		token_present_or_create,
		token_pcreate,
		token_private,
		token_firstprivate,
		token_deviceptr,
		token_device_resident,
		token_host,
		token_device,
		token_collapse,
		token_gang,
		token_worker,
		token_vector,
		token_seq,
		token_independent,
		token_registerRO,
		token_registerRW,
		token_noregister,
		token_sharedRO,
		token_sharedRW,
		token_noshared,
		token_texture,
		token_notexture,
		token_constant,
		token_noconstant,
		token_global,
		token_noreductionunroll,
		token_noploopswap,
		token_noloopcollapse,
		token_multisrccg,
		token_multisrcgc,
		token_conditionalsrc,
		token_enclosingloops,
		token_permute,
		token_unroll,
		token_num_simd_work_items,
		token_num_compute_units
	}
	
	public static enum cuda_tuningconf
	{
		conf_defaultGOptionSet,
		conf_excludedGOptionSet,
		conf_gpuMemTrOptLevel,
		conf_gpuMallocOptLevel,
		conf_AccPrivatization,
		conf_AccReduction,
		conf_localRedVarConf,
		conf_defaultNumWorkersSet,
		conf_defaultNumComputeUnits,
		conf_maxNumGangsSet,
		conf_defaultNumSIMDWorkItems,
		conf_UEPRemovalOptLevel
	}
	
	public static enum aspen_directives
	{
		aspen_enter,
		aspen_exit,
		aspen_modelregion,
		aspen_declare,
		aspen_control
	}
	
	public static enum aspen_clauses
	{
		aspen_param,
		aspen_data,
		aspen_loop,
		aspen_if,
		aspen_probability,
		aspen_ignore,
		aspen_parallelism,
		aspen_execute,
		aspen_label,
		aspen_flops,
		aspen_loads,
		aspen_stores,
		aspen_messages,
		aspen_intracomm,
		aspen_allocates,
		aspen_resizes,
		aspen_frees
	}
	
	public static enum nvl_directives
	{
		nvl_atomic,
	}
	
	public static enum nvl_clauses
	{
		nvl_heap,
		nvl_default,
		nvl_backup,
		nvl_backup_writeFirst,
		nvl_clobber,
		nvl_readonly,
		nvl_mpiGroup,
	}


}
