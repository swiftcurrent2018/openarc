package cetus.analysis;

import cetus.hir.Expression;
import cetus.hir.IntegerLiteral;
import cetus.hir.NameID;
import cetus.hir.PrintTools;
import cetus.hir.SomeExpression;
import cetus.hir.Tools;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import openacc.analysis.ACCParser.ExpressionParser;
import openacc.analysis.SubArray;

/**
 * an OpenMP directive parser
 */
public class OmpParser {

    private static String [] token_array;
    private static int token_index;
    private static HashMap omp_map;
    private int debug_level;
	private static final int infiniteLoopCheckTh = 1024; //used to detect possible infinite loop due to incorrect parsing.

    public OmpParser() {
    }

    private static String get_token() {
        return token_array[token_index++];
    }

	// get one token but do not consume the token
	private static String lookahead()
	{
		if( end_of_token() ) {
			return "";
		} else {
			return token_array[token_index];
		}
	}

    // consume one token
    private static void eat() {
        token_index++;
    }

    // match a token with the given string
    private static boolean match(String istr) {
        boolean answer = check(istr);
        if (answer == false) {
            System.out.println("OmpParser Syntax Error: " + istr + " is expected");
			System.out.println("    Current token: " + lookahead());
			System.out.println("    Token index: " + token_index);
            System.out.println(display_tokens());
            System.exit(0);
        }
        token_index++;
        return answer;
    }

    // match a token with the given string, but do not consume a token
    private static boolean check(String istr) {
        if (end_of_token()) { 
            return false;
        }
        return (token_array[token_index].compareTo(istr) == 0) ? true : false;
    }    

    private static String display_tokens() {
        StringBuilder str = new StringBuilder(160);
        for (int i = 0; i < token_array.length; i++) {
            str.append("token_array[").append(i).append("] = ");
            str.append(token_array[i]).append(PrintTools.line_sep);
        }
        return str.toString();
    }

    private static boolean end_of_token() {
        return (token_index >= token_array.length) ? true : false;
    }

    // returns TRUE, if the omp pragma will be attached to the following
    // non-pragma statement. Or, it returns false, if the pragma is a
    // standalone pragma
    public static boolean
    parse_omp_pragma(HashMap input_map, String [] str_array) {
    	omp_map = input_map;
    	token_array = str_array;
    	token_index = 3;    // "#", "pragma", "omp" have already been matched
    	PrintTools.println(display_tokens(), 9);
    	String construct = "omp_" + get_token();
    	switch (omp_pragma.valueOf(construct)) {
    	case omp_parallel       : parse_omp_parallel();     return true;
    	case omp_for            : parse_omp_for();          return true;
    	case omp_sections       : parse_omp_sections();     return true;
    	case omp_section        : parse_omp_section();      return true;
    	case omp_single         : parse_omp_single();       return true;
    	case omp_task           : parse_omp_task();         return true;
    	case omp_master         : parse_omp_master();       return true;
    	case omp_critical       : parse_omp_critical();     return true;
    	case omp_barrier        : parse_omp_barrier();      return false;
    	case omp_taskwait       : parse_omp_taskwait();     return false;
    	case omp_atomic         : parse_omp_atomic();       return true;
    	case omp_flush          : parse_omp_flush();        return false;
    	case omp_ordered        : parse_omp_ordered();      return true;
    	case omp_threadprivate  : parse_omp_threadprivate();return false;
    	case omp_teams          : parse_omp_teams();        return true;
    	case omp_distribute     : parse_omp_distribute();   return true;
    	case omp_simd           : parse_omp_simd();	        return true;
    	case omp_declare        :
    		boolean declare_ret;
    		if(check("simd"))
    			declare_ret = true;
    		else
    			declare_ret = false;
    		parse_omp_declare();
    		return declare_ret;
    	case omp_end            : parse_omp_end_declare();  return false;
    	case omp_target         :
    		boolean target_ret;
    		if(check("update") || check("enter") || check("exit"))
    			target_ret = false;
    		else
    			target_ret = true;
    		parse_omp_target();
    		return target_ret;
    	case omp_taskgroup      : parse_omp_taskgroup();    return true;
    	case omp_cancel         : parse_omp_cancel();       return false;
    	case omp_cancellation   : parse_omp_cancellation_point(); return false;
    	default                 : OmpParserError("Not Supported Construct" + construct);
    	}
    	return true;        // meaningless return because it is unreachable
    }

    /** ---------------------------------------------------------------
      *        2.4 parallel Construct
      *
      *        #pragma omp parallel [clause[[,] clause]...] new-line
      *            structured-block
      *
      *        where clause is one of the following
      *            if(scalar-expression)
      *            num_threads(integer-expression)
      *            default(shared|none)
      *            private(list)
      *            firstprivate(list)
      *            shared(list)
      *            copyin(list)
      *            reduction(operator:list)
      * --------------------------------------------------------------- */
    private static void parse_omp_parallel() {
        PrintTools.println("OmpParser is parsing [parallel] clause", 2);
        addToMap("parallel", "true");
        if (check("for")) {
            eat();
            parse_omp_parallel_for();
        } else if (check("sections")) {
            eat();
            parse_omp_parallel_sections();
        } else {
            while (end_of_token() == false) {
            	String tok = get_token();
            	if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
            	if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
                String clause = "token_" + tok;
                PrintTools.println("clause=" + clause, 2);
                switch (omp_clause.valueOf(clause)) {
                    case token_if           : parse_omp_if();           break;
                    case token_num_threads  : parse_omp_num_threads();  break;
                    case token_default      : parse_omp_default();      break;
                    case token_private      : parse_omp_private();      break;
                    case token_firstprivate : parse_omp_firstprivate(); break;
                    case token_shared       : parse_omp_shared();       break;
                    case token_copyin       : parse_omp_copyin();       break;
                    case token_reduction    : parse_omp_reduction();    break;
                    default                 :
                        OmpParserError("Not supported clause : " + clause);
                }
            }
        }
    }

    /** ---------------------------------------------------------------
      *        2.5 Worksharing Constructs
      *        OpenMP defines the following worksharing constructs
      *        - loop, sections, single, workshare(FORTRAN only) construct
      * --------------------------------------------------------------- */
    /** ---------------------------------------------------------------
      *        2.5.1 Loop Construct
      *
      *        #pragma omp for [clause[[,] clause]...] new-line
      *            for-loops
      *        where clause is one of the following
      *            private(list)
      *            firstprivate(list)
      *            lastprivate(list)
      *            reduction(operator:list)
      *            schedule(kind[, chunk_size])
      *            collapse(n)
      *            ordered
      *            nowait
      * --------------------------------------------------------------- */
    private static void parse_omp_for() {
        PrintTools.println("OmpParser is parsing [for] clause", 2);
        addToMap("for", "true");
        if (check("simd")) {
            eat();
            parse_omp_for_simd();
        } else {
            while (end_of_token() == false) {
            	String tok = get_token();
            	if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
            	if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
                String clause = "token_" + tok;
                switch (omp_clause.valueOf(clause)) {
                    case token_private      : parse_omp_private();      break;
                    case token_shared       : parse_omp_shared();       break;
                    case token_firstprivate : parse_omp_firstprivate(); break;
                    case token_lastprivate  : parse_omp_lastprivate();  break;
                    case token_reduction    : parse_omp_reduction();    break;
                    case token_schedule     : parse_omp_schedule();     break;
                    case token_collapse     : parse_omp_collapse();     break;
                    case token_ordered      : parse_omp_ordered();      break;
                    case token_nowait       : parse_omp_nowait();       break;
                    default : OmpParserError("Not supported clause: " + tok);
                }
            }
        }
    }

    /** ---------------------------------------------------------------
      *        2.5.2 sections Construct
      *
      *        #pragma omp sections [clause[[,] clause]...] new-line
      *        {
      *            [#pragma omp section new-line]
      *                structured-block
      *            [#pragma omp section new-line
      *                structured-block]
      *        }
      *        where clause is one of the following
      *            private(list)
      *            firstprivate(list)
      *            lastprivate(list)
      *            reduction(operator:list)
      *            nowait
      * --------------------------------------------------------------- */
    private static void parse_omp_sections() {
        PrintTools.println("OmpParser is parsing [sections] clause", 2);
        addToMap("sections", "true");
        while (end_of_token() == false) {
            String tok = get_token();
            if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
            if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
            String clause = "token_" + tok;
            switch (omp_clause.valueOf(clause)) {
                case token_private      : parse_omp_private();      break;
                case token_firstprivate : parse_omp_firstprivate(); break;
                case token_lastprivate  : parse_omp_lastprivate();  break;
                case token_reduction    : parse_omp_reduction();    break;
                case token_nowait       : parse_omp_nowait();       break;
                default : OmpParserError("Not supported clause: " + tok);
            }
        }
    }

    private static void parse_omp_section() {
        PrintTools.println("OmpParser is parsing [section] clause", 2);
        addToMap("section", "true");
    }

    /** ---------------------------------------------------------------
      *        2.5.3 single Construct
      *
      *        #pragma omp single [clause[[,] clause]...] new-line
      *                structured-block
      *
      *        where clause is one of the following
      *            private(list)
      *            firstprivate(list)
      *            copyprivate(list)
      *            nowait
      * --------------------------------------------------------------- */
    private static void parse_omp_single() {
        PrintTools.println("OmpParser is parsing [single] clause", 2);
        addToMap("single", "true");
        while (end_of_token() == false) {
            String tok = get_token();
            if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
            if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
            String clause = "token_" + tok;
            switch (omp_clause.valueOf(clause)) {
                case token_private      : parse_omp_private();      break;
                case token_firstprivate : parse_omp_firstprivate(); break;
                case token_copyprivate  : parse_omp_copyprivate();  break;
                case token_nowait       : parse_omp_nowait();       break;
                default : OmpParserError("Not supported clause: " + tok);
            }
        }
    }

    /** ---------------------------------------------------------------
      *        2.6 Combined Parallel Worksharing Constructs
      *
      *        2.6.1 parallel loop Construct
      *
      *        #pragma omp parallel for [clause[[,] clause]...] new-line
      *                for-loop
      *        
      *        where clause can be any of the clauses accepted by the parallel
      *        or for directives, except the nowait clause, with identical
      *        meanings and restrictions
      * --------------------------------------------------------------- */
    private static void parse_omp_parallel_for() {
        PrintTools.println("OmpParser is parsing [parallel for] clause", 2);
        addToMap("for", "true");
        if (check("simd")) {
            eat();
            parse_omp_parallel_for_simd();
        } else {
            while (end_of_token() == false) {
            	String tok = get_token();
            	if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
            	if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
                String clause = "token_" + tok;
                switch (omp_clause.valueOf(clause)) {
                    case token_if           : parse_omp_if();           break;
                    case token_num_threads  : parse_omp_num_threads();  break;
                    case token_default      : parse_omp_default();      break;
                    case token_private      : parse_omp_private();      break;
                    case token_firstprivate : parse_omp_firstprivate(); break;
                    case token_shared       : parse_omp_shared();       break;
                    case token_copyin       : parse_omp_copyin();       break;
                    case token_reduction    : parse_omp_reduction();    break;
                    case token_lastprivate  : parse_omp_lastprivate();  break;
                    case token_schedule     : parse_omp_schedule();     break;
                    case token_collapse     : parse_omp_collapse();     break;
                    case token_ordered      : parse_omp_ordered();      break;
                    default : OmpParserError("Not supported clause: " + tok);
                }
            }
        }
    }

    private static void parse_omp_parallel_for_simd() {
        PrintTools.println("OmpParser is parsing [simd] clause", 2);
        addToMap("simd", "true");

        while (end_of_token() == false) {
            String tok = get_token();
            if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
            if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
            String clause = "token_" + tok;
            switch (omp_clause.valueOf(clause)) {
                case token_if           : parse_omp_if();           break;
                case token_num_threads  : parse_omp_num_threads();  break;
                case token_default      : parse_omp_default();      break;
                case token_private      : parse_omp_private();      break;
                case token_firstprivate : parse_omp_firstprivate(); break;
                case token_shared       : parse_omp_shared();       break;
                case token_copyin       : parse_omp_copyin();       break;
                case token_reduction    : parse_omp_reduction();    break;
                case token_lastprivate  : parse_omp_lastprivate();  break;
                case token_schedule     : parse_omp_schedule();     break;
                case token_collapse     : parse_omp_collapse();     break;
                case token_ordered      : parse_omp_ordered();      break;
                case token_safelen      : parse_omp_safelen();      break;
                case token_linear       : parse_omp_linear();       break;
                case token_aligned      : parse_omp_aligned();      break;
                default : OmpParserError("Not supported clause: " + tok);
            }
        }

    }

    /** ---------------------------------------------------------------
      *        2.6.2 parallel sections Construct
      *
      *        #pragma omp sections [clause[[,] clause]...] new-line
      *        {
      *            [#pragma omp section new-line]
      *                structured-block
      *            [#pragma omp section new-line
      *                structured-block]
      *        }
      *        
      *        where clause can be any of the clauses accepted by the parallel
      *        or sections directives, except the nowait clause, with identical
      *        meanings and restrictions
      * --------------------------------------------------------------- */
    private static void parse_omp_parallel_sections() {
        PrintTools.println("OmpParser is parsing [parallel sections] clause",2);
        while (end_of_token() == false) {
            String tok = get_token();
            if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
            if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
            String clause = "token_" + tok;
            addToMap("sections", "true");
            switch (omp_clause.valueOf(clause)) {
                case token_if           : parse_omp_if();           break;
                case token_num_threads  : parse_omp_num_threads();  break;
                case token_default      : parse_omp_default();      break;
                case token_private      : parse_omp_private();      break;
                case token_firstprivate : parse_omp_firstprivate(); break;
                case token_shared       : parse_omp_shared();       break;
                case token_copyin       : parse_omp_copyin();       break;
                case token_lastprivate  : parse_omp_lastprivate();  break;
                case token_reduction    : parse_omp_reduction();    break;
                default : OmpParserError("Not supported clause: " + tok);
            }
        }
    }

    /** ---------------------------------------------------------------
      *        2.7 task Construct
      *
      *        #pragma omp task [clause[[,] clause]...] new-line
      *            structured-block
      *
      *        where clause is one of the following
      *            if(scalar-expression)
      *            untied
      *            default(shared|none)
      *            private(list)
      *            firstprivate(list)
      *            shared(list)
      * --------------------------------------------------------------- */
    private static void parse_omp_task() {
        PrintTools.println("OmpParser is parsing [task] clause", 2);
        addToMap("task", "true");
        while (end_of_token() == false) {
            String tok = get_token();
            if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
            if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
            String clause = "token_" + tok; 
            switch (omp_clause.valueOf(clause)) {
                case token_if           : parse_omp_if();           break;
                case token_untied       : parse_omp_untied();       break;
                case token_default      : parse_omp_default();      break;
                case token_private      : parse_omp_private();      break;
                case token_firstprivate : parse_omp_firstprivate(); break;
                case token_shared       : parse_omp_shared();       break;
                case token_depend       : parse_omp_depend();       break;
                default : OmpParserError("Not supported clause: " + tok);
            }
        }
    }

    /** ---------------------------------------------------------------
      *        2.8 Master and Synchronization Construct
      *
      *        -    master/critical/barrier/taskwait/atomic/flush/ordered
      *
      *        2.8.1 master Construct
      *
      *        #pragma omp master new-line
      *            structured-block
      *
      * --------------------------------------------------------------- */
    private static void parse_omp_master() {
        PrintTools.println("OmpParser is parsing [master] clause", 2);
        addToMap("master", "true");
    }

    private static void parse_omp_critical() {
        PrintTools.println("OmpParser is parsing [critical] clause", 2);
        String name = null;
        if (end_of_token() == false) {
            match("(");
            name = new String(get_token());
            match(")");
        }
        addToMap("critical", name);
    }

    private static void parse_omp_barrier() {
        PrintTools.println("OmpParser is parsing [barrier] clause", 2);
        addToMap("barrier", "true");
    }

    private static void parse_omp_taskwait() {
        PrintTools.println("OmpParser is parsing [taskwait] clause", 2);
        addToMap("taskwait", "true");
    }

    private static void parse_omp_taskgroup() {
        PrintTools.println("OmpParser is parsing [taskgroup] clause", 2);
        addToMap("taskgroup", "true");
    }

    private static void parse_omp_atomic() {
        PrintTools.println("OmpParser is parsing [atomic] clause", 2);
        addToMap("atomic", "true");
	if(check("read") || check("write") || check("update") || check("capture"))
	{
            addToMap("type", new String(get_token()));
	}
    }

    private static void parse_omp_cancel() {
        PrintTools.println("OmpParser is parsing [cancel] clause", 2);
        if(check("parallel") || check("sections") || check("for") || check("taskgroup"))
        {
            addToMap("cancel", new String(get_token()));
            while (end_of_token() == false) {
                String clause = "token_" + get_token();
                switch (omp_clause.valueOf(clause)) {
                    case token_if      : parse_omp_if();      break;
                    default : OmpParserError("Not supported clause: " + clause);
                }
            }
        } else
        {
            OmpParserError("Not supported clause: " + lookahead());
        }
    }

    private static void parse_omp_cancellation_point() {
        PrintTools.println("OmpParser is parsing [cancellation point] clause", 2);
        if(check("point")) {
            eat();
            if (check("parallel") || check("sections") || check("for") || check("taskgroup")) {
                addToMap("cancellation point", new String(get_token()));
            } else {
                OmpParserError("Not supported clause: " + lookahead());
            }
        } else {
            OmpParserError("Not supported clause: " + lookahead());
        }
    }

    private static void parse_omp_flush() {
        PrintTools.println("OmpParser is parsing [flush] clause", 2);
        Set set = new HashSet<String>();
        if (end_of_token() == false) {
            match("(");
            parse_commaSeparatedList(set);
            match(")");
        }
        addToMap("flush", set);
    }

    private static void parse_omp_ordered() {
        PrintTools.println("OmpParser is parsing [ordered] clause", 2);
        addToMap("ordered", "true");
    }

    /** ---------------------------------------------------------------
      *        2.9 Data Environment
      *
      *        2.9.1 read the specification
      *
      *        2.9.2 threadprivate Directive
      *
      * --------------------------------------------------------------- */
    /**
      * threadprivate(x) needs special handling: it should be a global
      * information that every parallel region has to be annotated as private(x)
      */
    private static void parse_omp_threadprivate() {
        PrintTools.println("OmpParser is parsing [threadprivate] clause", 2);
        match("(");
        Set set = new HashSet<String>();
        parse_commaSeparatedList(set);
        match(")");
        addToMap("threadprivate", set);
    }

    /** ---------------------------------------------------------------
     *        2.4 declare Construct
     *
     *        #pragma omp declare simd
     *        #pragma omp declare target
     *        #pragma omp declare target (extended-list)
     *        #pragma omp declare target to (extended-list)
     *        #pragma omp declare target link (list)
     *        
     * --------------------------------------------------------------- */
    private static void parse_omp_declare() {
        PrintTools.println("OmpParser is parsing [declare] clause", 2);
        if (check("simd")) {
            addToMap("declare", "simd");
            eat();
            parse_omp_declare_simd();
        } else if (check("target")) {
            addToMap("declare", "target");
            eat();
            parse_omp_declare_target();
        }else {
            OmpParserError("Not supported clause : " + lookahead());
        }
    }

    private static void parse_omp_end_declare() {
        PrintTools.println("OmpParser is parsing [end declare] clause", 2);
        if(check("declare"))
        {
            addToMap("end", "true");
            eat();
            if(check("target"))
            {
                addToMap("declare", "target");
                eat();
            } else {
                OmpParserError("Not supported clause : " + lookahead());
            }
        }else {
            OmpParserError("NoSuchParallelConstruct : " + lookahead());
        }
    }

    private static void parse_omp_declare_target() {
        PrintTools.println("OmpParser is parsing [declare target] clause", 2);
        if( check("(") ) {
        	parse_omp_to();
        }
        while (end_of_token() == false) {
            String tok = get_token();
            if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
            if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
            String clause = "token_" + tok;
            switch (omp_clause.valueOf(clause)) {
                case token_to           : parse_omp_to();           break;
                case token_link         : parse_omp_link();         break;
                default : OmpParserError("Not supported clause: " + tok);
            }
        }

    }

    /** ---------------------------------------------------------------
      *        2.8 SIMD Constructs
	  *           The simd construct can be applied to a loop to indicate 
	  *			  that the loop can be transformed into a SIMD loop
      * --------------------------------------------------------------- */
    /** ---------------------------------------------------------------
      *	        #pragma omp simd [clause[ [, ]clause] ...]
      *	            for-loops
      *	        clause:
      *	            safelen(length)
      *	            linear(list[:linear-step])
      *             aligned(list[:alignment])
      *             private(list)
      *             lastprivate(list)
      *             reduction(reduction-identifier: list)
      *             collapse(n)
      * --------------------------------------------------------------- */
    private static void parse_omp_simd() {
        PrintTools.println("OmpParser is parsing [simd] clause", 2);
        addToMap("simd", "true");
        while (end_of_token() == false) {
            String tok = get_token();
            if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
            if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
            String clause = "token_" + tok;
            switch (omp_clause.valueOf(clause)) {
                case token_safelen      : parse_omp_safelen();      break;
                case token_linear       : parse_omp_linear();       break;
                case token_aligned      : parse_omp_aligned();      break;
                case token_private      : parse_omp_private();      break;
                case token_lastprivate  : parse_omp_lastprivate();  break;
                case token_reduction    : parse_omp_reduction();    break;
                case token_collapse     : parse_omp_collapse();     break;
                default : OmpParserError("Not supported clause: " + tok);
            }
        }
    }

    /** ---------------------------------------------------------------
     *	        #pragma omp declare simd [clause[ [, ]clause] ...]
     *	            for-loops
     *	        clause:
     *	            safelen(length)
     *	            linear(list[:linear-step])
     *              aligned(list[:alignment])
     *              uniform(argument-list)
     *              inbranch
     *              notinbranch
     * --------------------------------------------------------------- */
    private static void parse_omp_declare_simd() {
        PrintTools.println("OmpParser is parsing [simd] clause", 2);
        addToMap("simd", "true");
        while (end_of_token() == false) {
            String tok = get_token();
            if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
            if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
            String clause = "token_" + tok;
            switch (omp_clause.valueOf(clause)) {
                case token_safelen      : parse_omp_safelen();      break;
                case token_linear       : parse_omp_linear();       break;
                case token_aligned      : parse_omp_aligned();      break;
                case token_uniform      : parse_omp_uniform();      break;
                case token_inbranch     : parse_omp_inbranch();     break;
                case token_notinbranch  : parse_omp_notinbranch();  break;
                default : OmpParserError("Not supported clause: " + tok);
            }
        }
    }

    /** ---------------------------------------------------------------
     *      #pragma omp for simd [clause[ [, ]clause] ...]
     *          for-loops
     *      clause:
     *          Any accepted by the simd or for directives with
     *          identical meanings and restrictions.
     * --------------------------------------------------------------- */
    private static void parse_omp_for_simd() {
        PrintTools.println("OmpParser is parsing [simd] clause", 2);
        addToMap("simd", "true");
        while (end_of_token() == false) {
            String tok = get_token();
            if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
            if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
            String clause = "token_" + tok;
            switch (omp_clause.valueOf(clause)) {
                case token_safelen      : parse_omp_safelen();      break;
                case token_linear       : parse_omp_linear();       break;
                case token_aligned      : parse_omp_aligned();      break;
                case token_private      : parse_omp_private();      break;
                case token_lastprivate  : parse_omp_lastprivate();  break;
                case token_reduction    : parse_omp_reduction();    break;
                case token_collapse     : parse_omp_collapse();     break;
                case token_shared       : parse_omp_shared();       break;
                case token_firstprivate : parse_omp_firstprivate(); break;
                case token_schedule     : parse_omp_schedule();     break;
                case token_ordered      : parse_omp_ordered();      break;
                case token_nowait       : parse_omp_nowait();       break;
                default : OmpParserError("Not supported clause: " + tok);
            }
        }
    }

    /** ---------------------------------------------------------------
     *        2.8 Target Constructs
     *           These constructs create a device data environment for the
     *           extent of the region. target also starts execution on the
     *           device.
     * --------------------------------------------------------------- */
    /** ---------------------------------------------------------------
     *       #pragma omp target [clause[ [, ]clause] ...]
     *          structured-block
     *       clause:
     *          device(integer-expression)
     *          map([map-type: ] list)
     *          if(scalar-expression)
     *          private(list)
     *          firstprivate(list)
     *          is_device_ptr(list)
     *          depend(dependence-type: list)
     *          nowait
     * --------------------------------------------------------------- */
    private static void parse_omp_target() {
        PrintTools.println("OmpParser is parsing [target] clause", 2);
        addToMap("target", "true");
        if (check("enter")) {
            eat();
            match("data");
            parse_omp_target_enter_data();
        } else if (check("exit")) {
            eat();
            match("data");
            parse_omp_target_exit_data();
        } else if (check("data")) {
            eat();
            parse_omp_target_data();
        } else if   (check("update")) {
            eat();
            parse_omp_target_update();
        } else if   (check("teams")) {
            eat();
            parse_omp_target_teams();
        } else if   (check("parallel")) {
            eat();
            parse_omp_target_parallel();
        } else if   (check("simd")) {
            eat();
            parse_omp_target_simd();
        } else {
            while (end_of_token() == false) {
            	String tok = get_token();
            	if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
            	if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
                String clause = "token_" + tok;
                switch (omp_clause.valueOf(clause)) {
                    case token_device       : parse_omp_device();       break;
                    case token_map          : parse_omp_map();          break;
                    case token_if           : parse_omp_if();           break;
                    case token_private      : parse_omp_private();      break;
                    case token_firstprivate : parse_omp_firstprivate(); break;
                    case token_is_device_ptr : parse_omp_is_device_ptr(); break;
                    case token_depend       : parse_omp_depend();       break;
                    case token_nowait		: parse_omp_nowait();		break;
                    default : OmpParserError("Not supported clause: " + tok);
                }
            }
        }
    }

    private static void parse_omp_target_update() {
        PrintTools.println("OmpParser is parsing [update] clause", 2);
        addToMap("update", "true");

        while (end_of_token() == false) {
            String tok = get_token();
            if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
            if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
            String clause = "token_" + tok;
            switch (omp_clause.valueOf(clause)) {
                case token_device       : parse_omp_device();       break;
                case token_if           : parse_omp_if();           break;
                case token_to           : parse_omp_to();           break;
                case token_from         : parse_omp_from();         break;
                case token_depend       : parse_omp_depend();       break;
                case token_nowait		: parse_omp_nowait();		break;
                default : OmpParserError("Not supported clause: " + tok);
            }
        }

    }

    private static void parse_omp_target_teams() {
        PrintTools.println("OmpParser is parsing [teams] clause", 2);
        addToMap("teams", "true");
        if (check("distribute")) {
            eat();
            parse_omp_target_teams_distribute();
        } else {
            while (end_of_token() == false) {
            	String tok = get_token();
            	if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
            	if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
                String clause = "token_" + tok;
                switch (omp_clause.valueOf(clause)) {
                    case token_device       : parse_omp_device();       break;
                    case token_map          : parse_omp_map();          break;
                    case token_if           : parse_omp_if();           break;
                    case token_num_teams    : parse_omp_num_teams();    break;
                    case token_thread_limit : parse_omp_thread_limit(); break;
                    case token_default      : parse_omp_default();      break;
                    case token_private      : parse_omp_private();      break;
                    case token_firstprivate : parse_omp_firstprivate(); break;
                    case token_shared       : parse_omp_shared();       break;
                    case token_reduction    : parse_omp_reduction();    break;
                    case token_collapse     : parse_omp_collapse();     break;
                    case token_is_device_ptr : parse_omp_is_device_ptr(); break;
                    case token_depend       : parse_omp_depend();       break;
                    case token_nowait		: parse_omp_nowait();		break;
                    default : OmpParserError("Not supported clause: " + tok);
                }
            }
        }
    }

    private static void parse_omp_target_teams_distribute() {
        PrintTools.println("OmpParser is parsing [distribute] clause", 2);
        addToMap("distribute", "true");
        if (check("simd")) {
            eat();
            parse_omp_target_teams_distribute_simd();
        } else  if (check("parallel")) {
            eat();
            if(check("for")){
                eat();
                parse_omp_target_teams_distribute_parallel_for();
            } else {
                OmpParserError("Not supported clause: " + lookahead());
            }
        }else {
            while (end_of_token() == false) {
            	String tok = get_token();
            	if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
            	if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
                String clause = "token_" + tok;
                switch (omp_clause.valueOf(clause)) {
                    case token_device       : parse_omp_device();       break;
                    case token_map          : parse_omp_map();          break;
                    case token_if           : parse_omp_if();           break;
                    case token_num_teams    : parse_omp_num_teams();    break;
                    case token_thread_limit : parse_omp_thread_limit(); break;
                    case token_default      : parse_omp_default();      break;
                    case token_private      : parse_omp_private();      break;
                    case token_firstprivate : parse_omp_firstprivate(); break;
                    case token_shared       : parse_omp_shared();       break;
                    case token_reduction    : parse_omp_reduction();    break;
                    case token_collapse     : parse_omp_collapse();         break;
                    case token_dist_schedule: parse_omp_dist_schedule();    break;
                    case token_is_device_ptr : parse_omp_is_device_ptr(); break;
                    case token_depend       : parse_omp_depend();       break;
                    case token_nowait		: parse_omp_nowait();		break;
                    default : OmpParserError("Not supported clause: " + tok);
                }
            }
        }
    }

    private static void parse_omp_target_teams_distribute_simd() {
        PrintTools.println("OmpParser is parsing [simd] clause", 2);
        addToMap("simd", "true");

        while (end_of_token() == false) {
            String tok = get_token();
            if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
            if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
            String clause = "token_" + tok;
            switch (omp_clause.valueOf(clause)) {
                case token_device       : parse_omp_device();       break;
                case token_map          : parse_omp_map();          break;
                case token_if           : parse_omp_if();           break;
                case token_num_teams    : parse_omp_num_teams();    break;
                case token_thread_limit : parse_omp_thread_limit(); break;
                case token_default      : parse_omp_default();      break;
                case token_private      : parse_omp_private();      break;
                case token_firstprivate : parse_omp_firstprivate(); break;
                case token_shared       : parse_omp_shared();       break;
                case token_reduction    : parse_omp_reduction();    break;
                case token_collapse         : parse_omp_collapse();         break;
                case token_dist_schedule    : parse_omp_dist_schedule();    break;
                case token_is_device_ptr : parse_omp_is_device_ptr(); break;
                case token_depend       : parse_omp_depend();       break;
                case token_nowait		: parse_omp_nowait();		break;
                default : OmpParserError("Not supported clause: " + tok);
            }
        }
    }

    private static void parse_omp_target_teams_distribute_parallel_for() {
        PrintTools.println("OmpParser is parsing [parallel for] clause", 2);
        addToMap("parallel", "true");
        addToMap("for", "true");
        if (check("simd")) {
            eat();
            parse_omp_target_teams_distribute_parallel_for_simd();
        } else {
            while (end_of_token() == false) {
            	String tok = get_token();
            	if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
            	if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
                String clause = "token_" + tok;
                switch (omp_clause.valueOf(clause)) {
                    case token_device       : parse_omp_device();       break;
                    case token_map          : parse_omp_map();          break;
                    case token_if           : parse_omp_if();           break;
                    case token_num_threads  : parse_omp_num_threads();  break;
                    case token_num_teams    : parse_omp_num_teams();    break;
                    case token_thread_limit : parse_omp_thread_limit(); break;
                    case token_default      : parse_omp_default();      break;
                    case token_private      : parse_omp_private();      break;
                    case token_firstprivate : parse_omp_firstprivate(); break;
                    case token_shared       : parse_omp_shared();       break;
                    case token_reduction    : parse_omp_reduction();    break;
                    case token_collapse         : parse_omp_collapse();         break;
                    case token_dist_schedule    : parse_omp_dist_schedule();    break;
                    case token_is_device_ptr : parse_omp_is_device_ptr(); break;
                    case token_depend       : parse_omp_depend();       break;
                    case token_nowait		: parse_omp_nowait();		break;
                    default : OmpParserError("Not supported clause: " + tok);
                }
            }
        }
    }

    private static void parse_omp_target_teams_distribute_parallel_for_simd() {
        PrintTools.println("OmpParser is parsing [simd] clause", 2);
        addToMap("simd", "true");

        while (end_of_token() == false) {
            String tok = get_token();
            if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
            if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
            String clause = "token_" + tok;
            switch (omp_clause.valueOf(clause)) {
                case token_device       : parse_omp_device();       break;
                case token_map          : parse_omp_map();          break;
                case token_if           : parse_omp_if();           break;
                case token_num_threads  : parse_omp_num_threads();  break;
                case token_num_teams    : parse_omp_num_teams();    break;
                case token_thread_limit : parse_omp_thread_limit(); break;
                case token_default      : parse_omp_default();      break;
                case token_private      : parse_omp_private();      break;
                case token_firstprivate : parse_omp_firstprivate(); break;
                case token_shared       : parse_omp_shared();       break;
                case token_reduction    : parse_omp_reduction();    break;
                case token_collapse         : parse_omp_collapse();         break;
                case token_dist_schedule    : parse_omp_dist_schedule();    break;
                case token_is_device_ptr : parse_omp_is_device_ptr(); break;
                case token_depend       : parse_omp_depend();       break;
                case token_nowait		: parse_omp_nowait();		break;
                default : OmpParserError("Not supported clause: " + tok);
            }
        }
    }

    private static void parse_omp_target_parallel() {
        PrintTools.println("OmpParser is parsing [parallel] clause", 2);
        addToMap("parallel", "true");
        if (check("for")) {
            eat();
            parse_omp_target_parallel_for();
        } else {
            while (end_of_token() == false) {
            	String tok = get_token();
            	if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
            	if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
                String clause = "token_" + tok;
                PrintTools.println("clause=" + clause, 2);
                switch (omp_clause.valueOf(clause)) {
                    case token_if           : parse_omp_if();           break;
                    case token_num_threads  : parse_omp_num_threads();  break;
                    case token_default      : parse_omp_default();      break;
                    case token_private      : parse_omp_private();      break;
                    case token_firstprivate : parse_omp_firstprivate(); break;
                    case token_shared       : parse_omp_shared();       break;
                    case token_copyin       : parse_omp_copyin();       break;
                    case token_reduction    : parse_omp_reduction();    break;
                    case token_device       : parse_omp_device();       break;
                    case token_map          : parse_omp_map();          break;
                    case token_is_device_ptr : parse_omp_is_device_ptr(); break;
                    case token_depend       : parse_omp_depend();       break;
                    case token_nowait		: parse_omp_nowait();		break;
                    default                 :
                        OmpParserError("Not supported clause : " + clause);
                }
            }
        }
    }

    private static void parse_omp_target_parallel_for() {
        PrintTools.println("OmpParser is parsing [parallel for] clause", 2);
        addToMap("for", "true");
        if (check("simd")) {
            eat();
            parse_omp_target_parallel_for_simd();
        } else {
            while (end_of_token() == false) {
            	String tok = get_token();
            	if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
            	if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
                String clause = "token_" + tok;
                switch (omp_clause.valueOf(clause)) {
                    case token_if           : parse_omp_if();           break;
                    case token_num_threads  : parse_omp_num_threads();  break;
                    case token_default      : parse_omp_default();      break;
                    case token_private      : parse_omp_private();      break;
                    case token_firstprivate : parse_omp_firstprivate(); break;
                    case token_shared       : parse_omp_shared();       break;
                    case token_copyin       : parse_omp_copyin();       break;
                    case token_reduction    : parse_omp_reduction();    break;
                    case token_lastprivate  : parse_omp_lastprivate();  break;
                    case token_schedule     : parse_omp_schedule();     break;
                    case token_collapse     : parse_omp_collapse();     break;
                    case token_ordered      : parse_omp_ordered();      break;
                    case token_device       : parse_omp_device();       break;
                    case token_map          : parse_omp_map();          break;
                    case token_is_device_ptr : parse_omp_is_device_ptr(); break;
                    case token_depend       : parse_omp_depend();       break;
                    case token_nowait		: parse_omp_nowait();		break;
                    default : OmpParserError("Not supported clause: " + tok);
                }
            }
        }
    }

    private static void parse_omp_target_parallel_for_simd() {
        PrintTools.println("OmpParser is parsing [simd] clause", 2);
        addToMap("simd", "true");

        while (end_of_token() == false) {
            String tok = get_token();
            if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
            if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
            String clause = "token_" + tok;
            switch (omp_clause.valueOf(clause)) {
                case token_if           : parse_omp_if();           break;
                case token_num_threads  : parse_omp_num_threads();  break;
                case token_default      : parse_omp_default();      break;
                case token_private      : parse_omp_private();      break;
                case token_firstprivate : parse_omp_firstprivate(); break;
                case token_shared       : parse_omp_shared();       break;
                case token_copyin       : parse_omp_copyin();       break;
                case token_reduction    : parse_omp_reduction();    break;
                case token_lastprivate  : parse_omp_lastprivate();  break;
                case token_schedule     : parse_omp_schedule();     break;
                case token_collapse     : parse_omp_collapse();     break;
                case token_ordered      : parse_omp_ordered();      break;
                case token_safelen      : parse_omp_safelen();      break;
                case token_linear       : parse_omp_linear();       break;
                case token_aligned      : parse_omp_aligned();      break;
                case token_device       : parse_omp_device();       break;
                case token_map          : parse_omp_map();          break;
                case token_is_device_ptr : parse_omp_is_device_ptr(); break;
                case token_depend       : parse_omp_depend();       break;
                case token_nowait		: parse_omp_nowait();		break;
                default : OmpParserError("Not supported clause: " + tok);
            }
        }

    }

    private static void parse_omp_target_simd() {
        PrintTools.println("OmpParser is parsing [simd] clause", 2);
        addToMap("simd", "true");
        while (end_of_token() == false) {
            String tok = get_token();
            if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
            if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
            String clause = "token_" + tok;
            switch (omp_clause.valueOf(clause)) {
                case token_safelen      : parse_omp_safelen();      break;
                case token_linear       : parse_omp_linear();       break;
                case token_aligned      : parse_omp_aligned();      break;
                case token_private      : parse_omp_private();      break;
                case token_lastprivate  : parse_omp_lastprivate();  break;
                case token_reduction    : parse_omp_reduction();    break;
                case token_collapse     : parse_omp_collapse();     break;
                case token_device       : parse_omp_device();       break;
                case token_map          : parse_omp_map();          break;
                case token_is_device_ptr : parse_omp_is_device_ptr(); break;
                case token_depend       : parse_omp_depend();       break;
                case token_nowait		: parse_omp_nowait();		break;
                default : OmpParserError("Not supported clause: " + tok);
            }
        }
    }

    /** ---------------------------------------------------------------
     *       #pragma omp target data [clause[ [, ]clause] ...]
     *          structured-block
     *       clause:
     *          device(integer-expression)
     *          map([map-type: ] list)
     *          if(scalar-expression)
     *          use_device_ptr(list)
     * --------------------------------------------------------------- */
    private static void parse_omp_target_data() {
        PrintTools.println("OmpParser is parsing [data] clause", 2);
        addToMap("data", "true");

        while (end_of_token() == false) {
			String tok = get_token();
			if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
			if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
            String clause = "token_" + tok;
            switch (omp_clause.valueOf(clause)) {
                case token_device       : parse_omp_device();       break;
                case token_map          : parse_omp_map();          break;
                case token_if           : parse_omp_if();           break;
                case token_use_device_ptr           : parse_omp_use_device_ptr();           break;
                default : OmpParserError("Not supported clause: " + tok);
            }
        }
    }

    /** ---------------------------------------------------------------
     *       #pragma omp target enter data [clause[ [, ]clause] ...]
     *       
     *       clause:
     *          device(integer-expression)
     *          map([map-type: ] list)
     *          if(scalar-expression)
     *          depend(dependence-type: list)
     *          nowait
     * --------------------------------------------------------------- */
    private static void parse_omp_target_enter_data() {
        PrintTools.println("OmpParser is parsing [enter data] clause", 2);
        addToMap("enter", "true");
        addToMap("data", "true");

        while (end_of_token() == false) {
            String tok = get_token();
            if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
            if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
            String clause = "token_" + tok;
            switch (omp_clause.valueOf(clause)) {
                case token_device       : parse_omp_device();       break;
                case token_map          : parse_omp_map();          break;
                case token_if           : parse_omp_if();           break;
                case token_depend       : parse_omp_depend();       break;
                case token_nowait		: parse_omp_nowait();		break;
                default : OmpParserError("Not supported clause: " + tok);
            }
        }
    }

    /** ---------------------------------------------------------------
     *       #pragma omp target exit data [clause[ [, ]clause] ...]
     *       
     *       clause:
     *          device(integer-expression)
     *          map([map-type: ] list)
     *          if(scalar-expression)
     *          depend(dependence-type: list)
     *          nowait
     * --------------------------------------------------------------- */
    private static void parse_omp_target_exit_data() {
        PrintTools.println("OmpParser is parsing [exit data] clause", 2);
        addToMap("exit", "true");
        addToMap("data", "true");

        while (end_of_token() == false) {
            String tok = get_token();
            if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
            if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
            String clause = "token_" + tok;
            switch (omp_clause.valueOf(clause)) {
                case token_device       : parse_omp_device();       break;
                case token_map          : parse_omp_map();          break;
                case token_if           : parse_omp_if();           break;
                case token_depend       : parse_omp_depend();       break;
                case token_nowait		: parse_omp_nowait();		break;
                default : OmpParserError("Not supported clause: " + tok);
            }
        }
    }

    /** ---------------------------------------------------------------
     *        2.8 Teams Constructs
     *           Creates a league of thread teams where the master thread
     *           of each team executes the region
     * --------------------------------------------------------------- */
    /** ---------------------------------------------------------------
     *	        #pragma omp teams [clause[ [, ]clause] ...]
     *	            structured-block
     *	        clause:
     *	           num_teams(integer-expression)
     *	           thread_limit(integer-expression)
     *	           default(shared | none)
     *	           private(list)
     *	           firstprivate(list)
     *	           shared(list)
     *	           reduction(reduction-identifier: list)
     * --------------------------------------------------------------- */
    private static void parse_omp_teams() {
        PrintTools.println("OmpParser is parsing [teams] clause", 2);
        addToMap("teams", "true");
        if (check("distribute")) {
            eat();
            parse_omp_teams_distribute();
        } else if (check("parallel")) {
            eat();
            if(check("for")){
                eat();
                parse_omp_teams_parallel_for();
            } else {
                OmpParserError("Not supported clause: " + lookahead());
            }
        } else {
            while (end_of_token() == false) {
            	String tok = get_token();
            	if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
            	if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
                String clause = "token_" + tok;
                switch (omp_clause.valueOf(clause)) {
                    case token_num_teams    : parse_omp_num_teams();    break;
                    case token_thread_limit : parse_omp_thread_limit(); break;
                    case token_default      : parse_omp_default();      break;
                    case token_private      : parse_omp_private();      break;
                    case token_firstprivate : parse_omp_firstprivate(); break;
                    case token_shared       : parse_omp_shared();       break;
                    case token_reduction    : parse_omp_reduction();    break;
                    default : OmpParserError("Not supported clause: " + tok);
                }
            }
        }
    }

    private static void parse_omp_teams_distribute() {
        PrintTools.println("OmpParser is parsing [distribute] clause", 2);
        addToMap("distribute", "true");
        if (check("simd")) {
            eat();
            parse_omp_teams_distribute_simd();
        } else  if (check("parallel")) {
            eat();
            if(check("for")){
                eat();
                parse_omp_teams_distribute_parallel_for();
            } else {
                OmpParserError("Not supported clause: " + lookahead());
            }
        } else {
            while (end_of_token() == false) {
            	String tok = get_token();
            	if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
            	if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
                String clause = "token_" + tok;
                switch (omp_clause.valueOf(clause)) {
                    case token_num_teams    : parse_omp_num_teams();    break;
                    case token_thread_limit : parse_omp_thread_limit(); break;
                    case token_default      : parse_omp_default();      break;
                    case token_private      : parse_omp_private();      break;
                    case token_firstprivate : parse_omp_firstprivate(); break;
                    case token_shared       : parse_omp_shared();       break;
                    case token_reduction    : parse_omp_reduction();    break;
                    case token_collapse         : parse_omp_collapse();         break;
                    case token_dist_schedule    : parse_omp_dist_schedule();    break;
                    default : OmpParserError("Not supported clause: " + tok);
                }
            }
        }
    }

    private static void parse_omp_teams_distribute_simd() {
        PrintTools.println("OmpParser is parsing [simd] clause", 2);
        addToMap("simd", "true");

        while (end_of_token() == false) {
            String tok = get_token();
            if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
            if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
            String clause = "token_" + tok;
            switch (omp_clause.valueOf(clause)) {
                case token_num_teams    : parse_omp_num_teams();    break;
                case token_thread_limit : parse_omp_thread_limit(); break;
                case token_default      : parse_omp_default();      break;
                case token_private      : parse_omp_private();      break;
                case token_firstprivate : parse_omp_firstprivate(); break;
                case token_shared       : parse_omp_shared();       break;
                case token_reduction    : parse_omp_reduction();    break;
                case token_collapse         : parse_omp_collapse();         break;
                case token_dist_schedule    : parse_omp_dist_schedule();    break;
                default : OmpParserError("Not supported clause: " + tok);
            }
        }
    }

    // [JL] Clang seems to ignore directives other than distribute after teams 
    // warning: extra tokens at the end of '#pragma omp teams' are ignored [-Wextra-tokens]
    private static void parse_omp_teams_parallel_for() {
        PrintTools.println("OmpParser is parsing [parallel for] clause", 2);
        addToMap("parallel", "true");
        addToMap("for", "true");
        if (check("simd")) {
            eat();
            parse_omp_teams_parallel_for_simd();
        } else {
            while (end_of_token() == false) {
            	String tok = get_token();
            	if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
            	if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
                String clause = "token_" + tok;
                switch (omp_clause.valueOf(clause)) {
                    case token_num_teams    : parse_omp_num_teams();    break;
                    case token_thread_limit : parse_omp_thread_limit(); break;
                    case token_default      : parse_omp_default();      break;
                    case token_private      : parse_omp_private();      break;
                    case token_firstprivate : parse_omp_firstprivate(); break;
                    case token_shared       : parse_omp_shared();       break;
                    case token_reduction    : parse_omp_reduction();    break;
                    case token_collapse         : parse_omp_collapse();         break;
                    default : OmpParserError("Not supported clause: " + tok);
                }
            }
        }
    }

    // [JL] Clang seems to ignore directives other than distribute after teams 
    // warning: extra tokens at the end of '#pragma omp teams' are ignored [-Wextra-tokens]
    private static void parse_omp_teams_parallel_for_simd() {
        PrintTools.println("OmpParser is parsing [simd] clause", 2);
        addToMap("simd", "true");

        while (end_of_token() == false) {
            String tok = get_token();
            if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
            if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
            String clause = "token_" + tok;
            switch (omp_clause.valueOf(clause)) {
                case token_num_teams    : parse_omp_num_teams();    break;
                case token_thread_limit : parse_omp_thread_limit(); break;
                case token_default      : parse_omp_default();      break;
                case token_private      : parse_omp_private();      break;
                case token_firstprivate : parse_omp_firstprivate(); break;
                case token_shared       : parse_omp_shared();       break;
                case token_reduction    : parse_omp_reduction();    break;
                case token_collapse         : parse_omp_collapse();         break;
                default : OmpParserError("Not supported clause: " + tok);
            }
        }
    }

    private static void parse_omp_teams_distribute_parallel_for() {
        PrintTools.println("OmpParser is parsing [parallel for] clause", 2);
        addToMap("parallel", "true");
        addToMap("for", "true");
        if (check("simd")) {
            eat();
            parse_omp_teams_distribute_parallel_for_simd();
        } else {
            while (end_of_token() == false) {
            	String tok = get_token();
            	if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
            	if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
                String clause = "token_" + tok;
                switch (omp_clause.valueOf(clause)) {
                    case token_num_teams    : parse_omp_num_teams();    break;
                    case token_num_threads  : parse_omp_num_threads();  break;
                    case token_thread_limit : parse_omp_thread_limit(); break;
                    case token_default      : parse_omp_default();      break;
                    case token_private      : parse_omp_private();      break;
                    case token_firstprivate : parse_omp_firstprivate(); break;
                    case token_shared       : parse_omp_shared();       break;
                    case token_reduction    : parse_omp_reduction();    break;
                    case token_collapse         : parse_omp_collapse();         break;
                    case token_dist_schedule    : parse_omp_dist_schedule();    break;
                    default : OmpParserError("Not supported clause: " + tok);
                }
            }
        }
    }

    private static void parse_omp_teams_distribute_parallel_for_simd() {
        PrintTools.println("OmpParser is parsing [simd] clause", 2);
        addToMap("simd", "true");

        while (end_of_token() == false) {
            String tok = get_token();
            if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
            if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
            String clause = "token_" + tok;
            switch (omp_clause.valueOf(clause)) {
                case token_num_teams    : parse_omp_num_teams();    break;
                case token_num_threads  : parse_omp_num_threads();  break;
                case token_thread_limit : parse_omp_thread_limit(); break;
                case token_default      : parse_omp_default();      break;
                case token_private      : parse_omp_private();      break;
                case token_firstprivate : parse_omp_firstprivate(); break;
                case token_shared       : parse_omp_shared();       break;
                case token_reduction    : parse_omp_reduction();    break;
                case token_collapse         : parse_omp_collapse();         break;
                case token_dist_schedule    : parse_omp_dist_schedule();    break;
                default : OmpParserError("Not supported clause: " + tok);
            }
        }
    }

    /** ---------------------------------------------------------------
     *        2.8 Distribute Constructs
     *           distribute specifies loops which are executed by the
     *           thread teams. distribute simd specifies loops which are
     *           executed concurrently using SIMD instructions.
     * --------------------------------------------------------------- */
    /** ---------------------------------------------------------------
     *	        #pragma omp distribute [clause[ [, ]clause] ...]
     *	            for-loops
     *	        clause:
     *	           private(list)
     *	           firstprivate(list)
     *	           collapse(n)
     *	           dist_schedule(kind[, chunk_size])
     * --------------------------------------------------------------- */
    private static void parse_omp_distribute() {
        PrintTools.println("OmpParser is parsing [distribute] clause", 2);
        addToMap("distribute", "true");
        if (check("simd")) {
            eat();
            parse_omp_distribute_simd();
        } else  if (check("parallel")) {
            eat();
            if(check("for")){
                eat();
                parse_omp_distribute_parallel_for();
            } else {
                OmpParserError("Not supported clause: " + lookahead());
            }
        } else {
            while (end_of_token() == false) {
            	String tok = get_token();
            	if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
            	if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
                String clause = "token_" + tok;
                switch (omp_clause.valueOf(clause)) {
                    case token_private          : parse_omp_private();          break;
                    case token_firstprivate     : parse_omp_firstprivate();     break;
                    case token_collapse         : parse_omp_collapse();         break;
                    case token_dist_schedule    : parse_omp_dist_schedule();    break;
                    default : OmpParserError("Not supported clause: " + tok);
                }
            }
        }
    }

    private static void parse_omp_distribute_simd() {
        PrintTools.println("OmpParser is parsing [simd] clause", 2);
        addToMap("simd", "true");
        while (end_of_token() == false) {
            String tok = get_token();
            if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
            if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
            String clause = "token_" + tok;
            switch (omp_clause.valueOf(clause)) {
                case token_private          : parse_omp_private();          break;
                case token_firstprivate     : parse_omp_firstprivate();     break;
                case token_collapse         : parse_omp_collapse();         break;
                case token_dist_schedule    : parse_omp_dist_schedule();    break;
                default : OmpParserError("Not supported clause: " + tok);
            }
        }
    }

    private static void parse_omp_distribute_parallel_for() {
        PrintTools.println("OmpParser is parsing [parallel for] clause", 2);
        addToMap("parallel", "true");
        addToMap("for", "true");
        if (check("simd")) {
            eat();
            parse_omp_distribute_parallel_for_simd();
        } else {
            while (end_of_token() == false) {
            	String tok = get_token();
            	if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
            	if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
                String clause = "token_" + tok;
                switch (omp_clause.valueOf(clause)) {
                    // [JL] parse num_threads clause here?
                    case token_private          : parse_omp_private();          break;
                    case token_firstprivate     : parse_omp_firstprivate();     break;
                    case token_collapse         : parse_omp_collapse();         break;
                    case token_dist_schedule    : parse_omp_dist_schedule();    break;
                    case token_shared           : parse_omp_shared();           break;
                    default : OmpParserError("NoSuchParallelConstruct: " + tok);
                }
            }
        }
    }

    private static void parse_omp_distribute_parallel_for_simd() {
        PrintTools.println("OmpParser is parsing [simd] clause", 2);
        addToMap("simd", "true");
        while (end_of_token() == false) {
            String tok = get_token();
            if( tok.equals("") ) continue; //Skip empty string, which may occur due to macro.
            if( tok.equals(",") ) continue; //Skip comma between clauses, if existing.
            String clause = "token_" + tok;
            switch (omp_clause.valueOf(clause)) {
                // [JL] parse num_threads clause here?
                case token_private          : parse_omp_private();          break;
                case token_firstprivate     : parse_omp_firstprivate();     break;
                case token_collapse         : parse_omp_collapse();         break;
                case token_dist_schedule    : parse_omp_dist_schedule();    break;
                default : OmpParserError("NoSuchParallelConstruct: " + tok);
            }
        }
    }

    /** ---------------------------------------------------------------
      *
      *        A collection of parser routines for OpenMP clauses
      *
      * --------------------------------------------------------------- */
    /**
      * This function parses a list of strings between a parenthesis, for
      * example, (scalar-expression) or (integer-expression).
      */
    private static String parse_ParenEnclosedExpr() {
        String str = null;
        int paren_depth = 1;
        match("(");
        while (true) {
            if (check("(")) {
                paren_depth++;
            }
            if (check(")")) {
                if (--paren_depth==0) {
                    break;
                }
            }
            if (str == null) {
                str = new String(get_token());
            } else {
                str.concat((" " + get_token()));
            }
        }
        match(")");
        return str;
    }

    // it is assumed that a (scalar-expression) is of the form (size < N)
    private static void parse_omp_if() {
        PrintTools.println("OmpParser is parsing [if] clause", 2);
        String str = parse_ParenEnclosedExpr();
        addToMap("if", str);
    }

    // it is assumed that a (integer-expression) is of the form (4)
    private static void parse_omp_num_threads() {
        PrintTools.println("OmpParser is parsing [omp_num_threads]", 2);
        String str = parse_ParenEnclosedExpr();
        addToMap("num_threads", str);    
    }

    /**
      * schedule(kind[, chunk_size])
      */
    private static void parse_omp_schedule() {
        PrintTools.println("OmpParser is parsing [schedule] clause", 2);
        String str = null;
        match("(");
        // schedule(static, chunk_size)
        // schedule(dynamic, chunk_size)
        // schedule(guided, chunk_size)
        if (check("static") || check("dynamic") || check("guided")) { 
            str = new String(get_token());
            if (check(",")) {
                match(",");
                eat();        // consume "chunk_size"    
            }
        // schedule(auto), schedule(runtime)
        } else if (check("auto") || check("runtime")) {
            str = new String(get_token());
        } else {
            OmpParserError("No such scheduling kind" + lookahead());
        }
        match(")");
        addToMap("schedule", str);
    }

    private static void parse_omp_collapse() {
        PrintTools.println("OmpParser is parsing [collapse] clause", 2);
        match("(");
        String int_str = new String(get_token());
        match(")");
        addToMap("collapse", int_str);    
    }

    private static void parse_omp_nowait() { 
        PrintTools.println("OmpParser is parsing [nowait] clause", 2);
        addToMap("nowait", "true"); 
    }

    private static void parse_omp_untied() { 
        PrintTools.println("OmpParser is parsing [untied] clause", 2);
        addToMap("untied", "true"); 
    }

    private static void parse_omp_default() {
        PrintTools.println("OmpParser is parsing [default] clause", 2);
        match("(");
        if (check("shared") || check("none")) {
            addToMap("default", new String(get_token()));
        } else {
            OmpParserError("NoSuchParallelDefaultCluase" + lookahead());
        }
        match(")");
    }

    private static void parse_omp_private() {
        PrintTools.println("OmpParser is parsing [private] clause", 2);
        match("(");
        Set set = new HashSet<String>();
        parse_commaSeparatedList(set);
        match(")");
        addToMap("private", set);
    }

    private static void parse_omp_firstprivate() {
        PrintTools.println("OmpParser is parsing [firstprivate] clause", 2);
        match("(");
        Set set = new HashSet<String>();
        parse_commaSeparatedList(set);
        match(")");
        addToMap("firstprivate", set);
    }

    private static void parse_omp_lastprivate() {
        PrintTools.println("OmpParser is parsing [lastprivate] clause", 2);
        match("(");
        Set set = new HashSet<String>();
        parse_commaSeparatedList(set);
        match(")");
        addToMap("lastprivate", set);
    }

    private static void parse_omp_copyprivate() {
        PrintTools.println("OmpParser is parsing [copyprivate] clause", 2);
        match("(");
        Set set = new HashSet<String>();
        parse_commaSeparatedList(set);
        match(")");
        addToMap("copyprivate", set);
    }

    private static void parse_omp_shared() {
        PrintTools.println("OmpParser is parsing [shared] clause", 2);
        match("(");
        Set set = new HashSet<String>();
        parse_commaSeparatedSubArrayList(set, 0);
//        parse_commaSeparatedList(set);
        match(")");
        addToMap("shared", set);
    }

    private static void parse_omp_copyin() {
        PrintTools.println("OmpParser is parsing [copyin] clause", 2);
        match("(");
        Set set = new HashSet<String>();
        parse_commaSeparatedList(set);
        match(")");
        addToMap("copyin", set);
    }

    private static void parse_omp_is_device_ptr() {
        PrintTools.println("OmpParser is parsing [is_device_ptr] clause", 2);
        match("(");
        Set set = new HashSet<String>();
        parse_commaSeparatedList(set);
        match(")");
        addToMap("is_device_ptr", set);
    }

    private static void parse_omp_use_device_ptr() {
        PrintTools.println("OmpParser is parsing [use_device_ptr] clause", 2);
        match("(");
        Set set = new HashSet<String>();
        parse_commaSeparatedList(set);
        match(")");
        addToMap("use_device_ptr", set);
    }

    // reduction(oprator:list)
    @SuppressWarnings("unchecked")
    private static void parse_omp_reduction() {
        PrintTools.println("OmpParser is parsing [reduction] clause", 2);
        HashMap reduction_map = null;
        Set set = null;
        String op = null;
        match("(");
        // Discover the kind of reduction operator (+, etc)
        if (check("+") || check("*") || check("-") || check("&") ||
            check("|") || check("^") || check("&&") || check("||")) {
            op = get_token();
            PrintTools.println("reduction op:" + op, 2);
        } else {
            OmpParserError("Undefined reduction operator" + lookahead());
        }

        // check if there is already a reduction annotation with the same
        // operator in the set
        for (String ikey : (Set<String>)(omp_map.keySet())) {
            if (ikey.compareTo("reduction") == 0) {
                reduction_map = (HashMap)(omp_map.get(ikey));
                set = (Set)(reduction_map.get(op));
                break;
            }
        }
        if (reduction_map == null) {
            reduction_map = new HashMap(4);
        } 
        if (match(":") == false) {
            OmpParserError(
                    "colon expected before a list of reduction variables");
        }
        // When reduction_map is not null, set can be null for the given
        // reduction operator
        if (set == null) {
            set = new HashSet<String>();
        }
        parse_commaSeparatedSubArrayList(set, 0);
//        parse_commaSeparatedList(set);
        match(")");
        reduction_map.put(op, set);
        addToMap("reduction", reduction_map);
    }

    /**
      * This function reads a list of comma-separated variables
      * It checks the right parenthesis to end the parsing, but does not
      * consume it.
      */
    @SuppressWarnings("unchecked")
    private static void parse_commaSeparatedList(Set set) {
    	String cTok = null;
    	String tTok = null;
    	while( !end_of_token() ) {
    		tTok = lookahead();
    		if(tTok.compareTo(")") == 0) {
    			if( cTok == null ) {
    				OmpParserError("missing entry in comma separated list");
    			} else {
    				set.add(cTok);
    			}
    			break;
    		} else if(tTok.compareTo(",") == 0) {
    			eat();
    			if( cTok == null ) {
    				OmpParserError("missing entry in comma separated list");
    			} else {
    				set.add(cTok);
    			}
    			cTok = null;
    		} else {
    			eat();
    			if( cTok == null ) {
    				cTok = tTok;
    			} else {
    				cTok = cTok+tTok;
    			}
    		}
    	}
//Old implementation
/*        for (;;) {
        	cTok = get_token();
        	while( check(":") ) {
        		match(":");
        		cTok = cTok+":"+get_token();
        	}
            set.add(cTok);
            if (check(")")) {
                break;
            } else if (match(",") == false) {
                OmpParserError("comma expected in comma separated list");
            }
        }*/
    }

	/**
		* This function reads a list of comma-separated subarrays
		* It checks the right parenthesis to end the parsing, but does not consume it.
		* 
		*/
	private static void parse_commaSeparatedSubArrayList(Set collect, int limit)
	{
		String tok;
		int counter = 0;
//		try {
			for (;;) {
				tok = get_token();
				Expression aName = ExpressionParser.parse(tok);
				if( aName == null ) {
					OmpParserError("null is returned where SubArray name is expected!");
				}
				//"*" is special variable used in resilience ftdata clause.
				if( aName.toString().equals("(*)") ) {
					aName = new NameID("*");
				} else if( (aName instanceof SomeExpression) ) {
					OmpParserError("Current implementation supports only simple scalar or array variables (but not class members) in OpenACC data clauses.");
				}
				SubArray subArr = new SubArray(aName);
				if ( check(")") )
				{
					collect.add(subArr.toString());
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
									OmpParserError("Can't find : token in the Subarray expressions");
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
										OmpParserError("Can't find ] token in the Subarray expressions");
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
						collect.add(subArr.toString());
						counter++;
						break;
					} else if( tok.equals(",") ) {
						collect.add(subArr.toString());
						counter++;
						eat();
						if( (limit > 0) && (counter == limit) ) {
							break;
						}
					} else {
						OmpParserError("comma expected in comma separated list");
					}
				}
				else if ( check(",") )
				{
					collect.add(subArr.toString());
					counter++;
					eat();
					if( (limit > 0) && (counter == limit) ) {
						break;
					}
				}
				else 
				{
					OmpParserError("comma expected in comma separated list");
				}
			}
/*		} catch( Exception e) {
			OmpParserError("unexpected error in parsing comma-separated SubArray list");
		}*/
	}

    private static void notSupportedWarning(String text) {
        System.out.println("Not Supported OpenMP feature: " + text); 
    }

    private static void OmpParserError(String text) {
        System.out.println("OpenMP Parser Syntax Error: " + text);
        System.out.println(display_tokens());
        System.exit(0);
    }

    @SuppressWarnings("unchecked")
    private static void addToMap(String key, String new_str) {
        if (omp_map.keySet().contains(key)) {
            Tools.exit("[Warning] OMP Parser detected duplicate pragma: "+key);
        } else {
            omp_map.put(key, new_str);
        }
    }

    // When a key already exists in the map
    // from page 31, 2.9.3 Data-Sharing Attribute Clauses
    // With the exception of the default clause, clauses may be repeated as
    // needed. A list item that specifies a given variable may not appear in
    // more than one clause on the same directive, except that a variable may
    // be specified in both firstprivate and lastprivate clauses.
    @SuppressWarnings("unchecked")
    private static void addToMap(String key, Set new_set) {
        if (omp_map.keySet().contains(key)) {
            Set set = (Set)omp_map.get(key);
            set.addAll(new_set);
        } else {
            omp_map.put(key, new_set);
        }
    }

    // reduction clause can be repeated as needed
    @SuppressWarnings("unchecked")
    private static void addToMap(String key, Map new_map) {
        if (omp_map.keySet().contains(key)) {
            Map orig_map = (Map)omp_map.get(key);
            for (String new_str : (Set<String>)new_map.keySet()) {
                Set new_set = (Set)new_map.get(new_str);
                if (orig_map.keySet().contains(new_str)) {
                    Set orig_set = (Set)orig_map.get(new_str);
                    orig_set.addAll(new_set);
                } else {
                    orig_map.put(new_str, new_set);
                }
            }
        } else {
            omp_map.put(key, new_map);
        }
    }

    // it is assumed that a (integer-expression) is of the form (4)
    private static void parse_omp_safelen() {
        PrintTools.println("OmpParser is parsing [safelen]", 2);
        String str = parse_ParenEnclosedExpr();
        addToMap("safelen", str);    
    }

    private static void parse_omp_device() {
        PrintTools.println("OmpParser is parsing [device]", 2);
        String str = parse_ParenEnclosedExpr();
        addToMap("device", str);
    }

    private static void parse_omp_linear() {
        PrintTools.println("OmpParser is parsing [linear] clause", 2);
        match("(");
        Set set = new HashSet<String>();
        parse_commaSeparatedList(set);
        match(")");
        addToMap("linear", set);
    }

    private static void parse_omp_aligned() {
        PrintTools.println("OmpParser is parsing [aligned] clause", 2);
        match("(");
        Set set = new HashSet<String>();
        parse_commaSeparatedList(set);
        match(")");
        addToMap("aligned", set);
    }

    private static void parse_omp_uniform() {
        PrintTools.println("OmpParser is parsing [uniform] clause", 2);
        match("(");
        Set set = new HashSet<String>();
        parse_commaSeparatedList(set);
        match(")");
        addToMap("uniform", set);
    }

    private static void parse_omp_map() {
    	PrintTools.println("OmpParser is parsing [map] clause", 2);
    	String mapType = "tofrom";
    	String mapType_prefix = null;
    	if( check("always") ) {
    		mapType_prefix = "always";
    		eat();
    		if( check(",") ) {
    			eat();
    		}
    	}
    	match("(");
    	if(check("to") || check("from") || check("alloc") || check("tofrom") || check("release") || check("delete"))
    	{
    		mapType = get_token();
    		match(":");
    	}
        Set set = new HashSet<String>();
        parse_commaSeparatedSubArrayList(set, 0);
//        parse_commaSeparatedList(set);
    	match(")");
    	if( mapType_prefix == null ) {
    		addToMap(mapType, set);
    	} else {
    		addToMap(mapType_prefix + " " + mapType, set);
    	}
    }

    private static void parse_omp_depend() {
    	PrintTools.println("OmpParser is parsing [depend] clause", 2);
    	String dependType = "inout";
    	match("(");
    	if( check("in") || check("out") || check("inout") || check("sink") )
    	{
    		dependType = get_token();
    		match(":");
    		Set set = new HashSet<String>();
    		parse_commaSeparatedSubArrayList(set, 0);
//        parse_commaSeparatedList(set);
    		match(")");
    		addToMap(dependType, set);
    	} else if( check("source") ) {
    		dependType = get_token();
    		match(")");
    		addToMap(dependType, "true");
        } else {
            OmpParserError("No such dependence-type" + lookahead());
    	}
    }

    private static void parse_omp_inbranch() {
        PrintTools.println("OmpParser is parsing [inbranch] clause", 2);
        addToMap("inbranch", "true");
    }

    private static void parse_omp_notinbranch() {
        PrintTools.println("OmpParser is parsing [notinbranch] clause", 2);
        addToMap("notinbranch", "true");
    }

    private static void parse_omp_num_teams() {
        PrintTools.println("OmpParser is parsing [num_teams]", 2);
        String str = parse_ParenEnclosedExpr();
        addToMap("num_teams", str);
    }

    private static void parse_omp_thread_limit() {
        PrintTools.println("OmpParser is parsing [thread_limit]", 2);
        String str = parse_ParenEnclosedExpr();
        addToMap("thread_limit", str);
    }

    private static void parse_omp_dist_schedule() {
        PrintTools.println("OmpParser is parsing [dist_schedule] clause", 2);
        String str = null;
        match("(");
        // schedule(static, chunk_size)
        if (check("static")) {
            str = new String(get_token());
            if (check(",")) {
                match(",");
                eat();        // consume "chunk_size"
            }
            // schedule(auto), schedule(runtime)
        } else {
            OmpParserError("No such scheduling kind" + lookahead());
        }
        match(")");
        addToMap("schedule", str);
    }

    private static void parse_omp_to() {
        PrintTools.println("OmpParser is parsing [to] clause", 2);
        match("(");
        Set set = new HashSet<String>();
        parse_commaSeparatedSubArrayList(set, 0);
//        parse_commaSeparatedList(set);
        match(")");
        addToMap("to", set);
    }

    private static void parse_omp_from() {
        PrintTools.println("OmpParser is parsing [from] clause", 2);
        match("(");
        Set set = new HashSet<String>();
        parse_commaSeparatedSubArrayList(set, 0);
//        parse_commaSeparatedList(set);
        match(")");
        addToMap("from", set);
    }

    private static void parse_omp_link() {
        PrintTools.println("OmpParser is parsing [link] clause", 2);
        match("(");
        Set set = new HashSet<String>();
        parse_commaSeparatedSubArrayList(set, 0);
//        parse_commaSeparatedList(set);
        match(")");
        addToMap("link", set);
    }

    public static enum omp_pragma {
        omp_parallel, 
        omp_for, 
        omp_sections, 
        omp_section, 
        omp_single, 
        omp_task, 
        omp_master, 
        omp_critical, 
        omp_barrier, 
        omp_taskwait,
        omp_atomic, 
        omp_flush, 
        omp_ordered,
        omp_threadprivate,
        omp_simd,
        omp_declare,
        omp_end,
        omp_target,
        omp_teams,
        omp_distribute,
        omp_taskgroup,
        omp_cancel,
        omp_cancellation,
    }

    public static enum omp_clause {
        token_if,
        token_num_threads,
        token_default,
        token_private,
        token_firstprivate,
        token_lastprivate,
        token_shared,
        token_copyprivate,
        token_copyin,
        token_schedule,
        token_nowait,
        token_ordered,
        token_untied,
        token_collapse,
        token_reduction,
        token_safelen,
        token_linear,
        token_aligned,
        token_uniform,
        token_inbranch,
        token_notinbranch,
        token_device,
        token_map,
        token_depend,
        token_num_teams,
        token_thread_limit,
        token_dist_schedule,
        token_to,
        token_from,
        token_link,
        token_use_device_ptr,
        token_is_device_ptr
    }
}
