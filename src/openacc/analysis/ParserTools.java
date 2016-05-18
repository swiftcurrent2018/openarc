package openacc.analysis;

import java.util.*;

import cetus.hir.*;

/**
 * <b>ParserTools</b> provides tools for parsing C code.
 *
 * @author Putt Sakdhnagool <psakdhna@purdue.edu>
 *         ParaMount Group
 *         School of ECE, Purdue University
 */
public final class ParserTools
{
    /**
     * Generate Expression from string. The tool will raise an exception
     * if the code cannot be compiled,
     * @param code C code
     */
    public static Expression strToExpression(String code)
    {
        Boolean success = true;
        Expression e = strToExpression(code, success);

        if (success.booleanValue() == false)
            PrintTools.println("[ParserTools] cannot compile: " + code + ". Return SomeExpression Instead", 3);
        return e;
    }

    /**
     * strToBinaryExpression changes a string of code into Expression object
     * for Cetus HIR.
     *
     * @param code Input C code
     * @param success [out] result of expression generation.
     * @return Expression object of the input code
     */
    public static Expression strToExpression(String code, Boolean success)
    {
    	Expression ex = null;
    	try {
    		List<Token> tokenList = lexer(code, success);

    		if(tokenList == null)
    		{
    			return null;
    		}

    		success = true;
    		PrintTools.println("[ParserTools] parsed token list: " + PrintTools.listToString(tokenList, ", "), 3);

    		ex = tokensToExpression(tokenList);
    	} catch (Exception e) {
    		ex = null;
    	}
    	return ex;
    }

    /**
     * tokensToExpression converts a token stream into Expression
     * Current implementation support only
     *  - BinaryExpression
     *  - UnaryExpression
     *  - Identifier
     *  - ArrayAccess
     *  - Constant
     *
     * @param tokenList token stream
     * @return Expression object of the input token stream
     */
    private static Expression tokensToExpression(List<Token> tokenList)
    {
        // Remove start and end parentheses at the beginning and the end of expression
    	// if they are the outermost parentheses.
        while ( (tokenList.size() > 2) && (tokenList.get(0).getType() == token_type.leftparenthesis) &&
                (tokenList.get(tokenList.size()-1).getType() == token_type.rightparenthesis)) {
        	int parenthesis_cnt = 0;
        	boolean RemoveParentheses = true;
        	for( int i=1; i<tokenList.size()-1; i++ ) {
        		if( tokenList.get(i).getType() == token_type.leftparenthesis ) {
        			parenthesis_cnt++;
        		} else if( tokenList.get(i).getType() == token_type.rightparenthesis ) {
        			parenthesis_cnt--;
        		}
        		if( parenthesis_cnt < 0 ) {
        			RemoveParentheses = false;
        			break;
        		}
        	}
        	if( RemoveParentheses && (parenthesis_cnt == 0) ) {
        		tokenList = tokenList.subList(1, tokenList.size()-1);
        	} else {
        		break;
        	}
        }

        if(tokenList.size() == 1)
        {
            Token t = tokenList.get(0);
            if(t.getType() == token_type.intconstant)
                return tokenToIntConstant(t);
            else if(t.getType() == token_type.floatconstant)
                return tokenToFloatConstant(t);
            else if(t.getType() == token_type.doubleconstant)
                return tokenToDoubleConstant(t);
            else if(t.getType() == token_type.identifier)
                return tokenToIdentifier(t);
        }
        // This is UnaryExpression
        else if((tokenList.size() == 2) &&
                ((tokenList.get(0).getType() == token_type.add) ||
                (tokenList.get(0).getType() == token_type.subtract) ||
                (tokenList.get(0).getType() == token_type.preincrement) ||
                (tokenList.get(0).getType() == token_type.predecrement) ||
                (tokenList.get(0).getType() == token_type.dereference) ||
                (tokenList.get(0).getType() == token_type.addressof) ||
                (tokenList.get(0).getType() == token_type.bitwisenot) ||
                (tokenList.get(1).getType() == token_type.postincrement) ||
                (tokenList.get(1).getType() == token_type.postdecrement) ||
                (tokenList.get(0).getType() == token_type.logicalnot)))
        {
            return tokensToUnaryExpression(tokenList);
        }
        else if( tokenList.size() > 2 ) {
        	if ((tokenList.get(0).getType() == token_type.identifier) &&
        			(tokenList.get(1).getType() == token_type.leftbracket) &&
        			(tokenList.get(tokenList.size()-1).getType() == token_type.rightbracket))
        	{
        		return tokensToArrayAccess(tokenList);
        	}
        	else if ((tokenList.get(0).getType() == token_type.identifier) &&
        			(tokenList.get(1).getType() == token_type.leftparenthesis) &&
        			(tokenList.get(tokenList.size()-1).getType() == token_type.rightparenthesis))
        	{
        		return tokensToFunctionCall(tokenList);
        	}
        	else if( (tokenList.size() > 3) && 
                (	((tokenList.get(0).getType() == token_type.add) ||
                		(tokenList.get(0).getType() == token_type.subtract) ||
                		(tokenList.get(0).getType() == token_type.preincrement) ||
                		(tokenList.get(0).getType() == token_type.predecrement) ||
                		(tokenList.get(0).getType() == token_type.dereference) ||
                		(tokenList.get(0).getType() == token_type.addressof) ||
                		(tokenList.get(0).getType() == token_type.bitwisenot) ||
                		(tokenList.get(0).getType() == token_type.logicalnot)) &&
                	(tokenList.get(1).getType() == token_type.leftparenthesis) &&
                	(tokenList.get(tokenList.size()-1).getType() == token_type.rightparenthesis)
                ) || (	((tokenList.get(tokenList.size()-1).getType() == token_type.postincrement) ||
                		(tokenList.get(tokenList.size()-1).getType() == token_type.postdecrement)) &&
                	(tokenList.get(0).getType() == token_type.leftparenthesis) &&
                	(tokenList.get(tokenList.size()-2).getType() == token_type.rightparenthesis)
                ) )
        	{
        		return tokensToUnaryExpression(tokenList);
        	}
        	else
        	{
        		return tokensToBinaryExpression(tokenList);
        	}
        }

        return  null;
    }

    private static Expression tokenToIdentifier(Token t)
    {
        return new NameID(t.getVal());
    }

    private static Expression tokenToIntConstant(Token t)
    {
        return  new IntegerLiteral(t.getVal());
    }
    
    private static Expression tokenToFloatConstant(Token t)
    {
        return  new FloatLiteral((new Double(t.getVal())).doubleValue(), "F");
    }
    
    private static Expression tokenToDoubleConstant(Token t)
    {
        return  new FloatLiteral((new Double(t.getVal())).doubleValue());
    }

    private static Expression tokensToUnaryExpression(List<Token> tokenList)
    {
    	UnaryOperator op = null;
    	token_type tType = tokenList.get(0).getType();
    	switch (tType)
    	{
    	case add: op = UnaryOperator.PLUS;break;
    	case subtract: op = UnaryOperator.MINUS; break;
    	case preincrement: op = UnaryOperator.PRE_INCREMENT; break;
    	case predecrement: op = UnaryOperator.PRE_DECREMENT; break;
    	case dereference: op = UnaryOperator.DEREFERENCE; break;
    	case addressof: op = UnaryOperator.ADDRESS_OF; break;
    	case logicalnot: op = UnaryOperator.LOGICAL_NEGATION; break;
    	case bitwisenot: op = UnaryOperator.BITWISE_COMPLEMENT; break;
    	}
    	Expression ex = null;
    	if( op != null ) {
    		ex = tokensToExpression(tokenList.subList(1, tokenList.size()));
    	} else {
    		tType = tokenList.get(tokenList.size()-1).getType();
    		switch (tType)
    		{
    		case postincrement: op = UnaryOperator.POST_INCREMENT; break;
    		case postdecrement: op = UnaryOperator.POST_DECREMENT; break;
    		}
    		if( op != null ) {
    			ex = tokensToExpression(tokenList.subList(0, tokenList.size()-1));
    		}
    	}
    	if( (op != null) && (ex != null) ) {
    		return  new UnaryExpression(op, ex);
    	} else {
    		return null;
    	}
    }

    private static Expression tokensToArrayAccess(List<Token> tokenList)
    {
        Expression var = tokenToIdentifier(tokenList.get(0));
        Expression idx = null;
        if( tokenList.size() > 3 ) {
        	idx = tokensToExpression(tokenList.subList(2, tokenList.size()-1));
        }
        
        if( idx != null ) {
        	return new ArrayAccess(var, idx);
        } else {
        	return null;
        }
    }
    
    private static List<Expression> tokensToExpressionList(List<Token> tokenList) {
    	List<Expression> expList = new LinkedList<Expression>();
        Token[] token_array = tokenList.toArray(new Token[0]);
        int op_idx = 0;
        int base_idx = 0;
        // Finding the comma operator.
        while (op_idx < token_array.length)
        {
        	token_type tType = token_array[op_idx].getType();
        	if( tType == token_type.comma ) {
        		Expression argExp = tokensToExpression(tokenList.subList(base_idx, op_idx));
        		if( argExp != null ) {
        			expList.add(argExp);
        		} else {
        			return null;
        		}
        		op_idx++;
        		base_idx = op_idx;
        	} else {
        		op_idx++;
        	}
        }
        Expression argExp = tokensToExpression(tokenList.subList(base_idx, op_idx));
        if( argExp != null ) {
        	expList.add(argExp);
        } else {
        	return null;
        }
        return expList;
    }
    
    private static Expression tokensToFunctionCall(List<Token> tokenList)
    {
        Expression var = tokenToIdentifier(tokenList.get(0));
        List<Expression> expList = null;
        if( tokenList.size() >  3) {
        	expList = tokensToExpressionList(tokenList.subList(2, tokenList.size()-1));
        } else {
        	expList = new ArrayList<Expression>();
        }

        if( expList == null ) {
        	return null;
        } else {
        	return new FunctionCall(var, expList);
        }
    }

    /**
     * tokensToBinaryExpression converts a token stream into BinaryExpression
     * in the form of
     *      [lhs] [op] [rhs]
     * lhs and rsh can be any of the following Expression
     *  - ( Expression )
     *  - Identifier
     *  - ArrayAccess
     *  - Constant
     *
     * @param tokenList token stream
     * @return Expression object of the input token stream
     */
    private static Expression tokensToBinaryExpression(List<Token> tokenList)
    {
        Token[] token_array = tokenList.toArray(new Token[0]);

        Expression lhs = null;
        BinaryOperator op = null;
        Expression rhs = null;

        int parentCount = 0;
        int op_idx = token_array.length-1;
        // Finding the operator of the BinaryExpression (||)
        while (op_idx >= 0)
        {
        	token_type tType = token_array[op_idx].getType();
        	// Found the operator
        	if(((tType == token_type.logicalor)) && parentCount == 0 && op_idx > 0)
        		break;
        	else if (token_array[op_idx].getVal().compareTo(")") == 0)
        		parentCount++;
        	else if (token_array[op_idx].getVal().compareTo("(") == 0)
        		parentCount--;
        	op_idx--;
        }

        if( (op_idx > 0) && (op_idx < token_array.length-1) ) { 
        	op = BinaryOperator.LOGICAL_OR;
        	lhs = tokensToExpression(tokenList.subList(0, op_idx));
        	rhs = tokensToExpression(tokenList.subList(op_idx+1, tokenList.size()));
        	if( (lhs == null) || (rhs == null ) ) {
        		return null;
        	} else {
        		return new BinaryExpression(lhs, op, rhs);
        	}
        }

        parentCount = 0;
        op_idx = token_array.length-1;
        // Finding the operator of the BinaryExpression (&&)
        while (op_idx >= 0)
        {
        	token_type tType = token_array[op_idx].getType();
        	// Found the operator
        	if(((tType == token_type.logicaland)) && parentCount == 0 && op_idx > 0)
        		break;
        	else if (token_array[op_idx].getVal().compareTo(")") == 0)
        		parentCount++;
        	else if (token_array[op_idx].getVal().compareTo("(") == 0)
        		parentCount--;
        	op_idx--;
        }

        if( (op_idx > 0) && (op_idx < token_array.length-1) ) { 
        	op = BinaryOperator.LOGICAL_AND;
        	lhs = tokensToExpression(tokenList.subList(0, op_idx));
        	rhs = tokensToExpression(tokenList.subList(op_idx+1, tokenList.size()));
        	if( (lhs == null) || (rhs == null ) ) {
        		return null;
        	} else {
        		return new BinaryExpression(lhs, op, rhs);
        	}
        }

        parentCount = 0;
        op_idx = token_array.length-1;
        // Finding the operator of the BinaryExpression (|)
        while (op_idx >= 0)
        {
        	token_type tType = token_array[op_idx].getType();
        	// Found the operator
        	if(((tType == token_type.bitwiseor)) && parentCount == 0 && op_idx > 0)
        		break;
        	else if (token_array[op_idx].getVal().compareTo(")") == 0)
        		parentCount++;
        	else if (token_array[op_idx].getVal().compareTo("(") == 0)
        		parentCount--;
        	op_idx--;
        }

        if( (op_idx > 0) && (op_idx < token_array.length-1) ) { 
        	op = BinaryOperator.BITWISE_INCLUSIVE_OR;
        	lhs = tokensToExpression(tokenList.subList(0, op_idx));
        	rhs = tokensToExpression(tokenList.subList(op_idx+1, tokenList.size()));
        	if( (lhs == null) || (rhs == null ) ) {
        		return null;
        	} else {
        		return new BinaryExpression(lhs, op, rhs);
        	}
        }

        parentCount = 0;
        op_idx = token_array.length-1;
        // Finding the operator of the BinaryExpression (^)
        while (op_idx >= 0)
        {
        	token_type tType = token_array[op_idx].getType();
        	// Found the operator
        	if(((tType == token_type.bitwisexor)) && parentCount == 0 && op_idx > 0)
        		break;
        	else if (token_array[op_idx].getVal().compareTo(")") == 0)
        		parentCount++;
        	else if (token_array[op_idx].getVal().compareTo("(") == 0)
        		parentCount--;
        	op_idx--;
        }

        if( (op_idx > 0) && (op_idx < token_array.length-1) ) { 
        	op = BinaryOperator.BITWISE_EXCLUSIVE_OR;
        	lhs = tokensToExpression(tokenList.subList(0, op_idx));
        	rhs = tokensToExpression(tokenList.subList(op_idx+1, tokenList.size()));
        	if( (lhs == null) || (rhs == null ) ) {
        		return null;
        	} else {
        		return new BinaryExpression(lhs, op, rhs);
        	}
        }

        parentCount = 0;
        op_idx = token_array.length-1;
        // Finding the operator of the BinaryExpression (&)
        while (op_idx >= 0)
        {
        	token_type tType = token_array[op_idx].getType();
        	// Found the operator
        	if(((tType == token_type.bitwiseand)) && parentCount == 0 && op_idx > 0)
        		break;
        	else if (token_array[op_idx].getVal().compareTo(")") == 0)
        		parentCount++;
        	else if (token_array[op_idx].getVal().compareTo("(") == 0)
        		parentCount--;
        	op_idx--;
        }

        if( (op_idx > 0) && (op_idx < token_array.length-1) ) { 
        	op = BinaryOperator.BITWISE_AND;
        	lhs = tokensToExpression(tokenList.subList(0, op_idx));
        	rhs = tokensToExpression(tokenList.subList(op_idx+1, tokenList.size()));
        	if( (lhs == null) || (rhs == null ) ) {
        		return null;
        	} else {
        		return new BinaryExpression(lhs, op, rhs);
        	}
        }

        parentCount = 0;
        op_idx = token_array.length-1;
        // Finding the operator of the BinaryExpression (==, !=)
        while (op_idx >= 0)
        {
        	token_type tType = token_array[op_idx].getType();
        	// Found the operator
        	if(((tType == token_type.EQ) || (tType == token_type.NE)) && parentCount == 0 && op_idx > 0)
        		break;
        	else if (token_array[op_idx].getVal().compareTo(")") == 0)
        		parentCount++;
        	else if (token_array[op_idx].getVal().compareTo("(") == 0)
        		parentCount--;
        	op_idx--;
        }

        if( (op_idx > 0) && (op_idx < token_array.length-1) ) { 
        	token_type tType = token_array[op_idx].getType();
        	switch (tType)
        	{
        	case EQ: op = BinaryOperator.COMPARE_EQ;break;
        	case NE: op = BinaryOperator.COMPARE_NE; break;
        	}
        	lhs = tokensToExpression(tokenList.subList(0, op_idx));
        	rhs = tokensToExpression(tokenList.subList(op_idx+1, tokenList.size()));
        	if( (lhs == null) || (rhs == null ) ) {
        		return null;
        	} else {
        		return new BinaryExpression(lhs, op, rhs);
        	}
        }
        
        parentCount = 0;
        op_idx = token_array.length-1;
        // Finding the operator of the BinaryExpression (<, <=, >, >=)
        while (op_idx >= 0)
        {
        	token_type tType = token_array[op_idx].getType();
        	// Found the operator
        	if(((tType == token_type.LT) || (tType == token_type.LE) || (tType == token_type.GT) || (tType == token_type.GE))
        			&& parentCount == 0 && op_idx > 0)
        		break;
        	else if (token_array[op_idx].getVal().compareTo(")") == 0)
        		parentCount++;
        	else if (token_array[op_idx].getVal().compareTo("(") == 0)
        		parentCount--;
        	op_idx--;
        }

        if( (op_idx > 0) && (op_idx < token_array.length-1) ) { 
        	token_type tType = token_array[op_idx].getType();
        	switch (tType)
        	{
        	case LT: op = BinaryOperator.COMPARE_LT;break;
        	case LE: op = BinaryOperator.COMPARE_LE;break;
        	case GT: op = BinaryOperator.COMPARE_GT; break;
        	case GE: op = BinaryOperator.COMPARE_GE; break;
        	}
        	lhs = tokensToExpression(tokenList.subList(0, op_idx));
        	rhs = tokensToExpression(tokenList.subList(op_idx+1, tokenList.size()));
        	if( (lhs == null) || (rhs == null ) ) {
        		return null;
        	} else {
        		return new BinaryExpression(lhs, op, rhs);
        	}
        }
        
        parentCount = 0;
        op_idx = token_array.length-1;
        // Finding the operator of the BinaryExpression (<<, >>)
        while (op_idx >= 0)
        {
        	token_type tType = token_array[op_idx].getType();
        	// Found the operator
        	if(((tType == token_type.leftshift) || (tType == token_type.rightshift)) && parentCount == 0 && op_idx > 0)
        		break;
        	else if (token_array[op_idx].getVal().compareTo(")") == 0)
        		parentCount++;
        	else if (token_array[op_idx].getVal().compareTo("(") == 0)
        		parentCount--;
        	op_idx--;
        }

        if( (op_idx > 0) && (op_idx < token_array.length-1) ) { 
        	token_type tType = token_array[op_idx].getType();
        	switch (tType)
        	{
        	case leftshift: op = BinaryOperator.SHIFT_LEFT;break;
        	case rightshift: op = BinaryOperator.SHIFT_RIGHT; break;
        	}
        	lhs = tokensToExpression(tokenList.subList(0, op_idx));
        	rhs = tokensToExpression(tokenList.subList(op_idx+1, tokenList.size()));
        	if( (lhs == null) || (rhs == null ) ) {
        		return null;
        	} else {
        		return new BinaryExpression(lhs, op, rhs);
        	}
        }
        
        parentCount = 0;
        op_idx = token_array.length-1;
        // Finding the operator of the BinaryExpression (+, -)
        while (op_idx >= 0)
        {
        	token_type tType = token_array[op_idx].getType();
        	// Found the operator
        	if(((tType == token_type.add) || (tType == token_type.subtract)) && parentCount == 0 && op_idx > 0)
        		break;
        	else if (token_array[op_idx].getVal().compareTo(")") == 0)
        		parentCount++;
        	else if (token_array[op_idx].getVal().compareTo("(") == 0)
        		parentCount--;
        	op_idx--;
        }

        if( (op_idx > 0) && (op_idx < token_array.length-1) ) { 
        	token_type tType = token_array[op_idx].getType();
        	switch (tType)
        	{
        	case add: op = BinaryOperator.ADD;break;
        	case subtract: op = BinaryOperator.SUBTRACT; break;
        	}
        	lhs = tokensToExpression(tokenList.subList(0, op_idx));
        	rhs = tokensToExpression(tokenList.subList(op_idx+1, tokenList.size()));
        	if( (lhs == null) || (rhs == null ) ) {
        		return null;
        	} else {
        		return new BinaryExpression(lhs, op, rhs);
        	}
        } 
        
        parentCount = 0;
        op_idx = token_array.length-1;
        // Finding the operator of the BinaryExpression (*, /)
        while (op_idx >= 0)
        {
        	token_type tType = token_array[op_idx].getType();
        	// Found the operator
        	if(((tType == token_type.multiply) || (tType == token_type.divide)) && parentCount == 0 && op_idx > 0)
        		break;
        	else if (token_array[op_idx].getVal().compareTo(")") == 0)
        		parentCount++;
        	else if (token_array[op_idx].getVal().compareTo("(") == 0)
        		parentCount--;
        	op_idx--;
        }

        // This expression is not a binary expression
        if((op_idx == 0) || (op_idx > token_array.length-2))
        	return null;

        token_type tType = token_array[op_idx].getType();
        switch (tType)
        {
        case multiply: op = BinaryOperator.MULTIPLY;break;
        case divide: op = BinaryOperator.DIVIDE; break;
        }
        lhs = tokensToExpression(tokenList.subList(0, op_idx));
        rhs = tokensToExpression(tokenList.subList(op_idx+1, tokenList.size()));
        if( (lhs == null) || (rhs == null) ) {
        	return null;
        } else {
        	return new BinaryExpression(lhs, op, rhs);
        }
    }
    
    /** 
     * checks if the given char c is a float suffix
     */
    private static boolean isFloatSuffix(final char c)
    {   
    	return "fFlL".indexOf(c) != -1; 
    }   

    /** 
     * checks if the given char c is an integer suffix
     */
    private static boolean isIntegerSuffix(final char c)
    {   
    	return "uUlL".indexOf(c) != -1; 
    }   

    /** 
     * checks if the given char c is an exponent
     */
    private static boolean isExponent(final char c)
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

    private static List<Token> lexer(String code, Boolean success)
    {
        List<Token> tokenList = new Vector<Token>();
        char[] code_array = code.toCharArray();

        Token prevToken = null;
        int base_ptr = 0;
        int ptr = 0;
        while(ptr < code_array.length)
        {
            // The token starts with an alphabet. The lexer found either identifier or reserved word
            // TODO: Current implementation does not support reserved word
            if(Character.isLetter(code_array[ptr]))
            {
                boolean canBeReserved = true;
                while (ptr+1 < code_array.length &&
                        (Character.isLetterOrDigit(code_array[ptr+1]) || code_array[ptr+1] == '_'))
                {
                    ptr++;
                    // No reserved word in C contains digit
                    if(Character.isDigit(code_array[ptr]))
                        canBeReserved = false;
                }
                prevToken = new Token(token_type.identifier, code.substring(base_ptr,ptr+1));
                tokenList.add(prevToken);
            }
            // The token starts with an alphabet. The lexer found a numerical constant
            else if(Character.isDigit(code_array[ptr]))
            {
                while (ptr+1 < code_array.length && Character.isDigit(code_array[ptr+1]))
                {
                    ptr++;
                }
                if( (ptr+1 < code_array.length) && isIntegerSuffix(code_array[ptr+1]) ) {
                	//Integer constant
                	ptr++;
                	if( (ptr+1 < code_array.length) && isIntegerSuffix(code_array[ptr+1]) ) {
                		ptr++;
                	}
                	prevToken = new Token(token_type.intconstant, code.substring(base_ptr,ptr+1));
                	tokenList.add(prevToken);
                } else if( (ptr+1 < code_array.length) && (code_array[ptr+1] == '.') ) {
                	//Float constant
                	ptr++;
                	while (ptr+1 < code_array.length && Character.isDigit(code_array[ptr+1]))
                	{
                		//Decimal fraction part of Float constant
                		ptr++;
                	}
                	if( (ptr+1 < code_array.length) && isExponent(code_array[ptr+1]) ) {
                		//Exponent character (e or E)
                		ptr++;
                		if( (ptr+1 < code_array.length) && isUnaryOp(code_array[ptr+1]) ) {
                			//Unary operator for exponent part.
                			ptr++;
                		}
                		while (ptr+1 < code_array.length && Character.isDigit(code_array[ptr+1]))
                		{
                			//Exponent part
                			ptr++;
                		}
                	}
                	if( (ptr+1 < code_array.length) && isFloatSuffix(code_array[ptr+1]) ) {
                		//Float suffix
                		ptr++;
                	}
                	String subStr = code.substring(base_ptr, ptr+1);
                	if( subStr.contains("f") || subStr.contains("F") ) {
                		prevToken = new Token(token_type.floatconstant, subStr);
                	} else {
                		prevToken = new Token(token_type.doubleconstant, subStr);
                	}
                	tokenList.add(prevToken);

                } else {
                	//Assume this is an Integer constant.
                	prevToken = new Token(token_type.intconstant, code.substring(base_ptr,ptr+1));
                	tokenList.add(prevToken);
                }
            }
            else
            {
                switch(code_array[ptr])
                {
                    case '(':prevToken = new Token(token_type.leftparenthesis, code.substring(base_ptr,ptr+1)); tokenList.add(prevToken); break;
                    case ')':prevToken = new Token(token_type.rightparenthesis, code.substring(base_ptr,ptr+1)); tokenList.add(prevToken); break;
                    case '[':prevToken = new Token(token_type.leftbracket, code.substring(base_ptr,ptr+1)); tokenList.add(prevToken); break;
                    case ']':prevToken = new Token(token_type.rightbracket, code.substring(base_ptr,ptr+1)); tokenList.add(prevToken); break;
                    case '+': if( code_array[ptr+1] == '+' ) {token_type tType = null; if( (prevToken.type == token_type.rightbracket) || 
                    		(prevToken.type == token_type.rightparenthesis) || (prevToken.type == token_type.identifier) ||
                    		(prevToken.type == token_type.floatconstant) || (prevToken.type == token_type.doubleconstant) ||
                    		(prevToken.type == token_type.intconstant)) {tType = token_type.postincrement;} else {tType = token_type.preincrement;}; 
                    		prevToken = new Token(tType, code.substring(base_ptr, ptr+2));ptr++;} 
                    	else {prevToken = new Token(token_type.add, code.substring(base_ptr,ptr+1));}; tokenList.add(prevToken); break;
                    case '-': if( code_array[ptr+1] == '-' ) {token_type tType = null; if( (prevToken.type == token_type.rightbracket) || 
                    		(prevToken.type == token_type.rightparenthesis) || (prevToken.type == token_type.identifier) ||
                    		(prevToken.type == token_type.floatconstant) || (prevToken.type == token_type.doubleconstant) ||
                    		(prevToken.type == token_type.intconstant)) {tType = token_type.postdecrement;} else {tType = token_type.predecrement;}; 
                    		prevToken = new Token(tType, code.substring(base_ptr, ptr+2));ptr++;} 
                    	else {prevToken = new Token(token_type.subtract, code.substring(base_ptr,ptr+1));}; tokenList.add(prevToken); break;
                    case '~': prevToken = new Token(token_type.bitwisenot, code.substring(base_ptr,ptr+1)); tokenList.add(prevToken); break;
                    case '*': {token_type tType = null; if( (prevToken.type == token_type.leftparenthesis) || 
                    		(prevToken.type == token_type.leftbracket) || (prevToken.type == token_type.logicalnot) ||
                    		(prevToken.type == token_type.preincrement) || (prevToken.type == token_type.predecrement) ||
                    		(prevToken.type == token_type.comma) || (prevToken.type == token_type.semicolon) ||
                    		(prevToken.type == token_type.dereference) ) {tType = token_type.dereference;}
                    	else {tType = token_type.multiply;}; prevToken = new Token(tType, code.substring(base_ptr,ptr+1)); 
                    	tokenList.add(prevToken);} break;
                    case '&': if( code_array[ptr+1] == '&' ) {prevToken = new Token(token_type.logicaland, 
                    		code.substring(base_ptr, ptr+2)); ptr++;} 
                    		else if( (prevToken.type == token_type.leftparenthesis) || 
                    		(prevToken.type == token_type.leftbracket) || (prevToken.type == token_type.logicalnot) ||
                    		(prevToken.type == token_type.preincrement) || (prevToken.type == token_type.predecrement) ||
                    		(prevToken.type == token_type.comma) || (prevToken.type == token_type.semicolon) ||
                    		(prevToken.type == token_type.dereference) ) {prevToken = new Token(token_type.addressof, code.substring(base_ptr, ptr+1));}
                    	else {prevToken = new Token(token_type.bitwiseand, code.substring(base_ptr,ptr+1));} 
                    	tokenList.add(prevToken); break;
                    case '/': prevToken = new Token(token_type.divide, code.substring(base_ptr,ptr+1)); tokenList.add(prevToken); break;
                    case '!': if( code_array[ptr+1] == '=' ) {prevToken = new Token(token_type.NE, code.substring(base_ptr, ptr+2));ptr++;} 
                    	else {prevToken = new Token(token_type.logicalnot, code.substring(base_ptr,ptr+1));}; tokenList.add(prevToken); break;
                    case '|': if( code_array[ptr+1] == '|' ) {prevToken = new Token(token_type.logicalor, code.substring(base_ptr, ptr+2));ptr++;} 
                    	else {prevToken = new Token(token_type.bitwiseor, code.substring(base_ptr,ptr+1));}; tokenList.add(prevToken); break;
                    case '<': if( code_array[ptr+1] == '<' ) {prevToken = new Token(token_type.leftshift, code.substring(base_ptr, ptr+2));ptr++;} 
                    	else if( code_array[ptr+1] == '=' ) {prevToken = new Token(token_type.LE, code.substring(base_ptr, ptr+2));ptr++;} 
                    	else {prevToken = new Token(token_type.LT, code.substring(base_ptr,ptr+1));}; tokenList.add(prevToken); break;
                    case '>': if( code_array[ptr+1] == '>' ) {prevToken = new Token(token_type.rightshift, code.substring(base_ptr, ptr+2));ptr++;} 
                    	else if( code_array[ptr+1] == '=' ) {prevToken = new Token(token_type.GE, code.substring(base_ptr, ptr+2));ptr++;} 
                    	else {prevToken = new Token(token_type.GT, code.substring(base_ptr,ptr+1));}; tokenList.add(prevToken); break;
                    case '=': if( code_array[ptr+1] == '=' ) {prevToken = new Token(token_type.EQ, code.substring(base_ptr, ptr+2));ptr++;} 
                    	else {prevToken = new Token(token_type.directassign, code.substring(base_ptr,ptr+1));}; tokenList.add(prevToken); break;
                    case '^': prevToken = new Token(token_type.bitwisexor, code.substring(base_ptr,ptr+1)); tokenList.add(prevToken); break;
                    case ',':prevToken = new Token(token_type.comma, code.substring(base_ptr,ptr+1)); tokenList.add(prevToken); break;
                    case ';':prevToken = new Token(token_type.semicolon, code.substring(base_ptr,ptr+1)); tokenList.add(prevToken); break;
                    default: return null;//Tools.exit("[ParserTools] cannot recognize token: " + code_array[ptr]);break;
                }
            }
            ptr++;
            base_ptr = ptr;
        }
        return tokenList;
    }
}

enum expression_type
{
    constant,
    identifier,
    binary_expression,
    array_access
}

enum token_type
{
    intconstant,
    floatconstant,
    doubleconstant,
    identifier,
    postincrement,
    postdecrement,
    leftparenthesis,
    rightparenthesis,
    leftbracket,
    rightbracket,
    preincrement,
    predecrement,
    dereference,
    addressof,
    multiply,
    divide,
    modulo,
    add,
    subtract,
    LT,
    LE,
    GT,
    GE,
    EQ,
    NE,
    directassign,
    comma,
    semicolon,
    logicalnot,
    logicaland,
    logicalor,
    bitwisenot,
    bitwiseand,
    bitwiseor,
    bitwisexor,
    leftshift,
    rightshift
}

/**
 * <b>Token</b>
 */
class Token
{
    token_type type;
    String val;

    Token(token_type type, String val) {
        this.type = type;
        this.val = val;
    }

    public token_type getType() {
        return type;
    }

    public void setType(token_type type) {
        this.type = type;
    }

    public String getVal() {
        return val;
    }

    public void setVal(String val) {
        this.val = val;
    }

    @Override
    public String toString() {
        return "Token{" + type + "(" + val + ")}";
    }
}

