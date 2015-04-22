package openacc.hir;

import cetus.hir.Printable;
import cetus.hir.BinaryOperator;
import java.util.HashMap;
import java.io.PrintWriter;

public class ReductionOperator implements Printable {

    private static HashMap<String, ReductionOperator> op_map =
            new HashMap<String, ReductionOperator>(32);

    private static String[] names = {
            "+", "&", "^", "|", "&&", "||", "*", "min", "max"};
    
    /**
    * +
    */
    public static final ReductionOperator ADD = new ReductionOperator(0);

    /**
    * &amp;
    */
    public static final ReductionOperator BITWISE_AND = new ReductionOperator(1);

    /**
    * ^
    */
    public static final ReductionOperator BITWISE_EXCLUSIVE_OR =
            new ReductionOperator(2);

    /**
    * |
    */
    public static final ReductionOperator BITWISE_INCLUSIVE_OR =
            new ReductionOperator(3);

    /**
    * &amp;&amp;
    */
    public static final ReductionOperator LOGICAL_AND = new ReductionOperator(4);

    /**
    * ||
    */
    public static final ReductionOperator LOGICAL_OR = new ReductionOperator(5);

    /**
    * *
    */
    public static final ReductionOperator MULTIPLY = new ReductionOperator(6);
    
    /**
    * min
    */
    public static final ReductionOperator MIN = new ReductionOperator(7);
    /**
    * max
    */
    public static final ReductionOperator MAX = new ReductionOperator(8);

    protected int value;
    
	protected ReductionOperator() {
		
	}
	
    /**
    * Used internally -- you may not create arbitrary reduction operators
    * and may only use the ones provided as static members.
    *
    * @param value The numeric code of the operator.
    */
    private ReductionOperator(int value) {
        this.value = value;
        op_map.put(names[value], this);
    }
    
    /**
    * Returns a reduction operator that matches the specified string <tt>s</tt>.
    * @param s the string to be matched.
    * @return the matching operator or null if not found.
    */
    public static ReductionOperator fromString(String s) {
        return op_map.get(s);
    }
    
    /**
     * Return a corresponding binary operator if existing.
     * 
     * @return the matching binary operator or null if not found.
     */
    public BinaryOperator getBinaryOperator() {
    	return BinaryOperator.fromString(names[value]);
    }
	
    /* It is not necessary to override equals or provide cloning, because
       all possible operators are provided as static objects. */

    public void print(PrintWriter o) {
        o.print(names[value]);
    }

    @Override
    public String toString() {
        return names[value];
    }

}
