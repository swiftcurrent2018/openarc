package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.util.ArrayList;

/**
 * Represents a string literal in the program.
 * 
 * [Modified by Joel E. Denny to handle wide string literals and to build
 * character value arrays.]
 */
public class StringLiteral extends Literal {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = StringLiteral.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }
    
    private String value;
    boolean wide;
    private long[] valueArray;

    /** Constructs a string literal with the specified string value. */
    public StringLiteral(String s) {
        object_print_method = class_print_method;
        value = s;
        wide = false;
        valueArray = null;
    }

    /** Returns a clone of the string literal. */
    @Override
    public StringLiteral clone() {
        StringLiteral o = (StringLiteral)super.clone();
        o.value = value;
        o.wide = wide;
        o.valueArray = valueArray != null ? valueArray.clone() : null;
        return o;
    }

    /**
    * Prints a literal to a stream.
    *
    * @param l The literal to print.
    * @param o The writer on which to print the literal.
    */
    public static void defaultPrint(StringLiteral l, PrintWriter o) {
        o.print(l.toString());
    }

    /** Returns a string representation of the string literal. */
    @Override
    public String toString() {
        return (wide?"L":"") + "\"" + value + "\"";
    }

    /** Compares the string literal with the specified object for equality. */
    @Override
    public boolean equals(Object o) {
        return (super.equals(o) && value.equals(((StringLiteral)o).value));
    }

    /** Returns the value of the string literal. */
    public String getValue() {
        return value;
    }

    /**
     * Is this a wide string literal? {@link #stripQuotes} must be invoked to
     * discover {@code L} prefix.
     */
    public boolean isWide() {
        return wide;
    }

    /**
     * Returns the character value array for this string's contents, or null
     * if {@link #stripQuotes} has not yet been invoked.
     */
    public long[] getValueArray() {
        return valueArray;
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
     * Sets the value of the literal with the specified string value.
     * Resets {@link #getValueArray} to return null.
     */
    public void setValue(String value) {
        this.value = value;
        this.wide = false;
        this.valueArray = null;
    }

    /**
     * Same as {@link #findFirstNotOf(String, int, int, String)} but with
     * {@code end = str.length()}.
     */
    private static int findFirstNotOf(String str, int begin, String set) {
        return findFirstNotOf(str, begin, str.length(), set);
    }

    /**
     * Search a string for any character not in a given set.
     * 
     * @param str
     *            the string to search
     * @param begin
     *            the index of the character at which to start the search
     * @param end
     *            the index of the character after the last character to
     *            search
     * @param set
     *            a string containing the set of characters at which the
     *            search should not stop
     * @return the index of the first character not in {@code set}, or the
     *         lesser of {@code end} and {@code str.length()} if none
     */
    private static int findFirstNotOf(String str, int begin, int end,
                                      String set)
    {
        if (str.length() < end)
            end = str.length();
        for (int i = begin; i < end; ++i) {
            char ch = str.charAt(i);
            int j;
            for (j = 0; j < set.length() && ch != set.charAt(j); ++j)
                ;
            if (j == set.length())
                return i;
        }
        return end;
    }

    /**
     * Strip any leading {@code L}, strip any outer quotes, and build
     * character value array.
     * 
     * <p>
     * Removes outer quotes from the string; this class automatically prints
     * quotes around the string so it does not need to store them. Thus, if
     * you create a StringLiteral with a String that already has quotes around
     * it, you will have double quotes, and may need to call this method.
     * </p>
     * 
     * <p>
     * This method also removes any {@code L} preceding the opening outer
     * quotes, and then sets {@link #isWide} to return true.
     * </p>
     * 
     * <p>
     * This method does not remove embedded quotes in the case of adjacent
     * string literals, such as {@code "abc""def"}, as doing so could corrupt
     * a hexadecimal or octal escape sequence that precedes closing quotes.
     * Thus, in the case of adjacent string literals, {@link #getValue} will
     * show embedded quotes but not outer quotes. This method also does not
     * remove any {@code L} preceding adjacent string literals.
     * </p>
     * 
     * <p>
     * This method skips both the removal of any leading {@code L} and the
     * removal of outer quotes if they are not present. In simple cases, that
     * means this method can be applied multiple times, and only the first
     * application will have any effect. However, there exist cases where that
     * is not true. For example, {@code "L""\""} becomes {@code L""\"} becomes
     * {@code "\}.
     * </p>
     * 
     * <p>
     * Finally, this method builds an array of the string's character values,
     * to be returned by {@link #getValueArray}. Each value is the raw value
     * as described by {@link CharLiteral#getValue} and
     * {@link EscapeLiteral#getValue}. This method also appends a null byte to
     * that array. Along the way, if there are adjacent string literals with a
     * preceding {@code L}, such as {@code "abc"L"def"}, this method sets
     * {@link #isWide} to return true.
     * </p>
     */
    public void stripQuotes() {
        if (value.startsWith("L\"") && value.endsWith("\"")) {
            wide = true;
            value = value.substring(2, value.length()-1);
        }
        else if (value.startsWith("\"") && value.endsWith("\""))
            value = value.substring(1, value.length() - 1);
        ArrayList<Long> valueList = new ArrayList<Long>();
        for (int i = 0; i < value.length(); ++i) {
            switch (value.charAt(i)) {
            case '"':
                ++i;
                if (i < value.length() && value.charAt(i) == 'L') {
                    wide = true;
                    ++i;
                }
                if (i < value.length() && value.charAt(i) == '"')
                    ++i;
                else {
                    PrintTools.printlnStatus(0, "Unmatched Quotes",
                                             toString());
                }
                --i;
                break;
            case '\\': {
                ++i;
                String esc;
                switch (value.charAt(i)) {
                case '0': case '1': case '2': case '3':
                case '4': case '5': case '6': case '7':
                {
                    final int end
                        = findFirstNotOf(value, i+1, i+3, "01234567");
                    esc = "'\\" + value.substring(i, end) + "'";
                    i = end;
                    break;
                }
                case 'x': {
                    ++i;
                    final int end
                        = findFirstNotOf(value, i, "0123456789ABCDEFabcdef");
                    esc = "'\\x" + value.substring(i, end) + "'";
                    i = end;
                    break;
                }
                default: {
                    esc = "'\\" + value.charAt(i) + "'";
                    ++i;
                    break;
                }
                }
                valueList.add(new EscapeLiteral(esc).getValue());
                --i;
                break;
            }
            default:
                valueList.add(Long.valueOf(value.charAt(i)));
                break;
            }
        }
        valueList.add(Long.valueOf(0));
        valueArray = new long[valueList.size()];
        for (int i = 0; i < valueList.size(); ++i)
            valueArray[i] = valueList.get(i);
    }

    @Override
    public void accept(TraversableVisitor v) { v.visit(this); }
}
