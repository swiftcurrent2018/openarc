package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;

/**
 * Represents an escape character.
 * 
 * [Modified by Joel E. Denny to handle wide characters.]
 */
public class EscapeLiteral extends Literal {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = EscapeLiteral.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }
    
    private String name;
    private boolean wide;
    private long value;

    /** Constructs an escape literal with the specified string name. */
    public EscapeLiteral(String name) {
        this.name = name;
        this.wide = name.charAt(0) == 'L';
        int start = this.wide ? 3 : 2;
        final char c = name.charAt(start);
        switch (c) {
        case 'a':
            this.value = '\7';
            break;
        case 'b':
            this.value = '\b';
            break;
        case 'f':
            this.value = '\f';
            break;
        case 'n':
            this.value = '\n';
            break;
        case 'r':
            this.value = '\r';
            break;
        case 't':
            this.value = '\t';
            break;
        case 'v':
            this.value = '\13';
            break;
        case '\\':
            this.value = '\\';
            break;
        case '?':
            this.value = '\77';
            break;
        case '\'':
            this.value = '\'';
            break;
        case '\"':
            this.value = '\"';
            break;
        case 'x':
            this.value = Long.parseLong(name.substring(
                            start+1, name.length()-1), 16);
            break;
        default:
            if (c <= '7' && c >= '0') {
                this.value = Long.parseLong(name.substring(
                                start, name.length()-1), 8);
            } else {
                this.value = '?';
                PrintTools.printlnStatus(0,
                        "Unrecognized Escape Sequence", name);
            }
            break;
        }
        object_print_method = class_print_method;
    }

    /** Returns a clone of the escape literal. */
    @Override
    public EscapeLiteral clone() {
        EscapeLiteral o = (EscapeLiteral)super.clone();
        o.name = name;
        o.wide = wide;
        o.value = value;
        return o;
    }

    /**
    * Prints a literal to a stream.
    *
    * @param l The literal to print.
    * @param o The writer on which to print the literal.
    */
    public static void defaultPrint(EscapeLiteral l, PrintWriter o) {
        o.print(l.name);
    }

    /** Returns a string representation of the escape literal. */
    @Override
    public String toString() {
        return name;
    }

    /** Compares the escape literal with the specified object for equality. */
    @Override
    public boolean equals(Object o) {
        return (super.equals(o) && value == ((EscapeLiteral)o).value);
    }

    /** Is this a wide char literal, like {@code L'\n'}? */
    public boolean isWide() {
        return wide;
    }

    /**
     * Returns the character value of the escape literal. This is the raw value
     * before the sign-extension to int specified in ISO C99 sec. 6.4.4.4p13.
     * Whether that should happen is implementation-specific.
     */
    public long getValue() {
        return value;
    }

    /** Returns the hash code of the escape literal. */
    @Override
    public int hashCode() {
        return toString().hashCode();
    }

    /**
    * Overrides the class print method, so that all subsequently created
    * objects will use the supplied method.
    *
    * @param m The new print method.
    */
    static public void setClassPrintMethod(Method m) {
        class_print_method = m;
    }

    /** Sets the value of the literal with the specified character. */
    public void setValue(char value) {
        this.name = "'" + value + "'";
        this.wide = false;
        this.value = value;
    }

    @Override
    public void accept(TraversableVisitor v) { v.visit(this); }
}
