package cetus.hir;

import java.io.PrintWriter;
import java.lang.reflect.Method;

/**
 * Represents an unescaped character literal in the program.
 * 
 * [Modified by Joel E. Denny to handle wide characters.]
 */
public class CharLiteral extends Literal {

    private static Method class_print_method;

    static {
        Class<?>[] params = new Class<?>[2];
        try {
            params[0] = CharLiteral.class;
            params[1] = PrintWriter.class;
            class_print_method = params[0].getMethod("defaultPrint", params);
        } catch(NoSuchMethodException e) {
            throw new InternalError();
        }
    }
    
    private String name;
    private boolean wide;
    private long value;

    /**
    * Constructs a char literal from the specified string.
    *
    * @param value the character value.
    */
    public CharLiteral(String name) {
        object_print_method = class_print_method;
        this.name = name;
        this.wide = name.charAt(0) == 'L';
        this.value = this.wide ? name.charAt(2) : name.charAt(1);
    }

    /** Returns a clone of the char literal. */
    @Override
    public CharLiteral clone() {
        CharLiteral o = (CharLiteral)super.clone();
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
    public static void defaultPrint(CharLiteral l, PrintWriter o) {
        o.print(l.name);
    }

    /** Returns a string representation of the char literal. */
    @Override
    public String toString() {
        return name;
    }

    /** Compares the char literal with the given object for equality. */
    @Override
    public boolean equals(Object o) {
        return (super.equals(o) && value == ((CharLiteral)o).value);
    }

    /** Is this a wide char literal, like {@code L'x'}? */
    public boolean isWide() {
        return wide;
    }

    /**
     * Returns the character value of the char literal. This is the raw value
     * before the sign-extension to int specified in ISO C99 sec. 6.4.4.4p13.
     * Whether that should happen is implementation-specific.
     */
    public long getValue() {
        return value;
    }

    /** Returns the hash code of the char literal. */
    @Override
    public int hashCode() {
        return toString().hashCode();
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

    /** Sets the value of the literal with the specified character. */
    public void setValue(char value) {
        this.name = "'" + value + "'";
        this.wide = false;
        this.value = value;
    }

    @Override
    public void accept(TraversableVisitor v) { v.visit(this); }
}
