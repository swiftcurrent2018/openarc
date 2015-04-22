package cetus.hir;

import java.io.PrintWriter;
import java.util.List;
import java.util.ArrayList;

/** Represents a C or C++ pointer. */
public class PointerSpecifier extends Specifier {

    /** * */
    public static final PointerSpecifier UNQUALIFIED = new PointerSpecifier();

    /** * const */
    public static final PointerSpecifier CONST =
            new PointerSpecifier(Specifier.CONST);

    /** * volatile */
    public static final PointerSpecifier VOLATILE =
            new PointerSpecifier(Specifier.VOLATILE);

    /** * restrict */
    public static final PointerSpecifier RESTRICT =
            new PointerSpecifier(Specifier.RESTRICT);

    /** * const volatile */
    public static final PointerSpecifier CONST_VOLATILE =
            new PointerSpecifier(Specifier.CONST, Specifier.VOLATILE);

    /** * const restrict */
    public static final PointerSpecifier CONST_RESTRICT =
            new PointerSpecifier(Specifier.CONST, Specifier.RESTRICT);

    /** * restrict volatile */
    public static final PointerSpecifier RESTRICT_VOLATILE =
            new PointerSpecifier(Specifier.RESTRICT, Specifier.VOLATILE);

    /** * const restrict volatile */
    public static final PointerSpecifier CONST_RESTRICT_VOLATILE =
            new PointerSpecifier(Specifier.CONST, Specifier.RESTRICT,
                                 Specifier.VOLATILE);

    private List<Specifier> qualifiers;

    private PointerSpecifier(Specifier ... specifiers) {
        if (specifiers.length > 0) {
            qualifiers = new ArrayList<Specifier>(specifiers.length);
            for (Specifier specifier : specifiers) {
                qualifiers.add(specifier);
            }
        } else {
            qualifiers = null;
        }
    }

    public void print(PrintWriter o) {
        o.print("*");
        if (qualifiers != null && !qualifiers.isEmpty()) {
            o.print(" ");
            PrintTools.printListWithSpace(qualifiers, o);
        }
    }

}
