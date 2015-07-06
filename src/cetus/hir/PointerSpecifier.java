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

    // [NVL support added by Joel E. Denny]

    /** * nvl */
    public static final PointerSpecifier NVL =
            new PointerSpecifier(Specifier.NVL);

    /** * const nvl */
    public static final PointerSpecifier CONST_NVL =
            new PointerSpecifier(Specifier.CONST, Specifier.NVL);

    /** * volatile nvl */
    public static final PointerSpecifier VOLATILE_NVL =
            new PointerSpecifier(Specifier.VOLATILE, Specifier.NVL);

    /** * restrict nvl */
    public static final PointerSpecifier RESTRICT_NVL =
            new PointerSpecifier(Specifier.RESTRICT, Specifier.NVL);

    /** * const volatile nvl */
    public static final PointerSpecifier CONST_VOLATILE_NVL =
            new PointerSpecifier(Specifier.CONST, Specifier.VOLATILE,
                                 Specifier.NVL);

    /** * const restrict nvl */
    public static final PointerSpecifier CONST_RESTRICT_NVL =
            new PointerSpecifier(Specifier.CONST, Specifier.RESTRICT,
                                 Specifier.NVL);

    /** * restrict volatile nvl */
    public static final PointerSpecifier RESTRICT_VOLATILE_NVL =
            new PointerSpecifier(Specifier.RESTRICT, Specifier.VOLATILE,
                                 Specifier.NVL);

    /** * const restrict volatile nvl */
    public static final PointerSpecifier CONST_RESTRICT_VOLATILE_NVL =
            new PointerSpecifier(Specifier.CONST, Specifier.RESTRICT,
                                 Specifier.VOLATILE, Specifier.NVL);

    /** * nvl_wp */
    public static final PointerSpecifier NVL_WP =
            new PointerSpecifier(Specifier.NVL_WP);

    /** * const nvl_wp */
    public static final PointerSpecifier CONST_NVL_WP =
            new PointerSpecifier(Specifier.CONST, Specifier.NVL_WP);

    /** * volatile nvl_wp */
    public static final PointerSpecifier VOLATILE_NVL_WP =
            new PointerSpecifier(Specifier.VOLATILE, Specifier.NVL_WP);

    /** * restrict nvl_wp */
    public static final PointerSpecifier RESTRICT_NVL_WP =
            new PointerSpecifier(Specifier.RESTRICT, Specifier.NVL_WP);

    /** * const volatile nvl_wp */
    public static final PointerSpecifier CONST_VOLATILE_NVL_WP =
            new PointerSpecifier(Specifier.CONST, Specifier.VOLATILE,
                                 Specifier.NVL_WP);

    /** * const restrict nvl_wp */
    public static final PointerSpecifier CONST_RESTRICT_NVL_WP =
            new PointerSpecifier(Specifier.CONST, Specifier.RESTRICT,
                                 Specifier.NVL_WP);

    /** * restrict volatile nvl_wp */
    public static final PointerSpecifier RESTRICT_VOLATILE_NVL_WP =
            new PointerSpecifier(Specifier.RESTRICT, Specifier.VOLATILE,
                                 Specifier.NVL_WP);

    /** * const restrict volatile nvl_wp */
    public static final PointerSpecifier CONST_RESTRICT_VOLATILE_NVL_WP =
            new PointerSpecifier(Specifier.CONST, Specifier.RESTRICT,
                                 Specifier.VOLATILE, Specifier.NVL_WP);

    /** * nvl nvl_wp */
    public static final PointerSpecifier NVL_NVL_WP =
            new PointerSpecifier(Specifier.NVL, Specifier.NVL_WP);

    /** * const nvl nvl_wp */
    public static final PointerSpecifier CONST_NVL_NVL_WP =
            new PointerSpecifier(Specifier.CONST, Specifier.NVL,
                                 Specifier.NVL_WP);

    /** * volatile nvl nvl_wp */
    public static final PointerSpecifier VOLATILE_NVL_NVL_WP =
            new PointerSpecifier(Specifier.VOLATILE, Specifier.NVL,
                                 Specifier.NVL_WP);

    /** * restrict nvl nvl_wp */
    public static final PointerSpecifier RESTRICT_NVL_NVL_WP =
            new PointerSpecifier(Specifier.RESTRICT, Specifier.NVL,
                                 Specifier.NVL_WP);

    /** * const volatile nvl nvl_wp */
    public static final PointerSpecifier CONST_VOLATILE_NVL_NVL_WP =
            new PointerSpecifier(Specifier.CONST, Specifier.VOLATILE,
                                 Specifier.NVL, Specifier.NVL_WP);

    /** * const restrict nvl nvl_wp */
    public static final PointerSpecifier CONST_RESTRICT_NVL_NVL_WP =
            new PointerSpecifier(Specifier.CONST, Specifier.RESTRICT,
                                 Specifier.NVL, Specifier.NVL_WP);

    /** * restrict volatile nvl nvl_wp */
    public static final PointerSpecifier RESTRICT_VOLATILE_NVL_NVL_WP =
            new PointerSpecifier(Specifier.RESTRICT, Specifier.VOLATILE,
                                 Specifier.NVL, Specifier.NVL_WP);

    /** * const restrict volatile nvl nvl_wp */
    public static final PointerSpecifier CONST_RESTRICT_VOLATILE_NVL_NVL_WP =
            new PointerSpecifier(Specifier.CONST, Specifier.RESTRICT,
                                 Specifier.VOLATILE, Specifier.NVL,
                                 Specifier.NVL_WP);

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

    // [Added by Joel E. Denny]
    public List<Specifier> getQualifiers() {
      return qualifiers == null ? new ArrayList<Specifier>() : qualifiers;
    }

    public void print(PrintWriter o) {
        o.print("*");
        if (qualifiers != null && !qualifiers.isEmpty()) {
            o.print(" ");
            PrintTools.printListWithSpace(qualifiers, o);
        }
    }

}
