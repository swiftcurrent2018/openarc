package openacc.hir;

import java.util.*;
import cetus.hir.*;

/**
 * Repository for CUDA-supported standard library functions. This class provides a basic
 * information about the standard library calls. Knowing that if a function call
 * may or must not have side effects can greatly improve the precision of a
 * program analysis in general. 
 * 
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future TechnologiesGroup, Oak Ridge National Laboratory
 *
 */
public class CUDAStdLibrary
{
  /** Only a single object is constructed. */
  private static final CUDAStdLibrary std = new CUDAStdLibrary();

  /** Predefined properties for each library functions */
  private Map<String, Set<Property>> catalog;

  /** Predefined set of properties */
  private enum Property
  {
    SIDE_EFFECT_GLOBAL,    // contains side effects on global variables.
    SIDE_EFFECT_PARAMETER, // contains side effects through parameters.
  }

  /**
  * Checks if the given function call is a standard library call
  *  supported by CUDA runtime system.
  * @param fcall the function call to be examined.
  * @return true if the function call exists in the entries.
  */
  public static boolean contains(FunctionCall fcall)
  {
    return (std.catalog.get(fcall.getName().toString()) != null);
  }

  /**
  * Checks if the given function call may have side effects.
  * @param fcall the function call to be examined.
  * @return true if the function call has a side effect.
  */
  public static boolean isSideEffectFree(FunctionCall fcall)
  {
    if ( !contains(fcall) )
      return false;
    Set<Property> properties = std.catalog.get(fcall.getName().toString());
    return (
      !properties.contains(Property.SIDE_EFFECT_GLOBAL) &&
      !properties.contains(Property.SIDE_EFFECT_PARAMETER)
    );
  }

  /** Constructs a new repository */
  private CUDAStdLibrary()
  {
    catalog = new HashMap<String, Set<Property>>();
    addEntries();
  }

  /**
  * Adds each entry to the repository. The same properties are assigned  
  * as the ones in cetus.hir.StandardLibrary
  */
  private void addEntries()
  {
    // Mathematical standard library functions supported 
	// by the CUDA runtime library version 4.0
	// When compiling for devices without native double type precision support,
	// double precision math functions are mapped to their single precision
	// equivalents by CUDA compiler.
    add("sqrtf");
    add("sqrt");
    add("rsqrtf");
    add("rsqrt");
    add("cbrtf");
    add("cbrt");
    add("rcbrtf");
    add("rcbrt");
    add("hypotf");
    add("hypot");
    add("expf");
    add("exp");
    add("exp2f");
    add("exp2");
    add("exp10f");
    add("exp10");
    add("expm1f");
    add("expm1");
    add("logf");
    add("log");
    add("log2f");
    add("log2");
    add("log10f");
    add("log10");
    add("log1pf");
    add("log1p");
    add("sinf");
    add("sin");
    add("sinpif");
    add("sinpi");
    add("cosf");
    add("cos");
    add("cospif");
    add("cospi");
    add("tanf");
    add("tan");
    add("tanpi");
    add("tanpif");
    add("sincosf");
    add("sincos");
    add("asinf");
    add("asin");
    add("acosf");
    add("acos");
    add("atanf");
    add("atan");
    add("atan2f");
    add("atan2");
    add("sinhf");
    add("sinh");
    add("coshf");
    add("cosh");
    add("tanhf");
    add("tanh");
    add("asinhf");
    add("asinh");
    add("coshf");
    add("cosh");
    add("tanhf");
    add("tanh");
    add("acoshf");
    add("acosh");
    add("atanhf");
    add("atanh");
    add("powf");
    add("pow");
    add("erff");
    add("erf");
    add("erfcf");
    add("erfc");
    add("lgammaf");
    add("lgamma");
    add("tgammaf");
    add("tgamma");
    add("fmaf");
    add("fma");
    add("frexpf",     Property.SIDE_EFFECT_PARAMETER);
    add("frexp",     Property.SIDE_EFFECT_PARAMETER);
    add("ldexpf");
    add("ldexp");
    add("scalbnf");
    add("scalbn");
    add("scalblnf");
    add("scalbln");
    add("logbf");
    add("logb");
    add("ilogbf");
    add("ilogb");
    add("fmodf");
    add("fmod");
    add("remainderf");
    add("remainder");
    add("remquof",    Property.SIDE_EFFECT_PARAMETER);
    add("remquo",    Property.SIDE_EFFECT_PARAMETER);
    add("modff",      Property.SIDE_EFFECT_PARAMETER);
    add("modf",      Property.SIDE_EFFECT_PARAMETER);
    add("fdimf");
    add("fdim");
    add("truncf");
    add("trunc");
    add("roundf");
    add("round");
    add("rintf");
    add("rint");
    add("nearbyintf");
    add("nearbyint");
    add("ceilf");
    add("ceil");
    add("floorf");
    add("floor");
    add("lrintf");
    add("lrint");
    add("lroundf");
    add("lround");
    add("llrintf");
    add("llrint");
    add("llroundf");
    add("llround");
    add("signbit");
    add("isinf");
    add("isnan");
    add("isfinite");
    add("copysignf");
    add("copysign");
    add("fminf");
    add("fmin");
    add("fmaxf");
    add("fmax");
    add("fabsf");
    add("fabs");
    add("nanf");
    add("nan");
    add("nextafterf");
    add("nextafter");
    add("printf");
    add("malloc");
    add("free", Property.SIDE_EFFECT_PARAMETER);
    add("memcpy", Property.SIDE_EFFECT_PARAMETER);
    add("memset", Property.SIDE_EFFECT_PARAMETER);
  }

  /** Adds the specified properties to the call */
  private void add(String name, Property... properties)
  {
    catalog.put(name, EnumSet.noneOf(Property.class));
    Set<Property> props = catalog.get(name);
    for ( Property property : properties )
      props.add(property);
  }
}
