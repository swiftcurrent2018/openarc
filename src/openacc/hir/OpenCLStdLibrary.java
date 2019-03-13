package openacc.hir;

import java.util.*;
import cetus.hir.*;

/**
 * Repository for OpenCL-supported standard library functions. This class provides a basic
 * information about the standard library calls. Knowing that if a function call
 * may or must not have side effects can greatly improve the precision of a
 * program analysis in general. 
 * 
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future TechnologiesGroup, Oak Ridge National Laboratory
 *
 */
public class OpenCLStdLibrary
{
  /** Only a single object is constructed. */
  private static final OpenCLStdLibrary std = new OpenCLStdLibrary();

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
  *  supported by OpenCL runtime system.
  * @param fcall the function call to be examined.
  * @return true if the function call exists in the entries.
  */
  public static boolean contains(FunctionCall fcall)
  {
    return (std.catalog.get(fcall.getName().toString()) != null);
  }
  
  public static boolean contains(String fcallName)
  {
    return (std.catalog.get(fcallName) != null);
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
  private OpenCLStdLibrary()
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
	// by the OpenCL runtime library version 1.1
	// When compiling for devices without native double type precision support,
	// double precision math functions are mapped to their single precision
	// equivalents by OpenCL compiler.
    add("sqrt");
    add("rsqrt");
    add("cbrt");
    add("rcbrt");
    add("hypot");
    add("exp");
    add("exp2");
    add("exp10");
    add("expm1");
    add("log");
    add("log2");
    add("log10");
    add("log1p");
    add("sin");
    add("sinpi");
    add("cos");
    add("cospi");
    add("tan");
    add("tanpi");
    add("sincos");
    add("asin");
    add("acos");
    add("atan");
    add("atan2");
    add("sinh");
    add("cosh");
    add("tanh");
    add("cosh");
    add("tanh");
    add("asinh");
    add("acosh");
    add("atanh");
    add("asinpi");
    add("acospi");
    add("atanpi");
    add("atan2pi");
    add("pow");
    add("erf");
    add("erfc");
    add("lgamma");
    add("tgamma");
    add("fma");
    add("frexp",     Property.SIDE_EFFECT_PARAMETER);
    add("ldexp");
    add("scalbn");
    add("scalbln");
    add("logb");
    add("ilogb");
    add("fmod");
    add("remainder");
    add("remquo",    Property.SIDE_EFFECT_PARAMETER);
    add("modf",      Property.SIDE_EFFECT_PARAMETER);
    add("fdim");
    add("trunc");
    add("round");
    add("rint");
    add("nearbyint");
    add("ceil");
    add("floor");
    add("lrint");
    add("lround");
    add("llrint");
    add("llround");
    add("isinf");
    add("isnan");
    add("isfinite");
    add("copysign");
    add("fmin");
    add("fmax");
    add("fabs");
    add("nan");
    add("nextafter");
    //Non-standard functions supported by OpenCL
    add("mad");
    add("fract");
    add("maxmag");
    add("minmag");
    add("powr");
    add("rootn");
    add("printf");
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
