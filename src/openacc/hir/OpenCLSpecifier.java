package openacc.hir;

import cetus.hir.Specifier;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.HashMap;

/**
 * <b>OpenCLSpecifier</b> represents type specifiers and modifiers for OpenCL.
 * 
 * @author Putt Sakdhnagool <psakdhna@purdue.edu>
 *         Future TechnologiesGroup, Oak Ridge National Laboratory
 *
 */
public class OpenCLSpecifier extends Specifier
{
  private static HashMap<String, OpenCLSpecifier> spec_map = new HashMap(24);

  private static String[] names =
    {   "__kernel", "__global", "__local", "__constant", "channel" };

  /* The following type-specifiers are added to support CUDA */
  public static final OpenCLSpecifier OPENCL_KERNEL	= new OpenCLSpecifier(0);
  public static final OpenCLSpecifier OPENCL_GLOBAL	= new OpenCLSpecifier(1);
  public static final OpenCLSpecifier OPENCL_LOCAL	= new OpenCLSpecifier(2);
  public static final OpenCLSpecifier OPENCL_CONSTANT	= new OpenCLSpecifier(3);
  public static final OpenCLSpecifier OPENCL_CHANNEL	= new OpenCLSpecifier(4);

  protected int cvalue;

  protected OpenCLSpecifier()
  {
    cvalue = -1;
  }

  private OpenCLSpecifier(int cvalue)
  {
    this.cvalue = cvalue;
    spec_map.put(names[cvalue], this);
  }

  /** Prints the specifier to the print writer. */
  public void print(PrintWriter o)
  {
    if (cvalue >= 0)
      o.print(names[cvalue]);
  }

  /** Returns a string representation of the specifier. */
  @Override
  public String toString()
  {
    StringWriter sw = new StringWriter(16);
    print(new PrintWriter(sw));
    return sw.toString();
  }
}
