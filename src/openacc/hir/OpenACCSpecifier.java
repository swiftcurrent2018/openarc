/**
 * @author Seyong Lee <lees2@ornl.gov>
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
package openacc.hir;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.HashMap;

import cetus.hir.Specifier;

/**
 * @author f6l
 *
 */
public class OpenACCSpecifier extends Specifier {

	private static HashMap<String, OpenACCSpecifier> spec_map = new HashMap(24);

	private static String[] names =
		{   "h_void", "d_void", "HI_device_mem_handle_t", 
		"size_t", "std::string", "\"C\"" };

	/* The following type-specifiers are added to support CUDA */
	public static final OpenACCSpecifier OpenACC_H_VOID	= new OpenACCSpecifier(0);
	public static final OpenACCSpecifier OpenACC_D_VOID	= new OpenACCSpecifier(1);
	public static final OpenACCSpecifier OpenACC_HI_DEV_MEM_HANDLE = new OpenACCSpecifier(2);
	/* size_t is a macro, but treat it as a specifier for convenience */
	public static final OpenACCSpecifier SIZE_T = new OpenACCSpecifier(3);
	public static final OpenACCSpecifier STRING	= new OpenACCSpecifier(4);
	public static final OpenACCSpecifier EXTERN_C	= new OpenACCSpecifier(5);

	protected int cvalue;

	protected OpenACCSpecifier()
	{
		cvalue = -1;
	}

	private OpenACCSpecifier(int cvalue)
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
