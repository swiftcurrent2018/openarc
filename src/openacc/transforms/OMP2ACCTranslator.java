/**
 * 
 */
package openacc.transforms;

import cetus.analysis.Reduction;
import cetus.hir.*;
import cetus.transforms.TransformPass;
import openacc.analysis.ParserTools;
import openacc.analysis.SubArray;
import openacc.hir.ACCAnnotation;

import java.util.*;

/**
 * @author Putt Sakdhnagool <psakdhna@purdue.edu>
 *
 */
public class OMP2ACCTranslator extends TransformPass {
    private enum OmpRegion
    {
        Target,
        Teams,
        Distribute,
        Parallel_For,
        Parallel,
        SIMD
    }

    protected String pass_name = "[OMP2ACCTranslator]";
    protected Program program;
    //Main refers either a procedure containing acc_init() call or main() procedure if no explicit acc_init() call exists.
    protected Procedure main;

    //Target information
    private boolean targetRegion = false;
    private Expression deviceExpr = null;

    private Stack<Traversable> parentStack = new Stack<Traversable>();
    private Stack<OmpRegion> regionStack = new Stack<OmpRegion>();

    public OMP2ACCTranslator(Program prog) {
        super(prog);
        program = prog;
    }

    @Override
    public String getPassName() {
        return pass_name;
    }

    @Override
    public void start() 
    {
		System.err.println(pass_name + " is under construction; exit");
		System.exit(1);	
    }

}
