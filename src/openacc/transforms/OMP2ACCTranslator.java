/**
 * 
 */
package openacc.transforms;

import cetus.analysis.Reduction;
import cetus.hir.*;
import cetus.transforms.TransformPass;
import openacc.analysis.ACCAnalysis;
import openacc.analysis.ACCParser;
import openacc.analysis.AnalysisTools;
import openacc.analysis.ParserTools;
import openacc.analysis.SubArray;
import openacc.hir.ACCAnnotation;
import openacc.hir.ARCAnnotation;
import openacc.hir.ASPENAnnotation;
import openacc.hir.NVLAnnotation;

import java.util.*;

/**
 * @author Putt Sakdhnagool <psakdhna@purdue.edu>
 *         Seyong Lee <lees2@ornl.gov>
 *
 */
public class OMP2ACCTranslator extends TransformPass {
    private enum OmpRegion
    {
        Target,
        Target_Data,
        Target_Enter_Data,
        Target_Exit_Data,
        Target_Update,
        Teams,
        Distribute,
        Parallel_For,
        Parallel,
        SIMD
    }
    
    private enum taskType
    {
    	innerTask,
    	outerTask
    }

    protected String pass_name = "[OMP2ACCTranslator]";
    protected Program program;
    //Main refers either a procedure containing acc_init() call or main() procedure if no explicit acc_init() call exists.
    protected Procedure main;

    //Target information
    private Expression deviceExpr = null;

    private Stack<Traversable> parentStack = new Stack<Traversable>();
    private Stack<OmpRegion> regionStack = new Stack<OmpRegion>();
    private List<OmpAnnotation> removeList = new LinkedList<OmpAnnotation>();
    private HashMap<String, String> macroMap = null;
    private List<Declaration> procDeclList = null;
	private	List<FunctionCall> funcCallList = null;
	private int defaultNumAsyncQueues = 4;

    public OMP2ACCTranslator(Program prog, int numAsyncQueues) {
        super(prog);
        program = prog;
        defaultNumAsyncQueues = numAsyncQueues;
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
