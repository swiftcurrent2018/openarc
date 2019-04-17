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
import openacc.hir.ReductionOperator;

import java.util.*;

/**
 * @author Jacob Lambert <jlambert@cs.uoregon.edu>
 *         Seyong Lee <lees2@ornl.gov>
 *
 */
public class ACCtoOMP3Translator extends TransformPass {
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

  private enum AccRegion
  {
    Data,
    Loop,
    Parallel,
    Update,
    Enter_Data,
    Exit_Data,
    Parallel_Loop
  }

  private enum taskType
  {
    innerTask,
    outerTask
  }

  static private boolean DEBUG = true;

  protected String pass_name = "[ACCtoOMP3Translator]";
  protected Program program;

  static private List<ACCAnnotation> removeList = new LinkedList<ACCAnnotation>();
  static private List<FunctionCall> funcCallList = null;
  private int defaultNumAsyncQueues = 4;

  public ACCtoOMP3Translator(Program prog, int numAsyncQueues) {
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
