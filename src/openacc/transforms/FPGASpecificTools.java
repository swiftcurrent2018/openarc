package openacc.transforms;

import cetus.analysis.LoopTools;
import cetus.analysis.LoopInfo;
import cetus.hir.*;
import openacc.hir.*;
import openacc.analysis.AnalysisTools;
import openacc.analysis.SubArray;
import openacc.hir.ACCAnnotation;
import openacc.hir.ARCAnnotation;
import openacc.hir.OpenCLSpecifier;
import openacc.hir.ReductionOperator;

import java.util.*;

public abstract class FPGASpecificTools {
  private static int tempIndexBase = 1000;

  private FPGASpecificTools()
  {
  }

  protected static Statement reductionTransformation(Procedure cProc, Statement region, String cRegionKind,
      FunctionCall call_to_new_proc, Procedure new_proc, boolean IRSymbolOnly, boolean isSingleTask) {
    Tools.exit("[ERROR in FPGASpecificTools.reductionTransformation()] Not implemented yet!\n");
    return null;
  }

  protected static void collapseTransformation(ForLoop accLoop, CompoundStatement cStmt, ArrayList<Symbol> indexSymbols, 
      ArrayList<Expression> iterspaceList, int collapseLevel) { 
    Tools.exit("[ERROR in FPGASpecificTools.collapseTransformation()] Not implemented yet!\n");
  }

  protected static boolean isFPGASingleWorkItemRegion(Statement region) {
      return false;
  }

  protected static Statement UnrollLoopsInRegion(Statement region) {
    Tools.exit("[ERROR in FPGASpecificTools.UnrollLoopsInRegion()] Not implemented yet!\n");
    return null;
  }


}
