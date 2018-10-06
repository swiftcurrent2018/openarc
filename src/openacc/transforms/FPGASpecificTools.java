package openacc.transforms;

import cetus.analysis.LoopTools;
import cetus.analysis.LoopInfo;
import cetus.exec.Driver;
import cetus.hir.*;
import openacc.hir.*;
import openacc.analysis.AnalysisTools;
import openacc.analysis.SubArray;
import openacc.hir.ACCAnnotation;
import openacc.hir.ARCAnnotation;
import openacc.hir.OpenCLSpecifier;
import openacc.hir.ReductionOperator;

import java.util.*;

/**
 * <b>FPGASpecificTools</b> provides tools for various transformation tools for FPGA-specific translation.
 * 
 * @author Jacob Lambert <uql@ornl.gov>
 *         Future Technologies Group, Computer Science and Mathematics Division,
 *         Oak Ridge National Laboratory
 */
public abstract class FPGASpecificTools {
  private static int tempIndexBase = 1000;

  /**
   * Java doesn't allow a class to be both abstract and final,
   * so this private constructor prevents any derivations.
   */
  private FPGASpecificTools()
  {
  }

  public static class OperatorPair {
    public Specifier spec;
    public ReductionOperator operator;

    public OperatorPair(ReductionOperator op, Specifier sp) {
      operator = op;
      spec = sp;
    }

    @Override
      public String toString() {
        return "(" + spec.toString() + ", " + operator.toString() + ")";
      }

    @Override
      public boolean equals(Object obj) {

        if(obj==null || !(obj instanceof OperatorPair))
          return false;

        OperatorPair op_pair = (OperatorPair) obj;

        return op_pair.operator.equals(operator) && op_pair.spec.equals(spec);
      }

    @Override
      public int hashCode() {
        return operator.hashCode() + spec.hashCode();
      }


  }

  /**
   * This method performs FPGA-specific reduction transformation.
   * This transformation is intraprocedural; functions called in a compute region should be handled
   * separately.
   *
   * CAUTION: 1) This function assumes that the applied region is a loop contained within a single work-item region,
   * with an FPGA device as the target accelerator. 
   *
   * @param region
   */
  protected static Statement reductionTransformation(Procedure cProc, Statement region, String cRegionKind,
      FunctionCall call_to_new_proc, Procedure new_proc, boolean IRSymbolOnly, boolean isSingleTask) {

    PrintTools.println("[OpenCL reductionTransformation() begins] current procedure: " + cProc.getSymbolName() +
        "\ncompute region type: " + cRegionKind + "\n", 2);

    /* Default operator costs */
    HashMap<OperatorPair, Integer> operatorCostMap = new HashMap<OperatorPair, Integer>();

    boolean STRATIX_V = false;
    boolean ARRIA_X = false;

    String fpga_env = System.getenv("OPENARC_FPGA");
    if (fpga_env == null)
      Tools.exit("[Error in FPGASpecificTools.reductionTransformation] Environment variable OPENARC_FPGA not set. Options include:\n" + 
          "  STRATIX_V\n  ARRIA_X");
    else if (fpga_env.equals("STRATIX_V")) 
      STRATIX_V = true;
    else if (fpga_env.equals("ARRIA_X")) 
      ARRIA_X = true;
    else
      Tools.exit("[Error in FPGASpecificTools.reductionTransformation] Environment variable OPENARC_FPGA='" +
          fpga_env.toString() + "' not supported. Options include:\n  STRATIX_V\n  ARRIA_X");

    if (ARRIA_X) {
      operatorCostMap.put(new OperatorPair(ReductionOperator.ADD, Specifier.INT), 1);
      operatorCostMap.put(new OperatorPair(ReductionOperator.MULTIPLY, Specifier.INT), 1);
      operatorCostMap.put(new OperatorPair(ReductionOperator.MAX, Specifier.INT), 1);
      operatorCostMap.put(new OperatorPair(ReductionOperator.MIN, Specifier.INT), 1);

      operatorCostMap.put(new OperatorPair(ReductionOperator.ADD, Specifier.FLOAT), 4); // change to 1 for accumulator
      operatorCostMap.put(new OperatorPair(ReductionOperator.MULTIPLY, Specifier.FLOAT), 4);
      operatorCostMap.put(new OperatorPair(ReductionOperator.MAX, Specifier.FLOAT), 1);
      operatorCostMap.put(new OperatorPair(ReductionOperator.MIN, Specifier.FLOAT), 1);

      operatorCostMap.put(new OperatorPair(ReductionOperator.ADD, Specifier.DOUBLE), 12);
      operatorCostMap.put(new OperatorPair(ReductionOperator.MULTIPLY, Specifier.DOUBLE), 16);
      operatorCostMap.put(new OperatorPair(ReductionOperator.MAX, Specifier.DOUBLE), 1);
      operatorCostMap.put(new OperatorPair(ReductionOperator.MIN, Specifier.DOUBLE), 1);
    }

    if (STRATIX_V) {
      operatorCostMap.put(new OperatorPair(ReductionOperator.ADD, Specifier.INT), 2);
      operatorCostMap.put(new OperatorPair(ReductionOperator.MULTIPLY, Specifier.INT), 2);
      operatorCostMap.put(new OperatorPair(ReductionOperator.MAX, Specifier.INT), 2);
      operatorCostMap.put(new OperatorPair(ReductionOperator.MIN, Specifier.INT), 2);

      operatorCostMap.put(new OperatorPair(ReductionOperator.ADD, Specifier.FLOAT), 8);
      operatorCostMap.put(new OperatorPair(ReductionOperator.MULTIPLY, Specifier.FLOAT), 6);
      operatorCostMap.put(new OperatorPair(ReductionOperator.MAX, Specifier.FLOAT), 3);
      operatorCostMap.put(new OperatorPair(ReductionOperator.MIN, Specifier.FLOAT), 3);

      operatorCostMap.put(new OperatorPair(ReductionOperator.ADD, Specifier.DOUBLE), 11);
      operatorCostMap.put(new OperatorPair(ReductionOperator.MULTIPLY, Specifier.DOUBLE), 16);
      operatorCostMap.put(new OperatorPair(ReductionOperator.MAX, Specifier.DOUBLE), 2);
      operatorCostMap.put(new OperatorPair(ReductionOperator.MIN, Specifier.DOUBLE), 2);
    }

    /* Collect reduction information */
    List<ACCAnnotation> reduction_annots = IRTools.collectPragmas(region, ACCAnnotation.class, "reduction");
    if( (reduction_annots == null) || reduction_annots.isEmpty() ) {
      PrintTools.println("[FPGA-specific reductionTransformation() ends] current procedure: " + cProc.getSymbolName() +
          "\ncompute region type: " + cRegionKind + "\n", 2);
      return region;
    }

    /* Return statement */
    Statement rStmt = region;

    //////////////////////////////////////////////////////////////////
    // Iterate over each sub-region annotated by a reduction clause //
    //////////////////////////////////////////////////////////////////
    // NOTE: This also handles the case where the entire region is annotated
    //       with a reduction clause (red_region == region)

    for ( ACCAnnotation pannot : reduction_annots) {

      // skip non-loops
      if (pannot.getAnnotatable().getClass() != ForLoop.class) continue;

      /* Host reduction symbol to reduction operator mapping */
      HashMap<Symbol, ReductionOperator> redOpMap = new HashMap<Symbol, ReductionOperator>();
      HashMap<Symbol, SubArray> redVarMap = new HashMap<Symbol, SubArray>();

      /* Set of allocated reduction symbols */
      Map<ReductionOperator, Set<SubArray>> redMap = pannot.get("reduction");

      for(ReductionOperator op : redMap.keySet() ) {
        Set<SubArray> redSet = redMap.get(op);
        for( SubArray sArray : redSet ) {
          Symbol rSym = AnalysisTools.subarrayToSymbol(sArray, IRSymbolOnly);
          redVarMap.put(rSym, sArray);
          redOpMap.put(rSym, op);
        }
      }

      /* Create enclosing compound statement */
      CompoundStatement cStmt = new CompoundStatement();

      Statement red_region = (Statement) pannot.getAnnotatable();
      ForLoop forLoop = (ForLoop) red_region.clone();
      cStmt.addStatement(forLoop);

      red_region.swapWith(cStmt);

      /* Record loop unroll factor */
      int unroll_factor = 1;

      List<PragmaAnnotation> pragmaList = forLoop.getAnnotations(PragmaAnnotation.class);
      for (PragmaAnnotation pragmaAnnot : pragmaList) {
        String keySet[] = pragmaAnnot.toString().split(" ");

        int index = 0;
        for( String tKey : keySet ) {
          // get unroll value
          if (tKey.equals("unroll"))  {
            if (index == keySet.length - 1)
              Tools.exit("[ERROR in ReductionTransformation: Annotation #pragma unroll must have" + 
                  " an associated unroll factor");
            unroll_factor = Integer.parseInt(keySet[index + 1]); 
          }
          index++;
        }
      }

      ////////////////////////////////////////////////////////////////////////////////
      // Perform shift-register reduction for each variable in the reduction clause //
      ////////////////////////////////////////////////////////////////////////////////
      for (Symbol rSym : redOpMap.keySet()) {

        /* Determine needed register depth */
        // DEBUG: reg_depth should have an option for user assignment
        List<Specifier> rSym_specs = rSym.getTypeSpecifiers(); 
        OperatorPair op_pair = new OperatorPair( redOpMap.get(rSym), rSym_specs.get(rSym_specs.size() - 1)); 
        Object obj = operatorCostMap.get(op_pair);
        if (obj == null)
          Tools.exit("[ERROR in ReductionTransformation] Reduction operator \"" + redOpMap.get(rSym) + "\"  on type \"" 
              + rSym_specs.get(rSym_specs.size() - 1) + "\" not supported for FPGA-specific reduction transformation");

        int op_cost = (int) obj; 
        if (op_cost < 2) continue;

        int reg_depth = op_cost * unroll_factor;

        /* Set up kernel arguments/parameters */
        String symNameBase = rSym.getSymbolName();
        String hostrName = "gpu__" + symNameBase;
        Symbol hostrSym = SymbolTools.getSymbolOfName(hostrName, cStmt);
        Identifier hostrVar = new Identifier(hostrSym);

        /* Create a new index variable for indexing the shift-register */
        Expression iter = LoopTools.getIndexVariable(forLoop);
        Symbol indexSym = SymbolTools.getSymbolOf(iter);
        List<Specifier> index_specs = new ArrayList<Specifier>();
        index_specs.addAll(indexSym.getTypeSpecifiers());
        symNameBase = indexSym.getSymbolName();
        Identifier srIndex = TransformTools.getTempScalar(forLoop, index_specs, "sr__" + symNameBase, 0);

        /* Define shift register array */
        List<Specifier> removeSpecs = new ArrayList<Specifier>();
        removeSpecs.add(Specifier.STATIC);
        removeSpecs.add(Specifier.CONST);
        removeSpecs.add(Specifier.EXTERN);
        List<Specifier> typeSpecs = new ArrayList<Specifier>();
        Symbol IRSym = rSym;
        if( rSym instanceof PseudoSymbol ) {
          IRSym = ((PseudoSymbol)rSym).getIRSymbol();
          Symbol tSym = rSym;
          while( tSym instanceof AccessSymbol ) {
            tSym = ((AccessSymbol)tSym).getMemberSymbol();
          }
          typeSpecs.addAll(((VariableDeclaration)tSym.getDeclaration()).getSpecifiers());
        } else {
          typeSpecs.addAll(((VariableDeclaration)IRSym.getDeclaration()).getSpecifiers());
        }
        typeSpecs.removeAll(removeSpecs);
        symNameBase = null;

        if( rSym instanceof AccessSymbol) {
          symNameBase = TransformTools.buildAccessSymbolName((AccessSymbol)rSym);
        } else {
          symNameBase = rSym.getSymbolName();
        }
        String shiftregSymName = "shift_reg__" + symNameBase;
        List tindices = new LinkedList();

        tindices.add(new IntegerLiteral(reg_depth + 1));
        ArraySpecifier aspec = new ArraySpecifier(tindices);
        List tailSpecs = new ArrayList(1);
        tailSpecs.add(aspec);
        VariableDeclarator srSym_declarator =
          new VariableDeclarator(new NameID(shiftregSymName), tailSpecs);
        Identifier shift_reg = new Identifier(srSym_declarator);
        VariableDeclaration srSym_decl = new VariableDeclaration(typeSpecs,
            srSym_declarator);

        cStmt.addDeclaration(srSym_decl);

        /* Initialize shift register array */
        // #pragma unroll
        PragmaAnnotation unrollpragma = new PragmaAnnotation("unroll");
        // sr = 0
        Expression expr1 = new AssignmentExpression(srIndex.clone(), AssignmentOperator.NORMAL, new IntegerLiteral(0));
        Statement expStmt = new ExpressionStatement(expr1.clone());
        // sr < reg_depth
        expr1 = new BinaryExpression(srIndex.clone(), BinaryOperator.COMPARE_LT, new IntegerLiteral(reg_depth + 1));
        // sr++
        Expression expr2 = new UnaryExpression(UnaryOperator.POST_INCREMENT, srIndex.clone());

        CompoundStatement forBody = new CompoundStatement();

        Expression init_value = TransformTools.getRInitValue( redOpMap.get(rSym), rSym.getTypeSpecifiers());

        Expression expr3 = new AssignmentExpression(new ArrayAccess(shift_reg.clone(), srIndex.clone()),
            AssignmentOperator.NORMAL, init_value.clone());
        forBody.addStatement(new ExpressionStatement(expr3));

        ForLoop srLoop = new ForLoop(expStmt, expr1, expr2, forBody);
        srLoop.annotate(unrollpragma.clone());

        Statement cStmt_ref = AnalysisTools.getFirstExecutableStatement(cStmt);
        cStmt.addStatementBefore(cStmt_ref, srLoop);

        /* Replace reduction variable with shift register access */
        List<Expression> redVarList = IRTools.findExpressions(cStmt, redVarMap.get(rSym).getArrayName());

        for (Expression oExp : redVarList) {
          Expression nExp = new ArrayAccess(shift_reg.clone(), new IntegerLiteral(reg_depth));

          oExp.swapWith(nExp);
        }

        /* Generate extra assignment statement to percolate values */
        // sr[REG_DEPTH] = sr[0];
        expr1 = new ArrayAccess(shift_reg.clone(), new IntegerLiteral(reg_depth));
        expr2 = new ArrayAccess(shift_reg.clone(), new IntegerLiteral(0));
        Statement stmt = new ExpressionStatement(new AssignmentExpression(expr1, AssignmentOperator.NORMAL, expr2));

        CompoundStatement forLoop_body = (CompoundStatement) forLoop.getBody();
        Statement forLoop_ref = AnalysisTools.getFirstExecutableStatement(forLoop_body);
        forLoop_body.addStatementBefore(forLoop_ref, stmt.clone());

        /* Shift the shift register */
        // sr = 0
        expr1 = new AssignmentExpression(srIndex.clone(), AssignmentOperator.NORMAL, new IntegerLiteral(0));
        expStmt = new ExpressionStatement(expr1.clone());
        // sr < reg_depth
        expr1 = new BinaryExpression(srIndex.clone(), BinaryOperator.COMPARE_LT, new IntegerLiteral(reg_depth));
        // sr++
        expr2 = new UnaryExpression(UnaryOperator.POST_INCREMENT, srIndex.clone());

        forBody = new CompoundStatement();
        // shift_reg[sr] = shift_reg[sr + 1];
        expr3 = new AssignmentExpression(
            new ArrayAccess(shift_reg.clone(), srIndex.clone()), AssignmentOperator.NORMAL,
            new ArrayAccess(shift_reg.clone(), new BinaryExpression(srIndex.clone(), BinaryOperator.ADD, new IntegerLiteral(1))));
        forBody.addStatement(new ExpressionStatement(expr3));

        srLoop = new ForLoop(expStmt, expr1, expr2, forBody);
        srLoop.annotate(unrollpragma.clone());
        forLoop_body.addStatement(srLoop);

        /* Accumulate partial values */
        // sr = 0
        expr1 = new AssignmentExpression(srIndex.clone(), AssignmentOperator.NORMAL, new IntegerLiteral(0));
        expStmt = new ExpressionStatement(expr1.clone());

        // sr < reg_depth
        expr1 = new BinaryExpression(srIndex.clone(), BinaryOperator.COMPARE_LT, new IntegerLiteral(reg_depth));

        // sr++
        expr2 = new UnaryExpression(UnaryOperator.POST_INCREMENT, srIndex.clone());

        // sum += shift_reg[sr]
        Expression exprLHS = redVarMap.get(rSym).getArrayName();

        expr3 = new AssignmentExpression(
            exprLHS,
            AssignmentOperator.ADD,
            new ArrayAccess(shift_reg.clone(), srIndex.clone()));

        forBody = new CompoundStatement();
        forBody.addStatement(new ExpressionStatement(expr3));

        srLoop = new ForLoop(expStmt, expr1, expr2, forBody);
        srLoop.annotate(unrollpragma.clone());

        cStmt.addStatement(srLoop);
      }

      /* Set return statement */
      // NOTE: If we've modified the original region and not a sub-region,
      //       we need to return the new modified region, and not the original
      if (red_region == region) {
        rStmt = cStmt;
      }
    }

    PrintTools.println("[FPGA-specific reductionTransformation() ends] current procedure: " + cProc.getSymbolName() +
        "\ncompute region type: " + cRegionKind + "\n", 2);

    return rStmt;
  }

  /**
   * This method performs FPGA-specific collapse variable incrementation.
   *
   * CAUTION: 1) This function assumes that the applied region is a loop contained within a single work-item region,
   * with an FPGA device as the target accelerator. 
   *
   * NOTE: 1) This function replaces the ForLoop (accLoop) with a CompoundStatement (collapseStmt) containing: 
   *            
   *              collapseStmt
   *                variable initializations
   *                accLoop
   *
   * @param region
   */
  protected static void collapseTransformation(ForLoop accLoop, CompoundStatement cStmt, ArrayList<Symbol> indexSymbols, 
      ArrayList<Expression> iterspaceList, ArrayList<Expression> lbList, int collapseLevel) { 
    PrintTools.println("[INFO] FPGASpecificTools.collapseTransformation() is called.",1);

    /* Verify loop meets restrictions for FPGA-specific transformation */
    /* Step size verification not currently needed, due to error checks in CompRegionConfAnalysis():
     *  "A worksharing loop with a stride > 1 is found; current implmentation can not handle the following loop" */
    //System.out.println("accLoop: " + accLoop.toString());
    //Expression step = accLoop.getStep();
    //Expression init = ((ExpressionStatement) accLoop.getInitialStatement()).getExpression();
    //System.out.println("step: " + step + ", init: " + init);
    //if (step instanceof BinaryExpression) {
    //  Expression rhs = ((AssignmentExpression) step).getRHS();
    //  if (!rhs.toString().equals("1"))
    //    Tools.exit("[ERROR in collapseTransformation]: In FPGA-specific collapse, step size must be 1");
    //} 
    //else if (step instanceof UnaryExpression) {
    //  UnaryOperator op =  ((UnaryExpression) step).getOperator();
    //  if (op != UnaryOperator.POST_INCREMENT)
    //    Tools.exit("[ERROR in collapseTransformation]: In FPGA-specific collapse, step size must be 1");
    //}
    
    Statement stmt;
    Expression expr1;

    /* Generate incrementation statements for each collapsed variable */
    for (int i = collapseLevel - 1; i > 0; i--) {
      Identifier tIndex = new Identifier(indexSymbols.get(i));
      Identifier nIndex = new Identifier(indexSymbols.get(i-1));

      // x = x + 1
      if (i == collapseLevel-1) { 
        expr1 = new BinaryExpression(tIndex.clone(), BinaryOperator.ADD, new IntegerLiteral(1));

        stmt = new ExpressionStatement(new AssignmentExpression(tIndex.clone(),
              AssignmentOperator.NORMAL, expr1));
        cStmt.addStatement(stmt);
      }

      // y = (x == X + init_value) ? y + 1 : y
      Expression expr_true = new BinaryExpression(nIndex.clone(), BinaryOperator.ADD,
          new IntegerLiteral(1));

      Expression expr_cond = new BinaryExpression(tIndex.clone(), BinaryOperator.COMPARE_EQ,
          new BinaryExpression(iterspaceList.get(i).clone(), BinaryOperator.ADD, lbList.get(i).clone() ));

      expr1 = new ConditionalExpression(expr_cond, expr_true, nIndex.clone());

      stmt = new ExpressionStatement(new AssignmentExpression(nIndex.clone(),
            AssignmentOperator.NORMAL, expr1));
      cStmt.addStatement(stmt);

      // x = (x == X + init_value) ? init_value : x; 
      expr_cond = new BinaryExpression(tIndex.clone(), BinaryOperator.COMPARE_EQ,
          new BinaryExpression(iterspaceList.get(i).clone(), BinaryOperator.ADD, lbList.get(i).clone() ));

      expr1 = new ConditionalExpression(expr_cond, lbList.get(i).clone(), tIndex.clone());

      stmt = new ExpressionStatement(new AssignmentExpression(tIndex.clone(),
            AssignmentOperator.NORMAL, expr1));
      cStmt.addStatement(stmt);
    }

    /* Add initializations before loop */
    CompoundStatement collapseStmt = new CompoundStatement();

    for (int i = 0; i < collapseLevel; ++i) {
      Identifier tIndex = new Identifier(indexSymbols.get(i));

      expr1 = new AssignmentExpression(tIndex.clone(), AssignmentOperator.NORMAL,
          lbList.get(i).clone());
      ExpressionStatement expStmt = new ExpressionStatement(expr1);
      collapseStmt.addStatement(expStmt);
    }

    /* Move relevant annotations from loop (accLoop) to the enclosing region (collapseStmt) or enclosing parallel region*/
    ACCAnnotation move_annot = new ACCAnnotation();
    ACCAnnotation keep_annot = new ACCAnnotation();

    Annotatable target_region;
    // Parallel-loop (move clauses to collapseStmt)
    if (accLoop.getAnnotation(ACCAnnotation.class, "parallel") != null ||
        accLoop.getAnnotation(ACCAnnotation.class, "kernels") != null) {
      target_region = collapseStmt;
      move_annot.put("parallel", "_directive");
    }
    // Nested-loop (move clauses to enclosing parallel region)
    else {
      target_region = (Annotatable) accLoop.getParent();
      while (target_region.getAnnotation(ACCAnnotation.class, "parallel") == null && 
            target_region.getAnnotation(ACCAnnotation.class, "kernels") == null) {
        target_region = (Annotatable) target_region.getParent();
      }
    }

    List<ACCAnnotation> ACCList = accLoop.getAnnotations(ACCAnnotation.class);
    accLoop.removeAnnotations(ACCAnnotation.class);

    for (ACCAnnotation tOldAnnot : ACCList) {
      Set<String> keySet = tOldAnnot.keySet();

      for (String tKey : keySet) {
        // move all data clauses up
        if (ACCAnnotation.dataClauses.contains(tKey) ) {
          move_annot.put(tKey, tOldAnnot.get(tKey));
        } 
        // move all parallel clauses up
        else if (tKey.equals("parallel") ||
            tKey.equals("firstprivate") || 
            tKey.equals("private") ||
            tKey.equals("num_gangs") || 
            tKey.equals("num_workers") || 
            tKey.equals("async")||
            tKey.equals("wait") || 
            tKey.equals("if")) {
          move_annot.put(tKey, tOldAnnot.get(tKey));
        } 
        // move and keep reduction clauses
        else if (tKey.equals("reduction")) {
          move_annot.put("reduction", tOldAnnot.get(tKey));
          keep_annot.put("reduction", tOldAnnot.get(tKey));
        } 
        // keep loop clauses 
        else if (tKey.equals("loop") || 
            tKey.equals("gang") || 
            tKey.equals("worker") ||
            tKey.equals("internal")){
          keep_annot.put(tKey, tOldAnnot.get(tKey));
        }
        // ignore unneeded clauses
        else if (tKey.equals("pragma") || 
            tKey.equals("collapse") ||
            tKey.equals("iterspace") ||
            tKey.equals("gangdim") ||
            tKey.equals("workerdim") ||
            tKey.equals("accreduction") ||
            tKey.equals("accshared") ||
            tKey.equals("accexplicitshared") ||
            tKey.equals("accreadonly") ||
            tKey.equals("accprivate")) {

        }
        else {
          System.out.println("[Warning in FPGASpecificTools.collapseTransformation] Unhandled clause in ACC annotation:\n" + 
              "  Annotation: " + tOldAnnot.toString() + "\n  Clause: " + tKey.toString());
        }

      }

    }

    //keep_annot.put("gang", "_directive");

    accLoop.annotate(keep_annot);
    target_region.annotate(move_annot);

    // Move OpenARC annotations
    List<ARCAnnotation> ARCList = accLoop.getAnnotations(ARCAnnotation.class);
    accLoop.removeAnnotations(ARCAnnotation.class);
    for (ARCAnnotation tAnnot : ARCList) 
      target_region.annotate(tAnnot);

    // Move Cetus annotations
    List<CetusAnnotation> CetusList = accLoop.getAnnotations(CetusAnnotation.class);
    accLoop.removeAnnotations(CetusAnnotation.class);
    for (CetusAnnotation tAnnot : CetusList) 
      target_region.annotate(tAnnot);

    /* Insert the collapse logic into the IR tree */
    accLoop.getBody().swapWith(cStmt);
    collapseStmt.addStatement(accLoop.clone());
    accLoop.swapWith(collapseStmt);

    List<ForLoop> fLoops = IRTools.getStatementsOfType(collapseStmt, ForLoop.class); 
    accLoop = fLoops.get(0);
  }

  /**
   * This method determines if a region is executed in both an FPGA and a single work-item context 
   *
   * @param region
   */
  protected static boolean isFPGASingleWorkItemRegion(Statement region) {
    /* Determine if single work-item */
    int isSingleWorkItem = 0;
    ACCAnnotation gAnnot, wAnnot;
    if (region == null) {
      Tools.exit("ERROR in isFPGASingleWorkItemRegion(...): region is null");
    }

    if( !region.containsAnnotation(ACCAnnotation.class, "kernels") &&
        !region.containsAnnotation(ACCAnnotation.class, "parallel")) {
      Tools.exit("ERROR in isFPGASingleWorkItemRegion(...): region does not contain" + 
          "kernels or parallel clause");
    }

    Expression totalnumgangs = null; 
    Expression totalnumworkers = null;

    if (region.containsAnnotation(ACCAnnotation.class, "parallel")) {
      gAnnot = region.getAnnotation(ACCAnnotation.class, "num_gangs");
      wAnnot = region.getAnnotation(ACCAnnotation.class, "num_workers");

      if( gAnnot == null )
        Tools.exit("ERROR in isFPGASingleWorkItemRegion(...): num_gangs null\n");
      if( wAnnot == null )
        Tools.exit("ERROR in isFPGASingleWorkItemRegion(...): num_workers null\n");
      totalnumgangs = ((Expression)gAnnot.get("num_gangs")).clone();
      totalnumworkers = ((Expression)wAnnot.get("num_workers")).clone();
    }

    if (region.containsAnnotation(ACCAnnotation.class, "kernels")) {
      gAnnot = region.getAnnotation(ACCAnnotation.class, "gang");
      wAnnot = region.getAnnotation(ACCAnnotation.class, "worker");

      if( gAnnot == null && wAnnot == null) {
        totalnumgangs = ((Expression) new IntegerLiteral(1));
        totalnumworkers = ((Expression) new IntegerLiteral(1));
      } else if( gAnnot == null ) {
        //[FIXME] kernels region does not have to have both gang and worker clauses; temporarily disabled.
        //Tools.exit("ERROR in isFPGASingleWorkItemRegion(...): gang null\n");
      } else if( wAnnot == null ) {
        //[FIXME] kernels region does not have to have both gang and worker clauses; temporarily disabled.
        //Tools.exit("ERROR in isFPGASingleWorkItemRegion(...): worker null\n");
      } else {
        totalnumgangs = ((Expression)gAnnot.get("gang")).clone();
        totalnumworkers = ((Expression)wAnnot.get("worker")).clone();
      }
    }

    if( (totalnumgangs == null) || (totalnumworkers == null) ) {
      isSingleWorkItem = 0;
    } else if( totalnumgangs.toString().equals("1") && totalnumworkers.toString().equals("1") )
      isSingleWorkItem = 1;
    else
      isSingleWorkItem = 0;

    /* Get target architecture */
    int targetArch;
    String value = null;
    value = Driver.getOptionValue("targetArch");
    if( value != null ) {
      targetArch = Integer.valueOf(value).intValue();
    } else {
      targetArch = Integer.valueOf(System.getenv("OPENARC_ARCH") ).intValue();
    }

    if (isSingleWorkItem == 1 && targetArch == 3)
      return true;
    else
      return false;
  }

  /**
   * This method unrolls all loops in a given region.
   *  
   *  If a given loop does not contain an unroll pragma, or if the loop has
   *  an unroll pragma with an unroll factor, these are substituted with a
   *  standard unroll pragma, and a warning is given.
   *
   * @param region
   */

  protected static Statement UnrollLoopsInRegion(Statement region) {

    PragmaAnnotation unrollPragma = new PragmaAnnotation("unroll");

    /* Annotate loops without an unroll, or with a partial unroll */
    DFIterator<ForLoop> it = new DFIterator<ForLoop>(region, ForLoop.class);
    while (it.hasNext()) {
      ForLoop at = it.next();

      ACCAnnotation loopAnnots = at.getAnnotation(ACCAnnotation.class, "loop");
      ACCAnnotation seqAnnots = at.getAnnotation(ACCAnnotation.class, "seq");

      if (loopAnnots != null || seqAnnots != null) 
        Tools.exit("[ERROR in SlidingWindowTransformation]: Nested loops cannot contain ACC annotations");

      List<PragmaAnnotation> pragmaList = at.getAnnotations(PragmaAnnotation.class);
      boolean unroll_bool = false;
      for (PragmaAnnotation pragmaAnnot : pragmaList) {
        String keySet[] = pragmaAnnot.toString().split(" ");
        int index = 0;
        for( String tKey : keySet ) {
          // get unroll value
          if (tKey.equals("unroll"))  {
            unroll_bool = true;
            if (index != keySet.length - 1) {
              System.out.println("[WARNING in SlidingWindowTransformation: " + 
                  "Nested loops must be fully unrolled. Partial unroll factor removed: " + pragmaAnnot.toString());
              at.removeAnnotations(PragmaAnnotation.class);
              at.annotate(unrollPragma);
            }
          }       
          index++;
        }
      }

      // no unroll pragma found
      if (unroll_bool == false) {
        at.annotate(unrollPragma);
        System.out.println("[WARNING in SlidingWindowTransformation: " + 
            "Nested loops must be fully unrolled. Unroll pragma added: " + at.toString());
      }
    }

    /* Perform Loop Unrollling */
    ArrayList<ForLoop> forLoops = new ArrayList<ForLoop>();
    forLoops = (ArrayList)IRTools.getStatementsOfType(region, ForLoop.class); 
    int num_loops = forLoops.size();

    // It make take several iterations, as new loops are created when unrolling loops with nested loops
    while (num_loops > 0) {

      for( ForLoop f : forLoops) {
        ForLoop newFor = f.clone();

        LoopInfo forLoopInfo = new LoopInfo(f);
        Expression index = forLoopInfo.getLoopIndex();
        Expression lb = forLoopInfo.getLoopLB();
        Expression ub = forLoopInfo.getLoopUB();
        Statement loopBody = f.getBody().clone();

        //Perform unrolling only if the increment is 1
        if (!forLoopInfo.getLoopIncrement().toString().equals("1")) {
          Tools.exit("[ERROR in SlidingWindowTransformation]: Nested loops must have a step size of 1 (to allow unrolling): "  +
              forLoopInfo.toString());
        }

        Expression itrSize = Symbolic.simplify(Symbolic.add(Symbolic.subtract(ub,lb),new IntegerLiteral(1)));

        int ub_int = (int) ( (IntegerLiteral) ub).getValue();
        int lb_int = (int) ( (IntegerLiteral) lb).getValue();

        CompoundStatement parent = (CompoundStatement) f.getParent();
        for (int i = lb_int; i <= ub_int; ++i) {
          Statement newBody = loopBody.clone();
          IRTools.replaceAll(newBody,index, new IntegerLiteral(i));
          parent.addStatementAfter(f, newBody);
        }

        parent.removeStatement(f);
      }

      forLoops = (ArrayList)IRTools.getStatementsOfType(region, ForLoop.class); 
      num_loops = forLoops.size();
    }

    return region;
  }


}
