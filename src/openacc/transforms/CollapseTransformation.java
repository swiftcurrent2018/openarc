package openacc.transforms;

import java.util.*;

import cetus.hir.*;
import cetus.analysis.LoopTools;
import cetus.transforms.LoopNormalization;
import cetus.transforms.TransformPass;
import openacc.hir.*;
import openacc.analysis.AnalysisTools;
import openacc.analysis.SubArray;

/**
 * <b>CollapseTransformation</b> performs source-level transformation to collapse
 * the iterations of all associated loops into one larger iteration space.
 * 
 * @author Seyong Lee <lees2@ornl.gov> and Jacob Lambert <jlambert@cs.uoregon.edu>
 *         Future Technologies Group, Oak Ridge National Laboratory
 */
public class CollapseTransformation extends TransformPass {
  private boolean simplifyExp = true;
  /**
   * @param program
   */
  public CollapseTransformation(Program program, boolean simplify) {
    super(program);
    verbosity = 1;
    simplifyExp = simplify;
  }

  /* (non-Javadoc)
   * @see cetus.transforms.TransformPass#getPassName()
   */
  @Override
    public String getPassName() {
      return new String("[collapse transformation]");
    }

  /* (non-Javadoc)
   * @see cetus.transforms.TransformPass#start()
   */
  @Override
    public void start() {
      List<ForLoop> outer_loops = new ArrayList<ForLoop>();
      // Find loops containing OpenACC collapse clauses with parameter value > 1.
      List<ACCAnnotation> collapseAnnotList = IRTools.collectPragmas(program, ACCAnnotation.class, "collapse");
      if( collapseAnnotList == null ) return;
      for( ACCAnnotation cAnnot : collapseAnnotList ) {
        Annotatable at = cAnnot.getAnnotatable();
        if( at instanceof ForLoop ) {
          Object cObj = cAnnot.get("collapse");
          if( !(cObj instanceof IntegerLiteral) ) {
            Tools.exit("[ERROR] the argument of OpenACC collapse clause must be a constant positive integer expression"
                + ", but the following loop construct has an incompatible argument:\n" + at.toString() + "\n");
          }
          int collapseLevel = (int)((IntegerLiteral)cObj).getValue();
          if( collapseLevel > 1 ) {
            outer_loops.add((ForLoop)at);
          }
        }
      }
      if( outer_loops.isEmpty() ) {
        return;
      } else {
        PrintTools.println("[INFO] Found OpenACC-collapse clauses that associate more than one loop.",1);
        int collapsedLoops = 0;
        for( ForLoop accLoop : outer_loops ) {
          collapsedLoops +=collapseLoop(accLoop, simplifyExp);
        }
        PrintTools.println("[INFO] Number of collapsed OpenACC loops: " + collapsedLoops, 0);
      }
    }
  
  // Collapse Header Only 
  public static int collapseLoopHeader(ForLoop accLoop, boolean simplifyE) {
    int collapsedLoops = 0;

    Traversable t = (Traversable)accLoop;
    while(t != null) {
      if (t instanceof Procedure) break;
      t = t.getParent(); 
    }

    if( t == null ) {
      Tools.exit("[ERROR in CollapseTransformatin.collapseLoop()] Cannot find an enclosing procedure for the following loop:\n"
          + accLoop + "\n");
    }

    Procedure proc = (Procedure)t;
    Statement enclosingCompRegion = null;
    if( !accLoop.containsAnnotation(ACCAnnotation.class, "kernels") && 
        !accLoop.containsAnnotation(ACCAnnotation.class, "parallel")) {
      Annotatable att = (Annotatable)accLoop.getParent();
      while( att != null ) {
        if( att.containsAnnotation(ACCAnnotation.class, "kernels" ) ||
            att.containsAnnotation(ACCAnnotation.class, "parallel")) {
          enclosingCompRegion = (Statement)att;
          break;
        } else {
          if( att.getParent() instanceof Annotatable ) {
            att = (Annotatable)att.getParent();
          } else {
            break;
          }
        }
      }
    }

    ArrayList<Symbol> indexSymbols = new ArrayList<Symbol>();
    ArrayList<ForLoop> indexedLoops = new ArrayList<ForLoop>();
    indexedLoops.add(accLoop);
    ACCAnnotation collapseAnnot = accLoop.getAnnotation(ACCAnnotation.class, "collapse");
    OmpAnnotation ompAnnot = accLoop.getAnnotation(OmpAnnotation.class, "for");
    int collapseLevel = (int)((IntegerLiteral)collapseAnnot.get("collapse")).getValue();
    boolean pnest = true;
    pnest = AnalysisTools.extendedPerfectlyNestedLoopChecking(accLoop, collapseLevel, indexedLoops, null);
    if( !pnest ) {
      Tools.exit("[ERROR] OpenACC collapse clause is applicable only to perfectly nested loops;\n"
          + "Procedure name: " + proc.getSymbolName() + "\nTarget loop: \n" +
          accLoop.toString() + "\n");
    }
    if( indexedLoops.size() < collapseLevel ) {
      PrintTools.println("\n[WARNING] Number of found loops (" + indexedLoops.size() + 
          ") is smaller then collapse parameter (" + collapseLevel + "); skip the loop in procedure " 
          + proc.getSymbolName() + ".\n",0);
      PrintTools.println("OpenACC loop\n" + accLoop + "\n", 2);
      return collapsedLoops;
    }
    for( ForLoop currLoop : indexedLoops ) {
      indexSymbols.add(LoopTools.getLoopIndexSymbol(currLoop));
    }
    collapsedLoops++;

    ForLoop innerLoop = indexedLoops.get(collapseLevel-1);
    Statement fBody = innerLoop.getBody();
    CompoundStatement cStmt = null;
    Statement tRefStmt = null;
    if( fBody instanceof CompoundStatement ) {
      cStmt = (CompoundStatement)fBody;
      tRefStmt = IRTools.getFirstNonDeclarationStatement(fBody);
    } else {
      cStmt = new CompoundStatement();
      cStmt.addStatement(fBody);
      tRefStmt = fBody;
    }
    //System.out.println("Loops to collapse: \n" + accLoop + "\n");
    //System.out.println("Innermost loop body: \n" + cStmt + "\n");
    //If const symbol declaration exists, old index variables may be used in its initialization.
    //In this case, expression statements assigning the old index variables should be put before
    //the const symbol declaration.
    //For simplicity, the new assignment expression statements added at the beginning whenever
    //const symbol declaration exists.
    boolean containsConstSymbol = false;
    Set<Symbol> lSymbolSet = cStmt.getSymbols();
    for(Symbol lSym : lSymbolSet) {
      if( SymbolTools.containsSpecifier(lSym, Specifier.CONST) ) {
        containsConstSymbol = true;
        break;
      }
    }
    if( containsConstSymbol ) {
      tRefStmt = (Statement)cStmt.getChildren().get(0);
    }
    if( tRefStmt == null ) {
      Tools.exit("[ERROR in CollapseTransformation] can not find referent statement "
          + "to insert old index calculation statements; exit!" + AnalysisTools.getEnclosingContext(accLoop));
    }
    ArrayList<Expression> iterspaceList = new ArrayList<Expression>();
    ArrayList<Expression> lbList = new ArrayList<Expression>();
    Expression collapsedIterSpace = null;
    if( simplifyE ) {
      for( int i=0; i<collapseLevel; i++ ) {
        ForLoop loop = indexedLoops.get(i);
        Expression lb = LoopTools.getLowerBoundExpression(loop);
        lbList.add(i, lb);
        Expression ub = LoopTools.getUpperBoundExpression(loop);
        Expression itrSpace = Symbolic.add(Symbolic.subtract(ub,lb),new IntegerLiteral(1));
        iterspaceList.add(i, itrSpace);
        if( i==0 ) {
          collapsedIterSpace = itrSpace;
        } else {
          collapsedIterSpace = Symbolic.multiply(collapsedIterSpace, itrSpace);
        }
      }
    } else {
      for( int i=0; i<collapseLevel; i++ ) {
        ForLoop loop = indexedLoops.get(i);
        Expression lb = LoopTools.getLowerBoundExpressionNS(loop);
        lbList.add(i, lb);
        Expression ub = LoopTools.getUpperBoundExpressionNS(loop);
        Expression itrSpace = Symbolic.add(Symbolic.subtract(ub,lb),new IntegerLiteral(1));
        iterspaceList.add(i, itrSpace);
        if( i==0 ) {
          collapsedIterSpace = itrSpace;
        } else {
          collapsedIterSpace = Symbolic.multiply(collapsedIterSpace, itrSpace);
        }
      }
    }
    //Create a new index variable for the newly collapsed loop.
    CompoundStatement procBody = proc.getBody();

    Identifier newIndex = null;
    if( enclosingCompRegion instanceof CompoundStatement ) {
      newIndex = TransformTools.getNewTempIndex(enclosingCompRegion);
    } else if( enclosingCompRegion instanceof ForLoop ) {
      newIndex = TransformTools.getNewTempIndex(((ForLoop)enclosingCompRegion).getBody());
    } else {
      newIndex = TransformTools.getNewTempIndex(procBody);

      //If the current accLoop is in a compute region, the above new private variable should be
      //added to the private clause.
      Set<SubArray> privateSet = null;
      if( collapseAnnot.containsKey("private") ) {
        privateSet = collapseAnnot.get("private");
      } else {
        privateSet = new HashSet<SubArray>();
        collapseAnnot.put("private", privateSet);
      }
      privateSet.add(AnalysisTools.createSubArray(newIndex.getSymbol(), true, null));
      if( ompAnnot != null ) {
        Set<String> ompPrivSet = ompAnnot.get("private");
        if( ompPrivSet == null ) {
          ompPrivSet = new HashSet<String>();
          ompAnnot.put("private", ompPrivSet);
        }
        ompPrivSet.add(newIndex.toString());
      }
    }
    if( !collapseAnnot.containsKey("gang") && !collapseAnnot.containsKey("worker") &&
        !collapseAnnot.containsKey("vector") && !collapseAnnot.containsKey("seq") ) {
      collapseAnnot.put("seq", "true");
    }

    /////////////////////////////////////////////////////////////////////////////////
    //Swap initialization statement, condition, and step of the OpenACC loop with  //
    //those of the new, collapsed loop.                                            //
    /////////////////////////////////////////////////////////////////////////////////

    Expression expr1 = new AssignmentExpression(newIndex.clone(), AssignmentOperator.NORMAL,
        new IntegerLiteral(0));
    Statement initStmt = new ExpressionStatement(expr1);
    accLoop.getInitialStatement().swapWith(initStmt);
    expr1 = new BinaryExpression((Identifier)newIndex.clone(), BinaryOperator.COMPARE_LT,
        collapsedIterSpace);
    accLoop.getCondition().swapWith(expr1);
    expr1 = new UnaryExpression(
        UnaryOperator.POST_INCREMENT, (Identifier)newIndex.clone());
    accLoop.getStep().swapWith(expr1);

    /////////////////////////////////////////////////////////////////////////
    //Swap the body of the OpenACC loop with the one of the innermost loop //
    //among associated loops.                                              //
    /////////////////////////////////////////////////////////////////////////
    accLoop.getBody().swapWith(cStmt);

    return collapsedLoops;
  }

  // Collapse Header and Collapse Incrementation
  public static int collapseLoop(ForLoop accLoop, boolean simplifyE) {
    int collapsedLoops = 0;

    Traversable t = (Traversable)accLoop;
    while(t != null) {
      if (t instanceof Procedure) break;
      t = t.getParent(); 
    }

    if( t == null ) {
      Tools.exit("[ERROR in CollapseTransformatin.collapseLoop()] Cannot find an enclosing procedure for the following loop:\n"
          + accLoop + "\n");
    }

    Procedure proc = (Procedure)t;
    Statement enclosingCompRegion = null;
    if( !accLoop.containsAnnotation(ACCAnnotation.class, "kernels") && 
        !accLoop.containsAnnotation(ACCAnnotation.class, "parallel")) {
      Annotatable att = (Annotatable)accLoop.getParent();
      while( att != null ) {
        if( att.containsAnnotation(ACCAnnotation.class, "kernels" ) ||
            att.containsAnnotation(ACCAnnotation.class, "parallel")) {
          enclosingCompRegion = (Statement)att;
          break;
        } else {
          if( att.getParent() instanceof Annotatable ) {
            att = (Annotatable)att.getParent();
          } else {
            break;
          }
        }
      }
    }

    // Determine if single work-item
    int isSingleWorkItem = 0;
    ACCAnnotation gAnnot, wAnnot;
    if (enclosingCompRegion != null) {
      gAnnot = enclosingCompRegion.getAnnotation(ACCAnnotation.class, "num_gangs");
      wAnnot = enclosingCompRegion.getAnnotation(ACCAnnotation.class, "num_workers");
    }
    else {
      gAnnot = accLoop.getAnnotation(ACCAnnotation.class, "num_gangs");
      wAnnot = accLoop.getAnnotation(ACCAnnotation.class, "num_workers");
    }

    if( gAnnot == null ) {
      Tools.exit("num_gangs null\n");
    }
    if( wAnnot == null ) {
      Tools.exit("num_workers null\n");
    }

    Expression totalnumgangs = ((Expression)gAnnot.get("num_gangs")).clone();
    Expression totalnumworkers = ((Expression)wAnnot.get("num_workers")).clone();

    if( totalnumgangs.toString().equals("1") && totalnumworkers.toString().equals("1") ) {
      isSingleWorkItem = 1;
    }
    else {
      isSingleWorkItem = 0;
    }

    ArrayList<Symbol> indexSymbols = new ArrayList<Symbol>();
    ArrayList<ForLoop> indexedLoops = new ArrayList<ForLoop>();
    indexedLoops.add(accLoop);
    ACCAnnotation collapseAnnot = accLoop.getAnnotation(ACCAnnotation.class, "collapse");
    OmpAnnotation ompAnnot = accLoop.getAnnotation(OmpAnnotation.class, "for");
    int collapseLevel = (int)((IntegerLiteral)collapseAnnot.get("collapse")).getValue();
    boolean pnest = true;
    pnest = AnalysisTools.extendedPerfectlyNestedLoopChecking(accLoop, collapseLevel, indexedLoops, null);
    if( !pnest ) {
      Tools.exit("[ERROR] OpenACC collapse clause is applicable only to perfectly nested loops;\n"
          + "Procedure name: " + proc.getSymbolName() + "\nTarget loop: \n" +
          accLoop.toString() + "\n");
    }
    if( indexedLoops.size() < collapseLevel ) {
      PrintTools.println("\n[WARNING] Number of found loops (" + indexedLoops.size() + 
          ") is smaller then collapse parameter (" + collapseLevel + "); skip the loop in procedure " 
          + proc.getSymbolName() + ".\n",0);
      PrintTools.println("OpenACC loop\n" + accLoop + "\n", 2);
      return collapsedLoops;
    }
    for( ForLoop currLoop : indexedLoops ) {
      //LoopNormalization.normalizeLoop(currLoop);
      indexSymbols.add(LoopTools.getLoopIndexSymbol(currLoop));
    }
    collapsedLoops++;

    ForLoop innerLoop = indexedLoops.get(collapseLevel-1);
    Statement fBody = innerLoop.getBody();
    CompoundStatement cStmt = null;
    Statement tRefStmt = null;
    if( fBody instanceof CompoundStatement ) {
      cStmt = (CompoundStatement)fBody;
      tRefStmt = IRTools.getFirstNonDeclarationStatement(fBody);
    } else {
      cStmt = new CompoundStatement();
      cStmt.addStatement(fBody);
      tRefStmt = fBody;
    }
    //System.out.println("Loops to collapse: \n" + accLoop + "\n");
    //System.out.println("Innermost loop body: \n" + cStmt + "\n");
    //If const symbol declaration exists, old index variables may be used in its initialization.
    //In this case, expression statements assigning the old index variables should be put before
    //the const symbol declaration.
    //For simplicity, the new assignment expression statements added at the beginning whenever
    //const symbol declaration exists.
    boolean containsConstSymbol = false;
    Set<Symbol> lSymbolSet = cStmt.getSymbols();
    for(Symbol lSym : lSymbolSet) {
      if( SymbolTools.containsSpecifier(lSym, Specifier.CONST) ) {
        containsConstSymbol = true;
        break;
      }
    }
    if( containsConstSymbol ) {
      tRefStmt = (Statement)cStmt.getChildren().get(0);
    }
    if( tRefStmt == null ) {
      Tools.exit("[ERROR in CollapseTransformation] can not find referent statement "
          + "to insert old index calculation statements; exit!" + AnalysisTools.getEnclosingContext(accLoop));
    }
    ArrayList<Expression> iterspaceList = new ArrayList<Expression>();
    ArrayList<Expression> lbList = new ArrayList<Expression>();
    Expression collapsedIterSpace = null;
    if( simplifyE ) {
      for( int i=0; i<collapseLevel; i++ ) {
        ForLoop loop = indexedLoops.get(i);
        Expression lb = LoopTools.getLowerBoundExpression(loop);
        lbList.add(i, lb);
        Expression ub = LoopTools.getUpperBoundExpression(loop);
        Expression itrSpace = Symbolic.add(Symbolic.subtract(ub,lb),new IntegerLiteral(1));
        iterspaceList.add(i, itrSpace);
        if( i==0 ) {
          collapsedIterSpace = itrSpace;
        } else {
          collapsedIterSpace = Symbolic.multiply(collapsedIterSpace, itrSpace);
        }
      }
    } else {
      for( int i=0; i<collapseLevel; i++ ) {
        ForLoop loop = indexedLoops.get(i);
        Expression lb = LoopTools.getLowerBoundExpressionNS(loop);
        lbList.add(i, lb);
        Expression ub = LoopTools.getUpperBoundExpressionNS(loop);
        Expression itrSpace = Symbolic.add(Symbolic.subtract(ub,lb),new IntegerLiteral(1));
        iterspaceList.add(i, itrSpace);
        if( i==0 ) {
          collapsedIterSpace = itrSpace;
        } else {
          collapsedIterSpace = Symbolic.multiply(collapsedIterSpace, itrSpace);
        }
      }
    }
    //Create a new index variable for the newly collapsed loop.
    CompoundStatement procBody = proc.getBody();

    Identifier newIndex = null;
    if( enclosingCompRegion instanceof CompoundStatement ) {
      newIndex = TransformTools.getNewTempIndex(enclosingCompRegion);
    } else if( enclosingCompRegion instanceof ForLoop ) {
      newIndex = TransformTools.getNewTempIndex(((ForLoop)enclosingCompRegion).getBody());
    } else {
      newIndex = TransformTools.getNewTempIndex(procBody);

      //If the current accLoop is in a compute region, the above new private variable should be
      //added to the private clause.
      Set<SubArray> privateSet = null;
      if( collapseAnnot.containsKey("private") ) {
        privateSet = collapseAnnot.get("private");
      } else {
        privateSet = new HashSet<SubArray>();
        collapseAnnot.put("private", privateSet);
      }
      privateSet.add(AnalysisTools.createSubArray(newIndex.getSymbol(), true, null));
      if( ompAnnot != null ) {
        Set<String> ompPrivSet = ompAnnot.get("private");
        if( ompPrivSet == null ) {
          ompPrivSet = new HashSet<String>();
          ompAnnot.put("private", ompPrivSet);
        }
        ompPrivSet.add(newIndex.toString());
      }
    }
    if( !collapseAnnot.containsKey("gang") && !collapseAnnot.containsKey("worker") &&
        !collapseAnnot.containsKey("vector") && !collapseAnnot.containsKey("seq") ) {
      collapseAnnot.put("seq", "true");
    }

    /////////////////////////////////////////////////////////////////////////////////
    //Swap initialization statement, condition, and step of the OpenACC loop with  //
    //those of the new, collapsed loop.                                            //
    /////////////////////////////////////////////////////////////////////////////////

    Expression expr1 = new AssignmentExpression(newIndex.clone(), AssignmentOperator.NORMAL,
        new IntegerLiteral(0));
    Statement initStmt = new ExpressionStatement(expr1);
    accLoop.getInitialStatement().swapWith(initStmt);
    expr1 = new BinaryExpression((Identifier)newIndex.clone(), BinaryOperator.COMPARE_LT,
        collapsedIterSpace);
    accLoop.getCondition().swapWith(expr1);
    expr1 = new UnaryExpression(
        UnaryOperator.POST_INCREMENT, (Identifier)newIndex.clone());
    accLoop.getStep().swapWith(expr1);

    int targetArch = Integer.valueOf(System.getenv("OPENARC_ARCH") ).intValue();  
    // single work-item FPGA execution
    if (isSingleWorkItem == 1 && targetArch == 3) {
      int i = 1;
      while (i < collapseLevel) {
        Identifier tIndex = new Identifier(indexSymbols.get(i));
        Identifier nIndex = new Identifier(indexSymbols.get(i-1));

        Statement stmt;

        // x = x + 1
        if (i == collapseLevel-1) { 
          expr1 = new BinaryExpression(tIndex.clone(), BinaryOperator.ADD, new IntegerLiteral(1));

          stmt = new ExpressionStatement(new AssignmentExpression(tIndex.clone(), 
                AssignmentOperator.NORMAL, expr1));
          cStmt.addStatement(stmt);
        }

        // y = (x == X) ? y + 1 : y
        Expression expr_true = new BinaryExpression(nIndex.clone(), BinaryOperator.ADD, 
            new IntegerLiteral(1));

        Expression expr_cond = new BinaryExpression(tIndex.clone(), BinaryOperator.COMPARE_EQ, 
            iterspaceList.get(i).clone());

        expr1 = new ConditionalExpression(expr_cond, expr_true, nIndex.clone());

        stmt = new ExpressionStatement(new AssignmentExpression(nIndex.clone(), 
              AssignmentOperator.NORMAL, expr1));
        cStmt.addStatement(stmt);

        // x = (x == X) ? 0 : x; 
        expr_cond = new BinaryExpression(tIndex.clone(), BinaryOperator.COMPARE_EQ, 
            iterspaceList.get(i).clone());

        expr1 = new ConditionalExpression(expr_cond, new IntegerLiteral(0), tIndex.clone());

        stmt = new ExpressionStatement(new AssignmentExpression(tIndex.clone(), 
              AssignmentOperator.NORMAL, expr1));
        cStmt.addStatement(stmt);

        i++;
      }

      // Add initalizations before loop
      CompoundStatement accLoopParent = (CompoundStatement) accLoop.getParent();
      for (i = 0; i < collapseLevel; ++i) {
        Identifier tIndex = new Identifier(indexSymbols.get(i));

        expr1 = new AssignmentExpression(tIndex.clone(), AssignmentOperator.NORMAL,
            new IntegerLiteral(0));
        ExpressionStatement expStmt = new ExpressionStatement(expr1);
        accLoopParent.addStatementBefore(accLoop, expStmt);
      }
    }
    // Multiple work item or non-FPGA execution
    else {
      Identifier accforIndex = new Identifier(indexSymbols.get(0));
      int i = collapseLevel-1;
      while( i>0 ) {
        Identifier tIndex = new Identifier(indexSymbols.get(i));
        if( i == (collapseLevel-1) ) {
          expr1 = new BinaryExpression(newIndex.clone(), BinaryOperator.MODULUS, 
              iterspaceList.get(i).clone());
        } else {
          expr1 = new BinaryExpression(accforIndex.clone(), BinaryOperator.MODULUS, 
              iterspaceList.get(i).clone());
        }
        Expression lbExp = lbList.get(i).clone();
        if( !(lbExp instanceof Literal) || !lbExp.toString().equals("0") ) {
          expr1 = new BinaryExpression(expr1, BinaryOperator.ADD, lbExp);
        }
        Statement stmt = new ExpressionStatement(new AssignmentExpression(tIndex.clone(), 
              AssignmentOperator.NORMAL, expr1));
        if( containsConstSymbol ) {
          cStmt.addStatementBefore(tRefStmt, stmt);
        } else {
          TransformTools.addStatementBefore(cStmt, tRefStmt, stmt);
        }
        if( i == (collapseLevel-1) ) {
          expr1 = new BinaryExpression(newIndex.clone(), BinaryOperator.DIVIDE, 
              iterspaceList.get(i).clone());
        } else {
          expr1 = new BinaryExpression(accforIndex.clone(), BinaryOperator.DIVIDE, 
              iterspaceList.get(i).clone());
        }
        if( i == 1 ) {
          lbExp = lbList.get(0).clone();
          if( !(lbExp instanceof Literal) || !lbExp.toString().equals("0") ) {
            expr1 = new BinaryExpression(expr1, BinaryOperator.ADD, lbExp);
          }
        }
        stmt = new ExpressionStatement(new AssignmentExpression(accforIndex.clone(), 
              AssignmentOperator.NORMAL, expr1));
        if( containsConstSymbol ) {
          cStmt.addStatementBefore(tRefStmt, stmt);
        } else {
          TransformTools.addStatementBefore(cStmt, tRefStmt, stmt);
        }
        i--;
      }
    }

    /////////////////////////////////////////////////////////////////////////
    //Swap the body of the OpenACC loop with the one of the innermost loop //
    //among associated loops.                                              //
    /////////////////////////////////////////////////////////////////////////
    accLoop.getBody().swapWith(cStmt);

    return collapsedLoops;
  }

}
