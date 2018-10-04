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
 * @author Seyong Lee <lees2@ornl.gov>
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
  
  /* This method performs the first half of loop collapsing. That is, it performs:
   *  > for (i=...)
   *  >   for (j=...)
   *
   *  < for(index=...)
   *
   *  This is needed in SlidingWindwTransformation.java, as
   *  the variable replacement and increment is handled internally in 
   *  that transformation
   *
   *  DEBUG: This method could be simplified, and possibly moved to 
   *         FPGASpecificTools.java.
   */
  public static int collapseLoopHead(ForLoop accLoop, boolean simplifyE) {
    int collapsedLoops = 0;

    Traversable t = (Traversable)accLoop;
    while(t != null) {
      if (t instanceof Procedure) break;
      t = t.getParent(); 
    }

    if( t == null ) {
      Tools.exit("[ERROR in CollapseTransformation.collapseLoop()] Cannot find an enclosing procedure for the following loop:\n"
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
    ACCAnnotation iAnnot = accLoop.getAnnotation(ACCAnnotation.class, "internal");

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
    //
    //Create a new index variable for the newly collapsed loop.
    CompoundStatement procBody = proc.getBody();

    Identifier newIndex = null;
    if( enclosingCompRegion instanceof CompoundStatement ) {
      newIndex = TransformTools.getNewTempIndex(enclosingCompRegion);
    } else if( enclosingCompRegion instanceof ForLoop ) {
      newIndex = TransformTools.getNewTempIndex(((ForLoop)enclosingCompRegion).getBody());
    } else {
      newIndex = TransformTools.getNewTempIndex(procBody);
    }

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
    for(Symbol indexSym : indexSymbols ) {
    	privateSet.add(AnalysisTools.createSubArray(indexSym, true, null));
    }
    if( ompAnnot != null ) {
    	Set<String> ompPrivSet = ompAnnot.get("private");
    	if( ompPrivSet == null ) {
    		ompPrivSet = new HashSet<String>();
    		ompAnnot.put("private", ompPrivSet);
    	}
    	ompPrivSet.add(newIndex.toString());
    	for(Symbol indexSym : indexSymbols ) {
    		ompPrivSet.add(indexSym.getSymbolName());
    	}
    }
    if( iAnnot != null) {
    	Set<Symbol> accPrivateSymbols = (Set<Symbol>)iAnnot.get("accprivate");
		if( accPrivateSymbols == null ) {
			accPrivateSymbols = new HashSet<Symbol>();
			iAnnot.put("accprivate", accPrivateSymbols);
		}
		accPrivateSymbols.add(newIndex.getSymbol());
    	for(Symbol indexSym : indexSymbols ) {
    		accPrivateSymbols.add(indexSym);
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

  public static int collapseLoop(ForLoop accLoop, boolean simplifyE) {
    int collapsedLoops = 0;

    Traversable t = (Traversable)accLoop;
    while(t != null) {
      if (t instanceof Procedure) break;
      t = t.getParent(); 
    }

    if( t == null ) {
      Tools.exit("[ERROR in CollapseTransformation.collapseLoop()] Cannot find an enclosing procedure for the following loop:\n"
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
    ACCAnnotation loopAnnot = accLoop.getAnnotation(ACCAnnotation.class, "loop");
    OmpAnnotation ompAnnot = accLoop.getAnnotation(OmpAnnotation.class, "for");
    ACCAnnotation iAnnot = accLoop.getAnnotation(ACCAnnotation.class, "internal");
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
    }

    //If the current accLoop is in a compute region, the above new private variable and 
    //original index variables should be added to the private clause.
    Set<SubArray> privateSet = null;
    if( collapseAnnot.containsKey("private") ) {
    	privateSet = collapseAnnot.get("private");
    } else {
    	privateSet = new HashSet<SubArray>();
    	collapseAnnot.put("private", privateSet);
    }
    privateSet.add(AnalysisTools.createSubArray(newIndex.getSymbol(), true, null));
    for(Symbol indexSym : indexSymbols ) {
    	privateSet.add(AnalysisTools.createSubArray(indexSym, true, null));
    }
    if( ompAnnot != null ) {
    	Set<String> ompPrivSet = ompAnnot.get("private");
    	if( ompPrivSet == null ) {
    		ompPrivSet = new HashSet<String>();
    		ompAnnot.put("private", ompPrivSet);
    	}
    	ompPrivSet.add(newIndex.toString());
    	for(Symbol indexSym : indexSymbols ) {
    		ompPrivSet.add(indexSym.getSymbolName());
    	}
    }
    if( iAnnot != null) {
    	Set<Symbol> accPrivateSymbols = (Set<Symbol>)iAnnot.get("accprivate");
		if( accPrivateSymbols == null ) {
			accPrivateSymbols = new HashSet<Symbol>();
			iAnnot.put("accprivate", accPrivateSymbols);
		}
		accPrivateSymbols.add(newIndex.getSymbol());
    	for(Symbol indexSym : indexSymbols ) {
    		accPrivateSymbols.add(indexSym);
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


    /* Determine the enclosing parallel region */
    Statement parallelRegion;
    parallelRegion = enclosingCompRegion == null ? accLoop : enclosingCompRegion;

    /* Single work-item FPGA kernel */
    if ( FPGASpecificTools.isFPGASingleWorkItemRegion(parallelRegion) ) {
      FPGASpecificTools.collapseTransformation(accLoop, cStmt, indexSymbols, iterspaceList, lbList, collapseLevel);
    }
    /* Multiple work item or non-FPGA kernel */
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
      /////////////////////////////////////////////////////////////////////////
      //Swap the body of the OpenACC loop with the one of the innermost loop //
      //among associated loops.                                              //
      /////////////////////////////////////////////////////////////////////////
      accLoop.getBody().swapWith(cStmt);

    }

    return collapsedLoops;
  }

}
